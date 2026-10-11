"""Phase 1 Task 3, step A0: go/no-go on the riskiest unknown (read-only: writes only to --out).

Does a render of the subject's OWN GUAVA avatar (enrolled with EHM-Tracker + GUAVA from the first
5 s of the track) point the student towards the correct pose? Compared with a DWPose 2D-keypoint
control under the same procedure.

Per track, on frames after the 5 s enrollment window (GT-keypoint crop, (c-vitpose) student):

Losses (each a function of pose rotations, camera translation and shape):
  sil        1 - soft IoU between the avatar's SOLID silhouette (all Gaussians opaque, scale >= 1 cm)
             and the StyleMatte matte
  part       set-valued cross-entropy between the avatar's rendered part map (15 parts, from each
             Gaussian's bound SMPL-X vertex) and Sapiens-Seg 0.3B labels; clothing labels allow
             every part they can cover, background pixels require no avatar
  feat_*     DINOv2 ViT-B/14 patch-feature cosine distance between the avatar's coarse RGB render,
             composited over the frame with the person greyed out, and the frame, in windows that
             follow the pose: upper body (mask-weighted), face, hands; feat = sum of the three
  kp         DWPose 2D keypoints (run on the same GT box), confidence-weighted Geman-McClure on the
             student's crop projection, averaged over 4 groups (body+feet, face, left hand, right hand)

Ranking accuracy: overall = fraction of frames with loss(GT pose) < loss(student pose), the GT pose
being the GT body rotations + a camera translation fitted to the projected GT joints (student shape
and fingers). Per part = fraction of frames with loss(GT pose) < loss(GT pose with that part's local
rotations taken from the student), so the two poses differ only by the student's error in that part
(legs; upper body incl. neck and head; hands = wrist rotations, 3DPW has no finger GT). (Replacing a
part of the STUDENT pose by GT is not a cleaner pose when the student's root is off: measured in the
smoke test, it raised MPJPE on one frame from 36 to 63 mm.)
Per-frame diagnostics: student front/back flip (facing angle > 120 deg) and leg swap (exchanging
left/right knee+ankle lowers their error by >= 50 mm), as defined in DESIGN.md section 4; DWPose
left/right flip (exchanging its left/right shoulder+hip keypoints lowers the distance to the
projected GT joints).

Fitting: from the student's pose, axis-angle offsets on root + 21 body joints and the camera
translation, Adam, N steps, objective loss/loss_0 + anchor * sum(offset^2); standard 3DPW errors
before and after.

Shape test: the student's prediction with its body shape replaced by the enrollment shape.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
EHM_ROOT = ROOT / "EHM-Tracker"

SAPIENS_CLASSES = ["Background", "Apparel", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm",
                   "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing",
                   "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
                   "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip",
                   "Lower_Teeth", "Upper_Teeth", "Tongue"]
PARTS = ["head", "torso", "pelvis", "l_uarm", "r_uarm", "l_larm", "r_larm", "l_hand", "r_hand",
         "l_uleg", "r_uleg", "l_lleg", "r_lleg", "l_foot", "r_foot"]
# SMPL-X joint (argmax skinning weight) -> part
JOINT_PART = {0: "pelvis", 1: "l_uleg", 2: "r_uleg", 3: "torso", 4: "l_lleg", 5: "r_lleg", 6: "torso",
              7: "l_foot", 8: "r_foot", 9: "torso", 10: "l_foot", 11: "r_foot", 12: "head", 13: "torso",
              14: "torso", 15: "head", 16: "l_uarm", 17: "r_uarm", 18: "l_larm", 19: "r_larm", 20: "l_hand",
              21: "r_hand", 22: "head", 23: "head", 24: "head"}
JOINT_PART.update({j: "l_hand" for j in range(25, 40)})
JOINT_PART.update({j: "r_hand" for j in range(40, 55)})
ALL_BODY = PARTS
SAPIENS_ALLOWED = {
    "Background": ["bg"], "Apparel": ALL_BODY, "Face_Neck": ["head"], "Hair": ["head"], "Lower_Lip": ["head"],
    "Upper_Lip": ["head"], "Lower_Teeth": ["head"], "Upper_Teeth": ["head"], "Tongue": ["head"],
    "Torso": ["torso", "pelvis"],
    "Upper_Clothing": ["torso", "pelvis", "l_uarm", "r_uarm", "l_larm", "r_larm"],
    "Lower_Clothing": ["pelvis", "l_uleg", "r_uleg", "l_lleg", "r_lleg"],
    "Left_Upper_Arm": ["l_uarm"], "Left_Lower_Arm": ["l_larm"], "Left_Hand": ["l_hand"],
    "Right_Upper_Arm": ["r_uarm"], "Right_Lower_Arm": ["r_larm"], "Right_Hand": ["r_hand"],
    "Left_Upper_Leg": ["l_uleg"], "Left_Lower_Leg": ["l_lleg"], "Left_Foot": ["l_foot"], "Left_Shoe": ["l_foot"],
    "Left_Sock": ["l_foot", "l_lleg"], "Right_Upper_Leg": ["r_uleg"], "Right_Lower_Leg": ["r_lleg"],
    "Right_Foot": ["r_foot"], "Right_Shoe": ["r_foot"], "Right_Sock": ["r_foot", "r_lleg"],
}
# SMPL-X body_pose indices (joint - 1) replaced by GT in the per-part counterfactuals
PART_JOINTS = {
    "legs": [0, 1, 3, 4, 6, 7, 9, 10],                       # hips, knees, ankles, feet
    "upper": [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18],      # spine x3, neck, collars, head, shoulders, elbows
    "hands": [19, 20],                                        # wrists
}
PART_DIAG = {"legs": "diag_lower_body_mm", "upper": "diag_upper_body_mm", "hands": "diag_hands_mm"}
UPPER_JOINTS = [0, 3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 21]
LOSSES = ["sil", "part", "feat_upper", "feat_face", "feat_hands", "feat", "kp"]
FIT_LOSSES = ["sil", "part", "feat", "kp"]


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("/raid/ubx858/outputs/phase1_task1/screen/c_vitpose/checkpoints/best.pt"))
    p.add_argument("--tracks", nargs="+", default=["courtyard_jumpBench_01", "outdoors_golf_00", "outdoors_parcours_01"])
    p.add_argument("--enroll-root", type=Path, default=Path("/raid/ubx858/outputs/phase1_task3/a0/ehm_run/tracked"))
    p.add_argument("--enroll-seconds", type=float, default=5.0)
    p.add_argument("--frames-per-track", type=int, default=50)
    p.add_argument("--fit-frames-per-track", type=int, default=10, help="evenly spaced subset of the ranking frames")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--anchor", type=float, default=0.1)
    p.add_argument("--sapiens", type=Path, default=Path("/raid/ubx858/models/sapiens/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-vis", type=int, default=3, help="overlay images per track")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)
    gpu_t0 = time.time()

    # --- PEAR side: student, EHM_v2, translation fit, SMPL-X -> DWPose mapping ---
    ev = load_by_path("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    cwd = Path.cwd()
    os.chdir(PEAR_ROOT)
    sys.path.insert(0, str(PEAR_ROOT))
    student, _ = ev.load_model("student_vitpose", args.checkpoint, "configs/student_l70_v2.yaml", 1, device)
    from models.modules.ehm import EHM_v2
    from models.smplx.smplx_head import smplx_joints_to_dwpose
    ehm_eval = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    while str(PEAR_ROOT) in sys.path:
        sys.path.remove(str(PEAR_ROOT))
    os.chdir(cwd)
    for name in list(sys.modules):
        f = getattr(sys.modules[name], "__file__", None) or ""
        if f.startswith(str(PEAR_ROOT)) and name.split(".")[0] in ("models", "utils", "dataset", "tools"):
            del sys.modules[name]

    # --- EHM-Tracker side: StyleMatte, DWPose (CPU onnxruntime: the CUDA provider needs CUDA 11 here) ---
    sys.path.insert(0, str(EHM_ROOT))
    os.chdir(ROOT)
    from src.modules.human_matting.matting import StyleMatteModel
    from src.modules.dwpose.onnxpose import inference_pose
    import onnxruntime as ort
    matte = StyleMatteModel(device=device, ckpt_fp=str(EHM_ROOT / "pretrained/matting/stylematte_synth.pt"))
    pose_session = ort.InferenceSession(str(EHM_ROOT / "pretrained/dwpose/dw-ll_ucoco_384.onnx"), providers=["CPUExecutionProvider"])
    sys.path.remove(str(EHM_ROOT))
    for name in list(sys.modules):
        f = getattr(sys.modules[name], "__file__", None) or ""
        if f.startswith(str(EHM_ROOT)) and name.split(".")[0] == "src":
            del sys.modules[name]

    # --- Sapiens, DINOv2 ---
    sapiens = torch.jit.load(str(args.sapiens)).eval().to(device)
    sap_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    sap_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    dino = torch.hub.load(os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"), "dinov2_vitb14",
                          source="local", pretrained=True).eval().to(device)
    for p_ in dino.parameters():
        p_.requires_grad_(False)
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # --- GUAVA side ---
    sys.path.insert(0, str(ROOT))
    import roma
    from dataset import TrackedData_infer
    from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
    from utils.general_utils import ConfigDict, add_extra_cfgs
    from utils.graphics_utils import get_full_proj_matrix
    cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "assets/GUAVA/config.yaml")))
    infer_model = Ubody_Gaussian_inferer(cfg.MODEL).to(device).eval()
    render_model = GaussianRenderer(cfg.MODEL).to(device).eval()
    ck = torch.load(ROOT / "assets/GUAVA/checkpoints/best_160000.pt", map_location="cpu", weights_only=True)
    infer_model.load_state_dict(ck["model"], strict=False)
    render_model.load_state_dict(ck["render_model"], strict=False)
    del ck

    size = 256
    view, _ = get_full_proj_matrix(torch.eye(4, device=device), torch.tensor(1.0 / 24, device=device))

    def make_cam(f, cx, cy):
        """Pinhole camera at the origin (OpenCV axes) for the 256 crop: focal f px, principal point (cx, cy) px.
        Same matrices as GUAVA's get_proj_matrix, plus the principal-point terms in [0, 2] and [1, 2]."""
        zn, zf = 0.01, 100.0
        P = torch.zeros(4, 4, device=device)
        P[0, 0] = 2.0 * f / size
        P[1, 1] = 2.0 * f / size
        P[0, 2] = 2.0 * cx / size - 1.0
        P[1, 2] = 2.0 * cy / size - 1.0
        P[3, 2] = 1.0
        P[2, 2] = zf / (zf - zn)
        P[2, 3] = -(zf * zn) / (zf - zn)
        full = view.unsqueeze(0).bmm(P.transpose(0, 1).unsqueeze(0)).squeeze(0)
        tan = torch.tensor([size / (2.0 * f)], device=device)
        return {"world_view_transform": view[None], "full_proj_transform": full[None], "tanfovx": tan, "tanfovy": tan,
                "image_height": torch.tensor([size], device=device), "image_width": torch.tensor([size], device=device),
                "camera_center": torch.zeros(1, 3, device=device), "f": float(f), "c": torch.tensor([cx, cy], device=device)}

    cam_student = make_cam(3072.0, 128.0, 128.0)       # PEAR crop camera: focal 24 on the normalised crop

    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    smpl_g = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female")}
    smpl_reg = ev.build_smpl(ev.DEFAULT_SMPL_DIR, "neutral", device).J_regressor.float()

    def gt_mesh(sample):
        v = smpl_g[sample.gender](global_orient=torch.from_numpy(sample.pose[None, :3]).to(device),
                                  body_pose=torch.from_numpy(sample.pose[None, 3:]).to(device),
                                  betas=torch.from_numpy(sample.betas[None]).to(device)).vertices
        return torch.einsum("ij,bvj->bvi", torch.from_numpy(sample.cam_rotation).to(device), v)

    def metrics(out, gt):
        pred = torch.sparse.mm(mapping, ev.predicted_smplx_vertices(out, ehm_eval)[0].float())[None]
        return {k: float(v[0]) for k, v in ev.frame_metrics(pred, gt, h36m, smpl_reg).items()}

    def rotmats(out):
        bp = out["body_param"]
        return torch.cat([bp["global_pose"].reshape(-1, 1, 3, 3), bp["body_pose"], bp["left_hand_pose"],
                          bp["right_hand_pose"]], 1)

    def as_output(out, rv, shape=None):
        new = dict(out); new["body_param"] = dict(out["body_param"])
        R = roma.rotvec_to_rotmat(rv)
        new["body_param"]["global_pose"] = R[:, 0].reshape(out["body_param"]["global_pose"].shape)
        new["body_param"]["body_pose"] = R[:, 1:22]
        if shape is not None:
            new["body_param"]["shape"] = shape
        return new

    def joints145(rv, out):
        o = as_output(out, rv)
        return ehm_eval(o["body_param"], o["flame_param"], pose_type="rotmat")["joints"]

    def project_px(joints, offset, camd):
        """Pixels in the 256 crop of body-model points translated by offset (OpenCV camera frame).
        Student camera: offset = (-tx, -ty, tz) of pd_cam, f = 3072, c = 128 (pd_cam R = diag(-1,-1,1))."""
        x = joints.float() + offset[:, None]
        return camd["f"] * x[..., :2] / x[..., 2:3].clamp_min(1e-3) + camd["c"]

    def to_offset(t):
        return torch.stack([-t[:, 0], -t[:, 1], t[:, 2]], 1)

    def gm(r2, sigma=0.05):
        return r2 / (r2 + sigma ** 2)

    results, per_track = [], {}
    timings = {k: [] for k in FIT_LOSSES}
    for track in args.tracks:
        # ---------------- enrollment: the subject's own avatar ----------------
        tdir = args.enroll_root / track
        dcfg = dict(cfg["DATASET"]); dcfg["data_path"] = str(tdir); cfg.update("DATASET", dcfg)
        src_ds = TrackedData_infer(cfg=cfg, split="test", device=device, test_full=True)
        vid = next(iter(src_ds.videos_info))
        keys = src_ds.videos_info[vid]["frames_keys"]
        track_pkl = pickle.load(open(tdir / "optim_tracking_ehm.pkl", "rb"))
        face_score = [float(np.mean(track_pkl[k]["dwpose_rlt"]["scores"][24:92])) for k in keys]
        key_idx = int(np.argmax(face_score))
        src = src_ds._load_source_info(vid, key_idx)
        t0 = time.time()
        with torch.no_grad():
            vertex_gs, uv_gs, _ = infer_model(src)
            avatar = Ubody_Gaussian(cfg.MODEL, vertex_gs, uv_gs, pruning=True)
            avatar.init_ehm(infer_model.ehm)
        torch.cuda.synchronize(device)
        build_s = time.time() - t0
        enroll_shape = src["smplx_coeffs"]["shape"].reshape(1, -1).float()
        n_vert_gs = avatar._smplx_xyz.shape[1]
        lbs_w = avatar.smplx.lbs_weights.detach().cpu()
        faces = avatar.smplx.faces_tensor.detach().cpu().long()
        binding = avatar._uv_binding_face.detach().cpu().long()
        print(json.dumps({"lbs_weights": list(lbs_w.shape), "vertex_gaussians": int(n_vert_gs), "faces": list(faces.shape),
                          "faces_max": int(faces.max()), "binding_max": int(binding.max()), "uv_gaussians": int(binding.numel())}), flush=True)
        vpart = torch.tensor([PARTS.index(JOINT_PART[int(j)]) for j in lbs_w.argmax(1).tolist()])
        gpart = torch.cat([vpart[:n_vert_gs], vpart[faces[binding, 0]]]).to(device)
        n_gs = gpart.shape[0]
        part_onehot = F.one_hot(gpart, len(PARTS)).float()[None]                          # (1, N, K)

        def render(rv, out, offset, camd, shape=None, solid_pass=True, real_pass=True):
            """offset: (1, 3) translation in the camera (OpenCV) frame. Solid pass: parts + solid alpha;
            real pass: coarse RGB + real alpha."""
            bp, fp = out["body_param"], out["flame_param"]
            smplx = {"exp": bp["exp"], "global_pose": rv[:, 0], "body_pose": rv[:, 1:22],
                     "left_hand_pose": rv[:, 22:37], "right_hand_pose": rv[:, 37:52],
                     "shape": bp["shape"] if shape is None else shape,
                     "joints_offset": src["smplx_coeffs"]["joints_offset"],
                     "head_scale": bp["head_scale"], "hand_scale": bp["hand_scale"]}
            flame = {k: fp[k] for k in ("expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params")}
            flame["shape_params"] = src["flame_coeffs"]["shape_params"]
            assets = avatar({"smplx_coeffs": smplx, "flame_coeffs": flame})
            assets = dict(assets, xyz=assets["xyz"] + offset[:, None])
            C = assets["features_color"].shape[-1]
            res = {}
            solid = dict(assets)
            fs = torch.zeros(1, n_gs, C, device=device)
            fs[..., 3:3 + len(PARTS)] = part_onehot
            fs[..., C - 1] = 1.0
            solid["features_color"] = fs
            solid["opacity"] = torch.ones_like(assets["opacity"])
            solid["scaling"] = assets["scaling"].clamp_min(0.01)
            if solid_pass:
                rs = render_model.forward_raw(solid, camd, bg=0.0)["raw_features"]
                res.update(parts=rs[:, 3:3 + len(PARTS)], alpha_solid=rs[:, C - 1])
            if real_pass:
                real = dict(assets)
                fr = assets["features_color"].clone()
                fr[..., C - 1] = 1.0
                real["features_color"] = fr
                rr = render_model.forward_raw(real, camd, bg=0.0)["raw_features"]
                res.update(rgb=rr[:, :3], alpha=rr[:, C - 1])
            return res

        # ---------------- frames ----------------
        samples_all = [s for s in ev.load_samples(ev.DEFAULT_DATASET.resolve(), "validation") if s.sequence == track]
        start = samples_all[0].frame
        samples = [s for s in samples_all if s.frame - start >= args.enroll_seconds * 30]
        pick = np.linspace(0, len(samples) - 1, args.frames_per_track).round().astype(int)
        fit_pick = set(pick[np.linspace(0, len(pick) - 1, args.fit_frames_per_track).round().astype(int)].tolist()) if args.fit_frames_per_track > 0 else set()
        ds = ev.CropDataset(samples, "gt_keypoints", None)
        seq = pickle.load(open(ev.DEFAULT_DATASET.resolve() / "sequenceFiles" / "validation" / f"{track}.pkl", "rb"), encoding="latin1")
        person = samples[0].person
        K = np.asarray(seq["cam_intrinsics"], np.float32)
        track_rows = []
        for n_vis, i in enumerate(pick):
            sample = samples[int(i)]
            item = ds[int(i)]
            affine = item["affine"].numpy().astype(np.float32)
            frame_rgb = cv2.imread(sample.image)[:, :, ::-1]
            image = item["image"][None].to(device).float() / 255.0
            t_frame = time.time()
            with torch.no_grad():
                out = student(image)
                out = {kk: ({k2: (v2.detach() if torch.is_tensor(v2) else v2) for k2, v2 in vv.items()} if isinstance(vv, dict) else vv.detach())
                       for kk, vv in out.items()}
                rv_s = roma.rotmat_to_rotvec(rotmats(out))
                t_s = out["pd_cam"][:, :3, 3].clone()
                gt = gt_mesh(sample)
                mask = matte(image, "alpha").clamp(0, 1)[None]                              # (1, 256, 256)

                # Sapiens on a 4:3 crop of the full frame around the same box, labels warped into the crop
                box = ev.gt_keypoint_box(sample.keypoints, frame_rgb.shape[1], frame_rgb.shape[0])
                cx, cy, side = box[0] + box[2] / 2, box[1] + box[3] / 2, max(box[2], box[3])
                Ap = np.array([[768 / (0.75 * side), 0, 384 - cx * 768 / (0.75 * side)],
                               [0, 1024 / side, 512 - cy * 1024 / side]], np.float32)
                crop_p = cv2.warpAffine(np.ascontiguousarray(frame_rgb), Ap, (768, 1024), flags=cv2.INTER_LINEAR)
                xp = (torch.from_numpy(crop_p).permute(2, 0, 1)[None].float().to(device) - sap_mean) / sap_std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = sapiens(xp)
                lab = F.interpolate(logits.float(), (1024, 768), mode="bilinear").argmax(1)[0].cpu().numpy().astype(np.uint8)
                M = np.vstack([affine, [0, 0, 1]]) @ np.linalg.inv(np.vstack([Ap, [0, 0, 1]]))
                lab256 = cv2.warpAffine(lab, M[:2].astype(np.float32), (256, 256), flags=cv2.INTER_NEAREST, borderValue=0)
                allowed = torch.zeros(1 + len(PARTS), 256, 256, dtype=torch.bool)
                for c, name in enumerate(SAPIENS_CLASSES):
                    sel = torch.from_numpy(lab256 == c)
                    for a in SAPIENS_ALLOWED[name]:
                        allowed[0 if a == "bg" else 1 + PARTS.index(a)] |= sel
                allowed = allowed.to(device)

                # DWPose on the same box (full frame), mapped into the crop
                bx = [float(box[0]), float(box[1]), float(box[0] + box[2]), float(box[1] + box[3])]
                kps, scs = inference_pose(pose_session, [bx], np.ascontiguousarray(frame_rgb))
                info = np.concatenate([kps, scs[..., None]], -1)
                neck = np.mean(info[:, [5, 6]], axis=1)
                neck[:, 2] = np.logical_and(info[:, 5, 2] > 0.3, info[:, 6, 2] > 0.3).astype(float)
                info = np.insert(info, 17, neck, axis=1)
                info[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] = info[:, [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]]
                kp_img = info[0, :, :2]
                kp_crop = kp_img @ affine[:, :2].T + affine[:, 2]
                kp_xy = torch.from_numpy(kp_crop).float().to(device)
                kp_conf = torch.from_numpy(info[0, :, 2]).float().to(device)
                kp_w = torch.where(kp_conf >= 0.3, kp_conf, torch.zeros_like(kp_conf))
                kp_w[1] = 0.0                                                                     # synthetic neck
                groups = [list(range(0, 24)), list(range(24, 92)), list(range(92, 113)), list(range(113, 134))]

                # GT pose: 3DPW SMPL joints 1-21 -> SMPL-X body joints 1-21, root rotated into the camera frame;
                # camera translation fitted to the projected GT SMPL joints (true intrinsics and extrinsics)
                gp = torch.from_numpy(np.asarray(sample.pose, np.float32)).to(device)
                rv_g = rv_s.clone()
                rv_g[:, 0] = roma.rotmat_to_rotvec(torch.from_numpy(sample.cam_rotation).to(device) @ roma.rotvec_to_rotmat(gp[:3]))
                rv_g[:, 1:22] = gp[3:66].reshape(21, 3)
                fr_idx = sample.frame
                pose_w = np.asarray(seq["poses"][person][fr_idx], np.float32)
                trans_w = np.asarray(seq["trans"][person][fr_idx], np.float32)
                Ew = np.asarray(seq["cam_poses"][fr_idx], np.float32)
                jw = smpl_g[sample.gender](global_orient=torch.from_numpy(pose_w[None, :3]).to(device),
                                           body_pose=torch.from_numpy(pose_w[None, 3:]).to(device),
                                           betas=torch.from_numpy(sample.betas[None]).to(device),
                                           transl=torch.from_numpy(trans_w[None]).to(device)).joints[0, :22]
                jc = jw @ torch.from_numpy(Ew[:3, :3]).to(device).T + torch.from_numpy(Ew[:3, 3]).to(device)
                uv = (jc[:, :2] / jc[:, 2:3]) * torch.tensor([K[0, 0], K[1, 1]], device=device) + torch.tensor([K[0, 2], K[1, 2]], device=device)
                uv_crop = uv @ torch.from_numpy(affine[:, :2]).to(device).T + torch.from_numpy(affine[:, 2]).to(device)
                # true crop camera: 3DPW intrinsics through the crop affine (no rotation, uniform scale)
                sc = float(affine[0, 0])
                pp = affine[:, :2] @ K[:2, 2] + affine[:, 2]
                cam_true = make_cam(float(K[0, 0]) * sc, float(pp[0]), float(pp[1]))
                jm_g = joints145(rv_g, out)[:, :22]
                o_g = (jc[0] - jm_g[0, 0])[None]                                          # GT root placed at its true position
                gt_joint_px_err = float((project_px(jm_g, o_g, cam_true)[0] - uv_crop).norm(dim=-1).mean())
                # student: its pd_cam root position, converted to the true focal at the same pixel and scale
                o_s = to_offset(t_s)
                r_s = joints145(rv_s, out)[:, 0]                                            # (1, 3) root, model coords
                p_root = r_s + o_s
                z_new = p_root[:, 2] * cam_true["f"] / 3072.0
                xy_new = (3072.0 * p_root[:, :2] / p_root[:, 2:3] + 128.0 - cam_true["c"]) * z_new[:, None] / cam_true["f"]
                o_s_true = torch.cat([xy_new, z_new[:, None]], 1) - r_s
                kp_s_own = project_px(smplx_joints_to_dwpose(joints145(rv_s, out))[0], o_s, cam_student)[0]
                kp_s_true = project_px(smplx_joints_to_dwpose(joints145(rv_s, out))[0], o_s_true, cam_true)[0]
                student_cam_conversion_px = float((kp_s_own - kp_s_true).norm(dim=-1).mean())

            frame = image                                                                  # (1, 3, 256, 256) in [0, 1]
            grey = frame * (1 - mask[:, None]) + 0.5 * mask[:, None]

            def window(img, c, s):
                """Square window centred at c (px) with side s (px) -> 224x224, differentiable."""
                theta = torch.zeros(img.shape[0], 2, 3, device=device)
                theta[:, 0, 0] = s / 256.0
                theta[:, 1, 1] = s / 256.0
                theta[:, 0, 2] = 2 * c[:, 0] / 256.0 - 1
                theta[:, 1, 2] = 2 * c[:, 1] / 256.0 - 1
                grid = F.affine_grid(theta, (img.shape[0], img.shape[1], 224, 224), align_corners=False)
                return F.grid_sample(img, grid, align_corners=False, padding_mode="zeros")

            def dino_tokens(x):
                return dino.forward_features((x - im_mean) / im_std)["x_norm_patchtokens"]    # (B, 256, 768)

            def all_losses(rv, offset, camd, shape=None, need=("sil", "part", "feat", "kp")):
                L = {}
                if any(n in need for n in ("sil", "part", "feat")):
                    r = render(rv, out, offset, camd, shape, solid_pass="sil" in need or "part" in need, real_pass="feat" in need)
                if "sil" in need:
                    a = r["alpha_solid"]
                    L["sil"] = 1 - (a * mask).sum() / (a + mask - a * mask).sum().clamp_min(1e-6)
                if "part" in need:
                    p = torch.cat([(1 - r["alpha_solid"]).clamp(0, 1)[:, None], r["parts"].clamp(0, 1)], 1)[0]
                    L["part"] = -torch.log((p * allowed).sum(0) + 1e-4).mean()
                if "feat" in need or "kp" in need:
                    j = joints145(rv, out)
                    pj = project_px(j, offset, camd)[0]                                      # (145, 2)
                if "feat" in need:
                    comp = r["rgb"] + (1 - r["alpha"][:, None]) * grey
                    up = pj[UPPER_JOINTS]
                    lo, hi = up.min(0).values, up.max(0).values
                    c_up = ((lo + hi) / 2)[None]
                    s_up = ((hi - lo).max() * 1.2).clamp_min(32.0)
                    c_face = pj[15][None]
                    s_face = ((pj[15] - pj[12]).norm() * 2.5).clamp_min(16.0)
                    s_hl = ((pj[20] - pj[28]).norm() * 3.0).clamp_min(16.0)
                    s_hr = ((pj[21] - pj[43]).norm() * 3.0).clamp_min(16.0)
                    c_hl = ((pj[20] + pj[28]) / 2)[None]
                    c_hr = ((pj[21] + pj[43]) / 2)[None]
                    crops_r = torch.cat([window(comp, c_up, s_up), window(comp, c_face, s_face),
                                         window(comp, c_hl, s_hl), window(comp, c_hr, s_hr)])
                    crops_f = torch.cat([window(frame, c_up, s_up), window(frame, c_face, s_face),
                                         window(frame, c_hl, s_hl), window(frame, c_hr, s_hr)])
                    tok = dino_tokens(torch.cat([crops_r, crops_f]))
                    d = 1 - F.cosine_similarity(tok[:4], tok[4:], dim=-1)                   # (4, 256)
                    union = torch.maximum(mask[:, None], r["alpha"][:, None].detach())
                    wu = F.adaptive_avg_pool2d(window(union, c_up, s_up), 16).flatten(1)[0]
                    L["feat_upper"] = (d[0] * wu).sum() / wu.sum().clamp_min(1e-6)
                    L["feat_face"] = d[1].mean()
                    L["feat_hands"] = (d[2].mean() + d[3].mean()) / 2
                    L["feat"] = L["feat_upper"] + L["feat_face"] + L["feat_hands"]
                if "kp" in need:
                    kp_pred = project_px(smplx_joints_to_dwpose(j)[0], offset, camd)[0]      # (134, 2); returns (kps, weights)
                    r2 = ((kp_pred - kp_xy) / 256.0).pow(2).sum(-1)
                    terms = []
                    for g in groups:
                        w = kp_w[g]
                        if w.sum() > 0:
                            terms.append((w * gm(r2[g])).sum() / w.sum())
                    L["kp"] = torch.stack(terms).mean()
                return L

            row = {"track": track, "frame": int(sample.frame), "true_focal_px": cam_true["f"],
                   "gt_joint_px_err_true_cam": gt_joint_px_err, "student_cam_conversion_px": student_cam_conversion_px}
            with torch.no_grad():
                # ranking under the true camera (both poses); fitting below uses the student's own camera
                poses = {"student": (rv_s, o_s_true), "gt": (rv_g, o_g)}
                for part, idx in PART_JOINTS.items():
                    rv_p = rv_g.clone()
                    rv_p[:, [k + 1 for k in idx]] = rv_s[:, [k + 1 for k in idx]]
                    poses[f"cf_{part}"] = (rv_p, o_g)
                for pname, (rv_, o_) in poses.items():
                    L = all_losses(rv_, o_, cam_true)
                    row.update({f"{pname}_{k}": float(v) for k, v in L.items()})
                    m = metrics(as_output(out, rv_), gt)
                    row.update({f"{pname}_m_{k}": v for k, v in m.items() if "h36m" not in k})
                m = metrics(as_output(out, rv_s, enroll_shape), gt)
                row.update({f"enrollshape_m_{k}": v for k, v in m.items() if "h36m" not in k})
                # diagnostics: student flip / leg swap vs GT (SMPL kinematic joints, camera frame, pelvis-centred)
                pk = torch.einsum("jv,bvc->bjc", smpl_reg, torch.sparse.mm(mapping, ev.predicted_smplx_vertices(out, ehm_eval)[0].float())[None])[0]
                gk = torch.einsum("jv,bvc->bjc", smpl_reg, gt)[0]
                pk, gk = pk - pk[:1], gk - gk[:1]
                def facing(k):
                    return F.normalize(torch.cross(k[1] - k[2], k[12] - k[0], dim=0), dim=0)
                row["diag_facing_angle_deg"] = float(torch.rad2deg(torch.arccos((facing(pk) * facing(gk)).sum().clamp(-1, 1))))
                legs, legs_sw = [4, 5, 7, 8], [5, 4, 8, 7]
                e0 = (pk[legs] - gk[legs]).norm(dim=-1).sum()
                e1 = (pk[legs_sw] - gk[legs]).norm(dim=-1).sum()
                row["diag_leg_swap_gain_mm"] = float((e0 - e1) * 1000)
                # DWPose left/right flip vs projected GT SMPL joints (L/R shoulder 16/17, hip 1/2)
                dw = kp_xy[[5, 2, 11, 8]]                                                    # L sh, R sh, L hip, R hip
                gt2 = uv_crop[[16, 17, 1, 2]]
                ok = (kp_conf[[5, 2, 11, 8]] >= 0.3).all()
                d0 = (dw - gt2).norm(dim=-1).sum(); d1 = (dw[[1, 0, 3, 2]] - gt2).norm(dim=-1).sum()
                row["diag_dwpose_lr_flip"] = bool(ok and d1 < d0)
                row["diag_dwpose_face_conf"] = float(kp_conf[24:92].mean())
                torch.cuda.synchronize(device)
                row["rank_s"] = time.time() - t_frame
                if n_vis < args.save_vis:
                    rs, rg = render(rv_s, out, o_s_true, cam_true), render(rv_g, out, o_g, cam_true)
                    pal = np.random.default_rng(1).integers(40, 255, (len(PARTS) + 1, 3)); pal[0] = 0
                    def pmap(r):
                        p = torch.cat([(1 - r["alpha_solid"])[:, None], r["parts"]], 1)[0].argmax(0).cpu().numpy()
                        return pal[p].astype(np.uint8)
                    lab_rgb = np.zeros((256, 256, 3), np.uint8)
                    for c in range(len(SAPIENS_CLASSES)):
                        lab_rgb[lab256 == c] = np.random.default_rng(c + 7).integers(40, 255, 3) if c else 0
                    comp_s = (rs["rgb"] + (1 - rs["alpha"][:, None]) * grey)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    comp_g = (rg["rgb"] + (1 - rg["alpha"][:, None]) * grey)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    kv = (frame[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                    for q in range(134):
                        if kp_w[q] > 0:
                            cv2.circle(kv, (int(kp_crop[q, 0]), int(kp_crop[q, 1])), 1, (255, 0, 0), -1)
                    tiles = [kv, (mask[0, :, :, None].expand(-1, -1, 3).cpu().numpy() * 255).astype(np.uint8), lab_rgb,
                             pmap(rs), pmap(rg), (comp_s * 255).astype(np.uint8), (comp_g * 255).astype(np.uint8)]
                    cv2.imwrite(str(args.out / f"{track}_{sample.frame:05d}.jpg"),
                                np.ascontiguousarray(np.concatenate(tiles, 1)[:, :, ::-1]))

            if int(i) in fit_pick:
                for lname in FIT_LOSSES:
                    need = ("feat",) if lname == "feat" else (lname,)
                    d_rot = torch.zeros(1, 22, 3, device=device, requires_grad=True)
                    d_t = torch.zeros(1, 3, device=device, requires_grad=True)
                    opt = torch.optim.Adam([{"params": [d_rot], "lr": 0.02}, {"params": [d_t], "lr": 0.01}])
                    scale = torch.tensor([1.0, 1.0, 20.0], device=device)
                    with torch.no_grad():
                        l0 = float(all_losses(rv_s, o_s, cam_student, need=need)[lname])
                    torch.cuda.synchronize(device); ts = time.time()
                    for _ in range(args.steps):
                        rv = torch.cat([rv_s[:, :22] + d_rot, rv_s[:, 22:]], 1)
                        loss = all_losses(rv, o_s + d_t * scale, cam_student, need=need)[lname] / max(l0, 1e-6) + args.anchor * (d_rot ** 2).sum()
                        opt.zero_grad(); loss.backward(); opt.step()
                    torch.cuda.synchronize(device)
                    timings[lname].append((time.time() - ts) / args.steps * 1000)
                    with torch.no_grad():
                        rv = torch.cat([rv_s[:, :22] + d_rot, rv_s[:, 22:]], 1)
                        l1 = float(all_losses(rv, o_s + d_t * scale, cam_student, need=need)[lname])
                        m = metrics(as_output(out, rv), gt)
                    row.update({f"fit_{lname}_loss0": l0, f"fit_{lname}_loss1": l1,
                                **{f"fit_{lname}_m_{k}": v for k, v in m.items() if "h36m" not in k}})
            track_rows.append(row)
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()
                              if k in ("track", "frame") or k.startswith("fit_") and k.endswith("mpjpe_mm")}), flush=True)
        per_track[track] = {"enroll_frames": len(keys), "source_key": keys[key_idx], "source_face_score": face_score[key_idx],
                            "avatar_build_s": build_s, "gaussians": int(n_gs), "frames": len(track_rows),
                            "fit_frames": sum(1 for r in track_rows if "fit_kp_m_mpjpe_mm" in r)}
        results.extend(track_rows)

    # ---------------- summary ----------------
    def frac(rows, a, b):
        return float(np.mean([r[a] < r[b] for r in rows])) if rows else float("nan")

    summary = {"tracks": per_track, "frames": len(results), "steps": args.steps, "anchor": args.anchor,
               "wall_s": time.time() - gpu_t0, "ms_per_fit_step": {k: float(np.median(v)) if v else None for k, v in timings.items()}}
    rank = {}
    for lname in LOSSES:
        rank[lname] = {"overall": frac(results, f"gt_{lname}", f"student_{lname}"),
                       **{part: frac(results, f"gt_{lname}", f"cf_{part}_{lname}") for part in PART_JOINTS}}
    summary["ranking_accuracy"] = rank
    summary["camera_sanity"] = {
        "gt_joint_px_err_true_cam_mean": float(np.mean([r["gt_joint_px_err_true_cam"] for r in results])),
        "student_cam_conversion_px_mean": float(np.mean([r["student_cam_conversion_px"] for r in results])),
        "true_focal_px_mean": float(np.mean([r["true_focal_px"] for r in results]))}
    summary["counterfactual_sanity"] = {
        "gt_lower_mpjpe_than_student": frac(results, "gt_m_mpjpe_mm", "student_m_mpjpe_mm"),
        **{f"gt_lower_{PART_DIAG[part]}_than_cf_{part}": frac(results, f"gt_m_{PART_DIAG[part]}", f"cf_{part}_m_{PART_DIAG[part]}")
           for part in PART_JOINTS},
        **{f"mean_{PART_DIAG[part]}_gt_vs_cf_{part}": [float(np.mean([r[f"gt_m_{PART_DIAG[part]}"] for r in results])),
                                                       float(np.mean([r[f"cf_{part}_m_{PART_DIAG[part]}"] for r in results]))]
           for part in PART_JOINTS}}
    flip = [r for r in results if r["diag_facing_angle_deg"] > 120]
    swapf = [r for r in results if r["diag_leg_swap_gain_mm"] >= 50]
    dwflip = [r for r in results if r["diag_dwpose_lr_flip"]]
    summary["diagnostics"] = {"student_flip_frames": len(flip), "student_leg_swap_frames": len(swapf),
                              "dwpose_lr_flip_frames": len(dwflip),
                              "ranking_overall_on_student_flip_frames": {l: frac(flip, f"gt_{l}", f"student_{l}") for l in LOSSES},
                              "ranking_legs_on_student_swap_frames": {l: frac(swapf, f"gt_{l}", f"cf_legs_{l}") for l in LOSSES},
                              "ranking_overall_on_dwpose_flip_frames": {l: frac(dwflip, f"gt_{l}", f"student_{l}") for l in LOSSES},
                              "median_rank_s_per_frame": float(np.median([r["rank_s"] for r in results]))}
    mkeys = ["mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "diag_lower_body_mm", "diag_upper_body_mm", "diag_hands_mm"]
    fit_rows = [r for r in results if "fit_kp_m_mpjpe_mm" in r]
    summary["fit"] = {} if not fit_rows else {"frames": len(fit_rows),
                      "before": {k: float(np.mean([r[f"student_m_{k}"] for r in fit_rows])) for k in mkeys},
                      **{lname: {k: float(np.mean([r[f"fit_{lname}_m_{k}"] for r in fit_rows])) for k in mkeys}
                         for lname in FIT_LOSSES},
                      "loss_ratio": {lname: float(np.mean([r[f"fit_{lname}_loss1"] / max(r[f"fit_{lname}_loss0"], 1e-9) for r in fit_rows]))
                                     for lname in FIT_LOSSES}}
    summary["shape_test"] = {k: {"student_shape": float(np.mean([r[f"student_m_{k}"] for r in results])),
                                 "enroll_shape": float(np.mean([r[f"enrollshape_m_{k}"] for r in results]))}
                             for k in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm")}
    summary["shape_test_frac_pve_better"] = frac(results, "enrollshape_m_pve_mm", "student_m_pve_mm")
    summary["gt_pose_errors"] = {k: float(np.mean([r[f"gt_m_{k}"] for r in results])) for k in mkeys}
    summary["per_track_ranking"] = {
        t: {lname: frac([r for r in results if r["track"] == t], f"gt_{lname}", f"student_{lname}") for lname in LOSSES}
        for t in per_track}
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "frames": results}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
