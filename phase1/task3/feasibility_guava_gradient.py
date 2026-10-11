"""Phase 1 Task 3 feasibility check (read-only: loads models and data, writes only to --out).

Riskiest unknown of AvatarLoop: does a GUAVA render give a *useful* gradient for the student's pose
on a real 3DPW frame, including the legs (GUAVA is an upper-body model; legs only enter through
their silhouette)?

Per frame of one single-person 3DPW validation sequence:
1. the (c-vitpose) student predicts EHM-s pose, shape and crop camera from the standard
   GT-keypoint crop;
2. a GUAVA avatar (an already-tracked example person: no 3DPW enrollment exists yet, so
   appearance is NOT the subject's; the student's predicted body shape is used) is deformed to
   that pose and rasterised at the student's camera, translation applied to the Gaussian means
   (the rasteriser returns no gradient for the view matrix); a second pass with all-ones colours
   and a black background gives the silhouette (accumulated opacity);
3. the silhouette is compared with a StyleMatte matte of the same crop (soft IoU / L1);
4. gradient flow: the silhouette loss computed from the student's output is backpropagated to the
   student's decoder (head) weights;
5. pose recovery: starting from the student's prediction, axis-angle offsets on the root and 21
   body joints plus the camera are optimised for N steps on the silhouette loss, once per anchor
   weight (squared offset norm); 3DPW errors (standard protocol functions) before and after,
   overall and per part;
6. reference: the ground-truth body pose rendered the same way (only the camera translation
   fitted), i.e. the silhouette IoU this avatar/matte pair reaches at the correct pose.
Also reports render/backward time per step.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PEAR_ROOT = ROOT / "third_party" / "PEAR"


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("/raid/ubx858/outputs/phase1_task1/screen/c_vitpose/checkpoints/best.pt"))
    p.add_argument("--sequence", default="courtyard_jumpBench_01")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--anchors", type=float, nargs="+", default=[1e-3, 1e-1, 1.0],
                   help="weights of the squared axis-angle offset anchor, one pose-recovery run each")
    p.add_argument("--sil-mode", choices=("gaussian", "solid"), default="gaussian",
                   help="gaussian: the avatar's own opacities and scales; solid: every Gaussian opaque with "
                        "scale >= 1 cm (approximates the posed body-mesh silhouette)")
    p.add_argument("--avatar", type=Path, default=ROOT / "assets/example/tracked_image/vicox_14_right")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)

    # --- PEAR side (student, PEAR's EHM_v2 for evaluation), then release PEAR's package names ---
    ev = load_by_path("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    cwd = Path.cwd()
    os.chdir(PEAR_ROOT)
    sys.path.insert(0, str(PEAR_ROOT))
    student, _ = ev.load_model("student_vitpose", args.checkpoint, "configs/student_l70_v2.yaml", 1, device)
    from models.modules.ehm import EHM_v2
    ehm_eval = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    while str(PEAR_ROOT) in sys.path:   # load_model also inserts it
        sys.path.remove(str(PEAR_ROOT))
    os.chdir(cwd)
    for name in list(sys.modules):
        f = getattr(sys.modules[name], "__file__", None) or ""
        if f.startswith(str(PEAR_ROOT)) and name.split(".")[0] in ("models", "utils", "dataset"):
            del sys.modules[name]

    # --- matting (EHM-Tracker's StyleMatte) ---
    sys.path.insert(0, str(ROOT / "EHM-Tracker"))
    from src.modules.human_matting.matting import StyleMatteModel
    matte = StyleMatteModel(device=device, ckpt_fp=str(ROOT / "EHM-Tracker/pretrained/matting/stylematte_synth.pt"))
    sys.path.remove(str(ROOT / "EHM-Tracker"))

    # --- GUAVA side ---
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
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
    dcfg = dict(cfg["DATASET"]); dcfg["data_path"] = str(args.avatar); cfg.update("DATASET", dcfg)
    src_ds = TrackedData_infer(cfg=cfg, split="test", device=device, test_full=True)
    src = src_ds._load_source_info(next(iter(src_ds.videos_info)))
    t0 = time.time()
    with torch.no_grad():   # not inference_mode: avatar tensors must be usable inside autograd
        vertex_gs, uv_gs, _ = infer_model(src)
        avatar = Ubody_Gaussian(cfg.MODEL, vertex_gs, uv_gs, pruning=True)
        avatar.init_ehm(infer_model.ehm)
    torch.cuda.synchronize(device)
    build_s = time.time() - t0
    n_gauss = avatar._smplx_xyz.shape[1] + avatar._uv_local_xyz.shape[1]

    # Camera: identity view (camera at origin, looking +z), focal 24 over the image, 256 px. The
    # student's translation is added to the Gaussian means: X_cam = X + (-tx, -ty, tz).
    size = 256
    w2c = torch.eye(4, device=device)
    view, proj = get_full_proj_matrix(w2c, torch.tensor(1.0 / 24, device=device))
    cam = {"world_view_transform": view[None], "full_proj_transform": proj[None],
           "tanfovx": torch.tensor([1.0 / 24], device=device), "tanfovy": torch.tensor([1.0 / 24], device=device),
           "image_height": torch.tensor([size], device=device), "image_width": torch.tensor([size], device=device),
           "camera_center": torch.zeros(1, 3, device=device)}

    def silhouette(rotvecs, out, t):
        """rotvecs: (1, 52, 3) axis-angle [root, 21 body, 15 left hand, 15 right hand]."""
        bp, fp = out["body_param"], out["flame_param"]
        smplx = {"exp": bp["exp"], "global_pose": rotvecs[:, 0], "body_pose": rotvecs[:, 1:22],
                 "left_hand_pose": rotvecs[:, 22:37], "right_hand_pose": rotvecs[:, 37:52],
                 "shape": bp["shape"], "joints_offset": src["smplx_coeffs"]["joints_offset"],
                 "head_scale": bp["head_scale"], "hand_scale": bp["hand_scale"]}
        flame = {k: fp[k] for k in ("expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params")}
        flame["shape_params"] = src["flame_coeffs"]["shape_params"]
        assets = avatar({"smplx_coeffs": smplx, "flame_coeffs": flame})
        offset = torch.stack([-t[:, 0], -t[:, 1], t[:, 2]], 1)
        assets = dict(assets, xyz=assets["xyz"] + offset[:, None])
        assets["features_color"] = torch.ones_like(assets["features_color"])
        if args.sil_mode == "solid":
            assets["opacity"] = torch.ones_like(assets["opacity"])
            assets["scaling"] = assets["scaling"].clamp_min(0.01)
        return render_model.forward_raw(assets, cam, bg=0.0)["raw_features"][:, 0]   # (1, H, W) alpha

    def rotmats(out):
        bp = out["body_param"]
        return torch.cat([bp["global_pose"].reshape(-1, 1, 3, 3), bp["body_pose"], bp["left_hand_pose"],
                          bp["right_hand_pose"]], 1)

    def metrics(out, sample):
        pred = torch.sparse.mm(mapping, ev.predicted_smplx_vertices(out, ehm_eval)[0].float())[None]
        g = {"male": smpl_m, "female": smpl_f}[sample.gender]
        v = g(global_orient=torch.from_numpy(sample.pose[None, :3]).to(device),
              body_pose=torch.from_numpy(sample.pose[None, 3:]).to(device),
              betas=torch.from_numpy(sample.betas[None]).to(device)).vertices
        gt = torch.einsum("ij,bvj->bvi", torch.from_numpy(sample.cam_rotation).to(device), v)
        m = ev.frame_metrics(pred, gt, h36m, smpl_reg)
        return {k: float(v[0]) for k, v in m.items()}

    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    smpl_m, smpl_f = (ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female"))
    smpl_reg = ev.build_smpl(ev.DEFAULT_SMPL_DIR, "neutral", device).J_regressor.float()

    samples = [s for s in ev.load_samples(ev.DEFAULT_DATASET.resolve(), "validation") if s.sequence == args.sequence]
    pick = np.linspace(len(samples) * 0.1, len(samples) - 1, args.frames).astype(int)
    ds = ev.CropDataset(samples, "gt_keypoints", None)
    results = []
    for k, i in enumerate(pick):
        item = ds[int(i)]
        sample = samples[int(i)]
        image = item["image"][None].to(device).float() / 255.0                      # (1, 3, 256, 256)
        with torch.no_grad():
            mask = matte(image, "alpha").clamp(0, 1)[None]                             # (1, 256, 256)

        # (4) gradient reaches the student's decoder weights
        student.zero_grad(set_to_none=True)
        out = student(image)
        rv = roma.rotmat_to_rotvec(rotmats(out))
        sil = silhouette(rv, out, out["pd_cam"][:, :3, 3])
        loss = (sil - mask).abs().mean()
        loss.backward()
        head_g = [p.grad for p in student.head.parameters() if p.grad is not None]
        g_norm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in head_g)))
        g_finite = all(torch.isfinite(g).all() for g in head_g)

        # (5) pose recovery on the silhouette alone
        with torch.no_grad():
            out = student(image)
            out = {kk: ({k2: (v2.detach() if torch.is_tensor(v2) else v2) for k2, v2 in vv.items()} if isinstance(vv, dict) else vv.detach())
                   for kk, vv in out.items()}
            rv0 = roma.rotmat_to_rotvec(rotmats(out))
            t0_ = out["pd_cam"][:, :3, 3].clone()
            before = metrics(out, sample)
        def soft_iou(a, b):
            return float((a * b).sum() / (a + b - a * b).sum().clamp_min(1e-6))

        def recover(rv_start, anchor, fit_pose=True):
            d_rot = torch.zeros(1, 22, 3, device=device, requires_grad=fit_pose)
            d_t = torch.zeros(1, 3, device=device, requires_grad=True)
            groups = [{"params": [d_t], "lr": 0.01}] + ([{"params": [d_rot], "lr": 0.02}] if fit_pose else [])
            opt = torch.optim.Adam(groups)
            scale = torch.tensor([1.0, 1.0, 20.0], device=device)
            iou_start = None
            torch.cuda.synchronize(device); ts = time.time()
            for step in range(args.steps):
                rv = torch.cat([rv_start[:, :22] + d_rot, rv_start[:, 22:]], 1)
                sil = silhouette(rv, out, t0_ + d_t * scale)
                if iou_start is None:
                    iou_start = soft_iou(sil.detach(), mask)
                loss = (sil - mask).abs().mean() + anchor * (d_rot ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(device); ms = (time.time() - ts) / args.steps * 1000
            with torch.no_grad():
                rv = torch.cat([rv_start[:, :22] + d_rot, rv_start[:, 22:]], 1)
                sil = silhouette(rv, out, t0_ + d_t * scale)
                new = dict(out); new["body_param"] = dict(out["body_param"])
                R = roma.rotvec_to_rotmat(rv)
                new["body_param"]["global_pose"] = R[:, 0].reshape(out["body_param"]["global_pose"].shape)
                new["body_param"]["body_pose"] = R[:, 1:22]
                return iou_start, soft_iou(sil, mask), metrics(new, sample), ms, sil

        r = {"frame": int(sample.frame), "head_grad_norm": g_norm, "head_grad_finite": bool(g_finite)}
        sils = []
        for anchor in args.anchors:
            iou0, iou1, after, step_ms, sil = recover(rv0, anchor)
            sils.append(sil)
            r.update({f"a{anchor:g}_iou_before": iou0, f"a{anchor:g}_iou_after": iou1, "ms_per_step": step_ms,
                      **{f"a{anchor:g}_after_{k2}": v2 for k2, v2 in after.items()}})
        r.update({f"before_{k2}": v2 for k2, v2 in before.items()})
        # Reference: ground-truth body pose (3DPW SMPL joints 1-21 = SMPL-X body joints 1-21, root rotated
        # into the camera frame), student shape and hands, camera translation fitted on the silhouette.
        with torch.no_grad():
            gt_pose = torch.from_numpy(np.asarray(sample.pose, np.float32)).to(device)
            gt_root = torch.from_numpy(np.asarray(sample.cam_rotation, np.float32)).to(device) @ roma.rotvec_to_rotmat(gt_pose[:3])
            rv_gt = rv0.clone()
            rv_gt[:, 0] = roma.rotmat_to_rotvec(gt_root)
            rv_gt[:, 1:22] = gt_pose[3:66].reshape(21, 3)
        iou_g0, iou_g1, gt_m, _, sil = recover(rv_gt, 0.0, fit_pose=False)
        sils.append(sil)
        r.update({"gtpose_iou_camfit": iou_g1, **{f"gtpose_{k2}": v2 for k2, v2 in gt_m.items()}})
        with torch.no_grad():
            # image | matte | silhouette after recovery, one per anchor | GT pose (camera fitted)
            vis = np.concatenate([item["image"].permute(1, 2, 0).numpy()[:, :, ::-1]] +
                                 [(m[0, :, :, None].expand(-1, -1, 3).cpu().numpy() * 255).astype(np.uint8)
                                  for m in [mask] + sils], 1)
            cv2.imwrite(str(args.out / f"frame_{k}_{sample.frame:05d}.jpg"), np.ascontiguousarray(vis))
        results.append(r)
        print(json.dumps({k2: (round(v2, 2) if isinstance(v2, float) else v2) for k2, v2 in r.items()
                          if "h36m" not in k2 and "hands" not in k2}), flush=True)

    keys = sorted(k2 for k2 in results[0] if k2 not in ("frame", "head_grad_norm", "head_grad_finite"))
    summary = {"sequence": args.sequence, "frames": len(results), "steps": args.steps, "anchors": args.anchors,
               "sil_mode": args.sil_mode,
               "avatar": str(args.avatar), "avatar_build_s": build_s, "gaussians": int(n_gauss),
               "mean": {k2: float(np.mean([r[k2] for r in results])) for k2 in keys},
               "head_grad_all_finite": all(r["head_grad_finite"] for r in results),
               "head_grad_norm_mean": float(np.mean([r["head_grad_norm"] for r in results]))}
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "frames": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
