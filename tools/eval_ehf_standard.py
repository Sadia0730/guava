#!/usr/bin/env python
"""EHF evaluator for EHM-s models, following the Hand4Whole protocol used by SMPL-X methods.

Protocol (Hand4Whole_RELEASE @ c94908654b81, data/EHF/EHF.py `evaluate`,
common/utils/human_models.py, common/utils/transforms.py `rigid_align`; OSX and SMPLer-X reuse it):
- ground truth: `NN_align.ply` (SMPL-X topology) rotated by the EHF camera rotation;
- prediction: EHM-s vertices[:10475], which are in SMPL-X topology (see REPORT.md);
- PVE all: translate so the pelvis joint (SMPL-X J_regressor row 0) matches;
- PVE hands: per hand (MANO_SMPLX_vertex_ids.pkl), translate at the wrist (rows 20 / 21), mean of both;
- PVE face: SMPL-X__FLAME_vertex_ids.npy vertices, translate at the neck (row 12);
- PA-PVE: similarity Procrustes (`rigid_align`) on each vertex set separately;
- PA-MPJPE body (SMPLX_to_J14.pkl) and hands (make_hand_regressor) after `rigid_align`.
The crop deviates from Hand4Whole, whose body boxes come from a separate EHF.json we do not
have: the box is built from EHF's OpenPose BODY_25 keypoints exactly like the 3DPW main crop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_3dpw_standard as ev  # noqa: E402

EHF_ROOT = Path("/raid/ubx858/datasets/EHF/EHF")
SMPLX_ASSETS = ev.PEAR_ROOT / "assets" / "SMPLX"
# EHF_camera.txt
FOCAL = (1498.22426237, 1498.22426237)
PRINCPT = (790.263706, 578.90334)
CAM_T = np.array([-0.03609917, 0.43416458, 2.37101226])
CAM_R_AA = np.array([-2.98747896, 0.01172457, -0.05704687])
J_IDX = {"pelvis": 0, "lwrist": 20, "rwrist": 21, "neck": 12}


def read_ply_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    return np.frombuffer(raw, np.float32, count=10475 * 3, offset=end).reshape(10475, 3).astype(np.float64)


def rigid_transform_3D(A, B):
    """Verbatim port of Hand4Whole common/utils/transforms.py."""
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)
    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    return np.transpose(np.dot(c * R, np.transpose(A))) + t


class Protocol:
    def __init__(self):
        smplx = np.load(SMPLX_ASSETS / "SMPLX_NEUTRAL_2020.npz", allow_pickle=True)
        self.J = np.asarray(smplx["J_regressor"], dtype=np.float64)
        with open(SMPLX_ASSETS / "SMPLX_to_J14.pkl", "rb") as f:
            self.j14 = np.asarray(pickle.load(f, encoding="latin1"), dtype=np.float64)
        with open(SMPLX_ASSETS / "MANO_SMPLX_vertex_ids.pkl", "rb") as f:
            self.hand_idx = pickle.load(f, encoding="latin1")
        self.face_idx = np.load(SMPLX_ASSETS / "SMPL-X__FLAME_vertex_ids.npy")
        eye = np.eye(10475)
        r = self.J
        self.hand_reg = {
            "left": np.concatenate((r[[20, 37, 38, 39]], eye[5361, None], r[[25, 26, 27]], eye[4933, None],
                                    r[[28, 29, 30]], eye[5058, None], r[[34, 35, 36]], eye[5169, None],
                                    r[[31, 32, 33]], eye[5286, None])),
            "right": np.concatenate((r[[21, 52, 53, 54]], eye[8079, None], r[[40, 41, 42]], eye[7669, None],
                                     r[[43, 44, 45]], eye[7794, None], r[[49, 50, 51]], eye[7905, None],
                                     r[[46, 47, 48]], eye[8022, None])),
        }

    def evaluate(self, mesh_out: np.ndarray, mesh_gt: np.ndarray) -> dict:
        """Hand4Whole EHF.evaluate for one frame; both meshes (10475, 3) in metres, camera axes."""
        e = lambda a, b: np.sqrt(np.sum((a - b) ** 2, 1)).mean() * 1000  # noqa: E731
        J = self.J
        out = {}
        out["pa_mpvpe_all"] = e(rigid_align(mesh_out, mesh_gt), mesh_gt)
        al = mesh_out - np.dot(J, mesh_out)[J_IDX["pelvis"], None] + np.dot(J, mesh_gt)[J_IDX["pelvis"], None]
        out["mpvpe_all"] = e(al, mesh_gt)
        gl, ol = mesh_gt[self.hand_idx["left_hand"]], mesh_out[self.hand_idx["left_hand"]]
        gr, orr = mesh_gt[self.hand_idx["right_hand"]], mesh_out[self.hand_idx["right_hand"]]
        out["pa_mpvpe_hand"] = (e(rigid_align(ol, gl), gl) + e(rigid_align(orr, gr), gr)) / 2.0
        al = ol - np.dot(J, mesh_out)[J_IDX["lwrist"], None] + np.dot(J, mesh_gt)[J_IDX["lwrist"], None]
        ar = orr - np.dot(J, mesh_out)[J_IDX["rwrist"], None] + np.dot(J, mesh_gt)[J_IDX["rwrist"], None]
        out["mpvpe_hand"] = (e(al, gl) + e(ar, gr)) / 2.0
        gf, of = mesh_gt[self.face_idx], mesh_out[self.face_idx]
        out["pa_mpvpe_face"] = e(rigid_align(of, gf), gf)
        af = of - np.dot(J, mesh_out)[J_IDX["neck"], None] + np.dot(J, mesh_gt)[J_IDX["neck"], None]
        out["mpvpe_face"] = e(af, gf)
        jg, jo = np.dot(self.j14, mesh_gt), np.dot(self.j14, mesh_out)
        out["pa_mpjpe_body"] = e(rigid_align(jo, jg), jg)
        hl = (np.dot(self.hand_reg["left"], mesh_gt), np.dot(self.hand_reg["left"], mesh_out))
        hr = (np.dot(self.hand_reg["right"], mesh_gt), np.dot(self.hand_reg["right"], mesh_out))
        out["pa_mpjpe_hand"] = (e(rigid_align(hl[1], hl[0]), hl[0]) + e(rigid_align(hr[1], hr[0]), hr[0])) / 2.0
        return out


def load_frames():
    rot, _ = cv2.Rodrigues(CAM_R_AA)
    frames = []
    for k in range(1, 101):
        stem = f"{k:02d}"
        kp = json.loads((EHF_ROOT / f"{stem}_2Djnt.json").read_text())["people"][0]["pose_keypoints_2d"]
        frames.append({
            "id": stem,
            "image": str(EHF_ROOT / f"{stem}_img.jpg"),
            "keypoints": np.asarray(kp, dtype=np.float32).reshape(25, 3),
            "gt_world": read_ply_vertices(EHF_ROOT / f"{stem}_align.ply"),
            "R": rot,
        })
    return frames


def gt_camera_mesh(frame) -> np.ndarray:
    """Hand4Whole: rotation only (translation is removed by every alignment)."""
    return frame["gt_world"] @ frame["R"].T


def project_gt(frame) -> np.ndarray:
    cam = frame["gt_world"] @ frame["R"].T + CAM_T
    return np.stack([FOCAL[0] * cam[:, 0] / cam[:, 2] + PRINCPT[0], FOCAL[1] * cam[:, 1] / cam[:, 2] + PRINCPT[1]], -1)


def project_points(points_world, frame) -> np.ndarray:
    """EHF world points -> image px (as project_gt, for any points)."""
    cam = points_world @ frame["R"].T + CAM_T
    return np.stack([FOCAL[0] * cam[:, 0] / cam[:, 2] + PRINCPT[0], FOCAL[1] * cam[:, 1] / cam[:, 2] + PRINCPT[1]], -1)


def load_by_path_ehf(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def crop(frame):
    image = cv2.imread(frame["image"])
    h, w = image.shape[:2]
    box = ev.gt_keypoint_box(frame["keypoints"], w, h)
    patch = cv2.cvtColor(cv2.warpAffine(image, ev.box_to_affine(box), (256, 256), flags=cv2.INTER_LINEAR),
                         cv2.COLOR_BGR2RGB)
    return torch.from_numpy(patch).permute(2, 0, 1), box


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("pear", "student", "student_vitpose", "student_vitpose_mq", "student_vitpose_mq_handcrop", "gt", "sprint", "sprint_raw", "pear_arch"), required=True,
                   help="'gt' feeds the ground truth as the prediction (must give 0 everywhere)")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32",
                   help="model forward under autocast in this dtype (fp32: unchanged path); same as the 3DPW evaluator")
    p.add_argument("--refiner", type=Path, default=None,
                   help="optional no-render refiner checkpoint (train_render_refiner.py) applied once, as in "
                        "tools/eval_author_protocol.py; fp32 only. Without it the path is unchanged.")
    p.add_argument("--hand-crops", choices=("input", "fullres", "oracle"), default="fullres",
                   help="student_vitpose_mq_handcrop only: hand-crop source = the 256 px input, the full-resolution EHF "
                        "image around the predicted hands, or the full-resolution image around the TRUE hands (diagnostic)")
    p.add_argument("--head", choices=ev.HEADS, default="ehm",
                   help="evaluated mesh head: ehm (SMPL-X body + FLAME head, default) or smplx (plain SMPL-X 2020); PEAR: ehm")
    p.add_argument("--camera", action="store_true",
                   help="SPRINT models only: pass cond={'cam': ...} with the known EHF camera (EHF_camera.txt focal, real 1). "
                        "Ignored for other models. Default off (unchanged outputs).")
    p.add_argument("--camera-width-only", action="store_true", help="with --camera: focal = image width, real 0 (no EHF calibration)")
    args = p.parse_args()
    head = "gt" if args.model == "gt" else ev.effective_head(args.model, args.head)
    if args.refiner and args.precision != "fp32":
        raise SystemExit("--refiner is supported with --precision fp32 only")
    t0 = time.time()
    frames = load_frames()
    proto = Protocol()
    device = torch.device(args.device)
    preds = {}
    if args.model != "gt":
        checkpoint = (args.checkpoint or ev.DEFAULT_TEACHER).resolve()
        cwd = Path.cwd()
        os.chdir(ev.PEAR_ROOT)
        try:
            model, load_report = ev.load_model(args.model, checkpoint, args.student_config, 25, device)
            from models.modules.ehm import EHM_v2
            ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
            refiner = None
            if args.refiner:
                sys.path.insert(0, str(ev.PEAR_ROOT))
                from models.render_refiner import RenderRefiner
                from train_render_refiner import cam_from_t, refined_output
                import student_gt_losses as L
                st = torch.load(args.refiner, map_location="cpu", weights_only=False)
                assert not st["config"]["use_render"], "only the no-render refiner is supported here"
                refiner = RenderRefiner(use_render=False).to(device).eval()
                refiner.load_state_dict(st["refiner"], strict=True)
        finally:
            os.chdir(cwd)
        crops = [crop(f) for f in frames]
        images = torch.stack([c[0] for c in crops])
        cams = None
        if args.camera:   # image centre in the formula as in training (cam_features); focal = EHF fx
            def _cf(c, f):
                w_, h_ = cv2.imread(f["image"]).shape[1::-1]
                return ev.cam_feat(ev.box_to_affine(c[1]), w_, h_, *((w_, 0.0) if args.camera_width_only else (FOCAL[0], 1.0)))
            cams = np.stack([_cf(c, f) for c, f in zip(crops, frames)])
        handcrop = args.model == "student_vitpose_mq_handcrop"
        if handcrop:
            hc = load_by_path_ehf("guava_handcrop_eval_utils", ROOT / "tools" / "handcrop_eval_utils.py")
            P_ = Protocol()
            affs = [ev.box_to_affine(c[1]) for c in crops]
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
        with torch.inference_mode():
            for s in range(0, len(frames), 25):
                ck = ev.camera_kwargs(args.model, args.camera, None if cams is None else cams[s:s + 25], device)
                if refiner is not None:
                    out, feats = model(images[s:s + 25].to(device).float().div_(255.0), return_features=True, **ck)
                    out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                               if isinstance(v, dict) else v.float()) for k, v in out.items()}
                    rot1, cam1 = refiner(feats.float(), None, L.predicted_rotmats(out)[:, :22], cam_from_t(out["pd_cam"][:, :3, 3]))
                    out = refined_output(out, rot1, cam1)
                elif handcrop and dtype == torch.float32:
                    x = images[s:s + 25].to(device).float().div_(255.0)
                    kw = {}
                    if args.hand_crops != "input":
                        fr = frames[s:s + 25]
                        kw["hand_source"] = hc.hand_source([cv2.cvtColor(cv2.imread(f["image"]), cv2.COLOR_BGR2RGB) for f in fr], affs[s:s + 25], device)
                        if args.hand_crops == "oracle":
                            kp = np.stack([np.stack([project_points(P_.hand_reg[side] @ f["gt_world"], f) for side in ("left", "right")]) for f in fr])
                            kw["hand_boxes"] = hc.oracle_boxes(kp, affs[s:s + 25]).to(device)
                    out = model(x, **kw, **ck)
                elif dtype == torch.float32:
                    out = model(images[s:s + 25].to(device).float().div_(255.0), **ck)
                else:   # as in eval_3dpw_standard.py: autocast forward, outputs back to fp32
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        out = model(images[s:s + 25].to(device).float().div_(255.0), **ck)
                    out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                               if isinstance(v, dict) else v.float()) for k, v in out.items()}
                verts = ev.predicted_smplx_vertices(out, ehm, head).double().cpu().numpy()
                for j, v in enumerate(verts):
                    preds[frames[s + j]["id"]] = v
    rows = []
    for f in frames:
        gt = gt_camera_mesh(f)
        pred = gt.copy() if args.model == "gt" else preds[f["id"]]
        rows.append({"frame": f["id"], **proto.evaluate(pred, gt)})
    keys = [k for k in rows[0] if k != "frame"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_frame.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["frame", *keys]); w.writeheader(); w.writerows(rows)
    report = {
        "protocol": "Hand4Whole EHF.evaluate (c94908654b81); crop from EHF OpenPose BODY_25 via PEAR get_bbox x1.2 + process_bbox x1.25",
        "model": args.model, "head": head, "frames": len(rows), "metrics_mm": summary, "precision": args.precision,
        "camera": bool(args.camera and args.model in ev.SPRINT_KINDS),
        "refiner": str(args.refiner) if args.refiner else None,
        "hand_crops": args.hand_crops if args.model == "student_vitpose_mq_handcrop" else None,
        "refiner_sha256": ev.sha256(args.refiner) if args.refiner else None,
        "checkpoint": None if args.model == "gt" else str(checkpoint),
        "checkpoint_sha256": None if args.model == "gt" else ev.sha256(checkpoint),
        "assets_sha256": {n: ev.sha256(SMPLX_ASSETS / n) for n in
                          ("SMPLX_NEUTRAL_2020.npz", "SMPLX_to_J14.pkl", "MANO_SMPLX_vertex_ids.pkl", "SMPL-X__FLAME_vertex_ids.npy")},
        "git": {"guava": ev.git_state(ev.ROOT), "pear": ev.git_state(ev.PEAR_ROOT)},
        "evaluator_sha256": ev.sha256(Path(__file__).resolve()),
        "command": " ".join(sys.argv), "seconds": round(time.time() - t0, 1),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
