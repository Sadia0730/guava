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


def crop(frame):
    image = cv2.imread(frame["image"])
    h, w = image.shape[:2]
    box = ev.gt_keypoint_box(frame["keypoints"], w, h)
    patch = cv2.cvtColor(cv2.warpAffine(image, ev.box_to_affine(box), (256, 256), flags=cv2.INTER_LINEAR),
                         cv2.COLOR_BGR2RGB)
    return torch.from_numpy(patch).permute(2, 0, 1), box


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("pear", "student", "gt"), required=True,
                   help="'gt' feeds the ground truth as the prediction (must give 0 everywhere)")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
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
        finally:
            os.chdir(cwd)
        images = torch.stack([crop(f)[0] for f in frames])
        with torch.inference_mode():
            for s in range(0, len(frames), 25):
                out = model(images[s:s + 25].to(device).float().div_(255.0))
                verts = ev.predicted_smplx_vertices(out, ehm).double().cpu().numpy()
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
        "model": args.model, "frames": len(rows), "metrics_mm": summary,
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
