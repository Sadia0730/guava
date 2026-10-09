#!/usr/bin/env python
"""Checkpoint 2: recover the geometric conventions of the BEDLAM2 (CameraHMR) labels.

Read-only. Builds the locked-head SMPL-X mesh from the labels under several hypotheses for
the translation and hand-pose convention, projects it with `cam_int`, and nearest-neighbour
matches the result against the stored `proj_verts` (437 points) and `gtkps` (171 points).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

LABELS = Path("/raid/ubx858/datasets/bedlam2_gt/expanded_labels/bedlam2_labels_processed")
LOCKED = Path("/home/ubx858/guava/assets/SMPLX/lockedhead/models_lockedhead/smplx/SMPLX_NEUTRAL.npz")


def build(flat_hand_mean: bool, num_betas: int = 16):
    import smplx
    return smplx.SMPLX(model_path=str(LOCKED), ext="npz", gender="neutral", num_betas=num_betas,
                       use_pca=False, flat_hand_mean=flat_hand_mean, num_expression_coeffs=10,
                       use_face_contour=True).eval()


def forward(model, pose, betas, transl):
    pose = torch.as_tensor(pose, dtype=torch.float32)
    return model(global_orient=pose[:, :3], body_pose=pose[:, 3:66], jaw_pose=pose[:, 66:69],
                 leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75], left_hand_pose=pose[:, 75:120],
                 right_hand_pose=pose[:, 120:165], betas=torch.as_tensor(betas, dtype=torch.float32),
                 transl=torch.as_tensor(transl, dtype=torch.float32),
                 expression=torch.zeros(len(pose), 10))


def project(points, K):
    uv = np.einsum("bij,bnj->bni", K, points)
    return uv[..., :2] / uv[..., 2:3]


def nn_match(pred_2d, target_2d):
    """For every target point, nearest predicted point: distance and index."""
    d = np.linalg.norm(pred_2d[:, None, :] - target_2d[None, :, :], axis=-1)
    return d.min(0), d.argmin(0)


def main():
    rng = np.random.default_rng(0)
    files = sorted(LABELS.glob("*.npz"))
    picks = []
    for path in rng.choice(files, 6, replace=False):
        d = np.load(path, allow_pickle=True)
        for i in rng.choice(len(d["imgname"]), 5, replace=False):
            picks.append({k: d[k][i] for k in ("pose_cam", "pose_world", "shape", "trans_cam", "trans_world",
                                                 "cam_int", "cam_ext", "gtkps", "proj_verts", "imgname")} | {"job": path.stem})
    pose = np.stack([p["pose_cam"] for p in picks]); posew = np.stack([p["pose_world"] for p in picks])
    betas = np.stack([p["shape"] for p in picks]); K = np.stack([p["cam_int"] for p in picks])
    ext = np.stack([p["cam_ext"] for p in picks]); tc = np.stack([p["trans_cam"] for p in picks])
    tw = np.stack([p["trans_world"] for p in picks])
    pv = np.stack([p["proj_verts"][:, :2] for p in picks]); kp = np.stack([p["gtkps"][:, :2] for p in picks])

    results = {}
    for flat in (True, False):
        model = build(flat)
        with torch.no_grad():
            cam = forward(model, pose, betas, np.zeros_like(tc))
            world = forward(model, posew, betas, tw)
        v0, j0 = cam.vertices.numpy(), cam.joints.numpy()
        hyp = {
            "H1: M(pose_cam) + trans_cam": (v0 + tc[:, None], j0 + tc[:, None]),
            "H2: M(pose_cam) + trans_cam + ext_t": (v0 + (tc + ext[:, :3, 3])[:, None], j0 + (tc + ext[:, :3, 3])[:, None]),
            "H3: ext @ (M(pose_world) + trans_world)": (
                np.einsum("bij,bnj->bni", ext[:, :3, :3], world.vertices.numpy()) + ext[:, None, :3, 3],
                np.einsum("bij,bnj->bni", ext[:, :3, :3], world.joints.numpy()) + ext[:, None, :3, 3]),
        }
        for name, (verts, joints) in hyp.items():
            vd = [nn_match(project(verts[b:b + 1], K[b:b + 1])[0], pv[b]) for b in range(len(picks))]
            kd = [nn_match(project(joints[b:b + 1], K[b:b + 1])[0], kp[b]) for b in range(len(picks))]
            key = f"flat_hand_mean={flat} | {name}"
            results[key] = {
                "proj_verts_nn_px_median": float(np.median(np.concatenate([x[0] for x in vd]))),
                "proj_verts_nn_px_max": float(np.max(np.concatenate([x[0] for x in vd]))),
                "proj_verts_index_consistent": bool(all(np.array_equal(vd[0][1], x[1]) for x in vd)),
                "gtkps_vs_joints_nn_px_median": float(np.median(np.concatenate([x[0] for x in kd]))),
                "gtkps_vs_joints_nn_px_p95": float(np.percentile(np.concatenate([x[0] for x in kd]), 95)),
                "joint_count": int(joints.shape[1]),
            }
            if name.startswith("H2") or name.startswith("H1") or name.startswith("H3"):
                results[key]["proj_verts_indices_first10"] = vd[0][1][:10].tolist()
                results[key]["gtkps_joint_indices"] = kd[0][1].tolist()
    print(json.dumps(results, indent=1))
    out = Path("/raid/ubx858/outputs/phase1_task1/checkpoint2/geometry_conventions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
