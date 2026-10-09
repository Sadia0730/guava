#!/usr/bin/env python
"""Diagnostic: is wrist / ankle error related to where the joint falls in the crop?

For every 3DPW person-frame and model, the ground-truth SMPL joints (gendered model, with
translation) are projected with the 3DPW camera into the 256x256 ground-truth-keypoint crop
used by eval_3dpw_standard.py. Each wrist / ankle is classified as inside the 192-wide centre
the network sees, inside the 32-px side strips it discards, or outside the crop. Per-joint
error is the non-standard SMPL-kinematic-joint error of the evaluator's diagnostic table
(neutral SMPL regressor on both SMPL-topology meshes, pelvis-joint centred, no alignment).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_3dpw_standard as ev  # noqa: E402

JOINTS = {"left_wrist": 20, "right_wrist": 21, "left_ankle": 7, "right_ankle": 8}
OPENPOSE = {"left_wrist": 7, "right_wrist": 4, "left_ankle": 13, "right_ankle": 10}
CENTRE = (32.0, 224.0)


def region(x: float, y: float) -> str:
    if not (0.0 <= x < ev.IMAGE_SIZE and 0.0 <= y < ev.IMAGE_SIZE):
        return "outside_crop"
    return "centre" if CENTRE[0] <= x < CENTRE[1] else "side_strip"


def camera_tables(dataset_root: Path, split: str) -> dict:
    tables = {}
    for path in sorted((dataset_root / "sequenceFiles" / split).glob("*.pkl")):
        with path.open("rb") as handle:
            seq = pickle.load(handle, encoding="latin1")
        tables[path.stem] = {
            "K": np.asarray(seq["cam_intrinsics"], dtype=np.float64),
            "extrinsics": np.asarray(seq["cam_poses"], dtype=np.float64),
            "trans": [np.asarray(t, dtype=np.float32) for t in seq["trans"]],
        }
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), args.split)
    cams = camera_tables(ev.DEFAULT_DATASET.resolve(), args.split)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female", "neutral")}
    reg = smpl["neutral"].J_regressor.float()

    # Ground-truth geometry: crop-space joint positions and visible OpenPose points.
    loader = DataLoader(ev.CropDataset(samples, "gt_keypoints", None), batch_size=64, num_workers=12)
    gt_crop_xy, affines = {}, {}
    for batch in loader:
        for j, i in enumerate(batch["index"].tolist()):
            affines[i] = batch["affine"][j].numpy().astype(np.float64)
    geometry, kp_check = [], []
    for i, s in enumerate(samples):
        cam = cams[s.sequence]
        with torch.no_grad():
            joints = smpl[s.gender](
                global_orient=torch.from_numpy(s.pose[None, :3]).to(device),
                body_pose=torch.from_numpy(s.pose[None, 3:]).to(device),
                betas=torch.from_numpy(s.betas[None]).to(device),
                transl=torch.from_numpy(cam["trans"][s.person][s.frame][None]).to(device),
            ).joints[0, :24].double().cpu().numpy()
        ext = cam["extrinsics"][s.frame]
        cam_xyz = joints @ ext[:3, :3].T + ext[:3, 3]
        uv = cam_xyz @ cam["K"].T
        uv = uv[:, :2] / uv[:, 2:3]
        crop = np.c_[uv, np.ones(len(uv))] @ affines[i].T
        geometry.append(crop)
        for name, op in OPENPOSE.items():
            x, y, c = s.keypoints[op]
            if c > 0:
                kp_check.append(float(np.linalg.norm(uv[JOINTS[name]] - [x, y])))
    print(f"projection check: |projected SMPL joint - OpenPose keypoint| median {np.median(kp_check):.1f} px, "
          f"p90 {np.percentile(kp_check, 90):.1f} px over {len(kp_check)} visible wrists/ankles")

    errors = {}
    for kind in ("pear", "student"):
        cwd = Path.cwd()
        os.chdir(ev.PEAR_ROOT)
        try:
            ckpt = ev.DEFAULT_TEACHER if kind == "pear" else args.student_checkpoint.resolve()
            model, _ = ev.load_model(kind, ckpt, "configs/student_l70_v2.yaml", 64, device)
            from models.modules.ehm import EHM_v2
            ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
        finally:
            os.chdir(cwd)
        per_joint = np.zeros((len(samples), 24), dtype=np.float32)
        for batch in loader:
            idx = batch["index"].tolist()
            bs = [samples[i] for i in idx]
            with torch.inference_mode():
                out = model(batch["image"].to(device).float().div_(255.0))
                pred = torch.stack([torch.sparse.mm(mapping, v) for v in ev.predicted_smplx_vertices(out, ehm)])
                pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(device)
                betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(device)
                rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(device)
                gt = torch.empty_like(pred)
                for g in ("male", "female"):
                    sel = [k for k, s in enumerate(bs) if s.gender == g]
                    if sel:
                        v = smpl[g](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                        gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], v)
                pk = torch.einsum("jv,bvc->bjc", reg, pred)
                gk = torch.einsum("jv,bvc->bjc", reg, gt)
                err = ((pk - pk[:, :1]) - (gk - gk[:, :1])).norm(dim=-1) * 1000.0
            per_joint[idx] = err.cpu().numpy()
        errors[kind] = per_joint
        del model, ehm
        torch.cuda.empty_cache()
        print(f"{kind}: done", flush=True)

    rows = []
    for i, s in enumerate(samples):
        for name, j in JOINTS.items():
            x, y = geometry[i][j]
            rows.append({"sequence": s.sequence, "person": s.person, "frame": s.frame, "joint": name,
                         "crop_x": round(float(x), 2), "crop_y": round(float(y), 2), "region": region(x, y),
                         "pear_error_mm": float(errors["pear"][i, j]),
                         "student_error_mm": float(errors["student"][i, j])})
    with (args.output_dir / "per_joint_instance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"projection_check_px": {"median": float(np.median(kp_check)),
                                       "p90": float(np.percentile(kp_check, 90)), "n": len(kp_check)}}
    for part, names in (("wrists", ("left_wrist", "right_wrist")), ("ankles", ("left_ankle", "right_ankle"))):
        summary[part] = {}
        for reg_name in ("centre", "side_strip", "outside_crop"):
            sel = [r for r in rows if r["joint"] in names and r["region"] == reg_name]
            if not sel:
                summary[part][reg_name] = {"instances": 0}
                continue
            pe = np.array([r["pear_error_mm"] for r in sel])
            se = np.array([r["student_error_mm"] for r in sel])
            summary[part][reg_name] = {"instances": len(sel), "pear_mean_mm": float(pe.mean()),
                                       "student_mean_mm": float(se.mean()), "gap_mm": float((se - pe).mean()),
                                       "student_over_pear": float(se.mean() / pe.mean())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
