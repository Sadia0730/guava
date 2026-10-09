#!/usr/bin/env python
"""Compare two SMPL-X -> SMPL conversions on real predictions (diagnostic).

A: fixed vertex correspondence (each SMPL vertex = barycentric blend of SMPL-X vertices).
B: per-frame fit of a neutral SMPL model (pose, 10 betas, translation) to the A vertices,
   i.e. A projected onto the SMPL body-model manifold.
Reports the fit residual, the J14 shift between A and B, the resulting 3DPW metric change,
and the time each conversion takes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_3dpw_standard as ev  # noqa: E402


def fit_smpl(smpl, targets, init_rotmats, iters: int):
    from pytorch3d.transforms import matrix_to_axis_angle
    batch = targets.shape[0]
    aa = matrix_to_axis_angle(init_rotmats)                 # (B, 22, 3): global + 21 body
    pose = torch.zeros(batch, 24, 3, device=targets.device)
    pose[:, :22] = aa
    pose = pose.reshape(batch, 72).clone().requires_grad_(True)
    betas = torch.zeros(batch, 10, device=targets.device, requires_grad=True)
    transl = torch.zeros(batch, 3, device=targets.device, requires_grad=True)
    with torch.no_grad():
        start = smpl(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=betas).vertices
        transl += (targets - start).mean(1)
    optim = torch.optim.Adam([pose, betas, transl], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, iters)
    for _ in range(iters):
        optim.zero_grad(set_to_none=True)
        verts = smpl(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=betas, transl=transl).vertices
        loss = (verts - targets).norm(dim=-1).mean() + 1e-4 * betas.square().mean()
        loss.backward()
        optim.step()
        sched.step()
    with torch.no_grad():
        return smpl(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=betas, transl=transl).vertices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), "test")[:: args.stride]
    cwd = Path.cwd()
    os.chdir(ev.PEAR_ROOT)
    try:
        model, _ = ev.load_model("pear", ev.DEFAULT_TEACHER, "", 64, device)
        from models.modules.ehm import EHM_v2
        ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    finally:
        os.chdir(cwd)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female", "neutral")}

    loader = DataLoader(ev.CropDataset(samples, "gt_keypoints", None), batch_size=64, num_workers=12)
    pred_x, rotmats, gts = [], [], []
    for batch in loader:
        bs = [samples[i] for i in batch["index"].tolist()]
        with torch.inference_mode():
            out = model(batch["image"].to(device).float().div_(255.0))
            pred_x.append(ev.predicted_smplx_vertices(out, ehm))
            bp = out["body_param"]
            rotmats.append(torch.cat([bp["global_pose"].reshape(-1, 1, 3, 3), bp["body_pose"]], 1))
            pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(device)
            betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(device)
            gt = torch.empty(len(bs), 6890, 3, device=device)
            for g in ("male", "female"):
                sel = [i for i, s in enumerate(bs) if s.gender == g]
                if sel:
                    v = smpl[g](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                    gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], v)
            gts.append(gt)
    pred_x, rotmats, gt = torch.cat(pred_x), torch.cat(rotmats), torch.cat(gts)

    torch.cuda.synchronize(); t0 = time.time()
    mapped = torch.stack([torch.sparse.mm(mapping, v) for v in pred_x])
    torch.cuda.synchronize(); t_map = time.time() - t0
    t0 = time.time()
    fitted = fit_smpl(smpl["neutral"], mapped.detach(), rotmats.detach(), args.iters)
    torch.cuda.synchronize(); t_fit = time.time() - t0

    reg = smpl["neutral"].J_regressor.float()
    m_a = ev.frame_metrics(mapped, gt, h36m, reg)
    m_b = ev.frame_metrics(fitted, gt, h36m, reg)
    j_a = torch.einsum("jv,bvc->bjc", h36m, mapped)[:, ev.H36M_TO_J14]
    j_b = torch.einsum("jv,bvc->bjc", h36m, fitted)[:, ev.H36M_TO_J14]
    j_a, j_b = j_a - ev.hip_centre(j_a)[:, None], j_b - ev.hip_centre(j_b)[:, None]
    residual = (fitted - mapped).norm(dim=-1).mean(1) * 1000
    shift = (j_a - j_b).norm(dim=-1).mean(1) * 1000
    n = len(samples)
    report = {
        "frames": n,
        "fit_iters": args.iters,
        "fit_vertex_residual_mm": {"mean": float(residual.mean()), "p95": float(residual.quantile(0.95))},
        "j14_shift_A_vs_B_mm": {"mean": float(shift.mean()), "p95": float(shift.quantile(0.95))},
        "metrics_A_fixed_mapping": {k: float(v.mean()) for k, v in m_a.items()},
        "metrics_B_per_frame_fit": {k: float(v.mean()) for k, v in m_b.items()},
        "seconds_per_frame": {"A_fixed_mapping": t_map / n, "B_per_frame_fit": t_fit / n},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
