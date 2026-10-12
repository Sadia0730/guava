"""Phase 1 Task 3: leg-swap and front/back-flip rates of a student on a 3DPW split (read-only).

Same data path, crop, model loading, GT meshes and SMPLX->SMPL mapping as tools/eval_3dpw_standard.py
(imported from it, unchanged). Per frame, on pelvis-centred SMPL kinematic joints (neutral SMPL
regressor on both meshes, camera frame, no rotation alignment), as defined in
phase1/task3/DESIGN.md section 4:
  leg swap:  exchanging the predicted left/right knee + ankle lowers their summed error vs GT by
             >= T mm (T = 50, sensitivity 25 and 100)
  flip:      angle between predicted and GT facing vectors, normalise((L_hip - R_hip) x (neck - pelvis)),
             > A degrees (A = 120, sensitivity 90 and 150)
Rates per 1,000 frames with a sequence-level bootstrap 95% CI (the evaluator's bootstrap_ci).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PEAR_ROOT = ROOT / "third_party" / "PEAR"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="student_vitpose")
    p.add_argument("--checkpoint", type=Path, default=Path("/raid/ubx858/outputs/phase1_task1/screen/c_vitpose/checkpoints/best.pt"))
    p.add_argument("--config", default="configs/student_l70_v2.yaml")
    p.add_argument("--split", default="validation")
    p.add_argument("--crop", default="gt_keypoints")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    spec = importlib.util.spec_from_file_location("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    ev = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ev; spec.loader.exec_module(ev)
    cwd = Path.cwd()
    os.chdir(PEAR_ROOT)
    sys.path.insert(0, str(PEAR_ROOT))
    model, _ = ev.load_model(args.model, args.checkpoint, args.config, 1, device)
    from models.modules.ehm import EHM_v2
    ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    os.chdir(cwd)

    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female", "neutral")}
    reg = smpl["neutral"].J_regressor.float()
    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), args.split)
    loader = DataLoader(ev.CropDataset(samples, args.crop, None), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True)
    swap_gain, facing, seqs = [], [], []
    for batch in loader:
        bs = [samples[i] for i in batch["index"].tolist()]
        images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
        with torch.inference_mode():
            out = model(images)
            out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                   for k, v in out.items()}
            pred = torch.stack([torch.sparse.mm(mapping, v) for v in ev.predicted_smplx_vertices(out, ehm)])
            gt = torch.empty_like(pred)
            pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(device)
            betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(device)
            for g in ("male", "female"):
                sel = [i for i, s in enumerate(bs) if s.gender == g]
                if sel:
                    v = smpl[g](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                    gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], v)
            pk = torch.einsum("jv,bvc->bjc", reg, pred); gk = torch.einsum("jv,bvc->bjc", reg, gt)
            pk, gk = pk - pk[:, :1], gk - gk[:, :1]
            legs, legs_sw = [4, 5, 7, 8], [5, 4, 8, 7]
            e0 = (pk[:, legs] - gk[:, legs]).norm(dim=-1).sum(1)
            e1 = (pk[:, legs_sw] - gk[:, legs]).norm(dim=-1).sum(1)
            swap_gain.append(((e0 - e1) * 1000).cpu())
            def fvec(k):
                return F.normalize(torch.cross(k[:, 1] - k[:, 2], k[:, 12] - k[:, 0], dim=1), dim=1)
            facing.append(torch.rad2deg(torch.arccos((fvec(pk) * fvec(gk)).sum(1).clamp(-1, 1))).cpu())
        seqs += [s.sequence for s in bs]
    swap_gain = torch.cat(swap_gain).numpy(); facing = torch.cat(facing).numpy(); seqs = np.array(seqs)
    res = {"model": args.model, "checkpoint": str(args.checkpoint), "split": args.split, "crop": args.crop,
           "frames": int(len(seqs)), "sequences": int(len(set(seqs))), "evaluator_sha256": ev.sha256(ROOT / "tools" / "eval_3dpw_standard.py"),
           "seconds": time.time() - t0}
    for T in (25, 50, 100):
        x = (swap_gain >= T).astype(float) * 1000
        res[f"leg_swap_per_1000_T{T}mm"] = {"rate": float(x.mean()), "ci95": ev.bootstrap_ci(x, seqs), "frames": int((swap_gain >= T).sum())}
    for A in (90, 120, 150):
        x = (facing > A).astype(float) * 1000
        res[f"flip_per_1000_A{A}deg"] = {"rate": float(x.mean()), "ci95": ev.bootstrap_ci(x, seqs), "frames": int((facing > A).sum())}
    res["per_sequence_leg_swap_T50"] = {s: int(((swap_gain >= 50) & (seqs == s)).sum()) for s in sorted(set(seqs))}
    res["per_sequence_flip_A120"] = {s: int(((facing > 120) & (seqs == s)).sum()) for s in sorted(set(seqs))}
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez(args.out / "per_frame.npz", swap_gain_mm=swap_gain, facing_deg=facing, sequence=seqs)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
