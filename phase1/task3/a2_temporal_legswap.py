"""Phase 1 Task 3, A2 (post-hoc follow-up, not pre-registered): leg swaps without an avatar.

A simple causal temporal rule on the student's own predictions. At every frame of a track (in time
order), two hypotheses:
  H0  legs as predicted;
  H1  legs swapped: left/right hip, knee, ankle and foot rotations exchanged and mirrored
      (R -> M R M, M = diag(-1, 1, 1), the same mirror as the training flip augmentation).
The rule keeps whichever hypothesis has its knees and ankles (pelvis-centred, camera frame) closer to
the legs of the previous frame's OUTPUT (the hypothesis chosen there). The first frame of a track
keeps H0. Variants: a margin m (switch to H1 only if it is closer by more than m mm).

Swap definition and metrics as in phase1/task3/swap_flip_rates.py and the standard evaluator
(tools/eval_3dpw_standard.py, imported unchanged): swap = exchanging the left/right knee+ankle
positions lowers their summed error vs GT by >= 50 mm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
L_LEG, R_LEG = [0, 3, 6, 9], [1, 4, 7, 10]       # body_pose indices: hip, knee, ankle, foot
LEGS = [4, 5, 7, 8]                               # SMPL kinematic joints: L knee, R knee, L ankle, R ankle
LEGS_SWAPPED = [1, 0, 3, 2]
METRICS = ["mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "diag_lower_body_mm"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("/raid/ubx858/outputs/phase1_task1/screen/c_vitpose/checkpoints/best.pt"))
    p.add_argument("--split", default="validation")
    p.add_argument("--crop", default="gt_keypoints")
    p.add_argument("--margins", type=float, nargs="+", default=[0.0, 50.0, 100.0])
    p.add_argument("--swap-threshold", type=float, default=50.0)
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
    cwd = Path.cwd(); os.chdir(PEAR_ROOT); sys.path.insert(0, str(PEAR_ROOT))
    model, _ = ev.load_model("student_vitpose", args.checkpoint, "configs/student_l70_v2.yaml", 1, device)
    from models.modules.ehm import EHM_v2
    ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    os.chdir(cwd)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female", "neutral")}
    reg = smpl["neutral"].J_regressor.float()
    M = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device))

    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), args.split)
    loader = DataLoader(ev.CropDataset(samples, args.crop, None), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True)
    rec = {k: [] for k in ("legs0", "legs1", "gt_legs")}
    met = {h: {m: [] for m in METRICS} for h in ("h0", "h1")}
    for batch in loader:
        bs = [samples[i] for i in batch["index"].tolist()]
        images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
        with torch.inference_mode():
            out = model(images)
            out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                   for k, v in out.items()}
            gt = None
            pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(device)
            betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(device)
            for h in ("h0", "h1"):
                o = out
                if h == "h1":
                    bp = dict(out["body_param"]); body = bp["body_pose"].clone()
                    body[:, L_LEG] = M @ out["body_param"]["body_pose"][:, R_LEG] @ M
                    body[:, R_LEG] = M @ out["body_param"]["body_pose"][:, L_LEG] @ M
                    bp["body_pose"] = body
                    o = dict(out); o["body_param"] = bp
                pred = torch.stack([torch.sparse.mm(mapping, v) for v in ev.predicted_smplx_vertices(o, ehm)])
                if gt is None:
                    gt = torch.empty_like(pred)
                    for g in ("male", "female"):
                        sel = [i for i, s in enumerate(bs) if s.gender == g]
                        if sel:
                            v = smpl[g](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                            gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], v)
                    gk = torch.einsum("jv,bvc->bjc", reg, gt); gk = gk - gk[:, :1]
                    rec["gt_legs"].append(gk[:, LEGS].cpu())
                m = ev.frame_metrics(pred, gt, h36m, reg)
                for k in METRICS:
                    met[h][k].append(m[k].cpu())
                pk = torch.einsum("jv,bvc->bjc", reg, pred); pk = pk - pk[:, :1]
                rec["legs0" if h == "h0" else "legs1"].append(pk[:, LEGS].cpu())
    legs0, legs1, gtl = (torch.cat(rec[k]).numpy() for k in ("legs0", "legs1", "gt_legs"))
    mets = {h: {k: torch.cat(v).numpy() for k, v in met[h].items()} for h in met}
    seqs = np.array([s.sequence for s in samples]); persons = np.array([s.person for s in samples])
    frames = np.array([s.frame for s in samples])

    def is_swap(legs):
        e0 = np.linalg.norm(legs - gtl, axis=-1).sum(1)
        e1 = np.linalg.norm(legs[:, LEGS_SWAPPED] - gtl, axis=-1).sum(1)
        return (e0 - e1) * 1000 >= args.swap_threshold

    swap_before = is_swap(legs0)
    tracks = defaultdict(list)
    for i, (s, pn) in enumerate(zip(seqs, persons)):
        tracks[(s, int(pn))].append(i)
    gaps = [int(np.max(np.diff(frames[sorted(ix, key=lambda j: frames[j])]))) for ix in tracks.values() if len(ix) > 1]

    res = {"split": args.split, "crop": args.crop, "frames": int(len(samples)), "tracks": len(tracks),
           "checkpoint": str(args.checkpoint), "evaluator_sha256": ev.sha256(ROOT / "tools" / "eval_3dpw_standard.py"),
           "swap_threshold_mm": args.swap_threshold, "max_frame_gap_in_a_track": int(max(gaps)),
           "swaps_before": int(swap_before.sum()), "rules": {}}
    swap_idx = np.nonzero(swap_before)[0]
    # oracle: per frame, the hypothesis with the lower lower-body error (upper bound of any H0/H1 selector)
    oracle = mets["h1"]["diag_lower_body_mm"] < mets["h0"]["diag_lower_body_mm"]
    for margin in args.margins:
        choose1 = np.zeros(len(samples), bool)
        for ix in tracks.values():
            ix = sorted(ix, key=lambda j: frames[j])
            prev = legs0[ix[0]]
            for a, j in enumerate(ix):
                if a > 0:
                    d0 = np.linalg.norm(legs0[j] - prev, axis=-1).sum() * 1000
                    d1 = np.linalg.norm(legs1[j] - prev, axis=-1).sum() * 1000
                    choose1[j] = d1 + margin < d0
                prev = legs1[j] if choose1[j] else legs0[j]
        res["rules"][f"margin_{margin:g}mm"] = summarise(choose1, legs0, legs1, mets, is_swap, swap_before, swap_idx, seqs)
    res["rules"]["oracle_lower_body"] = summarise(oracle, legs0, legs1, mets, is_swap, swap_before, swap_idx, seqs)
    res["seconds"] = time.time() - t0
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez(args.out / "per_frame.npz", legs0=legs0, legs1=legs1, gt_legs=gtl, sequence=seqs, person=persons, frame=frames,
             **{f"{h}_{k}": v for h in mets for k, v in mets[h].items()})
    print(json.dumps(res, indent=1))


def summarise(choose1, legs0, legs1, mets, is_swap, swap_before, swap_idx, seqs):
    legs = np.where(choose1[:, None, None], legs1, legs0)
    swap_after = is_swap(legs)
    out = {"frames_switched_to_swapped": int(choose1.sum()),
           "swaps_after": int(swap_after.sum()),
           "swaps_fixed": int((swap_before & ~swap_after).sum()),
           "swaps_created": int((~swap_before & swap_after).sum())}
    for k in METRICS:
        after = np.where(choose1, mets["h1"][k], mets["h0"][k])
        out[k] = {"all_before": float(mets["h0"][k].mean()), "all_after": float(after.mean()),
                  "swap_frames_before": float(mets["h0"][k][swap_idx].mean()), "swap_frames_after": float(after[swap_idx].mean())}
        if choose1.any():
            out[k]["switched_frames_before"] = float(mets["h0"][k][choose1].mean())
            out[k]["switched_frames_after"] = float(mets["h1"][k][choose1].mean())
    out["per_sequence"] = {s: {"swaps_before": int(swap_before[seqs == s].sum()), "swaps_after": int(swap_after[seqs == s].sum()),
                               "switched": int(choose1[seqs == s].sum())}
                           for s in sorted(set(seqs)) if swap_before[seqs == s].any() or swap_after[seqs == s].any() or choose1[seqs == s].any()}
    return out


if __name__ == "__main__":
    main()
