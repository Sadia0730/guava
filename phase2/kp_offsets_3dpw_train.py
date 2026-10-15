"""Phase 2, 3b cross-check: keypoint-definition offsets with TRUE meshes (no model error).

3DPW TRAIN split (never used for evaluation): GT SMPL meshes (gendered, 3DPW translation and camera extrinsics,
projected with the 3DPW intrinsics) -> J14 joints with the standard evaluator's H36M regressor (the same LSP-J14 style
definition as the hips / knees / ankles / shoulders / elbows / wrists of our 2D loss), compared with 3DPW's OpenPose
detections (poses2d, COCO-trained keypoint conventions). Offsets as in third_party/PEAR/tools/keypoint_definition_offsets.py:
detection - GT joint, in a body frame from the detected torso (down = shoulder mid -> hip mid; out = away from the
midline), in torso lengths; detections with confidence >= --min-conf only. The left/right assignment of the regressor's
joints is chosen per limb (ankle, knee, ... pair) for the whole dataset (the one that matches the detections better):
in the H36M regressor the leg and arm sides are labelled oppositely.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# J14 (via H36M_TO_J14) slots -> OpenPose-18 index, for one left/right assignment; the other swaps each pair
PAIRS = {"ankle": ((0, 5), (13, 10)), "knee": ((1, 4), (12, 9)), "hip": ((2, 3), (11, 8)),
         "wrist": ((11, 6), (7, 4)), "elbow": ((10, 7), (6, 3)), "shoulder": ((9, 8), (5, 2))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--every", type=int, default=5)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    ev = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ev; spec.loader.exec_module(ev)
    device = torch.device(args.device)
    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), "train")[::args.every]
    seqs = {}
    for path in sorted((ev.DEFAULT_DATASET.resolve() / "sequenceFiles" / "train").glob("*.pkl")):
        seqs[path.stem] = pickle.load(open(path, "rb"), encoding="latin1")
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female")}
    J, D = [], []                                                    # projected GT J14 (px), detections (18, 3)
    for i in range(0, len(samples), 256):
        bs = samples[i:i + 256]
        with torch.inference_mode():
            pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(device)
            betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(device)
            v = torch.empty(len(bs), 6890, 3, device=device)
            for g in ("male", "female"):
                sel = [k for k, s in enumerate(bs) if s.gender == g]
                if sel:
                    v[sel] = torch.einsum("bij,bvj->bvi", rot[sel], smpl[g](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices)
            off, K = [], []
            for s in bs:
                sq = seqs[s.sequence]
                E = np.asarray(sq["cam_poses"][s.frame], np.float64)
                off.append(E[:3, :3] @ np.asarray(sq["trans"][s.person][s.frame], np.float64) + E[:3, 3])
                K.append(np.asarray(sq["cam_intrinsics"], np.float64))
            j = torch.einsum("jv,bvc->bjc", h36m, v)[:, ev.H36M_TO_J14].cpu().numpy() + np.stack(off)[:, None]
            K = np.stack(K)
            px = np.einsum("bij,bkj->bki", K, j)
            J.append(px[..., :2] / px[..., 2:3])
        D.append(np.stack([s.keypoints for s in bs]))               # (B, 18, 3): x, y, conf
    J, D = np.concatenate(J), np.concatenate(D).astype(np.float64)
    conf = D[..., 2]
    det = D[..., :2]
    ok = (conf[:, [2, 5, 8, 11]] >= args.min_conf).all(1)
    sm, hm = (det[:, 2] + det[:, 5]) / 2, (det[:, 8] + det[:, 11]) / 2
    T = np.linalg.norm(hm - sm, axis=1)
    ok &= T > 20
    down = (hm - sm) / np.maximum(T, 1e-6)[:, None]
    perp = np.stack([-down[:, 1], down[:, 0]], 1)
    lr = (det[:, 5] - det[:, 2]) + (det[:, 11] - det[:, 8])                         # OpenPose: 5 l_sho, 2 r_sho, 11 l_hip, 8 r_hip
    perp *= np.sign((perp * lr).sum(1, keepdims=True) + 1e-9)
    # left/right assignment: compare the total distance of the two assignments over all pairs
    swaps, costs = {}, {}
    for name, ((a, b), (oa, ob)) in PAIRS.items():
        cost = [0.0, 0.0]
        for swap in (0, 1):
            ja, jb = (a, b) if not swap else (b, a)
            m = ok & (conf[:, oa] >= args.min_conf) & (conf[:, ob] >= args.min_conf)
            cost[swap] = float(np.linalg.norm(J[m][:, ja] - det[m][:, oa], axis=1).sum() + np.linalg.norm(J[m][:, jb] - det[m][:, ob], axis=1).sum())
        swaps[name], costs[name] = int(cost[1] < cost[0]), cost
    res = {"split": "train", "every": args.every, "samples": len(samples), "persons_used": int(ok.sum()),
           "lr_assignment_cost": costs, "swapped": swaps, "regressor": str(ev.DEFAULT_H36M_REGRESSOR), "offsets_T": {}}
    for name, ((a, b), (oa, ob)) in PAIRS.items():
        ja, jb = (a, b) if not swaps[name] else (b, a)
        for side, jj, oo, sgn in (("l", ja, oa, 1.0), ("r", jb, ob, -1.0)):
            m = ok & (conf[:, oo] >= args.min_conf)
            d = (det[m, oo] - J[m, jj]) / T[m, None]
            cd = (d * down[m]).sum(1); co = sgn * (d * perp[m]).sum(1)
            res["offsets_T"][f"{side}_{name}"] = {"n": int(m.sum()),
                "down_mean": float(cd.mean()), "down_ci95": float(1.96 * cd.std() / np.sqrt(len(cd))), "down_median": float(np.median(cd)),
                "out_mean": float(co.mean()), "out_ci95": float(1.96 * co.std() / np.sqrt(len(co))), "out_median": float(np.median(co))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["offsets_T"].items():
        print(f"{k:11s} n {v['n']:6d} down {v['down_mean']:+.3f}±{v['down_ci95']:.3f} (med {v['down_median']:+.3f})  out {v['out_mean']:+.3f}±{v['out_ci95']:.3f}")
    print("swapped", swaps)


if __name__ == "__main__":
    main()
