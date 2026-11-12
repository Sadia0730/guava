#!/usr/bin/env python
"""AGORA metrics for predictions in the official format (tools/eval_agora.py): our own implementation of the official protocol,
written from the source of github.com/pixelite1201/agora_evaluation @ 8c9dece (phase3_dnt/PREREG.md, "Pose table ... AGORA",
amendment 1: the official checkout is not executed here). Rules reproduced, with the official file they come from:
  matching.match_2d_greedy   pairs ranked by np.linalg.norm(pred[:24] - gt[:24], 2) (the matrix 2-norm, as in utils.l2_error);
                             greedy; a pair needs box IoU >= 0.1, boxes = min / max of the prediction's 24 joints and of ALL the
                             ground truth's 2D joints (get_bbx_overlap, +1 pixel convention); valid ground truth = isValid and
                             0 <= occlusion <= 90; predictions matched to an invalid person are dropped, not counted
  calculate_v2v_error        body: 22 joints / body_verts_smplx.npy, aligned at the hip midpoint (joints 1, 2); hands: wrist + 15
                             joints / MANO_SMPLX_vertex_ids, aligned at the wrist; face: neck + 51 landmarks /
                             SMPL-X__FLAME_vertex_ids, aligned at the neck; mean Euclidean distance, mm
  compute_average_error      counted persons = isValid, images with "bfh" or "smplx" in the name (--onlybfh); miss = valid person
                             without a match (all of them in an image without predictions); body MPJPE / MVE = mean over matched
                             valid persons; precision / recall / F1 from TP, FP, FN, rounded to 2 decimals; N-MVE = MVE / F1;
                             full body = body + (left hand + right hand + face) / 3. The official hand / face means include the
                             missed persons as zeros; both that value ("official") and the mean over matched persons are written.
The only departure: where the official inner loop would never end (all pair errors used up) this code stops the matching of that
image and counts the event (`matching_guard`, expected 0).
--self-test: the ground truth of every valid person is fed back as the prediction; every error must be 0 and F1 1.
Usage: python tools/score_agora.py --pred-dir DIR --gt-df DIR --indices DIR --output-dir DIR [--self-test]"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

GUARD = 0


def bbx_overlap(p1, p2):
    a0, a1, b0, b1 = p1.min(0), p1.max(0), p2.min(0), p2.max(0)
    xl, yt, xr, yb = max(a0[0], b0[0]), max(a0[1], b0[1]), min(a1[0], b1[0]), min(a1[1], b1[1])
    inter = max(0, xr - xl + 1) * max(0, yb - yt + 1)
    ar1 = (a1[0] - a0[0] + 1) * (a1[1] - a0[1] + 1); ar2 = (b1[0] - b0[0] + 1) * (b1[1] - b0[1] + 1)
    return inter / float(ar1 + ar2 - inter)


def match_2d_greedy(pred_kps, gtkp, valid, iou_thresh=0.1):
    """official matching.match_2d_greedy -> list of (pred, gt) | (pred, 'falsePositive') | ('miss', gt) | ('invalid', gt)."""
    global GUARD
    P, G = len(pred_kps), len(gtkp)
    combs = list(product(range(P), range(G)))
    err = np.array([np.linalg.norm(pred_kps[p][:24, :2] - gtkp[g][:24, :2], 2) for p, g in combs], dtype=np.float64)
    gt_a, op_a, best, excl, fpc, stop = np.zeros(G, bool), np.zeros(P, bool), [], [], 0, False
    while gt_a.sum() < G and op_a.sum() + fpc < P and not stop:
        found = fp = False
        while not found:
            exhausted = bool(np.all(np.isinf(err)))
            mi = int(np.argmin(err)); p, g = combs[mi]
            iou = bbx_overlap(pred_kps[p][:, :2], gtkp[g][:, :2])
            if not op_a[p] and not gt_a[g] and iou >= iou_thresh:
                found = True; err[mi] = np.inf
            else:
                err[mi] = np.inf
                if iou < iou_thresh:
                    found = fp = True; fpc += 1
                elif exhausted:                       # the official loop would spin here for ever
                    GUARD += 1; stop = True; break
        if stop:
            break
        if valid[g]:
            if not fp:
                best.append((p, g)); op_a[p] = True; gt_a[g] = True
        else:
            gt_a[g] = True; excl.append(g)
    op_ids, gt_ids = sorted(m[0] for m in best), sorted(m[1] for m in best)
    out = list(best) + [(p, "falsePositive") for p in np.setdiff1d(np.arange(P), op_ids)]
    for g in np.setdiff1d(np.arange(G), gt_ids):
        if valid[g]: out.append(("miss", int(g)))
        else: excl.append(int(g))
    return out + [("invalid", g) for g in excl]


def matching_dict(matching):
    d, fp = {}, 0
    for m in matching:
        if not (m[1] == "falsePositive") or m[0] == "invalid": d[int(m[1])] = m[0]
        elif m[1] == "falsePositive": fp += 1
    return d, fp


def part_error(gj, pj, gv, pv, root):
    """mean joint / vertex distance (m) after subtracting each side's own root joint."""
    gr = (gj[1] + gj[2]) / 2.0 if root == "pelvis" else gj[0]
    pr = (pj[1] + pj[2]) / 2.0 if root == "pelvis" else pj[0]
    return (float(np.sqrt((((gj - gr) - (pj - pr)) ** 2).sum(1)).mean()), float(np.sqrt((((gv - gr) - (pv - pr)) ** 2).sum(1)).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path); ap.add_argument("--gt-df", type=Path, required=True)
    ap.add_argument("--indices", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--self-test", action="store_true"); ap.add_argument("--name", default="SPRINT")
    a = ap.parse_args(); t0 = time.time(); a.output_dir.mkdir(parents=True, exist_ok=True)
    body = np.load(a.indices / "body_verts_smplx.npy"); face = np.load(a.indices / "SMPL-X__FLAME_vertex_ids.npy")
    hands = pickle.load(open(a.indices / "MANO_SMPLX_vertex_ids.pkl", "rb"), encoding="latin1")
    lh, rh = np.asarray(hands["left_hand"]), np.asarray(hands["right_hand"])
    preds = defaultdict(list)
    if not a.self_test:
        for f in a.pred_dir.glob("*.pkl"): preds["_".join(f.name.split("_")[0:-2]) + ".png"].append(str(f))
    rows, n_img, n_img_pred, fp_total, tot, miss_tot = [], 0, 0, 0, 0, 0
    for dfp in sorted(a.gt_df.glob("*.pkl")):
        df = pd.read_pickle(dfp)
        for _, r in df.iterrows():
            img = r["imgPath"]
            if "bfh" not in img and "smplx" not in img: continue
            n_img += 1
            is_valid = np.asarray(r["isValid"], bool); occ = np.asarray(r["occlusion"], np.float64)
            valid = (occ <= 90) & (occ >= 0) & is_valid
            g2 = [np.asarray(x, np.float64) for x in r["gt_joints_2d"]]; g3 = [np.asarray(x, np.float64) for x in r["gt_joints_3d"]]
            gv = [np.asarray(x, np.float64) for x in r["gt_verts"]]
            if a.self_test:
                P = [{"joints": g2[j][:24], "allSmplJoints3d": g3[j], "verts": gv[j]} for j in range(len(g2)) if valid[j]]
            else:
                P = [pickle.load(open(f, "rb"), encoding="latin1") for f in sorted(preds.get(img, []))]
            if not P:
                for j in np.nonzero(is_valid)[0]:
                    tot += 1; miss_tot += 1; rows.append({"image": img, "gt": int(j), "occlusion": float(occ[j]), "kid": bool(r["kid"][j]), "miss": 1})
                continue
            n_img_pred += 1
            md, fp = matching_dict(match_2d_greedy([np.asarray(q["joints"], np.float64).squeeze() for q in P], g2, valid)); fp_total += fp
            for j in np.nonzero(is_valid)[0]:
                tot += 1; row = {"image": img, "gt": int(j), "occlusion": float(occ[j]), "kid": bool(r["kid"][j])}
                m = md.get(int(j), "miss")
                if m in ("miss", "invalid"):
                    miss_tot += 1; row["miss"] = 1; rows.append(row); continue
                pj = np.asarray(P[m]["allSmplJoints3d"], np.float64).squeeze(); pv = np.asarray(P[m]["verts"], np.float64).squeeze(); gj = g3[j]
                row["miss"] = 0
                row["body-MPJPE"], row["body-MVE"] = part_error(gj[:22], pj[:22], gv[j][body], pv[body], "pelvis")
                for key, wr, sl, vi in (("lhand", 20, slice(25, 40), lh), ("rhand", 21, slice(40, 55), rh), ("face", 12, slice(76, 127), face)):
                    row[f"{key}-MPJPE"], row[f"{key}-MVE"] = part_error(np.concatenate([gj[wr][None], gj[sl]]), np.concatenate([pj[wr][None], pj[sl]]), gv[j][vi], pv[vi], "first")
                rows.append(row)
    keys = [f"{p}-{m}" for p in ("body", "lhand", "rhand", "face") for m in ("MPJPE", "MVE")]
    hit = [r for r in rows if not r["miss"]]; tp, fn = tot - miss_tot, miss_tot
    res = {"name": "ground truth as prediction (self-test)" if a.self_test else a.name, "images_scored": n_img, "images_with_predictions": n_img_pred,
           "valid_persons": tot, "matched": tp, "missed": fn, "false_positives": fp_total, "matching_guard": GUARD,
           "precision": tp / max(tp + fp_total, 1), "recall": tp / max(tp + fn, 1), "f1": tp / max(tp + 0.5 * (fp_total + fn), 1e-9)}
    f1r = round(res["f1"], 2)
    for k in keys:
        res[k] = 1000 * float(np.mean([r[k] for r in hit])) if hit else float("nan")                 # matched persons only
        if not k.startswith("body"):                                                                 # official: misses count as 0
            res[k + "_official"] = 1000 * float(np.sum([r[k] for r in hit])) / max(tot, 1)
    for m in ("MPJPE", "MVE"):
        res[f"fullbody-{m}"] = res[f"body-{m}"] + (res[f"lhand-{m}"] + res[f"rhand-{m}"] + res[f"face-{m}"]) / 3
        res[f"fullbody-{m}_official"] = round(res[f"body-{m}"], 1) + (round(res[f"lhand-{m}_official"], 1) + round(res[f"rhand-{m}_official"], 1) + round(res[f"face-{m}_official"], 1)) / 3
        res[f"body-N{'MJE' if m == 'MPJPE' else 'MVE'}_official"] = round(res[f"body-{m}"], 1) / f1r if f1r else float("nan")
        res[f"fullbody-N{'MJE' if m == 'MPJPE' else 'MVE'}_official"] = round(res[f"fullbody-{m}_official"], 1) / f1r if f1r else float("nan")
    res.update({"f1_rounded": f1r, "command": " ".join(sys.argv), "seconds": round(time.time() - t0, 1),
                "protocol": "own implementation of pixelite1201/agora_evaluation @ 8c9dece (see module docstring); mm"})
    (a.output_dir / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    with (a.output_dir / "per_person.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "gt", "occlusion", "kid", "miss", *keys]); w.writeheader()
        for r in rows: w.writerow({k: (round(1000 * v, 3) if k in keys else v) for k, v in r.items()})
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ("command", "protocol")}, indent=1))


if __name__ == "__main__":
    main()
