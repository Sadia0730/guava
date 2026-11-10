"""Teaser clip selection (PREREG "Extended benchmark, amendment 4 (a)"): per dataset, among clips whose test frames include a view
change >= 135 deg from the source (root orientation, smplx2020.npz; none -> all clips), the clip with the largest per-clip PSNR gain
fb_update100 - fb_gt (ties -> smaller clip id). Pools: every clip of the extended benchmark (existing roots: `_full` score files;
new clips: `--tag` files of finalbench_ext, or `_full` once they exist). Prints / writes JSON; reads scores only.
Usage: python teaser_select.py [--tag teaser] [--out sel.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="teaser"); ap.add_argument("--out", default=""); ap.add_argument("--thr", type=float, default=135.0)
a = ap.parse_args()
ROOTS = {"publicbench_v1": "full", "publicbench_v2": "full", "finalbench_ext": a.tag}


def scores(root, m, tag):
    f = O / root / "scores" / f"{m}_{tag}.json"
    if not f.exists() and tag != "full": f = O / root / "scores" / f"{m}_full.json"
    return json.loads(f.read_text())["per_clip"] if f.exists() else {}


def max_view(root, cid):
    b = O / root / "base" / cid; fj = json.loads((b / "frames.json").read_text()); s = np.load(b / "smplx2020.npz")
    idx = {str(k): i for i, k in enumerate(s["keys"])}; R = Rot.from_rotvec(s["global_orient"]).as_matrix(); Rs = R[idx[fj["source"]]]
    return max(float(np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ Rs.T).as_rotvec()))) for k in fj["test"] if k in idx)


cand = {}
for root, tag in ROOTS.items():
    g, u = scores(root, "fb_gt", tag), scores(root, "fb_update100", tag)
    for cid in sorted(set(g) & set(u)):
        ds = g[cid]["dataset"]
        cand.setdefault(ds, []).append({"clip": cid, "root": root, "gain": u[cid]["psnr"] - g[cid]["psnr"], "view": max_view(root, cid),
                                        "fb_gt": g[cid]["psnr"], "fb_update100": u[cid]["psnr"]})
sel = {}
for ds, cs in sorted(cand.items()):
    el = [c for c in cs if c["view"] >= a.thr] or cs
    best = sorted(el, key=lambda c: (-c["gain"], c["clip"]))[0]
    sel[ds] = {**best, "pool": len(cs), "eligible": len(el)}
    print(ds, json.dumps(sel[ds]))
if a.out: Path(a.out).write_text(json.dumps(sel, indent=1))
