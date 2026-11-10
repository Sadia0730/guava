"""Teaser candidates (PREREG amendment 5, presentation only; the user picks the teaser example by eye from this list): for every
clip with appearance-update renders, the test frames turned away from the source (view change >= 135 deg) and, on those, the PSNR
of fb_gt (photo only) and fb_update100 (after the video) against the GT crop; plus texture coverage photo -> after update.
CPU only, reads existing renders / GT; writes a JSON ranked by the mean turned-away gain.
Usage: python teaser_candidates.py --out cands.json"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=135.0); a = ap.parse_args()


def psnr(x, y): return float(10 * np.log10(1.0 / max(np.mean((x - y) ** 2), 1e-12)))
def rd(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255 if p.exists() else None


def clip(root, cid):
    b = O / root / "base" / cid; fj = json.loads((b / "frames.json").read_text()); s = np.load(b / "smplx2020.npz")
    idx = {str(k): i for i, k in enumerate(s["keys"])}; R = Rot.from_rotvec(s["global_orient"]).as_matrix(); Rs = R[idx[fj["source"]]]
    view = {k: float(np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ Rs.T).as_rotvec()))) for k in fj["test"] if k in idx}
    back = [k for k, v in view.items() if v >= a.thr]
    run = json.loads((O / root / "renders" / "fb_update100" / cid / "run.json").read_text())
    fr = []
    for k in back:
        g = rd(O / root / "gt512" / cid / f"{k}.png"); p = rd(O / root / "renders" / "fb_gt" / cid / f"{k}.png"); u = rd(O / root / "renders" / "fb_update100" / cid / f"{k}.png")
        if g is None or p is None or u is None: continue
        fr.append({"key": k, "view": view[k], "photo": psnr(p, g), "update": psnr(u, g)})
    gains = [f["update"] - f["photo"] for f in fr]
    return {"clip": cid, "root": root, "n_back": len(fr), "gain_back_mean": float(np.mean(gains)) if gains else None,
            "gain_back_max": float(np.max(gains)) if gains else None, "best_frame": max(fr, key=lambda f: f["update"] - f["photo"]) if fr else None,
            "coverage_photo": run["coverage_source"], "coverage_after": run["coverage_fused"]}


jobs = []
for root in ("finalbench_ext", "publicbench_v1", "publicbench_v2"):
    for d in sorted((O / root / "renders" / "fb_update100").glob("*/*")):
        if (d / "run.json").exists() and (O / root / "renders" / "fb_gt" / d.parent.name / d.name).exists():
            jobs.append((root, f"{d.parent.name}/{d.name}"))
with ThreadPoolExecutor(16) as ex: res = [r for r in ex.map(lambda j: clip(*j), jobs) if r["n_back"]]
res.sort(key=lambda r: -r["gain_back_mean"])
Path(a.out).write_text(json.dumps(res, indent=1))
for r in res[:25]:
    print(f"{r['clip']:40s} back {r['n_back']:3d}  gain mean {r['gain_back_mean']:+.2f} max {r['gain_back_max']:+.2f}  "
          f"cover {r['coverage_photo'] * 100:.0f}->{r['coverage_after'] * 100:.0f}%  best {r['best_frame']['key']} ({r['best_frame']['view']:.0f} deg)")
