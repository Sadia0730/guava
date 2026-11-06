"""Per-frame PSNR of arbitrary rows vs the source->test view change (diag_view_gap.py's angle: geodesic root-orientation change
in the crop camera, base/<ds>/<clip>/smplx2020.npz), binned; any E / split / method list (CPU).
Usage: python view_bins.py --E <root> --split <json> --methods fb_gt,gart,diag_fuse_reg [--out table.md] [--clips id,..]"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--methods", required=True); ap.add_argument("--out", default=""); ap.add_argument("--clips", default="")
ap.add_argument("--bins", default="0,30,60,90,135,181"); a = ap.parse_args()
E = Path(a.E); M = a.methods.split(","); BINS = [int(x) for x in a.bins.split(",")]
clips = json.loads(Path(a.split).read_text())["clips"]
if a.clips: clips = [c for c in clips if c["id"] in a.clips.split(",")]


def psnr(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2) / 255.0 ** 2; return 10 * np.log10(1.0 / max(mse, 1e-12))


def clip_rows(c):
    ds, clip = c["dataset"], c["clip"]; base = E / "base" / ds / clip; fj = json.loads((base / "frames.json").read_text())
    s = np.load(base / "smplx2020.npz"); idx = {str(k): i for i, k in enumerate(s["keys"])}
    R = Rot.from_rotvec(s["global_orient"]).as_matrix(); Rs = R[idx[fj["source"]]]; rows = []
    for k in fj["test"]:
        g = cv2.imread(str(E / "gt512" / ds / clip / f"{k}.png")); r = {"ds": ds, "clip": clip, "key": k,
                                                                        "view": float(np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ Rs.T).as_rotvec())))}
        for m in M:
            p = E / "renders" / m / ds / clip / f"{k}.png"
            if p.exists(): r[m] = psnr(cv2.imread(str(p)), g)
        rows.append(r)
    return rows


with ThreadPoolExecutor(8) as ex: rows = [r for rs in ex.map(clip_rows, clips) for r in rs]
lines = []
for ds in sorted({r["ds"] for r in rows}):
    R = [r for r in rows if r["ds"] == ds]
    lines += [f"## {ds} ({len(R)} frames)", "", "| view bin | n | " + " | ".join(M) + " |", "|---|---|" + "---|" * len(M)]
    for name, S in [("all", R)] + [(f"[{lo},{hi})", [r for r in R if lo <= r["view"] < hi]) for lo, hi in zip(BINS[:-1], BINS[1:])]:
        S2 = [r for r in S if all(m in r for m in M)]
        if not S2: continue
        lines.append(f"| {name} | {len(S2)} | " + " | ".join(f"{np.mean([r[m] for r in S2]):.2f}" for m in M) + " |")
    lines.append("")
    lines += ["| clip | n | " + " | ".join(M) + " |", "|---|---|" + "---|" * len(M)]
    for clip in sorted({r["clip"] for r in R}):
        S2 = [r for r in R if r["clip"] == clip and all(m in r for m in M)]
        if S2: lines.append(f"| {clip} | {len(S2)} | " + " | ".join(f"{np.mean([r[m] for r in S2]):.2f}" for m in M) + " |")
    lines.append("")
txt = "\n".join(lines); print(txt)
if a.out: Path(a.out).write_text(txt); json.dump(rows, open(Path(a.out).with_suffix(".json"), "w"))
