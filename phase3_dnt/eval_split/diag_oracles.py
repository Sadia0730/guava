"""Cheap oracle decompositions of a row's error on finished renders (PREREG "Gap diagnosis D2", 2026-11-05; descriptive, CPU).
Per test frame of each method, PSNR of
  plain      the scored render vs gt512 (= score.py);
  masked     render x the GT person mask (spill outside the person removed; upper bound of a perfect silhouette *outside* the body);
  fillmiss   person pixels the render leaves black (miss) replaced by the GT: upper bound of perfect coverage (hair, loose cloth);
  color      render after a per-frame least-squares affine colour map (3x3 + bias, fitted on mask pixels) -> upper bound of a
             global colour / exposure correction;
  lowpass    Gaussian-blurred (sigma 4 px) render vs blurred GT: structure / low frequencies only;
  residual   the high-pass parts (image - blurred): detail only (PSNR of the difference of residuals, same scale).
Rows are grouped by dataset and by the source->test view change (view_gap.json bins) and written as markdown + json.
Usage: python diag_oracles.py --E <root> --split <json> --methods fb_gt,gart,... --view-json <view_gap.json> --out <dir>"""
import argparse, json
from pathlib import Path
import cv2, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--methods", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--view-json", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/improve/view_gap/view_gap.json")
a = ap.parse_args(); E = Path(a.E); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
clips = json.loads(Path(a.split).read_text())["clips"]; methods = a.methods.split(",")
VIEW = {}
if Path(a.view_json).exists():
    for r in json.load(open(a.view_json)): VIEW[(r["ds"], r["clip"], r["key"])] = r["view"]
BINS = [0, 30, 60, 90, 135, 181]


def psnr(x, y):
    mse = float(np.mean((x - y) ** 2)); return 10 * np.log10(1.0 / max(mse, 1e-12))


def affine_fit(r, g, m):
    """per-frame 3x3 + bias colour map fitted on the mask pixels and applied to the rendered person only (render > 0.02 inside the
    mask); the background stays black, so the bias cannot move it."""
    X = np.concatenate([r[m], np.ones((int(m.sum()), 1))], 1); A, *_ = np.linalg.lstsq(X, g[m], rcond=None)
    out = np.clip((np.concatenate([r.reshape(-1, 3), np.ones((r.shape[0] * r.shape[1], 1))], 1) @ A).reshape(r.shape), 0, 1)
    keep = m & (r.max(-1) > 0.02)
    return np.where(keep[..., None], out, r)


rows = []
for c in clips:
    ds, name = c["dataset"], c["clip"]; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
    for k in fj["test"]:
        g = cv2.imread(str(E / "gt512" / ds / name / f"{k}.png")).astype(np.float64) / 255.0
        mp = E / "base" / ds / name / "masks" / f"{k}.png"
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0 if m is not None else (g.sum(-1) > 0).astype(np.float64)
        mb = m > 0.5; gb = cv2.GaussianBlur(g, (0, 0), 4)
        row = {"ds": ds, "clip": name, "key": k, "view": VIEW.get((ds, name, k)), "mask_frac": float(mb.mean())}
        for meth in methods:
            p = E / "renders" / meth / ds / name / f"{k}.png"
            if not p.exists(): continue
            r = cv2.imread(str(p)).astype(np.float64) / 255.0
            rm = r * m[..., None]; rc = affine_fit(r, g, mb) if mb.sum() > 100 else r; rb = cv2.GaussianBlur(r, (0, 0), 4)
            miss = (r.max(-1) <= 0.02) & mb; rf = np.where(miss[..., None], g, r)                 # coverage oracle: missed person pixels := GT
            row[meth] = {"plain": psnr(r, g), "masked": psnr(rm, g), "fillmiss": psnr(rf, g), "color": psnr(rc, g), "lowpass": psnr(rb, gb),
                         "residual": psnr((r - rb) + 0.5, (g - gb) + 0.5),
                         "spill_px": float(((r.max(-1) > 0.02) & ~mb).mean()), "miss_px": float(((r.max(-1) <= 0.02) & mb).mean())}
        rows.append(row)
json.dump(rows, open(OUT / "oracles_rows.json", "w"))
KEYS = ("plain", "masked", "fillmiss", "color", "lowpass", "residual", "spill_px", "miss_px")
lines = []
for ds in sorted({r["ds"] for r in rows}):
    R = [r for r in rows if r["ds"] == ds]
    lines += [f"## {ds} ({len(R)} test frames)", "", "| frames | method | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
    groups = [("all", R)] + [(f"view [{lo},{hi})", [r for r in R if r["view"] is not None and lo <= r["view"] < hi]) for lo, hi in zip(BINS[:-1], BINS[1:])]
    for gname, G in groups:
        if not G: continue
        for meth in methods:
            S = [r[meth] for r in G if meth in r]
            if not S: continue
            lines.append(f"| {gname} ({len(S)}) | {meth} | " + " | ".join(f"{np.mean([s[k] for s in S]):.3f}" if k.endswith("px") else f"{np.mean([s[k] for s in S]):.2f}" for k in KEYS) + " |")
    lines.append("")
(OUT / "oracles.md").write_text("\n".join(lines)); print("\n".join(lines))
