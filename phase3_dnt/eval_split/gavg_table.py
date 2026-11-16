"""Averaging decoded Gaussians vs fusing decoder inputs, ValBench-v1 (PREREG 2026-11-16 "Averaging decoded Gaussians instead of
fusing decoder inputs"). Reads score.py's <E>/scores/<row>_vb.json. Frame-weighted PSNR / SSIM / LPIPS per dataset and over all
test frames (the stored aggregate); paired per-clip differences with a percentile bootstrap (clips resampled within each dataset).
Usage: python gavg_table.py [--E /dev/shm/ubx858_eval/valbench_v1] [--out O/reports/gavg]"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--E", default="/dev/shm/ubx858_eval/valbench_v1")
ap.add_argument("--out", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/gavg")
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261109)
a = ap.parse_args(); E = Path(a.E); OUT = Path(a.out)
ROWS = [("fb_gt", "Photo only"), ("diag_fuse_fill", "Fill texels the photo did not see"), ("diag_fuse_wta", "Best-facing view per texel"),
        ("diag_fuse_reg", "Weighted mean with threshold 0.3 and cap 4"), ("diag_fuse_oracle", "Weighted mean, source weight 1"),
        ("p7_b28k_fused", "Weighted mean, source weight 2 (evaluated rule)"),
        ("diag_gavg_own", "Average of decoded Gaussians, each frame's own conditioning"),
        ("diag_gavg_src", "Average of decoded Gaussians, source conditioning")]
PAIRS = [("diag_gavg_own", "p7_b28k_fused"), ("diag_gavg_src", "p7_b28k_fused"), ("diag_gavg_own", "fb_gt"), ("diag_gavg_src", "fb_gt"), ("p7_b28k_fused", "fb_gt")]
MET = ("psnr", "ssim", "lpips"); DEC = {"psnr": 2, "ssim": 3, "lpips": 3}; DS = ("ubody", "bedlam2", "all"); DSN = {"ubody": "UBody", "bedlam2": "BEDLAM2.0", "all": "All"}
J = {m: json.loads((E / "scores" / f"{m}_vb.json").read_text()) for m, _ in ROWS if (E / "scores" / f"{m}_vb.json").exists()}
NAME = dict(ROWS); ref = J["fb_gt"]["per_clip"]; clips = sorted(c for c in ref if all(c in J[m]["per_clip"] for m in J))
L = [f"# Averaging decoded Gaussians vs fusing decoder inputs, validation set ({len(clips)} of {len(ref)} clips; written {time.strftime('%F %T')})", "",
     "PSNR / SSIM / LPIPS over all test frames of a group (frame weighted).", "",
     "| Rule | " + " | ".join(f"{DSN[d]} ({J['fb_gt']['aggregate'][d]['clips']} clips, {J['fb_gt']['aggregate'][d]['frames']} frames)" for d in DS) + " |", "|---|---|---|---|"]
for m, n in ROWS:
    if m not in J: L.append(f"| {n} | missing | | |"); continue
    ag = J[m]["aggregate"]; L.append(f"| {n} | " + " | ".join(" / ".join(f"{ag[d][k]:.{DEC[k]}f}" for k in MET) for d in DS) + " |")
rng = np.random.default_rng(a.seed); dsc = {d: [c for c in clips if ref[c]["dataset"] == d] for d in ("ubody", "bedlam2")}
draw = {d: rng.integers(0, len(v), size=(a.boot, len(v))) for d, v in dsc.items()}
res = {}
L += ["", "Paired per-clip differences (clip means), mean [95 % bootstrap interval, clips resampled within each dataset], clips better:", ""]
for m, r in PAIRS:
    if m not in J or r not in J: continue
    out = {}
    for k in MET:
        dd = {d: np.array([J[m]["per_clip"][c][k] - J[r]["per_clip"][c][k] for c in v]) for d, v in dsc.items()}
        allv = np.concatenate([dd["ubody"], dd["bedlam2"]]); sgn = -1 if k == "lpips" else 1
        bs = {d: dd[d][draw[d]].mean(1) for d in dd}; n = {d: len(dd[d]) for d in dd}
        ball = (bs["ubody"] * n["ubody"] + bs["bedlam2"] * n["bedlam2"]) / (n["ubody"] + n["bedlam2"])
        out[k] = {"all": [float(allv.mean()), *map(float, np.percentile(ball, [2.5, 97.5])), int((sgn * allv > 0).sum()), len(allv)],
                  **{d: [float(dd[d].mean()), *map(float, np.percentile(bs[d], [2.5, 97.5])), int((sgn * dd[d] > 0).sum()), n[d]] for d in dd}}
    res[f"{m} - {r}"] = out
    L.append(f"- {NAME[m]} - {NAME[r]}: " + "; ".join(f"{k.upper()} {out[k]['all'][0]:+.{DEC[k] + 1}f} [{out[k]['all'][1]:+.{DEC[k] + 1}f}, {out[k]['all'][2]:+.{DEC[k] + 1}f}], {out[k]['all'][3]}/{out[k]['all'][4]}" for k in MET)
             + ". PSNR per dataset: " + ", ".join(f"{DSN[d]} {out['psnr'][d][0]:+.2f} [{out['psnr'][d][1]:+.2f}, {out['psnr'][d][2]:+.2f}] ({out['psnr'][d][3]}/{out['psnr'][d][4]})" for d in ("ubody", "bedlam2")))
sc = {}
for m in ("diag_gavg_own", "diag_gavg_src"):
    v = []
    for c in clips:
        p = E / "renders" / m / c / "run.json"
        if p.exists(): v.append(json.loads(p.read_text()).get("selfcheck_gavg_source_only_vs_static_u8"))
    sc[m] = [len(v), max(v) if v else None]
L += ["", "Self-check (average of the source alone vs the static avatar, largest 8-bit difference over clips): " + "; ".join(f"{m}: {n} clips, max {mx}" for m, (n, mx) in sc.items()) + "."]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gavg_table.md").write_text("\n".join(L) + "\n")
(OUT / "gavg_table.json").write_text(json.dumps({"clips": clips, "aggregate": {m: J[m]["aggregate"] for m in J}, "paired": res, "selfcheck": sc, "boot": a.boot, "seed": a.seed}, indent=1))
for m in J: shutil.copyfile(E / "scores" / f"{m}_vb.json", OUT / f"{m}_vb.json")
print("\n".join(L))
