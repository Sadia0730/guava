"""Phase 7 decision (PREREG "Phase 7" + amendment 1) from the ValBench-v1 score files: table of the rows, rules (1)-(3) for arm A
(and C), attribution A - B, adoption rule for C (exposure-shift probe). Writes <out>/decision.md and the view-bin tables."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--E", default="/dev/shm/ubx858_eval/valbench_v1"); ap.add_argument("--split", default="valbench_v1.json")
ap.add_argument("--out", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/p7"); ap.add_argument("--no-bins", action="store_true")
a = ap.parse_args(); E = Path(a.E); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); HERE = Path(__file__).resolve().parent
ROWS = ["fb_gt", "diag_fuse_oracle", "p7_b28k_fused_exp"] + [f"p7_{x}_{r}" for x in "ABC" for r in ("single", "fused", "fused_exp")]
S = {}
for r in ROWS:
    f = E / "scores" / f"{r}_vb.json"
    if f.exists():
        d = json.load(open(f)); S[r] = d["aggregate"]; S[r]["_missing"] = len(d.get("missing_clips", []))
L = []
def P(s=""): L.append(s); print(s)
P("# Phase 7 decision (ValBench-v1, frame-weighted, score.py --tag vb)"); P()
P("| row | all PSNR | all LPIPS | UBody PSNR | BEDLAM2 PSNR | missing clips |"); P("|---|---|---|---|---|---|")
for r in ROWS:
    if r in S: P(f"| {r} | {S[r]['all']['psnr']:.3f} | {S[r]['all']['lpips']:.4f} | {S[r]['ubody']['psnr']:.3f} | {S[r]['bedlam2']['psnr']:.3f} | {S[r]['_missing']} |")
    else: P(f"| {r} | (not scored) | | | | |")
bins = {}
have = [r for r in ROWS if r in S and (E / "renders" / r).exists()]
if not a.no_bins and have:
    subprocess.run([sys.executable, str(HERE / "view_bins.py"), "--E", str(E), "--split", str(HERE / a.split), "--methods", ",".join(have), "--out", str(OUT / "valbench_viewbins.md")], check=True)
    rows = json.load(open(OUT / "valbench_viewbins.json"))
    for ds in ("ubody", "bedlam2"):
        R = [x for x in rows if x["ds"] == ds and x["view"] < 30]
        bins[ds] = {m: float(np.mean([x[m] for x in R if m in x])) for m in have if any(m in x for x in R)}; bins[ds]["_n"] = len(R)
    P(); P("Near-view bin [0,30) PSNR:"); P()
    P("| row | UBody [0,30) | BEDLAM2 [0,30) |"); P("|---|---|---|")
    for m in have: P(f"| {m} | {bins['ubody'].get(m, float('nan')):.3f} | {bins['bedlam2'].get(m, float('nan')):.3f} |")
def g(r, ds="all", k="psnr"): return S[r][ds][k] if r in S else None
P(); P("## Rules")
for X in "AC":
    fu, si = f"p7_{X}_fused", f"p7_{X}_single"
    if fu not in S or si not in S or "fb_gt" not in S or "diag_fuse_oracle" not in S: P(f"arm {X}: rows missing"); continue
    r1 = g(fu) >= g("diag_fuse_oracle") + 0.30 and g(fu, k="lpips") <= g("diag_fuse_oracle", k="lpips") + 0.002
    r2 = g(si) >= g("fb_gt") - 0.10 and g(si, k="lpips") <= g("fb_gt", k="lpips") + 0.002
    r3 = all(bins[ds].get(fu, -1e9) >= bins[ds].get(si, 1e9) - 0.10 for ds in ("ubody", "bedlam2")) if bins else None
    P(f"arm {X}: (1) fused {g(fu):.3f} vs naive {g('diag_fuse_oracle'):.3f} + 0.30 = {g('diag_fuse_oracle') + 0.30:.3f}, LPIPS {g(fu, k='lpips'):.4f} vs {g('diag_fuse_oracle', k='lpips') + 0.002:.4f} -> {'PASS' if r1 else 'FAIL'}")
    P(f"arm {X}: (2) single {g(si):.3f} vs fb_gt {g('fb_gt'):.3f} - 0.10 = {g('fb_gt') - 0.10:.3f}, LPIPS {g(si, k='lpips'):.4f} vs {g('fb_gt', k='lpips') + 0.002:.4f} -> {'PASS' if r2 else 'FAIL'}")
    if bins: P(f"arm {X}: (3) near-view fused vs single - 0.10: UBody {bins['ubody'].get(fu, float('nan')):.3f} vs {bins['ubody'].get(si, float('nan')) - 0.10:.3f}, BEDLAM2 {bins['bedlam2'].get(fu, float('nan')):.3f} vs {bins['bedlam2'].get(si, float('nan')) - 0.10:.3f} -> {'PASS' if r3 else 'FAIL'}")
    P(f"arm {X}: ALL RULES {'PASS' if (r1 and r2 and (r3 or r3 is None)) else 'FAIL'}")
if "p7_A_fused" in S and "p7_B_fused" in S:
    d = g("p7_A_fused") - g("p7_B_fused"); P(f"attribution: A_fused - B_fused = {d:+.3f} dB ({'fusion-aware training' if d >= 0.20 else 'NOT attributable to fusion-aware training (< 0.20): continued training'})")
if all(r in S for r in ("p7_A_fused", "p7_C_fused", "p7_A_single", "p7_C_single", "p7_A_fused_exp", "p7_C_fused_exp")):
    c1 = g("p7_C_fused") >= g("p7_A_fused") - 0.05; c2 = g("p7_C_single") >= g("p7_A_single") - 0.05; c3 = g("p7_C_fused_exp") >= g("p7_A_fused_exp") + 0.20
    P(f"C vs A: fused {g('p7_C_fused'):.3f} vs {g('p7_A_fused'):.3f} ({'ok' if c1 else 'cost'}), single {g('p7_C_single'):.3f} vs {g('p7_A_single'):.3f} ({'ok' if c2 else 'cost'}), exposure probe {g('p7_C_fused_exp'):.3f} vs {g('p7_A_fused_exp'):.3f} + 0.20 ({'gain' if c3 else 'no gain'})"
      + (f"; B28k probe {g('p7_b28k_fused_exp'):.3f}" if "p7_b28k_fused_exp" in S else "") + f" -> {'ADOPT C' if (c1 and c2 and c3) else 'keep A'} (if A passes)")
(OUT / "decision.md").write_text("\n".join(L) + "\n"); print("written", OUT / "decision.md")
