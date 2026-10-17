"""C2 baselines (PREREG 'C2'): per-identity means, gap closed = (B - S) / (O - S) on identity means, 95 % bootstrap CI
over identities (1,000 resamples, seed 0); secondary metrics per variant. Input: <results>/<clip>/summary.json."""
import glob, json, os, sys
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else "/raid/ubx858/outputs/dnt/c2/baselines"
split = json.load(open("/raid/ubx858/outputs/dnt/c1/split.json")); ident = {c["clip"]: c["identity"] for c in split["clips"]}
V = ["O", "S", "S:B1", "S:B2", "S:B2+OE"]; M = ["psnr", "lpips", "ssim", "kp_dist", "lpips_face", "lpips_hands", "t_lpips", "flicker"]
per = {}
for f in sorted(glob.glob(f"{R}/*/summary.json")):
    d = json.load(open(f)); clip = os.path.basename(os.path.dirname(f))
    per.setdefault(ident.get(clip, clip), []).append(d["means"])
ids = sorted(per); X = {v: {m: np.array([np.mean([c[v][m] for c in per[i]]) for i in ids]) for m in M} for v in V}
rng = np.random.default_rng(0); boots = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
def ci(fn):
    return fn(np.arange(len(ids))), np.percentile([fn(b) for b in boots], 2.5), np.percentile([fn(b) for b in boots], 97.5)
out = {"identities": len(ids), "means": {v: {m: float(X[v][m].mean()) for m in M} for v in V}, "gap_closed": {}}
lines = [f"C2 baselines, {len(ids)} identities (per-identity means; 95 % bootstrap CI over identities)", "",
         "| Variant | PSNR | LPIPS | kp dist | face LPIPS | hand LPIPS | t-LPIPS | flicker |", "|---|---|---|---|---|---|---|---|"]
for v in V:
    mm = out["means"][v]
    lines.append(f"| {v} | {mm['psnr']:.2f} | {mm['lpips']:.4f} | {mm['kp_dist']:.3f} | {mm['lpips_face']:.4f} | {mm['lpips_hands']:.4f} | {mm['t_lpips']:.4f} | {mm['flicker']:+.4f} |")
lines += ["", "| Baseline | PSNR gap closed | LPIPS gap closed | B − S PSNR (dB) | O − B PSNR (dB) |", "|---|---|---|---|---|"]
for v in ["S:B1", "S:B2", "S:B2+OE"]:
    g = {}
    for m, sgn in (("psnr", 1), ("lpips", -1)):
        g[m] = ci(lambda b, m=m: (X[v][m][b].mean() - X["S"][m][b].mean()) / (X["O"][m][b].mean() - X["S"][m][b].mean()))
    d1 = ci(lambda b: X[v]["psnr"][b].mean() - X["S"]["psnr"][b].mean()); d2 = ci(lambda b: X["O"]["psnr"][b].mean() - X[v]["psnr"][b].mean())
    out["gap_closed"][v] = {"psnr": g["psnr"], "lpips": g["lpips"], "B_minus_S_psnr": d1, "O_minus_B_psnr": d2}
    f3 = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
    f3p = lambda t: f"{100*t[0]:+.0f} % [{100*t[1]:+.0f}, {100*t[2]:+.0f}]"
    lines.append(f"| {v} | {f3p(g['psnr'])} | {f3p(g['lpips'])} | {f3(d1)} | {f3(d2)} |")
txt = "\n".join(lines); print(txt)
open(f"{R}/c2_table.md", "w").write(txt + "\n"); json.dump(out, open(f"{R}/c2_stats.json", "w"), indent=1, default=float)
