"""C1 G1 statistics from /raid/ubx858/outputs/dnt/c1/results/summary.json (per-identity means): the pre-registered G1
rule on all identities (primary) and without oracle-failure identities (sensitivity, O kp_dist > 0.10), plus the
descriptive attribution (part swaps, camera split) raw and aligned. Writes report_stats.json and attribution plots."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(sys.argv[1] if len(sys.argv) > 1 else "/raid/ubx858/outputs/dnt/c1/results")
s = json.loads((R / "summary.json").read_text())
ids_all = list(s["clips"])
flag = [c for c in ids_all if s["clips"][c]["means"]["O"]["kp_dist"] > 0.10]
sets = {"all": ids_all, "oracle_ok": [c for c in ids_all if c not in flag]}
rng = np.random.default_rng(0)

def val(ids, n, m):
    return np.array([s["clips"][c]["means"][n][m] for c in ids])

def boot(x):
    b = [np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(1000)]
    return float(np.nanmean(x)), [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]

out = {"flagged_oracle_failures": flag, "sets": {}}
for sname, ids in sets.items():
    o = {"n": len(ids)}
    for suf in ("", "_aligned"):
        dp = val(ids, "O", "psnr" + suf) - val(ids, "S", "psnr" + suf)          # positive = O better
        dl = val(ids, "S", "lpips" + suf) - val(ids, "O", "lpips" + suf)
        o["d_psnr" + suf] = boot(dp); o["d_lpips" + suf] = boot(dl)
        for drv in ("S+R", "P"):
            o[f"d_psnr_{drv}{suf}"] = boot(val(ids, "O", "psnr" + suf) - val(ids, drv, "psnr" + suf))
            o[f"d_lpips_{drv}{suf}"] = boot(val(ids, drv, "lpips" + suf) - val(ids, "O", "lpips" + suf))
        # attribution: gap recovered by giving S each part from O; gap lost by giving O each part from S
        for part in ("body", "hands", "face", "cam", "ray", "depth"):
            if f"S+O:{part}" in s["variants"]:
                o[f"recover_psnr_{part}{suf}"] = boot(val(ids, f"S+O:{part}", "psnr" + suf) - val(ids, "S", "psnr" + suf))
                o[f"recover_lpips_{part}{suf}"] = boot(val(ids, "S", "lpips" + suf) - val(ids, f"S+O:{part}", "lpips" + suf))
            if f"O+S:{part}" in s["variants"]:
                o[f"loss_psnr_{part}{suf}"] = boot(val(ids, "O", "psnr" + suf) - val(ids, f"O+S:{part}", "psnr" + suf))
                o[f"loss_lpips_{part}{suf}"] = boot(val(ids, f"O+S:{part}", "lpips" + suf) - val(ids, "O", "lpips" + suf))
    for m in ("lpips_face", "lpips_hands", "kp_dist", "t_lpips", "flicker", "ssim"):
        o[f"S_minus_O_{m}"] = boot(val(ids, "S", m) - val(ids, "O", m))
    dp, dl = o["d_psnr"], o["d_lpips"]
    o["G1"] = "GO" if (dp[0] >= 0.5 and dp[1][0] > 0) or (dl[0] >= 0.01 and dl[1][0] > 0) else "NO-GO"
    out["sets"][sname] = o
(R / "report_stats.json").write_text(json.dumps(out, indent=1) + "\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
for sname in sets:
    o = out["sets"][sname]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, suf, title in ((axes[0], "", "raw"), (axes[1], "_aligned", "after 2D similarity alignment")):
        parts = ["body", "hands", "face", "cam", "ray", "depth"]; x = np.arange(len(parts))
        rec = [o.get(f"recover_psnr_{p}{suf}", (np.nan, [np.nan, np.nan])) for p in parts]
        los = [o.get(f"loss_psnr_{p}{suf}", (np.nan, [np.nan, np.nan])) for p in parts]
        for off, data, lab in ((-0.2, rec, "S + part from O: PSNR gained over S"), (0.2, los, "O + part from S: PSNR lost vs O")):
            m = np.array([d[0] for d in data]); lo = np.array([d[1][0] for d in data]); hi = np.array([d[1][1] for d in data])
            ax.bar(x + off, m, 0.4, label=lab, yerr=[m - lo, hi - m], capsize=3)
        gap = o["d_psnr" + suf]
        ax.axhline(gap[0], color="k", ls="--", lw=1, label=f"total gap O − S = {gap[0]:+.2f} dB")
        ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(parts); ax.set_ylabel("Δ PSNR (dB)")
        ax.set_title(f"{title} — {sname} (n = {o['n']})"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(R / f"attribution_{sname}.png", dpi=130); plt.close(fig)
f = lambda t: f"{t[0]:+.3f} [{t[1][0]:+.3f}, {t[1][1]:+.3f}]"
for sname, o in out["sets"].items():
    print(f"== {sname} (n={o['n']}) G1 {o['G1']}")
    print("  O−S PSNR raw", f(o["d_psnr"]), "| aligned", f(o["d_psnr_aligned"]))
    print("  S−O LPIPS raw", f(o["d_lpips"]), "| aligned", f(o["d_lpips_aligned"]))
    print("  O−(S+R) PSNR", f(o["d_psnr_S+R"]), " O−P PSNR", f(o["d_psnr_P"]))
    for p in ("body", "hands", "face", "cam", "ray", "depth"):
        print(f"  {p:6s} recover PSNR raw {f(o.get('recover_psnr_'+p, (0,[0,0])))} aligned {f(o.get('recover_psnr_'+p+'_aligned', (0,[0,0])))} | loss raw {f(o.get('loss_psnr_'+p, (0,[0,0])))}")
    print("  S−O face LPIPS", f(o["S_minus_O_lpips_face"]), "hands LPIPS", f(o["S_minus_O_lpips_hands"]), "kp", f(o["S_minus_O_kp_dist"]), "tLPIPS", f(o["S_minus_O_t_lpips"]), "flicker", f(o["S_minus_O_flicker"]))
print("flagged:", flag)
