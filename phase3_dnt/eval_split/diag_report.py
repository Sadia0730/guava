"""Readable tables from diag_regions.py outputs (methods side by side). Usage: python diag_report.py <dir> m1,m2,... > report.md"""
import json, sys
from pathlib import Path
import numpy as np

D, M = Path(sys.argv[1]), sys.argv[2].split(",")
J = {m: json.loads((D / f"{m}.json").read_text()) for m in M}
for ds in ("ubody", "bedlam2"):
    if not all(ds in J[m]["datasets"] for m in M): continue
    print(f"\n## {ds} ({J[M[0]]['datasets'][ds]['frames']} frames)\n")
    print("| | " + " | ".join(M) + " |\n|---|" + "---|" * len(M))
    for k in ("psnr", "ssim", "lpips", "miss_frac_of_mask", "spill_frac_of_mask"):
        print(f"| image {k} | " + " | ".join(f"{J[m]['datasets'][ds][k]:.4f}" for m in M) + " |")
    cls = [c for c in J[M[0]]["datasets"][ds]["classes"]]
    print(f"\n| class (px share) | " + " | ".join(f"{m} PSNR / SSIM / SSE share" for m in M) + " |\n|---|" + "---|" * len(M))
    for c in cls:
        t0 = J[M[0]]["datasets"][ds]["classes"][c]
        print(f"| {c} ({100 * t0['px_share']:.1f} %) | " + " | ".join(
            (lambda t: f"{t['psnr']:.2f} / {t['ssim']:.3f} / {100 * t['sse_share']:.1f} %")(J[m]["datasets"][ds]["classes"].get(c, {"psnr": np.nan, "ssim": np.nan, "sse_share": np.nan}))
            for m in M) + " |")
    for b in ("angle", "motion"):
        print(f"\n| {b} bin (n) | " + " | ".join(f"{m} PSNR / SSIM / LPIPS" for m in M) + " |\n|---|" + "---|" * len(M))
        for k, v0 in J[M[0]]["datasets"][ds]["bins"][b].items():
            print(f"| {k} ({v0['n']}) | " + " | ".join(
                (lambda v: f"{v['psnr']:.2f} / {v['ssim']:.4f} / {v['lpips']:.4f}")(J[m]["datasets"][ds]["bins"][b][k]) for m in M) + " |")
