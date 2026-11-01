"""Where a change in squared error sits relative to the silhouette (CPU; validation only). Classes from the 512 StyleMatte mask of
the baseline package (the same alpha that composes gt512): interior = mask > 0.5 eroded by --band px, band = dilated XOR eroded,
outside = beyond the dilation. Per dataset: pooled PSNR per class, miss / spill (diag_regions' definitions), and the share of the
SSE change between --base and each other method that falls in each class (does a gain come from the outline or from the inside?).
Usage: python diag_silhouette.py --E <root> --methods fb_sprint,fb_sprint_k60_trkc --base fb_sprint --out <json>"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--methods", required=True); ap.add_argument("--base", required=True)
ap.add_argument("--band", type=int, default=5); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--out", required=True)
a = ap.parse_args(); E = Path(a.E); M = a.methods.split(","); assert a.base in M
CL = ("interior", "band", "outside")
rd = lambda p: np.asarray(Image.open(p).convert("RGB"), np.float64) / 255.0


def clip(cid):
    acc = {m: {c: [0.0, 0] for c in CL} | {"miss": 0, "spill": 0, "mask": 0, "frames": 0} for m in M}
    for g in sorted((E / "gt512" / cid).glob("*.png")):
        gt = rd(g); mk = np.asarray(Image.open(E / "base" / cid / "masks" / g.name), np.float64) / 255.0 > 0.5
        it = binary_erosion(mk, iterations=a.band); dl = binary_dilation(mk, iterations=a.band); cls = {"interior": it, "band": dl & ~it, "outside": ~dl}
        for m in M:
            r = rd(E / "renders" / m / cid / g.name); se = ((r - gt) ** 2).sum(-1); cov = (r > 0.02).any(-1)
            for c in CL: acc[m][c][0] += float(se[cls[c]].sum()); acc[m][c][1] += int(cls[c].sum())
            acc[m]["miss"] += int((mk & ~cov).sum()); acc[m]["spill"] += int((~mk & cov).sum()); acc[m]["mask"] += int(mk.sum()); acc[m]["frames"] += 1
    return cid, acc


if __name__ == "__main__":
    cids = sorted(str(p.relative_to(E / "gt512")) for p in (E / "gt512").glob("*/*") if p.is_dir())
    with ProcessPoolExecutor(a.workers) as ex: R = dict(ex.map(clip, cids))
    res = {"band_px": a.band, "base": a.base, "datasets": {}}
    for ds in ("ubody", "bedlam2"):
        T = {m: {c: [sum(R[k][m][c][0] for k in R if k.startswith(ds + "/")), sum(R[k][m][c][1] for k in R if k.startswith(ds + "/"))] for c in CL} for m in M}
        cnt = {m: {q: sum(R[k][m][q] for k in R if k.startswith(ds + "/")) for q in ("miss", "spill", "mask", "frames")} for m in M}
        D = {}
        for m in M:
            tot = sum(T[m][c][0] for c in CL); dtot = tot - sum(T[a.base][c][0] for c in CL)
            D[m] = {"frames": cnt[m]["frames"], "miss_frac_of_mask": cnt[m]["miss"] / cnt[m]["mask"], "spill_frac_of_mask": cnt[m]["spill"] / cnt[m]["mask"],
                    "classes": {c: {"px_share": T[m][c][1] / sum(T[m][q][1] for q in CL), "psnr": float(-10 * np.log10(T[m][c][0] / (3 * T[m][c][1]))),
                                    "sse_share": T[m][c][0] / tot, "share_of_sse_change_vs_base": ((T[m][c][0] - T[a.base][c][0]) / dtot) if m != a.base and dtot else None} for c in CL}}
        res["datasets"][ds] = D
    Path(a.out).write_text(json.dumps(res, indent=1))
    for ds, D in res["datasets"].items():
        print(f"\n{ds}: {D[M[0]]['frames']} frames")
        for m, v in D.items():
            print(f"  {m:22s} miss {100 * v['miss_frac_of_mask']:.2f} % spill {100 * v['spill_frac_of_mask']:.2f} % | " + " | ".join(
                f"{c} {v['classes'][c]['psnr']:.2f} dB" + (f" ({100 * v['classes'][c]['share_of_sse_change_vs_base']:.0f} % of change)" if v['classes'][c]['share_of_sse_change_vs_base'] is not None else "") for c in CL))
