"""Qualitative figure of the appearance update with zoom strips (PREREG "Extended benchmark", amendment 7f). CPU only; reads the
existing renders / GT / source images; writes only under --out.
Per row (clip:frame): one square crop for every column (union of the GT and all rendered silhouettes, 6 % margin, as in
fig_update_ext.py). Zoom window (registered rule): a 2:1 window of width --win x crop side, at least half covered by the GT person
mask, placed (4 px stride on the 512 px render) where mean |RGB error| of the other five render columns (GUAVA, LHM, LHM++, IDOL,
ours photo only) minus that of ours + update is largest -- i.e. the region of our largest advantage, by construction (say so in the
caption). The same window is boxed on every full view and shown enlarged below it. Renders are never altered.
Outputs: fig_update_zoom.{pdf,png} (the --fig rows, 6.88 in wide, vector text), row_<ds>_<clip>_<key>.png for every --rows entry,
tiles/<...>/<Dataset>_<clip>_<nn>_<column>{,_zoom}.png (1024 px; 512 px renders Lanczos-enlarged), zoom_selection.json.
Usage: python fig_update_zoom.py --out DIR --rows ubc/A1wrrhGRZmS:frame_000248,... [--fig ubc/A1wrrhGRZmS:frame_000248,...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
ROOT = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
SPLITS = [("avatar_test_v1", "avatar_test_v1.json", {"ubody"}), ("publicbench_v1", "publicbench_v1.json", None),
          ("publicbench_v2", "publicbench_v2_xhumans.json", None), ("finalbench_ext", "finalbench_ext.json", None)]
DSN = {"ubody": "UBody", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "ubc": "UBC-Fashion"}
COLS = [("guava", "GUAVA"), ("lhm_facefix", "LHM"), ("lhmpp", "LHM++"), ("idol", "IDOL"), ("fb_gt", "Ours, photo only"),
        ("fb_update100", "Ours + update")]
OTHERS = ["guava", "lhm_facefix", "lhmpp", "idol", "fb_gt"]
BOXC = (1.0, 0.67, 0.0)                                                    # zoom box / border colour

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--fig", default="")
ap.add_argument("--win", type=float, default=0.44); ap.add_argument("--min-mask", type=float, default=0.5)
ap.add_argument("--stride", type=int, default=4); ap.add_argument("--tile", type=int, default=1024)
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

ROOTOF = {}
for E, sj, only in SPLITS:
    if (HERE / sj).exists():
        for c in json.loads((HERE / sj).read_text())["clips"]:
            if c["dataset"] in DSN and (not only or c["dataset"] in only): ROOTOF[c["id"]] = E


def split(cid): return cid.split("/", 1)
def base(cid): ds, clip = split(cid); return ROOT / ROOTOF[cid] / "base" / ds / clip
def rd(p): p = Path(p); return np.asarray(Image.open(p).convert("RGB")) if p.exists() else None
def render(m, cid, k): ds, clip = split(cid); return rd(ROOT / ROOTOF[cid] / "renders" / m / ds / clip / f"{k}.png")
def gt(cid, k): ds, clip = split(cid); return rd(ROOT / ROOTOF[cid] / "gt512" / ds / clip / f"{k}.png")
def psnr(x, y): mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2) / 255.0 ** 2; return 10 * np.log10(1.0 / max(mse, 1e-12))


def view_deg(cid, k):
    f = json.loads((base(cid) / "frames.json").read_text()); s = np.load(base(cid) / "smplx2020.npz")
    idx = {str(q): i for i, q in enumerate(s["keys"])}; R = Rot.from_rotvec(s["global_orient"]).as_matrix()
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ R[idx[f["source"]]].T).as_rotvec())))


def square_box(mask, margin=0.06):
    H = mask.shape[0]; ys, xs = np.nonzero(mask)
    if len(ys) == 0: return (0, 0, H, H)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)); s = min(max(s, 64), H); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    t, l = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, H - s)); return (l, t, l + s, t + s)


def boxsum(img, h, w):
    """sum of img over every h x w window (top-left anchored), via an integral image."""
    I = np.pad(img.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]


def source_img(cid):
    f = json.loads((base(cid) / "frames.json").read_text()); img = rd(base(cid) / "images" / f"{f['source']}.png")
    msk = rd(base(cid) / "masks" / f"{f['source']}.png"); m = (msk[..., 0] > 127) if msk is not None else (img.sum(-1) > 0)
    l, t, r, b = square_box(m); return img[t:b, l:r]


def process(item):
    cid, k = item.rsplit(":", 1); ds, clip = split(cid); g = gt(cid, k)
    R = {m: render(m, cid, k) for m, _ in COLS}
    miss = [m for m, v in R.items() if v is None]
    assert not miss, f"{cid} {k}: missing renders {miss}"
    mask = g.sum(-1) > 0
    for im in R.values(): mask |= im.sum(-1) > 12
    l, t, r_, b = square_box(mask); s = r_ - l
    crop = lambda im: im[t:b, l:r_]
    G = crop(g).astype(np.float64); gm = crop(g).sum(-1) > 0
    err = {m: np.abs(crop(R[m]).astype(np.float64) - G).mean(-1) / 255.0 for m, _ in COLS}
    win = a.win
    while True:                       # amendment 7f fix: narrow people -> width shrinks by 0.04 until a window passes the mask rule
        w = int(round(win * s)); h = w // 2
        adv = (np.mean([boxsum(err[m], h, w) for m in OTHERS], 0) - boxsum(err["fb_update100"], h, w)) / (h * w)
        ok = boxsum(gm.astype(np.float64), h, w) / (h * w) >= a.min_mask
        grid = np.zeros_like(ok); grid[::a.stride, ::a.stride] = True
        score = np.where(ok & grid, adv, -np.inf); y0, x0 = np.unravel_index(int(np.argmax(score)), score.shape)   # ties -> top, left
        if np.isfinite(score[y0, x0]): break
        win = round(win - 0.04, 2); assert win >= 0.12, f"{cid} {k}: no window passes the mask rule"
    zwin = (int(x0), int(y0), int(x0 + w), int(y0 + h))
    zerr = {m: float(err[m][y0:y0 + h, x0:x0 + w].mean()) for m, _ in COLS}
    full = {"Ground truth": crop(g)}; full.update({lab: crop(R[m]) for m, lab in COLS})
    meta = {"clip": cid, "key": k, "dataset": DSN[ds], "view_deg": round(view_deg(cid, k), 1), "crop_box_512": [l, t, r_, b],
            "zoom_window_in_crop": list(zwin), "win_used": win, "zoom_window_frac": [round(v / s, 4) for v in zwin],
            "zoom_mean_abs_err": {m: round(v, 4) for m, v in zerr.items()},
            "zoom_advantage": round(float(score[y0, x0]), 4),
            "frame_psnr": {m: round(psnr(R[m], g), 2) for m, _ in COLS}}
    return meta, source_img(cid), full


def save_tiles(meta, src, full):
    ds, clip = split(meta["clip"]); td = OUT / "tiles" / f"{ds}_{clip}_{meta['key']}"; td.mkdir(parents=True, exist_ok=True)
    pre = f"{meta['dataset']}_{clip}"; z = a.tile
    Image.fromarray(src).resize((z, z), Image.LANCZOS).save(td / f"{pre}_00_Source_photo.png")
    x0, y0, x1, y1 = meta["zoom_window_in_crop"]
    for j, (lab, im) in enumerate(full.items(), start=1):
        n = lab.replace(" + ", "_").replace(", ", "_").replace(" ", "_")
        big = Image.fromarray(im).resize((z, z), Image.LANCZOS); sc = z / im.shape[0]
        from PIL import ImageDraw
        dr = ImageDraw.Draw(big); lw = max(3, z // 200)
        dr.rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=tuple(int(255 * c) for c in BOXC), width=lw)
        big.save(td / f"{pre}_{j:02d}_{n}.png")
        Image.fromarray(im[y0:y1, x0:x1]).resize((z, z // 2), Image.LANCZOS).save(td / f"{pre}_{j:02d}_{n}_zoom.png")


def figure(rows, name, width=6.88):
    """rows: list of (meta, src, full). One cell per column: full view (square) with the box, zoom (2:1) below."""
    ncol = 2 + len(COLS); gap = 0.02; cw = (width - gap * (ncol - 1)) / ncol; zh = cw / 2; head = 0.17
    rh = cw + zh + gap; H = head + len(rows) * (rh + 0.05)
    fig = plt.figure(figsize=(width, H), dpi=300)
    heads = ["Photo", "Ground truth"] + [lab for _, lab in COLS]
    for j, lab in enumerate(heads):
        fig.text((j * (cw + gap) + cw / 2) / width, 1 - 0.09 / H, lab, ha="center", va="center", fontsize=6.6,
                 fontweight="bold" if lab.startswith("Ours") else "normal")
    for i, (meta, src, full) in enumerate(rows):
        ytop = H - head - i * (rh + 0.05)
        ax = fig.add_axes([0, (ytop - cw) / H, cw / width, cw / H]); ax.imshow(src); ax.axis("off")
        lines = [meta["dataset"], meta["clip"].split("/", 1)[1], f"view {meta['view_deg']:.0f}°"]
        fig.text((cw / 2) / width, (ytop - cw - 0.05) / H, "\n".join(lines), ha="center", va="top", fontsize=5.2, linespacing=1.25)
        x0, y0, x1, y1 = meta["zoom_window_in_crop"]
        for j, (lab, im) in enumerate(full.items(), start=1):
            x = j * (cw + gap)
            ax = fig.add_axes([x / width, (ytop - cw) / H, cw / width, cw / H]); ax.imshow(im, interpolation="lanczos"); ax.axis("off")
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=BOXC, lw=0.7))
            key = {"GUAVA": "guava", "LHM": "lhm_facefix", "LHM++": "lhmpp", "IDOL": "idol", "Ours, photo only": "fb_gt", "Ours + update": "fb_update100"}.get(lab)
            if key:
                ax.text(0.97, 0.03, f"{meta['frame_psnr'][key]:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.6, color="white")
            az = fig.add_axes([x / width, (ytop - cw - gap / 2 - zh) / H, cw / width, zh / H])
            az.imshow(im[y0:y1, x0:x1], interpolation="lanczos"); az.set_xticks([]); az.set_yticks([])
            for sp in az.spines.values(): sp.set_edgecolor(BOXC); sp.set_linewidth(0.8)
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)
    print("wrote", OUT / f"{name}.pdf", f"{width} x {H:.2f} in", flush=True)


done = {}
for item in a.rows.split(","):
    meta, src, full = process(item); done[item] = (meta, src, full); save_tiles(meta, src, full)
    ds, clip = split(meta["clip"]); figure([done[item]], f"row_{ds}_{clip}_{meta['key']}")
(OUT / "zoom_selection.json").write_text(json.dumps({"rule": "PREREG ext. amendment 7f", "win": a.win, "min_mask": a.min_mask,
                                                     "stride": a.stride, "rows": [d[0] for d in done.values()]}, indent=1) + "\n")
if a.fig:
    figure([done[i] for i in a.fig.split(",")], "fig_update_zoom")
    cap = ("Appearance update on turned-away views. Each row: the source photo, the ground-truth test frame, four single-image "
           "methods, our avatar from the photo only, and our avatar after fusing the person's video (no gradient steps). Numbers: "
           "PSNR (dB) of the full 512 px frame. The boxed region, enlarged below each view, is placed automatically where our "
           "updated avatar's error is lowest relative to the other five columns (it shows our largest advantage by construction). "
           "Examples were selected from the clips with the largest gains on turned-away views; all numbers in the tables are over "
           "the full 163-clip benchmark.")
    (OUT / "fig_update_zoom_caption.md").write_text(cap + "\n")
for m, *_ in done.values():
    print(f"{m['clip']:34s} {m['key']}  window {m['zoom_window_frac']}  err others/ours "
          f"{np.mean([m['zoom_mean_abs_err'][o] for o in OTHERS]):.4f}/{m['zoom_mean_abs_err']['fb_update100']:.4f}")
