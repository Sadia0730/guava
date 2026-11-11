"""Paper teaser, hand-picked examples (PREREG "Extended benchmark", amendment 5, presentation only). Per row: source photo ->
avatar from the photo (re-rendered in the photo's pose; driven by a later video frame's pose) | the back the photo never showed,
zoomed: photo-only avatar | after the video update | real video frame. Renders: teaser_render.py --back-key K --zoom 1.4
--anim-keys <pose frame>. Vector PDF + PNG.

Compositing (v3, 2026-11-11): the renderer's RGB is already blended on black, so tiles are RGB + (1 - alpha) x white (v2 multiplied
by alpha a second time, which drew a dark outline around every figure). --rim-trim (default on): the avatar has a thin rim of
near-background-dark Gaussians at grazing angles (invisible on the black background it is rendered and scored on); where the
surface just inside the silhouette is bright, rim pixels much darker than it count as background. Dark hair / clothing is never
touched. --no-rim-trim shows the raw composite.
Usage: python fig_teaser_hero.py --rows ubc/A1YNmKj0sCS:frame_000232:frame_000280,ubc/91bxAN6BjAS:frame_000164:frame_000248
       --E <root> --renders <dir> --out <dir>      (row = clip : back frame : pose frame)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                      # noqa: E402
import numpy as np                                                                                   # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch                                       # noqa: E402
from PIL import Image                                                                                # noqa: E402
from scipy import ndimage                                                                            # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--E", required=True)
ap.add_argument("--renders", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="fig_teaser")
ap.add_argument("--zoom", default="1.4"); ap.add_argument("--width", type=float, default=6.875)
ap.add_argument("--no-rim-trim", action="store_true"); ap.add_argument("--dpi", type=int, default=600)
a = ap.parse_args(); RR = Path(a.renders); E = Path(a.E); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
INK, MUTED, ACC, ACC_F, PANEL, LINE = "#1f1f1f", "#5a5a5a", "#D9771E", "#FDF1E6", "#F2F5F9", "#D9DEE5"
NARROW = 0.40                                                                                        # full-body tile width / height


def rgba(p): return np.asarray(Image.open(p).convert("RGBA")).astype(np.float32) / 255
def hexf(c): return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def render_on(x, bg=(1, 1, 1), band=6):
    """x: RGB blended on black + rasterised coverage -> composite on bg; optional rim trim (see the module docstring)."""
    rgb, al = x[..., :3], x[..., 3].copy()
    if not a.no_rim_trim:
        inner = ndimage.binary_erosion(al > 0.5, iterations=band); rim = (al > 0.01) & ~inner; lum = rgb.max(-1)
        iy, ix = ndimage.distance_transform_edt(~inner, return_distances=False, return_indices=True)
        ref = ndimage.uniform_filter(lum, 5)[iy, ix]                                                 # brightness of the nearest interior surface
        cov = np.clip((lum / np.maximum(ref, 1e-6) - 0.25) / 0.30, 0, 1)
        new = np.where(rim & inner.any() & (ref > 0.5), np.minimum(al, cov), al)
        rgb = rgb * (new / np.maximum(al, 1e-6))[..., None]; al = new
    return np.clip(rgb + (1 - al[..., None]) * np.asarray(bg, np.float32).reshape(1, 1, 3), 0, 1), al


def photo_on(x, bg=(1, 1, 1)):
    """real frame with its matte (straight alpha)."""
    return x[..., :3] * x[..., 3:] + np.asarray(bg, np.float32).reshape(1, 1, 3) * (1 - x[..., 3:]), x[..., 3]


def crop_body(im, al, asp=NARROW, pad=0.035, bg=(1, 1, 1)):
    ys, xs = np.nonzero(al > 0.1); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * asp); hh = ww / asp; H, W = im.shape[:2]
    P = np.ones((H + 2000, W + 2000, 3), np.float32) * np.asarray(bg, np.float32); P[1000:1000 + H, 1000:1000 + W] = im
    Y0, X0 = int(round(cy - hh / 2)) + 1000, int(round(cx - ww / 2)) + 1000
    return P[Y0:Y0 + int(round(hh)), X0:X0 + int(round(ww))]


def source_photo(base, key, bg):
    """the 1024 source crop: studio backdrop kept inside the matte box, crop padding / backdrop replaced by the panel colour."""
    im = np.asarray(Image.open(next(base.glob(f"images/{key}.*"))).convert("RGB")).astype(np.float32) / 255
    m = np.asarray(Image.open(base / "masks" / f"{key}.png").convert("L")).astype(np.float32) / 255
    return crop_body(im * m[..., None] + np.asarray(bg, np.float32) * (1 - m[..., None]), m, bg=bg)


def fit(t, w_in, h_in):
    """resample a tile to its printed size (Lanczos, once) so the figure's own interpolation never mixes tile borders."""
    W, H = max(8, int(round(w_in * a.dpi))), max(8, int(round(h_in * a.dpi)))
    return np.asarray(Image.fromarray((np.clip(t, 0, 1) * 255).round().astype(np.uint8)).resize((W, H), Image.LANCZOS))


rows = []
for r in a.rows.split(","):
    cid, bkey, pkey = r.split(":"); ds, clip = cid.split("/"); d = RR / ds / clip; run = json.loads((d / "run.json").read_text())
    base = E / "base" / ds / clip; fj = json.loads((base / "frames.json").read_text()); z = f"zoom{a.zoom}"
    front = render_on(rgba(d / "orbit_photo_000.png")); pose = render_on(rgba(d / f"anim_photo_{pkey}.png"))
    rows.append({"cov": (run["coverage_photo"], run["coverage_after_update"]),
                 "narrow": [source_photo(base, fj["source"], (1, 1, 1)), crop_body(*front), crop_body(*pose)],
                 "square": [render_on(rgba(d / f"back_{z}_photo_{bkey}.png"))[0], render_on(rgba(d / f"back_{z}_update_{bkey}.png"))[0],
                            photo_on(rgba(d / f"back_{z}_gt_{bkey}.png"))[0]]})

LEFT, ARW, g, G = 0.10, 0.34, 0.035, 0.17
sq = (a.width - LEFT - ARW - G - 4 * g - 0.06) / (3 + 3 * NARROW); nw = sq * NARROW                  # square side; narrow width
top, RG = 0.47, 0.06; Hfig = top + len(rows) * (sq + RG) + 0.03
fig = plt.figure(figsize=(a.width, Hfig)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, a.width); ax.set_ylim(0, Hfig); ax.axis("off")
xs = [LEFT]; xs.append(xs[0] + nw + ARW); xs.append(xs[1] + nw + g); xs.append(xs[2] + nw + G); xs.append(xs[3] + sq + g); xs.append(xs[4] + sq + g)
for (x0, x1, fc) in ((xs[1] - 0.05, xs[2] + nw + 0.05, PANEL), (xs[3] - 0.06, xs[5] + sq + 0.06, ACC_F)):
    ax.add_patch(FancyBboxPatch((x0, 0.01), x1 - x0, Hfig - 0.02, boxstyle="round,pad=0,rounding_size=0.06", lw=0, fc=fc, zorder=0))
H1 = Hfig - 0.06; cB = (xs[1] + xs[2] + nw) / 2; cC = (xs[3] + xs[5] + sq) / 2
ax.text(xs[0] + nw / 2, H1, "One photo", ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
ax.text(cB, H1, "Avatar in 55 ms", ha="center", va="top", fontsize=8.2, weight="bold", color=INK)
for x, t in ((xs[1] + nw / 2, "same pose"), (xs[2] + nw / 2, "new pose")):
    ax.text(x, H1 - 0.185, t, ha="center", va="top", fontsize=6.6, style="italic", color=MUTED)
ax.text(cC, H1, "The back the photo never showed", ha="center", va="top", fontsize=8.2, weight="bold", color=ACC)
for i, t in enumerate(("from the photo only", "after the video update", "real video frame")):
    ax.text(xs[3 + i] + sq / 2, H1 - 0.185, t, ha="center", va="top", fontsize=6.8, style="italic", color=ACC if i == 1 else MUTED,
            weight="bold" if i == 1 else "normal")
for ri, r in enumerate(rows):
    y = Hfig - top - (ri + 1) * (sq + RG) + RG
    for ci, t in enumerate(r["narrow"]):
        ax.imshow(fit(t, nw, sq), extent=(xs[ci], xs[ci] + nw, y, y + sq), zorder=2, interpolation="none")
        if ci: ax.add_patch(FancyBboxPatch((xs[ci], y), nw, sq, boxstyle="square,pad=0", lw=0.4, ec=LINE, fc="none", zorder=3))
    for ci, t in enumerate(r["square"]):
        x0 = xs[3 + ci]; ax.imshow(fit(t, sq, sq), extent=(x0, x0 + sq, y, y + sq), zorder=2, interpolation="none")
        ax.add_patch(FancyBboxPatch((x0, y), sq, sq, boxstyle="square,pad=0", lw=0.9 if ci == 1 else 0.4, ec=ACC if ci == 1 else LINE, fc="none", zorder=3))
    ym = y + sq * 0.56
    ax.add_patch(FancyArrowPatch((xs[0] + nw + 0.045, ym), (xs[1] - 0.085, ym), arrowstyle="-|>,head_length=0.55,head_width=0.32",
                                 mutation_scale=8, lw=1.5, color=ACC, zorder=3))
    ax.text((xs[0] + nw + xs[1]) / 2 - 0.02, ym + 0.055, "RAGA", ha="center", va="bottom", fontsize=6.8, weight="bold", color=ACC)
    ax.text((xs[0] + nw + xs[1]) / 2 - 0.02, ym - 0.06, "feed-\nforward", ha="center", va="top", fontsize=5.6, style="italic", color=MUTED, linespacing=0.95)
    c0, c1 = r["cov"]
    ax.text(xs[4] + sq / 2, y + 0.045, f"body surface seen  {c0 * 100:.0f}% $\\rightarrow$ {c1 * 100:.0f}%", ha="center", va="bottom", fontsize=6.2,
            color=ACC, weight="bold", zorder=4, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=ACC, lw=0.5, alpha=0.92))
fig.savefig(OUT / f"{a.name}.pdf", dpi=a.dpi); fig.savefig(OUT / f"{a.name}.png", dpi=300)
print("wrote", OUT / f"{a.name}.pdf", f"{a.width:.2f} x {Hfig:.2f} in")
