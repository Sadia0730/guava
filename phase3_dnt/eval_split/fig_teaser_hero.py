"""Paper teaser, hand-picked examples (PREREG "Extended benchmark, amendment 5", presentation only). Per row: source photo ->
avatar from the photo (front view, camera turned 30 deg) | the back the photo never showed, zoomed: photo-only avatar | after the streaming
UV update | real video frame (teaser_render.py --back-key ... --zoom 1.6). Composited on white from RGBA renders; vector PDF.
Usage: python fig_teaser_hero.py --rows ubc/A1YNmKj0sCS:frame_000236,ubc/91bxAN6BjAS:frame_000164 --E <root> --renders <dir> --out <dir>"""
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
a = ap.parse_args(); RR = Path(a.renders); E = Path(a.E); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
INK, MUTED, ACC, ACC_F, PANEL = "#1f1f1f", "#555555", "#D9771E", "#FDF1E6", "#F3F6FA"
NARROW = 0.43                                                                                        # full-body tile width / height


def rgba(p): return np.asarray(Image.open(p).convert("RGBA")).astype(np.float32) / 255
def on(x, bg): bg = np.asarray(bg, np.float32).reshape(1, 1, 3); return x[..., :3] * x[..., 3:] + bg * (1 - x[..., 3:])
def hexf(c): return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def body(x, asp=NARROW, pad=0.04, bg=(1, 1, 1)):
    al = x[..., 3]; ys, xs = np.nonzero(al > 0.05); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2; hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * asp); hh = ww / asp
    im = on(x, bg); H, W = im.shape[:2]; P = np.ones((H + 2000, W + 2000, 3), np.float32) * np.asarray(bg, np.float32)
    P[1000:1000 + H, 1000:1000 + W] = im; Y0, X0 = int(round(cy - hh / 2)) + 1000, int(round(cx - ww / 2)) + 1000
    return P[Y0:Y0 + int(round(hh)), X0:X0 + int(round(ww))]


def photo(base, key):
    im = np.asarray(Image.open(next(base.glob(f"images/{key}.*"))).convert("RGB")).astype(np.float32) / 255
    lab, _ = ndimage.label(im.max(-1) < 0.06); edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    im = np.where(ndimage.binary_dilation(np.isin(lab, edge[edge > 0]), iterations=4)[..., None], 1.0, im)   # crop padding -> white
    m = np.asarray(Image.open(base / "masks" / f"{key}.png").convert("L")) > 127
    return body_from_mask(im, m)


def body_from_mask(im, m, asp=NARROW, pad=0.04):
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * asp); hh = ww / asp; H, W = m.shape
    P = np.ones((H + 2000, W + 2000, 3), np.float32); P[1000:1000 + H, 1000:1000 + W] = im
    Y0, X0 = int(round(cy - hh / 2)) + 1000, int(round(cx - ww / 2)) + 1000
    return P[Y0:Y0 + int(round(hh)), X0:X0 + int(round(ww))]


rows = []
for r in a.rows.split(","):
    cid, key = r.split(":"); ds, clip = cid.split("/"); d = RR / ds / clip; run = json.loads((d / "run.json").read_text())
    base = E / "base" / ds / clip; fj = json.loads((base / "frames.json").read_text()); z = f"zoom{a.zoom}"
    rows.append({"cov": (run["coverage_photo"], run["coverage_after_update"]),
                 "narrow": [photo(base, fj["source"]), body(rgba(d / "orbit_photo_000.png"), bg=hexf(PANEL)),
                            body(rgba(d / "orbit_photo_330.png"), bg=hexf(PANEL))],
                 "square": [on(rgba(d / f"back_{z}_{w}_{key}.png"), (1, 1, 1)) for w in ("photo", "update", "gt")]})

LEFT, ARW, g, G = 0.13, 0.30, 0.03, 0.16
sq = (a.width - LEFT - ARW - G - 4 * g - 0.03) / (3 + 3 * NARROW); nw = sq * NARROW                  # square side; narrow width
top, RG = 0.46, 0.05; Hfig = top + len(rows) * (sq + RG) + 0.02
fig = plt.figure(figsize=(a.width, Hfig)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, a.width); ax.set_ylim(0, Hfig); ax.axis("off")
xs = [LEFT]; xs.append(xs[0] + nw + ARW); xs.append(xs[1] + nw + g); xs.append(xs[2] + nw + G); xs.append(xs[3] + sq + g); xs.append(xs[4] + sq + g)
# panels + headers
for (x0, x1, fc) in ((xs[1] - 0.04, xs[2] + nw + 0.04, PANEL), (xs[3] - 0.05, xs[5] + sq + 0.05, ACC_F)):
    ax.add_patch(FancyBboxPatch((x0, 0.01), x1 - x0, Hfig - 0.03, boxstyle="round,pad=0,rounding_size=0.05", lw=0, fc=fc, zorder=0))
H1 = Hfig - 0.05
ax.text(xs[0] + nw / 2, H1, "One photo", ha="center", va="top", fontsize=8, weight="bold", color=INK)
ax.text((xs[1] + xs[2] + nw) / 2, H1, "Avatar in 55 ms", ha="center", va="top", fontsize=8, weight="bold", color=INK)
ax.text((xs[1] + xs[2] + nw) / 2, H1 - 0.15, "front  ·  turned 30°", ha="center", va="top", fontsize=6.4, style="italic", color=MUTED)
ax.text((xs[3] + xs[5] + sq) / 2, H1, "The back the photo never showed", ha="center", va="top", fontsize=8, weight="bold", color=ACC)
for i, t in enumerate(("photo only", "after live video update", "real video frame")):
    ax.text(xs[3 + i] + sq / 2, H1 - 0.17, t, ha="center", va="top", fontsize=6.6, style="italic", color=ACC if i == 1 else MUTED,
            weight="bold" if i == 1 else "normal")
for ri, r in enumerate(rows):
    y = Hfig - top - (ri + 1) * (sq + RG) + RG
    for ci, t in enumerate(r["narrow"]): ax.imshow(t, extent=(xs[ci], xs[ci] + nw, y, y + sq), zorder=2, interpolation="lanczos")
    for ci, t in enumerate(r["square"]): ax.imshow(t, extent=(xs[3 + ci], xs[3 + ci] + sq, y, y + sq), zorder=2, interpolation="lanczos")
    ax.add_patch(FancyArrowPatch((xs[0] + nw + 0.04, y + sq / 2), (xs[1] - 0.07, y + sq / 2), arrowstyle="-|>,head_length=0.5,head_width=0.3",
                                 mutation_scale=8, lw=1.4, color=ACC, zorder=3))
    ax.text((xs[0] + nw + xs[1]) / 2 - 0.015, y + sq / 2 + 0.05, "RAGA", ha="center", va="bottom", fontsize=6.6, weight="bold", color=ACC)
    c0, c1 = r["cov"]
    ax.text(xs[4] + sq / 2, y + 0.03, f"body seen {c0 * 100:.0f}% $\\rightarrow$ {c1 * 100:.0f}%", ha="center", va="bottom", fontsize=6.2,
            color=ACC, weight="bold", zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ACC, lw=0.5, alpha=0.9))
for ext in ("pdf", "png"): fig.savefig(OUT / f"{a.name}.{ext}", dpi=300)
print("wrote", OUT / f"{a.name}.pdf", f"{a.width:.2f} x {Hfig:.2f} in")
