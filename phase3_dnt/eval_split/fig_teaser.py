"""Paper teaser (PREREG "Extended benchmark, amendment 4", presentation only). One row per clip (the rule-selected update-showcase
clips), fixed columns: source photo | avatar from the photo at 0 / 90 / 180 deg (teaser_render.py orbit, source pose) | two new
poses (test frames at 25 / 75 % of the test list, GT poses) | back view (180 deg) from the photo vs after the streaming update, with
the texture coverage photo -> after update (run.json). Composited on white from the RGBA renders; nothing is scored here.
Timing numbers are the paper's (avatar creation 54.5 ms, render 12.24 ms / frame, idle L40S).
Usage: python fig_teaser.py --rows <E>:<ds/clip>,... --renders /dev/shm/ubx858_eval/teaser_renders --out <dir> [--name fig_teaser]"""
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

ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--renders", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--name", default="fig_teaser"); ap.add_argument("--width", type=float, default=6.875)
a = ap.parse_args(); RR = Path(a.renders); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
INK, MUTED, ACC, ACC_F = "#1f1f1f", "#555555", "#D9771E", "#FCE3CC"
DSNAME = {"peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "ubc": "UBC-Fashion", "ubody": "UBody"}
ASP = 0.72                                                                                           # tile width / height


def person(rgba, asp=ASP, pad=0.05, bg=1.0, upper=None):
    """crop to the alpha box (upper=f: only the top f of the body, e.g. head + torso), fixed aspect, composited on bg (RGB)."""
    al = rgba[..., 3]; ys, xs = np.nonzero(al > 0.05); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if upper: y1 = int(y0 + upper * (y1 - y0)); sel = (ys <= y1); x0, x1 = xs[sel].min(), xs[sel].max()
    bg = np.asarray(bg, np.float32).reshape(1, 1, -1)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2; hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * asp); hh = ww / asp
    rgb = rgba[..., :3] * rgba[..., 3:] + bg * (1 - rgba[..., 3:]); H, W = rgb.shape[:2]; P = np.ones((H + 2000, W + 2000, 3), np.float32) * bg
    P[1000:1000 + H, 1000:1000 + W] = rgb; Y0, X0 = int(round(cy - hh / 2)) + 1000, int(round(cx - ww / 2)) + 1000
    return P[Y0:Y0 + int(round(hh)), X0:X0 + int(round(ww))]


def rd_rgba(p): return np.asarray(Image.open(p).convert("RGBA")).astype(np.float32) / 255


def photo(base, key, asp=ASP, pad=0.05):
    p = next(base.glob(f"images/{key}.*")); im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
    from scipy import ndimage                                                                        # crop padding -> white:
    lab, _ = ndimage.label(im.max(-1) < 0.06); edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    pad_ = ndimage.binary_dilation(np.isin(lab, edge[edge > 0]), iterations=4)                      # near-black regions touching the border
    im = np.where(pad_[..., None], 1.0, im)
    m = np.asarray(Image.open(base / "masks" / f"{key}.png").convert("L")) > 127; ys, xs = np.nonzero(m)
    rgba = np.concatenate([im, np.ones_like(im[..., :1])], -1); al = np.zeros(m.shape, np.float32); al[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * asp); hh = ww / asp; H, W = m.shape
    Y0, X0 = int(np.clip(round(cy - hh / 2), 0, max(0, H - round(hh)))), int(np.clip(round(cx - ww / 2), 0, max(0, W - round(ww))))
    P = np.ones((H + 2000, W + 2000, 3), np.float32); P[1000:1000 + H, 1000:1000 + W] = im
    return P[Y0 + 1000:Y0 + 1000 + int(round(hh)), X0 + 1000:X0 + 1000 + int(round(ww))]


def hex2(c): return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]


PANEL = ["#FFFFFF", "#F2F5F9", "#F2F5F9", ACC_F]
rows = []
for r in a.rows.split(","):
    E, cid = r.split(":"); ds, clip = cid.split("/"); d = RR / ds / clip; run = json.loads((d / "run.json").read_text())
    base = Path(E) / "base" / ds / clip; fj = json.loads((base / "frames.json").read_text()); an = run["anim_frames"]
    rows.append({"ds": ds, "clip": clip, "cov": (run["coverage_photo"], run["coverage_after_update"]),
                 "tiles": [photo(base, fj["source"])] + [person(rd_rgba(d / f"orbit_photo_{g:03d}.png"), bg=hex2(PANEL[1])) for g in (0, 90, 180)]
                 + [person(rd_rgba(d / f"anim_photo_{k}.png"), bg=hex2(PANEL[2])) for k in (an[0], an[-1])]
                 + [person(rd_rgba(d / f"orbit_{w}_180.png"), bg=(1, 1, 1), upper=0.5) for w in ("photo", "update")]})

g = 0.03; G = 0.16; LEFT = 0.16; ARW = 0.26                                                         # tile gap, section gap, row label, arrow
tw = (a.width - LEFT - 4 * g - 3 * G - ARW - 0.02) / 8; th = tw / ASP
sections = [(0, 1), (1, 4), (4, 6), (6, 8)]
x = [LEFT]; xs = []
for si, (s0, s1) in enumerate(sections):
    for i in range(s0, s1): xs.append(x[0]); x[0] += tw + g
    x[0] += (G + (ARW if si == 0 else 0)) - g
Wfig = a.width; top = 0.36; RG = 0.04; Hfig = top + len(rows) * (th + RG) + 0.04
fig = plt.figure(figsize=(Wfig, Hfig)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wfig); ax.set_ylim(0, Hfig); ax.axis("off")

# panel backgrounds + headers
heads = ["One photo", "Full-body avatar, new views", "New poses", "Streaming update"]
subs = ["", "built feed-forward in 55 ms", "rendered in 12 ms / frame", "unseen back: photo only | after video"]
for si, (s0, s1) in enumerate(sections):
    x0, x1 = xs[s0] - 0.04, xs[s1 - 1] + tw + 0.04; y0 = 0.03
    fc = PANEL[si]
    if si: ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, Hfig - 0.05 - y0, boxstyle="round,pad=0,rounding_size=0.05", lw=0, fc=fc, zorder=0))
    ax.text((x0 + x1) / 2, Hfig - 0.06, heads[si], ha="center", va="top", fontsize=7.6, weight="bold", color=ACC if si == 3 else INK)
    if subs[si]: ax.text((x0 + x1) / 2, Hfig - 0.20, subs[si], ha="center", va="top", fontsize=6.4, style="italic", color=MUTED)
for ri, r in enumerate(rows):
    if ri == 0:
        for ci in (6, 7): pass
    y = Hfig - top - (ri + 1) * (th + RG) + RG
    for ci, t in enumerate(r["tiles"]):
        ax.imshow(t, extent=(xs[ci], xs[ci] + tw, y, y + th), zorder=2, interpolation="lanczos")
    ax.text(xs[0] - 0.035, y + th / 2, DSNAME.get(r["ds"], r["ds"]), rotation=90, ha="right", va="center", fontsize=6.2, color=MUTED)
    ax.add_patch(FancyArrowPatch((xs[0] + tw + 0.05, y + th / 2), (xs[1] - 0.09, y + th / 2), arrowstyle="-|>,head_length=0.5,head_width=0.3",
                                 mutation_scale=8, lw=1.4, color=ACC, zorder=3))
    for ci, lab in ((1, "0°"), (2, "90°"), (3, "180°")):
        ax.text(xs[ci] + tw - 0.02, y + 0.02, lab, ha="right", va="bottom", fontsize=5.6, color=MUTED, zorder=4)
    c0, c1 = r["cov"]
    ax.text(xs[7] - g / 2, y + 0.02, f"body seen {c0 * 100:.0f}% $\\rightarrow$ {c1 * 100:.0f}%", ha="center", va="bottom",
            fontsize=5.8, color=ACC, weight="bold", zorder=4, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
ax.text((xs[0] + tw + xs[1]) / 2 - 0.02, Hfig - top + 0.02, "RAGA", ha="center", va="bottom", fontsize=7, weight="bold", color=ACC)
for ext in ("pdf", "png"): fig.savefig(OUT / f"{a.name}.{ext}", dpi=300)
print("wrote", OUT / f"{a.name}.pdf", f"{Wfig:.2f} x {Hfig:.2f} in")
