"""Architecture figure, v2 (PREREG "Extended benchmark", amendment 4 (c), presentation only): the three stages drawn from the
code (models/UbodyAvatar/ubody_gaussian.py, guava_fb/fb_common.py, worker6/uv_fusion.py), every thumbnail an image of the model's
own tensors for one example (teaser_render.py --arch-key: SMPL-X mesh, DINO features (PCA), per-vertex features, UV maps, the
32-channel feature raster (PCA), renders). Gray = inherited from GUAVA, orange = ours. No numbers other than the paper's.
Usage: python fig_architecture.py --E <root> --clip ubc/A1YNmKj0sCS --renders /dev/shm/ubx858_eval/teaser_renders --back-key frame_000232 --out <dir>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                      # noqa: E402
import numpy as np                                                                                   # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image                                                                                # noqa: E402
from scipy import ndimage                                                                            # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--clip", required=True)
ap.add_argument("--renders", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="fig_architecture")
ap.add_argument("--back-key", required=True); ap.add_argument("--zoom", default="1.4"); ap.add_argument("--dpi", type=int, default=600)
ap.add_argument("--cands", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/paper_figures/teaser_candidates.json",
                help="teaser_candidates.py output: mean PSNR gain of the update on the clip's back-view test frames")
a = ap.parse_args(); ds, clip = a.clip.split("/"); D = Path(a.renders) / ds / clip; A = D / "arch"; E = Path(a.E)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); meta = json.loads((A / "arch.json").read_text()); run = json.loads((D / "run.json").read_text())
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 6.4, "pdf.fonttype": 42})
INK, MUTED, ARW = "#1f1f1f", "#5b6068", "#3b3f45"
GF, GE = "#ECEEF1", "#8C939D"                                                                        # inherited from GUAVA
OF, OE = "#FCE3CC", "#D9771E"                                                                        # ours
BAND = {"a": ("#F4F7FB", "#33507A"), "b": ("#F2F8F2", "#2F6B3A"), "c": ("#FEF5EC", "#B4590E")}
DARK = (0.105, 0.115, 0.135)
W, H = 6.875, 3.98
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


# ------------------------------------------------------------------------------------------------ images
def rgba(p): return np.asarray(Image.open(p).convert("RGBA")).astype(np.float32) / 255
def premul_on(x, bg=(1, 1, 1)): return np.clip(x[..., :3] + (1 - x[..., 3:]) * np.asarray(bg, np.float32), 0, 1), x[..., 3]
def straight_on(x, bg=(1, 1, 1)): return x[..., :3] * x[..., 3:] + (1 - x[..., 3:]) * np.asarray(bg, np.float32), x[..., 3]


def render_on(x, bg=(1, 1, 1), band=6):
    """as fig_teaser_hero.render_on: premultiplied composite + rim trim next to bright surfaces."""
    rgb, al = x[..., :3], x[..., 3].copy(); inner = ndimage.binary_erosion(al > 0.5, iterations=band); rim = (al > 0.01) & ~inner; lum = rgb.max(-1)
    iy, ix = ndimage.distance_transform_edt(~inner, return_distances=False, return_indices=True); ref = ndimage.uniform_filter(lum, 5)[iy, ix]
    new = np.where(rim & (ref > 0.5), np.minimum(al, np.clip((lum / np.maximum(ref, 1e-6) - 0.25) / 0.30, 0, 1)), al)
    return np.clip(rgb * (new / np.maximum(al, 1e-6))[..., None] + (1 - new[..., None]) * np.asarray(bg, np.float32), 0, 1), new


def crop_body(im, al, asp, pad=0.04, bg=(1, 1, 1)):
    ys, xs = np.nonzero(al > 0.1); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * asp); hh = ww / asp; Hh, Ww = im.shape[:2]
    P = np.ones((Hh + 2000, Ww + 2000, 3), np.float32) * np.asarray(bg, np.float32); P[1000:1000 + Hh, 1000:1000 + Ww] = im
    Y0, X0 = int(round(cy - hh / 2)) + 1000, int(round(cx - ww / 2)) + 1000
    return P[Y0:Y0 + int(round(hh)), X0:X0 + int(round(ww))], (X0 - 1000, Y0 - 1000, ww, hh)


def fit(t, w_in, h_in, nearest=False):
    Wp, Hp = max(8, int(round(w_in * a.dpi))), max(8, int(round(h_in * a.dpi)))
    return np.asarray(Image.fromarray((np.clip(t, 0, 1) * 255).round().astype(np.uint8)).resize((Wp, Hp), Image.NEAREST if nearest else Image.LANCZOS))


def tile(im, x, y, w, h, ec=GE, lw=0.5, z=3, r=0.03, nearest=False):
    art = ax.imshow(fit(im, w, h, nearest), extent=(x, x + w, y, y + h), zorder=z, interpolation="none")
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, ec=ec, fc="none", zorder=z + 0.1)
    ax.add_patch(box); art.set_clip_path(box); return box


def body_tile(path, x, y, w, h, kind="render", **kw):
    f = {"render": render_on, "premul": premul_on, "straight": straight_on}[kind]
    im, al = f(rgba(path)); t, _ = crop_body(im, al, w / h); return tile(t, x, y, w, h, **kw)


def uv_tile(path, x, y, w, h, **kw):
    x_ = rgba(path); return tile(np.clip(x_[..., :3] * x_[..., 3:] + (1 - x_[..., 3:]) * np.asarray(DARK, np.float32), 0, 1), x, y, w, h, **kw)


# ------------------------------------------------------------------------------------------------ drawing helpers
def label(x, y, s, size=6.0, color=MUTED, ha="center", va="top", **kw): ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, zorder=6, linespacing=1.05, **kw)
def arrow(p0, p1, color=ARW, lw=0.9, ls="-", z=5, ms=6.5, cs=None):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=0.5,head_width=0.28", mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                                 zorder=z, shrinkA=0, shrinkB=0, connectionstyle=cs or "arc3"))
def line(pts, color=ARW, lw=0.9, ls="-", z=5): ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
def box(x, y, w, h, title, sub="", ours=False, ts=6.6, ss=5.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045", lw=0.8, ec=OE if ours else GE, fc=OF if ours else GF, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.06 if sub else 0), title, ha="center", va="center", fontsize=ts, weight="bold", color=INK, zorder=6)
    if sub: ax.text(x + w / 2, y + h / 2 - 0.075, sub, ha="center", va="center", fontsize=ss, color=MUTED, zorder=6, linespacing=1.0)
def encoder(x, y, w, h, ours=False):
    ax.add_patch(Polygon([(x, y - h / 2), (x + w, y - h / 3.4), (x + w, y + h / 3.4), (x, y + h / 2)], closed=True, lw=0.8, ec=OE if ours else GE,
                         fc=OF if ours else GF, zorder=3, joinstyle="round"))
def unet(x, y, w, h, ours=False):
    ax.add_patch(Polygon([(x, y - h / 2), (x + w / 2, y - h / 4.2), (x + w, y - h / 2), (x + w, y + h / 2), (x + w / 2, y + h / 4.2), (x, y + h / 2)], closed=True,
                         lw=0.8, ec=OE if ours else GE, fc=OF if ours else GF, zorder=3, joinstyle="round"))
def band(key, y0, y1, title, sub):
    fc, tc = BAND[key]; ax.add_patch(FancyBboxPatch((0.02, y0), W - 0.04, y1 - y0, boxstyle="round,pad=0,rounding_size=0.07", lw=0, fc=fc, zorder=0))
    ax.text(0.10, y1 - 0.065, title, ha="left", va="top", fontsize=7.6, weight="bold", color=tc, zorder=6)
    tw = ax.text(0.10, y1 - 0.065, title, ha="left", va="top", fontsize=7.6, weight="bold", alpha=0).get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    ax.text(0.10 + tw + 0.06, y1 - 0.072, sub, ha="left", va="top", fontsize=6.6, style="italic", color=MUTED, zorder=6)


def fan(paths, x, y, w, h, dx, kind, ec=GE):
    for i, p in enumerate(paths):
        xi, yi = x + i * dx, y - i * 0.035
        ax.add_patch(Rectangle((xi + 0.012, yi - 0.012), w, h, fc="#000000", alpha=0.10, lw=0, zorder=3 + i * 0.3 - 0.05))
        (uv_tile if kind == "uv" else lambda p_, *q, **k: body_tile(p_, *q, kind=kind, **k))(p, xi, yi, w, h, z=3 + i * 0.3, ec=ec, r=0.02)


# ================================================================================================ bands
YA0, YA1 = 2.60, 3.95; YB0, YB1 = 1.36, 2.53; YC0, YC1 = 0.03, 1.29
band("a", YA0, YA1, "(a) Avatar creation", "one photo, one forward pass")
band("b", YB0, YB1, "(b) Real-time animation", "every frame")
band("c", YC0, YC1, "(c) Streaming appearance update", "background worker, no gradient steps")
src_key = run["source"]; base = E / "base" / ds / clip; dk = meta["drive"]; sk = meta["stream"]
gain = next((c["gain_back_mean"] for c in json.loads(Path(a.cands).read_text()) if c["clip"] == a.clip), None) if Path(a.cands).exists() else None
TH = 0.80; TW = 0.35

# ------------------------------------------------------------------------------------------------ (a) creation
ya = YA0 + 0.185; yc = ya + TH / 2
im = np.asarray(Image.open(next(base.glob(f"images/{src_key}.*"))).convert("RGB")).astype(np.float32) / 255
mk = np.asarray(Image.open(base / "masks" / f"{src_key}.png").convert("L")).astype(np.float32) / 255
t, _ = crop_body(im * mk[..., None] + (1 - mk[..., None]), mk, TW / TH); tile(t, 0.10, ya, TW, TH); label(0.10 + TW / 2, ya - 0.025, "photo $I_s$")
body_tile(A / "mesh_source.png", 0.52, ya, TW, TH, kind="premul", ec=OE, lw=0.9); label(0.52 + TW / 2, ya - 0.025, "SMPL-X fit", color=OE)
arrow((0.90, yc), (1.02, yc))
encoder(1.02, yc, 0.30, 0.66); ax.text(1.17, yc, "DINOv2", ha="center", va="center", rotation=90, fontsize=6.6, weight="bold", color=INK, zorder=6)
ax.text(1.115, ya - 0.062, "❄", ha="center", va="center", fontsize=6.2, color="#3F7FBF", family="DejaVu Sans", zorder=6)
label(1.165, ya - 0.025, "frozen", ha="left")
arrow((1.35, yc), (1.47, yc))
body_tile(A / "feat_pca.png", 1.47, ya, TW, TH, kind="premul"); label(1.47 + TW / 2, ya - 0.025, "features $F$")
# two branches
yt, yb = ya + TH - 0.19, ya + 0.19; xs0 = 1.47 + TW
line([(xs0 + 0.02, yc), (xs0 + 0.10, yc)]); line([(xs0 + 0.10, yb), (xs0 + 0.10, yt)])
arrow((xs0 + 0.10, yt), (xs0 + 0.22, yt)); arrow((xs0 + 0.10, yb), (xs0 + 0.22, yb))
# vertex branch: per-vertex features at the projected vertices
vx0, vw, vh = xs0 + 0.22, 0.20, 0.38; v = np.load(A / "verts.npz"); px, col = v["px"], v["rgb"]
x0_, x1_, y0_, y1_ = px[:, 0].min(), px[:, 0].max(), px[:, 1].min(), px[:, 1].max(); sc = min((vw - 0.03) / (x1_ - x0_), (vh - 0.03) / (y1_ - y0_))
ax.add_patch(FancyBboxPatch((vx0, yt - vh / 2), vw, vh, boxstyle="round,pad=0,rounding_size=0.02", lw=0.5, ec=GE, fc="white", zorder=3))
ax.scatter(vx0 + vw / 2 + (px[:, 0] - (x0_ + x1_) / 2) * sc, yt - (px[:, 1] - (y0_ + y1_) / 2) * sc, s=0.05, c=col, lw=0, zorder=4, rasterized=True)
arrow((vx0 + vw + 0.02, yt), (vx0 + vw + 0.14, yt))
bx = vx0 + vw + 0.14; box(bx, yt - 0.16, 0.86, 0.32, "Vertex decoder", "MLP, 1 Gaussian / vertex")
label(vx0 - 0.02, yt + 0.215, "$F$ sampled at the projected vertices", size=5.6, ha="left", va="bottom")
# UV branch: inverse texture mapping -> partial UV map -> StyleUNet
uw = 0.38; uv_tile(A / "uv_source.png", vx0 - 0.02, yb - uw / 2, uw, uw)
ux_ = vx0 + uw + 0.12; arrow((vx0 - 0.02 + uw + 0.02, yb), (ux_, yb)); unet(ux_, yb, 0.56, 0.32)
ax.text(ux_ + 0.28, yb, "UV decoder", ha="center", va="center", fontsize=5.7, weight="bold", color=INK, zorder=6)
label(ux_ + 0.30, yb - 0.175, "1 Gaussian / texel", size=5.6)
label(vx0 - 0.02 + uw / 2 - 0.04, yb - uw / 2 - 0.02, "UV map", size=5.6)
# avatar
xa = bx + 0.86 + 0.34; line([(bx + 0.86, yt), (xa - 0.12, yt)]); line([(ux_ + 0.56, yb), (xa - 0.12, yb)])
line([(xa - 0.12, yb), (xa - 0.12, yt)]); arrow((xa - 0.12, yc), (xa, yc))
body_tile(D / "orbit_photo_000.png", xa, ya, TW, TH, kind="render", ec=INK, lw=0.9); label(xa + TW / 2 + 0.06, ya - 0.075, "avatar $\\mathcal{G}_s$", color=INK, weight="bold", ha="left", va="center")
# magnifier: schematic 3D Gaussians (crisp, opaque, a few sizes and orientations)
mc = (xa + TW + 0.36, yc + 0.02); mr = 0.27; ax.add_patch(Circle(mc, mr, fc="white", ec=INK, lw=0.9, zorder=6))
rng = np.random.default_rng(11); pal = [("#F1E4D2", "#B79A78"), ("#E6D2B8", "#A88B66"), ("#D8BC98", "#967650"), ("#B98A63", "#7C5636"), ("#8A5A3C", "#55331D"), ("#F7EEE2", "#C2AA8C")]
clipc = Circle(mc, mr - 0.012, transform=ax.transData); pts = []
for _ in range(4000):
    r_, th = mr * 0.80 * np.sqrt(rng.random()), rng.random() * 2 * np.pi; q = (mc[0] + r_ * np.cos(th), mc[1] + r_ * np.sin(th))
    if all((q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2 > 0.112 ** 2 for o in pts): pts.append(q)
for j, q in enumerate(pts):
    fc_, ec_ = pal[j % len(pal)]; wd, ht, ang = 0.115 + 0.05 * rng.random(), 0.052 + 0.03 * rng.random(), rng.random() * 180
    e = Ellipse(q, wd, ht, angle=ang, fc=fc_, ec=ec_, lw=0.45, zorder=6.2); ax.add_patch(e); e.set_clip_path(clipc)
    h_ = Ellipse((q[0] - 0.012 * np.cos(np.radians(ang)), q[1] + 0.008), wd * 0.42, ht * 0.38, angle=ang, fc="white", ec="none", alpha=0.55, zorder=6.3); ax.add_patch(h_); h_.set_clip_path(clipc)
line([(xa + TW * 0.72, yc + 0.10), (mc[0] - mr * 0.93, mc[1] + 0.05)], color=INK, lw=0.5, z=5.5)
ax.add_patch(Circle((xa + TW * 0.66, yc + 0.10), 0.03, fc="none", ec=INK, lw=0.6, zorder=5.6))
label(mc[0] + mr + 0.08, mc[1] + 0.085, "$\\approx$207k 3D Gaussians", size=6.4, color=INK, ha="left", va="center", weight="bold")
label(mc[0] + mr + 0.08, mc[1] - 0.075, "position, scale, rotation,\nopacity, 32-d feature", size=5.8, ha="left", va="center")
# legend
lx, ly = 5.62, YA1 - 0.30
for i, (fc, ec, s_) in enumerate(((GF, GE, "inherited from GUAVA"), (OF, OE, "ours (new or changed)"))):
    ax.add_patch(FancyBboxPatch((lx, ly - i * 0.19), 0.17, 0.11, boxstyle="round,pad=0,rounding_size=0.025", lw=0.8, ec=ec, fc=fc, zorder=3))
    label(lx + 0.23, ly - i * 0.19 + 0.055, s_, size=6.0, color=INK, ha="left", va="center")
ax.text(lx + 0.085, ly - 0.38 + 0.055, "❄", ha="center", va="center", fontsize=7, color="#3F7FBF", family="DejaVu Sans", zorder=6)
label(lx + 0.23, ly - 0.38 + 0.055, "frozen weights", size=6.0, color=INK, ha="left", va="center")

# ------------------------------------------------------------------------------------------------ (b) animation: the avatar spine is x = gx
gx = xa + TW / 2; yb0 = YB0 + 0.185; ybc = yb0 + TH / 2; SW = 0.70; xl = gx - SW / 2
body_tile(A / f"drive_gt_{dk}.png", 0.10, yb0, TW, TH, kind="straight"); label(0.10 + TW / 2, yb0 - 0.025, "live frame")
arrow((0.48, ybc), (0.62, ybc))
# SPRINT (ours): image backbone + SMPL-X regressor
sx0, sw_, sh_ = 0.62, 1.66, 0.66
ax.add_patch(FancyBboxPatch((sx0, ybc - sh_ / 2), sw_, sh_, boxstyle="round,pad=0,rounding_size=0.05", lw=0.9, ec=OE, fc=OF, zorder=3))
ax.text(sx0 + 0.08, ybc + sh_ / 2 - 0.085, "SPRINT pose regressor", ha="left", va="center", fontsize=6.6, weight="bold", color=INK, zorder=6)
ax.add_patch(Polygon([(sx0 + 0.10, ybc - 0.27), (sx0 + 0.50, ybc - 0.20), (sx0 + 0.50, ybc + 0.06), (sx0 + 0.10, ybc + 0.13)], closed=True, lw=0.7, ec=OE, fc="white", zorder=4, joinstyle="round"))
ax.text(sx0 + 0.30, ybc - 0.07, "ViT\nbackbone", ha="center", va="center", fontsize=5.4, color=INK, zorder=6, linespacing=0.95)
arrow((sx0 + 0.52, ybc - 0.07), (sx0 + 0.66, ybc - 0.07), color=OE, lw=0.8)
ax.add_patch(FancyBboxPatch((sx0 + 0.66, ybc - 0.22), 0.88, 0.30, boxstyle="round,pad=0,rounding_size=0.035", lw=0.7, ec=OE, fc="white", zorder=4))
ax.text(sx0 + 1.10, ybc - 0.07, "SMPL-X regressor\npose, expression, camera", ha="center", va="center", fontsize=5.4, color=INK, zorder=6, linespacing=1.0)
arrow((sx0 + sw_ + 0.02, ybc), (sx0 + sw_ + 0.16, ybc)); mx = sx0 + sw_ + 0.16
body_tile(A / f"drive_mesh_{dk}.png", mx, yb0, TW, TH, kind="premul", ec=OE, lw=0.9); label(mx + TW / 2, yb0 - 0.025, "pose $\\theta_t$", color=OE)
arrow((mx + TW + 0.02, ybc), (xl - 0.02, ybc)); box(xl, ybc - 0.19, SW, 0.38, "Skinning", "LBS on SMPL-X")
arrow((gx, ya - 0.115), (gx, ybc + 0.195), lw=1.1)                                                    # avatar from (a), straight down
arrow((xl + SW + 0.02, ybc), (xl + SW + 0.14, ybc))
fx = xl + SW + 0.16; fpc, fal = premul_on(rgba(A / f"drive_featpca_{dk}.png")); ft, _ = crop_body(fpc, fal, TW / TH)
for i, perm in enumerate(((2, 0, 1), (1, 2, 0), (0, 1, 2))):
    tile(ft[..., list(perm)], fx + (2 - i) * 0.045, yb0 + (2 - i) * 0.03 - 0.03, TW, TH - 0.06, z=3 + i * 0.3, r=0.02)
label(fx + TW / 2 + 0.045, yb0 - 0.045, "32-channel raster", size=6.0)
arrow((fx + TW + 0.11, ybc), (fx + TW + 0.23, ybc)); rx = fx + TW + 0.23
unet(rx, ybc, 0.50, 0.40); ax.text(rx + 0.25, ybc, "refine", ha="center", va="center", fontsize=5.6, weight="bold", color=INK, zorder=6)
label(rx + 0.25, ybc - 0.24, "StyleUNet refiner", size=5.6)
arrow((rx + 0.52, ybc), (rx + 0.64, ybc)); ox = rx + 0.64
body_tile(A / f"drive_out_{dk}.png", ox, yb0, TW, TH, kind="render", ec=INK, lw=0.9); label(ox + TW / 2, yb0 - 0.025, "output $\\hat{I}_t$", color=INK, weight="bold")
for i, f_ in enumerate((D / "anim_photo_frame_000276.png", D / "orbit_photo_000.png")):
    if f_.exists(): body_tile(f_, ox + (i + 1) * (TW + 0.06), yb0, TW, TH, kind="render", lw=0.4)
label(ox + 1.5 * (TW + 0.06) + TW / 2, yb0 - 0.025, "other frames")

# ------------------------------------------------------------------------------------------------ (c) streaming update
yc0 = YC0 + 0.20; ycc = yc0 + TH / 2
fan([A / f"stream_frame_{k}.png" for k in sk], 0.10, yc0 + 0.07, 0.28, TH - 0.07, 0.13, "straight"); label(0.10 + 0.27, yc0 - 0.03, "video frames")
arrow((0.68, ycc), (0.79, ycc))
gx0, gw, gh = 0.79, 0.82, 0.74; ax.add_patch(FancyBboxPatch((gx0, ycc - gh / 2), gw, gh, boxstyle="round,pad=0,rounding_size=0.045", lw=0.8, ec=OE, fc=OF, zorder=3))
ax.text(gx0 + gw / 2, ycc + gh / 2 - 0.085, "Frame gates", ha="center", va="center", fontsize=6.6, weight="bold", color=INK, zorder=6)
for i, s_ in enumerate(("sharp image", "pose fits keypoints", "surface faces camera", "not an outlier frame")):
    yy = ycc + 0.13 - i * 0.125
    ax.text(gx0 + 0.065, yy, "✓", ha="center", va="center", fontsize=5.6, color="#2F8F46", family="DejaVu Sans", weight="bold", zorder=6)
    ax.text(gx0 + 0.12, yy, s_, ha="left", va="center", fontsize=5.5, color=INK, zorder=6)
arrow((gx0 + gw + 0.02, ycc), (gx0 + gw + 0.12, ycc)); ux = gx0 + gw + 0.12
fan([A / f"stream_uv_{k}.png" for k in sk], ux, yc0 + 0.23, 0.44, 0.44, 0.11, "uv"); label(ux + 0.33, yc0 + 0.06, "UV observations $U_t$,\nweights $w_t$", size=5.8)
arrow((ux + 0.69, ycc), (ux + 0.79, ycc)); sx = ux + 0.92
ax.add_patch(Circle((sx, ycc), 0.125, fc=OF, ec=OE, lw=0.9, zorder=3)); ax.text(sx, ycc - 0.005, "$\\Sigma$", ha="center", va="center", fontsize=8.6, color=INK, zorder=6)
label(sx, ycc + 0.16, "weighted\nmean", size=5.6, va="bottom", color=INK)
label(sx, ycc - 0.16, "$\\bar{U}=S/W$", size=6.0)
fw = 0.56; dw = 0.46; dx2 = gx - dw / 2; fx2 = dx2 - 0.13 - fw; arrow((sx + 0.13, ycc), (fx2 - 0.02, ycc))
fused = np.asarray(Image.open(D / "uv_rgb_fused.png").convert("RGB")).astype(np.float32) / 255; m_ = fused.max(-1, keepdims=True) > 0.004
tile(np.where(m_, fused, np.asarray(DARK, np.float32)), fx2, ycc - fw / 2, fw, fw, ec=OE, lw=0.9)
label(fx2 + fw / 2, ycc - fw / 2 - 0.025, "fused UV map", size=5.8, color=OE, weight="bold")
arrow((fx2 + fw + 0.02, ycc), (dx2 - 0.01, ycc))
unet(dx2, ycc, dw, 0.32); ax.text(gx, ycc, "UV", ha="center", va="center", fontsize=5.6, weight="bold", color=INK, zorder=6)
label(gx, ycc - 0.175, "same UV decoder,\nre-run once", size=5.6)
# updated avatar: straight up the spine into Skinning (ours), and shown on the unseen back
arrow((gx, ycc + 0.175), (gx, ybc - 0.195), color=OE, lw=1.4, ms=8)
label(gx + 0.07, YC1 - 0.085, "updated avatar replaces $\\mathcal{G}_s$,", size=6.0, color=OE, ha="left", va="center", weight="bold")
label(gx + 0.07, YC1 - 0.185, "rendering continues", size=6.0, color=OE, ha="left", va="center", style="italic")
bw = 0.60; bx2 = dx2 + dw + 0.16; arrow((dx2 + dw + 0.02, ycc), (bx2 - 0.02, ycc)); z = f"zoom{a.zoom}"; yt2 = ycc - bw / 2 - 0.045
for i, (kind, lab, ec, lw) in enumerate((("photo", "photo only", GE, 0.5), ("update", "after the update", OE, 1.0), ("gt", "real frame", GE, 0.5))):
    x_ = rgba(D / f"back_{z}_{kind}_{a.back_key}.png"); im_ = (straight_on(x_) if kind == "gt" else render_on(x_))[0]; n = im_.shape[0]; c0 = int(0.20 * n)
    tile(im_[: n - 2 * c0, c0: n - c0], bx2 + i * (bw + 0.06), yt2, bw, bw, ec=ec, lw=lw)
    label(bx2 + i * (bw + 0.06) + bw / 2, yt2 - 0.02, lab, size=5.6, color=OE if i == 1 else MUTED, weight="bold" if i == 1 else "normal")
if gain is not None:
    ax.text(bx2 + 3 * (bw + 0.06) + 0.04, ycc + 0.06, f"+{gain:.1f} dB", ha="left", va="center", fontsize=9.5, weight="bold", color=OE, zorder=6)
    label(bx2 + 3 * (bw + 0.06) + 0.04, ycc - 0.10, "PSNR, back views\nof this clip", size=5.6, color=OE, ha="left", va="center")
fig.savefig(OUT / f"{a.name}.pdf", dpi=a.dpi); fig.savefig(OUT / f"{a.name}.png", dpi=300)
print("wrote", OUT / f"{a.name}.pdf", f"{W:.2f} x {H:.2f} in")
