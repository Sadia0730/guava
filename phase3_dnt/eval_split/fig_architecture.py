"""Architecture figure, v4 (PREREG "Extended benchmark", amendment 4 (c), presentation only): the three stages drawn from the
code (models/UbodyAvatar/ubody_gaussian.py, guava_fb/fb_common.py, worker6/uv_fusion.py), every thumbnail an image of the model's
own tensors for one example (teaser_render.py --arch-key: SMPL-X mesh, DINO features (PCA), per-vertex features, UV maps, the
32-channel feature raster (PCA), renders). Modules are coloured by stage (creation blue, animation green, update orange; the UV
decoder keeps its creation colour where the update re-runs it); no "ours / inherited" legend in the drawing: the credit to GUAVA
for the creation and rendering design goes in the caption and the method text. The avatar sits on one vertical line:
creation -> skinning <- updated avatar. No timings; the only number is the clip's back-view PSNR gain (teaser_candidates.json).
Every element is drawn through T / RECT / POLY / SEG / ELL / IMG, which also record it; --export DIR writes layout.json + the
thumbnails for fig_architecture_pptx.py (an editable PowerPoint version of the same layout).
Usage: python fig_architecture.py --E <root> --clip ubc/A1YNmKj0sCS --renders /dev/shm/ubx858_eval/teaser_renders --back-key frame_000232
       --out <dir> [--export <dir>]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                      # noqa: E402
import numpy as np                                                                                   # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon                    # noqa: E402
from PIL import Image                                                                                # noqa: E402
from scipy import ndimage                                                                            # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--clip", required=True)
ap.add_argument("--renders", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", default="fig_architecture")
ap.add_argument("--back-key", required=True); ap.add_argument("--zoom", default="1.4"); ap.add_argument("--dpi", type=int, default=600)
ap.add_argument("--cands", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/paper_figures/teaser_candidates.json",
                help="teaser_candidates.py output: mean PSNR gain of the update on the clip's back-view test frames")
ap.add_argument("--export", default="", help="directory for layout.json + thumbnails (input of fig_architecture_pptx.py)")
a = ap.parse_args(); ds, clip = a.clip.split("/"); D = Path(a.renders) / ds / clip; A = D / "arch"; E = Path(a.E)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); meta = json.loads((A / "arch.json").read_text()); run = json.loads((D / "run.json").read_text())
EXP = Path(a.export) if a.export else None
if EXP: (EXP / "img").mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 6.4, "pdf.fonttype": 42})
INK, MUTED, ARW = "#1f1f1f", "#5b6068", "#3b3f45"
GE = "#8C939D"                                                                                       # thumbnail borders
PAL = {"a": ("#E4ECF7", "#5E82B5"), "b": ("#E2F0E4", "#4F9561"), "c": ("#FCE3CC", "#D9771E")}        # module fill / edge per stage
OE = PAL["c"][1]
BAND = {"a": ("#F4F7FB", "#33507A"), "b": ("#F2F8F2", "#2F6B3A"), "c": ("#FEF5EC", "#B4590E")}
DARK = (0.105, 0.115, 0.135); BLUE = "#3F7FBF"; GREEN = "#2F8F46"
W, H = 6.875, 3.98
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
PR = []                                                                                              # recorded primitives (export)


# ------------------------------------------------------------------------------------------------ primitives (draw + record)
def T(x, y, s, size=6.0, color=MUTED, ha="center", va="top", bold=False, italic=False, rot=0, family=None, ls=1.05):
    kw = {"family": family} if family else {}
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, zorder=6, linespacing=ls, weight="bold" if bold else "normal",
            style="italic" if italic else "normal", rotation=rot, **kw)
    PR.append(dict(t="text", z=6, x=x, y=y, s=s, size=size, color=color, ha=ha, va=va, bold=bold, italic=italic, rot=rot, family=family))


def RECT(x, y, w, h, fc, ec, lw, r=0.045, z=3.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, ec=ec if lw else "none", fc=fc, zorder=z))
    PR.append(dict(t="rect", z=z, x=x, y=y, w=w, h=h, fc=fc, ec=ec, lw=lw, r=r))


def POLY(pts, fc, ec, lw=0.8, z=3.0):
    ax.add_patch(Polygon(pts, closed=True, lw=lw, ec=ec, fc=fc, zorder=z, joinstyle="round")); PR.append(dict(t="poly", z=z, pts=[list(p) for p in pts], fc=fc, ec=ec, lw=lw))


def SEG(pts, color=ARW, lw=0.9, arrow=True, z=5.0, ms=6.5):
    """polyline; arrow: head at the last point."""
    body = pts if not arrow else pts[:-1]
    if len(body) > 1: ax.plot([p[0] for p in body], [p[1] for p in body], color=color, lw=lw, zorder=z, solid_capstyle="round")
    if arrow:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=0.5,head_width=0.28", mutation_scale=ms, lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))
    PR.append(dict(t="line", z=z, pts=[list(p) for p in pts], color=color, lw=lw, arrow=arrow))


def ELL(cx, cy, w, h, angle=0.0, fc="white", ec=INK, lw=0.8, z=6.0):
    ax.add_patch(Ellipse((cx, cy), w, h, angle=angle, fc=fc, ec=ec if lw else "none", lw=lw, zorder=z)); PR.append(dict(t="ell", z=z, cx=cx, cy=cy, w=w, h=h, angle=angle, fc=fc, ec=ec, lw=lw))


def IMG(im, x, y, w, h, ec=GE, lw=0.5, z=3.0, r=0.03, nearest=False):
    Wp, Hp = max(8, int(round(w * a.dpi))), max(8, int(round(h * a.dpi)))
    px = Image.fromarray((np.clip(im, 0, 1) * 255).round().astype(np.uint8)).resize((Wp, Hp), Image.NEAREST if nearest else Image.LANCZOS)
    art = ax.imshow(np.asarray(px), extent=(x, x + w, y, y + h), zorder=z, interpolation="none")
    box_ = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", lw=lw, ec=ec, fc="none", zorder=z + 0.1); ax.add_patch(box_); art.set_clip_path(box_)
    f = None
    if EXP: f = f"img/t{sum(p['t'] == 'img' for p in PR):02d}.png"; px.save(EXP / f)
    PR.append(dict(t="img", z=z, x=x, y=y, w=w, h=h, ec=ec, lw=lw, file=f))


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
    return P[Y0:Y0 + int(round(hh)), X0:X0 + int(round(ww))]


def body_tile(path, x, y, w, h, kind="render", **kw):
    im, al = {"render": render_on, "premul": premul_on, "straight": straight_on}[kind](rgba(path)); IMG(crop_body(im, al, w / h), x, y, w, h, **kw)


def uv_tile(path, x, y, w, h, **kw):
    x_ = rgba(path); IMG(np.clip(x_[..., :3] * x_[..., 3:] + (1 - x_[..., 3:]) * np.asarray(DARK, np.float32), 0, 1), x, y, w, h, **kw)


# ------------------------------------------------------------------------------------------------ composite helpers
def box(x, y, w, h, title, sub="", k="a", ts=6.6, ss=5.5):
    RECT(x, y, w, h, PAL[k][0], PAL[k][1], 0.8)
    T(x + w / 2, y + h / 2 + (0.06 if sub else 0), title, size=ts, color=INK, va="center", bold=True)
    if sub: T(x + w / 2, y + h / 2 - 0.075, sub, size=ss, va="center", ls=1.0)


def encoder(x, y, w, h, k="a"): POLY([(x, y - h / 2), (x + w, y - h / 3.4), (x + w, y + h / 3.4), (x, y + h / 2)], PAL[k][0], PAL[k][1], 0.8, 3.0)
def unet(x, y, w, h, k="a"): POLY([(x, y - h / 2), (x + w / 2, y - h / 4.2), (x + w, y - h / 2), (x + w, y + h / 2), (x + w / 2, y + h / 4.2), (x, y + h / 2)], PAL[k][0], PAL[k][1])


def band(key, y0, y1, title, sub):
    fc, tc = BAND[key]; RECT(0.02, y0, W - 0.04, y1 - y0, fc, fc, 0, r=0.07, z=0.0)
    tw = ax.text(0.10, y1 - 0.065, title, ha="left", va="top", fontsize=7.6, weight="bold", alpha=0).get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    T(0.10, y1 - 0.065, title, size=7.6, color=tc, ha="left", bold=True); T(0.10 + tw + 0.06, y1 - 0.072, sub, size=6.6, ha="left", italic=True)


def fan(paths, x, y, w, h, dx, kind, ec=GE):
    for i, p in enumerate(paths):
        xi, yi = x + i * dx, y - i * 0.035
        (uv_tile if kind == "uv" else lambda p_, *q, **k: body_tile(p_, *q, kind=kind, **k))(p, xi, yi, w, h, z=3 + i * 0.3, ec=ec, lw=0.6, r=0.02)


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
IMG(crop_body(im * mk[..., None] + (1 - mk[..., None]), mk, TW / TH), 0.10, ya, TW, TH); T(0.10 + TW / 2, ya - 0.025, "photo $I_s$")
body_tile(A / "mesh_source.png", 0.52, ya, TW, TH, kind="premul"); T(0.52 + TW / 2, ya - 0.025, "SMPL-X fit")
SEG([(0.90, yc), (1.02, yc)])
encoder(1.02, yc, 0.30, 0.66); T(1.17, yc, "DINOv2", size=6.6, color=INK, va="center", bold=True, rot=90)
T(1.115, ya - 0.062, "❄", size=6.2, color=BLUE, va="center", family="DejaVu Sans"); T(1.165, ya - 0.025, "frozen", ha="left")
SEG([(1.35, yc), (1.47, yc)])
body_tile(A / "feat_pca.png", 1.47, ya, TW, TH, kind="premul"); T(1.47 + TW / 2, ya - 0.025, "features $F$")
# two branches
yt, yb = ya + TH - 0.19, ya + 0.19; xs0 = 1.47 + TW
SEG([(xs0 + 0.02, yc), (xs0 + 0.10, yc), (xs0 + 0.10, yt), (xs0 + 0.22, yt)]); SEG([(xs0 + 0.10, yc), (xs0 + 0.10, yb), (xs0 + 0.22, yb)])
# vertex branch: per-vertex features at the projected vertices (drawn as dots into a small image)
vx0, vw, vh = xs0 + 0.22, 0.20, 0.38; v = np.load(A / "verts.npz"); px, col = v["px"], v["rgb"]; S_ = 1500
x0_, x1_, y0_, y1_ = px[:, 0].min(), px[:, 0].max(), px[:, 1].min(), px[:, 1].max(); sc = min((vw - 0.03) / (x1_ - x0_), (vh - 0.03) / (y1_ - y0_)) * S_
dots = np.ones((int(vh * S_), int(vw * S_), 3), np.float32)
u = np.clip((vw * S_ / 2 + (px[:, 0] - (x0_ + x1_) / 2) * sc).astype(int), 1, dots.shape[1] - 3); w_ = np.clip((vh * S_ / 2 + (px[:, 1] - (y0_ + y1_) / 2) * sc).astype(int), 1, dots.shape[0] - 3)
for du in (0, 1):
    for dv in (0, 1): dots[w_ + dv, u + du] = col
IMG(dots, vx0, yt - vh / 2, vw, vh, r=0.02)
SEG([(vx0 + vw + 0.02, yt), (vx0 + vw + 0.14, yt)])
bx = vx0 + vw + 0.14; box(bx, yt - 0.16, 0.86, 0.32, "Vertex decoder", "MLP, 1 Gaussian / vertex")
T(vx0 - 0.02, yt + 0.215, "$F$ sampled at the projected vertices", size=5.6, ha="left", va="bottom")
# UV branch: inverse texture mapping -> partial UV map -> StyleUNet
uw = 0.38; uv_tile(A / "uv_source.png", vx0 - 0.02, yb - uw / 2, uw, uw)
ux_ = vx0 + uw + 0.12; SEG([(vx0 - 0.02 + uw + 0.02, yb), (ux_, yb)]); unet(ux_, yb, 0.56, 0.32)
T(ux_ + 0.28, yb, "UV decoder", size=5.7, color=INK, va="center", bold=True)
T(ux_ + 0.30, yb - 0.175, "1 Gaussian / texel", size=5.6); T(vx0 - 0.02 + uw / 2 - 0.04, yb - uw / 2 - 0.02, "UV map", size=5.6)
# avatar
xa = bx + 0.86 + 0.34; SEG([(bx + 0.86, yt), (xa - 0.12, yt), (xa - 0.12, yc)], arrow=False); SEG([(ux_ + 0.56, yb), (xa - 0.12, yb), (xa - 0.12, yc), (xa, yc)])
body_tile(D / "orbit_photo_000.png", xa, ya, TW, TH, kind="render", ec=INK, lw=0.9); T(xa + TW / 2 + 0.06, ya - 0.075, "avatar $\\mathcal{G}_s$", color=INK, bold=True, ha="left", va="center")
# magnifier: schematic 3D Gaussians (crisp, opaque, all inside the lens)
mc = (xa + TW + 0.50, yc + 0.02); mr = 0.33; SEG([(xa + TW * 0.72, yc + 0.10), (mc[0] - mr * 0.93, mc[1] + 0.05)], color=INK, lw=0.5, arrow=False, z=5.5)
ELL(xa + TW * 0.66, yc + 0.10, 0.06, 0.06, fc="none", ec=INK, lw=0.6, z=5.6); ELL(mc[0], mc[1], 2 * mr, 2 * mr, fc="white", ec=INK, lw=0.9, z=6.0)
rng = np.random.default_rng(11); pal = [("#F1E4D2", "#B79A78"), ("#E6D2B8", "#A88B66"), ("#D8BC98", "#967650"), ("#B98A63", "#7C5636"), ("#8A5A3C", "#55331D"), ("#F7EEE2", "#C2AA8C")]
pts = []
for _ in range(6000):
    r_, th = (mr - 0.10) * np.sqrt(rng.random()), rng.random() * 2 * np.pi; q = (mc[0] + r_ * np.cos(th), mc[1] + r_ * np.sin(th))
    if all((q[0] - o[0]) ** 2 + (q[1] - o[1]) ** 2 > 0.118 ** 2 for o in pts): pts.append(q)
for j, q in enumerate(pts):
    fc_, ec_ = pal[j % len(pal)]; wd, ht, ang = 0.125 + 0.045 * rng.random(), 0.056 + 0.03 * rng.random(), rng.random() * 180
    hi = "#" + "".join(f"{int(0.45 * int(fc_[k:k + 2], 16) + 0.55 * 255):02X}" for k in (1, 3, 5))
    ELL(q[0], q[1], wd, ht, ang, fc_, ec_, 0.45, 6.2); ELL(q[0] - 0.012 * np.cos(np.radians(ang)), q[1] + 0.008, wd * 0.42, ht * 0.38, ang, hi, hi, 0, 6.3)
T(mc[0] + mr + 0.12, mc[1] + 0.10, "$\\approx$207k 3D Gaussians", size=7.4, color=INK, ha="left", va="center", bold=True)
T(mc[0] + mr + 0.12, mc[1] - 0.085, "each with position, scale, rotation,\nopacity and a 32-d appearance feature", size=6.0, ha="left", va="center")

# ------------------------------------------------------------------------------------------------ (b) animation: the avatar spine is x = gx
gx = xa + TW / 2; yb0 = YB0 + 0.185; ybc = yb0 + TH / 2; SW = 0.70; xl = gx - SW / 2
body_tile(A / f"drive_gt_{dk}.png", 0.10, yb0, TW, TH, kind="straight"); T(0.10 + TW / 2, yb0 - 0.025, "live frame")
SEG([(0.48, ybc), (0.62, ybc)])
sx0, sw_, sh_ = 0.62, 1.66, 0.66; PB = PAL["b"]; RECT(sx0, ybc - sh_ / 2, sw_, sh_, PB[0], PB[1], 0.9, r=0.05)   # pose regressor: backbone + SMPL-X regressor
T(sx0 + 0.08, ybc + sh_ / 2 - 0.085, "Pose regressor", size=6.6, color=INK, ha="left", va="center", bold=True)
POLY([(sx0 + 0.10, ybc - 0.27), (sx0 + 0.50, ybc - 0.20), (sx0 + 0.50, ybc + 0.06), (sx0 + 0.10, ybc + 0.13)], "white", PB[1], 0.7, 4.0)
T(sx0 + 0.30, ybc - 0.07, "ViT\nbackbone", size=5.4, color=INK, va="center", ls=0.95)
SEG([(sx0 + 0.52, ybc - 0.07), (sx0 + 0.66, ybc - 0.07)], color=PB[1], lw=0.8)
RECT(sx0 + 0.66, ybc - 0.22, 0.88, 0.30, "white", PB[1], 0.7, r=0.035, z=4.0); T(sx0 + 1.10, ybc - 0.07, "SMPL-X regressor\npose, expression, camera", size=5.4, color=INK, va="center", ls=1.0)
SEG([(sx0 + sw_ + 0.02, ybc), (sx0 + sw_ + 0.16, ybc)]); mx = sx0 + sw_ + 0.16
body_tile(A / f"drive_mesh_{dk}.png", mx, yb0, TW, TH, kind="premul"); T(mx + TW / 2, yb0 - 0.025, "pose $\\theta_t$")
SEG([(mx + TW + 0.02, ybc), (xl - 0.02, ybc)]); box(xl, ybc - 0.19, SW, 0.38, "Skinning", "LBS on SMPL-X", k="b")
SEG([(gx, ya - 0.115), (gx, ybc + 0.195)], lw=1.1)                                                     # avatar from (a), straight down
SEG([(xl + SW + 0.02, ybc), (xl + SW + 0.14, ybc)])
fx = xl + SW + 0.16; fpc, fal = premul_on(rgba(A / f"drive_featpca_{dk}.png")); ft = crop_body(fpc, fal, TW / TH)
for i, perm in enumerate(((2, 0, 1), (1, 2, 0), (0, 1, 2))): IMG(ft[..., list(perm)], fx + (2 - i) * 0.045, yb0 + (2 - i) * 0.03 - 0.03, TW, TH - 0.06, z=3 + i * 0.3, r=0.02)
T(fx + TW / 2 + 0.045, yb0 - 0.045, "32-channel raster", size=6.0)
SEG([(fx + TW + 0.11, ybc), (fx + TW + 0.23, ybc)]); rx = fx + TW + 0.23
unet(rx, ybc, 0.50, 0.40, k="b"); T(rx + 0.25, ybc, "refine", size=5.6, color=INK, va="center", bold=True); T(rx + 0.25, ybc - 0.24, "StyleUNet refiner", size=5.6)
SEG([(rx + 0.52, ybc), (rx + 0.64, ybc)]); ox = rx + 0.64
body_tile(A / f"drive_out_{dk}.png", ox, yb0, TW, TH, kind="render", ec=INK, lw=0.9); T(ox + TW / 2, yb0 - 0.025, "output $\\hat{I}_t$", color=INK, bold=True)
for i, f_ in enumerate((D / "anim_photo_frame_000276.png", D / "orbit_photo_000.png")):
    if f_.exists(): body_tile(f_, ox + (i + 1) * (TW + 0.06), yb0, TW, TH, kind="render", lw=0.4)
T(ox + 1.5 * (TW + 0.06) + TW / 2, yb0 - 0.025, "other frames")

# ------------------------------------------------------------------------------------------------ (c) streaming update
yc0 = YC0 + 0.20; ycc = yc0 + TH / 2
fan([A / f"stream_frame_{k}.png" for k in sk], 0.10, yc0 + 0.07, 0.28, TH - 0.07, 0.12, "straight"); T(0.10 + 0.26, yc0 - 0.03, "video frames")
SEG([(0.64, ycc), (0.74, ycc)])
gx0, gw, gh = 0.74, 0.80, 0.74; RECT(gx0, ycc - gh / 2, gw, gh, PAL["c"][0], PAL["c"][1], 0.8)
T(gx0 + gw / 2, ycc + gh / 2 - 0.085, "Frame gates", size=6.6, color=INK, va="center", bold=True)
for i, s_ in enumerate(("sharp image", "pose fits keypoints", "surface faces camera", "not an outlier")):
    yy = ycc + 0.13 - i * 0.125; T(gx0 + 0.065, yy, "✓", size=5.6, color=GREEN, va="center", bold=True, family="DejaVu Sans"); T(gx0 + 0.12, yy, s_, size=5.5, color=INK, ha="left", va="center")
SEG([(gx0 + gw + 0.02, ycc), (gx0 + gw + 0.12, ycc)]); ux = gx0 + gw + 0.12
fan([A / f"stream_uv_{k}.png" for k in sk], ux, yc0 + 0.23, 0.42, 0.42, 0.10, "uv"); T(ux + 0.31, yc0 + 0.075, "UV observations $U_t$,\nweights $w_t$", size=5.8)
SEG([(ux + 0.65, ycc), (ux + 0.75, ycc)]); sx = ux + 0.88
ELL(sx, ycc, 0.25, 0.25, fc=PAL["c"][0], ec=OE, lw=0.9, z=3.0); T(sx, ycc - 0.005, "$\\Sigma$", size=8.6, color=INK, va="center")
T(sx, ycc + 0.16, "weighted\nmean", size=5.6, va="bottom", color=INK); T(sx, ycc - 0.16, "over all\nframes", size=5.6)
fw = 0.54; dw = 0.56; dx2 = gx - dw / 2; fx2 = dx2 - 0.13 - fw; SEG([(sx + 0.135, ycc), (fx2 - 0.02, ycc)])
fused = np.asarray(Image.open(D / "uv_rgb_fused.png").convert("RGB")).astype(np.float32) / 255; m_ = fused.max(-1, keepdims=True) > 0.004
IMG(np.where(m_, fused, np.asarray(DARK, np.float32)), fx2, ycc - fw / 2, fw, fw, ec=OE, lw=0.9); T(fx2 + fw / 2, ycc - fw / 2 - 0.025, "fused UV map", size=5.8, color=OE, bold=True)
SEG([(fx2 + fw + 0.02, ycc), (dx2 - 0.01, ycc)]); unet(dx2, ycc, dw, 0.32); T(gx, ycc, "UV decoder", size=5.7, color=INK, va="center", bold=True)
# updated avatar: straight up the spine into Skinning (ours), and shown on the unseen back
SEG([(gx, ycc + 0.175), (gx, ybc - 0.195)], color=OE, lw=1.4, ms=8)
T(gx + 0.07, YC1 - 0.085, "updated avatar replaces $\\mathcal{G}_s$,", size=6.0, color=OE, ha="left", va="center", bold=True)
T(gx + 0.07, YC1 - 0.185, "rendering continues", size=6.0, color=OE, ha="left", va="center", italic=True)
bw = 0.60; bx2 = dx2 + dw + 0.16; SEG([(dx2 + dw + 0.02, ycc), (bx2 - 0.02, ycc)]); z = f"zoom{a.zoom}"; yt2 = ycc - bw / 2 - 0.045
for i, (kind, lab, ec, lw) in enumerate((("photo", "photo only", GE, 0.5), ("update", "after the update", OE, 1.0), ("gt", "real frame", GE, 0.5))):
    x_ = rgba(D / f"back_{z}_{kind}_{a.back_key}.png"); im_ = (straight_on(x_) if kind == "gt" else render_on(x_))[0]; n = im_.shape[0]; c0 = int(0.20 * n)
    IMG(im_[: n - 2 * c0, c0: n - c0], bx2 + i * (bw + 0.06), yt2, bw, bw, ec=ec, lw=lw)
    T(bx2 + i * (bw + 0.06) + bw / 2, yt2 - 0.02, lab, size=5.6, color=OE if i == 1 else MUTED, bold=i == 1)
if gain is not None:
    T(bx2 + 3 * (bw + 0.06) + 0.04, ycc + 0.06, f"+{gain:.1f} dB", size=9.5, color=OE, ha="left", va="center", bold=True)
    T(bx2 + 3 * (bw + 0.06) + 0.04, ycc - 0.10, "PSNR, back views\nof this clip", size=5.6, color=OE, ha="left", va="center")
fig.savefig(OUT / f"{a.name}.pdf", dpi=a.dpi); fig.savefig(OUT / f"{a.name}.png", dpi=300)
if EXP: (EXP / "layout.json").write_text(json.dumps({"W": W, "H": H, "prims": PR}, indent=1))
print("wrote", OUT / f"{a.name}.pdf", f"{W:.2f} x {H:.2f} in", f"; {len(PR)} elements" + (f" exported to {EXP}" if EXP else ""))
