"""Architecture diagram for the paper (PREREG "Extended benchmark, amendment 4 (c)", presentation only). Drawn from the code:
models/UbodyAvatar/ubody_gaussian.py (inferer: DINOv2, vertex branch, UV branch, decoders), fb_common.py (SMPL-X 2020 native head,
per-sample camera), worker6/uv_fusion.py (streaming update: gates, weights, running mean, prior 2, background decode). Inherited
GUAVA parts are grey, our changes are orange. Thumbnails are real images of one clip (teaser_render.py outputs + base images).
Writes <out>/fig_architecture.{pdf,png} (vector PDF, full text width 6.875 in).
Usage: python fig_architecture.py --renders /dev/shm/ubx858_eval/teaser_renders/peoplesnapshot/female-3-casual --base <E>/base/peoplesnapshot/female-3-casual --out <dir>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                      # noqa: E402
import numpy as np                                                                                   # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle                           # noqa: E402
from PIL import Image                                                                                # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--renders", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); R = Path(a.renders); BASE = Path(a.base); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1f1f1f"; MUTED = "#5a5a5a"
GUAVA_F, GUAVA_E = "#ECECEC", "#9A9A9A"                                                              # inherited
OURS_F, OURS_E = "#FCE3CC", "#D9771E"                                                                # ours (new / changed)
BAND = {"create": "#F5F8FC", "anim": "#F6FAF5", "stream": "#FDF8F1"}
W, H = 6.875, 3.45
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, txt, ours=False, fs=6.3, bold_first=True, dashed=False, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", lw=0.7, ls="--" if dashed else "-",
                                fc=OURS_F if ours else GUAVA_F, ec=OURS_E if ours else GUAVA_E, zorder=2))
    lines = txt.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w / 2 if align == "center" else x + 0.05, y + h - 0.045, lines[0], ha=align, va="top", fontsize=fs, weight="bold", color=INK, zorder=3)
        ax.text(x + w / 2 if align == "center" else x + 0.05, y + h - 0.045 - fs / 72 * 1.25, "\n".join(lines[1:]), ha=align, va="top",
                fontsize=fs - 0.6, color=MUTED, zorder=3, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold", color=INK, zorder=3)
    return (x, y, w, h)


def arrow(p, q, color=INK, ls="-", lw=0.8, rad=0.0, head=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={head / 10 * 0.9},head_width={head / 10 * 0.55}", mutation_scale=6,
                                 lw=lw, ls=ls, color=color, connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0))


def poly(pts, color=INK, ls="-", lw=0.8):
    """polyline whose last segment ends in an arrow head."""
    xs, ys = zip(*pts[:-1]); ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=4, solid_capstyle="butt")
    arrow(pts[-2], pts[-1], color=color, ls=ls, lw=lw)


def img(path, x, y, w, h=None, white=True, frame=True):
    im = Image.open(path); im = np.asarray(im.convert("RGBA")).astype(np.float32) / 255
    if white: im = im[..., :3] * im[..., 3:] + (1 - im[..., 3:])
    else: im = im[..., :3]
    if h is None: h = w * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(x, x + w, y, y + h), zorder=3, interpolation="lanczos")
    if frame: ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.4, ec="#7a7a7a", zorder=5))
    return (x, y, w, h)


def crop_person(path, aspect=0.62, pad=0.06):
    """RGBA render -> crop to the alpha box (pad), fixed aspect w/h; returns a temp PNG path (white background)."""
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255; al = im[..., 3]
    ys, xs = np.nonzero(al > 0.05); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2; hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * aspect); hh = ww / aspect
    Y0, X0 = int(round(cy - hh / 2)), int(round(cx - ww / 2)); Y1, X1 = Y0 + int(round(hh)), X0 + int(round(ww))
    P = np.zeros((im.shape[0] + 2000, im.shape[1] + 2000, 4), np.float32); P[1000:1000 + im.shape[0], 1000:1000 + im.shape[1]] = im
    c = P[Y0 + 1000:Y1 + 1000, X0 + 1000:X1 + 1000]; rgb = c[..., :3] * c[..., 3:] + (1 - c[..., 3:])
    p = OUT / "_tmp" / (Path(path).stem + "_crop.png"); p.parent.mkdir(exist_ok=True); Image.fromarray((rgb * 255).round().astype(np.uint8)).save(p); return p


def crop_photo(path, aspect=0.62, pad=0.06):
    """the photo (with its background) cropped to the person's matte box (masks/<key>.png), fixed aspect w/h."""
    im = np.asarray(Image.open(path).convert("RGB")); m = np.asarray(Image.open(path.parent.parent / "masks" / (path.stem + ".png")).convert("L")) > 127
    from scipy import ndimage                                                                        # crop padding -> white
    lab, _ = ndimage.label(im.max(-1) < 15); edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    im = np.where(ndimage.binary_dilation(np.isin(lab, edge[edge > 0]), iterations=4)[..., None], 255, im).astype(np.uint8)
    ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    hh = (y1 - y0) * (1 + 2 * pad); ww = max((x1 - x0) * (1 + 2 * pad), hh * aspect); hh = ww / aspect
    H0, W0 = im.shape[:2]; Y0, X0 = int(round(cy - hh / 2)), int(round(cx - ww / 2))
    if hh <= H0: Y0 = int(np.clip(Y0, 0, H0 - round(hh)))                                           # stay inside the photo
    if ww <= W0: X0 = int(np.clip(X0, 0, W0 - round(ww)))
    P = np.full((H0 + 4000, W0 + 4000, 3), 255, np.uint8); P[2000:2000 + H0, 2000:2000 + W0] = im
    c = P[Y0 + 2000:Y0 + 2000 + int(round(hh)), X0 + 2000:X0 + 2000 + int(round(ww))]
    p = OUT / "_tmp" / (path.stem + "_photo.png"); p.parent.mkdir(exist_ok=True); Image.fromarray(c).save(p); return p


def label(x, y, s, fs=6.0, color=MUTED, ha="center", style="normal", weight="normal"):
    ax.text(x, y, s, ha=ha, va="top", fontsize=fs, color=color, style=style, weight=weight, zorder=6)


run = json.loads((R / "run.json").read_text()); fj = json.loads((BASE / "frames.json").read_text())
src_img = next(BASE.glob("images/" + fj["source"] + ".*")); drive_key = run["anim_frames"][-1]
drive_img = next(BASE.glob("images/" + drive_key + ".*")); rec = [k for k in fj["recon"] if k != fj["source"]]
from scipy.spatial.transform import Rotation as Rot                                                  # noqa: E402
s_ = np.load(BASE / "smplx2020.npz"); idx = {str(k): i for i, k in enumerate(s_["keys"])}; Rg = Rot.from_rotvec(s_["global_orient"]).as_matrix()
vdeg = {k: float(np.degrees(np.linalg.norm(Rot.from_matrix(Rg[idx[k]] @ Rg[idx[fj["source"]]].T).as_rotvec()))) for k in rec if k in idx}
stream_key = max(rec, key=lambda k: (vdeg.get(k, -1), -rec.index(k)))                               # rule: largest view change from the source
stream_img = next(BASE.glob("images/" + stream_key + ".*"))

# ---------------------------------------------------------------- bands
bands = (("create", 2.06, 3.40, "Avatar creation (once per photo, feed-forward)"),
         ("anim", 1.00, 2.00, "Animation (every frame)"),
         ("stream", 0.05, 0.94, "Optional streaming appearance update (background worker, no gradient steps)"))
for k, y0, y1, t in bands:
    ax.add_patch(FancyBboxPatch((0.03, y0), 5.80, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06", lw=0, fc=BAND[k], zorder=0))
    ax.text(0.08, y1 - 0.03, t, ha="left", va="top", fontsize=6.6, style="italic", color="#3c4a5c", zorder=6)

# ---------------------------------------------------------------- creation
TW = 0.50; th = TW / 0.62
img(crop_photo(src_img), 0.10, 2.13, TW, 1.06)
label(0.35, 2.125, r"photo $I_s$", fs=5.8)
dino = box(0.74, 2.60, 0.78, 0.62, "DINOv2 encoder\nfrozen, 518 px\nfeature maps $F_1, F_2$\nglobal token $g$")
body = box(0.74, 2.13, 0.78, 0.40, "SMPL-X 2020\nnative head; $\\beta,\\theta_s,\\psi_s$\ncamera $\\kappa=2f/L$", ours=True, fs=6.0)
arrow((0.60, 2.86), (0.74, 2.86))
# vertex branch (upper row)
vprj = box(1.70, 2.84, 1.02, 0.38, "Vertex sampling\nproject vertices, sample $F_2$")
vdec = box(2.88, 2.84, 1.40, 0.38, "Vertex Gaussian decoder\nMLP on [feature, learned base, $g$]")
# UV branch (lower row)
itm = box(1.70, 2.20, 1.02, 0.44, "Inverse texture mapping\n$[\\mathrm{RGB}, F_1]$ onto visible texels\nUV map $512^2$")
uv0 = img(R / "uv_rgb_source.png", 2.80, 2.18, 0.42, 0.42, white=False)
uvdec = box(3.30, 2.20, 0.98, 0.44, "UV decoder\nStyleUNet (style $g$)\n+ base; 1 Gaussian/texel")
arrow((1.52, 3.03), (1.70, 3.03)); arrow((1.52, 2.42), (1.70, 2.42))
arrow((2.72, 3.03), (2.88, 3.03)); arrow((2.72, 2.42), (2.80, 2.42)); arrow((3.22, 2.42), (3.30, 2.42))
arrow((1.13, 2.53), (1.70, 2.92), color=OURS_E, rad=0.0); arrow((1.52, 2.30), (1.70, 2.30), color=OURS_E)
# avatar
av = box(4.44, 2.20, 0.82, 1.02, "Avatar $\\mathcal{G}_s$\n$\\approx$207k Gaussians\n$(\\mu,\\Sigma,\\alpha,\\mathbf{h})$\n$\\mathbf{h}\\in\\mathbb{R}^{32}$")
img(crop_person(R / "orbit_photo_000.png"), 5.31, 2.18, 0.46, 0.74)
label(5.54, 2.175, "source pose", fs=5.6)
arrow((4.28, 3.03), (4.44, 3.03)); arrow((4.28, 2.42), (4.44, 2.42))

# ---------------------------------------------------------------- animation
img(crop_photo(drive_img), 0.10, 1.07, TW, 0.80)
label(0.35, 1.065, "driving frame", fs=5.8)
sprint = box(0.74, 1.20, 0.98, 0.62, "SPRINT pose regressor\nViTPose+-S + SMPL-X head\n$\\approx$7 ms; shape fixed\n(or supplied SMPL-X)", ours=True, fs=6.1)
arrow((0.60, 1.51), (0.74, 1.51))
lbs = box(1.92, 1.28, 0.86, 0.48, "LBS deformation $\\mathcal{W}$\n$(\\beta,\\theta_t,\\psi_t)$, SMPL-X 2020")
arrow((1.72, 1.52), (1.92, 1.52)); label(1.82, 1.50, r"$\theta_t,\psi_t$", fs=5.6, color=INK)
ras = box(2.96, 1.28, 0.92, 0.48, "Rasterise features $\\mathcal{R}$\n32 ch, $512^2$, camera $C_t$ ($\\kappa$)")
ref = box(4.06, 1.28, 0.78, 0.48, "Refiner $\\mathcal{S}$\nStyleUNet, RGB $512^2$")
arrow((2.78, 1.52), (2.96, 1.52)); arrow((3.88, 1.52), (4.06, 1.52))
img(crop_person(R / f"anim_photo_{drive_key}.png"), 5.02, 1.06, 0.46, 0.74)
label(5.25, 1.055, r"$\hat I_t$", fs=6.0, color=INK)
arrow((4.84, 1.52), (5.02, 1.52))
poly([(4.85, 2.20), (4.85, 2.06), (2.35, 2.06), (2.35, 1.76)])                                     # avatar -> LBS
ax.add_patch(Rectangle((3.02, 1.06), 0.80, 0.0, lw=0))

# ---------------------------------------------------------------- streaming update
img(crop_photo(stream_img), 0.10, 0.12, TW, 0.70)
label(0.35, 0.115, "later frame", fs=5.8)
gates = box(0.74, 0.30, 0.98, 0.52, "Frame gates\nsharpness, keypoint error,\nhead frontal, outlier frames", ours=True, fs=6.1)
arrow((0.60, 0.56), (0.74, 0.56))
obs = box(1.90, 0.30, 0.98, 0.52, "UV observation $U_t$\nencoder + inverse texture map\n$w_t=v_t\\,c_t\\,q_t\\,s_t$", ours=True, fs=6.1)
acc = box(3.06, 0.30, 1.16, 0.52, "Confidence-weighted fusion\n$S_t=S_{t-1}+w_tU_t,\\ W_t=W_{t-1}+w_t$\n$\\bar U_t=S_t/W_t$; source prior $\\times 2$", ours=True, fs=6.1)
uvf = img(R / "uv_rgb_fused.png", 4.32, 0.34, 0.44, 0.44, white=False)
dec2 = box(4.86, 0.30, 0.86, 0.52, "Re-run UV decoder\nswap in updated $\\mathcal{G}_s$\nrendering continues", ours=True, fs=6.1)
arrow((1.72, 0.56), (1.90, 0.56)); arrow((2.88, 0.56), (3.06, 0.56)); arrow((4.22, 0.56), (4.32, 0.56)); arrow((4.76, 0.56), (4.86, 0.56))
poly([(5.72, 0.56), (5.79, 0.56), (5.79, 2.70), (5.77, 2.70)], color=OURS_E, ls=(0, (3, 1.5)))       # updated avatar -> avatar
label(4.54, 0.33, f"seen {run['coverage_photo'] * 100:.0f}% $\\rightarrow$ {run['coverage_after_update'] * 100:.0f}%", fs=5.5, color=OURS_E)
label(3.01, 2.175, f"seen in photo: {run['coverage_photo'] * 100:.0f}%", fs=5.5)

# ---------------------------------------------------------------- training (right column) + legend
tb = (5.95, 1.00, 0.88, 2.40)
ax.add_patch(FancyBboxPatch(tb[:2], tb[2], tb[3], boxstyle="round,pad=0,rounding_size=0.06", lw=0.7, ls=(0, (3, 1.5)), fc="white", ec="#7a7a7a", zorder=1))
ax.text(tb[0] + tb[2] / 2, tb[1] + tb[3] - 0.05, "Training only", ha="center", va="top", fontsize=6.6, weight="bold", color=INK)
y = tb[1] + tb[3] - 0.24
for t, ours in (("GUAVA losses", False), ("L1 + LPIPS:\nimage, head, hands", False), ("added (ours)", True),
                ("identity (face)", True), ("clothing region\nL1 + LPIPS", True), ("Sobel edges\n(self-reconstruction)", True),
                ("silhouette\n$\\|\\hat a_t - m_t\\|_1$", True), ("data mix", True), ("BEDLAM2, UBody,\nMotion-X, 4D-DRESS,\nUBC-Fashion", True)):
    head = t in ("GUAVA losses", "added (ours)", "data mix")
    ax.text(tb[0] + 0.06, y, t, ha="left", va="top", fontsize=6.2 if head else 5.9, weight="bold" if head else "normal",
            color=(OURS_E if ours else MUTED) if head else INK, linespacing=1.1)
    y -= 0.115 * (t.count("\n") + 1) + (0.05 if head else 0.035)
for i, (f, e, t) in enumerate(((GUAVA_F, GUAVA_E, "GUAVA (inherited)"), (OURS_F, OURS_E, "ours (new/changed)"))):
    yy = 0.70 - i * 0.20
    ax.add_patch(FancyBboxPatch((5.97, yy), 0.16, 0.11, boxstyle="round,pad=0,rounding_size=0.02", fc=f, ec=e, lw=0.7))
    ax.text(6.18, yy + 0.055, t, ha="left", va="center", fontsize=5.9, color=INK)
ax.text(5.97, 0.24, "--- updated avatar\n     replaces $\\mathcal{G}_s$", ha="left", va="center", fontsize=5.6, color=OURS_E)

for ext in ("pdf", "png"): fig.savefig(OUT / f"fig_architecture.{ext}", dpi=300)
print("wrote", OUT / "fig_architecture.pdf")
