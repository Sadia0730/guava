"""Qualitative figure of the appearance update with zoom strips (PREREG "Extended benchmark", amendment 7f). CPU only; reads the
existing renders / GT / source images; writes only under --out.
Per row (clip:frame): one square crop for every column (union of the GT and all rendered silhouettes, 6 % margin, as in
fig_update_ext.py). Zoom window (registered rule): a 2:1 window of width --win x crop side, at least half covered by the GT person
mask, placed (4 px stride on the 512 px render) where mean |RGB error| of the other five render columns (GUAVA, LHM, LHM++, IDOL,
ours photo only) minus that of ours + update is largest -- i.e. the region of our largest advantage, by construction (say so in the
caption). The same window is boxed on every full view and shown enlarged below it. Renders are never altered.
Outputs: fig_update_zoom.{pdf,png} (the --fig rows, 6.88 in wide, vector text), row_<ds>_<clip>_<key>.png for every --rows entry,
tiles/<...>/<Dataset>_<clip>_<nn>_<column>{,_zoom}.png (1024 px; 512 px renders Lanczos-enlarged), zoom_selection.json.
Usage: python fig_update_zoom.py --out DIR --rows ubc/A1wrrhGRZmS:frame_000248,... [--fig ubc/A1wrrhGRZmS:frame_000248,...]
Paper version (amendment 7g): --paper (dataset-only row labels, STIX TrueType text, 450 dpi images, a rule between examples),
--drop fb_gt (hide columns; scores and the window rule unchanged), --win-set "cid:key=x0,y0,x1,y1;..." (hand-placed window in
render px of the crop, recorded as such), --caption-extra "..." (sentence added to the caption)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from PIL import Image, ImageOps
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
ap.add_argument("--fig-name", default="fig_update_zoom"); ap.add_argument("--only-fig", action="store_true", help="write only the --fig figure")
ap.add_argument("--paper", action="store_true", help="paper format (amendment 7g)"); ap.add_argument("--drop", default="", help="columns to hide")
ap.add_argument("--win-set", default="", help="hand-placed windows: cid:key=x0,y0,x1,y1;...")
ap.add_argument("--caption-extra", default=""); ap.add_argument("--caption", default="", help="full caption text (overrides)")
ap.add_argument("--white-bg", action="store_true", help="display only: show GT and renders on white (scores unchanged)")
ap.add_argument("--despill", action="store_true", help="with --white-bg: remove green-screen spill from soft edges (display only)")
ap.add_argument("--white-mode", default="soft", choices=["soft", "cut"], help="with --white-bg: soft = recomposite soft edges (RUNLOG 179); cut = cut at about half opacity, GT by its stored matte (display only)")
ap.add_argument("--cut", default="0.35,0.75", help="with --white-mode cut: opacity range of the edge transition (lo,hi)")
ap.add_argument("--num-corner", default="br", choices=["br", "auto"], help="PSNR label: br = bottom right (as before); auto = first corner (br, tr, tl, bl) that no person and no zoom box touches in any tile")
ap.add_argument("--dpi", type=int, default=0, help="output resolution of both the pdf images and the png (default: 450 / 300)")
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
DROP = set(filter(None, a.drop.split(",")))
WSET = {k: tuple(int(v) for v in w.split(",")) for k, w in (e.split("=") for e in filter(None, a.win_set.split(";")))}
if a.paper: plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})

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
    l, t, r, b = square_box(m); c = img[t:b, l:r]
    v = c.max(-1).astype(np.float64)                                                # display only: drop black padding + its edge line
    cols, rows = np.nonzero(np.median(v, 0) > 30)[0], np.nonzero(np.median(v, 1) > 30)[0]
    if not (len(cols) and len(rows)): return c
    bands = [v[:, :max(cols.min() - 6, 0)], v[:, cols.max() + 7:], v[:max(rows.min() - 6, 0)], v[rows.max() + 7:]]
    if any(b.size and (b > 40).mean() > 0.01 for b in bands): return c[2:-2, 2:-2] if a.paper else c   # 7h fix: dark content, not padding
    e = lambda lo, hi, N: (lo + 6 if lo > 0 else 0, hi + 1 - 6 if hi < N - 1 else N)     # padded side: also drop the frame's edge line
    (y0, y1), (x0, x1) = e(rows.min(), rows.max(), c.shape[0]), e(cols.min(), cols.max(), c.shape[1])
    c = c[y0:y1, x0:x1]
    return c[2:-2, 2:-2] if a.paper else c                                         # 7g: drop the frame's 2 px edge gradient


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
    wsrc, adv_v = "rule (amendment 7f)", float(score[y0, x0])
    if item in WSET:                                                   # amendment 7g: window placed by hand, recorded as such
        x0, y0, x1h, y1h = WSET[item]; w, h = x1h - x0, y1h - y0; win = round(w / s, 4); wsrc = "hand (amendment 7g)"
        assert 0 <= x0 < x1h <= s and 0 <= y0 < y1h <= s, item
    zwin = (int(x0), int(y0), int(x0 + w), int(y0 + h))
    zerr = {m: float(err[m][y0:y0 + h, x0:x0 + w].mean()) for m, _ in COLS}
    if item in WSET: adv_v = float(np.mean([zerr[m] for m in OTHERS]) - zerr["fb_update100"])
    full = {"Ground truth": crop(g)}; full.update({lab: crop(R[m]) for m, lab in COLS})
    person = np.zeros(crop(g).shape[:2], bool)                                        # union of all shown silhouettes (label placement)
    for im in full.values(): person |= im.max(-1) > 12
    if a.white_bg and a.white_mode == "cut":                                          # display only; scores and window use g and R above
        img, mat = rd(base(cid) / "images" / f"{k}.png"), base(cid) / "masks" / f"{k}.png"
        mat = np.asarray(Image.open(mat).convert("L")) if mat.exists() else None
        exact = img is not None and mat is not None and img.shape == g.shape and \
            np.abs(img.astype(np.float64) * mat[..., None] / 255.0 - g).max() <= 1.5      # GT = frame x stored matte
        full = {"Ground truth": crop(gt_white(img, mat) if exact else render_white(g))}
        full.update({lab: crop(render_white(R[m])) for m, lab in COLS})
    meta = {"clip": cid, "key": k, "dataset": DSN[ds], "view_deg": round(view_deg(cid, k), 1), "crop_box_512": [l, t, r_, b],
            "zoom_window_in_crop": list(zwin), "win_used": win, "zoom_window_frac": [round(v / s, 4) for v in zwin],
            "zoom_mean_abs_err": {m: round(v, 4) for m, v in zerr.items()},
            "zoom_advantage": round(adv_v, 4), "window_source": wsrc,
            "frame_psnr": {m: round(psnr(R[m], g), 2) for m, _ in COLS}}
    meta["_person"] = person
    return meta, source_img(cid), full


def white_bg(im, thr=3, band=10, hole=500, hole_zero=0.8, despill=False):
    """Display only. Background = black pixels (max channel <= thr) connected to the image border, plus enclosed black regions
    (e.g. between legs whose feet touch) of at least `hole` px that are mostly exact zeros (>= hole_zero). A pixel within `band`
    px of it is treated as a soft edge blended on black, I = alpha * F, where F's brightness is the largest brightness met by
    walking straight inward (along the distance gradient) while the distance to the background keeps growing; thin dark parts
    (heels, straps, hair strands) end that walk at once and stay as they are. Soft edges are recomposited on white; interior
    pixels are left unchanged. despill: in that edge band, green is limited to max(red, blue) first (green-screen spill in
    a matted frame would otherwise show as a green fringe on white; skin, red, blue and neutral colours are unaffected)."""
    from scipy import ndimage as ndi
    mx = im.max(-1).astype(np.float64); dark = mx <= thr
    lab, n = ndi.label(dark); edge_ids = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
    ids = list(edge_ids)
    if n:
        sz = np.bincount(lab.ravel(), minlength=n + 1); zc = np.bincount(lab.ravel(), weights=(mx == 0).ravel(), minlength=n + 1)
        ids += [c for c in range(1, n + 1) if c not in edge_ids and sz[c] >= hole and zc[c] / sz[c] >= hole_zero]
    bg = np.isin(lab, ids)
    if not bg.any(): return im
    d = ndi.distance_transform_edt(~bg); H, W = mx.shape
    ys, xs = np.nonzero((~bg) & (d <= band))
    if despill:
        im = im.copy(); px = im[ys, xs].astype(np.int16); px[:, 1] = np.minimum(px[:, 1], np.maximum(px[:, 0], px[:, 2]))
        im[ys, xs] = px.astype(np.uint8); mx = im.max(-1).astype(np.float64)
    gy, gx = np.gradient(ndi.gaussian_filter(d, 1.0)); u = np.hypot(gy, gx) + 1e-9; uy, ux = gy[ys, xs] / u[ys, xs], gx[ys, xs] / u[ys, xs]
    F = mx[ys, xs].copy(); prev = d[ys, xs].copy(); live = np.ones(len(ys), bool)
    for k in range(1, band + 1):
        yy = np.clip(np.rint(ys + k * uy).astype(int), 0, H - 1); xx = np.clip(np.rint(xs + k * ux).astype(int), 0, W - 1)
        dk = d[yy, xx]; live &= dk > prev + 0.3; prev = np.where(live, dk, prev)
        F = np.where(live, np.maximum(F, mx[yy, xx]), F)
    Fm = np.zeros_like(mx); wm = np.zeros_like(mx); Fm[ys, xs] = F; wm[ys, xs] = 1.0          # smooth F along the edge (no streaks)
    Fs = ndi.gaussian_filter(Fm, 1.5) / np.maximum(ndi.gaussian_filter(wm, 1.5), 1e-6)
    F = np.maximum(Fs[ys, xs], mx[ys, xs])
    alpha = np.ones_like(mx); alpha[bg] = 0.0
    ok = F > 8; alpha[ys[ok], xs[ok]] = np.clip(mx[ys[ok], xs[ok]] / F[ok], 0.0, 1.0)
    out = im.astype(np.float64) + (1.0 - alpha)[..., None] * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _bg_mask(mx, thr=3, hole=500, hole_zero=0.8):
    """Black connected to the frame border, plus enclosed black regions of >= hole px that are mostly exact zeros (as white_bg)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(mx <= thr); edge_ids = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}; ids = list(edge_ids)
    if n:
        sz = np.bincount(lab.ravel(), minlength=n + 1); zc = np.bincount(lab.ravel(), weights=(mx == 0).ravel(), minlength=n + 1)
        ids += [c for c in range(1, n + 1) if c not in edge_ids and sz[c] >= hole and zc[c] / sz[c] >= hole_zero]
    return np.isin(lab, ids)


def _est_alpha(im, band=10, fsig=2.5):
    """Opacity of a render on black (no matte stored): brightness / brightness of the interior met along the inward normal (bilinear
    samples while the distance to the background grows, at most `band` px), smoothed along the edge. 1 beyond `band`, 0 on background."""
    from scipy import ndimage as ndi
    mx = im.max(-1).astype(np.float64); bg = _bg_mask(mx); al = np.ones_like(mx); al[bg] = 0.0
    if not bg.any(): return al
    d = ndi.distance_transform_edt(~bg); ys, xs = np.nonzero((~bg) & (d <= band))
    gy, gx = np.gradient(ndi.gaussian_filter(d, 1.5)); u = np.hypot(gy, gx) + 1e-9; uy, ux = gy[ys, xs] / u[ys, xs], gx[ys, xs] / u[ys, xs]
    F = mx[ys, xs].copy(); prev = d[ys, xs].copy(); live = np.ones(len(ys), bool)
    for k in range(1, band + 1):
        pos = [ys + k * uy, xs + k * ux]
        v = ndi.map_coordinates(mx, pos, order=1, mode="nearest"); dk = ndi.map_coordinates(d, pos, order=1, mode="nearest")
        live &= dk > prev + 0.3; prev = np.where(live, dk, prev); F = np.where(live, np.maximum(F, v), F)
    Fm = np.zeros_like(mx); wm = np.zeros_like(mx); Fm[ys, xs] = F; wm[ys, xs] = 1.0
    Fs = ndi.gaussian_filter(Fm, fsig) / np.maximum(ndi.gaussian_filter(wm, fsig), 1e-6)
    al[ys, xs] = np.clip(mx[ys, xs] / np.maximum(np.maximum(Fs[ys, xs], mx[ys, xs]), 1.0), 0, 1)
    return al


def _white_cut(F, al, spill=4):
    """F: colour without the black background (float 0-255), al: opacity. Opacity is remapped by a smooth step from lo to hi (a cut at
    about half opacity with a narrow transition: no faint halo, no dark rim) and the result is put on white. Within `spill` px of the
    new edge, green is limited to max(red, blue) (green-screen spill)."""
    from scipy import ndimage as ndi
    lo, hi = (float(v) for v in a.cut.split(",")); t = np.clip((al - lo) / (hi - lo), 0, 1); t = t * t * (3 - 2 * t)
    sel = (ndi.distance_transform_edt(t > 0.02) <= spill) & (t > 0)
    F = F.copy(); px = F[sel]; px[:, 1] = np.minimum(px[:, 1], np.maximum(px[:, 0], px[:, 2])); F[sel] = px
    return np.clip(np.rint(t[..., None] * F + (1 - t[..., None]) * 255.0), 0, 255).astype(np.uint8)


def render_white(im):
    """Display only: a render on black -> on white (--white-mode cut). Interior pixels (opacity 1) are unchanged."""
    from scipy import ndimage as ndi
    al = ndi.gaussian_filter(_est_alpha(im), 0.6); I = im.astype(np.float64)
    F = np.where((al >= 0.999)[..., None], I, np.minimum(I / np.maximum(al, 0.35)[..., None], 255.0))
    return _white_cut(F, al)


def gt_white(img, matte, bsig=3.0):
    """Display only: ground truth on white from the frame and its stored matte (GT = frame x matte). Edge colours come from the matting
    equation I = a F + (1 - a) B with B = the frame's local background colour (Gaussian mean over matte < 0.02)."""
    from scipy import ndimage as ndi
    I = img.astype(np.float64); al = matte.astype(np.float64) / 255.0; isbg = (al < 0.02).astype(np.float64)
    ws = ndi.gaussian_filter(isbg, bsig); B = np.stack([ndi.gaussian_filter(I[..., c] * isbg, bsig) for c in range(3)], -1) / np.maximum(ws, 1e-6)[..., None]
    ok = (ws > 1e-3) & (al < 0.999) & (al > 0.3); F = I.copy()
    F[ok] = np.clip((I[ok] - (1 - al[ok])[:, None] * B[ok]) / al[ok][:, None], 0, 255)
    return _white_cut(F, al)


def num_corner(rows, w=0.27, h=0.14):
    """PSNR label corner (--num-corner auto): the first of br, tr, tl, bl whose w x h box (fractions of the tile) contains no person
    pixel of any shown tile and does not meet the zoom box, for every row; if none is free everywhere, the corner with least person."""
    occ = {}
    for c in ("br", "tr", "tl", "bl"):
        worst = 0.0
        for meta, _, _ in rows:
            P = meta["_person"]; S = P.shape[0]; bw, bh = int(round(w * S)), int(round(h * S))
            ys = slice(S - bh, S) if c[0] == "b" else slice(0, bh); xs = slice(S - bw, S) if c[1] == "r" else slice(0, bw)
            x0, y0, x1, y1 = meta["zoom_window_in_crop"]
            hit = not (x1 < xs.start or x0 > xs.stop or y1 < ys.start or y0 > ys.stop)
            worst = max(worst, 1.0 if hit else float(P[ys, xs].mean()))
        occ[c] = worst
        if worst == 0.0: return c
    return min(occ, key=occ.get)


def save_tiles(meta, src, full):
    ds, clip = split(meta["clip"]); td = OUT / "tiles" / f"{ds}_{clip}_{meta['key']}"; td.mkdir(parents=True, exist_ok=True)
    pre = f"{meta['dataset']}_{clip}"; z = a.tile
    ImageOps.pad(Image.fromarray(src), (z, z), Image.LANCZOS, color="white").save(td / f"{pre}_00_Source_photo.png")
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
    P = a.paper; shown = [(m, lab) for m, lab in COLS if m not in DROP]
    ncol = 2 + len(shown); gap = 0.02; cw = (width - gap * (ncol - 1)) / ncol; zh = cw / 2; head = 0.19 if P else 0.17
    sep = 0.11 if P else 0.05                                          # paper: wider gap + thin rule between examples
    if P: rh = cw + gap / 2 + zh; H = head + len(rows) * rh + (len(rows) - 1) * sep + 0.01
    else: rh = cw + zh + gap; H = head + len(rows) * (rh + 0.05)
    fs_head, fs_lab, fs_num = (8.0, 7.5, 6.2) if P else (6.6, 5.2, 4.6)
    fig = plt.figure(figsize=(width, H), dpi=a.dpi or (450 if P else 300))
    heads = ["Photo", "Ground truth"] + [lab for _, lab in shown]
    for j, lab in enumerate(heads):
        fig.text((j * (cw + gap) + cw / 2) / width, 1 - (head / 2) / H, lab, ha="center", va="center", fontsize=fs_head,
                 fontweight="bold" if lab.startswith("Ours") else "normal")
    corner = num_corner(rows) if a.num_corner == "auto" else "br"
    for i, (meta, src, full) in enumerate(rows):
        ytop = H - head - i * (rh + (sep if P else 0.05))
        if P and i:
            yl = (ytop + sep / 2) / H
            fig.add_artist(plt.Line2D([0, 1], [yl, yl], transform=fig.transFigure, color="0.72", lw=0.5))
        ax = fig.add_axes([0, (ytop - cw) / H, cw / width, cw / H]); ax.imshow(src, interpolation="lanczos" if P else "antialiased"); ax.axis("off")
        lines = [meta["dataset"]] if P else [meta["dataset"], meta["clip"].split("/", 1)[1], f"view {meta['view_deg']:.0f}°"]
        fig.text((cw / 2) / width, (ytop - cw - (0.08 if P else 0.05)) / H, "\n".join(lines), ha="center", va="top", fontsize=fs_lab, linespacing=1.25)
        x0, y0, x1, y1 = meta["zoom_window_in_crop"]
        keep = {"Ground truth"} | {lab for _, lab in shown}
        for j, (lab, im) in enumerate([(l, v) for l, v in full.items() if l in keep], start=1):
            x = j * (cw + gap)
            if a.white_bg and a.white_mode == "soft": im = white_bg(im, despill=a.despill)
            ax = fig.add_axes([x / width, (ytop - cw) / H, cw / width, cw / H]); ax.imshow(im, interpolation="lanczos"); ax.axis("off")
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=BOXC, lw=0.7))
            key = {"GUAVA": "guava", "LHM": "lhm_facefix", "LHM++": "lhmpp", "IDOL": "idol", "Ours, photo only": "fb_gt", "Ours + update": "fb_update100"}.get(lab)
            if key:
                if a.num_corner == "auto":                                        # free corner: plain text, nothing behind it
                    ax.text(0.97 if corner[1] == "r" else 0.03, 0.03 if corner[0] == "b" else 0.97, f"{meta['frame_psnr'][key]:.2f}",
                            transform=ax.transAxes, ha="right" if corner[1] == "r" else "left", va="bottom" if corner[0] == "b" else "top",
                            fontsize=fs_num + 0.4, color="black" if a.white_bg else "white")
                else:
                    ax.text(0.97, 0.03, f"{meta['frame_psnr'][key]:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs_num,
                            color="black" if a.white_bg else "white",
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")] if a.white_bg else None)
            az = fig.add_axes([x / width, (ytop - cw - gap / 2 - zh) / H, cw / width, zh / H])
            az.imshow(im[y0:y1, x0:x1], interpolation="lanczos"); az.set_xticks([]); az.set_yticks([])
            for sp in az.spines.values(): sp.set_edgecolor(BOXC); sp.set_linewidth(0.8)
    fig.savefig(OUT / f"{name}.pdf", dpi=a.dpi or (450 if P else 300)); fig.savefig(OUT / f"{name}.png", dpi=a.dpi or 300); plt.close(fig)
    print("wrote", OUT / f"{name}.pdf", f"{width} x {H:.2f} in", flush=True)


done = {}
for item in a.rows.split(","):
    meta, src, full = process(item); done[item] = (meta, src, full)
    if a.only_fig: continue
    save_tiles(meta, src, full); ds, clip = split(meta["clip"]); figure([done[item]], f"row_{ds}_{clip}_{meta['key']}")
if not a.only_fig: (OUT / "zoom_selection.json").write_text(json.dumps({"rule": "PREREG ext. amendment 7f", "win": a.win, "min_mask": a.min_mask,
                                                     "stride": a.stride, "rows": [{k: v for k, v in d[0].items() if not k.startswith("_")} for d in done.values()]}, indent=1) + "\n")
if a.fig:
    figure([done[i] for i in a.fig.split(",")], a.fig_name)
    cap = ("Appearance update on turned-away views. Each row: the source photo, the ground-truth test frame, four single-image "
           "methods, our avatar from the photo only, and our avatar after fusing the person's video (no gradient steps). Numbers: "
           "PSNR (dB) of the full 512 px frame. The boxed region, enlarged below each view, is placed automatically where our "
           "updated avatar's error is lowest relative to the other five columns (it shows our largest advantage by construction). "
           "Examples were selected from the clips with the largest gains on turned-away views; all numbers in the tables are over "
           "the full 163-clip benchmark.")
    if a.paper:
        cap = ("Appearance update on turned-away views. Each row: the source photo, the ground-truth test frame, four single-image "
               "methods, and our avatar after fusing the person's video (no gradient steps). Numbers: PSNR (dB) of the full frame. "
               "The boxed region is enlarged below each view. " + (a.caption_extra + " " if a.caption_extra else "") +
               "Examples and zoom regions were selected to show the update; all reported metrics are over the full 163-clip benchmark.")
    if a.caption: cap = a.caption
    (OUT / f"{a.fig_name}_caption.md").write_text(cap + "\n")
    (OUT / f"{a.fig_name}.json").write_text(json.dumps([{k: v for k, v in done[i][0].items() if not k.startswith("_")} for i in a.fig.split(",")], indent=1) + "\n")
for m, *_ in done.values():
    print(f"{m['clip']:34s} {m['key']}  window {m['zoom_window_frac']}  err others/ours "
          f"{np.mean([m['zoom_mean_abs_err'][o] for o in OTHERS]):.4f}/{m['zoom_mean_abs_err']['fb_update100']:.4f}")
