"""FinalBench qualitative figures and videos from the existing renders (PREREG "FinalBench qualitative figures: example selection
rule", registered 2026-11-09 before any figure). CPU only; reads renders / GT / scores, writes only under --out.
Rows are chosen by the registered rules, never by eye:
  typical: per dataset the clip with the median per-clip PSNR of GUAVA-FB [SMPL-X GT] (lower median); frames = the test frame with
           the smallest and the one with the largest root-orientation change from the source frame (ties -> earlier key).
  hard:    per dataset the clip with the smallest per-clip PSNR margin of GUAVA-FB [SMPL-X GT] over the best single-image baseline
           (GUAVA, LHM, LHM++, IDOL); frame = the test frame with the median per-frame GUAVA-FB PSNR (lower median).
Each row is cropped to one square box (union of the GT and all rendered silhouettes of that frame, 6 % margin), the same box for
every column; the source photo is cropped to its own matte. Tiles show the frame's PSNR (dB, vs GT, full 512 image as scored).
Usage: python fig_qualitative.py --out O/reports/qualitative_finalbench [--size 384] [--no-video]"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

ROOT = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
BENCH = {"ubody": "avatar_test_v1", "peoplesnapshot": "publicbench_v1", "neuman": "publicbench_v1", "xhumans": "publicbench_v2",
         "bedlam2": "avatar_test_v1"}
DSNAME = {"ubody": "UBody", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "bedlam2": "BEDLAM2"}
OURS = "fb_gt"
BASE1 = ["guava", "lhm_facefix", "lhmpp", "idol"]
SINGLE = [("guava", "GUAVA"), ("lhm_facefix", "LHM"), ("lhmpp", "LHM++"), ("idol", "IDOL"), ("fb_sprint", "Ours (est. pose)"),
          ("fb_gt", "Ours (GT pose)")]
PERSON = [("gart", "GART"), ("gaussianavatar", "GaussianAvatar"), ("exavatar_gtface", "ExAvatar"), ("fb_gt", "Ours (GT pose)")]

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=384)
ap.add_argument("--no-video", action="store_true"); ap.add_argument("--fps", type=int, default=10); a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); (OUT / "tiles").mkdir(exist_ok=True)
try:
    F = ImageFont.truetype("DejaVuSans.ttf", max(12, a.size // 14)); FS = ImageFont.truetype("DejaVuSans.ttf", max(10, a.size // 17))
except OSError:
    F = FS = ImageFont.load_default()

_sc = {}


def clip_scores(m, ds):
    k = (m, BENCH[ds])
    if k not in _sc: _sc[k] = json.loads((ROOT / BENCH[ds] / "scores" / f"{m}_full.json").read_text())["per_clip"]
    return {cid: v for cid, v in _sc[k].items() if v["dataset"] == ds}


def E(ds): return ROOT / BENCH[ds]


def rd(path):
    p = Path(path); return np.asarray(Image.open(p).convert("RGB")) if p.exists() else None


def psnr(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2) / 255.0 ** 2; return 10 * np.log10(1.0 / max(mse, 1e-12))


def frames_json(ds, clip): return json.loads((E(ds) / "base" / ds / clip / "frames.json").read_text())


def view_angles(ds, clip):
    fj = frames_json(ds, clip); s = np.load(E(ds) / "base" / ds / clip / "smplx2020.npz"); idx = {str(k): i for i, k in enumerate(s["keys"])}
    R = Rot.from_rotvec(s["global_orient"]).as_matrix(); Rs = R[idx[fj["source"]]]
    return [(k, float(np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ Rs.T).as_rotvec())))) for k in fj["test"]]


def render(m, ds, clip, k): return rd(E(ds) / "renders" / m / ds / clip / f"{k}.png")


def gt(ds, clip, k): return rd(E(ds) / "gt512" / ds / clip / f"{k}.png")


def square_box(mask, margin=0.06, H=512):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0: return (0, 0, H, H)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)); s = min(max(s, 64), H); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    t, l = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, H - s)); return (l, t, l + s, t + s)


def tile(img, box, size, txt=None):
    if img is None:
        im = Image.new("RGB", (size, size), (60, 60, 60)); d = ImageDraw.Draw(im); d.text((size // 2 - 30, size // 2 - 8), "not run", fill="white", font=F); return im
    im = Image.fromarray(img).crop(box).resize((size, size), Image.BICUBIC)
    if txt:
        d = ImageDraw.Draw(im); w = d.textlength(txt, font=FS); d.rectangle([0, size - FS.size - 8, w + 10, size], fill="black")
        d.text((5, size - FS.size - 5), txt, fill="white", font=FS)
    return im


def source_tile(ds, clip, size):
    fj = frames_json(ds, clip); base = E(ds) / "base" / ds / clip
    img = rd(base / "images" / f"{fj['source']}.png"); msk = rd(base / "masks" / f"{fj['source']}.png")
    m = (msk[..., 0] > 127) if msk is not None else (img.sum(-1) > 0)
    return tile(img, square_box(m), size), fj["source"]


# ---- selection (registered rules) ----
sel = {"typical": [], "hard": []}
for ds in BENCH:
    ours = clip_scores(OURS, ds); order = sorted(ours, key=lambda c: (ours[c]["psnr"], c)); cid = order[(len(order) - 1) // 2]
    clip = cid.split("/", 1)[1]; va = view_angles(ds, clip)
    near = min(enumerate(va), key=lambda t: (t[1][1], t[0]))[1]; far = max(enumerate(va), key=lambda t: (t[1][1], -t[0]))[1]
    for k, ang in (near, far):
        sel["typical"].append({"ds": ds, "clip": clip, "key": k, "view_deg": round(ang, 1), "clip_psnr_ours": round(ours[cid]["psnr"], 2),
                               "why": f"median-PSNR {DSNAME[ds]} clip ({len(order)} clips); {'smallest' if (k, ang) == near else 'largest'} view change"})
    base = {m: clip_scores(m, ds) for m in BASE1}
    marg = sorted(((ours[c]["psnr"] - max(base[m][c]["psnr"] for m in BASE1), c) for c in ours))
    mval, cid = marg[0]; clip = cid.split("/", 1)[1]; fj = frames_json(ds, clip)
    pf = sorted(((psnr(render(OURS, ds, clip, k), gt(ds, clip, k)), i, k) for i, k in enumerate(fj["test"])))
    p, _, k = pf[(len(pf) - 1) // 2]
    best = max(BASE1, key=lambda m: base[m][cid]["psnr"])
    sel["hard"].append({"ds": ds, "clip": clip, "key": k, "frame_psnr_ours": round(p, 2), "margin_db": round(mval, 2), "best_baseline": best,
                        "why": f"smallest margin over the best single-image baseline in {DSNAME[ds]} ({mval:+.2f} dB vs {best}); median frame"})
(OUT / "selection.json").write_text(json.dumps(sel, indent=1))


def figure(rows, cols, name, title):
    z = a.size; lw = int(z * 0.85); hd = int(F.size * 2.2)
    canvas = Image.new("RGB", (lw + z * (len(cols) + 2), hd + z * len(rows)), "white"); dr = ImageDraw.Draw(canvas)
    for j, lab in enumerate(["Source photo", "Ground truth"] + [l for _, l in cols]):
        x = lw + j * z + (z - dr.textlength(lab, font=F)) / 2; dr.text((x, hd * 0.25), lab, fill="black", font=F)
    meta = []
    for i, r in enumerate(rows):
        ds, clip, k = r["ds"], r["clip"], r["key"]; y = hd + i * z
        g = gt(ds, clip, k); R = {m: render(m, ds, clip, k) for m, _ in cols}
        mask = g.sum(-1) > 0
        for im in R.values():
            if im is not None: mask |= im.sum(-1) > 12
        box = square_box(mask)
        st, sk = source_tile(ds, clip, z); tiles = [st, tile(g, box, z)]
        cell = {"ds": ds, "clip": clip, "key": k, "source": sk, "box": box, "psnr": {}}
        for m, _ in cols:
            p = psnr(R[m], g) if R[m] is not None else None; cell["psnr"][m] = None if p is None else round(p, 2)
            tiles.append(tile(R[m], box, z, None if p is None else f"{p:.2f} dB"))
        td = OUT / "tiles" / name / f"{i:02d}_{ds}_{clip}_{k}"; td.mkdir(parents=True, exist_ok=True)
        for j, (t, n) in enumerate(zip(tiles, ["source", "gt"] + [m for m, _ in cols])):
            canvas.paste(t, (lw + j * z, y)); t.save(td / f"{j:02d}_{n}.png")
        lab = [DSNAME[ds], clip[-22:], f"view {r.get('view_deg', '')}°" if "view_deg" in r else f"margin {r['margin_db']:+.2f} dB"]
        for t, line in enumerate(lab): dr.text((6, y + 8 + t * (FS.size + 6)), line, fill="black", font=F if t == 0 else FS)
        meta.append(cell)
    canvas.save(OUT / f"{name}.png"); canvas.convert("RGB").resize((canvas.width // 2, canvas.height // 2), Image.LANCZOS).save(OUT / f"{name}_preview.jpg", quality=90)
    (OUT / f"{name}.json").write_text(json.dumps({"title": title, "cols": cols, "rows": meta}, indent=1))
    print("wrote", OUT / f"{name}.png", canvas.size, flush=True)


figure(sel["typical"], SINGLE, "fig_single_image", "Single-image methods, typical clip per dataset, near and turned view")
figure(sel["typical"], PERSON, "fig_per_person", "Per-person video-trained methods vs ours, same frames")
figure(sel["hard"], SINGLE, "fig_hard_cases", "Hardest clip per dataset for ours (smallest margin over the best single-image baseline)")

if not a.no_video:
    import imageio.v2 as iio
    (OUT / "videos").mkdir(exist_ok=True); z = 256; hd = 28
    try: FV = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError: FV = ImageFont.load_default()
    done = set()
    for r in sel["typical"]:
        ds, clip = r["ds"], r["clip"]
        if (ds, clip) in done: continue
        done.add((ds, clip)); fj = frames_json(ds, clip)
        for name, cols in (("single", SINGLE), ("perperson", PERSON)):
            labs = ["Source photo", "Ground truth"] + [l for _, l in cols]; st, _ = source_tile(ds, clip, z)
            w = iio.get_writer(OUT / "videos" / f"{ds}_{clip}_{name}.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=16)
            for k in fj["test"]:
                g = gt(ds, clip, k); R = {m: render(m, ds, clip, k) for m, _ in cols}; full = (0, 0, 512, 512)
                fr = Image.new("RGB", (z * len(labs), z + hd), "white"); d = ImageDraw.Draw(fr)
                for j, lab in enumerate(labs): d.text((j * z + (z - d.textlength(lab, font=FV)) / 2, 5), lab, fill="black", font=FV)
                for j, t in enumerate([st, tile(g, full, z)] + [tile(R[m], full, z) for m, _ in cols]): fr.paste(t, (j * z, hd))
                arr = np.asarray(fr); hh, ww = (arr.shape[0] // 16) * 16, (arr.shape[1] // 16) * 16; w.append_data(arr[:hh, :ww])
            w.close(); print("video", ds, clip, name, len(fj["test"]), "frames", flush=True)
print(json.dumps(sel, indent=1))
