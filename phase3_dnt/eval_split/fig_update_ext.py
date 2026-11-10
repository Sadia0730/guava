"""Appearance-update figures for the extended benchmark (PREREG "Extended benchmark" visual results (2) + amendment 1, registered
before any per-clip update score of a new clip was looked at). CPU only; reads renders / GT / scores; writes only under --out.
Per dataset (UBody, People-Snapshot, NeuMan, X-Humans, UBC-Fashion), among clips whose test frames include a view change >= 135 deg
from the source (root-orientation change; if no clip qualifies, all clips, stated):
  median:   clip with the median per-clip PSNR gain fb_update100 - fb_gt (lower median); frame = its test frame with the largest
            view change (ties -> earlier key)                                                  [registered rule]
  showcase: clip with the LARGEST per-clip gain; frame = among its test frames with view change >= 135 deg (or all, if none), the
            largest per-frame PSNR gain (ties -> earlier key); captioned as the best case        [amendment 1]
Columns: source photo, ground truth, the single-image methods GUAVA / LHM / LHM++ / IDOL, ours from the photo, ours after 25 / 50 / 100 % of the video, UV texture before / after
(no per-person-trained method in the figures: amendment 2).
Each row uses one square crop (union of GT and all rendered silhouettes, 6 % margin); tiles saved at --size (1024: the 512 render
crop Lanczos-upsampled; the source from its own crop) with the frame's PSNR (full 512 image, as scored).
Usage: python fig_update_ext.py --out O/reports/qualitative_ext [--size 1024] [--no-video]"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
ROOT = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
SPLITS = [("avatar_test_v1", "avatar_test_v1.json", {"ubody"}), ("publicbench_v1", "publicbench_v1.json", None),
          ("publicbench_v2", "publicbench_v2_xhumans.json", None), ("finalbench_ext", "finalbench_ext.json", None)]
DS = ["ubody", "peoplesnapshot", "neuman", "xhumans", "ubc"]
DSN = {"ubody": "UBody", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "ubc": "UBC-Fashion"}
COLS = [("guava", "GUAVA"), ("lhm_facefix", "LHM"), ("lhmpp", "LHM++"), ("idol", "IDOL"), ("fb_gt", "Ours, photo only"), ("fb_update25", "+ update, 25 % video"), ("fb_update50", "+ update, 50 % video"),
        ("fb_update100", "+ update, 100 % video")]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--no-video", action="store_true"); ap.add_argument("--fps", type=int, default=10); ap.add_argument("--thr", type=float, default=135.0)
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
def font(s):
    try: return ImageFont.truetype("DejaVuSans.ttf", s)
    except OSError: return ImageFont.load_default()
F, FS = font(int(a.size * 0.045)), font(int(a.size * 0.04))

ROOTOF = {}
for E, sj, only in SPLITS:
    if (HERE / sj).exists():
        for c in json.loads((HERE / sj).read_text())["clips"]:
            if c["dataset"] in DS and (not only or c["dataset"] in only): ROOTOF[c["id"]] = E
_sc = {}
def score(m, E):
    if (m, E) not in _sc:
        f = ROOT / E / "scores" / f"{m}_full.json"; _sc[(m, E)] = json.loads(f.read_text())["per_clip"] if f.exists() else {}
    return _sc[(m, E)]
def base(cid): ds, clip = cid.split("/", 1); return ROOT / ROOTOF[cid] / "base" / ds / clip
def fj(cid): return json.loads((base(cid) / "frames.json").read_text())
def rd(p): p = Path(p); return np.asarray(Image.open(p).convert("RGB")) if p.exists() else None
def render(m, cid, k): ds, clip = cid.split("/", 1); return rd(ROOT / ROOTOF[cid] / "renders" / m / ds / clip / f"{k}.png")
def gt(cid, k): ds, clip = cid.split("/", 1); return rd(ROOT / ROOTOF[cid] / "gt512" / ds / clip / f"{k}.png")
def psnr(x, y): mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2) / 255.0 ** 2; return 10 * np.log10(1.0 / max(mse, 1e-12))
def views(cid):
    f = fj(cid); s = np.load(base(cid) / "smplx2020.npz"); idx = {str(k): i for i, k in enumerate(s["keys"])}
    R = Rot.from_rotvec(s["global_orient"]).as_matrix(); Rs = R[idx[f["source"]]]
    return {k: float(np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ Rs.T).as_rotvec()))) for k in f["test"]}
def square_box(mask, margin=0.06, H=512):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0: return (0, 0, H, H)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = int(max(y1 - y0, x1 - x0) * (1 + 2 * margin)); s = min(max(s, 64), H); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    t, l = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, H - s)); return (l, t, l + s, t + s)
def tile(img, box, z, txt=None):
    if img is None:
        im = Image.new("RGB", (z, z), (60, 60, 60)); ImageDraw.Draw(im).text((z // 3, z // 2), "not run", fill="white", font=F); return im
    im = Image.fromarray(img).crop(box).resize((z, z), Image.LANCZOS)
    if txt:
        d = ImageDraw.Draw(im); w = d.textlength(txt, font=FS); d.rectangle([0, z - FS.size - 12, w + 14, z], fill="black"); d.text((7, z - FS.size - 8), txt, fill="white", font=FS)
    return im
def source_tile(cid, z):
    f = fj(cid); img = rd(base(cid) / "images" / f"{f['source']}.png"); msk = rd(base(cid) / "masks" / f"{f['source']}.png")
    m = (msk[..., 0] > 127) if msk is not None else (img.sum(-1) > 0)
    return tile(img, square_box(m, H=img.shape[0]), z)
def uv_tile(cid, which, z):
    ds, clip = cid.split("/", 1); p = ROOT / ROOTOF[cid] / "renders" / "fb_update100" / ds / clip / f"uv_rgb_{which}.png"
    im = rd(p); return tile(im, (0, 0, im.shape[1], im.shape[0]), z) if im is not None else tile(None, None, z)

def coverage(cid):
    ds, clip = cid.split("/", 1); p = ROOT / ROOTOF[cid] / "renders" / "fb_update100" / ds / clip / "run.json"
    if not p.exists(): return None, None
    r = json.loads(p.read_text()); return r.get("coverage_source"), r.get("coverage_fused")


# ---- selection
sel = []
for ds in DS:
    cands = []
    for cid, E in ROOTOF.items():
        if not cid.startswith(ds + "/"): continue
        g, u = score("fb_gt", E).get(cid), score("fb_update100", E).get(cid)
        if g and u: cands.append((u["psnr"] - g["psnr"], cid))
    if not cands: print("no scored clips yet:", ds, flush=True); continue
    V = {cid: views(cid) for _, cid in cands}
    q = [(gn, cid) for gn, cid in cands if max(V[cid].values()) >= a.thr]; note = f"view change >= {a.thr:.0f} deg"
    if not q: q = cands; note = f"no clip reaches {a.thr:.0f} deg; all clips"
    q.sort(); gmed, cmed = q[(len(q) - 1) // 2]; gbest, cbest = q[-1]
    kmed = max(V[cmed].items(), key=lambda t: (t[1], [-i for i, k in enumerate(V[cmed]) if k == t[0]][0]))[0]
    ks = [k for k, v in V[cbest].items() if v >= a.thr] or list(V[cbest])
    pg = [(psnr(render("fb_update100", cbest, k), gt(cbest, k)) - psnr(render("fb_gt", cbest, k), gt(cbest, k)), -i, k) for i, k in enumerate(ks)]
    kbest = max(pg)[2]
    sel.append({"ds": ds, "kind": "median", "clip": cmed, "key": kmed, "view_deg": round(V[cmed][kmed], 1), "clip_gain_db": round(gmed, 2), "pool": len(q), "note": note})
    sel.append({"ds": ds, "kind": "showcase", "clip": cbest, "key": kbest, "view_deg": round(V[cbest][kbest], 1), "clip_gain_db": round(gbest, 2),
                "frame_gain_db": round(max(pg)[0], 2), "pool": len(q), "note": note + "; best case by the registered rule"})
(OUT / "update_selection.json").write_text(json.dumps(sel, indent=1)); print(json.dumps(sel, indent=1), flush=True)

# ---- figures: one PNG per row (paper), one per kind (all datasets), tiles
z = a.size; LAB = ["Source photo", "Ground truth"] + [l for _, l in COLS] + ["UV texture, photo", "UV texture, after update"]
def row_tiles(r):
    cid, k = r["clip"], r["key"]; g = gt(cid, k); R = {m: render(m, cid, k) for m, _ in COLS}; mask = g.sum(-1) > 0
    for im in R.values():
        if im is not None: mask |= im.sum(-1) > 12
    box = square_box(mask); ps = {m: (round(psnr(R[m], g), 2) if R[m] is not None else None) for m, _ in COLS}
    tl = [source_tile(cid, z), tile(g, box, z)] + [tile(R[m], box, z, None if ps[m] is None else f"{ps[m]:.2f} dB") for m, _ in COLS]
    return tl + [uv_tile(cid, "source", z), uv_tile(cid, "fused", z)], ps
def canvas(rows, name):
    hd = int(F.size * 2.4); lw = int(z * 0.9); cv = Image.new("RGB", (lw + z * len(LAB), hd + z * len(rows)), "white"); dr = ImageDraw.Draw(cv)
    for j, lab in enumerate(LAB): dr.text((lw + j * z + (z - dr.textlength(lab, font=F)) / 2, hd * 0.25), lab, fill="black", font=F)
    meta = []
    for i, r in enumerate(rows):
        tl, ps = row_tiles(r); y = hd + i * z
        td = OUT / "tiles" / f"{r['kind']}_{r['ds']}_{r['clip'].split('/', 1)[1]}_{r['key']}"; td.mkdir(parents=True, exist_ok=True)
        for j, t in enumerate(tl): cv.paste(t, (lw + j * z, y)); t.save(td / f"{j:02d}_{LAB[j].replace(' ', '_').replace(',', '').replace('%', 'pct')}.png")
        c0, c1 = coverage(r["clip"]); cov = f"texture seen {c0 * 100:.0f}% -> {c1 * 100:.0f}%" if c0 is not None else ""
        for t, line in enumerate([DSN[r["ds"]], r["clip"].split("/", 1)[1][-20:], f"view {r['view_deg']:.0f}°", cov, f"clip gain {r['clip_gain_db']:+.2f} dB",
                                  "best case" if r["kind"] == "showcase" else "median case"]):
            dr.text((10, y + 12 + t * (FS.size + 10)), line, fill="black", font=F if t == 0 else FS)
        meta.append({**r, "frame_psnr": ps, "coverage_photo": c0, "coverage_after_update": c1})
    cv.save(OUT / f"{name}.png"); cv.resize((cv.width // 4, cv.height // 4), Image.LANCZOS).save(OUT / f"{name}_preview.jpg", quality=90)
    (OUT / f"{name}.json").write_text(json.dumps({"columns": LAB, "tile_px": z, "note": "renders are 512 px; tiles are the row crop Lanczos-upsampled", "rows": meta}, indent=1))
    cap = [f"# {name}", "", "Columns: " + " | ".join(LAB) + ".", "",
           "What the update fixes: from one photo only the visible part of the body surface is observed; every single-image method "
           "(GUAVA, LHM, LHM++, IDOL, ours from the photo) has to guess the rest (back, sides, occluded parts). Our appearance update "
           "streams the person's video frames into the avatar's UV texture without any training, so the guessed regions are replaced "
           "by observed appearance. All renders are 512 px, shown enlarged; numbers on tiles = PSNR of that frame.", ""]
    for m in meta:
        p0, p1 = m["frame_psnr"].get("fb_gt"), m["frame_psnr"].get("fb_update100")
        bb = {k: v for k, v in m["frame_psnr"].items() if k in ("guava", "lhm_facefix", "lhmpp", "idol") and v is not None}
        cov = (f"texture coverage {m['coverage_photo'] * 100:.0f}% from the photo -> {m['coverage_after_update'] * 100:.0f}% after the update; "
               if m["coverage_photo"] is not None else "")
        cap.append(f"- {DSN[m['ds']]} ({m['kind']}, {m['clip']}, {m['key']}, view change {m['view_deg']:.0f} deg): {cov}"
                   f"frame PSNR ours {p0} -> {p1} dB after 100 % of the video" + (f"; best other single-image method {max(bb.values()):.2f} dB" if bb else "") + ".")
    (OUT / f"{name}_caption.md").write_text("\n".join(cap) + "\n")
    print("wrote", OUT / f"{name}.png", cv.size, flush=True)
canvas([r for r in sel if r["kind"] == "showcase"], "fig_update_showcase")
canvas([r for r in sel if r["kind"] == "median"], "fig_update_median")
for r in sel: canvas([r], f"row_{r['kind']}_{r['ds']}")

if not a.no_video:
    import imageio.v2 as iio
    (OUT / "videos").mkdir(exist_ok=True); zv = 384; FV = font(16); hv = 30; labs = ["Source photo", "Ground truth"] + [l for _, l in COLS]
    for r in sel:
        cid = r["clip"]; f = fj(cid); st = source_tile(cid, zv)
        w = iio.get_writer(OUT / "videos" / f"update_{r['kind']}_{r['ds']}_{cid.split('/', 1)[1]}.mp4", fps=a.fps, codec="libx264", quality=8, macro_block_size=16)
        for k in f["test"]:
            fr = Image.new("RGB", (zv * len(labs), zv + hv), "white"); d = ImageDraw.Draw(fr)
            for j, lab in enumerate(labs): d.text((j * zv + (zv - d.textlength(lab, font=FV)) / 2, 6), lab, fill="black", font=FV)
            full = (0, 0, 512, 512)
            for j, t in enumerate([st, tile(gt(cid, k), full, zv)] + [tile(render(m, cid, k), full, zv) for m, _ in COLS]): fr.paste(t, (j * zv, hv))
            arr = np.asarray(fr); w.append_data(arr[:(arr.shape[0] // 16) * 16, :(arr.shape[1] // 16) * 16])
        w.close(); print("video", r["kind"], cid, len(f["test"]), "frames", flush=True)
