"""GUAVA-FB pilot comparison (PREREG amendment 3), validation data only. Parts:
  eval    per model: (2) clothing-box LPIPS / SSIM / L1 (512, box = projected torso / hip / thigh / upper-arm vertices, 2-98 %
          quantiles, as the pilot-B loss), (3) zoomed clothing LPIPS / SSIM / PSNR on 4D-DRESS (2x zoom at the median matte
          pixel of the clothing band, against the 1024 crop), (4) Gaussian coverage vs matte, full-image PSNR / SSIM / LPIPS
          (consistency with the registered validator), (5) flicker: 4 held-out 4D-DRESS Outer groups, first non-source camera,
          the take's frames + slerp in-betweens; mean LPIPS between consecutive renders; one mp4 per group.
  timing  all models interleaved on one GPU: Gaussians after pruning, avatar build time (median of 20), per-frame render time
          (deform + rasterise + refine, median of 200).
  geometry (amendment 4, measure 7) mean cosine of the rendered normal and of the D-Normal (normal_prior.py) vs frozen
          Sapiens-normal-1B on the same validation targets, inside matte ∩ coverage (each eroded by 2 px); normal-map examples.
  compose side-by-side mp4s / zoom crops and the summary table from the per-model results.
  showcase --models baseline=ckpt candidate=ckpt, after eval + geometry of both: the validation targets with the largest
          per-target gain of the candidate (distinct take groups) for the D-Normal cosine (best per track + the median-gain
          target), coverage miss and zoomed clothing LPIPS; figures + selection.json (gains, share of targets improved).
  rawref  (amendment 6, diagnostic) per model: the raw rasterised RGB (first 3 of the 32 Gaussian feature channels, which
          GUAVA's loss also supervises) vs the refined render, on the full image, the clothing box (256) and the face box (128,
          as the validator): PSNR / SSIM / LPIPS / L1 and high-pass luminance detail (correlation with the ground truth and
          energy ratio render / ground truth, masked Gaussian blur sigma 2 px). --tracks may include ubody (a20 val rule).
          Example crops at fixed target positions (1/6, 1/2, 5/6 of each track's list, chosen before any result)."""
import argparse, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["eval", "timing", "geometry", "compose", "showcase", "rawref"], required=True)
ap.add_argument("--max-per-group", type=int, default=0, help="smoke tests only: at most this many targets per validation group")
ap.add_argument("--fig-labels", default="", help="rawref: labels whose example crops are composed into one figure once all exist")
ap.add_argument("--per", type=int, default=3, help="showcase: examples per measure")
ap.add_argument("--models", nargs="+", required=True, help="label=checkpoint")
ap.add_argument("--out", required=True); ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data")
ap.add_argument("--tracks", default="dress4d,bedlam2,motionx"); ap.add_argument("--val-step", type=int, default=4)
ap.add_argument("--val-caps", default="dress4d:30,bedlam2:20,motionx:10")
ap.add_argument("--flicker-groups", default="00127/Outer,00134/Outer,00137/Outer,00151/Outer"); ap.add_argument("--inbetween", type=int, default=3)
ap.add_argument("--sapiens", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/models/sapiens/sapiens_1b_normal.pt2")
ap.add_argument("--scan-root", default="", help="geometry: 4D-DRESS scan cache -> scan normal / depth / fold-detail / per-garment coverage (amendment 5)")
a = ap.parse_args()
models = dict(m.split("=", 1) for m in a.models); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import cv2, numpy as np, torch
from scipy.spatial.transform import Rotation, Slerp

if a.part == "compose":
    labels = list(models)
    res = {l: json.loads((out / l / "eval.json").read_text()) for l in labels if (out / l / "eval.json").exists()}
    for l in labels:
        if l in res and (out / l / "geometry.json").exists():
            for t, d in json.loads((out / l / "geometry.json").read_text())["summary"].items(): res[l]["summary"].setdefault(t, {}).update(d)
    tim = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    keys = [("dress4d", "cloth_lpips"), ("dress4d", "cloth_ssim"), ("bedlam2", "cloth_lpips"), ("motionx", "cloth_lpips"),
            ("dress4d", "zoom_lpips"), ("dress4d", "zoom_ssim"), ("dress4d", "zoom_psnr"), ("dress4d", "miss_frac"), ("dress4d", "spill_frac"),
            ("dress4d", "psnr"), ("bedlam2", "psnr"), ("motionx", "psnr"), ("flicker", "lpips_consecutive"),
            ("all", "geo_cos_dnormal"), ("all", "geo_cos_rendered"), ("dress4d", "geo_cos_dnormal"), ("bedlam2", "geo_cos_dnormal"), ("motionx", "geo_cos_dnormal"),
            ("all", "fold_sapiens_rendered"), ("dress4d", "scan_cos_rendered"), ("dress4d", "scan_cos_dnormal"), ("dress4d", "scan_depth_mae_cm"),
            ("dress4d", "fold_scan_rendered"), ("dress4d", "fold_scan_dnormal"), ("dress4d", "miss_scan"), ("dress4d", "miss_upper"),
            ("dress4d", "miss_lower"), ("dress4d", "miss_outer")]
    rows = ["| measure | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for t, k in keys:
        rows.append(f"| {t} {k} | " + " | ".join(f"{res[l]['summary'].get(t, {}).get(k, float('nan')):.4f}" if l in res else "-" for l in labels) + " |")
    for k in ("n_gaussians", "build_ms", "frame_ms"):
        rows.append(f"| {k} | " + " | ".join(f"{tim.get(l, {}).get(k, float('nan')):.2f}" for l in labels) + " |")
    (out / "summary.md").write_text("\n".join(rows) + "\n"); print("\n".join(rows))
    for g in a.flicker_groups.split(","):                                     # side-by-side mp4 per group
        caps = [cv2.VideoCapture(str(out / l / f"flicker_{g.replace('/', '_')}.mp4")) for l in labels]
        if not all(c.isOpened() for c in caps): continue
        vw = None
        while True:
            fr = [c.read() for c in caps]
            if not all(f[0] for f in fr): break
            row = np.concatenate([cv2.putText(f[1].copy(), l, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) for f, l in zip(fr, labels)], 1)
            if vw is None: vw = cv2.VideoWriter(str(out / f"flicker_{g.replace('/', '_')}_side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (row.shape[1], row.shape[0]))
            vw.write(row)
        if vw is not None: vw.release()
    ns = [cv2.imread(str(out / l / "normal_examples.jpg")) for l in labels]
    if all(z is not None for z in ns): cv2.imwrite(str(out / "normal_examples_all.jpg"), np.concatenate(ns, 0))
    zs = [cv2.imread(str(out / l / "zoom_examples.jpg")) for l in labels]
    if all(z is not None for z in zs): cv2.imwrite(str(out / "zoom_examples_all.jpg"), np.concatenate(zs, 0))
    sys.exit(0)

import fb_common as F
from fb_common import to_dev
os.chdir(F.ROOT); dev = torch.device("cuda:0")
from models.UbodyAvatar import Ubody_Gaussian
from utils.loss_utils import cal_psnr, cal_ssim
from submodules.lpipsPyTorch import LPIPS

# ----------------------------------------------------------------------------- validation set (= guava_fb_train.build_val, v1b grouping)
tracks = [t for t in a.tracks.split(",") if t]; jobs, meta = [], []
for s in tracks:
    if s == "ubody":                                                         # a20's validation clips and face conversion
        fconv, eye_copy = F.load_face_conv()
        for c in F.a20_split()[1]:
            jobs.append((str(c), fconv.get(c.name, {}), eye_copy)); meta.append({"source": s, "subject": c.name, "pair_group": c.name})
        continue
    for e in json.loads((Path(a.data_root) / s / "index.json").read_text())["clips"]:
        if e.get("split") == "val":
            jobs.append((str(Path(a.data_root) / s / e["name"]), None, True))
            meta.append({"source": s, "subject": str(e.get("subject", e["name"])), "pair_group": "/".join(str(e[k]) for k in ("subject", "outfit") if k in e)})
loaded = F.load_clips_parallel(jobs, workers=16); clips = []
drop = {s: {k: set(v) for k, v in json.loads((Path(a.data_root) / s / "drop_frames.json").read_text()).items()}
        for s in tracks if (Path(a.data_root) / s / "drop_frames.json").exists()}
for c, m in zip(loaded, meta):
    dk = drop.get(m["source"], {}).get(Path(c["dir"]).name)
    if dk and "keys" in c: c["keys"] = [k for k in c["keys"] if k not in dk]
    if "error" in c or len(c.get("keys", [])) < 2: continue
    c.update(m); clips.append(c)
cfg = F.get_cfg(); fbd = F.FBData(cfg, clips)
caps = {k: int(v) for k, v in (x.split(":") for x in a.val_caps.split(",") if x)}
VAL, GROUPSRC = {}, {}
for s in tracks:
    cis = [i for i, c in enumerate(clips) if c["source"] == s]; G = []
    if s == "dress4d":
        by = {}
        for i in cis: by.setdefault(clips[i]["pair_group"], []).append(i)
        for g, ii in sorted(by.items()):
            ii = sorted(ii, key=lambda i: clips[i]["name"]); take = ii[0]
            v0 = sorted({clips[take]["recs"][k]["view"] for k in clips[take]["keys"]})[0]
            src = (take, next(k for k in clips[take]["keys"] if clips[take]["recs"][k]["view"] == v0))
            tg = [(i, k) for i in ii for k in clips[i]["keys"] if clips[i]["recs"][k]["view"] != v0]
            step = max(1, math.ceil(len(tg) / caps[s])); G.append((src, tg[::step][:caps[s]])); GROUPSRC[g] = (src, v0)
    elif s == "ubody":                                                       # exact a20 rule
        for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::4]]))
    else:
        for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::a.val_step]][:caps.get(s, 150)]))
    VAL[s] = [(src, tg[:a.max_per_group]) for src, tg in G] if a.max_per_group else G

_, inf, ren, _ = F.setup_a20(dev); inf.eval(); ren.eval(); lp = LPIPS("vgg", "0.1").to(dev)
NV = F.NV; CLOTH_JOINTS = torch.tensor([0, 1, 2, 3, 6, 9, 13, 14, 16, 17], device=dev)
CLOTH_VID = torch.nonzero(torch.isin(inf.ehm.smplx.lbs_weights[:NV].argmax(1), CLOTH_JOINTS)).flatten()


def load(label):
    st = torch.load(models[label], map_location="cpu", weights_only=False)
    inf.load_state_dict(st["model"]); ren.load_state_dict(st["render_model"]); inf.eval(); ren.eval()
    step = st.get("step"); del st; return step


@torch.no_grad()
def avatar(src):
    torch.manual_seed(0); vg, ug, _ = inf(to_dev(F.collate1(fbd.source_info(*src)), dev))
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval(); return av


@torch.no_grad()
def render(av, t):
    assets = av(t); torch.manual_seed(0); rr = ren(assets, t["render_cam_params"], bg=0.0)
    ones = dict(assets); ones["features_color"] = torch.ones_like(assets["features_color"])
    alpha = ren.forward_raw(ones, t["render_cam_params"], bg=0.0)["raw_features"][:, :1].clamp(0, 1)
    return rr["renders"].clamp(0, 1), alpha


def crop256(x, box):
    l, r, tp, b = box
    return torch.nn.functional.interpolate(x[:, :, tp:b, l:r], (256, 256), mode="bilinear", antialias=True, align_corners=False)


def target_from_record(rec):
    """target dict from a (possibly interpolated) record without an image (flicker clips)."""
    sm = {k: torch.as_tensor(np.asarray(v, np.float32).copy()) for k, v in rec["smplx"].items()}
    sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
    RT = sm["camera_RT_params"]; c2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])); M4 = torch.eye(4); M4[:3, :4] = RT
    w2c = c2c @ M4; t = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c),
                         "invtanfov": float(rec["invtanfov"])}
    t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
    return t


def interp_records(r0, r1, n):
    outl = []
    rot_keys = [k for k, v in r0["smplx"].items() if k not in ("camera_RT_params", "shape", "exp") and np.asarray(v).shape[-1] == 3]
    for j in range(n):
        w = j / n; r = {"smplx": {}, "invtanfov": (1 - w) * r0["invtanfov"] + w * r1["invtanfov"]}
        for k, v in r0["smplx"].items():
            v0, v1 = np.asarray(v, np.float64), np.asarray(r1["smplx"][k], np.float64)
            if k in rot_keys:
                a0, a1 = v0.reshape(-1, 3), v1.reshape(-1, 3)
                r["smplx"][k] = np.stack([Slerp([0, 1], Rotation.from_rotvec([x, y]))(w).as_rotvec() for x, y in zip(a0, a1)]).reshape(v0.shape)
            elif k == "camera_RT_params":
                R = Slerp([0, 1], Rotation.from_matrix([v0[:, :3], v1[:, :3]]))(w).as_matrix(); r["smplx"][k] = np.c_[R, (1 - w) * v0[:, 3] + w * v1[:, 3]]
            else:
                r["smplx"][k] = (1 - w) * v0 + w * v1
        outl.append(r)
    return outl


# ----------------------------------------------------------------------------- showcase (selected examples, baseline vs candidate)
if a.part == "showcase":
    import normal_prior as NP
    la, lb = list(models); R = {}
    for l in (la, lb):
        for f in ("eval", "geometry"):
            for r in json.loads((out / l / f"{f}.json").read_text())["rows"]: R.setdefault(l, {}).setdefault((r["track"], r["tgt"]), {}).update(r)
    REF = {(s, f"{clips[ref[0]]['name']}/{ref[1]}"): (src, ref) for s, G in VAL.items() for src, tg in G for ref in tg}
    grp = lambda k: (k[0], k[1].split("/")[0].split("_Take")[0])

    def gains(m, sg):
        return sorted(((sg * (R[lb][k][m] - R[la][k][m]), k) for k in R[lb] if R[lb][k].get(m) is not None and R[la].get(k, {}).get(m) is not None), reverse=True)

    def top(g, n):
        seen, sel = set(), []
        for v, k in g:
            if v > 0 and grp(k) not in seen: seen.add(grp(k)); sel.append(("best", v, k))
            if len(sel) == n: break
        return sel
    MEAS = {"geometry": ("geo_cos_dnormal", 1), "coverage": ("miss_frac", -1), "zoom": ("zoom_lpips", -1)}
    G = {p: gains(*ms) for p, ms in MEAS.items()}
    SEL = {"geometry": [x for s in VAL for x in top([y for y in G["geometry"] if y[1][0] == s], 1)] + [("median", *G["geometry"][len(G["geometry"]) // 2])],
           "coverage": top(G["coverage"], a.per), "zoom": top(G["zoom"], a.per)}
    sap = NP.SapiensNormal(a.sapiens, dev, sign=(1.0, -1.0, -1.0)); T = {}
    for l in (la, lb):
        load(l); cache = {}
        with torch.no_grad():
            for part, sel in SEL.items():
                for _, _, k in sel:
                    src, ref = REF[k]; d = T.setdefault((part, k), {})
                    if src not in cache: cache[src] = avatar(src)
                    av = cache[src]; t = to_dev(F.collate1(fbd.target_info(*ref)), dev); d["gt"] = (t["image"] * t["mask"])[0].cpu(); d["m"] = (t["mask"][0, :1] > 0.5).cpu()
                    if part == "geometry":
                        if "Np" not in d: d["Np"] = sap(t["image"])[0].cpu()
                        Nr, Dr, Ar = NP.render_geometry(ren, av(t), t["render_cam_params"], t["w2c_cam"]); Nd = NP.depth_to_normal(Dr, F.batch_invtanfov(t, 1, dev))
                        d[l] = (Nr[0].cpu(), Nd[0].cpu(), (Ar[0] > 0.5).cpu())
                    elif part == "coverage":
                        r, al = render(av, t); d[l] = (r[0].cpu(), al[0].cpu())
                    else:
                        tz = to_dev(F.collate1(fbd.target_info_zoom(*ref, center="band_median", factor=2.0, min_cover=0.0)), dev)
                        d["gt"] = (tz["image"] * tz["mask"])[0].cpu(); d[l] = render(av, tz)[0][0].cpu()
        del cache

    def u8(x): return np.ascontiguousarray((x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1])

    def sq(m, pad=0.06):
        ys, xs = np.nonzero(m[0].numpy()); H = m.shape[-1]
        side = min(H, int(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * pad)))
        y0 = int(np.clip((ys.min() + ys.max() - side) / 2, 0, H - side)); x0 = int(np.clip((xs.min() + xs.max() - side) / 2, 0, H - side))
        return y0, x0, side

    def tile(img, b=None, S=400, txt=()):
        if b is not None: img = img[b[0]:b[0] + b[2], b[1]:b[1] + b[2]]
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA if img.shape[0] > S else cv2.INTER_LINEAR)
        for i, s in enumerate(txt):
            (w, h), bl = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, .55, 1); y = 6 + i * (h + bl + 8)
            cv2.rectangle(img, (3, y - 3), (9 + w, y + h + bl + 2), (0, 0, 0), -1)
            cv2.putText(img, s, (6, y + h), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    def grid(rows, heads, title, path):
        W = sum(t.shape[1] for t in rows[0]); hd = np.full((64, W, 3), 255, np.uint8); x = 0
        cv2.putText(hd, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
        for h, t in zip(heads, rows[0]): cv2.putText(hd, h, (x + 8, 54), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2, cv2.LINE_AA); x += t.shape[1]
        cv2.imwrite(str(path), np.concatenate([hd] + [np.concatenate(r, 1) for r in rows], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])

    od = out / "showcase"; od.mkdir(exist_ok=True); share = {}
    for part, (m, sg) in MEAS.items():
        share[part] = {"measure": m, "n": len(G[part]), "candidate_better": int(sum(v > 0 for v, _ in G[part])), "mean_gain": float(np.mean([v for v, _ in G[part]]))}
    rows = []
    for tag, v, k in SEL["geometry"]:
        d = T[("geometry", k)]; b = sq(d["m"]); rk = [R[l][k] for l in (la, lb)]
        nrm = lambda n, msk: u8((n * .5 + .5) * msk)
        rows.append([tile(u8(d["gt"]), b, txt=(f"{k[0]} ({tag} gain)", k[1].split("/")[0][:40])), tile(nrm(d["Np"], d["m"]), b)]
                    + [tile(nrm(d[l][0], d[l][2]), b, txt=(f"cos {r['geo_cos_rendered']:.3f}",)) for l, r in zip((la, lb), rk)]
                    + [tile(nrm(d[l][1], d[l][2]), b, txt=(f"cos {r['geo_cos_dnormal']:.3f}",)) for l, r in zip((la, lb), rk)])
    s = share["geometry"]
    grid(rows, ["input", "Sapiens (target)", f"{la} normal", f"{lb} normal", f"{la} D-Normal", f"{lb} D-Normal"],
         f"geometry: {lb} > {la} on D-Normal cosine for {s['candidate_better']}/{s['n']} targets; rows = best per track + median-gain target", od / "geometry_wins.jpg")
    rows = []
    for tag, v, k in SEL["coverage"]:
        d = T[("coverage", k)]; b = sq(d["m"]); m = d["m"][0]; ov = []
        for l in (la, lb):
            al = d[l][1][0]; base = d["gt"] * 0.35; miss = m & (al < 0.5); spill = ~m & (al > 0.5)
            base[0][miss], base[1][miss], base[2][miss] = 1.0, 0.1, 0.1; base[0][spill], base[1][spill], base[2][spill] = 0.1, 0.9, 1.0
            ov.append(tile(u8(base), b, txt=(f"miss {R[l][k]['miss_frac'] * 100:.1f}%  spill {R[l][k]['spill_frac'] * 100:.1f}%",)))
        rows.append([tile(u8(d["gt"]), b, txt=(k[0], k[1][:40])), tile(u8(d[la][0]), b), tile(u8(d[lb][0]), b)] + ov)
    s = share["coverage"]
    grid(rows, ["ground truth", la, lb, f"{la} coverage", f"{lb} coverage"],
         f"coverage (red = person not covered, cyan = spill): {lb} misses less on {s['candidate_better']}/{s['n']} targets", od / "coverage_wins.jpg")
    rows = []
    for tag, v, k in SEL["zoom"]:
        d = T[("zoom", k)]
        rows.append([tile(u8(d["gt"]), txt=(k[1][:40],))] + [tile(u8(d[l]), txt=(f"LPIPS {R[l][k]['zoom_lpips']:.3f}  SSIM {R[l][k]['zoom_ssim']:.3f}",)) for l in (la, lb)])
    s = share["zoom"]
    grid(rows, ["ground truth (2x zoom)", la, lb], f"2x clothing zoom: {lb} lower LPIPS on {s['candidate_better']}/{s['n']} targets", od / "zoom_wins.jpg")
    (od / "selection.json").write_text(json.dumps({"baseline": la, "candidate": lb, "share": share,
                                                   "selected": {p: [{"track": k[0], "tgt": k[1], "kind": t, "gain": v} for t, v, k in sel] for p, sel in SEL.items()}}, indent=1))
    print(json.dumps(share, indent=1)); sys.exit(0)


# ----------------------------------------------------------------------------- raw vs refined (amendment 6)
if a.part == "rawref":
    import scan_geom as SG
    fg = F.FaceGeom(inf, dev)

    def lum(x): return 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def detail(r, g, m, sigma=2.0):
        hr = lum(r) - SG.blur_masked(lum(r), m, sigma); hg = lum(g) - SG.blur_masked(lum(g), m, sigma)
        er, eg = (hr ** 2 * m).sum(), (hg ** 2 * m).sum()
        return float((hr * hg * m).sum() / (er * eg).sqrt().clamp(min=1e-12)), float((er / eg.clamp(min=1e-12)).sqrt())

    def crops_metrics(pre, x, g, m, row):
        row.update({f"{pre}_lpips": float(lp(x, g)), f"{pre}_ssim": float(cal_ssim(x, g)), f"{pre}_l1": float((x - g).abs().mean())})
        row[f"{pre}_detail_corr"], row[f"{pre}_detail_energy"] = detail(x, g, m)

    flat_t = {s: [ref for _, tg in G for ref in tg] for s, G in VAL.items()}
    EXPOS = {s: {tuple(v[int(len(v) * f)]) for f in (1 / 6, 1 / 2, 5 / 6)} for s, v in flat_t.items() if v}
    KEYS = [f"{st}_{k}" for st in ("raw", "ref") for k in ("psnr", "ssim", "lpips", "cloth_lpips", "cloth_ssim", "cloth_l1", "cloth_detail_corr", "cloth_detail_energy",
                                                          "face_lpips", "face_ssim", "face_l1", "face_detail_corr", "face_detail_energy")]
    for label in models:
        step = load(label); od = out / label; od.mkdir(exist_ok=True); rows = []; ex = {}; n_empty = {}
        with torch.no_grad():
            for s, G in VAL.items():
                for src, tg in G:
                    av = avatar(src)
                    for ref in tg:
                        t = to_dev(F.collate1(fbd.target_info(*ref)), dev); g = t["image"] * t["mask"]; m = t["mask"][:, :1]
                        if float((m > 0.5).sum()) < 64: n_empty[s] = n_empty.get(s, 0) + 1; continue     # empty matte (e.g. a UBody frame): PSNR undefined
                        torch.manual_seed(0); rr = ren(av(t), t["render_cam_params"], bg=0.0)
                        R_ = {"raw": rr["raw_renders"].clamp(0, 1), "ref": rr["renders"].clamp(0, 1)}
                        row = {"track": s, "tgt": f"{clips[ref[0]]['name']}/{ref[1]}"}
                        for st, x in R_.items(): row.update({f"{st}_psnr": float(cal_psnr(x, g)), f"{st}_ssim": float(cal_ssim(x, g)), f"{st}_lpips": float(lp(x, g))})
                        o = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"]); itf = F.batch_invtanfov(t, 1, dev)
                        px, _ = F.project_px(o["vertices"][:, CLOTH_VID], t["w2c_cam"], itf)
                        lo = torch.quantile(px[0], 0.02, dim=0); hi = torch.quantile(px[0], 0.98, dim=0)
                        box = [int(lo[0].clamp(0, 511)), int(hi[0].clamp(0, 511)), int(lo[1].clamp(0, 511)), int(hi[1].clamp(0, 511))]
                        E = {}
                        if box[1] - box[0] >= 8 and box[3] - box[2] >= 8:
                            gc, mc = crop256(g, box), (crop256(m, box) > 0.5).float(); E["cloth"] = [gc]
                            for st, x in R_.items(): xc = crop256(x, box); crops_metrics(f"{st}_cloth", xc, gc, mc, row); E["cloth"].append(xc)
                        hp, _ = F.project_px(o["vertices"][:, fg.head_vid], t["w2c_cam"], itf); fb_, side, inside = F.face_boxes(hp)
                        x0, y0, x1, y1 = fb_[0].clamp(0, 512).round().long().tolist()
                        if float(inside[0]) >= 0.5 and float(side[0]) >= 4 and x1 - x0 >= 2 and y1 - y0 >= 2:
                            fc = lambda x: torch.nn.functional.interpolate(x[:, :, y0:y1, x0:x1], (128, 128), mode="bilinear", antialias=True, align_corners=False)
                            gf, mf = fc(g), (fc(m) > 0.5).float(); E["face"] = [gf]
                            for st, x in R_.items(): xf = fc(x); crops_metrics(f"{st}_face", xf, gf, mf, row); E["face"].append(xf)
                        if tuple(ref) in EXPOS[s]: ex[(s, row["tgt"])] = {k: [z[0].cpu() for z in v] for k, v in E.items()} | {"row": dict(row)}
                        rows.append(row)
                print(label, s, "done", len(rows), flush=True)
        def mean(x): x = [v for v in x if v is not None]; return float(np.mean(x)) if x else None
        summ = {s: {k: mean([r.get(k) for r in rows if r["track"] == s]) for k in KEYS} | {"n": sum(r["track"] == s for r in rows),
                                                                                           "n_face": sum(r["track"] == s and "raw_face_lpips" in r for r in rows), "n_empty_matte_skipped": n_empty.get(s, 0)} for s in VAL}
        (od / "rawref.json").write_text(json.dumps({"label": label, "checkpoint": models[label], "step": step, "summary": summ, "rows": rows}, indent=0))
        torch.save(ex, od / "rawref_examples.pt")
        print(label, json.dumps({s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for s, d in summ.items()}), flush=True)
    labs = [l for l in a.fig_labels.split(",") if l]
    if labs and all((out / l / "rawref_examples.pt").exists() for l in labs):
        EXS = {l: torch.load(out / l / "rawref_examples.pt", weights_only=False) for l in labs}; keys = sorted(EXS[labs[0]], key=lambda k: (tracks.index(k[0]), k[1]))

        def u8(x, S=256):
            x = torch.nn.functional.interpolate(x[None], (S, S), mode="bilinear", align_corners=False)[0]
            return np.ascontiguousarray((x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1])

        def lab(img, txt):
            for i, s_ in enumerate(txt):
                (w, h), bl = cv2.getTextSize(s_, cv2.FONT_HERSHEY_SIMPLEX, .45, 1); y = 4 + i * (h + bl + 6)
                cv2.rectangle(img, (2, y - 2), (8 + w, y + h + bl + 1), (0, 0, 0), -1); cv2.putText(img, s_, (5, y + h), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            return cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for part in ("cloth", "face"):
            rows_img = []
            for k in keys:
                if not all(part in EXS[l].get(k, {}) for l in labs): continue
                tiles = [lab(u8(EXS[labs[0]][k][part][0]), (k[0], k[1].split("/")[0][:30]))]
                for l in labs:
                    r_ = EXS[l][k]["row"]
                    for j, st in ((1, "raw"), (2, "ref")):
                        tiles.append(lab(u8(EXS[l][k][part][j]), (f"{l} {'raw' if st == 'raw' else 'refined'}", f"LPIPS {r_[f'{st}_{part}_lpips']:.3f}  detail r {r_[f'{st}_{part}_detail_corr']:.2f}")))
                rows_img.append(np.concatenate(tiles, 1))
            if rows_img:
                hd = np.full((40, rows_img[0].shape[1], 3), 255, np.uint8)
                cv2.putText(hd, f"{part}: ground truth | " + " | ".join(f"{l} raw (Gaussians only), {l} refined" for l in labs) + " -- fixed target positions, not selected by result",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(str(out / f"rawref_{part}.jpg"), np.concatenate([hd] + rows_img, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    sys.exit(0)


# ----------------------------------------------------------------------------- timing
if a.part == "timing":
    src = VAL["dress4d"][0][0]; ref = VAL["dress4d"][0][1][0]; t = to_dev(F.collate1(fbd.target_info(*ref)), dev); si = to_dev(F.collate1(fbd.source_info(*src)), dev)
    states = {l: torch.load(p, map_location="cpu", weights_only=False) for l, p in models.items()}
    T = {l: {"build": [], "frame": []} for l in models}
    for rnd in range(5):                                                     # interleaved rounds
        for l, st in states.items():
            inf.load_state_dict(st["model"]); ren.load_state_dict(st["render_model"]); inf.eval(); ren.eval()
            with torch.no_grad():
                for _ in range(3): av = avatar(src); render(av, t)        # warm-up
                for _ in range(4):
                    torch.cuda.synchronize(); t0 = time.perf_counter(); torch.manual_seed(0); vg, ug, _ = inf(si)
                    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); torch.cuda.synchronize(); T[l]["build"].append((time.perf_counter() - t0) * 1e3)
                for _ in range(40):
                    torch.cuda.synchronize(); t0 = time.perf_counter(); ren(av(t), t["render_cam_params"], bg=0.0); torch.cuda.synchronize(); T[l]["frame"].append((time.perf_counter() - t0) * 1e3)
                T[l]["n_gaussians"] = int(av._xyz_deform.shape[1]) if hasattr(av, "_xyz_deform") else None
    res = {l: {"build_ms": float(np.median(v["build"])), "frame_ms": float(np.median(v["frame"])), "frame_ms_p90": float(np.percentile(v["frame"], 90)),
               "n_gaussians": v["n_gaussians"], "gpu": torch.cuda.get_device_name(0)} for l, v in T.items()}
    (out / "timing.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); sys.exit(0)

# ----------------------------------------------------------------------------- geometry (measure 7)
if a.part == "geometry":
    import normal_prior as NP, torchvision, scan_geom as SG
    sap = NP.SapiensNormal(a.sapiens, dev, sign=(1.0, -1.0, -1.0)); store = SG.ScanStore(a.scan_root) if a.scan_root else None
    GARM = {3: "upper", 4: "lower", 5: "outer"}
    for label in models:
        step = load(label); od = out / label; od.mkdir(exist_ok=True); rows = []; ex = []
        with torch.no_grad():
            for s, G in VAL.items():
                for src, tg in G:
                    av = avatar(src)
                    for ref in tg:
                        t = to_dev(F.collate1(fbd.target_info(*ref)), dev); Np = sap(t["image"]); itf = F.batch_invtanfov(t, 1, dev)
                        Nr, Dr, Ar = NP.render_geometry(ren, av(t), t["render_cam_params"], t["w2c_cam"])
                        Nd = NP.depth_to_normal(Dr, itf); m = t["mask"][:, :1] > 0.5
                        v = NP.erode(m.float(), 2) * NP.erode((Ar > 0.5).float(), 2); nv = v.sum().clamp(min=1); v2 = NP.erode(v, 2)
                        row = {"track": s, "tgt": f"{clips[ref[0]]['name']}/{ref[1]}",
                               "geo_cos_rendered": float(((Nr * Np).sum(1, keepdim=True) * v).sum() / nv),
                               "geo_cos_dnormal": float(((Nd * Np).sum(1, keepdim=True) * v).sum() / nv),
                               "fold_sapiens_rendered": float(SG.fold_detail(Nr, Np, v2)[0]), "fold_sapiens_dnormal": float(SG.fold_detail(Nd, Np, v2)[0])}
                        sc = store.get(clips[ref[0]]["name"], ref[1]) if (store is not None and s == "dress4d") else None
                        if sc is not None:
                            Ns, Ds, Ls = SG.render_scans([tuple(torch.from_numpy(x).to(dev) for x in sc)], t["w2c_cam"], itf)
                            on = (Ls >= 0) & m; vs = NP.erode(on.float(), 2) * NP.erode((Ar > 0.5).float(), 2); nvs = vs.sum().clamp(min=1)
                            row.update({"scan_cos_rendered": float(((Nr * Ns).sum(1, keepdim=True) * vs).sum() / nvs),
                                        "scan_cos_dnormal": float(((Nd * Ns).sum(1, keepdim=True) * vs).sum() / nvs),
                                        "scan_depth_mae_cm": float(((Dr - Ds).abs() * vs).sum() / nvs) * 100,
                                        "fold_scan_rendered": float(SG.fold_detail(Nr, Ns, NP.erode(vs, 2))[0]),
                                        "fold_scan_dnormal": float(SG.fold_detail(Nd, Ns, NP.erode(vs, 2))[0]),
                                        "miss_scan": float((on & (Ar < 0.5)).sum() / on.sum().clamp(min=1))})
                            for l, nm in GARM.items():
                                pix = (Ls == l) & m
                                if int(pix.sum()) >= 50: row[f"miss_{nm}"] = float((pix & (Ar < 0.5)).sum() / pix.sum())
                        rows.append(row)
                        if len(ex) < 4 and len(rows) % 60 == 1:
                            ex.append(torch.cat([t["image"][0] * t["mask"][0], (Np[0] * .5 + .5) * v[0], (Nr[0] * .5 + .5) * v[0], (Nd[0] * .5 + .5) * v[0]], 2))
                print(label, s, "done", len(rows), flush=True)
        KEYS = ("geo_cos_rendered", "geo_cos_dnormal", "fold_sapiens_rendered", "fold_sapiens_dnormal", "scan_cos_rendered", "scan_cos_dnormal",
                "scan_depth_mae_cm", "fold_scan_rendered", "fold_scan_dnormal", "miss_scan", "miss_upper", "miss_lower", "miss_outer")
        def mk(rr): return {k: float(np.mean([r[k] for r in rr if k in r])) for k in KEYS if any(k in r for r in rr)}
        summ = {s: mk([r for r in rows if r["track"] == s]) for s in VAL}
        summ["all"] = mk(rows)                                                            # over all targets (registered)
        (od / "geometry.json").write_text(json.dumps({"label": label, "checkpoint": models[label], "step": step, "summary": summ, "rows": rows}, indent=0))
        if ex: torchvision.utils.save_image(torch.stack(ex), od / "normal_examples.jpg", nrow=len(ex), padding=4, pad_value=1.0)
        print(label, json.dumps(summ), flush=True)
    sys.exit(0)

# ----------------------------------------------------------------------------- eval
for label in models:
    step = load(label); od = out / label; od.mkdir(exist_ok=True); rows = []; zoom_ex = []
    with torch.no_grad():
        for s, G in VAL.items():
            for src, tg in G:
                av = avatar(src)
                for ref in tg:
                    t = to_dev(F.collate1(fbd.target_info(*ref)), dev); r, al = render(av, t); g = t["image"] * t["mask"]; m = t["mask"][:, :1] > 0.5
                    V = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"])["vertices"][:, CLOTH_VID]
                    px, _ = F.project_px(V, t["w2c_cam"], F.batch_invtanfov(t, 1, dev))
                    lo = torch.quantile(px[0], 0.02, dim=0); hi = torch.quantile(px[0], 0.98, dim=0)
                    box = [int(lo[0].clamp(0, 511)), int(hi[0].clamp(0, 511)), int(lo[1].clamp(0, 511)), int(hi[1].clamp(0, 511))]
                    row = {"track": s, "tgt": f"{clips[ref[0]]['name']}/{ref[1]}", "psnr": float(cal_psnr(r, g)), "ssim": float(cal_ssim(r, g)), "lpips": float(lp(r, g)),
                           "miss_frac": float((m & (al < 0.5)).sum() / m.sum().clamp(min=1)), "spill_frac": float((~m & (al > 0.5)).sum() / m.sum().clamp(min=1))}
                    if box[1] - box[0] >= 8 and box[3] - box[2] >= 8:
                        rc, gc = crop256(r, box), crop256(g, box)
                        row.update({"cloth_lpips": float(lp(rc, gc)), "cloth_ssim": float(cal_ssim(rc, gc)), "cloth_l1": float((rc - gc).abs().mean())})
                    if s == "dress4d":
                        tz = fbd.target_info_zoom(*ref, center="band_median", factor=2.0, min_cover=0.0)
                        if tz is not None:
                            tz = to_dev(F.collate1(tz), dev); rz, az = render(av, tz); gz = tz["image"] * tz["mask"]; mz = tz["mask"][:, :1] > 0.5
                            row.update({"zoom_lpips": float(lp(rz, gz)), "zoom_ssim": float(cal_ssim(rz, gz)), "zoom_psnr": float(cal_psnr(rz, gz)),
                                        "zoom_miss_frac": float((mz & (az < 0.5)).sum() / mz.sum().clamp(min=1))})
                            if len(zoom_ex) < 6 and len(rows) % 40 == 0: zoom_ex.append(torch.cat([gz[0], rz[0]], 2))
                    rows.append(row)
            print(label, s, "done", len(rows), flush=True)
        # flicker clips
        fl = {}
        for gname in [g for g in a.flicker_groups.split(",") if g in GROUPSRC]:
            (src, v0) = GROUPSRC[gname]; av = avatar(src); ci = src[0]
            views = sorted({clips[ci]["recs"][k]["view"] for k in clips[ci]["keys"]} - {v0}); cam = views[0]
            keys = sorted([k for k in clips[ci]["keys"] if clips[ci]["recs"][k]["view"] == cam], key=lambda k: k.split("_", 1)[-1])
            recs = [clips[ci]["recs"][k] for k in keys]; seq = []
            for r0, r1 in zip(recs[:-1], recs[1:]): seq += interp_records(r0, r1, a.inbetween + 1)
            seq.append(recs[-1]); frames = []; d = []
            vw = cv2.VideoWriter(str(od / f"flicker_{gname.replace('/', '_')}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (512, 512))
            prev = None
            for rec in seq:
                t = to_dev(F.collate1(target_from_record(rec)), dev); r, _ = render(av, t)
                if prev is not None: d.append(float(lp(r, prev)))
                prev = r; vw.write(cv2.cvtColor((r[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            vw.release(); fl[gname] = {"frames": len(seq), "camera": cam, "lpips_consecutive": float(np.mean(d))}
    def mean(x): x = [v for v in x if v is not None]; return float(np.mean(x)) if x else None
    summ = {s: {k: mean([r.get(k) for r in rows if r["track"] == s]) for k in ("psnr", "ssim", "lpips", "cloth_lpips", "cloth_ssim", "cloth_l1",
                                                                              "zoom_lpips", "zoom_ssim", "zoom_psnr", "zoom_miss_frac", "miss_frac", "spill_frac")} for s in VAL}
    summ["flicker"] = {"lpips_consecutive": mean([v["lpips_consecutive"] for v in fl.values()]), "groups": fl}
    (od / "eval.json").write_text(json.dumps({"label": label, "checkpoint": models[label], "step": step, "summary": summ, "rows": rows}, indent=0))
    if zoom_ex:
        import torchvision; torchvision.utils.save_image(torch.stack(zoom_ex), od / "zoom_examples.jpg", nrow=len(zoom_ex), padding=4, pad_value=1.0)
    print(label, json.dumps({s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "groups"} for s, d in summ.items()}), flush=True)
