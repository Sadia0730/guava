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
  compose side-by-side mp4s / zoom crops and the summary table from the per-model results."""
import argparse, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["eval", "timing", "geometry", "compose"], required=True)
ap.add_argument("--models", nargs="+", required=True, help="label=checkpoint")
ap.add_argument("--out", required=True); ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data")
ap.add_argument("--tracks", default="dress4d,bedlam2,motionx"); ap.add_argument("--val-step", type=int, default=4)
ap.add_argument("--val-caps", default="dress4d:30,bedlam2:20,motionx:10")
ap.add_argument("--flicker-groups", default="00127/Outer,00134/Outer,00137/Outer,00151/Outer"); ap.add_argument("--inbetween", type=int, default=3)
ap.add_argument("--sapiens", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/models/sapiens/sapiens_1b_normal.pt2")
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
            ("all", "geo_cos_dnormal"), ("all", "geo_cos_rendered"), ("dress4d", "geo_cos_dnormal"), ("bedlam2", "geo_cos_dnormal"), ("motionx", "geo_cos_dnormal")]
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
    else:
        for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::a.val_step]][:caps.get(s, 150)]))
    VAL[s] = G

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
    import normal_prior as NP, torchvision
    sap = NP.SapiensNormal(a.sapiens, dev, sign=(1.0, -1.0, -1.0))
    for label in models:
        step = load(label); od = out / label; od.mkdir(exist_ok=True); rows = []; ex = []
        with torch.no_grad():
            for s, G in VAL.items():
                for src, tg in G:
                    av = avatar(src)
                    for ref in tg:
                        t = to_dev(F.collate1(fbd.target_info(*ref)), dev); Np = sap(t["image"])
                        Nr, Dr, Ar = NP.render_geometry(ren, av(t), t["render_cam_params"], t["w2c_cam"])
                        Nd = NP.depth_to_normal(Dr, F.batch_invtanfov(t, 1, dev))
                        v = NP.erode((t["mask"][:, :1] > 0.5).float(), 2) * NP.erode((Ar > 0.5).float(), 2); nv = v.sum().clamp(min=1)
                        rows.append({"track": s, "tgt": f"{clips[ref[0]]['name']}/{ref[1]}",
                                     "geo_cos_rendered": float(((Nr * Np).sum(1, keepdim=True) * v).sum() / nv),
                                     "geo_cos_dnormal": float(((Nd * Np).sum(1, keepdim=True) * v).sum() / nv)})
                        if len(ex) < 4 and len(rows) % 60 == 1:
                            ex.append(torch.cat([t["image"][0] * t["mask"][0], (Np[0] * .5 + .5) * v[0], (Nr[0] * .5 + .5) * v[0], (Nd[0] * .5 + .5) * v[0]], 2))
                print(label, s, "done", len(rows), flush=True)
        summ = {s: {k: float(np.mean([r[k] for r in rows if r["track"] == s])) for k in ("geo_cos_rendered", "geo_cos_dnormal")} for s in VAL}
        summ["all"] = {k: float(np.mean([r[k] for r in rows])) for k in ("geo_cos_rendered", "geo_cos_dnormal")}          # over all targets (registered)
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
