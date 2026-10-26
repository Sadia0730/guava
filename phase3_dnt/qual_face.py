"""DNT qualitative face check (inference only). Models: PEAR (released), L-real best.pt, A2-ctrl best.pt EMA, Track A
best.pt (live). Frames: 12 expressive UBody intra-scene test frames (face_valid; ranked by the z-scored sum of the UBody
pseudo-GT jaw angle and expression norm; at most one per video), 6 EHF frames (evenly spaced), 6 C1 test frames (first 6
tracked identities of the C1 split, middle future frame; the 256 px driver input). Per frame and model: full-mesh
overlay on the 256 px input crop, and a zoomed face crop with the FLAME-region wireframe and the 68 projected face
landmarks (EHM joints 55:123). Also per model: mean jaw opening (FLAME jaw angle, deg) and expression magnitude
(||FLAME expression||, mean |body exp|). Output: /raid/ubx858/outputs/dnt/qual_face/."""
import importlib.util, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--extra", action="append", default=[], help="NAME=student checkpoint (added after the four reference models)")
_ap.add_argument("--out", default="/raid/ubx858/outputs/dnt/qual_face")
_ap.add_argument("--head", choices=("ehm", "smplx"), default="ehm", help="mesh head of the --extra models (eval_3dpw_standard.HEADS); "
                 "the four reference models keep their EHM (FLAME) head")
_ap.add_argument("--extra-model", default="student_vitpose", choices=("student_vitpose", "sprint", "sprint_raw"), help="model kind of the --extra checkpoints")
_ap.add_argument("--camera", action="store_true", help="SPRINT --extra models only: cond={'cam': ...} for UBody (f = image width, real 0) and "
                 "EHF (EHF camera, real 1) frames; C1 driver frames (no original-image geometry) keep the empty camera token. Default off")
_ap.add_argument("--camera-width-only", action="store_true", help="with --camera: EHF frames also use focal = image width, real 0")
_ap.add_argument("--refiner", default=None, help="optional no-render refiner checkpoint applied to the --extra models only "
                 "(eval_3dpw_standard.RefinedModel); default none")
_args = _ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; OUT = Path(_args.out); OUT.mkdir(parents=True, exist_ok=True)
sys.argv = sys.argv[:1]
sys.path.insert(0, str(ROOT / "tools")); import eval_ehf_standard as E; ev = E.ev
def by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
U = by_path("guava_eval_ubody", ROOT / "tools/eval_ubody_standard.py")
LM = by_path("guava_lmdb", ROOT / "utils/lmdb.py")
dev = torch.device("cuda:0")

# ---- frames ----
frames = []                                                                     # (tag, rgb uint8 256x256, info)
meta = json.loads(Path("/raid/ubx858/datasets/eval_suite/ubody/ubody_intra_test.json").read_text())["items"]
cand = [it for it in meta if it["face_valid"]]
jaw = np.array([np.linalg.norm(it["smplx_param"]["jaw_pose"]) for it in cand]); ex = np.array([np.linalg.norm(it["smplx_param"]["expr"]) for it in cand])
score = (jaw - jaw.mean()) / jaw.std() + (ex - ex.mean()) / ex.std()
seen = set()
ds_ub = U.UBodyCrops(ev, cand, Path("/raid/ubx858/datasets/eval_suite/ubody/images"))
for i in np.argsort(-score):
    vid = (cand[i]["scene"], cand[i]["file_name"].rsplit("/", 1)[0] if "/" in cand[i]["file_name"] else cand[i]["file_name"].rsplit("_", 1)[0])
    if vid in seen: continue
    seen.add(vid); s = ds_ub[i]
    frames.append(("ubody", s["image"].permute(1, 2, 0).numpy(), {"file": cand[i]["file_name"], "gt_jaw_deg": float(np.degrees(jaw[i])), "gt_expr_norm": float(ex[i])}))
    if _args.camera:
        _h, _w = cv2.imread(str(Path("/raid/ubx858/datasets/eval_suite/ubody/images") / cand[i]["scene"] / cand[i]["file_name"])).shape[:2]
        frames[-1][2]["_cam"] = ev.cam_feat(s["affine"].numpy(), _w, _h, _w, 0.0)
    if len(seen) == 12: break
ehf = E.load_frames()
for j in np.linspace(0, len(ehf) - 1, 6).round().astype(int):
    img, _box = E.crop(ehf[j]); frames.append(("ehf", img.permute(1, 2, 0).numpy() if torch.is_tensor(img) else img, {"id": ehf[j]["id"]}))
    if _args.camera:
        _h, _w = cv2.imread(ehf[j]["image"]).shape[:2]
        frames[-1][2]["_cam"] = ev.cam_feat(ev.box_to_affine(_box), _w, _h, *((_w, 0.0) if _args.camera_width_only else (E.FOCAL[0], 1.0)))
split = json.loads(Path("/raid/ubx858/outputs/dnt/c1/split.json").read_text())["clips"]
n = 0
for c in split:
    d = Path("/raid/ubx858/outputs/dnt/c1/tracked") / c["clip"]
    if not (d / "optim_tracking_ehm.pkl").exists(): continue
    eng = LM.LMDBEngine(str(d / "img_lmdb"), write=False)
    keys = sorted(json.loads((d / "videos_info.json").read_text()).values().__iter__().__next__()["frames_keys"])
    k = keys[60 + (len(keys) - 60) // 2]
    x = eng[f"{k}/body_image"].float()[None] / 255.0
    x = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
    frames.append(("c1", (x[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy(), {"clip": c["clip"], "frame": k})); eng.close()
    n += 1
    if n == 6: break
print(len(frames), "frames", flush=True)
imgs = torch.stack([torch.from_numpy(np.ascontiguousarray(f[1])).permute(2, 0, 1) for f in frames]).to(dev).float() / 255.0

# ---- models ----
os.chdir(ev.PEAR_ROOT); sys.path.insert(0, str(ev.PEAR_ROOT))
for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]: sys.modules.pop(k)
from models.modules.ehm import EHM_v2
ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(dev).eval()
faces = ehm.smplx.faces_tensor.cpu().numpy(); faces = faces[(faces < 10475).all(1)]          # SMPL-X topology only
fid = np.load(E.SMPLX_ASSETS / "SMPL-X__FLAME_vertex_ids.npy"); fmask = np.zeros(10475, bool); fmask[fid] = True
face_tris = faces[fmask[faces].all(1)]
MODELS = {"PEAR": ("pear", None), "L-real": ("student_vitpose", "/raid/ubx858/outputs/phase1_task1/screen/L_real/checkpoints/best.pt"),
          "A2-ctrl EMA": ("student_vitpose", "/raid/ubx858/outputs/dnt/trackA2/eval/ckpt/A2ctrl_best_ema.pt"),
          "Track A": ("student_vitpose", "/raid/ubx858/outputs/dnt/trackA/run/checkpoints/best.pt")}
MODELS.update({e.split("=", 1)[0]: (_args.extra_model, e.split("=", 1)[1]) for e in _args.extra})
HEAD = {n: (ev.effective_head(MODELS[n][0], _args.head) if n in {e.split("=", 1)[0] for e in _args.extra} else "ehm") for n in MODELS}
res, stats = {}, {}
for name, (kind, ck) in MODELS.items():
    m, _ = ev.load_model(kind, Path(ck).resolve() if ck else ev.DEFAULT_TEACHER.resolve(), "configs/student_l70_v2.yaml", 8, dev)
    if _args.refiner and name in {e.split("=", 1)[0] for e in _args.extra}:
        m = ev.RefinedModel(m, Path(_args.refiner).resolve(), dev)
    outs = []
    cam_on = _args.camera and kind in ev.SPRINT_KINDS; step = 1 if cam_on else 8   # per frame: C1 rows have no camera
    with torch.inference_mode():
        for s in range(0, len(frames), step):
            c = frames[s][2].get("_cam") if cam_on else None
            o = m(imgs[s:s + step], **ev.camera_kwargs(kind, c is not None, None if c is None else c[None], dev)); o = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float()) for k, v in o.items()}
            mesh = ev.predicted_mesh(o, ehm, HEAD[name])
            cam = o["pd_cam"]
            def proj(P):
                c = torch.einsum("bij,bkj->bki", cam[:, :3, :3], P) + cam[:, None, :3, 3]
                return ((1.0 - 24.0 * c[..., :2] / c[..., 2:3].clamp_min(1e-4)) * 0.5 * 256).cpu().numpy(), c[..., 2].cpu().numpy()
            uv, z = proj(mesh["vertices"][:, :10475]); lm, _ = proj(mesh["joints"][:, 55:123])
            fp, bp = o["flame_param"], o["body_param"]
            for b in range(uv.shape[0]):
                outs.append({"uv": uv[b], "z": z[b], "lm": lm[b], "jaw_deg": float(np.degrees(fp["jaw_params"][b].norm().item())),
                             "flame_expr_norm": float(fp["expression_params"][b].norm()), "body_exp_abs": float(bp["exp"][b].abs().mean()),
                             "smplx_exp_norm": float(bp["exp"][b].norm()),   # SMPL-X 2020 expression (50): the face of pure SMPL-X heads (SPRINT)
                             "head_scale": float(bp["head_scale"][b].mean())})
    res[name] = outs; del m; torch.cuda.empty_cache()
    for tag in ("ubody", "ehf", "c1", "all"):
        sel = [o for o, f in zip(outs, frames) if tag == "all" or f[0] == tag]
        stats.setdefault(name, {})[tag] = {k: float(np.mean([o[k] for o in sel])) for k in ("jaw_deg", "flame_expr_norm", "smplx_exp_norm", "body_exp_abs", "head_scale")}
    print(name, stats[name]["all"], flush=True)

# ---- drawing ----
def overlay(img, o):
    out = img.copy(); uv, z = o["uv"], o["z"]
    tri = faces[np.argsort(-z[faces].mean(1))]
    layer = img.copy()
    for t in tri:
        p = uv[t].astype(np.int32)
        n = np.cross(np.r_[p[1] - p[0], 0], np.r_[p[2] - p[0], 0])[2]
        if n >= 0: continue                                                         # back-facing in the image
        cv2.fillConvexPoly(layer, p, (90, 160, 230))
    return cv2.addWeighted(layer, 0.55, out, 0.45, 0)
def face_zoom(img, o):
    q = o["uv"][fid]; lo, hi = q.min(0), q.max(0); c = (lo + hi) / 2; s = max(hi - lo) * 1.3 / 2 + 4
    x0, y0 = c - s; scale = 256.0 / (2 * s)
    A = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]], np.float32)
    z = cv2.warpAffine(img, A, (256, 256), flags=cv2.INTER_LINEAR)
    T = lambda P: (P - [x0, y0]) * scale
    uvz = T(o["uv"])
    for t in face_tris[::2]:
        p = uvz[t].astype(np.int32); cv2.polylines(z, [p], True, (80, 220, 80), 1, cv2.LINE_AA)
    for p in T(o["lm"]): cv2.circle(z, (int(p[0]), int(p[1])), 2, (255, 40, 40), -1)
    return z
font = cv2.FONT_HERSHEY_SIMPLEX
rows_all = []
for fi, (tag, img, info) in enumerate(frames):
    img = np.ascontiguousarray(img)
    top = [img] + [overlay(img, res[n][fi]) for n in MODELS]; bot = [np.zeros_like(img)] + [face_zoom(img, res[n][fi]) for n in MODELS]
    for j, n in enumerate(["input"] + list(MODELS)):
        cv2.putText(top[j], n, (4, 14), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if j: cv2.putText(bot[j], f"jaw {res[n][fi]['jaw_deg']:.1f}d |e|fl {res[n][fi]['flame_expr_norm']:.2f} |e|sx {res[n][fi]['smplx_exp_norm']:.2f}", (4, 250), font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    g = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    cv2.imwrite(str(OUT / f"{fi:02d}_{tag}.jpg"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    rows_all.append(g)
json.dump({**({"refiner": _args.refiner} if _args.refiner else {}), "frames": [{"tag": f[0], **{k: v for k, v in f[2].items() if k != "_cam"}} for f in frames], "stats": stats, "head": HEAD,
           **({"camera": {n: bool(_args.camera and MODELS[n][0] in ev.SPRINT_KINDS) for n in MODELS},
               "camera_frames": [f[0] for f in frames if "_cam" in f[2]]} if _args.camera else {"camera": False}),
           "per_frame": {n: [{k: v for k, v in o.items() if k not in ("uv", "z", "lm")} for o in res[n]] for n in MODELS}},
          open(OUT / "summary.json", "w"), indent=1)
print("QUAL_FACE_DONE", flush=True)
