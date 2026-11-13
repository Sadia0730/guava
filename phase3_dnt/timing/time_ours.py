"""RAGA / GUAVA laptop phase timing, same evaluated model configuration.

Source construction, animation, and feature accumulation are separate operations.
Supplied-input rows consume the evaluated masks/parameters; segmentation is null
because no fresh matte is computed in that path. Estimated source/support fitting
includes its actual matting and keypoint/refinement stages. Target driving is raw
shape-conditioned regression (no target fitting and no target segmentation).

Animation is measured after the updated avatar is constructed for update rows.
A phase-synchronized diagnostic pass and an independent uninstrumented serial pass
are reported separately. Neither establishes webcam/application throughput. Use
run_live_replay.py for measured application replay FPS with the worker off/on.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; P3 = HERE.parent; ROOT = P3.parent
sys.path.insert(0, str(HERE))
import phase_timer as PT                                                                             # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", required=True, help="timing subset root (manifest.json)"); ap.add_argument("--out-dir", required=True)
ap.add_argument("--rows", default="fb_gt,fb_sprint,fb_update,fb_sprint_update,guava"); ap.add_argument("--clips", default="")
ap.add_argument("--tag", default="", help="suffix of the output JSON names (e.g. ft for the fine-tuned weights)")
ap.add_argument("--assets", default=str(ROOT / "live_assets"), help="package asset root (checkpoints/)")
ap.add_argument("--fb-ckpt", default=""); ap.add_argument("--sprint-ckpt", default="")
ap.add_argument("--guava-ckpt", default=str(ROOT / "assets/GUAVA/checkpoints/best_160000.pt"))
ap.add_argument("--box", default="center", choices=("yolox", "center"), help="est. rows: person box per frame (live app --crop)")
ap.add_argument("--warmup", type=int, default=PT.PROTOCOL["warmup"]); ap.add_argument("--iters", type=int, default=PT.PROTOCOL["iters"])
ap.add_argument("--recon-warmup", type=int, default=PT.PROTOCOL["recon_warmup"]); ap.add_argument("--recon-reps", type=int, default=PT.PROTOCOL["recon_reps"])
ap.add_argument("--upd-frames", type=int, default=6); ap.add_argument("--src-iters", type=int, default=150); ap.add_argument("--upd-iters", type=int, default=60)
ap.add_argument("--dwpose-threads", type=int, default=4); ap.add_argument("--power-note", default="")
a = ap.parse_args()
T = Path(a.root).resolve(); OUTD = Path(a.out_dir).resolve(); ROWS = a.rows.split(",")
VALID_ROWS = {"fb_gt", "fb_sprint", "fb_update", "fb_sprint_update", "guava"}
if set(ROWS) - VALID_ROWS: ap.error(f"unknown rows: {set(ROWS) - VALID_ROWS}")
if not ROWS or min(a.iters,a.recon_reps,a.upd_frames)<1 or min(a.warmup,a.recon_warmup)<0:
    ap.error("positive timed counts and nonnegative warmups required")
if OUTD.is_relative_to(T) or OUTD.is_relative_to('/raid') or OUTD.is_relative_to('/vault'):
    ap.error("write results to a fresh local directory outside the input bundle, /raid and /vault")
OUTD.mkdir(parents=True, exist_ok=True)
NEED_POSE = any(r in ROWS for r in ("fb_sprint", "fb_sprint_update"))
NEED_FB = any(r != "guava" for r in ROWS)
ASSETS = Path(a.assets)
if (ASSETS / "eval_assets").exists(): os.environ.setdefault("SPRINT_EVAL_ASSETS", str(ASSETS / "eval_assets"))
ARC = ASSETS / "arcface" / "glintr100.onnx"
if ARC.exists(): os.environ.setdefault("ARCFACE_ONNX", str(ARC))


def first(*ps):
    for p in ps:
        if p and Path(p).exists(): return str(p)
    raise FileNotFoundError(f"none of {ps}")


FB_CKPT = (a.fb_ckpt or first(ASSETS / "checkpoints/guava_fb_B28k.pt")) if NEED_FB else None
SPRINT_CKPT = (a.sprint_ckpt or first(ASSETS / "checkpoints/sprint_scratch_ema128k.pt")) if NEED_POSE else None
MAN = json.loads((T / "manifest.json").read_text()); CLIPS = MAN["clips"]
if a.clips:
    requested = set(a.clips.split(",")); CLIPS = [c for c in CLIPS if c["id"] in requested]
    if requested - {c["id"] for c in CLIPS}: ap.error("unknown clip requested")
if not CLIPS: ap.error("no clips selected")

# SPRINT first (it loads PEAR's packages, then purges them), as eval_guava_rows.py / the live app
spec = importlib.util.spec_from_file_location("sprint_avatar_fit", P3 / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = None
if NEED_POSE:
    fitter = SAF.SprintFitter(SPRINT_CKPT, "cuda:0", a.src_iters, dwpose_threads=a.dwpose_threads)
    fitter.empty_matte_fix = True

import cv2, roma, torch                                                                              # noqa: E401,E402
sys.path.insert(0, str(P3 / "guava_fb")); sys.path.insert(0, str(P3 / "worker6")); sys.path.insert(0, str(P3 / "eval_split")); sys.path.insert(0, str(P3))
import fb_common as F                                                                                # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                         # noqa: E402
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer              # noqa: E402
dev = torch.device("cuda:0"); GPU = torch.cuda.get_device_name(0)
cfg = F.get_cfg(); inf = ren = None
if NEED_FB:
    cfg, inf, ren, META = F.setup_a20(dev, ckpt=FB_CKPT); inf.eval(); ren.eval()
UPD_FNS = None
if any(r in ROWS for r in ("fb_update", "fb_sprint_update")):
    import uv_fusion as UF                                                                           # noqa: E402
    import uv_update_fns as UPD                                                                      # noqa: E402
    UPD_FNS = UPD.make(inf, cfg, dev, F, UF)

YOLO = None; DET_NAME = "not used (supplied parameters)"
if NEED_POSE:
    try:                                                                                                 # the live app's GPU detector
        import onnx, onnx2torch                                                                          # noqa: E401

        class TorchYOLOX:
            class _In:
                name = "images"

            def __init__(self, path): self.m = onnx2torch.convert(onnx.load(str(path))).to(dev).eval()
            def get_inputs(self): return [self._In()]

            @torch.no_grad()
            def run(self, _, feed):
                x = torch.from_numpy(np.ascontiguousarray(next(iter(feed.values())))).to(dev, non_blocking=True)
                return [self.m(x).float().cpu().numpy()]
        YOLO, DET_NAME = TorchYOLOX(SAF.EHMT / "pretrained/dwpose/yolox_l.onnx"), "YOLOX-L on the GPU (onnx2torch), as the live app"
    except ImportError:
        YOLO, DET_NAME = fitter.s_det, "YOLOX-L on the CPU (onnxruntime; onnx2torch missing, so NOT the live app's GPU detector)"


def detect(img, ref):
    """largest box at the start, then the highest IoU with the previous box (live app's detect, without H1c's fallbacks)."""
    D = fitter.det.inference_detector(YOLO, img)
    if len(D) == 0: return np.asarray(ref, np.float64) if ref is not None else np.array([0, 0, img.shape[1] - 1, img.shape[0] - 1], np.float64)
    if ref is None: return np.asarray(D[int(np.argmax((D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1])))], np.float64)
    def iou(p, q):
        w, h = max(0.0, min(p[2], q[2]) - max(p[0], q[0])), max(0.0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
        return i / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i, 1e-9)
    return np.asarray(D[int(np.argmax([iou(d, ref) for d in D]))], np.float64)


def center_box(img):
    H, W = img.shape[:2]; h = H / 1.1 + 1e-6
    return np.array([W / 2 - h / 2, H / 2 - h / 2, W / 2 + h / 2, H / 2 + h / 2], np.float64)


def crop_up(img, box):
    M = SAF.box_crop_matrix(box, SAF.CROP); crop = cv2.warpAffine(img, M[:2], (SAF.CROP, SAF.CROP), flags=cv2.INTER_LINEAR)
    return torch.from_numpy(crop).to(dev).permute(2, 0, 1)[None].float() / 255.0, M


def sprint_raw(img, crop_t, M, shp):
    """live_sprint_guavafb.live_raw's SPRINT call + raw conversion."""
    H, W = img.shape[:2]; cam = fitter.ev.cam_feat((np.diag([256 / SAF.CROP, 256 / SAF.CROP, 1.0]) @ M)[:2], W, H, W, 0.0)
    with torch.no_grad(): out = fitter.sprint(crop_t, cam, shp)
    bp0 = out["body_param"]; aa = lambda Rm: roma.rotmat_to_rotvec(Rm.reshape(-1, 3, 3))                # noqa: E731
    flat = torch.cat([aa(bp0[k]).reshape(-1) for k in ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose")]
                     + [bp0["exp"][0, :50].reshape(-1), out["pd_cam"][0, :3, :4].reshape(-1)]).cpu().numpy().astype(np.float32)
    s = np.cumsum([0, 3, 63, 45, 45, 3, 50, 12])
    return {"global_pose": flat[s[0]:s[1]], "body_pose": flat[s[1]:s[2]].reshape(21, 3), "left_hand_pose": flat[s[2]:s[3]].reshape(15, 3),
            "right_hand_pose": flat[s[3]:s[4]].reshape(15, 3), "jaw_pose": flat[s[4]:s[5]].reshape(1, 3), "eye_pose": np.zeros((2, 3), np.float32),
            "exp": flat[s[5]:s[6]], "camera_RT_params": flat[s[6]:s[7]].reshape(3, 4), "body_cam": np.zeros(3, np.float32)}


def target_from_smplx(sm_np, invtanfov, flame=None):
    """render inputs from a compact SMPL-X record (the live app's render(), and FBData.info without the image read)."""
    sm = {k: torch.as_tensor(np.asarray(v).copy()) for k, v in sm_np.items()}
    sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
    M4 = torch.eye(4); M4[:3, :4] = sm["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
    t = {"smplx_coeffs": sm, "flame_coeffs": flame if flame is not None else {"expression_params": torch.zeros(50)}, "w2c_cam": w2c,
         "c2w_cam": torch.linalg.inv(w2c), "invtanfov": float(invtanfov)}
    t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
    return to_dev(F.collate1(t), dev)


def target_live(raw, box, M, shp):
    Mlr = SAF.box_crop_matrix(box, SAF.BODY_LR); Mp = SAF.box_crop_matrix(box, SAF.PART)
    rec = {"smplx_coeffs": raw, "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": M, "M_c2o-hd": np.linalg.inv(M).astype(np.float32)},
           **{p: {"M_o2c": Mp, "M_c2o": np.linalg.inv(Mp).astype(np.float32)} for p in ("head_crop", "left_hand_crop", "right_hand_crop")}}
    comp = F.compact_record(rec, shp); return target_from_smplx(comp["smplx"], comp["invtanfov"])


@torch.no_grad()
def deform_render(P, av, rr, t):
    with P("deformation"): torch.manual_seed(0); ga = av(t)
    with P("rendering_raster"): raw = rr.forward_raw(ga, t["render_cam_params"], bg=0.0)
    with P("rendering_refiner"): out = rr.refine_raw(raw)
    return out


ORIGINALS = {}
def orig(c, k):
    key = (c["id"],k)
    if key not in ORIGINALS:
        path = T / "orig" / c["dataset"] / c["clip"] / f"{k}.png"
        img = cv2.imread(str(path))
        if img is None: raise FileNotFoundError(path)
        ORIGINALS[key] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return ORIGINALS[key]


class Rel(F.FBData):
    """FBData + FLAME fields (released GUAVA's EHM reads them), as eval_guava_rows.FBDataRel."""
    def info(self, ci, k):
        r, img, msk = super().info(ci, k); c = self.clips[ci]
        fc = {kk: torch.as_tensor(np.asarray(v, np.float32).reshape(-1)) for kk, v in c["flame"][k].items()
              if kk in ("expression_params", "jaw_params", "pose_params", "neck_pose_params", "eye_pose_params", "eyelid_params")}
        fc["shape_params"] = torch.as_tensor(c["flame_shape"]); r["flame_coeffs"] = fc
        return r, img, msk


def loop(n_warm, n_time, fn):
    P = PT.Phases()
    for i in range(n_warm + n_time):
        P.start_run(record=i >= n_warm); fn(P, i); P.end_run()
    S = P.summary()
    for name,parts in {"rendering":("rendering_raster","rendering_refiner"),
                       "pose":("pose_detect","pose_keypoints","pose_sprint","pose_refine"),
                       "network":("network_inferer","network_assembly"),
                       "preprocess":("preprocess_crop","preprocess_record")}.items():
        if any(k in P.t for k in parts): S[name] = P.combined(*parts)
    return S


def fill(S, names):
    for n in names: S.setdefault(n, None)
    return S


def matte_of(img, box):
    crop_t, _ = crop_up(img, box)
    with torch.no_grad(): return fitter.matting(crop_t)


@torch.no_grad()
def build(i_, src, gs=Ubody_Gaussian, P=None):
    with P("network_inferer"): torch.manual_seed(0); vg, ug, _ = i_(src)
    with P("network_assembly"): av = gs(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(i_.ehm); av.eval()
    return av


def fit_to_batch(rs, key, name="timing"):
    with tempfile.TemporaryDirectory() as td:
        SAF.write_tracked(Path(td), name, key, rs)
        cl = F.load_clips_parallel([(td, None, True)], workers=1); fb = F.FBData(cfg, cl)
        return to_dev(F.collate1(fb.source_info(0, key)), dev)


def run_clip(row, c, models):
    ds, name = c["dataset"], c["clip"]; E = T / c["E"]; cdir = E / "data" / ds / name
    cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1)
    i_, r_ = models
    if row == "guava":
        raw = F.load_pkl(cdir / "optim_tracking_ehm.pkl"); ids = F.load_pkl(cdir / "id_share_params.pkl")
        cl[0]["flame"] = {k: raw[k]["flame_coeffs"] for k in raw}; cl[0]["flame_shape"] = np.asarray(ids["flame_shape"], np.float32).reshape(300)
        fbd = Rel(cfg, cl)
    else:
        fbd = F.FBData(cfg, cl)
    src_key, test = c["source"], c["test"]
    est = row in ("fb_sprint", "fb_sprint_update"); notes = []
    support = c.get("support", [k for k in c["recon"] if k != src_key])[:a.upd_frames]
    img_src = orig(c,src_key) if est else None
    box_src = center_box(img_src) if est else None
    if est:
        for key in [src_key]+test+support: orig(c,key)  # decoding is outside the component timers
    def add_fit_parts(P, result):
        times = result["times"]
        for label,key in [("pose_keypoints","dwpose_cpu"),("segmentation","matting"),
                          ("pose_sprint","sprint"),("pose_refine","refine")]:
            P.add(label,times[key]*1000)
    if not est: notes.append("supplied masks and parameters; no fresh segmentation or source fitting")
    out = {}

    # ---------------------------------------------------------------- reconstruction (photo -> avatar)
    st = {}

    def recon(P, i):
        if est:
            with P("pose_detect"): b = detect(img_src, None)
            with P("pose_fit"): rs = fitter.fit(img_src, box=b)                         # DWPose kp + StyleMatte + SPRINT + 150 steps
            add_fit_parts(P,rs)
            with P("preprocess"): src = fit_to_batch(rs, src_key)
            st["shape"] = rs["shape"]; st["box"] = rs["box"]
        else:
            with P("preprocess"): src = to_dev(F.collate1(fbd.source_info(0, src_key)), dev)
        st["src"] = src; st["av"] = build(i_, src, P=P)
    S = loop(a.recon_warmup, a.recon_reps, recon)
    out["reconstruction"] = fill(S, ["preprocess", "segmentation", "pose", "network"])
    av, src = st["av"], st["src"]

    # ---------------------------------------------------------------- independent accumulation / decoder-refresh timing
    if row in ("fb_update", "fb_sprint_update"):
        if not support or set(support)&set(test): raise ValueError("disjoint reconstruction support is required")
        FRAME_FEATS, DECODE, MEANFUSION = UPD_FNS
        seed_timer=PT.Phases();seed_timer.start_run()
        with seed_timer('source_features'):f0 = FRAME_FEATS(src)
        with seed_timer('accumulator_init'):fus = MEANFUSION(f0, prior=2.0)
        seed_timer.end_run();out['update_initialization']=seed_timer.summary()
        P = PT.Phases(); accepted = rejected = 0
        for n in range(3+len(support)):
            if n==3: fus = MEANFUSION(f0, prior=2.0)  # discard warm-up state
            k=support[(n-3)%len(support)]; P.start_run(record=n>=3)
            if est:
                with P("pose_detect"): b=detect(orig(c,k),None)
                with P("pose_fit"): rf=fitter.fit(orig(c,k),box=b,iters=a.upd_iters,shape_lock=st["shape"])
                add_fit_parts(P,rf)
                valid=(not rf["diag"]["matte_empty"] and np.isfinite(rf["diag"]["kp_err_px1024_after"])
                       and rf["diag"]["kp_err_px1024_after"]<=20)
                with P("preprocess"): b2=fit_to_batch(rf,k) if valid else None
            else:
                valid=True
                with P("preprocess"): b2=to_dev(F.collate1(fbd.source_info(0,k)),dev)
            if valid:
                with P("features"): feat=FRAME_FEATS(b2)
                with P("fuse"): fus.add(feat)
                with P("decode"): av=DECODE(src,f0,fus.uvmap())
                if n>=3: accepted+=1
            elif n>=3: rejected+=1
            P.end_run()
        out["update"]=P.summary()
        out["update"]["pose"] = P.combined("pose_detect","pose_keypoints","pose_sprint","pose_refine")
        for name in ["segmentation","pose","features","fuse","decode"]: out["update"].setdefault(name,None)
        out["update_counts"]={"attempted":len(support),"accepted":accepted,"rejected":rejected,"unique_support_keys":support}
        notes.append("Update profile: six-or-fewer unique reconstruction supports; warm-up state discarded. "
                     "Each accepted support refreshes the decoder. Rejected fits retain their cost. "
                     "Rendering below uses the final decoded avatar. This is a component profile, not the experimental worker.")

    # ---------------------------------------------------------------- animation (per driving frame)
    seq = [test[i % len(test)] for i in range(a.warmup + a.iters)]
    recs = cl[0]["recs"]; ctx = {"box": st.get("box", box_src)}

    def flame_of(k):                                                                     # released GUAVA's EHM reads the FLAME record
        if row != "guava": return None
        fc = {kk: torch.as_tensor(np.asarray(v, np.float32).reshape(-1)) for kk, v in cl[0]["flame"][k].items()
              if kk in ("expression_params", "jaw_params", "pose_params", "neck_pose_params", "eye_pose_params", "eyelid_params")}
        fc["shape_params"] = torch.as_tensor(cl[0]["flame_shape"]); return fc

    def anim(P, i):
        k = seq[i % len(seq)]
        if est:
            img = orig(c, k)                                                             # decoded frame (capture not timed)
            if a.box == "center": b = center_box(img)
            else:
                with P("pose_detect"): b = detect(img, ctx["box"])
            ctx["box"] = b
            with P("preprocess_crop"): crop_t, M = crop_up(img, b)
            with P("pose_sprint"): raw = sprint_raw(img, crop_t, M, st["shape"])
            with P("preprocess_record"): t = target_live(raw, b, M, st["shape"])
        else:
            with P("preprocess_record"): t = target_from_smplx(recs[k]["smplx"], recs[k]["invtanfov"], flame_of(k))
        deform_render(P, av, r_, t)
    A = loop(a.warmup, a.iters, anim)
    if not est: A["preprocess"] = A.get("preprocess_record")
    A["serial_total"] = PT.measure_serial(anim,a.warmup,a.iters)
    out["animation"] = fill(A, ["pose", "segmentation", "preprocess", "deformation", "rendering"])
    out["animation_boundary"] = "decoded frame/parameters to 512 GPU image; no capture, PNG decode, display or worker"
    with torch.no_grad():                                                                # the split equals the renderer's forward
        t = target_from_smplx(recs[test[0]]["smplx"], recs[test[0]]["invtanfov"], flame_of(test[0])); torch.manual_seed(0); ga = av(t)
        d = (r_(ga, t["render_cam_params"], bg=0.0)["renders"] - r_.refine_raw(r_.forward_raw(ga, t["render_cam_params"], bg=0.0))).abs().max()
        out["split_check_max_abs_diff"] = float(d)
    out["gaussians"] = int(av._smplx_xyz.shape[1] + av._uv_local_xyz.shape[1])

    out["notes"] = notes
    return out


results = {}
for row in ROWS:
    if row == "guava":
        if not Path(a.guava_ckpt).exists():
            raise FileNotFoundError(f"released GUAVA checkpoint missing: {a.guava_ckpt}")
        cfg_g = F.get_cfg(); ig = Ubody_Gaussian_inferer(cfg_g.MODEL).to(dev); rg = GaussianRenderer(cfg_g.MODEL).to(dev)
        ck = torch.load(a.guava_ckpt, map_location="cpu", weights_only=False)
        ig.load_state_dict(ck["model"], strict=False); rg.load_state_dict(ck["render_model"], strict=False); del ck
        F.patch_inferer(ig); ig.eval(); rg.eval(); models = (ig, rg)
    else:
        models = (inf, ren)
    clips = {}
    for c in CLIPS:
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        clips[c["id"]] = run_clip(row, c, models); clips[c["id"]]["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        A = clips[c["id"]]["animation"]; R = clips[c["id"]]["reconstruction"]
        print(f"{row:16s} {c['id']:32s} recon total {R['total']['median_ms']:8.1f} ms | frame total {A['total']['median_ms']:6.1f} ms "
              f"(deform {A['deformation']['median_ms']:.1f}, raster {A['rendering_raster']['median_ms']:.1f}, refiner {A['rendering_refiner']['median_ms']:.1f})"
              f" | {time.time() - t0:.0f} s", flush=True)
    names = {"fb_gt": "RAGA, photo only (supplied)", "fb_sprint": "RAGA, photo only (estimated)", "fb_update": "RAGA + appearance update (supplied)",
             "fb_sprint_update": "RAGA + appearance update (estimated)", "guava": "GUAVA"}
    notes = {"checkpoint": a.guava_ckpt if row == "guava" else FB_CKPT, "sprint_ckpt": SPRINT_CKPT, "source_detector": DET_NAME, "driving_crop": a.box,
             "phases": __doc__, "live_note": "Application throughput must be measured separately with run_live_replay.py; no FPS guarantee."}
    proto = dict(PT.PROTOCOL, warmup=a.warmup, iters=a.iters, recon_warmup=a.recon_warmup, recon_reps=a.recon_reps, upd_frames=a.upd_frames,
                 src_iters=a.src_iters, upd_iters=a.upd_iters, box=a.box)
    tag = f"_{a.tag}" if a.tag else ""
    results[row] = PT.write(OUTD / f"ours_{row}{tag}.json", f"ours_{row}{tag}", names[row] + (f" [{a.tag}]" if a.tag else ""), clips, notes,
                            protocol=proto, power_note=a.power_note)
