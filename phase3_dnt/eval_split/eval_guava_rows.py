"""AvatarTest-v1 GUAVA-family rows (PREREG "AvatarTest-v1 addendum 1"): every test frame rendered at 512 (8-bit PNG, black
background) into E/renders/<row>/<dataset>/<clip>/<key>.png, plus run.json per clip (avatar-creation time, render time).
  fb_gt      GUAVA-FB (guava_fb_B30k best.pt, 28k) from the source frame's GT record + crop, driven by the GT test records.
  guava      released GUAVA (best_160000.pt; EHM model with the FLAME head), same source / targets; FLAME fields from the
             converter (flame_shape fit, expression = exp50, jaw / eyes); camera patch (per-record invtanfov). The EHM head
             vs the GT SMPL-X 2020 head is measured on 3 frames per clip (mm) and logged.
  fb_sprint  GUAVA-FB from SPRINT's fit of the source frame's ORIGINAL image (sprint_avatar_fit, 150 refinement steps), driven
             per test frame by SPRINT raw with the shape lock (Avatar A's SL); rendered at 512 in SPRINT's crop camera (GUAVA's
             output size) and warped into the metric crop with the exact homography (SPRINT crop -> original -> metric crop).
Seed 0 before every inference / render call (StyleUNet noise), as fb_common / compare_fb.
--timing N: reconstruction time only (PREREG: avatar creation from the source, CUDA-synchronised, GPU stated): N clips alternating
UBody / BEDLAM2, 3 warm-up builds, --reps timed builds per clip (median), median over clips; the SPRINT row adds SPRINT's fit of
the source image timed the same way. -> E/reports/recon_timing_<row>.json
Usage: CUDA_VISIBLE_DEVICES=1 python eval_guava_rows.py --row fb_gt [--clips id,id] [--timing 10]"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--row", required=True, choices=("fb_gt", "guava", "fb_sprint"))
ap.add_argument("--clips", default=""); ap.add_argument("--sprint-iters", type=int, default=150)
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
ap.add_argument("--guava-ckpt", default=str(ROOT / "assets/GUAVA/checkpoints/best_160000.pt"))
ap.add_argument("--sprint-ckpt", default="/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
ap.add_argument("--data-root", default="", help="local copy of E/data (random LMDB reads over SMB are slow)")
ap.add_argument("--timing", type=int, default=0, help="clean recon timing on an idle GPU: N clips (alternating datasets), no renders")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--split", default=str(HERE / "avatar_test_v1.json"))
ap.add_argument("--E", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1")
ap.add_argument("--track-nearest", action="store_true", help="H1c: if no detection overlaps the previous box, take the nearest-centre detection (not the stale box)")
ap.add_argument("--track-subject", action="store_true", help="H1b: select the subject once (source box) and track it by IoU, instead of the largest detection per frame")
ap.add_argument("--name", default="", help="output row name (default: the row, + _k<iters> for driving refinement)")
ap.add_argument("--drive-iters", type=int, default=0, help="H1: refine each SPRINT driving frame with k iterations of sprint_avatar_fit's objective (0 = raw, deployed)")
ap.add_argument("--raw", action="store_true", help="also write the raw Gaussian render (first 3 feature channels, no refiner) to <row>_raw")
ap.add_argument("--fast-backend", action="store_true", help="opt-in backend skipping unused work for raw driving; source fit unchanged")
ap.add_argument("--dwpose-provider", default="CPUExecutionProvider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"))
ap.add_argument("--driving-matte-size", type=int, default=1024, choices=(512, 1024), help="fast backend only: matte input for driving refinement; source always 1024")
ap.add_argument("--detector-fp16", action="store_true", help="fast CUDA backend: FP16 person-only detector for driving; source detector unchanged")
ap.add_argument("--sprint-graph", action="store_true", help="fast backend: CUDA Graph replay of unchanged FP32 SPRINT driving")
ap.add_argument("--box-source", default="yolox", choices=("yolox", "mediapipe", "track"), help="PREREG phase 5: driving-frame person box (yolox = H1b/H1c path)")
ap.add_argument("--box-margin", type=float, default=1.25, help="phase 5: margin on the landmark / keypoint box (mediapipe, track)")
ap.add_argument("--dwpose-threads", type=int, default=0, help="DWPose onnxruntime threads (0 = SprintFitter's default, 16); bound it on a CPU-saturated host")
a = ap.parse_args()
if not a.fast_backend and a.dwpose_provider != 'CPUExecutionProvider':
    ap.error('--dwpose-provider CUDAExecutionProvider requires --fast-backend')
if a.driving_matte_size != 1024 and not a.fast_backend:
    ap.error('--driving-matte-size 512 requires --fast-backend')
if (a.detector_fp16 or a.sprint_graph) and (not a.fast_backend or a.drive_iters != 0):
    ap.error('--detector-fp16/--sprint-graph require --fast-backend with k=0')
if a.detector_fp16 and a.dwpose_provider != 'CUDAExecutionProvider':
    ap.error('--detector-fp16 requires --dwpose-provider CUDAExecutionProvider')
SPLIT = json.loads(Path(a.split).read_text())
E = Path(a.E)
ROWNAME = a.name or (a.row + (f"_k{a.drive_iters}" if a.drive_iters else ""))
OUT = E / "renders" / ROWNAME
DATA = Path(a.data_root) if a.data_root else E / "data"
LOCAL = Path("/dev/shm/ubx858_eval/renders") / ROWNAME                                               # written here, then one bulk copy per clip
clips = SPLIT["clips"]
if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]

fitter = None
if a.row == "fb_sprint":                                    # SPRINT first: it loads PEAR's packages, then purges them
    fitter_file = 'sprint_fast_driving.py' if a.fast_backend else 'sprint_avatar_fit.py'
    spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / 'phase3_dnt' / fitter_file)
    SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
    fitter_kwargs = {'dwpose_provider': a.dwpose_provider, 'driving_matte_size': a.driving_matte_size,
                     'detector_fp16': a.detector_fp16, 'sprint_graph': a.sprint_graph} if a.fast_backend else {}
    if a.dwpose_threads: fitter_kwargs['dwpose_threads'] = a.dwpose_threads
    fitter = SAF.SprintFitter(a.sprint_ckpt, "cuda:0", a.sprint_iters, **fitter_kwargs)

import cv2, torch, torchvision                                                                       # noqa: E402
if a.box_source != "yolox":                                                                          # phase 5 box sources
    assert a.track_subject and a.track_nearest and a.drive_iters == 0, "box sources need H1c linking for re-detection, k = 0"
    import box_sources as BS
if os.environ.get("OMP_NUM_THREADS"): cv2.setNumThreads(int(os.environ["OMP_NUM_THREADS"]))          # bounded OpenCV threads when asked
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import fb_common as F                                                                                # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                         # noqa: E402
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer              # noqa: E402
dev = torch.device("cuda:0"); GPU = torch.cuda.get_device_name(0)


def load_ckpt(inf, ren, path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    r1 = inf.load_state_dict(st["model"], strict=False); r2 = ren.load_state_dict(st["render_model"], strict=False)
    return {"ckpt": path, "step": st.get("step", st.get("global_iter")), "missing": len(r1.missing_keys) + len(r2.missing_keys),
            "unexpected": len(r1.unexpected_keys) + len(r2.unexpected_keys)}


if a.row == "guava":
    cfg = F.get_cfg(); inf = Ubody_Gaussian_inferer(cfg.MODEL).to(dev); ren = GaussianRenderer(cfg.MODEL).to(dev)
    CK = load_ckpt(inf, ren, a.guava_ckpt); F.patch_inferer(inf)
else:
    cfg, inf, ren, _ = F.setup_a20(dev); CK = load_ckpt(inf, ren, a.fb_ckpt)
inf.eval(); ren.eval()
print("model", CK, GPU, flush=True)


class FBDataRel(F.FBData):
    """FBData + the record's FLAME fields (released GUAVA's EHM reads them)."""
    def info(self, ci, k):
        r, img, msk = super().info(ci, k); c = self.clips[ci]
        fc = {kk: torch.as_tensor(np.asarray(v, np.float32).reshape(-1)) for kk, v in c["flame"][k].items()
              if kk in ("expression_params", "jaw_params", "pose_params", "neck_pose_params", "eye_pose_params", "eyelid_params")}
        fc["shape_params"] = torch.as_tensor(c["flame_shape"]); r["flame_coeffs"] = fc
        return r, img, msk


@torch.no_grad()
def build(src):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    torch.manual_seed(0); vg, ug, _ = inf(src)
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    torch.cuda.synchronize(); return av, (time.perf_counter() - t0) * 1000


_RAW = {}


@torch.no_grad()
def render(av, t):
    torch.manual_seed(0); rr = ren(av(t), t["render_cam_params"], bg=0.0)
    _RAW["last"] = rr["raw_renders"].clamp(0, 1)[0] if a.raw else None
    return rr["renders"].clamp(0, 1)[0]


def u8(x):
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def original_frames(c, frames):
    """{frame: RGB uint8} of the ORIGINAL video on the GT's timeline."""
    import convert_bedlam2 as cb
    if c["dataset"] == "bedlam2": return cb.decode_frames(c["job"], c["seq"], frames)
    if c["dataset"] == "peoplesnapshot":                                                             # PublicBench: undistorted frames saved by the converter
        b = E / "base" / c["dataset"] / c["clip"] / "orig"
        return {f: cv2.cvtColor(cv2.imread(str(b / f"frame_{f:06d}.png")), cv2.COLOR_BGR2RGB) for f in frames}
    if c["dataset"] == "neuman":
        import convert_publicbench as PB; return PB.neuman_images(c["clip"], frames)
    if c["dataset"] == "xhumans":                                                                    # PublicBench-v2: the X-Humans color pngs
        d = np.load(E / "base" / c["dataset"] / c["clip"] / "crop_h.npz"); src = {int(str(k).split("_")[1]): str(p) for k, p in zip(d["keys"], d["orig_src"])}
        return {f: cv2.cvtColor(cv2.imread(src[f]), cv2.COLOR_BGR2RGB) for f in frames}
    import convert_test as CT
    CT.TMP.mkdir(parents=True, exist_ok=True); out = {}
    with tempfile.TemporaryDirectory(dir=CT.TMP) as td:
        subprocess.run([CT.FFMPEG, "-loglevel", "error", "-i", c["video"], "-f", "image2", "-r", "30", f"{td}/%06d.png"], check=True)
        for f in frames: out[f] = cv2.cvtColor(cv2.imread(f"{td}/{f + 1:06d}.png"), cv2.COLOR_BGR2RGB)
    return out


_ANN, _MAPS = {}, {}


def metric_H(c):
    """original px -> 1024 metric crop px (the converter's geometry, recomputed), per GT frame."""
    if c["dataset"] in ("peoplesnapshot", "neuman", "xhumans"):                                      # PublicBench: the converter's stored H
        d = np.load(E / "base" / c["dataset"] / c["clip"] / "crop_h.npz")
        return {int(str(k).split("_")[1]): h for k, h in zip(d["keys"], d["H"])}
    import convert_bedlam2 as cb
    import convert_test as CT
    if c["dataset"] == "bedlam2":
        d = cb.job_npz(c["job"]); bm = np.load(cb.BMAP)
        betas = bm["M"].astype(np.float64) @ d["shape16"][c["rows"][0]].astype(np.float64) + bm["c"].astype(np.float64)
        g = cb.clip_geometry({"rows": c["rows"]}, d, betas)
    else:
        if c["scene"] not in _ANN:
            _ANN.clear(); _ANN[c["scene"]] = json.load(zipfile.ZipFile("/raid/ubx858/datasets/raw/UBody/annotations.zip").open(f"annotations/{c['scene']}/smplx_annotation.json"))
        sx = _ANN[c["scene"]]; _MAPS.setdefault("m", CT.maps()); g, _, _ = CT.ub_geometry(c, [sx[str(i)] for i in c["ann_ids"]], _MAPS["m"])
    return {f: gi["H"] for f, gi in zip(c["frames"], g)}


S512 = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])                                    # pixel-index 1024 -> 512


def _iou(a, b):
    w, h = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def subject_box(fitter, img, ref):
    """H1b: detection with the highest IoU to `ref` (None: the largest); `ref` kept if no detection overlaps it (IoU < 0.1)."""
    B = fitter.det.inference_detector(fitter.s_det, img)
    if len(B) == 0:
        if ref is None: h, w = img.shape[:2]; return np.array([0, 0, w - 1, h - 1], np.float64), 1
        return np.asarray(ref, np.float64), 1
    if ref is None: return B[int(np.argmax((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])))], 0
    v = [_iou(b, ref) for b in B]; j = int(np.argmax(v))
    if v[j] >= 0.1: return B[j], 0
    if not a.track_nearest: return np.asarray(ref, np.float64), 1                                  # H1b (rejected): stale box
    c0 = np.array([(ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2]); sz = max(ref[2] - ref[0], ref[3] - ref[1], 1.0)   # H1c: nearest centre
    d = [np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - c0) / sz for b in B]
    return B[int(np.argmin(d))], 1


def fit_track(fitter, img, prev_box, **kw):
    """SPRINT fit with a tracking fallback: if the detector finds no person, reuse the previous frame's box (else the whole image)."""
    try:
        return fitter.fit(img, **kw), 0
    except RuntimeError as e:
        if "no person" not in str(e): raise
        h, w = img.shape[:2]; box = prev_box if prev_box is not None else [0, 0, w - 1, h - 1]
        return fitter.fit(img, box=box, **kw), 1


def main():
    OUT.mkdir(parents=True, exist_ok=True); warm = False
    for c in clips:
        ds, name = c["dataset"], c["clip"]; od = OUT / ds / name
        if (od / "run.json").exists(): continue
        cdir = DATA / ds / name; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
        if not (cdir / "meta.json").exists(): print("missing data", c["id"], flush=True); continue
        od_final = od; od = LOCAL / ds / name
        if od.exists(): shutil.rmtree(od)
        od.mkdir(parents=True, exist_ok=True); info = {"method": ROWNAME, **CK, "gpu": GPU, "n_test": len(fj["test"]), "drive_iters": a.drive_iters}
        if a.row in ("fb_gt", "guava"):
            cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1)
            if a.row == "guava":
                raw = F.load_pkl(cdir / "optim_tracking_ehm.pkl"); ids = F.load_pkl(cdir / "id_share_params.pkl")
                cl[0]["flame"] = {k: raw[k]["flame_coeffs"] for k in raw}; cl[0]["flame_shape"] = np.asarray(ids["flame_shape"], np.float32).reshape(300)
                fbd = FBDataRel(cfg, cl)
            else:
                fbd = F.FBData(cfg, cl)
            src = to_dev(F.collate1(fbd.source_info(0, fj["source"])), dev)
            if not warm: build(src); warm = True                                                   # CUDA / cuDNN warm-up (not timed)
            av, ms = build(src); info["recon_ms"] = ms; info["recon_input"] = "first frame (GT SMPL-X record)"
            if a.row == "guava":                                                                   # EHM (FLAME head) vs GT SMPL-X 2020 head
                from smplx_lean import SMPLXLean
                lean = SMPLXLean(str(ROOT / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz"), 200, dtype=torch.float32).to(dev)
                fid = torch.as_tensor(np.load(ROOT / "assets/SMPLX/SMPL-X__FLAME_vertex_ids.npy"), device=dev); errs = []
                for k in fj["test"][:: max(1, len(fj["test"]) // 3)][:3]:
                    t = to_dev(F.collate1(fbd.target_info(0, k)), dev); sc = t["smplx_coeffs"]
                    ve = inf.ehm(sc, t["flame_coeffs"])["vertices"][0, :F.NV]
                    pose = torch.cat([sc["global_pose"].reshape(1, 3), sc["body_pose"].reshape(1, 63), sc["jaw_pose"].reshape(1, 3),
                                      sc["eye_pose"].reshape(1, 6), sc["left_hand_pose"].reshape(1, 45), sc["right_hand_pose"].reshape(1, 45)], 1)
                    vl, _ = lean(pose, sc["shape"].reshape(1, 200), sc["exp"].reshape(1, 50))
                    d = (ve - vl[0]).norm(dim=-1) * 1000; errs.append([float(d[fid].mean()), float(d[fid].max()), float(d.mean())])
                info["ehm_vs_smplx2020_mm"] = {"head_mean": float(np.mean([e[0] for e in errs])), "head_max": float(np.max([e[1] for e in errs])),
                                               "all_mean": float(np.mean([e[2] for e in errs]))}
            tr = []
            for k in fj["test"]:
                t = to_dev(F.collate1(fbd.target_info(0, k)), dev)
                torch.cuda.synchronize(); t0 = time.perf_counter(); r = render(av, t); torch.cuda.synchronize(); tr.append(time.perf_counter() - t0)
                torchvision.io.write_png(u8(r), str(od / f"{k}.png"))
                if a.raw:
                    (od / "raw").mkdir(exist_ok=True); torchvision.io.write_png(u8(_RAW["last"]), str(od / "raw" / f"{k}.png"))
        else:                                                                                      # fb_sprint
            info.update(fast_backend=a.fast_backend, dwpose_provider=a.dwpose_provider, source_fit_iters=a.sprint_iters,
                        driving_matte_size=a.driving_matte_size, detector_fp16=a.detector_fp16, sprint_graph=a.sprint_graph)
            if hasattr(fitter, 'set_source_mode'): fitter.set_source_mode()
            sf = c["source_frame"]; tf = [int(k.split("_")[1]) for k in fj["test"]]
            ori = original_frames(c, [sf] + tf); Hm = metric_H(c)
            if not warm: fit_track(fitter, ori[sf], None)                                         # warm-up (not timed)
            if a.track_subject:                                                                  # H1b: identity selection in the source frame
                ref = None
                if c["dataset"] == "ubody": b0 = c["bbox_xywh"][c["frames"].index(sf)]; ref = [b0[0], b0[1], b0[0] + b0[2], b0[1] + b0[3]]
                sbox, src_fb = subject_box(fitter, ori[sf], ref); rs = fitter.fit(ori[sf], box=sbox)
            else:
                rs, src_fb = fit_track(fitter, ori[sf], None)
            shp = rs["shape"]
            with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
                key = fj["source"]; SAF.write_tracked(Path(td), name, key, rs)
                cl = F.load_clips_parallel([(td, None, True)], workers=1); fbd = F.FBData(cfg, cl)
                src = to_dev(F.collate1(fbd.source_info(0, key)), dev)
            if not warm: build(src); warm = True
            av, ms = build(src)
            info["recon_ms"] = ms + rs["times"]["total"] * 1000; info["recon_ms_model"] = ms; info["sprint_fit_ms"] = rs["times"]["total"] * 1000
            info["sprint_fit_times_s"] = rs["times"]; info["recon_input"] = "first frame (SPRINT fit, own box / crop / matte)"
            info['gaussians'] = int(av._smplx_xyz.shape[1] + av._uv_local_xyz.shape[1])
            setup_start = time.perf_counter()
            if hasattr(fitter, 'set_driving_mode'): fitter.set_driving_mode(shp)
            info['driving_setup_ms'] = (time.perf_counter() - setup_start) * 1000
            tr, tfit, nfb, pbox = [], [], 0, rs["box"]
            if a.box_source == "mediapipe": boxer = BS.MediaPipeBox(a.box_margin)
            elif a.box_source == "track": boxer = BS.TrackBox(a.box_margin, fitter)
            pres, perf = rs, {}
            for k, f in zip(fj["test"], tf):
                torch.cuda.synchronize(); te0 = time.perf_counter(); lost = 0
                if a.box_source != "yolox":                                                     # phase 5: box from MediaPipe / tracking
                    bx = boxer(ori[f], pbox) if a.box_source == "mediapipe" else boxer(ori[f], pbox, pres, shp)
                    fb_ = 0
                    if bx is None: lost = 1; bx, fb_ = subject_box(fitter, ori[f], pbox)       # lost -> YOLOX + H1c
                    tdet = time.perf_counter() - te0
                    r_ = fitter.fit(ori[f], box=bx, iters=a.drive_iters, shape_lock=shp); r_["times"]["total"] += tdet
                elif a.track_subject:
                    td0 = time.perf_counter(); bx, fb_ = subject_box(fitter, ori[f], pbox); tdet = time.perf_counter() - td0
                    r_ = fitter.fit(ori[f], box=bx, iters=a.drive_iters, shape_lock=shp); r_["times"]["total"] += tdet
                else:
                    r_, fb_ = fit_track(fitter, ori[f], pbox, iters=a.drive_iters, shape_lock=shp)
                nfb += fb_; pbox = r_["box"]; pres = r_; tfit.append(r_["times"]["total"])
                rr = dict(r_["record"])
                if a.drive_iters == 0: rr["smplx_coeffs"] = r_["raw_smplx"]                       # deployed: raw SPRINT; H1: refined record
                comp = F.compact_record(rr, shp); sm = {kk: torch.as_tensor(v.copy()) for kk, v in comp["smplx"].items()}
                sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
                RT = sm["camera_RT_params"]; M4 = torch.eye(4); M4[:3, :4] = RT; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
                t = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c),
                     "invtanfov": comp["invtanfov"]}
                t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
                t = to_dev(F.collate1(t), dev)
                torch.cuda.synchronize(); t0 = time.perf_counter(); r = render(av, t); torch.cuda.synchronize(); tr.append(time.perf_counter() - t0)
                perf[k] = {"box": [round(float(v), 2) for v in pbox], "lost": lost, "e2e_ms": (time.perf_counter() - te0) * 1000,
                           "render_ms": tr[-1] * 1000}                                          # box -> SPRINT -> render, synchronised
                W = S512 @ Hm[f] @ np.linalg.inv(r_["M"].astype(np.float64)) @ np.linalg.inv(S512)
                img = u8(r).permute(1, 2, 0).numpy()
                wp = cv2.warpPerspective(img, W, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                torchvision.io.write_png(torch.from_numpy(wp).permute(2, 0, 1).contiguous(), str(od / f"{k}.png"))
            info["sprint_drive_s_per_frame"] = float(np.mean(tfit)); info["detector_fallback_frames"] = nfb; info["detector_fallback_source"] = src_fb
            info.update(box_source=a.box_source, box_margin=a.box_margin if a.box_source != "yolox" else None, per_frame=perf,
                        lost_frames=int(sum(v["lost"] for v in perf.values())), load_avg=os.getloadavg())
        info["render_ms_per_frame"] = float(np.mean(tr) * 1000); info["gpu"] = GPU
        od_final.mkdir(parents=True, exist_ok=True)
        for p_ in sorted(od.iterdir()):
            if p_.is_dir():                                                                        # raw renders -> renders/<row>_raw/...
                rd = OUT.parent / f"{ROWNAME}_raw" / ds / name; rd.mkdir(parents=True, exist_ok=True)
                for q_ in sorted(p_.iterdir()): shutil.copyfile(q_, rd / q_.name)
                (rd / "run.json").write_text(json.dumps({"method": f"{ROWNAME}_raw", "note": "first 3 Gaussian feature channels, no refiner"}))
            else: shutil.copyfile(p_, od_final / p_.name)
        (od_final / "run.json").write_text(json.dumps(info, indent=1)); shutil.rmtree(od); print(json.dumps({"id": c["id"], **{k: v for k, v in info.items() if k != "sprint_fit_times_s"}}), flush=True)


def timing():
    ub = [c for c in clips if c["dataset"] == "ubody"]; b2 = [c for c in clips if c["dataset"] == "bedlam2"]
    sel = [x for pair in zip(ub, b2) for x in pair][:a.timing]; per = []
    for c in sel:
        ds, name = c["dataset"], c["clip"]; fj = json.loads((E / "base" / ds / name / "frames.json").read_text()); cdir = DATA / ds / name
        if a.row == "fb_sprint":
            ori = original_frames(c, [c["source_frame"]])[c["source_frame"]]
            for _ in range(3): fitter.fit(ori)
            tf = []
            for _ in range(a.reps):
                torch.cuda.synchronize(); t0 = time.perf_counter(); rs = fitter.fit(ori); torch.cuda.synchronize(); tf.append((time.perf_counter() - t0) * 1000)
            td = tempfile.mkdtemp(dir="/dev/shm"); SAF.write_tracked(Path(td), name, fj["source"], rs)
            cl = F.load_clips_parallel([(td, None, True)], workers=1); fbd = F.FBData(cfg, cl)
        else:
            cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1); tf = [0.0]
            if a.row == "guava":
                raw = F.load_pkl(cdir / "optim_tracking_ehm.pkl"); ids = F.load_pkl(cdir / "id_share_params.pkl")
                cl[0]["flame"] = {k: raw[k]["flame_coeffs"] for k in raw}; cl[0]["flame_shape"] = np.asarray(ids["flame_shape"], np.float32).reshape(300)
                fbd = FBDataRel(cfg, cl)
            else:
                fbd = F.FBData(cfg, cl)
        src = to_dev(F.collate1(fbd.source_info(0, fj["source"])), dev)
        for _ in range(3): build(src)
        tb = [build(src)[1] for _ in range(a.reps)]
        per.append({"id": c["id"], "build_ms": float(np.median(tb)), "fit_ms": float(np.median(tf))})
        print(per[-1], flush=True)
    out = {"row": a.row, "gpu": GPU, "n_clips": len(per), "reps": a.reps, "build_ms_median": float(np.median([p["build_ms"] for p in per])),
           "fit_ms_median": float(np.median([p["fit_ms"] for p in per])), "per_clip": per,
           "note": "build = GUAVA inference from the prepared source (image features -> Gaussians); fit = SPRINT source fit (DWPose CPU + "
                   "StyleMatte + SPRINT + 150 refinement steps) for the SPRINT row; GT rows assume the source record is given"}
    out["recon_ms_median"] = out["build_ms_median"] + out["fit_ms_median"]
    (E / "reports").mkdir(parents=True, exist_ok=True); (E / "reports" / f"recon_timing_{a.row}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_clip"}), flush=True)


timing() if a.timing else main()
