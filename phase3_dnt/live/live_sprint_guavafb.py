"""Live full-body avatar (laptop): webcam / video -> person box (YOLOX-L on the GPU + H1c subject linking) -> SPRINT (k = 0,
shape locked to the session's source fit) -> [+ background-worker pose correction] -> GUAVA-FB (B28k) render -> display.

Session start: a reference frame is captured (webcam: after a countdown, stand 2-3 m from the camera with the whole body in view;
video: --ref-frame), SPRINT fits it (150 refinement steps, ~1 s) and GUAVA-FB builds the avatar from it once.
Per frame (main thread, high-priority CUDA stream, never waits for the worker): detector -> SPRINT -> correction -> render.
--worker: off | pose (PREREG phase 6 step 1, phase3_dnt/worker6/worker_core.py) | uv (live appearance update = phase 6 step 2,
worker6/uv_fusion.py: accepted frames -> image features -> GUAVA-FB's UV map -> fused -> decoded -> the live loop renders the
latest published avatar; EXPERIMENTAL, its validation is pending) | pose+uv | pose+uv+geo (not available). Default off.
--record DIR: saves the session (frames, timestamps, live outputs) for offline replay / scoring (written by a background thread).
--input REC_DIR: replays a --record folder (its reference.png, then frames/ at the recorded capture times, every frame processed).
--score-every N: every N-th processed frame is marked "scored" in the recording and never used by the uv update (ablation scoring).
--timing-json PATH: per-stage and end-to-end ms p50 / p95, FPS, worker update rate.
RAGA speed-ups (laptop, 2026-11-05): --fourcc MJPG (default; webcam 1280x720 at 30 instead of 10 FPS), --crop center (no detector
in the live loop: a fixed full-height centre square), --parallel (default since 2026-11-06; crop + SPRINT of the next frame in a
second thread while the current frame renders; --no-parallel for the serial loop).

YOLOX-L is the EHM-Tracker DWPose detector ONNX (yolox_l.onnx) converted to PyTorch with onnx2torch (max abs diff to
onnxruntime 4e-5 on the same input) and run on the GPU; pre- / post-processing are EHM's own (onnxdet.inference_detector).

Usage:
  python phase3_dnt/live/live_sprint_guavafb.py --input 0 [--calib webcam_intrinsics.json] [--worker off] [--record rec_dir]
  python phase3_dnt/live/live_sprint_guavafb.py --input clip.mp4 --no-display --frames 600 --timing-json timing.json
  bash phase3_dnt/live/run_ablation.sh REC_DIR      # live appearance update off vs on on the same recording (README section 8)"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; P3 = ROOT / "phase3_dnt"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--input", default="0", help="webcam index, video path or a --record folder"); ap.add_argument("--video", default="", help="alias of --input PATH")
ap.add_argument("--calib", default="", help="webcam intrinsics JSON (tools/calibrate_webcam.py); default: focal = image width")
ap.add_argument("--worker", default="off", choices=("off", "pose", "uv", "pose+uv", "pose+uv+geo"))
ap.add_argument("--score-every", type=int, default=3, help="every N-th processed frame is marked scored, never used by the uv update (0 = none)")
ap.add_argument("--uv-prior", type=float, default=2.0, help="uv: source anchor prior weight (registered tuning grid {2, 4})")
ap.add_argument("--uv-thr", type=float, default=0.3, help="uv: texel update threshold (registered tuning grid {0.3, 0.5})")
ap.add_argument("--uv-tmpl-avg", action="store_true", help="uv: template branch with averaged vertex features (tuning option)")
ap.add_argument("--no-pace", action="store_true", help="replay of a --record folder: as fast as possible, not at the capture times")
ap.add_argument("--record", default="", help="directory: save frames + timestamps + live outputs")
ap.add_argument("--frames", type=int, default=0, help="stop after N live frames (0 = until q / end of video)")
ap.add_argument("--no-display", action="store_true"); ap.add_argument("--view", default="side", choices=("side", "overlay", "avatar"))
ap.add_argument("--timing-json", default=""); ap.add_argument("--warmup", type=int, default=20, help="frames excluded from timing")
ap.add_argument("--ref-delay", type=float, default=5.0, help="webcam: countdown (s) before the reference frame is captured")
ap.add_argument("--ref-frame", type=int, default=0, help="video: index of the reference frame")
ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
ap.add_argument("--fourcc", default="MJPG", help="webcam pixel format: MJPG (default; this laptop's webcam gives 1280x720 at 30 FPS "
                "only in MJPG, 10 FPS in uncompressed YUYV) | YUYV | none (the camera's default = the behaviour before this option)")
ap.add_argument("--crop", default="yolox", choices=("yolox", "center"),
                help="live person crop: yolox (YOLOX-L + H1c every frame) | center (no detector: a fixed square of the full frame height "
                     "at the frame centre; stay inside it; untested with --worker pose / uv). The reference frame always uses YOLOX")
ap.add_argument("--print-fps", type=int, default=30, metavar="N",
                help="print the frame rate in the terminal every N frames (0 = never): smoothed FPS, mean FPS since warm-up, stage ms")
ap.add_argument("--parallel", dest="parallel", action="store_true", default=True,
                help="(default) person crop + SPRINT for the next frame run in a second thread (LIVE stream) while the main thread "
                     "renders the current one; every frame read is rendered once, recorded under its own index; about one frame of pose lag")
ap.add_argument("--no-parallel", "--serial", dest="parallel", action="store_false", help="serial loop: crop, SPRINT, render in one thread")
ap.add_argument("--assets", default=str(ROOT / "live_assets"), help="package asset root (checkpoints/)")
ap.add_argument("--sprint-ckpt", default=""); ap.add_argument("--fb-ckpt", default="")
ap.add_argument("--dwpose-threads", type=int, default=4, help="CPU threads of DWPose (reference fit and worker only)")
a = ap.parse_args()
SRC = a.video or a.input
ASSETS = Path(a.assets)
SPRINT_CKPT = a.sprint_ckpt or str(ASSETS / "checkpoints" / "sprint_scratch_ema128k.pt")
FB_CKPT = a.fb_ckpt or str(ASSETS / "checkpoints" / "guava_fb_B28k.pt")
if a.worker in ("pose+uv", "pose+uv+geo"):
    ap.error(f"--worker {a.worker}: not available (PREREG phase 6 step 2 / 3 have not passed their validation rule)")
ARC = Path(os.environ.get("ARCFACE_ONNX", ASSETS / "arcface" / "glintr100.onnx"))
if a.worker == "uv":
    if not ARC.exists(): ap.error(f"--worker uv needs the ArcFace model {ARC} (README_LIVE.md section 8: download glintr100.onnx)")
    os.environ["ARCFACE_ONNX"] = str(ARC)                                                          # read by onnx_arcface_torch
    print("--worker uv: live appearance update (PREREG phase 6 step 2) is EXPERIMENTAL; its validation is pending", flush=True)

# eval assets (SPRINT's evaluator constants point at the server); the package ships them under ASSETS/eval_assets
if (ASSETS / "eval_assets").exists(): os.environ.setdefault("SPRINT_EVAL_ASSETS", str(ASSETS / "eval_assets"))

spec = importlib.util.spec_from_file_location("sprint_avatar_fit", P3 / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = SAF.SprintFitter(SPRINT_CKPT, "cuda:0", 150, dwpose_threads=a.dwpose_threads); fitter.empty_matte_fix = True

import cv2, onnx, onnx2torch, roma, torch                                                            # noqa: E402
sys.path.insert(0, str(P3 / "guava_fb")); sys.path.insert(0, str(P3 / "worker6"))
import fb_common as F                                                                                 # noqa: E402
import worker_core as WC                                                                              # noqa: E402
if a.worker == "uv": import uv_fusion as UF                                                           # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                          # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                         # noqa: E402
dev = torch.device("cuda:0"); torch.backends.cudnn.benchmark = True
lo_pri, hi_pri = torch.cuda.Stream.priority_range()
LIVE = torch.cuda.Stream(priority=hi_pri); WORK = torch.cuda.Stream(priority=lo_pri)
cfg, inf, ren, fb_meta = F.setup_a20(dev, ckpt=FB_CKPT); inf.eval(); ren.eval()


# ------------------------------------------------------------------------------------------------ detector (GPU)
class TorchYOLOX:
    """onnxruntime-session-like wrapper (get_inputs / run) around the onnx2torch module, so EHM's inference_detector runs as is."""
    class _In:
        name = "images"

    def __init__(self, path):
        self.m = onnx2torch.convert(onnx.load(str(path))).to(dev).eval()

    def get_inputs(self): return [self._In()]

    @torch.no_grad()
    def run(self, _, feed):
        x = torch.from_numpy(np.ascontiguousarray(next(iter(feed.values())))).to(dev, non_blocking=True)
        return [self.m(x).float().cpu().numpy()]


YOLO = TorchYOLOX(SAF.EHMT / "pretrained/dwpose/yolox_l.onnx")


def iou(p, q):
    w, h = max(0.0, min(p[2], q[2]) - max(p[0], q[0])), max(0.0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    return i / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i, 1e-9)


def detect(img, ref):
    """YOLOX + H1c: highest IoU with the previous box (>= 0.1), else the nearest centre; no detection -> previous box."""
    D = fitter.det.inference_detector(YOLO, img)
    if len(D) == 0: return (np.asarray(ref, np.float64) if ref is not None else None), True
    if ref is None: return np.asarray(D[int(np.argmax((D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1])))], np.float64), False
    v = [iou(d, ref) for d in D]; j = int(np.argmax(v))
    if v[j] >= 0.1: return np.asarray(D[j], np.float64), False
    c0 = np.array([(ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2]); sz = max(ref[2] - ref[0], ref[3] - ref[1], 1.0)
    return np.asarray(D[int(np.argmin([np.linalg.norm(np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2]) - c0) / sz for d in D]))], np.float64), False


# ------------------------------------------------------------------------------------------------ SPRINT + render
CALIB = json.loads(Path(a.calib).read_text()) if a.calib else None
if CALIB is not None:                                                                                # every SPRINT call (reference fit, worker, live)
    _cam_feat = fitter.ev.cam_feat                                                                   # sees the calibrated focal, rescaled to the frame
    fitter.ev.cam_feat = lambda A, W, H, f, real: _cam_feat(A, W, H, CALIB["fx"] * W / CALIB["image_size_wh"][0], 1.0)


def cam_feat(M, W, H):
    return fitter.ev.cam_feat((np.diag([256 / SAF.CROP, 256 / SAF.CROP, 1.0]) @ M)[:2], W, H, W, 0.0)


def live_raw(img, box, shp):
    """SPRINT raw on the detector crop (sprint_avatar_fit.fit's raw conversion); syncs only the current stream."""
    H, W = img.shape[:2]; M = SAF.box_crop_matrix(box, SAF.CROP)
    crop = cv2.warpAffine(img, M[:2], (SAF.CROP, SAF.CROP), flags=cv2.INTER_LINEAR)
    crop_t = torch.from_numpy(crop).pin_memory().to(dev, non_blocking=True).permute(2, 0, 1)[None].float() / 255.0
    with torch.no_grad(): out = fitter.sprint(crop_t, cam_feat(M, W, H), shp)
    bp0 = out["body_param"]; aa = lambda Rm: roma.rotmat_to_rotvec(Rm.reshape(-1, 3, 3))                # noqa: E731
    flat = torch.cat([aa(bp0[k]).reshape(-1) for k in ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose")]
                     + [bp0["exp"][0, :50].reshape(-1), out["pd_cam"][0, :3, :4].reshape(-1)]).cpu().numpy().astype(np.float32)
    s = np.cumsum([0, 3, 63, 45, 45, 3, 50, 12])
    raw = {"global_pose": flat[s[0]:s[1]], "body_pose": flat[s[1]:s[2]].reshape(21, 3), "left_hand_pose": flat[s[2]:s[3]].reshape(15, 3),
           "right_hand_pose": flat[s[3]:s[4]].reshape(15, 3), "jaw_pose": flat[s[4]:s[5]].reshape(1, 3), "eye_pose": np.zeros((2, 3), np.float32),
           "exp": flat[s[5]:s[6]], "camera_RT_params": flat[s[6]:s[7]].reshape(3, 4), "body_cam": np.zeros(3, np.float32)}
    return raw, M


def record_of(smplx, box, M):
    Mlr = SAF.box_crop_matrix(box, SAF.BODY_LR); Mp = SAF.box_crop_matrix(box, SAF.PART)
    return {"smplx_coeffs": smplx, "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": M, "M_c2o-hd": np.linalg.inv(M).astype(np.float32)},
            **{p: {"M_o2c": Mp, "M_c2o": np.linalg.inv(Mp).astype(np.float32)} for p in ("head_crop", "left_hand_crop", "right_hand_crop")}}


@torch.no_grad()
def render(av, smplx, box, M, shp):
    comp = F.compact_record(record_of(smplx, box, M), shp); sm = {k: torch.as_tensor(v.copy()) for k, v in comp["smplx"].items()}
    sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
    M4 = torch.eye(4); M4[:3, :4] = sm["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
    t = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c), "invtanfov": comp["invtanfov"]}
    t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
    t = to_dev(F.collate1(t), dev); torch.manual_seed(0); rr = ren(av(t), t["render_cam_params"], bg=0.0)
    return rr["renders"][0], rr["depths"][0, 0]


def center_box(img):
    """--crop center: the box whose SPRINT crop (box_crop_matrix: square, 1.1 x the long side) is the full frame height, centred."""
    H, W = img.shape[:2]; h = H / 1.1 + 1e-6
    return np.array([W / 2 - h / 2, H / 2 - h / 2, W / 2 + h / 2, H / 2 + h / 2], np.float64)


def live_box(img, ref):
    if a.crop == "center": return center_box(img)
    return detect(img, ref)[0]


def pose_step(img, pbox, shp):
    """Person crop (detector or the fixed one) + SPRINT -> (box, raw, M, det ms, SPRINT ms, t0); None until a person is found.
    Serial: the detector on the default stream (as before); --parallel: also on the LIVE stream, off the render's stream."""
    t0 = time.perf_counter()
    if a.parallel:
        with torch.cuda.stream(LIVE): box = live_box(img, pbox)
    else:
        box = live_box(img, pbox)
    t1 = time.perf_counter()
    if box is None: return None
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(LIVE):
        e0.record(LIVE); raw, M = live_raw(img, box, shp); e1.record(LIVE)
    e1.synchronize()
    return box, raw, M, (t1 - t0) * 1000, e0.elapsed_time(e1), t0


def posed_serial(cap, pbox, shp):
    """Frames with their pose, one after the other (the behaviour before --parallel)."""
    last = -1
    while True:
        fr = cap.read(last)
        if fr is None: return
        last, t_cap, img = fr
        r = pose_step(img, pbox, shp)
        if r is None: continue
        pbox = r[0]
        yield last, t_cap, img, r


class PoseThread(threading.Thread):
    """--parallel: reads a frame and poses it while the main thread renders the previous one. It hands each result over and
    waits until the main thread has taken it before reading the next frame: one frame ahead at most, every frame rendered once."""
    def __init__(self, cap, pbox, shp):
        super().__init__(daemon=True); self.cap, self.pbox, self.shp = cap, pbox, shp
        self.q = queue.Queue(maxsize=1); self.taken = threading.Event(); self.stop = threading.Event(); self.start()

    def run(self):
        last = -1
        try:
            while not self.stop.is_set():
                fr = self.cap.read(last)
                if fr is None: break
                last, t_cap, img = fr
                r = pose_step(img, self.pbox, self.shp)
                if r is None: continue
                self.pbox = r[0]; self.taken.clear(); self.q.put((last, t_cap, img, r))
                while not self.taken.wait(0.1):
                    if self.stop.is_set(): return
        except BaseException as e:                                                                     # re-raised in the main thread
            self.q.put(e); return
        self.q.put(None)

    def get(self):
        it = self.q.get(); self.taken.set()
        if isinstance(it, BaseException): raise it
        return it


def posed_parallel(cap, pbox, shp):
    th = PoseThread(cap, pbox, shp)
    try:
        while True:
            it = th.get()
            if it is None: return
            yield it
    finally:                                       # join: a thread still in a CUDA call at interpreter exit aborts the process
        th.stop.set(); th.join(timeout=30)


def build_avatar(img):
    """Source: SPRINT fit (150 steps) of the reference frame + GUAVA-FB avatar build. -> avatar, shape, box, fit, ms, source batch."""
    torch.cuda.synchronize(); t0 = time.perf_counter()
    box, _ = detect(img, None)
    if box is None: raise RuntimeError("no person in the reference frame")
    rs = fitter.fit(img, box=box); shp = rs["shape"]
    with tempfile.TemporaryDirectory() as td:
        SAF.write_tracked(Path(td), "live", "frame_000000", rs)
        cl = F.load_clips_parallel([(td, None, True)], workers=1); fbd = F.FBData(cfg, cl); src = to_dev(F.collate1(fbd.source_info(0, "frame_000000")), dev)
    with torch.no_grad():
        torch.manual_seed(0); vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    torch.cuda.synchronize(); return av, shp, box, rs, (time.perf_counter() - t0) * 1000, src


# ------------------------------------------------------------------------------------------------ capture / recording
class Capture:
    """Webcam: a thread keeps only the latest frame (live semantics). Video: sequential frames (every frame is processed).
    --record folder: its frames/ in order, each released at its recorded capture time (frames.jsonl t_capture) unless --no-pace."""
    def __init__(self, src):
        self.dir = Path(src) if Path(src).is_dir() else None; self.cam = False; self.desc = None            # desc: webcam mode
        if self.dir is not None:
            tc = {}
            if (self.dir / "frames.jsonl").exists():
                for line in open(self.dir / "frames.jsonl"): m = json.loads(line); tc[int(m["i"])] = float(m["t_capture"])
            self.items = [(int(f.stem), f, tc.get(int(f.stem))) for f in sorted((self.dir / "frames").glob("*.jpg"))]
            if not self.items: raise SystemExit(f"no frames/*.jpg in {src}")
            self.k = 0; self.t0 = None; self.fps = 30.0; self.ok = True; return
        self.cam = src.isdigit(); self.cap = cv2.VideoCapture(int(src) if self.cam else src)
        if self.cam:
            if a.fourcc.lower() != "none": self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*a.fourcc))   # before the size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
            self.desc = (f"{int(self.cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')} "
                         f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                         f"@ {self.cap.get(cv2.CAP_PROP_FPS):.0f} fps")
            print(f"webcam: {self.desc}", flush=True)
            self.lock = threading.Lock(); self.latest = None; self.n = 0; self.ok = True
            threading.Thread(target=self._loop, daemon=True).start()
        if not self.cap.isOpened(): raise SystemExit(f"cannot open {src}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _loop(self):
        while self.ok:
            ok, fr = self.cap.read()
            if not ok: self.ok = False; break
            with self.lock: self.latest = (self.n, time.perf_counter(), cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); self.n += 1

    def read(self, last_idx=-1):
        if self.dir is not None:
            if self.k >= len(self.items): return None
            i, f, tc = self.items[self.k]; self.k += 1; img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            if tc is not None and not a.no_pace:                                                       # the recorded camera clock
                if self.t0 is None: self.t0 = (time.perf_counter(), tc)
                wait = self.t0[0] + (tc - self.t0[1]) - time.perf_counter()
                if wait > 0: time.sleep(wait)
            return i, time.perf_counter(), img
        if not self.cam:
            ok, fr = self.cap.read(); return (None if not ok else (int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, time.perf_counter(), cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        while self.ok:
            with self.lock: cur = self.latest
            if cur is not None and cur[0] != last_idx: return cur
            time.sleep(0.001)
        return None


class Recorder(threading.Thread):
    """Background writer: frames/<i>.jpg, avatar/<i>.jpg, frames.jsonl (timestamps, box, raw / used SMPL-X, stage ms)."""
    def __init__(self, out):
        super().__init__(daemon=True); self.out = Path(out); self.q = queue.Queue(maxsize=256); self.dropped = 0
        for d in ("frames", "avatar"): (self.out / d).mkdir(parents=True, exist_ok=True)
        self.f = open(self.out / "frames.jsonl", "w"); self.start()

    def put(self, item):
        try: self.q.put_nowait(item)
        except queue.Full: self.dropped += 1

    def run(self):
        while True:
            it = self.q.get()
            if it is None: break
            i, img, ava, meta = it
            cv2.imwrite(str(self.out / "frames" / f"{i:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(self.out / "avatar" / f"{i:06d}.jpg"), cv2.cvtColor(ava, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.f.write(json.dumps(meta) + "\n")
        self.f.close()

    def close(self): self.q.put(None); self.join()


def jsonable(d): return {k: (np.asarray(v).round(6).tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


class Scored:
    """Processed frame n is scored (never fused by the uv update) iff n % every == 0; the same frames in every --worker run."""
    def __init__(self, every): self.every = every
    def __contains__(self, n): return self.every > 0 and n % self.every == 0


class HeldOut:
    """Stop-rule held-out frames of the uv update (refined, never fused): every 6th non-scored frame (replay.py --heldout-every 6)."""
    def __init__(self, every): self.every = every
    def __contains__(self, n):
        if n < 1 or (self.every > 0 and n % self.every == 0): return False
        return (n - (n // self.every if self.every > 0 else 0)) % 6 == 0


# ------------------------------------------------------------------------------------------------ main
def compose(img, ava, dep, M, fps, ms, wv):
    if a.view == "avatar": out = ava
    elif a.view == "overlay":
        Hs, Ws = img.shape[:2]; A = np.linalg.inv(np.diag([0.5, 0.5, 1.0]) @ M)[:2]                    # 512 render px -> frame px
        wa = cv2.warpAffine(ava, A, (Ws, Hs)); wm = cv2.warpAffine((dep > 1e-4).astype(np.uint8), A, (Ws, Hs)) > 0
        out = img.copy(); out[wm] = wa[wm]
    else:
        h = 512; l = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        if a.crop == "center":                                                                        # the fixed crop: stay inside it
            Mi = np.linalg.inv(M); s = h / img.shape[0]
            (x0, y0), (x1, y1) = (Mi @ [0, 0, 1])[:2] * s, (Mi @ [SAF.CROP, SAF.CROP, 1])[:2] * s
            cv2.rectangle(l, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (0, 255, 0), 2)
        out = np.concatenate([l, ava], 1)
    out = np.ascontiguousarray(out[..., ::-1])
    txt = f"{fps:5.1f} FPS | det {ms['det']:.0f} sprint {ms['sprint']:.0f} render {ms['render']:.0f} total {ms['total']:.0f} ms | worker {a.worker} v{wv}"
    cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    cap = Capture(SRC); rec = Recorder(a.record) if a.record else None
    # warm-up of every network on a dummy frame (not timed)
    dummy = np.zeros((a.height, a.width, 3), np.uint8); fitter.det.inference_detector(YOLO, dummy)
    # reference frame
    if cap.cam:
        t_end = time.perf_counter() + a.ref_delay; last = -1
        while True:
            fr = cap.read(last); last = fr[0]; left = t_end - time.perf_counter()
            if left <= 0: ref = fr[2]; break
            if not a.no_display:
                show = np.ascontiguousarray(fr[2][..., ::-1]).copy()
                cv2.putText(show, f"Stand 2-3 m away, whole body in view: {left:.1f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                if a.crop == "center":                                                                # where to stand: the fixed crop
                    Hs, Ws = show.shape[:2]; x0 = (Ws - Hs) // 2
                    cv2.rectangle(show, (x0, 0), (x0 + Hs - 1, Hs - 1), (0, 255, 0), 2)
                    cv2.putText(show, "--crop center: stay inside the green square", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("live_sprint_guavafb", show); cv2.waitKey(1)
    elif cap.dir is not None and (cap.dir / "reference.png").exists():                                  # replay: the session's reference
        ref = cv2.cvtColor(cv2.imread(str(cap.dir / "reference.png")), cv2.COLOR_BGR2RGB)
    else:
        for _ in range(a.ref_frame + 1): fr = cap.read()
        ref = fr[2]
    av, shp, pbox, rs, ref_ms, src = build_avatar(ref)
    print(f"reference: SPRINT fit + avatar build {ref_ms:.0f} ms (kp err {rs['diag']['kp_err_px1024_after']:.1f} px)", flush=True)
    if rec:
        cv2.imwrite(str(Path(a.record) / "reference.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
        (Path(a.record) / "session.json").write_text(json.dumps({"input": SRC, "calib": CALIB, "worker": a.worker, "ref_ms": ref_ms, "source_box": pbox.tolist(),
                                                                 "source_smplx": jsonable(rs["record"]["smplx_coeffs"]), "shape": rs["shape"].tolist(),
                                                                 "source_raw": jsonable(rs["raw_smplx"]), "score_every": a.score_every,
                                                                 "uv": {"prior": a.uv_prior, "upd_thr": a.uv_thr, "tmpl_avg": a.uv_tmpl_avg},
                                                                 "fb_ckpt": FB_CKPT, "sprint_ckpt": SPRINT_CKPT, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                 "camera": cap.desc, "crop": a.crop, "parallel": a.parallel}))
    worker = None; fu = None; SC = Scored(a.score_every)
    if a.worker == "pose":
        worker = WC.Worker(fitter, shp, WORK, "pose"); worker.start()
    elif a.worker == "uv":
        t_uv = time.perf_counter()
        fu = UF.UVFusion(cfg, inf, ren, src, rs, ref, shp, fitter, SAF, prior=a.uv_prior, upd_thr=a.uv_thr, tmpl_avg=a.uv_tmpl_avg, dev="cuda:0")
        worker = WC.Worker(fitter, shp, WORK, "uv", fusion=fu, static_av=av, src_raw=rs["raw_smplx"])
        worker.scored, worker.heldout = SC, HeldOut(a.score_every); worker.start()                  # scored frames are never fused
        print(f"live appearance update ready ({(time.perf_counter() - t_uv) * 1000:.0f} ms)", flush=True)
    # session start (reference fit, avatar build): cuDNN autotuning may briefly borrow most of the free GPU memory here
    build_peak = torch.cuda.max_memory_allocated() / 1e9; torch.cuda.reset_peak_memory_stats()
    T = {k: [] for k in ("det", "sprint", "render", "total", "frame_interval", "capture_to_display")}
    n, t_prev, fps_s, t_first, t_last = 0, None, 0.0, None, None
    rv = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    src_it = (posed_parallel if a.parallel else posed_serial)(cap, pbox, shp)                         # frames with their pose
    for last, t_cap, img, (box, raw, M, ms_det, ms_sprint, t0) in src_it:
        with torch.cuda.stream(LIVE):
            corr = worker.published if a.worker == "pose" else None; used = corr.apply(raw) if corr is not None else raw
            if worker: worker.offer(n, img, box, raw)
        av_cur, wv = worker.published_av if a.worker == "uv" else (av, worker.version if worker else 0)       # one read per frame
        # render on the default stream: the rasterizer's kernels ignore the current stream (legacy default stream)
        rv[0].record(torch.cuda.default_stream()); rgb, dep = render(av_cur, used, box, M, shp); rv[1].record(torch.cuda.default_stream())
        ava = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy(); depn = dep.cpu().numpy()
        t2 = time.perf_counter()
        # --parallel: total = latency from this frame's crop start to its render end (the frame waits while the previous renders)
        ms = {"det": ms_det, "sprint": ms_sprint, "render": rv[0].elapsed_time(rv[1]), "total": (t2 - t0) * 1000}
        if t_prev: fps_s = 1.0 / (t2 - t_prev) if fps_s == 0.0 else 0.9 * fps_s + 0.1 / (t2 - t_prev)
        if not a.no_display:
            cv2.imshow("live_sprint_guavafb", compose(img, ava, depn, M, fps_s, ms, wv))
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        t3 = time.perf_counter()
        if n >= a.warmup:
            for k in ("det", "sprint", "render", "total"): T[k].append(ms[k])
            if t_prev: T["frame_interval"].append((t3 - t_prev) * 1000)
            T["capture_to_display"].append((t3 - t_cap) * 1000)
            if t_first is None: t_first = t3
            t_last = t3
        if a.print_fps and n % a.print_fps == 0 and n:
            mean_fps = (len(T["total"]) - 1) / (t_last - t_first) if t_first and t_last > t_first and len(T["total"]) > 1 else float("nan")
            print(f"[fps] frame {n}: {fps_s:5.1f} now | {mean_fps:5.1f} mean since warm-up | det {ms['det']:.1f} sprint {ms['sprint']:.1f} "
                  f"render {ms['render']:.1f} total {ms['total']:.1f} ms | capture->display {(t3 - t_cap) * 1000:.0f} ms", flush=True)
        if rec: rec.put((last, img, ava, {"i": last, "n": n, "scored": n in SC, "t_capture": t_cap, "t_display": t3, "box": box.tolist(),
                                         "M": np.asarray(M, np.float64).tolist(), "raw": jsonable(raw),
                                         "corr_version": worker.version if worker else 0, "av_version": wv, "ms": ms}))
        t_prev = t3; n += 1
        if a.frames and n >= a.frames: break
    src_it.close()                                                                                     # stops the pose thread
    if worker: worker.wait_idle(); worker.stop()
    if rec: rec.close()
    cap.ok = False; cv2.destroyAllWindows()
    pct = lambda v, q: float(np.percentile(v, q)) if v else None                                           # noqa: E731
    res = {"input": SRC, "worker": a.worker, "frames": n, "timed_frames": len(T["total"]), "gpu": torch.cuda.get_device_name(0),
           "cpu": platform.processor() or platform.machine(), "torch": torch.__version__, "ref_build_ms": ref_ms,
           "stages_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95), "mean": float(np.mean(v)) if v else None} for k, v in T.items()},
           "camera": cap.desc, "crop": a.crop, "parallel": a.parallel,
           # first to last displayed frame (before, the end was taken after the recorder flushed its queue, which grows with speed)
           "fps_observed": (len(T["total"]) - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first else None,
           "worker_updates": (worker.av_version if a.worker == "uv" else worker.version) if worker else 0,
           "worker_update_ms": [round((e["t_end"] - e["t_start"]) * 1000, 1) for e in worker.log if e.get("gate") == "accepted"] if worker else [],
           "worker_offers": len(worker.log) if worker else 0, "recorder_dropped": rec.dropped if rec else 0,
           "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),                         # live loop (+ worker)
           "gpu_mem_session_start_peak_gb": round(build_peak, 2)}
    if res["worker_updates"] and t_first: res["worker_updates_per_s"] = res["worker_updates"] / max(1e-6, time.perf_counter() - t_first)
    if fu is not None:
        from collections import Counter
        res["uv"] = {"settings": {"prior": a.uv_prior, "upd_thr": a.uv_thr, "tmpl_avg": a.uv_tmpl_avg, "score_every": a.score_every},
                     "gates": dict(Counter(str(e.get("gate")).split(":")[0] for e in worker.log)), "rollbacks": fu.n_rollback,
                     "final_coverage": dict(zip(UF.REGIONS, (float(c) for c in fu.coverage()))),
                     "stopped": [UF.REGIONS[k] for k in range(4) if fu.stop[k]],
                     "errors": [e["gate"] for e in worker.log if str(e.get("gate")).startswith("error")][:5]}
    print(json.dumps({k: v for k, v in res.items() if k != "worker_update_ms"}, indent=1), flush=True)
    if a.timing_json: Path(a.timing_json).parent.mkdir(parents=True, exist_ok=True); Path(a.timing_json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
