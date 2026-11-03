"""Live full-body avatar (laptop): webcam / video -> person box (YOLOX-L on the GPU + H1c subject linking) -> SPRINT (k = 0,
shape locked to the session's source fit) -> [+ background-worker pose correction] -> GUAVA-FB (B28k) render -> display.

Session start: a reference frame is captured (webcam: after a countdown, stand 2-3 m from the camera with the whole body in view;
video: --ref-frame), SPRINT fits it (150 refinement steps, ~1 s) and GUAVA-FB builds the avatar from it once.
Per frame (main thread, high-priority CUDA stream, never waits for the worker): detector -> SPRINT -> correction -> render.
--worker: off | pose (PREREG phase 6 step 1, phase3_dnt/worker6/worker_core.py) | pose+uv | pose+uv+geo (not available until
those steps pass their registered validation; the flag exists so the interface is final). Default off.
--record DIR: saves the session (frames, timestamps, live outputs) for offline replay / scoring (written by a background thread).
--timing-json PATH: per-stage and end-to-end ms p50 / p95, FPS, worker update rate.

YOLOX-L is the EHM-Tracker DWPose detector ONNX (yolox_l.onnx) converted to PyTorch with onnx2torch (max abs diff to
onnxruntime 4e-5 on the same input) and run on the GPU; pre- / post-processing are EHM's own (onnxdet.inference_detector).

Usage:
  python phase3_dnt/live/live_sprint_guavafb.py --input 0 [--calib webcam_intrinsics.json] [--worker off] [--record rec_dir]
  python phase3_dnt/live/live_sprint_guavafb.py --input clip.mp4 --no-display --frames 600 --timing-json timing.json"""
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
ap.add_argument("--input", default="0", help="webcam index or video path"); ap.add_argument("--video", default="", help="alias of --input PATH")
ap.add_argument("--calib", default="", help="webcam intrinsics JSON (tools/calibrate_webcam.py); default: focal = image width")
ap.add_argument("--worker", default="off", choices=("off", "pose", "pose+uv", "pose+uv+geo"))
ap.add_argument("--record", default="", help="directory: save frames + timestamps + live outputs")
ap.add_argument("--frames", type=int, default=0, help="stop after N live frames (0 = until q / end of video)")
ap.add_argument("--no-display", action="store_true"); ap.add_argument("--view", default="side", choices=("side", "overlay", "avatar"))
ap.add_argument("--timing-json", default=""); ap.add_argument("--warmup", type=int, default=20, help="frames excluded from timing")
ap.add_argument("--ref-delay", type=float, default=5.0, help="webcam: countdown (s) before the reference frame is captured")
ap.add_argument("--ref-frame", type=int, default=0, help="video: index of the reference frame")
ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720)
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

# eval assets (SPRINT's evaluator constants point at the server); the package ships them under ASSETS/eval_assets
if (ASSETS / "eval_assets").exists(): os.environ.setdefault("SPRINT_EVAL_ASSETS", str(ASSETS / "eval_assets"))

spec = importlib.util.spec_from_file_location("sprint_avatar_fit", P3 / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = SAF.SprintFitter(SPRINT_CKPT, "cuda:0", 150, dwpose_threads=a.dwpose_threads); fitter.empty_matte_fix = True

import cv2, onnx, onnx2torch, roma, torch                                                            # noqa: E402
sys.path.insert(0, str(P3 / "guava_fb")); sys.path.insert(0, str(P3 / "worker6"))
import fb_common as F                                                                                 # noqa: E402
import worker_core as WC                                                                              # noqa: E402
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


def build_avatar(img):
    """Source: SPRINT fit (150 steps) of the reference frame + GUAVA-FB avatar build. -> avatar, shape, box, ms."""
    torch.cuda.synchronize(); t0 = time.perf_counter()
    box, _ = detect(img, None)
    if box is None: raise RuntimeError("no person in the reference frame")
    rs = fitter.fit(img, box=box); shp = rs["shape"]
    with tempfile.TemporaryDirectory() as td:
        SAF.write_tracked(Path(td), "live", "frame_000000", rs)
        cl = F.load_clips_parallel([(td, None, True)], workers=1); fbd = F.FBData(cfg, cl); src = to_dev(F.collate1(fbd.source_info(0, "frame_000000")), dev)
    with torch.no_grad():
        torch.manual_seed(0); vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    torch.cuda.synchronize(); return av, shp, box, rs, (time.perf_counter() - t0) * 1000


# ------------------------------------------------------------------------------------------------ capture / recording
class Capture:
    """Webcam: a thread keeps only the latest frame (live semantics). Video: sequential frames (every frame is processed)."""
    def __init__(self, src):
        self.cam = src.isdigit(); self.cap = cv2.VideoCapture(int(src) if self.cam else src)
        if self.cam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
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


# ------------------------------------------------------------------------------------------------ main
def compose(img, ava, dep, M, fps, ms, wv):
    if a.view == "avatar": out = ava
    elif a.view == "overlay":
        Hs, Ws = img.shape[:2]; A = np.linalg.inv(np.diag([0.5, 0.5, 1.0]) @ M)[:2]                    # 512 render px -> frame px
        wa = cv2.warpAffine(ava, A, (Ws, Hs)); wm = cv2.warpAffine((dep > 1e-4).astype(np.uint8), A, (Ws, Hs)) > 0
        out = img.copy(); out[wm] = wa[wm]
    else:
        h = 512; l = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)); out = np.concatenate([l, ava], 1)
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
                cv2.imshow("live_sprint_guavafb", show); cv2.waitKey(1)
    else:
        for _ in range(a.ref_frame + 1): fr = cap.read()
        ref = fr[2]
    av, shp, pbox, rs, ref_ms = build_avatar(ref)
    print(f"reference: SPRINT fit + avatar build {ref_ms:.0f} ms (kp err {rs['diag']['kp_err_px1024_after']:.1f} px)", flush=True)
    if rec:
        cv2.imwrite(str(Path(a.record) / "reference.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
        (Path(a.record) / "session.json").write_text(json.dumps({"input": SRC, "calib": CALIB, "worker": a.worker, "ref_ms": ref_ms, "source_box": pbox.tolist(),
                                                                 "source_smplx": jsonable(rs["record"]["smplx_coeffs"]), "shape": rs["shape"].tolist(),
                                                                 "fb_ckpt": FB_CKPT, "sprint_ckpt": SPRINT_CKPT, "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
    worker = None
    if a.worker != "off":
        worker = WC.Worker(fitter, shp, WORK, "pose"); worker.start()
    T = {k: [] for k in ("det", "sprint", "render", "total", "frame_interval", "capture_to_display")}
    last, n, t_prev, fps_s, t_first = -1, 0, None, 0.0, None
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    while True:
        fr = cap.read(last)
        if fr is None: break
        last, t_cap, img = fr; t0 = time.perf_counter()
        box, _ = detect(img, pbox); t1 = time.perf_counter()
        if box is None: continue
        with torch.cuda.stream(LIVE):
            ev[0].record(LIVE); raw, M = live_raw(img, box, shp); ev[1].record(LIVE)
            corr = worker.published if worker else None; used = corr.apply(raw) if corr is not None else raw
            if worker: worker.offer(n, img, box, raw)
        # render on the default stream: the rasterizer's kernels ignore the current stream (legacy default stream)
        rgb, dep = render(av, used, box, M, shp); ev[2].record(torch.cuda.default_stream())
        ava = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy(); depn = dep.cpu().numpy()
        t2 = time.perf_counter(); pbox = box
        ms = {"det": (t1 - t0) * 1000, "sprint": ev[0].elapsed_time(ev[1]), "render": ev[1].elapsed_time(ev[2]), "total": (t2 - t0) * 1000}
        if t_prev: fps_s = 1.0 / (t2 - t_prev) if fps_s == 0.0 else 0.9 * fps_s + 0.1 / (t2 - t_prev)
        if not a.no_display:
            cv2.imshow("live_sprint_guavafb", compose(img, ava, depn, M, fps_s, ms, worker.version if worker else 0))
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        t3 = time.perf_counter()
        if n >= a.warmup:
            for k in ("det", "sprint", "render", "total"): T[k].append(ms[k])
            if t_prev: T["frame_interval"].append((t3 - t_prev) * 1000)
            T["capture_to_display"].append((t3 - t_cap) * 1000)
            if t_first is None: t_first = t3
        if rec: rec.put((last, img, ava, {"i": last, "t_capture": t_cap, "t_display": t3, "box": box.tolist(), "raw": jsonable(raw),
                                         "corr_version": worker.version if worker else 0, "ms": ms}))
        t_prev = t3; n += 1
        if a.frames and n >= a.frames: break
    if worker: worker.wait_idle(); worker.stop()
    if rec: rec.close()
    cap.ok = False; cv2.destroyAllWindows()
    pct = lambda v, q: float(np.percentile(v, q)) if v else None                                           # noqa: E731
    res = {"input": SRC, "worker": a.worker, "frames": n, "timed_frames": len(T["total"]), "gpu": torch.cuda.get_device_name(0),
           "cpu": platform.processor() or platform.machine(), "torch": torch.__version__, "ref_build_ms": ref_ms,
           "stages_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95), "mean": float(np.mean(v)) if v else None} for k, v in T.items()},
           "fps_observed": (len(T["total"]) - 1) / (time.perf_counter() - t_first) if t_first and len(T["total"]) > 1 else None,
           "worker_updates": worker.version if worker else 0,
           "worker_update_ms": [round((e["t_end"] - e["t_start"]) * 1000, 1) for e in worker.log if e.get("gate") == "accepted"] if worker else [],
           "worker_offers": len(worker.log) if worker else 0, "recorder_dropped": rec.dropped if rec else 0,
           "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    if res["worker_updates"] and t_first: res["worker_updates_per_s"] = res["worker_updates"] / max(1e-6, time.perf_counter() - t_first)
    print(json.dumps({k: v for k, v in res.items() if k != "worker_update_ms"}, indent=1), flush=True)
    if a.timing_json: Path(a.timing_json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
