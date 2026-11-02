"""PREREG phase 5: end-to-end live timing of the three box sources on any video (server or laptop).
Per frame, synchronised: person box (yolox + H1c linking | mediapipe | track) -> crop + SPRINT forward (k = 0, lean backend) ->
GUAVA-FB render. Source avatar from the first frame (largest YOLOX box, full SPRINT fit, not timed). First --warmup frames per source
are not counted. Prints and writes p50 / p95 ms, lost-frame counts, device and host load.
Needs: this repo with its assets, an NVIDIA GPU (GUAVA-FB's rasteriser is CUDA-only), the guava env (mediapipe 0.10.21 included).
Usage: python time_live_boxes.py --video clip.mp4 [--frames 300] [--sources yolox,mediapipe,track] [--margin-mp M] [--margin-track M]
       [--out timing.json]   (bounded threads: set OMP_NUM_THREADS; --dwpose-threads for ONNX)"""
from __future__ import annotations

import argparse, importlib.util, json, os, platform, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--sources", default="yolox,mediapipe,track")
ap.add_argument("--margin-mp", type=float, default=1.25); ap.add_argument("--margin-track", type=float, default=1.25)
ap.add_argument("--dwpose-threads", type=int, default=4); ap.add_argument("--out", default="")
ap.add_argument("--sprint-ckpt", default="/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / "phase3_dnt/sprint_fast_driving.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = SAF.SprintFitter(a.sprint_ckpt, "cuda:0", 150, dwpose_threads=a.dwpose_threads)

import cv2, tempfile, torch                                                                           # noqa: E402
if os.environ.get("OMP_NUM_THREADS"): cv2.setNumThreads(int(os.environ["OMP_NUM_THREADS"]))
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import fb_common as F                                                                                 # noqa: E402
import box_sources as BS                                                                              # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                          # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                         # noqa: E402
dev = torch.device("cuda:0")
cfg, inf, ren, _ = F.setup_a20(dev)
st = torch.load(a.fb_ckpt, map_location="cpu", weights_only=False)
inf.load_state_dict(st["model"], strict=False); ren.load_state_dict(st["render_model"], strict=False); inf.eval(); ren.eval()

cap = cv2.VideoCapture(a.video); frames = []
while len(frames) < a.frames + 1:
    ok, fr = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
assert len(frames) > a.warmup + 2, "video too short"


def h1c_box(img, ref):
    """YOLOX + H1c linking (eval_guava_rows.subject_box with --track-nearest). -> (box, fallback)."""
    B = fitter.det.inference_detector(fitter.s_det, img)
    if len(B) == 0: return np.asarray(ref, np.float64), 1
    v = [BS.iou(b, ref) for b in B]; j = int(np.argmax(v))
    if v[j] >= 0.1: return B[j], 0
    c0 = np.array([(ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2]); sz = max(ref[2] - ref[0], ref[3] - ref[1], 1.0)
    return B[int(np.argmin([np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - c0) / sz for b in B]))], 1


rs = fitter.fit(frames[0]); shp = rs["shape"]                                                           # source: full fit (not timed)
with tempfile.TemporaryDirectory(dir="/dev/shm" if Path("/dev/shm").exists() else None) as td:
    SAF.write_tracked(Path(td), "live", "frame_000000", rs)
    cl = F.load_clips_parallel([(td, None, True)], workers=1); fbd = F.FBData(cfg, cl)
    src = to_dev(F.collate1(fbd.source_info(0, "frame_000000")), dev)
with torch.no_grad():
    torch.manual_seed(0); vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()


@torch.no_grad()
def drive_render(img, box):
    r = fitter.fit(img, box=box, iters=0, shape_lock=shp)
    comp = F.compact_record(dict(r["record"]), shp); sm = {k: torch.as_tensor(v.copy()) for k, v in comp["smplx"].items()}
    sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
    M4 = torch.eye(4); M4[:3, :4] = sm["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
    t = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c),
         "invtanfov": comp["invtanfov"]}
    t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
    tt = to_dev(F.collate1(t), dev); torch.manual_seed(0); out = ren(av(tt), tt["render_cam_params"], bg=0.0)["renders"]
    return r, out


res = {"video": a.video, "frames": len(frames) - 1, "warmup": a.warmup, "gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or platform.machine(),
       "cpu_count": os.cpu_count(), "omp_threads": os.environ.get("OMP_NUM_THREADS"), "dwpose_threads": a.dwpose_threads, "sources": {}}
for s in a.sources.split(","):
    boxer = BS.MediaPipeBox(a.margin_mp) if s == "mediapipe" else BS.TrackBox(a.margin_track, fitter) if s == "track" else None
    pbox, pres, ms, lost = np.asarray(rs["box"], np.float64), rs, [], 0
    load0 = os.getloadavg()
    for i, img in enumerate(frames[1:]):
        torch.cuda.synchronize(); t0 = time.perf_counter(); l_ = 0
        if s == "yolox":
            box, _ = h1c_box(img, pbox)
        else:
            box = boxer(img, pbox) if s == "mediapipe" else boxer(img, pbox, pres, shp)
            if box is None: l_ = 1; box, _ = h1c_box(img, pbox)
        r, _ = drive_render(img, box); torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
        pbox, pres = np.asarray(r["box"], np.float64), r
        if i >= a.warmup: ms.append(dt); lost += l_
    res["sources"][s] = {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)), "n": len(ms), "lost_frames": lost,
                         "margin": a.margin_mp if s == "mediapipe" else a.margin_track if s == "track" else None,
                         "load_avg_before_after": [load0, os.getloadavg()]}
    print(s, json.dumps(res["sources"][s]), flush=True)
print(json.dumps(res, indent=1))
if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
