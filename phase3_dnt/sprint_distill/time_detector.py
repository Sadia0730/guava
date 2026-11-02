"""PREREG phase 4 (H4 smoke): CPU time of the two parts of SprintFitter's "dwpose_cpu" stage on real frames -- the YOLOX-L person
detector alone (all a lean k = 0 driving loop needs besides the crop and SPRINT's forward) and the DWPose-L whole-body keypoints
(needed only by the k = 60 refinement). Same ONNX files, runtime and thread count as SprintFitter (16 intra-op threads, CPU).
Usage: python time_detector.py --video V [--n 60] [--out JSON]"""
import argparse, importlib.util, json, time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

EHMT = Path(__file__).resolve().parents[2] / "EHM-Tracker"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--n", type=int, default=60)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--out", default=""); a = ap.parse_args()
det = load("ehm_dwpose_onnxdet", EHMT / "src/modules/dwpose/onnxdet.py"); pose = load("ehm_dwpose_onnxpose", EHMT / "src/modules/dwpose/onnxpose.py")
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
s_det = ort.InferenceSession(str(EHMT / "pretrained/dwpose/yolox_l.onnx"), so, providers=["CPUExecutionProvider"])
s_pose = ort.InferenceSession(str(EHMT / "pretrained/dwpose/dw-ll_ucoco_384.onnx"), so, providers=["CPUExecutionProvider"])
cap = cv2.VideoCapture(a.video); frames = []
while len(frames) < a.n + 5:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
td, tp = [], []
for i, img in enumerate(frames):
    t0 = time.perf_counter(); B = det.inference_detector(s_det, img); t1 = time.perf_counter()
    box = B[int(np.argmax((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])))] if len(B) else np.array([0, 0, img.shape[1] - 1, img.shape[0] - 1], np.float64)
    pose.inference_pose(s_pose, np.asarray(box, np.float64)[None], img); t2 = time.perf_counter()
    if i >= 5: td.append(t1 - t0); tp.append(t2 - t1)                                     # first 5 = warm-up
out = {"video": a.video, "wh": list(frames[0].shape[1::-1]), "n": len(td), "threads": a.threads,
       "detector_ms_median": 1000 * float(np.median(td)), "keypoints_ms_median": 1000 * float(np.median(tp)),
       "detector_ms_p90": 1000 * float(np.percentile(td, 90)), "keypoints_ms_p90": 1000 * float(np.percentile(tp, 90))}
print(json.dumps(out, indent=1))
if a.out: Path(a.out).write_text(json.dumps(out, indent=1))
