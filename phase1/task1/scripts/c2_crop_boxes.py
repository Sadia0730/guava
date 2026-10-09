#!/usr/bin/env python
"""Checkpoint 2, section 4: YOLOX boxes vs ground-truth-mesh boxes on BEDLAM2, plus joint visibility.

Read-only. Samples labelled frames of single-person render jobs, runs YOLOX-L (CPU provider,
parallel), builds the box of the full projected ground-truth mesh, and compares them. Then
classifies GT wrists/ankles as visible centre / side strips / outside the 256x256 crop for
square x1.25 crops from (a) the GT-mesh box, (b) the YOLOX box, (c) the GT-mesh box with
+/-15% scale and +/-10% shift jitter. Also times detection and measures frame JPEG size.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import c2_geometry_conventions as g

ROOT = Path("/home/ubx858/guava")
FRAMES = Path("/raid/ubx858/datasets/expanded/BEDLAM")
OUT = Path("/raid/ubx858/outputs/phase1_task1/checkpoint2/crop_boxes")
DETECTOR = ROOT / "EHM-Tracker" / "pretrained" / "dwpose" / "yolox_l.onnx"
JOINTS = {"wrist": (20, 21), "ankle": (7, 8)}
_DET = None


def detect(image_bgr):
    global _DET
    if _DET is None:
        import onnxruntime
        sys.path.insert(0, str(ROOT / "EHM-Tracker"))
        from src.modules.dwpose.onnxdet import inference_detector
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        sess = onnxruntime.InferenceSession(str(DETECTOR), opts, providers=["CPUExecutionProvider"])
        _DET = lambda im: inference_detector(sess, im)  # noqa: E731
    t0 = time.time()
    boxes = _DET(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return ([] if boxes is None or len(boxes) == 0 else np.asarray(boxes, float)[:, :4].tolist()), time.time() - t0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def square(box, scale=1.25):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = scale * max(box[2] - box[0], box[3] - box[1])
    return cx, cy, side


def region(p, cx, cy, side):
    x = (p[0] - (cx - side / 2)) * 256 / side
    y = (p[1] - (cy - side / 2)) * 256 / side
    if not (0 <= x < 256 and 0 <= y < 256):
        return "outside_crop"
    return "centre" if 32 <= x < 224 else "side_strip"


def main():
    rng = np.random.default_rng(11)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sorted(p.name for p in FRAMES.iterdir() if (p / "mp4").is_dir())
    model = g.build(True)
    samples = []
    for job in jobs:
        d = np.load(g.LABELS / f"{job}.npz", allow_pickle=True)
        for i in rng.choice(len(d["imgname"]), 10, replace=False):
            seq, name = d["imgname"][i].split("/")
            frame_no = int(name.split("_")[-1].split(".")[0])
            cap = cv2.VideoCapture(str(FRAMES / job / "mp4" / f"{seq}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, image = cap.read()
            cap.release()
            if not ok:
                continue
            t = d["trans_cam"][i] + d["cam_ext"][i][:3, 3]
            with torch.no_grad():
                out = g.forward(model, d["pose_cam"][i][None], d["shape"][i][None], t[None])
            uv = g.project(out.vertices.numpy(), d["cam_int"][i][None])[0]
            ju = g.project(out.joints[:, :55].numpy(), d["cam_int"][i][None])[0]
            samples.append({"job": job, "image": image, "mesh_box": [*uv.min(0), *uv.max(0)], "joints": ju,
                            "hw": image.shape[:2]})
    print(f"{len(samples)} frames sampled from {len(jobs)} jobs", flush=True)

    t0 = time.time()
    with mp.get_context("spawn").Pool(32) as pool:
        dets = pool.map(detect, [s["image"] for s in samples], chunksize=4)
    wall = time.time() - t0
    jpeg_kb = [len(cv2.imencode(".jpg", s["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])[1]) / 1024 for s in samples[:100]]

    ious, centre_err, scale_ratio, misses = [], [], [], 0
    vis = {k: {} for k in ("gt_box", "yolox_box", "gt_box_jitter")}
    for s, (boxes, _) in zip(samples, dets):
        h, w = s["hw"]
        mb = s["mesh_box"]
        clipped = [max(0, mb[0]), max(0, mb[1]), min(w, mb[2]), min(h, mb[3])]
        best = max(boxes, key=lambda b: iou(b, clipped), default=None)
        if best is None or iou(best, clipped) < 0.1:
            misses += 1
            best = None
        else:
            ious.append(iou(best, clipped))
            gs = max(clipped[2] - clipped[0], clipped[3] - clipped[1])
            centre_err.append(np.hypot((best[0] + best[2] - clipped[0] - clipped[2]) / 2,
                                       (best[1] + best[3] - clipped[1] - clipped[3]) / 2) / gs)
            scale_ratio.append(max(best[2] - best[0], best[3] - best[1]) / gs)
        crops = {"gt_box": square(mb)}
        if best is not None:
            crops["yolox_box"] = square(best)
        cx, cy, side = square(mb)
        jitter = side * (1 + rng.uniform(-0.15, 0.15))
        crops["gt_box_jitter"] = (cx + rng.uniform(-0.1, 0.1) * side, cy + rng.uniform(-0.1, 0.1) * side, jitter)
        for policy, (bx, by, bs) in crops.items():
            for part, idx in JOINTS.items():
                for j in idx:
                    p = s["joints"][j]
                    key = f"{part}:{region(p, bx, by, bs)}" + (":off_image" if not (0 <= p[0] < w and 0 <= p[1] < h) else "")
                    vis[policy][key] = vis[policy].get(key, 0) + 1

    det_times = [d[1] for d in dets]
    report = {
        "frames": len(samples),
        "yolox_vs_gt_mesh_box": {
            "gt_box": "full projected mesh, clipped to the image",
            "detector_misses_iou_lt_0.1": misses,
            "iou": {"mean": float(np.mean(ious)), "median": float(np.median(ious)), "p10": float(np.percentile(ious, 10))},
            "centre_offset_over_box_size": {"mean": float(np.mean(centre_err)), "p90": float(np.percentile(centre_err, 90))},
            "scale_ratio_yolox_over_gt": {"mean": float(np.mean(scale_ratio)), "p10": float(np.percentile(scale_ratio, 10)),
                                          "p90": float(np.percentile(scale_ratio, 90))},
        },
        "timing": {"detect_seconds_per_frame_single_process": float(np.median(det_times)),
                   "wall_seconds_32_processes": wall, "frames_per_second_32_processes": len(samples) / wall},
        "full_frame_jpeg_q95_kb": {"mean": float(np.mean(jpeg_kb))},
        "joint_visibility": vis,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
