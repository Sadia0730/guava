"""PREREG phase 5: person-box sources for live SPRINT driving (used by eval_guava_rows.py --box-source).
  mediapipe  MediaPipe Pose (legacy solution, full landmark model, static mode) on a square region of 2x the previous box's long side;
             box = landmarks with visibility >= 0.5, scaled by the margin about its centre. Lost: no / < 6 visible landmarks, or
             IoU < 0.1 with the previous box.
  track      box of SPRINT's previous-frame projected OpenPose body points 0-17 that fall inside the image, scaled by the margin.
             Lost: < 4 such points, or long side changed by > 1.5x vs the previous box.
On loss the caller re-detects (YOLOX + H1c subject linking). Boxes are x1 y1 x2 y2 in original-frame pixels; SPRINT's crop rule
(square, 1.1 x long side) is applied downstream by the fitter."""
from __future__ import annotations

import numpy as np
import torch


def iou(a, b):
    w, h = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def scaled(pts, m):
    lo, hi = pts.min(0), pts.max(0); c, h = (lo + hi) / 2, (hi - lo) / 2 * m
    return np.array([c[0] - h[0], c[1] - h[1], c[0] + h[0], c[1] + h[1]], np.float64)


class MediaPipeBox:
    def __init__(self, margin):
        import mediapipe as mp
        self.m = margin
        self.pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

    def __call__(self, img, prev_box, prev_res=None):
        """-> box or None (lost)."""
        H, W = img.shape[:2]
        c = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2]); s = max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1])
        x0, y0 = int(max(0, c[0] - s)), int(max(0, c[1] - s)); x1, y1 = int(min(W, c[0] + s)), int(min(H, c[1] + s))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        r = self.pose.process(np.ascontiguousarray(img[y0:y1, x0:x1]))
        if r.pose_landmarks is None:
            return None
        lm = np.array([[p.x * (x1 - x0) + x0, p.y * (y1 - y0) + y0, p.visibility] for p in r.pose_landmarks.landmark])
        v = lm[lm[:, 2] >= 0.5, :2]
        if len(v) < 6:
            return None
        box = scaled(v, self.m)
        return box if iou(box, prev_box) >= 0.1 else None


class TrackBox:
    def __init__(self, margin, fitter):
        import roma
        self.m, self.f, self.roma = margin, fitter, roma

    @torch.no_grad()
    def body_points(self, res, shape):
        """OpenPose body points 0-17 of a SPRINT result in original-frame pixels (raw driving result or a full fit's refined record)."""
        f = self.f; dev = f.dev
        if "uv_after" in res:                                                             # full fit (source frame): its own projection
            uv = np.asarray(res["uv_after"], np.float64)[:18]
        else:
            sx = res["raw_smplx"]; R = lambda v, n: self.roma.rotvec_to_rotmat(torch.as_tensor(np.asarray(v, np.float32)).reshape(-1, 3).to(dev)).reshape(1, n, 3, 3)
            beta = torch.as_tensor(np.asarray(shape, np.float32)[:200], device=dev)[None]
            exp = torch.as_tensor(np.asarray(sx["exp"], np.float32)[:50], device=dev)[None]
            _, K = f.lean(R(sx["global_pose"], 1), R(sx["body_pose"], 21), R(sx["jaw_pose"], 1), f._st["eye"],
                          R(sx["left_hand_pose"], 15), R(sx["right_hand_pose"], 15), beta, exp)
            RT = torch.as_tensor(np.asarray(sx["camera_RT_params"], np.float32), device=dev)
            uv = f.project(RT[:, :3], RT[:, 3], K[0]).cpu().numpy().astype(np.float64)[:18]
        Minv = np.linalg.inv(np.asarray(res["M"], np.float64))
        return (np.c_[uv * 1024.0, np.ones(len(uv))] @ Minv.T)[:, :2]

    def __call__(self, img, prev_box, prev_res, shape):
        H, W = img.shape[:2]
        p = self.body_points(prev_res, shape)
        p = p[(p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)]
        if len(p) < 4:
            return None
        box = scaled(p, self.m)
        ls, lp = max(box[2] - box[0], box[3] - box[1]), max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1])
        return box if 1 / 1.5 <= ls / max(lp, 1e-6) <= 1.5 else None
