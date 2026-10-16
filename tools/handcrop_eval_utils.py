"""Evaluation helpers for the part-wise-query student with hand-crop tokens (PEAR models/pipeline/student_pipeline_vitpose_mq.py).

hand_source(): the FULL-RESOLUTION image as the hand-crop source: image (B, 1, 3, H, W) uint8 (images of a batch
zero-padded to a common size, top-left aligned, so pixel coordinates are unchanged), M (B, 2, 2, 3) = network-crop px ->
image px (the inverse of the evaluator's crop affine, the same for both hands), has (B,) ones.
oracle_boxes(): diagnostic (iv): hand boxes from the TRUE hand keypoints (21 per hand, image px), with the model's own
rule (student_pipeline_vitpose_mq.hand_boxes: box centre, side = 1.5 x extent clamped to [16, 96] / 256), in [0, 1]
units of the network crop. Additive only: the evaluators' metric code is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

MIN_SIDE, MAX_SIDE, MARGIN = 16 / 256, 96 / 256, 1.5


def inverse_affine(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32)


def hand_source(images: list[np.ndarray], affines: list[np.ndarray], device) -> dict:
    """images: list of (H, W, 3) uint8 RGB; affines: list of (2, 3) image -> network-crop px."""
    H = max(im.shape[0] for im in images); W = max(im.shape[1] for im in images)
    canvas = np.zeros((len(images), H, W, 3), np.uint8)
    for i, im in enumerate(images):
        canvas[i, :im.shape[0], :im.shape[1]] = im
    M = np.stack([np.stack([inverse_affine(a)] * 2) for a in affines])
    return {"image": torch.from_numpy(canvas).permute(0, 3, 1, 2)[:, None].to(device),
            "M": torch.from_numpy(M).to(device), "has": torch.ones(len(images), device=device)}


def oracle_boxes(hand_kp_img: np.ndarray, affines: list[np.ndarray], crop: int = 256) -> torch.Tensor:
    """hand_kp_img (B, 2, 21, 2) image px (left, right) -> (B, 2, 3) centre x, y, side in [0, 1] crop units."""
    out = np.zeros((len(affines), 2, 3), np.float32)
    for i, A in enumerate(affines):
        for s in range(2):
            p = hand_kp_img[i, s] @ A[:, :2].T + A[:, 2]
            lo, hi = p.min(0), p.max(0)
            out[i, s, :2] = (lo + hi) / 2 / crop
            out[i, s, 2] = np.clip(MARGIN * (hi - lo).max() / crop, MIN_SIDE, MAX_SIDE)
    return torch.from_numpy(out)
