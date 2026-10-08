"""Shared PEAR crop, camera, and pose-evaluation geometry."""

from __future__ import annotations

import numpy as np
import torch


IMAGE_SIZE = 256
PEAR_MODEL_X_MIN = 32
PEAR_MODEL_X_MAX = 224

# SMPL/SMPL-X body-joint indices. SMPLX_to_J14.pkl replaces these SMPL-X
# joint locations with the same 14-joint regressor used by PEAR evaluation.
J14_INDICES = np.asarray([8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15])
LOWER6_INDICES = np.asarray([8, 5, 2, 1, 4, 7])
LOWER8_INDICES = np.asarray([1, 2, 4, 5, 7, 8, 10, 11])

# 3DPW poses2D uses the OpenPose-18 ordering documented by its annotations.
OPENPOSE18_TO_SMPL = {
    1: 12,
    2: 17,
    3: 19,
    4: 21,
    5: 16,
    6: 18,
    7: 20,
    8: 2,
    9: 5,
    10: 8,
    11: 1,
    12: 4,
    13: 7,
}


def bbox_from_keypoints(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    visible = np.asarray(confidence) > 0
    if int(visible.sum()) < 2:
        raise ValueError("at least two visible keypoints are required")
    selected = np.asarray(points, dtype=np.float32)[visible]
    return np.asarray(
        [selected[:, 0].min(), selected[:, 1].min(), selected[:, 0].max(), selected[:, 1].max()],
        dtype=np.float32,
    )


def square_crop_transform(
    box: np.ndarray,
    crop_scale: float = 1.25,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    center_x = float(box[0] + box[2]) * 0.5
    center_y = float(box[1] + box[3]) * 0.5
    side = max(float(box[2] - box[0]), float(box[3] - box[1])) * crop_scale
    if not np.isfinite(side) or side <= 1.0:
        raise ValueError(f"invalid crop box: {box.tolist()}")
    scale = image_size / side
    return np.asarray(
        [
            [scale, 0.0, image_size * 0.5 - scale * center_x],
            [0.0, scale, image_size * 0.5 - scale * center_y],
        ],
        dtype=np.float32,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1), np.float32)), axis=-1)
    return np.einsum("ij,...j->...i", transform, homogeneous)


def invert_affine(transform: np.ndarray) -> np.ndarray:
    full = np.eye(3, dtype=np.float64)
    full[:2] = np.asarray(transform, dtype=np.float64)
    return np.linalg.inv(full)[:2].astype(np.float32)


def crop_extent_in_source(transform: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    corners = np.asarray(
        [[0.0, 0.0], [image_size, 0.0], [image_size, image_size], [0.0, image_size]],
        dtype=np.float32,
    )
    return transform_points(corners, invert_affine(transform))


def pear_camera_project_normalized(rt: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """Project PEAR joints into normalized coordinates of the 256-square crop.

    This is algebraically equivalent to PEAR's GS_Camera perspective projection
    divided by its 1024-pixel rendering coordinate system.
    """

    rotated = torch.einsum("bij,bkj->bki", rt[:, :3, :3].float(), joints.float())
    camera = rotated + rt[:, None, :3, 3].float()
    perspective = 24.0 * camera[..., :2] / camera[..., 2:3].clamp_min(1e-4)
    return (1.0 - perspective) * 0.5


def root_center(joints: torch.Tensor, root_index: int = 0) -> torch.Tensor:
    return joints - joints[:, root_index : root_index + 1]


def similarity_align(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batched similarity-Procrustes alignment of prediction to target."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must both have shape BxJx3")
    prediction = prediction.float()
    target = target.float()
    pred_mean = prediction.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    pred_norm = torch.linalg.vector_norm(pred_centered, dim=(1, 2), keepdim=True).clamp_min(1e-8)
    target_norm = torch.linalg.vector_norm(target_centered, dim=(1, 2), keepdim=True).clamp_min(1e-8)
    pred_unit = pred_centered / pred_norm
    target_unit = target_centered / target_norm
    covariance = torch.matmul(pred_unit.transpose(1, 2), target_unit)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = torch.matmul(u, vh)
    determinant = torch.det(rotation)
    correction = torch.eye(3, device=prediction.device, dtype=prediction.dtype)[None].repeat(
        len(prediction), 1, 1
    )
    correction[:, -1, -1] = torch.where(determinant < 0, -1.0, 1.0)
    rotation = torch.matmul(torch.matmul(u, correction), vh)
    scale = target_norm / pred_norm
    return scale * torch.matmul(pred_centered, rotation) + target_mean


def mpjpe_mm(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction.float() - target.float(), dim=-1).mean(dim=-1) * 1000.0


def pa_mpjpe_mm(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mpjpe_mm(similarity_align(prediction, target), target)

