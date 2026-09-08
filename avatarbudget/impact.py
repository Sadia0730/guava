from __future__ import annotations

import torch
import torch.nn.functional as F

from .contracts import PARTS, PART_PARAMETER_KEYS, Part, clone_pose_record, validate_pose_record


def _ssim_like(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Small differentiable global SSIM surrogate; inputs are [B, C, H, W]."""
    dims = (-3, -2, -1)
    mean_left = left.mean(dim=dims)
    mean_right = right.mean(dim=dims)
    var_left = left.var(dim=dims, unbiased=False)
    var_right = right.var(dim=dims, unbiased=False)
    covariance = ((left - mean_left[:, None, None, None]) * (right - mean_right[:, None, None, None])).mean(dim=dims)
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left.square() + mean_right.square() + c1) * (var_left + var_right + c2)
    )


def _roi_l1(left: torch.Tensor, right: torch.Tensor, roi: torch.Tensor, size: int = 64):
    batch = left.shape[0]
    if roi.ndim == 1:
        roi = roi.unsqueeze(0).expand(batch, -1)
    if tuple(roi.shape) != (batch, 4):
        raise ValueError(f"roi must be [B,4], got {tuple(roi.shape)}")
    axis = torch.linspace(0.0, 1.0, size, device=left.device, dtype=left.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x1, y1, x2, y2 = roi.to(left).unbind(dim=-1)
    grid_x = x1[:, None, None] + xx * (x2 - x1)[:, None, None]
    grid_y = y1[:, None, None] + yy * (y2 - y1)[:, None, None]
    grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
    left_crop = F.grid_sample(left, grid, mode="bilinear", padding_mode="border", align_corners=True)
    right_crop = F.grid_sample(right, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return F.l1_loss(left_crop, right_crop, reduction="none").mean(dim=(-3, -2, -1))


def counterfactual_render_damage(
    reference: torch.Tensor,
    skipped: torch.Tensor,
    silhouette_threshold: float = 0.02,
    roi: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Render-space damage labels for router training.

    Both inputs have shape ``[B, 3, H, W]`` and values in ``[0, 1]``.
    """
    if reference.shape != skipped.shape or reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError("reference and skipped must have identical [B, 3, H, W] shapes")
    l1 = F.l1_loss(skipped, reference, reduction="none").mean(dim=(-3, -2, -1))
    ssim_damage = 1.0 - _ssim_like(reference, skipped)
    reference_mask = reference.mean(dim=1) > silhouette_threshold
    skipped_mask = skipped.mean(dim=1) > silhouette_threshold
    intersection = (reference_mask & skipped_mask).sum(dim=(-2, -1)).float()
    union = (reference_mask | skipped_mask).sum(dim=(-2, -1)).float().clamp_min(1.0)
    silhouette_damage = 1.0 - intersection / union
    roi_l1 = l1 if roi is None else _roi_l1(reference, skipped, roi)
    total = l1 + 2.0 * roi_l1 + 0.25 * ssim_damage + 0.5 * silhouette_damage
    return {
        "total": total,
        "l1": l1,
        "roi_l1": roi_l1,
        "ssim_damage": ssim_damage,
        "silhouette_damage": silhouette_damage,
    }


class CounterfactualRenderLabeler:
    """Render full and part-skipped states to produce router supervision."""

    def __init__(self, render_pose):
        self.render_pose = render_pose

    @staticmethod
    def skip_part(
        teacher_pose: dict[str, torch.Tensor],
        predicted_pose: dict[str, torch.Tensor],
        part: Part,
    ) -> dict[str, torch.Tensor]:
        validate_pose_record(teacher_pose)
        validate_pose_record(predicted_pose)
        value = clone_pose_record(teacher_pose)
        for key in PART_PARAMETER_KEYS[part]:
            value[key] = predicted_pose[key].detach().clone()
        return value

    def __call__(
        self,
        teacher_pose: dict[str, torch.Tensor],
        predicted_pose: dict[str, torch.Tensor],
        rois: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reference = self.render_pose(teacher_pose)
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        damages = []
        components: dict[str, list[torch.Tensor]] = {}
        for part_index, part in enumerate(PARTS):
            skipped = self.render_pose(self.skip_part(teacher_pose, predicted_pose, part))
            if skipped.ndim == 3:
                skipped = skipped.unsqueeze(0)
            roi = None if rois is None else rois[:, part_index]
            damage = counterfactual_render_damage(reference, skipped, roi=roi)
            damages.append(damage["total"])
            for key, value in damage.items():
                components.setdefault(key, []).append(value)
        return {key: torch.stack(values, dim=1) for key, values in components.items()}
