from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ScoutConfig
from .contracts import PARTS, ScoutOutput


DEFAULT_ROIS = torch.tensor(
    [
        [0.34, 0.02, 0.66, 0.34],  # face
        [0.02, 0.25, 0.40, 0.82],  # left side of image / subject's right hand
        [0.60, 0.25, 0.98, 0.82],  # right side of image / subject's left hand
        [0.18, 0.12, 0.82, 0.98],  # upper body
    ],
    dtype=torch.float32,
)


class CheapScout(nn.Module):
    """Always-on low-resolution motion and appearance observer.

    Input: RGB float tensor ``[B, 3, H, W]`` in ``[0, 1]``.
    Output scalar fields: ``[B, 4]`` in Part enum order; ROIs: ``[B, 4, 4]``;
    per-ROI features: ``[B, 4, 6]``.
    """

    def __init__(self, config: ScoutConfig):
        super().__init__()
        self.config = config
        self.register_buffer("default_rois", DEFAULT_ROIS.clone(), persistent=False)
        self._previous: torch.Tensor | None = None
        self._previous_features: torch.Tensor | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_features = None

    def _expand_rois(self, rois: torch.Tensor) -> torch.Tensor:
        center = (rois[..., :2] + rois[..., 2:]) * 0.5
        half = (rois[..., 2:] - rois[..., :2]) * (0.5 + self.config.roi_expand)
        return torch.cat((center - half, center + half), dim=-1).clamp(0.0, 1.0)

    @staticmethod
    def _extract_rois(image: torch.Tensor, rois: torch.Tensor, size: int = 16) -> torch.Tensor:
        """Vectorized normalized ROI sampling, returning [B, 4, C, size, size]."""
        batch, channels = image.shape[:2]
        axis = torch.linspace(0.0, 1.0, size, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        x1, y1, x2, y2 = rois.unbind(dim=-1)
        grid_x = x1[..., None, None] + xx * (x2 - x1)[..., None, None]
        grid_y = y1[..., None, None] + yy * (y2 - y1)[..., None, None]
        grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
        repeated = image[:, None].expand(-1, len(PARTS), -1, -1, -1)
        sampled = F.grid_sample(
            repeated.reshape(batch * len(PARTS), channels, *image.shape[-2:]),
            grid.reshape(batch * len(PARTS), size, size, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(batch, len(PARTS), channels, size, size)

    @classmethod
    def _roi_features(cls, image: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        crops = cls._extract_rois(image, rois)
        gray = crops.mean(dim=2, keepdim=True)
        dx = (gray[..., 1:] - gray[..., :-1]).abs().mean(dim=(-3, -2, -1))
        dy = (gray[..., 1:, :] - gray[..., :-1, :]).abs().mean(dim=(-3, -2, -1))
        channels = crops.mean(dim=(-2, -1))
        contrast = gray.flatten(2).std(dim=-1)
        return torch.cat((channels, contrast[..., None], dx[..., None], dy[..., None]), dim=-1)

    @classmethod
    def _roi_motion(cls, current: torch.Tensor, previous: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        difference = (current - previous).abs().mean(dim=1, keepdim=True)
        return cls._extract_rois(difference, rois).mean(dim=(-3, -2, -1))

    def forward(self, frame: torch.Tensor, rois: torch.Tensor | None = None) -> ScoutOutput:
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError(f"frame must have shape [B, 3, H, W], got {tuple(frame.shape)}")
        if not frame.is_floating_point():
            raise TypeError("frame must be floating point and scaled to [0, 1]")
        low = F.interpolate(
            frame,
            size=(self.config.height, self.config.width),
            mode="bilinear",
            align_corners=False,
        )
        batch = frame.shape[0]
        if rois is None:
            rois = self.default_rois.to(frame).unsqueeze(0).expand(batch, -1, -1)
        if tuple(rois.shape) != (batch, len(PARTS), 4):
            raise ValueError(f"rois must have shape {(batch, len(PARTS), 4)}")
        rois = self._expand_rois(rois)
        features = self._roi_features(low, rois)

        if self._previous is None or self._previous.shape != low.shape:
            motion = torch.ones((batch, len(PARTS)), device=frame.device, dtype=frame.dtype)
            appearance = motion.clone()
        else:
            motion = self._roi_motion(low, self._previous, rois)
            motion = (motion * self.config.motion_gain).clamp(0.0, 1.0)
            appearance = (features - self._previous_features).abs().mean(dim=-1)
            appearance = (appearance * self.config.appearance_gain).clamp(0.0, 1.0)

        contrast = features[..., 3]
        brightness = features[..., :3].mean(dim=-1)
        visibility = ((contrast * 8.0).clamp(0.0, 1.0) * (brightness > 0.02)).to(frame.dtype)
        confidence = (0.25 + 0.75 * visibility).clamp(0.0, 1.0)
        output = ScoutOutput(
            motion_score=motion,
            visibility=visibility,
            rois=rois,
            confidence=confidence,
            appearance_delta=appearance,
            features=features,
        )
        output.validate()
        self._previous = low.detach()
        self._previous_features = features.detach()
        return output
