from __future__ import annotations

from typing import Mapping

import torch

from .config import RenderingConfig
from .contracts import RenderLevel


def progressive_indices(total: int, body_count: int, fraction: float, device) -> torch.Tensor:
    if not 0 <= body_count <= total:
        raise ValueError("body_count must lie inside the Gaussian array")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    body = torch.arange(body_count, device=device)
    uv_count = total - body_count
    if fraction >= 1.0 or uv_count == 0:
        uv = torch.arange(body_count, total, device=device)
    else:
        kept = max(1, int(round(uv_count * fraction)))
        uv = torch.linspace(body_count, total - 1, steps=kept, device=device).round().long().unique()
    return torch.cat((body, uv))


def select_gaussians(
    assets: Mapping[str, torch.Tensor | int], fraction: float
) -> dict[str, torch.Tensor | int]:
    """Keep all mesh-vertex Gaussians and a deterministic fraction of UV Gaussians."""
    total = int(assets["xyz"].shape[1])
    body_count = int(assets["smplx_xyz_deform"].shape[1])
    indices = progressive_indices(total, body_count, fraction, assets["xyz"].device)
    selected: dict[str, torch.Tensor | int] = {}
    for key, value in assets.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == total:
            selected[key] = value.index_select(1, indices)
        else:
            selected[key] = value
    return selected


class ProgressiveGuavaRenderer:
    """Split GUAVA deformation, rasterization, and optional refinement for profiling."""

    def __init__(self, avatar, render_model, camera, config: RenderingConfig, profiler=None):
        self.avatar = avatar
        self.render_model = render_model
        self.camera = camera
        self.config = config
        self.profiler = profiler

    def _profile(self, name: str):
        if self.profiler is None:
            from contextlib import nullcontext

            return nullcontext()
        return self.profiler.cuda(name)

    def render(self, target, level: RenderLevel) -> torch.Tensor:
        setting = getattr(self.config, level.value)
        with self._profile("guava_deformation"):
            assets = self.avatar(target)
            assets = select_gaussians(assets, setting.gaussian_fraction)
        with self._profile("gaussian_rasterization"):
            rasterized = self.render_model.forward_raw(assets, self.camera, bg=0.0)
        if setting.refine:
            with self._profile("neural_refiner"):
                rendered = self.render_model.refine_raw(rasterized)
        else:
            rendered = rasterized["raw_renders"]
        image = (rendered[0].clamp(0.0, 1.0).flip(0) * 255.0).permute(1, 2, 0)
        return image.to(torch.uint8).contiguous()
