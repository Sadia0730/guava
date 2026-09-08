from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ScoutConfig:
    height: int = 96
    width: int = 128
    roi_expand: float = 0.12
    motion_gain: float = 4.0
    appearance_gain: float = 2.0


@dataclass(frozen=True)
class TemporalConfig:
    history: int = 3
    velocity_damping: float = 0.85
    uncertainty_growth: float = 0.12
    max_prediction_gap: int = 3


@dataclass(frozen=True)
class RouterWeights:
    motion: float = 1.0
    uncertainty: float = 1.1
    staleness: float = 0.8
    render_impact: float = 1.3
    visibility: float = 0.4
    appearance: float = 0.7


@dataclass(frozen=True)
class RouterCosts:
    fixed_ms: float = 1.8
    expert_shared_ms: float = 7.5
    face_head_ms: float = 0.15
    left_hand_head_ms: float = 0.10
    right_hand_head_ms: float = 0.10
    body_head_ms: float = 0.15
    render_low_ms: float = 5.0
    render_medium_ms: float = 7.5
    render_high_ms: float = 10.0


@dataclass(frozen=True)
class RouterConfig:
    weights: RouterWeights = field(default_factory=RouterWeights)
    costs: RouterCosts = field(default_factory=RouterCosts)
    max_staleness: int = 3
    minimum_risk: float = 0.35
    render_quality_weight: float = 0.6
    impact_weights: tuple[float, float, float, float] = (1.0, 1.2, 1.2, 0.8)


@dataclass(frozen=True)
class RenderLevelConfig:
    gaussian_fraction: float
    refine: bool


@dataclass(frozen=True)
class RenderingConfig:
    low: RenderLevelConfig = field(
        default_factory=lambda: RenderLevelConfig(gaussian_fraction=0.35, refine=False)
    )
    medium: RenderLevelConfig = field(
        default_factory=lambda: RenderLevelConfig(gaussian_fraction=0.65, refine=True)
    )
    high: RenderLevelConfig = field(
        default_factory=lambda: RenderLevelConfig(gaussian_fraction=1.0, refine=True)
    )


@dataclass(frozen=True)
class SchedulerConfig:
    target_fps: float = 50.0
    deadline_ms: float = 20.0
    sleep_to_rate: bool = True
    warmup_frames: int = 10


@dataclass(frozen=True)
class ProfilingConfig:
    warmup_frames: int = 10
    report_window: int = 300
    report_every: int = 60


@dataclass(frozen=True)
class AvatarBudgetConfig:
    scout: ScoutConfig = field(default_factory=ScoutConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def __post_init__(self) -> None:
        if self.scheduler.target_fps <= 0 or self.scheduler.deadline_ms <= 0:
            raise ValueError("target_fps and deadline_ms must be positive")
        expected_ms = 1000.0 / self.scheduler.target_fps
        if abs(expected_ms - self.scheduler.deadline_ms) > 0.25:
            raise ValueError(
                "scheduler.deadline_ms must match 1000 / target_fps; "
                f"got {self.scheduler.deadline_ms} ms and {self.scheduler.target_fps} FPS"
            )
        if self.temporal.max_prediction_gap < 1:
            raise ValueError("temporal.max_prediction_gap must be positive")
        if self.router.max_staleness > self.temporal.max_prediction_gap:
            raise ValueError(
                "router.max_staleness cannot exceed temporal.max_prediction_gap"
            )
        for level in (self.rendering.low, self.rendering.medium, self.rendering.high):
            if not 0.0 < level.gaussian_fraction <= 1.0:
                raise ValueError("gaussian fractions must be in (0, 1]")


def _construct(cls, value: Mapping[str, Any] | None):
    return cls(**dict(value or {}))


def from_mapping(value: Mapping[str, Any]) -> AvatarBudgetConfig:
    router_value = dict(value.get("router", {}))
    router = RouterConfig(
        weights=_construct(RouterWeights, router_value.pop("weights", None)),
        costs=_construct(RouterCosts, router_value.pop("costs", None)),
        **router_value,
    )
    rendering_value = dict(value.get("rendering", {}))
    rendering_defaults = RenderingConfig()
    rendering = RenderingConfig(
        low=(
            _construct(RenderLevelConfig, rendering_value["low"])
            if "low" in rendering_value
            else rendering_defaults.low
        ),
        medium=(
            _construct(RenderLevelConfig, rendering_value["medium"])
            if "medium" in rendering_value
            else rendering_defaults.medium
        ),
        high=(
            _construct(RenderLevelConfig, rendering_value["high"])
            if "high" in rendering_value
            else rendering_defaults.high
        ),
    )
    return AvatarBudgetConfig(
        scout=_construct(ScoutConfig, value.get("scout")),
        temporal=_construct(TemporalConfig, value.get("temporal")),
        router=router,
        rendering=rendering,
        scheduler=_construct(SchedulerConfig, value.get("scheduler")),
        profiling=_construct(ProfilingConfig, value.get("profiling")),
    )


def load_config(path: str | Path) -> AvatarBudgetConfig:
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping in {path}")
    return from_mapping(payload)
