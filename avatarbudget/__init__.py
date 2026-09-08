"""Budget-aware PEAR-to-GUAVA runtime components."""

from .config import AvatarBudgetConfig, load_config
from .contracts import (
    PARAMETER_SHAPES,
    PART_PARAMETER_KEYS,
    Part,
    PoseState,
    RenderLevel,
    RouteDecision,
    ScoutOutput,
    TemporalPrediction,
    validate_pose_record,
)

__all__ = [
    "AvatarBudgetConfig",
    "PARAMETER_SHAPES",
    "PART_PARAMETER_KEYS",
    "Part",
    "PoseState",
    "RenderLevel",
    "RouteDecision",
    "ScoutOutput",
    "TemporalPrediction",
    "load_config",
    "validate_pose_record",
]
