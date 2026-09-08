from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch


class Part(str, Enum):
    FACE = "face"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    BODY = "body"


PARTS = tuple(Part)


class RenderLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Unbatched record emitted by main.live_pear_guava.PearRunner.
PARAMETER_SHAPES: dict[str, tuple[int, ...]] = {
    "global_pose": (1, 6),
    "body_pose": (21, 6),
    "left_hand_pose": (15, 6),
    "right_hand_pose": (15, 6),
    "exp": (50,),
    "expression_params": (50,),
    "jaw_params": (3,),
    "pose_params": (3,),
    "eye_pose_params": (6,),
    "eyelid_params": (2,),
}


PART_PARAMETER_KEYS: dict[Part, tuple[str, ...]] = {
    Part.FACE: (
        "exp",
        "expression_params",
        "jaw_params",
        "pose_params",
        "eye_pose_params",
        "eyelid_params",
    ),
    Part.LEFT_HAND: ("left_hand_pose",),
    Part.RIGHT_HAND: ("right_hand_pose",),
    Part.BODY: ("global_pose", "body_pose"),
}


def validate_pose_record(values: Mapping[str, torch.Tensor]) -> None:
    missing = set(PARAMETER_SHAPES).difference(values)
    extra = set(values).difference(PARAMETER_SHAPES)
    if missing or extra:
        raise ValueError(f"Pose keys differ from contract; missing={sorted(missing)}, extra={sorted(extra)}")
    for key, expected in PARAMETER_SHAPES.items():
        value = values[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{key} must be a tensor")
        if tuple(value.shape) != expected:
            raise ValueError(f"{key}: expected shape {expected}, got {tuple(value.shape)}")


def clone_pose_record(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in values.items()}


@dataclass
class PoseState:
    values: dict[str, torch.Tensor]
    staleness: dict[Part, int]
    observed: dict[Part, bool]

    def __post_init__(self) -> None:
        validate_pose_record(self.values)
        if set(self.staleness) != set(PARTS) or set(self.observed) != set(PARTS):
            raise ValueError("PoseState staleness and observed maps must contain every Part")


@dataclass(frozen=True)
class ScoutOutput:
    motion_score: torch.Tensor
    visibility: torch.Tensor
    rois: torch.Tensor
    confidence: torch.Tensor
    appearance_delta: torch.Tensor
    features: torch.Tensor

    def validate(self) -> None:
        batch = self.motion_score.shape[0]
        expected = (batch, len(PARTS))
        for name in ("motion_score", "visibility", "confidence", "appearance_delta"):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name}: expected {expected}, got {tuple(tensor.shape)}")
        if tuple(self.rois.shape) != (batch, len(PARTS), 4):
            raise ValueError(f"rois: expected {(batch, len(PARTS), 4)}, got {tuple(self.rois.shape)}")
        if self.features.ndim != 3 or self.features.shape[:2] != expected:
            raise ValueError("features must have shape [B, 4, F]")


@dataclass(frozen=True)
class TemporalPrediction:
    values: dict[str, torch.Tensor]
    uncertainty: torch.Tensor

    def validate(self) -> None:
        validate_pose_record(self.values)
        if tuple(self.uncertainty.shape) != (len(PARTS),):
            raise ValueError(
                f"uncertainty: expected {(len(PARTS),)}, got {tuple(self.uncertainty.shape)}"
            )


@dataclass(frozen=True)
class RouteDecision:
    update_parts: frozenset[Part]
    forced_parts: frozenset[Part]
    risk: torch.Tensor
    render_level: RenderLevel
    estimated_ms: float
    deadline_feasible: bool

    def validate(self) -> None:
        if tuple(self.risk.shape) != (len(PARTS),):
            raise ValueError(f"risk: expected {(len(PARTS),)}, got {tuple(self.risk.shape)}")
        if not self.forced_parts.issubset(self.update_parts):
            raise ValueError("forced_parts must be included in update_parts")
