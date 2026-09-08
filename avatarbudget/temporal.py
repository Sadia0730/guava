from __future__ import annotations

from collections import deque
from typing import Mapping

import torch

from .config import TemporalConfig
from .contracts import (
    PARTS,
    PART_PARAMETER_KEYS,
    Part,
    PoseState,
    TemporalPrediction,
    clone_pose_record,
    validate_pose_record,
)


class ConstantVelocityPredictor:
    """Short-gap predictor over PEAR's continuous 6D rotation records."""

    def __init__(self, config: TemporalConfig):
        self.config = config
        self._history: deque[dict[str, torch.Tensor]] = deque(maxlen=max(2, config.history))
        self._staleness = {part: 0 for part in PARTS}

    @property
    def ready(self) -> bool:
        return bool(self._history)

    @property
    def state(self) -> PoseState:
        if not self._history:
            raise RuntimeError("predictor has no pose state")
        return PoseState(
            values=clone_pose_record(self._history[-1]),
            staleness=dict(self._staleness),
            observed={part: self._staleness[part] == 0 for part in PARTS},
        )

    def initialize(self, values: Mapping[str, torch.Tensor]) -> PoseState:
        validate_pose_record(values)
        self._history.clear()
        self._history.append(clone_pose_record(values))
        self._staleness = {part: 0 for part in PARTS}
        return self.state

    def predict(self) -> TemporalPrediction:
        if not self._history:
            raise RuntimeError("initialize the predictor before predict()")
        latest = self._history[-1]
        if len(self._history) == 1:
            predicted = clone_pose_record(latest)
            acceleration = {key: value.new_zeros(()) for key, value in latest.items()}
        else:
            previous = self._history[-2]
            predicted = {
                key: latest[key] + self.config.velocity_damping * (latest[key] - previous[key])
                for key in latest
            }
            acceleration = {
                key: (latest[key] - previous[key]).abs().mean() for key in latest
            }

        uncertainty = []
        for part in PARTS:
            keys = PART_PARAMETER_KEYS[part]
            motion = torch.stack([acceleration[key] for key in keys]).mean()
            growth = self.config.uncertainty_growth * self._staleness[part]
            uncertainty.append((motion + growth).clamp(0.0, 1.0))
        result = TemporalPrediction(values=predicted, uncertainty=torch.stack(uncertainty))
        result.validate()
        return result

    def commit(
        self,
        prediction: TemporalPrediction,
        observed: Mapping[str, torch.Tensor] | None,
        update_parts: set[Part] | frozenset[Part],
    ) -> PoseState:
        prediction.validate()
        values = clone_pose_record(prediction.values)
        if observed is not None:
            validate_pose_record(observed)
            for part in update_parts:
                for key in PART_PARAMETER_KEYS[part]:
                    values[key] = observed[key].detach().clone()

        for part in PARTS:
            self._staleness[part] = 0 if part in update_parts else self._staleness[part] + 1
        self._history.append(values)
        return PoseState(
            values=clone_pose_record(values),
            staleness=dict(self._staleness),
            observed={part: part in update_parts for part in PARTS},
        )

    def override_parts(
        self,
        values: Mapping[str, torch.Tensor],
        parts: set[Part] | frozenset[Part],
        *,
        observed: bool = False,
        reset_staleness: bool = True,
    ) -> PoseState:
        """Replace selected parts in the latest state.

        Live occlusion handling uses this to put hidden parts into a neutral
        pose after the normal prediction/update merge, without reinitializing
        the whole temporal history.
        """
        if not self._history:
            raise RuntimeError("initialize the predictor before override_parts()")
        validate_pose_record(values)
        latest = clone_pose_record(self._history[-1])
        for part in parts:
            for key in PART_PARAMETER_KEYS[part]:
                latest[key] = values[key].detach().clone()
            if reset_staleness:
                self._staleness[part] = 0
        self._history[-1] = latest
        return PoseState(
            values=clone_pose_record(latest),
            staleness=dict(self._staleness),
            observed={part: observed and part in parts for part in PARTS},
        )
