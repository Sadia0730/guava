from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import nn

from .config import RouterConfig
from .contracts import PARTS, Part, PoseState, RenderLevel, RouteDecision, ScoutOutput


class RenderImpactRouterNet(nn.Module):
    """Predict counterfactual render damage for each part.

    Input shape: ``[B, 4, 7]``. Output shape: ``[B, 4]``.
    """

    def __init__(self, feature_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (len(PARTS), 7):
            raise ValueError(f"router features must be [B, 4, 7], got {tuple(features.shape)}")
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class Candidate:
    score: float
    cost: float
    parts: frozenset[Part]
    render_level: RenderLevel


class BudgetRouter:
    """Enumerate the small part/quality action space under one frame budget."""

    def __init__(self, config: RouterConfig, model: RenderImpactRouterNet | None = None):
        self.config = config
        self.model = model

    def feature_tensor(
        self,
        scout: ScoutOutput,
        uncertainty: torch.Tensor,
        staleness: dict[Part, int],
    ) -> torch.Tensor:
        scout.validate()
        if scout.motion_score.shape[0] != 1:
            raise ValueError("online router currently expects batch size 1")
        device = scout.motion_score.device
        dtype = scout.motion_score.dtype
        stale = torch.tensor(
            [staleness[part] / max(self.config.max_staleness, 1) for part in PARTS],
            device=device,
            dtype=dtype,
        )
        impact = torch.tensor(self.config.impact_weights, device=device, dtype=dtype)
        return torch.stack(
            (
                scout.motion_score[0],
                scout.appearance_delta[0],
                uncertainty.to(device=device, dtype=dtype),
                stale,
                impact,
                scout.visibility[0],
                scout.confidence[0],
            ),
            dim=-1,
        ).unsqueeze(0)

    def risk_scores(
        self,
        scout: ScoutOutput,
        uncertainty: torch.Tensor,
        state: PoseState,
    ) -> torch.Tensor:
        features = self.feature_tensor(scout, uncertainty, state.staleness)
        if self.model is not None:
            return self.model(features)[0]
        weights = self.config.weights
        risk = (
            weights.motion * features[0, :, 0]
            + weights.appearance * features[0, :, 1]
            + weights.uncertainty * features[0, :, 2]
            + weights.staleness * features[0, :, 3]
            + weights.render_impact * features[0, :, 4]
            + weights.visibility * features[0, :, 5]
        )
        normalizer = (
            weights.motion
            + weights.appearance
            + weights.uncertainty
            + weights.staleness
            + weights.render_impact * max(self.config.impact_weights)
            + weights.visibility
        )
        return (risk / max(normalizer, 1e-6)).clamp(0.0, 1.0)

    def _cost(self, parts: frozenset[Part], level: RenderLevel) -> float:
        costs = self.config.costs
        head_cost = {
            Part.FACE: costs.face_head_ms,
            Part.LEFT_HAND: costs.left_hand_head_ms,
            Part.RIGHT_HAND: costs.right_hand_head_ms,
            Part.BODY: costs.body_head_ms,
        }
        render_cost = {
            RenderLevel.LOW: costs.render_low_ms,
            RenderLevel.MEDIUM: costs.render_medium_ms,
            RenderLevel.HIGH: costs.render_high_ms,
        }[level]
        expert = costs.expert_shared_ms if parts else 0.0
        return costs.fixed_ms + render_cost + expert + sum(head_cost[part] for part in parts)

    def route(
        self,
        scout: ScoutOutput,
        uncertainty: torch.Tensor,
        state: PoseState,
        budget_ms: float,
    ) -> RouteDecision:
        risk = self.risk_scores(scout, uncertainty, state)
        forced = frozenset(
            part for part in PARTS if state.staleness[part] >= self.config.max_staleness
        )
        eligible = tuple(
            part for index, part in enumerate(PARTS) if risk[index] >= self.config.minimum_risk
        )
        candidates: list[Candidate] = []
        for count in range(len(eligible) + 1):
            for subset in itertools.combinations(eligible, count):
                parts = frozenset(subset).union(forced)
                for level_index, level in enumerate(RenderLevel):
                    cost = self._cost(parts, level)
                    if cost > budget_ms:
                        continue
                    update_score = sum(float(risk[PARTS.index(part)]) for part in parts)
                    render_score = self.config.render_quality_weight * level_index * float(risk.max())
                    candidates.append(Candidate(update_score + render_score, cost, parts, level))

        feasible = bool(candidates)
        if feasible:
            candidate = max(candidates, key=lambda item: (item.score, -item.cost))
        else:
            parts = forced
            candidate = Candidate(0.0, self._cost(parts, RenderLevel.LOW), parts, RenderLevel.LOW)
        decision = RouteDecision(
            update_parts=candidate.parts,
            forced_parts=forced,
            risk=risk.detach(),
            render_level=candidate.render_level,
            estimated_ms=candidate.cost,
            deadline_feasible=feasible,
        )
        decision.validate()
        return decision
