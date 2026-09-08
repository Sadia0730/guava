from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from avatarbudget.cache import TeacherCacheWriter, TeacherOutputCache
from avatarbudget.config import AvatarBudgetConfig, load_config
from avatarbudget.contracts import (
    PARAMETER_SHAPES,
    PARTS,
    Part,
    PoseState,
    ScoutOutput,
    validate_pose_record,
)
from avatarbudget.impact import CounterfactualRenderLabeler, counterfactual_render_damage
from avatarbudget.profiler import StageProfiler
from avatarbudget.rendering import select_gaussians
from avatarbudget.router import BudgetRouter, RenderImpactRouterNet
from avatarbudget.scheduler import HardFrameScheduler
from avatarbudget.scout import CheapScout
from avatarbudget.temporal import ConstantVelocityPredictor


ROOT = Path(__file__).resolve().parents[1]


def pose_record(offset=0.0):
    return {
        key: torch.full(shape, float(offset), dtype=torch.float32)
        for key, shape in PARAMETER_SHAPES.items()
    }


def scout_output(value=0.0):
    return ScoutOutput(
        motion_score=torch.full((1, 4), value),
        visibility=torch.ones(1, 4),
        rois=torch.zeros(1, 4, 4),
        confidence=torch.ones(1, 4),
        appearance_delta=torch.full((1, 4), value),
        features=torch.zeros(1, 4, 6),
    )


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class ContractTests(unittest.TestCase):
    def test_pose_shapes_are_explicit_and_validated(self):
        values = pose_record()
        validate_pose_record(values)
        self.assertEqual(tuple(values["body_pose"].shape), (21, 6))
        self.assertEqual(tuple(values["left_hand_pose"].shape), (15, 6))
        self.assertEqual(tuple(values["expression_params"].shape), (50,))

    def test_wrong_pose_shape_is_rejected(self):
        values = pose_record()
        values["jaw_params"] = torch.zeros(4)
        with self.assertRaisesRegex(ValueError, "jaw_params"):
            validate_pose_record(values)


class ModuleInterfaceTests(unittest.TestCase):
    def test_config_loads_50fps_deadline(self):
        config = load_config(ROOT / "configs/avatarbudget_rtx3080_laptop.yaml")
        self.assertEqual(config.scheduler.target_fps, 50.0)
        self.assertEqual(config.scheduler.deadline_ms, 20.0)

    def test_scout_shapes_and_motion(self):
        scout = CheapScout(AvatarBudgetConfig().scout)
        first = scout(torch.zeros(2, 3, 240, 320))
        second = scout(torch.ones(2, 3, 240, 320))
        first.validate()
        second.validate()
        self.assertEqual(tuple(second.motion_score.shape), (2, 4))
        self.assertEqual(tuple(second.rois.shape), (2, 4, 4))
        self.assertEqual(tuple(second.features.shape), (2, 4, 6))
        self.assertTrue(torch.all(second.motion_score > 0.9))

    def test_temporal_prediction_and_conditional_merge(self):
        config = AvatarBudgetConfig().temporal
        predictor = ConstantVelocityPredictor(config)
        predictor.initialize(pose_record(0.0))
        prediction = predictor.predict()
        predictor.commit(prediction, pose_record(1.0), frozenset(PARTS))
        extrapolated = predictor.predict()
        self.assertTrue(torch.allclose(extrapolated.values["body_pose"], torch.full((21, 6), 1.85)))

        observed = pose_record(5.0)
        state = predictor.commit(extrapolated, observed, frozenset({Part.FACE}))
        self.assertTrue(torch.all(state.values["expression_params"] == 5.0))
        self.assertTrue(torch.all(state.values["body_pose"] == 1.85))
        self.assertEqual(state.staleness[Part.FACE], 0)
        self.assertEqual(state.staleness[Part.BODY], 1)

    def test_temporal_override_parts_for_occlusion_rest(self):
        predictor = ConstantVelocityPredictor(AvatarBudgetConfig().temporal)
        predictor.initialize(pose_record(0.0))
        prediction = predictor.predict()
        predictor.commit(prediction, pose_record(1.0), frozenset({Part.BODY}))
        rest = pose_record(0.0)
        state = predictor.override_parts(rest, frozenset({Part.LEFT_HAND}), reset_staleness=True)
        self.assertTrue(torch.all(state.values["left_hand_pose"] == 0.0))
        self.assertTrue(torch.all(state.values["body_pose"] == 1.0))
        self.assertEqual(state.staleness[Part.LEFT_HAND], 0)

    def test_router_forces_stale_parts_and_obeys_feasible_budget(self):
        config = AvatarBudgetConfig().router
        state = PoseState(
            pose_record(),
            staleness={part: config.max_staleness for part in PARTS},
            observed={part: False for part in PARTS},
        )
        prediction_uncertainty = torch.zeros(4)
        decision = BudgetRouter(config).route(
            scout_output(), prediction_uncertainty, state, budget_ms=20.0
        )
        self.assertEqual(decision.update_parts, frozenset(PARTS))
        self.assertTrue(decision.deadline_feasible)
        self.assertLessEqual(decision.estimated_ms, 20.0)

    def test_router_network_shapes(self):
        output = RenderImpactRouterNet()(torch.zeros(3, 4, 7))
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertTrue(torch.all(output >= 0))

    def test_progressive_gaussians_keep_all_body_points(self):
        assets = {
            "xyz": torch.arange(60).reshape(1, 20, 3).float(),
            "opacity": torch.ones(1, 20, 1),
            "smplx_xyz_deform": torch.zeros(1, 8, 3),
            "sh_degree": 0,
        }
        selected = select_gaussians(assets, 0.5)
        self.assertEqual(tuple(selected["xyz"].shape), (1, 14, 3))
        self.assertTrue(torch.equal(selected["xyz"][:, :8], assets["xyz"][:, :8]))


class StorageAndMetricTests(unittest.TestCase):
    def test_teacher_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = [root / "a.jpg", root / "b.jpg"]
            output = {
                "output": {"pd_cam": torch.arange(32).reshape(2, 4, 4).float()},
                "features": torch.ones(2, 8, 2, 2),
            }
            writer = TeacherCacheWriter(root / "cache", metadata={"features": True})
            writer.add_shard(image_paths, output)
            writer.close()
            cache = TeacherOutputCache(root / "cache")
            batch = cache.batch(reversed(image_paths), "cpu")
            self.assertEqual(tuple(batch["output"]["pd_cam"].shape), (2, 4, 4))
            self.assertEqual(float(batch["output"]["pd_cam"][0, 0, 0]), 16.0)

    def test_render_damage_and_labeler_shapes(self):
        reference = torch.ones(2, 3, 8, 8)
        skipped = torch.zeros_like(reference)
        damage = counterfactual_render_damage(reference, skipped)
        self.assertEqual(tuple(damage["total"].shape), (2,))

        def render(values):
            scalar = values["body_pose"].mean().clamp(0, 1)
            return scalar.expand(3, 8, 8)

        labels = CounterfactualRenderLabeler(render)(pose_record(1.0), pose_record(0.0))
        self.assertEqual(tuple(labels["total"].shape), (1, 4))

    def test_profiler_reports_mean_p95_p99(self):
        profiler = StageProfiler(window=10, warmup_frames=0)
        for value in (1.0, 2.0, 3.0, 4.0):
            profiler.add("stage", value)
        summary = profiler.as_dict()["stage"]
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertGreaterEqual(summary["p99_ms"], summary["p95_ms"])

    def test_scheduler_paces_and_reports_deadline_miss(self):
        clock = FakeClock()
        scheduler = HardFrameScheduler(
            AvatarBudgetConfig().scheduler, clock=clock, sleeper=clock.sleep
        )
        scheduler.begin_frame()
        clock.value += 0.005
        result = scheduler.finish_frame()
        self.assertAlmostEqual(result.work_ms, 5.0)
        self.assertAlmostEqual(result.frame_ms, 20.0)
        self.assertFalse(result.deadline_missed)

        scheduler.begin_frame()
        clock.value += 0.025
        result = scheduler.finish_frame()
        self.assertTrue(result.deadline_missed)


if __name__ == "__main__":
    unittest.main()
