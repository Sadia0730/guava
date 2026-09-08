from __future__ import annotations

import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from tools.compare_teacher_student import sequence_metrics, shared_preprocess
from tools.domain_investigation_utils import (
    deterministic_random_frames,
    deterministic_sequential_clips,
    embed_in_canvas,
    load_manifest_records,
    remap_directory,
    square_person_crop,
    validate_manifest_files,
)
from tools.pear_debug_utils import (
    as_primitive,
    compare_state_dicts,
    find_state_dict,
    prefix_coverage,
)


class CheckpointAuditUtilityTests(unittest.TestCase):
    def test_checkpoint_source_is_unambiguous(self):
        state = OrderedDict((("backbone.weight", torch.ones(2, 3)),))
        source, selected = find_state_dict({"step": 1, "student": state})
        self.assertEqual(source, "student")
        self.assertIs(selected, state)

    def test_ambiguous_model_states_are_rejected(self):
        state = {"weight": torch.ones(1)}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            find_state_dict({"student": state, "ema": state})

    def test_shape_mismatch_is_reported(self):
        report = compare_state_dicts(
            {"backbone.weight": torch.ones(2, 3)},
            {"backbone.weight": torch.ones(3, 2), "head.weight": torch.ones(1)},
        )
        self.assertEqual(report["matched_tensor_count"], 0)
        self.assertEqual(report["missing_keys"], ["head.weight"])
        self.assertEqual(report["shape_mismatched_keys"][0]["key"], "backbone.weight")

    def test_prefix_coverage_requires_every_shape(self):
        model = {"head.a": torch.ones(2), "head.b": torch.ones(3)}
        checkpoint = {"head.a": torch.ones(2), "head.b": torch.ones(4)}
        coverage = prefix_coverage(checkpoint, model, "head")
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["matching_numel"], 2)

    def test_paths_are_removed_from_sanitized_metadata(self):
        converted = as_primitive({"output": Path("/tmp/run"), "items": (1, Path("a"))})
        self.assertEqual(converted, {"output": "/tmp/run", "items": [1, "a"]})


class IdenticalPreprocessingTests(unittest.TestCase):
    def test_shared_preprocess_shapes_ranges_and_determinism(self):
        frame = np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3)
        first = shared_preprocess(frame, crop_scale=1.5, device="cpu")
        second = shared_preprocess(frame, crop_scale=1.5, device="cpu")
        self.assertEqual(tuple(first["model_tensor"].shape), (1, 3, 256, 256))
        self.assertEqual(tuple(first["normalized_tensor"].shape), (1, 3, 256, 192))
        self.assertGreaterEqual(float(first["model_tensor"].min()), 0.0)
        self.assertLessEqual(float(first["model_tensor"].max()), 1.0)
        self.assertTrue(torch.equal(first["normalized_tensor"], second["normalized_tensor"]))

    def test_motion_retention_is_reported(self):
        teacher = [torch.tensor([[0.0]]), torch.tensor([[2.0]]), torch.tensor([[4.0]])]
        student = [torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[2.0]])]
        report = sequence_metrics(teacher, student)
        self.assertAlmostEqual(report["teacher_motion_amplitude_retained_percent"], 50.0)
        self.assertAlmostEqual(report["teacher_frame_delta_mean"], 2.0)
        self.assertAlmostEqual(report["student_frame_delta_mean"], 1.0)


class DomainInvestigationUtilityTests(unittest.TestCase):
    def test_absolute_manifest_root_is_remapped_without_editing_manifest(self):
        path = remap_directory(
            "/old/processed/frames/clip_a",
            Path("/tmp/val.jsonl"),
            Path("/old/processed"),
            Path("/new/processed"),
        )
        self.assertEqual(path, Path("/new/processed/frames/clip_a"))

    def test_manifest_validation_and_sampling_are_deterministic(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            for index in range(8):
                (frames / f"{index:06d}.jpg").write_bytes(b"frame")
            manifest = root / "val.jsonl"
            manifest.write_text(
                '{"source":"UBody","sequence":"clip","directory":"frames",'
                '"num_frames":8,"fps":30,"frame_step":1}\n',
                encoding="utf-8",
            )
            records = load_manifest_records(manifest)
            self.assertTrue(validate_manifest_files(records)["valid"])
            first = deterministic_random_frames(records, 4, seed=17)
            second = deterministic_random_frames(records, 4, seed=17)
            self.assertEqual(first, second)
            clips = deterministic_sequential_clips(records, 2, 3, 2, seed=23)
            self.assertEqual(len(clips), 2)
            self.assertTrue(all(len(clip) == 3 for clip in clips))

    def test_crop_and_wide_canvas_shapes_are_explicit(self):
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        crop = square_person_crop(frame, (160, 40, 480, 440), crop_scale=1.2)
        canvas = embed_in_canvas(crop, (640, 480), scale=0.5)
        self.assertEqual(crop.shape, (256, 256, 3))
        self.assertEqual(canvas.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
