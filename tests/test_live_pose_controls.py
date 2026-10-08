import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import roma

from main.live_pear_guava import (
    FLAME_KEYS,
    TargetBuilder,
    draw_pose_skeleton,
    parse_args,
    read_source_frame_interactive,
    matrix_to_axis_angle,
)


IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
FLAME_SIZES = {
    "expression_params": 50,
    "jaw_params": 3,
    "pose_params": 3,
    "eye_pose_params": 6,
    "eyelid_params": 2,
}


def pose_params():
    values = {
        "global_pose": IDENTITY_6D.repeat(1, 1),
        "body_pose": IDENTITY_6D.repeat(21, 1),
        "left_hand_pose": IDENTITY_6D.repeat(15, 1),
        "right_hand_pose": IDENTITY_6D.repeat(15, 1),
        "exp": torch.ones(50),
    }
    values.update({key: torch.ones(size) for key, size in FLAME_SIZES.items()})
    return values


def source_identity():
    return {
        "shape": torch.zeros(1, 10),
        "joints_offset": torch.zeros(1, 1),
        "head_scale": torch.ones(1, 1),
        "hand_scale": torch.ones(1, 1),
        "flame_shape": torch.zeros(1, 100),
        "source_exp": torch.full((1, 50), 2.0),
        "source_body_pose": torch.full((1, 21, 3), 0.25),
        "source_flame": {
            key: torch.full((1, size), 3.0) for key, size in FLAME_SIZES.items()
        },
    }


def args(**overrides):
    values = {
        "smooth": False,
        "freeze_face": False,
        "face_mode": "live",
        "rest_lower_body": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LivePoseControlTests(unittest.TestCase):
    def test_interactive_source_capture_waits_after_s_key(self):
        class FakeCapture:
            def __init__(self):
                self.index = 0

            def read(self):
                self.index += 1
                return True, np.full((24, 32, 3), self.index, dtype=np.uint8)

        capture_args = SimpleNamespace(
            source_capture_skip_frames=0,
            source_capture_delay=20.0,
            source_capture_max_attempts=3,
        )
        with mock.patch("main.live_pear_guava.cv2.namedWindow"), \
             mock.patch("main.live_pear_guava.cv2.imshow"), \
             mock.patch("main.live_pear_guava.cv2.destroyWindow"), \
             mock.patch("main.live_pear_guava.cv2.waitKey", side_effect=[ord("s"), 0]), \
             mock.patch("main.live_pear_guava.time.monotonic", side_effect=[100.0, 119.0, 121.0]):
            captured = read_source_frame_interactive(FakeCapture(), "0", capture_args)

        self.assertTrue(np.all(captured == 3))

    def test_live_yaml_and_cli_override(self):
        config = Path(__file__).resolve().parents[1] / "configs/live_pear_student.yaml"
        runtime = parse_args(["--live_config", str(config)])
        self.assertEqual(runtime.input, "0")
        self.assertFalse(runtime.smooth)
        self.assertEqual(runtime.avatar_view, "full")
        self.assertEqual(runtime.input_framing, "centered")
        self.assertEqual(runtime.compile_targets, ["pear", "refiner"])
        self.assertIsInstance(runtime.student_ckpt, Path)
        self.assertIsNone(runtime.record_dir)
        self.assertEqual(runtime.source_capture_delay, 0.0)
        override = parse_args(["--live_config", str(config), "--smooth",
                               "--input_framing", "whole", "--compile_targets",
                               "--record_dir", "outputs/live_test", "--record_fps", "24",
                               "--source_capture_delay", "20"])
        self.assertTrue(override.smooth)
        self.assertEqual(override.input_framing, "whole")
        self.assertEqual(override.compile_targets, [])
        self.assertEqual(override.record_dir, Path("outputs/live_test"))
        self.assertEqual(override.record_fps, 24.0)
        self.assertEqual(override.source_capture_delay, 20.0)

    def test_live_yaml_rejects_unknown_options(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "live.yaml"
            config.write_text("auto_person_crop: true\n")
            with self.assertRaises(SystemExit):
                parse_args(["--live_config", str(config)])

    def test_rotation_conversion_preserves_half_turns(self):
        angles = torch.tensor([[torch.pi, 0., 0.], [0., torch.pi - 1e-5, 0.], [0., 0., 0.]])
        rotations = roma.rotvec_to_rotmat(angles)
        restored = roma.rotvec_to_rotmat(matrix_to_axis_angle(rotations))
        torch.testing.assert_close(restored, rotations, atol=1e-6, rtol=1e-6)

    def test_skeleton_uses_rasterizer_vertical_direction(self):
        joints = torch.zeros(1, 22, 3)
        joints[0, :, 1] = 0.5
        camera = {"full_proj_transform": torch.eye(4).unsqueeze(0)}
        panel = draw_pose_skeleton(joints, camera, 256, 256)
        self.assertGreater(np.count_nonzero(panel[185:200, 120:140]), 0)
        self.assertEqual(np.count_nonzero(panel[55:75, 120:140]), 0)


    def test_source_delta_preserves_source_baseline_and_live_change(self):
        builder = TargetBuilder(source_identity(), args(face_mode="source-delta"))
        first_params = pose_params()
        first = builder(first_params, 0.0)
        second_params = {key: value.clone() for key, value in first_params.items()}
        second_params["jaw_params"] += 0.5
        second = builder(second_params, 1.0)

        self.assertTrue(torch.equal(first["flame_coeffs"]["jaw_params"], torch.full((1, 3), 3.0)))
        self.assertTrue(torch.equal(second["flame_coeffs"]["jaw_params"], torch.full((1, 3), 3.5)))

    def test_lower_body_uses_source_pose_without_freezing_spine(self):
        builder = TargetBuilder(source_identity(), args(rest_lower_body=True))
        result = builder(pose_params(), 0.0)["smplx_coeffs"]["body_pose"]

        lower = (0, 1, 3, 4, 6, 7, 9, 10)
        self.assertTrue(torch.equal(result[:, lower], torch.full((1, len(lower), 3), 0.25)))
        self.assertTrue(torch.equal(result[:, 2], torch.zeros(1, 3)))

    def test_skeleton_panel_draws_projected_joints(self):
        joints = torch.zeros(1, 22, 3)
        joints[0, :, 0] = torch.linspace(-0.5, 0.5, 22)
        joints[0, :, 1] = torch.linspace(-0.5, 0.5, 22)
        camera = {"full_proj_transform": torch.eye(4).unsqueeze(0)}

        panel = draw_pose_skeleton(joints, camera, 256, 256)

        self.assertEqual(panel.shape, (256, 256, 3))
        self.assertGreater(np.count_nonzero(panel), 0)


if __name__ == "__main__":
    unittest.main()
