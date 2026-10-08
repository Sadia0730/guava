from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.pear_geometry import (
    bbox_from_keypoints,
    invert_affine,
    pa_mpjpe_mm,
    pear_camera_project_normalized,
    similarity_align,
    square_crop_transform,
    transform_points,
)


class PearGeometryTests(unittest.TestCase):
    def test_crop_round_trip(self) -> None:
        transform = square_crop_transform(np.asarray([30.0, 50.0, 190.0, 250.0]), 1.25)
        points = np.asarray([[30.0, 50.0], [90.0, 120.0], [190.0, 250.0]], np.float32)
        crop = transform_points(points, transform)
        recovered = transform_points(crop, invert_affine(transform))
        np.testing.assert_allclose(recovered, points, atol=2e-5)

    def test_bbox_ignores_invisible_points(self) -> None:
        points = np.asarray([[10.0, 20.0], [50.0, 70.0], [1000.0, 1000.0]])
        confidence = np.asarray([1.0, 0.5, 0.0])
        np.testing.assert_array_equal(bbox_from_keypoints(points, confidence), [10.0, 20.0, 50.0, 70.0])

    def test_camera_translation_changes_projection(self) -> None:
        joints = torch.tensor([[[0.1, 0.2, 0.1]]]).expand(1, 22, 3).clone()
        teacher = torch.eye(4).unsqueeze(0)
        teacher[:, 2, 3] = 3.0
        student = teacher.clone()
        student[:, 0, 3] += 0.1
        teacher_xy = pear_camera_project_normalized(teacher, joints)
        student_xy = pear_camera_project_normalized(student, joints)
        self.assertGreater(float((student_xy - teacher_xy).abs().max()), 0.1)

    def test_similarity_alignment_removes_similarity_transform(self) -> None:
        generator = torch.Generator().manual_seed(7)
        target = torch.randn(4, 14, 3, generator=generator)
        angle = torch.tensor(0.6)
        rotation = torch.tensor(
            [[torch.cos(angle), -torch.sin(angle), 0.0],
             [torch.sin(angle), torch.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        prediction = 1.7 * torch.matmul(target, rotation) + torch.tensor([2.0, -1.0, 0.4])
        aligned = similarity_align(prediction, target)
        torch.testing.assert_close(aligned, target, atol=2e-5, rtol=2e-5)
        self.assertLess(float(pa_mpjpe_mm(prediction, target).max()), 0.03)


if __name__ == "__main__":
    unittest.main()
