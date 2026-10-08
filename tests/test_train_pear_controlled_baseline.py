from __future__ import annotations

import unittest

import torch

from tools.train_pear_controlled_baseline import project_joints


class ControlledBaselineTests(unittest.TestCase):
    def test_projection_uses_camera_translation(self) -> None:
        joints = torch.zeros(1, 22, 3)
        first = torch.eye(4).unsqueeze(0)
        first[:, 2, 3] = 3.0
        second = first.clone()
        second[:, 0, 3] = 0.2

        first_xy = project_joints({"pd_cam": first}, joints)
        second_xy = project_joints({"pd_cam": second}, joints)

        self.assertGreater(float((first_xy - second_xy).abs().max()), 0.1)


if __name__ == "__main__":
    unittest.main()
