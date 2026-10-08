from __future__ import annotations

import unittest

import torch

from tools.train_pear_student_3dpw import pcgrad_merge


class ThreeDPWTrainingTests(unittest.TestCase):
    def test_pcgrad_projects_conflicting_gradients(self) -> None:
        first = (torch.tensor([1.0, 0.0]),)
        second = (torch.tensor([-1.0, 1.0]),)
        merged, cosine = pcgrad_merge(first, second)

        self.assertLess(float(cosine), 0.0)
        torch.testing.assert_close(merged[0], torch.tensor([0.5, 1.5]))

    def test_pcgrad_keeps_aligned_gradient_sum(self) -> None:
        first = (torch.tensor([1.0, 0.0]),)
        second = (torch.tensor([2.0, 0.0]),)
        merged, cosine = pcgrad_merge(first, second)

        self.assertGreater(float(cosine), 0.99)
        torch.testing.assert_close(merged[0], torch.tensor([3.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
