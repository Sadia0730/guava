from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from distillation.teacher_cache import TeacherCacheWriter, TeacherOutputCache


class TeacherCacheTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
