"""Person-only output must preserve the original class-aware person NMS."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Session:
    def __init__(self, values):
        self.values = values

    def get_inputs(self):
        return [SimpleNamespace(name='images')]

    def run(self, _, inputs):
        assert inputs['images'].dtype == np.float32
        return [self.values.copy()]


def assert_person_only_matches_original_class_aware_nms(seed):
    detector = load('original_detector', ROOT / 'EHM-Tracker/src/modules/dwpose/onnxdet.py')
    backend = load('fast_driving_test', ROOT / 'phase3_dnt/sprint_fast_driving.py')
    fitter = object.__new__(backend.SprintFitter)
    fitter._source_detector_module = detector
    rng = np.random.default_rng(seed)
    values = np.zeros((1, 8400, 85), np.float32)
    values[0, :, :2] = rng.uniform(-.2, .8, (8400, 2))
    values[0, :, 2:4] = rng.uniform(-.5, 1.5, (8400, 2))
    if seed:
        # Include multiple classes at the same locations and scores near both
        # thresholds, plus person overlap that exercises independent NMS.
        ids = np.array([0, 1, 2, 80, 81, 82, 1100, 1101])
        values[0, ids, 4] = [.9, .9, .5, .31, .29, .8, .8, .9]
        values[0, ids, 5] = [.9, .8, .5, .99, .99, .7, .9, .9]
        values[0, ids, 8] = .99
    image = np.zeros((576, 1024, 3), np.uint8)
    original = detector.inference_detector(Session(values), image)
    person_only = fitter._person_detector(Session(values[:, :, :6]), image)
    if len(original) == 0:
        assert len(person_only) == 0
    else:
        np.testing.assert_array_equal(original, person_only)


class PersonDetectorTest(unittest.TestCase):
    def test_class_aware_person_equivalence(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                assert_person_only_matches_original_class_aware_nms(seed)


if __name__ == '__main__':
    unittest.main()
