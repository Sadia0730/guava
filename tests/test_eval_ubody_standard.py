"""Tests for tools/eval_ubody_standard.py metric code (no model, no GPU)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ub = _load("eval_ubody_standard", ROOT / "tools" / "eval_ubody_standard.py")
ehf = _load("guava_eval_ehf_standard", ROOT / "tools" / "eval_ehf_standard.py")
P = ehf.Protocol()
CAM = {"focal": [1500.0, 1500.0], "princpt": [960.0, 540.0]}
WH = [1920, 1080]


def _mesh(seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.3, (10475, 3))
    m[:, 2] += 4.0                          # in front of the camera, mostly inside the image
    return m


def test_gt_as_prediction_scores_zero():
    m = _mesh()
    r = ub.evaluate_one(P, ehf, m.copy(), m, CAM, WH)
    assert set(r) == {"pa_mpvpe_all", "mpvpe_all", "pa_mpvpe_hand", "mpvpe_hand", "pa_mpvpe_face", "mpvpe_face",
                      "pa_mpjpe_body", "pa_mpjpe_hand"}
    assert all(abs(v) < 1e-6 for v in r.values())


def test_translation_is_removed_by_every_alignment():
    m = _mesh(1)
    r = ub.evaluate_one(P, ehf, m + np.array([0.3, -0.2, 0.5]), m, CAM, WH)
    assert all(abs(v) < 1e-3 for v in r.values())       # regressor rows sum to 1 within 1e-8


def test_nothing_in_image_gives_no_metrics():
    m = _mesh(2)
    m[:, 2] = -1.0                          # whole GT mesh behind the camera: every part is skipped
    assert ub.evaluate_one(P, ehf, m.copy(), m, CAM, WH) == {}


def test_in_image_projection():
    pts = np.array([[0.0, 0.0, 2.0], [10.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    v = ub.in_image(pts, CAM, WH)
    assert v.tolist() == [True, False, False]
