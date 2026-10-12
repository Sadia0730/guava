"""Tests for tools/eval_pck_hmr2.py (no model, no GPU)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_pck_hmr2", ROOT / "tools" / "eval_pck_hmr2.py")
pk = importlib.util.module_from_spec(spec)
sys.modules["eval_pck_hmr2"] = pk
spec.loader.exec_module(pk)
HSMR_PCK = Path("/raid/ubx858/datasets/eval_assets/reference_code/HSMR/lib/evaluation/hmr2_utils/pck_accuracy.py")


def _data(seed=0, n=50, k=44):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0, 500, (n, k, 2))
    conf = (rng.uniform(0, 1, (n, k)) > 0.3).astype(float)
    box = rng.uniform(80, 300, n)
    return gt, conf, box


def test_gt_as_prediction_is_perfect():
    gt, conf, box = _data()
    res = pk.pck(gt.copy(), gt, conf, box)
    assert res[0.05]["pck"] == 1.0 and res[0.1]["pck"] == 1.0


def test_known_offset():
    gt, conf, box = _data(1)
    pred = gt.copy()
    pred[..., 0] += 0.07 * box[:, None]          # 7% of the box: wrong at 0.05, right at 0.1
    res = pk.pck(pred, gt, conf, box)
    assert res[0.05]["pck"] == 0.0 and res[0.1]["pck"] == 1.0


def test_invisible_keypoints_ignored():
    gt, conf, box = _data(2)
    pred = gt.copy()
    pred[conf <= 0.5] += 1e4                    # huge errors where GT is not visible
    assert pk.pck(pred, gt, conf, box)[0.05]["pck"] == 1.0


@pytest.mark.skipif(not HSMR_PCK.exists(), reason="HSMR reference code not on disk")
def test_matches_hsmr_reference_implementation():
    spec2 = importlib.util.spec_from_file_location("hsmr_pck", HSMR_PCK)
    ref = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ref)
    gt, conf, box = _data(3)
    pred = gt + np.random.default_rng(4).normal(0, 15, gt.shape)
    norm = np.stack([box, box], 1)
    for thr in (0.05, 0.1):
        a, b, c = ref.keypoint_pck_accuracy(pred, gt, conf > 0.5, thr, norm.copy())
        a2, b2, c2 = pk.keypoint_pck_accuracy(pred, gt, conf > 0.5, thr, norm.copy())
        assert np.allclose(a, a2) and abs(b - b2) < 1e-12 and c == c2


def test_openpose_mapping_matches_hmr2_constants():
    # HMR2.0 / SPIN: OpenPose order from the 45 smplx-SMPL joints
    assert pk.OPENPOSE_FROM_SMPL45 == [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7,
                                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
