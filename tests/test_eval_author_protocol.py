"""Tests for tools/eval_author_protocol.py: each ported function against HSMR's own code (no model, no GPU)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HSMR = Path("/raid/ubx858/datasets/eval_assets/reference_code/HSMR")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ap = _load("eval_author_protocol", ROOT / "tools" / "eval_author_protocol.py")


def test_similarity_transform_matches_hsmr():
    ref = _load("hsmr_hmr2_utils", HSMR / "lib/evaluation/hmr2_utils/__init__.py")
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(8, 14, 3, generator=g), torch.randn(8, 14, 3, generator=g)        # float32, as HMR2 runs it
    assert torch.allclose(ap.compute_similarity_transform(a, b), ref.compute_similarity_transform(a, b), atol=1e-5)
    assert torch.allclose(ap.compute_similarity_transform(a.double(), b.double()).float(), ref.compute_similarity_transform(a, b), atol=1e-4)


def test_gt_as_prediction_scores_zero():
    g = torch.randn(5, 14, 3, dtype=torch.float64)
    m, r = ap.eval_pose_hmr2(g.clone(), g)
    assert m.abs().max() < 1e-9 and r.abs().max() < 1e-6


def test_expand_to_aspect_ratio_matches_hsmr():
    def ref(input_shape, target_aspect_ratio=(192, 256)):   # HSMR utils.expand_to_aspect_ratio, verbatim logic
        w, h = input_shape
        w_t, h_t = target_aspect_ratio
        if h / w < h_t / w_t:
            return np.array([w, max(w * h_t / w_t, h)])
        return np.array([max(h * w_t / h_t, w), h])
    for wh in ([300.0, 372.5], [34.85, 73.4], [400.0, 100.0], [100.0, 400.0]):
        assert np.allclose(ap.expand_to_aspect_ratio(np.array(wh)), ref(wh))


def test_crop_affine_matches_gen_trans_from_patch_cv():
    import cv2

    def rotate_2d(pt, rad):
        sn, cs = np.sin(rad), np.cos(rad)
        return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs], dtype=np.float32)

    def gen_trans(c_x, c_y, w, h, dw, dh):          # HSMR utils.gen_trans_from_patch_cv with scale=1, rot=0
        src = np.float32([[c_x, c_y], [c_x, c_y + h * 0.5], [c_x + w * 0.5, c_y]])
        dst = np.float32([[dw * 0.5, dh * 0.5], [dw * 0.5, dh], [dw, dh * 0.5]])
        return cv2.getAffineTransform(src, dst)
    ev = _load("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    for cx, cy, s in ((223.6, 392.3, 496.7), (509.3, 220.2, 97.9)):
        assert np.allclose(ev.box_to_affine([cx - s / 2, cy - s / 2, s, s]), gen_trans(cx, cy, s, s, 256, 256), atol=1e-4)


def test_reference_tables_parse():
    if not ap.REFERENCE.exists():
        return
    t = ap.reference_tables()
    assert len(t["JOINT_NAMES"]) == 44 and len(t["smpl_to_h36m"]) == 44 and len(t["smpl_to_openpose"]) == 25
