from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PYLIB = Path("/raid/ubx858/datasets/eval_assets/pylib")
for path in (ROOT, PYLIB):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools import eval_3dpw_standard as ev  # noqa: E402
from tools import eval_ehf_standard as eh  # noqa: E402


def test_rigid_align_port_matches_our_procrustes():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 3)), rng.normal(size=(300, 3))
    ours = ev.similarity_align(torch.tensor(a[None]), torch.tensor(b[None]))[0].numpy()
    assert np.allclose(eh.rigid_align(a, b), ours, atol=1e-12)


def test_ground_truth_scores_zero_and_shift_scores_zero_after_alignment():
    proto = eh.Protocol()
    frames = eh.load_frames()[:3]
    for frame in frames:
        gt = eh.gt_camera_mesh(frame)
        for value in proto.evaluate(gt.copy(), gt).values():
            assert value < 1e-6
        shifted = proto.evaluate(gt + np.array([0.3, -0.1, 1.0]), gt)
        for key, value in shifted.items():
            assert value < 1e-3, key


def test_hand_regressors_have_21_joints_and_vertex_sets_are_disjoint():
    proto = eh.Protocol()
    assert proto.hand_reg["left"].shape == (21, 10475)
    assert proto.hand_reg["right"].shape == (21, 10475)
    face = set(proto.face_idx.tolist())
    assert not face & set(proto.hand_idx["left_hand"]) and not face & set(proto.hand_idx["right_hand"])
