from __future__ import annotations

import ast
import math
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


def _pear_crop_functions():
    """Load PEAR's original crop helpers without importing ultralytics."""
    source = (ROOT / "third_party/PEAR/inference_images.py").read_text()
    wanted = {"get_bbox", "sanitize_bbox", "process_bbox", "rotate_2d", "gen_trans_from_patch_cv"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {"np": np, "cv2": ev.cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pear_crop", "exec"), namespace)
    return namespace


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def test_j14_hips_are_the_h36m_hips():
    assert len(ev.H36M_TO_J14) == 14
    assert {ev.H36M_TO_J14[i] for i in ev.J14_HIPS} == {1, 4}


def test_similarity_align_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(1, 14, 3))
    rot = _rotation([0.3, 1.0, -0.2], 1.1)
    source = (target - 0.5) @ rot.T * 0.7
    aligned = ev.similarity_align(torch.tensor(source), torch.tensor(target))
    assert torch.allclose(aligned, torch.tensor(target), atol=1e-9)


def test_similarity_align_never_reflects():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(1, 14, 3))
    mirrored = target * np.array([-1.0, 1.0, 1.0])
    aligned = ev.similarity_align(torch.tensor(mirrored), torch.tensor(target))
    assert (aligned - torch.tensor(target)).norm(dim=-1).mean() > 0.1


def _smpl_vertices(pose_seed: int):
    smpl = ev.build_smpl(ev.DEFAULT_SMPL_DIR, "neutral", "cpu")
    gen = torch.Generator().manual_seed(pose_seed)
    body = 0.3 * torch.randn(1, 69, generator=gen)
    with torch.no_grad():
        verts = smpl(global_orient=torch.zeros(1, 3), body_pose=body, betas=torch.zeros(1, 10)).vertices
    return smpl, verts


def test_metrics_are_zero_for_identical_meshes():
    smpl, verts = _smpl_vertices(0)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float()
    metrics = ev.frame_metrics(verts.clone(), verts, h36m, smpl.J_regressor.float())
    for name, value in metrics.items():
        assert float(value) < 1e-3, (name, float(value))


def test_translation_invariance_within_regressor_tolerance():
    # The standard H36M regressor's rows sum to 0.9996-1.0, so a shifted mesh moves its
    # regressed joints by slightly less than the shift. Both meshes are built at the origin
    # in the evaluator, as in SPIN; this bounds the effect for an extreme 3 m shift.
    smpl, verts = _smpl_vertices(0)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float()
    shift = torch.tensor([0.4, -0.2, 3.0])
    metrics = ev.frame_metrics(verts + shift, verts, h36m, smpl.J_regressor.float())
    bound_mm = 2 * float((1 - h36m.sum(1)).abs().max()) * float(shift.norm()) * 1000.0
    for name in ("mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "mpjpe_h36m_pelvis_mm"):
        assert float(metrics[name]) < bound_mm, (name, float(metrics[name]), bound_mm)
    for name in ("diag_lower_body_mm", "diag_upper_body_mm", "diag_hands_mm"):
        assert float(metrics[name]) < 1e-3, name


def test_rotation_raises_mpjpe_but_not_pa_mpjpe():
    smpl, verts = _smpl_vertices(1)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float()
    rot = torch.tensor(_rotation([0, 1, 0], math.radians(30)), dtype=torch.float32)
    metrics = ev.frame_metrics(verts @ rot.T, verts, h36m, smpl.J_regressor.float())
    assert float(metrics["mpjpe_mm"]) > 50.0
    assert float(metrics["pa_mpjpe_mm"]) < 1e-3


def test_crop_helpers_match_pear():
    pear = _pear_crop_functions()
    rng = np.random.default_rng(2)
    for _ in range(20):
        kp = np.column_stack([rng.uniform(0, 1920, 18), rng.uniform(0, 1080, 18), rng.integers(0, 2, 18)])
        kp[:2, 2] = 1
        valid = (kp[:, 2] > 0).astype(np.int64)
        ours = ev.process_bbox(ev.get_bbox(kp[:, :2], valid, 1.2), 1920, 1080, (256, 256), 1.25)
        theirs = pear["process_bbox"](pear["get_bbox"](kp[:, :2], valid, 1.2), 1920, 1080, [256, 256], 1.25)
        assert np.allclose(ours, theirs)
        cx, cy = ours[0] + 0.5 * ours[2], ours[1] + 0.5 * ours[3]
        theirs_affine = pear["gen_trans_from_patch_cv"](cx, cy, ours[2], ours[3], 256, 256, 1.0, 0.0)
        assert np.allclose(ev.box_to_affine(ours), theirs_affine, atol=1e-4)


def test_training_square_box_matches_data_preparation():
    box = ev.training_square_box([100.0, 200.0, 300.0, 700.0])
    side = 500.0 * 1.25
    assert np.allclose(box, [200.0 - side / 2, 450.0 - side / 2, side, side])


def test_smplx_to_smpl_mapping_is_a_convex_correspondence():
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to_dense()
    assert tuple(mapping.shape) == (6890, 10475)
    assert torch.allclose(mapping.sum(1), torch.ones(6890), atol=1e-5)
