#!/usr/bin/env python
"""Checkpoint 2, section 2: can BEDLAM2 shape/pose be represented by our body model?

Read-only. BEDLAM2 bodies use the locked-head SMPL-X (16 betas); our EHM-s body is
SMPL-X 2020 with 200 betas (head replaced by FLAME). Shape enters linearly, so the
least-squares map from locked-head betas to 2020 betas is a fixed affine map, fitted on
non-head vertices. Errors are measured on real BEDLAM2 rows, in T-pose and posed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

LABELS = Path("/raid/ubx858/datasets/bedlam2_gt/expanded_labels/bedlam2_labels_processed")
LOCKED = Path("/home/ubx858/guava/assets/SMPLX/lockedhead/models_lockedhead/smplx/SMPLX_NEUTRAL.npz")
PEAR = Path("/home/ubx858/guava/third_party/PEAR")
V2020 = PEAR / "assets" / "SMPLX" / "SMPLX_NEUTRAL_2020.npz"
OUR_BETAS = 200


def stats(x):
    x = np.asarray(x).ravel()
    return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def smplx_model(path, num_betas, num_expr):
    import smplx
    return smplx.SMPLX(model_path=str(path), ext="npz", gender="neutral", num_betas=num_betas, use_pca=False,
                       flat_hand_mean=True, num_expression_coeffs=num_expr).eval()


def posed(model, pose, betas):
    pose = torch.as_tensor(pose, dtype=torch.float32)
    n = len(pose)
    with torch.no_grad():
        out = model(global_orient=pose[:, :3], body_pose=pose[:, 3:66], jaw_pose=pose[:, 66:69],
                    leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75], left_hand_pose=pose[:, 75:120],
                    right_hand_pose=pose[:, 120:165], betas=torch.as_tensor(betas, dtype=torch.float32),
                    expression=torch.zeros(n, model.num_expression_coeffs))
    return out.vertices.numpy(), out.joints[:, :55].numpy()


def main():
    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(LABELS.glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        idx = rng.choice(len(d["imgname"]), 15, replace=False)
        rows += [(d["pose_cam"][i], d["shape"][i]) for i in idx]
    pose = np.stack([r[0] for r in rows]).astype(np.float32)
    beta_l = np.stack([r[1] for r in rows]).astype(np.float64)

    lk, v20 = np.load(LOCKED, allow_pickle=True), np.load(V2020, allow_pickle=True)
    head = np.load(PEAR / "assets" / "SMPLX" / "SMPL-X__FLAME_vertex_ids.npy")
    body = np.setdiff1d(np.arange(10475), head)
    vt_l, s_l = lk["v_template"], lk["shapedirs"][:, :, :16]
    vt_o, s_o = v20["v_template"], v20["shapedirs"][:, :, :OUR_BETAS]

    # Affine map: beta_ours = c + M beta_locked, least squares over non-head vertices.
    a = s_o[body].reshape(-1, OUR_BETAS)
    pinv = np.linalg.pinv(a)
    c = pinv @ (vt_l - vt_o)[body].reshape(-1)
    m = pinv @ s_l[body].reshape(-1, 16)
    beta_o = beta_l @ m.T + c
    naive = np.zeros((len(beta_l), OUR_BETAS)); naive[:, :16] = beta_l

    def tpose(vt, s, b):
        return vt[None] + np.einsum("vcb,nb->nvc", s, b)

    gt_t = tpose(vt_l, s_l, beta_l)
    conv_t = tpose(vt_o, s_o, beta_o)
    naive_t = tpose(vt_o, s_o, naive)
    j_l = np.einsum("jv,nvc->njc", lk["J_regressor"], gt_t)
    j_o = np.einsum("jv,nvc->njc", v20["J_regressor"], conv_t)

    report = {
        "rows_sampled": len(rows),
        "head_vertices_excluded": int(len(head)),
        "tpose_body_vertex_err_mm": {"converted": stats(np.linalg.norm(conv_t - gt_t, axis=-1)[:, body] * 1000),
                                     "naive_copy_16": stats(np.linalg.norm(naive_t - gt_t, axis=-1)[:, body] * 1000)},
        "tpose_head_vertex_err_mm_converted": stats(np.linalg.norm(conv_t - gt_t, axis=-1)[:, head] * 1000),
        "tpose_joint_err_mm_converted": {"body_22": stats(np.linalg.norm(j_o - j_l, axis=-1)[:, :22] * 1000),
                                         "hands_25_54": stats(np.linalg.norm(j_o - j_l, axis=-1)[:, 25:55] * 1000)},
        "converted_beta_abs_max": float(np.abs(beta_o).max()),
    }

    # Posed comparison with the reference smplx implementation (both flat_hand_mean=True).
    gt_model = smplx_model(LOCKED, 16, 10)
    our_model = smplx_model(V2020, OUR_BETAS, 10)
    gv, gj = posed(gt_model, pose, beta_l)
    ov, oj = posed(our_model, pose, beta_o)
    # Root-relative: our model's pelvis sits elsewhere for the same shape.
    pelvis_offset = (oj[:, 0] - gj[:, 0]) * 1000
    gv_r, ov_r = gv - gj[:, :1], ov - oj[:, :1]
    gj_r, oj_r = gj - gj[:, :1], oj - oj[:, :1]
    report["posed"] = {
        "pelvis_offset_mm": stats(np.linalg.norm(pelvis_offset, axis=-1)),
        "body_vertex_err_rootrel_mm": stats(np.linalg.norm(ov_r - gv_r, axis=-1)[:, body] * 1000),
        "body_joint_err_rootrel_mm_22": stats(np.linalg.norm(oj_r - gj_r, axis=-1)[:, 1:22] * 1000),
        "hand_joint_err_rootrel_mm": stats(np.linalg.norm(oj_r - gj_r, axis=-1)[:, 25:55] * 1000),
    }

    # Does PEAR's EHM_v2 body path equal reference smplx SMPL-X 2020 (no FLAME head)?
    sys.path.insert(0, str(PEAR))
    cwd = Path.cwd()
    os.chdir(PEAR)
    try:
        from models.modules.ehm import EHM_v2
        from pytorch3d.transforms import axis_angle_to_matrix
        ehm = EHM_v2("assets/FLAME", "assets/SMPLX").eval()
        p = torch.as_tensor(pose[:32])
        rot = lambda x, n: axis_angle_to_matrix(x.reshape(-1, n, 3))  # noqa: E731
        body_param = {"global_pose": rot(p[:, :3], 1), "body_pose": rot(p[:, 3:66], 21),
                      "left_hand_pose": rot(p[:, 75:120], 15), "right_hand_pose": rot(p[:, 120:165], 15),
                      "shape": torch.as_tensor(beta_o[:32], dtype=torch.float32), "exp": torch.zeros(32, 50),
                      "hand_scale": None, "head_scale": None}
        with torch.no_grad():
            ev = ehm(body_param, None, pose_type="rotmat")["vertices"][:, :10475].numpy()
        ref_v, _ = posed(smplx_model(V2020, OUR_BETAS, 50), pose[:32], beta_o[:32])
        report["ehm_vs_reference_smplx2020_vertex_err_mm"] = stats(np.linalg.norm(ev - ref_v, axis=-1) * 1000)
    finally:
        os.chdir(cwd)

    out = Path("/raid/ubx858/outputs/phase1_task1/checkpoint2")
    np.savez(out / "betas_locked16_to_2020_200.npz", M=m, c=c, fitted_on="non-head vertices",
             locked=str(LOCKED), target=str(V2020))
    (out / "body_model_compat.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
