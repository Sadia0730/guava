#!/usr/bin/env python
"""Representation floor of EHM-s on EHF: how close can our output mesh get to the GT at best?

Fits (a) EHM-s (SMPL-X 2020 body, 200 betas, FLAME head with shape/expression/jaw/eyes/eyelids
and head scale) and (b) plain SMPL-X 2020 (300 betas, 100 expression, jaw, eyes) to each of the
100 EHF ground-truth meshes, initialised from PEAR's predictions, and scores the fits with the
EHF protocol. (a) is the best any EHM-s predictor can do; (a) - (b) is the cost of the FLAME head.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ubx858/guava")
sys.path.insert(0, str(ROOT))
from tools import eval_3dpw_standard as ev  # noqa: E402
from tools import eval_ehf_standard as eh  # noqa: E402

OUT = Path("/raid/ubx858/outputs/phase1_task1/eval_ehf/mapping_floor")
ITERS = 2000


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    frames = eh.load_frames()
    proto = eh.Protocol()
    gt = torch.tensor(np.stack([eh.gt_camera_mesh(f) for f in frames]), dtype=torch.float32, device=device)
    os.chdir(ev.PEAR_ROOT)
    model, _ = ev.load_model("pear", ev.DEFAULT_TEACHER, "", 25, device)
    from models.modules.ehm import EHM_v2
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
    ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    os.chdir(ROOT)
    images = torch.stack([eh.crop(f)[0] for f in frames])
    with torch.no_grad():
        outs = [model(images[s:s + 25].to(device).float() / 255) for s in range(0, 100, 25)]
    bp = {k: torch.cat([o["body_param"][k] for o in outs]) for k in
          ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose", "shape", "exp", "head_scale", "hand_scale")}
    fp = {k: torch.cat([o["flame_param"][k] for o in outs]) for k in
          ("eye_pose_params", "pose_params", "jaw_params", "eyelid_params", "expression_params", "shape_params")}
    del model
    torch.cuda.empty_cache()

    def leaf(x):
        return x.detach().clone().float().requires_grad_(True)

    aa = {k: leaf(matrix_to_axis_angle(bp[k])) for k in ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose")}
    free = {k: leaf(bp[k]) for k in ("shape", "exp", "head_scale", "hand_scale")}
    free.update({k: leaf(v) for k, v in fp.items()})
    transl = leaf(torch.zeros(100, 3, device=device))

    def ehm_mesh():
        body = {k: axis_angle_to_matrix(v) for k, v in aa.items()}
        body.update({k: free[k] for k in ("shape", "exp", "head_scale", "hand_scale")})
        flame = {k: free[k] for k in fp}
        return ehm(body, flame, pose_type="rotmat")["vertices"][:, :10475] + transl[:, None]

    with torch.no_grad():
        start = ehm_mesh()
        transl += (gt - start).mean(1)
    report = {}
    params = [*aa.values(), *free.values(), transl]
    opt = torch.optim.Adam(params, lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for _ in range(ITERS):
        opt.zero_grad(set_to_none=True)
        loss = (ehm_mesh() - gt).norm(dim=-1).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        fit = ehm_mesh().double().cpu().numpy()
    gt_np = gt.double().cpu().numpy()
    m = [proto.evaluate(a, b) for a, b in zip(fit, gt_np)]
    report["ehm_s_fit"] = {k: float(np.mean([x[k] for x in m])) for k in m[0]}

    # Control: plain SMPL-X 2020 (reference smplx implementation), same init.
    import smplx
    sx = smplx.SMPLX(model_path=str(ev.PEAR_ROOT / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz"), ext="npz",
                     gender="neutral", num_betas=300, num_expression_coeffs=100, use_pca=False,
                     flat_hand_mean=True, use_face_contour=True).to(device)
    p = {"global_orient": leaf(aa["global_pose"].detach().reshape(100, 3)),
         "body_pose": leaf(aa["body_pose"].detach().reshape(100, 63)),
         "left_hand_pose": leaf(aa["left_hand_pose"].detach().reshape(100, 45)),
         "right_hand_pose": leaf(aa["right_hand_pose"].detach().reshape(100, 45)),
         "jaw_pose": leaf(torch.zeros(100, 3, device=device)),
         "leye_pose": leaf(torch.zeros(100, 3, device=device)), "reye_pose": leaf(torch.zeros(100, 3, device=device)),
         "betas": leaf(torch.zeros(100, 300, device=device)), "expression": leaf(torch.zeros(100, 100, device=device)),
         "transl": leaf(torch.zeros(100, 3, device=device))}
    with torch.no_grad():
        p["transl"] += (gt - sx(**p).vertices).mean(1)
    opt = torch.optim.Adam(list(p.values()), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for _ in range(ITERS):
        opt.zero_grad(set_to_none=True)
        loss = (sx(**p).vertices - gt).norm(dim=-1).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        fit_sx = sx(**p).vertices.double().cpu().numpy()
    m = [proto.evaluate(a, b) for a, b in zip(fit_sx, gt_np)]
    report["smplx_2020_fit"] = {k: float(np.mean([x[k] for x in m])) for k in m[0]}
    report["iterations"] = ITERS
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
