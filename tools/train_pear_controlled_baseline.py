#!/usr/bin/env python
"""Run matched PEAR continuation controls from an immutable student checkpoint.

This is a thin experiment wrapper around PEAR's distillation trainer.  It keeps
the model and data path unchanged while making the disputed joint-2D term the
only difference between experimental arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
sys.path.insert(0, str(PEAR_ROOT))

import train_pear_student_distill as trainer  # noqa: E402


MODES = ("legacy", "joint2d_off", "camera_projected")


def pop_wrapper_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--controlled_mode", choices=MODES, required=True)
    parser.add_argument("--controlled_seed", type=int, default=20260927)
    parser.add_argument(
        "--load_optimizer_state",
        action="store_true",
        help="Load optimizer momentum from the source checkpoint. The matched default is fresh.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def load_weights_only(resume_path, student, optimizer, scaler, device, feature_adapter):
    del optimizer, scaler, device, feature_adapter
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    student.load_state_dict(checkpoint["student"], strict=True)
    return int(checkpoint["step"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_joints(output: dict, joints: torch.Tensor) -> torch.Tensor:
    """Project EHM joints through PEAR's full perspective camera into [0, 1]."""
    rt = output["pd_cam"].float()
    camera = torch.einsum("bij,bkj->bki", rt[:, :3, :3], joints.float())
    camera = camera + rt[:, None, :3, 3]
    perspective = 24.0 * camera[..., :2] / camera[..., 2:3].clamp_min(1e-4)
    return (1.0 - perspective) * 0.5


def camera_projected_joint_losses(
    smplx_model,
    student_output: dict,
    teacher_output: dict,
    want_3d: bool,
    want_2d: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_mesh = smplx_model(trainer.add_body_cam(student_output), pose_type="rotmat")
    with torch.no_grad():
        teacher_mesh = smplx_model(trainer.add_body_cam(teacher_output), pose_type="rotmat")
    student_3d = student_mesh["joints"][:, :22]
    teacher_3d = teacher_mesh["joints"][:, :22].detach()

    joint3d = student_3d.new_zeros(())
    if want_3d:
        joint3d = F.l1_loss(student_3d, teacher_3d)

    joint2d = student_3d.new_zeros(())
    if want_2d:
        joint2d = F.l1_loss(
            project_joints(student_output, student_3d),
            project_joints(teacher_output, teacher_3d).detach(),
        )
    return joint3d, joint2d


def main() -> None:
    wrapper = pop_wrapper_args()
    random.seed(wrapper.controlled_seed)
    np.random.seed(wrapper.controlled_seed)
    torch.manual_seed(wrapper.controlled_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(wrapper.controlled_seed)
    original_parse_args = trainer.parse_args

    def controlled_parse_args():
        args = original_parse_args()
        args.controlled_mode = wrapper.controlled_mode
        args.controlled_seed = wrapper.controlled_seed
        args.optimizer_initialization = (
            "checkpoint" if wrapper.load_optimizer_state else "fresh"
        )
        if wrapper.controlled_mode == "joint2d_off":
            args.joint2d_weight = 0.0
        return args

    trainer.parse_args = controlled_parse_args
    if not wrapper.load_optimizer_state:
        trainer.load_resume = load_weights_only
    if wrapper.controlled_mode == "camera_projected":
        trainer.joint_losses = camera_projected_joint_losses

    args = trainer.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "controlled_mode": wrapper.controlled_mode,
        "optimizer_initialization": args.optimizer_initialization,
        "seed": wrapper.controlled_seed,
        "source_checkpoint": str(args.resume),
        "source_checkpoint_sha256": sha256_file(args.resume),
        "source_step": 235000,
        "difference_from_legacy": {
            "legacy": "none",
            "joint2d_off": "joint2d_weight forced to zero",
            "camera_projected": "joint-2D loss uses full PEAR R/T and perspective projection",
        }[wrapper.controlled_mode],
    }
    (args.output_dir / "controlled_experiment.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    trainer.train(args)


if __name__ == "__main__":
    main()
