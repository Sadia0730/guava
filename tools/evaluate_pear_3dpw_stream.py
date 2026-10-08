#!/usr/bin/env python
"""Stream full-split paired PEAR teacher/student 3DPW evaluation.

This evaluator is intentionally separate from the 111-frame visual pilot. It
does not cache crops, so it can process the full official 3DPW folders. Its
camera-coordinate SMPL/SMPL-X metric is a reproducible local protocol, not a
replacement for an author-provided benchmark evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_pear_3dpw import (
    DEFAULT_TEACHER,
    PEAR_ROOT,
    collect_candidates,
    load_models,
    sha256_file,
    tree_float,
)
from tools.pear_geometry import (
    IMAGE_SIZE,
    J14_INDICES,
    LOWER6_INDICES,
    LOWER8_INDICES,
    bbox_from_keypoints,
    mpjpe_mm,
    pa_mpjpe_mm,
    root_center,
    square_crop_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-config", default="configs/infer.yaml")
    parser.add_argument("--smplx-assets", type=Path, default=PEAR_ROOT / "assets/SMPLX")
    parser.add_argument("--crop-scale", type=float, default=1.25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--max-frames-per-split",
        type=int,
        default=0,
        help="Optional smoke-test limit. Zero evaluates each entire split.",
    )
    return parser.parse_args()


METRICS = (
    "body22_mpjpe_mm",
    "j14_mpjpe_mm",
    "j14_pa_mpjpe_mm",
    "lower6_mpjpe_mm",
    "lower8_mpjpe_mm",
)


def confidence_interval(groups: dict[str, list[float]], count: int, seed: int) -> list[float]:
    names = sorted(groups)
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(count):
        selected = generator.choice(names, size=len(names), replace=True)
        values = [value for name in selected for value in groups[name]]
        estimates.append(float(np.mean(values)))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def summarize(values: dict[str, dict[str, list[float]]], args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model, model_metrics in values.items():
        result[model] = {}
        for name, metric_values in model_metrics.items():
            flat = [value for sequence_values in metric_values.values() for value in sequence_values]
            result[model][name] = {
                "mean": float(np.mean(flat)),
                "median": float(np.median(flat)),
                "sequence_bootstrap_95ci_mean": confidence_interval(
                    metric_values, args.bootstrap_samples, args.seed
                ),
            }
    return result


def crop_image(sample, crop_scale: float) -> torch.Tensor:
    image_bgr = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(sample.image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    keypoints = np.asarray(sample.keypoints_2d, dtype=np.float32)
    confidence = np.asarray(sample.confidence, dtype=np.float32)
    transform = square_crop_transform(bbox_from_keypoints(keypoints, confidence), crop_scale)
    crop = cv2.warpAffine(image_rgb, transform, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)
    return torch.from_numpy(crop.copy()).permute(2, 0, 1).contiguous()


def evaluate(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_samples = {}
    for split in ("train", "validation", "test"):
        items = [sample for candidates in collect_candidates(args.dataset_root, split).values() for sample in candidates]
        if args.max_frames_per_split:
            items = items[: args.max_frames_per_split]
        split_samples[split] = items
    samples = [(split, sample) for split, items in split_samples.items() for sample in items]
    teacher, student, smplx_model, step = load_models(args)
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    indices = {
        "body22_mpjpe_mm": None,
        "j14_mpjpe_mm": torch.as_tensor(J14_INDICES, dtype=torch.long, device=device),
        "j14_pa_mpjpe_mm": torch.as_tensor(J14_INDICES, dtype=torch.long, device=device),
        "lower6_mpjpe_mm": torch.as_tensor(LOWER6_INDICES, dtype=torch.long, device=device),
        "lower8_mpjpe_mm": torch.as_tensor(LOWER8_INDICES, dtype=torch.long, device=device),
    }
    modes = {
        "all_test_mode": {"train", "validation", "test"},
        "train_test_mode": {"test"},
        "validation_mode": {"train", "test"},
    }
    values = {
        mode: {
            model: {metric: defaultdict(list) for metric in METRICS}
            for model in ("teacher", "student")
        }
        for mode in modes
    }

    for start in range(0, len(samples), args.batch_size):
        records = samples[start : start + args.batch_size]
        images = torch.stack([crop_image(sample, args.crop_scale) for _, sample in records]).float().div_(255.0)
        gt = torch.tensor([sample.joints_3d for _, sample in records], dtype=torch.float32, device=device)
        gt_camera = torch.tensor([sample.camera_pose for _, sample in records], dtype=torch.float32, device=device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            teacher_output = tree_float(teacher(images.to(device, non_blocking=True)))
            student_output = tree_float(student(images.to(device, non_blocking=True)))
        with torch.inference_mode():
            outputs = {"teacher": teacher_output, "student": student_output}
            predictions = {}
            for model, output in outputs.items():
                joints = smplx_model({**output["body_param"], "body_cam": torch.stack((24.0 / output["pd_cam"][:, 2, 3].clamp_min(1e-6), output["pd_cam"][:, 0, 3], output["pd_cam"][:, 1, 3]), dim=1)}, pose_type="rotmat")["joints"][:, :22]
                camera_joints = torch.einsum("bij,bkj->bki", output["pd_cam"][:, :3, :3], joints) + output["pd_cam"][:, None, :3, 3]
                predictions[model] = root_center(camera_joints * camera_joints.new_tensor([-1.0, -1.0, 1.0]))
            gt_camera_joints = torch.einsum("bij,bkj->bki", gt_camera[:, :3, :3], gt[:, :22]) + gt_camera[:, None, :3, 3]
            gt_centered = root_center(gt_camera_joints)
            batch_metrics = {model: {} for model in outputs}
            for model, prediction in predictions.items():
                batch_metrics[model]["body22_mpjpe_mm"] = mpjpe_mm(prediction, gt_centered)
                for metric, joint_indices in indices.items():
                    if joint_indices is None:
                        continue
                    source = prediction.index_select(1, joint_indices)
                    target = gt_centered.index_select(1, joint_indices)
                    batch_metrics[model][metric] = (
                        pa_mpjpe_mm(source, target) if metric == "j14_pa_mpjpe_mm" else mpjpe_mm(source, target)
                    )

        for local_index, (split, sample) in enumerate(records):
            sequence_key = f"{split}/{sample.sequence}"
            for mode, included_splits in modes.items():
                if split not in included_splits:
                    continue
                for model in outputs:
                    for metric in METRICS:
                        values[mode][model][metric][sequence_key].append(
                            float(batch_metrics[model][metric][local_index].cpu())
                        )
        print(f"inference: {min(start + args.batch_size, len(samples))}/{len(samples)}", flush=True)

    summary = {
        "protocol": {
            "full_folder_evaluation": True,
            "split_counts": {split: len(items) for split, items in split_samples.items()},
            "all_test_mode": "train + validation + test",
            "train_test_mode": "test only; train may train and validation may select",
            "validation_mode": "train + test; valid only for models not trained on 3DPW train labels",
            "crop": "annotated 3DPW OpenPose-18 box, square scale 1.25, 256 x 256",
            "joint_protocol": "camera-frame, pelvis-centered corresponding SMPL/SMPL-X joints; PEAR xy basis converted to OpenCV",
            "warning": "This local protocol is not the PEAR Table 3 evaluator until its exact crop, joint regressor, and metric code are verified.",
        },
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "student_step": step,
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
        "results": {mode: summarize(mode_values, args) for mode, mode_values in values.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
