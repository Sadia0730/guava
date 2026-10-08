#!/usr/bin/env python
"""Rank PEAR student checkpoints on fixed single-frame and temporal samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_pear_student_quality import (
    clip_metrics,
    infer,
    load_images,
    load_models,
    read_manifest,
    sample_clips,
    sample_frames,
    single_frame_metrics,
    summarize_clip_metrics,
    tree_cat,
    tree_index_range,
    tree_to_cpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    parser.add_argument("--teacher-config", default="configs/infer.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--random-frames", type=int, default=100)
    parser.add_argument("--motion-clips", type=int, default=10)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    return parser.parse_args()


def nested(value: dict, *keys: str) -> float:
    for key in keys:
        value = value[key]
    return float(value)


def selection_components(single: dict, temporal: dict) -> dict[str, float]:
    pose = (
        nested(single, "body", "global_pose_geodesic_deg", "mean") / 3.0
        + nested(single, "body", "body_joint_geodesic_deg", "mean") / 5.0
        + nested(single, "hands", "combined_joint_geodesic_deg", "mean") / 10.0
        + nested(single, "face_expression", "jaw_geodesic_deg", "mean") / 5.0
    ) / 4.0

    amplitude_errors = []
    direction_errors = []
    relative_delta_errors = []
    for group in ("body", "hands", "face_expression"):
        metrics = temporal[group]
        retained = float(metrics["motion_amplitude_retained_percent"])
        amplitude_errors.append(abs(retained - 100.0) / 100.0)
        direction_errors.append(1.0 - float(metrics["motion_direction_cosine"]))
        teacher_delta = max(float(metrics["teacher_frame_delta_mean"]), 1e-8)
        relative_delta_errors.append(float(metrics["frame_delta_mae"]) / teacher_delta)

    temporal_amplitude = sum(amplitude_errors) / len(amplitude_errors)
    temporal_direction = sum(direction_errors) / len(direction_errors)
    temporal_delta = sum(relative_delta_errors) / len(relative_delta_errors)
    total = pose + temporal_amplitude + temporal_direction + temporal_delta
    return {
        "pose_target_error": pose,
        "temporal_amplitude_error": temporal_amplitude,
        "temporal_direction_error": temporal_direction,
        "relative_frame_delta_error": temporal_delta,
        "selection_score": total,
    }


def infer_student(images: torch.Tensor, model, args) -> dict:
    outputs = []
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.precision
    ]
    for start in range(0, len(images), args.batch_size):
        batch = images[start : start + args.batch_size].to(device, non_blocking=True)
        autocast = torch.autocast(
            device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
        )
        with torch.inference_mode(), autocast:
            output = model(batch)
        outputs.append(tree_to_cpu(output))
    return tree_cat(outputs)


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.checkpoint_dir.resolve().glob("step_*.pt"))
    if not checkpoints:
        checkpoints = sorted(args.checkpoint_dir.resolve().glob("finetune_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no step or fine-tune checkpoints under {args.checkpoint_dir}")

    sequences = read_manifest(args.manifest)
    import random

    rng = random.Random(args.seed)
    random_refs = sample_frames(sequences, args.random_frames, rng)
    clips = sample_clips(sequences, args.motion_clips, args.clip_length, rng)
    refs = random_refs + [ref for clip in clips for ref in clip]
    images = load_images(sequences, refs)

    model_args = SimpleNamespace(
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        student_config=args.student_config,
        teacher_config=args.teacher_config,
        teacher_checkpoint=args.teacher_checkpoint,
        student_checkpoint=checkpoints[0],
    )
    teacher_model, student_model, _ = load_models(model_args)
    teacher_output, first_student_output = infer(images, teacher_model, student_model, model_args)

    results = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        step = int(checkpoint["step"])
        if checkpoint_index == 0:
            student_output = first_student_output
        else:
            student_model.load_state_dict(checkpoint["student"], strict=True)
            student_output = infer_student(images, student_model, model_args)
        del checkpoint

        single = single_frame_metrics(
            tree_index_range(teacher_output, 0, args.random_frames),
            tree_index_range(student_output, 0, args.random_frames),
        )
        clip_reports = []
        offset = args.random_frames
        for clip in clips:
            end = offset + len(clip)
            clip_reports.append(
                clip_metrics(
                    tree_index_range(teacher_output, offset, end),
                    tree_index_range(student_output, offset, end),
                )
            )
            offset = end
        temporal = summarize_clip_metrics(clip_reports)
        components = selection_components(single, temporal)
        result = {
            "checkpoint": str(checkpoint_path),
            "step": step,
            "single_frame_pose_agreement": single,
            "sequential_motion_retention": temporal,
            "selection": components,
        }
        results.append(result)
        print(
            f"checkpoint {checkpoint_index + 1}/{len(checkpoints)} step={step} "
            f"score={components['selection_score']:.4f} "
            f"body_retained={temporal['body']['motion_amplitude_retained_percent']:.1f}%",
            flush=True,
        )

    ranked = sorted(results, key=lambda item: item["selection"]["selection_score"])
    report = {
        "method": {
            "manifest": str(args.manifest.resolve()),
            "random_frames": args.random_frames,
            "motion_clips": args.motion_clips,
            "clip_length": args.clip_length,
            "seed": args.seed,
            "precision": args.precision,
            "rendering_in_sweep": False,
            "selection_score": (
                "mean normalized global/body/hand/jaw target error + mean motion-amplitude "
                "error + mean motion-direction error + mean relative frame-delta error; lower is better"
            ),
        },
        "ranked": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("top checkpoints:")
    for rank, item in enumerate(ranked[:5], start=1):
        print(
            f"{rank}. step {item['step']} score={item['selection']['selection_score']:.4f} "
            f"body={item['sequential_motion_retention']['body']['motion_amplitude_retained_percent']:.1f}% "
            f"hands={item['sequential_motion_retention']['hands']['motion_amplitude_retained_percent']:.1f}% "
            f"face={item['sequential_motion_retention']['face_expression']['motion_amplitude_retained_percent']:.1f}%"
        )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
