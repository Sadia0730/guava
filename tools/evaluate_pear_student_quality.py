#!/usr/bin/env python
"""Evaluate PEAR teacher/student pose, temporal motion, and GUAVA renders."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

extra_site_packages = os.environ.get("GUAVA_EXTRA_SITE_PACKAGES")
if extra_site_packages:
    sys.path.extend(path for path in extra_site_packages.split(os.pathsep) if path)

import cv2
import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image


ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
sys.path.insert(0, str(ROOT))
TEACHER_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/"
    "513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt"
)
ROTATION_KEYS = ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose")


@dataclass(frozen=True)
class Sequence:
    source: str
    name: str
    directory: Path
    num_frames: int
    fps: float
    frame_step: int


@dataclass(frozen=True)
class FrameRef:
    sequence_index: int
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-config", default="configs/student_l70.yaml")
    parser.add_argument("--teacher-config", default="configs/infer.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path, default=TEACHER_CHECKPOINT)
    parser.add_argument("--random-frames", type=int, default=100)
    parser.add_argument("--motion-clips", type=int, default=10)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--render-guava", action="store_true")
    parser.add_argument("--model-path", type=Path, default=ROOT / "assets/GUAVA")
    parser.add_argument(
        "--source-data-path",
        type=Path,
        default=ROOT / "assets/example/tracked_image/NTFbJBzjlts__047",
    )
    parser.add_argument("--render-size", type=int, choices=(256, 512), default=256)
    parser.add_argument(
        "--baseline-render-dir",
        type=Path,
        help="Optional prior render directory used to save teacher/baseline/student triplets.",
    )
    parser.add_argument("--baseline-label", default="baseline student")
    parser.add_argument("--student-label", default="evaluated student")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[Sequence]:
    sequences = []
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            directory = Path(item["directory"])
            if not directory.is_absolute():
                directory = (path.resolve().parent / directory).resolve()
            sequences.append(
                Sequence(
                    source=str(item["source"]),
                    name=str(item["sequence"]),
                    directory=directory,
                    num_frames=int(item["num_frames"]),
                    fps=float(item["fps"]),
                    frame_step=int(item["frame_step"]),
                )
            )
    if not sequences:
        raise ValueError(f"empty manifest: {path}")
    return sequences


def sample_frames(
    sequences: list[Sequence], count: int, rng: random.Random
) -> list[FrameRef]:
    cumulative = []
    total = 0
    for sequence in sequences:
        total += sequence.num_frames
        cumulative.append(total)
    if count > total:
        raise ValueError(f"requested {count} random frames from only {total}")
    refs = []
    for flat_index in rng.sample(range(total), count):
        sequence_index = bisect.bisect_right(cumulative, flat_index)
        previous = cumulative[sequence_index - 1] if sequence_index else 0
        refs.append(FrameRef(sequence_index, flat_index - previous))
    return refs


def sample_clips(
    sequences: list[Sequence], count: int, length: int, rng: random.Random
) -> list[list[FrameRef]]:
    eligible = [index for index, sequence in enumerate(sequences) if sequence.num_frames >= length]
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} sequences can provide {length}-frame clips")
    clips = []
    for sequence_index in rng.sample(eligible, count):
        sequence = sequences[sequence_index]
        start = rng.randint(0, sequence.num_frames - length)
        clips.append([FrameRef(sequence_index, start + offset) for offset in range(length)])
    return clips


def load_images(sequences: list[Sequence], refs: list[FrameRef]) -> torch.Tensor:
    images = []
    for ref in refs:
        path = sequences[ref.sequence_index].directory / f"{ref.frame_index:06d}.jpg"
        image = read_image(str(path), mode=ImageReadMode.RGB)
        if tuple(image.shape) != (3, 256, 256):
            raise ValueError(f"expected 3x256x256 image, got {tuple(image.shape)}: {path}")
        images.append(image.float().div_(255.0))
    return torch.stack(images)


def load_models(args: argparse.Namespace):
    original_directory = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from train_pear_student_distill import load_student, load_teacher

        device = torch.device(args.device)
        teacher = load_teacher(args.teacher_config, args.teacher_checkpoint.resolve(), device)
        student = load_student(args.student_config, args.batch_size, 1, device)
        checkpoint = torch.load(
            args.student_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        student.load_state_dict(checkpoint["student"], strict=True)
        student.eval()
        step = int(checkpoint["step"])
        del checkpoint
        return teacher, student, step
    finally:
        os.chdir(original_directory)
        sys.path.remove(str(PEAR_ROOT))


def tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().float().cpu()
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items() if item is not None}
    return value


def tree_cat(items: list[Any]) -> Any:
    first = items[0]
    if torch.is_tensor(first):
        return torch.cat(items, dim=0)
    if isinstance(first, dict):
        return {key: tree_cat([item[key] for item in items]) for key in first}
    return first


def tree_index(value: Any, index: int, device: str | None = None) -> Any:
    if torch.is_tensor(value):
        selected = value[index : index + 1]
        return selected.to(device) if device else selected
    if isinstance(value, dict):
        return {key: tree_index(item, index, device) for key, item in value.items()}
    return value


def infer(
    images: torch.Tensor,
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    teacher_batches = []
    student_batches = []
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.precision
    ]
    for start in range(0, len(images), args.batch_size):
        batch = images[start : start + args.batch_size].to(device, non_blocking=True)
        autocast = torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)
        with torch.inference_mode(), autocast:
            teacher_output = teacher(batch)
            student_output = student(batch)
        teacher_batches.append(tree_to_cpu(teacher_output))
        student_batches.append(tree_to_cpu(student_output))
        print(f"inference: {min(start + args.batch_size, len(images))}/{len(images)}", flush=True)
    return tree_cat(teacher_batches), tree_cat(student_batches)


def distribution(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten()
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def rotation_error_deg(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    relative = torch.matmul(student.float(), teacher.float().transpose(-1, -2))
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    return angle * (180.0 / math.pi)


def axis_angle_matrix(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    x, y, z = values.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1)
    return torch.matrix_exp(skew.reshape(*values.shape[:-1], 3, 3))


def l1_per_frame(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (student.float() - teacher.float()).abs().flatten(1).mean(1)


def single_frame_metrics(teacher: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    teacher_body = teacher["body_param"]
    student_body = student["body_param"]
    teacher_face = teacher["flame_param"]
    student_face = student["flame_param"]
    left = rotation_error_deg(student_body["left_hand_pose"], teacher_body["left_hand_pose"])
    right = rotation_error_deg(student_body["right_hand_pose"], teacher_body["right_hand_pose"])
    jaw = rotation_error_deg(
        axis_angle_matrix(student_face["jaw_params"]),
        axis_angle_matrix(teacher_face["jaw_params"]),
    )
    camera_translation = torch.linalg.norm(
        student["pd_cam"][:, :3, 3] - teacher["pd_cam"][:, :3, 3], dim=-1
    )
    return {
        "body": {
            "global_pose_geodesic_deg": distribution(
                rotation_error_deg(student_body["global_pose"], teacher_body["global_pose"])
            ),
            "body_joint_geodesic_deg": distribution(
                rotation_error_deg(student_body["body_pose"], teacher_body["body_pose"])
            ),
        },
        "hands": {
            "left_joint_geodesic_deg": distribution(left),
            "right_joint_geodesic_deg": distribution(right),
            "combined_joint_geodesic_deg": distribution(torch.cat((left.flatten(), right.flatten()))),
        },
        "face_expression": {
            "expression_l1": distribution(
                l1_per_frame(student_face["expression_params"], teacher_face["expression_params"])
            ),
            "jaw_geodesic_deg": distribution(jaw),
            "face_pose_l1": distribution(
                l1_per_frame(student_face["pose_params"], teacher_face["pose_params"])
            ),
            "eye_pose_l1": distribution(
                l1_per_frame(student_face["eye_pose_params"], teacher_face["eye_pose_params"])
            ),
            "eyelid_l1": distribution(
                l1_per_frame(student_face["eyelid_params"], teacher_face["eyelid_params"])
            ),
        },
        "camera": {"translation_l2": distribution(camera_translation)},
    }


def rotation_motion(values: torch.Tensor) -> torch.Tensor:
    return rotation_error_deg(values[1:], values[:-1]).flatten(1).mean(1)


def vector_motion(values: torch.Tensor) -> torch.Tensor:
    return (values[1:].float() - values[:-1].float()).abs().flatten(1).mean(1)


def motion_report(teacher_motion: torch.Tensor, student_motion: torch.Tensor) -> dict[str, Any]:
    teacher_mean = float(teacher_motion.mean())
    student_mean = float(student_motion.mean())
    return {
        "teacher_frame_delta_mean": teacher_mean,
        "student_frame_delta_mean": student_mean,
        "frame_delta_mae": float((student_motion - teacher_motion).abs().mean()),
        "motion_amplitude_retained_percent": (
            100.0 * student_mean / teacher_mean if teacher_mean > 1e-12 else None
        ),
    }


def raw_motion_agreement(teacher: torch.Tensor, student: torch.Tensor) -> dict[str, float]:
    teacher_delta = (teacher[1:].float() - teacher[:-1].float()).flatten(1)
    student_delta = (student[1:].float() - student[:-1].float()).flatten(1)
    cosine = torch.nn.functional.cosine_similarity(student_delta, teacher_delta, dim=1, eps=1e-8)
    velocity_mae = (student_delta - teacher_delta).abs().mean(1)
    if len(teacher) > 2:
        teacher_accel = teacher_delta[1:] - teacher_delta[:-1]
        student_accel = student_delta[1:] - student_delta[:-1]
        acceleration_mae = float((student_accel - teacher_accel).abs().mean())
    else:
        acceleration_mae = 0.0
    return {
        "velocity_mae": float(velocity_mae.mean()),
        "motion_direction_cosine": float(cosine.mean()),
        "acceleration_mae": acceleration_mae,
    }


def clip_metrics(teacher: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    tb, sb = teacher["body_param"], student["body_param"]
    tf, sf = teacher["flame_param"], student["flame_param"]
    body_teacher = torch.cat((tb["global_pose"], tb["body_pose"]), dim=1)
    body_student = torch.cat((sb["global_pose"], sb["body_pose"]), dim=1)
    hands_teacher = torch.cat((tb["left_hand_pose"], tb["right_hand_pose"]), dim=1)
    hands_student = torch.cat((sb["left_hand_pose"], sb["right_hand_pose"]), dim=1)
    face_keys = ("expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params")
    face_teacher = torch.cat([tf[key].flatten(1) for key in face_keys], dim=1)
    face_student = torch.cat([sf[key].flatten(1) for key in face_keys], dim=1)
    groups = {
        "body": (body_teacher, body_student, rotation_motion),
        "hands": (hands_teacher, hands_student, rotation_motion),
        "face_expression": (face_teacher, face_student, vector_motion),
    }
    result = {}
    for name, (teacher_values, student_values, motion_fn) in groups.items():
        result[name] = motion_report(motion_fn(teacher_values), motion_fn(student_values))
        result[name].update(raw_motion_agreement(teacher_values, student_values))
    return result


def summarize_clip_metrics(clips: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for group in ("body", "hands", "face_expression"):
        keys = clips[0][group]
        summary[group] = {}
        for key in keys:
            values = [clip[group][key] for clip in clips if clip[group][key] is not None]
            summary[group][key] = float(np.mean(values)) if values else None
    return summary


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float32) / 255.0
    second = second.astype(np.float32) / 255.0
    c1, c2 = 0.01**2, 0.03**2
    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    sigma_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first * mu_first
    sigma_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second * mu_second
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_first * mu_second
    score = ((2 * mu_first * mu_second + c1) * (2 * covariance + c2)) / (
        (mu_first * mu_first + mu_second * mu_second + c1)
        * (sigma_first + sigma_second + c2)
    )
    return float(np.mean(score))


def render_pair_metrics(teacher: np.ndarray, student: np.ndarray) -> dict[str, float]:
    difference = teacher.astype(np.float32) / 255.0 - student.astype(np.float32) / 255.0
    mse = float(np.mean(difference * difference))
    teacher_mask = teacher.max(axis=2) > 3
    student_mask = student.max(axis=2) > 3
    union = np.logical_or(teacher_mask, student_mask).sum()
    iou = float(np.logical_and(teacher_mask, student_mask).sum() / union) if union else 1.0
    return {
        "l1": float(np.mean(np.abs(difference))),
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "ssim": ssim(teacher, student),
        "silhouette_iou": iou,
    }


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create video: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def labeled_panel(frame: np.ndarray, text: str) -> np.ndarray:
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_metrics(
    teacher: dict[str, Any],
    student: dict[str, Any],
    random_count: int,
    clips: list[list[FrameRef]],
    sequences: list[Sequence],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from tools.compare_teacher_student import (
        cleanup_pear_modules,
        initialize_guava_renderer,
        render_output,
    )

    cleanup_pear_modules()
    runtime_args = SimpleNamespace(
        source_data_path=args.source_data_path,
        model_path=args.model_path,
        device=args.device,
        render_size=args.render_size,
        precision="fp32",
        compile_targets=[],
        compile_mode="default",
        warmup=1,
        smooth=False,
        smooth_min_cutoff=2.0,
        smooth_beta=0.3,
        smooth_d_cutoff=1.0,
    )
    first_teacher = tree_index(teacher, 0, args.device)
    renderer, target_builder, source_dataset = initialize_guava_renderer(runtime_args, first_teacher)
    pair_metrics = []
    teacher_renders = []
    student_renders = []
    total_render_count = random_count + sum(len(clip) for clip in clips)
    render_dir = args.output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    improvement_dir = args.output_dir / "render_comparisons"
    if args.baseline_render_dir:
        improvement_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(total_render_count):
            teacher_output = tree_index(teacher, index, args.device)
            student_output = tree_index(student, index, args.device)
            with torch.inference_mode():
                teacher_render = render_output(renderer, target_builder, teacher_output)
                student_render = render_output(renderer, target_builder, student_output)
            teacher_renders.append(teacher_render)
            student_renders.append(student_render)
            pair_metrics.append(render_pair_metrics(teacher_render, student_render))
            if index < random_count:
                cv2.imwrite(str(render_dir / f"single_{index:03d}.png"), np.hstack((teacher_render, student_render)))
                if args.baseline_render_dir:
                    baseline_path = args.baseline_render_dir / f"single_{index:03d}.png"
                    baseline_pair = cv2.imread(str(baseline_path), cv2.IMREAD_COLOR)
                    if baseline_pair is None:
                        raise FileNotFoundError(baseline_path)
                    baseline_student = baseline_pair[:, baseline_pair.shape[1] // 2 :]
                    triplet = np.hstack(
                        (
                            labeled_panel(teacher_render, "PEAR teacher"),
                            labeled_panel(baseline_student, args.baseline_label),
                            labeled_panel(student_render, args.student_label),
                        )
                    )
                    cv2.imwrite(str(improvement_dir / f"comparison_{index:03d}.png"), triplet)
            print(f"GUAVA render pairs: {index + 1}/{total_render_count}", flush=True)
    finally:
        source_dataset._lmdb_engine.close()

    single_summary = {
        key: float(np.mean([item[key] for item in pair_metrics[:random_count]]))
        for key in pair_metrics[0]
    }
    clip_summaries = []
    offset = random_count
    for clip_index, clip in enumerate(clips):
        length = len(clip)
        teacher_clip = teacher_renders[offset : offset + length]
        student_clip = student_renders[offset : offset + length]
        teacher_delta = np.array(
            [np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))) / 255.0 for a, b in zip(teacher_clip, teacher_clip[1:])]
        )
        student_delta = np.array(
            [np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))) / 255.0 for a, b in zip(student_clip, student_clip[1:])]
        )
        teacher_mean = float(teacher_delta.mean())
        summary = {
            "clip": clip_index,
            "sequence": sequences[clip[0].sequence_index].name,
            "teacher_frame_delta_mean": teacher_mean,
            "student_frame_delta_mean": float(student_delta.mean()),
            "frame_delta_mae": float(np.mean(np.abs(student_delta - teacher_delta))),
            "motion_amplitude_retained_percent": (
                100.0 * float(student_delta.mean()) / teacher_mean if teacher_mean > 1e-12 else None
            ),
        }
        clip_summaries.append(summary)
        fps = sequences[clip[0].sequence_index].fps / sequences[clip[0].sequence_index].frame_step
        panels = [np.hstack((teacher_frame, student_frame)) for teacher_frame, student_frame in zip(teacher_clip, student_clip)]
        write_video(render_dir / f"motion_clip_{clip_index:02d}.mp4", panels, fps)
        offset += length
    temporal_summary = {}
    for key in ("teacher_frame_delta_mean", "student_frame_delta_mean", "frame_delta_mae"):
        temporal_summary[key] = float(np.mean([clip[key] for clip in clip_summaries]))
    retained = [clip["motion_amplitude_retained_percent"] for clip in clip_summaries if clip["motion_amplitude_retained_percent"] is not None]
    temporal_summary["motion_amplitude_retained_percent"] = float(np.mean(retained)) if retained else None
    return {
        "single_frame": single_summary,
        "sequential_summary": temporal_summary,
        "sequential_clips": clip_summaries,
        "saved_single_frame_pairs": random_count,
        "saved_motion_videos": len(clips),
        "saved_teacher_baseline_student_triplets": random_count if args.baseline_render_dir else 0,
    }


def frame_metadata(sequences: list[Sequence], ref: FrameRef) -> dict[str, Any]:
    sequence = sequences[ref.sequence_index]
    return {
        "source": sequence.source,
        "sequence": sequence.name,
        "frame_index": ref.frame_index * sequence.frame_step,
    }


def main() -> None:
    args = parse_args()
    if args.random_frames < 100:
        raise ValueError("--random-frames must be at least 100")
    if args.motion_clips < 10:
        raise ValueError("--motion-clips must be at least 10")
    if args.clip_length < 3:
        raise ValueError("--clip-length must be at least 3")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequences = read_manifest(args.manifest)
    rng = random.Random(args.seed)
    random_refs = sample_frames(sequences, args.random_frames, rng)
    clips = sample_clips(sequences, args.motion_clips, args.clip_length, rng)
    all_refs = random_refs + [ref for clip in clips for ref in clip]
    images = load_images(sequences, all_refs)

    teacher_model, student_model, student_step = load_models(args)
    teacher, student = infer(images, teacher_model, student_model, args)
    del teacher_model, student_model
    torch.cuda.empty_cache()

    random_teacher = tree_index_range(teacher, 0, args.random_frames)
    random_student = tree_index_range(student, 0, args.random_frames)
    single = single_frame_metrics(random_teacher, random_student)

    clip_reports = []
    offset = args.random_frames
    for clip_index, clip in enumerate(clips):
        end = offset + len(clip)
        report = clip_metrics(
            tree_index_range(teacher, offset, end), tree_index_range(student, offset, end)
        )
        report["clip"] = clip_index
        report["sequence"] = sequences[clip[0].sequence_index].name
        report["start_frame"] = clip[0].frame_index * sequences[clip[0].sequence_index].frame_step
        report["frames"] = len(clip)
        clip_reports.append(report)
        offset = end
    temporal = summarize_clip_metrics(clip_reports)

    report = {
        "evaluation": {
            "student_checkpoint": str(args.student_checkpoint.resolve()),
            "student_checkpoint_sha256": sha256_file(args.student_checkpoint.resolve()),
            "student_step": student_step,
            "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
            "validation_manifest": str(args.manifest.resolve()),
            "seed": args.seed,
            "precision": args.precision,
            "random_validation_frames": args.random_frames,
            "sequential_motion_clips": args.motion_clips,
            "frames_per_motion_clip": args.clip_length,
            "random_frame_samples": [frame_metadata(sequences, ref) for ref in random_refs],
            "motion_clip_samples": [
                {
                    "source": sequences[clip[0].sequence_index].source,
                    "sequence": sequences[clip[0].sequence_index].name,
                    "start_frame": clip[0].frame_index * sequences[clip[0].sequence_index].frame_step,
                    "end_frame": clip[-1].frame_index * sequences[clip[0].sequence_index].frame_step,
                    "fps": sequences[clip[0].sequence_index].fps,
                }
                for clip in clips
            ],
            "scout_temporal_predictor_router_used": False,
        },
        "single_frame_pose_agreement": single,
        "sequential_motion_retention": {
            "summary": temporal,
            "clips": clip_reports,
        },
        "direct_guava_rendering": None,
    }
    report_path = args.output_dir / "evaluation_report.json"
    # Persist the numerical evaluation before optional rendering. Rendering
    # uses external CUDA extensions and must not discard a completed pose run
    # if that separate environment is unavailable.
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    rendering = None
    if args.render_guava:
        rendering = render_metrics(
            teacher, student, args.random_frames, clips, sequences, args
        )
        report["direct_guava_rendering"] = rendering
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "single_frame_pose_agreement": single,
        "sequential_motion_retention": temporal,
        "direct_guava_rendering": rendering and {
            "single_frame": rendering["single_frame"],
            "sequential_summary": rendering["sequential_summary"],
        },
    }, indent=2))
    print(f"wrote {report_path}")


def tree_index_range(value: Any, start: int, end: int) -> Any:
    if torch.is_tensor(value):
        return value[start:end]
    if isinstance(value, dict):
        return {key: tree_index_range(item, start, end) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
