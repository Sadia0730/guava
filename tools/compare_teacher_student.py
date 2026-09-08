#!/usr/bin/env python
"""Compare PEAR teacher and student on one shared, deterministic input tensor path.

The default mode is FP32/eager. Supply one or more input categories. Live input
is captured to a finite clip first and only then evaluated, so teacher and
student never see different webcam frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
DEFAULT_STUDENT = ROOT / "outputs/checkpoints/student_step250000_inference.pt"
DEFAULT_SOURCE = ROOT / "assets/example/tracked_image/random_google_pic/blue_shirt"
sys.path.insert(0, str(ROOT))

from tools.pear_debug_utils import find_state_dict, tensor_summary, tensor_tree  # noqa: E402


ROTATION_KEYS = ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose")
IDENTITY_KEYS = (
    "body_param.shape",
    "body_param.joints_offset",
    "body_param.head_scale",
    "body_param.hand_scale",
    "flame_param.shape_params",
)
IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-config", type=Path, default=Path("configs/student_l70.yaml"))
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-config", default="configs/infer.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--download-teacher", action="store_true")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--webcam-video", type=Path)
    parser.add_argument("--live-input", help="Camera index or stream captured before comparison")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--input-crop-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/debug/teacher_student")
    parser.add_argument("--render-guava", action="store_true")
    parser.add_argument("--source-data-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-path", type=Path, default=ROOT / "assets/GUAVA")
    parser.add_argument("--render-size", type=int, choices=(256, 512), default=256)
    return parser.parse_args()


def resolve_pear_path(path: Path) -> Path:
    for candidate in (path, ROOT / path, PEAR_ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def cleanup_pear_modules() -> None:
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if module_file and str(module_file).startswith(str(PEAR_ROOT)):
            if name == "models" or name.startswith("models.") or name == "utils" or name.startswith("utils."):
                del sys.modules[name]


def load_models(args: argparse.Namespace):
    original_directory = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from huggingface_hub import hf_hub_download
        from models.pipeline.ehm_pipeline import Ehm_Pipeline
        from models.pipeline.student_pipeline import PearStudentPipeline
        from utils.general_utils import ConfigDict, add_extra_cfgs

        teacher_cfg = add_extra_cfgs(ConfigDict(model_config_path=args.teacher_config))
        student_config_path = resolve_pear_path(args.student_config)
        student_cfg = add_extra_cfgs(ConfigDict(model_config_path=str(student_config_path)))
        teacher = Ehm_Pipeline(teacher_cfg)
        student = PearStudentPipeline(student_cfg)

        if args.teacher_checkpoint is None:
            try:
                teacher_path = Path(
                    hf_hub_download(
                        repo_id="BestWJH/PEAR_models",
                        filename="pear_model.pt",
                        repo_type="model",
                        local_files_only=not args.download_teacher,
                    )
                )
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    "Teacher checkpoint is not cached. Pass --teacher-checkpoint or "
                    "explicitly allow --download-teacher."
                ) from error
        else:
            teacher_path = args.teacher_checkpoint.resolve()
        teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
        teacher.backbone.load_state_dict(teacher_checkpoint["backbone"], strict=True)
        teacher.head.load_state_dict(teacher_checkpoint["head"], strict=True)
        del teacher_checkpoint

        student_path = args.student_checkpoint.resolve()
        try:
            student_checkpoint = torch.load(student_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError(
                "Student comparison requires the sanitized weights-only checkpoint. Run "
                "tools/audit_student_checkpoint.py first."
            ) from error
        _, student_state = find_state_dict(student_checkpoint)
        student.load_state_dict(student_state, strict=True)
        del student_checkpoint
    finally:
        os.chdir(original_directory)
        sys.path.remove(str(PEAR_ROOT))

    teacher = teacher.to(args.device).eval()
    student = student.to(args.device).eval()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.precision
    ]
    if dtype != torch.float32:
        teacher.backbone.to(dtype=dtype)
        student.backbone.to(dtype=dtype)
    return teacher, student, teacher_path, student_config_path


def center_crop(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return frame.copy()
    height, width = frame.shape[:2]
    crop_height = max(1, int(round(height / scale)))
    crop_width = max(1, int(round(width / scale)))
    y0 = max(0, (height - crop_height) // 2)
    x0 = max(0, (width - crop_width) // 2)
    return frame[y0 : y0 + crop_height, x0 : x0 + crop_width].copy()


def letterbox(frame: np.ndarray, size: int = 256) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(size / height, size / width)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    result = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    result[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return result


def shared_preprocess(frame_bgr: np.ndarray, crop_scale: float, device: str) -> dict[str, Any]:
    crop_bgr = center_crop(frame_bgr, crop_scale)
    model_bgr = letterbox(crop_bgr)
    model_rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
    model_tensor = (
        torch.as_tensor(model_rgb, device=device).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    )
    mean = IMAGE_MEAN.to(device=device, dtype=model_tensor.dtype)
    std = IMAGE_STD.to(device=device, dtype=model_tensor.dtype)
    normalized = ((model_tensor - mean) / std)[:, :, :, 32:-32].contiguous()
    return {
        "original_bgr": frame_bgr,
        "person_crop_bgr": crop_bgr,
        "model_bgr": model_bgr,
        "model_tensor": model_tensor,
        "normalized_tensor": normalized,
    }


def read_video(path: Path, limit: int, step: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    index = 0
    try:
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frames.append(frame)
            index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video contains no readable frames: {path}")
    return frames, fps / step


def read_manifest(path: Path, limit: int, step: int) -> tuple[list[np.ndarray], float]:
    records = []
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    frames = []
    fps = 30.0
    for record in records:
        directory = Path(record["directory"])
        if not directory.is_absolute():
            directory = (path.resolve().parent / directory).resolve()
        fps = float(record.get("fps", fps)) / max(int(record.get("frame_step", 1)), 1)
        for index in range(0, int(record["num_frames"]), step):
            frame = cv2.imread(str(directory / f"{index:06d}.jpg"), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(directory / f"{index:06d}.jpg")
            frames.append(frame)
            if len(frames) >= limit:
                return frames, fps / step
    if not frames:
        raise RuntimeError(f"manifest contains no readable frames: {path}")
    return frames, fps / step


def capture_live(value: str, limit: int, output: Path) -> tuple[list[np.ndarray], float]:
    source = int(value) if value.isdigit() else value
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open live input: {value}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    try:
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame.copy())
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"live input produced no frames: {value}")
    write_video(output, frames, reported_fps)
    return frames, reported_fps


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create video: {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


def label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(result, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return result


def image_tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    return tensor_summary(value.detach().cpu())


def run_model_from_shared_normalized(model, normalized: torch.Tensor, dtype: torch.dtype):
    token_capture = {}

    def capture_token(_module, _inputs, output):
        token_capture["tokens"] = output.detach()

    handle = model.head.transformer.register_forward_hook(capture_token)
    try:
        backbone_input = normalized.to(dtype=dtype)
        if hasattr(model.backbone, "forward") and model.__class__.__name__ == "PearStudentPipeline":
            features, stages = model.backbone(backbone_input, return_stages=True)
        else:
            features = model.backbone(backbone_input)
            stages = {}
        # The deployed reduced-precision path hands FP32 features to the FP32 head.
        outputs = model.head(features.float())
    finally:
        handle.remove()
    return outputs, features, token_capture["tokens"], stages


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    skew = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    axis = skew / (2.0 * torch.sin(angle).abs().clamp_min(1e-6)[..., None])
    return axis * angle[..., None]


def converted_guava_parameters(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    body = output["body_param"]
    flame = output["flame_param"]
    result = {key: matrix_to_axis_angle(body[key]) for key in ROTATION_KEYS}
    result["exp"] = body["exp"]
    for key in ("expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params"):
        result[key] = flame[key]
    result["camera"] = output["pd_cam"]
    return result


def rotation_validity(output: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in ROTATION_KEYS:
        matrix = output["body_param"][key].detach().float()
        identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        orthogonality = torch.matmul(matrix.transpose(-1, -2), matrix) - identity
        result[key] = {
            "max_orthogonality_error": float(orthogonality.abs().max()),
            "mean_abs_determinant_error": float((torch.linalg.det(matrix) - 1.0).abs().mean()),
        }
    return result


def sequence_metrics(teacher_values: list[torch.Tensor], student_values: list[torch.Tensor]):
    teacher = torch.cat([value.detach().float().cpu() for value in teacher_values], dim=0)
    student = torch.cat([value.detach().float().cpu() for value in student_values], dim=0)
    teacher_delta = (teacher[1:] - teacher[:-1]).abs().mean() if len(teacher) > 1 else teacher.new_zeros(())
    student_delta = (student[1:] - student[:-1]).abs().mean() if len(student) > 1 else student.new_zeros(())
    teacher_range = (teacher.max(dim=0).values - teacher.min(dim=0).values).abs().mean()
    student_range = (student.max(dim=0).values - student.min(dim=0).values).abs().mean()
    finite_student = student[torch.isfinite(student)]
    return {
        "shape_per_frame": list(teacher.shape[1:]),
        "teacher_temporal_std_mean": float(teacher.std(dim=0, unbiased=False).mean()),
        "student_temporal_std_mean": float(student.std(dim=0, unbiased=False).mean()),
        "teacher_frame_delta_mean": float(teacher_delta),
        "student_frame_delta_mean": float(student_delta),
        "teacher_motion_range_mean": float(teacher_range),
        "student_motion_range_mean": float(student_range),
        "student_to_teacher_mae": float((student - teacher).abs().mean()),
        "teacher_motion_amplitude_retained_percent": (
            100.0 * float(student_delta / teacher_delta) if float(teacher_delta) > 1e-12 else None
        ),
        "teacher_nan_count": int(torch.isnan(teacher).sum()),
        "teacher_inf_count": int(torch.isinf(teacher).sum()),
        "student_nan_count": int(torch.isnan(student).sum()),
        "student_inf_count": int(torch.isinf(student).sum()),
        "student_near_zero_count": int((finite_student.abs() < 1e-6).sum()),
        "student_near_zero_fraction": (
            float((finite_student.abs() < 1e-6).float().mean()) if finite_student.numel() else None
        ),
        "student_large_magnitude_count_abs_ge_10": int((finite_student.abs() >= 10.0).sum()),
    }


def activation_sequence_metrics(values: list[torch.Tensor]) -> dict[str, Any]:
    activations = torch.cat([value.detach().float().cpu() for value in values], dim=0)
    flattened = activations.flatten(1)
    temporal_std = flattened.std(dim=0, unbiased=False)
    magnitude = flattened.abs().mean()
    return {
        "shape_per_frame": list(activations.shape[1:]),
        "temporal_std_mean": float(temporal_std.mean()),
        "mean_abs_magnitude": float(magnitude),
        "signal_ratio_std_over_magnitude": float(temporal_std.mean() / magnitude.clamp_min(1e-12)),
        "temporally_dead_element_fraction_std_lt_1e-4": float(
            (temporal_std < 1e-4).float().mean()
        ),
        "nan_count": int(torch.isnan(activations).sum()),
        "inf_count": int(torch.isinf(activations).sum()),
    }


def student_batch_norm_ablation(
    student,
    normalized_frames: list[torch.Tensor],
    teacher_accumulated: dict[str, list[torch.Tensor]],
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Compare eval running statistics with batch statistics without retaining mutations."""
    batch_norms = [module for module in student.backbone.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    if not batch_norms or len(normalized_frames) < 2:
        return {"available": False, "reason": "student has no BatchNorm2d or fewer than two frames"}
    snapshots = [
        (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
        for module in batch_norms
    ]
    student.eval()
    for module in batch_norms:
        module.train()
    try:
        with torch.inference_mode():
            output, features, tokens, stages = run_model_from_shared_normalized(
                student, torch.cat(normalized_frames, dim=0), dtype
            )
    finally:
        for module, (running_mean, running_var, batches) in zip(batch_norms, snapshots):
            module.running_mean.copy_(running_mean)
            module.running_var.copy_(running_var)
            module.num_batches_tracked.copy_(batches)
        student.eval()

    converted = converted_guava_parameters(output)
    result = {
        "available": True,
        "mode": "student eval model with backbone BatchNorm2d layers using current-sequence batch statistics",
        "batch_size": len(normalized_frames),
        "backbone_features": activation_sequence_metrics([features]),
        "decoder_token": activation_sequence_metrics([tokens]),
        "student_backbone_stages": {
            name: activation_sequence_metrics([value]) for name, value in stages.items()
        },
        "parameter_statistics": {},
    }
    for key, student_value in converted.items():
        teacher_key = f"guava.{key}"
        teacher_values = teacher_accumulated.get(teacher_key)
        if teacher_values is not None and student_value.shape[0] == len(teacher_values):
            result["parameter_statistics"][teacher_key] = sequence_metrics(
                teacher_values, [student_value]
            )
    return result


def pose_record_for_live(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    def matrix_to_rotation_6d(matrix):
        return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)

    body, flame = output["body_param"], output["flame_param"]
    params = {key: matrix_to_rotation_6d(body[key].float())[0] for key in ROTATION_KEYS}
    params["exp"] = body["exp"][0].float()
    for key in ("expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params"):
        params[key] = flame[key][0].float()
    return params


def initialize_guava_renderer(args, warmup_output):
    cleanup_pear_modules()
    from main.live_pear_guava import TargetBuilder, initialize_guava

    runtime_args = SimpleNamespace(
        source_data_path=args.source_data_path.resolve(),
        model_path=args.model_path.resolve(),
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
    renderer, identity, source_dataset = initialize_guava(
        runtime_args, pose_record_for_live(warmup_output)
    )
    return renderer, TargetBuilder(identity, runtime_args), source_dataset


def render_output(renderer, target_builder, output):
    target = target_builder(pose_record_for_live(output), time.perf_counter())
    return renderer.render(target).cpu().numpy()


def compare_category(
    name: str,
    frames: list[np.ndarray],
    fps: float,
    teacher,
    student,
    args,
) -> dict[str, Any]:
    category_dir = args.output_dir / name
    category_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.precision
    ]
    accumulated = {"teacher": defaultdict(list), "student": defaultdict(list)}
    activation_series = {"teacher": defaultdict(list), "student": defaultdict(list)}
    normalized_frames = []
    crop_video = []
    teacher_renders, student_renders, avatar_pairs = [], [], []
    first_frame_report = None
    guava = None

    with torch.inference_mode():
        for index, frame in enumerate(frames):
            stages = shared_preprocess(frame, args.input_crop_scale, args.device)
            normalized = stages["normalized_tensor"]
            normalized_frames.append(normalized.detach().clone())
            teacher_output, teacher_features, teacher_tokens, _ = run_model_from_shared_normalized(
                teacher, normalized, dtype
            )
            student_output, student_features, student_tokens, student_stages = run_model_from_shared_normalized(
                student, normalized, dtype
            )
            activation_series["teacher"]["backbone_features"].append(teacher_features)
            activation_series["teacher"]["decoder_token"].append(teacher_tokens)
            activation_series["student"]["backbone_features"].append(student_features)
            activation_series["student"]["decoder_token"].append(student_tokens)
            for stage_name, activation in student_stages.items():
                activation_series["student"][f"backbone.{stage_name}"].append(activation)
            teacher_flat = tensor_tree(teacher_output)
            student_flat = tensor_tree(student_output)
            teacher_converted = {
                f"guava.{key}": value for key, value in converted_guava_parameters(teacher_output).items()
            }
            student_converted = {
                f"guava.{key}": value for key, value in converted_guava_parameters(student_output).items()
            }
            teacher_flat.update(teacher_converted)
            student_flat.update(student_converted)
            for key in sorted(set(teacher_flat) & set(student_flat)):
                if teacher_flat[key].shape == student_flat[key].shape:
                    accumulated["teacher"][key].append(teacher_flat[key])
                    accumulated["student"][key].append(student_flat[key])

            crop = stages["model_bgr"]
            crop_video.append(
                cv2.hconcat((label(crop, "teacher input"), label(crop, "student input: exact same tensor")))
            )
            if first_frame_report is None:
                first_frame_report = {
                    "original_rgb": image_tensor_stats(
                        torch.as_tensor(cv2.cvtColor(stages["original_bgr"], cv2.COLOR_BGR2RGB))
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    ),
                    "detected_or_center_crop_rgb": image_tensor_stats(
                        torch.as_tensor(cv2.cvtColor(stages["person_crop_bgr"], cv2.COLOR_BGR2RGB))
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    ),
                    "resized_model_rgb": image_tensor_stats(stages["model_tensor"].cpu()),
                    "shared_normalized_input": image_tensor_stats(normalized.cpu()),
                    "teacher_backbone_features": image_tensor_stats(teacher_features.cpu()),
                    "student_backbone_features": image_tensor_stats(student_features.cpu()),
                    "teacher_transformer_tokens": image_tensor_stats(teacher_tokens.cpu()),
                    "student_transformer_tokens": image_tensor_stats(student_tokens.cpu()),
                    "student_backbone_stages": {
                        key: image_tensor_stats(value.cpu()) for key, value in student_stages.items()
                    },
                    "teacher_outputs": {
                        key: tensor_summary(value.cpu()) for key, value in teacher_flat.items()
                    },
                    "student_outputs": {
                        key: tensor_summary(value.cpu()) for key, value in student_flat.items()
                    },
                    "identical_input_assertion": {
                        "same_tensor_object_used_for_both_backbones": True,
                        "max_abs_difference": 0.0,
                    },
                    "rotation_validity": {
                        "teacher": rotation_validity(teacher_output),
                        "student": rotation_validity(student_output),
                    },
                }

            if args.render_guava:
                if guava is None:
                    guava = initialize_guava_renderer(args, teacher_output)
                renderer, target_builder, _source_dataset = guava
                teacher_render = render_output(renderer, target_builder, teacher_output)
                student_render = render_output(renderer, target_builder, student_output)
                teacher_renders.append(teacher_render)
                student_renders.append(student_render)
                avatar_pairs.append(
                    cv2.hconcat(
                        (label(teacher_render, "PEAR teacher -> GUAVA"), label(student_render, "student -> GUAVA"))
                    )
                )
            print(f"{name}: {index + 1}/{len(frames)}", flush=True)

    metrics = {
        key: sequence_metrics(accumulated["teacher"][key], accumulated["student"][key])
        for key in accumulated["teacher"]
    }
    identity_variation = {
        key: metrics[key]
        for key in IDENTITY_KEYS
        if key in metrics
    }
    activation_metrics = {
        model_name: {
            stage_name: activation_sequence_metrics(values)
            for stage_name, values in stages.items()
        }
        for model_name, stages in activation_series.items()
    }
    batch_norm_ablation = student_batch_norm_ablation(
        student, normalized_frames, accumulated["teacher"], dtype
    )
    report = {
        "category": name,
        "frame_count": len(frames),
        "source_fps": fps,
        "precision": args.precision,
        "compiled": False,
        "first_frame_stage_statistics": first_frame_report,
        "sequence_parameter_statistics": metrics,
        "sequence_activation_statistics": activation_metrics,
        "student_batch_norm_ablation": batch_norm_ablation,
        "identity_parameter_variation": identity_variation,
        "guava_rendered": args.render_guava,
    }
    (category_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    for key, values in metrics.items():
        rows.append({"parameter": key, **values})
    with (category_dir / "parameter_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["parameter"])
        writer.writeheader()
        writer.writerows(rows)
    write_video(category_dir / "input_crops_side_by_side.mp4", crop_video, fps)
    if args.render_guava:
        write_video(category_dir / "teacher_driven_guava.mp4", teacher_renders, fps)
        write_video(category_dir / "student_driven_guava.mp4", student_renders, fps)
        write_video(category_dir / "avatar_side_by_side.mp4", avatar_pairs, fps)
        guava[2]._lmdb_engine.close()
    return report


def diagnosis(reports: dict[str, Any]) -> dict[str, Any]:
    key = "guava.body_pose"
    category_motion = {}
    for name, report in reports.items():
        values = report["sequence_parameter_statistics"].get(key)
        if values:
            category_motion[name] = values["teacher_motion_amplitude_retained_percent"]
    training_retained = category_motion.get("training_distribution")
    webcam_values = [
        value for name, value in category_motion.items() if name != "training_distribution" and value is not None
    ]
    if training_retained is not None and training_retained < 25.0:
        gate = "student_fails_training_distribution"
        next_action = "Investigate training collapse/output learning; checkpoint loading and shape compatibility already pass."
    elif training_retained is not None and webcam_values and max(webcam_values) < 0.5 * training_retained:
        gate = "webcam_domain_or_preprocessing_shift"
        next_action = "Student retains motion on training crops but loses it on webcam crops; fix crop/domain alignment."
    elif training_retained is None and webcam_values and max(webcam_values) < 25.0:
        gate = "webcam_motion_collapse_training_distribution_required"
        next_action = (
            "The student loses most teacher motion before GUAVA conversion on prerecorded or "
            "capture-first live frames. Run the same tool with the original validation manifest to distinguish "
            "training collapse from webcam domain shift."
        )
    elif category_motion:
        gate = "parameter_motion_present_check_guava_conversion_and_video"
        next_action = "Inspect converted GUAVA metrics and side-by-side renders before any routing work."
    else:
        gate = "insufficient_inputs"
        next_action = "Run at least one multi-frame category."
    return {
        "body_motion_retained_percent_by_category": category_motion,
        "acceptance_gate_result": gate,
        "next_action": next_action,
        "routing_allowed": False,
        "reason": "Routing remains blocked until direct student-to-GUAVA animation is visibly correct.",
    }


def main() -> None:
    args = parse_args()
    if args.frames < 2 or args.frame_step < 1:
        raise ValueError("--frames must be at least 2 and --frame-step must be positive")
    if not any((args.training_manifest, args.webcam_video, args.live_input)):
        raise ValueError("provide --training-manifest, --webcam-video, and/or --live-input")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this comparison in the host guava environment")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {}
    if args.training_manifest:
        inputs["training_distribution"] = read_manifest(
            args.training_manifest, args.frames, args.frame_step
        )
    if args.webcam_video:
        inputs["prerecorded_webcam"] = read_video(args.webcam_video, args.frames, args.frame_step)
    if args.live_input:
        live_path = args.output_dir / "live_webcam_captured_before_inference.mp4"
        inputs["live_webcam_preprocessing"] = capture_live(args.live_input, args.frames, live_path)

    teacher, student, teacher_path, student_config_path = load_models(args)
    reports = {}
    try:
        for name, (frames, fps) in inputs.items():
            reports[name] = compare_category(name, frames, fps, teacher, student, args)
    finally:
        cleanup_pear_modules()

    summary = {
        "teacher_checkpoint": str(teacher_path),
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "student_config": str(student_config_path),
        "device": args.device,
        "precision": args.precision,
        "compile": "disabled",
        "shared_preprocessing": (
            "One RGB [0,1] 256x256 tensor is ImageNet-normalized and center-cropped to "
            "[B,3,256,192] once; that same tensor object is passed to both backbones."
        ),
        "categories": {name: str((args.output_dir / name / "comparison.json")) for name in reports},
        "diagnosis": diagnosis(reports),
    }
    summary_path = args.output_dir / "teacher_student_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Comparison summary: {summary_path}")
    print(json.dumps(summary["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
