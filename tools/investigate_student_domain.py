#!/usr/bin/env python
"""Run controlled PEAR L70 crop/domain and BatchNorm adaptation experiments.

This tool is intentionally independent of AvatarBudget's scout, temporal model,
and router. It tests student correctness before any budget-policy decisions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from tools.compare_teacher_student import (
    DEFAULT_SOURCE,
    DEFAULT_STUDENT,
    cleanup_pear_modules,
    converted_guava_parameters,
    initialize_guava_renderer,
    label,
    load_models,
    read_video,
    render_output,
    run_model_from_shared_normalized,
    sequence_metrics,
    shared_preprocess,
    write_video,
)
from tools.domain_investigation_utils import (
    embed_in_canvas,
    mean_frame_pixel_motion,
    square_person_crop,
)
from tools.pear_debug_utils import tensor_tree


ROOT = Path(__file__).resolve().parents[1]
DYNAMIC_KEYS = (
    "guava.global_pose",
    "guava.body_pose",
    "guava.left_hand_pose",
    "guava.right_hand_pose",
    "guava.expression_params",
    "guava.jaw_params",
    "guava.eye_pose_params",
    "guava.eyelid_params",
)
KEY_LABELS = {
    "guava.global_pose": "global",
    "guava.body_pose": "body",
    "guava.left_hand_pose": "left hand",
    "guava.right_hand_pose": "right hand",
    "guava.expression_params": "expression",
    "guava.jaw_params": "jaw",
    "guava.eye_pose_params": "eyes",
    "guava.eyelid_params": "eyelids",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--input-label", default="webcam")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--crop-scales", type=float, nargs="+", default=(1.0, 1.2, 1.4, 1.8))
    parser.add_argument(
        "--manual-person-box",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional fixed xyxy person box for a tight-crop condition.",
    )
    parser.add_argument("--manual-crop-scale", type=float, default=1.25)
    parser.add_argument("--student-config", type=Path, default=Path("configs/student_l70.yaml"))
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-config", default="configs/infer.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--download-teacher", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-iterations", type=int, default=100)
    parser.add_argument("--render-guava", action="store_true")
    parser.add_argument("--source-data-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-path", type=Path, default=ROOT / "assets/GUAVA")
    parser.add_argument("--render-size", type=int, choices=(256, 512), default=256)
    parser.add_argument(
        "--validation-plan",
        type=Path,
        help="JSON from inspect_student_manifest.py; enables original-distribution evidence.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/domain_investigation"
    )
    return parser.parse_args()


def preprocess_crop(crop_bgr: np.ndarray, device: str) -> dict[str, Any]:
    return shared_preprocess(crop_bgr, crop_scale=1.0, device=device)


def build_conditions(
    frames: list[np.ndarray], args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    conditions: dict[str, list[dict[str, Any]]] = {}
    for scale in args.crop_scales:
        name = f"center_scale_{scale:g}"
        conditions[name] = [shared_preprocess(frame, scale, args.device) for frame in frames]
    if args.manual_person_box is not None:
        conditions["tight_manual_person"] = [
            preprocess_crop(
                square_person_crop(frame, args.manual_person_box, args.manual_crop_scale),
                args.device,
            )
            for frame in frames
        ]
    return conditions


def flatten_outputs(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    values = tensor_tree(output)
    values.update(
        {f"guava.{key}": value for key, value in converted_guava_parameters(output).items()}
    )
    return values


def output_metrics(teacher_output: dict[str, Any], student_output: dict[str, Any]) -> dict[str, Any]:
    teacher = flatten_outputs(teacher_output)
    student = flatten_outputs(student_output)
    result = {}
    for key in sorted(set(teacher) & set(student)):
        if teacher[key].shape == student[key].shape:
            result[key] = sequence_metrics([teacher[key]], [student[key]])
    return result


def infer_batched(model, tensors: list[torch.Tensor], dtype: torch.dtype, batch_size: int):
    output_parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.cat(tensors[start : start + batch_size], dim=0)
            output, _features, _tokens, _stages = run_model_from_shared_normalized(
                model, batch, dtype
            )
            for key, value in tensor_tree(output).items():
                output_parts[key].append(value.detach().clone())
    merged = {key: torch.cat(values, dim=0) for key, values in output_parts.items()}
    template: dict[str, Any] = {}
    for dotted_key, value in merged.items():
        cursor = template
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return template


def slice_output(output: Any, index: int) -> Any:
    if torch.is_tensor(output):
        return output[index : index + 1]
    if isinstance(output, dict):
        return {key: slice_output(value, index) for key, value in output.items()}
    return output


def batch_norm_layers(student) -> list[torch.nn.BatchNorm2d]:
    return [
        module for module in student.backbone.modules() if isinstance(module, torch.nn.BatchNorm2d)
    ]


@contextmanager
def preserve_batch_norm(student):
    layers = batch_norm_layers(student)
    states = [
        {
            "running_mean": layer.running_mean.detach().clone(),
            "running_var": layer.running_var.detach().clone(),
            "num_batches_tracked": layer.num_batches_tracked.detach().clone(),
            "momentum": layer.momentum,
            "training": layer.training,
        }
        for layer in layers
    ]
    try:
        yield layers
    finally:
        for layer, state in zip(layers, states):
            layer.running_mean.copy_(state["running_mean"])
            layer.running_var.copy_(state["running_var"])
            layer.num_batches_tracked.copy_(state["num_batches_tracked"])
            layer.momentum = state["momentum"]
            layer.train(state["training"])
        student.eval()


def infer_with_current_batch_bn(student, tensors, dtype, batch_size):
    with preserve_batch_norm(student) as layers:
        student.eval()
        for layer in layers:
            layer.train()
        return infer_batched(student, tensors, dtype, batch_size)


def infer_with_recalibrated_bn(student, tensors, dtype, batch_size):
    """Recompute cumulative BN statistics from unlabeled deployment crops, then infer."""
    with preserve_batch_norm(student) as layers:
        student.eval()
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        with torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = torch.cat(tensors[start : start + batch_size], dim=0)
                run_model_from_shared_normalized(student, batch, dtype)
        student.eval()
        return infer_batched(student, tensors, dtype, batch_size)


def retained_summary(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        KEY_LABELS[key]: metrics.get(key, {}).get("teacher_motion_amplitude_retained_percent")
        for key in DYNAMIC_KEYS
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    values = [
        metrics.get(key, {}).get("teacher_motion_amplitude_retained_percent")
        for key in (
            "guava.body_pose",
            "guava.left_hand_pose",
            "guava.right_hand_pose",
            "guava.expression_params",
        )
    ]
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return statistics.fmean(finite) if finite else float("-inf")


def benchmark_student(student, tensor, dtype, warmup: int, iterations: int) -> dict[str, Any]:
    if warmup < 1 or iterations < 1:
        return {"skipped": True}
    for _ in range(warmup):
        with torch.inference_mode():
            run_model_from_shared_normalized(student, tensor, dtype)
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)
    cuda_ms = []
    wall_ms = []
    for _ in range(iterations):
        start_event = end_event = None
        if tensor.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        with torch.inference_mode():
            run_model_from_shared_normalized(student, tensor, dtype)
        if end_event is not None:
            end_event.record()
            end_event.synchronize()
            cuda_ms.append(float(start_event.elapsed_time(end_event)))
        wall_ms.append((time.perf_counter() - start) * 1000.0)

    def summarize(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(array.mean()),
            "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
            "p99_ms": float(np.percentile(array, 99)),
        }

    return {
        "input_shape": list(tensor.shape),
        "precision": str(dtype),
        "batch_size": 1,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "cuda": summarize(cuda_ms) if cuda_ms else None,
        "wall": summarize(wall_ms),
    }


def make_crop_video(conditions: dict[str, list[dict[str, Any]]], fps: float, path: Path) -> None:
    names = list(conditions)
    frames = []
    for index in range(len(next(iter(conditions.values())))):
        panels = [label(conditions[name][index]["model_bgr"], name) for name in names]
        frames.append(cv2.hconcat(panels))
    write_video(path, frames, fps)


def render_variants(
    args,
    fps: float,
    condition_name: str,
    condition: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    comparison_name: str = "webcam_before_after.mp4",
    video_prefix: str = "guava",
) -> dict[str, Any]:
    teacher_first = slice_output(outputs["teacher"], 0)
    renderer, target_builder, source_dataset = initialize_guava_renderer(args, teacher_first)
    rendered: dict[str, list[np.ndarray]] = defaultdict(list)
    try:
        for variant, output in outputs.items():
            if variant == "frozen_bn_control":
                continue
            for index in range(len(condition)):
                rendered[variant].append(
                    render_output(renderer, target_builder, slice_output(output, index))
                )
        rendered["frozen_bn_control"] = [frame.copy() for frame in rendered["stored_bn"]]
    finally:
        source_dataset._lmdb_engine.close()

    input_frames = [stage["model_bgr"] for stage in condition]
    comparison = []
    for index, crop in enumerate(input_frames):
        comparison.append(
            cv2.hconcat(
                (
                    label(crop, f"{args.input_label}: {condition_name}"),
                    label(rendered["teacher"][index], "teacher -> GUAVA"),
                    label(rendered["stored_bn"][index], "L70 stored BN"),
                    label(rendered["recalibrated_bn"][index], "L70 recalibrated BN"),
                )
            )
        )
    write_video(args.output_dir / comparison_name, comparison, fps)
    for variant, frames in rendered.items():
        write_video(args.output_dir / f"{video_prefix}_{variant}.mp4", frames, fps)
    return {
        variant: {
            "mean_consecutive_frame_pixel_motion": mean_frame_pixel_motion(frames),
            "frame_count": len(frames),
        }
        for variant, frames in rendered.items()
    }


def read_validation_plan(path: Path, limit: int) -> tuple[list[np.ndarray], float, dict[str, Any]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    clips = plan.get("sequential_clips", [])
    if not clips:
        raise ValueError("validation plan contains no sequential clips")
    selected = clips[0][:limit]
    frames = []
    for item in selected:
        frame = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(item["path"])
        frames.append(frame)
    return frames, 30.0, plan


def write_report(path: Path, report: dict[str, Any]) -> None:
    crop_rows = []
    for name, condition in report["crop_conditions"].items():
        retained = condition["stored_bn"]["motion_retained_percent"]
        crop_rows.append(
            f"| {name} | {retained.get('global')} | {retained.get('body')} | "
            f"{retained.get('left hand')} | {retained.get('right hand')} | "
            f"{retained.get('expression')} |"
        )
    bn = report["selected_crop_bn_comparison"]
    bn_rows = []
    for variant, values in bn.items():
        retained = values["motion_retained_percent"]
        pixel = values.get("rendered_pixel_motion")
        bn_rows.append(
            f"| {variant} | {retained.get('body')} | {retained.get('left hand')} | "
            f"{retained.get('right hand')} | {retained.get('expression')} | {pixel} |"
        )
    validation = report["original_validation"]
    overfit = report["webcam_overfit"]
    decision = report["decision"]
    text = f"""# PEAR L70 Domain Investigation

Status: **{decision['status']}**

This investigation isolates PEAR teacher/student correctness. AvatarBudget scout,
temporal prediction, and routing were not executed or modified. Solving student
motion does not establish 50 FPS.

## Evidence boundaries

- Input: `{report['input']['path']}` ({report['input']['label']})
- Frames: {report['input']['frame_count']} at source {report['input']['fps']} FPS
- Shared model tensor: `[B, 3, 256, 192]`; teacher and student receive the exact same tensor per condition.
- Original validation: {validation['status']}
- Webcam overfit: {overfit['status']}

## Crop comparison

Values are student/teacher frame-delta percentages. Very small teacher motion can
make a ratio unstable, so the exact crop video must be inspected with the numbers.

| condition | global % | body % | left hand % | right hand % | expression % |
|---|---:|---:|---:|---:|---:|
{chr(10).join(crop_rows)}

Best measured crop by mean body/hand/expression retention: **{report['selected_crop']}**.

## BatchNorm comparison

| mode | body % | left hand % | right hand % | expression % | GUAVA pixel motion |
|---|---:|---:|---:|---:|---:|
{chr(10).join(bn_rows)}

`frozen_bn_control` is deliberately identical to stored-BN inference because no
weights are optimized in this diagnostic. It defines the correct frozen-BN baseline
for a later fine-tuning experiment; it is not presented as an adaptation result.

## Answers

1. Original validation behavior: **{validation['answer']}**
2. Sequential validation: **{validation['sequential_answer']}**
3. Best crop: **{report['selected_crop']}** on this input.
4. Tight crop effect: **{decision['tight_crop_answer']}**
5. BatchNorm effect: **{decision['batch_norm_answer']}**
6. Small webcam overfit: **{overfit['answer']}**
7. Recommended action: **{decision['recommended_action']}**
8. Adapted student directly animates GUAVA: **{decision['adapted_guava_answer']}**
9. Remaining student latency: `{report['student_latency']}`

## Required next evidence

{chr(10).join(f'- {item}' for item in decision['required_next_evidence'])}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.frames < 2 or args.frame_step < 1 or args.batch_size < 2:
        raise ValueError("frames >= 2, frame-step >= 1, and batch-size >= 2 are required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run in the host guava environment")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(args.input_video.resolve(), args.frames, args.frame_step)
    conditions = build_conditions(frames, args)
    make_crop_video(conditions, fps, args.output_dir / "crop_comparison.mp4")

    teacher, student, teacher_path, student_config_path = load_models(args)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.precision
    ]
    condition_reports = {}
    condition_outputs = {}
    try:
        for name, stages in conditions.items():
            tensors = [stage["normalized_tensor"] for stage in stages]
            teacher_output = infer_batched(teacher, tensors, dtype, args.batch_size)
            stored_output = infer_batched(student, tensors, dtype, args.batch_size)
            current_output = infer_with_current_batch_bn(
                student, tensors, dtype, args.batch_size
            )
            recalibrated_output = infer_with_recalibrated_bn(
                student, tensors, dtype, args.batch_size
            )
            variants = {
                "stored_bn": stored_output,
                "current_batch_bn": current_output,
                "recalibrated_bn": recalibrated_output,
                "frozen_bn_control": stored_output,
            }
            condition_reports[name] = {}
            for variant, output in variants.items():
                metrics = output_metrics(teacher_output, output)
                condition_reports[name][variant] = {
                    "motion_retained_percent": retained_summary(metrics),
                    "full_parameter_metrics": metrics,
                }
            condition_outputs[name] = {"teacher": teacher_output, **variants}
            print(f"completed {name}", flush=True)

        selected = max(
            condition_reports,
            key=lambda name: score_metrics(
                condition_reports[name]["stored_bn"]["full_parameter_metrics"]
            ),
        )
        latency = benchmark_student(
            student,
            conditions[selected][0]["normalized_tensor"],
            dtype,
            args.latency_warmup,
            args.latency_iterations,
        )
        rendered = None
        if args.render_guava:
            rendered = render_variants(
                args, fps, selected, conditions[selected], condition_outputs[selected]
            )
            for variant, values in condition_reports[selected].items():
                values["rendered_pixel_motion"] = rendered[variant][
                    "mean_consecutive_frame_pixel_motion"
                ]

        validation_report = {
            "status": "not run: no --validation-plan supplied",
            "answer": "Unknown on this machine; the original manifest/data are on the server.",
            "sequential_answer": "Unknown; old side-by-side stills do not establish temporal tracking.",
        }
        if args.validation_plan:
            validation_frames, validation_fps, plan = read_validation_plan(
                args.validation_plan.resolve(), args.frames
            )
            validation_conditions = {
                "original_crop": [preprocess_crop(frame, args.device) for frame in validation_frames],
                "wide_640x480": [
                    shared_preprocess(embed_in_canvas(frame), 1.0, args.device)
                    for frame in validation_frames
                ],
            }
            validation_metrics = {}
            validation_outputs = {}
            for name, stages in validation_conditions.items():
                tensors = [stage["normalized_tensor"] for stage in stages]
                teacher_output = infer_batched(teacher, tensors, dtype, args.batch_size)
                student_output = infer_batched(student, tensors, dtype, args.batch_size)
                metrics = output_metrics(teacher_output, student_output)
                validation_metrics[name] = {
                    "motion_retained_percent": retained_summary(metrics),
                    "full_parameter_metrics": metrics,
                }
                validation_outputs[name] = (teacher_output, student_output)
            validation_report = {
                "status": "one deterministic sequential clip evaluated",
                "answer": "See original_crop per-frame and sequential metrics; broaden to all planned clips before a global conclusion.",
                "sequential_answer": "Measured on one planned clip; aggregate all 10+ clips for the requested decision.",
                "manifest": plan.get("manifest"),
                "conditions": validation_metrics,
            }
            if args.render_guava:
                teacher_output, student_output = validation_outputs["original_crop"]
                render_inputs = {
                    "teacher": teacher_output,
                    "stored_bn": student_output,
                    "current_batch_bn": student_output,
                    "recalibrated_bn": student_output,
                    "frozen_bn_control": student_output,
                }
                render_variants(
                    SimpleNamespace(**{**vars(args), "input_label": "original validation"}),
                    validation_fps,
                    "original_crop",
                    validation_conditions["original_crop"],
                    render_inputs,
                    comparison_name="original_validation_comparison.mp4",
                    video_prefix="validation_guava",
                )

        selected_values = condition_reports[selected]
        recal_score = score_metrics(selected_values["recalibrated_bn"]["full_parameter_metrics"])
        stored_score = score_metrics(selected_values["stored_bn"]["full_parameter_metrics"])
        has_tight = "tight_manual_person" in condition_reports
        tight_answer = (
            "Measured; inspect tight_manual_person against center-scale rows."
            if has_tight
            else "Not tested because no manual person box or detector track was supplied."
        )
        pixel_stored = selected_values["stored_bn"].get("rendered_pixel_motion")
        pixel_recal = selected_values["recalibrated_bn"].get("rendered_pixel_motion")
        bn_substantial = (
            recal_score >= max(stored_score * 2.0, stored_score + 20.0)
            and pixel_stored is not None
            and pixel_recal is not None
            and pixel_recal >= pixel_stored * 1.5
        )
        decision = {
            "status": "preliminary local domain test; original validation and webcam overfit remain gated",
            "tight_crop_answer": tight_answer,
            "batch_norm_answer": (
                "Substantial parameter and rendered-motion improvement measured."
                if bn_substantial
                else "No substantial direct-render improvement established by the configured gate."
            ),
            "recommended_action": (
                "Do not choose preprocessing, L70 adaptation, or v2 until original validation and a motion-rich webcam overfit are run."
            ),
            "adapted_guava_answer": (
                "No. BN recalibration is an unlabeled statistics test, not a trained adapted checkpoint."
            ),
            "required_next_evidence": [
                "Run inspect_student_manifest.py and this tool on the server validation manifest.",
                "Record 200-500 motion-rich deployment crops plus 50-100 held-out frames.",
                "Cache teacher labels and run the constrained L70 overfit before any v2 training.",
            ],
        }
        report = {
            "experiment": "PEAR L70 crop/domain and BatchNorm investigation",
            "teacher_checkpoint": str(teacher_path),
            "student_checkpoint": str(args.student_checkpoint.resolve()),
            "student_config": str(student_config_path),
            "device": args.device,
            "precision": args.precision,
            "input": {
                "path": str(args.input_video.resolve()),
                "label": args.input_label,
                "frame_count": len(frames),
                "fps": fps,
                "original_shape": list(frames[0].shape),
            },
            "tensor_contract": {
                "pre_letterbox": "uint8 BGR [H,W,3]",
                "model_rgb": "float32 RGB [B,3,256,256] in [0,1]",
                "normalized_backbone_input": "[B,3,256,192] ImageNet normalized",
            },
            "crop_conditions": condition_reports,
            "selected_crop": selected,
            "selected_crop_bn_comparison": selected_values,
            "rendered_motion": rendered,
            "student_latency": latency,
            "original_validation": validation_report,
            "webcam_overfit": {
                "status": "not run: requires a motion-rich 200-500/50-100 train/held-out dataset",
                "answer": "Unknown; no optimization was started from insufficient local data.",
            },
            "decision": decision,
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        write_report(args.output_dir / "final_report.md", report)
        print(f"report: {args.output_dir / 'final_report.md'}")
        print(f"metrics: {args.output_dir / 'metrics.json'}")
    finally:
        cleanup_pear_modules()


if __name__ == "__main__":
    main()
