#!/usr/bin/env python
"""Run the budget-aware PEAR/student -> GUAVA live avatar pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from avatarbudget import RenderLevel, RouteDecision, load_config
from avatarbudget.contracts import PARTS, Part, validate_pose_record
from avatarbudget.profiler import StageProfiler, percentile
from avatarbudget.rendering import ProgressiveGuavaRenderer
from avatarbudget.router import BudgetRouter, RenderImpactRouterNet
from avatarbudget.scheduler import HardFrameScheduler
from avatarbudget.scout import CheapScout
from avatarbudget.temporal import ConstantVelocityPredictor
from main.live_pear_guava import (
    DEFAULT_SOURCE,
    DTYPES,
    ROOT,
    TargetBuilder,
    initialize_guava,
    initialize_pear,
    open_capture,
)


DEFAULT_CONFIG = ROOT / "configs" / "avatarbudget_rtx3080_laptop.yaml"
STAGE_CLOCKS = {
    "capture": "cpu_perf_counter",
    "upload_preprocess": "cuda_event",
    "cheap_scout": "cuda_event",
    "temporal_prediction": "cuda_event",
    "budget_router_gpu": "cuda_event",
    "budget_router_host": "cpu_perf_counter",
    "pear_estimator": "cuda_event",
    "conditional_pose_merge": "cuda_event",
    "ehm_target_conversion": "cuda_event",
    "guava_deformation": "cuda_event",
    "gaussian_rasterization": "cuda_event",
    "neural_refiner": "cuda_event",
    "readback": "cpu_perf_counter",
    "display": "cpu_perf_counter",
    "end_to_end_work": "cpu_perf_counter",
    "frame_release_period": "cpu_perf_counter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", default="0", help="webcam index, URL, or video file")
    parser.add_argument("--source_data_path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model_path", type=Path, default=ROOT / "assets" / "GUAVA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render_size", type=int, choices=(256, 512), default=256)
    parser.add_argument(
        "--input_crop_scale",
        type=float,
        default=1.0,
        help="Center-crop the live frame by this zoom factor before PEAR.",
    )
    parser.add_argument("--pear_backend", choices=("teacher", "student"), default="student")
    parser.add_argument("--student_config", type=Path, default=Path("configs/student_l70.yaml"))
    parser.add_argument("--student_ckpt", type=Path)
    parser.add_argument("--router_ckpt", type=Path)
    parser.add_argument("--precision", choices=tuple(DTYPES), default="fp16")
    parser.add_argument(
        "--compile_targets",
        nargs="*",
        choices=("pear", "deform", "refiner"),
        default=["pear", "refiner"],
    )
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/avatarbudget/live_report.json")
    parser.add_argument("--no_smooth", "--no-smooth", dest="smooth", action="store_false")
    parser.set_defaults(smooth=True)
    parser.add_argument(
        "--rest_occluded_parts",
        action="store_true",
        help="Put low-visibility parts into a neutral pose instead of predicting them.",
    )
    parser.add_argument(
        "--occlusion_visibility_threshold",
        type=float,
        default=0.25,
        help="Scout visibility below this value marks a part as occluded.",
    )
    parser.add_argument("--smooth_min_cutoff", type=float, default=2.0)
    parser.add_argument("--smooth_beta", type=float, default=0.3)
    parser.add_argument("--smooth_d_cutoff", type=float, default=1.0)
    args = parser.parse_args()

    # Compatibility fields consumed by the proven loaders in live_pear_guava.
    args.pipeline = "serial"
    args.pear_stride = 1
    args.window = 300
    args.capture_source_from_stream = False
    if args.pear_backend == "student" and args.student_ckpt is None:
        parser.error("--student_ckpt is required with --pear_backend student")
    return args


def load_router_checkpoint(path: Path | None, device: str):
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = RenderImpactRouterNet(
        feature_dim=int(checkpoint.get("feature_dim", 7)),
        hidden_dim=int(checkpoint.get("hidden_dim", 32)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def rgb_tensor(frame_bgr, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(frame_bgr, device=device)
    return tensor[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def initial_decision(device: str) -> RouteDecision:
    decision = RouteDecision(
        update_parts=frozenset(PARTS),
        forced_parts=frozenset(PARTS),
        risk=torch.ones(len(PARTS), device=device),
        render_level=RenderLevel.LOW,
        estimated_ms=0.0,
        deadline_feasible=False,
    )
    decision.validate()
    return decision


def shape_report(values: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    validate_pose_record(values)
    return {key: list(value.shape) for key, value in values.items()}


def neutral_pose_like(reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    validate_pose_record(reference)
    pose = {key: torch.zeros_like(value) for key, value in reference.items()}
    identity_6d = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        device=reference["body_pose"].device,
        dtype=reference["body_pose"].dtype,
    )
    for key in ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose"):
        pose[key] = identity_6d.expand_as(reference[key]).clone()
    return pose


def occluded_parts_from_scout(args, scout_output) -> frozenset[Part]:
    if not args.rest_occluded_parts:
        return frozenset()
    visibility = scout_output.visibility[0].detach()
    threshold = float(args.occlusion_visibility_threshold)
    return frozenset(part for index, part in enumerate(PARTS) if visibility[index] < threshold)


def report_payload(
    args,
    config,
    profiler,
    scheduler,
    frame_count,
    steady_work_ms,
    steady_period_ms,
    part_updates,
    render_levels,
    estimator_calls,
    duplicated_frames,
    occlusion_rest_counts,
    setup_seconds,
    shapes,
):
    work_count = len(steady_work_ms)
    period_count = len(steady_period_ms)
    measured = min(work_count, period_count)
    achieved_fps = 1000.0 / (sum(steady_period_ms) / period_count) if period_count else 0.0
    deadline_rate = (
        sum(value <= config.scheduler.deadline_ms for value in steady_work_ms) / work_count
        if work_count
        else 0.0
    )
    p99_work = percentile(steady_work_ms, 0.99)
    actual_hardware = torch.cuda.get_device_name(torch.device(args.device))
    hardware_match = "RTX 3080 Laptop GPU" in actual_hardware
    verified = (
        measured >= 100
        and hardware_match
        and achieved_fps >= config.scheduler.target_fps * 0.99
        and deadline_rate >= 0.99
        and p99_work <= config.scheduler.deadline_ms
    )
    stage_latency = profiler.as_dict()
    for name, clock in STAGE_CLOCKS.items():
        stage_latency.setdefault(
            name,
            {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "clock": clock},
        )
    stage_latency = {name: stage_latency[name] for name in STAGE_CLOCKS}
    return {
        "target_hardware": "RTX 3080 Laptop GPU",
        "actual_hardware": actual_hardware,
        "target_hardware_match": hardware_match,
        "target_fps": config.scheduler.target_fps,
        "deadline_ms": config.scheduler.deadline_ms,
        "frames_total": frame_count,
        "frames_measured_after_warmup": measured,
        "setup_seconds_excluded": setup_seconds,
        "pear_backend": args.pear_backend,
        "estimator_calls": estimator_calls,
        "estimator_skipped_frames": frame_count - estimator_calls,
        "duplicated_output_frames": duplicated_frames,
        "part_update_counts": {part.value: part_updates[part] for part in PARTS},
        "occlusion_rest_counts": {part.value: occlusion_rest_counts[part] for part in PARTS},
        "render_level_counts": {level.value: render_levels[level] for level in RenderLevel},
        "pose_tensor_shapes_unbatched": shapes,
        "stage_latency": stage_latency,
        "end_to_end": {
            "achieved_fps_from_frame_release_timestamps": achieved_fps,
            "work_mean_ms": sum(steady_work_ms) / work_count if work_count else 0.0,
            "work_p95_ms": percentile(steady_work_ms, 0.95),
            "work_p99_ms": p99_work,
            "deadline_met_fraction": deadline_rate,
            "deadline_misses": sum(value > config.scheduler.deadline_ms for value in steady_work_ms),
            "sustained_50fps_verified": verified,
            "verification_rule": (
                ">=100 post-warmup end-to-end frames, achieved FPS >=99% of target, "
                "deadline met on >=99% of frames, and end-to-end work p99 <= deadline"
            ),
        },
        "timing_scope": (
            "Setup is excluded. End-to-end work includes capture, upload/preprocess, always-on "
            "scout, routing, optional PEAR/student inference, temporal merge, GUAVA deformation, "
            "Gaussian rasterization, optional refinement, synchronization, and readback. GUI "
            f"display is {'included' if not args.no_display else 'disabled for this run'}."
        ),
    }


def print_progress(payload):
    end = payload["end_to_end"]
    stages = payload["stage_latency"]
    stage_text = " | ".join(
        f"{name} {values['mean_ms']:.2f}/{values['p95_ms']:.2f}/{values['p99_ms']:.2f} ms"
        for name, values in stages.items()
    )
    print(
        f"frames={payload['frames_total']} measured={payload['frames_measured_after_warmup']} "
        f"e2e={end['achieved_fps_from_frame_release_timestamps']:.2f} FPS "
        f"deadline={100.0 * end['deadline_met_fraction']:.1f}% | {stage_text}",
        flush=True,
    )


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("The integrated PEAR/GUAVA live pipeline requires CUDA")
    config = load_config(args.config)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    setup_start = time.perf_counter()
    print("Loading PEAR estimator...")
    pear, warmup_params = initialize_pear(args)
    print("Loading the cached source identity and creating the GUAVA avatar...")
    base_renderer, identity, source_dataset = initialize_guava(args, warmup_params)
    setup_seconds = time.perf_counter() - setup_start

    profiler = StageProfiler(
        window=config.profiling.report_window,
        warmup_frames=config.profiling.warmup_frames,
    )
    scout = CheapScout(config.scout).to(args.device).eval()
    predictor = ConstantVelocityPredictor(config.temporal)
    router_model = load_router_checkpoint(args.router_ckpt, args.device)
    router = BudgetRouter(config.router, router_model)
    scheduler = HardFrameScheduler(config.scheduler)
    renderer = ProgressiveGuavaRenderer(
        base_renderer.avatar,
        base_renderer.render_model,
        base_renderer.camera,
        config.rendering,
        profiler,
    )
    build_target = TargetBuilder(identity, args)
    capture = open_capture(args.input)

    frame_count = 0
    estimator_calls = 0
    part_updates = Counter()
    render_levels = Counter()
    steady_work_ms: list[float] = []
    steady_period_ms: list[float] = []
    shapes = None
    previous_release_time = None
    previous_render = None
    duplicated_frames = 0
    rest_pose = None
    occlusion_rest_counts = Counter()

    print(
        f"AvatarBudget ready: {config.scheduler.target_fps:.1f} FPS / "
        f"{config.scheduler.deadline_ms:.1f} ms. Press q to stop."
    )
    try:
        with torch.inference_mode():
            while args.max_frames == 0 or frame_count < args.max_frames:
                profiler.next_frame()
                scheduler.begin_frame()
                with profiler.cpu("capture"):
                    ok, frame_bgr = capture.read()
                if not ok:
                    # Do not leave the scheduler in an active-frame state at EOF.
                    scheduler.finish_frame()
                    break

                with profiler.cuda("upload_preprocess"):
                    scout_input = rgb_tensor(frame_bgr, args.device)
                with profiler.cuda("cheap_scout"):
                    scout_output = scout(scout_input)

                reuse_previous = False
                if not predictor.ready:
                    decision = initial_decision(args.device)
                    observed = pear._infer(frame_bgr)
                    estimator_calls += 1
                    profiler.add("pear_estimator", pear.durations[-1], clock="cuda_event")
                    state = predictor.initialize(observed)
                    rest_pose = neutral_pose_like(observed)
                    shapes = shape_report(observed)
                else:
                    with profiler.cuda("temporal_prediction"):
                        prediction = predictor.predict()
                    with profiler.cpu("budget_router_host"):
                        with profiler.cuda("budget_router_gpu"):
                            elapsed = scheduler.elapsed_ms()
                            fixed = config.router.costs.fixed_ms
                            decision_budget = max(
                                1.0, scheduler.routing_budget_ms - elapsed + fixed
                            )
                            decision = router.route(
                                scout_output,
                                prediction.uncertainty,
                                predictor.state,
                                decision_budget,
                            )

                    # A duplicated frame is acceptable only when no part is
                    # forced stale. Otherwise the hard-budget fallback can
                    # suppress PEAR forever once capture/display already exceed
                    # 20 ms, which freezes the avatar.
                    reuse_previous = (
                        not decision.deadline_feasible
                        and not decision.forced_parts
                        and previous_render is not None
                    )
                    if reuse_previous:
                        decision = RouteDecision(
                            update_parts=frozenset(),
                            forced_parts=frozenset(),
                            risk=decision.risk,
                            render_level=RenderLevel.LOW,
                            estimated_ms=0.0,
                            deadline_feasible=False,
                        )
                        duplicated_frames += 1

                    observed = None
                    if decision.update_parts:
                        observed = pear._infer(frame_bgr)
                        estimator_calls += 1
                        profiler.add("pear_estimator", pear.durations[-1], clock="cuda_event")
                    with profiler.cuda("conditional_pose_merge"):
                        state = predictor.commit(
                            prediction, observed, decision.update_parts
                        )
                    occluded_parts = occluded_parts_from_scout(args, scout_output)
                    if occluded_parts and rest_pose is not None:
                        state = predictor.override_parts(
                            rest_pose,
                            occluded_parts,
                            observed=False,
                            reset_staleness=True,
                        )
                        for part in occluded_parts:
                            occlusion_rest_counts[part] += 1

                for part in decision.update_parts:
                    part_updates[part] += 1
                if not reuse_previous:
                    render_levels[decision.render_level] += 1

                if reuse_previous:
                    image = previous_render
                else:
                    with profiler.cuda("ehm_target_conversion"):
                        target = build_target(state.values, time.perf_counter())
                    image = renderer.render(target, decision.render_level)
                    previous_render = image

                with profiler.cpu("readback"):
                    render_bgr = image.cpu().numpy()
                profiler.collect_cuda(synchronize=False)
                if not args.no_display:
                    with profiler.cpu("display"):
                        preview = cv2.resize(
                            frame_bgr, (render_bgr.shape[1], render_bgr.shape[0])
                        )
                        combined = cv2.hconcat((preview, render_bgr))
                        selected = ",".join(part.value for part in decision.update_parts) or "predict"
                        if decision.forced_parts:
                            selected = f"{selected} forced"
                        if args.rest_occluded_parts:
                            occluded = occluded_parts_from_scout(args, scout_output)
                            if occluded:
                                selected = f"{selected} rest:{','.join(part.value for part in occluded)}"
                        label = f"{decision.render_level.value} | {selected}"
                        cv2.putText(
                            combined,
                            label,
                            (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("AvatarBudget: camera -> GUAVA avatar", combined)
                        should_stop = cv2.waitKey(1) & 0xFF == ord("q")

                result = scheduler.finish_frame()
                release_time = time.perf_counter()
                frame_count += 1
                if profiler.frame_index >= config.profiling.warmup_frames:
                    steady_work_ms.append(result.work_ms)
                    if previous_release_time is not None:
                        steady_period_ms.append((release_time - previous_release_time) * 1000.0)
                    profiler.add("end_to_end_work", result.work_ms)
                    if previous_release_time is not None:
                        profiler.add(
                            "frame_release_period",
                            (release_time - previous_release_time) * 1000.0,
                        )
                previous_release_time = release_time

                if frame_count == 1 or frame_count % config.profiling.report_every == 0:
                    payload = report_payload(
                        args,
                        config,
                        profiler,
                        scheduler,
                        frame_count,
                        steady_work_ms,
                        steady_period_ms,
                        part_updates,
                        render_levels,
                        estimator_calls,
                        duplicated_frames,
                        occlusion_rest_counts,
                        setup_seconds,
                        shapes or {},
                    )
                    print_progress(payload)
                if not args.no_display and should_stop:
                    break
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        source_dataset._lmdb_engine.close()

    profiler.collect_cuda(synchronize=True)
    payload = report_payload(
        args,
        config,
        profiler,
        scheduler,
        frame_count,
        steady_work_ms,
        steady_period_ms,
        part_updates,
        render_levels,
        estimator_calls,
        duplicated_frames,
        occlusion_rest_counts,
        setup_seconds,
        shapes or {},
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_progress(payload)
    print(f"Full end-to-end report: {args.report.resolve()}")
    if not payload["end_to_end"]["sustained_50fps_verified"]:
        print("50 FPS is not verified by this run; inspect end-to-end p99 and deadline misses.")
    return payload


if __name__ == "__main__":
    run(parse_args())
