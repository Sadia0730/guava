#!/usr/bin/env python
"""Paired PEAR teacher/student evaluation on official 3DPW annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pear_geometry import (
    IMAGE_SIZE,
    J14_INDICES,
    LOWER6_INDICES,
    LOWER8_INDICES,
    PEAR_MODEL_X_MAX,
    PEAR_MODEL_X_MIN,
    bbox_from_keypoints,
    mpjpe_mm,
    pa_mpjpe_mm,
    pear_camera_project_normalized,
    root_center,
    square_crop_transform,
    transform_points,
)


PEAR_ROOT = ROOT / "third_party" / "PEAR"
EHM_ROOT = ROOT / "EHM-Tracker"
DEFAULT_TEACHER = (
    Path.home()
    / ".cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/"
    "513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt"
)
DEFAULT_DETECTOR = EHM_ROOT / "pretrained/dwpose/yolox_l.onnx"
BODY_EDGES = (
    (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (12, 16), (12, 17),
    (16, 18), (17, 19), (18, 20), (19, 21),
)
OPENPOSE_EDGES = (
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
)


@dataclass(frozen=True)
class Sample:
    sequence: str
    sequence_path: str
    person: int
    frame: int
    image_path: str
    joints_3d: list[list[float]]
    camera_pose: list[list[float]]
    keypoints_2d: list[list[float]]
    confidence: list[float]

    @property
    def sample_id(self) -> str:
        return f"{self.sequence}__p{self.person:02d}__f{self.frame:05d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "infer", "all"), default="all")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/3dpw")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--mode", choices=("annotated", "detector"), default="annotated")
    parser.add_argument("--samples", type=int, default=111)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--crop-scale", type=float, default=1.25)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--detector-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-config", default="configs/infer.yaml")
    parser.add_argument("--smplx-assets", type=Path, default=PEAR_ROOT / "assets/SMPLX")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_candidates(dataset_root: Path, split: str) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for sequence_path in sorted((dataset_root / "sequenceFiles" / split).glob("*.pkl")):
        with sequence_path.open("rb") as handle:
            sequence = pickle.load(handle, encoding="latin1")
        sequence_name = str(sequence["sequence"])
        image_root = dataset_root / "imageFiles" / sequence_name
        candidates = []
        for person_index, (poses2d, joint_positions) in enumerate(
            zip(sequence["poses2d"], sequence["jointPositions"])
        ):
            for frame_index in range(min(len(poses2d), len(joint_positions))):
                confidence = np.asarray(poses2d[frame_index, 2], dtype=np.float32)
                joints = np.asarray(joint_positions[frame_index], dtype=np.float32).reshape(24, 3)
                image_path = image_root / f"image_{frame_index:05d}.jpg"
                if int((confidence > 0).sum()) < 6 or not np.isfinite(joints).all() or not image_path.is_file():
                    continue
                points = np.asarray(poses2d[frame_index, :2].T, dtype=np.float32)
                candidates.append(
                    Sample(
                        sequence=sequence_name,
                        sequence_path=str(sequence_path.resolve()),
                        person=person_index,
                        frame=frame_index,
                        image_path=str(image_path.resolve()),
                        joints_3d=joints.tolist(),
                        camera_pose=np.asarray(sequence["cam_poses"][frame_index], dtype=np.float32).tolist(),
                        keypoints_2d=points.tolist(),
                        confidence=confidence.tolist(),
                    )
                )
        if candidates:
            groups[sequence_name] = candidates
    return groups


def stratified_sample(groups: dict[str, list[Sample]], count: int, seed: int) -> list[Sample]:
    if count < len(groups):
        raise ValueError(f"{count} samples cannot cover all {len(groups)} sequences")
    if count > sum(map(len, groups.values())):
        raise ValueError("requested more samples than available")
    rng = random.Random(seed)
    queues = {}
    for name, candidates in groups.items():
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        queues[name] = shuffled
    names = sorted(queues)
    selected = []
    round_index = 0
    while len(selected) < count:
        name = names[round_index % len(names)]
        if queues[name]:
            selected.append(queues[name].pop())
        round_index += 1
    rng.shuffle(selected)
    return selected


def create_detector(model_path: Path, device: str):
    providers = onnxruntime.get_available_providers()
    if device == "cuda":
        selected = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        selected = ["CPUExecutionProvider"]
    sys.path.insert(0, str(EHM_ROOT))
    from src.modules.dwpose.onnxdet import inference_detector

    session = onnxruntime.InferenceSession(str(model_path), providers=selected)
    return lambda image: inference_detector(session, image)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    top_left = np.maximum(first[:2], second[:2])
    bottom_right = np.minimum(first[2:], second[2:])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(size[0] * size[1])
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def crop_diagnostics(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    transform: np.ndarray,
) -> dict[str, Any]:
    crop_points = transform_points(keypoints, transform)
    visible = confidence > 0
    inside = np.logical_and(crop_points >= 0.0, crop_points < IMAGE_SIZE).all(axis=1)
    inside_width = np.logical_and(
        crop_points[:, 0] >= PEAR_MODEL_X_MIN,
        crop_points[:, 0] < PEAR_MODEL_X_MAX,
    )
    ankle_visible = visible[[10, 13]]
    ankles = crop_points[[10, 13]]
    ankle_inside = np.logical_and(ankles >= 0.0, ankles < IMAGE_SIZE).all(axis=1)
    ankle_edge = np.min(np.concatenate((ankles, IMAGE_SIZE - ankles), axis=1), axis=1)
    return {
        "visible_keypoints": int(visible.sum()),
        "keypoints_outside_crop": int(np.logical_and(visible, ~inside).sum()),
        "keypoints_outside_internal_width": int(np.logical_and(visible, ~inside_width).sum()),
        "visible_ankles": int(ankle_visible.sum()),
        "ankles_outside_crop": int(np.logical_and(ankle_visible, ~ankle_inside).sum()),
        "ankles_within_5px_edge": int(np.logical_and(ankle_visible, ankle_edge < 5.0).sum()),
    }


def prepare(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = collect_candidates(dataset_root, args.split)
    samples = stratified_sample(groups, args.samples, args.seed)
    detector = create_detector(args.detector.resolve(), args.detector_device)
    mode_images: dict[str, list[torch.Tensor]] = {"annotated": [], "detector": []}
    mode_transforms: dict[str, list[torch.Tensor]] = {"annotated": [], "detector": []}
    mode_valid: dict[str, list[bool]] = {"annotated": [], "detector": []}
    mode_meta: dict[str, list[dict[str, Any]]] = {"annotated": [], "detector": []}

    for index, sample in enumerate(samples):
        image_bgr = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(sample.image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        keypoints = np.asarray(sample.keypoints_2d, dtype=np.float32)
        confidence = np.asarray(sample.confidence, dtype=np.float32)
        annotation_box = bbox_from_keypoints(keypoints, confidence)

        boxes = np.asarray(detector(image_rgb), dtype=np.float32)
        detector_box = None
        detector_iou = None
        if boxes.size:
            boxes = boxes.reshape(-1, boxes.shape[-1])[:, :4]
            detector_box = max(boxes, key=lambda box: box_iou(box, annotation_box)).copy()
            detector_iou = box_iou(detector_box, annotation_box)

        for mode, box in (("annotated", annotation_box), ("detector", detector_box)):
            valid = box is not None
            if valid:
                transform = square_crop_transform(box, args.crop_scale)
                crop = cv2.warpAffine(
                    image_rgb,
                    transform,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    flags=cv2.INTER_LINEAR,
                )
                tensor = torch.from_numpy(crop.copy()).permute(2, 0, 1).contiguous()
                diagnostics = crop_diagnostics(keypoints, confidence, transform)
            else:
                transform = np.zeros((2, 3), dtype=np.float32)
                tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
                diagnostics = {
                    "visible_keypoints": int((confidence > 0).sum()),
                    "keypoints_outside_crop": None,
                    "keypoints_outside_internal_width": None,
                    "visible_ankles": int((confidence[[10, 13]] > 0).sum()),
                    "ankles_outside_crop": None,
                    "ankles_within_5px_edge": None,
                }
            mode_images[mode].append(tensor)
            mode_transforms[mode].append(torch.from_numpy(transform))
            mode_valid[mode].append(valid)
            mode_meta[mode].append(
                {
                    **diagnostics,
                    "box": box.tolist() if valid else None,
                    "annotation_box": annotation_box.tolist(),
                    "detector_iou_to_annotation": detector_iou if mode == "detector" else None,
                }
            )
        print(f"prepare: {index + 1}/{len(samples)}", flush=True)

    prepared_path = args.prepared or args.output_dir / "prepared_3dpw.pt"
    payload = {
        "format_version": 1,
        "protocol": {
            "split": args.split,
            "samples": args.samples,
            "sequences": len(groups),
            "seed": args.seed,
            "crop_scale": args.crop_scale,
            "selection": "seeded round-robin across every sequence; exact laptop frame list unavailable",
            "detector_association": "highest IoU to annotated person box; annotation not used for crop",
        },
        "samples": [asdict(sample) for sample in samples],
        "modes": {
            mode: {
                "images": torch.stack(mode_images[mode]),
                "transforms": torch.stack(mode_transforms[mode]),
                "valid": torch.tensor(mode_valid[mode], dtype=torch.bool),
                "metadata": mode_meta[mode],
            }
            for mode in mode_images
        },
    }
    torch.save(payload, prepared_path)
    (args.output_dir / "selected_samples.json").write_text(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "sequence": sample.sequence,
                        "person": sample.person,
                        "frame": sample.frame,
                        "image_path": sample.image_path,
                    }
                    for sample in samples
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return prepared_path


def tree_float(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, dict):
        return {key: tree_float(item) for key, item in value.items()}
    return value


def load_models(args: argparse.Namespace):
    original = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from models.smplx.SMPLXV2 import SMPLX
        from train_pear_student_distill import load_student, load_teacher

        device = torch.device(args.device)
        teacher = load_teacher(args.teacher_config, args.teacher_checkpoint.resolve(), device)
        student = load_student(args.student_config, args.batch_size, 1, device)
        checkpoint = torch.load(args.student_checkpoint.resolve(), map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["student"], strict=True)
        student.eval()
        model = SMPLX(str(args.smplx_assets.resolve()), n_shape=300, n_exp=50).to(device).eval()
        return teacher, student, model, int(checkpoint["step"])
    finally:
        os.chdir(original)
        sys.path.remove(str(PEAR_ROOT))


def add_body_cam(output: dict[str, Any]) -> dict[str, Any]:
    result = dict(output["body_param"])
    rt = output["pd_cam"]
    result["body_cam"] = torch.stack(
        (24.0 / rt[:, 2, 3].clamp_min(1e-6), rt[:, 0, 3], rt[:, 1, 3]), dim=1
    )
    return result


def confidence_interval_by_sequence(
    rows: list[dict[str, Any]], key: str, samples: int, seed: int
) -> list[float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row["sequence"], []).append(float(row[key]))
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.choice(names, size=len(names), replace=True)
        values = [value for name in selected for value in groups[name]]
        estimates.append(float(np.mean(values)))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "teacher_body22_mpjpe_mm",
        "teacher_j14_mpjpe_mm",
        "teacher_j14_pa_mpjpe_mm",
        "teacher_lower6_mpjpe_mm",
        "teacher_lower8_mpjpe_mm",
        "student_body22_mpjpe_mm",
        "student_j14_mpjpe_mm",
        "student_j14_pa_mpjpe_mm",
        "student_lower6_mpjpe_mm",
        "student_lower8_mpjpe_mm",
    )
    metrics = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        metrics[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sequence_bootstrap_95ci_mean": confidence_interval_by_sequence(
                rows, key, args.bootstrap_samples, args.seed
            ),
        }
    student_worse = np.asarray(
        [row["student_j14_mpjpe_mm"] > row["teacher_j14_mpjpe_mm"] for row in rows]
    )
    return {
        "eligible_frames": len(rows),
        "sequences": len({row["sequence"] for row in rows}),
        "student_j14_mpjpe_worse_than_teacher_frames": int(student_worse.sum()),
        "metrics": metrics,
    }


def draw_skeleton(
    image: np.ndarray,
    points: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    visible: np.ndarray | None = None,
) -> None:
    points = np.rint(points).astype(np.int32)
    for first, second in edges:
        if visible is None or (visible[first] and visible[second]):
            cv2.line(image, tuple(points[first]), tuple(points[second]), color, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        if visible is None or visible[index]:
            cv2.circle(image, tuple(point), 3, color, -1, cv2.LINE_AA)


def save_visuals(
    output_dir: Path,
    payload: dict[str, Any],
    mode: str,
    rows: list[dict[str, Any]],
    teacher_2d: np.ndarray,
    student_2d: np.ndarray,
) -> None:
    valid_indices = [row["prepared_index"] for row in rows]
    row_by_index = {row["prepared_index"]: row for row in rows}
    worst = sorted(valid_indices, key=lambda i: row_by_index[i]["student_lower8_mpjpe_mm"], reverse=True)[:12]
    rng = random.Random(20260927)
    random_indices = rng.sample(valid_indices, min(12, len(valid_indices)))
    mode_payload = payload["modes"][mode]
    valid_to_output = {prepared_index: output_index for output_index, prepared_index in enumerate(valid_indices)}

    for name, indices in (("worst_lower_body", worst), ("seeded_random", random_indices)):
        panels = []
        for prepared_index in indices:
            output_index = valid_to_output[prepared_index]
            image = mode_payload["images"][prepared_index].permute(1, 2, 0).numpy()[:, :, ::-1].copy()
            sample = payload["samples"][prepared_index]
            transform = mode_payload["transforms"][prepared_index].numpy()
            gt_points = transform_points(np.asarray(sample["keypoints_2d"], np.float32), transform)
            confidence = np.asarray(sample["confidence"]) > 0
            gt_panel = image.copy()
            teacher_panel = image.copy()
            student_panel = image.copy()
            draw_skeleton(gt_panel, gt_points, OPENPOSE_EDGES, (60, 220, 60), confidence)
            draw_skeleton(teacher_panel, teacher_2d[output_index] * IMAGE_SIZE, BODY_EDGES, (40, 80, 240))
            draw_skeleton(student_panel, student_2d[output_index] * IMAGE_SIZE, BODY_EDGES, (240, 160, 40))
            for panel, label, color in (
                (gt_panel, "3DPW 2D", (60, 220, 60)),
                (teacher_panel, "teacher", (40, 80, 240)),
                (student_panel, "student", (240, 160, 40)),
            ):
                cv2.rectangle(panel, (0, 0), (IMAGE_SIZE, 25), (0, 0, 0), -1)
                cv2.putText(panel, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            combined = np.hstack((gt_panel, teacher_panel, student_panel))
            row = row_by_index[prepared_index]
            text = (
                f"{row['sample_id']}  student J14/lower8 "
                f"{row['student_j14_mpjpe_mm']:.1f}/{row['student_lower8_mpjpe_mm']:.1f} mm"
            )
            cv2.rectangle(combined, (0, combined.shape[0] - 22), (combined.shape[1], combined.shape[0]), (0, 0, 0), -1)
            cv2.putText(combined, text, (5, combined.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            panels.append(combined)
        if panels:
            cv2.imwrite(str(output_dir / f"{name}.jpg"), np.vstack(panels))


def infer(args: argparse.Namespace, prepared_path: Path) -> None:
    if args.student_checkpoint is None:
        raise ValueError("--student-checkpoint is required for inference")
    payload = torch.load(prepared_path, map_location="cpu", weights_only=False)
    mode_payload = payload["modes"][args.mode]
    valid_indices = torch.nonzero(mode_payload["valid"], as_tuple=False).flatten().tolist()
    images = mode_payload["images"][valid_indices].float().div_(255.0)
    gt = torch.tensor(
        [payload["samples"][index]["joints_3d"] for index in valid_indices], dtype=torch.float32
    )
    gt_camera = torch.tensor(
        [payload["samples"][index]["camera_pose"] for index in valid_indices], dtype=torch.float32
    )
    teacher, student, smplx_model, step = load_models(args)
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.precision
    ]
    j14_indices = torch.as_tensor(J14_INDICES, dtype=torch.long, device=device)
    lower6_indices = torch.as_tensor(LOWER6_INDICES, dtype=torch.long, device=device)
    lower8_indices = torch.as_tensor(LOWER8_INDICES, dtype=torch.long, device=device)
    rows = []
    teacher_2d_all = []
    student_2d_all = []

    for start in range(0, len(images), args.batch_size):
        end = min(start + args.batch_size, len(images))
        batch = images[start:end].to(device, non_blocking=True)
        gt_batch = gt[start:end].to(device)
        gt_camera_batch = gt_camera[start:end].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=dtype != torch.float32
        ):
            teacher_output = teacher(batch)
            student_output = student(batch)
        teacher_output = tree_float(teacher_output)
        student_output = tree_float(student_output)
        with torch.inference_mode():
            teacher_joints = smplx_model(add_body_cam(teacher_output), pose_type="rotmat")["joints"][:, :22]
            student_joints = smplx_model(add_body_cam(student_output), pose_type="rotmat")["joints"][:, :22]
            teacher_2d = pear_camera_project_normalized(teacher_output["pd_cam"], teacher_joints)
            student_2d = pear_camera_project_normalized(student_output["pd_cam"], student_joints)
        teacher_2d_all.append(teacher_2d.cpu())
        student_2d_all.append(student_2d.cpu())

        gt_camera_joints = torch.einsum(
            "bij,bkj->bki", gt_camera_batch[:, :3, :3], gt_batch[:, :22]
        ) + gt_camera_batch[:, None, :3, 3]
        teacher_camera_joints = torch.einsum(
            "bij,bkj->bki", teacher_output["pd_cam"][:, :3, :3], teacher_joints
        ) + teacher_output["pd_cam"][:, None, :3, 3]
        student_camera_joints = torch.einsum(
            "bij,bkj->bki", student_output["pd_cam"][:, :3, :3], student_joints
        ) + student_output["pd_cam"][:, None, :3, 3]
        # PEAR/PyTorch3D camera coordinates use +x left and +y up, whereas
        # 3DPW/OpenCV uses +x right and +y down. PEAR's own screen projection
        # applies this xy flip; apply the same fixed basis change for 3D error.
        pear_to_opencv = teacher_camera_joints.new_tensor([-1.0, -1.0, 1.0])
        teacher_camera_joints = teacher_camera_joints * pear_to_opencv
        student_camera_joints = student_camera_joints * pear_to_opencv
        gt_centered = root_center(gt_camera_joints)
        teacher_centered = root_center(teacher_camera_joints)
        student_centered = root_center(student_camera_joints)
        batch_metrics = {}
        for model_name, prediction in (("teacher", teacher_centered), ("student", student_centered)):
            batch_metrics[f"{model_name}_body22_mpjpe_mm"] = mpjpe_mm(prediction, gt_centered)
            batch_metrics[f"{model_name}_j14_mpjpe_mm"] = mpjpe_mm(
                prediction.index_select(1, j14_indices), gt_centered.index_select(1, j14_indices)
            )
            batch_metrics[f"{model_name}_j14_pa_mpjpe_mm"] = pa_mpjpe_mm(
                prediction.index_select(1, j14_indices), gt_centered.index_select(1, j14_indices)
            )
            batch_metrics[f"{model_name}_lower6_mpjpe_mm"] = mpjpe_mm(
                prediction.index_select(1, lower6_indices), gt_centered.index_select(1, lower6_indices)
            )
            batch_metrics[f"{model_name}_lower8_mpjpe_mm"] = mpjpe_mm(
                prediction.index_select(1, lower8_indices), gt_centered.index_select(1, lower8_indices)
            )

        for local_index in range(end - start):
            prepared_index = valid_indices[start + local_index]
            sample = payload["samples"][prepared_index]
            rows.append(
                {
                    "prepared_index": prepared_index,
                    "sample_id": (
                        f"{sample['sequence']}__p{sample['person']:02d}__f{sample['frame']:05d}"
                    ),
                    "sequence": sample["sequence"],
                    "person": sample["person"],
                    "frame": sample["frame"],
                    **{
                        key: float(value[local_index].cpu())
                        for key, value in batch_metrics.items()
                    },
                }
            )
        print(f"{args.mode} inference: {end}/{len(images)}", flush=True)

    teacher_2d_array = torch.cat(teacher_2d_all).numpy()
    student_2d_array = torch.cat(student_2d_all).numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": payload["protocol"],
        "mode": args.mode,
        "prepared_samples": len(payload["samples"]),
        "detector_or_crop_valid_samples": len(valid_indices),
        "checkpoint": str(args.student_checkpoint.resolve()),
        "checkpoint_step": step,
        "checkpoint_sha256": sha256_file(args.student_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
        "joint_protocol": {
            "root": "SMPL/SMPL-X pelvis index 0",
            "coordinates": (
                "3DPW and PEAR joints rotated into their respective camera frames; "
                "PEAR x/y converted from PyTorch3D to OpenCV axes; then pelvis centered"
            ),
            "body22": list(range(22)),
            "j14": J14_INDICES.tolist(),
            "lower6": LOWER6_INDICES.tolist(),
            "lower8": LOWER8_INDICES.tolist(),
            "units": "millimeters",
            "pa_alignment": "per-frame similarity transform on J14 only",
        },
        "results": summarize(rows, args),
        "crop_diagnostics": mode_payload["metadata"],
        "limitations": [
            "This is a deterministic reconstruction of the reported 111-frame/24-sequence pilot; the original laptop frame list/evaluator was unavailable.",
            "Detector-person association uses annotation IoU, but the detector box itself is used for cropping.",
            "3DPW ground truth is SMPL while predictions are EHM/SMPL-X; corresponding body joints are compared after pelvis centering.",
            "3DPW does not independently evaluate PEAR hand articulation or facial expression.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output_dir / "predicted_joints_2d.npz",
        prepared_indices=np.asarray(valid_indices),
        teacher=teacher_2d_array,
        student=student_2d_array,
    )
    save_visuals(args.output_dir, payload, args.mode, rows, teacher_2d_array, student_2d_array)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    prepared_path = args.prepared.resolve() if args.prepared else args.output_dir / "prepared_3dpw.pt"
    if args.stage in ("prepare", "all"):
        prepared_path = prepare(args)
    if args.stage in ("infer", "all"):
        infer(args, prepared_path)


if __name__ == "__main__":
    main()
