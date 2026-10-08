#!/usr/bin/env python
"""Audit PEAR/3DPW camera, crop, joint, and foot-visibility conventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pear_geometry import (
    IMAGE_SIZE,
    OPENPOSE18_TO_SMPL,
    PEAR_MODEL_X_MAX,
    PEAR_MODEL_X_MIN,
    bbox_from_keypoints,
    crop_extent_in_source,
    invert_affine,
    pear_camera_project_normalized,
    square_crop_transform,
    transform_points,
)


RIGHT_LEFT_SWAP = {
    2: 16,
    3: 18,
    4: 20,
    5: 17,
    6: 19,
    7: 21,
    8: 1,
    9: 4,
    10: 7,
    11: 2,
    12: 5,
    13: 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/3dpw"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crop-scale", type=float, default=1.25)
    parser.add_argument("--preview-sequences", type=int, default=24)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_world(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    camera = np.einsum("ij,nj->ni", extrinsic[:3, :3], points) + extrinsic[:3, 3]
    projected = np.einsum("ij,nj->ni", intrinsic, camera)
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)


def projection_error(
    joints: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    mapping: dict[int, int],
) -> list[float]:
    projected = project_world(joints, extrinsic, intrinsic)
    errors = []
    for openpose_index, smpl_index in mapping.items():
        if confidence[openpose_index] > 0:
            errors.append(float(np.linalg.norm(projected[smpl_index] - observed[openpose_index])))
    return errors


def draw_preview(
    image: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    transform: np.ndarray,
    title: str,
) -> np.ndarray:
    original = image.copy()
    box_corners = crop_extent_in_source(transform).astype(np.int32)
    cv2.polylines(original, [box_corners], True, (40, 220, 255), 3, cv2.LINE_AA)
    for point, visible in zip(points, confidence > 0):
        if visible:
            cv2.circle(original, tuple(np.rint(point).astype(np.int32)), 4, (60, 220, 60), -1)
    original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    crop = cv2.warpAffine(image, transform, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)
    crop_points = transform_points(points, transform)
    for index, (point, visible) in enumerate(zip(crop_points, confidence > 0)):
        if visible:
            color = (40, 80, 240) if index in (10, 13) else (60, 220, 60)
            cv2.circle(crop, tuple(np.rint(point).astype(np.int32)), 4, color, -1)
    cv2.line(crop, (PEAR_MODEL_X_MIN, 0), (PEAR_MODEL_X_MIN, IMAGE_SIZE), (255, 180, 40), 2)
    cv2.line(crop, (PEAR_MODEL_X_MAX, 0), (PEAR_MODEL_X_MAX, IMAGE_SIZE), (255, 180, 40), 2)
    panel = np.hstack((original, crop))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(panel, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return panel


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash_before = sha256_file(args.checkpoint)

    roundtrip_errors = []
    world_to_camera_errors = []
    inverse_camera_errors = []
    direct_mapping_errors = []
    swapped_mapping_errors = []
    crop_padding_fractions = []
    visible_keypoints = 0
    keypoints_outside_crop = 0
    keypoints_outside_model_width = 0
    visible_ankles = 0
    ankles_outside_crop = 0
    ankles_near_crop_edge_5px = 0
    frames = 0
    people = 0
    previews = []

    sequence_paths = sorted((dataset_root / "sequenceFiles" / args.split).glob("*.pkl"))
    for sequence_index, sequence_path in enumerate(sequence_paths):
        with sequence_path.open("rb") as handle:
            sequence = pickle.load(handle, encoding="latin1")
        preview_saved = False
        for person_index, (poses2d, positions) in enumerate(
            zip(sequence["poses2d"], sequence["jointPositions"])
        ):
            people += 1
            for frame_index in range(min(len(poses2d), len(positions))):
                points = np.asarray(poses2d[frame_index, :2].T, dtype=np.float32)
                confidence = np.asarray(poses2d[frame_index, 2], dtype=np.float32)
                if int((confidence > 0).sum()) < 6:
                    continue
                frames += 1
                box = bbox_from_keypoints(points, confidence)
                transform = square_crop_transform(box, args.crop_scale)
                inverse = invert_affine(transform)
                crop_points = transform_points(points, transform)
                recovered = transform_points(crop_points, inverse)
                roundtrip_errors.append(float(np.abs(recovered - points).max()))

                width = int(round(float(sequence["cam_intrinsics"][0, 2]) * 2.0))
                height = int(round(float(sequence["cam_intrinsics"][1, 2]) * 2.0))
                extent = crop_extent_in_source(transform)
                crop_min = extent.min(axis=0)
                crop_max = extent.max(axis=0)
                crop_area = max(1.0, float((crop_max[0] - crop_min[0]) * (crop_max[1] - crop_min[1])))
                inside_w = max(0.0, min(crop_max[0], width) - max(crop_min[0], 0.0))
                inside_h = max(0.0, min(crop_max[1], height) - max(crop_min[1], 0.0))
                crop_padding_fractions.append(1.0 - inside_w * inside_h / crop_area)

                visible = confidence > 0
                visible_keypoints += int(visible.sum())
                inside_crop = np.logical_and(crop_points >= 0.0, crop_points < IMAGE_SIZE).all(axis=1)
                keypoints_outside_crop += int(np.logical_and(visible, ~inside_crop).sum())
                inside_model_width = np.logical_and(
                    crop_points[:, 0] >= PEAR_MODEL_X_MIN,
                    crop_points[:, 0] < PEAR_MODEL_X_MAX,
                )
                keypoints_outside_model_width += int(np.logical_and(visible, ~inside_model_width).sum())

                for ankle_index in (10, 13):
                    if visible[ankle_index]:
                        visible_ankles += 1
                        ankle = crop_points[ankle_index]
                        if not bool(np.logical_and(ankle >= 0.0, ankle < IMAGE_SIZE).all()):
                            ankles_outside_crop += 1
                        if float(np.min(np.r_[ankle, IMAGE_SIZE - ankle])) < 5.0:
                            ankles_near_crop_edge_5px += 1

                joints = np.asarray(positions[frame_index], dtype=np.float64).reshape(24, 3)
                camera_pose = np.asarray(sequence["cam_poses"][frame_index], dtype=np.float64)
                intrinsic = np.asarray(sequence["cam_intrinsics"], dtype=np.float64)
                camera_valid = bool(sequence["campose_valid"][person_index][frame_index])
                if camera_valid:
                    observed = points.astype(np.float64)
                    direct = projection_error(
                        joints,
                        observed,
                        confidence,
                        camera_pose,
                        intrinsic,
                        OPENPOSE18_TO_SMPL,
                    )
                    inverse_errors = projection_error(
                        joints,
                        observed,
                        confidence,
                        np.linalg.inv(camera_pose),
                        intrinsic,
                        OPENPOSE18_TO_SMPL,
                    )
                    swapped = projection_error(
                        joints,
                        observed,
                        confidence,
                        camera_pose,
                        intrinsic,
                        RIGHT_LEFT_SWAP,
                    )
                    world_to_camera_errors.extend(direct)
                    direct_mapping_errors.extend(direct)
                    inverse_camera_errors.extend(inverse_errors)
                    swapped_mapping_errors.extend(swapped)

                if (
                    not preview_saved
                    and len(previews) < args.preview_sequences
                    and sequence_index < args.preview_sequences
                ):
                    image_path = (
                        dataset_root
                        / "imageFiles"
                        / str(sequence["sequence"])
                        / f"image_{frame_index:05d}.jpg"
                    )
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is not None:
                        previews.append(
                            draw_preview(
                                image,
                                points,
                                confidence,
                                transform,
                                f"{sequence['sequence']} p{person_index} f{frame_index}",
                            )
                        )
                        preview_saved = True

    joints = torch.tensor([[[0.1, 0.2, 0.1]]]).expand(1, 22, 3).clone()
    teacher_camera = torch.eye(4).unsqueeze(0)
    teacher_camera[:, 2, 3] = 3.0
    student_camera = teacher_camera.clone()
    student_camera[:, 0, 3] += 0.1
    old_teacher = joints[..., :2] / joints[..., 2:3].clamp_min(0.1)
    old_student = joints[..., :2] / joints[..., 2:3].clamp_min(0.1)
    correct_teacher = pear_camera_project_normalized(teacher_camera, joints)
    correct_student = pear_camera_project_normalized(student_camera, joints)

    def median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    report = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "sequences": len(sequence_paths),
        "people": people,
        "eligible_person_frames": frames,
        "crop_scale": args.crop_scale,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": sha256_file(args.checkpoint),
        "crop_roundtrip_max_abs_px": max(roundtrip_errors),
        "crop_padding_fraction_mean": float(np.mean(crop_padding_fractions)),
        "crop_padding_fraction_p95": float(np.quantile(crop_padding_fractions, 0.95)),
        "visible_keypoints": visible_keypoints,
        "visible_keypoints_outside_256_crop": keypoints_outside_crop,
        "visible_keypoints_outside_internal_192_width": keypoints_outside_model_width,
        "visible_ankles": visible_ankles,
        "ankles_outside_256_crop": ankles_outside_crop,
        "ankles_within_5px_of_crop_edge": ankles_near_crop_edge_5px,
        "camera_convention": {
            "world_to_camera_median_reprojection_px": median(world_to_camera_errors),
            "inverse_camera_median_reprojection_px": median(inverse_camera_errors),
            "world_to_camera_preferred": median(world_to_camera_errors) < median(inverse_camera_errors),
        },
        "left_right_mapping": {
            "direct_median_reprojection_px": median(direct_mapping_errors),
            "swapped_median_reprojection_px": median(swapped_mapping_errors),
            "direct_mapping_preferred": median(direct_mapping_errors) < median(swapped_mapping_errors),
        },
        "joint2d_camera_response_toy": {
            "old_helper_mean_l1": float(torch.nn.functional.l1_loss(old_student, old_teacher)),
            "camera_aware_mean_l1": float(
                torch.nn.functional.l1_loss(correct_student, correct_teacher)
            ),
            "camera_aware_horizontal_shift_256px": float(
                (correct_student - correct_teacher)[..., 0].abs().mean() * IMAGE_SIZE
            ),
        },
        "normalization": {
            "backbone": "GroupNorm",
            "head": "LayerNorm",
            "batchnorm_recalibration_applicable": False,
        },
        "limitations": [
            "3DPW poses2D contains ankles but not toe landmarks, so foot clipping is measured at ankles.",
            "Projection residuals compare fitted SMPL joints with detected OpenPose-18 joints and are convention checks, not annotation accuracy metrics.",
            "Detector-crop clipping is evaluated by evaluate_pear_3dpw.py, not this annotated-crop audit.",
        ],
    }
    if report["checkpoint_sha256_before"] != report["checkpoint_sha256_after"]:
        raise RuntimeError("checkpoint hash changed during read-only audit")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if previews:
        rows = []
        for start in range(0, len(previews), 3):
            row = previews[start : start + 3]
            while len(row) < 3:
                row.append(np.zeros_like(previews[0]))
            rows.append(np.hstack(row))
        cv2.imwrite(str(args.output_dir / "crop_audit_contact_sheet.jpg"), np.vstack(rows))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
