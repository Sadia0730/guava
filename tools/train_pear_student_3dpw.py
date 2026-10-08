#!/usr/bin/env python
"""Fine-tune PEAR student with 3DPW lower-body 3D labels and PEAR replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
sys.path.insert(0, str(ROOT))

from tools.pear_geometry import (  # noqa: E402
    IMAGE_SIZE,
    LOWER6_INDICES,
    LOWER8_INDICES,
    bbox_from_keypoints,
    square_crop_transform,
)

sys.path.insert(0, str(PEAR_ROOT))

from dataset.student_distill_dataset import build_distillation_dataloader  # noqa: E402
from train_pear_student_distill import (  # noqa: E402
    add_body_cam,
    autocast_context,
    distillation_loss,
    load_student,
    load_teacher,
)
LOWER8_EDGES = torch.tensor(((0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--crop_scale", type=float, default=1.25)
    parser.add_argument("--original_manifest", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--student_config", default="configs/student_l70_v2.yaml")
    parser.add_argument("--teacher_config", default="configs/infer.yaml")
    parser.add_argument("--teacher_ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--supervised_batch_size", type=int, default=6)
    parser.add_argument("--original_batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--train_scope", choices=("all", "head"), default="all")
    parser.add_argument("--gradient_method", choices=("sum", "pcgrad"), default="sum")
    parser.add_argument("--camera_weight", type=float, default=1.0)
    parser.add_argument("--body_weight", type=float, default=1.0)
    parser.add_argument("--flame_weight", type=float, default=2.0)
    parser.add_argument("--feature_weight", type=float, default=1.0)
    parser.add_argument("--rot_weight", type=float, default=1.0)
    parser.add_argument("--lower_position_weight", type=float, default=10.0)
    parser.add_argument("--lower_bone_weight", type=float, default=5.0)
    parser.add_argument("--lower_direction_weight", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=300)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(root: Path, split: str, frame_stride: int) -> list[dict]:
    records = []
    for sequence_path in sorted((root / "sequenceFiles" / split).glob("*.pkl")):
        with sequence_path.open("rb") as handle:
            sequence = pickle.load(handle, encoding="latin1")
        name = str(sequence["sequence"])
        image_root = root / "imageFiles" / name
        camera_poses = np.asarray(sequence["cam_poses"], dtype=np.float32)
        camera_valid = np.asarray(sequence["campose_valid"], dtype=bool)
        for person, (joint_track, pose2d_track) in enumerate(
            zip(sequence["jointPositions"], sequence["poses2d"])
        ):
            valid_track = camera_valid[person] if camera_valid.ndim == 2 else camera_valid
            for frame in range(0, len(joint_track), frame_stride):
                if frame >= len(valid_track) or not valid_track[frame]:
                    continue
                image_path = image_root / f"image_{frame:05d}.jpg"
                if not image_path.is_file():
                    continue
                pose2d = np.asarray(pose2d_track[frame], dtype=np.float32)
                points = pose2d[:2].T
                confidence = pose2d[2]
                visible = confidence > 0.1
                if int(visible.sum()) < 6:
                    continue
                box = bbox_from_keypoints(points, np.where(visible, confidence, 0.0))
                joints = np.asarray(joint_track[frame], dtype=np.float32).reshape(-1, 3)
                camera = camera_poses[frame]
                camera_joints = (camera[:3, :3] @ joints.T).T + camera[:3, 3]
                camera_joints = camera_joints - camera_joints[0:1]
                if not np.isfinite(camera_joints[:22]).all():
                    continue
                records.append(
                    {
                        "sequence": name,
                        "person": person,
                        "frame": frame,
                        "image": str(image_path),
                        "box": box.astype(np.float32),
                        "joints": camera_joints[:22].astype(np.float32),
                    }
                )
    if not records:
        raise ValueError(f"no valid 3DPW records in {root} split {split}")
    return records


class ThreeDPWDataset(Dataset):
    def __init__(self, records: list[dict], crop_scale: float):
        self.records = records
        self.crop_scale = crop_scale

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["image"])
        transform = square_crop_transform(record["box"], self.crop_scale, IMAGE_SIZE)
        crop = cv2.warpAffine(
            image,
            transform,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return {
            "image": torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0),
            "joints": torch.from_numpy(record["joints"]),
            "sample_id": (
                f"{record['sequence']}__p{record['person']:02d}__f{record['frame']:05d}"
            ),
        }


def infinite_batches(loader):
    while True:
        yield from loader


def slice_output(output: dict, count: int) -> dict:
    sliced = {}
    for key, value in output.items():
        if isinstance(value, dict):
            sliced[key] = slice_output(value, count)
        elif torch.is_tensor(value):
            sliced[key] = value[:count]
        else:
            sliced[key] = value
    return sliced


def predicted_camera_joints(output: dict, smplx_model) -> torch.Tensor:
    mesh = smplx_model(add_body_cam(output), pose_type="rotmat")
    joints = mesh["joints"][:, :22].float()
    rt = output["pd_cam"].float()
    camera = torch.einsum("bij,bkj->bki", rt[:, :3, :3], joints)
    camera = camera + rt[:, None, :3, 3]
    camera = camera * camera.new_tensor((-1.0, -1.0, 1.0))
    return camera - camera[:, 0:1]


def lower_body_losses(
    output: dict,
    smplx_model,
    target_joints: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = predicted_camera_joints(output, smplx_model)
    indices = torch.as_tensor(LOWER8_INDICES, device=prediction.device)
    pred_lower = prediction.index_select(1, indices)
    target_lower = target_joints.float().index_select(1, indices)
    position = F.smooth_l1_loss(pred_lower, target_lower, beta=0.02)

    edges = LOWER8_EDGES.to(prediction.device)
    pred_bones = pred_lower[:, edges[:, 1]] - pred_lower[:, edges[:, 0]]
    target_bones = target_lower[:, edges[:, 1]] - target_lower[:, edges[:, 0]]
    bone = F.smooth_l1_loss(pred_bones, target_bones, beta=0.02)
    direction = (1.0 - F.cosine_similarity(pred_bones, target_bones, dim=-1)).mean()

    lower6 = torch.as_tensor(LOWER6_INDICES, device=prediction.device)
    lower6_mm = (
        torch.linalg.vector_norm(
            prediction.index_select(1, lower6)
            - target_joints.float().index_select(1, lower6),
            dim=-1,
        ).mean()
        * 1000.0
    )
    lower8_mm = torch.linalg.vector_norm(pred_lower - target_lower, dim=-1).mean() * 1000.0
    return {
        "position": position,
        "bone": bone,
        "direction": direction,
        "lower6_mpjpe_mm": lower6_mm,
        "lower8_mpjpe_mm": lower8_mm,
    }


def distillation_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "velocity_weight": 0.0,
            "body_delta_magnitude_weight": 0.0,
            "body_delta_direction_weight": 0.0,
            "hand_delta_magnitude_weight": 0.0,
            "hand_delta_direction_weight": 0.0,
            "face_delta_magnitude_weight": 0.0,
            "face_delta_direction_weight": 0.0,
            "joint3d_weight": 0.0,
            "joint2d_weight": 0.0,
        }
    )
    return argparse.Namespace(**values)


def pcgrad_merge(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
) -> tuple[list[torch.Tensor | None], torch.Tensor]:
    """Symmetric two-task PCGrad merge and pre-projection cosine."""
    pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    if not pairs:
        raise ValueError("PCGrad received no shared gradients")
    dot = torch.stack([(a.float() * b.float()).sum() for a, b in pairs]).sum()
    first_norm_sq = torch.stack([a.float().square().sum() for a, _ in pairs]).sum()
    second_norm_sq = torch.stack([b.float().square().sum() for _, b in pairs]).sum()
    cosine = dot / (first_norm_sq.sqrt() * second_norm_sq.sqrt()).clamp_min(1e-12)
    conflict = dot < 0
    first_scale = torch.where(conflict, dot / second_norm_sq.clamp_min(1e-12), dot.new_zeros(()))
    second_scale = torch.where(conflict, dot / first_norm_sq.clamp_min(1e-12), dot.new_zeros(()))
    merged = []
    for first_grad, second_grad in zip(first, second):
        if first_grad is None:
            merged.append(second_grad)
        elif second_grad is None:
            merged.append(first_grad)
        else:
            first_projected = first_grad - first_scale.to(first_grad.dtype) * second_grad
            second_projected = second_grad - second_scale.to(second_grad.dtype) * first_grad
            merged.append(first_projected + second_projected)
    return merged, cosine.detach()


def write_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, separators=(",", ":")) + "\n")


def save_checkpoint(output_dir, step, source_step, student, optimizer, scaler, args):
    checkpoint = {
        "format_version": 1,
        "kind": "pear_student_3dpw_lower_body_3d_finetune",
        "step": source_step + step,
        "source_step": source_step,
        "finetune_step": step,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    torch.save(checkpoint, checkpoint_dir / f"finetune_{step:07d}.pt")


@torch.no_grad()
def validate(student, teacher, smplx_model, loader, device, args) -> dict[str, float]:
    student.eval()
    totals = {f"{model}_{metric}": 0.0 for model in ("student", "teacher") for metric in ("lower6_mpjpe_mm", "lower8_mpjpe_mm")}
    batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["joints"].to(device, non_blocking=True)
        with autocast_context(device, args.precision):
            student_output = student(images)
            teacher_output = teacher(images)
        for model, output in (("student", student_output), ("teacher", teacher_output)):
            losses = lower_body_losses(output, smplx_model, target)
            for metric in ("lower6_mpjpe_mm", "lower8_mpjpe_mm"):
                totals[f"{model}_{metric}"] += float(losses[metric])
        batches += 1
        if batches >= args.val_batches:
            break
    student.train()
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.original_manifest = args.original_manifest.resolve()
    args.resume = args.resume.resolve()
    args.teacher_ckpt = args.teacher_ckpt.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_records = build_records(args.dataset_root, args.train_split, args.frame_stride)
    val_records = build_records(args.dataset_root, args.val_split, args.frame_stride)
    os.chdir(PEAR_ROOT)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    total_batch = args.supervised_batch_size + args.original_batch_size
    student = load_student(args.student_config, total_batch, 1, device)
    source = torch.load(args.resume, map_location="cpu", weights_only=False)
    student.load_state_dict(source["student"], strict=True)
    if args.train_scope == "head":
        for parameter in student.backbone.parameters():
            parameter.requires_grad_(False)
    source_step = int(source.get("step", 235000))
    teacher = load_teacher(args.teacher_config, args.teacher_ckpt, device)

    from models.smplx.SMPLXV2 import SMPLX

    smplx_model = SMPLX(str(PEAR_ROOT / "assets" / "SMPLX"), n_shape=300, n_exp=50).to(device).eval()
    for parameter in smplx_model.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ThreeDPWDataset(train_records, args.crop_scale),
        batch_size=args.supervised_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        ThreeDPWDataset(val_records, args.crop_scale),
        batch_size=args.supervised_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    original_loader = build_distillation_dataloader(
        manifest=args.original_manifest,
        batch_size=args.original_batch_size,
        clip_length=1,
        train=True,
        balance_sources=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    supervised_iter = infinite_batches(train_loader)
    original_iter = infinite_batches(original_loader)

    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.precision == "fp16")
    loss_args = distillation_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    metadata = {
        "source_checkpoint": str(args.resume),
        "source_step": source_step,
        "source_checkpoint_sha256": file_sha256(args.resume),
        "train_split": args.train_split,
        "validation_split": args.val_split,
        "test_labels_used_for_training": False,
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)

    baseline = validate(student, teacher, smplx_model, val_loader, device, args)
    write_jsonl(log_path, {"split": "val", "finetune_step": 0, **baseline})
    print(f"baseline validation: {baseline}", flush=True)

    student.train()
    last_time = time()
    for step in range(1, args.steps + 1):
        supervised = next(supervised_iter)
        original = next(original_iter)
        supervised_images = supervised["image"].to(device, non_blocking=True)
        original_images = original["images"][:, 0].to(device, non_blocking=True)
        images = torch.cat((supervised_images, original_images), dim=0)
        target_joints = supervised["joints"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast_context(device, args.precision):
            teacher_features = teacher.forward_features(images)
            teacher_output = teacher.head(teacher_features)
        with autocast_context(device, args.precision):
            student_output, student_features = student(images, return_features=True)
            distill, parts = distillation_loss(
                student_output,
                teacher_output,
                batch_size=total_batch,
                clip_length=1,
                args=loss_args,
                student_feats=student_features,
                teacher_feats=teacher_features,
            )
            lower = lower_body_losses(
                slice_output(student_output, args.supervised_batch_size),
                smplx_model,
                target_joints,
            )
            supervised_loss = (
                args.lower_position_weight * lower["position"]
                + args.lower_bone_weight * lower["bone"]
                + args.lower_direction_weight * lower["direction"]
            )
            loss = distill + supervised_loss

        gradient_cosine = loss.new_zeros(())
        if args.gradient_method == "pcgrad":
            distill_gradients = torch.autograd.grad(
                distill, trainable_parameters, retain_graph=True, allow_unused=True
            )
            supervised_gradients = torch.autograd.grad(
                supervised_loss, trainable_parameters, allow_unused=True
            )
            merged, gradient_cosine = pcgrad_merge(
                distill_gradients, supervised_gradients
            )
            for parameter, gradient in zip(trainable_parameters, merged):
                parameter.grad = gradient
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        if step % args.log_every == 0:
            elapsed = max(time() - last_time, 1e-9)
            last_time = time()
            item = {
                "split": "train",
                "finetune_step": step,
                "source_equivalent_step": source_step + step,
                "lr": args.lr,
                "batches_per_second": args.log_every / elapsed,
                "total": float(loss.detach()),
                "distillation": float(distill.detach()),
                "lower_position": float(lower["position"].detach()),
                "lower_bone": float(lower["bone"].detach()),
                "lower_direction": float(lower["direction"].detach()),
                "train_lower8_mpjpe_mm": float(lower["lower8_mpjpe_mm"].detach()),
                "gradient_cosine": float(gradient_cosine),
                **{f"distill_{key}": float(value) for key, value in parts.items()},
            }
            write_jsonl(log_path, item)
            print(
                "finetune {finetune_step:06d} | total {total:.4f} | distill "
                "{distillation:.4f} | lower pos/bone/dir {lower_position:.4f}/"
                "{lower_bone:.4f}/{lower_direction:.4f} | lower8 "
                "{train_lower8_mpjpe_mm:.1f} mm | grad cos {gradient_cosine:.3f} | "
                "{batches_per_second:.2f} batch/s".format(
                    **item
                ),
                flush=True,
            )

        if step % args.val_every == 0:
            metrics = validate(student, teacher, smplx_model, val_loader, device, args)
            write_jsonl(log_path, {"split": "val", "finetune_step": step, **metrics})
            print(f"validation {step}: {metrics}", flush=True)

        if step % args.save_every == 0:
            save_checkpoint(args.output_dir, step, source_step, student, optimizer, scaler, args)

    if args.steps % args.save_every:
        save_checkpoint(args.output_dir, args.steps, source_step, student, optimizer, scaler, args)


if __name__ == "__main__":
    main()
