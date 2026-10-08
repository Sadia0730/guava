#!/usr/bin/env python
"""Fine-tune PEAR student from an immutable checkpoint using LSP lower-body 2D labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image


ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
sys.path.insert(0, str(PEAR_ROOT))

from dataset.student_distill_dataset import build_distillation_dataloader  # noqa: E402
from train_pear_student_distill import (  # noqa: E402
    add_body_cam,
    autocast_context,
    distillation_loss,
    load_student,
    load_teacher,
)


LSP_LEFT_RIGHT = torch.tensor([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13])
LSP_LOWER_SMPLX = torch.tensor([8, 5, 2, 1, 4, 7])
LOWER_EDGES = torch.tensor([[2, 1], [1, 0], [3, 4], [4, 5]])
LOWER_WEIGHTS = torch.tensor([2.0, 1.5, 1.0, 1.0, 1.5, 2.0])
LOWER_DRAW_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lsp_train_manifest", type=Path, required=True)
    parser.add_argument("--lsp_val_manifest", type=Path, required=True)
    parser.add_argument("--original_manifest", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--student_config", default="configs/student_l70_v2.yaml")
    parser.add_argument("--teacher_config", default="configs/infer.yaml")
    parser.add_argument("--teacher_ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lsp_batch_size", type=int, default=12)
    parser.add_argument("--original_batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--flip_probability", type=float, default=0.5)
    parser.add_argument("--camera_weight", type=float, default=1.0)
    parser.add_argument("--body_weight", type=float, default=1.0)
    parser.add_argument("--flame_weight", type=float, default=0.5)
    parser.add_argument("--feature_weight", type=float, default=1.0)
    parser.add_argument("--rot_weight", type=float, default=1.0)
    parser.add_argument("--lower_reprojection_weight", type=float, default=5.0)
    parser.add_argument("--lower_shape_weight", type=float, default=2.0)
    parser.add_argument("--lower_direction_weight", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    if args.lsp_batch_size < 1 or args.original_batch_size < 1:
        parser.error("batch sizes must be positive")
    if not 0 <= args.flip_probability <= 1:
        parser.error("--flip_probability must be in [0, 1]")
    return args


class LspDataset(Dataset):
    def __init__(self, manifest: Path, train: bool, flip_probability: float = 0.5):
        self.records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"empty LSP manifest: {manifest}")
        self.train = train
        self.flip_probability = flip_probability if train else 0.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = read_image(record["image"], mode=ImageReadMode.RGB).float().div_(255.0)
        joints = torch.tensor(record["joints_2d"], dtype=torch.float32)
        visible = torch.tensor(record["visible"], dtype=torch.bool)
        if self.train and torch.rand(()) < self.flip_probability:
            image = torch.flip(image, dims=(-1,))
            joints[:, 0] = 1.0 - joints[:, 0]
            joints = joints.index_select(0, LSP_LEFT_RIGHT)
            visible = visible.index_select(0, LSP_LEFT_RIGHT)
        return {
            "image": image,
            "joints_2d": joints,
            "visible": visible,
            "path": record["image"],
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


def project_joints(output: dict, joints: torch.Tensor) -> torch.Tensor:
    """Match PEAR's 1024-space GS camera, returned as normalized [0, 1] XY."""
    rt = output["pd_cam"].float()
    rotated = torch.einsum("bij,bkj->bki", rt[:, :3, :3], joints.float())
    camera = rotated + rt[:, None, :3, 3]
    perspective = 24.0 * camera[..., :2] / camera[..., 2:3].clamp_min(1e-4)
    return (1.0 - perspective) * 0.5


def normalized_points(points: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    weights = visible.float().unsqueeze(-1)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    center = (points * weights).sum(dim=1, keepdim=True) / count
    centered = points - center
    scale = torch.sqrt(
        (centered.square() * weights).sum(dim=(1, 2), keepdim=True)
        / (count * 2.0)
    ).clamp_min(1e-4)
    return centered / scale


def masked_weighted_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    weights = LOWER_WEIGHTS.to(prediction.device)[None, :, None]
    mask = visible.float().unsqueeze(-1) * weights
    error = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.02)
    return (error * mask).sum() / (mask.sum() * prediction.shape[-1]).clamp_min(1.0)


def lower_body_losses(
    output: dict,
    smplx_model,
    target: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mesh = smplx_model(add_body_cam(output), pose_type="rotmat")
    indices = LSP_LOWER_SMPLX.to(mesh["joints"].device)
    prediction = project_joints(output, mesh["joints"].index_select(1, indices))

    reprojection = masked_weighted_l1(prediction, target, visible)
    shape = masked_weighted_l1(
        normalized_points(prediction, visible),
        normalized_points(target, visible),
        visible,
    )

    edges = LOWER_EDGES.to(prediction.device)
    pred_bones = prediction[:, edges[:, 1]] - prediction[:, edges[:, 0]]
    target_bones = target[:, edges[:, 1]] - target[:, edges[:, 0]]
    edge_visible = visible[:, edges[:, 0]] & visible[:, edges[:, 1]]
    cosine = F.cosine_similarity(pred_bones, target_bones, dim=-1, eps=1e-6)
    direction = ((1.0 - cosine) * edge_visible.float()).sum() / edge_visible.sum().clamp_min(1)
    return {
        "lower_reprojection": reprojection,
        "lower_shape": shape,
        "lower_direction": direction,
        "lower_prediction": prediction,
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


def write_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, separators=(",", ":")) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keypoint_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    mask = visible.float()
    denominator = mask.sum().clamp_min(1.0)
    return {
        "lower_pixel_error": (distance * mask).sum() * IMAGE_SIZE / denominator,
        "lower_pck_005": ((distance <= 0.05).float() * mask).sum() / denominator,
        "lower_pck_010": ((distance <= 0.10).float() * mask).sum() / denominator,
    }


def save_checkpoint(
    output_dir: Path,
    step: int,
    source_step: int,
    student,
    optimizer,
    scaler,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "format_version": 1,
        "kind": "pear_student_lsp_lower_body_finetune",
        "step": source_step + step,
        "source_step": source_step,
        "finetune_step": step,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    torch.save(checkpoint, checkpoint_dir / f"finetune_{step:07d}.pt")


def draw_lower_body(image: np.ndarray, points: np.ndarray, visible: np.ndarray, color) -> None:
    pixel = np.rint(points * image.shape[0]).astype(np.int32)
    for first, second in LOWER_DRAW_EDGES:
        if visible[first] and visible[second]:
            cv2.line(image, tuple(pixel[first]), tuple(pixel[second]), color, 2, cv2.LINE_AA)
    for index, point in enumerate(pixel):
        if visible[index]:
            cv2.circle(image, tuple(point), 4, color, -1, cv2.LINE_AA)


@torch.no_grad()
def validate_and_visualize(
    student,
    teacher,
    smplx_model,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
) -> dict[str, float]:
    student.eval()
    totals: dict[str, float] = {}
    count = 0
    visual_batch = None
    visual_student = None
    visual_teacher = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["joints_2d"][:, :6].to(device, non_blocking=True)
        visible = batch["visible"][:, :6].to(device, non_blocking=True)
        with autocast_context(device, args.precision):
            student_output = student(images)
            teacher_output = teacher(images)
        student_losses = lower_body_losses(student_output, smplx_model, target, visible)
        teacher_losses = lower_body_losses(teacher_output, smplx_model, target, visible)
        student_losses.update(
            keypoint_metrics(student_losses["lower_prediction"], target, visible)
        )
        teacher_losses.update(
            keypoint_metrics(teacher_losses["lower_prediction"], target, visible)
        )
        for name in (
            "lower_reprojection",
            "lower_shape",
            "lower_direction",
            "lower_pixel_error",
            "lower_pck_005",
            "lower_pck_010",
        ):
            totals[f"student_{name}"] = totals.get(f"student_{name}", 0.0) + float(student_losses[name])
            totals[f"teacher_{name}"] = totals.get(f"teacher_{name}", 0.0) + float(teacher_losses[name])
        if visual_batch is None:
            visual_batch = batch
            visual_student = student_losses["lower_prediction"].float().cpu()
            visual_teacher = teacher_losses["lower_prediction"].float().cpu()
        count += 1
        if count >= args.val_batches:
            break

    if visual_batch is not None:
        rows = []
        for index in range(min(8, len(visual_batch["image"]))):
            image = visual_batch["image"][index].mul(255).byte().permute(1, 2, 0).numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            visible = visual_batch["visible"][index, :6].numpy()
            gt = visual_batch["joints_2d"][index, :6].numpy()
            panels = []
            for label, points, color in (
                ("GT", gt, (60, 220, 60)),
                ("teacher", visual_teacher[index].numpy(), (60, 80, 240)),
                ("student", visual_student[index].numpy(), (240, 160, 40)),
            ):
                panel = image.copy()
                draw_lower_body(panel, gt, visible, (60, 220, 60))
                if label != "GT":
                    draw_lower_body(panel, points, visible, color)
                cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                panels.append(panel)
            rows.append(np.concatenate(panels, axis=1))
        visual_dir = args.output_dir / "visuals"
        visual_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(visual_dir / f"step_{step:07d}.jpg"), np.concatenate(rows, axis=0))

    student.train()
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    args.lsp_train_manifest = args.lsp_train_manifest.resolve()
    args.lsp_val_manifest = args.lsp_val_manifest.resolve()
    args.original_manifest = args.original_manifest.resolve()
    args.resume = args.resume.resolve()
    args.teacher_ckpt = args.teacher_ckpt.resolve()
    args.output_dir = args.output_dir.resolve()
    if not os.access(args.resume, os.R_OK):
        raise FileNotFoundError(args.resume)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.chdir(PEAR_ROOT)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    total_batch = args.lsp_batch_size + args.original_batch_size
    student = load_student(args.student_config, total_batch, 1, device)
    source = torch.load(args.resume, map_location="cpu", weights_only=False)
    student.load_state_dict(source["student"], strict=True)
    source_step = int(source.get("step", 235000))
    teacher = load_teacher(args.teacher_config, args.teacher_ckpt, device)

    from models.smplx.SMPLXV2 import SMPLX

    smplx_model = SMPLX(str(PEAR_ROOT / "assets" / "SMPLX"), n_shape=300, n_exp=50).to(device).eval()
    for parameter in smplx_model.parameters():
        parameter.requires_grad_(False)

    lsp_train = DataLoader(
        LspDataset(args.lsp_train_manifest, train=True, flip_probability=args.flip_probability),
        batch_size=args.lsp_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    lsp_val = DataLoader(
        LspDataset(args.lsp_val_manifest, train=False),
        batch_size=args.lsp_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    original = build_distillation_dataloader(
        manifest=args.original_manifest,
        batch_size=args.original_batch_size,
        clip_length=1,
        train=True,
        balance_sources=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    lsp_iter = infinite_batches(lsp_train)
    original_iter = infinite_batches(original)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.precision == "fp16")
    loss_args = distillation_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    metadata = {
        "source_checkpoint": str(args.resume),
        "source_step": source_step,
        "source_checkpoint_sha256": file_sha256(args.resume),
        "lsp_train_images": len(lsp_train.dataset),
        "lsp_val_images": len(lsp_val.dataset),
        "original_images": len(original.dataset),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)

    baseline = validate_and_visualize(
        student, teacher, smplx_model, lsp_val, device, args, step=0
    )
    write_jsonl(log_path, {"split": "val", "finetune_step": 0, **baseline})
    print(f"baseline validation: {baseline}", flush=True)

    student.train()
    last_time = time()
    for step in range(1, args.steps + 1):
        lsp_batch = next(lsp_iter)
        original_batch = next(original_iter)
        lsp_images = lsp_batch["image"].to(device, non_blocking=True)
        original_images = original_batch["images"][:, 0].to(device, non_blocking=True)
        images = torch.cat((lsp_images, original_images), dim=0)
        target = lsp_batch["joints_2d"][:, :6].to(device, non_blocking=True)
        visible = lsp_batch["visible"][:, :6].to(device, non_blocking=True)

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
                slice_output(student_output, args.lsp_batch_size),
                smplx_model,
                target,
                visible,
            )
            loss = (
                distill
                + args.lower_reprojection_weight * lower["lower_reprojection"]
                + args.lower_shape_weight * lower["lower_shape"]
                + args.lower_direction_weight * lower["lower_direction"]
            )

        scaler.scale(loss).backward()
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
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
                "lower_reprojection": float(lower["lower_reprojection"].detach()),
                "lower_shape": float(lower["lower_shape"].detach()),
                "lower_direction": float(lower["lower_direction"].detach()),
                **{f"distill_{key}": float(value) for key, value in parts.items()},
            }
            write_jsonl(log_path, item)
            print(
                "finetune {finetune_step:06d} | total {total:.4f} | distill {distillation:.4f} | "
                "lower reproj/shape/dir {lower_reprojection:.4f}/{lower_shape:.4f}/"
                "{lower_direction:.4f} | {batches_per_second:.2f} batch/s".format(**item),
                flush=True,
            )

        if step % args.val_every == 0:
            metrics = validate_and_visualize(
                student, teacher, smplx_model, lsp_val, device, args, step
            )
            write_jsonl(log_path, {"split": "val", "finetune_step": step, **metrics})
            print(f"validation {step}: {metrics}", flush=True)

        if step % args.save_every == 0:
            save_checkpoint(args.output_dir, step, source_step, student, optimizer, scaler, args)

    if args.steps % args.save_every:
        save_checkpoint(args.output_dir, args.steps, source_step, student, optimizer, scaler, args)


if __name__ == "__main__":
    main()
