#!/usr/bin/env python
"""Prepare LSP-Extended images and 2D joints for PEAR lower-body fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


IMAGE_SIZE = 256
LSP_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
    (2, 8), (3, 9), (8, 12), (9, 12), (12, 13),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lsp_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--crop_scale", type=float, default=1.35)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--min_lower_visible", type=int, default=4)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.crop_scale <= 0:
        parser.error("--crop_scale must be positive")
    if not 0 < args.val_fraction < 1:
        parser.error("--val_fraction must be in (0, 1)")
    if not 1 <= args.min_lower_visible <= 6:
        parser.error("--min_lower_visible must be in [1, 6]")
    return args


def stable_split(name: str, val_fraction: float) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if value < val_fraction else "train"


def crop_transform(
    xy: np.ndarray,
    visible: np.ndarray,
    crop_scale: float,
) -> np.ndarray:
    points = xy[visible]
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    side = max(float((maximum - minimum).max()) * crop_scale, 32.0)
    scale = IMAGE_SIZE / side
    return np.asarray(
        [
            [scale, 0.0, IMAGE_SIZE * 0.5 - scale * center[0]],
            [0.0, scale, IMAGE_SIZE * 0.5 - scale * center[1]],
        ],
        dtype=np.float32,
    )


def transform_points(xy: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (xy.astype(np.float32), np.ones((len(xy), 1), dtype=np.float32)), axis=1
    )
    return homogeneous @ transform.T


def draw_preview(image: np.ndarray, xy: np.ndarray, visible: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    for first, second in LSP_EDGES:
        if visible[first] and visible[second]:
            cv2.line(
                canvas,
                tuple(np.rint(xy[first]).astype(int)),
                tuple(np.rint(xy[second]).astype(int)),
                (30, 220, 80),
                2,
                cv2.LINE_AA,
            )
    for index, (x, y) in enumerate(xy):
        if visible[index]:
            color = (40, 80, 255) if index < 6 else (255, 180, 40)
            cv2.circle(canvas, (round(float(x)), round(float(y))), 3, color, -1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    lsp_root = args.lsp_root.resolve()
    output_root = args.output_root.resolve()
    image_root = lsp_root / "images"
    joints_path = lsp_root / "joints.mat"
    if not image_root.is_dir() or not joints_path.is_file():
        raise FileNotFoundError(f"invalid LSP root: {lsp_root}")
    if (output_root / "train.jsonl").exists() or (output_root / "val.jsonl").exists():
        raise FileExistsError(f"prepared manifests already exist: {output_root}")

    joints = loadmat(joints_path)["joints"]
    if joints.shape != (14, 3, 10000):
        raise ValueError(f"unexpected joints.mat shape: {joints.shape}")

    records: dict[str, list[dict]] = {"train": [], "val": []}
    previews = []
    rejected = []
    total = min(joints.shape[-1], args.limit or joints.shape[-1])
    for source_index in range(total):
        image_name = f"im{source_index + 1:05d}.jpg"
        source_path = image_root / image_name
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": image_name, "reason": "unreadable"})
            continue

        xy = joints[:, :2, source_index].astype(np.float32)
        visible = joints[:, 2, source_index] > 0.5
        height, width = image.shape[:2]
        visible &= (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        if int(visible[:6].sum()) < args.min_lower_visible:
            rejected.append({"image": image_name, "reason": "lower_body_visibility"})
            continue

        transform = crop_transform(xy, visible, args.crop_scale)
        crop = cv2.warpAffine(
            image,
            transform,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        crop_xy = transform_points(xy, transform)
        visible &= (
            (crop_xy[:, 0] >= 0)
            & (crop_xy[:, 0] < IMAGE_SIZE)
            & (crop_xy[:, 1] >= 0)
            & (crop_xy[:, 1] < IMAGE_SIZE)
        )

        split = stable_split(image_name, args.val_fraction)
        destination = output_root / "images" / split / image_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        ):
            raise RuntimeError(f"could not write {destination}")

        records[split].append(
            {
                "image": str(destination),
                "source_image": str(source_path),
                "source_index": source_index,
                "joints_2d": (crop_xy / IMAGE_SIZE).tolist(),
                "visible": visible.astype(int).tolist(),
                "lower_visible": int(visible[:6].sum()),
            }
        )
        if len(previews) < 16:
            previews.append(draw_preview(crop, crop_xy, visible))

    output_root.mkdir(parents=True, exist_ok=True)
    for split, split_records in records.items():
        manifest = output_root / f"{split}.jsonl"
        with manifest.open("w", encoding="utf-8") as file:
            for record in split_records:
                file.write(json.dumps(record, separators=(",", ":")) + "\n")
    with (output_root / "rejected.jsonl").open("w", encoding="utf-8") as file:
        for record in rejected:
            file.write(json.dumps(record, separators=(",", ":")) + "\n")

    if previews:
        rows = []
        for start in range(0, len(previews), 4):
            row = previews[start : start + 4]
            while len(row) < 4:
                row.append(np.zeros_like(previews[0]))
            rows.append(np.concatenate(row, axis=1))
        cv2.imwrite(str(output_root / "annotation_preview.jpg"), np.concatenate(rows, axis=0))

    summary = {
        "lsp_root": str(lsp_root),
        "source_commit": "8fabf3183516b1be5f9d44bd1c516609637e983e",
        "crop_scale": args.crop_scale,
        "min_lower_visible": args.min_lower_visible,
        "train_images": len(records["train"]),
        "val_images": len(records["val"]),
        "rejected_images": len(rejected),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
