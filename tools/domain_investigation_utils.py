"""CPU-safe utilities for PEAR validation and input-domain investigations."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ManifestRecord:
    source: str
    sequence: str
    directory: Path
    num_frames: int
    fps: float
    frame_step: int
    line_number: int


@dataclass(frozen=True)
class FrameReference:
    source: str
    sequence: str
    frame_index: int
    path: Path


def remap_directory(
    directory: str | Path,
    manifest_path: Path,
    root_from: Path | None = None,
    root_to: Path | None = None,
) -> Path:
    """Resolve a manifest directory, optionally replacing one absolute root."""
    value = Path(directory)
    if value.is_absolute() and root_from is not None:
        if root_to is None:
            raise ValueError("root_to is required when root_from is set")
        try:
            value = root_to / value.relative_to(root_from)
        except ValueError:
            pass
    elif not value.is_absolute():
        value = manifest_path.resolve().parent / value
    return value.resolve()


def load_manifest_records(
    manifest_path: Path,
    root_from: Path | None = None,
    root_to: Path | None = None,
) -> list[ManifestRecord]:
    manifest_path = manifest_path.resolve()
    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            record = ManifestRecord(
                source=str(item["source"]),
                sequence=str(item["sequence"]),
                directory=remap_directory(
                    item["directory"], manifest_path, root_from=root_from, root_to=root_to
                ),
                num_frames=int(item["num_frames"]),
                fps=float(item["fps"]),
                frame_step=int(item.get("frame_step", 1)),
                line_number=line_number,
            )
            if record.num_frames < 1 or record.fps <= 0 or record.frame_step < 1:
                raise ValueError(f"invalid sequence metadata on line {line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return records


def frame_path(record: ManifestRecord, index: int) -> Path:
    if not 0 <= index < record.num_frames:
        raise IndexError(index)
    return record.directory / f"{index:06d}.jpg"


def validate_manifest_files(records: Iterable[ManifestRecord]) -> dict[str, Any]:
    """Check every frame named by a manifest, not only sequence directories."""
    checked = 0
    missing: list[str] = []
    missing_directories: list[str] = []
    for record in records:
        if not record.directory.is_dir():
            missing_directories.append(str(record.directory))
            continue
        for index in range(record.num_frames):
            checked += 1
            path = frame_path(record, index)
            if not path.is_file():
                missing.append(str(path))
    return {
        "checked_frame_count": checked,
        "missing_frame_count": len(missing),
        "missing_frames": missing,
        "missing_directory_count": len(missing_directories),
        "missing_directories": missing_directories,
        "valid": not missing and not missing_directories,
    }


def deterministic_random_frames(
    records: list[ManifestRecord], count: int, seed: int
) -> list[FrameReference]:
    """Sample unique frames uniformly over all manifest frames without expanding all paths."""
    if count < 1:
        return []
    total = sum(record.num_frames for record in records)
    count = min(count, total)
    offsets = []
    running = 0
    for record in records:
        offsets.append(running)
        running += record.num_frames
    sampled = sorted(random.Random(seed).sample(range(total), count))
    references = []
    record_index = 0
    for flat_index in sampled:
        while (
            record_index + 1 < len(records)
            and flat_index >= offsets[record_index] + records[record_index].num_frames
        ):
            record_index += 1
        record = records[record_index]
        local_index = flat_index - offsets[record_index]
        references.append(
            FrameReference(
                record.source,
                record.sequence,
                local_index,
                frame_path(record, local_index),
            )
        )
    return references


def deterministic_sequential_clips(
    records: list[ManifestRecord],
    count: int,
    clip_length: int,
    temporal_stride: int,
    seed: int,
) -> list[list[FrameReference]]:
    if count < 1:
        return []
    if clip_length < 2 or temporal_stride < 1:
        raise ValueError("clip_length must be >= 2 and temporal_stride must be positive")
    span = (clip_length - 1) * temporal_stride + 1
    candidates = [record for record in records if record.num_frames >= span]
    if not candidates:
        raise ValueError(f"no sequence is long enough for a {span}-frame span")
    rng = random.Random(seed)
    clips = []
    for _ in range(count):
        record = rng.choice(candidates)
        start = rng.randrange(record.num_frames - span + 1)
        clips.append(
            [
                FrameReference(
                    record.source,
                    record.sequence,
                    index,
                    frame_path(record, index),
                )
                for index in range(start, start + span, temporal_stride)
            ]
        )
    return clips


def reference_to_json(reference: FrameReference) -> dict[str, Any]:
    value = asdict(reference)
    value["path"] = str(reference.path)
    return value


def square_person_crop(
    frame: np.ndarray,
    box: np.ndarray | tuple[float, float, float, float],
    crop_scale: float = 1.25,
    output_size: int = 256,
) -> np.ndarray:
    """Affine square crop around an xyxy person box, with black padding when needed."""
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (4,):
        raise ValueError("box must contain x1,y1,x2,y2")
    center_x = float(box[0] + box[2]) * 0.5
    center_y = float(box[1] + box[3]) * 0.5
    side = max(float(box[2] - box[0]), float(box[3] - box[1])) * crop_scale
    if side <= 1.0 or crop_scale <= 0:
        raise ValueError("person box and crop scale must define a positive crop")
    scale = output_size / side
    transform = np.array(
        [
            [scale, 0.0, output_size * 0.5 - scale * center_x],
            [0.0, scale, output_size * 0.5 - scale * center_y],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(frame, transform, (output_size, output_size), flags=cv2.INTER_LINEAR)


def embed_in_canvas(
    crop: np.ndarray,
    canvas_size: tuple[int, int] = (640, 480),
    scale: float = 0.6,
) -> np.ndarray:
    """Center one validation crop in a wider deployment-like BGR canvas."""
    width, height = canvas_size
    if width < 1 or height < 1 or not 0 < scale <= 1:
        raise ValueError("invalid canvas dimensions or scale")
    target = max(1, int(min(width, height) * scale))
    resized = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=crop.dtype)
    x0 = (width - target) // 2
    y0 = (height - target) // 2
    canvas[y0 : y0 + target, x0 : x0 + target] = resized
    return canvas


def mean_frame_pixel_motion(frames: list[np.ndarray]) -> float | None:
    if len(frames) < 2:
        return None
    deltas = [
        np.abs(right.astype(np.float32) - left.astype(np.float32)).mean()
        for left, right in zip(frames, frames[1:])
    ]
    return float(np.mean(deltas))
