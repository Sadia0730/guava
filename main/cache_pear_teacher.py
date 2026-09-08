#!/usr/bin/env python
"""Cache PEAR teacher outputs and backbone features for distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatarbudget.cache import TeacherCacheWriter, file_fingerprint


PEAR_ROOT = ROOT / "third_party" / "PEAR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--teacher_config", default="configs/infer.yaml")
    parser.add_argument("--teacher_ckpt", type=Path, default=None)
    parser.add_argument("--download_teacher", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--no_features", action="store_true")
    return parser.parse_args()


def manifest_images(manifests: list[Path]) -> list[Path]:
    images: dict[str, Path] = {}
    for manifest in manifests:
        manifest = manifest.resolve()
        with manifest.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                directory = Path(item["directory"])
                if not directory.is_absolute():
                    directory = (manifest.parent / directory).resolve()
                for index in range(int(item["num_frames"])):
                    image = (directory / f"{index:06d}.jpg").resolve()
                    if not image.is_file():
                        raise FileNotFoundError(
                            f"Missing frame for {manifest}:{line_number}: {image}"
                        )
                    images[str(image)] = image
    return list(images.values())


def load_images(paths: list[Path], device: torch.device) -> torch.Tensor:
    images = []
    for path in paths:
        image = read_image(str(path), mode=ImageReadMode.RGB)
        if tuple(image.shape) != (3, 256, 256):
            raise ValueError(f"Expected [3, 256, 256], got {tuple(image.shape)}: {path}")
        images.append(image.float().div_(255.0))
    return torch.stack(images).to(device, non_blocking=True)


def tensor_shapes(value, prefix="") -> dict[str, list[int]]:
    shapes = {}
    if torch.is_tensor(value):
        shapes[prefix] = list(value.shape)
    elif isinstance(value, dict):
        for key, item in value.items():
            shapes.update(tensor_shapes(item, f"{prefix}.{key}" if prefix else key))
    return shapes


def autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    paths = manifest_images(args.manifest)
    if not paths:
        raise ValueError("manifests contain no images")

    original_directory = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from huggingface_hub import hf_hub_download
        from models.pipeline.ehm_pipeline import Ehm_Pipeline
        from utils.general_utils import ConfigDict, add_extra_cfgs

        if args.teacher_ckpt is None:
            if not args.download_teacher:
                raise ValueError("Pass --teacher_ckpt or --download_teacher")
            args.teacher_ckpt = Path(
                hf_hub_download(
                    repo_id="BestWJH/PEAR_models",
                    filename="pear_model.pt",
                    repo_type="model",
                )
            )
        checkpoint_path = args.teacher_ckpt.resolve()
        config = add_extra_cfgs(ConfigDict(model_config_path=args.teacher_config))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        teacher = Ehm_Pipeline(config)
        teacher.backbone.load_state_dict(checkpoint["backbone"], strict=False)
        teacher.head.load_state_dict(checkpoint["head"], strict=False)
        teacher = teacher.to(args.device).eval()
        del checkpoint

        writer = TeacherCacheWriter(
            args.output_dir.resolve(),
            metadata={
                "teacher_checkpoint": str(checkpoint_path),
                "teacher_sha256": file_fingerprint(checkpoint_path),
                "teacher_config": args.teacher_config,
                "features": not args.no_features,
                "manifests": [str(path.resolve()) for path in args.manifest],
                "storage_dtype": "float16",
            },
        )
        first_shapes = None
        device = torch.device(args.device)
        with torch.inference_mode():
            for start in range(0, len(paths), args.batch_size):
                batch_paths = paths[start : start + args.batch_size]
                images = load_images(batch_paths, device)
                with autocast(device, args.precision):
                    output = teacher(images)
                    features = None if args.no_features else teacher.forward_features(images)
                payload = {"output": output, "features": features}
                if first_shapes is None:
                    first_shapes = tensor_shapes(payload)
                    writer.metadata["tensor_shapes"] = first_shapes
                    print("Cached tensor shapes (batch dimension shown):")
                    for name, shape in sorted(first_shapes.items()):
                        print(f"  {name}: {shape}")
                writer.add_shard(batch_paths, payload)
                print(f"cached {min(start + len(batch_paths), len(paths))}/{len(paths)}", flush=True)
        index_path = writer.close()
    finally:
        os.chdir(original_directory)
        sys.path.remove(str(PEAR_ROOT))

    print(f"Teacher cache: {index_path}")


if __name__ == "__main__":
    main()
