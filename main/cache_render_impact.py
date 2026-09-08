#!/usr/bin/env python
"""Render PEAR/GUAVA counterfactuals and cache router training labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.io import ImageReadMode, read_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatarbudget import load_config
from avatarbudget.cache import TeacherOutputCache
from avatarbudget.contracts import PARTS, validate_pose_record
from avatarbudget.impact import CounterfactualRenderLabeler
from avatarbudget.router import BudgetRouter
from avatarbudget.scout import CheapScout
from avatarbudget.temporal import ConstantVelocityPredictor
from main.live_pear_guava import (
    FLAME_KEYS,
    ROTATION_KEYS,
    TargetBuilder,
    initialize_guava,
    matrix_to_rotation_6d,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher_cache", type=Path, required=True)
    parser.add_argument("--source_data_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/avatarbudget_rtx3080_laptop.yaml")
    parser.add_argument("--model_path", type=Path, default=ROOT / "assets/GUAVA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render_size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args()


def manifest_sequences(manifest: Path) -> list[list[Path]]:
    sequences = []
    manifest = manifest.resolve()
    with manifest.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            directory = Path(item["directory"])
            if not directory.is_absolute():
                directory = (manifest.parent / directory).resolve()
            paths = [(directory / f"{index:06d}.jpg").resolve()
                     for index in range(int(item["num_frames"]))]
            missing = next((path for path in paths if not path.is_file()), None)
            if missing is not None:
                raise FileNotFoundError(f"Missing frame at {manifest}:{line_number}: {missing}")
            sequences.append(paths)
    return sequences


def output_to_pose(output: dict[str, object], device: str | None = None) -> dict[str, torch.Tensor]:
    body = output["body_param"]
    flame = output["flame_param"]
    pose = {key: matrix_to_rotation_6d(body[key].float()) for key in ROTATION_KEYS}
    pose["exp"] = body["exp"].float()
    pose.update({key: flame[key].float() for key in FLAME_KEYS})
    if device is not None:
        pose = {key: value.to(device) for key, value in pose.items()}
    validate_pose_record(pose)
    return pose


def load_rgb(path: Path, device: str) -> torch.Tensor:
    image = read_image(str(path), mode=ImageReadMode.RGB).float().div_(255.0)
    if tuple(image.shape) != (3, 256, 256):
        raise ValueError(f"Expected [3,256,256], got {tuple(image.shape)}: {path}")
    return image.unsqueeze(0).to(device)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Counterfactual GUAVA rendering requires CUDA")
    config = load_config(args.config)
    cache = TeacherOutputCache(args.teacher_cache)
    sequences = manifest_sequences(args.manifest)
    if not sequences:
        raise ValueError("manifest contains no sequences")

    first_pose = output_to_pose(cache.get(sequences[0][0])["output"], args.device)
    loader_args = SimpleNamespace(
        model_path=args.model_path,
        source_data_path=args.source_data_path,
        device=args.device,
        render_size=args.render_size,
        precision=args.precision,
        compile_targets=[],
        compile_mode="default",
        warmup=1,
        smooth=False,
        smooth_min_cutoff=2.0,
        smooth_beta=0.3,
        smooth_d_cutoff=1.0,
    )
    setup_start = time.perf_counter()
    renderer, identity, source_dataset = initialize_guava(loader_args, first_pose)
    setup_seconds = time.perf_counter() - setup_start
    target_builder = TargetBuilder(identity, loader_args)

    def render_pose(pose):
        target = target_builder(pose, time.perf_counter())
        assets = renderer.avatar(target)
        return renderer.render_model(assets, renderer.camera, bg=0.0)["renders"][0].clamp(0.0, 1.0)

    labeler = CounterfactualRenderLabeler(render_pose)
    router = BudgetRouter(config.router)
    feature_rows = []
    damage_rows: dict[str, list[torch.Tensor]] = {
        "total": [],
        "l1": [],
        "roi_l1": [],
        "ssim_damage": [],
        "silhouette_damage": [],
    }
    processed = 0
    try:
        with torch.inference_mode():
            for paths in sequences:
                scout = CheapScout(config.scout).to(args.device).eval()
                predictor = ConstantVelocityPredictor(config.temporal)
                for path in paths:
                    teacher_pose = output_to_pose(cache.get(path)["output"], args.device)
                    scout_output = scout(load_rgb(path, args.device))
                    if not predictor.ready:
                        predictor.initialize(teacher_pose)
                        continue
                    prediction = predictor.predict()
                    features = router.feature_tensor(
                        scout_output, prediction.uncertainty, predictor.state.staleness
                    )[0]
                    damage = labeler(
                        teacher_pose, prediction.values, rois=scout_output.rois
                    )
                    feature_rows.append(features.cpu())
                    for key in damage_rows:
                        damage_rows[key].append(damage[key][0].cpu())
                    predictor.commit(prediction, teacher_pose, frozenset(PARTS))
                    processed += 1
                    print(f"counterfactual frames: {processed}", flush=True)
                    if args.max_frames and processed >= args.max_frames:
                        break
                if args.max_frames and processed >= args.max_frames:
                    break
    finally:
        source_dataset._lmdb_engine.close()

    if not feature_rows:
        raise ValueError("no labels produced; each sequence needs at least two frames")
    output = {
        "features": torch.stack(feature_rows),
        "damage": torch.stack(damage_rows.pop("total")),
        **{key: torch.stack(values) for key, values in damage_rows.items()},
        "parts": [part.value for part in PARTS],
        "metadata": {
            "manifest": str(args.manifest.resolve()),
            "teacher_cache": str(args.teacher_cache.resolve()),
            "source_data_path": str(args.source_data_path.resolve()),
            "render_size": args.render_size,
            "setup_seconds_excluded": setup_seconds,
            "label_definition": (
                "global L1 + 2 * part-ROI L1 + 0.25 * global-SSIM damage + "
                "0.5 * silhouette-IoU damage"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"features: {list(output['features'].shape)}")
    print(f"damage: {list(output['damage'].shape)}")
    print(f"render-impact cache: {args.output.resolve()}")


if __name__ == "__main__":
    main()
