#!/usr/bin/env python
"""Strictly audit a PEAR student checkpoint and export a tensor-only copy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
sys.path.insert(0, str(ROOT))

from tools.pear_debug_utils import (  # noqa: E402
    OUTPUT_HEAD_PREFIXES,
    as_primitive,
    compare_state_dicts,
    find_state_dict,
    grouped_parameter_norms,
    prefix_coverage,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sanitized-output", type=Path, required=True)
    return parser.parse_args()


def resolve_config(path: Path) -> Path:
    candidates = [path, ROOT / path, PEAR_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"student config not found; tried: {candidates}")


def instantiate_student(config_path: Path):
    original_directory = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from models.pipeline.student_pipeline import PearStudentPipeline
        from utils.general_utils import ConfigDict, add_extra_cfgs

        config = add_extra_cfgs(ConfigDict(model_config_path=str(config_path)))
        return PearStudentPipeline(config), config
    finally:
        os.chdir(original_directory)
        sys.path.remove(str(PEAR_ROOT))


def top_level_report(checkpoint) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        return {"type": type(checkpoint).__name__, "keys": []}
    return {
        "type": type(checkpoint).__name__,
        "keys": list(checkpoint),
        "value_types": {key: type(value).__name__ for key, value in checkpoint.items()},
    }


def config_match_report(saved_args: dict, requested: Path, compatibility: dict) -> dict[str, object]:
    saved_name = saved_args.get("student_config") if isinstance(saved_args, dict) else None
    named_match = saved_name is not None and Path(str(saved_name)).name == requested.name
    architecture_compatible = (
        not compatibility["missing_keys"]
        and not compatibility["unexpected_keys"]
        and not compatibility["shape_mismatched_keys"]
    )
    return {
        "requested_config": str(requested),
        "checkpoint_saved_config": saved_name,
        "saved_path_name_matches": named_match,
        "instantiated_architecture_matches_all_tensors": architecture_compatible,
        "exact_training_yaml_hash_available": False,
        "exact_match": None,
        "verdict": (
            "The saved config path and instantiated tensor architecture match, but the checkpoint "
            "does not embed the original YAML or its hash, so byte-for-byte training-config "
            "identity cannot be proven."
            if named_match and architecture_compatible
            else "The requested config does not match the checkpoint metadata and tensor architecture."
        ),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    config_path = resolve_config(args.config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    print(f"Trusted local load (weights_only=False): {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_key, checkpoint_state = find_state_dict(checkpoint)
    model, config = instantiate_student(config_path)
    model_state = model.state_dict()
    model_parameters = dict(model.named_parameters())
    model_buffers = dict(model.named_buffers())
    checkpoint_parameters = {
        key: checkpoint_state[key] for key in model_parameters if key in checkpoint_state
    }
    compatibility = compare_state_dicts(checkpoint_state, model_state)

    coverage_prefixes = (
        "backbone",
        "backbone.stem",
        "backbone.stage1",
        "backbone.stage2",
        "backbone.stage3",
        "backbone.stage4",
        "backbone.transformer",
        "head",
        "head.transformer",
        *OUTPUT_HEAD_PREFIXES,
    )
    coverage = {
        prefix: prefix_coverage(checkpoint_state, model_state, prefix)
        for prefix in coverage_prefixes
    }
    essential = ("backbone", "head.transformer", *OUTPUT_HEAD_PREFIXES)
    essential_missing = [prefix for prefix in essential if not coverage[prefix]["complete"]]

    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    config_match = config_match_report(saved_args, config_path, compatibility)
    teacher_head_possible = int(config.HEAD.context_dim) == 1280 and int(config.HEAD.depth) == 6
    phase_report = {
        "phase_one_frozen_teacher_head": {
            "present": bool(
                teacher_head_possible
                and saved_args.get("init_head_from_teacher")
                and saved_args.get("freeze_head")
                and coverage["head"]["complete"]
            ),
            "teacher_head_architecture_compatible": teacher_head_possible,
            "checkpoint_flags": {
                "init_head_from_teacher": saved_args.get("init_head_from_teacher"),
                "freeze_head": saved_args.get("freeze_head"),
            },
            "note": (
                "This L70 checkpoint has a 640-context, five-layer student head, so it cannot "
                "contain the frozen 1280-context, six-layer teacher head used by the v2 recipe."
                if not teacher_head_possible
                else "Presence is based on architecture plus explicit checkpoint flags."
            ),
        },
        "phase_two_updated_head": {
            "weights_present": coverage["head"]["complete"],
            "proven_updated_by_optimizer": False,
            "note": (
                "All student-head tensors are present, but this older checkpoint has no phase or "
                "freeze-head metadata and no initial-head snapshot, so an optimizer update cannot "
                "be proven from the checkpoint alone."
            ),
        },
    }

    norm_prefixes = [
        "backbone",
        "backbone.stem",
        "backbone.stage1",
        "backbone.stage2",
        "backbone.stage3",
        "backbone.stage4",
        "backbone.proj",
        "backbone.transformer",
        "head.transformer",
        *OUTPUT_HEAD_PREFIXES,
    ]
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "top_level": top_level_report(checkpoint),
        "training_step": as_primitive(checkpoint.get("step")),
        "saved_training_arguments": as_primitive(saved_args),
        "weight_source_key": source_key,
        "compatibility": compatibility,
        "model_parameter_tensor_count": len(model_parameters),
        "model_parameter_count": sum(value.numel() for value in model_parameters.values()),
        "model_buffer_tensor_count": len(model_buffers),
        "model_buffer_element_count": sum(value.numel() for value in model_buffers.values()),
        "coverage": coverage,
        "essential_missing_or_incompatible": essential_missing,
        "cnn_backbone_loaded": coverage["backbone"]["complete"],
        "spatial_transformer_loaded": coverage["backbone.transformer"]["complete"],
        "decoder_transformer_loaded": coverage["head.transformer"]["complete"],
        "all_output_heads_loaded": all(coverage[prefix]["complete"] for prefix in OUTPUT_HEAD_PREFIXES),
        "training_phase_evidence": phase_report,
        "config_match": config_match,
        "parameter_norms": grouped_parameter_norms(checkpoint_parameters, norm_prefixes),
        "strict_load_passed": False,
        "sanitized_checkpoint": str(args.sanitized_output.resolve()),
    }

    if (
        essential_missing
        or compatibility["missing_keys"]
        or compatibility["unexpected_keys"]
        or compatibility["shape_mismatched_keys"]
    ):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(
            "essential checkpoint weights are missing or incompatible; see " f"{args.output.resolve()}"
        )

    model.load_state_dict(checkpoint_state, strict=True)
    report["strict_load_passed"] = True
    sanitized = {
        "format_version": 1,
        "kind": "pear_student_inference",
        "step": int(checkpoint.get("step", 0)),
        "student": {key: tensor.detach().cpu().clone() for key, tensor in checkpoint_state.items()},
        "metadata": {
            "source_checkpoint_sha256": report["checkpoint_sha256"],
            "source_weight_key": source_key,
            "student_config": str(config_path.relative_to(PEAR_ROOT)),
            "saved_training_arguments": as_primitive(saved_args),
            "strict_load_passed": True,
        },
    }
    args.sanitized_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sanitized, args.sanitized_output)
    # Verify the new file through the safe loader before declaring success.
    safe_copy = torch.load(args.sanitized_output, map_location="cpu", weights_only=True)
    safe_source, safe_state = find_state_dict(safe_copy)
    safe_compatibility = compare_state_dicts(safe_state, model_state)
    report["sanitized_verification"] = {
        "weights_only_load_passed": True,
        "weight_source_key": safe_source,
        "strict_compatibility": not any(
            safe_compatibility[key]
            for key in ("missing_keys", "unexpected_keys", "shape_mismatched_keys")
        ),
        "size_bytes": args.sanitized_output.stat().st_size,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Strict match: {compatibility['matched_tensor_count']}/{compatibility['model_tensor_count']} "
        f"tensors, {compatibility['matched_parameter_percent_of_model']:.2f}% model parameters"
    )
    print(f"Audit: {args.output.resolve()}")
    print(f"Sanitized checkpoint: {args.sanitized_output.resolve()}")


if __name__ == "__main__":
    main()
