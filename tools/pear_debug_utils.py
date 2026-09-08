"""Shared, CPU-testable utilities for PEAR checkpoint and output diagnostics."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


STATE_DICT_KEYS = ("student", "state_dict", "model", "ema")
OUTPUT_HEAD_PREFIXES = (
    "head.smplx_poses_decoder",
    "head.smplx_scale_decoder",
    "head.smplx_shape_decoder",
    "head.smplx_expression_decoder",
    "head.smplx_joint_decoder",
    "head.flame_poses_decoder",
    "head.flame_shape_decoder",
    "head.flame_expression_decoder",
    "head.cam_decoder",
)


def as_primitive(value: Any) -> Any:
    """Convert checkpoint metadata to weights-only-safe Python primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): as_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_primitive(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    return repr(value)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_tensor_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(item) for key, item in value.items()
    )


def find_state_dict(checkpoint: Any) -> tuple[str, Mapping[str, torch.Tensor]]:
    """Find model tensors without silently guessing among ambiguous candidates."""
    if is_tensor_state_dict(checkpoint):
        return "<root>", checkpoint
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")

    candidates = [key for key in STATE_DICT_KEYS if is_tensor_state_dict(checkpoint.get(key))]
    if not candidates:
        available = ", ".join(str(key) for key in checkpoint)
        raise KeyError(f"no tensor state dict found; top-level keys: {available}")
    if len(candidates) > 1:
        raise ValueError(f"ambiguous checkpoint: multiple model states found: {candidates}")
    key = candidates[0]
    return key, checkpoint[key]


def compare_state_dicts(
    checkpoint_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    matched = []
    shape_mismatched = []
    unexpected = []
    for key, tensor in checkpoint_state.items():
        expected = model_state.get(key)
        if expected is None:
            unexpected.append(key)
        elif tuple(tensor.shape) != tuple(expected.shape):
            shape_mismatched.append(
                {
                    "key": key,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(expected.shape),
                }
            )
        else:
            matched.append(key)
    missing = sorted(set(model_state) - set(checkpoint_state))
    matched.sort()
    unexpected.sort()

    checkpoint_numel = sum(tensor.numel() for tensor in checkpoint_state.values())
    model_numel = sum(tensor.numel() for tensor in model_state.values())
    matched_numel = sum(checkpoint_state[key].numel() for key in matched)
    return {
        "checkpoint_tensor_count": len(checkpoint_state),
        "checkpoint_numel": checkpoint_numel,
        "model_tensor_count": len(model_state),
        "model_numel": model_numel,
        "matched_tensor_count": len(matched),
        "matched_numel": matched_numel,
        "matched_tensor_percent_of_model": 100.0 * len(matched) / max(len(model_state), 1),
        "matched_parameter_percent_of_model": 100.0 * matched_numel / max(model_numel, 1),
        "matched_parameter_percent_of_checkpoint": 100.0 * matched_numel / max(checkpoint_numel, 1),
        "matched_keys": matched,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatched_keys": shape_mismatched,
    }


def prefix_coverage(
    checkpoint_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, Any]:
    expected = {
        key: value for key, value in model_state.items() if key == prefix or key.startswith(prefix + ".")
    }
    present = {
        key: value
        for key, value in checkpoint_state.items()
        if key == prefix or key.startswith(prefix + ".")
    }
    matching = [
        key for key, tensor in present.items() if key in expected and tensor.shape == expected[key].shape
    ]
    expected_numel = sum(tensor.numel() for tensor in expected.values())
    matching_numel = sum(present[key].numel() for key in matching)
    return {
        "prefix": prefix,
        "expected_tensors": len(expected),
        "present_tensors": len(present),
        "matching_tensors": len(matching),
        "expected_numel": expected_numel,
        "matching_numel": matching_numel,
        "coverage_percent": 100.0 * matching_numel / max(expected_numel, 1),
        "complete": bool(expected) and matching_numel == expected_numel,
    }


def grouped_parameter_norms(
    state: Mapping[str, torch.Tensor], prefixes: list[str] | tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    result = {}
    for prefix in prefixes:
        tensors = [
            tensor.detach().float()
            for key, tensor in state.items()
            if (key == prefix or key.startswith(prefix + ".")) and tensor.is_floating_point()
        ]
        squared = sum(float(torch.sum(tensor * tensor)) for tensor in tensors)
        count = sum(tensor.numel() for tensor in tensors)
        result[prefix] = {
            "floating_tensor_count": len(tensors),
            "parameter_count": count,
            "l2_norm": math.sqrt(squared),
            "rms": math.sqrt(squared / count) if count else 0.0,
        }
    return result


def tensor_tree(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if torch.is_tensor(value):
        result[prefix] = value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(tensor_tree(item, child))
    return result


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "nan_count": int(torch.isnan(value).sum()),
        "inf_count": int(torch.isinf(value).sum()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean()),
                "std": float(finite_values.std(unbiased=False)),
                "near_zero_count": int((finite_values.abs() < 1e-6).sum()),
            }
        )
    return result
