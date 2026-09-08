#!/usr/bin/env python
"""Build render-impact router labels from saved full and part-skipped renders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatarbudget.contracts import PARTS
from avatarbudget.impact import counterfactual_render_damage


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="PT file with features [N,4,7], reference [N,3,H,W], skipped [N,4,3,H,W].",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = torch.load(args.input, map_location="cpu", weights_only=True)
    features = bundle["features"].float()
    reference = bundle["reference"].float()
    skipped = bundle["skipped"].float()
    if features.ndim != 3 or tuple(features.shape[1:]) != (len(PARTS), 7):
        raise ValueError(f"features must be [N,4,7], got {tuple(features.shape)}")
    if reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError(f"reference must be [N,3,H,W], got {tuple(reference.shape)}")
    if skipped.shape[:3] != (len(reference), len(PARTS), 3):
        raise ValueError(f"skipped must be [N,4,3,H,W], got {tuple(skipped.shape)}")

    labels = []
    components = {"l1": [], "roi_l1": [], "ssim_damage": [], "silhouette_damage": []}
    for part_index in range(len(PARTS)):
        result = counterfactual_render_damage(reference, skipped[:, part_index])
        labels.append(result["total"])
        for key in components:
            components[key].append(result[key])
    output = {
        "features": features,
        "damage": torch.stack(labels, dim=1),
        **{key: torch.stack(values, dim=1) for key, values in components.items()},
        "parts": [part.value for part in PARTS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"features: {list(output['features'].shape)}")
    print(f"damage: {list(output['damage'].shape)}")
    print(f"router dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
