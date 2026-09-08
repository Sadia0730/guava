#!/usr/bin/env python
"""Evaluate a trained AvatarBudget router on held-out render-damage labels."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.train_budget_router import evaluate, load_data
from avatarbudget.router import RenderImpactRouterNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RenderImpactRouterNet(
        feature_dim=int(checkpoint.get("feature_dim", 7)),
        hidden_dim=int(checkpoint.get("hidden_dim", 32)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    features, damage = load_data(args.data)
    report = {
        "features_shape": list(features.shape),
        "damage_shape": list(damage.shape),
        **evaluate(model, features, damage, torch.device(args.device)),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
