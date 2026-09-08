#!/usr/bin/env python
"""Train the render-impact-aware AvatarBudget router."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatarbudget.contracts import PARTS
from avatarbudget.router import RenderImpactRouterNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def load_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    features = value["features"].float()
    damage = value["damage"].float()
    if features.ndim != 3 or tuple(features.shape[1:]) != (len(PARTS), 7):
        raise ValueError(f"features must be [N,4,7], got {tuple(features.shape)}")
    if tuple(damage.shape) != tuple(features.shape[:2]):
        raise ValueError(f"damage must be [N,4], got {tuple(damage.shape)}")
    return features, damage


@torch.no_grad()
def evaluate(model, features, damage, device) -> dict[str, float]:
    model.eval()
    prediction = model(features.to(device)).cpu()
    absolute = (prediction - damage).abs()
    report = {
        "mae": float(absolute.mean()),
        "rmse": float((prediction - damage).square().mean().sqrt()),
    }
    for index, part in enumerate(PARTS):
        report[f"mae/{part.value}"] = float(absolute[:, index].mean())
    model.train()
    return report


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_features, train_damage = load_data(args.train_data)
    if args.val_data is None:
        split = max(1, int(0.9 * len(train_features)))
        val_features, val_damage = train_features[split:], train_damage[split:]
        train_features, train_damage = train_features[:split], train_damage[:split]
    else:
        val_features, val_damage = load_data(args.val_data)
    if len(val_features) == 0:
        raise ValueError("validation split is empty")

    print(f"train features: {list(train_features.shape)}; labels: {list(train_damage.shape)}")
    print(f"val features: {list(val_features.shape)}; labels: {list(val_damage.shape)}")
    loader = DataLoader(
        TensorDataset(train_features, train_damage),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    device = torch.device(args.device)
    model = RenderImpactRouterNet(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iterator = iter(loader)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"

    for step in range(1, args.steps + 1):
        try:
            features, damage = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            features, damage = next(iterator)
        features = features.to(device)
        damage = damage.to(device)
        prediction = model(features)
        loss = F.smooth_l1_loss(prediction, damage)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics = evaluate(model, val_features, val_damage, device)
            item = {"step": step, "train_loss": float(loss), **metrics}
            print(json.dumps(item), flush=True)
            with log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(item) + "\n")

    checkpoint = {
        "model": model.state_dict(),
        "feature_dim": 7,
        "hidden_dim": args.hidden_dim,
        "parts": [part.value for part in PARTS],
        "steps": args.steps,
        "validation": evaluate(model, val_features, val_damage, device),
    }
    output = args.output_dir / "router.pt"
    torch.save(checkpoint, output)
    print(f"router checkpoint: {output.resolve()}")


if __name__ == "__main__":
    main()
