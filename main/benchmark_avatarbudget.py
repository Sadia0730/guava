#!/usr/bin/env python
"""Run and verify the complete AvatarBudget pipeline on a finite video."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source_data_path", type=Path, required=True)
    parser.add_argument("--student_ckpt", type=Path, required=True)
    parser.add_argument("--student_config", type=Path, default=Path("configs/student_l70.yaml"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/avatarbudget_rtx3080_laptop.yaml")
    parser.add_argument("--router_ckpt", type=Path)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render_size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/avatarbudget/benchmark.json")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    if args.frames < 110:
        raise ValueError("Use at least 110 frames so 100 remain after the default warmup")

    command = [
        sys.executable,
        "main/live_avatarbudget.py",
        "--input", str(args.input.resolve()),
        "--source_data_path", str(args.source_data_path.resolve()),
        "--student_ckpt", str(args.student_ckpt.resolve()),
        "--student_config", str(args.student_config),
        "--config", str(args.config.resolve()),
        "--device", args.device,
        "--render_size", str(args.render_size),
        "--precision", args.precision,
        "--max_frames", str(args.frames),
        "--no_display",
        "--report", str(args.report.resolve()),
    ]
    if args.router_ckpt is not None:
        command.extend(("--router_ckpt", str(args.router_ckpt.resolve())))
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(json.dumps(report["end_to_end"], indent=2))
    if not report["end_to_end"]["sustained_50fps_verified"]:
        print("Result: complete-pipeline 50 FPS was not verified.")
    else:
        print("Result: complete-pipeline 50 FPS verification criteria passed.")


if __name__ == "__main__":
    main()
