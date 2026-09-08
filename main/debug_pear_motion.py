"""Measure whether PEAR/student outputs change on a live stream.

This is a lightweight diagnostic for the live PEAR -> GUAVA path.  It does not
load GUAVA or render; it only reports consecutive-frame pose deltas so we can
separate a frozen pose estimator from a GUAVA pose-consumption issue.
"""

from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path

import cv2
import torch

from live_pear_guava import FLAME_KEYS, ROTATION_KEYS, initialize_pear


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="0", help="webcam index, /dev/video path, video path, or auto")
    parser.add_argument("--pear_backend", choices=("teacher", "student"), default="student")
    parser.add_argument("--student_config", type=Path, default=Path("configs/student_l70.yaml"))
    parser.add_argument("--student_ckpt", type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--compile_targets", nargs="*", default=(), choices=("pear", "deform", "refiner"))
    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--input_crop_scale",
        type=float,
        default=1.0,
        help="Center-crop the live frame by this zoom factor before PEAR.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--pear_stride", type=int, default=1)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def open_debug_capture(value):
    candidates = []
    if value == "auto":
        devices = sorted(glob.glob("/dev/video*"))
        candidates.extend(devices)
        candidates.extend(str(index) for index in range(8))
    else:
        candidates.append(value)

    tried = []
    for candidate in candidates:
        source = int(candidate) if str(candidate).isdigit() else candidate
        capture = cv2.VideoCapture(source)
        tried.append(candidate)
        if capture.isOpened():
            ok, _ = capture.read()
            if ok:
                print(f"Opened camera/input: {candidate}")
                return capture
        capture.release()
    raise RuntimeError(
        "Could not open any camera/input. Tried: "
        f"{', '.join(tried) or value}. Close the live demo or browser camera first, "
        "then try --input auto or a specific /dev/videoN."
    )


def flatten_params(params):
    keys = ROTATION_KEYS + ("exp",) + FLAME_KEYS
    return {key: params[key].detach().float().cpu().reshape(-1) for key in keys}


def main():
    args = parse_args()
    if args.pear_backend == "student" and args.student_ckpt is None:
        raise SystemExit("--student_ckpt is required with --pear_backend student")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA is not available")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print("Loading PEAR/student only...")
    pear, _ = initialize_pear(args)
    capture = open_debug_capture(args.input)

    for _ in range(max(0, args.skip)):
        capture.read()

    previous = None
    rows = []
    totals = []
    try:
        with torch.inference_mode():
            for index in range(args.frames):
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.perf_counter()
                params = flatten_params(pear._infer(frame))
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if previous is None:
                    deltas = {key: 0.0 for key in params}
                    total = 0.0
                else:
                    deltas = {
                        key: float((params[key] - previous[key]).abs().mean())
                        for key in params
                    }
                    total = sum(deltas.values()) / len(deltas)
                totals.append(total)
                row = {"frame": index, "infer_ms": elapsed_ms, "mean_delta": total, **deltas}
                rows.append(row)
                print(
                    f"frame={index:03d} infer={elapsed_ms:7.1f} ms "
                    f"mean_delta={total:.6f} body={deltas['body_pose']:.6f} "
                    f"lh={deltas['left_hand_pose']:.6f} rh={deltas['right_hand_pose']:.6f} "
                    f"face_exp={deltas['expression_params']:.6f}"
                )
                previous = params
    finally:
        capture.release()

    if totals:
        nonzero = [value for value in totals[1:] if value > 1e-5]
        print(
            f"summary frames={len(totals)} mean_delta={sum(totals[1:]) / max(len(totals)-1, 1):.6f} "
            f"max_delta={max(totals):.6f} changing_pairs={len(nonzero)}/{max(len(totals)-1, 0)}"
        )
    if args.csv is not None and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
