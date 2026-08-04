import argparse
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import lmdb
import numpy as np
import torch
import torchvision


ROOT = Path(__file__).resolve().parent
EHM_ROOT = ROOT / "EHM-Tracker"


def decode_lmdb_image(transaction, key):
    payload = transaction.get(key.encode())
    if payload is None:
        raise KeyError(f"LMDB key not found: {key}")
    try:
        encoded = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
        return torchvision.io.decode_image(encoded).permute(1, 2, 0).numpy()
    except Exception:
        return torch.load(io.BytesIO(payload), weights_only=True).permute(1, 2, 0).numpy()


def make_target_clip(lmdb_path, video_info_path, output_path, frame_count):
    video_info = json.loads(video_info_path.read_text())
    sequence = next(iter(video_info.values()))
    frame_keys = sequence["frames_keys"][:frame_count]
    if len(frame_keys) < frame_count:
        raise ValueError(f"Requested {frame_count} frames, but only {len(frame_keys)} are available")

    environment = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=True,
    )
    with environment.begin(write=False) as transaction:
        frames = [
            decode_lmdb_image(transaction, f"{frame_key}/ori_image")
            for frame_key in frame_keys
        ]
    environment.close()
    imageio.mimwrite(output_path, frames, fps=30, quality=8)


def timed_run(name, command, cwd):
    print(f"\n[{name}] {' '.join(map(str, command))}", flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True)
    seconds = time.perf_counter() - start
    print(f"[{name}] {seconds:.3f} seconds", flush=True)
    return seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark image tracking, video tracking, and GUAVA cross animation."
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=ROOT
        / "assets/example/tracked_image/random_google_pic/blue_shirt/blue_shirt.jpg",
    )
    parser.add_argument(
        "--target-tracked-dir",
        type=Path,
        default=ROOT / "assets/example/tracked_video/6gvP8f5WQyo__056",
        help="Existing tracked sequence used only to reconstruct a raw test clip.",
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/e2e_benchmark",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    input_dir = run_dir / "inputs"
    source_tracking_root = run_dir / "source_tracking"
    target_tracking_root = run_dir / "target_tracking"
    guava_output_root = run_dir / "guava_output"
    input_dir.mkdir(parents=True)

    target_clip = input_dir / f"target_{args.frames}f.mp4"
    prep_start = time.perf_counter()
    make_target_clip(
        args.target_tracked_dir / "img_lmdb",
        args.target_tracked_dir / "videos_info.json",
        target_clip,
        args.frames,
    )
    input_preparation_seconds = time.perf_counter() - prep_start
    stage_timings_path = run_dir / "stage_timings.json"
    stage_timings = {
        "input_preparation_seconds_excluded": input_preparation_seconds,
    }
    stage_timings_path.write_text(json.dumps(stage_timings, indent=2))

    source_tracking_seconds = timed_run(
        "source EHM tracking",
        [
            sys.executable,
            "-m",
            "src.tracking_single_image",
            "--input_dir",
            str(args.source_image.resolve()),
            "--output_dir",
            str(source_tracking_root.resolve()),
        ],
        EHM_ROOT,
    )
    stage_timings["source_ehm_tracking_seconds"] = source_tracking_seconds
    stage_timings_path.write_text(json.dumps(stage_timings, indent=2))

    target_tracking_seconds = timed_run(
        "target PEAR tracking",
        [
            sys.executable,
            "main/pear_tracking.py",
            "--in_root",
            str(target_clip.resolve()),
            "--output_dir",
            str(target_tracking_root.resolve()),
        ],
        ROOT,
    )
    stage_timings["target_pear_tracking_seconds"] = target_tracking_seconds
    stage_timings_path.write_text(json.dumps(stage_timings, indent=2))

    source_name = args.source_image.stem
    target_name = target_clip.stem
    source_tracked_dir = source_tracking_root / source_name
    target_tracked_dir = target_tracking_root / target_name
    guava_seconds = timed_run(
        "GUAVA avatar and animation",
        [
            sys.executable,
            "-m",
            "main.test",
            "-d",
            "0",
            "-m",
            "assets/GUAVA",
            "-s",
            str(guava_output_root.resolve()),
            "-n",
            "e2e",
            "--data_path",
            str(target_tracked_dir.resolve()),
            "--source_data_path",
            str(source_tracked_dir.resolve()),
            "--skip_self_act",
            "--render_cross_act",
        ],
        ROOT,
    )
    stage_timings["guava_avatar_and_animation_seconds"] = guava_seconds
    stage_timings_path.write_text(json.dumps(stage_timings, indent=2))

    render_dir = (
        guava_output_root
        / "e2e_cross_act"
        / source_name
        / f"{source_name}_{target_name}"
        / "render"
    )
    rendered_frames = len(list(render_dir.glob("*.png")))
    if rendered_frames == 0:
        raise RuntimeError(f"GUAVA produced no rendered frames in {render_dir}")

    total_seconds = (
        source_tracking_seconds + target_tracking_seconds + guava_seconds
    )
    report = {
        "source_image": str(args.source_image.resolve()),
        "target_clip": str(target_clip.resolve()),
        "requested_target_frames": args.frames,
        "rendered_frames": rendered_frames,
        "input_preparation_seconds_excluded": input_preparation_seconds,
        "source_ehm_tracking_seconds": source_tracking_seconds,
        "target_pear_tracking_seconds": target_tracking_seconds,
        "target_pear_tracking_fps": rendered_frames / target_tracking_seconds,
        "guava_avatar_and_animation_seconds": guava_seconds,
        "guava_cold_command_effective_fps": rendered_frames / guava_seconds,
        "end_to_end_seconds": total_seconds,
        "end_to_end_fps": rendered_frames / total_seconds,
        "timing_scope": (
            "Cold subprocess wall time. Includes model initialization, tracking parameter "
            "saving, GUAVA avatar inference, frame rendering, PNG saving, and MP4 encoding."
        ),
        "paper_fps_comparison": (
            "The paper's animation/rendering FPS excludes tracking, cold model loading, "
            "input loading, image saving, and MP4 encoding. This report does not measure that "
            "compute-only steady-state metric."
        ),
        "output_video": str(
            render_dir.parent / f"{source_name}_{target_name}_video.mp4"
        ),
    }
    report_path = run_dir / "timing_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n{json.dumps(report, indent=2)}")
    print(f"\nTiming report: {report_path}")


if __name__ == "__main__":
    main()
