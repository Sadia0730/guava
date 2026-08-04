#!/usr/bin/env python
"""Compare EHM- and PEAR-driven GUAVA animation on the same video frames."""

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EHM_DATA = ROOT / "assets/example/tracked_video/6gvP8f5WQyo__056"
DEFAULT_SOURCE_DATA = ROOT / "assets/example/tracked_image/random_google_pic/blue_shirt"
DEFAULT_OUTPUT = ROOT / "outputs/pear_ehm_comparison/6gvP8f5WQyo__056_35f_canonical"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ehm-data", type=Path, default=DEFAULT_EHM_DATA)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=35)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def save_pickle(path, value):
    with path.open("wb") as file:
        pickle.dump(value, file)


def subset_frame_dict(data, frame_keys):
    return {frame_key: data[frame_key] for frame_key in frame_keys}


def prepare_ehm_subset(source_dir, destination_dir, frame_count, motion_path, fps):
    videos_info = json.loads((source_dir / "videos_info.json").read_text())
    source_video_id, source_video_info = next(iter(videos_info.items()))
    frame_keys = source_video_info["frames_keys"][:frame_count]
    if len(frame_keys) != frame_count:
        raise ValueError(f"Requested {frame_count} frames, found {len(frame_keys)}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    tracking = load_pickle(source_dir / "optim_tracking_ehm.pkl")
    save_pickle(destination_dir / "optim_tracking_ehm.pkl", subset_frame_dict(tracking, frame_keys))
    shutil.copy2(source_dir / "id_share_params.pkl", destination_dir / "id_share_params.pkl")

    for filename in ("base_tracking.pkl", "optim_tracking_flame.pkl"):
        source_path = source_dir / filename
        if source_path.exists():
            save_pickle(destination_dir / filename,
                        subset_frame_dict(load_pickle(source_path), frame_keys))

    target_video_id = motion_path.stem
    output_info = {
        target_video_id: {
            "frames_num": len(frame_keys),
            "frames_keys": frame_keys,
            "source_video_id": source_video_id,
        }
    }
    (destination_dir / "videos_info.json").write_text(json.dumps(output_info, indent=2))

    source_lmdb = lmdb.open(str(source_dir / "img_lmdb"), readonly=True, lock=False,
                            readahead=False, meminit=True)
    destination_lmdb = lmdb.open(str(destination_dir / "img_lmdb"), map_size=2**32)
    video_writer = None
    with source_lmdb.begin(write=False) as source_txn, destination_lmdb.begin(write=True) as destination_txn:
        for frame_key in frame_keys:
            prefix = f"{frame_key}/".encode()
            cursor = source_txn.cursor()
            if not cursor.set_range(prefix):
                raise KeyError(f"No LMDB entries found for {frame_key}")
            for key, value in cursor:
                if not key.startswith(prefix):
                    break
                destination_txn.put(key, value)

            encoded_image = source_txn.get(f"{frame_key}/ori_image".encode())
            if encoded_image is None:
                raise KeyError(f"Missing original image for {frame_key}")
            frame = cv2.imdecode(np.frombuffer(encoded_image, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not decode original image for {frame_key}")
            if video_writer is None:
                height, width = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(motion_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            video_writer.write(frame)

    if video_writer is None:
        raise RuntimeError("No motion frames were written")
    video_writer.release()
    source_lmdb.close()
    destination_lmdb.close()


def run(command):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def run_guava(data_path, source_path, output_root, run_name, device):
    run([
        "conda", "run", "-n", "guava", "env", "PYTHONPATH=.",
        "python", "main/test.py",
        "-d", device,
        "-m", "assets/GUAVA",
        "-s", output_root,
        "-n", run_name,
        "--data_path", data_path,
        "--source_data_path", source_path,
        "--skip_self_act",
        "--render_cross_act",
        "--canonical_camera",
    ])


def find_animation(run_root, run_name):
    matches = list((run_root / f"{run_name}_cross_act").glob("*/*/*_video.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {run_name} animation, found: {matches}")
    return matches[0]


def fit_panel(frame, size=512):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)),
                         interpolation=cv2.INTER_AREA)
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def add_label(panel, label):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(panel, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


def make_comparison(motion_path, ehm_path, pear_path, output_path, metrics_path):
    captures = [cv2.VideoCapture(str(path)) for path in (motion_path, ehm_path, pear_path)]
    fps = captures[0].get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (512 * 4, 512))
    differences = []

    while True:
        reads = [capture.read() for capture in captures]
        if not all(success for success, _ in reads):
            break
        motion, ehm, pear = [fit_panel(frame) for _, frame in reads]
        difference = cv2.absdiff(ehm, pear)
        differences.append(float(difference.mean()))
        difference = cv2.applyColorMap(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY),
                                       cv2.COLORMAP_INFERNO)
        panels = [motion, ehm, pear, difference]
        labels = ["Driving motion", "EHM -> GUAVA", "PEAR -> GUAVA", "Absolute difference"]
        for panel, label in zip(panels, labels):
            add_label(panel, label)
        writer.write(np.concatenate(panels, axis=1))

    for capture in captures:
        capture.release()
    writer.release()
    if not differences:
        raise RuntimeError("No matching animation frames were available for comparison")

    metrics = {
        "compared_frames": len(differences),
        "mean_absolute_pixel_difference": float(np.mean(differences)),
        "motion_video": str(motion_path),
        "ehm_animation": str(ehm_path),
        "pear_animation": str(pear_path),
        "comparison_video": str(output_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))


def main():
    args = parse_args()
    ehm_data = args.ehm_data.resolve()
    source_data = args.source_data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_path = output_dir / "motion.mp4"
    ehm_tracking = output_dir / "ehm_tracking" / motion_path.stem
    pear_tracking_root = output_dir / "pear_tracking"
    pear_tracking = pear_tracking_root / motion_path.stem
    guava_runs = output_dir / "guava_runs"

    prepare_ehm_subset(ehm_data, ehm_tracking, args.frames, motion_path, args.fps)
    run([
        "conda", "run", "-n", "pear", "python", "main/pear_tracking.py",
        "--in_root", motion_path,
        "--output_dir", pear_tracking_root,
        "--device", f"cuda:{args.device}",
    ])

    run_guava(ehm_tracking, source_data, guava_runs, "ehm", args.device)
    run_guava(pear_tracking, source_data, guava_runs, "pear", args.device)

    ehm_animation = output_dir / "ehm_animation.mp4"
    pear_animation = output_dir / "pear_animation.mp4"
    shutil.copy2(find_animation(guava_runs, "ehm"), ehm_animation)
    shutil.copy2(find_animation(guava_runs, "pear"), pear_animation)
    make_comparison(
        motion_path,
        ehm_animation,
        pear_animation,
        output_dir / "comparison.mp4",
        output_dir / "comparison_metrics.json",
    )
    print(f"\nComparison workspace: {output_dir}")
    print(f"Debug video: {output_dir / 'comparison.mp4'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
