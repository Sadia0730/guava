#!/usr/bin/env python
"""Validate a PEAR student manifest and create deterministic evaluation sample plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.domain_investigation_utils import (
    deterministic_random_frames,
    deterministic_sequential_clips,
    load_manifest_records,
    reference_to_json,
    validate_manifest_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root-from", type=Path)
    parser.add_argument("--root-to", type=Path)
    parser.add_argument("--random-frames", type=int, default=100)
    parser.add_argument("--sequential-clips", type=int, default=10)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--skip-full-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.root_from is None) != (args.root_to is None):
        raise ValueError("--root-from and --root-to must be supplied together")
    records = load_manifest_records(args.manifest, args.root_from, args.root_to)
    random_frames = deterministic_random_frames(records, args.random_frames, args.seed)
    clips = deterministic_sequential_clips(
        records,
        args.sequential_clips,
        args.clip_length,
        args.temporal_stride,
        args.seed + 1,
    )
    report = {
        "manifest": str(args.manifest.resolve()),
        "root_remap": {
            "from": str(args.root_from.resolve()) if args.root_from else None,
            "to": str(args.root_to.resolve()) if args.root_to else None,
        },
        "record_count": len(records),
        "total_frame_count": sum(record.num_frames for record in records),
        "sources": sorted({record.source for record in records}),
        "full_file_validation": (
            {"skipped": True}
            if args.skip_full_validation
            else validate_manifest_files(records)
        ),
        "random_frames": [reference_to_json(item) for item in random_frames],
        "sequential_clips": [
            [reference_to_json(item) for item in clip] for clip in clips
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("record_count", "total_frame_count", "sources", "full_file_validation")}, indent=2))
    print(f"sample plan: {args.output.resolve()}")


if __name__ == "__main__":
    main()
