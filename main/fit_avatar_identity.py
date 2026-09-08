#!/usr/bin/env python
"""Fit and cache one GUAVA/EHM identity from a source image."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from main.live_pear_guava import ROOT, track_source_image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_image", type=Path, required=True)
    parser.add_argument(
        "--output_dir", type=Path, default=ROOT / "outputs" / "avatar_identities"
    )
    parser.add_argument("--tracking_python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_image = args.source_image.resolve()
    if not source_image.is_file():
        raise FileNotFoundError(source_image)
    tracking_args = SimpleNamespace(
        source_tracking_output_dir=args.output_dir.resolve(),
        source_tracking_python=args.tracking_python,
        force_source_tracking=args.force,
    )
    tracked_path, seconds = track_source_image(source_image, tracking_args)
    report = {
        "source_image": str(source_image),
        "tracked_identity": str(tracked_path.resolve()),
        "identity_parameters": str((tracked_path / "id_share_params.pkl").resolve()),
        "tracking_seconds": seconds,
        "cache_hit": seconds == 0.0,
    }
    report_path = tracked_path / "avatarbudget_identity.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
