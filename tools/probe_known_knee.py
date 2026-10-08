"""Compare pose estimators on the saved controlled GUAVA knee-bend renders.

This is a synthetic probe, not a validation of real webcam accuracy.
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from roma import rotmat_to_rotvec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compare_teacher_student import parse_args, load_models, compare_category, cleanup_pear_modules


def summarize(output_dir):
    saved = np.load(output_dir / "synthetic_known_knee/body_rotations.npz")
    responses = []
    for frame, (pose, joint, angle) in enumerate((
        ("left_knee", 3, 1.2), ("right_knee", 4, 1.2),
        ("left_hip_side", 0, 0.6), ("right_hip_side", 1, -0.6),
        ("left_hip_lift", 0, -0.8)), start=1):
        row = {"pose": pose, "commanded_deg": abs(float(np.rad2deg(angle)))}
        for model in ("teacher", "student"):
            rotations = torch.from_numpy(saved[model])
            relative = rotations[frame, joint] @ rotations[0, joint].T
            row[f"{model}_response_deg"] = float(torch.rad2deg(rotmat_to_rotvec(relative).norm()))
        responses.append(row)
    summary = {"scope": "Synthetic rendered probes; not real webcam validation", "responses": responses}
    (output_dir / "controlled_response.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    frames = []
    for name in ("full_rest", "full_left_knee", "full_right_knee",
                 "full_left_hip_side", "full_right_hip_side", "full_left_hip_lift"):
        path = Path("outputs/debug/live_leg_geometry") / f"{name}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        frames.append(image[:, :image.shape[1] // 2].copy())
    teacher, student, _, _ = load_models(args)
    try:
        compare_category("synthetic_known_knee", frames, 1., teacher, student, args)
        summarize(args.output_dir)
    finally:
        cleanup_pear_modules()


if __name__ == "__main__":
    main()
