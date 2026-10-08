"""Run a reproducible, sampled 3DPW test comparison of PEAR teacher and student.

This is a GT-2D-crop pilot, not a claim to reproduce PEAR Table 3's crop protocol.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile

import cv2
import joblib
import numpy as np
from scipy import sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party/PEAR"
SMPL_PATH = Path("/data/AnimatedHumanExp/third_parties/smplx/SMPL/smpl/SMPL_NEUTRAL.pkl")
STUDENT_CKPT = ROOT / "outputs/checkpoints/student_v2_step0235000_inference.pt"
TEACHER_CKPT = Path("/home/virlab/.cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt")
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def align_similarity(pred, gt):
    """Per-person similarity fit, with a proper (non-reflecting) rotation."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    x = pred - pred.mean(axis=0)
    y = gt - gt.mean(axis=0)
    variance = np.sum(x * x)
    if variance < 1e-12:
        return np.broadcast_to(gt.mean(axis=0), gt.shape).copy()
    u, _, vt = np.linalg.svd(x.T @ y)
    diag = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ diag @ vt
    scale = np.trace(rotation.T @ x.T @ y) / variance
    return scale * (x @ rotation) + gt.mean(axis=0)


def joint_errors(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != (24, 3) or gt.shape != (24, 3):
        raise ValueError("Expected matching 24x3 SMPL joints")
    pred = pred - pred[0]
    gt = gt - gt[0]
    raw = np.linalg.norm(pred - gt, axis=1) * 1000.0
    pa = np.linalg.norm(align_similarity(pred, gt) - gt, axis=1) * 1000.0
    return raw, pa


def gt_camera_joints(sequence, person, frame):
    world = np.asarray(sequence["jointPositions"][person][frame], dtype=np.float64).reshape(24, 3)
    camera = np.asarray(sequence["cam_poses"][frame], dtype=np.float64)
    return world @ camera[:3, :3].T + camera[:3, 3]


def person_crop(image, pose2d, scale):
    points = np.asarray(pose2d, dtype=np.float64).T
    visible = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if visible.sum() < 6:
        raise ValueError("Fewer than 6 visible 2D joints")
    xy = points[visible, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    center = (lo + hi) / 2
    height = max((hi[1] - lo[1]) * scale, (hi[0] - lo[0]) * scale / 0.75, 16.0)
    width = 0.75 * height
    x0, y0 = center - [width / 2, height / 2]
    transform = np.array([[192 / width, 0, -x0 * 192 / width],
                          [0, 256 / height, -y0 * 256 / height]], dtype=np.float32)
    crop = cv2.warpAffine(image, transform, (192, 256), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    canvas = cv2.copyMakeBorder(crop, 0, 0, 32, 32, cv2.BORDER_CONSTANT)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), [float(x0), float(y0), float(width), float(height)]


def valid_frames(sequence, person):
    poses = np.asarray(sequence["poses2d"][person])
    valid_cam = np.asarray(sequence["campose_valid"][person]).astype(bool)
    if poses.shape[1:] != (3, 18):
        raise ValueError(f"Unexpected 2D annotation shape: {poses.shape}")
    visible = np.isfinite(poses).all(axis=1) & (poses[:, 2, :] > 0)
    return np.flatnonzero(valid_cam & (visible.sum(axis=1) >= 6))


def select_samples(dataset_root, split, samples_per_sequence, max_sequences, scale):
    ann_dir = dataset_root / "sequenceFiles/sequenceFiles" / split
    paths = sorted(ann_dir.glob("*.pkl"))
    if max_sequences:
        paths = paths[:max_sequences]
    if not paths:
        raise FileNotFoundError(f"No annotations in {ann_dir}")
    samples = []
    for path in paths:
        with path.open("rb") as file:
            sequence = pickle.load(file, encoding="latin1")
        for person in range(len(sequence["genders"])):
            valid = valid_frames(sequence, person)
            if not len(valid):
                continue
            frames = valid[np.linspace(0, len(valid) - 1,
                                        min(samples_per_sequence, len(valid)), dtype=int)]
            for frame in frames:
                image_path = dataset_root / "imageFiles" / path.stem / f"image_{frame:05d}.jpg"
                image = cv2.imread(str(image_path))
                if image is None:
                    raise FileNotFoundError(image_path)
                rgb, box = person_crop(image, sequence["poses2d"][person][frame], scale)
                gt = gt_camera_joints(sequence, person, int(frame))
                if not np.isfinite(gt).all():
                    continue
                samples.append({"sequence": path.stem, "person": person, "frame": int(frame),
                                "image": str(image_path), "box_xywh": box,
                                "rgb": rgb, "gt": gt})
        print(f"Selected {path.stem}: {len(sequence['genders'])} person(s)", flush=True)
    return samples


def load_regressors():
    archive = PEAR_ROOT / "assets/SMPLX2SMPL.zip"
    with zipfile.ZipFile(archive) as file:
        names = [name for name in file.namelist() if name.endswith("body_models/smplx2smpl.pkl")]
        if len(names) != 1:
            raise RuntimeError("Could not uniquely locate SMPLX-to-SMPL mapping")
        conversion = joblib.load(io.BytesIO(file.read(names[0])))["matrix"]
    mapping = sparse.csr_matrix(conversion, dtype=np.float32)
    del conversion
    # Older licensed SMPL pickles still use the pre-NumPy-1.24 aliases.
    import inspect
    for alias, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                         "object": object, "str": str, "unicode": str}.items():
        if alias not in np.__dict__:
            setattr(np, alias, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    import smplx
    regressor = smplx.SMPL(str(SMPL_PATH), gender="neutral").J_regressor.cpu().numpy()
    if mapping.shape != (6890, 10475) or regressor.shape != (24, 6890):
        raise RuntimeError(f"Unexpected conversion shapes: {mapping.shape}, {regressor.shape}")
    return mapping, sparse.csr_matrix(regressor.astype(np.float32))


def predict_all(kind, samples, args, mapping, regressor):
    from models.modules.ehm import EHM_v2
    from utils.general_utils import ConfigDict, add_extra_cfgs
    if kind == "student":
        from models.pipeline.student_pipeline import PearStudentPipeline
        cfg = add_extra_cfgs(ConfigDict(model_config_path=str(PEAR_ROOT / "configs/student_l70_v2.yaml")))
        model = PearStudentPipeline(cfg)
        ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["student"], strict=True)
    else:
        from models.pipeline.ehm_pipeline import Ehm_Pipeline
        cfg = add_extra_cfgs(ConfigDict(model_config_path=str(PEAR_ROOT / "configs/infer.yaml")))
        model = Ehm_Pipeline(cfg)
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(ckpt["backbone"], strict=False)
        model.head.load_state_dict(ckpt["head"], strict=False)
    del ckpt
    model = model.to(args.device).eval()
    ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(args.device).eval()
    rows = []
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            image = torch.from_numpy(sample["rgb"].copy()).permute(2, 0, 1).unsqueeze(0)
            output = model(image.to(args.device, dtype=torch.float32) / 255.0)
            mesh = ehm(output["body_param"], output["flame_param"], pose_type="rotmat")
            vertices = mesh["vertices"][0, :10475].cpu().numpy().astype(np.float32)
            joints = regressor @ (mapping @ vertices)
            raw, pa = joint_errors(joints, sample["gt"])
            rows.append({"model": kind, "sequence": sample["sequence"],
                         "person": sample["person"], "frame": sample["frame"],
                         "mpjpe_mm": float(raw.mean()), "pa_mpjpe_mm": float(pa.mean()),
                         "lower_mpjpe_mm": float(raw[LOWER_JOINTS].mean()),
                         "left_knee_mm": float(raw[4]), "right_knee_mm": float(raw[5]),
                         "left_ankle_mm": float(raw[7]), "right_ankle_mm": float(raw[8])})
            if (i + 1) % 10 == 0 or i + 1 == len(samples):
                print(f"{kind}: {i+1}/{len(samples)}", flush=True)
    del model, ehm
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "assets/3DPW")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--samples-per-sequence", type=int, default=3)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--crop-scale", type=float, default=1.25)
    parser.add_argument("--student-ckpt", type=Path, default=STUDENT_CKPT)
    parser.add_argument("--teacher-ckpt", type=Path, default=TEACHER_CKPT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/debug/3dpw_pilot")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.samples_per_sequence < 1 or args.max_sequences < 0 or args.crop_scale <= 0:
        parser.error("Sample counts must be positive and crop scale must be > 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    samples = select_samples(args.dataset_root, args.split, args.samples_per_sequence,
                             args.max_sequences, args.crop_scale)
    mapping, regressor = load_regressors()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    try:
        os.chdir(PEAR_ROOT)
        rows = []
        for kind in ("student", "teacher"):
            rows.extend(predict_all(kind, samples, args, mapping, regressor))
    finally:
        os.chdir(original_cwd)
        sys.path.remove(str(PEAR_ROOT))
    with (args.output_dir / "per_frame.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    columns = ["mpjpe_mm", "pa_mpjpe_mm", "lower_mpjpe_mm", "left_knee_mm",
               "right_knee_mm", "left_ankle_mm", "right_ankle_mm"]
    summary = {kind: {col: float(np.mean([row[col] for row in rows if row["model"] == kind]))
                      for col in columns} for kind in ("student", "teacher")}
    report = {"scope": "Sampled local 3DPW GT-2D-crop pilot; not PEAR Table 3 replication",
              "split": args.split, "samples_per_sequence": args.samples_per_sequence,
              "max_sequences": args.max_sequences, "crop_scale": args.crop_scale,
              "total_person_frames": len(samples), "joint_definition": "SMPL 24, neutral J regressor over SMPLX-to-SMPL vertices",
              "gt": "3DPW jointPositions transformed by cam_poses; see protocol caveat in docs",
              "student_ckpt": str(args.student_ckpt), "teacher_ckpt": str(args.teacher_ckpt),
              "summary": summary,
              "sample_manifest": [{k: v for k, v in sample.items() if k not in ("rgb", "gt")}
                                  for sample in samples]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
