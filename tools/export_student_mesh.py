"""Export camera-aligned EHM mesh previews from a student and saved RGB video.

These are qualitative predictions, not a ground-truth accuracy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party/PEAR"
sys.path.insert(0, str(ROOT))
from main.live_pear_guava import pad_and_resize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/student_l70_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", nargs="+", type=float, default=[14, 28, 42, 63, 70])
    parser.add_argument("--input-framing", choices=["whole", "centered"], default="centered")
    parser.add_argument("--input-crop-scale", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.image_size < 64 or args.input_crop_scale < 1 or any(t < 0 for t in args.seconds):
        parser.error("image-size must be >=64, crop scale >=1, and seconds nonnegative")
    for name in ["video", "checkpoint", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    if not args.config.is_absolute():
        args.config = PEAR_ROOT / args.config
    return args


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def labeled(image, title):
    band = np.full((34, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return np.concatenate([band, image], axis=0)


def main():
    args = parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or any(round(t * fps) >= count for t in args.seconds):
        capture.release()
        raise ValueError("Requested times must be inside a video with valid FPS metadata")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        capture.release()
        raise RuntimeError("CUDA is unavailable in this execution environment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_directory = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    try:
        os.chdir(PEAR_ROOT)
        from models.pipeline.student_pipeline import PearStudentPipeline
        from models.modules.ehm import EHM_v2
        from utils.general_utils import ConfigDict, add_extra_cfgs
        from utils.graphics_utils import GS_BaseMeshRenderer

        config = add_extra_cfgs(ConfigDict(model_config_path=str(args.config)))
        model = PearStudentPipeline(config)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["student"], strict=True)
        step = int(checkpoint.get("step", 0))
        del checkpoint
        model = model.to(args.device).eval()
        ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(args.device).eval()
        renderer = GS_BaseMeshRenderer(image_size=args.image_size, focal_length=24,
                                       inverse_light=True).to(args.device)
        rows, records = [], []
        faces = ehm.smplx.faces_tensor
        with torch.inference_mode():
            for second in args.seconds:
                frame_id = round(second * fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read frame {frame_id}")
                crop = pad_and_resize(frame, crop_scale=args.input_crop_scale,
                                      framing=args.input_framing)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).to(args.device).permute(2, 0, 1)
                output = model(tensor.float().div(255).unsqueeze(0))
                mesh = ehm(output["body_param"], output["flame_param"], pose_type="rotmat")
                vertices = mesh["vertices"]
                if not torch.isfinite(vertices).all() or not torch.isfinite(output["pd_cam"]).all():
                    raise RuntimeError(f"Nonfinite geometry/camera at frame {frame_id}")
                rgba = renderer.render_mesh(vertices, transform_matrix=output["pd_cam"],
                                            faces=faces)[0].permute(1, 2, 0).cpu().numpy()
                mesh_bgr = rgba[..., :3][..., ::-1].clip(0, 255)
                mask = rgba[..., 3:] / 255.0
                if not np.any(mask > 0):
                    raise RuntimeError(f"Mesh is outside the predicted camera at frame {frame_id}")
                image = cv2.resize(crop, (args.image_size, args.image_size))
                alpha = 0.55 * mask
                overlay = (image * (1 - alpha) + mesh_bgr * alpha).clip(0, 255).astype(np.uint8)
                prefix = f"frame_{frame_id:06d}"
                write_image(args.output_dir / f"{prefix}_input.png", image)
                write_image(args.output_dir / f"{prefix}_mesh.png", mesh_bgr.astype(np.uint8))
                write_image(args.output_dir / f"{prefix}_overlay.png", overlay)
                np.savez_compressed(args.output_dir / f"{prefix}_geometry.npz",
                                    vertices=vertices[0].cpu().numpy(), faces=faces.cpu().numpy(),
                                    camera=output["pd_cam"][0].cpu().numpy())
                row = np.concatenate([
                    labeled(image, f"Input crop | frame {frame_id} | {frame_id / fps:.2f}s"),
                    labeled(mesh_bgr.astype(np.uint8), f"Student EHM mesh | step {step}"),
                    labeled(overlay, "Student mesh overlay | predicted camera"),
                ], axis=1)
                rows.append(row)
                records.append({"frame_id": frame_id, "playback_seconds": frame_id / fps,
                                "mesh_coverage": float(mask.mean())})
                print(f"Exported {prefix}", flush=True)
        write_image(args.output_dir / "student_mesh_comparison.jpg", np.concatenate(rows, axis=0))
        with args.checkpoint.open("rb") as file:
            digest = hashlib.sha256()
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata = {
            "video": str(args.video), "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": digest.hexdigest(), "step": step,
            "config": str(args.config), "device": args.device, "precision": "fp32",
            "input_framing": args.input_framing, "input_crop_scale": args.input_crop_scale,
            "frames": records,
            "scope": "Qualitative EHM predictions over the shared 256x256 input canvas; no GT metrics.",
            "backbone_support": "Only columns 32:224 of the input canvas reach the network.",
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Results: {args.output_dir}", flush=True)
    finally:
        capture.release()
        os.chdir(original_directory)
        sys.path.remove(str(PEAR_ROOT))


if __name__ == "__main__":
    main()
