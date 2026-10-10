#!/usr/bin/env python
"""Standard 3DPW protocol evaluator for EHM-s models (PEAR teacher and students).

Protocol (one definition for every model):
- every `campose_valid` person-frame of the chosen 3DPW split (35,515 on test);
- ground truth: gendered SMPL mesh from `poses`/`betas`, rotated into the camera frame;
- prediction: EHM-s mesh (SMPL-X topology, FLAME head) mapped to SMPL topology with a
  fixed SMPL-X-to-SMPL vertex correspondence;
- 14 LSP joints from the Human3.6M joint regressor on both meshes;
- MPJPE after hip-midpoint centring, PA-MPJPE after per-frame similarity Procrustes on
  the 14 joints, PVE over the 6,890 SMPL vertices after hip-midpoint centring.

Crops: `gt_keypoints` (main, comparable to published numbers) uses PEAR's own box code on
the ground-truth OpenPose keypoints; `detector` (secondary) uses the training YOLOX box,
square, x1.25, exactly as the training data were prepared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
ASSETS = Path("/raid/ubx858/datasets/eval_assets")
DEFAULT_DATASET = ROOT / "datasets" / "3dpw"
DEFAULT_H36M_REGRESSOR = ASSETS / "data" / "J_regressor_h36m.npy"
DEFAULT_SMPL_DIR = ASSETS / "smpl"
DEFAULT_SMPLX2SMPL = PEAR_ROOT / "assets" / "SMPLX2SMPL.zip"
DEFAULT_DETECTOR = ROOT / "EHM-Tracker" / "pretrained" / "dwpose" / "yolox_l.onnx"
DEFAULT_TEACHER = Path.home() / (
    ".cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/"
    "513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt"
)

IMAGE_SIZE = 256
H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
J14_HIPS = (2, 3)
# Non-standard diagnostic: SMPL kinematic joints from the same (neutral) SMPL regressor
# applied to both SMPL-topology meshes.
SMPL_PARTS = {
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "upper_body": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19],
    "hands": [20, 21, 22, 23],
}


# ---------------------------------------------------------------------------
# Crops. The first four functions are copied from third_party/PEAR/inference_images.py
# (that module imports ultralytics at load time); tests check they match.
# ---------------------------------------------------------------------------


def get_bbox(joint_img, joint_valid, extend_ratio=1.2):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin, ymin, xmax, ymax = min(x_img), min(y_img), max(x_img), max(y_img)
    x_center = (xmin + xmax) / 2.0
    width = xmax - xmin
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    y_center = (ymin + ymax) / 2.0
    height = ymax - ymin
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio
    return np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        return np.array([x1, y1, x2 - x1, y2 - y1])
    return None


def process_bbox(bbox, img_width, img_height, input_img_shape, ratio=1.25):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox
    w, h = bbox[2], bbox[3]
    c_x, c_y = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * ratio
    bbox[3] = h * ratio
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0
    return bbox.astype(np.float32)


def box_to_affine(box_xywh) -> np.ndarray:
    """Affine from an (x, y, w, h) box to the 256x256 patch; equals PEAR's
    gen_trans_from_patch_cv with scale 1 and no rotation."""
    x, y, w, h = [float(v) for v in box_xywh]
    cx, cy = x + 0.5 * w, y + 0.5 * h
    src = np.float32([[cx, cy], [cx, cy + 0.5 * h], [cx + 0.5 * w, cy]])
    half = IMAGE_SIZE * 0.5
    dst = np.float32([[half, half], [half, half + half], [half + half, half]])
    return cv2.getAffineTransform(src, dst).astype(np.float32)


def gt_keypoint_box(keypoints_18x3: np.ndarray, width: int, height: int):
    xy = keypoints_18x3[:, :2]
    valid = (keypoints_18x3[:, 2] > 0).astype(np.int64)
    box = get_bbox(xy, valid, extend_ratio=1.2)
    return process_bbox(box, width, height, (IMAGE_SIZE, IMAGE_SIZE), ratio=1.25)


def training_square_box(box_xyxy, crop_scale: float = 1.25):
    """The training-data crop: square, side = crop_scale * max(w, h), no clipping
    (third_party/PEAR/tools/prepare_student_distill_data.py:square_crop)."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    side = max(x2 - x1, y2 - y1) * crop_scale
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return np.float32([cx - side / 2, cy - side / 2, side, side])


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def hip_centre(j14: torch.Tensor) -> torch.Tensor:
    return 0.5 * (j14[:, J14_HIPS[0]] + j14[:, J14_HIPS[1]])


def similarity_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Batched similarity Procrustes of pred onto gt (B, N, 3), proper rotations only."""
    pred = pred.double()
    gt = gt.double()
    mu_p, mu_g = pred.mean(1, keepdim=True), gt.mean(1, keepdim=True)
    x, y = pred - mu_p, gt - mu_g
    var = (x * x).sum((1, 2))
    u, s, vt = torch.linalg.svd(x.transpose(1, 2) @ y)
    diag = torch.ones_like(s)
    diag[:, -1] = torch.sign(torch.linalg.det(u @ vt))
    rot = u @ torch.diag_embed(diag) @ vt
    scale = (diag * s).sum(1) / var.clamp_min(1e-12)
    return scale[:, None, None] * (x @ rot) + mu_g


def frame_metrics(pred_verts, gt_verts, h36m_regressor, smpl_regressor) -> dict:
    """All inputs in metres, SMPL topology (B, 6890, 3), camera frame (OpenCV axes)."""
    pred_j14 = torch.einsum("jv,bvc->bjc", h36m_regressor, pred_verts)[:, H36M_TO_J14]
    gt_j14 = torch.einsum("jv,bvc->bjc", h36m_regressor, gt_verts)[:, H36M_TO_J14]
    pred_pelvis, gt_pelvis = hip_centre(pred_j14)[:, None], hip_centre(gt_j14)[:, None]
    pj, gj = pred_j14 - pred_pelvis, gt_j14 - gt_pelvis
    out = {
        "mpjpe_mm": (pj - gj).norm(dim=-1).mean(1) * 1000.0,
        "pa_mpjpe_mm": (similarity_align(pj, gj) - gj.double()).norm(dim=-1).mean(1).float() * 1000.0,
        "pve_mm": ((pred_verts - pred_pelvis) - (gt_verts - gt_pelvis)).norm(dim=-1).mean(1) * 1000.0,
    }
    # SPIN-style variant: centre on the regressed H36M pelvis (joint 0). Diagnostic only.
    pred_h0 = torch.einsum("jv,bvc->bjc", h36m_regressor[:1], pred_verts)
    gt_h0 = torch.einsum("jv,bvc->bjc", h36m_regressor[:1], gt_verts)
    out["mpjpe_h36m_pelvis_mm"] = ((pred_j14 - pred_h0) - (gt_j14 - gt_h0)).norm(dim=-1).mean(1) * 1000.0
    # Non-standard per-part diagnostic on SMPL kinematic joints (same regressor both sides).
    pk = torch.einsum("jv,bvc->bjc", smpl_regressor, pred_verts)
    gk = torch.einsum("jv,bvc->bjc", smpl_regressor, gt_verts)
    pk, gk = pk - pk[:, :1], gk - gk[:, :1]
    err = (pk - gk).norm(dim=-1) * 1000.0
    for name, idx in SMPL_PARTS.items():
        out[f"diag_{name}_mm"] = err[:, idx].mean(1)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    sequence: str
    person: int
    frame: int
    gender: str
    pose: np.ndarray       # (72,)
    betas: np.ndarray      # (10,)
    cam_rotation: np.ndarray  # (3, 3) world -> camera
    keypoints: np.ndarray  # (18, 3) OpenPose
    image: str


def load_samples(dataset_root: Path, split: str) -> list[Sample]:
    samples = []
    for path in sorted((dataset_root / "sequenceFiles" / split).glob("*.pkl")):
        with path.open("rb") as handle:
            seq = pickle.load(handle, encoding="latin1")
        for person in range(len(seq["genders"])):
            valid = np.asarray(seq["campose_valid"][person]).astype(bool)
            for frame in np.flatnonzero(valid):
                samples.append(Sample(
                    sequence=path.stem, person=person, frame=int(frame),
                    gender={"m": "male", "f": "female"}[seq["genders"][person]],
                    pose=np.asarray(seq["poses"][person][frame], dtype=np.float32),
                    betas=np.asarray(seq["betas"][person][:10], dtype=np.float32),
                    cam_rotation=np.asarray(seq["cam_poses"][frame], dtype=np.float32)[:3, :3],
                    keypoints=np.asarray(seq["poses2d"][person][frame], dtype=np.float32).T,
                    image=str(dataset_root / "imageFiles" / path.stem / f"image_{frame:05d}.jpg"),
                ))
    return samples


class CropDataset(Dataset):
    def __init__(self, samples, crop: str, detections: dict | None):
        self.samples, self.crop, self.detections = samples, crop, detections

    def __len__(self):
        return len(self.samples)

    def box_for(self, index, sample, width, height):
        gt_box = gt_keypoint_box(sample.keypoints, width, height)
        if self.crop == "gt_keypoints":
            return gt_box, 0
        visible = sample.keypoints[sample.keypoints[:, 2] > 0, :2]
        tight = [visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()]
        boxes = self.detections.get(Path(sample.image).relative_to(Path(sample.image).parents[1]).as_posix(), [])
        best = max(boxes, key=lambda b: iou_xyxy(b, tight), default=None)
        if best is None or iou_xyxy(best, tight) < 0.1:
            return gt_box, 1  # detector miss: fall back, and count it
        return training_square_box(best), 0

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(sample.image)
        if image is None:
            raise FileNotFoundError(sample.image)
        height, width = image.shape[:2]
        box, miss = self.box_for(index, sample, width, height)
        affine = box_to_affine(box)
        crop = cv2.warpAffine(image, affine, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return {
            "index": index,
            "image": torch.from_numpy(crop).permute(2, 0, 1).contiguous(),
            "affine": torch.from_numpy(affine),
            "detector_miss": miss,
        }


# ---------------------------------------------------------------------------
# Models and meshes
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def load_smplx_to_smpl(path: Path) -> torch.Tensor:
    with zipfile.ZipFile(path) as archive:
        name = [n for n in archive.namelist() if n.endswith("body_models/smplx2smpl.pkl")]
        matrix = joblib.load(io.BytesIO(archive.read(name[0])))["matrix"]
    m = sparse.coo_matrix(matrix, dtype=np.float32)
    return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data, m.shape).coalesce()


def build_smpl(smpl_dir: Path, gender: str, device):
    from smplx import SMPL
    from smplx.utils import Struct
    arrays = dict(np.load(smpl_dir / f"SMPL_{gender.upper()}.npz", allow_pickle=False))
    return SMPL(model_path="unused", data_struct=Struct(**arrays), gender=gender, num_betas=10).to(device).eval()


def load_model(kind: str, checkpoint: Path, student_config: str, batch_size: int, device):
    sys.path.insert(0, str(PEAR_ROOT))
    from utils.general_utils import ConfigDict, add_extra_cfgs
    from omegaconf import OmegaConf
    if kind == "student":
        from models.pipeline.student_pipeline import PearStudentPipeline
        cfg = add_extra_cfgs(ConfigDict(model_config_path=student_config))
        OmegaConf.set_readonly(cfg, False); cfg.TRAIN.batch_size = batch_size; OmegaConf.set_readonly(cfg, True)
        model = PearStudentPipeline(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["student"], strict=True)
        load_report = {"strict": True, "step": int(state.get("step", -1))}
    else:
        from models.pipeline.ehm_pipeline import Ehm_Pipeline
        cfg = add_extra_cfgs(ConfigDict(model_config_path="configs/infer.yaml"))
        model = Ehm_Pipeline(cfg)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        backbone = model.backbone.load_state_dict(state["backbone"], strict=False)
        head = model.head.load_state_dict(state["head"], strict=False)
        load_report = {
            "strict": False,
            "backbone_missing": list(backbone.missing_keys), "backbone_unexpected": list(backbone.unexpected_keys),
            "head_missing": list(head.missing_keys), "head_unexpected": list(head.unexpected_keys),
        }
    return model.to(device).eval(), load_report


def predicted_smplx_vertices(model_output, ehm) -> torch.Tensor:
    """EHM-s vertices (SMPL-X topology) in OpenCV camera axes, metres.

    PEAR's camera is R = diag(-1, -1, 1) (PyTorch3D axes); converting PyTorch3D to OpenCV
    multiplies x and y by -1 again, so body-model coordinates are already OpenCV camera
    axes up to translation, which every metric here removes."""
    mesh = ehm(model_output["body_param"], model_output["flame_param"], pose_type="rotmat")
    return mesh["vertices"][:, :10475]


def project_to_image(verts_cam, pd_cam, affine) -> torch.Tensor:
    """PEAR camera projection of points to original-image pixels (sanity checks)."""
    rt = pd_cam.float()
    cam = torch.einsum("bij,bkj->bki", rt[:, :3, :3], verts_cam.float()) + rt[:, None, :3, 3]
    ndc = 24.0 * cam[..., :2] / cam[..., 2:3].clamp_min(1e-4)
    patch = (1.0 - ndc) * 0.5 * IMAGE_SIZE
    inv = torch.linalg.inv(torch.cat([affine, affine.new_tensor([0, 0, 1]).expand(affine.shape[0], 1, 3)], 1))
    homog = torch.cat([patch, torch.ones_like(patch[..., :1])], -1)
    return torch.einsum("bij,bkj->bki", inv[:, :2], homog)


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    by_group = {k: values[groups == k] for k in keys}
    means = []
    for _ in range(n):
        picked = rng.choice(keys, size=len(keys), replace=True)
        means.append(np.concatenate([by_group[k] for k in picked]).mean())
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("pear", "student"), required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="student checkpoint; teacher default is the HF snapshot")
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--split", choices=("test", "validation", "train"), default="test")
    p.add_argument("--crop", choices=("gt_keypoints", "detector"), default="gt_keypoints")
    p.add_argument("--detections", type=Path, help="JSON from --detect-only (required for --crop detector)")
    p.add_argument("--detect-only", action="store_true", help="run YOLOX on every image of the split and exit")
    p.add_argument("--detect-processes", type=int, default=32)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--h36m-regressor", type=Path, default=DEFAULT_H36M_REGRESSOR)
    p.add_argument("--smpl-dir", type=Path, default=DEFAULT_SMPL_DIR)
    p.add_argument("--smplx2smpl", type=Path, default=DEFAULT_SMPLX2SMPL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--stride", type=int, default=1, help="evaluate every Nth sample (smoke tests only)")
    p.add_argument("--save-overlays", type=int, default=0, help="save N 2D reprojection overlays")
    return p.parse_args()


_DETECTOR = None


def _detect_worker(paths: list[str]) -> dict:
    # CPU execution provider on purpose: the onnxruntime CUDA provider in this env needs
    # CUDA 11 libraries and silently falls back to CPU anyway.
    global _DETECTOR
    if _DETECTOR is None:
        import onnxruntime
        sys.path.insert(0, str(ROOT / "EHM-Tracker"))
        from src.modules.dwpose.onnxdet import inference_detector
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4
        session = onnxruntime.InferenceSession(str(DEFAULT_DETECTOR), options, providers=["CPUExecutionProvider"])
        _DETECTOR = lambda frame: inference_detector(session, frame)  # noqa: E731
    out = {}
    for path in paths:
        boxes = _DETECTOR(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        key = Path(path).relative_to(Path(path).parents[1]).as_posix()
        out[key] = [] if boxes is None or len(boxes) == 0 else np.asarray(boxes, dtype=float)[:, :4].tolist()
    return out


def run_detection(args, samples):
    import multiprocessing as mp
    images = sorted({s.image for s in samples})[:: args.stride]
    chunks = [images[i:i + 64] for i in range(0, len(images), 64)]
    out, start = {}, time.time()
    with mp.get_context("spawn").Pool(args.detect_processes) as pool:
        for i, part in enumerate(pool.imap(_detect_worker, chunks)):
            out.update(part)
            if (i + 1) % 20 == 0:
                print(f"detect {len(out)}/{len(images)}  {time.time() - start:.0f} s", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"detections_{args.split}.json"
    target.write_text(json.dumps({
        "detector": str(DEFAULT_DETECTOR), "detector_sha256": sha256(DEFAULT_DETECTOR),
        "provider": "CPUExecutionProvider", "images": len(out), "seconds": round(time.time() - start, 1),
        "boxes_xyxy": out}))
    print(f"wrote {target} ({len(out)} images, {time.time() - start:.0f} s)")


def main():
    args = parse_args()
    t_start = time.time()
    samples = load_samples(args.dataset_root.resolve(), args.split)
    print(f"{args.split}: {len(samples)} person-frames")
    if args.detect_only:
        return run_detection(args, samples)
    if args.stride > 1:
        samples = samples[::args.stride]
    detections = None
    if args.crop == "detector":
        detections = json.loads(args.detections.read_text())["boxes_xyxy"]

    checkpoint = (args.checkpoint or DEFAULT_TEACHER).resolve()
    device = torch.device(args.device)
    cwd = Path.cwd()
    os.chdir(PEAR_ROOT)
    try:
        model, load_report = load_model(args.model, checkpoint, args.student_config, args.batch_size, device)
        from models.modules.ehm import EHM_v2
        ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    finally:
        os.chdir(cwd)

    h36m = torch.from_numpy(np.load(args.h36m_regressor)).float().to(device)
    mapping = load_smplx_to_smpl(args.smplx2smpl).to(device)
    smpl = {g: build_smpl(args.smpl_dir, g, device) for g in ("male", "female", "neutral")}
    smpl_regressor = smpl["neutral"].J_regressor.float()

    loader = DataLoader(CropDataset(samples, args.crop, detections), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.precision]
    rows, overlay_dir = [], args.output_dir / "overlays"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reproj = []
    for batch in loader:
        idx = batch["index"].tolist()
        batch_samples = [samples[i] for i in idx]
        images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                output = model(images)
            output = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                          if isinstance(v, dict) else v.float()) for k, v in output.items()}
            pred_x = predicted_smplx_vertices(output, ehm)
            pred = torch.stack([torch.sparse.mm(mapping, v) for v in pred_x])
            gt = torch.empty_like(pred)
            pose = torch.from_numpy(np.stack([s.pose for s in batch_samples])).to(device)
            betas = torch.from_numpy(np.stack([s.betas for s in batch_samples])).to(device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in batch_samples])).to(device)
            for gender in ("male", "female"):
                sel = [i for i, s in enumerate(batch_samples) if s.gender == gender]
                if sel:
                    v = smpl[gender](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                    gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], v)
            metrics = frame_metrics(pred, gt, h36m, smpl_regressor)
            # Every 50th predicted vertex, projected back to the original image for overlays.
            px = project_to_image(pred_x[:, ::50], output["pd_cam"], batch["affine"].to(device))
        for j, s in enumerate(batch_samples):
            rows.append({"sequence": s.sequence, "person": s.person, "frame": s.frame, "gender": s.gender,
                         "detector_miss": int(batch["detector_miss"][j]),
                         **{k: float(v[j]) for k, v in metrics.items()}})
        if args.save_overlays and len(reproj) < args.save_overlays:
            overlay_dir.mkdir(exist_ok=True)
            for j, s in enumerate(batch_samples[: args.save_overlays - len(reproj)]):
                img = cv2.imread(s.image)
                for x, y in px[j].cpu().numpy():
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
                for x, y, c in s.keypoints:
                    if c > 0:
                        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), 2)
                name = f"{s.sequence}_p{s.person}_f{s.frame:05d}.jpg"
                cv2.imwrite(str(overlay_dir / name), img)
                reproj.append(name)
        if (len(rows) // args.batch_size) % 50 == 0:
            print(f"{len(rows)}/{len(samples)}  running MPJPE {np.mean([r['mpjpe_mm'] for r in rows]):.1f}", flush=True)

    keys = [k for k in rows[0] if k.endswith("_mm")]
    seqs = np.array([r["sequence"] for r in rows])
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        summary[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                      "sequence_bootstrap_95ci": bootstrap_ci(v, seqs)}
    with (args.output_dir / "per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "protocol": {
            "split": args.split, "person_frames": len(rows), "stride": args.stride,
            "crop": args.crop, "precision": args.precision,
            "gt": "gendered SMPL (poses, 10 betas) rotated by cam_poses into camera frame",
            "pred": "EHM-s vertices[:10475] -> SMPL topology via fixed SMPLX2SMPL correspondence",
            "joints": "Human3.6M regressor -> 14 LSP joints (H36M_TO_J14)",
            "centring": "hip midpoint of the 14 joints (indices 2, 3)",
            "pa": "per-frame similarity Procrustes on the 14 joints",
            "pve": "6890 SMPL vertices after hip-midpoint centring",
            "diagnostics": "mpjpe_h36m_pelvis_mm is SPIN-style centring; diag_* use SMPL kinematic joints from the "
                           "neutral SMPL regressor on both meshes (NON-STANDARD)",
            "detector_misses": int(sum(r["detector_miss"] for r in rows)),
        },
        "model": args.model, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "load_report": load_report,
        "assets": {
            "h36m_regressor_sha256": sha256(args.h36m_regressor), "smplx2smpl_zip_sha256": sha256(args.smplx2smpl),
            "smpl_npz_sha256": {g: sha256(args.smpl_dir / f"SMPL_{g}.npz") for g in ("MALE", "FEMALE", "NEUTRAL")},
        },
        "git": {"guava": git_state(ROOT), "pear": git_state(PEAR_ROOT)},
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "command": " ".join(sys.argv),
        "seconds": round(time.time() - t_start, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
        "metrics": summary,
        "overlays": reproj,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: round(v["mean"], 2) for k, v in summary.items()}, indent=2))
    print(f"{len(rows)} frames in {report['seconds']} s -> {args.output_dir}")


if __name__ == "__main__":
    main()
