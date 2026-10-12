"""2D PCK evaluator for the HMR2.0 / HSMR benchmarks (COCO-val, LSP-Extended, PoseTrack-val).

Phase 2 (PEAR Table 3). Additive: reuses tools/eval_3dpw_standard.py (imported, unchanged) for model
loading, cropping, EHM-s vertices, the SMPLX->SMPL map and the crop camera projection.

Protocol (phase2/EVAL_SUITE.md):
  CONFIRMED (HSMR commit 88486dc1 / HMR2.0 code):
    labels      hmr2_evaluation_data/{coco_val,hr-lspet_train,posetrack_2018_val}.npz
    GT kps      44 = 25 OpenPose body ('body_keypoints_2d') + 19 extra ('extra_keypoints_2d'); conf > 0.5 counts
    normaliser  max side of the npz box, scale*200 (HMR2: crop-normalised units x crop/box ratio)
    score       mmpose keypoint_pck_accuracy: per-keypoint accuracy over valid GT, averaged over keypoints
  ASSUMED (to check against PEAR's author's script):
    keypoints   EHM-s -> SMPL topology (fixed map) -> HMR2's SMPL keypoints: OpenPose-25 from the 24 SMPL
                joints (neutral J_regressor) + smplx vertex joints, and the 19 extra joints from SMPL_to_J19.pkl
    crop        square crop from the npz box via PEAR's process_bbox (x1.25), 256 px
    projection  PEAR crop camera, mapped back to image pixels (eval_3dpw_standard.project_to_image)
  --keypoints pear_native (diagnostic alternative to the assumed mapping): the OpenPose body keypoints from
    PEAR's own EHM_v2 joints (SMPLXV2 with its extra joint regressor) through PEAR's smplx_joints_to_dwpose
    (OpenPose-18 order); mid-hip = mean of the hips; feet and the 19 extra joints stay as in smpl_hmr2.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path("/raid/ubx858/datasets/eval_suite")
DATASETS = {
    "coco": ("coco_val.npz", EVAL_ROOT / "coco"),
    "lspet": ("hr-lspet_train.npz", EVAL_ROOT / "hr-lspet"),
    "posetrack": ("posetrack_2018_val.npz", EVAL_ROOT / "posetrack" / "posetrack2018" / "posetrack_data"),
}
DEFAULT_LABELS = EVAL_ROOT / "hmr2_evaluation_data"
DEFAULT_J19 = Path("/raid/ubx858/datasets/eval_assets/hmr2/SMPL_to_J19.pkl")
# OpenPose-25 from the 45 smplx-SMPL joints (24 kinematic + nose, reye, leye, rear, lear, LBigToe, LSmallToe,
# LHeel, RBigToe, RSmallToe, RHeel, ...), as in HMR2.0 / SPIN (constants.JOINT_MAP / smpl_to_openpose).
OPENPOSE_FROM_SMPL45 = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
THRESHOLDS = (0.05, 0.1)


def load_eval_module():
    name = "guava_eval_3dpw_standard"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "eval_3dpw_standard.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """mmpose's keypoint_pck_accuracy (as vendored by HSMR, lib/evaluation/hmr2_utils/pck_accuracy.py)."""
    n, k, _ = pred.shape
    m = mask.copy()
    m[np.where((normalize == 0).sum(1))[0], :] = False
    dist = np.full((n, k), -1, dtype=np.float32)
    normalize = normalize.copy()
    normalize[np.where(normalize <= 0)] = 1e6
    dist[m] = np.linalg.norm(((pred - gt) / normalize[:, None, :])[m], axis=-1)
    dist = dist.T
    acc = []
    for d in dist:
        valid = d != -1
        acc.append((d[valid] < thr).sum() / valid.sum() if valid.sum() > 0 else -1)
    acc = np.array(acc)
    valid_acc = acc[acc >= 0]
    return acc, (valid_acc.mean() if len(valid_acc) else 0.0), len(valid_acc)


def pck(pred_px, gt_px, gt_conf, box_max, thresholds=THRESHOLDS):
    """pred_px, gt_px (N, K, 2) image pixels; gt_conf (N, K); box_max (N,) = max(scale*200).
    Same numbers as HMR2's EvaluatorPCK (crop-normalised x expand factor = pixels / box_max)."""
    norm = np.stack([box_max, box_max], 1).astype(np.float64)
    out = {}
    for t in thresholds:
        acc, avg, cnt = keypoint_pck_accuracy(pred_px.astype(np.float64), gt_px.astype(np.float64), gt_conf > 0.5, t, norm)
        out[t] = {"pck": float(avg), "per_keypoint": [float(a) for a in acc], "keypoints_counted": int(cnt)}
    return out


class SMPLKeypoints:
    """HMR2's 44 keypoints from SMPL-topology vertices (B, 6890, 3)."""

    def __init__(self, smpl_neutral, j19_path: Path, device):
        from smplx.vertex_ids import vertex_ids
        self.J = smpl_neutral.J_regressor.float().to(device)                           # (24, 6890)
        ids = vertex_ids["smplh"]
        names = ["nose", "reye", "leye", "rear", "lear", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
        self.vjoints = torch.tensor([ids[n] for n in names], device=device)
        j19 = pickle.load(open(j19_path, "rb"), encoding="latin1")
        self.J19 = torch.from_numpy(np.asarray(j19, dtype=np.float32)).to(device)        # (19, 6890)

    def __call__(self, verts):
        j24 = torch.einsum("jv,bvc->bjc", self.J, verts)
        j35 = torch.cat([j24, verts[:, self.vjoints]], 1)
        openpose = j35[:, OPENPOSE_FROM_SMPL45]
        extra = torch.einsum("jv,bvc->bjc", self.J19, verts)
        return torch.cat([openpose, extra], 1)                                          # (B, 44, 3)


class NpzCropDataset(Dataset):
    """One crop per npz row: box (center, scale*200) -> PEAR process_bbox (x1.25) -> 256x256 affine crop."""

    def __init__(self, ev, npz_path: Path, img_root: Path, stride: int = 1):
        d = np.load(npz_path)
        self.ev = ev
        self.img_root = img_root
        n = len(d["imgname"])
        idx = np.arange(0, n, stride)
        self.imgname = [str(x.decode() if isinstance(x, bytes) else x) for x in d["imgname"][idx]]
        self.center = d["center"][idx].astype(np.float64)
        self.box_wh = d["scale"][idx].reshape(len(idx), -1).astype(np.float64)
        if self.box_wh.shape[1] == 1:
            self.box_wh = np.tile(self.box_wh, (1, 2))
        body = d["body_keypoints_2d"][idx] if "body_keypoints_2d" in d.files else np.zeros((len(idx), 25, 3))
        extra = d["extra_keypoints_2d"][idx] if "extra_keypoints_2d" in d.files else np.zeros((len(idx), 19, 3))
        self.kps = np.concatenate([body, extra], 1).astype(np.float64)                  # (N, 44, 3)

    def __len__(self):
        return len(self.imgname)

    def __getitem__(self, i):
        image = cv2.imread(str(self.img_root / self.imgname[i]))
        if image is None:
            raise FileNotFoundError(self.img_root / self.imgname[i])
        h, w = image.shape[:2]
        cx, cy = self.center[i]
        bw, bh = self.box_wh[i]
        box = np.array([cx - bw / 2, cy - bh / 2, bw, bh], np.float32)
        box = self.ev.process_bbox(box, w, h, (self.ev.IMAGE_SIZE, self.ev.IMAGE_SIZE), ratio=1.25)
        affine = self.ev.box_to_affine(box)
        crop = cv2.warpAffine(image, affine, (self.ev.IMAGE_SIZE, self.ev.IMAGE_SIZE), flags=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return {"index": i, "image": torch.from_numpy(crop).permute(2, 0, 1).contiguous(), "affine": torch.from_numpy(affine)}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--model", choices=("pear", "student", "student_vitpose", "gt"), required=True,
                   help="gt: GT keypoints as prediction (pipeline sanity check, no model)")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--img-root", type=Path, default=None)
    p.add_argument("--j19", type=Path, default=DEFAULT_J19)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--keypoints", choices=("smpl_hmr2", "pear_native"), default="smpl_hmr2")
    args = p.parse_args()
    t0 = time.time()
    ev = load_eval_module()
    device = torch.device(args.device)
    npz_name, img_root = DATASETS[args.dataset]
    img_root = args.img_root or img_root
    ds = NpzCropDataset(ev, args.labels / npz_name, img_root, args.stride)
    gt_px, gt_conf = ds.kps[:, :, :2], ds.kps[:, :, 2]
    box_max = ds.box_wh.max(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "gt":
        pred_px, load_report, checkpoint = gt_px.copy(), {}, None
    else:
        checkpoint = (args.checkpoint or ev.DEFAULT_TEACHER).resolve()
        import os
        cwd = Path.cwd()
        os.chdir(ev.PEAR_ROOT)
        try:
            model, load_report = ev.load_model(args.model, checkpoint, args.student_config, args.batch_size, device)
            from models.modules.ehm import EHM_v2
            from models.smplx.smplx_head import smplx_joints_to_dwpose
            ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
        finally:
            os.chdir(cwd)
        mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
        kps44 = SMPLKeypoints(ev.build_smpl(ev.DEFAULT_SMPL_DIR, "neutral", device), args.j19, device)
        pred_px = np.zeros_like(gt_px)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        for batch in loader:
            idx = batch["index"].numpy()
            images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
            with torch.inference_mode():
                out = model(images)
                out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                       for k, v in out.items()}
                v_smplx = ev.predicted_smplx_vertices(out, ehm)
                v_smpl = torch.stack([torch.sparse.mm(mapping, v.float()) for v in v_smplx])
                k3 = kps44(v_smpl)
                if args.keypoints == "pear_native":
                    j145 = ehm(out["body_param"], out["flame_param"], pose_type="rotmat")["joints"]
                    op18 = smplx_joints_to_dwpose(j145)[0][:, :18]                         # OpenPose-18 order
                    to25 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                    k3 = k3.clone()
                    k3[:, to25] = op18
                    k3[:, 8] = 0.5 * (op18[:, 8] + op18[:, 11])
                px = ev.project_to_image(k3, out["pd_cam"], batch["affine"].to(device).float())
            pred_px[idx] = px.cpu().numpy()

    res_pck = pck(pred_px, gt_px, gt_conf, box_max)
    summary = {
        "dataset": args.dataset, "labels": str(args.labels / npz_name), "labels_sha256": sha256(args.labels / npz_name),
        "img_root": str(img_root), "persons": len(ds), "stride": args.stride, "model": args.model,
        "checkpoint": str(checkpoint) if checkpoint else None, "load_report": load_report,
        "protocol": "HMR2.0/HSMR PCK (confirmed: labels, normaliser, scoring); assumed: SMPL 44-keypoint mapping, "
                    "PEAR process_bbox x1.25 crop from the npz box, PEAR crop camera",
        "keypoints": args.keypoints,
        "pck": {str(t): v["pck"] for t, v in res_pck.items()},
        "per_keypoint": {str(t): v["per_keypoint"] for t, v in res_pck.items()},
        "keypoints_counted": res_pck[THRESHOLDS[0]]["keypoints_counted"],
        "evaluator_sha256": sha256(Path(__file__)), "eval_3dpw_standard_sha256": sha256(ROOT / "tools" / "eval_3dpw_standard.py"),
        "j19_sha256": sha256(args.j19), "seconds": time.time() - t0,
        "command": " ".join(sys.argv),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(args.output_dir / "per_person.npz", pred_px=pred_px.astype(np.float32), gt_px=gt_px.astype(np.float32),
             gt_conf=gt_conf.astype(np.float32), box_max=box_max)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_keypoint"}, indent=2))


if __name__ == "__main__":
    main()
