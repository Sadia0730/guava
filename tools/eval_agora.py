#!/usr/bin/env python
"""AGORA predictions of a SPRINT-type model in the official AGORA evaluation format
(phase3_dnt/PREREG.md, "Pose table (PEAR Table 3 format): AGORA Body MVE for SPRINT").

Pipeline (fixed in the PREREG): 1280x720 images -> YOLOX-L person boxes (DWPose ONNX, as tools/eval_3dpw_standard.py's
detector path) -> training_square_box crop (1.25 x the longer side, no clipping) -> 256 px -> model -> plain SMPL-X mesh.
Per detected person one pickle `<image>_personId_<k>.pkl` (image = the 3840x2160 name, i.e. without "_1280x720"):
  joints           (24, 2)  first 24 kinematic joints projected with the predicted camera, x 3 -> 3840x2160 pixels (matching)
  allSmplJoints3d  (127, 3) 55 kinematic joints of the model's SMPL-X layer (no J14 replacement) + 21 vertex joints +
                            51 static face landmarks (barycentric, SMPL-X npz lmk_faces_idx / lmk_bary_coords), metres
  verts            (10475, 3) SMPL-X vertices, OpenCV camera axes up to a translation (every AGORA error aligns a root joint)
Camera input (SPRINT models): --focal width (default: focal = image width, real 0) | known (AGORA focal from the image name,
constants of the official projection.py, real 1; validation diagnostic only).
The official evaluation is run on the written folder by tools/run_agora_official.py.
Usage: python tools/eval_agora.py --images DIR --checkpoint best_ema.pt --pred-dir DIR --output-dir DIR [--focal known]"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_3dpw_standard as ev  # noqa: E402

# smplx.vertex_ids.vertex_ids["smplx"], in the order of smplx's VertexJointSelector (joints 55..75)
EXTRA_VERTS = [9120, 9929, 9448, 616, 6, 5770, 5780, 8846, 8463, 8474, 8635,
               5361, 4933, 5058, 5169, 5286, 8079, 7669, 7794, 7905, 8022]
SMPLX_NPZ = ev.PEAR_ROOT / "assets" / "SMPLX" / "SMPLX_NEUTRAL_2020.npz"
_DET = None


def agora_focal_px(name: str, width: int) -> float:
    """official agora_evaluation/projection.py: 50 mm (hdri), 18 mm (cam00..cam03), 28 mm (others) on a 36 mm sensor."""
    mm = 50.0 if "hdri" in name else (18.0 if any(c in name for c in ("cam00", "cam01", "cam02", "cam03")) else 28.0)
    return mm / 36.0 * width


def detect_worker(paths):
    """as ev._detect_worker (CPU provider, same detector file and call); returns {path: [[x1, y1, x2, y2], ...]}."""
    global _DET
    if _DET is None:
        import onnxruntime
        sys.path.insert(0, str(ROOT / "EHM-Tracker"))
        from src.modules.dwpose.onnxdet import inference_detector
        opt = onnxruntime.SessionOptions(); opt.intra_op_num_threads = 4
        sess = onnxruntime.InferenceSession(str(ev.DEFAULT_DETECTOR), opt, providers=["CPUExecutionProvider"])
        _DET = lambda frame: inference_detector(sess, frame)  # noqa: E731
    out = {}
    for p in paths:
        boxes = _DET(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        out[p] = [] if boxes is None or len(boxes) == 0 else np.asarray(boxes, dtype=float)[:, :4].tolist()
    return out


class Persons(Dataset):
    def __init__(self, items, focal):
        self.items, self.focal = items, focal

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box = self.items[i]
        image = cv2.imread(path)
        h, w = image.shape[:2]
        affine = ev.box_to_affine(ev.training_square_box(box))
        crop = cv2.cvtColor(cv2.warpAffine(image, affine, (ev.IMAGE_SIZE, ev.IMAGE_SIZE), flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        f, real = (float(w), 0.0) if self.focal == "width" else (agora_focal_px(Path(path).name, w), 1.0)
        return {"index": i, "image": torch.from_numpy(crop).permute(2, 0, 1).contiguous(), "affine": torch.from_numpy(affine),
                "cam_feat": torch.from_numpy(ev.cam_feat(affine, w, h, f, real))}


def smplx_mesh(out):
    """ev.pure_smplx_mesh without the J14 replacement: vertices (B, 10475, 3) and the 55 kinematic joints."""
    bp = out["body_param"]; fp = out.get("flame_param") or {}
    shape = bp["shape"].float(); b = shape.shape[0]; layer = ev.pure_smplx_layer(shape.device)
    jaw = bp.get("jaw_pose")
    if jaw is None and fp.get("jaw_params") is not None:
        from pytorch3d.transforms import axis_angle_to_matrix
        jaw = axis_angle_to_matrix(fp["jaw_params"].float().reshape(b, 3))[:, None]
    exp = bp.get("exp"); exp = shape.new_zeros(b, 50) if exp is None else exp.float()[:, :50]
    params = {"global_pose": bp["global_pose"].float(), "body_pose": bp["body_pose"].float(),
              "left_hand_pose": bp["left_hand_pose"].float(), "right_hand_pose": bp["right_hand_pose"].float(),
              "shape": shape, "exp": exp, "jaw_pose": None if jaw is None else jaw.float().reshape(b, 1, 3, 3),
              "eye_pose": None if bp.get("eye_pose") is None else bp["eye_pose"].float().reshape(b, 2, 3, 3),
              "body_cam": shape.new_tensor([[1.0, 0.0, 0.0]]).expand(b, 3)}
    o = layer(params, pose_type="rotmat")
    return o["vertices"][:, :10475], o["joints"][:, :55]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", choices=("sprint", "sprint_raw", "pear_arch"), default="sprint_raw")
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--focal", choices=("width", "known"), default="width")
    p.add_argument("--pred-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--detections", type=Path, default=None, help="reuse a detections.json written by an earlier run")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12); p.add_argument("--detect-processes", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N images only")
    a = p.parse_args()
    t0 = time.time(); a.output_dir.mkdir(parents=True, exist_ok=True); a.pred_dir.mkdir(parents=True, exist_ok=True)
    assert not any(a.pred_dir.iterdir()), f"{a.pred_dir} is not empty"
    images = sorted(str(x) for x in a.images.glob("*.png"))
    if a.limit: images = images[:a.limit]
    det_file = a.detections or (a.output_dir / "detections.json")
    if det_file.exists():
        det = json.loads(det_file.read_text())["boxes_xyxy"]
    else:
        import multiprocessing as mp
        chunks = [images[i:i + 16] for i in range(0, len(images), 16)]; det = {}
        with mp.get_context("spawn").Pool(a.detect_processes) as pool:
            for part in pool.imap(detect_worker, chunks): det.update(part)
        det = {Path(k).name: v for k, v in det.items()}
        det_file.write_text(json.dumps({"detector": str(ev.DEFAULT_DETECTOR), "detector_sha256": ev.sha256(ev.DEFAULT_DETECTOR),
                                        "images": len(det), "seconds": round(time.time() - t0, 1), "boxes_xyxy": det}))
    t_det = time.time() - t0
    items = [(x, b) for x in images for b in det.get(Path(x).name, [])]
    print(f"{len(images)} images, {len(items)} detected persons, {sum(1 for x in images if det.get(Path(x).name))} images with a detection", flush=True)
    device = torch.device(a.device); cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT)
    try:
        model, load_report = ev.load_model(a.model, a.checkpoint.resolve(), a.student_config, a.batch_size, device)
    finally:
        os.chdir(cwd)
    m = np.load(SMPLX_NPZ, allow_pickle=True)
    faces = torch.from_numpy(m["f"].astype(np.int64)).to(device); lf = torch.from_numpy(m["lmk_faces_idx"].astype(np.int64)).to(device)
    lb = torch.from_numpy(m["lmk_bary_coords"].astype(np.float32)).to(device); jreg = torch.from_numpy(np.asarray(m["J_regressor"], np.float32)).to(device)
    extra = torch.tensor(EXTRA_VERTS, device=device); count = {}; chk = {"reg_vs_kin_max_m": 0.0, "joints_in_box": [], "verts_vs_pure_max_m": 0.0}
    ds = Persons(items, a.focal)
    for batch in DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False):
        with torch.inference_mode():
            out = model(batch["image"].to(device).float() / 255.0, **ev.camera_kwargs(a.model, True, batch["cam_feat"], device))
            out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                   for k, v in out.items()}
            verts, kin = smplx_mesh(out)
            lmk = torch.einsum("lk,blkc->blc", lb, verts[:, faces[lf]])
            j127 = torch.cat([kin, verts[:, extra], lmk], 1)
            px = ev.project_to_image(kin[:, :24], out["pd_cam"], batch["affine"].to(device).float())
            if not count:                                                                    # checks on the first batch
                chk["verts_vs_pure_max_m"] = float((ev.pure_smplx_mesh(out)["vertices"][:, :10475] - verts).abs().max())
            chk["reg_vs_kin_max_m"] = max(chk["reg_vs_kin_max_m"], float((torch.einsum("jv,bvc->bjc", jreg, verts) - kin).norm(dim=-1).max()))
        verts, j127, px = verts.cpu().numpy().astype(np.float32), j127.cpu().numpy().astype(np.float32), px.cpu().numpy().astype(np.float32)
        for n, i in enumerate(batch["index"].tolist()):
            path, box = items[i]; stem = Path(path).stem.replace("_1280x720", ""); k = count.get(stem, 0); count[stem] = k + 1
            side = max(box[2] - box[0], box[3] - box[1]); c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            chk["joints_in_box"].append(float((np.abs(px[n] - c).max(1) <= 0.625 * side).mean()))
            with open(a.pred_dir / f"{stem}_personId_{k}.pkl", "wb") as fh:
                pickle.dump({"joints": px[n] * 3.0, "allSmplJoints3d": j127[n], "verts": verts[n]}, fh, protocol=2)
    summary = {"images": len(images), "images_with_detection": len(count), "persons": len(items), "model": a.model, "focal": a.focal,
               "checkpoint": str(a.checkpoint.resolve()), "checkpoint_sha256": ev.sha256(a.checkpoint.resolve()), "load_report": load_report,
               "detector": str(ev.DEFAULT_DETECTOR), "detections": str(det_file), "crop": "training_square_box 1.25, 256 px",
               "checks": {"verts_vs_pure_smplx_mesh_max_m": chk["verts_vs_pure_max_m"], "J_regressor_vs_kinematic_max_m": chk["reg_vs_kin_max_m"],
                          "share_of_projected_joints_inside_crop": float(np.mean(chk["joints_in_box"])) if chk["joints_in_box"] else None},
               "git": {"guava": ev.git_state(ev.ROOT), "pear": ev.git_state(ev.PEAR_ROOT)}, "evaluator_sha256": ev.sha256(Path(__file__).resolve()),
               "command": " ".join(sys.argv), "seconds_detection": round(t_det, 1), "seconds": round(time.time() - t0, 1)}
    (a.output_dir / "predict_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("images", "images_with_detection", "persons", "checks", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
