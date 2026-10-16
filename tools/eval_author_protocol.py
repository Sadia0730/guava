"""Phase 2: PEAR author's evaluation protocol (evaluation_kp.py), ported to run on our machines.

The author's script (kept UNTRACKED in phase2/reference/evaluation_kp/, no redistribution permission yet) imports
his private code base, so it cannot run as is. This file reproduces its protocol with our model loading
(tools/eval_3dpw_standard.py, imported unchanged) and HSMR/HMR2's data pipeline; the joint index tables are NOT
copied here: they are parsed (ast, no execution) from the untracked reference file at run time.

Protocol (author / HSMR / HMR2.0):
  data      hmr2_evaluation_data/{coco_val, hr-lspet_train, posetrack_2018_val, 3dpw_test}.npz + original images
  crop      HMR2 ImageDataset (eval): bbox_size = expand_to_aspect_ratio(scale*200, [192, 256]).max(), square crop
            of that size at the npz centre -> 256x256 (gen_trans_from_patch_cv, no scale/rotation/flip); fed to the
            model as [0, 1] RGB (PEAR normalises internally)                              [ASSUMED: [0,1] input]
  vertices  EHM-s vertices[:10475] -> SMPL with smplx2smpl.pkl['matrix'] (same file as our evaluators)
  joints    COCO / LSP: 45 smplx-SMPL joints (24 J_regressor + vertex joints) ++ 9 SPIN extra (J_regressor_extra.npy),
                        reindexed by JOINT_MAP over JOINT_NAMES (44)
            PoseTrack / 3DPW: 45 joints[smpl_to_openpose] ++ 19 (SMPL_to_J19.pkl), then [smpl_to_h36m] (44)
  2D        crop camera projection -> crop-normalised [-0.5, 0.5]; PCK as HMR2 EvaluatorPCK (x bbox_expand_factor,
            conf > 0.5, mmpose keypoint_pck_accuracy)
  3D (3DPW) HMR2 Evaluator: pelvis = joint 39 subtracted from prediction and GT (npz extra_keypoints_3d), joints
            [25..37, 43]; MPJPE and PA-MPJPE (compute_similarity_transform) in mm
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "phase2" / "reference" / "evaluation_kp" / "evaluation_kp.py"
EVAL_ROOT = Path("/raid/ubx858/datasets/eval_suite")
DATASETS = {   # name: (npz, image root, task)
    "COCO-VAL": ("coco_val.npz", EVAL_ROOT / "coco", "pck"),
    "LSP-EXTENDED": ("hr-lspet_train.npz", EVAL_ROOT / "hr-lspet", "pck"),
    "POSETRACK-VAL": ("posetrack_2018_val.npz", EVAL_ROOT / "posetrack" / "posetrack2018" / "posetrack_data", "pck"),
    "3DPW-TEST": ("3dpw_test.npz", ROOT / "datasets" / "3dpw", "3d"),
}
SMPLX2SMPL_ZIP = ROOT / "third_party" / "PEAR" / "assets" / "SMPLX2SMPL.zip"
BBOX_SHAPE = (192, 256)
PATCH = 256


def load_module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


def reference_tables(path: Path = REFERENCE) -> dict:
    """Module-level literal tables of the author's script (ast.literal_eval only, nothing executed)."""
    tree = ast.parse(path.read_text())
    want = {"smpl_to_openpose", "smpl_to_h36m", "JOINT_MAP", "JOINT_NAMES"}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in want:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = want - set(out)
    if missing:
        raise KeyError(f"tables not found in {path}: {missing}")
    return out


def expand_to_aspect_ratio(wh, target=BBOX_SHAPE):
    """HMR2 / HSMR utils.expand_to_aspect_ratio."""
    w, h = wh
    w_t, h_t = target
    if h / w < h_t / w_t:
        return np.array([w, max(w * h_t / w_t, h)])
    return np.array([max(h * w_t / h_t, w), h])


def compute_similarity_transform(S1, S2):
    """HMR2 / HSMR hmr2_utils.compute_similarity_transform (batched, torch)."""
    b = S1.shape[0]
    S1, S2 = S1.permute(0, 2, 1), S2.permute(0, 2, 1)
    mu1, mu2 = S1.mean(dim=2, keepdim=True), S2.mean(dim=2, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2))
    K = torch.matmul(X1, X2.permute(0, 2, 1))
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)
    Z = torch.eye(U.shape[1], device=U.device, dtype=U.dtype).unsqueeze(0).repeat(b, 1, 1)   # dtype added (HMR2: float32 only)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    t = mu2 - scale * torch.matmul(R, mu1)
    return (scale * torch.matmul(R, S1) + t).permute(0, 2, 1)


def eval_pose_hmr2(pred, gt):
    """HMR2 Evaluator 3D part for already pelvis-centred, selected joints (B, J, 3), metres -> mm."""
    mpjpe = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1) * 1000
    re = torch.sqrt(((compute_similarity_transform(pred, gt) - gt) ** 2).sum(dim=-1)).mean(dim=-1) * 1000
    return mpjpe, re


class HMR2Crops(Dataset):
    def __init__(self, ev, npz_path: Path, img_root: Path, stride: int = 1):
        d = np.load(npz_path)
        idx = np.arange(0, len(d["imgname"]), stride)
        self.ev, self.img_root = ev, img_root
        self.imgname = [str(x.decode() if isinstance(x, bytes) else x) for x in d["imgname"][idx]]
        self.center = d["center"][idx].astype(np.float64)
        sc = d["scale"][idx].reshape(len(idx), -1).astype(np.float64)
        self.box_wh = np.tile(sc, (1, 2)) if sc.shape[1] == 1 else sc                         # = (scale/200)*200
        self.bbox_size = np.array([expand_to_aspect_ratio(b).max() for b in self.box_wh])
        body = d["body_keypoints_2d"][idx] if "body_keypoints_2d" in d.files else np.zeros((len(idx), 25, 3))
        extra = d["extra_keypoints_2d"][idx] if "extra_keypoints_2d" in d.files else np.zeros((len(idx), 19, 3))
        self.kp2d = np.concatenate([body, extra], 1).astype(np.float64)
        b3 = d["body_keypoints_3d"][idx] if "body_keypoints_3d" in d.files else np.zeros((len(idx), 25, 4))
        e3 = d["extra_keypoints_3d"][idx] if "extra_keypoints_3d" in d.files else np.zeros((len(idx), 19, 4))
        b3 = b3.copy(); b3[:, 1:15, -1] = 0                                                    # as HMR2 ImageDataset
        self.kp3d = np.concatenate([b3, e3], 1).astype(np.float64)

    def __len__(self):
        return len(self.imgname)

    def affine(self, i):
        s = self.bbox_size[i]
        cx, cy = self.center[i]
        return self.ev.box_to_affine([cx - s / 2, cy - s / 2, s, s])   # == gen_trans_from_patch_cv(cx, cy, s, s, 256, 256, 1, 0)

    def __getitem__(self, i):
        image = cv2.imread(str(self.img_root / self.imgname[i]))
        if image is None:
            raise FileNotFoundError(self.img_root / self.imgname[i])
        a = self.affine(i)
        crop = cv2.cvtColor(cv2.warpAffine(image, a, (PATCH, PATCH), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0), cv2.COLOR_BGR2RGB)
        return {"index": i, "image": torch.from_numpy(crop).permute(2, 0, 1).contiguous(), "affine": torch.from_numpy(a)}


class AuthorJoints:
    def __init__(self, ev, tables, device):
        self.t = tables
        self.smpl = ev.build_smpl(ev.DEFAULT_SMPL_DIR, "neutral", device)
        with zipfile.ZipFile(SMPLX2SMPL_ZIP) as z:
            extra9 = np.load(io.BytesIO(z.read("SMPLX2SMPL/data/J_regressor_extra.npy")))
            j19 = pickle.load(io.BytesIO(z.read("SMPLX2SMPL/SMPL_to_J19.pkl")), encoding="latin1")
        self.extra9 = torch.from_numpy(np.asarray(extra9, np.float32)).to(device)                # (9, 6890)
        self.j19 = torch.from_numpy(np.asarray(j19, np.float32)).to(device)                      # (19, 6890)
        self.joint_map = torch.tensor([tables["JOINT_MAP"][n] for n in tables["JOINT_NAMES"]], device=device)

    def j45(self, v):
        j24 = torch.einsum("jv,bvc->bjc", self.smpl.J_regressor.float(), v)
        return self.smpl.vertex_joint_selector(v, j24)                                           # (B, 45, 3)

    def __call__(self, v, dataset):
        if dataset in ("LSP-EXTENDED", "COCO-VAL"):
            j = torch.cat([self.j45(v), torch.einsum("jv,bvc->bjc", self.extra9, v)], 1)        # (B, 54, 3)
            return j[:, self.joint_map]
        j = torch.cat([self.j45(v)[:, self.t["smpl_to_openpose"]], torch.einsum("jv,bvc->bjc", self.j19, v)], 1)
        return j[:, self.t["smpl_to_h36m"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--model", choices=("pear", "student", "student_vitpose", "student_vitpose_mq", "student_vitpose_mq_handcrop"), required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--refiner", type=Path, default=None, help="optional A4 refiner checkpoint (no-render) applied once")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    t0 = time.time()
    ev = load_module("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    pck_mod = load_module("guava_eval_pck_hmr2", ROOT / "tools" / "eval_pck_hmr2.py")
    tables = reference_tables()
    device = torch.device(args.device)
    npz, img_root, task = DATASETS[args.dataset]
    labels = EVAL_ROOT / "hmr2_evaluation_data" / npz
    ds = HMR2Crops(ev, labels, img_root, args.stride)
    checkpoint = (args.checkpoint or ev.DEFAULT_TEACHER).resolve()
    cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT)
    try:
        model, load_report = ev.load_model(args.model, checkpoint, args.student_config, args.batch_size, device)
        from models.modules.ehm import EHM_v2
        ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
        refiner = None
        if args.refiner:
            sys.path.insert(0, str(ev.PEAR_ROOT))
            from models.render_refiner import RenderRefiner
            from train_render_refiner import cam_from_t, refined_output
            import student_gt_losses as L
            st = torch.load(args.refiner, map_location="cpu", weights_only=False)
            assert not st["config"]["use_render"], "only the no-render refiner is supported here"
            refiner = RenderRefiner(use_render=False).to(device).eval()
            refiner.load_state_dict(st["refiner"], strict=True)
    finally:
        os.chdir(cwd)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    joints = AuthorJoints(ev, tables, device)
    pred2d = np.zeros((len(ds), 44, 2)); pred3d = np.zeros((len(ds), 44, 3))
    for batch in DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True):
        idx = batch["index"].numpy()
        with torch.inference_mode():
            if refiner is None:
                out = model(batch["image"].to(device).float() / 255.0)
            else:
                out, feats = model(batch["image"].to(device).float() / 255.0, return_features=True)
            out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                   for k, v in out.items()}
            if refiner is not None:
                rot1, cam1 = refiner(feats.float(), None, L.predicted_rotmats(out)[:, :22], cam_from_t(out["pd_cam"][:, :3, 3]))
                out = refined_output(out, rot1, cam1)
            v = torch.stack([torch.sparse.mm(mapping, x.float()) for x in ev.predicted_smplx_vertices(out, ehm)])
            j = joints(v, args.dataset)
            px = ev.project_to_image(j, out["pd_cam"], batch["affine"].to(device).float())       # image pixels
        pred2d[idx] = px.cpu().numpy(); pred3d[idx] = j.cpu().numpy()
    res = {"dataset": args.dataset, "persons": len(ds), "stride": args.stride, "model": args.model,
           "checkpoint": str(checkpoint), "load_report": load_report, "refiner": str(args.refiner) if args.refiner else None,
           "labels": str(labels), "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
           "reference_script_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
           "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "protocol": "PEAR author's evaluation_kp.py (HSMR/HMR2 data, crop and metrics); [0,1] input assumed"}
    if task == "pck":
        # HMR2: crop-normalised x expand factor == image-pixel distance / max(scale*200)
        r = pck_mod.pck(pred2d, ds.kp2d[:, :, :2], ds.kp2d[:, :, 2], ds.box_wh.max(1))
        res["pck"] = {str(t): v["pck"] for t, v in r.items()}
        res["per_keypoint"] = {str(t): v["per_keypoint"] for t, v in r.items()}
    else:
        kl = list(range(25, 38)) + [43]
        pr = torch.from_numpy(pred3d - pred3d[:, [39]])[:, kl].double()
        gt = torch.from_numpy(ds.kp3d[:, :, :3] - ds.kp3d[:, [39], :3])[:, kl].double()
        mpjpe, re = eval_pose_hmr2(pr, gt)
        res["mpjpe_mm"], res["pa_mpjpe_mm"] = float(mpjpe.mean()), float(re.mean())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / "per_frame_mpjpe_pa.npy", np.stack([mpjpe.numpy(), re.numpy()], 1))
    res["seconds"] = time.time() - t0
    res["command"] = " ".join(sys.argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_keypoint"}, indent=2))


if __name__ == "__main__":
    main()
