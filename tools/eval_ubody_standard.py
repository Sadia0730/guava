"""UBody "intra-scene" evaluator (OSX protocol), Phase 2. Additive; reuses tools/eval_3dpw_standard.py (model
loading, crop, EHM-s vertices) and tools/eval_ehf_standard.py (rigid_align, SMPL-X regressors / vertex sets, J_IDX),
both imported unchanged.

Data: phase2/build_ubody_intra_test.py -> /raid/ubx858/datasets/eval_suite/ubody/{ubody_intra_test.json,images/}
(OSX test selection: every 100th annotation of intra-scene test videos).
Metrics (OSX data/UBody/UBody.py evaluate): pa_mpvpe_all/hand/face, mpvpe_all (pelvis-aligned), mpvpe_hand (wrist-
aligned), mpvpe_face (neck-aligned), pa_mpjpe_body (J14) / hand (21-joint regressor); only GT vertices / joints that
project inside the image count; a hand / the face is skipped when none of its vertices is inside.
GT mesh: SMPL-X v1.1 neutral layer as in OSX (smplx.create(..., 'smplx', gender='NEUTRAL', use_pca=False,
use_face_contour=True), 10 betas, 10 expression, flat_hand_mean=False) from the annotation's smplx_param, + trans.
DEVIATION (recorded): OSX projects the GT mesh for the in-image test with the EVALUATED MODEL's predicted camera;
here the annotation camera (focal, princpt) is used, so the mask is the same for every model.
ASSUMED: crop = annotation bbox -> PEAR process_bbox (x1.25) -> 256x256.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/raid/ubx858/datasets/eval_suite/ubody")


def load_module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


class UBodyCrops(Dataset):
    def __init__(self, ev, items, img_root: Path):
        self.ev, self.items, self.img_root = ev, items, img_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        image = cv2.imread(str(self.img_root / it["scene"] / it["file_name"]))
        if image is None:
            raise FileNotFoundError(self.img_root / it["scene"] / it["file_name"])
        h, w = image.shape[:2]
        box = self.ev.process_bbox(np.asarray(it["bbox"], np.float32), w, h, (self.ev.IMAGE_SIZE, self.ev.IMAGE_SIZE), ratio=1.25)
        affine = self.ev.box_to_affine(box)
        crop = cv2.cvtColor(cv2.warpAffine(image, affine, (self.ev.IMAGE_SIZE, self.ev.IMAGE_SIZE), flags=cv2.INTER_LINEAR),
                            cv2.COLOR_BGR2RGB)
        return {"index": i, "image": torch.from_numpy(crop).permute(2, 0, 1).contiguous()}


def gt_meshes(items, smplx_dir: Path, device):
    import smplx
    layer = smplx.create(str(smplx_dir), "smplx", gender="NEUTRAL", use_pca=False, use_face_contour=True,
                         num_betas=10, num_expression_coeffs=10, flat_hand_mean=False).to(device).eval()
    out = []
    for k in range(0, len(items), 256):
        ch = items[k:k + 256]
        P = {key: torch.tensor(np.array([c["smplx_param"][key] for c in ch]), dtype=torch.float32, device=device)
             for key in ("root_pose", "body_pose", "lhand_pose", "rhand_pose", "jaw_pose", "shape", "expr", "trans")}
        z = torch.zeros(len(ch), 3, device=device)
        with torch.no_grad():
            v = layer(global_orient=P["root_pose"], body_pose=P["body_pose"], left_hand_pose=P["lhand_pose"],
                      right_hand_pose=P["rhand_pose"], jaw_pose=P["jaw_pose"], leye_pose=z, reye_pose=z,
                      betas=P["shape"], expression=P["expr"], transl=P["trans"]).vertices
        out.append(v.cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def in_image(points, cam, wh):
    f, c = np.asarray(cam["focal"], np.float64), np.asarray(cam["princpt"], np.float64)
    xy = points[:, :2] / np.maximum(points[:, 2:3], 1e-4) * f + c
    return (points[:, 2] > 0) & (0 < xy[:, 0]) & (xy[:, 0] < wh[0]) & (0 < xy[:, 1]) & (xy[:, 1] < wh[1])


def evaluate_one(P, ehf, mesh_out, mesh_gt, cam, wh):
    """OSX UBody.evaluate for one sample (P: ehf.Protocol)."""
    e = lambda a, b, m: np.sqrt(np.sum((a - b) ** 2, 1))[m].mean() * 1000  # noqa: E731
    J, J_IDX = P.J, ehf.J_IDX
    mesh_valid = in_image(mesh_gt, cam, wh)
    lh, rh, fi = P.hand_idx["left_hand"], P.hand_idx["right_hand"], P.face_idx
    jb_valid = in_image(np.dot(P.j14, mesh_gt), cam, wh)
    jl_valid = in_image(np.dot(P.hand_reg["left"], mesh_gt), cam, wh)
    jr_valid = in_image(np.dot(P.hand_reg["right"], mesh_gt), cam, wh)
    r = {}
    if mesh_valid.any():
        r["pa_mpvpe_all"] = e(ehf.rigid_align(mesh_out, mesh_gt), mesh_gt, mesh_valid)
        al = mesh_out - np.dot(J, mesh_out)[J_IDX["pelvis"], None] + np.dot(J, mesh_gt)[J_IDX["pelvis"], None]
        r["mpvpe_all"] = e(al, mesh_gt, mesh_valid)
    pa_h, h_ = [], []
    for idx, wrist in ((lh, "lwrist"), (rh, "rwrist")):
        mv = mesh_valid[idx]
        if mv.sum() == 0:
            continue
        g, o = mesh_gt[idx], mesh_out[idx]
        pa_h.append(e(ehf.rigid_align(o, g), g, mv))
        al = o - np.dot(J, mesh_out)[J_IDX[wrist], None] + np.dot(J, mesh_gt)[J_IDX[wrist], None]
        h_.append(e(al, g, mv))
    if pa_h:
        r["pa_mpvpe_hand"], r["mpvpe_hand"] = float(np.mean(pa_h)), float(np.mean(h_))
    fv = mesh_valid[fi]
    if fv.sum():
        g, o = mesh_gt[fi], mesh_out[fi]
        r["pa_mpvpe_face"] = e(ehf.rigid_align(o, g), g, fv)
        al = o - np.dot(J, mesh_out)[J_IDX["neck"], None] + np.dot(J, mesh_gt)[J_IDX["neck"], None]
        r["mpvpe_face"] = e(al, g, fv)
    jg, jo = np.dot(P.j14, mesh_gt), np.dot(P.j14, mesh_out)
    if jb_valid.any():
        r["pa_mpjpe_body"] = e(ehf.rigid_align(jo, jg), jg, jb_valid)
    pj = []
    for side, v in (("left", jl_valid), ("right", jr_valid)):
        if v.sum():
            g, o = np.dot(P.hand_reg[side], mesh_gt), np.dot(P.hand_reg[side], mesh_out)
            pj.append(e(ehf.rigid_align(o, g), g, v))
    if pj:
        r["pa_mpjpe_hand"] = float(np.mean(pj))
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("pear", "student", "student_vitpose", "gt"), required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--smplx-dir", type=Path, required=True, help="folder with SMPLX_NEUTRAL.npz (SMPL-X v1.1)")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-items", type=int, default=0)
    args = p.parse_args()
    t0 = time.time()
    ev = load_module("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    ehf = load_module("guava_eval_ehf_standard", ROOT / "tools" / "eval_ehf_standard.py")
    device = torch.device(args.device)
    meta = json.loads((args.data / "ubody_intra_test.json").read_text())
    items = meta["items"][:args.max_items] if args.max_items else meta["items"]
    P = ehf.Protocol()
    mesh_gt = gt_meshes(items, args.smplx_dir, device)
    if args.model == "gt":
        mesh_out, load_report, checkpoint = mesh_gt.copy(), {}, None
    else:
        checkpoint = (args.checkpoint or ev.DEFAULT_TEACHER).resolve()
        cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT)
        try:
            model, load_report = ev.load_model(args.model, checkpoint, args.student_config, args.batch_size, device)
            from models.modules.ehm import EHM_v2
            ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
        finally:
            os.chdir(cwd)
        mesh_out = np.zeros_like(mesh_gt)
        for batch in DataLoader(UBodyCrops(ev, items, args.data / "images"), batch_size=args.batch_size,
                                num_workers=args.num_workers, shuffle=False):
            with torch.inference_mode():
                out = model(batch["image"].to(device).float() / 255.0)
                out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                       for k, v in out.items()}
                mesh_out[batch["index"].numpy()] = ev.predicted_smplx_vertices(out, ehm).cpu().numpy()
    rows = [evaluate_one(P, ehf, mesh_out[i], mesh_gt[i], it["cam_param"], it["img_wh"]) for i, it in enumerate(items)]
    keys = ["pa_mpvpe_all", "pa_mpvpe_hand", "pa_mpvpe_face", "mpvpe_all", "mpvpe_hand", "mpvpe_face", "pa_mpjpe_body", "pa_mpjpe_hand"]
    summary = {"model": args.model, "checkpoint": str(checkpoint) if checkpoint else None, "load_report": load_report,
               "items": len(items), "smplx_dir": str(args.smplx_dir),
               "smplx_neutral_sha256": hashlib.sha256((args.smplx_dir / "SMPLX_NEUTRAL.npz").read_bytes()).hexdigest()
               if (args.smplx_dir / "SMPLX_NEUTRAL.npz").exists() else None,
               "metrics_mm": {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys},
               "counts": {k: int(sum(k in r for r in rows)) for k in keys},
               "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seconds": time.time() - t0,
               "command": " ".join(sys.argv)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
