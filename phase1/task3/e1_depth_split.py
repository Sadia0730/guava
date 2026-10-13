"""Phase 1 Task 3, E1 (analysis only): split each joint's 3D error into depth and image-plane components.

3DPW split (default validation), GT-keypoint crop, standard evaluator (tools/eval_3dpw_standard.py, imported
unchanged): H36M regressor -> 14 LSP joints, hip-midpoint centring, i.e. exactly the vectors MPJPE averages.
For joint j with centred error e_j and the camera ray r_j = unit vector from the camera centre to the TRUE joint
(GT SMPL with 3DPW's translation and camera extrinsics), the depth component is d_j = e_j . r_j and the image-plane
component is p_j = e_j - d_j r_j (|e|^2 = d^2 + |p|^2). Reported: mean |d|, mean |p|, mean |e| (= MPJPE), and the
depth share of the squared error sum(d^2) / sum(|e|^2), overall and per part.
Models: PEAR (released), (L-0.1) best, (L-0.1) + A4 no-render refiner (ii).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PARTS = {"legs (ankles, knees, hips)": [0, 1, 2, 3, 4, 5], "arms (wrists, elbows, shoulders)": [6, 7, 8, 9, 10, 11],
         "thorax + head": [12, 13], "all 14": list(range(14))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="validation")
    p.add_argument("--student", type=Path, default=Path("/raid/ubx858/outputs/phase1_task1/screen/L_0.1/checkpoints/best.pt"))
    p.add_argument("--refiner", type=Path, default=Path("/raid/ubx858/outputs/phase1_task3/a4/norender/checkpoints/latest.pt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    ev = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ev; spec.loader.exec_module(ev)
    device = torch.device(args.device)
    os.chdir(ev.PEAR_ROOT); sys.path.insert(0, str(ev.PEAR_ROOT))
    pear, _ = ev.load_model("pear", ev.DEFAULT_TEACHER.resolve(), "configs/student_l70_v2.yaml", args.batch_size, device)
    student, _ = ev.load_model("student_vitpose", args.student, "configs/student_l70_v2.yaml", args.batch_size, device)
    from models.modules.ehm import EHM_v2
    from models.render_refiner import RenderRefiner
    from train_render_refiner import cam_from_t, refined_output
    import student_gt_losses as L
    ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    st = torch.load(args.refiner, map_location="cpu", weights_only=False)
    refiner = RenderRefiner(use_render=False).to(device).eval(); refiner.load_state_dict(st["refiner"], strict=True)
    os.chdir(ROOT)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female")}
    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), args.split)
    seqs = {}
    for path in sorted((ev.DEFAULT_DATASET.resolve() / "sequenceFiles" / args.split).glob("*.pkl")):
        seqs[path.stem] = pickle.load(open(path, "rb"), encoding="latin1")

    def j14(verts):
        j = torch.einsum("jv,bvc->bjc", h36m, verts)[:, ev.H36M_TO_J14]
        return j, ev.hip_centre(j)[:, None]

    acc = {m: {"d": [], "p": []} for m in ("PEAR", "L-0.1", "L-0.1 + refiner (ii)")}
    loader = DataLoader(ev.CropDataset(samples, "gt_keypoints", None), batch_size=args.batch_size, num_workers=12, shuffle=False)
    for batch in loader:
        bs = [samples[i] for i in batch["index"].tolist()]
        images = batch["image"].to(device).float() / 255.0
        with torch.inference_mode():
            pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(device)
            betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(device)
            gt = torch.empty(len(bs), 6890, 3, device=device)
            for g in ("male", "female"):
                sel = [i for i, s in enumerate(bs) if s.gender == g]
                if sel:
                    v = smpl[g](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                    gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], v)
            # absolute camera-frame offset of the GT body: R_cam trans_world + t_cam (SMPL translation is additive)
            off = []
            for s in bs:
                sq = seqs[s.sequence]
                E = np.asarray(sq["cam_poses"][s.frame], np.float64)
                off.append(E[:3, :3] @ np.asarray(sq["trans"][s.person][s.frame], np.float64) + E[:3, 3])
            off = torch.from_numpy(np.stack(off)).float().to(device)
            gj, gc = j14(gt)
            ray = torch.nn.functional.normalize(gj + off[:, None], dim=-1)                 # (B, 14, 3)
            outs = {}
            o = pear(images); outs["PEAR"] = o
            o, feats = student(images, return_features=True)
            o = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float()) for k, v in o.items()}
            outs["L-0.1"] = o
            r1, c1 = refiner(feats.float(), None, L.predicted_rotmats(o)[:, :22], cam_from_t(o["pd_cam"][:, :3, 3]))
            outs["L-0.1 + refiner (ii)"] = refined_output(o, r1, c1)
            for name, o in outs.items():
                pred = torch.stack([torch.sparse.mm(mapping, x.float()) for x in ev.predicted_smplx_vertices(o, ehm)])
                pj, pc = j14(pred)
                e = (pj - pc) - (gj - gc)                                                    # (B, 14, 3), metres
                d = (e * ray).sum(-1)
                pl = (e - d[..., None] * ray).norm(dim=-1)
                acc[name]["d"].append(d.cpu().numpy()); acc[name]["p"].append(pl.cpu().numpy())
    res = {"split": args.split, "frames": len(samples), "student": str(args.student), "refiner": str(args.refiner),
           "evaluator_sha256": ev.sha256(ROOT / "tools" / "eval_3dpw_standard.py"), "models": {}}
    for name, a in acc.items():
        d = np.concatenate(a["d"]) * 1000; pl = np.concatenate(a["p"]) * 1000
        e = np.sqrt(d ** 2 + pl ** 2)
        res["models"][name] = {part: {"mean_abs_depth_mm": float(np.abs(d[:, ix]).mean()), "mean_plane_mm": float(pl[:, ix].mean()),
                                      "mean_error_mm": float(e[:, ix].mean()),
                                      "depth_share_of_squared_error": float((d[:, ix] ** 2).sum() / (e[:, ix] ** 2).sum()),
                                      "frac_joints_depth_dominant": float((np.abs(d[:, ix]) > pl[:, ix]).mean())}
                               for part, ix in PARTS.items()}
        args.out.mkdir(parents=True, exist_ok=True)
        tag = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        np.savez(args.out / f"per_joint_{tag}.npz", depth_mm=d, plane_mm=pl)
    res["seconds"] = time.time() - t0
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["models"], indent=1))


if __name__ == "__main__":
    main()
