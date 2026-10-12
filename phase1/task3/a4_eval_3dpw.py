"""Phase 1 Task 3, A4: evaluate the render-and-compare refiner on a 3DPW split (read-only).

(i) the frozen student ((L-0.1) best.pt) as is; (ii) refiner without the render; (iii) refiner with the
render; each refiner applied once (and twice, secondary). Data path, crop, model loading, GT meshes,
SMPLX->SMPL mapping and metrics are the standard evaluator's (tools/eval_3dpw_standard.py, imported
unchanged); the student runs in fp32 so (i) equals the standard evaluator's numbers. Differences are
reported with the evaluator's sequence bootstrap on paired per-frame differences.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
KEYS = ["mpjpe_mm", "pa_mpjpe_mm", "pve_mm", "diag_lower_body_mm", "diag_upper_body_mm", "diag_hands_mm"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", type=Path, default=Path("/raid/ubx858/outputs/phase1_task1/screen/L_0.1/checkpoints/best.pt"))
    p.add_argument("--refiner-norender", type=Path, required=True)
    p.add_argument("--refiner-render", type=Path, required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--crop", default="gt_keypoints")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all (smoke tests: a prefix)")
    p.add_argument("--timing-frames", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    spec = importlib.util.spec_from_file_location("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    ev = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ev; spec.loader.exec_module(ev)
    os.chdir(PEAR_ROOT); sys.path.insert(0, str(PEAR_ROOT))
    student, report = ev.load_model("student_vitpose", args.student, "configs/student_l70_v2.yaml", 1, device)
    from models.modules.ehm import EHM_v2
    import student_gt_losses as L
    from models.render_refiner import BodyRenderer, RenderRefiner
    from train_render_refiner import cam_from_t, refined_output, t_from_cam
    ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(device).eval()
    smplx_kin = L.kinematic_smplx(str(PEAR_ROOT / "assets/SMPLX"), device)
    renderer = BodyRenderer(smplx_kin.lbs_weights.detach().cpu(), smplx_kin.faces_tensor.detach().cpu(), device)
    refiners, ref_info = {}, {}
    for name, path in (("norender", args.refiner_norender), ("render", args.refiner_render)):
        st = torch.load(path, map_location="cpu", weights_only=False)
        net = RenderRefiner(use_render=bool(st["config"]["use_render"])).to(device).eval()
        net.load_state_dict(st["refiner"], strict=True)
        assert net.use_render == (name == "render"), path
        refiners[name] = net
        ref_info[name] = {"checkpoint": str(path), "step": int(st["step"]), "val": st.get("val"), "sha256": ev.sha256(path)}

    mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(device)
    h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(device)
    smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, device) for g in ("male", "female", "neutral")}
    reg = smpl["neutral"].J_regressor.float()
    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), args.split)
    if args.max_frames:
        samples = samples[:args.max_frames]
    loader = DataLoader(ev.CropDataset(samples, args.crop, None), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=True)

    def render(out, rot, cam):
        r = torch.cat([rot, L.predicted_rotmats(out)[:, 22:]], 1)
        verts, _ = L.body_mesh(smplx_kin, r, out["body_param"]["shape"].float(), out["body_param"]["exp"].float())
        return renderer(verts, t_from_cam(cam))

    def refine(name, out, feats, iters):
        rot = L.predicted_rotmats(out)[:, :22]
        cam = cam_from_t(out["pd_cam"][:, :3, 3])
        for _ in range(iters):
            rot, cam = refiners[name](feats, render(out, rot, cam) if name == "render" else None, rot, cam)
        return refined_output(out, rot, cam)

    variants = ["student", "norender_x1", "render_x1", "norender_x2", "render_x2"]
    per = {v: {k: [] for k in KEYS} for v in variants}
    seqs = []
    for batch in loader:
        bs = [samples[i] for i in batch["index"].tolist()]
        images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
        with torch.inference_mode():
            out, feats = student(images, return_features=True)
            out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                   for k, v in out.items()}
            gt = None
            pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(device)
            betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(device)
            outs = {"student": out}
            for name in ("norender", "render"):
                outs[f"{name}_x1"] = refine(name, out, feats.float(), 1)
                outs[f"{name}_x2"] = refine(name, out, feats.float(), 2)
            for v, o in outs.items():
                pred = torch.stack([torch.sparse.mm(mapping, x) for x in ev.predicted_smplx_vertices(o, ehm)])
                if gt is None:
                    gt = torch.empty_like(pred)
                    for g in ("male", "female"):
                        sel = [i for i, s in enumerate(bs) if s.gender == g]
                        if sel:
                            vv = smpl[g](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                            gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], vv)
                m = ev.frame_metrics(pred, gt, h36m, reg)
                for k in KEYS:
                    per[v][k].append(m[k].cpu())
        seqs += [s.sequence for s in bs]
    per = {v: {k: torch.cat(x).numpy() for k, x in d.items()} for v, d in per.items()}
    seqs = np.array(seqs)

    # runtime per frame, batch 1 (CUDA-synchronised), after warm-up
    ds = ev.CropDataset(samples, args.crop, None)
    timing = {"student": [], "render": [], "refiner_render": [], "refiner_norender": []}
    with torch.inference_mode():
        for n in range(args.timing_frames + 10):
            img = ds[n % len(ds)]["image"][None].to(device).float() / 255.0
            torch.cuda.synchronize(); a = time.perf_counter()
            out, feats = student(img, return_features=True)
            torch.cuda.synchronize(); b_ = time.perf_counter()
            rot0 = L.predicted_rotmats(out)[:, :22].float(); cam0 = cam_from_t(out["pd_cam"][:, :3, 3].float())
            rnd = render(out, rot0, cam0)
            torch.cuda.synchronize(); c = time.perf_counter()
            refiners["render"](feats.float(), rnd, rot0, cam0)
            torch.cuda.synchronize(); d_ = time.perf_counter()
            refiners["norender"](feats.float(), None, rot0, cam0)
            torch.cuda.synchronize(); e = time.perf_counter()
            if n >= 10:
                timing["student"].append((b_ - a) * 1000); timing["render"].append((c - b_) * 1000)
                timing["refiner_render"].append((d_ - c) * 1000); timing["refiner_norender"].append((e - d_) * 1000)

    res = {"split": args.split, "crop": args.crop, "frames": int(len(seqs)), "student": str(args.student), "student_load": report,
           "refiners": ref_info, "evaluator_sha256": ev.sha256(ROOT / "tools" / "eval_3dpw_standard.py"),
           "gpu": torch.cuda.get_device_name(device), "seconds": None,
           "mean": {v: {k: float(per[v][k].mean()) for k in KEYS} for v in variants},
           "ms_per_frame_batch1_median": {k: float(np.median(x)) for k, x in timing.items()},
           "differences": {}}
    for a_, b in (("render_x1", "student"), ("render_x1", "norender_x1"), ("norender_x1", "student"),
                  ("render_x2", "student"), ("render_x2", "norender_x2")):
        res["differences"][f"{a_} - {b}"] = {k: {"mean": float((per[a_][k] - per[b][k]).mean()),
                                                 "ci95": ev.bootstrap_ci(per[a_][k] - per[b][k], seqs)} for k in KEYS}
    res["seconds"] = time.time() - t0
    (args.out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    np.savez(args.out / "per_frame.npz", sequence=seqs, **{f"{v}_{k}": per[v][k] for v in variants for k in KEYS})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
