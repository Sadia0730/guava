#!/usr/bin/env python
"""Laptop timing of GART (Lei et al., CVPR 2024; per-subject, trained on the person's video) on neuman/bike.

Reconstruction = GART's training on the clip's 83 recon frames exactly as evaluated (phase3_dnt/baselines/gart/run_gart.py:
profiles/people/people_1m.yaml unchanged, 2000 steps, --fast, StyleMatte masks of the package, training-pose refinement on,
fp32, 512 px); one run, no repetition (n = 1). Animation = the trained avatar driven by each test frame's GT SMPL with K[key]
(identity extrinsics, black background, 512 px, fp32), 10 warm-up + 50 timed frames cycling over the test keys.

LAPTOP NOTE (RTX 3080 Laptop, 16 GB)
Reuse the matching evaluated environment, code and assets; set the paths in
laptop.env.example.sh. Laptop GPU memory and speed remain unmeasured. Server
CUDA extensions may lack sm_86 kernels; use the method's pinned installation
procedure to build compatible extensions in a fresh build directory. The
wrappers do not rebuild dependencies automatically or change precision.
See TIMING_ON_LAPTOP.md for the short protocol, thermal guard and boundaries.

Usage (env as run_clip.sh): bash run_gart.sh SUBSET_ROOT OUT_DIR   -> OUT_DIR/gart.json
  or  /dev/shm/ubx858_baselines/gart/env/bin/python time_gart.py --root SUBSET_ROOT --out OUT_DIR/gart.json
  --max-steps N (tests only) stops GART's loop after N steps; the learning-rate schedules keep the evaluated 2000-step
  horizon, so every executed step is the evaluated step; the steps run are recorded in the JSON.
"""
import time

T_SCRIPT_START = time.perf_counter()

import sys  # noqa: E402

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only GART tree

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
GART_ADAPTER = osp.join(osp.dirname(HERE), "baselines", "gart")  # avatartest_loader.py, run_gart.py (evaluated adapter)
ROOT = os.environ.get("GART_ROOT", "/dev/shm/ubx858_baselines/gart")
CODE = f"{ROOT}/code"
PROFILE = f"{CODE}/profiles/people/people_1m.yaml"
SMPL_PATH = f"{CODE}/data/smpl_model/SMPL_NEUTRAL.pkl"
RES = 512


def find_clip(root, clip):
    """<root>/<E>/base/<dataset>/<clip>, E from manifest.json (fallback: any <root>/*/base/<clip>)."""
    man = osp.join(root, "manifest.json")
    if osp.exists(man):
        for c in json.load(open(man))["clips"]:
            if c["id"] == clip:
                p = osp.join(root, c["E"], "base", clip)
                if osp.isdir(p):
                    return p, c["E"]
    for E in sorted(os.listdir(root)):
        p = osp.join(root, E, "base", clip)
        if osp.isdir(p):
            return p, E
    raise SystemExit(f"clip {clip} not found under {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="timing subset root (contains manifest.json)")
    ap.add_argument("--clip", default="neuman/bike")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="TEST ONLY: stop GART's training loop after N steps (default: the evaluated 2000)")
    ap.add_argument("--work-root", default="/dev/shm", help="a fresh work dir (checkpoints) is made here and deleted at the end")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--png", type=int, default=0, help="also save this many renders as PNG next to --out (gart_renders/)")
    ap.add_argument("--power-note", default="")
    a = ap.parse_args()
    root, out = osp.abspath(a.root), osp.abspath(a.out)
    clip_dir, E = find_clip(root, a.clip)

    if not osp.exists(osp.join(GART_ADAPTER, "avatartest_loader.py")):
        raise SystemExit(f"evaluated GART adapter not found at {GART_ADAPTER} (needs phase3_dnt/baselines/gart next to timing/)")
    os.chdir(CODE)  # as run_gart.py: GART's create_log reads lib_gart/ and *.ipynb relative to the repo root
    sys.path.insert(0, CODE)
    sys.path.insert(0, GART_ADAPTER)
    sys.path.insert(0, HERE)
    import numpy as np
    import torch
    from phase_timer import PROTOCOL, Phases, stats, write
    from run_gart import GART_COMMIT, stub_smal  # the evaluated adapter's import shim (placeholder for the dog model)
    stub_smal()
    import solver as gart_solver
    from solver import TGFitter
    from lib_data.data_provider import RealDataOptimizablePoseProviderPose
    from lib_render.gauspl_renderer import render_cam_pcl
    from avatartest_loader import AvatarTestV1Dataset

    def now():
        torch.cuda.synchronize()
        return time.perf_counter()

    ms = lambda t0, t1: (t1 - t0) * 1000.0  # noqa: E731

    # CUDA context creation is a per-process cost, not GART's: done before the reconstruction clock starts.
    t_imp = time.perf_counter()
    device = torch.device("cuda:0")
    torch.zeros(1, device=device)
    t_cuda = now()
    torch.cuda.reset_peak_memory_stats()

    # Truncatable, timed training loop: solver.run iterates `for step in tqdm(range(self.TOTAL_steps))`; GART's module-level
    # tqdm is replaced by a wrapper that marks the first and the last step (CUDA-synchronised) and, for --max-steps, stops
    # early without touching TOTAL_steps (so the lr schedules are the evaluated ones).
    orig_tqdm, loop = gart_solver.tqdm, {}

    def timed_tqdm(iterable, *args, **kw):
        n = len(iterable)
        steps = n if a.max_steps is None else max(1, min(n, a.max_steps))
        loop["profile_steps"] = n

        def gen():
            loop["t0"] = now()
            for s in itertools.islice(iterable, steps):
                yield s
            loop["t1"] = now()
            loop["steps"] = steps
        return orig_tqdm(gen(), *args, total=steps, **kw)

    gart_solver.tqdm = timed_tqdm

    os.makedirs(a.work_root, exist_ok=True)
    work = tempfile.mkdtemp(prefix="gart_timing_", dir=a.work_root)  # fresh, ours; deleted at the end
    try:
        # ---------------- reconstruction (one run, n = 1) ----------------
        t0 = now()
        trainset = AvatarTestV1Dataset(clip_dir, "recon", load_images=True)  # PNG decode + mask composite (GART's loader)
        t1 = now()
        solver = TGFitter(log_dir=work, profile_fn=PROFILE, mode="human", template_model_path=SMPL_PATH, device=device,
                          FAST_TRAINING=True)  # --fast as in script/fit_people_*.sh and the evaluated run
        t2 = now()
        provider = RealDataOptimizablePoseProviderPose(trainset, balance=getattr(solver, "VIEW_BALANCE_FLAG", False))
        provider.to(device)
        if getattr(solver, "DATA_STAY_GPU_FLAG", True):
            provider.move_images_to_device(device)
        t3 = now()
        _, optimized_seq = solver.run(provider)  # model / optimiser init -> loop (timed by timed_tqdm) -> save
        t4 = now()
        assert "t1" in loop, "GART training loop was not reached through the tqdm wrapper"
        n_gauss_train = None
        try:
            n_gauss_train = int(torch.load(osp.join(work, "model.pth"), map_location="cpu")["_xyz"].shape[0])
        except Exception:  # noqa: BLE001
            pass
        recon_peak = torch.cuda.max_memory_allocated() / 2**30
        recon_peak_res = torch.cuda.max_memory_reserved() / 2**30
        del provider, optimized_seq
        torch.cuda.empty_cache()

        pre_ms = ms(t0, t3) + ms(t3, loop["t0"])  # data loading / conversion + solver setup + model init before step 0
        recon = {"preprocess": stats([pre_ms]), "segmentation": None, "pose": None, "network": None,
                 "training": stats([ms(loop["t0"], loop["t1"])]), "total": stats([ms(t0, t4)])}
        recon_details = {
            "training_steps_run": loop["steps"], "training_steps_evaluated": loop["profile_steps"],
            "training_complete": loop["steps"] == loop["profile_steps"],
            "it_per_s": round(loop["steps"] / (loop["t1"] - loop["t0"]), 2),
            "n_recon_frames": len(trainset), "n_gaussians_end": n_gauss_train,
            "preprocess_breakdown_ms": {"load_recon_frames": round(ms(t0, t1), 1),
                                        "TGFitter_init_profile_log": round(ms(t1, t2), 1),
                                        "pose_provider_to_gpu": round(ms(t2, t3), 1),
                                        "model_optimizer_init_in_run": round(ms(t3, loop["t0"]), 1)},
            "post_loop_save_ms": round(ms(loop["t1"], t4), 1),
            "python_startup_imports_ms": round(ms(T_SCRIPT_START, t_imp), 1),
            "cuda_context_init_ms": round(ms(t_imp, t_cuda), 1),
            "train_s_equivalent": round(t4 - T_SCRIPT_START, 2),
        }

        # ---------------- animation (per driving frame, GT SMPL given) ----------------
        torch.cuda.reset_peak_memory_stats()
        ta = now()
        model = solver.load_saved_model()  # as the evaluated run / GART's test_utils.test
        model.eval()
        tb = now()
        testset = AvatarTestV1Dataset(clip_dir, "test", load_images=False)
        frames = [testset[i] for i in range(len(testset))]  # (numpy dict, meta): GT pose / trans / K per test key
        act = model.max_sph_order
        bg = [0.0, 0.0, 0.0]
        P = Phases()
        pngs = {}
        with torch.no_grad():
            for i in range(a.warmup + a.iters):
                d, meta = frames[i % len(frames)]
                P.start_run(record=i >= a.warmup)
                with P("preprocess"):  # GT SMPL params + K (numpy) -> model input tensors on the GPU
                    pose = torch.as_tensor(d["smpl_pose"]).float().to(device)[None]
                    trans = torch.as_tensor(d["smpl_trans"]).float().to(device)[None]
                    K = torch.as_tensor(d["K"]).float().to(device)
                with P("deformation"):  # GaussianTemplateModel.forward: skeleton + voxel skinning query, LBS, pose-MLP bones
                    mu, fr, sc, op, sph, _ = model(pose, trans, additional_dict={"t": None}, active_sph_order=act)
                with P("rendering"):  # render_cam_pcl: covariance, SH->RGB, CUDA rasteriser; alpha==0 -> background
                    pkg = render_cam_pcl(mu[0], fr[0], sc[0], op[0], sph[0], RES, RES, K, False, act, bg)
                    mask = (pkg["alpha"].squeeze(0) > 0.0).bool()
                    pkg["rgb"][:, ~mask] = bg[0]
                rgb = pkg["rgb"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()  # device->host as evaluated (in "total" only)
                P.end_run()
                if i >= a.warmup and len(pngs) < a.png and meta["key"] not in pngs:
                    pngs[meta["key"]] = rgb.copy()
        anim_peak = torch.cuda.max_memory_allocated() / 2**30
        anim_peak_res = torch.cuda.max_memory_reserved() / 2**30
        s = P.summary()
        anim = {"pose": None, "segmentation": None, "preprocess": s["preprocess"], "deformation": s["deformation"],
                "rendering": s["rendering"], "rendering_raster": None, "rendering_refiner": None, "update": None,
                "total": s["total"]}
        anim_details = {"avatar_load_ms": round(ms(ta, tb), 1), "n_gaussians": int(model.N), "active_sph_order": int(act),
                        "test_keys": testset.keys, "warmup": a.warmup, "iters": a.iters}
        if pngs:
            import cv2
            pd = osp.join(osp.dirname(out), "gart_renders")
            os.makedirs(pd, exist_ok=True)
            for k, rgb in pngs.items():
                cv2.imwrite(osp.join(pd, f"{k}.png"), (rgb * 255.0 + 0.5).astype(np.uint8)[..., ::-1])
    finally:
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)

    steps_txt = (f"{recon_details['training_steps_run']} of {recon_details['training_steps_evaluated']} steps"
                 + ("" if recon_details["training_complete"] else " (TEST RUN: training stopped early by --max-steps; "
                    "reconstruction times are NOT the evaluated training)"))
    clip_notes = (f"Package {E}/base/{a.clip}: {len(trainset)} recon frames (all), {len(testset.keys)} test keys. "
                  f"Training {steps_txt}. Peak GPU memory (torch max_memory_allocated): reconstruction {recon_peak:.2f} GB "
                  f"(reserved {recon_peak_res:.2f}), animation {anim_peak:.2f} GB (reserved {anim_peak_res:.2f}).")
    clips = {a.clip: {"reconstruction": recon, "animation": anim, "notes": clip_notes,
                      "peak_mem_gb": {"reconstruction": round(recon_peak, 3), "reconstruction_reserved": round(recon_peak_res, 3),
                                      "animation": round(anim_peak, 3), "animation_reserved": round(anim_peak_res, 3)},
                      "details": {"reconstruction": recon_details, "animation": anim_details}}}
    notes = (
        f"GART official code {GART_COMMIT[:12]} with the evaluated adapter (phase3_dnt/baselines/gart: avatartest_loader.py, "
        "run_gart.py's lines); profiles/people/people_1m.yaml unchanged (2000 steps, 1 frame/step, SH order 1, voxel "
        "skinning 64^3, 4 pose-MLP bones, training-pose refinement from step 500), FAST_TRAINING (no viz), fp32, 512 px. "
        "RECONSTRUCTION (one run, n=1, CUDA context created beforehand; python start-up / imports excluded, see details): "
        "preprocess = AvatarTestV1Dataset recon load (83 PNG + StyleMatte mask decode, composite on white) + TGFitter init "
        "(profile, log dir) + RealDataOptimizablePoseProviderPose build and move to GPU + the part of TGFitter.run before "
        "step 0 (_get_model_optimizer: SMPL template, voxel skinning field, on-mesh Gaussian init, Adam; _get_pose_optimizer); "
        "segmentation = null (the package's StyleMatte masks are used, alpha > 127); pose = null (GT SMPL given; GART's "
        "training-frame pose/translation refinement runs inside the loop and is counted in training); network = null; "
        "training = wall time of TGFitter.run's optimisation loop (first to last step, CUDA-synchronised, via a wrapper on "
        "solver.tqdm; includes densify/prune); total = end to end from data loading to the end of TGFitter.run (includes "
        "saving model.pth / training_poses.pth). ANIMATION (per test frame, GT SMPL + K[key], identity extrinsics, black "
        "background, torch.no_grad, 10 warm-up + 50 timed frames cycling over the test keys): preprocess = numpy pose / "
        "trans / K -> CUDA tensors; deformation = GaussianTemplateModel.forward (SMPL skeleton forward kinematics, voxel "
        "skinning-weight query per Gaussian, LBS, pose-MLP additional bones, posed means / frames); rendering = "
        "lib_render.gauspl_renderer.render_cam_pcl (covariances, view-dependent SH->RGB in PyTorch, diff_gaussian_rasterization "
        "forward) + alpha==0 background fill; total = preprocess + deformation + rendering + device->host copy of the 512x512 "
        "image (the evaluated render_s_per_frame window); one-off avatar loading (load_saved_model) is in details, not per frame. "
        "pose/segmentation/update/rendering_raster/rendering_refiner = null (GT pose given, no matting, no refiner, no update).")
    protocol = dict(PROTOCOL, warmup=a.warmup, iters=a.iters, recon_warmup=0, recon_reps=1,
                    training_steps=recon_details["training_steps_run"], max_steps=a.max_steps or None, precision="fp32", render_px=RES)
    write(out, method="gart", row="GART", clips=clips, notes=notes, protocol=protocol, power_note=a.power_note)
    print(json.dumps({"reconstruction": recon, "animation": anim, "peak_mem_gb": clips[a.clip]["peak_mem_gb"]}, indent=1))


if __name__ == "__main__":
    main()
