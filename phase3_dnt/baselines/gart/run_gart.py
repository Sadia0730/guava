#!/usr/bin/env python
"""GART (Lei et al., CVPR 2024; official code JiahuiLei/GART) on one AvatarTest-v1 clip (PREREG addendum 1, CONTRACT.md).

Train on frames.json "recon" keys only with GART's People-Snapshot profile (people_1m.yaml, the default of solver.py /
example.ipynb; hyper-parameters unchanged; training-frame pose refinement on, as in the profile), then render every "test"
key with that frame's GT SMPL (smpl.npz) and K[key], identity extrinsics, black background, 512x512 8-bit PNG.
No test-time pose optimisation (GART's TTO for PeopleSnapshot is NOT run in this main variant).

Usage (env from setup.sh):
  /dev/shm/ubx858_baselines/gart/env/bin/python run_gart.py --dataset ubody --clip ConductMusic_S21_Trim8
  /dev/shm/ubx858_baselines/gart/env/bin/python run_gart.py --dataset ubody --clip ... --sanity 8   # camera check only
"""
import time

T_SCRIPT_START = time.time()  # train_s counts from here (python start-up, imports, data loading, init, training, save)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import os.path as osp  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

ROOT = "/dev/shm/ubx858_baselines/gart"
CODE = f"{ROOT}/code"
GART_COMMIT = "16c11f8a5bb3ae249a9d04dc9d98c316e10f1126"
E_DEFAULT = "/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1"
HERE = osp.dirname(osp.abspath(__file__))

LOADER_CHANGES = [
    "New dataset class avatartest_loader.AvatarTestV1Dataset (GART's own files unmodified) with GART's dataset interface "
    "(rgb composited on white, mask, K, smpl_beta, smpl_pose (24,3), smpl_trans), fed to GART's "
    "RealDataOptimizablePoseProviderPose exactly as TGFitter.prepare_real_seq does (balance=VIEW_BALANCE_FLAG, data on GPU).",
    "File layout: E/base/<dataset>/<clip>/ (frames.json, images/<key>.png, masks/<key>.png, cameras.npz, smpl.npz).",
    "Frame selection: frames.json 'recon' keys for training, 'test' keys for rendering (instead of START_END slices).",
    "Per-frame intrinsics K[key] (GART's provider already stores one K per frame; its PeopleSnapshot loader repeats one K).",
    "Masks: StyleMatte alpha > 127 (binary) from masks/<key>.png.",
    "No resize (IMAGE_ZOOM_RATIO not applied): the crops are already 512 px; GART's 0.5 brings PeopleSnapshot's 1080 px "
    "frames to 540 px. Training and rendering at 512 as registered.",
    "Body model: neutral SMPL v1.1.0 (code/data/smpl_model/SMPL_NEUTRAL.pkl) with the clip's smpl.npz betas10 and per-frame "
    "pose/transl (GART's PeopleSnapshot runs use gendered SMPL with InstantAvatar-refined poses).",
    "Test rendering: GART's _save_render_image_from_pose lines (model forward, render_cam_pcl, alpha>0 mask) with bg=[0,0,0], "
    "saving only the prediction as 8-bit PNG named <key>.png; no test-time pose optimisation.",
    "Not run (visualisation / benchmarking only, no effect on the model): viz_human_all, eval_fps.",
    "Import shim (not a loader change): lib_gart/templates.py imports the dog model smal.smal_tpg unconditionally, which "
    "reads BITE's dog data at import; a placeholder module is registered in sys.modules (human path never uses SMAL).",
]


def stub_smal():
    """lib_gart/templates.py imports the dog model (smal.smal_tpg) unconditionally, and that module reads BITE's
    smal_data/mean_dog_bone_lengths.txt at import time (data the README asks for only 'if you want to work with dogs').
    The human path never uses SMAL, so a placeholder module is registered instead of downloading the dog data; it raises
    if anything tries to build a SMAL template. No GART file is changed."""
    import types
    mod = types.ModuleType("smal.smal_tpg")

    class SMAL:  # noqa: D401
        def __init__(self, *a, **k):
            raise RuntimeError("SMAL (dog) template is not installed in this human-only GART setup")

    mod.SMAL = SMAL
    sys.modules["smal.smal_tpg"] = mod


def pick_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    used = {int(a): float(b) for a, b in (l.split(",") for l in out.strip().splitlines())}
    for g in [0, 1, 2, 3, 4, 5, 6, 7]:
        if used.get(g, 1e9) < 2048:
            return g
    raise SystemExit(f"no GPU with < 2 GB in use: {used}")


def gpu_name(idx):
    return subprocess.check_output(["nvidia-smi", "-i", str(idx), "--query-gpu=name", "--format=csv,noheader"], text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ubody", "bedlam2", "peoplesnapshot", "neuman"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--E", default=E_DEFAULT)
    ap.add_argument("--profile", default=f"{CODE}/profiles/people/people_1m.yaml")
    ap.add_argument("--work_root", default=f"{ROOT}/work")
    ap.add_argument("--sanity", type=int, default=0, help="camera check on N recon frames, no training")
    ap.add_argument("--keep_work", action="store_true")
    a = ap.parse_args()

    gpu = pick_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    clip_dir = osp.join(a.E, "base", a.dataset, a.clip)
    out_dir = osp.join(a.E, "renders", "gart", a.dataset, a.clip)
    work = osp.join(a.work_root, a.dataset, a.clip + ("_sanity" if a.sanity else ""))
    if osp.isdir(work):
        shutil.rmtree(work)  # our own scratch dir
    os.makedirs(work)

    os.chdir(CODE)  # GART's create_log / relative paths expect the repo root as cwd
    sys.path.insert(0, CODE)
    sys.path.insert(0, HERE)
    import numpy as np
    import torch
    import cv2
    stub_smal()
    from solver import TGFitter
    from lib_data.data_provider import RealDataOptimizablePoseProviderPose
    from lib_render.gauspl_renderer import render_cam_pcl
    from avatartest_loader import AvatarTestV1Dataset

    device = torch.device("cuda:0")
    smpl_path = f"{CODE}/data/smpl_model/SMPL_NEUTRAL.pkl"

    t_load0 = time.time()
    trainset = AvatarTestV1Dataset(clip_dir, "recon", load_images=True)
    t_load = time.time() - t_load0
    solver = TGFitter(log_dir=work, profile_fn=a.profile, mode="human", template_model_path=smpl_path, device=device,
                      FAST_TRAINING=True)  # --fast as in script/fit_people_*.sh

    if a.sanity:
        return sanity(a, solver, trainset, work, out_dir, gpu, render_cam_pcl, np, torch, cv2)

    # ---- training (GART solver.run, unchanged) ----
    provider = RealDataOptimizablePoseProviderPose(trainset, balance=getattr(solver, "VIEW_BALANCE_FLAG", False))
    provider.to(device)
    if getattr(solver, "DATA_STAY_GPU_FLAG", True):
        provider.move_images_to_device(device)
    t_fit0 = time.time()
    _, optimized_seq = solver.run(provider)
    torch.cuda.synchronize()
    t_train_end = time.time()
    train_s = t_train_end - T_SCRIPT_START
    fit_s = t_train_end - t_fit0
    with torch.no_grad():  # (GART's provider.pose_diff property references a missing attribute, so computed here)
        pose_opt = torch.cat([optimized_seq.pose_base_list, optimized_seq.pose_rest_list], 1)
        pose_diff = pose_opt - optimized_seq.pose_list_original
        trans_diff = optimized_seq.global_trans_list - optimized_seq.global_trans_list_original
    refine = {"pose_refine_mean_abs_rad": float(pose_diff.abs().mean()), "pose_refine_max_abs_rad": float(pose_diff.abs().max()),
              "trans_refine_mean_abs_m": float(trans_diff.abs().mean()), "trans_refine_max_abs_m": float(trans_diff.abs().max())}
    del provider, optimized_seq
    torch.cuda.empty_cache()

    # ---- test rendering with the GT SMPL of each test frame ----
    t_r0 = time.time()
    model = solver.load_saved_model()  # as test_utils.test
    model.eval()
    testset = AvatarTestV1Dataset(clip_dir, "test", load_images=False)
    tmp_out = osp.join(work, "renders")
    os.makedirs(tmp_out)
    bg = [0.0, 0.0, 0.0]
    act = model.max_sph_order
    per_frame = []
    with torch.no_grad():
        for i in range(len(testset)):
            d, meta = testset[i]
            torch.cuda.synchronize()
            t0 = time.time()
            pose = torch.as_tensor(d["smpl_pose"]).float().to(device)[None]
            trans = torch.as_tensor(d["smpl_trans"]).float().to(device)[None]
            K = torch.as_tensor(d["K"]).float().to(device)
            mu, fr, sc, op, sph, _ = model(pose, trans, additional_dict={"t": None}, active_sph_order=act)
            pkg = render_cam_pcl(mu[0], fr[0], sc[0], op[0], sph[0], 512, 512, K, False, act, bg)
            mask = (pkg["alpha"].squeeze(0) > 0.0).bool()
            pkg["rgb"][:, ~mask] = bg[0]
            rgb = pkg["rgb"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            torch.cuda.synchronize()
            per_frame.append(time.time() - t0)
            u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
            cv2.imwrite(osp.join(tmp_out, f"{meta['key']}.png"), u8[..., ::-1])
    render_wall_local = time.time() - t_r0

    # ---- copy to /vault (plain copies), run.json, small logs ----
    os.makedirs(out_dir, exist_ok=True)
    for k in testset.keys:
        shutil.copyfile(osp.join(tmp_out, f"{k}.png"), osp.join(out_dir, f"{k}.png"))
    n_out = sum(osp.exists(osp.join(out_dir, f"{k}.png")) for k in testset.keys)
    assert n_out == len(testset.keys), (n_out, len(testset.keys))
    logs = sorted(f for f in os.listdir(work) if f.endswith(".log"))
    log_dst = []
    for f in logs:
        shutil.copyfile(osp.join(work, f), osp.join(out_dir, "gart_train.log"))
        log_dst.append("gart_train.log")
    shutil.copyfile(a.profile, osp.join(out_dir, osp.basename(a.profile)))
    run = {
        "method": "GART",
        "commit": GART_COMMIT,
        "train_s": round(train_s, 2),
        "render_s_per_frame": round(float(np.mean(per_frame)), 5),
        "gpu": gpu_name(gpu),
        "iterations": int(solver.TOTAL_steps),
        "notes": {
            "profile": "profiles/people/people_1m.yaml (People-Snapshot, default of solver.py and example.ipynb), unchanged",
            "gpu_index": gpu,
            "n_recon": len(trainset), "n_test": len(testset),
            "train_breakdown_s": {"startup_imports": round(t_load0 - T_SCRIPT_START, 2), "load_recon_frames": round(t_load, 2),
                                  "provider_init_fit_save": round(fit_s, 2)},
            "render_s_per_frame": "CUDA-synchronised model forward + rasterisation + device->host, mean over test frames; "
                                  "excludes PNG encoding and the copy to /vault",
            "render_wall_incl_load_and_png_s": round(render_wall_local, 2),
            "n_gaussians": int(model.N),
            "training_pose_refinement": refine,
            "test_time_pose_optimisation": False,
            "background": "black (0); pixels with rendered alpha == 0 set to 0 as in GART's test code",
            "body_model": "neutral SMPL v1.1.0, smpl.npz (fitted to GT SMPL-X), one betas10 per clip",
            "logs": log_dst,
        },
        "loader_changes": LOADER_CHANGES,
    }
    with open(osp.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=1)
    print(json.dumps(run, indent=1), flush=True)
    if not a.keep_work:
        shutil.rmtree(work)  # checkpoints / caches stay off /vault and are deleted once the renders are written


def sanity(a, solver, ds, work, out_dir, gpu, render_cam_pcl, np, torch, cv2):
    """Camera-convention check: render the UNTRAINED GART template (Gaussians on the posed SMPL mesh) and the SMPL mesh
    silhouette with our K / identity extrinsics; IoU against masks/<key>.png (alpha > 127)."""
    device = torch.device("cuda:0")
    model = solver._get_model_optimizer(betas=ds.betas)[0].to(device).eval()
    tl = model.template._template_layer.to(device)
    faces = tl.faces_tensor.cpu().numpy() if hasattr(tl, "faces_tensor") else np.asarray(tl.faces)
    from pytorch3d.transforms import axis_angle_to_matrix
    idx = np.unique(np.round(np.linspace(0, len(ds) - 1, a.sanity)).astype(int))
    res, tiles = [], []
    with torch.no_grad():
        for i in idx:
            d, meta = ds[i]
            pose = torch.as_tensor(d["smpl_pose"]).float().to(device)[None]
            trans = torch.as_tensor(d["smpl_trans"]).float().to(device)[None]
            K = torch.as_tensor(d["K"]).float().to(device)
            mu, fr, sc, op, sph, _ = model(pose, trans, additional_dict={"t": None}, active_sph_order=0)
            pkg = render_cam_pcl(mu[0], fr[0], sc[0], op[0], sph[0], 512, 512, K, False, 0, [0.0, 0.0, 0.0])
            ga = (pkg["alpha"][0] > 0.5).cpu().numpy()
            R = axis_angle_to_matrix(pose[0])
            out = tl(betas=torch.as_tensor(d["smpl_beta"]).float().to(device)[None], body_pose=R[None, 1:],
                     global_orient=R[None, 0], transl=trans)
            v = out.vertices[0].cpu().numpy()
            Kn = d["K"]
            uv = v[:, :2] / v[:, 2:] * np.array([Kn[0, 0], Kn[1, 1]]) + np.array([Kn[0, 2], Kn[1, 2]])
            sil = np.zeros((512, 512), np.uint8)
            # continuous coords -> pixel-index coords (centre of pixel i at i + 0.5): shift by -0.5, 4-bit subpixel
            pts = np.round((uv - 0.5) * 16).astype(np.int32)
            for tri in pts[faces]:
                cv2.fillConvexPoly(sil, tri, 1, lineType=cv2.LINE_8, shift=4)
            gm = d["mask"] > 0.5
            iou = lambda p, q: float((p & q).sum() / max((p | q).sum(), 1))  # noqa: E731
            r = {"key": meta["key"], "iou_gart_template": iou(ga, gm), "iou_smpl_mesh": iou(sil > 0, gm),
                 "iou_gart_vs_mesh": iou(ga, sil > 0), "mask_px": int(gm.sum())}
            res.append(r)
            print(r, flush=True)
            img = (d["rgb"] * 255).astype(np.uint8).copy()
            ov = img.copy()
            ov[ga] = (0.5 * ov[ga] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            cnt, _ = cv2.findContours((sil > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnt, -1, (0, 160, 0), 1)
            cnt, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnt, -1, (0, 0, 255), 1)
            tiles.append(ov)
    summ = {"clip": a.clip, "dataset": a.dataset, "gpu_index": gpu, "frames": res,
            "mean_iou_gart_template": float(np.mean([r["iou_gart_template"] for r in res])),
            "mean_iou_smpl_mesh": float(np.mean([r["iou_smpl_mesh"] for r in res])),
            "mean_iou_gart_vs_mesh": float(np.mean([r["iou_gart_vs_mesh"] for r in res])),
            "legend": "magenta = untrained GART template alpha>0.5, green = SMPL mesh silhouette contour, blue = mask contour"}
    print(json.dumps({k: v for k, v in summ.items() if k != "frames"}, indent=1), flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(work, "camera_sanity.json"), "w") as f:
        json.dump(summ, f, indent=1)
    cv2.imwrite(osp.join(work, "camera_sanity.jpg"), np.concatenate(tiles, 1)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    shutil.copyfile(osp.join(work, "camera_sanity.json"), osp.join(out_dir, "camera_sanity.json"))
    shutil.copyfile(osp.join(work, "camera_sanity.jpg"), osp.join(out_dir, "camera_sanity.jpg"))
    return summ


if __name__ == "__main__":
    main()
