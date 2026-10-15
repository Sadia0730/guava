"""Phase 2, G1: the live gap. One GUAVA avatar per clip, built from the clip's first tracked frame, driven by
  (a) oracle: the offline EHM-Tracker parameters of every frame (smoothed, fitted over the whole clip);
  (b) PEAR (released); (c) L-real; (d) L-real + refiner v2 (no-render pose/camera refiner);
and rendered at 512 px for every frame; compared with the real frame (EHM's union-box crop x StyleMatte mask, black
background, exactly GUAVA's self-reenactment GT, main/test.py).

Same avatar, identity and renderer for every driver: drivers (b)-(d) replace only the per-frame pose, hand, expression,
jaw / eye / eyelid and camera parameters of the EHM records (main/pear_tracking.py convert_frame); shape, FLAME shape,
joints offset, head / hand scale stay EHM's per-clip identity, as in GUAVA's cross-reenactment. Driver input: the same
union-box crop (EHM LMDB body_image, 1024 px) resized to 256 px, so the predicted camera is in GUAVA's frame (focal 24).
The StyleUNet noise is re-seeded identically for every frame (GUAVA's default draws new noise each frame, which
would add flicker; its randomize_noise=False path fails at this resolution).

Per frame (frame 0 = the source is excluded): PSNR, SSIM, LPIPS-VGG (main/metrics.py functions, 8-bit quantised as when
GUAVA saves and reloads PNGs), MAE. Render jitter: excess temporal change mean |(R_t - R_t-1) - (G_t - G_t-1)| and the
render / GT motion energies mean |R_t - R_t-1|, mean |G_t - G_t-1|. Driver time: batch 1, CUDA-synchronised, per frame
(this server's shared GPU; the oracle's is EHM-Tracker's wall time per frame, offline).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
SCREEN = Path("/raid/ubx858/outputs/phase1_task1/screen")
DRIVERS = {"pear": ("pear", None, None),
           "L-real": ("student_vitpose", SCREEN / "L_real/checkpoints/best.pt", None),
           "L-real + refiner v2": ("student_vitpose", SCREEN / "L_real/checkpoints/best.pt",
                                   Path("/raid/ubx858/outputs/phase2/refiner_v2_Lreal/checkpoints/latest.pt"))}


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", nargs="+", type=Path, required=True, help="EHM-Tracker output directories (one clip each)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--drivers", nargs="+", default=["oracle", *DRIVERS])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-frames", type=int, default=4, help="comparison images per clip")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    # --- PEAR side: drivers ---
    ev = load_by_path("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    cwd = Path.cwd(); os.chdir(PEAR_ROOT); sys.path.insert(0, str(PEAR_ROOT))
    import student_gt_losses as L
    from models.render_refiner import RenderRefiner
    from train_render_refiner import cam_from_t, refined_output
    models, cache = {}, {}
    for name in args.drivers:
        if name == "oracle":
            continue
        kind, ck, ref = DRIVERS[name]
        key = (kind, str(ck))
        if key not in cache:
            cache[key], _ = ev.load_model(kind, (ck or ev.DEFAULT_TEACHER).resolve(), "configs/student_l70_v2.yaml", 1, device)
        r = None
        if ref:
            st = torch.load(ref, map_location="cpu", weights_only=False)
            r = RenderRefiner(use_render=False).to(device).eval(); r.load_state_dict(st["refiner"], strict=True)
        models[name] = (kind, cache[key], r)
    while str(PEAR_ROOT) in sys.path:
        sys.path.remove(str(PEAR_ROOT))
    os.chdir(cwd)
    for name in list(sys.modules):
        f = getattr(sys.modules[name], "__file__", None) or ""
        if f.startswith(str(PEAR_ROOT)) and name.split(".")[0] in ("models", "utils", "dataset", "tools"):
            del sys.modules[name]

    # --- GUAVA side ---
    sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
    import roma
    from omegaconf import OmegaConf
    from dataset import TrackedData_infer
    from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
    from utils.general_utils import ConfigDict, add_extra_cfgs
    from submodules.lpipsPyTorch import LPIPS
    met = load_by_path("guava_metrics", ROOT / "main" / "metrics.py")
    cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "assets/GUAVA/config.yaml")))
    infer_model = Ubody_Gaussian_inferer(cfg.MODEL).to(device).eval()
    render_model = GaussianRenderer(cfg.MODEL).to(device).eval()
    ck = torch.load(ROOT / "assets/GUAVA/checkpoints/best_160000.pt", map_location="cpu", weights_only=True)
    infer_model.load_state_dict(ck["model"], strict=False); render_model.load_state_dict(ck["render_model"], strict=False)
    del ck
    refine_fwd = render_model.nerual_refiner.forward
    def fixed_noise_refine(x, *a, **k):                                 # same StyleUNet noise for every frame
        torch.manual_seed(20261014)                                        # (its randomize_noise=False buffers do not
        return refine_fwd(x, *a, **k)                                      # match this resolution)
    render_model.nerual_refiner.forward = fixed_noise_refine
    lpips = LPIPS("vgg", "0.1").to(device)

    def drive(name, img256):
        """One driver forward at batch 1 -> (smplx_coeffs, flame_coeffs) updates, seconds."""
        kind, model, refiner = models[name]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            if refiner is None:
                out = model(img256)
            else:
                out, feats = model(img256, return_features=True)
            out = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                   for k, v in out.items()}
            if refiner is not None:
                rot1, cam1 = refiner(feats.float(), None, L.predicted_rotmats(out)[:, :22], cam_from_t(out["pd_cam"][:, :3, 3]))
                out = refined_output(out, rot1, cam1)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        bp, fp = out["body_param"], out["flame_param"]
        aa = lambda r: roma.rotmat_to_rotvec(r.reshape(-1, 3, 3).float()).cpu().numpy()
        smplx = {"exp": bp["exp"][0].cpu().numpy(), "global_pose": aa(bp["global_pose"])[0], "body_pose": aa(bp["body_pose"]),
                 "left_hand_pose": aa(bp["left_hand_pose"]), "right_hand_pose": aa(bp["right_hand_pose"]),
                 "camera_RT_params": out["pd_cam"][0, :3, :4].cpu().numpy()}
        flame = {k: fp[k][0].cpu().numpy() for k in ("expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params")}
        flame["neck_pose_params"] = np.zeros(3, np.float32)
        return smplx, flame, dt

    q8 = lambda x: (x.clamp(0, 1) * 255).round() / 255
    summary = {"drivers": args.drivers, "clips": {}}
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        OmegaConf.set_readonly(cfg.DATASET, False)                      # cfg["DATASET"] would return a detached dict
        cfg.DATASET.data_path = str(clip)
        ds = TrackedData_infer(cfg=cfg, split="test", device=device, test_full=True)
        vid = list(ds.videos_info.keys())[0]
        keys = ds.videos_info[vid]["frames_keys"]
        with torch.no_grad():
            src = ds._load_source_info(vid)
            vg, ug, _ = infer_model(src)
            avatar = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avatar.init_ehm(infer_model.ehm); avatar.eval()
        original = {k: copy.deepcopy(ds.traked_data[k]) for k in keys}
        # driver records for every frame (batch 1, timed)
        records, times = {"oracle": original}, {}
        for name in args.drivers:
            if name == "oracle":
                continue
            recs, ts = {}, []
            for k in keys:
                img = ds._lmdb_engine[f"{k}/body_image"].float()[None].to(device) / 255.0
                img = F.interpolate(img, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
                smplx, flame, dt = drive(name, img)
                r = copy.deepcopy(original[k]); r["smplx_coeffs"].update(smplx); r["flame_coeffs"].update(flame)
                recs[k] = r; ts.append(dt)
            records[name], times[name] = recs, ts
        res = {"frames": len(keys), "drivers": {}}
        prev = {}
        per = {n: {m: [] for m in ("psnr", "ssim", "lpips", "mae", "jitter", "motion_render", "motion_gt", "render_ms")} for n in args.drivers}
        grid_idx = set(np.linspace(1, len(keys) - 1, args.save_frames).round().astype(int).tolist())
        grids = {}
        for i, k in enumerate(keys):
            rows = {}
            for name in args.drivers:
                ds.traked_data[k] = records[name][k]
                with torch.no_grad():
                    tgt = ds._load_target_info(vid, k)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    rr = render_model(avatar(tgt), tgt["render_cam_params"], bg=0.0)
                    torch.cuda.synchronize(); rms = (time.perf_counter() - t0) * 1000
                    R = q8(rr["renders"][0:1]); m = tgt["mask"][0:1]
                    G = q8(tgt["image"][0:1] * m)
                if i > 0:                                                            # frame 0 = source
                    per[name]["psnr"].append(float(met.psnr(R, G).mean())); per[name]["ssim"].append(float(met.ssim(R, G)))
                    per[name]["lpips"].append(float(lpips(R, G))); per[name]["mae"].append(float((R - G).abs().mean()))
                    per[name]["render_ms"].append(rms)
                    if name in prev:
                        pR, pG = prev[name]
                        per[name]["jitter"].append(float(((R - pR) - (G - pG)).abs().mean()))
                        per[name]["motion_render"].append(float((R - pR).abs().mean())); per[name]["motion_gt"].append(float((G - pG).abs().mean()))
                prev[name] = (R, G)
                if i in grid_idx:
                    rows[name] = R[0]
                    rows["gt"] = G[0]
            if rows:
                grids[i] = torch.cat([rows["gt"]] + [rows[n] for n in args.drivers], 2)
        for k in keys:
            ds.traked_data[k] = original[k]
        for name in args.drivers:
            d = {m: float(np.mean(v)) for m, v in per[name].items() if v}
            if name in times:
                d["driver_ms_median"] = float(np.median(times[name]) * 1000); d["driver_fps"] = 1000.0 / d["driver_ms_median"]
            res["drivers"][name] = d
        out_clip = args.out / clip.name
        out_clip.mkdir(parents=True, exist_ok=True)
        np.savez(out_clip / "per_frame.npz", **{f"{n}|{m}": np.array(v) for n in args.drivers for m, v in per[n].items()})
        import torchvision
        if grids:
            torchvision.utils.save_image(torch.cat([grids[i] for i in sorted(grids)], 1), out_clip / "compare_gt_then_drivers.jpg")
        (out_clip / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
        summary["clips"][clip.name] = res
        print(clip.name, json.dumps({n: {m: round(v, 4) for m, v in d.items()} for n, d in res["drivers"].items()}), flush=True)
        ds._lmdb_engine.close()
    agg = {}
    for name in args.drivers:
        vals = [c["drivers"][name] for c in summary["clips"].values()]
        agg[name] = {m: float(np.mean([v[m] for v in vals if m in v])) for m in vals[0]}
    summary["mean_over_clips"] = agg
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
