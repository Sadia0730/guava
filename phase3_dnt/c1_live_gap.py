"""DNT Track B, C1: the live-driving gap on GUAVA (B2-B4).

For every clip of the C1 split (EHM-Tracker output directory per clip): one GUAVA avatar from the first tracked frame,
then every later frame (>= --future-from) rendered under
  drivers   O = EHM-Tracker (offline, oracle)   S = L-real student   S+R = L-real + refiner v2   P = PEAR (ViT-H)
  swaps     S with ONE part replaced by O:  body (global + body pose), hands (hand poses), face (FLAME params + head
            scale), cam (camera)  ->  "S+O:body" ...;  leave-one-out: O with ONE part from S -> "O+S:body" ...
Identity (shape, FLAME shape, joints offset, hand scale) is EHM's per-clip identity for every variant (GUAVA cross-act
convention); head scale is part of the face. Driver records are cached per clip (drivers.npz) and reused.

Metrics per frame, computed on the GT upper-body mask (EHM's StyleMatte alpha; both images multiplied by the mask, PSNR
over mask pixels): PSNR, SSIM (GUAVA's), LPIPS-VGG; face-crop and hand-crop LPIPS (crops = EHM's head / hand boxes of
the frame, the same boxes for every variant); keypoint distance (ViTPose+ small, COCO expert, on the render vs the real
frame, keypoints with real-frame score >= 0.5, in units of the real torso length); temporal LPIPS (render_t vs
render_t-1) and flicker (mean |render_t - render_t-1| - mean |real_t - real_t-1|). Every metric RAW and after a
per-frame 2D similarity alignment of the render to the real frame (Umeyama on the detected keypoints).
Aggregation: per-identity means, then 95% bootstrap CIs over identities (1,000 resamples) for every variant and for the
difference of every variant to O. Outputs: <out>/<clip>/{drivers.npz, per_frame.npz, compare.jpg}, <out>/summary.json,
<out>/table.md, <out>/attribution.png. The StyleUNet noise is re-seeded identically for every render.
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

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
SCREEN = Path("/raid/ubx858/outputs/phase1_task1/screen")
DRIVERS = {"P": ("pear", None, None),
           "S": ("student_vitpose", SCREEN / "L_real/checkpoints/best.pt", None),
           "S+R": ("student_vitpose", SCREEN / "L_real/checkpoints/best.pt", Path("/raid/ubx858/outputs/phase2/refiner_v2_Lreal/checkpoints/latest.pt"))}
PARTS = {"body": (["global_pose", "body_pose"], []), "hands": (["left_hand_pose", "right_hand_pose"], []),
         "face": (["exp"], ["expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params"]),
         "cam": (["camera_RT_params"], [])}
# Head scale belongs to the face part, but GUAVA's loader overwrites every record's smplx head_scale with the clip
# identity (data_loader._load_one_info). A driver's per-frame head scale is therefore kept beside the record
# ("_head_scale") and written into the loaded target after loading; records without it (O) keep EHM's identity value.
METRICS = ("psnr", "ssim", "lpips", "lpips_face", "lpips_hands", "kp_dist", "t_lpips", "flicker")


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def swap(rec_from, rec_to, part):
    """rec_from's `part` written into a copy of rec_to."""
    r = copy.deepcopy(rec_to)
    sk, fk = PARTS[part]
    for k in sk:
        r["smplx_coeffs"][k] = copy.deepcopy(rec_from["smplx_coeffs"][k])
    for k in fk:
        r["flame_coeffs"][k] = copy.deepcopy(rec_from["flame_coeffs"][k])
    if part == "face":
        if "_head_scale" in rec_from:
            r["_head_scale"] = copy.deepcopy(rec_from["_head_scale"])
        else:
            r.pop("_head_scale", None)
    return r


def umeyama_similarity(src, dst):
    """2D similarity (s, R, t) mapping src -> dst (N, 2), least squares."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2); S[1, 1] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / (xs ** 2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return np.concatenate([s * R, t[:, None]], 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, default=Path("/raid/ubx858/outputs/dnt/c1/split.json"))
    p.add_argument("--tracked", type=Path, default=Path("/raid/ubx858/outputs/dnt/c1/tracked"))
    p.add_argument("--out", type=Path, default=Path("/raid/ubx858/outputs/dnt/c1/results"))
    p.add_argument("--future-from", type=int, default=60)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--variants", nargs="*", default=None, help="subset of variant names (smoke tests)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    split = json.loads(args.split.read_text())
    clips = [c for c in split["clips"] if (args.tracked / c["clip"] / "optim_tracking_ehm.pkl").exists()]
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"{len(clips)} tracked clips of {len(split['clips'])}", flush=True)

    # --- PEAR side ---
    ev = load_by_path("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
    cwd = Path.cwd(); os.chdir(PEAR_ROOT); sys.path.insert(0, str(PEAR_ROOT))
    import student_gt_losses as L
    from models.render_refiner import RenderRefiner
    from train_render_refiner import cam_from_t, refined_output
    models, cache = {}, {}
    for name, (kind, ck, ref) in DRIVERS.items():
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
    from transformers import VitPoseForPoseEstimation
    met = load_by_path("guava_metrics", ROOT / "main" / "metrics.py")
    cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "assets/GUAVA/config.yaml")))
    infer_model = Ubody_Gaussian_inferer(cfg.MODEL).to(device).eval()
    render_model = GaussianRenderer(cfg.MODEL).to(device).eval()
    ck = torch.load(ROOT / "assets/GUAVA/checkpoints/best_160000.pt", map_location="cpu", weights_only=True)
    infer_model.load_state_dict(ck["model"], strict=False); render_model.load_state_dict(ck["render_model"], strict=False)
    del ck
    refine_fwd = render_model.nerual_refiner.forward

    def fixed_noise_refine(x, *a, **k):
        torch.manual_seed(20261015)
        return refine_fwd(x, *a, **k)
    render_model.nerual_refiner.forward = fixed_noise_refine
    lpips = LPIPS("vgg", "0.1").to(device)
    det = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-small").to(device).eval()
    det_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    det_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def keypoints(img512):
        """(1, 3, 512, 512) in [0, 1] -> (17, 2) px in the 512 frame, (17,) scores."""
        x = F.interpolate(img512, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - det_mean) / det_std)[:, :, :, 32:-32]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hm = det(pixel_values=x, dataset_index=torch.zeros(1, dtype=torch.long, device=device)).heatmaps.float()[0]
        K, H, W = hm.shape
        flat = hm.reshape(K, -1); sc, idx = flat.max(-1)
        y, xk = (idx // W).float(), (idx % W).float()
        kp = torch.stack([((xk + 0.5) * 192 / W + 32) * 2, (y + 0.5) * 256 / H * 2], -1)
        return kp.cpu().numpy(), sc.cpu().numpy()

    def drive(name, img256):
        kind, model, refiner = models[name]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out, feats = model(img256, return_features=True) if refiner is not None else (model(img256), None)
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
        return smplx, flame, bp["head_scale"][0].cpu().numpy(), dt

    q8 = lambda x: (x.clamp(0, 1) * 255).round() / 255

    def masked_metrics(R, G, M, boxes, kpG, scG):
        """R, G (1,3,512,512), M (1,1,512,512). Returns dict of the per-frame metrics (not temporal) and the render keypoints."""
        Rm, Gm = R * M, G * M
        mse = (((Rm - Gm) ** 2).sum() / (M.sum() * 3).clamp_min(1)).item()
        d = {"psnr": float(10 * np.log10(1.0 / max(mse, 1e-10))), "ssim": float(met.ssim(Rm, Gm)), "lpips": float(lpips(Rm, Gm))}
        crops = {"lpips_face": [boxes["head"]], "lpips_hands": [boxes["lhand"], boxes["rhand"]]}
        for key, bl in crops.items():
            vals = []
            for (l, r_, t, b) in bl:
                if r_ - l >= 8 and b - t >= 8:
                    cr = F.interpolate(Rm[:, :, t:b, l:r_], size=(128, 128), mode="bilinear", align_corners=False)
                    cg = F.interpolate(Gm[:, :, t:b, l:r_], size=(128, 128), mode="bilinear", align_corners=False)
                    vals.append(float(lpips(cr, cg)))
            d[key] = float(np.mean(vals)) if vals else float("nan")
        kpR, scR = keypoints(Rm)
        ok = scG >= 0.5
        torso = np.linalg.norm((kpG[5] + kpG[6]) / 2 - (kpG[11] + kpG[12]) / 2)
        d["kp_dist"] = float(np.linalg.norm(kpR[ok] - kpG[ok], axis=1).mean() / max(torso, 1.0)) if ok.sum() >= 4 and torso > 8 else float("nan")
        return d, kpR, scR

    summary = {"split": str(args.split), "clips": {}, "variants": None}
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        cdir = args.tracked / clip["clip"]; odir = args.out / clip["clip"]; odir.mkdir(parents=True, exist_ok=True)
        OmegaConf.set_readonly(cfg.DATASET, False); cfg.DATASET.data_path = str(cdir)
        ds = TrackedData_infer(cfg=cfg, split="test", device=device, test_full=True)
        vid = list(ds.videos_info.keys())[0]; keys = ds.videos_info[vid]["frames_keys"]
        with torch.no_grad():
            src = ds._load_source_info(vid)
            vg, ug, _ = infer_model(src)
            avatar = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avatar.init_ehm(infer_model.ehm); avatar.eval()
        O = {k: copy.deepcopy(ds.traked_data[k]) for k in keys}
        # drivers (cached)
        dcache = odir / "drivers.npz"
        if dcache.exists():
            recs = np.load(dcache, allow_pickle=True)["records"].item(); times = np.load(dcache, allow_pickle=True)["times"].item()
        else:
            recs, times = {}, {}
            for name in DRIVERS:
                recs[name], times[name] = {}, []
                for k in keys:
                    img = ds._lmdb_engine[f"{k}/body_image"].float()[None].to(device) / 255.0
                    img = F.interpolate(img, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
                    smplx, flame, hs, dt = drive(name, img)
                    r = copy.deepcopy(O[k]); r["smplx_coeffs"].update(smplx); r["flame_coeffs"].update(flame); r["_head_scale"] = hs
                    recs[name][k] = r; times[name].append(dt)
            np.savez(dcache, records=np.array(recs, dtype=object), times=np.array(times, dtype=object))
        variants = {"O": O, **recs}
        for part in PARTS:
            variants[f"S+O:{part}"] = {k: swap(O[k], recs["S"][k], part) for k in keys}
            variants[f"O+S:{part}"] = {k: swap(recs["S"][k], O[k], part) for k in keys}
        if args.variants:
            variants = {n: v for n, v in variants.items() if n in args.variants}
        summary["variants"] = list(variants)
        per = {n: {f"{m}{s}": [] for m in METRICS for s in ("", "_aligned")} for n in variants}
        prev = {}; prevG = None; grid = []
        eval_keys = [k for i, k in enumerate(keys) if i >= args.future_from]
        for i, k in enumerate(eval_keys):
            ds.traked_data[k] = O[k]
            with torch.no_grad():
                tgt = ds._load_target_info(vid, k)
            M = tgt["mask"][0:1]; G = q8(tgt["image"][0:1] * M)
            boxes = {n: [int(x) for x in tgt[f"{b}_box"][0].tolist()] for n, b in (("head", "head"), ("lhand", "left_hand"), ("rhand", "right_hand"))}
            kpG, scG = keypoints(G)
            row = {"gt": G[0]}
            for name, V in variants.items():
                ds.traked_data[k] = V[k]
                with torch.no_grad():
                    tgt_v = ds._load_target_info(vid, k)
                    if "_head_scale" in V[k]:
                        tgt_v["smplx_coeffs"]["head_scale"] = torch.as_tensor(V[k]["_head_scale"], dtype=torch.float32, device=device).reshape(1, 3)
                    R = q8(render_model(avatar(tgt_v), tgt_v["render_cam_params"], bg=0.0)["renders"][0:1])
                d, kpR, scR = masked_metrics(R, G, M, boxes, kpG, scG)
                # similarity alignment of the render to the real frame (detected keypoints, both confident)
                ok = (scG >= 0.5) & (scR >= 0.5)
                if ok.sum() >= 4:
                    A = umeyama_similarity(kpR[ok], kpG[ok])
                    Ra = torch.from_numpy(cv2.warpAffine(R[0].permute(1, 2, 0).cpu().numpy(), A, (512, 512), flags=cv2.INTER_LINEAR)).permute(2, 0, 1)[None].to(device)
                    da, _, _ = masked_metrics(Ra, G, M, boxes, kpG, scG)
                else:
                    Ra, da = R, dict(d)
                for m in d:
                    per[name][m].append(d[m]); per[name][m + "_aligned"].append(da[m])
                if name in prev:
                    pR, pRa = prev[name]
                    tl, tla = float(lpips(R * M, pR * M)), float(lpips(Ra * M, pRa * M))
                    fg = float((G - prevG).abs().mean())
                    per[name]["t_lpips"].append(tl); per[name]["t_lpips_aligned"].append(tla)
                    per[name]["flicker"].append(float((R - pR).abs().mean()) - fg); per[name]["flicker_aligned"].append(float((Ra - pRa).abs().mean()) - fg)
                prev[name] = (R, Ra)
                row[name] = R[0]
            prevG = G
            if i in {0, len(eval_keys) // 2, len(eval_keys) - 1}:
                grid.append(torch.cat([row["gt"]] + [row[n] for n in variants], 2))
        for k in keys:
            ds.traked_data[k] = O[k]
        ds._lmdb_engine.close()
        import torchvision
        torchvision.utils.save_image(torch.cat(grid, 1), odir / "compare.jpg")
        np.savez(odir / "per_frame.npz", **{f"{n}|{m}": np.array(v) for n in variants for m, v in per[n].items()})
        res = {"identity": clip["identity"], "frames_evaluated": len(eval_keys),
               "means": {n: {m: float(np.nanmean(v)) if len(v) else float("nan") for m, v in per[n].items()} for n in variants},
               "driver_ms_median": {n: float(np.median(t) * 1000) for n, t in times.items()}}
        (odir / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
        summary["clips"][clip["clip"]] = res
        print(clip["clip"], {n: round(res["means"][n]["psnr"], 2) for n in variants}, flush=True)

    # --- aggregation: bootstrap over identities ---
    names = summary["variants"]; ids = list(summary["clips"])
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
    agg = {}
    for n in names:
        agg[n] = {}
        for m in summary["clips"][ids[0]]["means"][n]:
            x = np.array([summary["clips"][c]["means"][n][m] for c in ids]); xo = np.array([summary["clips"][c]["means"]["O"][m] for c in ids])
            bs = [np.nanmean(x[b]) for b in boot_idx]; bd = [np.nanmean((x - xo)[b]) for b in boot_idx]
            agg[n][m] = {"mean": float(np.nanmean(x)), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                         "diff_vs_O": float(np.nanmean(x - xo)), "diff_ci95": [float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))]}
    summary["aggregate"] = agg
    summary["driver_ms_median"] = {n: float(np.median([summary["clips"][c]["driver_ms_median"][n] for c in ids])) for n in DRIVERS}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    # table.md
    lines = [f"# C1 live-driving gap — {len(ids)} identities, frames >= {args.future_from} after the source; mean [95% CI over identities]\n"]
    for suffix, title in (("", "Raw"), ("_aligned", "After per-frame 2D similarity alignment")):
        lines.append(f"\n## {title}\n\n| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | face LPIPS ↓ | hands LPIPS ↓ | kp dist (torso) ↓ | temporal LPIPS ↓ | flicker | ΔPSNR vs O | ΔLPIPS vs O |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for n in names:
            a = agg[n]
            cell = lambda m: f"{a[m + suffix]['mean']:.3f} [{a[m + suffix]['ci95'][0]:.3f}, {a[m + suffix]['ci95'][1]:.3f}]"
            dcell = lambda m: f"{a[m + suffix]['diff_vs_O']:+.3f} [{a[m + suffix]['diff_ci95'][0]:+.3f}, {a[m + suffix]['diff_ci95'][1]:+.3f}]"
            lines.append(f"| {n} | {cell('psnr')} | {cell('ssim')} | {cell('lpips')} | {cell('lpips_face')} | {cell('lpips_hands')} | {cell('kp_dist')} | {cell('t_lpips')} | {cell('flicker')} | {dcell('psnr')} | {dcell('lpips')} |")
    lines.append("\nDriver time per frame (batch 1, this server's shared GPU, median ms): " + ", ".join(f"{n} {v:.1f}" for n, v in summary["driver_ms_median"].items()))
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    # attribution chart: PSNR / LPIPS recovered by each part swap (S+O:part - S) and lost by each leave-one-out (O+S:part - O)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, m, lab in ((axes[0], "psnr", "PSNR (dB)"), (axes[1], "lpips", "LPIPS")):
            parts = list(PARTS); x = np.arange(len(parts))
            gain = [agg[f"S+O:{p_}"][m]["mean"] - agg["S"][m]["mean"] for p_ in parts]
            loss = [agg[f"O+S:{p_}"][m]["mean"] - agg["O"][m]["mean"] for p_ in parts]
            ax.bar(x - 0.2, gain, 0.4, label="S with part from O (gain over S)"); ax.bar(x + 0.2, loss, 0.4, label="O with part from S (loss vs O)")
            ax.axhline(0, color="k", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(parts); ax.set_ylabel(f"Δ {lab}"); ax.legend(fontsize=8)
            ax.set_title(f"S−O gap: {agg['S'][m]['diff_vs_O']:+.3f} {lab}")
        fig.tight_layout(); fig.savefig(args.out / "attribution.png", dpi=130)
    except Exception as e:  # noqa: BLE001
        print("chart skipped:", e)
    print(json.dumps({n: {m: round(agg[n][m]["mean"], 4) for m in ("psnr", "lpips", "kp_dist")} for n in names}, indent=1))


if __name__ == "__main__":
    main()
