"""Avatar task A, step 2: tracker-free (SPRINT-fit) vs EHM-tracker-made GUAVA (a20) avatars on the C1 test identities.

Avatars (one per identity, from the FIRST tracked frame = C1's source frame; GUAVA (a20) checkpoint --guava-ckpt):
  A   EHM-Tracker fit of the source frame converted to a20 (tracker smplx_shape; exp50 + jaw = per-row LBS refit of the
      tracker's FLAME expression / jaw / eyelids, phase3_dnt/avatar_a_convert.py; eye rotations copied); source image = EHM's
      union-box body crop x EHM's StyleMatte alpha (GUAVA's own convention).
  B   SPRINT fit of the same source frame's ORIGINAL image (phase3_dnt/sprint_avatar_fit.py: own YOLOX box, own crop, own
      StyleMatte mask, SPRINT + 150-step refinement on DWPose keypoints + mask), written in GUAVA's tracked format and loaded
      through GUAVA's TrackedData_infer.
  B0  ablation: B without the refinement (raw SPRINT on the same crop / mask), driven by O only.
Drivers on every C1 future frame (tracked frames with index >= --future-from, EHM frames GUAVA cannot read dropped, as C1):
  O   EHM-Tracker frame fit converted to a20 as for A (tracker camera, tracker shape).
  S   SPRINT (--ckpt, sprint_raw, camera condition f = image width, real 0) on the frame's EHM body crop (as C1 drives the
      student), its own per-frame shape / exp / jaw, its crop camera.
  SL  S with the shape lock: cond["shape"] = the avatar's OWN reference-fit shape (A: tracker smplx_shape; B: SPRINT-fit
      betas), shape_on = 1.
  Variants: A|O A|S A|SL B|O B|S B|SL B0|O.
Metrics per frame (C1's code: GT upper-body mask = EHM StyleMatte alpha, both images x mask): PSNR, SSIM, LPIPS-VGG, face /
hand crop LPIPS, ViTPose keypoint distance; raw and after a per-frame 2D similarity alignment of the render (Umeyama on
ViTPose+ keypoints). Identity: ArcFace (insightface antelopev2 glintr100, IResNet-100 / Glint360K, executed on the GPU by
phase3_dnt/onnx_arcface_torch.py) cosine similarity between the render (on black) and the reference image (source frame
crop x mask); faces aligned with the REAL frame's SCRFD landmarks (phase3_dnt/avatar_a_face_lmk.py, identical crop for every
variant; frames without a detected face skipped); row "GT" = the real future frames vs the reference (ceiling).
Creation time: (a) EHM-Tracker wall time of the clip (c1/tracking.log; the tracker needs the whole clip) + GUAVA avatar
build; (b) SPRINT fit (DWPose CPU + StyleMatte + SPRINT + refinement) + avatar build, measured here.
Aggregation: per-identity means, 95% bootstrap CIs over identities (1,000 resamples, seed 0), paired differences.

Pre-registered verdict criterion (written before the run): (b) is CLEARLY WORSE than (a) if, under driver O (primary: same
driving, only the source fit differs) OR under the deployment driver SL, B - A satisfies at least one of
  LPIPS +0.010 or more with the 95% CI excluding 0;  PSNR -0.5 dB or more with the CI excluding 0;
  ArcFace similarity -0.05 or more with the CI excluding 0
(the C1 G1 thresholds, plus an identity threshold). Otherwise "not clearly worse" (differences reported either way).
Outputs: <out>/<clip>/{per_frame.npz, summary.json, compare.jpg}, <out>/summary.json, <out>/table.md; SPRINT fits in
<fits>/<clip>/{refined, raw}/ (GUAVA tracked format)."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phase3_dnt"))
import sprint_avatar_fit as SAF                     # noqa: E402

VARIANTS = ["A|O", "A|S", "A|SL", "B|O", "B|S", "B|SL", "B0|O"]
METRICS = ("psnr", "ssim", "lpips", "lpips_face", "lpips_hands", "kp_dist", "id_sim")
PAIRS = [("B|O", "A|O"), ("B|S", "A|S"), ("B|SL", "A|SL"), ("B0|O", "A|O"), ("B0|O", "B|O"), ("A|SL", "A|S"), ("B|SL", "B|S"),
         ("A|S", "A|O"), ("B|SL", "A|O")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=Path, default=Path("/raid/ubx858/outputs/dnt/c1/split.json"))
    p.add_argument("--tracked", type=Path, default=Path("/raid/ubx858/outputs/dnt/c1/tracked"))
    p.add_argument("--tracking-log", type=Path, default=Path("/raid/ubx858/outputs/dnt/c1/tracking.log"))
    p.add_argument("--out", type=Path, default=Path("/raid/ubx858/outputs/dnt/avatar_A/compare"))
    p.add_argument("--fits", type=Path, default=Path("/raid/ubx858/outputs/dnt/avatar_A/sprint_fits"))
    p.add_argument("--ckpt", type=Path, default=SAF.DEFAULT_CKPT, help="SPRINT checkpoint (sprint_raw) for the fit and the S drivers")
    p.add_argument("--guava-ckpt", type=Path, default=Path("/raid/ubx858/outputs/dnt/guava_a20/run/checkpoints/best.pt"))
    p.add_argument("--face-conv", type=Path, default=Path("/raid/ubx858/outputs/dnt/avatar_A/conversion_c1/face_a20_c1.npz"))
    p.add_argument("--face-lmk", type=Path, default=Path("/raid/ubx858/outputs/dnt/avatar_A/face_lmk"))
    p.add_argument("--future-from", type=int, default=60); p.add_argument("--iters", type=int, default=150)
    p.add_argument("--max-clips", type=int, default=0); p.add_argument("--max-eval-frames", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--variants", nargs="*", default=None, help="subset of variants (extra: 'B|O:own' = O driving avatar B with B's own fitted shape)")
    p.add_argument("--reuse-fits", action="store_true", help="load avatars B / B0 from existing <fits>/<clip>/ instead of re-fitting (timing fields copied from <out-main>)")
    p.add_argument("--out-main", type=Path, default=Path("/raid/ubx858/outputs/dnt/avatar_A/compare"))
    args = p.parse_args()
    global VARIANTS
    if args.variants:
        VARIANTS = list(args.variants)
    device = torch.device(args.device); torch.manual_seed(0)
    fitter = SAF.SprintFitter(args.ckpt, args.device, args.iters)          # PEAR side first (module purge)

    sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
    import roma
    from omegaconf import OmegaConf
    from dataset import TrackedData_infer
    from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
    from models.modules.ehm import EHM as _EHM  # noqa: F401
    from utils.general_utils import ConfigDict, add_extra_cfgs
    from utils.graphics_utils import get_full_proj_matrix
    from submodules.lpipsPyTorch import LPIPS
    from transformers import VitPoseForPoseEstimation
    import torch.utils.data._utils.collate as collate
    import ast
    import onnx_arcface_torch as AF
    ehm_mod = sys.modules["models.modules.ehm.EHM"]
    met = SAF.load_by_path("guava_metrics", ROOT / "main" / "metrics.py")
    cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "configs/train/ubody_512.yaml")))
    inf = Ubody_Gaussian_inferer(cfg.MODEL).to(device).eval(); ren = GaussianRenderer(cfg.MODEL).to(device).eval()
    ck = torch.load(args.guava_ckpt, map_location="cpu", weights_only=False)
    rep_load = (inf.load_state_dict(ck["model"], strict=False), ren.load_state_dict(ck["render_model"], strict=False))
    print("GUAVA a20:", args.guava_ckpt, "iter", ck.get("global_iter"), "variant", ck.get("variant"), "val", ck.get("val"),
          "missing/unexpected", [len(r.missing_keys) for r in rep_load], [len(r.unexpected_keys) for r in rep_load], flush=True)
    assert ck.get("variant") == "a20"
    del ck
    src = (ROOT / "phase3_dnt/guava_finetune.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "smplx_only_forward")
    ns = {"ehm_mod": ehm_mod, "np": np, "torch": torch}; exec(compile(ast.Module([fn], []), "guava_finetune.py", "exec"), ns)
    NV = 10475
    with torch.no_grad():                                                   # a20: EHM's FLAME-template insertion undone
        s_ = inf.ehm.smplx; inf.ehm.v_template[:NV] = s_.v_template[:NV]; inf.v_template.data[:NV] = s_.v_template[:NV]
    inf.ehm.forward = types.MethodType(ns["smplx_only_forward"], inf.ehm)
    refine_fwd = ren.nerual_refiner.forward

    def fixed_noise_refine(x, *a, **k):
        torch.manual_seed(20261015)
        return refine_fwd(x, *a, **k)
    ren.nerual_refiner.forward = fixed_noise_refine
    lpips = LPIPS("vgg", "0.1").to(device)
    det = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-small").to(device).eval()
    det_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1); det_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    arc = AF.OnnxArcFace().to(device).eval()
    fc = np.load(args.face_conv)
    conv = {(str(c), str(k)): (e, j) for c, k, e, j in zip(fc["clip"], fc["frame_key"], fc["exp50"], fc["jaw"])}; eye_copy = bool(fc["eye_copy"])
    track_time = {}
    if args.tracking_log.exists():
        for line in args.tracking_log.read_text().splitlines():
            m = re.match(r"(\S+) exit 0 (\d+) s", line)
            if m: track_time[m.group(1)] = float(m.group(2))

    # ------------------------------------------------------------------ helpers (C1's metric code)
    def keypoints(img512):
        x = F.interpolate(img512, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - det_mean) / det_std)[:, :, :, 32:-32]
        with torch.inference_mode():
            hm = det(pixel_values=x, dataset_index=torch.zeros(x.shape[0], dtype=torch.long, device=device)).heatmaps.float()
        N, K, H, W = hm.shape
        flat = hm.reshape(N, K, -1); sc, idx = flat.max(-1)
        y, xk = (idx // W).float(), (idx % W).float()
        kp = torch.stack([((xk + 0.5) * 192 / W + 32) * 2, (y + 0.5) * 256 / H * 2], -1)
        return kp.cpu().numpy(), sc.cpu().numpy()

    def lpips_items(x, y):
        fx, fy = lpips.net(x), lpips.net(y)
        return torch.stack([l((a - b) ** 2).mean((1, 2, 3)) for a, b, l in zip(fx, fy, lpips.lin)], 0).sum(0)

    def arc_emb(imgs, kps5):
        th = AF.align_theta(kps5, imgs.shape[-1])[None].expand(imgs.shape[0], -1, -1)
        return arc(arc.preprocess(AF.crops(imgs, th)))

    def batch_metrics(R, G, M, boxes, kpG, scG, kps5, ref_emb):
        N = R.shape[0]
        Rm, Gm = R * M, (G * M).expand(N, -1, -1, -1)
        mse = (((Rm - Gm) ** 2).sum((1, 2, 3)) / (M.sum() * 3).clamp_min(1)).clamp_min(1e-10)
        psnr = (10 * torch.log10(1.0 / mse)).cpu().numpy()
        ss = met.ssim(Rm, Gm, size_average=False).cpu().numpy()
        lp = lpips_items(Rm, Gm).cpu().numpy()
        crop_vals = {}
        for key, bl in {"lpips_face": [boxes["head"]], "lpips_hands": [boxes["lhand"], boxes["rhand"]]}.items():
            vals = []
            for (l, r_, t_, b) in bl:
                if r_ - l >= 8 and b - t_ >= 8:
                    cr = F.interpolate(Rm[:, :, t_:b, l:r_], size=(128, 128), mode="bilinear", align_corners=False)
                    cg = F.interpolate(Gm[:, :, t_:b, l:r_], size=(128, 128), mode="bilinear", align_corners=False)
                    vals.append(lpips_items(cr, cg).cpu().numpy())
            crop_vals[key] = np.mean(vals, 0) if vals else np.full(N, np.nan)
        kpR, scR = keypoints(Rm)
        ok = scG >= 0.5
        torso = np.linalg.norm((kpG[5] + kpG[6]) / 2 - (kpG[11] + kpG[12]) / 2)
        ids = (arc_emb(R, kps5) @ ref_emb).cpu().numpy() if kps5 is not None else np.full(N, np.nan)
        out = []
        for n in range(N):
            kd = float(np.linalg.norm(kpR[n][ok] - kpG[ok], axis=1).mean() / max(torso, 1.0)) if ok.sum() >= 4 and torso > 8 else float("nan")
            out.append({"psnr": float(psnr[n]), "ssim": float(ss[n]), "lpips": float(lp[n]), "lpips_face": float(crop_vals["lpips_face"][n]),
                        "lpips_hands": float(crop_vals["lpips_hands"][n]), "kp_dist": kd, "id_sim": float(ids[n])})
        return out, kpR, scR

    q8 = lambda x: (x.clamp(0, 1) * 255).round() / 255
    aa = lambda r: roma.rotmat_to_rotvec(r.reshape(-1, 3, 3).float()).cpu().numpy().astype(np.float32)

    def a20_rec(rec, clip, key):
        """Tracker record -> a20 smplx dict (exp50 / jaw refit, eyes copied)."""
        sc = rec["smplx_coeffs"]; e50, jaw = conv[(clip, key)]
        return {"global_pose": np.asarray(sc["global_pose"], np.float32).reshape(3), "body_pose": np.asarray(sc["body_pose"], np.float32).reshape(21, 3),
                "left_hand_pose": np.asarray(sc["left_hand_pose"], np.float32).reshape(15, 3), "right_hand_pose": np.asarray(sc["right_hand_pose"], np.float32).reshape(15, 3),
                "exp": np.asarray(e50, np.float32).reshape(50), "jaw_pose": np.asarray(jaw, np.float32).reshape(1, 3),
                "eye_pose": (np.asarray(rec["flame_coeffs"]["eye_pose_params"], np.float32).reshape(2, 3) if eye_copy else np.zeros((2, 3), np.float32)),
                "camera_RT_params": np.asarray(sc["camera_RT_params"], np.float32).reshape(3, 4)}

    def sprint_rec(out):
        bp = out["body_param"]
        return {"global_pose": aa(bp["global_pose"])[0], "body_pose": aa(bp["body_pose"]), "left_hand_pose": aa(bp["left_hand_pose"]),
                "right_hand_pose": aa(bp["right_hand_pose"]), "exp": bp["exp"][0, :50].cpu().numpy().astype(np.float32),
                "jaw_pose": aa(bp["jaw_pose"]).reshape(1, 3), "eye_pose": np.zeros((2, 3), np.float32),
                "camera_RT_params": out["pd_cam"][0, :3, :4].cpu().numpy().astype(np.float32)}, bp["shape"][0, :200].cpu().numpy().astype(np.float32)

    def target_batch(sm, shape, tanfov, size):
        ti = {"smplx_coeffs": {k: torch.as_tensor(v, dtype=torch.float32) for k, v in sm.items()}, "flame_coeffs": {}}
        ti["smplx_coeffs"]["shape"] = torch.as_tensor(np.asarray(shape, np.float32).reshape(200))
        RT_mat = torch.eye(4); RT_mat[:3, :4] = ti["smplx_coeffs"]["camera_RT_params"]
        w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ RT_mat; c2w = torch.linalg.inv(w2c)
        view, full = get_full_proj_matrix(w2c, tanfov)
        ti["render_cam_params"] = {"world_view_transform": view, "full_proj_transform": full, "tanfovx": tanfov, "tanfovy": tanfov,
                                   "image_height": size, "image_width": size, "camera_center": c2w[:3, 3]}
        b = collate.default_collate([ti])
        mv = lambda x: x.to(device) if torch.is_tensor(x) else ({k: mv(v) for k, v in x.items()} if isinstance(x, dict) else x)
        return mv(b)

    def build_avatar(src):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            vg, ug, _ = inf(src)
            av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
        torch.cuda.synchronize()
        return av, time.perf_counter() - t

    def set_path(path):
        OmegaConf.set_readonly(cfg.DATASET, False); cfg.DATASET.data_path = str(path)

    split = json.loads(args.split.read_text())
    clips = [c for c in split["clips"] if (args.tracked / c["clip"] / "optim_tracking_ehm.pkl").exists()]
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f"{len(clips)} tracked clips", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "variants": VARIANTS, "clips": {}}

    def process(clip):
        name = clip["clip"]; odir = args.out / name; odir.mkdir(parents=True, exist_ok=True)
        set_path(args.tracked / name)
        ds = TrackedData_infer(cfg=cfg, split="test", device=device, test_full=True)
        vid = list(ds.videos_info.keys())[0]; keys_all = ds.videos_info[vid]["frames_keys"]
        ds._load_source_info(vid)
        keys, dropped = [], []
        for k in keys_all:
            try:
                ds._load_target_info(vid, k); keys.append(k)
            except (ValueError, RuntimeError):
                dropped.append(k)
        key0 = keys_all[0]; O_raw = {k: copy.deepcopy(ds.traked_data[k]) for k in keys_all}
        ids_ = ds.traked_data_id_share; shape_A = np.asarray(ids_["smplx_shape"][0], np.float32).reshape(200)
        ori = ds._lmdb_engine[f"{key0}/ori_image"].permute(1, 2, 0).numpy().copy()             # the reference photo (RGB)
        Hh, Ww = ori.shape[:2]
        # --- (b) SPRINT fit of the reference photo
        if args.reuse_fits:
            prev = json.loads((args.out_main / name / "summary.json").read_text())
            res = {"diag": prev["fit"]["diag"], "box": np.asarray(prev["fit"]["box"]), "cam_feat": prev["fit"]["cam_feat"]}
            fit_t = prev["fit"]["times_s"]
            shape_B = pickle.load(open(args.fits / name / "refined" / "id_share_params.pkl", "rb"))["smplx_shape"][0]
            shape_B0 = pickle.load(open(args.fits / name / "raw" / "id_share_params.pkl", "rb"))["smplx_shape"][0]
        else:
            res = fitter.fit(ori)
            fit_t = dict(res["times"])
            SAF.write_tracked(args.fits / name / "refined", name, key0, res, ori)
            SAF.write_tracked(args.fits / name / "raw", name, key0, res, ori, raw=True)
            SAF.overlay(res, args.fits / name / "overlay.jpg")
            shape_B, shape_B0 = res["shape"], res["raw_shape"]
        # --- avatars
        ds.traked_data[key0] = {**O_raw[key0], "smplx_coeffs": {**O_raw[key0]["smplx_coeffs"], **a20_rec(O_raw[key0], name, key0)}}
        srcA = ds._load_source_info(vid); ds.traked_data[key0] = O_raw[key0]
        avA, tA = build_avatar(srcA)
        avs, tbuild = {"A": avA}, {"A": tA}
        for tag, sub in (("B", "refined"), ("B0", "raw")):
            set_path(args.fits / name / sub)
            dsb = TrackedData_infer(cfg=cfg, split="test", device=device, test_full=True)
            srcb = dsb._load_source_info(list(dsb.videos_info.keys())[0]); dsb._lmdb_engine.close()
            avs[tag], tbuild[tag] = build_avatar(srcb)
        set_path(args.tracked / name)
        # --- reference embedding (source frame crop x mask, 512) and landmarks
        lm = np.load(args.face_lmk / f"{name}.npz"); lmk = {str(k): (s, kp) for k, s, kp in zip(lm["keys"], lm["score"], lm["kps"])}
        with torch.no_grad():
            t0_ = ds._load_target_info(vid, key0)
            G0 = q8(t0_["image"][0:1] * t0_["mask"][0:1])
            s0, k0 = lmk[key0]
            ref_emb = arc_emb(G0, k0)[0] if s0 > 0 else None
        # --- drivers
        M_hd = lambda k: np.asarray(O_raw[k]["body_crop"]["M_o2c-hd"], np.float64)
        per = {n: {m: [] for m in [x + s for x in METRICS for s in ("", "_aligned")]} for n in VARIANTS + ["GT"]}
        eval_keys = [k for i, k in enumerate(keys) if i >= args.future_from]
        if args.max_eval_frames:
            eval_keys = eval_keys[:args.max_eval_frames]
        drv_ms, grid = [], []
        t_loop = time.time()
        for i, k in enumerate(eval_keys):
            with torch.no_grad():
                tgt = ds._load_target_info(vid, k)
                M = tgt["mask"][0:1]; G = q8(tgt["image"][0:1] * M)
                boxes = {n: [int(x) for x in tgt[f"{b}_box"][0].tolist()] for n, b in (("head", "head"), ("lhand", "left_hand"), ("rhand", "right_hand"))}
                kpG, scG = keypoints(G); kpG, scG = kpG[0], scG[0]
                crop_t = ds._lmdb_engine[f"{k}/body_image"].float()[None].to(device) / 255.0
                cam = fitter.ev.cam_feat((np.diag([0.25, 0.25, 1.0]) @ M_hd(k))[:2], Ww, Hh, Ww, 0.0)
                torch.cuda.synchronize(); td = time.perf_counter()
                S, shS = sprint_rec(fitter.sprint(crop_t, cam))
                torch.cuda.synchronize(); drv_ms.append((time.perf_counter() - td) * 1000)
                SLA, _ = sprint_rec(fitter.sprint(crop_t, cam, shape_A))
                SLB, _ = sprint_rec(fitter.sprint(crop_t, cam, shape_B))
                O = a20_rec(O_raw[k], name, k)
                spec = {"A|O": ("A", O, shape_A), "A|S": ("A", S, shS), "A|SL": ("A", SLA, shape_A),
                        "B|O": ("B", O, shape_A), "B|S": ("B", S, shS), "B|SL": ("B", SLB, shape_B), "B0|O": ("B0", O, shape_A),
                        "B|O:own": ("B", O, shape_B)}
                Rs = []
                for n in VARIANTS:
                    av, sm, sh = spec[n]
                    tb = target_batch(sm, sh, ds.tanfov, ds.image_size)
                    Rs.append(q8(ren(avs[av](tb), tb["render_cam_params"], bg=0.0)["renders"][0:1]))
                    if i == 0 and n == "A|O" and clip is clips[0]:                       # builder == GUAVA's loader path
                        ds.traked_data[k] = {**O_raw[k], "smplx_coeffs": {**O_raw[k]["smplx_coeffs"], **O}}
                        tl = ds._load_target_info(vid, k); ds.traked_data[k] = O_raw[k]
                        R_ref = q8(ren(avs["A"](tl), tl["render_cam_params"], bg=0.0)["renders"][0:1])
                        dmax = float((Rs[-1] - R_ref).abs().max()); print(f"loader-path check A|O: max |diff| {dmax:.2e}", flush=True)
                        assert dmax < 1.5 / 255
                Rb = torch.cat(Rs + [G])                                                   # last row: GT (identity ceiling)
                s_, kp5 = lmk.get(k, (0.0, None))
                kps5 = kp5 if (s_ > 0 and ref_emb is not None) else None
                raw_m, kpR, scR = batch_metrics(Rb, G, M, boxes, kpG, scG, kps5, ref_emb)
                Ra = []
                for n in range(Rb.shape[0]):
                    okk = (scG >= 0.5) & (scR[n] >= 0.5)
                    if okk.sum() >= 4 and n < len(VARIANTS):
                        A_ = SAF_umeyama(kpR[n][okk], kpG[okk])
                        Ra.append(torch.from_numpy(cv2.warpAffine(Rb[n].permute(1, 2, 0).cpu().numpy(), A_, (512, 512), flags=cv2.INTER_LINEAR)).permute(2, 0, 1)[None].to(device))
                    else:
                        Ra.append(Rb[n:n + 1])
                al_m, _, _ = batch_metrics(torch.cat(Ra), G, M, boxes, kpG, scG, kps5, ref_emb)
            for n, nm in enumerate(VARIANTS + ["GT"]):
                for m in METRICS:
                    per[nm][m].append(raw_m[n][m]); per[nm][m + "_aligned"].append(al_m[n][m])
            if i in {0, len(eval_keys) // 2, len(eval_keys) - 1}:
                grid.append(torch.cat([G[0]] + [r[0] for r in Rs], 2))
        print(f"{name}: {len(eval_keys)} frames x {len(VARIANTS)} variants in {time.time() - t_loop:.0f} s", flush=True)
        ds._lmdb_engine.close()
        import torchvision
        torchvision.utils.save_image(torch.cat(grid, 1), odir / "compare.jpg")
        np.savez(odir / "per_frame.npz", keys=np.array(eval_keys), **{f"{n}|{m}": np.array(v) for n in per for m, v in per[n].items()})
        r = {"identity": clip["identity"], "frames_evaluated": len(eval_keys), "frames_dropped_unreadable": len(dropped),
             "means": {n: {m: (float(np.nanmean(v)) if np.isfinite(np.asarray(v, float)).any() else float("nan")) for m, v in per[n].items()} for n in per},
             "id_frames_with_face": int(np.isfinite(np.asarray(per["A|O"]["id_sim"], float)).sum()),
             "fit": {"times_s": fit_t, "diag": res["diag"], "box": res["box"].tolist(), "cam_feat": np.asarray(res["cam_feat"]).tolist()},
             "avatar_build_s": tbuild, "tracker_clip_s": track_time.get(name),
             "creation_s": {"a": (track_time.get(name) or float("nan")) + tbuild["A"], "b": fit_t["total"] + tbuild["B"], "b0": fit_t["total"] - fit_t["refine"] + tbuild["B0"]},
             "sprint_driver_ms_median": float(np.median(drv_ms)), "shape_B_minus_A_l2": float(np.linalg.norm(shape_B - shape_A))}
        (odir / "summary.json").write_text(json.dumps(r, indent=1) + "\n")
        return r

    SAF_umeyama = AF.umeyama2d
    failed = {}
    for clip in clips:
        f = args.out / clip["clip"] / "summary.json"
        if f.exists():
            summary["clips"][clip["clip"]] = json.loads(f.read_text()); print(clip["clip"], "reused", flush=True); continue
        try:
            summary["clips"][clip["clip"]] = process(clip)
            m = summary["clips"][clip["clip"]]["means"]
            print(clip["clip"], {n: (round(m[n]["psnr"], 2), round(m[n]["lpips"], 3), round(m[n]["id_sim"], 3)) for n in m}, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); failed[clip["clip"]] = repr(e); print(clip["clip"], "FAILED", repr(e), flush=True)
    summary["failed_clips"] = failed
    aggregate(summary, args.out)


def boot(x, rng_seed=0, n=1000):
    x = np.asarray(x, float); rng = np.random.default_rng(rng_seed)
    bs = [np.nanmean(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return {"mean": float(np.nanmean(x)), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}


def aggregate(summary, out: Path):
    ids = list(summary["clips"]); names = VARIANTS + ["GT"]
    names = [n for n in names if n in summary["clips"][ids[0]]["means"]]
    mk = list(summary["clips"][ids[0]]["means"]["A|O"])
    val = lambda n, m: np.array([summary["clips"][c]["means"][n][m] for c in ids], float)
    agg = {n: {m: boot(val(n, m)) for m in mk} for n in names}
    pairs = {f"{a} - {b}": {m: boot(val(a, m) - val(b, m)) for m in mk} for a, b in PAIRS + [("B|O:own", "A|O"), ("B|O:own", "B|O")] if a in names and b in names}
    cs = {k: boot([summary["clips"][c]["creation_s"][k] for c in ids]) for k in ("a", "b", "b0")}
    ft = {k: boot([summary["clips"][c]["fit"]["times_s"][k] for c in ids]) for k in summary["clips"][ids[0]]["fit"]["times_s"]}
    kpd = {k: boot([summary["clips"][c]["fit"]["diag"][k] for c in ids]) for k in ("kp_err_px1024_before", "kp_err_px1024_after")}
    build = {k: boot([summary["clips"][c]["avatar_build_s"][k] for c in ids]) for k in ("A", "B", "B0")}
    verdict = {}
    for drv in ("O", "SL"):
        if f"B|{drv} - A|{drv}" not in pairs:
            continue
        d = pairs[f"B|{drv} - A|{drv}"]
        crit = {"lpips": d["lpips"]["mean"] >= 0.01 and d["lpips"]["ci95"][0] > 0,
                "psnr": d["psnr"]["mean"] <= -0.5 and d["psnr"]["ci95"][1] < 0,
                "id_sim": d["id_sim"]["mean"] <= -0.05 and d["id_sim"]["ci95"][1] < 0}
        verdict[drv] = crit
    clearly_worse = any(any(v.values()) for v in verdict.values())
    summary.update({"n_identities": len(ids), "aggregate": agg, "paired": pairs, "creation_s": cs, "fit_times_s": ft, "fit_kp_err": kpd,
                    "avatar_build_s": build, "verdict": {"criteria_met": verdict, "b_clearly_worse_than_a": clearly_worse}})
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    c = lambda d, f=".3f": f"{d['mean']:{f}} [{d['ci95'][0]:{f}}, {d['ci95'][1]:{f}}]"
    L = [f"# Avatar A — SPRINT-fit vs EHM-tracker GUAVA (a20) avatars, {len(ids)} C1 identities; mean [95% CI over identities]\n"]
    for suf, title in (("", "Raw"), ("_aligned", "After per-frame 2D similarity alignment")):
        L += [f"\n## {title}\n", "| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | face LPIPS ↓ | hands LPIPS ↓ | kp dist ↓ | ArcFace sim ↑ |", "|---|---|---|---|---|---|---|---|"]
        for n in names:
            a = agg[n]; L.append(f"| {n} | " + " | ".join(c(a[m + suf], ".2f" if m == "psnr" else ".3f") for m in METRICS) + " |")
        L += [f"\n### Paired differences ({title.lower()})\n", "| Difference | ΔPSNR | ΔSSIM | ΔLPIPS | Δface LPIPS | Δhands LPIPS | Δkp dist | ΔArcFace |", "|---|---|---|---|---|---|---|---|"]
        for pn, d in pairs.items():
            L.append(f"| {pn} | " + " | ".join(c(d[m + suf], "+.2f" if m == "psnr" else "+.3f") for m in METRICS) + " |")
    L.append("\n## Creation time (s, per identity)\n")
    L.append(f"- (a) EHM-Tracker on the whole clip + avatar build: {c(cs['a'], '.0f')}")
    L.append(f"- (b) SPRINT fit + avatar build: {c(cs['b'])}; without the refinement (B0): {c(cs['b0'])}")
    L.append("- (b) fit components: " + ", ".join(f"{k} {c(v)}" for k, v in ft.items()))
    L.append("- avatar build: " + ", ".join(f"{k} {c(v)}" for k, v in build.items()))
    L.append(f"- DWPose keypoint error of the fit (px in the 1024 crop): SPRINT {c(kpd['kp_err_px1024_before'], '.1f')} -> refined {c(kpd['kp_err_px1024_after'], '.1f')}")
    L.append(f"\n## Verdict (pre-registered)\n\ncriteria met: {json.dumps(verdict)} -> **(b) clearly worse than (a): {clearly_worse}**")
    (out / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
