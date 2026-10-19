"""E2 "avatar improves during a session" (phase3_dnt/AVATAR_PLAN.md): test-time, gradient-free fusion of GUAVA's per-frame
UV features (the 35-channel map = RGB + DINO f_map1, inverse-texture-mapped, before the UV decoder; models/UbodyAvatar/
ubody_gaussian.py convert_pixel_feature_to_uv). The vertex branch, the global / style feature and cam_dirs stay frame 0's.
Per accepted frame i: weight w_i = visible_i x facing_i x conf_i (facing = clamp(cos(normal, -view)), conf = 1 here);
frame 0 is the prior with weight PRIOR x w_0. Outlier texels (RGB channels > OUTLIER from the current fused RGB where the
fused weight >= 1) get weight 0. A frame is accepted if it adds >= NEW_COV of new UV coverage or its camera-frame root
orientation differs >= 15 deg from every accepted frame. The UV decoder is re-run whenever coverage grew >= 2 % since
the last decode; stop when coverage >= 0.9 and the held-out LPIPS gain (held-out frames inside the fusion window) is
< 0.002 for 3 consecutive refreshes. Fusion window: frames [1, FUTURE); held-out: every 6th of them. Evaluation: frames
>= FUTURE (never fused), driven by the EHM tracker (O), static frame-0 avatar vs the final fused avatar: masked PSNR /
LPIPS (as C1), plus identity drift = LPIPS at frame 0 (fused vs static)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="/raid/ubx858/outputs/dnt/c1/split.json"); ap.add_argument("--tracked", default="/raid/ubx858/outputs/dnt/c1/tracked")
ap.add_argument("--out", default="/raid/ubx858/outputs/dnt/e2/smoke"); ap.add_argument("--clips", type=int, default=1); ap.add_argument("--future-from", type=int, default=60)
ap.add_argument("--max-eval", type=int, default=0); ap.add_argument("--prior", type=float, default=2.0); ap.add_argument("--outlier", type=float, default=0.25); ap.add_argument("--new-cov", type=float, default=0.005)
a = ap.parse_args(); out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda:0")
from omegaconf import OmegaConf
from dataset import TrackedData_infer
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
from models.UbodyAvatar.ubody_gaussian import get_cam_dirs
from utils.general_utils import ConfigDict, add_extra_cfgs

from submodules.lpipsPyTorch import LPIPS
import roma
cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "assets/GUAVA/config.yaml")))
inf = Ubody_Gaussian_inferer(cfg.MODEL).to(dev).eval(); ren = GaussianRenderer(cfg.MODEL).to(dev).eval()
ck = torch.load(ROOT / "assets/GUAVA/checkpoints/best_160000.pt", map_location="cpu", weights_only=True)
inf.load_state_dict(ck["model"], strict=False); ren.load_state_dict(ck["render_model"], strict=False); del ck
_rf = ren.nerual_refiner.forward
def _fixed(x, *a_, **k_):
    torch.manual_seed(20261015); return _rf(x, *a_, **k_)
ren.nerual_refiner.forward = _fixed
lp = LPIPS("vgg", "0.1").to(dev)
def lpips_item(x, y):
    fx, fy = lp.net(x), lp.net(y); return float(torch.stack([l((p - q) ** 2).mean((1, 2, 3)) for p, q, l in zip(fx, fy, lp.lin)], 0).sum())
q8 = lambda x: (x.clamp(0, 1) * 255).round() / 255
M = inf.cfg if hasattr(inf, "cfg") else cfg.MODEL
@torch.no_grad()
def frame_uv(src):
    """35-ch UV map, visibility, facing for one source dict (B=1); also the per-frame pieces of the static forward."""
    d = inf.dino_encoder(src["image"], output_size=inf.cfg.image_size)
    deform = inf.ehm(src["smplx_coeffs"], src["flame_coeffs"]); V = deform["vertices"]
    rgb = F.interpolate(src["image"], (inf.cfg.image_size, inf.cfg.image_size), mode="bilinear", align_corners=False)
    feat = torch.cat([rgb, d["f_map1"]], 1)
    vis, _ = inf.mesh_renderer.render_fragments(V, transform_matrix=src["w2c_cam"])
    uv = inf.convert_pixel_feature_to_uv(feat, V, src["w2c_cam"], visble_faces=vis)          # zeroed where invisible
    sm = inf.smplx; faces = sm.faces_tensor; fidx = sm.uvmap_f_idx
    nb = faces.shape[0]; visf = torch.isin(torch.arange(nb, device=dev, dtype=torch.int32), torch.unique(vis))
    visible = (sm.uvmap_mask.bool() & visf[fidx]).float()
    # facing: face normal (camera frame) vs the direction to the camera at the texel
    v0, v1, v2 = V[0][faces[:, 0]], V[0][faces[:, 1]], V[0][faces[:, 2]]
    nrm = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1); R = src["w2c_cam"][0, :3, :3]
    nc = nrm @ R.T; ctr = ((v0 + v1 + v2) / 3); cc = torch.cat([ctr, torch.ones_like(ctr[:, :1])], 1) @ src["w2c_cam"][0].T
    cosf = (-(nc * F.normalize(cc[:, :3], dim=-1)).sum(-1)).abs().clamp(0, 1)                  # sign-free: |cos| (convention checked below)
    facing = cosf[fidx] * visible
    return {"uv": uv[0], "visible": visible, "facing": facing, "d": d, "deform": deform}
@torch.no_grad()
def decode(prior, uvmap):
    """The rest of Ubody_Gaussian_inferer.forward with the (fused) UV map; vertex branch / style / cam_dirs from the prior."""
    src = prior["src"]; d = prior["d"]; B = 1
    gfe = inf.global_feature_mapping(d["f_global"]); cam = inf.harmo_encoder(get_cam_dirs(src["w2c_cam"]))
    vsf, _ = inf.sample_prj_feature(prior["deform"]["vertices"], d["f_map2"], src["w2c_cam"])
    vsf = torch.cat([vsf, inf.vertex_base_feature[None].expand(B, -1, -1), gfe[:, None].expand(-1, vsf.shape[1], -1)], -1)
    vg = inf.vertex_gs_decoder(vsf, cam); vg["positions"] = inf.v_template.clone()[None].expand(B, -1, -1)
    x = inf.uv_feature_decoder(uvmap[None], extra_style=inf.uv_style_mapping(d["f_global"]))
    x = torch.cat([x, inf.uv_base_feature[None].expand(B, -1, -1, -1)], 1); ug = inf.uv_point_decoder(x, cam)
    for k in list(ug.keys()):
        ug[k] = ug[k].reshape(B, inf.uvmap_size * inf.uvmap_size, -1)[:, inf.uv_mask_flat, :]
    ug["binding_face"] = inf.smplx.uvmap_f_idx.clone().reshape(1, -1, 1)[:, inf.uv_mask_flat, :]
    ug["face_bary"] = inf.smplx.uvmap_f_bary.clone().reshape(1, inf.uvmap_size * inf.uvmap_size, -1)[:, inf.uv_mask_flat, :]
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval(); return av
@torch.no_grad()
def render_eval(av, ds, vid, keys):
    ps, ls = [], []
    for k in keys:
        t = ds._load_target_info(vid, k); Mk = t["mask"][0:1]; G = q8(t["image"][0:1] * Mk)
        R_ = q8(ren(av(t), t["render_cam_params"], bg=0.0)["renders"][0:1])
        mse = (((R_ * Mk - G) ** 2).sum() / (Mk.sum() * 3).clamp_min(1)).clamp_min(1e-10)
        ps.append(float(10 * torch.log10(1 / mse))); ls.append(lpips_item(R_ * Mk, G))
    return float(np.mean(ps)), float(np.mean(ls))
def root_rot(src):
    gp = src["smplx_coeffs"]["global_pose"].reshape(-1)[:3].float(); R = roma.rotvec_to_rotmat(gp)
    return src["w2c_cam"][0, :3, :3].float() @ R
split = json.loads(Path(a.split).read_text())["clips"]; done = 0; summary = {}
for c in split:
    cdir = Path(a.tracked) / c["clip"]
    if not (cdir / "optim_tracking_ehm.pkl").exists(): continue
    t0 = time.time()
    OmegaConf.set_readonly(cfg.DATASET, False); cfg.DATASET.data_path = str(cdir)
    ds = TrackedData_infer(cfg=cfg, split="test", device=dev, test_full=True)
    vid = list(ds.videos_info.keys())[0]; keys = ds.videos_info[vid]["frames_keys"]
    ok = []
    for i, k in enumerate(keys):
        try: ds._load_target_info(vid, k); ok.append(i)
        except (ValueError, RuntimeError): pass
    win = [i for i in ok if 1 <= i < a.future_from]; held = win[::6]; fuse_cand = [i for i in win if i not in held]
    fut = [keys[i] for i in ok if i >= a.future_from]; fut = fut[:a.max_eval] if a.max_eval else fut
    src0 = ds._load_source_info(vid, 0); f0 = frame_uv(src0); prior = {"src": src0, "d": f0["d"], "deform": f0["deform"]}
    w0 = a.prior * f0["facing"]; S = f0["uv"] * w0; W = w0.clone()
    static = decode(prior, f0["uv"]); cur = static
    heldk = [keys[i] for i in held]; base_held = render_eval(static, ds, vid, heldk)[1]; last_held = base_held
    mask_n = float(inf.smplx.uvmap_mask.sum()); cov = lambda W_: float((W_ > 1e-3).sum()) / mask_n
    cov_last = cov(W); accepted = [0]; rots = [root_rot(src0)]; small_gain = 0; refreshes = []; log = []
    for i in fuse_cand:
        s = ds._load_source_info(vid, i); f = frame_uv(s); R = root_rot(s)
        ang = min(float(torch.rad2deg(roma.rotmat_geodesic_distance(R, r))) for r in rots)
        w = f["facing"].clone()
        fused_rgb = S[:3] / W.clamp_min(1e-6)
        out = ((f["uv"][:3] - fused_rgb).abs().mean(0) > a.outlier) & (W >= 1.0); w[out] = 0
        new = float(((W <= 1e-3) & (w > 1e-3)).sum()) / mask_n
        acc = new >= a.new_cov or ang >= 15.0
        log.append({"frame": i, "new_cov": new, "min_angle_deg": ang, "outlier_frac": float(out.sum()) / mask_n, "accepted": acc})
        if not acc: continue
        S += f["uv"] * w; W += w; accepted.append(i); rots.append(R)
        if cov(W) - cov_last >= 0.02:
            cur = decode(prior, S / W.clamp_min(1e-6)); cov_last = cov(W)
            h = render_eval(cur, ds, vid, heldk)[1]; gain = last_held - h; last_held = h
            refreshes.append({"after_frame": i, "coverage": cov_last, "held_lpips": h, "gain": gain})
            small_gain = small_gain + 1 if gain < 0.002 else 0
            if cov_last >= 0.9 and small_gain >= 3: break
    if not refreshes or refreshes[-1]["after_frame"] != accepted[-1]:
        cur = decode(prior, S / W.clamp_min(1e-6))
    ps_s, lp_s = render_eval(static, ds, vid, fut); ps_f, lp_f = render_eval(cur, ds, vid, fut)
    _, id_s = render_eval(static, ds, vid, [keys[0]]); _, id_f = render_eval(cur, ds, vid, [keys[0]])
    r = {"identity": c["identity"], "future_frames": len(fut), "accepted_frames": len(accepted), "coverage_static": cov(w0), "coverage_final": cov(W),
         "refreshes": refreshes, "static": {"psnr": ps_s, "lpips": lp_s}, "fused": {"psnr": ps_f, "lpips": lp_f},
         "lpips_rel_gain": (lp_s - lp_f) / lp_s, "frame0_lpips": {"static": id_s, "fused": id_f}, "seconds": time.time() - t0, "log": log}
    summary[c["clip"]] = r; ds._lmdb_engine.close()
    print(json.dumps({k: v for k, v in r.items() if k not in ("log",)}, indent=1), flush=True)
    done += 1
    if done >= a.clips: break
(out_root / "summary.json").write_text(json.dumps(summary, indent=1))
