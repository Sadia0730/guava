"""GT-pose appearance fusion over a clip's training frames (PREREG "Gap diagnosis D1", 2026-11-05).

The single-image row (fb_gt) builds the avatar from the source frame only. This row feeds the clip's training frames (frames.json
"recon": the frames the per-person baselines train on) through GUAVA-FB's image encoder + inverse texture mapping with the GT pose
of each frame and fuses them into the UV feature map by a weighted running mean (uv_fusion.py's rule without its live gates), then
re-runs the UV decoder once and renders the test frames with the GT poses, exactly as fb_gt. No gradient step, no per-person
training: it is a feed-forward oracle for "what the single-image model is missing because it never saw the other sides".
  texel weight w = visibility x |cos(surface normal, view)|; source anchor prior x its own weight; a texel is updated where
  w >= --thr and its accumulated weight W < --cap (0 = no cap); vertex branch, style and camera features stay the source's.
Arms (fixed before running, PREREG): registered = prior 2, thr 0.3, cap 4 (step-2 defaults); oracle = prior 1, thr 0, no cap;
D1b (2026-11-05, after D1): wta = per texel the best-facing frame only (prior 1, thr 0); fill = source texels kept, unseen filled.
Writes E/renders/<name>/<ds>/<clip>/<key>.png (+ run.json: settings, frames used, coverage, texels per frame, self-check of the
source-only decode against the static build) and uv_rgb_{source,fused}.png (RGB channels of the UV map, for figures).
Usage: CUDA_VISIBLE_DEVICES=1 python fusion_gt_row.py --E <root> --split <json> --name diag_fuse_reg --prior 2 --thr 0.3 --cap 4"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--E", required=True); ap.add_argument("--split", required=True); ap.add_argument("--clips", default="")
ap.add_argument("--name", required=True)
ap.add_argument("--prior", type=float, default=2.0); ap.add_argument("--thr", type=float, default=0.3); ap.add_argument("--cap", type=float, default=4.0)
ap.add_argument("--mode", default="mean", choices=("mean", "wta", "fill"), help="mean: weighted running mean (as registered); wta: per texel the single "
                "best-facing frame's features (source weight x prior); fill: source texels untouched, unseen texels from the best-facing frame")
ap.add_argument("--every", type=int, default=1, help="use every n-th training frame (1 = all)")
ap.add_argument("--max-frac", type=float, default=0.0, help="use the first ceil(frac x n) training frames after the source, in time order (extended benchmark 25 / 50 %% rows; 0 = all)")
ap.add_argument("--max-frames", type=int, default=0, help="use at most N training frames after the source (0 = all)")
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
ap.add_argument("--data-root", default=""); ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--support-gain", type=float, default=1.0, help="exposure-shift probe (PREREG phase 7 amendment 1): brightness gain applied to every training frame fused after the source")
ap.add_argument("--support-sat", type=float, default=1.0, help="exposure-shift probe: saturation factor applied to every training frame fused after the source")
a = ap.parse_args()
E = Path(a.E); SPLIT = json.loads(Path(a.split).read_text()); clips = SPLIT["clips"]
if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]
OUT = E / "renders" / a.name; DATA = Path(a.data_root) if a.data_root else E / "data"
LOCAL = Path("/dev/shm/ubx858_eval/renders") / a.name

import cv2, torch, torchvision                                                                       # noqa: E402
import torch.nn.functional as Fn                                                                     # noqa: E402
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE.parent / "worker6")); sys.path.insert(0, str(HERE.parent))
import fb_common as F                                                                                # noqa: E402
os.chdir(F.ROOT)
import uv_fusion as UF                                                                               # noqa: E402
from fb_common import to_dev                                                                         # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                        # noqa: E402
from models.UbodyAvatar.ubody_gaussian import get_cam_dirs                                          # noqa: E402

dev = torch.device("cuda:0"); GPU = torch.cuda.get_device_name(0)
cfg, inf, ren, _ = F.setup_a20(dev)
st = torch.load(a.fb_ckpt, map_location="cpu", weights_only=False)
r1 = inf.load_state_dict(st["model"], strict=False); r2 = ren.load_state_dict(st["render_model"], strict=False)
CK = {"ckpt": a.fb_ckpt, "step": st.get("step", st.get("global_iter")), "missing": len(r1.missing_keys) + len(r2.missing_keys)}
del st; inf.eval(); ren.eval()
sm = inf.smplx; FACES = sm.faces_tensor; FIDX = sm.uvmap_f_idx.long(); UMASK = sm.uvmap_mask.bool(); NF = FACES.shape[0]
DEC = UF.seeded_decoder(inf.uv_feature_decoder, dev)
print("model", CK, GPU, flush=True)


@torch.no_grad()
def exposure_shift(batch, gain, sat):
    """probe: brightness gain + saturation factor on a masked [0, 1] image batch (background stays 0)."""
    if gain == 1.0 and sat == 1.0: return batch
    im = batch["image"]; lum = (0.299 * im[:, 0] + 0.587 * im[:, 1] + 0.114 * im[:, 2])[:, None]
    batch["image"] = ((lum + (im - lum) * sat) * gain).clamp(0, 1); return batch


@torch.no_grad()
def frame_feats(batch):
    """uv_fusion.UVFusion.frame_feats without the vertex-template extras: 35-channel UV map (RGB + DINO f_map1, zero where not
    visible), per-texel visibility and facing |cos|, the DINO dict (global style) and the vertex features of this frame."""
    itf = F.batch_invtanfov(batch, 1, dev)
    d = inf.dino_encoder(batch["image"], output_size=inf.cfg.image_size)
    deform = inf.ehm(batch["smplx_coeffs"], batch["flame_coeffs"]); V = deform["vertices"]
    rgb = Fn.interpolate(batch["image"], (inf.cfg.image_size, inf.cfg.image_size), mode="bilinear", align_corners=False)
    feat = torch.cat([rgb, d["f_map1"]], 1)
    p2f, _ = F.render_fragments_fb(inf.mesh_renderer, V, batch["w2c_cam"], itf)
    uv = F.convert_pixel_feature_to_uv_fb(inf, feat, V, batch["w2c_cam"], visble_faces=p2f, invtanfov=itf)[0]
    fv = torch.zeros(NF, dtype=torch.bool, device=dev); fv[p2f[p2f >= 0].long() % NF] = True
    vis = UMASK & fv[FIDX]
    v0, v1, v2 = V[0][FACES[:, 0]], V[0][FACES[:, 1]], V[0][FACES[:, 2]]
    nrm = Fn.normalize(torch.linalg.cross(v1 - v0, v2 - v0, dim=-1), dim=-1); R = batch["w2c_cam"][0, :3, :3]; t = batch["w2c_cam"][0, :3, 3]
    nc = nrm @ R.T; cc = ((v0 + v1 + v2) / 3) @ R.T + t
    cosf = (nc * Fn.normalize(cc, dim=-1)).sum(-1).abs().clamp(0, 1)
    vf, _ = F.sample_prj_feature_fb(inf, V, d["f_map2"], batch["w2c_cam"], invtanfov=itf)
    return {"d": d, "uv": uv, "vis": vis, "facing": cosf[FIDX] * vis.float(), "w2c": batch["w2c_cam"], "vfeat": vf[0]}


@torch.no_grad()
def decode(src, f0, uvmap):
    """uv_fusion.UVFusion.decode: the rest of the inferer forward on a UV map; vertex branch / style / camera = source's."""
    d = f0["d"]; B = 1
    gfe = inf.global_feature_mapping(d["f_global"]); cam = inf.harmo_encoder(get_cam_dirs(src["w2c_cam"]))
    vsf = torch.cat([f0["vfeat"][None], inf.vertex_base_feature[None].expand(B, -1, -1), gfe[:, None].expand(-1, F.NV if f0["vfeat"].shape[0] == F.NV else f0["vfeat"].shape[0], -1)], -1)
    vg = inf.vertex_gs_decoder(vsf, cam); vg["positions"] = inf.v_template.clone()[None].expand(B, -1, -1)
    x = DEC(uvmap[None], extra_style=inf.uv_style_mapping(d["f_global"]))
    x = torch.cat([x, inf.uv_base_feature[None].expand(B, -1, -1, -1)], 1); ug = inf.uv_point_decoder(x, cam)
    n = inf.uvmap_size
    for k in list(ug.keys()):
        ug[k] = ug[k].reshape(B, n * n, -1)[:, inf.uv_mask_flat, :]
    ug["binding_face"] = inf.smplx.uvmap_f_idx.clone().reshape(1, n * n, -1)[:, inf.uv_mask_flat, :]
    ug["face_bary"] = inf.smplx.uvmap_f_bary.clone().reshape(1, n * n, -1)[:, inf.uv_mask_flat, :]
    av = Ubody_Gaussian(cfg.MODEL, UF.clone_assets(vg), UF.clone_assets(ug), pruning=True); av.init_ehm(inf.ehm); av.eval()
    return av


@torch.no_grad()
def build_static(src):
    torch.manual_seed(0); vg, ug, _ = inf(src)
    av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval(); return av


@torch.no_grad()
def render(av, t):
    torch.manual_seed(0); rr = ren(av(t), t["render_cam_params"], bg=0.0); return rr["renders"].clamp(0, 1)[0]


def u8(x): return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def uv_rgb_png(uv, path):
    rgb = (uv[:3].clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    OUT.mkdir(parents=True, exist_ok=True); cap = a.cap if a.cap > 0 else float("inf")
    for c in clips:
        ds, name = c["dataset"], c["clip"]; od_final = OUT / ds / name
        if (od_final / "run.json").exists() and not a.overwrite: print("done", c["id"], flush=True); continue
        cdir = DATA / ds / name; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
        if not (cdir / "meta.json").exists(): print("missing data", c["id"], flush=True); continue
        od = LOCAL / ds / name
        if od.exists(): shutil.rmtree(od)
        od.mkdir(parents=True, exist_ok=True)
        cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1); fbd = F.FBData(cfg, cl); recs = set(cl[0]["recs"])
        t0 = time.perf_counter()
        src = to_dev(F.collate1(fbd.source_info(0, fj["source"])), dev); f0 = frame_feats(src)
        w0 = a.prior * f0["facing"]; S = f0["uv"] * w0; W = w0.clone(); uv0 = f0["uv"]; touched = torch.zeros_like(UMASK)
        seen = f0["vis"].clone(); n_uv = float(UMASK.sum())
        B = w0.clone(); FB = uv0.clone()                                                             # wta / fill: best weight + its features
        if a.mode == "fill": B = torch.where(f0["vis"], torch.full_like(B, float("inf")), B)
        # self-check: the source-only decode == the static build (first test frame, max abs diff in 8-bit units)
        t_first = to_dev(F.collate1(fbd.target_info(0, fj["test"][0])), dev)
        r_static = u8(render(build_static(src), t_first)); r_dec0 = u8(render(decode(src, f0, uv0), t_first))
        selfcheck = float((r_static.int() - r_dec0.int()).abs().max())
        train = [k for k in fj["recon"] if k != fj["source"] and k in recs][:: a.every]
        if a.max_frames: train = train[:a.max_frames]
        if a.max_frac: import math; train = train[:math.ceil(a.max_frac * len(train))]
        per = []
        for k in train:
            f = frame_feats(exposure_shift(to_dev(F.collate1(fbd.source_info(0, k)), dev), a.support_gain, a.support_sat)); seen |= f["vis"]
            w = f["facing"]
            if a.mode == "mean":
                upd = (w > 0) & (w >= a.thr) & (W < cap) & UMASK
                wu = torch.where(upd, w, torch.zeros_like(w)); S += f["uv"] * wu; W += wu; touched |= upd
            else:
                upd = (w > B) & (w >= a.thr) & UMASK
                FB = torch.where(upd[None], f["uv"], FB); B = torch.where(upd, w, B); touched |= upd; W = torch.where(upd, w, W)
            per.append({"key": k, "n_texels": int(upd.sum()), "visible": int(f["vis"].sum()), "mem_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 3)})
        uvmap = torch.where(touched[None], S / W.clamp_min(1e-6), uv0) if a.mode == "mean" else torch.where(touched[None], FB, uv0)
        av = decode(src, f0, uvmap); t_build = (time.perf_counter() - t0) * 1000
        tr = []
        for k in fj["test"]:
            t = to_dev(F.collate1(fbd.target_info(0, k)), dev)
            torch.cuda.synchronize(); t1 = time.perf_counter(); r = render(av, t); torch.cuda.synchronize(); tr.append(time.perf_counter() - t1)
            torchvision.io.write_png(u8(r), str(od / f"{k}.png"))
        uv_rgb_png(uv0, od / "uv_rgb_source.png"); uv_rgb_png(uvmap, od / "uv_rgb_fused.png")
        cov0 = float(f0["vis"].sum()) / n_uv; cov = float((W > 0).sum()) / n_uv; covt = float(touched.sum()) / n_uv
        info = {"method": a.name, **CK, "gpu": GPU, "settings": {"mode": a.mode, "prior": a.prior, "thr": a.thr, "cap": a.cap, "every": a.every, "max_frames": a.max_frames, "max_frac": a.max_frac, "support_gain": a.support_gain, "support_sat": a.support_sat},
                "source": fj["source"], "n_train_used": len(train), "n_train_available": len(fj["recon"]) - 1, "n_test": len(fj["test"]),
                "coverage_source": cov0, "coverage_seen_any_frame": float(seen.sum()) / n_uv, "coverage_fused": cov, "texels_updated_frac": covt,
                "mean_W_updated": float(W[touched].mean()) if covt > 0 else 0.0, "selfcheck_source_only_vs_static_u8": selfcheck,
                "fusion_ms_total": t_build, "render_ms_per_frame": float(np.mean(tr) * 1000), "per_frame": per,
                "note": "GT-pose fusion of the training frames into the UV map, feed-forward, no per-person optimisation (PREREG gap diagnosis D1)"}
        od_final.mkdir(parents=True, exist_ok=True)
        for p_ in sorted(od.iterdir()): shutil.copyfile(p_, od_final / p_.name)
        (od_final / "run.json").write_text(json.dumps(info, indent=1)); shutil.rmtree(od)
        print(json.dumps({k: v for k, v in info.items() if k != "per_frame"}), flush=True)


main()
