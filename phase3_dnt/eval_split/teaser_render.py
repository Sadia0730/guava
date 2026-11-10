"""Teaser renders (presentation only; PREREG "Extended benchmark, amendment 4"): for one clip, build the avatar from the source photo
(the fb_gt build) and the updated avatar (the fb_update100 build: every recon frame fused into the UV map with its GT pose, mean,
prior 2, thr 0, cap 0), then render
  orbit:  both avatars in the source pose, the camera turned about the body's long axis by the given angles (no GT exists);
  anim:   the photo-only avatar at the test frames at the given fractions of the test list, GT poses (+ the GT crop);
each as RGBA PNG at 512 (alpha = rasterised Gaussian coverage, the silhouette term's a_t), plus the 1024 source photo and the UV RGB
maps. Nothing is scored; the clip and frames are fixed by rule (fig_teaser.py), never by eye.
Usage: CUDA_VISIBLE_DEVICES=6 python teaser_render.py --E <root> --clip <ds/clip> --out <dir> [--angles 0,90,180] [--anim 0.25,0.5,0.75]"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--E", required=True); ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--angles", default="0,45,90,135,180,225,270,315"); ap.add_argument("--anim", default="0.25,0.5,0.75")
ap.add_argument("--back-key", default="", help="test frame to also render with a zoomed camera (amendment 5): photo-only, updated, GT")
ap.add_argument("--zoom", type=float, default=2.0)
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
a = ap.parse_args()
E = Path(a.E); ds, name = a.clip.split("/"); OUT = Path(a.out) / ds / name; OUT.mkdir(parents=True, exist_ok=True)

import cv2, torch, torchvision                                                                       # noqa: E402
import numpy as np                                                                                   # noqa: E402
import torch.nn.functional as Fn                                                                     # noqa: E402
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE.parent / "worker6")); sys.path.insert(0, str(HERE.parent))
import fb_common as F                                                                                # noqa: E402
os.chdir(F.ROOT)
import uv_fusion as UF                                                                               # noqa: E402
from fb_common import to_dev                                                                         # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                        # noqa: E402
from models.UbodyAvatar.ubody_gaussian import get_cam_dirs                                          # noqa: E402

dev = torch.device("cuda:0")
cfg, inf, ren, _ = F.setup_a20(dev)
st = torch.load(a.fb_ckpt, map_location="cpu", weights_only=False)
inf.load_state_dict(st["model"], strict=False); ren.load_state_dict(st["render_model"], strict=False); del st; inf.eval(); ren.eval()
sm = inf.smplx; FACES = sm.faces_tensor; FIDX = sm.uvmap_f_idx.long(); UMASK = sm.uvmap_mask.bool(); NF = FACES.shape[0]
DEC = UF.seeded_decoder(inf.uv_feature_decoder, dev)


# frame_feats / decode / build_static: identical to fusion_gt_row.py (the fb_update rows)
@torch.no_grad()
def frame_feats(batch):
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
    return {"d": d, "uv": uv, "vis": vis, "facing": cosf[FIDX] * vis.float(), "vfeat": vf[0], "V": V}


@torch.no_grad()
def decode(src, f0, uvmap):
    d = f0["d"]; B = 1
    gfe = inf.global_feature_mapping(d["f_global"]); cam = inf.harmo_encoder(get_cam_dirs(src["w2c_cam"]))
    vsf = torch.cat([f0["vfeat"][None], inf.vertex_base_feature[None].expand(B, -1, -1), gfe[:, None].expand(-1, f0["vfeat"].shape[0], -1)], -1)
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
def render_rgba(av, t, rcp):
    torch.manual_seed(0); g = av(t); rgb = ren(g, rcp, bg=0.0)["renders"].clamp(0, 1)[0]
    g1 = dict(g); g1["features_color"] = torch.ones_like(g["features_color"])
    alpha = ren.forward_raw(g1, rcp, bg=0.0)["raw_features"][0, :1].clamp(0, 1)
    return torch.cat([rgb, alpha], 0)


def save(x, p):
    x = (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()                 # RGBA -> BGRA for cv2
    cv2.imwrite(str(p), cv2.cvtColor(x, cv2.COLOR_RGBA2BGRA))


def orbit_cam(t, V, deg):
    """w2c' = [R_u(deg) | c - R_u c] w2c: the camera turned about the body's long axis u (first principal axis of the posed
    vertices in camera coordinates, sign towards camera +y) through the body centre c; render_cam_params for 512 px with the
    record's own invtanfov. (Camera-y as the axis tilts the body when the perspective crop is rotated.)"""
    w2c = t["w2c_cam"][0].double(); Vc = V[0].double() @ w2c[:3, :3].T + w2c[:3, 3]; c = Vc.mean(0)
    u = torch.linalg.svd(Vc - c, full_matrices=False).Vh[0]; u = u if float(u[1]) >= 0 else -u
    r = math.radians(deg); K = torch.tensor([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=torch.float64, device=dev)
    R = torch.eye(3, dtype=torch.float64, device=dev) + math.sin(r) * K + (1 - math.cos(r)) * (K @ K)
    M = torch.eye(4, dtype=torch.float64, device=dev); M[:3, :3] = R; M[:3, 3] = c - R @ c
    w = (M @ w2c).float(); itf = float(t["invtanfov"][0]) if torch.is_tensor(t["invtanfov"]) else float(t["invtanfov"])
    return to_dev(F.collate1(F.render_cam_params(w.cpu(), torch.linalg.inv(w).cpu(), itf, 512)), dev)


def main():
    cdir = E / "data" / ds / name; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
    cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1); fbd = F.FBData(cfg, cl); recs = set(cl[0]["recs"])
    src = to_dev(F.collate1(fbd.source_info(0, fj["source"])), dev); f0 = frame_feats(src)
    # the fb_update100 fusion (fusion_gt_row.py --mode mean --prior 2 --thr 0 --cap 0, all recon frames after the source)
    w0 = 2.0 * f0["facing"]; S = f0["uv"] * w0; W = w0.clone(); touched = torch.zeros_like(UMASK)
    for k in [k for k in fj["recon"] if k != fj["source"] and k in recs]:
        f = frame_feats(to_dev(F.collate1(fbd.source_info(0, k)), dev)); w = f["facing"]
        upd = (w > 0) & UMASK; wu = torch.where(upd, w, torch.zeros_like(w)); S += f["uv"] * wu; W += wu; touched |= upd
    uvmap = torch.where(touched[None], S / W.clamp_min(1e-6), f0["uv"])
    avs = {"photo": build_static(src), "update": decode(src, f0, uvmap)}
    ts = to_dev(F.collate1(fbd.target_info(0, fj["source"])), dev)
    for deg in [float(x) for x in a.angles.split(",")]:
        rcp = orbit_cam(ts, f0["V"], deg)
        for kind, av in avs.items(): save(render_rgba(av, ts, rcp), OUT / f"orbit_{kind}_{int(deg):03d}.png")
    test = fj["test"]; anim = []
    for fr in [float(x) for x in a.anim.split(",")]:
        k = test[min(len(test) - 1, int(round(fr * (len(test) - 1))))]; anim.append(k)
        t = to_dev(F.collate1(fbd.target_info(0, k)), dev)
        save(render_rgba(avs["photo"], t, t["render_cam_params"]), OUT / f"anim_photo_{k}.png")
        save(render_rgba(avs["update"], t, t["render_cam_params"]), OUT / f"anim_update_{k}.png")
        save(torch.cat([t["image"][0], t["mask"][0][:1]], 0), OUT / f"anim_gt_{k}.png")
    if a.back_key:                                                                                   # zoomed back view (amendment 5)
        _, _, tm = fbd.info(0, a.back_key); ys, xs = np.nonzero(tm[0].numpy() > 0.5)                 # centre: 38 % down the body
        cen = (float(np.median(xs)), float(ys.min() + 0.38 * (ys.max() - ys.min())))
        tz = fbd.target_info_zoom(0, a.back_key, factor=a.zoom, center=cen, min_cover=0.0)
        tz = to_dev(F.collate1(tz), dev); z = f"zoom{a.zoom:g}"
        save(render_rgba(avs["photo"], tz, tz["render_cam_params"]), OUT / f"back_{z}_photo_{a.back_key}.png")
        save(render_rgba(avs["update"], tz, tz["render_cam_params"]), OUT / f"back_{z}_update_{a.back_key}.png")
        save(torch.cat([tz["image"][0], tz["mask"][0][:1]], 0), OUT / f"back_{z}_gt_{a.back_key}.png")
        t = to_dev(F.collate1(fbd.target_info(0, a.back_key)), dev)
        for kind in ("photo", "update"): save(render_rgba(avs[kind], t, t["render_cam_params"]), OUT / f"back_full_{kind}_{a.back_key}.png")
        save(torch.cat([t["image"][0], t["mask"][0][:1]], 0), OUT / f"back_full_gt_{a.back_key}.png")
    img = sorted((E / "base" / ds / name / "images").glob(fj["source"] + ".*"))
    if img: shutil.copyfile(img[0], OUT / ("source" + img[0].suffix))
    for kk, uv in (("source", f0["uv"]), ("fused", uvmap)):
        cv2.imwrite(str(OUT / f"uv_rgb_{kk}.png"), cv2.cvtColor((uv[:3].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    n_uv = float(UMASK.sum())
    (OUT / "run.json").write_text(json.dumps({"clip": a.clip, "source": fj["source"], "anim_frames": anim, "angles": a.angles,
                                              "coverage_photo": float(f0["vis"].sum()) / n_uv, "coverage_after_update": float((W > 0).sum()) / n_uv,
                                              "n_recon_fused": int(len([k for k in fj["recon"] if k != fj["source"] and k in recs])),
                                              "ckpt": a.fb_ckpt}, indent=1))
    print("done", a.clip, flush=True)


main()
