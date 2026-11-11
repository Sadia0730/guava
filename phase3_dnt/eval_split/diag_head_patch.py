"""Diagnosis of the pale side-of-face patch (RUNLOG 135; no benchmark output touched): the updated avatar of one clip rendered at one
zoomed test frame with only the UV or only the vertex Gaussians; the mesh faces under pale head pixels -> their UV texels -> what each
recon frame samples there (image pixel, matte value). Built from teaser_render.py (lines 1-96) + the diagnostic code below; writes to --out."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

HERE = Path("/home/ubx858/guava/phase3_dnt/eval_split")
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
    x = (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(p), cv2.cvtColor(x, cv2.COLOR_RGBA2BGRA))


@torch.no_grad()
def render_mask(av, t, rcp, keep):
    """keep: 'all' | 'uv' (vertex Gaussians' opacity 0) | 'vertex' (UV Gaussians' opacity 0)."""
    torch.manual_seed(0); g = dict(av(t)); nv = av._smplx_opacity.shape[1]; op = g["opacity"].clone()
    if keep == "uv": op[:, :nv] = 0
    if keep == "vertex": op[:, nv:] = 0
    g["opacity"] = op; rgb = ren(g, rcp, bg=0.0)["renders"].clamp(0, 1)[0]
    g1 = dict(g); g1["features_color"] = torch.ones_like(g["features_color"])
    al = ren.forward_raw(g1, rcp, bg=0.0)["raw_features"][0, :1].clamp(0, 1)
    return torch.cat([rgb, al], 0)


cdir = E / "data" / ds / name; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
cl = F.load_clips_parallel([(str(cdir), None, True)], workers=1); fbd = F.FBData(cfg, cl); recs = set(cl[0]["recs"])
src = to_dev(F.collate1(fbd.source_info(0, fj["source"])), dev); f0 = frame_feats(src)
w0 = 2.0 * f0["facing"]; Sx = f0["uv"] * w0; W = w0.clone(); touched = torch.zeros_like(UMASK)
for k in [k for k in fj["recon"] if k != fj["source"] and k in recs]:
    f = frame_feats(to_dev(F.collate1(fbd.source_info(0, k)), dev)); w = f["facing"]
    upd = (w > 0) & UMASK; wu = torch.where(upd, w, torch.zeros_like(w)); Sx += f["uv"] * wu; W += wu; touched |= upd
uvmap = torch.where(touched[None], Sx / W.clamp_min(1e-6), f0["uv"])
avs = {"photo": build_static(src), "update": decode(src, f0, uvmap)}
k = a.back_key; _, _, tm = fbd.info(0, k); ys, xs = np.nonzero(tm[0].numpy() > 0.5)
cen = (float(np.median(xs)), float(ys.min() + 0.38 * (ys.max() - ys.min())))
tz = to_dev(F.collate1(fbd.target_info_zoom(0, k, factor=a.zoom, center=cen, min_cover=0.0)), dev)
for kind, av in avs.items():
    print(kind, "vertex Gaussians", av._smplx_opacity.shape[1], "uv Gaussians", av._uv_opacity.shape[1])
    for keep in ("all", "uv", "vertex"): save(render_mask(av, tz, tz["render_cam_params"], keep), OUT / f"abl_{kind}_{keep}_{k}.png")

# --- which texels draw the pale patch? mesh faces seen at pale head pixels of the updated render (zoom view) ---
import torch.nn.functional as Fn2
upd_rgb = render_mask(avs["update"], tz, tz["render_cam_params"], "uv")[:3]                       # 3 x 512 x 512
gt = tz["image"][0]; gm = tz["mask"][0][0]
itf = F.batch_invtanfov(tz, 1, dev); V = inf.ehm(tz["smplx_coeffs"], tz["flame_coeffs"])["vertices"]
p2f, _ = F.render_fragments_fb(inf.mesh_renderer, V, tz["w2c_cam"], itf); p2f = p2f[0, ..., 0] if p2f.dim() == 4 else p2f[0]
R = p2f.shape[-1]; print("raster", tuple(p2f.shape))
rs = lambda x: Fn2.interpolate(x[None, None] if x.dim() == 2 else x[None], (R, R), mode="bilinear")[0]
lr, lg = rs(upd_rgb).mean(0), rs(gt).mean(0)
jj = sm.lbs_weights.argmax(1); hv = torch.isin(jj, torch.tensor([15, 22, 23, 24], device=jj.device)).long()
fl = hv[FACES.long()]; hf = torch.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0]).to(dev)
pf = p2f.long() % FACES.shape[0]; head_px = (p2f >= 0) & (hf[pf] == 1)
patch = head_px & (lr > 0.45) & (lr - lg > 0.15)
print("head pixels", int(head_px.sum()), "patch pixels", int(patch.sum()))
fs = torch.unique(pf[patch]); tex = torch.isin(FIDX, fs) & UMASK
print("patch faces", len(fs), "texels", int(tex.sum()))
print("render RGB at patch", rs(upd_rgb)[:, patch].mean(1).tolist(), "GT RGB at patch", rs(gt)[:, patch].mean(1).tolist())
print("touched share", float(touched[tex].float().mean()), "source-visible share", float(f0["vis"][tex].float().mean()))
print("fused RGB at those texels", uvmap[:3][:, tex].mean(1).tolist(), "source RGB (visible ones)", f0["uv"][:3][:, tex & f0["vis"]].mean(1).tolist() if int((tex & f0["vis"]).sum()) else None)
print("fused weight W mean", float(W[tex].mean()), "median", float(W[tex].median()))
im = (rs(upd_rgb).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy(); im[patch.cpu().numpy()] = (255, 0, 255)
cv2.imwrite(str(OUT / "patch_px.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
uvv = (uvmap[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy(); uvv[tex.cpu().numpy()] = (255, 0, 255)
cv2.imwrite(str(OUT / "patch_uv.png"), cv2.cvtColor(uvv[:240, :320], cv2.COLOR_RGB2BGR))

def coords_uv(batch, n):
    itf = F.batch_invtanfov(batch, 1, dev); V = inf.ehm(batch["smplx_coeffs"], batch["flame_coeffs"])["vertices"]
    S_ = inf.cfg.image_size; yy, xx = torch.meshgrid(torch.arange(S_, device=dev).float(), torch.arange(S_, device=dev).float(), indexing="ij")
    feat = torch.cat([torch.stack([xx, yy])[None], torch.zeros(1, 33, S_, S_, device=dev)], 1)
    p2f, _ = F.render_fragments_fb(inf.mesh_renderer, V, batch["w2c_cam"], itf)
    return F.convert_pixel_feature_to_uv_fb(inf, feat, V, batch["w2c_cam"], visble_faces=p2f, invtanfov=itf)[0][:2], S_
crops = []
for k2 in ("frame_000088", "frame_000136"):
    b = to_dev(F.collate1(fbd.source_info(0, k2)), dev); f = frame_feats(b); m = tex & f["vis"]
    xy, S_ = coords_uv(b, uvmap.shape[-1]); _, img, msk = fbd.info(0, k2); s = img.shape[-1] / S_
    pts = (xy[:, m].T * s).round().long().cpu().numpy(); H = img.shape[-1]; pts = pts.clip(0, H - 1)
    im = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy(); mm = msk[0].numpy()
    print(k2, "matte at samples: mean", round(float(mm[pts[:, 1], pts[:, 0]].mean()), 3), "share < 0.5", round(float((mm[pts[:, 1], pts[:, 0]] < 0.5).mean()), 3),
          "raw image RGB at samples", (img[:, pts[:, 1], pts[:, 0]].mean(1)).tolist())
    ov = im.copy(); ov[pts[:, 1], pts[:, 0]] = (255, 0, 255); mv = np.stack([(mm * 255).astype(np.uint8)] * 3, -1); mv[pts[:, 1], pts[:, 0]] = (255, 0, 255)
    y0, y1, x0, x1 = max(0, pts[:, 1].min() - 80), pts[:, 1].max() + 80, max(0, pts[:, 0].min() - 80), pts[:, 0].max() + 80
    cv2.imwrite(str(OUT / f"patch_src_{k2}.png"), cv2.cvtColor(np.concatenate([im[y0:y1, x0:x1], ov[y0:y1, x0:x1], mv[y0:y1, x0:x1]], 1), cv2.COLOR_RGB2BGR))
