"""Multi-frame appearance fusion for GUAVA-FB (PREREG "Gap diagnosis D1" / "phase 7"): per-frame UV features of several frames of
the same person are fused into one UV map (texel weight = visibility x |cos(normal, view)|), then decoded by the UV decoder with
the ANCHOR frame's global style, camera and vertex branch. Same maths as eval_split/fusion_gt_row.py (whose source-only decode
equals the static build bit-identically), written for training as well (gradients flow through the frames' feature maps and the
decoders; the fusion weights are constants). Modes: mean (weighted running mean, anchor prior x its weight, optional threshold
/ confidence cap), wta (per texel the single best-facing frame), fill (anchor texels untouched, unseen texels from the best frame).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

import fb_common as F
from models.UbodyAvatar.ubody_gaussian import get_cam_dirs


class UVGeom:
    """UV-map constants of the inferer's SMPL-X (texel -> face, mask)."""
    def __init__(self, inf):
        sm = inf.smplx; self.faces = sm.faces_tensor; self.fidx = sm.uvmap_f_idx.long(); self.umask = sm.uvmap_mask.bool(); self.nF = self.faces.shape[0]


def frame_feats(inf, batch, geom, dev, need_vertex=True):
    """one frame (B = 1 batch with 'image' at the DINO size, GT pose + camera) -> {uv (35, H, W): RGB + DINO f_map1 on visible texels,
    vis, facing (|cos| x vis), d (DINO dict), deform, vfeat (projected vertex features of this frame)}."""
    itf = F.batch_invtanfov(batch, 1, dev)
    d = inf.dino_encoder(batch["image"], output_size=inf.cfg.image_size)
    deform = inf.ehm(batch["smplx_coeffs"], batch["flame_coeffs"]); V = deform["vertices"]
    rgb = Fn.interpolate(batch["image"], (inf.cfg.image_size, inf.cfg.image_size), mode="bilinear", align_corners=False)
    feat = torch.cat([rgb, d["f_map1"]], 1)
    with torch.no_grad():
        p2f, _ = F.render_fragments_fb(inf.mesh_renderer, V, batch["w2c_cam"], itf)
        fv = torch.zeros(geom.nF, dtype=torch.bool, device=dev); fv[p2f[p2f >= 0].long() % geom.nF] = True
        vis = geom.umask & fv[geom.fidx]
        v0, v1, v2 = V[0][geom.faces[:, 0]], V[0][geom.faces[:, 1]], V[0][geom.faces[:, 2]]
        nrm = Fn.normalize(torch.linalg.cross(v1 - v0, v2 - v0, dim=-1), dim=-1); R = batch["w2c_cam"][0, :3, :3]; t = batch["w2c_cam"][0, :3, 3]
        cosf = ((nrm @ R.T) * Fn.normalize(((v0 + v1 + v2) / 3) @ R.T + t, dim=-1)).sum(-1).abs().clamp(0, 1)
        facing = cosf[geom.fidx] * vis.float()
    uv = F.convert_pixel_feature_to_uv_fb(inf, feat, V, batch["w2c_cam"], visble_faces=p2f, invtanfov=itf)[0]
    out = {"d": d, "deform": deform, "uv": uv, "vis": vis, "facing": facing, "w2c": batch["w2c_cam"]}
    if need_vertex:
        vf, _ = F.sample_prj_feature_fb(inf, V, d["f_map2"], batch["w2c_cam"], invtanfov=itf); out["vfeat"] = vf[0]
    return out


def fuse(f0, frames, geom, mode="mean", prior=2.0, thr=0.0, cap=0.0):
    """anchor feats f0 + list of frame feats -> fused UV map (35, H, W). Texels no frame updates keep the anchor's map exactly."""
    uv0 = f0["uv"]; w0 = prior * f0["facing"]; touched = torch.zeros_like(geom.umask); capv = cap if cap > 0 else float("inf")
    if mode == "mean":
        S = uv0 * w0; W = w0.clone()
        for f in frames:
            w = f["facing"]; upd = (w > 0) & (w >= thr) & (W < capv) & geom.umask
            wu = torch.where(upd, w, torch.zeros_like(w)); S = S + f["uv"] * wu; W = W + wu; touched = touched | upd
        return torch.where(touched[None], S / W.clamp_min(1e-6), uv0), touched
    B = w0.clone(); FB = uv0
    if mode == "fill": B = torch.where(f0["vis"], torch.full_like(B, float("inf")), B)
    for f in frames:
        w = f["facing"]; upd = (w > B) & (w >= thr) & geom.umask
        FB = torch.where(upd[None], f["uv"], FB); B = torch.where(upd, w, B); touched = touched | upd
    return torch.where(touched[None], FB, uv0), touched


def decode(inf, src, f0, uvmap, seeded_dec=None):
    """the rest of the inferer forward (fb_common.inferer_forward_fb) on a UV map; vertex branch / style / camera = the anchor's.
    Returns (vertex_gs_dict, uv_point_gs_dict) for B = 1, as inf(src) would. seeded_dec: uv_fusion.seeded_decoder (inference)."""
    d = f0["d"]; B = 1; n = inf.uvmap_size
    gfe = inf.global_feature_mapping(d["f_global"]); cam = inf.harmo_encoder(get_cam_dirs(src["w2c_cam"]))
    vsf = torch.cat([f0["vfeat"][None], inf.vertex_base_feature[None].expand(B, -1, -1), gfe[:, None].expand(-1, f0["vfeat"].shape[0], -1)], -1)
    vg = inf.vertex_gs_decoder(vsf, cam); vg["positions"] = inf.v_template.clone()[None].expand(B, -1, -1)
    dec = seeded_dec if seeded_dec is not None else inf.uv_feature_decoder
    x = dec(uvmap[None], extra_style=inf.uv_style_mapping(d["f_global"]))
    x = torch.cat([x, inf.uv_base_feature[None].expand(B, -1, -1, -1)], 1); ug = inf.uv_point_decoder(x, cam)
    for k in list(ug.keys()):
        ug[k] = ug[k].reshape(B, n * n, -1)[:, inf.uv_mask_flat, :]
    ug["binding_face"] = inf.smplx.uvmap_f_idx.clone().reshape(1, n * n, -1)[:, inf.uv_mask_flat, :]
    ug["face_bary"] = inf.smplx.uvmap_f_bary.clone().reshape(1, n * n, -1)[:, inf.uv_mask_flat, :]
    inf.smplx_deform_res = f0["deform"]                                                             # as inferer_forward_fb (loss extras)
    return vg, ug


def fused_forward(inf, src, supports, geom, dev, mode="mean", prior=2.0, thr=0.0, cap=0.0):
    """training forward of one anchor (B = 1 batch) + its support frames (list of B = 1 batches; [] = single image, which equals
    inf(src) up to the StyleUNet noise draw). Returns (vg, ug, info)."""
    f0 = frame_feats(inf, src, geom, dev, need_vertex=True)
    frames = [frame_feats(inf, s, geom, dev, need_vertex=False) for s in supports]
    uvmap, touched = fuse(f0, frames, geom, mode, prior, thr, cap)
    vg, ug = decode(inf, src, f0, uvmap)
    n_uv = float(geom.umask.sum())
    return vg, ug, {"n_support": len(frames), "coverage_source": float(f0["vis"].sum()) / n_uv, "texels_touched": float(touched.sum()) / n_uv}


def cat_assets(dicts):
    """concatenate per-source (B = 1) asset dicts along the batch dim (tensors only; the binding / bary tensors are shared)."""
    out = {}
    for k in dicts[0]:
        v = dicts[0][k]
        out[k] = torch.cat([d[k] for d in dicts], 0) if torch.is_tensor(v) and v.shape[0] == 1 and k not in ("binding_face", "face_bary") else v
    return out
