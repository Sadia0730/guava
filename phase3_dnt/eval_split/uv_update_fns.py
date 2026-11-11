"""Appearance-update functions shared by the SPRINT-pose update row (PREREG extended benchmark amendment 6 (a)).
frame_feats / decode are copied verbatim from fusion_gt_row.py (the fb_update rows; that file stays unchanged), bound to an
inferer by make(); fuse_mean is fusion_gt_row's "mean" rule (prior x source facing, threshold, cap)."""
from __future__ import annotations

import torch
import torch.nn.functional as Fn


def make(inf, cfg, dev, F, UF):
    from models.UbodyAvatar import Ubody_Gaussian
    from models.UbodyAvatar.ubody_gaussian import get_cam_dirs
    sm = inf.smplx; FACES = sm.faces_tensor; FIDX = sm.uvmap_f_idx.long(); UMASK = sm.uvmap_mask.bool(); NF = FACES.shape[0]
    DEC = UF.seeded_decoder(inf.uv_feature_decoder, dev)

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
        return {"d": d, "uv": uv, "vis": vis, "facing": cosf[FIDX] * vis.float(), "w2c": batch["w2c_cam"], "vfeat": vf[0]}

    @torch.no_grad()
    def decode(src, f0, uvmap):
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

    class MeanFusion:
        """fusion_gt_row.py --mode mean: S / W running weighted mean, source anchor prior x facing."""
        def __init__(self, f0, prior=2.0, thr=0.0, cap=0.0):
            self.f0 = f0; w0 = prior * f0["facing"]; self.S = f0["uv"] * w0; self.W = w0.clone(); self.uv0 = f0["uv"]
            self.touched = torch.zeros_like(UMASK); self.seen = f0["vis"].clone(); self.thr = thr; self.cap = cap if cap > 0 else float("inf")

        @torch.no_grad()
        def add(self, f):
            self.seen |= f["vis"]; w = f["facing"]
            upd = (w > 0) & (w >= self.thr) & (self.W < self.cap) & UMASK
            wu = torch.where(upd, w, torch.zeros_like(w)); self.S += f["uv"] * wu; self.W += wu; self.touched |= upd
            return int(upd.sum())

        def uvmap(self): return torch.where(self.touched[None], self.S / self.W.clamp_min(1e-6), self.uv0)

        def coverage(self):
            n = float(UMASK.sum())
            return {"coverage_source": float(self.f0["vis"].sum()) / n, "coverage_seen_any_frame": float(self.seen.sum()) / n,
                    "coverage_fused": float((self.W > 0).sum()) / n, "texels_updated_frac": float(self.touched.sum()) / n}

    return frame_feats, decode, MeanFusion
