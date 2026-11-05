"""PREREG phase 6 step 2 (C3 appearance update): gradient-free fusion of GUAVA-FB's per-frame UV features, used by
worker_core.Worker(mode="uv") in the background thread. Port of e2_uv_fusion.py (GUAVA upper body) to GUAVA-FB's inferer
(fb_common's per-sample camera: render_fragments_fb, convert_pixel_feature_to_uv_fb, sample_prj_feature_fb).

Per frame: GUAVA-FB's image encoder on the frame's body crop x matte (the source build's path: JPEG q95 round trip, 518 px) ->
inverse texture mapping with the REFINED pose -> 35-channel UV map (RGB + DINO f_map1, zero where invisible) + visibility.
Texel weight = visibility x facing |cos| x exp(-kp_err / 20 px) x min(1, LapVar(crop) / LapVar(source crop)). Running weighted
mean S / W; the source is the anchor with prior weight (2 or 4) x its own weight; a texel is updated only where the frame weight
>= the update threshold AND W < 4; outlier texels (mean |RGB - fused RGB| > 0.25 where W >= 1) get weight 0; outlier frames
(render-vs-image LPIPS under the current avatar > median + 3 MAD of the accepted frames, after >= 5) are rejected. Head-region
texels only if the head is frontal (yaw <= 30, pitch <= 20 deg), the face crop is sharp (LapVar >= 0.8 x the source's) and the
face keypoint error <= 10 px. Decode = the rest of the inferer forward on the fused map (vertex branch, style and camera features
stay the source's; option: template branch with averaged vertex features). ArcFace rollback, per-region stop rule (held-out
reservoir) and the luminance reopen as registered.

Determinism: the decoder and the worker's renderer are deep copies whose StyleConv noise comes from a private CUDA generator
re-seeded with 0 before every call; this reproduces torch.manual_seed(0) before the original module's call (bit-identical), so
decode(source-only map) == the static avatar, and the live thread's global-RNG noise (re-seeded before every live render) is
never touched by the worker."""
from __future__ import annotations

import copy
import time
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision

import fb_common as F
import onnx_arcface_torch as AF
from models.UbodyAvatar import Ubody_Gaussian
from models.UbodyAvatar.ubody_gaussian import get_cam_dirs
from models.modules.net_module.styleunet.styleunet import StyleConv
from submodules.lpipsPyTorch import LPIPS

REGIONS = ("head", "torso", "arms", "legs")
_HEAD = {15, 22, 23, 24}
_LEGS = {1, 2, 4, 5, 7, 8, 10, 11}
_ARMS = set(range(16, 22)) | set(range(25, 55))
FACE68 = 24                                    # openpose134 layout of SprintFitter: face landmarks 24..91 (iBUG 68 order)


# ------------------------------------------------------------------------------------------------ seeded module copies
def _seeded_styleconv_forward(self, x, style, noise=None):
    """models/modules/net_module/styleunet/styleunet.py StyleConv.forward, noise drawn from self._p6_gen."""
    out = self.modulated_conv(x, style) * 2 ** 0.5
    if noise is None:
        b, _, h, w = out.shape
        noise = out.new_empty(b, 1, h, w).normal_(generator=self._p6_gen)
    out = out + self.weight * noise
    out = out + self.bias
    return self.activate(out)


def seed_styleconvs(mod, gen):
    for s in mod.modules():
        if isinstance(s, StyleConv):
            s._p6_gen = gen; s.forward = types.MethodType(_seeded_styleconv_forward, s)
    return mod


def seeded_decoder(dec, device, seed=0):
    m = seed_styleconvs(copy.deepcopy(dec), torch.Generator(device=device)); gen = next(s._p6_gen for s in m.modules() if isinstance(s, StyleConv))

    def call(*a, **k):
        gen.manual_seed(seed); return m(*a, **k)
    return call


def seeded_renderer(ren, device, seed=0):
    r = copy.deepcopy(ren); gen = torch.Generator(device=device); seed_styleconvs(r.nerual_refiner, gen)

    def refine_raw(self, raw):
        gen.manual_seed(seed); return self.nerual_refiner(raw["raw_features"])
    r.refine_raw = types.MethodType(refine_raw, r)
    return r


# ------------------------------------------------------------------------------------------------ small helpers
def lapvar(rgb_u8):
    return float(cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def luminance(crop_u8, alpha):
    m = alpha > 0.5
    if m.sum() < 1: return float("nan")
    y = crop_u8.astype(np.float64) @ np.array([0.299, 0.587, 0.114]) / 255.0
    return float(y[m].mean())


def jpeg_rt(chw_u8):
    return torchvision.io.decode_jpeg(torchvision.io.encode_jpeg(chw_u8.contiguous(), quality=95))


def face_kp_err(r, dyn, SAF):
    """mean 2D error (px at the 1024 crop) of the refined fit on the face landmarks it used (score >= 0.3, in the crop, not a
    contour point), as diag['kp_err_px1024_after'] restricted to face indices; inf if fewer than 5 landmarks."""
    kc = np.asarray(r["kp_crop"]).reshape(-1, 2); sc = np.asarray(r["kp_score"]).reshape(-1); uv = np.asarray(r["uv_after"]).reshape(-1, 2)
    n = min(len(kc), len(uv), len(sc), len(dyn)); idx = np.arange(FACE68, FACE68 + 68); idx = idx[idx < n]
    ok = (sc[idx] >= SAF.KP_THR) & (kc[idx] >= 0).all(1) & (kc[idx] <= 1).all(1) & ~dyn[idx]
    if ok.sum() < 5: return float("inf")
    return float((np.linalg.norm(uv[idx][ok] - kc[idx][ok], axis=1) * SAF.CROP).mean())


def five_points(kp_crop, kp_score, size=512, thr=0.3):
    """ArcFace 5 landmarks (image-left eye, image-right eye, nose tip, mouth corners) from the iBUG-68 face points; None if weak."""
    kp = np.asarray(kp_crop).reshape(-1, 2) * size; sc = np.asarray(kp_score).reshape(-1)
    g = lambda ids: [FACE68 + i for i in ids]                                                     # noqa: E731
    sets = [g(range(36, 42)), g(range(42, 48)), g([30]), g([48]), g([54])]
    if max(max(s) for s in sets) >= len(sc) or any(sc[s].min() < thr for s in sets): return None
    return np.stack([kp[s].mean(0) for s in sets]).astype(np.float32)


def head_angles(deform, w2c):
    """(yaw, pitch) in deg of the head's forward axis (+z of the SMPL-X head joint frame) against the head -> camera direction."""
    Rh = deform["joint_transform_mat"][0, 15, :3, :3].double(); ph = deform["joints_transform"][0, 15].double()
    Rc = w2c[0, :3, :3].double(); tc = w2c[0, :3, 3].double()
    f = Rc @ Rh @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=Rh.device); p = Rc @ ph + tc
    z = -p / p.norm(); ey = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=Rh.device)
    y = ey - (ey @ z) * z; y = y / y.norm(); x = torch.linalg.cross(y, z)
    fx, fy, fz = float(f @ x), float(f @ y), float(f @ z)
    return float(np.degrees(abs(np.arctan2(fx, fz)))), float(np.degrees(abs(np.arctan2(fy, np.hypot(fx, fz)))))


def clone_assets(d):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()}


# ------------------------------------------------------------------------------------------------ visibility helpers (also used by the replay)
def pose_batch(smplx, shp, dev):
    """the minimal batch (pose + camera) of a SPRINT smplx dict, for visibility / mesh rasterisation only."""
    rec = F.compact_record({"smplx_coeffs": smplx, **{k: {"M_o2c": np.eye(3, dtype=np.float32), "M_c2o": np.eye(3, dtype=np.float32)}
                                                      for k in F.CROP_KEYS}}, shp)
    sm = {k: torch.from_numpy(v.copy()) for k, v in rec["smplx"].items()}
    sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
    M4 = torch.eye(4); M4[:3, :4] = sm["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
    b = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "invtanfov": float(rec["invtanfov"])}
    return F.to_dev(F.collate1(b), dev)


@torch.no_grad()
def face_visibility(inf, batch, dev):
    """(faces visible in the batch's camera (bool, nF), pix_to_face (1, 512, 512, K), posed vertices)."""
    nF = inf.smplx.faces_tensor.shape[0]; itf = F.batch_invtanfov(batch, 1, dev)
    V = inf.ehm(batch["smplx_coeffs"], batch["flame_coeffs"])["vertices"]
    p2f, _ = F.render_fragments_fb(inf.mesh_renderer, V, batch["w2c_cam"], itf)
    fv = torch.zeros(nF, dtype=torch.bool, device=dev); fv[p2f[p2f >= 0].long() % nF] = True
    return fv, p2f, V


@torch.no_grad()
def unseen_mask(inf, src_face_vis, smplx, shp, dev):
    """512 x 512 uint8 (crop camera): pixels whose surface (SMPL-X face under the pixel) had zero source visibility; and the mesh
    silhouette (alignment check against the Gaussian render)."""
    nF = inf.smplx.faces_tensor.shape[0]; _, p2f, _ = face_visibility(inf, pose_batch(smplx, shp, dev), dev)
    fl = p2f[0, :, :, 0].long(); m = (fl >= 0) & ~src_face_vis[fl.clamp_min(0) % nF]
    return (m.to(torch.uint8) * 255).cpu().numpy(), ((fl >= 0).to(torch.uint8) * 255).cpu().numpy()


class UVFusion:
    """Fusion state + checks of one session. Methods called from the worker thread only (except face_visibility / render_unseen,
    used by the replay after the live loop)."""

    def __init__(self, cfg, inf, ren, src, rs, img0, shp, fitter, saf, prior=2.0, upd_thr=0.3, tmpl_avg=False, conf_cap=4.0,
                 outlier_rgb=0.25, heldout_k=4, dev="cuda:0"):
        self.cfg, self.inf, self.shp, self.fitter, self.saf, self.dev = cfg, inf, shp, fitter, saf, torch.device(dev)
        self.prior, self.thr, self.tmpl_avg, self.cap, self.out_rgb, self.K = prior, upd_thr, tmpl_avg, conf_cap, outlier_rgb, heldout_k
        self.dec = seeded_decoder(inf.uv_feature_decoder, self.dev); self.wren = seeded_renderer(ren, self.dev)
        self.lp = LPIPS("vgg", "0.1").to(self.dev).eval(); self.arc = AF.OnnxArcFace().to(self.dev).eval()
        self.dyn = fitter.lean.dyn.detach().cpu().numpy().astype(bool)
        sm = inf.smplx; self.faces = sm.faces_tensor; self.fidx = sm.uvmap_f_idx.long(); self.umask = sm.uvmap_mask.bool()
        self.nF, self.nV = self.faces.shape[0], sm.lbs_weights.shape[0]; self.n_uv = float(self.umask.sum())
        # regions: SMPL-X skinning argmax -> head / torso / arms / legs, per vertex, face (majority of 3) and texel
        j = sm.lbs_weights.argmax(1).cpu().numpy(); lv = np.full(len(j), 1)
        lv[np.isin(j, list(_HEAD))] = 0; lv[np.isin(j, list(_ARMS))] = 2; lv[np.isin(j, list(_LEGS))] = 3
        fl = lv[self.faces.cpu().numpy()]; lf = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0])
        self.vreg = torch.from_numpy(lv).to(self.dev); self.freg = torch.from_numpy(lf).to(self.dev)
        self.treg = torch.where(self.umask, self.freg[self.fidx], torch.full_like(self.fidx, -1))
        # anchor (source frame): its batch is the one the static avatar was built from
        self.src, self.rs = src, rs
        with torch.no_grad():
            f0 = self.frame_feats(src)
        self.src_face_vis = f0["face_vis"]
        self.fitq0 = float(np.exp(-rs["diag"]["kp_err_px1024_after"] / 20.0)); self.lap0 = lapvar(rs["crop"])
        self.face_lap0 = lapvar(self._face_crop(img0, rs["record"]))
        w0 = self.prior * f0["facing"] * self.fitq0
        self.S = f0["uv"] * w0; self.W = w0.clone(); self.seen = f0["vis"].clone()
        self.uv0 = f0["uv"]; self.touched = torch.zeros_like(self.umask)                           # texels no frame updated keep the anchor map exactly
        self.src_yaw_pitch = head_angles(f0["deform"], f0["w2c"])
        if self.tmpl_avg:
            wv0 = self.prior * f0["vfacing"] * self.fitq0
            self.Sv = f0["vfeat"] * wv0[:, None]; self.Wv = wv0.clone(); self.vtouched = torch.zeros(self.nV, dtype=torch.bool, device=self.dev)
        self.d0, self.V0, self.itf0 = f0["d"], f0["V"], f0["itf"]; self.vfeat0 = f0["vfeat"]
        with torch.no_grad():
            self.av_work = self.decode()                                                         # == the static avatar (checked by the caller)
        # ArcFace reference: source crop x matte at 512, aligned with the source's DWPose face landmarks
        self.kps5 = five_points(rs["kp_crop"], rs["kp_score"])
        if self.kps5 is not None:
            self.arc_ref = self._arc(self._img512(rs["crop"], rs["alpha"]))
            self.arc_sim = self._arc_sim(self.av_work)
        else:
            self.arc_ref = None; self.arc_sim = None
        self.lp_acc = []; self.res = []; self.lp_cache = {}; self.ver = 0
        self.stop = {r: False for r in range(4)}; self.hist = {r: [] for r in range(4)}; self.lum_stop = {r: None for r in range(4)}
        self.n_rollback = 0

    # -------------------------------------------------------------------------------------------- per-frame pieces
    def batch_from_fit(self, r):
        """FBData.source_info of SAF.write_tracked(r) without the disk: JPEG q95 round trip of the crop and the matte."""
        rec = F.compact_record(r["record"], r["shape"])
        sm = {k: torch.from_numpy(v.copy()) for k, v in rec["smplx"].items()}
        sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
        M4 = torch.eye(4); M4[:3, :4] = sm["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
        img = jpeg_rt(torch.from_numpy(r["crop"]).permute(2, 0, 1)).float() / 255.0
        a = torch.from_numpy((np.clip(r["alpha"], 0, 1) * 255).round().astype(np.uint8))[None].expand(3, -1, -1)
        msk = jpeg_rt(a).float() / 255.0
        image = torchvision.transforms.functional.resize(img * msk, (self.cfg.MODEL.feature_img_size,) * 2, antialias=True)
        b = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c),
             "invtanfov": float(rec["invtanfov"]), "image": image}
        return F.to_dev(F.collate1(b), self.dev)

    def pose_batch(self, smplx):
        return pose_batch(smplx, self.shp, self.dev)

    def face_visibility(self, batch):
        return face_visibility(self.inf, batch, self.dev)

    @torch.no_grad()
    def frame_feats(self, batch):
        inf = self.inf; itf = F.batch_invtanfov(batch, 1, self.dev)
        d = inf.dino_encoder(batch["image"], output_size=inf.cfg.image_size)
        deform = inf.ehm(batch["smplx_coeffs"], batch["flame_coeffs"]); V = deform["vertices"]
        rgb = Fn.interpolate(batch["image"], (inf.cfg.image_size, inf.cfg.image_size), mode="bilinear", align_corners=False)
        feat = torch.cat([rgb, d["f_map1"]], 1)
        p2f, _ = F.render_fragments_fb(inf.mesh_renderer, V, batch["w2c_cam"], itf)
        uv = F.convert_pixel_feature_to_uv_fb(inf, feat, V, batch["w2c_cam"], visble_faces=p2f, invtanfov=itf)[0]
        fv = torch.zeros(self.nF, dtype=torch.bool, device=self.dev); fv[p2f[p2f >= 0].long() % self.nF] = True
        vis = self.umask & fv[self.fidx]
        v0, v1, v2 = V[0][self.faces[:, 0]], V[0][self.faces[:, 1]], V[0][self.faces[:, 2]]
        nrm = Fn.normalize(torch.linalg.cross(v1 - v0, v2 - v0, dim=-1), dim=-1); R = batch["w2c_cam"][0, :3, :3]; t = batch["w2c_cam"][0, :3, 3]
        nc = nrm @ R.T; cc = ((v0 + v1 + v2) / 3) @ R.T + t
        cosf = (nc * Fn.normalize(cc, dim=-1)).sum(-1).abs().clamp(0, 1)
        out = {"d": d, "V": V, "deform": deform, "itf": itf, "uv": uv, "vis": vis, "facing": cosf[self.fidx] * vis.float(), "face_vis": fv,
               "w2c": batch["w2c_cam"]}
        vf, _ = F.sample_prj_feature_fb(inf, V, d["f_map2"], batch["w2c_cam"], invtanfov=itf); out["vfeat"] = vf[0]
        if self.tmpl_avg or not hasattr(self, "S"):
            vvis = torch.zeros(self.nV, dtype=torch.bool, device=self.dev); vvis[self.faces[fv].reshape(-1)] = True
            vn = torch.zeros_like(V[0]).index_add_(0, self.faces.reshape(-1), nrm.repeat_interleave(3, 0))
            vc = V[0] @ R.T + t; cosv = (Fn.normalize(vn @ R.T, dim=-1) * Fn.normalize(vc, dim=-1)).sum(-1).abs().clamp(0, 1)
            out["vfacing"] = cosv * vvis.float()
        return out

    @torch.no_grad()
    def coverage_gain(self, smplx):
        fv, _, _ = self.face_visibility(self.pose_batch(smplx))
        return float((self.umask & fv[self.fidx] & (self.W <= 0)).sum()) / self.n_uv

    @torch.no_grad()
    def decode(self):
        """the rest of the inferer forward (fb_common.inferer_forward_fb) on the fused map; vertex branch / style / camera = source's.
        Texels (vertices) that no frame updated keep the anchor's own features exactly (decode of the source-only state == static)."""
        inf, src, d = self.inf, self.src, self.d0; B = 1
        uvmap = torch.where(self.touched[None], self.S / self.W.clamp_min(1e-6), self.uv0)
        gfe = inf.global_feature_mapping(d["f_global"]); cam = inf.harmo_encoder(get_cam_dirs(src["w2c_cam"]))
        vsf = self.vfeat0[None]
        if self.tmpl_avg:
            vsf = torch.where(self.vtouched[:, None], self.Sv / self.Wv.clamp_min(1e-6)[:, None], self.vfeat0)[None]
        vsf = torch.cat([vsf, inf.vertex_base_feature[None].expand(B, -1, -1), gfe[:, None].expand(-1, vsf.shape[1], -1)], -1)
        vg = inf.vertex_gs_decoder(vsf, cam); vg["positions"] = inf.v_template.clone()[None].expand(B, -1, -1)
        x = self.dec(uvmap[None], extra_style=inf.uv_style_mapping(d["f_global"]))
        x = torch.cat([x, inf.uv_base_feature[None].expand(B, -1, -1, -1)], 1); ug = inf.uv_point_decoder(x, cam)
        n = inf.uvmap_size
        for k in list(ug.keys()):
            ug[k] = ug[k].reshape(B, n * n, -1)[:, inf.uv_mask_flat, :]
        ug["binding_face"] = inf.smplx.uvmap_f_idx.clone().reshape(1, n * n, -1)[:, inf.uv_mask_flat, :]
        ug["face_bary"] = inf.smplx.uvmap_f_bary.clone().reshape(1, n * n, -1)[:, inf.uv_mask_flat, :]
        self._last_assets = (vg, ug)
        av = Ubody_Gaussian(self.cfg.MODEL, clone_assets(vg), clone_assets(ug), pruning=True); av.init_ehm(inf.ehm); av.eval()
        return av

    def live_copy(self):
        """a second avatar instance (the live loop's) from the last decode: Ubody_Gaussian.forward stores per-call state on the
        instance, so the live thread and the worker never share one."""
        vg, ug = self._last_assets
        av = Ubody_Gaussian(self.cfg.MODEL, clone_assets(vg), clone_assets(ug), pruning=True); av.init_ehm(self.inf.ehm); av.eval()
        return av

    # -------------------------------------------------------------------------------------------- rendering / scores (default stream)
    def target(self, rec):
        comp = F.compact_record(rec, self.shp); sm = {k: torch.as_tensor(v.copy()) for k, v in comp["smplx"].items()}
        sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
        M4 = torch.eye(4); M4[:3, :4] = sm["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
        t = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c),
             "invtanfov": comp["invtanfov"]}
        t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
        return F.to_dev(F.collate1(t), self.dev)

    @torch.no_grad()
    def render(self, av, rec):
        t = self.target(rec); rr = self.wren(av(t), t["render_cam_params"], bg=0.0)
        return rr["renders"][0:1].clamp(0, 1), rr["depths"][0, 0]

    def _img512(self, crop_u8, alpha):
        x = torch.from_numpy(np.ascontiguousarray(crop_u8)).to(self.dev).permute(2, 0, 1)[None].float() / 255.0
        a = torch.from_numpy(np.asarray(alpha, np.float32)).to(self.dev)[None, None]
        x = Fn.interpolate(x, (512, 512), mode="bilinear", antialias=True, align_corners=False)
        a = Fn.interpolate(a, (512, 512), mode="bilinear", antialias=True, align_corners=False).clamp(0, 1)
        return x * a, a[0, 0]

    def _arc(self, img_a):
        img = img_a[0] if isinstance(img_a, tuple) else img_a
        th = AF.align_theta(self.kps5, img.shape[-1])[None]
        return self.arc(self.arc.preprocess(AF.crops(img, th)))[0]

    def _arc_sim(self, av):
        rgb, _ = self.render(av, self.rs["record"]); return float((self._arc(rgb) * self.arc_ref).sum())

    def lpips_map(self, x, y):
        fx, fy = self.lp.net(x), self.lp.net(y)
        return sum(Fn.interpolate(l((a - b) ** 2), size=x.shape[-2:], mode="bilinear", align_corners=False) for a, b, l in zip(fx, fy, self.lp.lin))[0, 0]

    def _face_crop(self, img, rec):
        M = np.asarray(rec["head_crop"]["M_o2c"], np.float64)
        return cv2.warpAffine(img, M[:2], (self.saf.PART, self.saf.PART), flags=cv2.INTER_LINEAR)

    # -------------------------------------------------------------------------------------------- held-out reservoir / stop rule
    @torch.no_grad()
    def add_heldout(self, i, r):
        img, alpha = self._img512(r["crop"], r["alpha"])
        _, p2f, _ = self.face_visibility(self.pose_batch(r["record"]["smplx_coeffs"]))
        fl = p2f[0, :, :, 0].long(); lab = torch.where(fl >= 0, self.freg[fl.clamp_min(0) % self.nF], torch.full_like(fl, -1))
        reg = torch.stack([(lab == k) & (alpha > 0.5) for k in range(4)])
        self.res.append({"i": i, "img": img, "rec": r["record"], "reg": reg}); self.res = self.res[-self.K:]

    @torch.no_grad()
    def region_lpips(self, av, ver):
        vals = [[] for _ in range(4)]
        for e in self.res:
            key = (ver, e["i"])
            if key not in self.lp_cache:
                rgb, _ = self.render(av, e["rec"]); m = self.lpips_map(rgb, e["img"])
                self.lp_cache[key] = [float(m[e["reg"][k]].mean()) if int(e["reg"][k].sum()) >= 1000 else None for k in range(4)]
            for k, v in enumerate(self.lp_cache[key]):
                if v is not None: vals[k].append(v)
        return [float(np.mean(v)) if v else None for v in vals]

    def coverage(self):
        cov = []
        for k in range(4):
            s = self.seen & (self.treg == k); n = int(s.sum())
            cov.append(float((s & (self.W > 0)).sum()) / n if n else 0.0)
        return cov

    def check_reopen(self, lum):
        out = []
        for k in range(4):
            if self.stop[k] and self.lum_stop[k] and np.isfinite(lum) and abs(lum - self.lum_stop[k]) / self.lum_stop[k] > 0.15:
                self.stop[k] = False; self.hist[k] = []; out.append(REGIONS[k])
        return out

    # -------------------------------------------------------------------------------------------- one accepted (well-fitted) frame
    @torch.no_grad()
    def process(self, i, img, r, work_stream):
        """returns (log dict, new live avatar or None). Fusion / decode on the worker stream; renders on the default stream
        (the Gaussian rasterizer launches its kernels on the legacy default stream)."""
        T = {}; t0 = time.perf_counter(); ent = {}
        lum = luminance(r["crop"], r["alpha"]); ent["lum"] = lum; ent["reopened"] = self.check_reopen(lum)
        fitq = float(np.exp(-r["diag"]["kp_err_px1024_after"] / 20.0)); sharp = min(1.0, lapvar(r["crop"]) / max(self.lap0, 1e-9))
        ent.update(fitq=fitq, sharp=sharp)
        with torch.cuda.stream(work_stream):
            f = self.frame_feats(self.batch_from_fit(r))
            self.seen |= f["vis"]
        work_stream.synchronize(); T["feats"] = time.perf_counter() - t0; t1 = time.perf_counter()
        # outlier frame: render-vs-image LPIPS under the current avatar
        rgb, _ = self.render(self.av_work, r["record"]); img512, _ = self._img512(r["crop"], r["alpha"])
        lp = float(self.lp(rgb, img512).mean()); ent["lpips_cur"] = lp
        if len(self.lp_acc) >= 5:
            med = float(np.median(self.lp_acc)); mad = float(np.median(np.abs(np.array(self.lp_acc) - med)))
            if lp > med + 3 * mad:
                ent["gate"] = "outlier_frame"; T["check"] = time.perf_counter() - t1; ent["ms"] = {k: v * 1000 for k, v in T.items()}; return ent, None
        T["outlier"] = time.perf_counter() - t1; t1 = time.perf_counter()
        # face rule (head region)
        yaw, pitch = head_angles(f["deform"], f["w2c"]); fe = face_kp_err(r, self.dyn, self.saf); fl = lapvar(self._face_crop(img, r["record"]))
        face_ok = yaw <= 30.0 and pitch <= 20.0 and fl >= 0.8 * self.face_lap0 and fe <= 10.0
        ent.update(yaw=yaw, pitch=pitch, face_kp_err=fe, face_lap_ratio=fl / max(self.face_lap0, 1e-9), face_ok=face_ok)
        with torch.cuda.stream(work_stream):
            w = f["facing"] * (fitq * sharp)
            fused_rgb = self.S[:3] / self.W.clamp_min(1e-6)
            out = ((f["uv"][:3] - fused_rgb).abs().mean(0) > self.out_rgb) & (self.W >= 1.0)
            w = torch.where(out, torch.zeros_like(w), w)
            open_t = torch.zeros_like(self.umask)
            for k in range(4):
                if not self.stop[k] and (k != 0 or face_ok): open_t |= self.treg == k
            upd = (w >= self.thr) & (self.W < self.cap) & open_t & self.umask
            S0, W0, T0 = self.S.clone(), self.W.clone(), self.touched.clone()
            wu = torch.where(upd, w, torch.zeros_like(w))
            self.S += f["uv"] * wu; self.W += wu; self.touched |= upd
            n_upd = int(upd.sum()); ent.update(n_texels=n_upd, n_outlier_texels=int(out.sum()))
            if self.tmpl_avg:
                Sv0, Wv0, VT0 = self.Sv.clone(), self.Wv.clone(), self.vtouched.clone()
                wv = f["vfacing"] * (fitq * sharp)
                open_v = torch.zeros(self.nV, dtype=torch.bool, device=self.dev)
                for k in range(4):
                    if not self.stop[k] and (k != 0 or face_ok): open_v |= self.vreg == k
                uv_ = (wv >= self.thr) & (self.Wv < self.cap) & open_v
                wvu = torch.where(uv_, wv, torch.zeros_like(wv)); self.Sv += f["vfeat"] * wvu[:, None]; self.Wv += wvu; self.vtouched |= uv_
                ent["n_vertices"] = int(uv_.sum())
        work_stream.synchronize(); T["fuse"] = time.perf_counter() - t1; t1 = time.perf_counter()
        if n_upd == 0 and not (self.tmpl_avg and ent.get("n_vertices", 0) > 0):
            ent["gate"] = "no_update"; self.lp_acc.append(lp); ent["ms"] = {k: v * 1000 for k, v in T.items()}; return ent, None
        with torch.cuda.stream(work_stream):
            new_work = self.decode()
        work_stream.synchronize(); T["decode"] = time.perf_counter() - t1; t1 = time.perf_counter()
        # ArcFace rollback
        if self.arc_ref is not None:
            sim = self._arc_sim(new_work); ent["arc_sim"] = sim; ent["arc_sim_prev"] = self.arc_sim
            if sim < self.arc_sim - 0.01:
                self.S, self.W, self.touched = S0, W0, T0
                if self.tmpl_avg: self.Sv, self.Wv, self.vtouched = Sv0, Wv0, VT0
                self.n_rollback += 1; ent["gate"] = "rollback"; T["arcface"] = time.perf_counter() - t1
                ent["ms"] = {k: v * 1000 for k, v in T.items()}; return ent, None
        T["arcface"] = time.perf_counter() - t1; t1 = time.perf_counter()
        # held-out gain per region (previous vs new version on the current reservoir) -> stop rule
        prev = self.region_lpips(self.av_work, self.ver); newv = self.ver + 1
        new = self.region_lpips(new_work, newv); cov = self.coverage(); ent.update(coverage=cov, heldout_prev=prev, heldout_new=new, heldout_n=len(self.res))
        stopped = []
        for k in range(4):
            if prev[k] is not None and new[k] is not None: self.hist[k].append(prev[k] - new[k])
            if not self.stop[k] and cov[k] >= 0.9 and len(self.hist[k]) >= 3 and all(g < 0.002 for g in self.hist[k][-3:]):
                self.stop[k] = True; self.lum_stop[k] = lum; stopped.append(REGIONS[k])
        ent["stopped"] = stopped; ent["stop_state"] = [REGIONS[k] for k in range(4) if self.stop[k]]
        self.lp_cache = {k: v for k, v in self.lp_cache.items() if k[0] == newv}
        T["heldout"] = time.perf_counter() - t1
        if self.arc_ref is not None: self.arc_sim = ent["arc_sim"]
        self.av_work = new_work; self.ver = newv; self.lp_acc.append(lp)
        live = self.live_copy()
        ent["gate"] = "accepted"; ent["version"] = self.ver; ent["ms"] = {k: v * 1000 for k, v in T.items()}
        return ent, live

    def all_stopped(self):
        return all(self.stop.values())

