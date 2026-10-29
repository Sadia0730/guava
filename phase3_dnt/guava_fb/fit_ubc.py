"""GUAVA-FB, UBC-Fashion converter (PREREG "GUAVA-FB"; format phase3_dnt/guava_fb/FORMAT.md).

Per official TRAIN video (GART's 6 and identity matches of evaluation videos skipped, see ubc_identity.py):
  frames: every `stride`-th frame, stride = max(4, ceil(n / cap)) (cap 100) -> <= 100 uniformly spaced frames.
  1. DWPose (SprintFitter.dwpose, CPU onnxruntime; largest YOLOX person box), in a prefetch thread (next video).
  2. EHM body crop (square 1.1 x long side, 1024) -> SPRINT (from scratch EMA) at 256 px with cam_feat(focal = W, real 0)
     (setting ii); pass 1 per frame, then pass 2 with the shape locked to the median of pass-1 shapes (cond shape_on).
  3. PERSPECTIVE crop: virtual camera (f = W, principal point = image centre) rotated by the minimal rotation R_v onto the
     ray through the person-box centre; homography K' R_v K^-1 warp, side S = 1.1 x long box side, 1024 px, f' = 1024 f / S,
     invtanfov = 2 f / S. SPRINT's root / translation (CLIFF-style: crop camera focal 24 -> full camera f = W) are moved into
     the virtual camera: global <- R_v global, T_v = R_v (J0 + T_full) - J0 (rotation about the pelvis J0).
  4. StyleMatte on the 1024 perspective crop.
  5. JOINT video refinement (all frames of the video batched on the GPU, Adam): ONE shared shape (200 betas = median of
     SPRINT shapes + shared offset on the first 20, regularised), per-frame residual rotations (root, 21 body, 2 x 15 hands,
     jaw; axis-angle right-multiplied), exp offsets, translation (x, y, log z). Data terms as sprint_avatar_fit (DWPose
     Geman-McClure sigma 3 %, feet on; StyleMatte mask in / coverage), priors to SPRINT, plus temporal smoothing:
     second differences of the final rotations (6D) and translations over consecutive sampled frames.
  6. Alignment filter (registered): (a) projected mesh pixels outside the matte (alpha < 0.5) < 5 % of mesh pixels;
     (b) max per-limb error < 3 % of body height (DWPose person-box height in the crop): limbs head (nose, eyes), arms
     (shoulder, elbow, wrist), legs (hip, knee, ankle, big / small toe, heel); limb error = mean pixel error of its keypoints
     with DWPose score > 0.5; and >= 15 such keypoints among these 21;
     video kept iff >= 80 % of its frames pass (index step, make_index_ubc.py).
  7. Write tracked clip O/data/ubc/<clip>/ (no gt_proj.npz: no exact labels); record invtanfov, view "0". The "original"
     image of body_crop / head / hand matrices is the 1024 perspective crop itself (a homography cannot go through GUAVA's
     affine box product): body_crop M_o2c-hd = I, M_o2c = 224/1024; head / hand crops from the projected fitted SMPL-X head /
     hand vertices; the original-image homography is kept in record["persp_crop"].
Diagnostics: O/runs/ubc/clips/<clip>/{fit.json, frames.npz, overlay_*.jpg}.
Usage: python fit_ubc.py --videos a b c | --list file [--shard i --nshard n] --device cuda:0"""
from __future__ import annotations

import argparse, json, math, os, pickle, queue, sys, threading, time, traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sprint_avatar_fit as SAF                                     # noqa: E402

D = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/ubc_fashion")
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
CKPT = "/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt"
CROP, PART = 1024, 512
DIAG = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
FILT_IDX = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23])   # openpose134 body + feet (no neck / ears)
# limbs for the filter's "max per-limb error" (openpose134: 18-20 left big toe / small toe / heel, 21-23 right)
LIMBS = {"head": [0, 14, 15], "r_arm": [2, 3, 4], "l_arm": [5, 6, 7], "r_leg": [8, 9, 10, 21, 22, 23], "l_leg": [11, 12, 13, 18, 19, 20]}
MR = 512                                                           # matte resolution of the mask terms
WTS = {"kp": 1.0, "mask_in": 0.3, "mask_out": float(os.environ.get("UBC_W_MASK_OUT", 1.0)), "mask_cov": 0.2, "prior": 0.05, "sm_rot": 1.0, "sm_hand": 0.1, "sm_t": 1.0}


# ---------------------------------------------------------------- geometry helpers
def rot_to_z(d):
    """Minimal rotation R with R d = e_z (d unit)."""
    z = np.array([0.0, 0.0, 1.0]); ax = np.cross(d, z); s = np.linalg.norm(ax); c = float(d @ z)
    if s < 1e-12:
        return np.eye(3)
    k = ax / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def persp_crop(box, W, H, f):
    l, t, r, b = [float(v) for v in box]
    cx, cy = (l + r) / 2, (t + b) / 2; S = 1.1 * max(r - l, b - t)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    d = np.linalg.inv(K) @ np.array([cx, cy, 1.0]); d /= np.linalg.norm(d)
    Rv = rot_to_z(d); fp = f * CROP / S
    Kp = np.array([[fp, 0, CROP / 2], [0, fp, CROP / 2], [0, 0, 1.0]])
    Hm = Kp @ Rv @ np.linalg.inv(K)
    return Hm, Rv, 2 * f / S, S


def hom(Hm, pts):
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


def rot6(R):
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def lean_batch(L, rots, beta, exp):
    """Batched LeanSMPLX: rots (B, 55, 3, 3), beta (200,) shared, exp (B, 50) -> verts (B, V, 3), dwpose pts (B, 134, 3)."""
    B = rots.shape[0]; dev = rots.device
    v_sh = L.v_template + torch.einsum("vcs,s->vc", L.shapedirs[:, :, :beta.shape[0]], beta)
    v_sh = v_sh[None] + torch.einsum("vcs,bs->bvc", L.shapedirs[:, :, L.n_shape:L.n_shape + 50], exp)
    J = torch.einsum("jv,bvc->bjc", L.J_regressor, v_sh)
    eye3 = torch.eye(3, device=dev)
    v_po = v_sh + ((rots[:, 1:] - eye3).reshape(B, -1) @ L.posedirs).reshape(B, -1, 3)
    rel = J - J[:, L.parents] * L.nonroot[None]
    Tl = torch.cat([torch.cat([rots, rel[..., None]], 3), L.hrow.expand(B, 55, 1, 4)], 2)
    Tw = Tl; bi = torch.arange(B, device=dev)[:, None]
    for lev in L.levels:
        Tw = Tw.index_put((bi, lev[None]), Tw[:, L.parents[lev]] @ Tl[:, lev])
    Jt = Tw[:, :, :3, 3]
    A = torch.cat([Tw[:, :, :3, :3], (Jt - (Tw[:, :, :3, :3] @ J[..., None])[..., 0])[..., None]], 3)
    Tv = torch.einsum("vj,bjk->bvk", L.lbs_weights, A.reshape(B, 55, 12)).reshape(B, -1, 3, 4)
    verts = (Tv[..., :3] @ v_po[..., None])[..., 0] + Tv[..., 3]
    P = torch.einsum("kv,bvc->bkc", L.Wv, verts) + torch.einsum("kj,bjc->bkc", L.Sj, Jt)
    P = torch.cat([P[:, :1], torch.cat([(P[:, 2:3, :2] + P[:, 5:6, :2]) / 2, P[:, 1:2, 2:]], 2), P[:, 2:]], 1)
    return verts, P, J[:, 0]


def proj(P, inv):
    """OpenCV-aligned camera points (B, N, 3), invtanfov (B,) -> normalised crop coords [0, 1]."""
    return 0.5 + 0.5 * inv[:, None, None] * P[..., :2] / P[..., 2:3].clamp_min(1e-3)


# ---------------------------------------------------------------- registered alignment filter (shared by all real-video sources)
OUT_FRAC_MAX, KP_ERR_MAX, KP_MIN, KP_SCORE = 0.05, 0.03, 15, 0.5
CLIP_PASS = 0.8


@torch.no_grad()
def silhouettes(vimg, depth, faces, size=512, chunk=25):
    """Mesh silhouettes (B, size, size) uint8 with pytorch3d's rasteriser. vimg (B, V, 2) normalised crop coords [0, 1],
    depth (B, V) > 0, faces (F, 3) long tensor on the same device. (A single cv2.fillPoly over all triangles is NOT a
    silhouette: overlapping triangles cancel under its even-odd rule.)"""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    out = []
    for i in range(0, vimg.shape[0], chunk):
        v = torch.cat([1 - 2 * vimg[i:i + chunk], depth[i:i + chunk, :, None]], -1)
        p2f = rasterize_meshes(Meshes(verts=list(v), faces=[faces] * v.shape[0]), image_size=size, blur_radius=0.0, faces_per_pixel=1, bin_size=0)[0]
        out.append((p2f[..., 0] >= 0).to(torch.uint8).cpu().numpy())
    return np.concatenate(out, 0)


def frame_filter(sil, alpha, uv134, tgt, sc, box, Hm):
    """Registered per-frame filter. sil (512, 512) mesh silhouette; alpha (1024, 1024) StyleMatte; uv134 / tgt (134, 2)
    fitted-model DWPose-layout points / DWPose keypoints in normalised crop coords; sc (134,) scores; box person box
    (original px); Hm original -> 1024 crop homography.
    (a) mesh pixels outside the matte (alpha < 0.5) < 5 % of mesh pixels; (b) >= 15 of the 21 FILT_IDX keypoints with score
    > 0.5 and max per-limb error < 3 % of body height (limb error = mean error of its valid keypoints, LIMBS; body height =
    DWPose box height mapped into the crop)."""
    a512 = cv2.resize(alpha, (512, 512), interpolation=cv2.INTER_AREA)
    ns = int(sil.sum()); out_frac = float((sil.astype(bool) & (a512 < 0.5)).sum() / max(ns, 1))
    cx = (box[0] + box[2]) / 2; ends = hom(Hm, np.array([[cx, box[1]], [cx, box[3]]])); bh = float(np.linalg.norm(ends[1] - ends[0]))
    good = FILT_IDX[sc[FILT_IDX] > KP_SCORE]
    err = np.linalg.norm(uv134[good] - tgt[good], axis=1) * CROP / bh if len(good) else np.zeros(0)
    ek = dict(zip(good.tolist(), err.tolist()))
    limb = {ln: float(np.mean([ek[k] for k in ks if k in ek])) for ln, ks in LIMBS.items() if any(k in ek for k in ks)}
    maxerr = max(limb.values()) if limb else float("inf")
    maxkp = float(err.max()) if len(good) else float("inf")
    each = [round(ek[k], 4) if k in ek else None for k in FILT_IDX.tolist()]
    pa = out_frac < OUT_FRAC_MAX; pb = len(good) >= KP_MIN and maxerr < KP_ERR_MAX
    return {"mesh_out_frac": out_frac, "mesh_px512": ns, "n_kp05": int(len(good)), "max_kp_err_rel": maxerr, "max_single_kp_err_rel": maxkp,
            "limb_err_rel": {k: round(v, 4) for k, v in limb.items()}, "body_h_px1024": bh, "pass_a": bool(pa), "pass_b": bool(pb),
            "pass": bool(pa and pb), "kp_err_rel_each": each}


# ---------------------------------------------------------------- per-video CPU stage (prefetch thread)
def cpu_stage(fitter, name, stride_min, cap):
    cap_ = cv2.VideoCapture(str(D / "train" / f"{name}.mp4")); n = int(cap_.get(7)); W = int(cap_.get(3)); H = int(cap_.get(4))
    stride = max(stride_min, math.ceil(n / cap)); want = set(range(0, n, stride))
    frames, t0, i, fails = [], time.perf_counter(), 0, []
    while True:
        ok, fr = cap_.read()
        if not ok:
            break
        if i in want:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            try:
                box, kp, sc = fitter.dwpose(rgb)
                frames.append({"idx": i, "rgb": rgb, "box": box, "kp": kp, "sc": sc})
            except Exception as e:  # noqa: BLE001
                fails.append([i, repr(e)])
        i += 1
    return {"name": name, "n": n, "W": W, "H": H, "stride": stride, "frames": frames, "dw_fail": fails, "t_dwpose": time.perf_counter() - t0}


# ---------------------------------------------------------------- GPU stage
class VideoFit:
    def __init__(self, fitter, iters):
        self.f = fitter; self.dev = fitter.dev; self.iters = iters; L = fitter.lean
        lw = L.lbs_weights
        self.head_v = torch.nonzero(lw[:, [15, 22, 23, 24]].sum(1) > 0.5)[:, 0]
        self.lh_v = torch.nonzero(lw[:, 25:40].sum(1) > 0.5)[:, 0]; self.rh_v = torch.nonzero(lw[:, 40:55].sum(1) > 0.5)[:, 0]
        self.faces = fitter.layer.faces_tensor.long().cpu().numpy()
        w = SAF.kp_weights(); w[18:24] = 0.5; w[[20, 23]] = 0.1       # full body: feet on; heels low (high heels / SMPL-X heel definition)
        self.w_kp = torch.from_numpy(w).to(self.dev) * (~L.dyn).float()

    def silhouettes(self, vimg, depth):
        return silhouettes(vimg, depth, torch.from_numpy(self.faces).to(self.dev))

    @torch.no_grad()
    def sprint_all(self, V):
        f = self.f; W, H = V["W"], V["H"]; out = []
        for fr in V["frames"]:
            M = SAF.box_crop_matrix(fr["box"], CROP)
            crop = cv2.warpAffine(fr["rgb"], M[:2], (CROP, CROP), flags=cv2.INTER_LINEAR)
            ct = torch.from_numpy(crop).to(self.dev).permute(2, 0, 1)[None].float() / 255.0
            fr["c256"] = F.interpolate(ct, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
            A256 = (np.diag([256 / CROP, 256 / CROP, 1.0]) @ M)[:2]
            fr["cam"] = f.ev.cam_feat(A256, W, H, W, 0.0); fr["size_aff"] = int(max(fr["box"][2] - fr["box"][0], fr["box"][3] - fr["box"][1]) * 1.1 + 1e-7)
        def run(shape):
            res = []
            for fr in V["frames"]:
                cond = {"cam": torch.as_tensor(fr["cam"], device=self.dev).float().reshape(1, 5)}
                if shape is not None:
                    cond["shape"] = shape[None]; cond["shape_on"] = torch.ones(1, device=self.dev)
                o = f.model(fr["c256"], cond=cond)
                res.append({"bp": {k: v.float() for k, v in o["body_param"].items() if torch.is_tensor(v)}, "cam": o["pd_cam"][0].float()})
            return res
        p1 = run(None)
        shape_med = torch.stack([r["bp"]["shape"][0, :200] for r in p1]).median(0).values
        p2 = run(shape_med)
        return p1, p2, shape_med

    def fit(self, V):
        f = self.f; dev = self.dev; T = {}; L = f.lean
        W, H = V["W"], V["H"]; fl = float(W)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        p1, p2, shape_med = self.sprint_all(V)
        torch.cuda.synchronize(); T["sprint"] = time.perf_counter() - t0; t0 = time.perf_counter()
        B = len(V["frames"])
        # perspective crops, matting, targets
        inv, Rv_l, Hm_l, crops, alphas, tgt, conf, dts, mpts, mw = [], [], [], [], [], [], [], [], [], []
        rng = np.random.default_rng(0); NM = 1000
        for fr in V["frames"]:
            Hm, Rv, iv, S = persp_crop(fr["box"], W, H, fl)
            crop = cv2.warpPerspective(fr["rgb"], Hm, (CROP, CROP), flags=cv2.INTER_LINEAR)
            ct = torch.from_numpy(crop).to(dev).permute(2, 0, 1)[None].float() / 255.0
            with torch.no_grad():
                a = f.matting(ct).clamp(0, 1)
            a_small = F.interpolate(a[None, None], size=(MR, MR), mode="bilinear", antialias=True, align_corners=False)[0, 0]
            m_np = (a_small > 0.5).cpu().numpy().astype(np.uint8)
            dts.append(cv2.distanceTransform(1 - m_np, cv2.DIST_L2, 5) / MR)
            ys, xs = np.nonzero(m_np); sel = rng.choice(len(xs), min(NM, len(xs)), replace=False) if len(xs) else np.zeros(0, int)
            mp = np.zeros((NM, 2), np.float32); w_ = np.zeros(NM, np.float32)
            mp[:len(sel)] = np.stack([(xs[sel] + 0.5) / MR, (ys[sel] + 0.5) / MR], 1); w_[:len(sel)] = 1
            mpts.append(mp); mw.append(w_)
            kpc = hom(Hm, fr["kp"]) / CROP
            ok = (fr["sc"] >= SAF.KP_THR) & (kpc >= 0).all(1) & (kpc <= 1).all(1)
            tgt.append(kpc); conf.append(np.where(ok, fr["sc"], 0.0))
            inv.append(iv); Rv_l.append(Rv); Hm_l.append(Hm); crops.append(crop); alphas.append(a.cpu().numpy())
            fr["S"] = S
        tt = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(dev)
        inv_t, Rv_t, tgt_t = tt(inv), tt(np.stack(Rv_l)), tt(np.stack(tgt))
        conf_t = tt(np.stack(conf)) * self.w_kp[None]
        dt_t = tt(np.stack(dts))[:, None]; mp_t, mw_t = tt(np.stack(mpts)), tt(np.stack(mw))
        torch.cuda.synchronize(); T["persp_matting"] = time.perf_counter() - t0; t0 = time.perf_counter()
        # SPRINT (pass 2) -> virtual-camera init
        cat = lambda k: torch.cat([r["bp"][k] for r in p2], 0)
        g0, b0, l0, r0, j0 = cat("global_pose"), cat("body_pose"), cat("left_hand_pose"), cat("right_hand_pose"), cat("jaw_pose")
        exp0 = cat("exp")[:, :50]
        Rc = torch.stack([r["cam"][:3, :3] for r in p2]); tc = torch.stack([r["cam"][:3, 3] for r in p2])
        Dg = tt(DIAG)
        assert float((Rc - Dg).abs().max()) < 1e-4, "SPRINT crop camera R is not diag(-1, -1, 1)"
        with torch.no_grad():
            rots0 = torch.cat([g0, b0, j0, torch.eye(3, device=dev).expand(B, 2, 3, 3), l0, r0], 1)
            _, _, J0 = lean_batch(L, rots0, shape_med, exp0)
            Tc = tc @ Dg                                                     # OpenCV translation in SPRINT's crop camera
            size_aff = tt([fr["size_aff"] for fr in V["frames"]]); cxy = tt([[(fr["box"][0] + fr["box"][2]) / 2 - W / 2, (fr["box"][1] + fr["box"][3]) / 2 - H / 2] for fr in V["frames"]])
            tz = Tc[:, 2] * fl / (12.0 * size_aff)
            Tfull = torch.cat([Tc[:, :2] + cxy / fl * tz[:, None], tz[:, None]], 1)
            Tv = (Rv_t @ (J0 + Tfull)[..., None])[..., 0] - J0
            gv = Rv_t[:, None] @ g0
        # ---- joint refinement
        z = lambda *s: torch.zeros(*s, device=dev, requires_grad=True)
        dg, db, dl, dr, dj, dexp = z(B, 1, 3), z(B, 21, 3), z(B, 15, 3), z(B, 15, 3), z(B, 1, 3), z(B, 50)
        dbeta = z(20); txy = Tv[:, :2].clone().requires_grad_(True); lz = torch.log(Tv[:, 2:3].clamp_min(0.1)).clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [dg, db, dj], "lr": 0.02}, {"params": [dl, dr], "lr": 0.02}, {"params": [dexp], "lr": 0.05},
                                {"params": [txy, lz], "lr": 0.01}, {"params": [dbeta], "lr": 0.05}])
        eye2 = torch.eye(3, device=dev).expand(B, 2, 3, 3); s2 = 0.03 ** 2
        def build():
            rots = torch.cat([gv @ SAF.rodrigues(dg), b0 @ SAF.rodrigues(db), j0 @ SAF.rodrigues(dj), eye2, l0 @ SAF.rodrigues(dl), r0 @ SAF.rodrigues(dr)], 1)
            beta = shape_med + torch.cat([dbeta, torch.zeros(180, device=dev)])
            return rots, beta, exp0 + dexp, torch.cat([txy, torch.exp(lz)], 1)
        def losses(rots, beta, exp, Tt):
            Vv, K, _ = lean_batch(L, rots, beta, exp)
            uv = proj(K + Tt[:, None], inv_t); r2 = ((uv - tgt_t) ** 2).sum(-1)
            Ls = {"kp": ((conf_t * SAF.gm(r2, s2)).sum(1) / conf_t.sum(1).clamp_min(1e-6)).mean()}
            v = proj(Vv[:, ::3] + Tt[:, None], inv_t)
            inside = ((v >= 0) & (v <= 1)).all(-1).float()
            d = F.grid_sample(dt_t, (v * 2 - 1)[:, None], mode="bilinear", align_corners=False)[:, 0, 0]
            Ls["mask_in"] = ((inside * SAF.gm(d ** 2, s2)).sum(1) / inside.sum(1).clamp_min(1.0)).mean()
            if WTS.get("mask_out", 0) > 0:      # tight silhouette term: L1 distance (in units of 1 % of the crop) outside the matte
                d2 = F.grid_sample(dt_t, (proj(Vv + Tt[:, None], inv_t) * 2 - 1)[:, None], mode="bilinear", align_corners=False)[:, 0, 0]
                Ls["mask_out"] = (d2 / 0.01).mean()
            with torch.no_grad():
                idx = torch.cdist(mp_t, v[:, ::2]).argmin(2)
            vn = torch.gather(v[:, ::2], 1, idx[..., None].expand(-1, -1, 2))
            Ls["mask_cov"] = ((mw_t * SAF.gm(((mp_t - vn) ** 2).sum(-1), s2)).sum(1) / mw_t.sum(1).clamp_min(1.0)).mean()
            Ls["prior"] = (0.5 * (dg ** 2).sum((1, 2)) + 4.0 / 21 * 0.5 * (db ** 2).sum((1, 2)) + 0.05 / 15 * ((dl ** 2).sum((1, 2)) + (dr ** 2).sum((1, 2)))
                           + 1.0 * (dj ** 2).sum((1, 2)) + 0.02 * (dexp ** 2).sum(1)).mean() + 0.05 * (dbeta ** 2).sum()
            if B >= 3:
                s6 = rot6(rots); d2 = s6[2:] - 2 * s6[1:-1] + s6[:-2]
                Ls["sm_rot"] = (d2[:, :22] ** 2).sum(-1).mean()
                Ls["sm_hand"] = (d2[:, 25:] ** 2).sum(-1).mean()
                t2 = Tt[2:] - 2 * Tt[1:-1] + Tt[:-2]
                Ls["sm_t"] = ((t2[:, :2] ** 2).sum(-1) / 0.02 ** 2 + (t2[:, 2] ** 2) / 0.1 ** 2).mean() * 0.01
            return Ls, uv, Vv
        with torch.no_grad():
            Lb, uvb, _ = losses(*build())
        for it in range(self.iters):
            opt.zero_grad(set_to_none=True)
            Ls, _, _ = losses(*build())
            sum(WTS[k] * v for k, v in Ls.items()).backward(); opt.step()
        with torch.no_grad():
            rots, beta, exp, Tt = build(); La, uva, Vv = losses(rots, beta, exp, Tt)
            vimg = proj(Vv + Tt[:, None], inv_t)                           # (B, V, 2) normalised
        torch.cuda.synchronize(); T["refine"] = time.perf_counter() - t0; t0 = time.perf_counter()
        pxe = lambda uv: float(((((uv - tgt_t).norm(dim=-1) * CROP) * (conf_t > 0)).sum() / (conf_t > 0).sum().clamp_min(1)))
        diag = {"loss_before": {k: float(v) for k, v in Lb.items()}, "loss_after": {k: float(v) for k, v in La.items()},
                "kp_err_px1024_before": pxe(uvb), "kp_err_px1024_after": pxe(uva)}
        # ---- alignment filter
        vimg_np = vimg.cpu().numpy(); uva_np = uva.cpu().numpy()
        sils = self.silhouettes(vimg, (Vv + Tt[:, None])[..., 2])
        filt = []
        for b, fr in enumerate(V["frames"]):
            r = frame_filter(sils[b], alphas[b], uva_np[b], tgt[b], fr["sc"], fr["box"], Hm_l[b])
            if b in getattr(self, "debug_frames", ()):
                a512 = cv2.resize(alphas[b], (512, 512), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(self.debug_dir / f"sil_{fr['idx']:06d}.png"), np.stack([sils[b] * 255, (a512 >= 0.5).astype(np.uint8) * 255, np.zeros_like(sils[b])], -1))
            Rg = rots[b, 0].cpu().numpy(); facing = float(-(Rg @ np.array([0, 0, 1.0]))[2])    # > 0: body faces the camera
            filt.append({"key": f"frame_{fr['idx']:06d}", **r, "facing": facing, "invtanfov": float(inv[b])})
        T["filter"] = time.perf_counter() - t0
        return {"rots": rots, "beta": beta, "exp": exp, "Tt": Tt, "inv": inv, "Hm": Hm_l, "Rv": Rv_l, "crops": crops, "alphas": alphas,
                "vimg": vimg_np, "uv": uva_np, "tgt": tgt, "conf": conf, "filt": filt, "diag": diag, "times": T, "shape_med": shape_med,
                "sprint_shape_std": float(torch.stack([r["bp"]["shape"][0, :200] for r in p1]).std(0)[:10].mean())}


# ---------------------------------------------------------------- writer
def write_clip(out_dir, name, V, R, vf):
    import lmdb, roma, torchvision
    out_dir.mkdir(parents=True, exist_ok=True)
    aa = lambda Rm: roma.rotmat_to_rotvec(Rm.reshape(-1, 3, 3)).cpu().numpy().astype(np.float32)
    rots = R["rots"]; recs, keys = {}, []
    z50 = np.zeros(50, np.float32)
    Mlr = np.diag([SAF.BODY_LR / CROP, SAF.BODY_LR / CROP, 1.0]).astype(np.float32); I3 = np.eye(3, dtype=np.float32)
    for b, fr in enumerate(V["frames"]):
        key = f"frame_{fr['idx']:06d}"; keys.append(key)
        t = (DIAG @ R["Tt"][b].cpu().numpy()).astype(np.float32)
        RT = np.concatenate([DIAG, t[:, None]], 1).astype(np.float32)
        smplx = {"global_pose": aa(rots[b, 0])[0], "body_pose": aa(rots[b, 1:22]), "jaw_pose": aa(rots[b, 22]).reshape(1, 3),
                 "left_hand_pose": aa(rots[b, 25:40]), "right_hand_pose": aa(rots[b, 40:55]), "eye_pose": np.zeros((2, 3), np.float32),
                 "exp": R["exp"][b].cpu().numpy().astype(np.float32), "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}
        vi = R["vimg"][b] * CROP
        def part(vidx, sl, sc):
            M = SAF.box_crop_matrix(SAF.landmark_box(vi[vidx], sl), PART, sc)
            return {"M_o2c": M, "M_c2o": np.linalg.inv(M).astype(np.float32)}
        crops = {"head_crop": part(vf.head_v.cpu().numpy(), 1.1, 1.3), "left_hand_crop": part(vf.lh_v.cpu().numpy(), 1.5, 1.3),
                 "right_hand_crop": part(vf.rh_v.cpu().numpy(), 1.5, 1.3)}
        recs[key] = {"smplx_coeffs": smplx,
                     "flame_coeffs": {"expression_params": z50, "jaw_params": np.zeros(3, np.float32), "pose_params": np.zeros(3, np.float32),
                                      "neck_pose_params": np.zeros(3, np.float32), "eye_pose_params": np.zeros(6, np.float32),
                                      "eyelid_params": np.zeros(2, np.float32), "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
                     "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3.copy(), "M_c2o-hd": I3.copy()},
                     "persp_crop": {"H_o2c": R["Hm"][b].astype(np.float32), "R_virtual": R["Rv"][b].astype(np.float32), "focal_px_orig": float(V["W"]),
                                    "orig_wh": [V["W"], V["H"]], "frame_index": int(fr["idx"])},
                     "dwpose_raw": {"keypoints": fr["kp"].astype(np.float32), "scores": fr["sc"].astype(np.float32), "bbox": np.asarray(fr["box"], np.float32),
                                    "keypoints_crop1024": (R["tgt"][b] * CROP).astype(np.float32)},
                     "invtanfov": float(R["inv"][b]), "view": "0", **crops}
    with open(out_dir / "optim_tracking_ehm.pkl", "wb") as fh:
        pickle.dump(recs, fh)
    with open(out_dir / "id_share_params.pkl", "wb") as fh:
        pickle.dump(SAF.id_share(R["beta"].cpu().numpy()), fh)
    (out_dir / "videos_info.json").write_text(json.dumps({name: {"frames_num": len(keys), "frames_keys": keys}}, indent=1))
    env = lmdb.open(str(out_dir / "img_lmdb"), map_size=1 << 32); txn = env.begin(write=True)
    for b, key in enumerate(keys):
        enc = torchvision.io.encode_jpeg(torch.from_numpy(R["crops"][b]).permute(2, 0, 1).contiguous(), quality=95)
        txn.put(f"{key}/body_image".encode(), bytes(enc.numpy().tobytes()))
        a = torch.from_numpy((np.clip(R["alphas"][b], 0, 1) * 255).round().astype(np.uint8))[None].expand(3, -1, -1).contiguous()
        txn.put(f"{key}/body_mask".encode(), bytes(torchvision.io.encode_jpeg(a, quality=95).numpy().tobytes()))
    txn.commit(); env.close()


def shaded(img, vimg, faces, verts_cam, size=512):
    """Painter's-algorithm flat-shaded mesh over a dimmed image (overlay only)."""
    out = (cv2.resize(img, (size, size)) * 0.45).astype(np.uint8)
    tri = vimg[faces] * size; tv = verts_cam[faces]
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    shade = np.abs(n[:, 2]); order = np.argsort(-tv[:, :, 2].mean(1))
    for i in order:
        if shade[i] <= 0:
            continue
        c = int(60 + 190 * shade[i]); cv2.fillConvexPoly(out, tri[i].round().astype(np.int32), (c, int(c * 0.85), int(c * 0.6)))
    return out


def overlays(diag_dir, V, R, vf, Vverts_cam):
    """2 composites per clip (first frame, frame with min |facing| -> most back-facing): top image crop, bottom mesh."""
    fac = np.array([r["facing"] for r in R["filt"]])
    picks = {"front": int(np.argmax(fac)), "back": int(np.argmin(fac))}
    for tag, b in picks.items():
        img = cv2.resize(R["crops"][b], (512, 512))
        mesh = shaded(R["crops"][b], R["vimg"][b], vf.faces, Vverts_cam[b])
        k = (R["tgt"][b] * 512); ok = R["conf"][b] > 0
        for (u, v), g in zip(k, ok):
            if g: cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0), -1)
        f = R["filt"][b]
        comp = np.concatenate([img, mesh], 0)
        txt = f"{V['name']} {f['key']} out={f['mesh_out_frac']:.3f} kp={f['max_kp_err_rel']:.3f} n={f['n_kp05']} {'PASS' if f['pass'] else 'FAIL'} facing={f['facing']:+.2f}"
        cv2.putText(comp, txt[:70], (5, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(diag_dir / f"overlay_{tag}.jpg"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*"); ap.add_argument("--list", type=Path)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--dwpose-threads", type=int, default=8); ap.add_argument("--out", type=Path, default=O / "data" / "ubc")
    ap.add_argument("--diag", type=Path, default=O / "runs" / "ubc" / "clips"); ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    names = list(a.videos or []) + ([l.strip() for l in a.list.read_text().splitlines() if l.strip()] if a.list else [])
    names = names[a.shard::a.nshard]
    def done(n):
        f = a.diag / n / "fit.json"
        return f.exists() and json.loads(f.read_text()).get("ok", False)
    names = [n for n in names if a.redo or not done(n)]
    print(f"{len(names)} videos to fit", flush=True)
    fitter = SAF.SprintFitter(CKPT, a.device, iters=0, dwpose_threads=a.dwpose_threads, cuda_graph=False)
    assert "sprint_scratch" in fitter.ckpt
    vf = VideoFit(fitter, a.iters)
    q: queue.Queue = queue.Queue(maxsize=1)
    def producer():
        for n in names:
            try:
                q.put(cpu_stage(fitter, n, a.stride, a.cap))
            except Exception as e:  # noqa: BLE001
                q.put({"name": n, "error": repr(e)})
        q.put(None)
    wq: queue.Queue = queue.Queue(maxsize=1)
    def writer():
        while True:
            item = wq.get()
            if item is None:
                return
            name, dd, V, R, vc, extra = item
            try:
                t1 = time.perf_counter()
                write_clip(a.out / name, name, V, R, vf)
                t_write = time.perf_counter() - t1
                overlays(dd, V, R, vf, vc)
                np.savez_compressed(dd / "frames.npz", keys=np.array([r["key"] for r in R["filt"]]),
                                    **{k: np.array([r[k] for r in R["filt"]]) for k in ("mesh_out_frac", "max_kp_err_rel", "max_single_kp_err_rel", "n_kp05", "pass_a", "pass_b", "pass", "facing", "invtanfov")})
                npass = sum(r["pass"] for r in R["filt"])
                info = {"name": name, "ok": True, "n_video_frames": V["n"], "W": V["W"], "H": V["H"], "stride": V["stride"], "n_frames": len(V["frames"]),
                        "dwpose_fail": V["dw_fail"], "pass_frac": npass / len(R["filt"]), "pass_a_frac": float(np.mean([r["pass_a"] for r in R["filt"]])),
                        "pass_b_frac": float(np.mean([r["pass_b"] for r in R["filt"]])), "kept_by_filter": npass / len(R["filt"]) >= CLIP_PASS,
                        "diag": R["diag"], "sprint_shape_std10": R["sprint_shape_std"],
                        "times_s": {"dwpose_cpu": V["t_dwpose"], "wait_for_cpu": extra["wait_for_cpu"], **R["times"], "write": t_write, "gpu_total": extra["gpu_total"]},
                        "gpu_peak_gb": extra["gpu_peak_gb"], "ckpt": CKPT, "iters": a.iters, "weights": WTS, "frames": R["filt"]}
            except Exception as e:  # noqa: BLE001
                info = {"name": name, "ok": False, "error": repr(e), "trace": traceback.format_exc()}
            (dd / "fit.json").write_text(json.dumps(info, indent=1))
            print(name, "ok" if info["ok"] else info["error"][:200], f"pass={info.get('pass_frac', 0):.2f}", json.dumps(info.get("times_s", {})), flush=True)
    wt = threading.Thread(target=writer); wt.start()
    threading.Thread(target=producer, daemon=True).start()
    while True:
        tw = time.perf_counter(); V = q.get(); t_wait = time.perf_counter() - tw
        if V is None:
            break
        name = V["name"]; dd = a.diag / name; dd.mkdir(parents=True, exist_ok=True)
        try:
            if "error" in V or len(V["frames"]) < 3:
                raise RuntimeError(V.get("error", f"only {len(V.get('frames', []))} frames with a person"))
            t0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats(fitter.dev)
            for attempt in range(4):                                   # shared GPUs: retry on OOM after a pause
                try:
                    R = vf.fit(V); break
                except torch.OutOfMemoryError:
                    if attempt == 3:
                        raise
                    torch.cuda.empty_cache(); print(name, "OOM, retry in 120 s", flush=True); time.sleep(120)
            with torch.no_grad():
                Vv, _, _ = lean_batch(fitter.lean, R["rots"], R["beta"], R["exp"])
                vc = (Vv + R["Tt"][:, None]).cpu().numpy()
            R["rots"], R["beta"], R["exp"], R["Tt"] = R["rots"].cpu(), R["beta"].cpu(), R["exp"].cpu(), R["Tt"].cpu()
            extra = {"wait_for_cpu": t_wait, "gpu_total": time.perf_counter() - t0, "gpu_peak_gb": round(torch.cuda.max_memory_allocated(fitter.dev) / 1e9, 2)}
            wq.put((name, dd, V, R, vc, extra))                         # write (lmdb on /vault, jpeg) + overlays in the writer thread
        except Exception as e:  # noqa: BLE001
            info = {"name": name, "ok": False, "error": repr(e), "trace": traceback.format_exc()}
            (dd / "fit.json").write_text(json.dumps(info, indent=1)); print(name, info["error"][:200], flush=True)
    wq.put(None); wt.join()
    print("FIT_UBC_DONE", flush=True)


if __name__ == "__main__":
    main()
