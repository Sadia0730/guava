"""Avatar task A, step 1: tracker-free SMPL-X 2020 fit of ONE reference photo with SPRINT, written in GUAVA's tracked format
so GUAVA (a20) (pure SMPL-X 2020 head, phase3_dnt/guava_finetune.py --variant a20) can build an avatar from it. No FLAME,
no EHM-Tracker optimisation.

Pipeline (models loaded once; times measured per photo with CUDA synchronisation):
  1. person box + 2D keypoints: DWPose (EHM-Tracker/pretrained/dwpose: YOLOX-L person detector + DWPose-L 384 whole-body,
     133 COCO-WholeBody keypoints, re-ordered to EHM's 134-point OpenPose layout with the neck; the ONNX files run standalone
     with onnxruntime on the CPU -- this env's onnxruntime-gpu 1.18 lacks its CUDA 11 libraries). Largest person box
     (or --box). DWPose is a detector EHM-Tracker also consumes as input; it is independent of the tracker's FIT.
  2. crop: EHM's body-crop rule (crop_image_by_bbox: square, 1.1 x the box's long side, 1024 px), so the source image has
     GUAVA's training framing.
  3. person mask: StyleMatte (EHM-Tracker/pretrained/matting/stylematte_synth.pt, Mask2Former pixel decoder from
     assets/StyleMatte; the matting GUAVA's data used), alpha on the 1024 crop.
  4. SPRINT (sprint_raw, --ckpt, default SPRINT-ft EMA) on the crop at 256 px with the camera condition cond["cam"]
     (tools/eval_3dpw_standard.cam_feat: focal = image width, real 0 -- no calibration); optional shape lock
     (cond["shape"], cond["shape_on"] = 1). -> SMPL-X 2020 root / body / hands (rotation matrices), shape 200, exp 50, jaw
     (body_param jaw_pose), crop camera (PEAR camera: R = diag(-1, -1, 1), focal 24 on the crop = GUAVA's tanfov 1/24).
  5. refinement (--iters Adam steps, default 150, all on the GPU): residual rotations (axis-angle, right-multiplied) of the
     root, 21 body joints, 2 x 15 hand joints and the jaw; exp 50 and the first 20 betas as small regularised offsets; camera
     translation (x, y, log z). Data terms: (i) DWPose keypoints (score >= 0.3; body, face 68, hands 2 x 21; feet off)
     vs the SMPL-X joints / landmarks of the same layout (PEAR smplx_joints_to_dwpose) projected with the crop camera,
     Geman-McClure (sigma 3 % of the crop); (ii) the StyleMatte mask: projected vertices outside the mask (distance
     transform) and mask pixels far from every projected vertex (coverage), both robust and down-weighted (hair / loose
     clothing are in the mask, not in SMPL-X). Priors: squared residual rotations / offsets towards SPRINT's prediction.
  6. output (tracked format read by dataset/data_loader.TrackedData_infer, flat single-video layout): optim_tracking_ehm.pkl
     {frame_key: record}, id_share_params.pkl (smplx_shape = fitted betas; no joint offsets, head / hand scale 1),
     videos_info.json, img_lmdb/{frame_key}/{body_image, body_mask, ori_image}; record smplx_coeffs = global_pose,
     body_pose, left/right_hand_pose (axis-angle), exp (50), jaw_pose (1, 3), eye_pose zeros, camera_RT_params; FLAME
     coeffs zero (unused by a20); crop matrices (body 224 / 1024, head / hands 512 from DWPose keypoints) for GUAVA's boxes.
     Plus fit.json (timings, keypoint / mask diagnostics) and overlay.jpg.

Usage: python phase3_dnt/sprint_avatar_fit.py --image photo.jpg --out DIR [--box x1 y1 x2 y2] [--ckpt CKPT]
       python phase3_dnt/sprint_avatar_fit.py --lmdb TRACKED_CLIP/img_lmdb --key frame_000000 --out DIR   (ori_image)
Importable: SprintFitter (used by phase3_dnt/avatar_a_compare.py). Construct it BEFORE importing GUAVA's packages (it loads
PEAR's `models` / `utils` and then removes them from sys.modules, like c1_live_gap.py)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
EHMT = ROOT / "EHM-Tracker"
DEFAULT_CKPT = Path("/raid/ubx858/outputs/dnt/eval_programme/sprint_ft/best_ema.pt")
CROP, BODY_LR, PART = 1024, 224, 512           # EHM body_hd_size, body_crop_size, head / hand crop size
FOCAL = 24.0                                   # PEAR / GUAVA: tanfov = 1 / 24 over the square crop
KP_THR = 0.3


def load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def box_crop_matrix(box, dsize, scale=1.1):
    """EHM-Tracker src/utils/crop.crop_image_by_bbox (no rotation): original px -> crop px, (3, 3)."""
    l, t, r, b = [float(v) for v in box]
    size = int(max(r - l, b - t) * scale + 1e-7)
    s = dsize / (size + 1e-7)
    cx, cy = (l + r) / 2, (t + b) / 2
    return np.array([[s, 0, dsize / 2 - s * cx], [0, s, dsize / 2 - s * cy], [0, 0, 1]], np.float32)


def landmark_box(pts, scale=1.5):
    """EHM-Tracker parse_bbox_from_landmark_lite (square, x scale)."""
    lt, rb = pts.min(0), pts.max(0); c = (lt + rb) / 2; m = float((rb - lt).max()) * scale
    return [c[0] - m / 2, c[1] - m / 2, c[0] + m / 2, c[1] + m / 2]


def to_openpose134(kp133, sc133):
    """DWPose Wholebody.__call__ re-ordering (EHM-Tracker src/modules/dwpose/wholebody.py): neck inserted at 17 (mean of
    the shoulders, score = both shoulders > 0.3), first 18 points in OpenPose order. (133, 2), (133,) -> (134, 2), (134,)."""
    info = np.concatenate([kp133, sc133[:, None]], -1)[None]
    neck = np.mean(info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3, info[:, 6, 2:4] > 0.3).astype(int)
    new = np.insert(info, 17, neck, axis=1)
    mm = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]; op = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new[:, op] = new[:, mm]
    return new[0, :, :2], new[0, :, 2]


def kp_weights():
    w = np.zeros(134, np.float32)
    w[:14] = 1.0; w[14:18] = 0.5           # body; eyes / ears (SMPL-X eyeball / ear landmarks are approximate)
    w[18:24] = 0.0                          # feet (upper-body framings)
    w[24:41] = 0.2; w[41:92] = 0.4          # face contour, inner face
    w[92:134] = 0.4                         # hands
    return w


def rodrigues(d):
    """Axis-angle (..., 3) -> rotation matrices, finite gradients at 0 (no normalisation of the axis)."""
    th2 = (d * d).sum(-1)[..., None, None] + 1e-12; th = th2.sqrt()
    zero = torch.zeros_like(d[..., 0])
    K = torch.stack([zero, -d[..., 2], d[..., 1], d[..., 2], zero, -d[..., 0], -d[..., 1], d[..., 0], zero], -1).reshape(*d.shape[:-1], 3, 3)
    return torch.eye(3, device=d.device) + torch.sin(th) / th * K + (1 - torch.cos(th)) / th2 * (K @ K)


def gm(r2, s2):
    """Geman-McClure on squared residuals, scaled to ~ r2 / s2 for small residuals."""
    return r2 / (r2 + s2)


class LeanSMPLX(torch.nn.Module):
    """The pure SMPL-X 2020 layer of the evaluators (PEAR SMPLXV2 via tools/eval_3dpw_standard.pure_smplx_mesh, EHM's J14
    replacement subset) re-implemented for the refinement loop without its per-call overheads (rotation-matrix -> axis-angle
    -> matrix round trip, dynamic landmarks, host syncs): blend shapes + LBS from rotation matrices (rigid chain by tree
    level), and the 134 DWPose points as fixed linear maps of the posed vertices (static face landmarks, extra joints,
    J14 regressor rows) or kinematic joints. The 17 dynamic face-contour landmarks (and the two ears, which map to contour
    points) are NOT reproduced (rows left zero, weight 0 in the fit). Checked against the layer by check_against()."""

    def __init__(self, layer, ev, to_dw):
        super().__init__()
        L = layer; dev = L.v_template.device
        self.register_buffer("v_template", L.v_template.float())
        self.register_buffer("shapedirs", L.shapedirs.float())                     # (V, 3, 300 + 50)
        self.register_buffer("posedirs", L.posedirs.float())                       # (486, V * 3)
        self.register_buffer("J_regressor", L.J_regressor.float())
        self.register_buffer("lbs_weights", L.lbs_weights.float())
        self.n_shape = int(L.n_shape); par = L.parents.long().cpu().numpy(); par[0] = -1
        self.register_buffer("parents", torch.from_numpy(np.where(par < 0, 0, par)).long().to(dev))
        depth = np.zeros(len(par), int)
        for j in range(1, len(par)):
            depth[j] = depth[par[j]] + 1
        self.levels = [torch.from_numpy(np.flatnonzero(depth == d)).to(dev) for d in range(1, depth.max() + 1)]
        V = L.v_template.shape[0]; F_ = L.faces_tensor.long()
        rows = np.zeros((145, V), np.float32); from_j = np.zeros(145, bool); from_j[:55] = True
        lf, lb = L.lmk_faces_idx.long(), L.lmk_bary_coords.float()
        for i in range(lf.shape[0]):                                                 # static landmarks 55..105
            for c in range(3):
                rows[55 + i, int(F_[lf[i], c])] += float(lb[i, c])
        sel = L.extra_joint_selector
        for i in range(sel.face_ids.shape[0]):                                       # extra joints 123..144
            for c in range(3):
                rows[123 + i, int(F_[sel.face_ids[i], c])] += float(sel.bcs[i, c])
        idx = torch.tensor(ev.EHM_J14_REPLACE)
        src = L.source_idxs[idx].long().cpu().numpy(); tgt = L.target_idxs[idx].long().cpu().numpy()
        R = L.extra_joint_regressor.float().cpu().numpy()
        for s_, t_ in zip(src, tgt):
            rows[s_] = R[t_]; from_j[s_] = False
        # DWPose mapping (smplx_joints_to_dwpose): which of the 145 joints each of the 134 points takes
        probe = torch.arange(145, dtype=torch.float32)[None, :, None].expand(1, 145, 3).clone()
        m = to_dw(probe)[0][0, :, 0].long().cpu().numpy(); m[1] = 12                   # row 1 (neck): x, y overwritten by the shoulder mean (forward)
        self.map = m
        self.register_buffer("Wv", torch.from_numpy(rows[m]).to(dev))                # (134, V)
        Sj = np.zeros((134, 55), np.float32)
        for i, j in enumerate(m):
            if j < 55 and from_j[j]:
                Sj[i, j] = 1.0
        self.register_buffer("Sj", torch.from_numpy(Sj).to(dev))
        self.register_buffer("hrow", torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev))
        self.register_buffer("nonroot", (torch.arange(55, device=dev) > 0).float()[:, None])
        self.register_buffer("dyn", torch.from_numpy(((m >= 106) & (m <= 122)).astype(bool)).to(dev))

    def forward(self, glob, body, jaw, eye, lh, rh, beta, exp):
        """Rotation matrices (1, n, 3, 3), beta (1, <= 300), exp (1, 50) -> vertices (1, V, 3), dwpose points (1, 134, 3)."""
        rots = torch.cat([glob, body, jaw, eye, lh, rh], 1)[0]                        # (55, 3, 3)
        b = torch.cat([beta, beta.new_zeros(1, self.n_shape - beta.shape[1]), exp], 1)[0]
        v_sh = self.v_template + torch.einsum("vcs,s->vc", self.shapedirs, b)
        J = self.J_regressor @ v_sh
        eye3 = torch.eye(3, device=rots.device)
        v_po = v_sh + ((rots[1:] - eye3).reshape(1, -1) @ self.posedirs).reshape(-1, 3)
        rel = J - J[self.parents] * self.nonroot
        Tl = torch.cat([torch.cat([rots, rel[:, :, None]], 2), self.hrow.expand(55, 1, 4)], 1)
        Tw = Tl
        for lev in self.levels:
            Tw = Tw.index_put((lev,), Tw[self.parents[lev]] @ Tl[lev])
        Jt = Tw[:, :3, 3]
        A = torch.cat([Tw[:, :3, :3], (Jt - (Tw[:, :3, :3] @ J[:, :, None])[:, :, 0])[:, :, None]], 2)   # (55, 3, 4)
        Tv = (self.lbs_weights @ A.reshape(55, 12)).reshape(-1, 3, 4)
        verts = (Tv[:, :, :3] @ v_po[:, :, None])[:, :, 0] + Tv[:, :, 3]
        P = self.Wv @ verts + self.Sj @ Jt
        P = torch.cat([P[:1], torch.cat([(P[2:3, :2] + P[5:6, :2]) / 2, P[1:2, 2:]], 1), P[2:]], 0)
        return verts[None], P[None]

    @torch.no_grad()
    def check_against(self, mesh_fn, bp):
        """Max |difference| (m) of vertices and of the non-dynamic DWPose points vs the evaluators' layer."""
        V0, K0 = mesh_fn(bp)
        eye = bp.get("eye_pose") if bp.get("eye_pose") is not None else torch.eye(3, device=V0.device).expand(1, 2, 3, 3)
        V1, K1 = self(bp["global_pose"], bp["body_pose"], bp["jaw_pose"], eye, bp["left_hand_pose"], bp["right_hand_pose"], bp["shape"], bp["exp"][:, :50])
        keep = ~self.dyn
        return float((V0 - V1).abs().max()), float((K0[0, keep] - K1[0, keep]).abs().max())


class SprintFitter:
    def __init__(self, ckpt=DEFAULT_CKPT, device="cuda:0", iters=150, use_mask=True, dwpose_threads=16,
                 student_config="configs/student_l70_v2.yaml", cuda_graph=True):
        self.dev = torch.device(device); self.iters = iters; self.use_mask = use_mask; self.ckpt = str(ckpt)
        # --- PEAR side (SPRINT, pure SMPL-X 2020 layer, DWPose joint mapping) ---
        self.ev = load_by_path("guava_eval_3dpw_standard", ROOT / "tools" / "eval_3dpw_standard.py")
        cwd = Path.cwd(); os.chdir(PEAR_ROOT); sys.path.insert(0, str(PEAR_ROOT))
        self.model, self.load_report = self.ev.load_model("sprint_raw", Path(ckpt).resolve(), student_config, 1, self.dev)
        self.layer = self.ev.pure_smplx_layer(self.dev)
        from models.smplx.smplx_head import smplx_joints_to_dwpose
        self.to_dw = smplx_joints_to_dwpose
        self.lean = LeanSMPLX(self.layer, self.ev, smplx_joints_to_dwpose).to(self.dev)
        while str(PEAR_ROOT) in sys.path:
            sys.path.remove(str(PEAR_ROOT))
        os.chdir(cwd)
        for name in list(sys.modules):
            f = getattr(sys.modules[name], "__file__", None) or ""
            if f.startswith(str(PEAR_ROOT)) and name.split(".")[0] in ("models", "utils", "dataset", "tools"):
                del sys.modules[name]
        for n in [n for n in sys.modules if n == "models" or n.startswith("models.") or n == "utils" or n.startswith("utils.")]:
            if not (getattr(sys.modules[n], "__file__", None) or "").startswith(str(ROOT / "models")) and \
               not (getattr(sys.modules[n], "__file__", None) or "").startswith(str(ROOT / "utils")):
                sys.modules.pop(n, None)
        # --- DWPose (CPU onnxruntime) ---
        import onnxruntime as ort
        self.det = load_by_path("ehm_dwpose_onnxdet", EHMT / "src/modules/dwpose/onnxdet.py")
        self.pose = load_by_path("ehm_dwpose_onnxpose", EHMT / "src/modules/dwpose/onnxpose.py")
        so = ort.SessionOptions(); so.intra_op_num_threads = dwpose_threads
        self.s_det = ort.InferenceSession(str(EHMT / "pretrained/dwpose/yolox_l.onnx"), so, providers=["CPUExecutionProvider"])
        self.s_pose = ort.InferenceSession(str(EHMT / "pretrained/dwpose/dw-ll_ucoco_384.onnx"), so, providers=["CPUExecutionProvider"])
        # --- StyleMatte (GPU) ---
        sm = load_by_path("ehm_stylematte", EHMT / "src/modules/human_matting/stylematte.py")
        cwd = Path.cwd(); os.chdir(ROOT)                     # cache_dir="assets/StyleMatte" is relative
        self.matte = sm.StyleMatte()
        os.chdir(cwd)
        self.matte.load_state_dict(torch.load(EHMT / "pretrained/matting/stylematte_synth.pt", map_location="cpu", weights_only=True))
        self.matte = self.matte.to(self.dev).eval()
        self.w_kp = torch.from_numpy(kp_weights()).to(self.dev)
        self.cuda_graph = cuda_graph
        self._setup_refine()
        self._warm()

    def _warm(self):
        img = np.zeros((480, 640, 3), np.uint8); img[100:400, 250:400] = 200
        self.fit(img, box=[250, 100, 400, 400], iters=5, _warm=True)
        self.fit(img, box=[250, 100, 400, 400], iters=5, _warm=True)

    # ---------------------------------------------------------------- components
    def dwpose(self, img_rgb, box=None):
        """EHM's usage: RGB frame; largest person box. Returns box (4,), keypoints (134, 2) original px, scores (134,)."""
        if box is None:
            boxes = self.det.inference_detector(self.s_det, img_rgb)
            if len(boxes) == 0:
                raise RuntimeError("no person detected")
            box = boxes[int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))]
        box = np.asarray(box, np.float64)
        kp, sc = self.pose.inference_pose(self.s_pose, box[None], img_rgb)
        k, s = to_openpose134(kp[0], sc[0])
        return box, k, s

    @torch.no_grad()
    def matting(self, crop_t):
        """crop_t (1, 3, 1024, 1024) in [0, 1] -> alpha (1024, 1024) (EHM-Tracker StyleMatteModel.forward, 'alpha')."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        return self.matte((crop_t - mean) / std)[0][0]

    @torch.no_grad()
    def sprint(self, crop_t, cam, shape=None):
        x = F.interpolate(crop_t, size=(256, 256), mode="bilinear", antialias=True, align_corners=False)
        cond = {"cam": torch.as_tensor(cam, device=self.dev).float().reshape(1, 5)}
        if shape is not None:
            cond["shape"] = torch.as_tensor(np.asarray(shape, np.float32).reshape(1, -1)[:, :200], device=self.dev)
            cond["shape_on"] = torch.ones(1, device=self.dev)
        out = self.model(x, cond=cond)
        return {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v.float())
                for k, v in out.items()}

    def mesh(self, bp):
        o = self.ev.pure_smplx_mesh({"body_param": bp, "flame_param": {}})
        kp = self.to_dw(o["joints"])[0][..., :3]
        return o["vertices"], kp

    @staticmethod
    def project(R, t, X):
        """PEAR / GUAVA crop camera: normalised crop coordinates in [0, 1] (tpd.project_joints)."""
        cam = X @ R.T + t
        return (1.0 - FOCAL * cam[..., :2] / cam[..., 2:3].clamp_min(1e-4)) * 0.5


    # ---------------------------------------------------------------- refinement (static tensors, CUDA graph)
    NM = 1500
    WTS = {"kp": 1.0, "mask_in": 0.3, "mask_cov": 0.2, "prior": 0.05}

    def _setup_refine(self):
        dev = self.dev; z = lambda *s: torch.zeros(*s, device=dev)
        st = self._st = {}
        for n, sh in (("dg", (1, 3)), ("db", (21, 3)), ("dl", (15, 3)), ("dr", (15, 3)), ("dj", (1, 3)), ("dexp", (50,)), ("dbeta", (20,)),
                      ("txy", (2,)), ("lz", (1,))):
            st[n] = z(*sh).requires_grad_(True)
        st.update(R=z(3, 3), g0=z(1, 1, 3, 3), b0=z(1, 21, 3, 3), l0=z(1, 15, 3, 3), r0=z(1, 15, 3, 3), j0=z(1, 1, 3, 3), shape0=z(200), exp0=z(50),
                  tgt=z(134, 2), conf=z(134), dt=z(1, 1, 256, 256), mpts=z(self.NM, 2), mw=z(self.NM), beta_on=z(1),
                  eye=torch.eye(3, device=dev).expand(1, 2, 3, 3).contiguous())
        self._opt = torch.optim.Adam([{"params": [st["dg"], st["db"], st["dj"]], "lr": 0.02}, {"params": [st["dl"], st["dr"]], "lr": 0.02},
                                      {"params": [st["dexp"]], "lr": 0.05}, {"params": [st["txy"], st["lz"]], "lr": 0.01},
                                      {"params": [st["dbeta"]], "lr": 0.05}], capturable=True)
        self._graph = None

    def _build(self):
        st = self._st; dev = self.dev
        bp = {"global_pose": st["g0"] @ rodrigues(st["dg"])[None], "body_pose": st["b0"] @ rodrigues(st["db"])[None],
              "left_hand_pose": st["l0"] @ rodrigues(st["dl"])[None], "right_hand_pose": st["r0"] @ rodrigues(st["dr"])[None],
              "jaw_pose": st["j0"] @ rodrigues(st["dj"])[None],
              "shape": (st["shape0"] + torch.cat([st["dbeta"] * st["beta_on"], torch.zeros(180, device=dev)]))[None],
              "exp": (st["exp0"] + st["dexp"])[None], "eye_pose": None}
        return bp, torch.cat([st["txy"], torch.exp(st["lz"])])

    def _losses(self, bp, tt):
        st = self._st; s2 = 0.03 ** 2
        V, K = self.lean(bp["global_pose"], bp["body_pose"], bp["jaw_pose"], st["eye"], bp["left_hand_pose"], bp["right_hand_pose"], bp["shape"], bp["exp"])
        uv = self.project(st["R"], tt, K[0]); r2 = ((uv - st["tgt"]) ** 2).sum(-1)
        L = {"kp": (st["conf"] * gm(r2, s2)).sum() / st["conf"].sum().clamp_min(1e-6)}
        v = self.project(st["R"], tt, V[0][::3])
        inside = ((v >= 0) & (v <= 1)).all(-1).float() * (st["mw"].sum() > 0).float()
        d = F.grid_sample(st["dt"], (v * 2 - 1)[None, None], mode="bilinear", align_corners=False)[0, 0, 0]
        L["mask_in"] = (inside * gm(d ** 2, s2)).sum() / inside.sum().clamp_min(1.0)
        L["mask_cov"] = (st["mw"] * gm(torch.cdist(st["mpts"], v).min(1)[0] ** 2, s2)).sum() / st["mw"].sum().clamp_min(1.0)
        L["prior"] = (0.5 * (st["dg"] ** 2).sum() + 4.0 / 21 * 0.5 * (st["db"] ** 2).sum() + 0.05 / 15 * ((st["dl"] ** 2).sum() + (st["dr"] ** 2).sum())
                      + 1.0 * (st["dj"] ** 2).sum() + 0.02 * (st["dexp"] ** 2).sum() + 0.05 * (st["dbeta"] ** 2).sum() * st["beta_on"].sum())
        return L, uv

    def _total(self):
        L, _ = self._losses(*self._build())
        return sum(self.WTS[k] * v for k, v in L.items())

    def _step(self):
        self._opt.zero_grad(set_to_none=True); self._total().backward(); self._opt.step()

    @torch.no_grad()
    def _reset(self, t_init):
        st = self._st
        for n in ("dg", "db", "dl", "dr", "dj", "dexp", "dbeta"):
            st[n].zero_()
        st["txy"].copy_(t_init[:2]); st["lz"].copy_(torch.log(t_init[2:3]))
        for g in self._opt.param_groups:
            for p_ in g["params"]:
                s_ = self._opt.state.get(p_)
                if s_:
                    s_["step"].zero_(); s_["exp_avg"].zero_(); s_["exp_avg_sq"].zero_()

    def _capture(self, t_init):
        try:
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph(); self._opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(g):
                self._total().backward(); self._opt.step()
            self._graph = g
        except Exception as e:  # noqa: BLE001 -- eager fallback
            print("CUDA graph capture failed, eager refinement:", repr(e), flush=True); self._graph = None; self.cuda_graph = False
        self._reset(t_init)

    # ---------------------------------------------------------------- fit
    def fit(self, img_rgb, box=None, iters=None, shape_lock=None, _warm=False, _check=False):
        import roma
        iters = self.iters if iters is None else iters
        dev = self.dev; T = {}
        H, W = img_rgb.shape[:2]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        box, kp, sc = self.dwpose(img_rgb, box)
        T["dwpose_cpu"] = time.perf_counter() - t0
        M = box_crop_matrix(box, CROP)
        crop = cv2.warpAffine(img_rgb, M[:2], (CROP, CROP), flags=cv2.INTER_LINEAR)
        crop_t = torch.from_numpy(crop).to(dev).permute(2, 0, 1)[None].float() / 255.0
        t1 = time.perf_counter()
        alpha = self.matting(crop_t)
        torch.cuda.synchronize(); T["matting"] = time.perf_counter() - t1; t1 = time.perf_counter()
        A256 = (np.diag([256 / CROP, 256 / CROP, 1.0]) @ M)[:2]
        cam = self.ev.cam_feat(A256, W, H, W, 0.0)
        out = self.sprint(crop_t, cam, shape_lock)
        torch.cuda.synchronize(); T["sprint"] = time.perf_counter() - t1; t1 = time.perf_counter()
        bp0 = out["body_param"]
        R = out["pd_cam"][0, :3, :3].float(); t_init = out["pd_cam"][0, :3, 3].float()
        # targets in normalised crop coordinates
        kpc = (np.c_[kp, np.ones(len(kp))] @ M.T)[:, :2] / CROP
        ok = (sc >= KP_THR) & (kpc >= 0).all(1) & (kpc <= 1).all(1)
        tgt = torch.from_numpy(kpc).float().to(dev); conf = torch.from_numpy(np.where(ok, sc, 0.0)).float().to(dev) * self.w_kp
        conf = conf * (~self.lean.dyn).float()                                                    # contour / ears not modelled
        a_small = F.interpolate(alpha[None, None], size=(256, 256), mode="bilinear", antialias=True, align_corners=False)[0, 0]
        m_np = (a_small > 0.5).cpu().numpy().astype(np.uint8)
        dt_out = cv2.distanceTransform((1 - m_np).astype(np.uint8), cv2.DIST_L2, 5) / 256.0              # 0 inside the mask
        dt_t = torch.from_numpy(dt_out).float().to(dev)[None, None]
        ys, xs = np.nonzero(m_np)
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xs), min(1500, len(xs)), replace=False) if len(xs) else np.zeros(0, int)
        mpts = torch.from_numpy(np.stack([(xs[sel] + 0.5) / 256, (ys[sel] + 0.5) / 256], 1)).float().to(dev)
        # refinement: static buffers + one captured CUDA graph (forward, backward, Adam step) replayed `iters` times
        st = self._st; lock = shape_lock is not None
        with torch.no_grad():
            st["R"].copy_(R); st["g0"].copy_(bp0["global_pose"]); st["b0"].copy_(bp0["body_pose"]); st["l0"].copy_(bp0["left_hand_pose"])
            st["r0"].copy_(bp0["right_hand_pose"]); st["j0"].copy_(bp0["jaw_pose"]); st["shape0"].copy_(bp0["shape"][0, :200])
            st["exp0"].copy_(bp0["exp"][0, :50]); st["tgt"].copy_(tgt); st["conf"].copy_(conf); st["dt"].copy_(dt_t)
            st["mpts"].zero_(); st["mw"].zero_()
            if len(mpts) and self.use_mask:
                st["mpts"][:len(mpts)] = mpts; st["mw"][:len(mpts)] = 1.0
            st["beta_on"].fill_(0.0 if lock else 1.0)
        self._reset(t_init)
        if _check:
            with torch.no_grad():
                b_, _ = self._build(); self.lean_check = self.lean.check_against(self.mesh, b_)
        diag = {}
        with torch.no_grad():
            L0, uv0 = self._losses(*self._build())
        tgt_, conf_ = st["tgt"], st["conf"]
        px = lambda uv: float((((uv - tgt_).norm(dim=-1) * CROP) * (conf_ > 0)).sum() / (conf_ > 0).sum().clamp_min(1))
        diag["kp_err_px1024_before"] = px(uv0); diag["loss_before"] = {k: float(v) for k, v in L0.items()}
        if iters > 0:
            if self._graph is None and self.cuda_graph:
                self._capture(t_init)
            if self._graph is not None:
                for _ in range(iters):
                    self._graph.replay()
            else:
                for _ in range(iters):
                    self._step()
        with torch.no_grad():
            bp, tt = self._build(); L1, uv1 = self._losses(bp, tt)
        torch.cuda.synchronize(); T["refine"] = time.perf_counter() - t1; t1 = time.perf_counter()
        diag["kp_err_px1024_after"] = px(uv1); diag["loss_after"] = {k: float(v) for k, v in L1.items()}
        diag["n_kp"] = int((conf > 0).sum()); diag["iters"] = iters; diag["cuda_graph"] = self._graph is not None
        aa = lambda Rm: roma.rotmat_to_rotvec(Rm.reshape(-1, 3, 3)).detach().cpu().numpy().astype(np.float32)
        RT = torch.cat([R, tt[:, None]], 1).detach().cpu().numpy().astype(np.float32)
        exp0 = bp0["exp"][0, :50].detach(); shape0 = bp0["shape"][0].detach()
        smplx = {"global_pose": aa(bp["global_pose"])[0], "body_pose": aa(bp["body_pose"]), "left_hand_pose": aa(bp["left_hand_pose"]),
                 "right_hand_pose": aa(bp["right_hand_pose"]), "jaw_pose": aa(bp["jaw_pose"]).reshape(1, 3),
                 "eye_pose": np.zeros((2, 3), np.float32), "exp": bp["exp"][0].detach().cpu().numpy().astype(np.float32),
                 "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}
        shape = bp["shape"][0].detach().cpu().numpy().astype(np.float32)
        raw = {"global_pose": aa(bp0["global_pose"])[0], "body_pose": aa(bp0["body_pose"]), "left_hand_pose": aa(bp0["left_hand_pose"]),
               "right_hand_pose": aa(bp0["right_hand_pose"]), "jaw_pose": aa(bp0["jaw_pose"]).reshape(1, 3), "eye_pose": np.zeros((2, 3), np.float32),
               "exp": exp0.cpu().numpy().astype(np.float32), "camera_RT_params": out["pd_cam"][0, :3, :4].cpu().numpy().astype(np.float32),
               "body_cam": np.zeros(3, np.float32)}
        # part crops (GUAVA's head / hand boxes) from DWPose keypoints
        Mlr = box_crop_matrix(box, BODY_LR)
        def part(idx, scale_lmk, scale_crop):
            g = idx[sc[idx] >= KP_THR]
            if len(g) < 4:
                return {"M_o2c": M.copy() * np.array([[PART / CROP], [PART / CROP], [1]], np.float32), "M_c2o": None}
            return {"M_o2c": box_crop_matrix(landmark_box(kp[g], scale_lmk), PART, scale_crop), "M_c2o": None}
        crops = {"head_crop": part(np.arange(24, 92), 1.5, 1.3), "left_hand_crop": part(np.arange(92, 113), 1.5, 1.3),
                 "right_hand_crop": part(np.arange(113, 134), 1.5, 1.3)}
        for v in crops.values():
            v["M_c2o"] = np.linalg.inv(v["M_o2c"]).astype(np.float32)
        z50 = np.zeros(50, np.float32)
        rec = {"smplx_coeffs": smplx,
               "flame_coeffs": {"expression_params": z50, "jaw_params": np.zeros(3, np.float32), "pose_params": np.zeros(3, np.float32),
                                "neck_pose_params": np.zeros(3, np.float32), "eye_pose_params": np.zeros(6, np.float32),
                                "eyelid_params": np.zeros(2, np.float32), "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
               "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": M, "M_c2o-hd": np.linalg.inv(M).astype(np.float32)},
               "dwpose_raw": {"keypoints": kp.astype(np.float32), "scores": sc.astype(np.float32), "bbox": np.asarray(box, np.float32)},
               **crops}
        T["total"] = time.perf_counter() - t0
        return {"record": rec, "shape": shape, "raw_smplx": raw, "raw_shape": shape0.cpu().numpy().astype(np.float32),
                "crop": crop, "alpha": alpha.clamp(0, 1).cpu().numpy(), "box": np.asarray(box), "M": M, "kp_crop": kpc, "kp_score": sc,
                "uv_before": uv0.cpu().numpy(), "uv_after": uv1.cpu().numpy(), "times": T, "diag": diag, "cam_feat": cam}


def id_share(shape):
    return {"smplx_shape": np.asarray(shape, np.float32).reshape(1, 200), "flame_shape": np.zeros((1, 300), np.float32),
            "left_mano_shape": np.zeros((1, 10), np.float32), "right_mano_shape": np.zeros((1, 10), np.float32),
            "head_scale": np.ones((1, 3), np.float32), "hand_scale": np.ones((1, 3), np.float32), "joints_offset": np.zeros((1, 55, 3), np.float32)}


def write_tracked(out_dir: Path, name: str, key: str, res: dict, ori_rgb=None, raw=False):
    """GUAVA tracked directory (flat single-video layout of TrackedData_infer). raw=True writes SPRINT's unrefined fit."""
    import lmdb, io, torchvision
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = dict(res["record"])
    if raw:
        rec = {**rec, "smplx_coeffs": res["raw_smplx"]}
    with open(out_dir / "optim_tracking_ehm.pkl", "wb") as f:
        pickle.dump({key: rec}, f)
    with open(out_dir / "id_share_params.pkl", "wb") as f:
        pickle.dump(id_share(res["raw_shape"] if raw else res["shape"]), f)
    (out_dir / "videos_info.json").write_text(json.dumps({name: {"frames_num": 1, "frames_keys": [key]}}, indent=1))
    env = lmdb.open(str(out_dir / "img_lmdb"), map_size=1 << 32); txn = env.begin(write=True)
    def put(k, chw_uint8):
        enc = torchvision.io.encode_jpeg(chw_uint8.to(torch.uint8), quality=95)
        txn.put(f"{key}/{k}".encode(), bytes(enc.numpy().tobytes()))
    put("body_image", torch.from_numpy(res["crop"]).permute(2, 0, 1).contiguous())
    a = torch.from_numpy((np.clip(res["alpha"], 0, 1) * 255).round().astype(np.uint8))[None].expand(3, -1, -1).contiguous()
    put("body_mask", a)
    if ori_rgb is not None:
        put("ori_image", torch.from_numpy(np.ascontiguousarray(ori_rgb)).permute(2, 0, 1).contiguous())
    txn.commit(); env.close()


def overlay(res, path):
    img = res["crop"].copy(); a = res["alpha"][..., None]
    img = (img * (0.5 + 0.5 * a)).astype(np.uint8)
    ok = res["kp_score"] >= KP_THR
    for (u, v), g in zip(res["kp_crop"] * CROP, ok):
        if g: cv2.circle(img, (int(u), int(v)), 4, (0, 255, 0), -1)
    for uv, col in ((res["uv_before"], (255, 0, 0)), (res["uv_after"], (0, 128, 255))):
        for (u, v), g in zip(uv * CROP, ok):
            if g: cv2.circle(img, (int(u), int(v)), 3, col, -1)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path); p.add_argument("--lmdb", type=Path); p.add_argument("--key", default="frame_000000")
    p.add_argument("--box", type=float, nargs=4, default=None, help="person box x1 y1 x2 y2 (original px); default: YOLOX largest")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--name", default=None)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT); p.add_argument("--iters", type=int, default=150)
    p.add_argument("--no-mask", action="store_true"); p.add_argument("--device", default="cuda:0"); p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()
    if a.image:
        img = cv2.cvtColor(cv2.imread(str(a.image)), cv2.COLOR_BGR2RGB); name = a.name or a.image.stem
    else:
        import lmdb
        env = lmdb.open(str(a.lmdb), readonly=True, lock=False); buf = env.begin().get(f"{a.key}/ori_image".encode())
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); name = a.name or a.lmdb.parent.name
    fitter = SprintFitter(a.ckpt, a.device, a.iters, use_mask=not a.no_mask)
    times = []
    for _ in range(max(1, a.repeat)):
        res = fitter.fit(img, a.box); times.append(res["times"])
    write_tracked(a.out, name, a.key, res, img)
    overlay(res, a.out / "overlay.jpg")
    info = {"ckpt": str(a.ckpt), "iters": a.iters, "mask": not a.no_mask, "times_s": times, "diag": res["diag"], "box": res["box"].tolist(),
            "cam_feat": np.asarray(res["cam_feat"]).tolist(), "gpu": torch.cuda.get_device_name(fitter.dev)}
    (a.out / "fit.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
