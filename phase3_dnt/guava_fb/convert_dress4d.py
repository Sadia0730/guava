"""GUAVA-FB / 4D-DRESS converter (format: phase3_dnt/guava_fb/FORMAT.md; registered: PREREG "GUAVA-FB").

Input  (read-only): 4D-DRESS extracted/<subj>/<Inner|Outer>/<Take>/ (Capture/<cam>/images, masks, cameras.pkl, basic_info.pkl) and
       the SMPLX_NEUTRAL_2020 refit of fit_dress4d.py (<refit>/<subj>.npz: one shape per subject, per-frame poses / exp / transl).
Output: O/data/dress4d/<subj>_<outfit>_<take>/ (one clip per take): optim_tracking_ehm.pkl, id_share_params.pkl, videos_info.json,
       img_lmdb/ (<frame_key>/body_image, <frame_key>/body_mask; JPEG q95 as sprint_avatar_fit.write_tracked), gt_proj.npz,
       stats.json (per-row checks). Rows = every 5th scan frame x the 4 cameras; frame_key '<cam>_<frame>' (e.g. '0004_00011'),
       record["view"] = cam id, keys ordered camera-major (frames_keys[0] = first frame of camera 0004).

Per row:
  1. person box (original image px) = union of (a) the projected source mesh (4D-DRESS gendered SMPL-X v1.1) and (b) StyleMatte
     mask pixels (pass 1: StyleMatte on the whole frame letterboxed to 512; connected components that touch the projected mesh).
  2. perspective crop: virtual camera rotated (minimal rotation) onto the ray through the box centre (2 re-centering iterations,
     box re-measured in the rotated camera), square side 1.1 x the long side of the box (in the rotated camera), 1024 px,
     principal point at the crop centre; H = K_v R_v K^-1 (cv2.warpPerspective, bilinear, black border).
     invtanfov = 2 f_v / 1024. Continuous crop px (pixel i spans [i, i+1)) = 1024 * GUAVA's normalised uv = 512 + f_v x / z.
  3. StyleMatte (EHM-Tracker stylematte_synth.pt, as sprint_avatar_fit) pass 2 on the 1024 crop -> body_mask.
  4. SMPL-X 2020 record: camera rotation R_v R_c folded into global_pose (about the pelvis joint J0 of the shaped rest mesh incl.
     expression): R' = R_v R_c R_g, t' = R_v R_c (J0 + transl) + R_v t_c - J0; camera_RT_params = [diag(-1,-1,1) | diag(-1,-1,1) t']
     (GUAVA w2c = diag(-1,-1,1,1) [R|T] -> X_cv = X_model + t'). Hands axis-angle, flat hand mean; eye_pose from the refit;
     dummy flame_coeffs as sprint_avatar_fit.
  5. crops: body_crop M_o2c-hd = I (the stored crop is the "original"), M_o2c = 224/1024 scale; head_crop from the projected SMPL-X
     head vertices (SMPL-X->FLAME ids minus FLAME's lower-head / neck ids), square x1.25 (about EHM's x1.75 on 68 face landmarks);
     left / right_hand_crop from the projected MANO vertices, square x1.2 (AVATAR_B_PLAN 4.1); 512 px part crops.
  6. checks (stats.json per row): StyleMatte IoU vs the provided rendered mask (warped with the same H; alpha > 0.5), mesh-vs-mask
     precision (refit mesh silhouette rasterised at 256 px inside StyleMatte mask / silhouette, counted only over crop pixels that come
     from inside the original image; also without that restriction) and recall, trunc_frac (fraction of source-mesh vertices that
     project outside the original 1280 x 940 image; also in record["dress4d"] so a loader can filter), reprojection error of the
     record (GUAVA camera formula) vs the source mesh in the crop (px at 1024).
  gt_proj.npz: up to 10 rows (spread over cameras / frames): frame_key, v2d_1024 (10475, 2) = SOURCE gendered mesh, original camera,
     projected into the crop (continuous px); also v2d_conv_1024 (converted record, same formula), invtanfov.

Usage: python convert_dress4d.py --refit DIR --out O/data/dress4d [--takes 00122/Inner/Take2 ...] [--gpus 1,2] [--workers-per-gpu 3]
       [--limit N]; index.json is written by --index-only (after the run) from the clip dirs + refit summary."""
import argparse, glob, hashlib, json, os, pickle, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path("/home/ubx858/guava")
EHMT = ROOT / "EHM-Tracker"
DATA = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/4d_dress/extracted")
CAMS = ["0004", "0028", "0052", "0076"]
STRIDE = 5
CROP, BODY_LR, PART = 1024, 224, 512
HEAD_SCALE, HAND_SCALE = 1.25, 1.2
P1 = 512                                   # pass-1 (box) StyleMatte resolution
LICENCE = ("4D-DRESS (Wang et al., CVPR 2024; ETH Zurich AIT lab, https://4d-dress.ait.ethz.ch). Obtained with a personal download "
           "link after accepting the website's terms of use (non-commercial scientific research only, no redistribution). The download "
           "contains NO licence file; 4D-DRESS/README.md describes the data but states no terms, so the terms are the website's.")


def subjects():
    return sorted(s for s in os.listdir(DATA) if s.isdigit())


def split_of():
    """Registered: hold out 2 female + 2 male subjects with the lowest SHA-1 (hex of the subject id string) within each gender."""
    g = {}
    for s in subjects():
        T = sorted(glob.glob(str(DATA / s / "*" / "Take*")))[0]
        g[s] = pickle.load(open(Path(T) / "basic_info.pkl", "rb"))["gender"]
    val = []
    for gen in ("female", "male"):
        ss = sorted((hashlib.sha1(s.encode()).hexdigest(), s) for s in g if g[s] == gen)
        val += [s for _, s in ss[:2]]
    return {s: ("val" if s in val else "train") for s in g}, g


def rot_to(d):
    """Minimal rotation R with R d / |d| = z."""
    d = d / np.linalg.norm(d); z = np.array([0.0, 0.0, 1.0]); v = np.cross(d, z); s = np.linalg.norm(v); c = float(d @ z)
    if s < 1e-12: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def box_crop_matrix(box, dsize, scale=1.1):
    """EHM-Tracker crop_image_by_bbox (no rotation), as sprint_avatar_fit."""
    l, t, r, b = [float(v) for v in box]
    size = int(max(r - l, b - t) * scale + 1e-7); s = dsize / (size + 1e-7); cx, cy = (l + r) / 2, (t + b) / 2
    return np.array([[s, 0, dsize / 2 - s * cx], [0, s, dsize / 2 - s * cy], [0, 0, 1]], np.float32)


def square_box(pts):
    lt, rb = pts.min(0), pts.max(0); c = (lt + rb) / 2; m = float((rb - lt).max())
    return [c[0] - m / 2, c[1] - m / 2, c[0] + m / 2, c[1] + m / 2]


def aa2R(a):
    import cv2
    return cv2.Rodrigues(np.asarray(a, np.float64).reshape(3, 1))[0]


def R2aa(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].reshape(3)


class Converter:
    def __init__(self, refit_dir, device="cuda:0", matte=True):
        import torch, cv2
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fit_dress4d import Model, V11, N2020
        self.torch, self.cv2 = torch, cv2
        self.dev = torch.device(device); self.refit_dir = Path(refit_dir)
        self.m20 = Model(N2020, 200, 50, self.dev)
        self.mg = {g: Model(V11 / f"SMPLX_{g.upper()}.npz", 10, 10, self.dev) for g in ("male", "female")}
        self.faces = torch.as_tensor(self.m20.faces, device=self.dev)
        # head (SMPL-X -> FLAME ids minus FLAME lower head / neck) and MANO vertex ids
        s2f = np.load(ROOT / "assets/SMPLX/SMPL-X__FLAME_vertex_ids.npy")
        low = np.load(ROOT / "assets/FLAME/selected_lowerhead.npy") if (ROOT / "assets/FLAME/selected_lowerhead.npy").exists() else np.zeros(0, int)
        self.head_ids = s2f[np.setdiff1d(np.arange(len(s2f)), np.asarray(low).reshape(-1))]
        mano = pickle.load(open(ROOT / "assets/SMPLX/MANO_SMPLX_vertex_ids.pkl", "rb"), encoding="latin1")
        self.lh_ids, self.rh_ids = np.asarray(mano["left_hand"]), np.asarray(mano["right_hand"])
        self._refit = {}
        if not matte: return
        # StyleMatte (as sprint_avatar_fit.SprintFitter)
        import importlib.util
        spec = importlib.util.spec_from_file_location("ehm_stylematte", EHMT / "src/modules/human_matting/stylematte.py")
        sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
        cwd = Path.cwd(); os.chdir(ROOT); self.matte = sm.StyleMatte(); os.chdir(cwd)
        self.matte.load_state_dict(torch.load(EHMT / "pretrained/matting/stylematte_synth.pt", map_location="cpu", weights_only=True))
        self.matte = self.matte.to(self.dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        self._refit = {}

    # ------------------------------------------------------------------ helpers
    def matting(self, imgs_u8):
        """list of (S, S, 3) uint8 RGB -> (B, S, S) alpha float32 numpy."""
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(np.stack(imgs_u8)).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            a = self.matte((x - self.mean) / self.std)[:, 0]
        return a.clamp(0, 1).cpu().numpy()

    def refit(self, subj):
        if subj not in self._refit:
            self._refit = {subj: dict(np.load(self.refit_dir / f"{subj}.npz", allow_pickle=True))}
        return self._refit[subj]

    def silhouettes(self, v2d, z, S=256):
        """v2d (B, V, 2) continuous crop px (1024), z (B, V) depth -> (B, S, S) bool silhouette (pytorch3d rasteriser)."""
        t = self.torch
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        v = t.as_tensor(v2d, device=self.dev).float() * (S / CROP)
        xn = -(v[..., 0] / S * 2 - 1); yn = -(v[..., 1] / S * 2 - 1)
        verts = t.stack([xn, yn, t.as_tensor(z, device=self.dev).float()], -1)
        mesh = Meshes(verts=list(verts), faces=[self.faces] * len(verts))
        p2f, _, _, _ = rasterize_meshes(mesh, image_size=S, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
        return (p2f[..., 0] >= 0).cpu().numpy()

    # ------------------------------------------------------------------ one take
    def convert_take(self, T, out_root, n_gt=10, timing=None):
        t, cv2 = self.torch, self.cv2
        T = Path(T); subj, outfit, take = T.parts[-3], T.parts[-2], T.parts[-1]
        name = f"{subj}_{outfit}_{take}"; out = Path(out_root) / name; tmp = Path(out_root) / f".{name}.tmp"
        if (out / "stats.json").exists(): return "skip"
        if tmp.exists():
            import shutil; shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        tm = {"read": 0.0, "gpu_mesh": 0.0, "matte1": 0.0, "crop": 0.0, "matte2": 0.0, "checks": 0.0, "write": 0.0}; t0 = time.perf_counter()
        bi = pickle.load(open(T / "basic_info.pkl", "rb")); gender = bi["gender"]
        frames = list(bi["scan_frames"])[::STRIDE]
        cams = pickle.load(open(T / "Capture" / "cameras.pkl", "rb"))
        R = self.refit(subj); sel = np.flatnonzero(R["take"] == name)
        fr_r = list(R["frame"][sel]); assert fr_r == frames, (name, fr_r[:3], frames[:3])
        # source (gendered) params
        P = [pickle.load(open(T / "SMPLX" / f"mesh-f{fr}_smplx.pkl", "rb")) for fr in frames]
        P = {k: np.stack([p[k] for p in P]).astype(np.float32) for k in P[0]}
        from concurrent.futures import ThreadPoolExecutor
        def rd(a):
            c, fr = a
            im = cv2.cvtColor(cv2.imread(str(T / "Capture" / c / "images" / f"capture-f{fr}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mk = cv2.imread(str(T / "Capture" / c / "masks" / f"mask-f{fr}.png"), cv2.IMREAD_GRAYSCALE)
            return im, mk
        jobs = [(c, fr) for c in CAMS for fr in frames]
        with ThreadPoolExecutor(8) as ex: IM = dict(zip(jobs, ex.map(rd, jobs)))
        tm["read"] += time.perf_counter() - t0; t1 = time.perf_counter()
        # meshes
        from fit_dress4d import gendered_verts
        with t.no_grad():
            Vs, _ = gendered_verts(self.mg[gender], P, self.dev)
            B = len(frames); dev = self.dev
            beta = t.as_tensor(R["betas"], device=dev)[None]; exp = t.as_tensor(R["exp"][sel], device=dev)
            pose = t.cat([t.as_tensor(R[k][sel], device=dev).reshape(B, -1, 3) for k in
                          ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose")], 1)
            tr = t.as_tensor(R["transl"][sel], device=dev)
            Vn, _ = self.m20(beta, exp, pose, tr)
            vsh = self.m20.vt[None] + t.einsum("vcs,bs->bvc", self.m20.sd, t.cat([beta.expand(B, -1), exp], 1))
            J0 = t.einsum("v,bvc->bc", self.m20.J[0], vsh)
        Vs, Vn, J0 = Vs.cpu().numpy().astype(np.float64), Vn.cpu().numpy().astype(np.float64), J0.cpu().numpy().astype(np.float64)
        tm["gpu_mesh"] += time.perf_counter() - t1
        records, keys, stats, crops, alphas, gt_rows = {}, [], [], {}, {}, []
        for c in CAMS:
            K = cams[c]["intrinsics"].astype(np.float64); E = cams[c]["extrinsics"].astype(np.float64); Rc, tc = E[:, :3], E[:, 3]
            Kinv = np.linalg.inv(K)
            # pass 1 StyleMatte (box only): whole frame letterboxed to 512 (batch over frames of this camera, chunks of 8)
            t1 = time.perf_counter()
            H0, W0 = IM[(c, frames[0])][0].shape[:2]; L0 = max(H0, W0); s0 = P1 / L0
            ox, oy = (L0 - W0) / 2, (L0 - H0) / 2
            Mlb = np.array([[s0, 0, s0 * ox], [0, s0, s0 * oy]], np.float64)
            A1 = []
            for i in range(0, B, 8):
                A1 += list(self.matting([cv2.warpAffine(IM[(c, fr)][0], Mlb, (P1, P1), flags=cv2.INTER_AREA) for fr in frames[i:i + 8]]))
            tm["matte1"] += time.perf_counter() - t1; t1 = time.perf_counter()
            cache = []
            for i, fr in enumerate(frames):
                img, pmask = IM[(c, fr)]
                Xs = Vs[i] @ Rc.T + tc; us = Xs @ K.T; us = us[:, :2] / us[:, 2:]
                # StyleMatte pass-1 pixels back in original px, components touching the projected mesh
                a1 = cv2.warpAffine((A1[i] > 0.5).astype(np.uint8), Mlb, (W0, H0), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                n, lab = cv2.connectedComponents(a1)
                ui = np.clip(np.round(us).astype(int), [0, 0], [W0 - 1, H0 - 1])
                keep = np.setdiff1d(np.unique(lab[ui[:, 1], ui[:, 0]]), [0])
                ys, xs = np.nonzero(np.isin(lab, keep)) if len(keep) else (np.zeros(0, int), np.zeros(0, int))
                pts = np.concatenate([us, np.stack([xs, ys], 1).astype(np.float64)], 0)
                # perspective crop (box measured in the rotated camera, re-centred twice)
                rays = np.c_[pts, np.ones(len(pts))] @ Kinv.T
                Rv = np.eye(3)
                for _ in range(3):
                    q = rays @ Rv.T; q = q[:, :2] / q[:, 2:]
                    lo, hi = q.min(0), q.max(0); cn = (lo + hi) / 2
                    Rv = rot_to(Rv.T @ np.array([cn[0], cn[1], 1.0])) if np.abs(cn).max() > 1e-9 else Rv
                q = rays @ Rv.T; q = q[:, :2] / q[:, 2:]; lo, hi = q.min(0), q.max(0)
                side = 1.1 * float((hi - lo).max()); fv = CROP / side; inv = 2 * fv / CROP
                Kv = np.array([[fv, 0, CROP / 2 - 0.5], [0, fv, CROP / 2 - 0.5], [0, 0, 1]])
                Hm = Kv @ Rv @ Kinv
                crop = cv2.warpPerspective(img, Hm, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                pm = cv2.warpPerspective(pmask, Hm, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                # 256-px map of crop pixels that come from inside the original image (truncated people: black border)
                valid = cv2.warpPerspective(np.full((H0, W0), 255, np.uint8), np.diag([256 / CROP, 256 / CROP, 1.0]) @ Hm, (256, 256),
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
                trunc = float(1.0 - ((us[:, 0] >= 0) & (us[:, 0] < W0) & (us[:, 1] >= 0) & (us[:, 1] < H0)).mean())
                # converted record
                Rg = aa2R(R["global_orient"][sel][i]); Rn = Rv @ Rc @ Rg
                tp = Rv @ Rc @ (J0[i] + R["transl"][sel][i]) + Rv @ tc - J0[i]
                proj = lambda X: np.stack([CROP / 2 + fv * X[:, 0] / X[:, 2], CROP / 2 + fv * X[:, 1] / X[:, 2]], 1)
                v_src = proj((Vs[i] @ Rc.T + tc) @ Rv.T)                       # source mesh, original camera -> crop
                Xn = (Vn[i] @ Rc.T + tc) @ Rv.T; v_cv = proj(Xn)               # refit mesh, same camera
                cache.append(dict(fr=fr, crop=crop, pm=pm, valid=valid, trunc=trunc, Rn=Rn, tp=tp, inv=inv, fv=fv, Rv=Rv, Hm=Hm, v_src=v_src, v_cv=v_cv, z=Xn[:, 2]))
            tm["crop"] += time.perf_counter() - t1; t1 = time.perf_counter()
            A2 = []
            for i in range(0, B, 4):
                A2 += list(self.matting([cc["crop"] for cc in cache[i:i + 4]]))
            tm["matte2"] += time.perf_counter() - t1; t1 = time.perf_counter()
            # record-level reprojection through GUAVA's camera formula (independent LBS of the record)
            with t.no_grad():
                gp = t.as_tensor(np.stack([R2aa(cc["Rn"]) for cc in cache]), device=dev).float()
                pose_r = pose.clone(); pose_r[:, 0] = gp
                Vr, _ = self.m20(beta, exp, pose_r, t.zeros(B, 3, device=dev))
            Vr = Vr.cpu().numpy().astype(np.float64)
            sil = self.silhouettes(np.stack([cc["v_cv"] for cc in cache]), np.stack([cc["z"] for cc in cache]))
            for i, cc in enumerate(cache):
                key = f"{c}_{cc['fr']}"; keys.append(key)
                Xr = Vr[i] + cc["tp"]; v_rec = np.stack([CROP / 2 + cc["inv"] * CROP / 2 * Xr[:, 0] / Xr[:, 2],
                                                         CROP / 2 + cc["inv"] * CROP / 2 * Xr[:, 1] / Xr[:, 2]], 1)
                a = A2[i]; am = a > 0.5; pmb = cc["pm"] > 127
                iou = float((am & pmb).sum() / max(1, (am | pmb).sum()))
                a256 = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA) > 0.5
                sl = sil[i]; slv = sl & cc["valid"]                       # precision inside the original image only (truncation)
                prec = float((slv & a256).sum() / max(1, slv.sum())); rec_ = float((sl & a256).sum() / max(1, a256.sum()))
                prec_all = float((sl & a256).sum() / max(1, sl.sum()))
                # part crops from projected refit vertices (crop px)
                def part(ids, sc):
                    M = box_crop_matrix(square_box(cc["v_cv"][ids]), PART, sc)
                    return {"M_o2c": M, "M_c2o": np.linalg.inv(M).astype(np.float32)}
                Mlr = np.diag([BODY_LR / CROP, BODY_LR / CROP, 1.0]).astype(np.float32); I3 = np.eye(3, dtype=np.float32)
                RT = np.concatenate([np.diag([-1.0, -1.0, 1.0]), (np.diag([-1.0, -1.0, 1.0]) @ cc["tp"])[:, None]], 1).astype(np.float32)
                z50 = np.zeros(50, np.float32)
                smplx = {"global_pose": R2aa(cc["Rn"]).astype(np.float32), "body_pose": R["body_pose"][sel][i].astype(np.float32),
                         "left_hand_pose": R["left_hand_pose"][sel][i].astype(np.float32), "right_hand_pose": R["right_hand_pose"][sel][i].astype(np.float32),
                         "jaw_pose": R["jaw_pose"][sel][i].reshape(1, 3).astype(np.float32),
                         "eye_pose": np.stack([R["leye_pose"][sel][i], R["reye_pose"][sel][i]]).astype(np.float32),
                         "exp": R["exp"][sel][i].astype(np.float32), "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}
                records[key] = {"smplx_coeffs": smplx,
                                "flame_coeffs": {"expression_params": z50, "jaw_params": np.zeros(3, np.float32), "pose_params": np.zeros(3, np.float32),
                                                 "neck_pose_params": np.zeros(3, np.float32), "eye_pose_params": np.zeros(6, np.float32),
                                                 "eyelid_params": np.zeros(2, np.float32), "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
                                "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3, "M_c2o-hd": I3.copy()},
                                "head_crop": part(self.head_ids, HEAD_SCALE), "left_hand_crop": part(self.lh_ids, HAND_SCALE),
                                "right_hand_crop": part(self.rh_ids, HAND_SCALE),
                                "invtanfov": float(cc["inv"]), "view": c,
                                "dress4d": {"subject": subj, "outfit": outfit, "take": take, "frame": cc["fr"], "gender": gender, "trunc_frac": cc["trunc"],
                                            "H_orig_to_crop": cc["Hm"].astype(np.float64), "R_virtual": cc["Rv"]}}
                inimg = ((cc["v_src"] >= 0) & (cc["v_src"] <= CROP)).all(1).mean()
                stats.append({"key": key, "iou_provided": iou, "mesh_prec": prec, "mesh_prec_incl_outside_image": prec_all, "mesh_rec": rec_,
                              "trunc_frac": cc["trunc"], "invtanfov": float(cc["inv"]),
                              "reproj_rec_vs_src_px1024": float(np.linalg.norm(v_rec - cc["v_src"], axis=1).mean()),
                              "reproj_rec_vs_refit_px1024": float(np.linalg.norm(v_rec - cc["v_cv"], axis=1).mean()),
                              "src_in_crop": float(inimg)})
                crops[key] = cc["crop"]; alphas[key] = a
                gt_rows.append((key, cc["v_src"].astype(np.float32), v_rec.astype(np.float32), float(cc["inv"])))
            tm["checks"] += time.perf_counter() - t1
        t1 = time.perf_counter()
        # write clip
        import lmdb, torchvision
        with open(tmp / "optim_tracking_ehm.pkl", "wb") as f: pickle.dump(records, f)
        with open(tmp / "id_share_params.pkl", "wb") as f:
            pickle.dump({"smplx_shape": R["betas"].astype(np.float32).reshape(1, 200), "flame_shape": np.zeros((1, 300), np.float32),
                         "left_mano_shape": np.zeros((1, 10), np.float32), "right_mano_shape": np.zeros((1, 10), np.float32),
                         "head_scale": np.ones((1, 3), np.float32), "hand_scale": np.ones((1, 3), np.float32),
                         "joints_offset": np.zeros((1, 55, 3), np.float32)}, f)
        (tmp / "videos_info.json").write_text(json.dumps({name: {"frames_num": len(keys), "frames_keys": keys}}, indent=1))
        env = lmdb.open(str(tmp / "img_lmdb"), map_size=1 << 33); txn = env.begin(write=True)
        def enc(chw):
            return bytes(torchvision.io.encode_jpeg(chw.to(t.uint8), quality=95).numpy().tobytes())
        def encode_row(k):
            im = enc(t.from_numpy(crops[k]).permute(2, 0, 1).contiguous())
            mk = enc(t.from_numpy((np.clip(alphas[k], 0, 1) * 255).round().astype(np.uint8))[None].expand(3, -1, -1).contiguous())
            return k, im, mk
        with ThreadPoolExecutor(4) as ex:
            for k, im, mk in ex.map(encode_row, keys):
                txn.put(f"{k}/body_image".encode(), im); txn.put(f"{k}/body_mask".encode(), mk)
        txn.commit(); env.close()
        idx = np.unique(np.linspace(0, len(gt_rows) - 1, min(n_gt, len(gt_rows))).round().astype(int))
        np.savez(tmp / "gt_proj.npz", frame_key=np.array([gt_rows[j][0] for j in idx]), v2d_1024=np.stack([gt_rows[j][1] for j in idx]),
                 v2d_conv_1024=np.stack([gt_rows[j][2] for j in idx]), invtanfov=np.array([gt_rows[j][3] for j in idx]),
                 convention=np.array("continuous crop px: pixel i spans [i, i+1); = 1024 x GUAVA normalised uv; v2d_1024 = source "
                                     "gendered SMPL-X v1.1 mesh with the original 4D-DRESS camera; v2d_conv_1024 = converted record"))
        tm["write"] += time.perf_counter() - t1; tm["total"] = time.perf_counter() - t0
        summ = {"clip": name, "subject": subj, "outfit": outfit, "take": take, "gender": gender, "frames": len(frames), "rows": len(keys),
                "views": CAMS, "timing_s": tm, "rows_stats": stats}
        (tmp / "stats.json").write_text(json.dumps(summ))
        if out.exists():
            import shutil; shutil.rmtree(out)
        os.rename(tmp, out)
        if timing is not None: timing.append(tm)
        return summ


def worker(gpu, takes, refit, out, logf):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch, cv2
    torch.set_num_threads(4); cv2.setNumThreads(4)
    cv = Converter(refit, "cuda:0")
    with open(logf, "a") as lg:
        for T in takes:
            for attempt in range(4):                       # shared GPUs: retry on CUDA OOM after a pause
                try:
                    t0 = time.time(); r = cv.convert_take(DATA / T, out)
                    msg = "skip" if r == "skip" else f"{r['rows']} rows {time.time() - t0:.1f}s"
                    lg.write(f"{time.strftime('%H:%M:%S')} gpu{gpu} {T} {msg} peakGB {torch.cuda.max_memory_allocated() / 1e9:.1f}\n"); lg.flush()
                    break
                except Exception as e:
                    oom = "out of memory" in str(e)
                    lg.write(f"{time.strftime('%H:%M:%S')} gpu{gpu} {T} FAILED attempt {attempt}{' (OOM, retry)' if oom and attempt < 3 else ''}\n"
                             f"{traceback.format_exc() if not oom else ''}\n"); lg.flush()
                    torch.cuda.empty_cache()
                    if not oom: break
                    time.sleep(120)


def refit_summary(refit):
    """Per-subject refit results written so far (fit_dress4d.py <subj>.json)."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(refit).glob("[0-9]*.json"))}


def write_index(out, refit):
    out = Path(out); split, gender = split_of()
    summ = refit_summary(refit)
    dropped = sorted(s for s, v in summ.items() if not v["pass"])
    clips = []
    for d in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith(".")):
        st = d / "stats.json"
        if not st.exists(): continue
        s = json.loads(st.read_text())
        clips.append({"name": d.name, "subject": s["subject"], "gender": s["gender"], "outfit": s["outfit"], "take": s["take"],
                      "split": split[s["subject"]], "frames": s["rows"], "scan_frames": s["frames"], "views": s["views"]})
    idx = {"source": "dress4d", "licence": LICENCE, "clips": clips,
           "notes": ("One clip per subject/outfit/take; rows = every 5th scan frame x 4 calibrated cameras, frame_key '<cam>_<frame>', "
                     "record['view'] = camera id. Pairs for GUAVA-FB: source and target from DIFFERENT cameras of the same subject (any "
                     "take / outfit). id_share_params.smplx_shape = the subject's single SMPLX_NEUTRAL_2020 shape (refit of the dataset's "
                     "gendered SMPL-X v1.1). Validation subjects (2 female + 2 male, lowest SHA-1 of the id): "
                     + ", ".join(sorted(s for s in split if split[s] == "val")) + ". Subjects failing the < 5 mm refit check (dropped, no "
                     "clips written): " + (", ".join(dropped) if dropped else "none") + ".")}
    (out / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--takes", nargs="*", default=None); ap.add_argument("--gpus", default="2"); ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.index_only:
        idx = write_index(a.out, a.refit); print(len(idx["clips"]), "clips"); return
    summ = refit_summary(a.refit)
    ok = {s for s, v in summ.items() if v["pass"]}
    takes = a.takes or sorted(str(Path(p).relative_to(DATA)) for p in glob.glob(str(DATA / "*" / "*" / "Take*")))
    takes = [x for x in takes if x.split("/")[0] in ok]
    if a.limit: takes = takes[:a.limit]
    gpus = [g for g in a.gpus.split(",") for _ in range(a.workers_per_gpu)]
    import multiprocessing as mp
    ctx = mp.get_context("spawn"); logf = Path(a.out).parent.parent / "work" / "dress4d" / "convert.log"; logf.parent.mkdir(parents=True, exist_ok=True)
    # interleave takes (largest subjects spread over workers)
    ps = [ctx.Process(target=worker, args=(g, takes[i::len(gpus)], a.refit, a.out, str(logf))) for i, g in enumerate(gpus)]
    for p in ps: p.start()
    for p in ps: p.join()
    write_index(a.out, a.refit)


if __name__ == "__main__":
    main()
