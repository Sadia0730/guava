"""PublicBench-v2 X-Humans part (PREREG "PublicBench-v2"): subjects 00028, 00034, 00087, official train / test split, in
AvatarTest-v1's formats so that the existing renderers, baseline adapters and scorer run unchanged:
  E3/data/xhumans/<subject>/    GUAVA tracked format (FORMAT.md): SMPL-X 2020 records FITTED to the GT SMPL-X v1.1 mesh + FLAME fields
  E3/gt512/xhumans/<subject>/   <key>.png for the test keys: image x depth mask at 512 on black (8 bit)
  E3/base/xhumans/<subject>/    baseline package (phase3_dnt/baselines/CONTRACT.md) + crop_h.npz + smplx_gt.npz (the GT itself)
  E3/reports/                   conversion log
E3 = O/eval/publicbench_v2 (O = the GUAVA-FB output root on /vault). X-Humans root: /vault/research/COS/VIRLab/ubx858/ubx858/X_Humans.

Stages
  split  -> eval_split/publicbench_v2_xhumans.json (AvatarTest-v1 split format). CPU only; reads file lists, cameras and GT only.
  run    -> converts the split's clips (GPU: fits). Resumable (a clip with base/.../frames.json is skipped).

Protocol as implemented
  frames  = every frame of every take; key frame_<take * 10000 + n> (n = the 1-based number of render/image/color_<n>.png =
            SMPLX/mesh-f<n>_smplx.pkl); recon = all frames of the official train takes, test = all frames of the official test takes;
            source = the first training frame (lowest train take, lowest n).
  GT      = the dataset's SMPL-X registration through SMPL-X v1.1 of the subject's gender (gender.txt), betas = mean_shape_smplx.npy
            (= every pkl's betas), hands + the v1.1 hand mean (flat_hand_mean False: reproduces the shipped .ply meshes to 0.00 mm,
            checked per clip on 2 frames), 10 expression coefficients. Camera: x_cam = extrinsic[i] @ [x_world; 1], K = intrinsic.
  SMPL-X 2020 (SMPLX_NEUTRAL_2020, 200 betas, 50 exp, flat hands) FITTED to the GT mesh (same topology, all 10,475 vertices): one
            betas200 per subject (ridge init on the rest shape, then fitted jointly on 64 frames), per-frame root, 21 body joints,
            jaw, eyes, 30 hand joints, exp50 and translation (Adam); initialised by the copy conversion (poses copied, exp50 =
            v11_expr10_to_2020_exp50 @ expr10). Residual (mm) reported per frame / clip.
  Crop / records / FLAME fields: convert_test.py's UBody path: perspective crop aimed at and sized by the projected GT mesh box
            clipped to the image, 1.1 x, 1024 px; FLAME shape = maps()['Wf'] @ betas200, expression = exp50, jaw copied.
  Mask    = rendered depth < DEPTH_BG (background depth ~1000, body ~6 m; exact for these renders of scans), warped like the image
            (bilinear) -> alpha; NOT StyleMatte. GT = image x alpha on black.
  smpl.npz (CONTRACT: neutral SMPL, one betas10 per clip, crop-camera frame) = neutral SMPL fitted to the GT mesh mapped by the
            SMPL-X -> SMPL correspondence (the dataset's own SMPL files are not used).

Usage (env: source phase3_dnt/guava_fb/env_bedlam2.sh):
  $PY convert_xhumans.py split
  CUDA_VISIBLE_DEVICES=0 $PY convert_xhumans.py run [--clips xhumans/00028,...]"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import convert_bedlam2 as cb                                                                       # noqa: E402
import convert_publicbench as CP                                                                   # noqa: E402
import convert_test as CT                                                                          # noqa: E402

SPLIT = HERE / "publicbench_v2_xhumans.json"
E3 = cb.O / "eval" / "publicbench_v2"
TMP = Path("/dev/shm/ubx858_eval/xhumans_tmp")
XH = Path("/vault/research/COS/VIRLab/ubx858/ubx858/X_Humans")
SUBJECTS = ("00028", "00034", "00087")
V11_DIR = Path("/raid/ubx858/datasets/models/smplx_v1_1/models/smplx")
CROP, PART, BODY_LR = cb.CROP, cb.PART, cb.BODY_LR
DEPTH_BG = 100.0                                                                                    # metres; background is ~1000
SHAPE_FRAMES, SHAPE_ITERS, POSE_ITERS, CHUNK = 64, 1500, 600, 256
README_COUNTS = {"00028": (1444, 448), "00034": (1411, 300), "00087": (1780, 360)}                  # README (train, test) frames
LICENCE = ("X-Humans (Shen et al., CVPR 2023; ETH Zurich AIT, https://xhumans.ait.ethz.ch): non-commercial academic research only; "
           "no redistribution of the data or of converted data; never used for training. Body model: SMPL-X (MPI non-commercial licence).")


def takes(subject, mode):
    return sorted((int(p.name[4:]) for p in (XH / subject / mode).iterdir() if p.name.startswith("Take")))


def take_frames(subject, mode, t):
    d = XH / subject / mode / f"Take{t}"
    im = sorted(int(Path(p).stem.split("_")[1]) for p in glob.glob(str(d / "render/image/color_*.png")))
    px = sorted(int(Path(p).stem.split("-f")[1].split("_")[0]) for p in glob.glob(str(d / "SMPLX/mesh-f*_smplx.pkl")))
    assert im == px, (subject, mode, t, len(im), len(px))
    return im


def key_of(t, n): return t * 10000 + n


def split(a):
    clips = []
    for s in SUBJECTS:
        tr_t, te_t = takes(s, "train"), takes(s, "test")
        tr = [key_of(t, n) for t in tr_t for n in take_frames(s, "train", t)]
        te = [key_of(t, n) for t in te_t for n in take_frames(s, "test", t)]
        assert not set(tr) & set(te) and (len(tr), len(te)) == README_COUNTS[s], (s, len(tr), len(te))
        cam = np.load(XH / s / "train" / f"Take{tr_t[0]}" / "render/cameras.npz")
        im0 = XH / s / "train" / f"Take{tr_t[0]}" / "render/image" / f"color_{take_frames(s, 'train', tr_t[0])[0]:06d}.png"
        import cv2
        h, w = cv2.imread(str(im0)).shape[:2]
        clips.append({"id": f"xhumans/{s}", "dataset": "xhumans", "part": "full body, rendered RGB of textured scans (moving virtual camera, white background)",
                      "identity": f"xhumans_{s}", "clip": s, "gender": (XH / s / "gender.txt").read_text().strip(),
                      "video": str(XH / s), "fps": 30, "wh": [w, h],
                      "frame_index": "frame = take * 10000 + n, n = 1-based number of render/image/color_<n>.png (= SMPLX/mesh-f<n>_smplx.pkl)",
                      "published_split": {"source": "X-Humans official train / test folders (README table)", "train_takes": tr_t, "test_takes": te_t,
                                          "readme_train_test_frames": list(README_COUNTS[s])},
                      "gt": {"kind": "X-Humans SMPL-X registration (SMPL-X v1.1, gendered, flat_hand_mean False, 10 expr), betas = mean_shape_smplx.npy",
                             "body_model": str(V11_DIR / f"SMPLX_{(XH / s / 'gender.txt').read_text().strip().upper()}.npz"),
                             "camera": {"K": cam["intrinsic"].tolist(), "extrinsic": "per frame, render/cameras.npz (x_cam = E [x_world; 1])"}},
                      "frames": sorted(tr + te), "n_frames": len(tr) + len(te), "source_frame": tr[0], "n_recon": len(tr), "recon_frames": tr, "test_frames": te,
                      "checks": {"train_test_disjoint": True, "counts_eq_readme": True, "images_eq_smplx_pkls": True}})
    comp = [{"dataset": "xhumans", "clips": len(clips), "identities": len(clips), "frames": sum(c["n_frames"] for c in clips),
             "test_frames": sum(len(c["test_frames"]) for c in clips), "recon_frames": sum(c["n_recon"] for c in clips),
             "pool": "the 3 X-Humans subjects of ExAvatar's X-Humans evaluation (user decision)"}]
    J = {"name": "PublicBench-v2 (X-Humans part)", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "phase3_dnt/eval_split/convert_xhumans.py split",
         "git_head_at_build": CP.git_head(),
         "why": "PREREG 'PublicBench-v2' = PublicBench-v1 (publicbench_v1.json, unchanged) + these X-Humans subjects; same fairness rule.",
         "protocol": {"source": "source_frame = the first training frame; single-image methods see only this frame",
                      "recon": "recon_frames = all frames of the official TRAIN takes; per-subject methods train on these only",
                      "test": "test_frames = all frames of the official TEST takes; every method is scored on exactly these",
                      "driving": "GT rows: the dataset's SMPL-X of each test frame (SMPL-X 2020 fitted to it; SMPL fitted for SMPL methods); SPRINT rows: SPRINT",
                      "mask": "rendered depth < 100 (exact); GT = image x mask on black",
                      "crop": "AvatarTest-v1 perspective crop: projected GT mesh box clipped to the image, 1.1 x, 1024 px; scored at 512"},
         "composition": comp, "excluded_clips": [], "licences": {"xhumans": LICENCE}, "clips": clips}
    SPLIT.write_text(json.dumps(J, indent=1))
    print(json.dumps({"clips": len(clips), "composition": comp, "sha256": CP.sha256(SPLIT)}, indent=1))


def frame_path(s, mode, t, n, kind):
    d = XH / s / mode / f"Take{t}"
    return {"img": d / "render/image" / f"color_{n:06d}.png", "depth": d / "render/depth" / f"depth_{n:06d}.tiff",
            "pkl": d / "SMPLX" / f"mesh-f{n:05d}_smplx.pkl", "ply": d / "SMPLX" / f"mesh-f{n:05d}_smplx.ply", "cam": d / "render/cameras.npz"}[kind]


def read_ply(p):
    b = open(p, "rb").read(); h = b.index(b"end_header\n") + 11
    return np.frombuffer(b[h:h + 10475 * 12], dtype="<f4").reshape(-1, 3).astype(np.float64)


def load_gt(c):
    """Per frame (c['frames'] order): GT SMPL-X v1.1 params (world), camera R, t (world -> cam), K; mode / take / n."""
    s = c["clip"]; tr = set(c["recon_frames"]); rows = []
    cams = {}
    for f in c["frames"]:
        t, n = divmod(f, 10000); mode = "train" if f in tr else "test"
        if (mode, t) not in cams:
            z = np.load(frame_path(s, mode, t, 1, "cam")); nums = take_frames(s, mode, t)
            cams[(mode, t)] = (z["intrinsic"].astype(np.float64), {m: z["extrinsic"][i].astype(np.float64) for i, m in enumerate(nums)})
            assert len(z["extrinsic"]) == len(nums)
        K, ex = cams[(mode, t)]
        p = pickle.load(open(frame_path(s, mode, t, n, "pkl"), "rb"))
        rows.append({"mode": mode, "take": t, "n": n, "K": K, "E": ex[n], **{k: np.asarray(v, np.float64).reshape(-1) for k, v in p.items()}})
    return rows


def fit_x2020(X, VS, init, b0, dev):
    """SMPL-X 2020 (GUAVA's lean layer, flat hands) fitted to GT SMPL-X meshes VS (n, 10475, 3), camera frame.
    init: pose165 (n, 165), exp50 (n, 50), transl (n, 3). Stage 1: betas + params on SHAPE_FRAMES frames; stage 2: per-frame params, betas fixed."""
    torch = X.torch; n = VS.shape[0]
    T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)                      # noqa: E731

    def run(idx, be, iters, fit_be):
        P = T(init["pose"][idx]).requires_grad_(True); ex = T(init["exp"][idx]).requires_grad_(True); tr = T(init["transl"][idx]).requires_grad_(True)
        params = [P, ex, tr] + ([be] if fit_be else [])
        opt = torch.optim.Adam(params, lr=0.005); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters); V_ = VS[idx]
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            V, _ = X.X(P, be.expand(len(idx), -1), ex); d = (V + tr[:, None] - V_).norm(dim=-1)
            (d.mean() + 1e-6 * be.square().sum() + 1e-6 * ex.square().sum()).backward(); opt.step(); sch.step()
        with torch.no_grad():
            V, _ = X.X(P, be.expand(len(idx), -1), ex); d = (V + tr[:, None] - V_).norm(dim=-1) * 1000
        return P.detach(), ex.detach(), tr.detach(), d
    be = T(b0)[None].requires_grad_(True)
    sub = np.unique(np.round(np.linspace(0, n - 1, min(SHAPE_FRAMES, n))).astype(int))
    run(sub, be, SHAPE_ITERS, True); be = be.detach()
    pose, exp, transl, res, p95, mx = [], [], [], [], [], []
    for i in range(0, n, CHUNK):
        idx = np.arange(i, min(n, i + CHUNK)); P, ex, tr, d = run(idx, be, POSE_ITERS, False)
        pose.append(P.cpu().numpy()); exp.append(ex.cpu().numpy()); transl.append(tr.cpu().numpy())
        res.append(d.mean(1).cpu().numpy()); p95.append(torch.quantile(d, 0.95, dim=1).cpu().numpy()); mx.append(d.max(1).values.cpu().numpy())
    cat = lambda L: np.concatenate(L).astype(np.float64)                                            # noqa: E731
    return {"pose": cat(pose), "exp": cat(exp), "transl": cat(transl), "betas": be[0].cpu().numpy().astype(np.float64),
            "res_mm": cat(res), "res_mm_p95": cat(p95), "res_mm_max": cat(mx)}


def convert_clip(c, X, M, J, log):
    import cv2, torch, torchvision
    from smplx_lean import SMPLXLean
    t00 = time.time(); ds, name = c["dataset"], c["clip"]; frames = c["frames"]; n = len(frames); dev = X.dev
    keys = [f"frame_{f:06d}" for f in frames]; G = cb.geo(); s = name
    gt = load_gt(c); g = c["gender"]
    L11 = SMPLXLean(str(V11_DIR / f"SMPLX_{g.upper()}.npz"), 10, n_exp=10, dtype=torch.float32).to(dev)
    d11 = np.load(V11_DIR / f"SMPLX_{g.upper()}.npz", allow_pickle=True); hm = np.concatenate([d11["hands_meanl"], d11["hands_meanr"]]).astype(np.float64)
    stack = lambda k: np.stack([r[k] for r in gt])                                                  # noqa: E731
    pose11 = np.concatenate([stack("global_orient"), stack("body_pose"), stack("jaw_pose"), stack("leye_pose"), stack("reye_pose"),
                             stack("left_hand_pose") + hm[:45], stack("right_hand_pose") + hm[45:]], 1)
    b10 = stack("betas"); e10 = stack("expression"); trw = stack("transl")
    assert np.abs(b10 - b10[0]).max() < 1e-6, "betas vary per frame"
    f32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)                    # noqa: E731
    # 1. GT mesh (world) -> camera frame; .ply reproduction check on the first / last frame
    VW = []
    with torch.no_grad():
        for i in range(0, n, CHUNK):
            V, _ = L11(f32(pose11[i:i + CHUNK]), f32(b10[i:i + CHUNK]), f32(e10[i:i + CHUNK])); VW.append(V + f32(trw[i:i + CHUNK])[:, None])
    VW = torch.cat(VW); Rc = np.stack([r["E"][:3, :3] for r in gt]); tc = np.stack([r["E"][:3, 3] for r in gt])
    VS0 = torch.einsum("nij,nvj->nvi", f32(Rc), VW) + f32(tc)[:, None]
    ply_mm = []
    for i in (0, n - 1):
        r = gt[i]; P_ = read_ply(frame_path(s, r["mode"], r["take"], r["n"], "ply"))
        ply_mm.append(float(np.linalg.norm(VW[i].cpu().numpy().astype(np.float64) - P_, axis=1).max() * 1000))
    assert max(ply_mm) < 1.0, ply_mm
    # 2. SMPL-X 2020 fit (camera frame). Init: copy conversion; root folded into the camera frame, translation by mean offset
    t1 = time.time()
    with torch.no_grad():
        v11r, _ = L11(torch.zeros(1, 165, device=dev), f32(b10[:1]), None)
    T20 = X.T20; B20 = X.B20.reshape(10475 * 3, 200)
    A = np.concatenate([B20, np.tile(np.eye(3), (10475, 1))], 1); y = (v11r[0].cpu().numpy().astype(np.float64) - T20).reshape(-1)
    reg = np.sqrt(CP.RIDGE) * np.concatenate([np.eye(200), np.zeros((200, 3))], 1)
    b0 = np.linalg.lstsq(np.concatenate([A, reg]), np.concatenate([y, np.zeros(200)]), rcond=None)[0][:200]
    pose0 = pose11.copy(); pose0[:, :3] = CP.aa_np(Rc @ CP.R_np(pose11[:, :3]))
    exp0 = e10 @ M["Wex"].T
    with torch.no_grad():
        V0 = torch.cat([X.X(f32(pose0[i:i + CHUNK]), f32(b0)[None].expand(len(pose0[i:i + CHUNK]), -1), f32(exp0[i:i + CHUNK]))[0] for i in range(0, n, CHUNK)])
        tr0 = (VS0 - V0).mean(1).cpu().numpy()
    del V0
    fx = fit_x2020(X, VS0, {"pose": pose0, "exp": exp0, "transl": tr0}, b0, dev); t_fit = time.time() - t1
    b200, pose, exp50, t = fx["betas"], fx["pose"], fx["exp"], fx["transl"]
    # 3. float64 GUAVA-layer geometry of the fitted params (chunks), records as convert_test.ub_geometry
    VN, JN = [], []
    for i in range(0, n, CHUNK):
        a_, b_ = G["N"](torch.tensor(pose[i:i + CHUNK]), torch.tensor(b200)[None].expand(len(pose[i:i + CHUNK]), -1), torch.tensor(exp50[i:i + CHUNK]))
        VN.append(a_.numpy()); JN.append(b_.numpy())
    VN, JN = np.concatenate(VN), np.concatenate(JN)
    VSn = VS0.cpu().numpy().astype(np.float64); W, H = c["wh"]; geo = []
    for i in range(n):
        K = gt[i]["K"]; fxx, fyy = K[0, 0], K[1, 1]; f = float(fxx); Ki = np.linalg.inv(K)
        XS = VSn[i]; XN = VN[i] + t[i]
        res_mm = float(np.linalg.norm(XN - XS, axis=1).mean() * 1000)
        p = XS @ K.T; p = p[:, :2] / p[:, 2:]
        lo = np.clip(p.min(0), 0, [W, H]); hi = np.clip(p.max(0), 0, [W, H])
        Rv = cb.rot_onto_z(Ki @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.0]))
        cor = np.array([[lo[0], lo[1], 1], [hi[0], lo[1], 1], [lo[0], hi[1], 1], [hi[0], hi[1], 1.0]]) @ Ki.T @ Rv.T
        q = f * cor[:, :2] / cor[:, 2:]
        Lc = cb.BODY_SCALE * float((q.max(0) - q.min(0)).max()); sc = CROP / Lc
        Kv = np.array([[f * sc, 0, CROP / 2], [0, f * sc, CROP / 2], [0, 0, 1.0]])
        Hm = Kv @ Rv @ Ki; invtanfov = 2 * f / Lc
        pj = lambda X3: (lambda u: (lambda w: w[:, :2] / w[:, 2:])((u / u[:, 2:]) @ Hm.T))(X3 @ K.T)   # noqa: E731
        v2d_src = pj(XN); v2d_gt = pj(XS)
        Rroot = Rv @ cv2.Rodrigues(pose[i, :3])[0]; J0x = JN[i, 0]; tp = Rv @ (J0x + t[i]) - J0x
        Dm = np.diag([-1.0, -1.0, 1.0]); RT = np.concatenate([Dm, (Dm @ tp)[:, None]], 1).astype(np.float32)
        Vmodel = (VN[i] - J0x) @ Rv.T + J0x
        v2d = cb.project_guava(Vmodel, RT.astype(np.float64), invtanfov)
        fold_px512 = float(np.linalg.norm(v2d - v2d_src, axis=1).mean() * 512 / CROP)
        fit_px512 = float(np.linalg.norm(v2d_src - v2d_gt, axis=1).mean() * 512 / CROP)
        pc = pose[i].astype(np.float32)
        smplx = {"global_pose": cb.aa_from_R(Rroot).astype(np.float32), "body_pose": pc[3:66].reshape(21, 3).copy(),
                 "jaw_pose": pc[66:69].reshape(1, 3).copy(), "eye_pose": pc[69:75].reshape(2, 3).copy(),
                 "left_hand_pose": pc[75:120].reshape(15, 3).copy(), "right_hand_pose": pc[120:165].reshape(15, 3).copy(),
                 "exp": exp50[i].astype(np.float32), "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}

        def part(ids, scale):
            pts = v2d[ids]; lt, rb = pts.min(0), pts.max(0)
            Mo = cb.box_crop_matrix([lt[0], lt[1], rb[0], rb[1]], PART, scale)
            return {"M_o2c": Mo, "M_c2o": np.linalg.inv(Mo).astype(np.float32)}, bool(((pts >= 0) & (pts < CROP)).all(1).mean() > 0.5)
        hc, hv = part(G["head"], cb.HEAD_SCALE); lc, lv = part(G["lhand"], cb.HAND_SCALE); rc, rv = part(G["rhand"], cb.HAND_SCALE)
        Mlr = np.diag([BODY_LR / CROP, BODY_LR / CROP, 1.0]).astype(np.float32); I3 = np.eye(3, dtype=np.float32)
        rec = {"smplx_coeffs": smplx,
               "flame_coeffs": {"expression_params": exp50[i].astype(np.float32), "jaw_params": pc[66:69].copy(),
                                "pose_params": np.zeros(3, np.float32), "neck_pose_params": np.zeros(3, np.float32),
                                "eye_pose_params": pc[69:75].copy(), "eyelid_params": np.zeros(2, np.float32),
                                "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
               "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3.copy(), "M_c2o-hd": I3.copy()},
               "head_crop": hc, "left_hand_crop": lc, "right_hand_crop": rc, "invtanfov": float(invtanfov), "view": "0"}
        geo.append({"rec": rec, "H": Hm, "Rv": Rv, "v2d_src": v2d_src.astype(np.float32), "v2d": v2d.astype(np.float32), "v2d_gt": v2d_gt.astype(np.float32),
                    "res_mm": res_mm, "res_px512": fold_px512, "fit_px512": fit_px512, "invtanfov": float(invtanfov), "crop_side_px": Lc,
                    "depth_m": float(XS[:, 2].mean()), "vis": {"head": hv, "lhand": lv, "rhand": rv},
                    "box_clipped": bool((p.min(0) < 0).any() or (p.max(0) > [W, H]).any())})
    t2 = time.time()
    # 4. original frames -> 1024 crops (+ JPEG q95); depth mask -> alpha (same warp)
    crops_j, alphas, iou_white = [], [], []
    for i, gi in enumerate(geo):
        r = gt[i]; im = cv2.cvtColor(cv2.imread(str(frame_path(s, r["mode"], r["take"], r["n"], "img"))), cv2.COLOR_BGR2RGB)
        dep = cv2.imread(str(frame_path(s, r["mode"], r["take"], r["n"], "depth")), cv2.IMREAD_UNCHANGED)
        assert im.shape[:2] == (H, W) and dep.shape[:2] == (H, W)
        m = (dep < DEPTH_BG).astype(np.uint8) * 255
        if i % 50 == 0: nw = im.astype(int).sum(-1) < 750; fm = m > 0; iou_white.append(float((nw & fm).sum() / max((nw | fm).sum(), 1)))
        cr = cv2.warpPerspective(im, gi["H"], (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        al = cv2.warpPerspective(m, gi["H"], (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        crops_j.append(torchvision.io.encode_jpeg(torch.from_numpy(cr).permute(2, 0, 1).contiguous(), quality=95).numpy().tobytes()); alphas.append(al)
    t3 = time.time()
    fs = M["Wf"] @ b200
    res = {"g": geo, "jpgs": crops_j, "betas200": b200}
    meta = {"id": c["id"], "dataset": ds, "clip": name, "frames_keys": keys, "missing_frames": [], "dropped_zmin": [],
            "res_mm": [x["res_mm"] for x in geo], "res_px512": [x["res_px512"] for x in geo], "fit_px512": [x["fit_px512"] for x in geo],
            "invtanfov": [x["invtanfov"] for x in geo], "crop_side_px": [x["crop_side_px"] for x in geo], "depth_m": [x["depth_m"] for x in geo],
            "part_visible": [x["vis"] for x in geo], "alpha_mean": [float(x.mean() / 255) for x in alphas],
            "shape_info": {"betas200_init": "ridge lstsq on the GT rest shape (all vertices), then fitted on 64 frames", "betas200_norm": float(np.linalg.norm(b200))},
            "times": {"fit": t_fit, "geom": t2 - t1 - t_fit, "decode_crop_mask": t3 - t2}, "licence": LICENCE,
            "note": "res_mm: SMPL-X 2020 fit vs GT SMPL-X v1.1 mesh (mm, all vertices); res_px512: camera-fold check; fit_px512: fit error projected; mask = depth"}
    CT.TMP = TMP
    mjpg = CT.write_tracked(E3 / "data" / ds / name, res, keys, alphas, fs, meta)
    # 5. base package (local, then one copy) + gt512
    dec = lambda b: torchvision.io.decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).float() / 255.0    # noqa: E731
    rs = lambda x: torchvision.transforms.functional.resize(x, (512, 512), antialias=True)                           # noqa: E731
    u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8)                                                      # noqa: E731
    LB = TMP / "base" / ds / name; LG = TMP / "gt512" / ds / name
    for p in (LB, LG):
        if p.exists(): shutil.rmtree(p)
    for p in (LB / "images", LB / "masks", LG): p.mkdir(parents=True, exist_ok=True)
    fset = set(f"frame_{f:06d}" for f in c["test_frames"]); msk512 = {}
    for k, jb, mb in zip(keys, crops_j, mjpg):
        im, ms = rs(dec(jb)), rs(dec(mb))
        torchvision.io.write_png(u8(im), str(LB / "images" / f"{k}.png")); torchvision.io.write_png(u8(ms[:1]), str(LB / "masks" / f"{k}.png"))
        msk512[k] = u8(ms[0]).numpy() > 127
        if k in fset: torchvision.io.write_png(u8(im * ms), str(LG / f"{k}.png"))
    R = [x["rec"]["smplx_coeffs"] for x in geo]; Dm = np.diag([-1.0, -1.0, 1.0])
    itf = np.array([x["invtanfov"] for x in geo])
    K5 = np.zeros((n, 3, 3)); K5[:, 0, 0] = K5[:, 1, 1] = 256 * itf; K5[:, 0, 2] = K5[:, 1, 2] = 256; K5[:, 2, 2] = 1
    sx = {"keys": np.array(keys), "betas": np.asarray(b200, np.float32),
          "global_orient": np.stack([r["global_pose"] for r in R]).astype(np.float32), "body_pose": np.stack([r["body_pose"] for r in R]).astype(np.float32),
          "jaw_pose": np.stack([np.asarray(r["jaw_pose"]).reshape(3) for r in R]).astype(np.float32),
          "leye_pose": np.stack([np.asarray(r["eye_pose"]).reshape(2, 3)[0] for r in R]).astype(np.float32),
          "reye_pose": np.stack([np.asarray(r["eye_pose"]).reshape(2, 3)[1] for r in R]).astype(np.float32),
          "left_hand_pose": np.stack([r["left_hand_pose"] for r in R]).astype(np.float32), "right_hand_pose": np.stack([r["right_hand_pose"] for r in R]).astype(np.float32),
          "expression": np.stack([r["exp"] for r in R]).astype(np.float32),
          "transl": np.stack([Dm @ np.asarray(r["camera_RT_params"], np.float64)[:, 3] for r in R]).astype(np.float32)}
    np.savez(LB / "smplx2020.npz", **sx); np.savez(LB / "cameras.npz", keys=np.array(keys), K=K5.astype(np.float32))
    pz = np.concatenate([sx["global_orient"], sx["body_pose"].reshape(-1, 63), sx["jaw_pose"], sx["leye_pose"], sx["reye_pose"],
                         sx["left_hand_pose"].reshape(-1, 45), sx["right_hand_pose"].reshape(-1, 45)], 1)
    q = []
    with torch.no_grad():
        for i in range(0, n, CHUNK):
            Vx, _ = X.X(torch.tensor(pz[i:i + CHUNK], device=dev), torch.tensor(sx["betas"], device=dev)[None].expand(len(pz[i:i + CHUNK]), -1),
                        torch.tensor(sx["expression"][i:i + CHUNK], device=dev))
            Vx = Vx + torch.tensor(sx["transl"][i:i + CHUNK], device=dev)[:, None]
            q.append((Vx[..., :2] / Vx[..., 2:]).cpu().numpy() * K5[i:i + CHUNK, 0, 0][:, None, None] + 256)
    q = np.concatenate(q)
    chk_px = float(np.abs(q - np.stack([x["v2d"] for x in geo]) / 2).max())
    # GT SMPL-X itself in the crop camera frame (smplx_gt.npz) and the contract SMPL (smpl.npz: neutral, one betas10) fitted to it
    Rv = np.stack([x["Rv"] for x in geo])
    VSv = torch.einsum("nij,nvj->nvi", f32(Rv), VS0)
    np.savez(LB / "smplx_gt.npz", keys=np.array(keys), gender=np.array(g), model=np.array(str(V11_DIR / f"SMPLX_{g.upper()}.npz")),
             betas=b10[0].astype(np.float32), pose_world_v11=pose11.astype(np.float32), expression=e10.astype(np.float32), transl_world=trw.astype(np.float32),
             world_to_crop_R=np.einsum("nij,njk->nik", Rv, Rc).astype(np.float64), world_to_crop_t=np.einsum("nij,nj->ni", Rv, tc).astype(np.float64),
             note=np.array("the dataset's GT SMPL-X v1.1 (gendered, hands incl. the hand mean, i.e. absolute) in its WORLD frame; "
                           "x_crop = world_to_crop_R x_world + world_to_crop_t (OpenCV axes, the crop camera of cameras.npz)"))
    VSm = X.mapped(VSv)
    with torch.no_grad(): _, jm = X.smpl_fwd("neutral", torch.zeros(1, 3, device=dev), torch.zeros(1, 69, device=dev), torch.zeros(1, 10, device=dev))
    go_v = CP.aa_np(Rv @ Rc @ CP.R_np(pose11[:, :3])); bp0 = np.concatenate([pose11[:, 3:66], np.zeros((n, 6))], 1)
    tr_init = VSm.mean(1).cpu().numpy() - jm[0].cpu().numpy()
    sms = [CP.fit_smpl_neutral(X, VSm[i:i + CHUNK], go_v[i:i + CHUNK], bp0[i:i + CHUNK], np.zeros(10), tr_init[i:i + CHUNK], CP.SMPL_ITERS, 0.02)
           for i in range(0, n, CHUNK)]
    # one betas10 per clip (contract): the median of the per-chunk betas, then pose / transl refitted with it
    bmed = np.median(np.stack([m["betas"] for m in sms]), 0)
    sm_go, sm_bp, sm_tr, sm_err = [], [], [], []
    torch_ = torch
    for i, m in zip(range(0, n, CHUNK), sms):
        go_, bp_, tr_ = (f32(m["go"]).requires_grad_(True), f32(m["bp"]).requires_grad_(True), f32(m["transl"]).requires_grad_(True))
        be_ = f32(bmed).reshape(1, 10); opt = torch_.optim.Adam([go_, bp_, tr_], lr=0.005)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            v, _ = X.smpl_fwd("neutral", go_, bp_, be_.expand(len(go_), -1), tr_); (v - VSm[i:i + CHUNK]).norm(dim=-1).mean().backward(); opt.step()
        with torch_.no_grad():
            v, _ = X.smpl_fwd("neutral", go_, bp_, be_.expand(len(go_), -1), tr_); sm_err.append(((v - VSm[i:i + CHUNK]).norm(dim=-1).mean(1) * 1000).cpu().numpy())
        sm_go.append(go_.detach().cpu().numpy()); sm_bp.append(bp_.detach().cpu().numpy()); sm_tr.append(tr_.detach().cpu().numpy())
    sm = {"go": np.concatenate(sm_go), "bp": np.concatenate(sm_bp), "transl": np.concatenate(sm_tr), "betas": bmed, "err_mm": np.concatenate(sm_err)}
    np.savez(LB / "smpl.npz", keys=np.array(keys), betas=np.asarray(sm["betas"], np.float32), global_orient=np.asarray(sm["go"], np.float32),
             body_pose=np.asarray(sm["bp"], np.float32).reshape(n, 23, 3), transl=np.asarray(sm["transl"], np.float32), fit_err_mm=np.asarray(sm["err_mm"], np.float32))
    np.savez(LB / "crop_h.npz", keys=np.array(keys), H=np.stack([x["H"] for x in geo]).astype(np.float64), orig_wh=np.array([W, H]),
             orig_src=np.array([str(frame_path(s, r["mode"], r["take"], r["n"], "img")) for r in gt]), K_orig=np.stack([r["K"] for r in gt]).astype(np.float64),
             Rv=Rv.astype(np.float64), invtanfov=itf, crop_side_px=np.array([x["crop_side_px"] for x in geo]),
             note=np.array("H: original image pixel (x, y, 1) -> 1024 crop pixel (cv2.warpPerspective(orig, H, (1024, 1024)), bilinear, border 0). "
                           "orig_src: the X-Humans color png (cv2.imread, BGR->RGB)."))
    # 6. checks: silhouette IoU (SMPL-X of smplx2020.npz; contract SMPL) vs the depth mask at 512; overlay
    with torch.no_grad():
        qn = []
        for i in range(0, n, CHUNK):
            vsm, _ = X.smpl_fwd("neutral", f32(sm["go"][i:i + CHUNK]), f32(sm["bp"][i:i + CHUNK]), f32(sm["betas"]).reshape(1, 10).expand(len(sm["go"][i:i + CHUNK]), -1), f32(sm["transl"][i:i + CHUNK]))
            qn.append((vsm[..., :2] / vsm[..., 2:]).cpu().numpy() * K5[i:i + CHUNK, 0, 0][:, None, None] + 256)
        qn = np.concatenate(qn)
    chk_i = np.unique(np.round(np.linspace(0, n - 1, 40)).astype(int))
    iou_x = [CP.raster_iou(q[i], X.faces_x, msk512[keys[i]])[0] for i in chk_i]
    iou_s = [CP.raster_iou(qn[i], X.faces_s, msk512[keys[i]])[0] for i in chk_i]
    tiles = []
    for i in np.unique(np.round(np.linspace(0, n - 1, 4)).astype(int)):
        im = cv2.cvtColor(cv2.imread(str(LB / "images" / f"{keys[i]}.png")), cv2.COLOR_BGR2RGB)
        for pts, col in ((q[i][::7], (0, 255, 0)), (qn[i][::5], (255, 0, 255))):
            for x_, y_ in pts.astype(int): cv2.circle(im, (int(x_), int(y_)), 1, col, -1)
        cv2.putText(im, keys[i], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA); tiles.append(im)
    cv2.imwrite(str(LB / "overlay.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    tk = [k for k in keys if k in fset]; rk = [f"frame_{f:06d}" for f in c["recon_frames"]]
    fj = {"dataset": ds, "clip": name, "id": c["id"], "keys": keys, "source": f"frame_{c['source_frame']:06d}", "recon": rk, "test": tk,
          "fps": c["fps"], "dropped_from_split": [],
          "note": "frame_XXXXXX = take * 10000 + n (render/image/color_<n>.png); recon = official X-Humans train takes, test = official test takes"}
    assert fj["source"] == rk[0] and set(tk) == fset and not (set(rk) & set(tk)) and len(rk) == c["n_recon"]
    (LB / "frames.json").write_text(json.dumps(fj, indent=1))
    for src, dst in ((LG, E3 / "gt512" / ds / name), (LB, E3 / "base" / ds / name)):
        tmp = dst.parent / (dst.name + ".partial")
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for p in sorted(src.rglob("*")):
            if p.is_dir() or p.name == "frames.json": continue
            (tmp / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, tmp / p.relative_to(src))
        if (src / "frames.json").exists(): shutil.copyfile(src / "frames.json", tmp / "frames.json")
        if dst.exists(): shutil.rmtree(dst)                                                        # only our own earlier output
        os.rename(tmp, dst); shutil.rmtree(src)
    rec = {"id": c["id"], "frames": n, "recon": len(rk), "test": len(tk), "gender": g, "ply_reproduction_mm_max": ply_mm,
           "smplx_fit_mm_mean": float(np.mean(meta["res_mm"])), "smplx_fit_mm_max_frame": float(np.max(meta["res_mm"])),
           "smplx_fit_mm_p95_vertex_mean": float(np.mean(fx["res_mm_p95"])), "smplx_fit_mm_max_vertex": float(np.max(fx["res_mm_max"])),
           "smplx_fit_px512_mean": float(np.mean(meta["fit_px512"])), "betas200_norm": float(np.linalg.norm(b200)),
           "fold_px512_mean": float(np.mean(meta["res_px512"])), "contract_cam_check_px": chk_px,
           "smpl_npz_err_mm_mean": float(np.mean(sm["err_mm"])), "smpl_npz_err_mm_max": float(np.max(sm["err_mm"])),
           "iou_smplx_mean": float(np.mean(iou_x)), "iou_smplx_min": float(np.min(iou_x)), "iou_smpl_mean": float(np.mean(iou_s)),
           "depth_mask_vs_nonwhite_iou_mean": float(np.mean(iou_white)), "alpha_mean": float(np.mean(meta["alpha_mean"])),
           "box_clipped_frames": int(sum(x["box_clipped"] for x in geo)), "crop_side_px_mean": float(np.mean(meta["crop_side_px"])),
           "part_visible_frac": {p: float(np.mean([v[p] for v in meta["part_visible"]])) for p in ("head", "lhand", "rhand")},
           "overlay": str(E3 / "base" / ds / name / "overlay.jpg"),
           "times_s": {"fit": round(t_fit, 1), "total": round(time.time() - t00, 1)}}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(json.dumps(rec), flush=True)
    return rec


def run(a):
    import torch
    J = json.loads(SPLIT.read_text()); clips = J["clips"]
    if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]
    todo = [c for c in clips if not (E3 / "base" / c["dataset"] / c["clip"] / "frames.json").exists()]
    print(f"{len(clips)} clips, {len(todo)} to do", flush=True)
    M = CT.maps(); (E3 / "reports").mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0"); X = CP.Ctx(dev)
    with open(E3 / "reports" / "convert_log.jsonl", "a") as log:
        for c in todo:
            convert_clip(c, X, M, J, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("split", "run")); ap.add_argument("--clips", default="")
    a = ap.parse_args()
    {"split": split, "run": run}[a.stage](a)


if __name__ == "__main__":
    main()
