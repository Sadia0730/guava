"""GUAVA-FB: BEDLAM2 videos -> GUAVA tracked clip format (phase3_dnt/guava_fb/FORMAT.md; PREREG "GUAVA-FB";
AVATAR_B_PLAN §2a, §3, §4.1 steps 1-7).

Stages
  plan   clip list + labelled rows + per-subject shape + camera motion -> O/reports/bedlam2/plan.json
         - strict split (bedlam_split_proposal.json): training = pilot subset (re-derived here with bedlam_split.py's rule
           and checked equal to the proposal's list: <= 2 clips / subject, 8 for moyo, prefer full-body median height >= 400 px,
           cap 1,500 clips); test (40 clips) NOT converted now.
         - val: 10 clips from the SPRINT / PEAR-student BEDLAM2 val jobs, subjects in neither test nor the training pool
           (subject-disjoint pool, i.e. never trainable), MOYO excluded; subjects drawn by the same SHA-1 hash rule as
           test, one clip per subject chosen as for test (moving-camera job first, most full-body rows).
         - rows: full-body rows of the census definition (all body / face / foot joints inside the image with a 1 % margin,
           >= 90 % of hand joints inside), <= 40 per clip spread evenly (linspace over the clip's full-body rows).
         - shape: shape16 (locked head) per SUBJECT (identical on every row of every clip of a subject; checked) ->
           SMPL-X 2020 betas200 = M @ shape16 + c (phase1_task1/checkpoint2/betas_locked16_to_2020_200.npz).
         - camera motion per clip from the BEDLAM2 camera csv: "translating" (position range > 1 cm), "rotating" (fixed
           position, pan / tilt / roll range > 0.5 deg or focal change), else "static".
  run    convert clips (CPU workers decode + geometry + perspective crop + JPEG; main process StyleMatte batched on its GPU,
         LMDB built on local disk then copied to O/data/bedlam2/<clip>). Resumable (a clip with meta.json is skipped).
  index  O/data/bedlam2/index.json.

Per row (record = sprint_avatar_fit.py conventions):
  locked-head mesh  V_L = SMPLX_lockedhead(pose_cam, shape16) + t_cam    (camera frame, OpenCV axes)
  2020 mesh         V_N = SMPLX_2020(pose_cam, betas200) + t_cam + (J0_L - J0_N)   (pelvis-aligned; same pose; checked
                    mean vertex distance to V_L, sub-mm -> no pose refit needed, threshold 5 mm)
  perspective crop  ray through the centre of the projected mesh box (2 iterations so the box is centred), virtual camera
                    rotation Rv (z onto the ray), square side L = 1.1 x long side of the mesh box in the virtual image,
                    K' = [[f s, 0, 512], [0, f s, 512]], s = 1024 / L, homography H = K' Rv K^-1 (cv2.warpPerspective,
                    bilinear), invtanfov = 2 f / L.
  fold              global_pose' = Rv R_root;  t' = Rv (J0_N + t) - J0_N  (t = t_cam + J0_L - J0_N)
  camera_RT_params  [diag(-1,-1,1) | diag(-1,-1,1) t']  -> GUAVA projection u = 0.5 + 0.5 invtanfov x / z (crop units).
  face              exp50 = 0, jaw = pose_cam 66:69, eyes = pose_cam 69:75 (also flame_coeffs.eye_pose_params).
  crops             body_crop M_o2c-hd = I (the stored 1024 image is the "original"), M_o2c = 224 / 1024 scale;
                    head_crop / hand crops from the projected converted SMPL-X 2020 vertices: FLAME head vertices
                    (SMPL-X__FLAME_vertex_ids minus FLAME selected_lowerhead = GUAVA's head_index set) square box x 1.2
                    (~ the tracker's 1.75 x face-landmark box), MANO vertices square box x 1.2 (AVATAR_B_PLAN §4.1.6), 512 px.
  drop rule         rows whose label mesh has a vertex < 0.2 m from the camera are dropped (degenerate crop).
  mask              StyleMatte (EHM-Tracker stylematte_synth.pt) alpha on the 1024 crop, stored as sprint_avatar_fit
                    (3-channel uint8 JPEG q95); image JPEG q95 (torchvision.io.encode_jpeg, RGB).

Usage (env as guava_a20/run_chain.sh):
  python convert_bedlam2.py plan
  CUDA_VISIBLE_DEVICES=1 nice python convert_bedlam2.py run --shard 0/3 --workers 10 [--limit 20] [--split train|val]
  python convert_bedlam2.py index
"""
from __future__ import annotations

import argparse
import collections as C
import csv
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GUAVA = Path("/home/ubx858/guava")
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
DATA = O / "data" / "bedlam2"
REP = O / "reports" / "bedlam2"
PLAN = REP / "plan.json"
AB = Path("/raid/ubx858/outputs/dnt/avatar_B")
PROC = Path("/raid/ubx858/datasets/processed/bedlam2_gt")
RAW = Path("/raid/ubx858/datasets/raw/bedlam2.0")
BMAP = Path("/raid/ubx858/outputs/phase1_task1/checkpoint2/betas_locked16_to_2020_200.npz")
VAL_JOBS = ("20240731_1_1827_hdri", "20240808_1_250_ai1105_vcam")
MAX_ROWS, CROP, BODY_LR, PART = 40, 1024, 224, 512
HEAD_SCALE, HAND_SCALE, BODY_SCALE = 1.2, 1.2, 1.1
ZMIN = 0.2      # rows whose label mesh comes closer than 0.2 m to the camera are dropped (1 row in the pilot: a hand at the lens)
LOCAL_TMP = Path(os.environ.get("GFB_TMP", "/tmp/claude-926700958/-home-ubx858-guava/474a66f8-c9aa-43f4-8a23-a36807857f2a/scratchpad/bedlam2_tmp"))
LICENCE = ("BEDLAM2: MPI non-commercial scientific research licence (Max Planck Institute for Intelligent Systems; "
           "https://bedlam2.is.tuebingen.mpg.de/license.html). Research use only; no redistribution. Body model: SMPL-X 2020 "
           "(MPI non-commercial licence).")


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16) / 16 ** 40


def clip_name(job, seq):
    return f"{job}__{seq}"


def full_body_mask(d):
    """bedlam_census.py's full-body row definition."""
    wh = d["img_wh"]; j2 = d["joints2d_img"]
    m = 0.01 * wh.max(1, keepdims=True)
    inside = ((j2[:, :, 0] > -m) & (j2[:, :, 0] < wh[:, :1] + m) & (j2[:, :, 1] > -m) & (j2[:, :, 1] < wh[:, 1:] + m))
    return inside[:, list(range(25))].all(1) & (inside[:, 25:55].mean(1) >= 0.9)


# ------------------------------------------------------------------------------------------------------------ plan
def camera_motion(job, seqs):
    out = {}
    p = RAW / f"{job}_gt_centersubframe_exr_meta_csv.tar.gz"
    want = {f"{s}_camera.csv": s for s in seqs}
    with tarfile.open(p) as tf:
        for m in tf:
            b = m.name.rsplit("/", 1)[-1]
            if b in want:
                rows = list(csv.DictReader(io.TextIOWrapper(tf.extractfile(m))))
                X = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
                A = np.array([[float(r[k]) for k in ("yaw", "pitch", "roll")] for r in rows])
                A = np.degrees(np.unwrap(np.radians(A), axis=0))
                f = np.array([float(r["focal_length"]) for r in rows])
                pr, ar, fr = float(np.ptp(X, 0).max()), float(np.ptp(A, 0).max()), float(np.ptp(f) / max(f.mean(), 1e-6))
                kind = "translating" if pr > 1.0 else ("rotating" if (ar > 0.5 or fr > 1e-3) else "static")
                out[want[b]] = {"moving": kind != "static", "motion": kind, "pos_range_cm": round(pr, 2),
                                "ang_range_deg": round(ar, 3), "focal_rel_range": round(fr, 5), "hfov_deg": float(rows[0]["hfov"])}
    missing = set(seqs) - set(out)
    for s in missing:
        out[s] = {"moving": None}
    return out


def plan(a):
    R = json.load(open(AB / "bedlam_census.json")); sp = json.load(open(AB / "bedlam_split_proposal.json"))
    S = [r for r in R if not r["multi"] and r["n_full"] >= 30 and r["body_h_px_full_med"] >= 350]
    for r in S:
        r["subject"] = "moyo" if r["body"].startswith("moyo") else r["subject"]
        r["outfit"] = r["texture_clothing"].split("_texture")[0] if r["texture_clothing"] else ""
    test_sub = {c[2] for c in sp["test"]["clips"]}
    test_outfits = {r["outfit"] for r in S if (r["job"], r["seq"]) in {(c[0], c[1]) for c in sp["test"]["clips"]}}
    train_pool = [r for r in S if r["subject"] not in test_sub and r["job"] not in VAL_JOBS]
    strict = [r for r in train_pool if r["outfit"] not in test_outfits]
    by = C.defaultdict(list)
    for r in strict:
        by[r["subject"]].append(r)
    pilot = []
    for s in sorted(by, key=h):
        c = sorted(by[s], key=lambda r: (-(r["body_h_px_full_med"] >= 400), h(r["job"] + r["seq"])))
        pilot += c[: (8 if s == "moyo" else 2)]
    pilot = sorted(pilot, key=lambda r: h(r["job"] + r["seq"]))[:1500]
    prop = [tuple(x) for x in sp["pilot_train"]["list"]]
    assert [(r["job"], r["seq"], r["subject"]) for r in pilot] == prop, "pilot rule does not reproduce the proposal list"
    # val: val-job clips, subjects in neither test nor the training pool, hash rule as test
    pool_sub = {r["subject"] for r in train_pool}
    vc = [r for r in S if r["job"] in VAL_JOBS and r["subject"] not in test_sub and r["subject"] not in pool_sub and r["subject"] != "moyo"]
    val = []
    for s in sorted({r["subject"] for r in vc}, key=h)[:10]:
        c = sorted([r for r in vc if r["subject"] == s], key=lambda r: (r["job"] != "20240808_1_250_ai1105_vcam", -r["n_full"], r["job"], r["seq"]))
        val.append(c[0])
    assert len(val) == 10
    clips = [dict(r, split="train") for r in pilot] + [dict(r, split="val") for r in val]
    bm = np.load(BMAP); M, c0 = bm["M"].astype(np.float64), bm["c"].astype(np.float64)
    jobs = C.defaultdict(list)
    for r in clips:
        jobs[r["job"]].append(r)
    subj_shape = {}; shape_dev = {}
    out = []
    for job in sorted(jobs):
        d = np.load(PROC / f"{job}.npz", allow_pickle=True)
        full = full_body_mask(d); seq = d["seq"]
        cams = camera_motion(job, [r["seq"] for r in jobs[job]])
        for r in jobs[job]:
            sel = np.flatnonzero((seq == r["seq"]) & full)
            sel = sel[np.argsort(d["frame"][sel], kind="stable")]
            assert len(sel) == r["n_full"], (job, r["seq"], len(sel), r["n_full"])
            k = min(MAX_ROWS, len(sel))
            pick = sel[np.unique(np.round(np.linspace(0, len(sel) - 1, k)).astype(int))]
            sh = d["shape16"][np.flatnonzero(seq == r["seq"])].astype(np.float64)
            s = r["subject"]
            if s not in subj_shape:
                subj_shape[s] = sh[0]
            shape_dev[s] = max(shape_dev.get(s, 0.0), float(np.abs(sh - subj_shape[s]).max()))
            out.append({"name": clip_name(job, r["seq"]), "job": job, "seq": r["seq"], "subject": s, "split": r["split"],
                        "rows": [int(i) for i in pick], "frames": [int(f) for f in d["frame"][pick]], "n_full": int(r["n_full"]),
                        "video_frames": int(r["video_frames"]), "img_wh": [int(r["img_w"]), int(r["img_h"])],
                        "body_h_px_full_med": float(r["body_h_px_full_med"]), "camera": cams[r["seq"]],
                        "outfit": r["outfit"], "texture_body": r["texture_body"], "hair": r["hair"]})
    betas = {s: (M @ v + c0).astype(np.float32).tolist() for s, v in subj_shape.items()}
    P = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": out,
         "subject_shape16": {s: v.tolist() for s, v in subj_shape.items()}, "subject_betas200": betas,
         "subject_shape16_max_abs_dev": shape_dev,
         "val_rule": "val jobs " + ", ".join(VAL_JOBS) + "; subjects not in test and not in the subject-disjoint training pool, "
                     "MOYO excluded; first 10 subjects by SHA-1(subject); one clip per subject (moving-camera job first, most full-body rows)",
         "val_candidates_subjects": len({r["subject"] for r in vc}),
         "pilot_check": "re-derived pilot list == bedlam_split_proposal.json pilot_train.list (strict variant)"}
    REP.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(P))
    n = C.Counter(c["split"] for c in out)
    print(json.dumps({"clips": n, "rows": {s: sum(len(c["rows"]) for c in out if c["split"] == s) for s in n},
                      "subjects": {s: len({c["subject"] for c in out if c["split"] == s}) for s in n},
                      "max_shape_dev": max(shape_dev.values()),
                      "motion": C.Counter(str(c["camera"].get("motion")) for c in out)}, indent=1))


# ------------------------------------------------------------------------------------------------------------ geometry
_G = {}


def geo():
    if not _G:
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, str(HERE))
        from smplx_lean import SMPLXLean, LOCKED, SMPLX2020
        _G["L"] = SMPLXLean(LOCKED, 16); _G["N"] = SMPLXLean(SMPLX2020, 200)
        fl = np.load(GUAVA / "assets/SMPLX/SMPL-X__FLAME_vertex_ids.npy")
        low = np.load(GUAVA / "assets/FLAME/selected_lowerhead.npy")
        _G["head"] = fl[np.setdiff1d(np.arange(len(fl)), low)]
        with open(GUAVA / "assets/SMPLX/MANO_SMPLX_vertex_ids.pkl", "rb") as f:
            mano = pickle.load(f, encoding="latin1")
        _G["lhand"], _G["rhand"] = np.asarray(mano["left_hand"]), np.asarray(mano["right_hand"])
        _G["torch"] = torch
    return _G


def rot_onto_z(ray):
    ray = ray / np.linalg.norm(ray); z = np.array([0, 0, 1.0])
    ax = np.cross(ray, z); sa = np.linalg.norm(ax); ang = np.arctan2(sa, ray @ z)
    if sa < 1e-12:
        return np.eye(3)
    ax /= sa; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def aa_from_R(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].reshape(3)


def box_crop_matrix(box, dsize, scale):
    """sprint_avatar_fit.box_crop_matrix (EHM crop_image_by_bbox, no rotation): original px -> crop px."""
    l, t, r, b = [float(v) for v in box]
    size = int(max(r - l, b - t) * scale + 1e-7)
    s = dsize / (size + 1e-7); cx, cy = (l + r) / 2, (t + b) / 2
    return np.array([[s, 0, dsize / 2 - s * cx], [0, s, dsize / 2 - s * cy], [0, 0, 1]], np.float32)


def project_guava(V_model, RT, invtanfov):
    """GUAVA / PEAR crop camera (sprint_avatar_fit.project) in 1024 px: u = (1 - F * cam_xy / cam_z) / 2."""
    cam = V_model @ RT[:, :3].T + RT[:, 3]
    return (1.0 - invtanfov * cam[..., :2] / cam[..., 2:3]) * 0.5 * CROP


def clip_geometry(clip, d, betas200):
    G = geo(); torch = G["torch"]
    rows = np.asarray(clip["rows"])
    P = torch.tensor(d["pose_cam"][rows], dtype=torch.float64)
    S16 = torch.tensor(d["shape16"][rows], dtype=torch.float64)
    B = torch.tensor(np.asarray(betas200, np.float64))[None].expand(len(rows), -1)
    VL, JL = G["L"](P, S16); VN, JN = G["N"](P, B)
    VL, JL, VN, JN = VL.numpy(), JL.numpy(), VN.numpy(), JN.numpy()
    tc = d["t_cam"][rows].astype(np.float64)
    t = tc + JL[:, 0] - JN[:, 0]                                    # 2020 translation (pelvis aligned)
    out = []
    for i, r in enumerate(rows):
        K = d["K"][r].astype(np.float64); f = K[0, 0]
        XL = VL[i] + tc[i]; XN = VN[i] + t[i]                       # camera frame meshes (source, converted)
        res_mm = float(np.linalg.norm(XL - XN, axis=1).mean() * 1000)
        Rv = np.eye(3)
        for _ in range(3):                                          # aim the virtual camera at the mesh-box centre
            Xr = XL @ Rv.T; p = f * Xr[:, :2] / Xr[:, 2:]
            lo, hi = p.min(0), p.max(0); cc = (lo + hi) / 2
            Rv = rot_onto_z(np.array([cc[0] / f, cc[1] / f, 1.0])) @ Rv
        Xr = XL @ Rv.T; p = f * Xr[:, :2] / Xr[:, 2:]; lo, hi = p.min(0), p.max(0)
        L = BODY_SCALE * float((hi - lo).max())
        s = CROP / L
        Kv = np.array([[f * s, 0, CROP / 2], [0, f * s, CROP / 2], [0, 0, 1.0]])
        H = Kv @ Rv @ np.linalg.inv(K)
        invtanfov = 2 * f / L
        # source label mesh in the crop (gt_proj): original camera projection mapped by H
        q = XL @ K.T; q = q / q[:, 2:]; q = q @ H.T; v2d_src = q[:, :2] / q[:, 2:]
        # fold Rv into the root about the 2020 pelvis
        Rroot = Rv @ __import__("cv2").Rodrigues(d["pose_cam"][r][:3].astype(np.float64))[0]
        J0 = JN[i, 0]
        tp = Rv @ (J0 + t[i]) - J0
        Dm = np.diag([-1.0, -1.0, 1.0])
        RT = np.concatenate([Dm, (Dm @ tp)[:, None]], 1).astype(np.float32)
        Vmodel = (VN[i] - J0) @ Rv.T + J0                            # model output with the folded root (no translation)
        v2d = project_guava(Vmodel, RT.astype(np.float64), invtanfov)
        res_px512 = float(np.linalg.norm(v2d - v2d_src, axis=1).mean() * 512 / CROP)
        pc = d["pose_cam"][r].astype(np.float32)
        smplx = {"global_pose": aa_from_R(Rroot).astype(np.float32), "body_pose": pc[3:66].reshape(21, 3).copy(),
                 "jaw_pose": pc[66:69].reshape(1, 3).copy(), "eye_pose": pc[69:75].reshape(2, 3).copy(),
                 "left_hand_pose": pc[75:120].reshape(15, 3).copy(), "right_hand_pose": pc[120:165].reshape(15, 3).copy(),
                 "exp": np.zeros(50, np.float32), "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}

        def part(ids, scale):
            pts = v2d[ids]; lt, rb = pts.min(0), pts.max(0)
            Mo = box_crop_matrix([lt[0], lt[1], rb[0], rb[1]], PART, scale)
            vis = bool(((pts >= 0) & (pts < CROP)).all(1).mean() > 0.5)
            return {"M_o2c": Mo, "M_c2o": np.linalg.inv(Mo).astype(np.float32)}, vis
        hc, hv = part(G["head"], HEAD_SCALE); lc, lv = part(G["lhand"], HAND_SCALE); rc, rv = part(G["rhand"], HAND_SCALE)
        Mlr = np.diag([BODY_LR / CROP, BODY_LR / CROP, 1.0]).astype(np.float32)
        I3 = np.eye(3, dtype=np.float32)
        rec = {"smplx_coeffs": smplx,
               "flame_coeffs": {"expression_params": np.zeros(50, np.float32), "jaw_params": pc[66:69].copy(),
                                "pose_params": np.zeros(3, np.float32), "neck_pose_params": np.zeros(3, np.float32),
                                "eye_pose_params": pc[69:75].copy(), "eyelid_params": np.zeros(2, np.float32),
                                "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
               "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3.copy(), "M_c2o-hd": I3.copy()},
               "head_crop": hc, "left_hand_crop": lc, "right_hand_crop": rc,
               "invtanfov": float(invtanfov), "view": "0"}
        out.append({"rec": rec, "H": H, "v2d_src": v2d_src.astype(np.float32), "v2d": v2d.astype(np.float32),
                    "res_mm": res_mm, "res_px512": res_px512, "invtanfov": float(invtanfov), "crop_side_px": L,
                    "depth_m": float(XL[:, 2].mean()), "zmin_m": float(XL[:, 2].min()), "vis": {"head": hv, "lhand": lv, "rhand": rv}})
    return out


# ------------------------------------------------------------------------------------------------------------ worker
_NPZ = {}


def job_npz(job):
    if job not in _NPZ:
        _NPZ.clear()
        z = np.load(PROC / f"{job}.npz", allow_pickle=True)
        _NPZ[job] = {k: z[k] for k in ("pose_cam", "shape16", "t_cam", "K", "frame", "seq", "img_wh")}
    return _NPZ[job]


def decode_frames(job, seq, frames):
    import cv2
    want = sorted(set(frames)); got = {}
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    with tarfile.open(RAW / f"{job}_mp4.tar") as tf:
        mem = tf.getmember(f"{job}/mp4/{seq}.mp4")
        fd, path = tempfile.mkstemp(suffix=".mp4", dir=str(LOCAL_TMP))
        with os.fdopen(fd, "wb") as fo:
            shutil.copyfileobj(tf.extractfile(mem), fo, 1 << 22)
    try:
        cap = cv2.VideoCapture(path); n = 0; last = want[-1]; ws = set(want)
        while n <= last:
            if n in ws:
                ok, im = cap.read()
                if not ok:
                    break
                got[n] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            else:
                if not cap.grab():
                    break
            n += 1
        cap.release()
    finally:
        os.unlink(path)
    return got


def work(args):
    """CPU worker: one clip -> geometry, crops (RGB uint8), image JPEG bytes."""
    clip, betas = args
    import cv2, torch, torchvision
    cv2.setNumThreads(1); torch.set_num_threads(1)
    t0 = time.time()
    try:
        d = job_npz(clip["job"])
        rows = np.asarray(clip["rows"])
        assert (d["seq"][rows] == clip["seq"]).all()
        g = clip_geometry(clip, d, betas)
        t1 = time.time()
        frames = decode_frames(clip["job"], clip["seq"], clip["frames"])
        t2 = time.time()
        crops, jpgs, keep = [], [], []
        for i, (fr, gi) in enumerate(zip(clip["frames"], g)):
            if fr not in frames or gi["zmin_m"] < ZMIN:                  # degenerate perspective crop (body part at the lens)
                continue
            im = frames[fr]
            if im.shape[1] != clip["img_wh"][0] or im.shape[0] != clip["img_wh"][1]:
                raise RuntimeError(f"frame size {im.shape} != label img_wh {clip['img_wh']}")
            c = cv2.warpPerspective(im, gi["H"], (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            crops.append(c)
            jpgs.append(torchvision.io.encode_jpeg(torch.from_numpy(c).permute(2, 0, 1).contiguous(), quality=95).numpy().tobytes())
            keep.append(i)
        t3 = time.time()
        return {"clip": clip, "g": [g[i] for i in keep], "frames": [clip["frames"][i] for i in keep], "crops": np.stack(crops) if crops else None,
                "jpgs": jpgs, "missing": [f for f in clip["frames"] if f not in frames],
                "dropped_zmin": [f for f, gi in zip(clip["frames"], g) if gi["zmin_m"] < ZMIN],
                "t": {"geom": t1 - t0, "decode": t2 - t1, "crop_jpg": t3 - t2}, "err": None}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"clip": clip, "err": repr(e) + "\n" + traceback.format_exc()}


# ------------------------------------------------------------------------------------------------------------ main (GPU)
class Matter:
    def __init__(self):
        import torch
        sys.path.insert(0, str(GUAVA / "EHM-Tracker"))
        from src.modules.human_matting.stylematte import StyleMatte
        cwd = os.getcwd(); os.chdir(GUAVA)                         # StyleMatte cache_dir is relative (as sprint_avatar_fit)
        m = StyleMatte(); os.chdir(cwd)
        m.load_state_dict(torch.load(GUAVA / "EHM-Tracker/pretrained/matting/stylematte_synth.pt", map_location="cpu", weights_only=True))
        self.m = m.cuda().eval(); self.torch = torch
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def __call__(self, crops, bs=8):
        torch = self.torch; out = []
        with torch.no_grad():
            for i in range(0, len(crops), bs):
                x = torch.from_numpy(crops[i:i + bs]).cuda().permute(0, 3, 1, 2).float() / 255.0
                a = self.m((x - self.mean) / self.std)[:, 0].clamp(0, 1)
                out.append((a * 255).round().to(torch.uint8).cpu())
        return torch.cat(out).numpy()


def write_clip(res, alphas, betas):
    import lmdb, torch, torchvision
    clip = res["clip"]; name = clip["name"]
    loc = LOCAL_TMP / "clips" / name
    if loc.exists():
        shutil.rmtree(loc)
    loc.mkdir(parents=True)
    keys = [f"frame_{f:06d}" for f in res["frames"]]
    recs = {k: g["rec"] for k, g in zip(keys, res["g"])}
    with open(loc / "optim_tracking_ehm.pkl", "wb") as f:
        pickle.dump(recs, f)
    ids = {"smplx_shape": np.asarray(betas, np.float32).reshape(1, 200), "flame_shape": np.zeros((1, 300), np.float32),
           "left_mano_shape": np.zeros((1, 10), np.float32), "right_mano_shape": np.zeros((1, 10), np.float32),
           "head_scale": np.ones((1, 3), np.float32), "hand_scale": np.ones((1, 3), np.float32), "joints_offset": np.zeros((1, 55, 3), np.float32)}
    with open(loc / "id_share_params.pkl", "wb") as f:
        pickle.dump(ids, f)
    (loc / "videos_info.json").write_text(json.dumps({name: {"frames_num": len(keys), "frames_keys": keys}}, indent=1))
    env = lmdb.open(str(loc / "img_lmdb"), map_size=1 << 32); txn = env.begin(write=True)
    msk_bytes = 0; img_bytes = 0
    for k, jb, a in zip(keys, res["jpgs"], alphas):
        txn.put(f"{k}/body_image".encode(), jb); img_bytes += len(jb)
        mb = torchvision.io.encode_jpeg(torch.from_numpy(a)[None].expand(3, -1, -1).contiguous(), quality=95).numpy().tobytes()
        txn.put(f"{k}/body_mask".encode(), mb); msk_bytes += len(mb)
    txn.commit(); env.close()
    gi = np.unique(np.round(np.linspace(0, len(keys) - 1, min(10, len(keys)))).astype(int))
    np.savez_compressed(loc / "gt_proj.npz", frame_key=np.array([keys[i] for i in gi]),
                        v2d_1024=np.stack([res["g"][i]["v2d_src"] for i in gi]),
                        v2d_converted_1024=np.stack([res["g"][i]["v2d"] for i in gi]),
                        note=np.array("v2d_1024: BEDLAM2 label mesh (SMPL-X locked head, shape16, pose_cam + t_cam) projected with the original K "
                                      "and mapped into the 1024 perspective crop by H; v2d_converted_1024: converted SMPL-X 2020 mesh with the "
                                      "record camera_RT_params + invtanfov (GUAVA projection)"))
    meta = {"name": name, "job": clip["job"], "seq": clip["seq"], "subject": clip["subject"], "split": clip["split"],
            "frames_keys": keys, "rows": [int(clip["rows"][clip["frames"].index(f)]) for f in res["frames"]],
            "missing_frames": res["missing"], "dropped_zmin": res.get("dropped_zmin", []), "camera": clip["camera"], "img_wh": clip["img_wh"],
            "res_mm": [g["res_mm"] for g in res["g"]], "res_px512": [g["res_px512"] for g in res["g"]],
            "invtanfov": [g["invtanfov"] for g in res["g"]], "crop_side_px": [g["crop_side_px"] for g in res["g"]],
            "depth_m": [g["depth_m"] for g in res["g"]], "part_visible": [g["vis"] for g in res["g"]],
            "alpha_mean": [float(a.mean() / 255) for a in alphas], "bytes_img": img_bytes, "bytes_mask": msk_bytes,
            "times": res["t"], "licence": "BEDLAM2 MPI non-commercial research licence"}
    (loc / "meta.json").write_text(json.dumps(meta))
    # copy to O (meta.json last = done marker), then remove the local copy
    dst = DATA / name; tmp = DATA / (name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for p in ("optim_tracking_ehm.pkl", "id_share_params.pkl", "videos_info.json", "gt_proj.npz"):
        shutil.copyfile(loc / p, tmp / p)
    (tmp / "img_lmdb").mkdir()
    for p in (loc / "img_lmdb").iterdir():
        shutil.copyfile(p, tmp / "img_lmdb" / p.name)
    shutil.copyfile(loc / "meta.json", tmp / "meta.json")
    if dst.exists():                                                # only our own partial / re-run output
        shutil.rmtree(dst)
    os.rename(tmp, dst)
    shutil.rmtree(loc)


def run(a):
    import multiprocessing as mp
    P = json.loads(PLAN.read_text())
    clips = [c for c in P["clips"] if a.split in (None, c["split"])]
    k, n = map(int, a.shard.split("/"))
    clips = [c for i, c in enumerate(sorted(clips, key=lambda c: (c["split"] != "val", c["name"]))) if i % n == k]
    if a.limit:
        clips = clips[:a.limit]
    todo = [c for c in clips if not (DATA / c["name"] / "meta.json").exists()]
    print(f"shard {a.shard}: {len(clips)} clips, {len(todo)} to do", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    matter = Matter()
    t0 = time.time(); nrows = 0; fails = []
    log = open(REP / f"run_log_shard{k}of{n}.jsonl", "a")
    writer = None
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        it = pool.imap_unordered(work, [(c, P["subject_betas200"][c["subject"]]) for c in todo], chunksize=1)
        for j, res in enumerate(it):
            if res["err"] or res.get("crops") is None:
                fails.append({"clip": res["clip"]["name"], "err": res["err"] or "no frames"})
                log.write(json.dumps({"clip": res["clip"]["name"], "err": res["err"] or "no frames"}) + "\n"); log.flush()
                print("FAIL", res["clip"]["name"], (res["err"] or "no frames")[:300], flush=True)
                continue
            tm = time.time(); alphas = matter(res["crops"]); tm = time.time() - tm
            res["crops"] = None
            if writer is not None:
                writer.join()
            betas = P["subject_betas200"][res["clip"]["subject"]]
            writer = threading.Thread(target=write_clip, args=(res, alphas, betas)); writer.start()
            nrows += len(alphas)
            el = time.time() - t0
            log.write(json.dumps({"clip": res["clip"]["name"], "rows": len(alphas), "matte_s": round(tm, 2), **{k_: round(v, 2) for k_, v in res["t"].items()},
                                  "elapsed_s": round(el, 1)}) + "\n"); log.flush()
            if (j + 1) % 10 == 0 or j + 1 == len(todo):
                print(f"[{k}/{n}] {j + 1}/{len(todo)} clips, {nrows} rows, {el:.0f} s, {nrows / el:.2f} rows/s", flush=True)
    if writer is not None:
        writer.join()
    el = time.time() - t0
    summ = {"shard": a.shard, "clips_done": len(todo) - len(fails), "fails": fails, "rows": nrows, "seconds": round(el, 1),
            "rows_per_s": round(nrows / max(el, 1e-6), 2), "workers": a.workers}
    print(json.dumps(summ), flush=True)
    (REP / f"run_summary_shard{k}of{n}{'_limit' + str(a.limit) if a.limit else ''}.json").write_text(json.dumps(summ, indent=1))


def index(a):
    P = json.loads(PLAN.read_text())
    clips = []
    for c in P["clips"]:
        mp_ = DATA / c["name"] / "meta.json"
        if not mp_.exists():
            continue
        m = json.loads(mp_.read_text())
        clips.append({"name": c["name"], "subject": c["subject"], "split": c["split"], "frames": len(m["frames_keys"]), "views": 1,
                      "job": c["job"], "seq": c["seq"], "camera_moving": c["camera"]["moving"], "camera_motion": c["camera"].get("motion")})
    idx = {"source": "bedlam2", "licence": LICENCE, "clips": clips,
           "notes": ("BEDLAM2 (Black et al. / Tesch et al., MPI) single-person rendered videos; labels exact (SMPL-X locked-head "
                     "pose_cam / shape16 / t_cam, exact pinhole K). Converted to SMPLX_NEUTRAL_2020: shape16 -> betas200 by the linear map "
                     "betas_locked16_to_2020_200.npz (one shape per subject = shape lock), same pose (pelvis-aligned; mean vertex distance "
                     "to the source mesh sub-mm, see reports/bedlam2/report.json), exp50 = 0, jaw / eyes from pose_cam. Perspective crop "
                     "(virtual camera on the ray through the mesh-box centre, 1.1 x box, 1024 px) with per-record invtanfov = 2 f / L. "
                     "Masks: StyleMatte. Rows: <= 40 full-body labelled rows (6 fps labels) per clip, spread evenly. One camera per clip "
                     "(view '0'); camera_moving from the BEDLAM2 camera csv. Split: AVATAR_B_PLAN strict split; train = pilot subset "
                     "(1,500 clips), val = 10 val-job clips of subjects outside test and the training pool; the 40 test clips are not "
                     "converted yet. Frame key frame_XXXXXX = video frame index (30 fps mp4).")}
    (DATA / "index.json").write_text(json.dumps(idx, indent=1))
    print(json.dumps({"clips": len(clips), "by_split": C.Counter(c["split"] for c in clips),
                      "frames": sum(c["frames"] for c in clips)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("plan", "run", "index"))
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--split", default=None)
    a = ap.parse_args()
    {"plan": plan, "run": run, "index": index}[a.stage](a)


if __name__ == "__main__":
    main()
