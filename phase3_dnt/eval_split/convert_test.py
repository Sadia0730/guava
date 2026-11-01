"""AvatarTest-v1 test data (PREREG "AvatarTest-v1 addendum 1"). Every GT frame of every clip of eval_split/avatar_test_v1.json ->
  E/data/<dataset>/<clip>/    GUAVA tracked format (FORMAT.md; GT SMPL-X 2020 records + FLAME fields for released GUAVA)
  E/gt512/<dataset>/<clip>/   <key>.png for the test keys: the scoring GT = image x StyleMatte alpha at 512 on black (8 bit)
  E/base/<dataset>/<clip>/    baseline input package (phase3_dnt/baselines/CONTRACT.md) incl. a neutral SMPL fit
E = O/eval/avatar_test_v1 (O = the GUAVA-FB output root on /vault).

BEDLAM2: convert_bedlam2.py's geometry (clip_geometry), frame decode, perspective crop and StyleMatte, unchanged, on all of the
clip's frames. UBody: SMPL-X v1.1 (flat_hand_mean False, as OSX) -> SMPL-X 2020 here: poses copied (same template, joint
regressor, skinning, pose correctives), hands + v1.1 hand mean, jaw copied, eyes 0, exp50 = v11_expr10_to_2020_exp50 @ expr10,
betas200 = least-squares map (all vertices) of the clip's MEDIAN betas10, translation pelvis-aligned; camera = the annotation's
per-frame (focal, princpt); crop aimed at / sized by the projected source-mesh box clipped to the image; frames extracted with
`ffmpeg -i clip.mp4 -f image2 -r 30` (the GT's timeline, as OSX). FLAME fields (both datasets): flame_shape = least-squares fit
of FLAME 2020's shape basis to the SMPL-X 2020 head (SMPL-X__FLAME_vertex_ids) of the clip's betas200; expression = exp50 (same
basis on the head); jaw / eyes copied. Images and masks of the baseline package / GT are the stored JPEGs decoded and resized
exactly as GUAVA-FB's loader does (fb_common.FBData: float, antialiased resize to 512).
SMPL (GART / GaussianAvatar): the GT SMPL-X 2020 camera-frame mesh mapped by PEAR's smplx2smpl correspondence, then a neutral
SMPL (eval_assets npz = the SMPL v1.1.0 neutral pkl) fitted per clip: per-frame pose + translation, one betas10; 600 Adam steps.

Usage: CUDA_VISIBLE_DEVICES=1 python convert_test.py [--clips id1,id2] [--workers 6]"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guava_fb"))
import convert_bedlam2 as cb                                                                       # noqa: E402

SPLIT = HERE / "avatar_test_v1.json"
E = cb.O / "eval" / "avatar_test_v1"
TMP = Path("/dev/shm/ubx858_eval/tmp")
FFMPEG = "/home/ubx858/miniconda3/envs/guava/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
V11 = Path("/raid/ubx858/datasets/models/smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz")
X2020 = cb.GUAVA / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz"
EXPR_MAP = Path("/raid/ubx858/outputs/dnt/sprint/data/expr_maps_to_smplx2020.npz")
FLAME_PKL = cb.GUAVA / "assets/FLAME/FLAME2020/generic_model.pkl"
FID = cb.GUAVA / "assets/SMPLX/SMPL-X__FLAME_vertex_ids.npy"
SMPL_DIR = Path("/raid/ubx858/datasets/eval_assets/smpl")
SMPLX2SMPL = cb.GUAVA / "third_party/PEAR/assets/SMPLX2SMPL.zip"
CROP, PART, BODY_LR = cb.CROP, cb.PART, cb.BODY_LR


# ------------------------------------------------------------------------------------------------ linear maps (CPU, once)
def maps():
    d11 = np.load(V11, allow_pickle=True); d20 = np.load(X2020, allow_pickle=True)
    B11 = d11["shapedirs"][:, :, :10].reshape(-1, 10).astype(np.float64); B20 = d20["shapedirs"][:, :, :200].reshape(-1, 200).astype(np.float64)
    Wsh = np.linalg.lstsq(B20, B11, rcond=None)[0]                                                 # (200, 10): betas200 = Wsh @ betas10
    sh_err = np.linalg.norm((B20 @ Wsh - B11).reshape(-1, 3, 10), axis=1).mean(0) * 1000          # mm per unit beta, per component
    for n in ("bool", "int", "float", "complex", "object", "str"):
        if n not in np.__dict__: setattr(np, n, getattr(__import__("builtins"), n))
    if "unicode" not in np.__dict__: np.unicode = str
    fl = pickle.load(open(FLAME_PKL, "rb"), encoding="latin1")
    Bf = np.asarray(fl["shapedirs"])[:, :, :300].reshape(-1, 300).astype(np.float64)
    fid = np.load(FID); Bxh = d20["shapedirs"][fid][:, :, :200].reshape(-1, 200).astype(np.float64)
    Wf = np.linalg.lstsq(Bf, Bxh, rcond=None)[0]                                                    # (300, 200): flame_shape = Wf @ betas200
    f_err = float(np.linalg.norm((Bf @ Wf - Bxh).reshape(-1, 3, 200), axis=1).mean() * 1000)
    Wex = np.load(EXPR_MAP)["v11_expr10_to_2020_exp50"].astype(np.float64)                          # (50, 10)
    hm = np.concatenate([d11["hands_meanl"], d11["hands_meanr"]]).astype(np.float64)                # (90,)
    return {"Wsh": Wsh, "Wf": Wf, "Wex": Wex, "hm": hm,
            "report": {"shape_map_err_mm_per_unit_beta": sh_err.round(3).tolist(), "flame_shape_map_err_mm_per_unit_beta": round(f_err, 4)}}


# ------------------------------------------------------------------------------------------------ UBody geometry (worker)
_U = {}


def ub_models():
    if not _U:
        g = cb.geo()
        from smplx_lean import SMPLXLean
        _U["V11"] = SMPLXLean(str(V11), 10, n_exp=10); _U["N"] = g["N"]
    return _U


def ub_geometry(clip, P, M):
    """P: per-frame UBody params (lists). Returns per-frame dicts like convert_bedlam2.clip_geometry."""
    import cv2, torch
    G = cb.geo(); U = ub_models()
    n = len(P); f64 = lambda x: torch.tensor(np.asarray(x, np.float64))                            # noqa: E731
    sp = [p["smplx_param"] for p in P]
    pose = np.concatenate([np.stack([s["root_pose"] for s in sp]), np.stack([s["body_pose"] for s in sp]), np.stack([s["jaw_pose"] for s in sp]),
                           np.zeros((n, 6)), np.stack([s["lhand_pose"] for s in sp]) + M["hm"][:45], np.stack([s["rhand_pose"] for s in sp]) + M["hm"][45:]], 1)
    b10 = np.stack([s["shape"] for s in sp]).astype(np.float64); e10 = np.stack([s["expr"] for s in sp]).astype(np.float64)
    tr = np.stack([s["trans"] for s in sp]).astype(np.float64)
    b200 = M["Wsh"] @ np.median(b10, 0); exp50 = e10 @ M["Wex"].T
    VL, JL = U["V11"](f64(pose), f64(b10), f64(e10)); VN, JN = U["N"](f64(pose), f64(b200)[None].expand(n, -1), f64(exp50))
    VL, JL, VN, JN = VL.numpy(), JL.numpy(), VN.numpy(), JN.numpy()
    t = tr + JL[:, 0] - JN[:, 0]
    W, H = clip["wh"]; out = []
    for i in range(n):
        cam = P[i]["cam_param"]; fx, fy = cam["focal"]; cx, cy = cam["princpt"]; f = float(fx)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); Ki = np.linalg.inv(K)
        XL = VL[i] + tr[i]; XN = VN[i] + t[i]
        res_mm = float(np.linalg.norm(XL - XN, axis=1).mean() * 1000)
        p = XL @ K.T; p = p[:, :2] / p[:, 2:]
        lo = np.clip(p.min(0), 0, [W, H]); hi = np.clip(p.max(0), 0, [W, H])                       # mesh box clipped to the image
        Rv = cb.rot_onto_z(Ki @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.0]))
        cor = np.array([[lo[0], lo[1], 1], [hi[0], lo[1], 1], [lo[0], hi[1], 1], [hi[0], hi[1], 1.0]]) @ Ki.T @ Rv.T
        q = f * cor[:, :2] / cor[:, 2:]
        L = cb.BODY_SCALE * float((q.max(0) - q.min(0)).max()); s = CROP / L
        Kv = np.array([[f * s, 0, CROP / 2], [0, f * s, CROP / 2], [0, 0, 1.0]])
        Hm = Kv @ Rv @ Ki; invtanfov = 2 * f / L
        qs = XL @ K.T; qs = qs / qs[:, 2:]; qs = qs @ Hm.T; v2d_src = qs[:, :2] / qs[:, 2:]
        Rroot = Rv @ cv2.Rodrigues(pose[i, :3])[0]; J0 = JN[i, 0]; tp = Rv @ (J0 + t[i]) - J0
        Dm = np.diag([-1.0, -1.0, 1.0]); RT = np.concatenate([Dm, (Dm @ tp)[:, None]], 1).astype(np.float32)
        Vmodel = (VN[i] - J0) @ Rv.T + J0
        v2d = cb.project_guava(Vmodel, RT.astype(np.float64), invtanfov)
        res_px512 = float(np.linalg.norm(v2d - v2d_src, axis=1).mean() * 512 / CROP)
        pc = pose[i].astype(np.float32)
        smplx = {"global_pose": cb.aa_from_R(Rroot).astype(np.float32), "body_pose": pc[3:66].reshape(21, 3).copy(),
                 "jaw_pose": pc[66:69].reshape(1, 3).copy(), "eye_pose": np.zeros((2, 3), np.float32),
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
                                "eye_pose_params": np.zeros(6, np.float32), "eyelid_params": np.zeros(2, np.float32),
                                "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
               "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3.copy(), "M_c2o-hd": I3.copy()},
               "head_crop": hc, "left_hand_crop": lc, "right_hand_crop": rc, "invtanfov": float(invtanfov), "view": "0"}
        out.append({"rec": rec, "H": Hm, "v2d_src": v2d_src.astype(np.float32), "v2d": v2d.astype(np.float32), "res_mm": res_mm,
                    "res_px512": res_px512, "invtanfov": float(invtanfov), "crop_side_px": L, "depth_m": float(XL[:, 2].mean()),
                    "zmin_m": float(XL[:, 2].min()), "vis": {"head": hv, "lhand": lv, "rhand": rv}, "focal_xy_rel": abs(fx - fy) / fx})
    return out, b200, {"betas10_median": np.median(b10, 0).tolist(), "betas10_max_abs_dev": float(np.abs(b10 - np.median(b10, 0)).max())}


def ub_work(args):
    clip, P, M = args
    import cv2, torch, torchvision
    cv2.setNumThreads(1); torch.set_num_threads(1); t0 = time.time()
    try:
        g, b200, sinfo = ub_geometry(clip, P, M); t1 = time.time()
        TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as td:
            subprocess.run([FFMPEG, "-loglevel", "error", "-i", clip["video"], "-f", "image2", "-r", "30", f"{td}/%06d.png"], check=True)
            n_ext = len(os.listdir(td))
            if n_ext != clip["video_frames"]:
                raise RuntimeError(f"ffmpeg -r 30 gave {n_ext} frames, annotations {clip['video_frames']}")
            t2 = time.time(); crops, jpgs = [], []
            for fr, gi in zip(clip["frames"], g):
                im = cv2.cvtColor(cv2.imread(f"{td}/{fr + 1:06d}.png"), cv2.COLOR_BGR2RGB)
                c = cv2.warpPerspective(im, gi["H"], (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                crops.append(c); jpgs.append(torchvision.io.encode_jpeg(torch.from_numpy(c).permute(2, 0, 1).contiguous(), quality=95).numpy().tobytes())
        return {"clip": clip, "g": g, "frames": list(clip["frames"]), "crops": np.stack(crops), "jpgs": jpgs, "betas200": b200, "shape_info": sinfo,
                "missing": [], "dropped_zmin": [], "t": {"geom": t1 - t0, "decode": t2 - t1, "crop_jpg": time.time() - t2}, "err": None}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"clip": clip, "err": repr(e) + "\n" + traceback.format_exc()}


def b2_work(args):
    clip, betas = args
    r = cb.work(({"name": clip["clip"], "job": clip["job"], "seq": clip["seq"], "subject": clip["identity"], "split": "test",
                  "rows": clip["rows"], "frames": clip["frames"], "img_wh": clip["wh"], "camera": clip["camera"]}, betas))
    r["clip"] = clip; r["betas200"] = np.asarray(betas, np.float64); r["shape_info"] = {}
    return r


def work(args):
    return ub_work(args) if args[0]["dataset"] == "ubody" else b2_work(args)


# ------------------------------------------------------------------------------------------------ main process (GPU)
def write_tracked(dst, res, keys, alphas, flame_shape, meta):
    import lmdb, torch, torchvision
    loc = TMP / "clips" / dst.name
    if loc.exists(): shutil.rmtree(loc)
    loc.mkdir(parents=True)
    with open(loc / "optim_tracking_ehm.pkl", "wb") as f:
        pickle.dump({k: g["rec"] for k, g in zip(keys, res["g"])}, f)
    ids = {"smplx_shape": np.asarray(res["betas200"], np.float32).reshape(1, 200), "flame_shape": np.asarray(flame_shape, np.float32).reshape(1, 300),
           "left_mano_shape": np.zeros((1, 10), np.float32), "right_mano_shape": np.zeros((1, 10), np.float32),
           "head_scale": np.ones((1, 3), np.float32), "hand_scale": np.ones((1, 3), np.float32), "joints_offset": np.zeros((1, 55, 3), np.float32)}
    with open(loc / "id_share_params.pkl", "wb") as f: pickle.dump(ids, f)
    (loc / "videos_info.json").write_text(json.dumps({dst.name: {"frames_num": len(keys), "frames_keys": keys}}, indent=1))
    env = lmdb.open(str(loc / "img_lmdb"), map_size=1 << 33); txn = env.begin(write=True); mjpg = []
    for k, jb, a in zip(keys, res["jpgs"], alphas):
        txn.put(f"{k}/body_image".encode(), jb)
        mb = torchvision.io.encode_jpeg(torch.from_numpy(a)[None].expand(3, -1, -1).contiguous(), quality=95).numpy().tobytes()
        txn.put(f"{k}/body_mask".encode(), mb); mjpg.append(mb)
    txn.commit(); env.close()
    gi = np.unique(np.round(np.linspace(0, len(keys) - 1, min(10, len(keys)))).astype(int))
    np.savez_compressed(loc / "gt_proj.npz", frame_key=np.array([keys[i] for i in gi]), v2d_1024=np.stack([res["g"][i]["v2d_src"] for i in gi]),
                        v2d_converted_1024=np.stack([res["g"][i]["v2d"] for i in gi]))
    (loc / "meta.json").write_text(json.dumps(meta))
    tmp = dst.parent / (dst.name + ".partial")
    if tmp.exists(): shutil.rmtree(tmp)
    shutil.copytree(loc, tmp, copy_function=shutil.copyfile)                                       # SMB: no metadata copy
    if dst.exists(): shutil.rmtree(dst)                                                            # only our own earlier output
    os.rename(tmp, dst); shutil.rmtree(loc)
    return mjpg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clips", default=""); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smpl-iters", type=int, default=600); a = ap.parse_args()
    import cv2, torch, torchvision
    import multiprocessing as mp
    sys.path.insert(0, str(cb.GUAVA)); from tools import eval_3dpw_standard as ev
    J = json.loads(SPLIT.read_text()); clips = J["clips"]
    if a.clips: want = a.clips.split(","); clips = sorted([c for c in clips if c["id"] in want], key=lambda c: want.index(c["id"]))
    todo = [c for c in clips if not (E / "base" / c["dataset"] / c["clip"] / "frames.json").exists()]
    print(f"{len(clips)} clips, {len(todo)} to do", flush=True)
    M = maps(); (E / "reports").mkdir(parents=True, exist_ok=True)
    (E / "reports" / "conversion_maps.json").write_text(json.dumps(M["report"], indent=1))
    # UBody GT params (per scene annotation file, only the split's annotation ids)
    z = zipfile.ZipFile("/raid/ubx858/datasets/raw/UBody/annotations.zip"); P = {}
    for scene in sorted({c["scene"] for c in todo if c["dataset"] == "ubody"}):
        sx = json.load(z.open(f"annotations/{scene}/smplx_annotation.json"))
        for c in todo:
            if c["dataset"] == "ubody" and c["scene"] == scene: P[c["id"]] = [sx[str(i)] for i in c["ann_ids"]]
        del sx
    bmap = np.load(cb.BMAP); Mb, cb0 = bmap["M"].astype(np.float64), bmap["c"].astype(np.float64)
    b2s = {}
    for c in todo:
        if c["dataset"] == "bedlam2":
            d = cb.job_npz(c["job"]); b2s[c["id"]] = (Mb @ d["shape16"][c["rows"][0]].astype(np.float64) + cb0).tolist()
    jobs = [(c, P[c["id"]], M) if c["dataset"] == "ubody" else (c, b2s[c["id"]]) for c in todo]
    dev = torch.device("cuda:0"); matter = cb.Matter()
    mapping = ev.load_smplx_to_smpl(SMPLX2SMPL).to(dev); smpl = ev.build_smpl(SMPL_DIR, "neutral", dev)
    from smplx_lean import SMPLXLean
    N20 = SMPLXLean(str(X2020), 200, dtype=torch.float32).to(dev)
    log = open(E / "reports" / "convert_log.jsonl", "a"); t00 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for res in pool.imap(work, jobs, chunksize=1):
            c = res["clip"]; ds, name = c["dataset"], c["clip"]
            if res["err"] or res.get("crops") is None:
                print("FAIL", c["id"], (res["err"] or "no frames")[:400], flush=True); log.write(json.dumps({"id": c["id"], "err": res["err"]}) + "\n"); continue
            t0 = time.time(); alphas = matter(res["crops"]); tm = time.time() - t0; res["crops"] = None
            keys = [f"frame_{f:06d}" for f in res["frames"]]
            fs = M["Wf"] @ np.asarray(res["betas200"], np.float64)
            meta = {"id": c["id"], "dataset": ds, "clip": name, "frames_keys": keys, "missing_frames": res["missing"], "dropped_zmin": res["dropped_zmin"],
                    "res_mm": [g["res_mm"] for g in res["g"]], "res_px512": [g["res_px512"] for g in res["g"]], "invtanfov": [g["invtanfov"] for g in res["g"]],
                    "crop_side_px": [g["crop_side_px"] for g in res["g"]], "depth_m": [g["depth_m"] for g in res["g"]], "part_visible": [g["vis"] for g in res["g"]],
                    "alpha_mean": [float(x.mean() / 255) for x in alphas], "shape_info": res["shape_info"], "times": res["t"],
                    "licence": J["licences"][ds]}
            mjpg = write_tracked(E / "data" / ds / name, res, keys, alphas, fs, meta)
            # decoded JPEG -> 512, exactly as fb_common.FBData (float, antialiased resize)
            dec = lambda b: torchvision.io.decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).float() / 255.0    # noqa: E731
            rs = lambda x: torchvision.transforms.functional.resize(x, (512, 512), antialias=True)                           # noqa: E731
            u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8)                                                      # noqa: E731
            B = E / "base" / ds / name; G5 = E / "gt512" / ds / name
            for p in (B / "images", B / "masks", G5): p.mkdir(parents=True, exist_ok=True)
            fset = set(f"frame_{f:06d}" for f in c["test_frames"]); kept = set(keys)
            for k, jb, mb in zip(keys, res["jpgs"], mjpg):
                im, ms = rs(dec(jb)), rs(dec(mb))
                torchvision.io.write_png(u8(im), str(B / "images" / f"{k}.png")); torchvision.io.write_png(u8(ms[:1]), str(B / "masks" / f"{k}.png"))
                if k in fset: torchvision.io.write_png(u8(im * ms), str(G5 / f"{k}.png"))
            # GT SMPL-X 2020 in the crop camera (OpenCV axes) + K at 512
            R = [g["rec"]["smplx_coeffs"] for g in res["g"]]; Dm = np.diag([-1.0, -1.0, 1.0])
            itf = np.array([g["invtanfov"] for g in res["g"]])
            K = np.zeros((len(keys), 3, 3)); K[:, 0, 0] = K[:, 1, 1] = 256 * itf; K[:, 0, 2] = K[:, 1, 2] = 256; K[:, 2, 2] = 1
            sx = {"keys": np.array(keys), "betas": np.asarray(res["betas200"], np.float32),
                  "global_orient": np.stack([r["global_pose"] for r in R]).astype(np.float32), "body_pose": np.stack([r["body_pose"] for r in R]).astype(np.float32),
                  "jaw_pose": np.stack([np.asarray(r["jaw_pose"]).reshape(3) for r in R]).astype(np.float32),
                  "leye_pose": np.stack([np.asarray(r["eye_pose"]).reshape(2, 3)[0] for r in R]).astype(np.float32),
                  "reye_pose": np.stack([np.asarray(r["eye_pose"]).reshape(2, 3)[1] for r in R]).astype(np.float32),
                  "left_hand_pose": np.stack([r["left_hand_pose"] for r in R]).astype(np.float32), "right_hand_pose": np.stack([r["right_hand_pose"] for r in R]).astype(np.float32),
                  "expression": np.stack([r["exp"] for r in R]).astype(np.float32),
                  "transl": np.stack([Dm @ np.asarray(r["camera_RT_params"], np.float64)[:, 3] for r in R]).astype(np.float32)}
            np.savez(B / "smplx2020.npz", **sx); np.savez(B / "cameras.npz", keys=np.array(keys), K=K.astype(np.float32))
            pose = np.concatenate([sx["global_orient"], sx["body_pose"].reshape(-1, 63), sx["jaw_pose"], sx["leye_pose"], sx["reye_pose"],
                                   sx["left_hand_pose"].reshape(-1, 45), sx["right_hand_pose"].reshape(-1, 45)], 1)
            with torch.no_grad():
                Vx, Jx = N20(torch.tensor(pose, device=dev), torch.tensor(sx["betas"], device=dev)[None].expand(len(keys), -1), torch.tensor(sx["expression"], device=dev))
                Vx = Vx + torch.tensor(sx["transl"], device=dev)[:, None]
            q = (Vx[..., :2] / Vx[..., 2:]).cpu().numpy() * K[:, 0, 0][:, None, None] + 256
            v2d = np.stack([g["v2d"] for g in res["g"]]) / 2
            chk_px = float(np.abs(q - v2d).max())                                                  # contract camera vs GUAVA projection
            # SMPL fit (one betas per clip)
            t1 = time.time(); tg = torch.stack([torch.sparse.mm(mapping, v) for v in Vx])
            nf = len(keys); po = torch.zeros(nf, 72, device=dev); po[:, :66] = torch.tensor(pose[:, :66], device=dev)
            po.requires_grad_(True); be = torch.zeros(1, 10, device=dev, requires_grad=True)
            with torch.no_grad(): trl = (tg - smpl(global_orient=po[:, :3], body_pose=po[:, 3:], betas=be.expand(nf, -1)).vertices).mean(1)
            trl.requires_grad_(True); opt = torch.optim.Adam([po, be, trl], lr=0.02); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.smpl_iters)
            for _ in range(a.smpl_iters):
                opt.zero_grad(set_to_none=True)
                v = smpl(global_orient=po[:, :3], body_pose=po[:, 3:], betas=be.expand(nf, -1), transl=trl).vertices
                ((v - tg).norm(dim=-1).mean() + 1e-4 * be.square().mean()).backward(); opt.step(); sch.step()
            with torch.no_grad():
                v = smpl(global_orient=po[:, :3], body_pose=po[:, 3:], betas=be.expand(nf, -1), transl=trl).vertices
                ferr = ((v - tg).norm(dim=-1).mean(1) * 1000).cpu().numpy()
            np.savez(B / "smpl.npz", keys=np.array(keys), betas=be[0].detach().cpu().numpy(), global_orient=po[:, :3].detach().cpu().numpy(),
                     body_pose=po[:, 3:].detach().cpu().numpy().reshape(nf, 23, 3), transl=trl.detach().cpu().numpy(), fit_err_mm=ferr)
            ts = time.time() - t1
            # overlay: 4 frames, GT SMPL-X (green) and SMPL fit (magenta) vertices
            tiles = []
            for i in np.unique(np.round(np.linspace(0, nf - 1, 4)).astype(int)):
                im = cv2.cvtColor(cv2.imread(str(B / "images" / f"{keys[i]}.png")), cv2.COLOR_BGR2RGB)
                for pts, col in ((q[i][::7], (0, 255, 0)), (((v[i, :, :2] / v[i, :, 2:]).cpu().numpy() * K[i, 0, 0] + 256)[::5], (255, 0, 255))):
                    for x, y in pts.astype(int): cv2.circle(im, (int(x), int(y)), 1, col, -1)
                tiles.append(im)
            cv2.imwrite(str(B / "overlay.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            tk = [k for k in keys if k in fset]; rk = [f"frame_{f:06d}" for f in c["frames"][:c["n_recon"]] if f"frame_{f:06d}" in kept]
            fj = {"dataset": ds, "clip": name, "id": c["id"], "keys": keys, "source": f"frame_{c['source_frame']:06d}", "recon": rk, "test": tk,
                  "fps": 30 if ds == "ubody" else 6, "dropped_from_split": sorted(set(f"frame_{f:06d}" for f in c["frames"]) - kept),
                  "note": "frame_XXXXXX = UBody ffmpeg -r 30 index (0-based)" if ds == "ubody" else "frame_XXXXXX = BEDLAM2 video frame (labels at 6 fps)"}
            assert fj["source"] in kept, (c["id"], "source frame dropped")
            (B / "frames.json").write_text(json.dumps(fj, indent=1))
            rec = {"id": c["id"], "frames": nf, "test": len(tk), "res_mm_mean": float(np.mean(meta["res_mm"])), "res_mm_max": float(np.max(meta["res_mm"])),
                   "res_px512_mean": float(np.mean(meta["res_px512"])), "contract_cam_check_px": chk_px, "smpl_fit_mm_mean": float(ferr.mean()),
                   "smpl_fit_mm_max": float(ferr.max()), "alpha_mean": float(np.mean(meta["alpha_mean"])), "matte_s": round(tm, 1), "smpl_s": round(ts, 1),
                   **{k: round(v, 1) for k, v in res["t"].items()}, "dropped": fj["dropped_from_split"], "elapsed_s": round(time.time() - t00, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
