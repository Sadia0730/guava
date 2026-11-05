"""PREREG phase 6: replay of recorded clips through the live full-body pipeline with an asynchronous background worker.

Live loop (every frame, never waits for the worker): cached YOLOX + H1c box -> SPRINT raw on its detector crop (shape lock to the
source fit) -> + the latest published pose correction -> GUAVA-FB render (512, SPRINT crop camera) -> warped into the metric crop.
Frames are released at their 30 fps timestamps; clock_t = max(timestamp_t, finish_t-1) (no drops). Live work runs on a
high-priority CUDA stream and synchronises only that stream (no device-wide sync on the render path).
Worker (separate thread, low-priority CUDA stream, at most one update in flight): offered the current frame whenever idle ->
gate (pose novelty >= 15 deg | every n-th offered frame since the last acceptance) -> k = 60 refinement (empty-matte fix on) ->
fit-quality gate (keypoint error, matte) -> step 1: correction = refined - raw, EMA, published by reference swap (double buffer).

Arms: k0 (worker off), pose (step 1), k60 (offline upper bound: every scored frame refined with k = 60, no async), uv (step 2
alone on k = 0 + H1c: C3 appearance update, uv_fusion.py; the live loop renders the latest published avatar).
Outputs: renders/<name>/<dataset>/<clip>/<key>.png (+ sil/<key>.png: depth > 0 silhouette; unseen/<key>.png: pixels whose SMPL-X
surface had zero source visibility, for the unseen-region LPIPS; written after the timed loop) and run.json (per-frame timings,
worker log) + live_poses.npz (live SPRINT pose / box / crop / warp of every scored frame) under E6 = O/eval/phase6_val (4D-DRESS
sessions and copies of the ValBench-v1 BEDLAM2 / ValBench-S UBody base / gt512).
Usage: python replay.py --arm pose --part tuning [--clips id,..] [--alpha 0.5] [--corr all|body|cam] [--name p6_pose_a05_all]
       python replay.py --arm uv --part tuning [--prior 2] [--upd-thr 0.3] [--tmpl-avg]
       python replay.py --arm k0|uv --part valbench_s (ValBench-S UBody, upper-body no-regression check)"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=("k0", "pose", "k60", "uv")); ap.add_argument("--part", default="tuning", choices=("tuning", "verdict", "all", "valbench_s"))
ap.add_argument("--clips", default=""); ap.add_argument("--name", default="")
ap.add_argument("--alpha", type=float, default=0.5, help="EMA rate of the pose correction (1 = latest refined frame only)")
ap.add_argument("--corr", default="all", choices=("all", "body", "cam"), help="corrected parts: all joints + camera | root + body + camera | camera only")
ap.add_argument("--cap-deg", type=float, default=30.0); ap.add_argument("--novelty-deg", type=float, default=15.0); ap.add_argument("--every-n", type=int, default=10)
ap.add_argument("--kp-thr", type=float, default=20.0, help="max mean 2D keypoint error (px at 1024) of the refined fit")
ap.add_argument("--drive-iters", type=int, default=60); ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--split", default=str(HERE / "p6_val.json")); ap.add_argument("--E", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/phase6_val")
ap.add_argument("--sprint-ckpt", default="/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
ap.add_argument("--fb-ckpt", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
ap.add_argument("--dwpose-threads", type=int, default=4); ap.add_argument("--max-frames", type=int, default=0, help="smoke: replay only the first N frames")
ap.add_argument("--boxes-only", action="store_true", help="only build the per-clip YOLOX + H1c box cache (shared by every arm)")
ap.add_argument("--prior", type=float, default=2.0, help="step 2: anchor (source) prior weight, x its own weight")
ap.add_argument("--upd-thr", type=float, default=0.3, help="step 2: a texel is updated only where the frame weight >= this")
ap.add_argument("--tmpl-avg", action="store_true", help="step 2 tuning option: template branch with averaged vertex features")
ap.add_argument("--heldout-every", type=int, default=6, help="step 2: every n-th released non-scored frame is held out (never fused)")
ap.add_argument("--heldout-k", type=int, default=4, help="step 2: held-out reservoir size (most recent refined held-out frames)")
ap.add_argument("--heldout-gap", type=int, default=30, help="step 2: refine a held-out frame only if the reservoir is not full or "
                "the newest entry is >= this many released frames old")
ap.add_argument("--vbs-split", default=str(HERE.parent / "eval_split" / "valbench_s.json"))
ap.add_argument("--vbs-E", default="/dev/shm/ubx858_eval/valbench_s", help="ValBench-S base / gt512 (copied into E6 if missing)")
a = ap.parse_args()
NAME = a.name or {"k0": "p6_k0", "k60": "p6_k60", "pose": f"p6_pose_a{a.alpha:g}_{a.corr}",
                  "uv": f"p6_uv_p{a.prior:g}_t{a.upd_thr:g}_{'tavg' if a.tmpl_avg else 'notavg'}"}[a.arm]

spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / "phase3_dnt" / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = SAF.SprintFitter(a.sprint_ckpt, "cuda:0", 150, dwpose_threads=a.dwpose_threads); fitter.empty_matte_fix = True

import cv2, roma, tempfile, torch, torchvision                                                       # noqa: E402
if os.environ.get("OMP_NUM_THREADS"): cv2.setNumThreads(int(os.environ["OMP_NUM_THREADS"]))
sys.path.insert(0, str(ROOT / "phase3_dnt" / "guava_fb")); sys.path.insert(0, str(ROOT / "phase3_dnt" / "eval_split"))
import fb_common as F                                                                                 # noqa: E402
sys.path.insert(0, str(HERE)); import worker_core as WC                                             # noqa: E402
import uv_fusion as UF                                                                                # noqa: E402
import convert_bedlam2 as cb                                                                          # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                          # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                         # noqa: E402
dev = torch.device("cuda:0"); GPU = torch.cuda.get_device_name(0)
if not a.boxes_only:
    cfg, inf, ren, _ = F.setup_a20(dev)
    st_ = torch.load(a.fb_ckpt, map_location="cpu", weights_only=False)
    inf.load_state_dict(st_["model"], strict=False); ren.load_state_dict(st_["render_model"], strict=False); inf.eval(); ren.eval(); del st_
lo_pri, hi_pri = torch.cuda.Stream.priority_range()                                                  # (low = 0, high = negative)
LIVE = torch.cuda.Stream(priority=hi_pri); WORK = torch.cuda.Stream(priority=lo_pri)
E = Path(a.E); S512 = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])


# ------------------------------------------------------------------------------------------------ data
def clip_frames(c):
    if c["dataset"] == "dress4d":                                                                   # preloaded: disk reads are not camera time
        fj = json.loads((E / "base" / "dress4d" / c["clip"] / "frames.json").read_text()); fr = c["frames"][:a.max_frames] if a.max_frames else c["frames"]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16) as ex:
            ims = dict(zip(fr, ex.map(lambda f: cv2.cvtColor(cv2.imread(fj["frame_src"][f"frame_{f:06d}"]), cv2.COLOR_BGR2RGB), fr)))
        return fr, (lambda f: ims[f]), ims
    fr = list(range(c["source_frame"], max(c["test_frames"]) + 1))
    if c["dataset"] == "ubody":                                                                     # ValBench-S: the video on the 30 fps timeline
        import subprocess, convert_test as CT
        CT.TMP.mkdir(parents=True, exist_ok=True); ims = {}
        with tempfile.TemporaryDirectory(dir=CT.TMP) as td:
            subprocess.run([CT.FFMPEG, "-loglevel", "error", "-i", c["video"], "-f", "image2", "-r", "30", f"{td}/%06d.png"], check=True)
            for f in fr: ims[f] = cv2.cvtColor(cv2.imread(f"{td}/{f + 1:06d}.png"), cv2.COLOR_BGR2RGB)
        return fr, (lambda f: ims[f]), ims
    ims = cb.decode_frames(c["job"], c["seq"], fr)
    return fr, (lambda f: ims[f]), ims


def metric_H(c):
    if c["dataset"] == "dress4d":
        d = np.load(E / "base" / "dress4d" / c["clip"] / "crop_h.npz"); return {int(str(k).split("_")[1]): h for k, h in zip(d["keys"], d["H"])}
    if c["dataset"] == "ubody":                                                                     # eval_guava_rows.metric_H (UBody branch)
        import zipfile, convert_test as CT
        sx = json.load(zipfile.ZipFile("/raid/ubx858/datasets/raw/UBody/annotations.zip").open(f"annotations/{c['scene']}/smplx_annotation.json"))
        g, _, _ = CT.ub_geometry(c, [sx[str(i)] for i in c["ann_ids"]], CT.maps()); return {f: gi["H"] for f, gi in zip(c["frames"], g)}
    d = cb.job_npz(c["job"]); bm = np.load(cb.BMAP)
    betas = bm["M"].astype(np.float64) @ d["shape16"][c["rows"][0]].astype(np.float64) + bm["c"].astype(np.float64)
    g = cb.clip_geometry({"rows": c["rows"]}, d, betas); return {f: gi["H"] for f, gi in zip(c["frames"], g)}


def iou(p, q):
    w, h = max(0.0, min(p[2], q[2]) - max(p[0], q[0])), max(0.0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    return i / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i, 1e-9)


def boxes_cached(c, frames, get):
    """YOLOX + H1c linking per frame (identical for every arm); cached with the detector's CPU ms. UBody (ValBench-S): the source
    frame's subject = the split's subject box (as H1c on ValBench-S); otherwise the largest box."""
    p = Path("/dev/shm/ubx858_eval/p6_boxes") / f"{c['id'].replace('/', '__')}.npz"
    if p.exists():
        z = np.load(p)
        if set(frames) <= set(int(f) for f in z["frames"]):                                     # a shorter (smoke) cache is recomputed
            return {int(f): b for f, b in zip(z["frames"], z["boxes"])}, {int(f): float(t) for f, t in zip(z["frames"], z["det_ms"])}
    B, T, ref = {}, {}, None
    if c["dataset"] == "ubody":
        b0 = c["bbox_xywh"][c["frames"].index(frames[0])]; ref = np.array([b0[0], b0[1], b0[0] + b0[2], b0[1] + b0[3]], np.float64)
    for f in frames:
        img = get(f); t0 = time.perf_counter(); D = fitter.det.inference_detector(fitter.s_det, img); dt = (time.perf_counter() - t0) * 1000
        if len(D) == 0: b = np.asarray(ref if ref is not None else [0, 0, img.shape[1] - 1, img.shape[0] - 1], np.float64)
        elif ref is None: b = D[int(np.argmax((D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1])))]
        else:
            v = [iou(d, ref) for d in D]; j = int(np.argmax(v))
            if v[j] >= 0.1: b = D[j]
            else:
                c0 = np.array([(ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2]); sz = max(ref[2] - ref[0], ref[3] - ref[1], 1.0)
                b = D[int(np.argmin([np.linalg.norm(np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2]) - c0) / sz for d in D]))]
        B[f] = np.asarray(b, np.float64)[:4]; T[f] = dt; ref = B[f]
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, frames=np.array(frames), boxes=np.stack([B[f] for f in frames]), det_ms=np.array([T[f] for f in frames]))
    return B, T


# ------------------------------------------------------------------------------------------------ live pieces
def live_raw(img, box, shp):
    """SPRINT raw on the detector crop (fit()'s raw conversion; the caller's current stream only)."""
    H, W = img.shape[:2]; M = SAF.box_crop_matrix(box, SAF.CROP)
    crop = cv2.warpAffine(img, M[:2], (SAF.CROP, SAF.CROP), flags=cv2.INTER_LINEAR)
    crop_t = torch.from_numpy(crop).pin_memory().to(dev, non_blocking=True).permute(2, 0, 1)[None].float() / 255.0
    cam = fitter.ev.cam_feat((np.diag([256 / SAF.CROP, 256 / SAF.CROP, 1.0]) @ M)[:2], W, H, W, 0.0)
    with torch.no_grad(): out = fitter.sprint(crop_t, cam, shp)
    bp0 = out["body_param"]; aa = lambda Rm: roma.rotmat_to_rotvec(Rm.reshape(-1, 3, 3))                # noqa: E731
    flat = torch.cat([aa(bp0[k]).reshape(-1) for k in ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose")]
                     + [bp0["exp"][0, :50].reshape(-1), out["pd_cam"][0, :3, :4].reshape(-1)]).cpu().numpy().astype(np.float32)   # one stream sync
    o = 0; raw = {}
    for k, n in (("global_pose", 3), ("body_pose", 63), ("left_hand_pose", 45), ("right_hand_pose", 45), ("jaw_pose", 3), ("exp", 50), ("camera_RT_params", 12)):
        raw[k] = flat[o:o + n]; o += n
    raw = {"global_pose": raw["global_pose"], "body_pose": raw["body_pose"].reshape(21, 3), "left_hand_pose": raw["left_hand_pose"].reshape(15, 3),
           "right_hand_pose": raw["right_hand_pose"].reshape(15, 3), "jaw_pose": raw["jaw_pose"].reshape(1, 3), "eye_pose": np.zeros((2, 3), np.float32),
           "exp": raw["exp"], "camera_RT_params": raw["camera_RT_params"].reshape(3, 4), "body_cam": np.zeros(3, np.float32)}
    return raw, M


def record_of(smplx, box, M):
    Mlr = SAF.box_crop_matrix(box, SAF.BODY_LR); Mp = SAF.box_crop_matrix(box, SAF.PART)
    return {"smplx_coeffs": smplx, "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": M, "M_c2o-hd": np.linalg.inv(M).astype(np.float32)},
            **{p: {"M_o2c": Mp, "M_c2o": np.linalg.inv(Mp).astype(np.float32)} for p in ("head_crop", "left_hand_crop", "right_hand_crop")}}


@torch.no_grad()
def render_rec(av, rec, shp):
    comp = F.compact_record(rec, shp); sm = {k: torch.as_tensor(v.copy()) for k, v in comp["smplx"].items()}
    sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
    M4 = torch.eye(4); M4[:3, :4] = sm["camera_RT_params"]; w2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])) @ M4
    t = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)}, "w2c_cam": w2c, "c2w_cam": torch.linalg.inv(w2c), "invtanfov": comp["invtanfov"]}
    t["render_cam_params"] = F.render_cam_params(t["w2c_cam"], t["c2w_cam"], t["invtanfov"], 512)
    t = to_dev(F.collate1(t), dev); torch.manual_seed(0); rr = ren(av(t), t["render_cam_params"], bg=0.0)
    return rr["renders"][0], rr["depths"][0, 0]


# ------------------------------------------------------------------------------------------------ main
def u8(x): return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def run_clip(c):
    ds, name = c["dataset"], c["clip"]; od = E / "renders" / NAME / ds / name
    if (od / "run.json").exists(): return
    loc = Path("/dev/shm/ubx858_eval/p6_renders") / NAME / ds / name
    if loc.exists(): shutil.rmtree(loc)
    (loc / "sil").mkdir(parents=True)
    frames, get, _ = clip_frames(c); frames = frames[:a.max_frames] if a.max_frames else frames
    scored = [f for f in c["test_frames"] if f in set(frames)]; sset = set(scored); Hm = metric_H(c)
    B, DET = boxes_cached(c, frames, get)
    sf = frames[0]; img0 = get(sf); rs = fitter.fit(img0, box=B[sf]); shp = rs["shape"]
    with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
        key = f"frame_{sf:06d}"; SAF.write_tracked(Path(td), name, key, rs)
        cl = F.load_clips_parallel([(td, None, True)], workers=1); fbd = F.FBData(cfg, cl); src = to_dev(F.collate1(fbd.source_info(0, key)), dev)
    with torch.no_grad():
        torch.manual_seed(0); vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    info = {"method": NAME, "arm": a.arm, "args": vars(a), "gpu": GPU, "frames": len(frames), "scored": len(scored), "source_fit_kp_err": rs["diag"]["kp_err_px1024_after"]}
    worker = None; fu = None
    idx_scored = {i for i, f in enumerate(frames) if f in sset}                                   # step 2: never fused
    nonsc = [i for i in range(1, len(frames)) if i not in idx_scored]; idx_held = set(nonsc[a.heldout_every - 1::a.heldout_every])
    if a.arm == "pose":
        worker = WC.Worker(fitter, shp, WORK, "pose", alpha=a.alpha, corr=a.corr, cap_deg=a.cap_deg, novelty_deg=a.novelty_deg,
                           every_n=a.every_n, kp_thr=a.kp_thr, iters=a.drive_iters); worker.start()
    elif a.arm == "uv":
        t0 = time.perf_counter()
        fu = UF.UVFusion(cfg, inf, ren, src, rs, img0, shp, fitter, SAF, prior=a.prior, upd_thr=a.upd_thr, tmpl_avg=a.tmpl_avg,
                         heldout_k=a.heldout_k, dev="cuda:0")
        info["fusion"] = {"init_ms": (time.perf_counter() - t0) * 1000, "static_identity": gaussian_diff(av, fu.av_work),
                          "arc_sim0": fu.arc_sim, "source_face_landmarks": fu.kps5 is not None, "held_out_frames": len(idx_held),
                          "source_head_yaw_pitch_deg": fu.src_yaw_pitch}
        worker = WC.Worker(fitter, shp, WORK, "uv", novelty_deg=a.novelty_deg, every_n=a.every_n, kp_thr=a.kp_thr, iters=a.drive_iters,
                           fusion=fu, static_av=av, src_raw=rs["raw_smplx"], scored=idx_scored, heldout=idx_held,
                           heldout_gap=a.heldout_gap); worker.start()
    wq = queue.Queue()

    def writer():
        while (it := wq.get()) is not None:
            d, k, im, sl, W = it
            cv2.imwrite(str(d / f"{k}.png"), cv2.cvtColor(cv2.warpPerspective(im, W, (512, 512), flags=cv2.INTER_LINEAR, borderValue=0), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(d / "sil" / f"{k}.png"), cv2.warpPerspective(sl, W, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0))
    wt = threading.Thread(target=writer, daemon=True); wt.start()
    per = {}; t_start = time.perf_counter(); clock_lag = []; poses = {}; sil_crop = {}
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i, f in enumerate(frames):
        if a.arm == "k60" and f not in sset: continue                                              # offline upper bound: scored frames only
        img = get(f); box = B[f]
        target = t_start + i / a.fps; now = time.perf_counter()
        if a.arm != "k60":
            if now < target: time.sleep(target - now)
            clock_lag.append(max(0.0, now - target))
        te0 = time.perf_counter()
        with torch.cuda.stream(LIVE):
            ev0.record(LIVE)
            if a.arm == "k60":
                r = fitter.fit(img, box=box, iters=a.drive_iters, shape_lock=shp); sm, M = r["record"]["smplx_coeffs"], r["M"]; corr_v = -1
            else:
                raw, M = live_raw(img, box, shp); corr = worker.published if worker else None; corr_v = worker.version if worker else 0
                sm = corr.apply(raw) if corr is not None else raw
                if worker: worker.offer(i, img, box, raw)
        # render on the default stream: the rasterizer launches its kernels on the legacy default stream (no stream argument), so
        # its buffers must be allocated there too; SPRINT's stream is already synchronised by its .cpu() read-back
        av_cur, av_v = worker.published_av if a.arm == "uv" else (av, 0)                           # one read: the latest published avatar
        rgb, dep = render_rec(av_cur, record_of(sm, box, M), shp)
        ev1.record(torch.cuda.default_stream())
        img_u8 = u8(rgb).permute(1, 2, 0).cpu().numpy(); sil = (dep > 1e-4).to(torch.uint8).mul(255).cpu().numpy()
        gpu_ms = ev0.elapsed_time(ev1); e2e = (time.perf_counter() - te0) * 1000
        per[f] = {"gpu_ms": gpu_ms, "e2e_ms": e2e, "det_ms": DET[f], "corr_version": corr_v, "av_version": av_v}
        if f in sset:
            W = S512 @ Hm[f] @ np.linalg.inv(np.asarray(M, np.float64)) @ np.linalg.inv(S512); k = f"frame_{f:06d}"
            wq.put((loc, k, img_u8, sil, W))                                                         # warp + PNG in the writer thread
            poses[k] = ({kk: np.array(v) for kk, v in sm.items()}, np.asarray(box, np.float64), np.asarray(M, np.float64), W)
            if len(sil_crop) < 30: sil_crop[k] = sil
    wq.put(None); wt.join()
    if worker:
        worker.wait_idle(); worker.stop()
        info["worker"] = {"log": [{k: (v - t_start if k.startswith("t_") else v) for k, v in e.items()} for e in worker.log],
                          "updates": worker.av_version if a.arm == "uv" else worker.version, "offered": len(worker.log),
                          "update_ms": [ (e["t_end"] - e["t_start"]) * 1000 for e in worker.log if e.get("gate") == "accepted"],
                          "replay_s": time.perf_counter() - t_start}
        if fu is not None:
            from collections import Counter
            info["worker"].update(gates=dict(Counter(str(e.get("gate")).split(":")[0] for e in worker.log)), rollbacks=fu.n_rollback,
                                  final_coverage=dict(zip(UF.REGIONS, fu.coverage())), stopped=[UF.REGIONS[k] for k in range(4) if fu.stop[k]],
                                  heldout_reservoir=len(fu.res))
    # after the timed loop: live poses of the scored frames and the unseen-region masks (identical for every arm of a clip)
    src_vis = fu.src_face_vis if fu is not None else UF.face_visibility(inf, src, dev)[0]
    (loc / "unseen").mkdir(exist_ok=True); ious = []
    for k, (sm_, box_, M_, W_) in poses.items():
        um, ms = UF.unseen_mask(inf, src_vis, sm_, shp, dev)
        cv2.imwrite(str(loc / "unseen" / f"{k}.png"), cv2.warpPerspective(um, W_, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0))
        if k in sil_crop:
            p_, q_ = ms > 0, sil_crop[k] > 0; ious.append(float((p_ & q_).sum()) / max(float((p_ | q_).sum()), 1.0))
    info["mesh_vs_render_sil_iou"] = float(np.mean(ious)) if ious else None
    ks = list(poses)
    if ks:
        np.savez_compressed(loc / "live_poses.npz", keys=np.array(ks), box=np.stack([poses[k][1] for k in ks]), M=np.stack([poses[k][2] for k in ks]),
                            W=np.stack([poses[k][3] for k in ks]), shape=np.asarray(shp, np.float32), src_face_vis=src_vis.cpu().numpy(),
                            **{f"smplx_{kk}": np.stack([poses[k][0][kk] for k in ks]) for kk in poses[ks[0]][0]})
    g = np.array([v["gpu_ms"] for v in per.values()]); e = np.array([v["e2e_ms"] for v in per.values()])
    info.update(per_frame=per, gpu_ms_p50=float(np.percentile(g, 50)), gpu_ms_p95=float(np.percentile(g, 95)),
                e2e_ms_p50=float(np.percentile(e, 50)), e2e_ms_p95=float(np.percentile(e, 95)), clock_lag_ms_p95=float(np.percentile(clock_lag, 95) * 1000) if clock_lag else None,
                load_avg=os.getloadavg())
    (loc / "run.json").write_text(json.dumps(info))
    od.parent.mkdir(parents=True, exist_ok=True)
    if od.exists(): shutil.rmtree(od)                                                              # only our own earlier, unfinished output
    shutil.copytree(loc, od, copy_function=shutil.copyfile); shutil.rmtree(loc)
    print(json.dumps({"id": c["id"], **{k: info[k] for k in ("frames", "scored", "gpu_ms_p50", "gpu_ms_p95", "e2e_ms_p50")},
                      "updates": info.get("worker", {}).get("updates")}), flush=True)


def gaussian_diff(p, q):
    """max |difference| of the static avatar's Gaussians vs the fusion's decode of the source-only map (should be ~0)."""
    out = {}
    for k in ("_smplx_xyz", "_smplx_features_color", "_smplx_opacity", "_uv_local_xyz", "_uv_features_color", "_uv_opacity", "_uv_scaling"):
        x, y = getattr(p, k), getattr(q, k)
        out[k] = float((x - y).abs().max()) if x.shape == y.shape else f"shape {tuple(x.shape)} vs {tuple(y.shape)}"
    return out


def vbs_clips():
    """ValBench-S clips (part = 'valbench_s'); base / gt512 copied into E6 if missing (no overwrite)."""
    J = json.loads(Path(a.vbs_split).read_text()); out = []
    for c in J["clips"]:
        c = dict(c, part="valbench_s")
        for sub in ("base", "gt512"):
            src_, dst_ = Path(a.vbs_E) / sub / c["dataset"] / c["clip"], E / sub / c["dataset"] / c["clip"]
            if not dst_.exists(): dst_.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src_, dst_, copy_function=shutil.copyfile)
        out.append(c)
    return out


if __name__ == "__main__":
    J = {"clips": vbs_clips()} if a.part == "valbench_s" else json.loads(Path(a.split).read_text())
    clips = [c for c in J["clips"] if a.part == "all" or c["part"] == a.part]
    if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]
    for c in clips:
        if a.boxes_only:
            fr, get, _ = clip_frames(c); boxes_cached(c, fr, get); print("boxes", c["id"], len(fr), flush=True)
        else: run_clip(c)
