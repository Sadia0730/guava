"""PREREG phase 6: replay of recorded clips through the live full-body pipeline with an asynchronous background worker.

Live loop (every frame, never waits for the worker): cached YOLOX + H1c box -> SPRINT raw on its detector crop (shape lock to the
source fit) -> + the latest published pose correction -> GUAVA-FB render (512, SPRINT crop camera) -> warped into the metric crop.
Frames are released at their 30 fps timestamps; clock_t = max(timestamp_t, finish_t-1) (no drops). Live work runs on a
high-priority CUDA stream and synchronises only that stream (no device-wide sync on the render path).
Worker (separate thread, low-priority CUDA stream, at most one update in flight): offered the current frame whenever idle ->
gate (pose novelty >= 15 deg | every n-th offered frame since the last acceptance) -> k = 60 refinement (empty-matte fix on) ->
fit-quality gate (keypoint error, matte) -> step 1: correction = refined - raw, EMA, published by reference swap (double buffer).

Arms: k0 (worker off), pose (step 1), k60 (offline upper bound: every scored frame refined with k = 60, no async).
Outputs: renders/<name>/<dataset>/<clip>/<key>.png (+ sil/<key>.png: depth > 0 silhouette) and run.json (per-frame timings,
worker log) under E6 = O/eval/phase6_val (4D-DRESS sessions and copies of the ValBench-v1 BEDLAM2 base / gt512).
Usage: python replay.py --arm pose --part tuning [--clips id,..] [--alpha 0.5] [--corr all|body|cam] [--name p6_pose_a05_all]"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=("k0", "pose", "k60")); ap.add_argument("--part", default="tuning", choices=("tuning", "verdict", "all"))
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
a = ap.parse_args()
NAME = a.name or {"k0": "p6_k0", "k60": "p6_k60", "pose": f"p6_pose_a{a.alpha:g}_{a.corr}"}[a.arm]

spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / "phase3_dnt" / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = SAF.SprintFitter(a.sprint_ckpt, "cuda:0", 150, dwpose_threads=a.dwpose_threads); fitter.empty_matte_fix = True

import cv2, roma, tempfile, torch, torchvision                                                       # noqa: E402
if os.environ.get("OMP_NUM_THREADS"): cv2.setNumThreads(int(os.environ["OMP_NUM_THREADS"]))
sys.path.insert(0, str(ROOT / "phase3_dnt" / "guava_fb")); sys.path.insert(0, str(ROOT / "phase3_dnt" / "eval_split"))
import fb_common as F                                                                                 # noqa: E402
import convert_bedlam2 as cb                                                                          # noqa: E402
os.chdir(F.ROOT)
from fb_common import to_dev                                                                          # noqa: E402
from models.UbodyAvatar import Ubody_Gaussian                                                         # noqa: E402
dev = torch.device("cuda:0"); GPU = torch.cuda.get_device_name(0)
cfg, inf, ren, _ = F.setup_a20(dev)
st_ = torch.load(a.fb_ckpt, map_location="cpu", weights_only=False)
inf.load_state_dict(st_["model"], strict=False); ren.load_state_dict(st_["render_model"], strict=False); inf.eval(); ren.eval(); del st_
lo_pri, hi_pri = torch.cuda.Stream.priority_range()                                                  # (low = 0, high = negative)
LIVE = torch.cuda.Stream(priority=hi_pri); WORK = torch.cuda.Stream(priority=lo_pri)
E = Path(a.E); S512 = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])
ROT_KEYS = {"all": ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose"), "body": ("global_pose", "body_pose"), "cam": ()}[a.corr]


# ------------------------------------------------------------------------------------------------ data
def clip_frames(c):
    if c["dataset"] == "dress4d":
        fj = json.loads((E / "base" / "dress4d" / c["clip"] / "frames.json").read_text())
        return c["frames"], (lambda f: cv2.cvtColor(cv2.imread(fj["frame_src"][f"frame_{f:06d}"]), cv2.COLOR_BGR2RGB)), None
    fr = list(range(c["source_frame"], max(c["test_frames"]) + 1)); ims = cb.decode_frames(c["job"], c["seq"], fr)
    return fr, (lambda f: ims[f]), ims


def metric_H(c):
    if c["dataset"] == "dress4d":
        d = np.load(E / "base" / "dress4d" / c["clip"] / "crop_h.npz"); return {int(str(k).split("_")[1]): h for k, h in zip(d["keys"], d["H"])}
    d = cb.job_npz(c["job"]); bm = np.load(cb.BMAP)
    betas = bm["M"].astype(np.float64) @ d["shape16"][c["rows"][0]].astype(np.float64) + bm["c"].astype(np.float64)
    g = cb.clip_geometry({"rows": c["rows"]}, d, betas); return {f: gi["H"] for f, gi in zip(c["frames"], g)}


def iou(p, q):
    w, h = max(0.0, min(p[2], q[2]) - max(p[0], q[0])), max(0.0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    return i / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i, 1e-9)


def boxes_cached(c, frames, get):
    """YOLOX + H1c linking per frame (identical for every arm); cached with the detector's CPU ms."""
    p = Path("/dev/shm/ubx858_eval/p6_boxes") / f"{c['id'].replace('/', '__')}.npz"
    if p.exists():
        z = np.load(p); return {int(f): b for f, b in zip(z["frames"], z["boxes"])}, {int(f): float(t) for f, t in zip(z["frames"], z["det_ms"])}
    B, T, ref = {}, {}, None
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


# ------------------------------------------------------------------------------------------------ step 1: pose correction
def so3_log(R): return roma.rotmat_to_rotvec(torch.as_tensor(R, dtype=torch.float64)).numpy()
def so3_exp(v): return roma.rotvec_to_rotmat(torch.as_tensor(v, dtype=torch.float64)).numpy()


def cap(v, deg):
    n = np.linalg.norm(v, axis=-1, keepdims=True); m = np.deg2rad(deg); return np.where(n > m, v * (m / np.maximum(n, 1e-12)), v)


class PoseCorrection:
    """Per-joint offsets R_off = R_ref R_raw^T (parent frame), EMA in the tangent space at the current mean; camera bias."""
    def __init__(self, alpha, keys, cap_deg):
        self.alpha, self.keys, self.cap = alpha, keys, cap_deg; self.R = {}; self.cam = np.zeros(3); self.n = 0

    @staticmethod
    def _t(RT): return np.diag([-1.0, -1.0, 1.0]) @ np.asarray(RT, np.float64)[:, 3]

    def update(self, raw, ref):
        new = PoseCorrection(self.alpha, self.keys, self.cap); new.n = self.n + 1; al = 1.0 if self.n == 0 else self.alpha
        for k in self.keys:
            Rr = so3_exp(np.asarray(raw[k], np.float64).reshape(-1, 3)); Rf = so3_exp(np.asarray(ref[k], np.float64).reshape(-1, 3))
            off = so3_exp(cap(so3_log(Rf @ Rr.transpose(0, 2, 1)), self.cap))
            cur = self.R.get(k, np.repeat(np.eye(3)[None], len(off), 0))
            new.R[k] = cur @ so3_exp(al * so3_log(cur.transpose(0, 2, 1) @ off))
        tr, tf = self._t(raw["camera_RT_params"]), self._t(ref["camera_RT_params"])
        b = np.array([tf[0] / tf[2] - tr[0] / tr[2], tf[1] / tf[2] - tr[1] / tr[2], np.log(tf[2] / tr[2])])
        new.cam = self.cam + al * (b - self.cam); return new

    def apply(self, raw):
        out = dict(raw)
        for k, Ro in self.R.items():
            out[k] = so3_log(Ro @ so3_exp(np.asarray(raw[k], np.float64).reshape(-1, 3))).astype(np.float32).reshape(np.asarray(raw[k]).shape)
        if self.n:
            RT = np.asarray(raw["camera_RT_params"], np.float64).copy(); t = self._t(RT); z = t[2] * np.exp(self.cam[2])
            t2 = np.array([(t[0] / t[2] + self.cam[0]) * z, (t[1] / t[2] + self.cam[1]) * z, z]); RT[:, 3] = np.diag([-1.0, -1.0, 1.0]) @ t2
            out["camera_RT_params"] = RT.astype(np.float32)
        return out


def root_cam(raw):
    """camera-frame root orientation of a raw / refined SPRINT pose: R_cam(pd_cam) R_root."""
    return np.asarray(raw["camera_RT_params"], np.float64)[:3, :3] @ so3_exp(np.asarray(raw["global_pose"], np.float64).reshape(1, 3))[0]


class Worker(threading.Thread):
    def __init__(self, shp):
        super().__init__(daemon=True); self.shp = shp; self.slot = None; self.cv = threading.Condition(); self.busy = False; self.stop = False
        self.published = PoseCorrection(a.alpha, ROT_KEYS, a.cap_deg); self.version = 0; self.accepted_R = []; self.since = 0; self.log = []

    def offer(self, i, f, img, box, raw, now_frame):
        """Non-blocking: hand the frame to the worker if it is idle (budget: one in flight)."""
        with self.cv:
            if self.busy: return False
            self.since += 1; self.busy = True; self.slot = (i, f, img, box, raw, now_frame); self.cv.notify(); return True

    def run(self):
        while True:
            with self.cv:
                while self.slot is None and not self.stop: self.cv.wait()
                if self.stop: return
                i, f, img, box, raw, _ = self.slot; self.slot = None
            ent = {"i": i, "frame": f, "t_start": time.perf_counter()}
            try:
                Rc = root_cam(raw); nov = min([float(np.rad2deg(np.linalg.norm(so3_log((Rc @ R.T)[None])))) for R in self.accepted_R] or [180.0])
                ent["novelty_deg"] = nov
                if not (nov >= a.novelty_deg or self.since >= a.every_n):
                    ent["gate"] = "not_novel"; continue
                with torch.cuda.stream(WORK):
                    r = fitter.fit(img, box=box, iters=a.drive_iters, shape_lock=self.shp)
                d = r["diag"]; ent.update(kp_err=d["kp_err_px1024_after"], matte_frac=d["matte_frac"], fit_ms=r["times"]["total"] * 1000)
                if d["matte_empty"] or not np.isfinite(d["kp_err_px1024_after"]) or d["kp_err_px1024_after"] > a.kp_thr:
                    ent["gate"] = "bad_fit"; continue
                new = self.published.update(r["raw_smplx"], r["record"]["smplx_coeffs"])
                self.published = new; self.version += 1                                             # atomic reference swap
                self.accepted_R.append(Rc); self.since = 0; ent.update(gate="accepted", version=self.version)
            except Exception as e:                                                                  # noqa: BLE001
                ent["gate"] = f"error: {e}"
            finally:
                ent["t_end"] = time.perf_counter(); self.log.append(ent)
                with self.cv: self.busy = False


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
    worker = None
    if a.arm == "pose": worker = Worker(shp); worker.start()
    per = {}; t_start = time.perf_counter(); clock_lag = []
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
                sm = corr.apply(raw) if corr is not None and corr.n else raw
                if worker: worker.offer(i, f, img, box, raw, i)
            rgb, dep = render_rec(av, record_of(sm, box, M), shp)
            ev1.record(LIVE)
            img_u8 = u8(rgb).permute(1, 2, 0).cpu().numpy(); sil = (dep > 1e-4).to(torch.uint8).mul(255).cpu().numpy()   # LIVE stream sync only
        gpu_ms = ev0.elapsed_time(ev1); e2e = (time.perf_counter() - te0) * 1000
        per[f] = {"gpu_ms": gpu_ms, "e2e_ms": e2e, "det_ms": DET[f], "corr_version": corr_v}
        if f in sset:
            W = S512 @ Hm[f] @ np.linalg.inv(np.asarray(M, np.float64)) @ np.linalg.inv(S512); k = f"frame_{f:06d}"
            cv2.imwrite(str(loc / f"{k}.png"), cv2.cvtColor(cv2.warpPerspective(img_u8, W, (512, 512), flags=cv2.INTER_LINEAR, borderValue=0), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(loc / "sil" / f"{k}.png"), cv2.warpPerspective(sil, W, (512, 512), flags=cv2.INTER_NEAREST, borderValue=0))
    if worker:
        while worker.busy: time.sleep(0.05)
        with worker.cv: worker.stop = True; worker.cv.notify()
        info["worker"] = {"log": [{k: (v - t_start if k.startswith("t_") else v) for k, v in e.items()} for e in worker.log],
                          "updates": worker.version, "offered": len(worker.log),
                          "update_ms": [ (e["t_end"] - e["t_start"]) * 1000 for e in worker.log if e.get("gate") == "accepted"],
                          "replay_s": time.perf_counter() - t_start}
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


if __name__ == "__main__":
    J = json.loads(Path(a.split).read_text()); clips = [c for c in J["clips"] if a.part == "all" or c["part"] == a.part]
    if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]
    for c in clips: run_clip(c)
