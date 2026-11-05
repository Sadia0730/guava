"""PREREG phase 6 background worker, shared by the replay harness (replay.py) and the laptop live app (live_sprint_guavafb.py).

Worker (separate thread, low-priority CUDA stream, at most one update in flight): offered the live frame whenever idle ->
gate (camera-frame root novelty >= novelty_deg | every_n-th offered frame since the last acceptance) -> k = 60 refinement
(SprintFitter.fit, empty-matte fix on) -> fit-quality gate (keypoint error <= kp_thr px at 1024, matte non-empty) ->
step 1: correction = refined - raw (per-joint offsets in the parent frame + camera bias), EMA, published by a reference swap
(the live loop reads `worker.published` once per frame; no locks on the render path).
Mode 'uv' (step 2, on k = 0 + H1c): gate also passes on new UV coverage >= 0.005 (raw pose); step 2 = uv_fusion.UVFusion.process;
the new avatar is published by a reference swap (`worker.published_av`, read once per frame by the live loop)."""
from __future__ import annotations

import threading
import time

import numpy as np
import roma
import torch

ROT_KEYS = {"all": ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose"), "body": ("global_pose", "body_pose"), "cam": ()}


def so3_log(R): return roma.rotmat_to_rotvec(torch.as_tensor(R, dtype=torch.float64)).numpy()
def so3_exp(v): return roma.rotvec_to_rotmat(torch.as_tensor(v, dtype=torch.float64)).numpy()


def cap(v, deg):
    n = np.linalg.norm(v, axis=-1, keepdims=True); m = np.deg2rad(deg); return np.where(n > m, v * (m / np.maximum(n, 1e-12)), v)


class PoseCorrection:
    """Per-joint offsets R_off = R_ref R_raw^T (parent frame), EMA in the tangent space at the current mean; camera bias
    (normalised x / y shift and log depth ratio of the crop-camera translation)."""
    def __init__(self, alpha=0.5, keys=ROT_KEYS["all"], cap_deg=30.0):
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
        if not self.n: return raw
        out = dict(raw)
        for k, Ro in self.R.items():
            out[k] = so3_log(Ro @ so3_exp(np.asarray(raw[k], np.float64).reshape(-1, 3))).astype(np.float32).reshape(np.asarray(raw[k]).shape)
        RT = np.asarray(raw["camera_RT_params"], np.float64).copy(); t = self._t(RT); z = t[2] * np.exp(self.cam[2])
        t2 = np.array([(t[0] / t[2] + self.cam[0]) * z, (t[1] / t[2] + self.cam[1]) * z, z]); RT[:, 3] = np.diag([-1.0, -1.0, 1.0]) @ t2
        out["camera_RT_params"] = RT.astype(np.float32)
        return out


def root_cam(raw):
    """Camera-frame root orientation of a SPRINT pose: R_cam(pd_cam) R_root."""
    return np.asarray(raw["camera_RT_params"], np.float64)[:3, :3] @ so3_exp(np.asarray(raw["global_pose"], np.float64).reshape(1, 3))[0]


class Worker(threading.Thread):
    """mode: 'pose' (step 1) | 'uv' (step 2 alone on the k = 0 + H1c live loop: C3 appearance update, no pose correction; step 1
    failed its verdict). fitter: a SprintFitter (sprint_avatar_fit) with empty_matte_fix = True. 'uv' needs fusion (a
    uv_fusion.UVFusion of the session), the static avatar (published until the first update), the source's raw SPRINT pose (its
    root orientation counts as accepted for the novelty gate), the scored frame indices (never fused) and the held-out ones."""
    def __init__(self, fitter, shp, stream, mode="pose", alpha=0.5, corr="all", cap_deg=30.0, novelty_deg=15.0, every_n=10,
                 kp_thr=20.0, iters=60, fusion=None, static_av=None, src_raw=None, scored=(), heldout=(), heldout_gap=30,
                 new_cov=0.005):
        super().__init__(daemon=True)
        if mode not in ("pose", "uv"): raise NotImplementedError(f"worker mode {mode!r}: only 'pose' (step 1) and 'uv' (step 2)")
        self.mode = mode
        self.fitter, self.shp, self.stream = fitter, shp, stream
        self.novelty_deg, self.every_n, self.kp_thr, self.iters = novelty_deg, every_n, kp_thr, iters
        self.slot = None; self.cv = threading.Condition(); self.busy = False; self.stop_flag = False
        self.published = PoseCorrection(alpha, ROT_KEYS[corr], cap_deg); self.version = 0       # identity in 'uv' mode
        self.accepted_R = []; self.since = 0; self.log = []
        self.fusion, self.published_av, self.av_version = fusion, (static_av, 0), 0                 # (avatar, version): one atomic read
        self.scored, self.heldout, self.heldout_gap, self.new_cov = set(scored), set(heldout), heldout_gap, new_cov
        self.last_heldout = -10 ** 9
        if mode == "uv": self.accepted_R.append(root_cam(src_raw))

    def offer(self, i, img, box, raw):
        """Non-blocking: hand the frame to the worker if it is idle (budget: one update in flight)."""
        with self.cv:
            if self.busy: return False
            if self.mode == "pose": self.since += 1
            self.busy = True; self.slot = (i, img, box, raw); self.cv.notify(); return True

    def stop(self):
        with self.cv: self.stop_flag = True; self.cv.notify()

    def wait_idle(self, timeout=30.0):
        t0 = time.perf_counter()
        while self.busy and time.perf_counter() - t0 < timeout: time.sleep(0.02)

    def run(self):
        while True:
            with self.cv:
                while self.slot is None and not self.stop_flag: self.cv.wait()
                if self.stop_flag: return
                i, img, box, raw = self.slot; self.slot = None
            ent = {"i": i, "t_start": time.perf_counter()}
            if self.mode == "uv":
                try: self._run_uv(ent, i, img, box, raw)
                except Exception as e:                                                                 # noqa: BLE001
                    import traceback; ent["gate"] = f"error: {type(e).__name__}: {e}"; ent["trace"] = traceback.format_exc()[-1500:]
                finally:
                    ent["t_end"] = time.perf_counter(); self.log.append(ent)
                    with self.cv: self.busy = False
                continue
            try:
                Rc = root_cam(raw)
                nov = min([float(np.rad2deg(np.linalg.norm(so3_log((Rc @ R.T)[None])))) for R in self.accepted_R] or [180.0])
                ent["novelty_deg"] = nov
                if not (nov >= self.novelty_deg or self.since >= self.every_n):
                    ent["gate"] = "not_novel"; continue
                with torch.cuda.stream(self.stream):
                    r = self.fitter.fit(img, box=box, iters=self.iters, shape_lock=self.shp)
                d = r["diag"]; ent.update(kp_err=d["kp_err_px1024_after"], matte_frac=d["matte_frac"], fit_ms=r["times"]["total"] * 1000)
                if d["matte_empty"] or not np.isfinite(d["kp_err_px1024_after"]) or d["kp_err_px1024_after"] > self.kp_thr:
                    ent["gate"] = "bad_fit"; continue
                self.published = self.published.update(r["raw_smplx"], r["record"]["smplx_coeffs"])      # atomic reference swap
                self.version += 1; self.accepted_R.append(Rc); self.since = 0; ent.update(gate="accepted", version=self.version)
            except Exception as e:                                                                     # noqa: BLE001
                ent["gate"] = f"error: {type(e).__name__}: {e}"
            finally:
                ent["t_end"] = time.perf_counter(); self.log.append(ent)
                with self.cv: self.busy = False

    def _good(self, r, ent):
        d = r["diag"]; ent.update(kp_err=d["kp_err_px1024_after"], matte_frac=d["matte_frac"], fit_ms=r["times"]["total"] * 1000)
        return not (d["matte_empty"] or not np.isfinite(d["kp_err_px1024_after"]) or d["kp_err_px1024_after"] > self.kp_thr)

    def _run_uv(self, ent, i, img, box, raw):
        """step 2: scored frames are never touched; held-out frames (refined, never fused) refresh the stop rule's reservoir;
        the other frames go through the registered gate -> k = 60 refinement -> fit gate -> UVFusion.process -> publish."""
        fu = self.fusion
        if i in self.scored: ent["gate"] = "scored"; return
        if i in self.heldout:
            if len(fu.res) >= fu.K and i - self.last_heldout < self.heldout_gap: ent["gate"] = "heldout_skip"; return
            with torch.cuda.stream(self.stream):
                r = self.fitter.fit(img, box=box, iters=self.iters, shape_lock=self.shp)
            if not self._good(r, ent): ent["gate"] = "heldout_bad_fit"; return
            fu.add_heldout(i, r); self.last_heldout = i; ent["gate"] = "heldout_added"; return
        self.since += 1
        Rc = root_cam(raw)
        nov = min([float(np.rad2deg(np.linalg.norm(so3_log((Rc @ R.T)[None])))) for R in self.accepted_R] or [180.0]); ent["novelty_deg"] = nov
        if fu.all_stopped():                                                                     # only the luminance reopen check
            if not (nov >= self.novelty_deg or self.since >= self.every_n): ent["gate"] = "stopped"; return
            import cv2, uv_fusion as UF
            M = fu.saf.box_crop_matrix(box, fu.saf.CROP)
            crop = cv2.warpAffine(img, M[:2], (fu.saf.CROP, fu.saf.CROP), flags=cv2.INTER_LINEAR)
            with torch.cuda.stream(self.stream):
                ct = torch.from_numpy(crop).to(self.fitter.dev).permute(2, 0, 1)[None].float() / 255.0
                al = self.fitter.matting(ct).clamp(0, 1).cpu().numpy()
            lum = UF.luminance(crop, al); ent.update(gate="stopped_check", lum=lum, reopened=fu.check_reopen(lum)); self.since = 0; return
        if not (nov >= self.novelty_deg or self.since >= self.every_n):
            with torch.cuda.stream(self.stream):
                cov = fu.coverage_gain(raw)
            self.stream.synchronize(); ent["cov_gain"] = cov
            if cov < self.new_cov: ent["gate"] = "not_novel"; return
        with torch.cuda.stream(self.stream):
            r = self.fitter.fit(img, box=box, iters=self.iters, shape_lock=self.shp)
        if not self._good(r, ent): ent["gate"] = "bad_fit"; return
        e2, live = fu.process(i, img, r, self.stream); ent.update(e2)
        if live is not None:
            self.av_version += 1; self.published_av = (live, self.av_version)                       # atomic reference swap
            self.accepted_R.append(Rc); self.since = 0
