"""RAGA quality gate: score replays of one recording against a baseline replay, on pose and avatar quality, plus timing.

Make every run from the same recording, with the same arguments except the change under test:
  live_sprint_guavafb.py --input REC --worker off --no-display --no-pace --record RUN --timing-json RUN/timing.json

Pose   : each run's recorded driving pose (SPRINT raw output, head / hand scale 1 as render() drives the avatar), projected into
         the frame, vs DWPose keypoints on the baseline's box (one reference for every run). Mean distance in frame pixels,
         for body, face and hands.
Avatar : each run's recorded 512 render vs the camera frame x StyleMatte alpha, both in the BASELINE's crop. A run whose crop
         differs (detector or crop changes) is warped into the baseline's crop first, so its scores stay comparable.
         PSNR / SSIM / LPIPS-VGG, the same functions as phase3_dnt/live/score_ablation.py.
Gate   : a run passes when no metric is worse than the baseline by more than its tolerance. With --repeat LABEL (a second
         replay of the baseline), each tolerance becomes max(default, 2 x |repeat - baseline|): the run-to-run noise.

usage: score_replays.py --session REC --runs base=DIR cand=DIR [repeat=DIR --repeat repeat] --out OUT [--frames auto|all|scored]
Run it with the guava env: /home/virlab/miniconda3/envs/guava/bin/python
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PKG = Path(os.environ.get("RAGA_PKG", "/data/GUAVA/guava_live")); P3 = PKG / "phase3_dnt"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", required=True, help="the recording that was replayed (frames/, session.json)")
ap.add_argument("--runs", nargs="+", required=True, help="label=replay_dir ...; the first one is the baseline")
ap.add_argument("--repeat", default="", help="label of a second replay of the baseline: tolerances follow the run-to-run noise")
ap.add_argument("--frames", default="auto", choices=("auto", "all", "scored"),
                help="auto: only frames marked 'scored' when any run used a worker (the uv update never fuses them), else all")
ap.add_argument("--tol-psnr", type=float, default=0.05, help="allowed PSNR loss, dB")
ap.add_argument("--tol-ssim", type=float, default=0.002, help="allowed SSIM loss")
ap.add_argument("--tol-lpips", type=float, default=0.002, help="allowed LPIPS increase")
ap.add_argument("--tol-pose", type=float, default=0.02, help="allowed relative increase of the mean keypoint error, per group")
ap.add_argument("--dwpose-threads", type=int, default=16,
                help="CPU threads of the DWPose reference; fewer run cooler (16 took this laptop's CPU to 100 C), results identical")
ap.add_argument("--max-cpu", type=float, default=92.0, help="pause while the CPU package is at least this hot (C); 0 = off")
ap.add_argument("--max-gpu", type=float, default=88.0, help="pause while the GPU is at least this hot (C); 0 = off")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heat_guard import HeatGuard  # noqa: E402

hg = HeatGuard(a.max_cpu, a.max_gpu)

runs = {k: Path(v).resolve() for k, v in (r.split("=", 1) for r in a.runs)}; labels = list(runs); base = labels[0]
if a.repeat and a.repeat not in runs:
    ap.error(f"--repeat {a.repeat}: no run has that label")
REC = Path(a.session).resolve(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
sess = json.loads((runs[base] / "session.json").read_text())
os.chdir(PKG)

os.environ.setdefault("SPRINT_EVAL_ASSETS", str(PKG / "live_assets" / "eval_assets"))
spec = importlib.util.spec_from_file_location("sprint_avatar_fit", P3 / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = SAF.SprintFitter(sess["sprint_ckpt"], "cuda:0", 150, dwpose_threads=a.dwpose_threads)  # DWPose, StyleMatte, SMPL-X layer

import roma   # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, str(PKG))
from main.metrics import psnr, ssim                                    # noqa: E402
from submodules.lpipsPyTorch import LPIPS                              # noqa: E402

dev = torch.device("cuda:0"); lp = LPIPS("vgg", "0.1").to(dev).eval()


def meta(d):
    m = {}
    for line in open(d / "frames.jsonl"):
        e = json.loads(line); m[int(e["i"])] = e
    return m


M = {lab: meta(d) for lab, d in runs.items()}
worker = {lab: json.loads((d / "session.json").read_text()).get("worker", "off") for lab, d in runs.items()}
scored_only = a.frames == "scored" or (a.frames == "auto" and any(w != "off" for w in worker.values()))
keys = set.intersection(*[set(m) for m in M.values()])
if scored_only:
    keys = {i for i in keys if all(M[lab][i].get("scored") for lab in labels)}
keys = sorted(i for i in keys if all((runs[lab] / "avatar" / f"{i:06d}.jpg").exists() for lab in labels))
if not keys:
    raise SystemExit("no frame is present in every run (same recording? same --score-every?)")

shape = torch.as_tensor(np.asarray(sess["shape"], np.float32).reshape(1, -1)[:, :200], device=dev)
dyn = fitter.lean.dyn.cpu().numpy(); wk = SAF.kp_weights()
GROUPS = {"body": np.arange(0, 14), "face": np.arange(41, 92), "hands": np.arange(92, 134)}
S = np.diag([0.5, 0.5, 1.0])                                            # 1024 crop px -> 512 render px
tens = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().div(255.0).contiguous().to(dev)  # noqa: E731
CORNERS = np.array([[0, 0, 1], [511, 0, 1], [0, 511, 1], [511, 511, 1]], np.float64)


def same(Mr, Mb):
    """Same crop up to GPU rounding: the run's 512 render maps onto the baseline's within 0.25 px (no resampling then)."""
    A = (S @ Mb) @ np.linalg.inv(S @ Mr)
    return bool(np.abs(CORNERS @ A.T - CORNERS)[:, :2].max() < 0.25)


def rot(v, n):
    return roma.rotvec_to_rotmat(torch.as_tensor(np.asarray(v, np.float32).reshape(n, 3), device=dev))[None]


@torch.no_grad()
def keypoints(raw, Mi):
    """Recorded pose -> 134 DWPose-order keypoints in frame px (head / hand scale 1, as render() drives the avatar)."""
    bp = {"global_pose": rot(raw["global_pose"], 1), "body_pose": rot(raw["body_pose"], 21), "jaw_pose": rot(raw["jaw_pose"], 1),
          "left_hand_pose": rot(raw["left_hand_pose"], 15), "right_hand_pose": rot(raw["right_hand_pose"], 15),
          "exp": torch.as_tensor(np.asarray(raw["exp"], np.float32).reshape(1, -1), device=dev), "shape": shape,
          "hand_scale": torch.ones(1, 3, device=dev), "head_scale": torch.ones(1, 3, device=dev), "eye_pose": None, "joints_offset": None}
    _, K = fitter.mesh(bp)
    RT = torch.as_tensor(np.asarray(raw["camera_RT_params"], np.float32).reshape(3, 4), device=dev)
    uv = fitter.project(RT[:, :3], RT[:, 3], K.reshape(-1, 3)).cpu().numpy()
    return (np.c_[uv * SAF.CROP, np.ones(len(uv))] @ np.linalg.inv(Mi).T)[:, :2]


def gt_in(cam, Mi):
    """The benchmarks' ground truth in a crop: camera frame warped into the 512 render crop x StyleMatte alpha (on black)."""
    c1024 = cv2.warpAffine(cam, Mi[:2], (SAF.CROP, SAF.CROP), flags=cv2.INTER_LINEAR)
    al = fitter.matting(torch.from_numpy(c1024).to(dev).permute(2, 0, 1)[None].float() / 255.0).clamp(0, 1)
    al = torch.nn.functional.interpolate(al.reshape(1, 1, *al.shape[-2:]), size=(512, 512), mode="bilinear", align_corners=False)
    return tens(cv2.warpAffine(cam, (S @ Mi)[:2], (512, 512), flags=cv2.INTER_LINEAR)) * al


def scores(img, g):
    r = tens(img)
    return {"psnr": float(psnr(r, g).mean()), "ssim": float(ssim(r, g)), "lpips": float(lp(r, g).mean())}


avg = lambda p, q: {k: (p[k] + q[k]) / 2 for k in p}  # noqa: E731
rows = []
with torch.no_grad():
    for n, i in enumerate(keys):
        hg.guard()
        cam = cv2.cvtColor(cv2.imread(str(REC / "frames" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB); H, W = cam.shape[:2]
        eb = M[base][i]; Mb = np.asarray(eb["M"], np.float64)
        _, kp, sc = fitter.dwpose(cam, np.asarray(eb["box"], np.float64))
        valid = (sc >= SAF.KP_THR) & ~dyn & (wk > 0) & (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H)
        g_b = gt_in(cam, Mb); gt_cache = {}                                       # runs sharing a crop share its matte
        img_b = cv2.cvtColor(cv2.imread(str(runs[base] / "avatar" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB); s_b = scores(img_b, g_b)
        row = {"i": i}
        for lab in labels:
            e = M[lab][i]; Mr = np.asarray(e["M"], np.float64); differs = not same(Mr, Mb)
            img = img_b if lab == base else cv2.cvtColor(cv2.imread(str(runs[lab] / "avatar" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
            if not differs:
                s, s_ref = (s_b if lab == base else scores(img, g_b)), s_b
            else:
                # Another crop (detector / crop change). Resampling a render blurs it, which moves PSNR / SSIM up and LPIPS down
                # by as much as the tolerances, so score BOTH runs in BOTH crops: each once as rendered, once resampled.
                key = np.round(Mr, 4).tobytes()
                if key not in gt_cache:
                    A = (S @ Mb) @ np.linalg.inv(S @ Mr)                                 # run's render px -> baseline's render px
                    g_r = gt_in(cam, Mr)
                    gt_cache[key] = (A, g_r, avg(s_b, scores(cv2.warpAffine(img_b, np.linalg.inv(A)[:2], (512, 512), flags=cv2.INTER_LINEAR), g_r)))
                A, g_r, s_ref = gt_cache[key]
                s = avg(scores(img, g_r), scores(cv2.warpAffine(img, A[:2], (512, 512), flags=cv2.INTER_LINEAR), g_b))
            row.update({f"{lab}_{k}": v for k, v in s.items()})
            row.update({f"{lab}_ref_{k}": v for k, v in s_ref.items()})                 # the baseline, scored the same way
            row[f"{lab}_crop_differs"] = int(differs)
            err = np.linalg.norm(keypoints(e.get("used", e["raw"]), Mr) - kp, axis=1)
            for gname, idx in GROUPS.items():
                sel = idx[valid[idx]]
                row[f"{lab}_{gname}_px"] = float(err[sel].mean()) if len(sel) else float("nan")
        rows.append(row)
        if n % 50 == 0:
            print(f"{n}/{len(keys)} frames", flush=True)

METRICS = ["psnr", "ssim", "lpips"] + [f"{g}_px" for g in GROUPS]
mean = {lab: {m: float(np.nanmean([r[f"{lab}_{m}"] for r in rows])) for m in METRICS} for lab in labels}
ref = {lab: {m: float(np.mean([r[f"{lab}_ref_{m}"] for r in rows])) for m in ("psnr", "ssim", "lpips")} for lab in labels}


def worse_by(lab):
    """How much worse than the baseline each metric is (positive = worse). Avatar metrics: against the baseline scored the
    same way (differs from the baseline's own score only on frames where the crop differs). Pose: fraction of the baseline's."""
    d = {"psnr": ref[lab]["psnr"] - mean[lab]["psnr"], "ssim": ref[lab]["ssim"] - mean[lab]["ssim"],
         "lpips": mean[lab]["lpips"] - ref[lab]["lpips"]}
    for g in GROUPS:
        d[f"{g}_px"] = (mean[lab][f"{g}_px"] - mean[base][f"{g}_px"]) / mean[base][f"{g}_px"]
    return d


tol = {"psnr": a.tol_psnr, "ssim": a.tol_ssim, "lpips": a.tol_lpips, **{f"{g}_px": a.tol_pose for g in GROUPS}}
noise = worse_by(a.repeat) if a.repeat else None
if noise:
    tol = {m: max(tol[m], 2 * abs(noise[m])) for m in tol}


def timing(lab):
    p = runs[lab] / "timing.json"
    if not p.exists():
        return None
    t = json.loads(p.read_text()); s = t.get("stages_ms", {})
    q = lambda k, pc: (s.get(k) or {}).get(pc)  # noqa: E731
    return {"fps": t.get("fps_observed"), "frame_p50": q("frame_interval", "p50"), "frame_p95": q("frame_interval", "p95"),
            "det_p50": q("det", "p50"), "sprint_p50": q("sprint", "p50"), "render_p50": q("render", "p50"), "worker": t.get("worker")}


f1 = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
res = {"session": str(REC), "baseline": base, "frames": len(rows), "frames_kind": "scored" if scored_only else "all",
       "mean": mean, "tolerance": tol, "repeat_noise": noise, "runs": {}}
lines = [f"Session {REC}: {len(rows)} {'scored' if scored_only else ''} frames in every run; baseline = {base}", "",
         "| run | PSNR | SSIM | LPIPS | body px | face px | hands px | replay FPS | frame p50 / p95 ms | det / SPRINT / render p50 ms | gate |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for lab in labels:
    mn, t = mean[lab], timing(lab)
    if lab == base:
        verdict, d = "baseline", None
    else:
        d = worse_by(lab); failed = [m for m in METRICS if d[m] > tol[m]]
        verdict = ("noise repeat" if lab == a.repeat else "pass") if not failed else "FAIL: " + ", ".join(failed)
    res["runs"][lab] = {"worse_by": d, "verdict": verdict, "timing": t, "crop_differs_frames": int(sum(r[f"{lab}_crop_differs"] for r in rows))}
    dd = lambda m, fmt: "" if d is None else f" ({-d[m] if m in ('psnr', 'ssim') else d[m]:+{fmt}})"  # noqa: E731
    pose = lambda g: f"{mn[g + '_px']:.1f}" + ("" if d is None else f" ({d[g + '_px'] * 100:+.1f}%)")  # noqa: E731
    lines.append(f"| {lab} | {mn['psnr']:.2f}{dd('psnr', '.2f')} | {mn['ssim']:.4f}{dd('ssim', '.4f')} | {mn['lpips']:.4f}{dd('lpips', '.4f')} | "
                 f"{pose('body')} | {pose('face')} | {pose('hands')} | {f1(t and t['fps'], '.1f')} | "
                 f"{f1(t and t['frame_p50'], '.0f')} / {f1(t and t['frame_p95'], '.0f')} | "
                 f"{f1(t and t['det_p50'], '.0f')} / {f1(t and t['sprint_p50'], '.0f')} / {f1(t and t['render_p50'], '.0f')} | {verdict} |")
lines += ["", "Tolerances (worse than baseline by at most): " + ", ".join(
    f"{m} {tol[m] * 100:.1f}%" if m.endswith("_px") else f"{m} {tol[m]:.4f}" for m in METRICS)
    + (" (from the repeat run's noise where larger)" if noise else " (defaults; pass --repeat to use the run-to-run noise)")]
for lab in labels:
    c = res["runs"][lab]["crop_differs_frames"]
    if c:
        lines.append(f"{lab}: crop differs from the baseline's on {c} of {len(rows)} frames; there both runs are scored in both crops "
                     f"(as rendered + resampled), so its PSNR / SSIM / LPIPS change is against the baseline scored the same way")
lines.append("Replay FPS = throughput with --no-pace (JPEG decoding in the loop; no camera, no display). Confirm 27-30 FPS live.")
(out / "gate.json").write_text(json.dumps(res, indent=1))
with open(out / "per_frame.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines.append(hg.summary())
(out / "gate.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
