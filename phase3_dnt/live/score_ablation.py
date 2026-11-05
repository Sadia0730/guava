"""Ablation of the live appearance update on one recorded session: the same recording replayed with the update off and on
(run_ablation.sh: live_sprint_guavafb.py --input REC --worker off|uv --record OUT/<arm>), scored on the frames marked "scored"
(every --score-every-th processed frame; the update never fuses them).

GT = the camera frame warped into the 512 render crop of that frame x StyleMatte alpha on black (the benchmarks' GT); prediction =
the recorded 512 render (black background). Metrics = score.py's: main/metrics psnr / ssim and LPIPS-VGG (lpipsPyTorch 0.1) on the
full 512 crop, mean over frames. "Turned away" = SPRINT's camera-frame root orientation >= --away-deg from the reference frame's.
Both arms use SPRINT k = 0 driving, so pose error is in both; the difference is the update.

Usage: python phase3_dnt/live/score_ablation.py --session REC --runs off=OUT/off uv=OUT/uv --out OUT [--video]"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; P3 = ROOT / "phase3_dnt"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", required=True, help="the recording (reference.png, session.json, frames/)")
ap.add_argument("--runs", nargs="+", required=True, help="label=replay_dir, e.g. off=OUT/off uv=OUT/uv (first = baseline)")
ap.add_argument("--out", required=True); ap.add_argument("--away-deg", type=float, default=90.0)
ap.add_argument("--video", action="store_true", help="side-by-side mp4 of the scored frames: camera crop x matte | runs")
ap.add_argument("--assets", default=str(ROOT / "live_assets")); ap.add_argument("--sprint-ckpt", default="")
a = ap.parse_args()
runs = dict(r.split("=", 1) for r in a.runs); labels = list(runs)
sess = json.loads((Path(runs[labels[0]]) / "session.json").read_text())                              # written by the replay itself
ASSETS = Path(a.assets)
if (ASSETS / "eval_assets").exists(): os.environ.setdefault("SPRINT_EVAL_ASSETS", str(ASSETS / "eval_assets"))
spec = importlib.util.spec_from_file_location("sprint_avatar_fit", P3 / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
fitter = SAF.SprintFitter(a.sprint_ckpt or sess.get("sprint_ckpt") or str(ASSETS / "checkpoints" / "sprint_scratch_ema128k.pt"), "cuda:0", 150)  # StyleMatte

import cv2, torch                                                                                      # noqa: E402
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(P3 / "worker6"))
from main.metrics import psnr, ssim                                                                   # noqa: E402
from submodules.lpipsPyTorch import LPIPS                                                            # noqa: E402
import worker_core as WC                                                                              # noqa: E402

dev = torch.device("cuda:0"); lp = LPIPS("vgg", "0.1").to(dev).eval()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
R0 = WC.root_cam({k: np.asarray(v) for k, v in sess["source_raw"].items()})


def meta(d):
    m = {}
    for line in open(Path(d) / "frames.jsonl"): e = json.loads(line); m[int(e["i"])] = e
    return m


M = {l: meta(d) for l, d in runs.items()}
keys = sorted(set.intersection(*[{i for i, e in m.items() if e.get("scored")} for m in M.values()]))
keys = [i for i in keys if all((Path(runs[l]) / "avatar" / f"{i:06d}.jpg").exists() for l in labels)]
if not keys: raise SystemExit("no frame is scored in every run (record with the same --score-every)")
tens = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().div(255.0).contiguous().to(dev)  # noqa: E731
S = np.diag([0.5, 0.5, 1.0])                                                                          # 1024 crop px -> 512 render px
rows, vw = [], None
with torch.no_grad():
    for i in keys:
        cam = cv2.cvtColor(cv2.imread(str(Path(a.session) / "frames" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)    # the replayed input
        e0 = M[labels[0]][i]; ang = float(np.degrees(np.linalg.norm(cv2.Rodrigues(
            WC.root_cam({k: np.asarray(v) for k, v in e0["raw"].items()}) @ R0.T)[0])))
        row = {"i": i, "view_deg": ang}; tiles = []
        for l in labels:
            Mi = np.asarray(M[l][i]["M"], np.float64)                                                 # frame px -> 1024 crop px
            c1024 = cv2.warpAffine(cam, Mi[:2], (SAF.CROP, SAF.CROP), flags=cv2.INTER_LINEAR)
            al = fitter.matting(torch.from_numpy(c1024).to(dev).permute(2, 0, 1)[None].float() / 255.0).clamp(0, 1)
            al = torch.nn.functional.interpolate(al.reshape(1, 1, *al.shape[-2:]), size=(512, 512), mode="bilinear", align_corners=False)
            g = tens(cv2.warpAffine(cam, (S @ Mi)[:2], (512, 512), flags=cv2.INTER_LINEAR)) * al
            r = tens(cv2.cvtColor(cv2.imread(str(Path(runs[l]) / "avatar" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB))
            row.update({f"{l}_psnr": float(psnr(r, g).mean()), f"{l}_ssim": float(ssim(r, g)), f"{l}_lpips": float(lp(r, g).mean()),
                        f"{l}_version": M[l][i].get("av_version", 0)})
            if a.video:
                if not tiles: tiles.append((g[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                t = cv2.cvtColor(cv2.imread(str(Path(runs[l]) / "avatar" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
                cv2.putText(t, f"{l} v{row[f'{l}_version']} {row[f'{l}_psnr']:.2f} dB", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                tiles.append(t)
        rows.append(row)
        if a.video:
            fr = np.concatenate(tiles, 1); cv2.putText(fr, f"frame {i}, view change {ang:.0f} deg", (8, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if vw is None: vw = cv2.VideoWriter(str(out / "side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (fr.shape[1], fr.shape[0]))
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
if vw is not None: vw.release()


def summary(sel):
    s = {"frames": len(sel)}
    for l in labels:
        for m in ("psnr", "ssim", "lpips"): s[f"{l}_{m}"] = float(np.mean([r[f"{l}_{m}"] for r in sel])) if sel else None
    return s


away = [r for r in rows if r["view_deg"] >= a.away_deg]
res = {"session": str(a.session), "runs": runs, "scored_frames": len(rows), "away_deg": a.away_deg,
       "all": summary(rows), "turned_away": summary(away), "front": summary([r for r in rows if r["view_deg"] < 30.0])}
for l in labels:                                                                                      # live cost and update count
    tj = Path(runs[l]) / "timing.json"
    if tj.exists():
        t = json.loads(tj.read_text()); res[f"{l}_timing"] = {"total_ms_p50": t["stages_ms"]["total"]["p50"], "total_ms_p95": t["stages_ms"]["total"]["p95"],
                                                              "fps": t.get("fps_observed"), "updates": t.get("worker_updates"), "gpu": t.get("gpu")}
(out / "ablation.json").write_text(json.dumps(res, indent=1))
with open(out / "per_frame.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
b = labels[0]
lines = [f"Session {a.session}: {len(rows)} scored frames ({len(away)} turned away >= {a.away_deg:.0f} deg)", "",
         "| frames | run | PSNR | SSIM | LPIPS |", "|---|---|---|---|---|"]
for name, key in (("all scored", "all"), ("front (< 30 deg)", "front"), (f"turned away", "turned_away")):
    s = res[key]
    for l in labels:
        if s[f"{l}_psnr"] is None: continue
        d = "" if l == b else f" ({s[f'{l}_psnr'] - s[f'{b}_psnr']:+.2f} dB, LPIPS {s[f'{l}_lpips'] - s[f'{b}_lpips']:+.4f})"
        lines.append(f"| {name} ({s['frames']}) | {l} | {s[f'{l}_psnr']:.2f}{d} | {s[f'{l}_ssim']:.4f} | {s[f'{l}_lpips']:.4f} |")
for l in labels:
    if f"{l}_timing" in res:
        t = res[f"{l}_timing"]; lines.append(f"{l}: live total {t['total_ms_p50']:.1f} / {t['total_ms_p95']:.1f} ms (p50 / p95), {t['updates']} updates")
(out / "ablation.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
