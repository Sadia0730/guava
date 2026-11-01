"""H1 timing (PREREG "GUAVA-FB improvement phase 2"): SPRINT driving cost per frame for k refinement iterations, GPU-synchronised,
on an idle GPU. Same 200 original frames (ValBench clips, deterministic order) for every k, 10 warm-up fits first. Reports the median
of sprint_avatar_fit's own stage timers (dwpose_cpu, matting, sprint, refine, total) and peak GPU memory per k. -> <out json>
Usage: CUDA_VISIBLE_DEVICES=5 python time_sprint.py --out <json> [--ks 0,20,60 --n 200]"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--ks", default="0,20,60"); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--split", default=str(HERE / "valbench_v1.json")); ap.add_argument("--sprint-ckpt", default="/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / "phase3_dnt/sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF); fitter = SAF.SprintFitter(a.sprint_ckpt, "cuda:0", 150)
import torch                                                                                         # noqa: E402
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import convert_bedlam2 as cb                                                                         # noqa: E402
clips = [c for c in json.loads(Path(a.split).read_text())["clips"] if c["dataset"] == "bedlam2"][:5]
frames = []
for c in clips:                                                                                      # 40 test frames of each of 5 BEDLAM2 clips
    got = cb.decode_frames(c["job"], c["seq"], c["test_frames"][:40]); frames += [got[f] for f in c["test_frames"][:40] if f in got]
frames = frames[:a.n]
src = fitter.fit(frames[0]); shp = src["shape"]
for f in frames[:10]: fitter.fit(f, iters=0, shape_lock=shp); fitter.fit(f, iters=20, shape_lock=shp)
out = {"gpu": torch.cuda.get_device_name(0), "n_frames": len(frames), "per_k": {}}
for k in [int(x) for x in a.ks.split(",")]:
    T = []; torch.cuda.reset_peak_memory_stats()
    for f in frames:
        r = fitter.fit(f, iters=k, shape_lock=shp); T.append(r["times"])
    out["per_k"][k] = {s: float(np.median([t[s] for t in T if s in t]) * 1000) for s in T[0]}
    out["per_k"][k]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    print(k, json.dumps(out["per_k"][k]), flush=True)
Path(a.out).write_text(json.dumps(out, indent=1))
