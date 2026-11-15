"""Far-view subset scores (PREREG "Far-view subset: SSIM, LPIPS and the other methods", 2026-11-15; descriptive).
Frames = teaser_candidates.py's rule: test frames whose global-root rotation differs from the source frame's by >= 135 deg, on
the roots finalbench_ext / publicbench_v1 / publicbench_v2; a frame is used only if every method's render exists. Metrics =
score.py's functions (PSNR, SSIM 11 px Gaussian, LPIPS-VGG 0.1) on the stored 8-bit 512 renders vs gt512. Mean over a clip's
qualifying frames, then over clips (per dataset, all). Paired fb_update100 - fb_gt with 95 % percentile bootstrap intervals
(10,000 draws, seed 20261109, people resampled within datasets as ext_tables.py). Reads only; writes --out/far_view_scores.{json,md}.
Usage: CUDA_VISIBLE_DEVICES=4 python far_view_scores.py --out O/reports/far_view"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchvision
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
ROOTS = [("finalbench_ext", "finalbench_ext.json"), ("publicbench_v1", "publicbench_v1.json"), ("publicbench_v2", "publicbench_v2_xhumans.json")]
METHODS = ["guava", "lhm_facefix", "lhmpp", "idol", "fb_gt", "fb_update25", "fb_update50", "fb_update100", "gart", "gaussianavatar", "exavatar_gtface", "ft_gt"]
NAME = {"guava": "GUAVA", "lhm_facefix": "LHM", "lhmpp": "LHM++", "idol": "IDOL", "fb_gt": "RAGA, photo", "fb_update25": "RAGA, update 25 %",
        "fb_update50": "RAGA, update 50 %", "fb_update100": "RAGA, update 100 %", "gart": "GART", "gaussianavatar": "GaussianAvatar",
        "exavatar_gtface": "ExAvatar", "ft_gt": "RAGA, fine tuned"}
DSN = {"ubc": "UBC-Fashion", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans"}
MET = ("psnr", "ssim", "lpips")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=135.0)
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261109); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for f in ("far_view_scores.json", "far_view_scores.md"): assert not (out / f).exists(), f"{out / f} exists (never overwrite)"
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from main.metrics import psnr, ssim                                                                  # noqa: E402
from submodules.lpipsPyTorch import LPIPS                                                            # noqa: E402
dev = torch.device("cuda:0"); lp = LPIPS("vgg", "0.1").to(dev).eval()

# ---- clips, people, qualifying frames
clips = []
for E, sj in ROOTS:
    for c in json.loads((HERE / sj).read_text())["clips"]:
        ds, name = c["dataset"], c["clip"]; cid = f"{ds}/{name}"
        if ds == "ubody" or not (O / E / "renders" / "fb_update100" / cid / "run.json").exists(): continue
        person = (name.rsplit("-", 1)[0] if ds == "peoplesnapshot" else c.get("person") or c["id"]) if ds != "ubc" else f"ubc:{c['person']}"
        b = O / E / "base" / cid; fj = json.loads((b / "frames.json").read_text()); s = np.load(b / "smplx2020.npz")
        idx = {str(k): i for i, k in enumerate(s["keys"])}; R = Rot.from_rotvec(s["global_orient"]).as_matrix(); Rs = R[idx[fj["source"]]]
        back = [k for k in fj["test"] if k in idx and np.degrees(np.linalg.norm(Rot.from_matrix(R[idx[k]] @ Rs.T).as_rotvec())) >= a.thr]
        keys = [k for k in back if (O / E / "gt512" / cid / f"{k}.png").exists()
                and all((O / E / "renders" / m / cid / f"{k}.png").exists() for m in METHODS)]
        if keys: clips.append({"id": cid, "E": E, "ds": ds, "person": person, "keys": keys, "n_back": len(back)})
print("clips", len(clips), "frames", sum(len(c["keys"]) for c in clips), "dropped for a missing render", sum(c["n_back"] - len(c["keys"]) for c in clips), flush=True)


def rd(p): return torchvision.io.read_image(str(p))[:3]


def score_clip(c):
    res = {}
    with ThreadPoolExecutor(8) as ex:
        G = torch.stack(list(ex.map(lambda k: rd(O / c["E"] / "gt512" / c["id"] / f"{k}.png"), c["keys"]))).float().div(255).to(dev)
        for m in METHODS:
            Rr = torch.stack(list(ex.map(lambda k: rd(O / c["E"] / "renders" / m / c["id"] / f"{k}.png"), c["keys"]))).float().div(255).to(dev)
            v = []
            with torch.no_grad():
                for i in range(len(G)):
                    r, g = Rr[i:i + 1].contiguous(), G[i:i + 1].contiguous()
                    v.append([float(psnr(r, g).mean()), float(ssim(r, g)), float(lp(r, g).mean())])
            res[m] = np.asarray(v).mean(0).tolist()
    return res


for n, c in enumerate(clips):
    c["score"] = score_clip(c)
    if n % 20 == 0: print(n, c["id"], len(c["keys"]), {m: round(c["score"][m][0], 2) for m in ("fb_gt", "fb_update100")}, flush=True)

# ---- aggregates and paired bootstrap
S = {m: np.array([c["score"][m] for c in clips]) for m in METHODS}                                   # (clips, 3)
dss = [d for d in DSN if any(c["ds"] == d for c in clips)]
sel = {d: np.array([i for i, c in enumerate(clips) if c["ds"] == d]) for d in dss}; sel["all"] = np.arange(len(clips))
agg = {m: {d: {"clips": int(len(ix)), **{k: float(S[m][ix, j].mean()) for j, k in enumerate(MET)}} for d, ix in sel.items()} for m in METHODS}
rng = np.random.default_rng(a.seed); W = np.zeros((a.boot, len(clips)))
for d in dss:
    people = sorted({clips[i]["person"] for i in sel[d]}); pc = {p: [i for i in sel[d] if clips[i]["person"] == p] for p in people}
    draws = rng.integers(0, len(people), size=(a.boot, len(people)))
    for b in range(a.boot):
        for j in draws[b]:
            W[b, pc[people[j]]] += 1
D = S["fb_update100"] - S["fb_gt"]; paired = {}
for d, ix in sel.items():
    w = W[:, ix]; paired[d] = {}
    for j, k in enumerate(MET):
        bs = (w * D[ix, j]).sum(1) / w.sum(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        better = int(((D[ix, j] > 0) if k != "lpips" else (D[ix, j] < 0)).sum())
        paired[d][k] = {"mean": float(D[ix, j].mean()), "lo": float(lo), "hi": float(hi), "clips_better": better, "clips": int(len(ix))}
res = {"rule": f"test frames turned >= {a.thr} deg from the source; every method's render exists", "clips": len(clips),
       "frames": int(sum(len(c["keys"]) for c in clips)), "frames_by_dataset": {d: int(sum(len(clips[i]["keys"]) for i in sel[d])) for d in dss},
       "aggregate": agg, "paired_update100_minus_photo": paired, "boot": a.boot, "seed": a.seed,
       "per_clip": {c["id"]: {"n": len(c["keys"]), **{m: dict(zip(MET, c["score"][m])) for m in METHODS}} for c in clips}}
(out / "far_view_scores.json").write_text(json.dumps(res, indent=1) + "\n")
f = lambda k, x: f"{x:.2f}" if k == "psnr" else f"{x:.3f}"                                              # noqa: E731
L = [f"# Far-view subset ({res['frames']} test frames turned >= {a.thr:.0f} deg from the source, {len(clips)} clips)", "",
     "Mean over a clip's qualifying frames, then over clips. PSNR / SSIM / LPIPS.", "",
     "| Method | " + " | ".join(f"{DSN[d]} ({len(sel[d])})" for d in dss) + f" | All ({len(clips)}) |", "|---|" + "---|" * (len(dss) + 1)]
for m in METHODS:
    L.append(f"| {NAME[m]} | " + " | ".join(" / ".join(f(k, agg[m][d][k]) for k in MET) for d in dss + ["all"]) + " |")
L += ["", "Paired RAGA update 100 % - RAGA photo, mean [95 % bootstrap interval], clips better:", ""]
for d in dss + ["all"]:
    L.append(f"- {DSN.get(d, 'All')}: " + "; ".join(f"{k.upper()} {paired[d][k]['mean']:+.3f} [{paired[d][k]['lo']:+.3f}, {paired[d][k]['hi']:+.3f}], "
                                                    f"{paired[d][k]['clips_better']}/{paired[d][k]['clips']}" for k in MET))
(out / "far_view_scores.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
