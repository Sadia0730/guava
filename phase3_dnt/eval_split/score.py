"""AvatarTest-v1 scoring (PREREG "AvatarTest-v1 addendum 1"; one script for every row). Renders E/renders/<method>/<dataset>/<clip>/<key>.png
(8-bit, 512) vs E/gt512/<dataset>/<clip>/<key>.png (image x StyleMatte alpha on black), test keys of base/.../frames.json.
Metrics = GUAVA main/metrics.py functions: PSNR, L1 (MAE), SSIM (11 px Gaussian), LPIPS-VGG (lpipsPyTorch 0.1); mean per clip,
frame-weighted over clips (= mean over frames), per dataset and overall. A clip counts only if every test render exists.
Usage: python score.py --methods fb_gt,guava,fb_sprint [--subset eval_split/avatar_test_v1_subset24.json] [--tag full]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torchvision

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--methods", required=True); ap.add_argument("--subset", default="")
ap.add_argument("--tag", default="full"); ap.add_argument("--split", default=str(HERE / "avatar_test_v1.json"))
ap.add_argument("--E", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1"); a = ap.parse_args(); a.split = str(Path(a.split).resolve())
E = Path(a.E)
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from main.metrics import psnr, ssim, l1_loss                                                         # noqa: E402
from submodules.lpipsPyTorch import LPIPS                                                            # noqa: E402
dev = torch.device("cuda:0"); lp = LPIPS("vgg", "0.1").to(dev).eval()
split = json.loads(Path(a.split).read_text())["clips"]
if a.subset: keep = set(json.loads(Path(a.subset).read_text())["clips"]); split = [c for c in split if c["id"] in keep]
load = lambda p: (torchvision.io.read_image(str(p))[:3].float() / 255.0)[None].contiguous().to(dev)              # noqa: E731
(E / "scores").mkdir(parents=True, exist_ok=True)
for m in a.methods.split(","):
    per, missing = {}, {}
    for c in split:
        ds, name = c["dataset"], c["clip"]; fj = json.loads((E / "base" / ds / name / "frames.json").read_text())
        rd = E / "renders" / m / ds / name; miss = [k for k in fj["test"] if not (rd / f"{k}.png").exists()]
        if miss: missing[c["id"]] = len(miss); continue
        v = []
        with torch.no_grad():
            for k in fj["test"]:
                r, g = load(rd / f"{k}.png"), load(E / "gt512" / ds / name / f"{k}.png")
                v.append([float(psnr(r, g).mean()), float(l1_loss(r, g)), float(ssim(r, g)), float(lp(r, g).mean())])
        v = np.asarray(v); per[c["id"]] = {"dataset": ds, "n": len(v), **dict(zip(("psnr", "l1", "ssim", "lpips"), v.mean(0).tolist()))}
        rj = rd / "run.json"
        if rj.exists(): per[c["id"]]["run"] = json.loads(rj.read_text())
    agg = {}
    for part in ("ubody", "bedlam2", "peoplesnapshot", "neuman", "xhumans", "dress4d", "all"):
        cs = [p for p in per.values() if part == "all" or p["dataset"] == part]
        if not cs: continue
        n = np.array([p["n"] for p in cs], float)
        agg[part] = {"clips": len(cs), "frames": int(n.sum()), **{k: float((np.array([p[k] for p in cs]) * n).sum() / n.sum()) for k in ("psnr", "l1", "ssim", "lpips")}}
    out = {"method": m, "tag": a.tag, "subset": a.subset or None, "aggregate": agg, "missing_clips": missing, "per_clip": per}
    (E / "scores" / f"{m}_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(m, a.tag, json.dumps(agg), "missing:", len(missing), flush=True)
