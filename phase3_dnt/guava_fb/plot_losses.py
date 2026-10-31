"""Training-loss curves of GUAVA-FB runs: total loss and every logged loss term (one panel each), plus the registered
validation measures (full-body SSIM, UBody SSIM, UBody face LPIPS with the best.pt thresholds). Reads train_log.jsonl only
(no GPU). Logged values are single iterations every --log-every steps: drawn faint, with a rolling mean over --smooth points.
A run may chain several run directories (e.g. a pilot and its resumed continuation): label=dir1,dir2 (later dirs win per iter).
Usage: python plot_losses.py --runs B=<runs/guava_fb_pilotB> C=<...> --out <dir>/losses   (writes <out>_train.png, <out>_val.png)"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--smooth", type=int, default=10); ap.add_argument("--title", default="GUAVA-FB training losses")
a = ap.parse_args()
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # categorical slots, fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
THR = {"ubody_ssim": 0.8193, "face_lpips": 0.1873}                                                  # registered best.pt thresholds


def load(dirs):
    tr, va = {}, {}
    for d in dirs.split(","):
        f = Path(d) / "train_log.jsonl"
        if not f.exists(): continue
        for line in open(f):
            try: j = json.loads(line)
            except json.JSONDecodeError: continue                                                  # a line being written
            (tr if j.get("split") == "train" else va if j.get("split") == "val" else {})[j.get("iter")] = j
    return [tr[k] for k in sorted(tr)], [va[k] for k in sorted(va)]


runs = {}
for spec in a.runs:
    label, dirs = spec.split("=", 1); runs[label] = load(dirs)
terms = ["loss"] + sorted({k for tr, _ in runs.values() for j in tr for k, v in j.items() if k.endswith("_loss") and isinstance(v, (int, float))})
NAMES = {"loss": "total loss (sum of the run's own terms)", "gan_d_loss": "gan_d_loss (discriminator; not in total)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})


def roll(y, n):
    y = np.asarray(y, float); out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - n + 1):i + 1]; w = w[np.isfinite(w)]; out[i] = w.mean() if len(w) else np.nan
    return out


nc = 4; nr = int(np.ceil(len(terms) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 2.8 * nr + 0.6), squeeze=False)
for ax, t in zip(axs.flat, terms):
    for (label, (tr, _)), c in zip(runs.items(), SERIES):
        pts = [(j["iter"], j[t]) for j in tr if isinstance(j.get(t), (int, float))]
        if not pts: continue
        x, y = map(np.array, zip(*pts))
        ax.plot(x, y, color=c, lw=0.6, alpha=0.18)
        ax.plot(x, roll(y, a.smooth), color=c, lw=2, label=label)
    ax.set_title(NAMES.get(t, t), fontsize=9, loc="left"); ax.grid(True, color=GRID, lw=0.6); ax.tick_params(length=0)
    ax.spines[["top", "right"]].set_visible(False); ax.set_xlabel("iteration")
for ax in list(axs.flat)[len(terms):]: ax.axis("off")
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=len(l), frameon=False)
fig.suptitle(f"{a.title} — faint: logged iteration; line: rolling mean of {a.smooth} log points", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{a.out}_train.png", dpi=110); plt.close(fig)

# validation (registered validator, every 2k iterations)
VM = [("fullbody_mean_ssim", "full-body mean SSIM (dress4d / bedlam2 / motionx) ↑", lambda j: j.get("fullbody_mean_ssim")),
      ("ubody_ssim", "UBody SSIM ↑ (best.pt needs ≥ 0.8193)", lambda j: j.get("ubody", {}).get("ssim")),
      ("face_lpips", "UBody face LPIPS ↓ (best.pt needs ≤ 0.1873)", lambda j: j.get("ubody", {}).get("face_lpips"))]
for s in ("dress4d", "bedlam2", "motionx"): VM.append((f"{s}_psnr", f"{s} PSNR (dB) ↑", lambda j, s=s: j.get(s, {}).get("psnr")))
fig, axs = plt.subplots(2, 3, figsize=(12.6, 6.2), squeeze=False)
for ax, (k, name, f) in zip(axs.flat, VM):
    for (label, (_, va)), c in zip(runs.items(), SERIES):
        pts = [(j["iter"], f(j)) for j in va if f(j) is not None]
        if pts: x, y = map(np.array, zip(*pts)); ax.plot(x, y, color=c, lw=2, marker="o", ms=4, label=label)
    if k in THR: ax.axhline(THR[k], color=INK2, lw=1, ls="--"); ax.annotate("threshold", (ax.get_xlim()[0], THR[k]), color=INK2, fontsize=8, va="bottom")
    ax.set_title(name, fontsize=9, loc="left"); ax.grid(True, color=GRID, lw=0.6); ax.tick_params(length=0)
    ax.spines[["top", "right"]].set_visible(False); ax.set_xlabel("iteration")
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=len(l), frameon=False)
fig.suptitle(f"{a.title} — registered validation", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{a.out}_val.png", dpi=110); plt.close(fig)
print(f"{a.out}_train.png", f"{a.out}_val.png", {l: (len(tr), tr[-1]["iter"] if tr else None) for l, (tr, _) in runs.items()})
