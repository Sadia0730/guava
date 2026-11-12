"""Main results table as an image (PNG for slides, vector PDF), from results_ext_a6.csv (ext_tables.py --amend6).
Two sections: feed-forward methods (no training on the person; our video-update rows fuse the person's video without gradient
steps) and methods trained per subject on the person's video (first half). Cells: PSNR / SSIM / LPIPS, per dataset = mean over
clips, "All clips" = mean over all clips. Bold: best per metric and column within each section.
Timing columns are copied from the author's table (2026-11-11); fine-tuned rows use the same network and Gaussian count as the
base model, so their per-frame cost is the base model's; fine-tune time = median over clips of subject_ft.py's train_s.
Usage: python fig_results_table.py --csv ../results_ext_a6.csv --out <dir> [--name table_results]"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                      # noqa: E402
import numpy as np                                                                                   # noqa: E402
from matplotlib.patches import Rectangle                                                             # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--name", default="table_results")
ap.add_argument("--ft-logs", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/subject_ft")
a = ap.parse_args(); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)

V = {}
for r in csv.DictReader(open(a.csv)): V[(r["method"], r["column"], r["metric"])] = float(r["mean"])
ft_min = np.median([json.loads(f.read_text())["train_s"] for f in Path(a.ft_logs).glob("*/*.json")]) / 60
FT = f"video + {ft_min:.0f} min train"
COLS = [("ubody", "UBody (30)"), ("peoplesnapshot", "People-Snapshot\n(24)"), ("neuman", "NeuMan (6)"), ("xhumans", "X-Humans (3)"),
        ("ubc", "UBC-Fashion (100)"), ("all_clips", "All clips (163)")]
SECTIONS = [
    ("Feed-forward, no training on the person:", [
        ("guava", "GUAVA (released)", "0.55 s", "24.3 ms (41 FPS)"),
        ("lhmpp", "LHM++", "1.5 s", "53.9 ms (19 FPS)"),
        ("lhm_facefix", "LHM", "7.5 s", "22.2 ms (45 FPS)"),
        ("idol", "IDOL", "2.0 s", "36.7 ms (27 FPS)"),
        ("fb_gt", "Ours (GT pose)", "0.55 s", "23.4 ms (43 FPS)"),
        ("fb_sprint", "Ours (SPRINT pose,\nautomatic)", "0.55 s", "23.4 ms; 51 FPS live"),
        ("fb_update100", "Ours + video update\n(GT pose)", "0.55 s", "28.7 ms; 38 FPS live"),
        ("fb_sprint_update100", "Ours + video update\n(SPRINT pose)", "0.55 s", "TBD")]),
    ("Trained per subject on the person's video (first half):", [
        ("ft_gt", "Ours, trained per subject\n(GT pose)", FT, "23.4 ms (43 FPS)"),
        ("ft_update100", "Ours, trained per subject\n+ video update (GT pose)", FT, "28.7 ms; 38 FPS live"),
        ("ft_sprint_update100", "Ours, trained per subject\n+ video update (SPRINT pose)", FT, "TBD"),
        ("gart", "GART", "video + 36.5 s train", "16.9 ms (59 FPS)"),
        ("gaussianavatar", "GaussianAvatar", "video + train (TBD)", "TBD"),
        ("exavatar_gtface", "ExAvatar", "video + 29.5 min", "53.1 ms (19 FPS)")])]
MET = (("psnr", ".2f", max), ("ssim", ".3f", max), ("lpips", ".3f", min))
missing = [(m, c, k) for _, rows in SECTIONS for m, *_ in rows for c, _ in COLS for k, *_ in MET if (m, c, k) not in V]
assert not missing, f"missing in {a.csv}: {missing[:5]}"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
HEAD, GRID, INK, SEC = "#56708F", "#D5D9DF", "#1f1f1f", "#444444"
cw = [1.95] + [1.3] * len(COLS) + [1.3, 1.45]; Wt = sum(cw); hh, rh, sh = 0.42, 0.36, 0.30
nrow = sum(len(r) for _, r in SECTIONS); fh = 0.40; Ht = hh + nrow * rh + len(SECTIONS) * sh + fh
fig = plt.figure(figsize=(Wt, Ht)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wt); ax.set_ylim(Ht, 0); ax.axis("off")
xs = np.concatenate([[0], np.cumsum(cw)])
ax.add_patch(Rectangle((0, 0), Wt, hh, fc=HEAD, ec="none"))
for i, t in enumerate(["Method"] + [c[1] for c in COLS] + ["Photo → avatar", "Frame → screen"]):
    ax.text(xs[i] + 0.08, hh / 2, t, ha="left", va="center", fontsize=7.6, weight="bold", color="white", linespacing=1.05)
y = hh
for title, rows in SECTIONS:
    ax.add_patch(Rectangle((0, y), Wt, sh, fc="white", ec=GRID, lw=0.6))
    ax.text(0.08, y + sh / 2, title, ha="left", va="center", fontsize=7.2, style="italic", color=SEC); y += sh
    best = {(c, k): f([round(V[(m, c, k)], int(fm[1])) for m, *_ in rows]) for c, _ in COLS for k, fm, f in MET}
    for m, label, t_av, t_fr in rows:
        ax.add_patch(Rectangle((0, y), Wt, rh, fc="white", ec=GRID, lw=0.6))
        for i in range(1, len(xs) - 1): ax.plot([xs[i], xs[i]], [y, y + rh], color=GRID, lw=0.6)
        ax.text(0.08, y + rh / 2, label, ha="left", va="center", fontsize=7.2, weight="bold", color=INK, linespacing=1.05)
        for ci, (c, _) in enumerate(COLS):
            x = xs[ci + 1] + 0.08
            for ki, (k, fmt, _) in enumerate(MET):
                v = V[(m, c, k)]; s = format(v, fmt) + (" / " if ki < 2 else "")
                tt = ax.text(x, y + rh / 2, s, ha="left", va="center", fontsize=7.0, color=INK,
                             weight="bold" if round(v, int(fmt[1])) == best[(c, k)] else "normal")
                fig.canvas.draw() if ki == 0 and ci == 0 and m == rows[0][0] else None
                bb = tt.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted()); x = bb.x1
        ax.text(xs[-3] + 0.08, y + rh / 2, t_av, ha="left", va="center", fontsize=7.0, color=INK)
        ax.text(xs[-2] + 0.08, y + rh / 2, t_fr, ha="left", va="center", fontsize=7.0, color=INK)
        y += rh
ax.text(0.04, y + 0.06, f"Cells: PSNR / SSIM / LPIPS; per dataset, mean over clips. Bold numbers: best within each section. Video update: frames from the first half of the person's video are fused into the avatar\nwithout any training steps. Trained per subject: our model fine-tuned on the same first-half frames the per-subject baselines train on (median {ft_min:.1f} min per person).", ha="left", va="top", fontsize=6.6, color=SEC, linespacing=1.3)
fig.savefig(OUT / f"{a.name}.pdf"); fig.savefig(OUT / f"{a.name}.png", dpi=200)
md = ["| Method | " + " | ".join(c[1].replace("\n", " ") for c in COLS) + " | Photo → avatar | Frame → screen |", "|" + "---|" * (len(COLS) + 3)]
for title, rows in SECTIONS:
    best = {(c, k): f([round(V[(m, c, k)], int(fm[1])) for m, *_ in rows]) for c, _ in COLS for k, fm, f in MET}
    md.append(f"| *{title}* |" + " |" * (len(COLS) + 2))
    for m, label, t_av, t_fr in rows:
        cell = lambda c: " / ".join((f"**{format(V[(m, c, k)], fmt)}**" if round(V[(m, c, k)], int(fmt[1])) == best[(c, k)] else format(V[(m, c, k)], fmt)) for k, fmt, _ in MET)
        md.append(f"| {label.replace(chr(10), ' ')} | " + " | ".join(cell(c) for c, _ in COLS) + f" | {t_av} | {t_fr} |")
(OUT / f"{a.name}.md").write_text("\n".join(md) + f"\n\nCells: PSNR / SSIM / LPIPS; per dataset, mean over clips. Bold numbers: best within each section. Video update: frames from the first half of the person's video are fused into the avatar without any training steps. Trained per subject: our model fine-tuned on the same first-half frames the per-subject baselines train on (median {ft_min:.1f} min per person).\n")
print("wrote", OUT / f"{a.name}.png", f"{Wt:.2f} x {Ht:.2f} in")
