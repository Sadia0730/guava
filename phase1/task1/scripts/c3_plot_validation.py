"""Phase 1 Task 1 screening: 3DPW validation curves with PEAR and student 235000 as reference lines.

Reads the in-loop validation entries of each run's train_log.jsonl and the standalone standard
evaluator's summary.json for the two reference models (same split, crop and protocol).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("/raid/ubx858/outputs/phase1_task1")
REFERENCES = {
    "PEAR (released ViT-H)": OUT / "eval_3dpw/pear/validation_gt_keypoints/summary.json",
    "Student 235000": OUT / "eval_3dpw/student_235000/validation_gt_keypoints_clean/summary.json",
}
METRICS = (("pa_mpjpe_mm", "PA-MPJPE"), ("mpjpe_mm", "MPJPE"), ("pve_mm", "PVE"))
COLOURS = {"a": "tab:blue", "b": "tab:orange", "c": "tab:green"}
LABELS = {"a": "(a) PEAR distillation", "b": "(b) BEDLAM2 GT", "c": "(c) GT + distillation"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=OUT / "screen")
    p.add_argument("--output", type=Path, default=OUT / "screen/validation_curves.png")
    args = p.parse_args()

    refs = {}
    for name, path in REFERENCES.items():
        summary = json.loads(path.read_text())
        refs[name] = {k: summary["metrics"][k]["mean"] for k, _ in METRICS}

    curves = {}
    for variant in "abc":
        log = args.runs / variant / "train_log.jsonl"
        if log.exists():
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            curves[variant] = [r for r in rows if r.get("split") == "val3dpw"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 4.5))
    for ax, (key, title) in zip(axes, METRICS):
        for variant, rows in curves.items():
            if rows:
                ax.plot([r["step"] for r in rows], [r[key] for r in rows], "o-", ms=3,
                        color=COLOURS[variant], label=LABELS[variant])
        for (name, values), style in zip(refs.items(), ("--", ":")):
            ax.axhline(values[key], color="black", ls=style, lw=1.2, label=f"{name}: {values[key]:.1f}")
        ax.set_title(f"3DPW validation {title} (mm)")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        lo = min([v[key] for v in refs.values()] + [r[key] for rows in curves.values() for r in rows])
        ax.set_ylim(bottom=max(0.0, lo - 15), top=max(v[key] for v in refs.values()) * 1.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
