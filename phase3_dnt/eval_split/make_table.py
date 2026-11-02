"""AvatarTest-v1 table in the layout of GUAVA's Table 2, extended per dataset (PREREG "AvatarTest-v1 addendum 1"):
PSNR / L1 / SSIM / LPIPS for UBody, BEDLAM2 and all clips, reconstruction input and reconstruction time, registered caveats
in the caption. Reads E/scores/<method>_<tag>.json, E/reports/recon_timing_<row>.json (GUAVA-family rows, idle GPU) and the
baselines' run.json train_s (median over the clips scored). Writes E/table_<tag>.md (and prints it).
Usage: python make_table.py --tag full --methods guava,exavatar,gart,gaussianavatar,fb_gt,fb_sprint"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="full"); ap.add_argument("--methods", required=True)
ap.add_argument("--title", default=""); a = ap.parse_args()
E = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1")
NAME = {"guava": "GUAVA (released)", "exavatar": "ExAvatar", "gart": "GART", "gaussianavatar": "GaussianAvatar",
        "fb_gt": "GUAVA-FB [SMPL-X GT]", "fb_sprint": "GUAVA-FB [SPRINT]"}
INPUT = {"guava": "first frame", "fb_gt": "first frame", "fb_sprint": "first frame"}


def fmt_time(ms):
    s = ms / 1000
    return f"≈ {ms:.0f} ms" if s < 10 else (f"≈ {s / 60:.0f} min" if s < 3600 else f"≈ {s / 3600:.1f} h")


rows, gpus = [], set()
for m in a.methods.split(","):
    p = E / "scores" / f"{m}_{a.tag}.json"
    if not p.exists(): rows.append((m, None, "", "", {})); continue
    s = json.loads(p.read_text()); agg = s["aggregate"]
    tj = E / "reports" / f"recon_timing_{m}.json"
    if tj.exists():
        t = json.loads(tj.read_text()); rt = fmt_time(t["recon_ms_median"]); gpus.add(t["gpu"])
    else:
        tr = [c["run"]["train_s"] for c in s["per_clip"].values() if "run" in c and "train_s" in c["run"]]
        rt = fmt_time(1000 * float(np.median(tr))) if tr else "-"
        gpus |= {c["run"].get("gpu") for c in s["per_clip"].values() if "run" in c and c["run"].get("gpu")}
    rows.append((m, agg, INPUT.get(m, "first half of the clip"), rt, s.get("missing_clips") or {}))
hdr = "| Method | " + " | ".join(f"{d} PSNR ↑ | {d} L1 ↓ | {d} SSIM ↑ | {d} LPIPS ↓" for d in ("UBody", "BEDLAM2", "All")) + " | Recon. input | Recon. time |"
L = [f"### AvatarTest-v1 ({a.tag}){' — ' + a.title if a.title else ''}", "", hdr, "|---|" + "---|" * 14]
for m, agg, inp, rt, _ in rows:
    if agg is None: L.append(f"| {NAME.get(m, m)} | " + " | ".join(["-"] * 12) + " | | |"); continue
    cell = []
    for d in ("ubody", "bedlam2", "all"):
        g = agg.get(d)
        cell += [f"{g['psnr']:.2f}", f"{g['l1']:.4f}", f"{g['ssim']:.4f}", f"{g['lpips']:.4f}"] if g else ["-"] * 4
    L.append(f"| {NAME.get(m, m)} | " + " | ".join(cell) + f" | {inp} | {rt} |")
n = {d: next((r[1][d]["clips"] for r in rows if r[1] and d in r[1]), 0) for d in ("ubody", "bedlam2", "all")}
f = {d: next((r[1][d]["frames"] for r in rows if r[1] and d in r[1]), 0) for d in ("ubody", "bedlam2", "all")}
L += ["", (f"Caption. Self-reenactment on AvatarTest-v1 ({n['all']} clips / {f['all']:,} test frames: UBody {n['ubody']} real upper-body clips, "
           f"BEDLAM2 {n['bedlam2']} synthetic full-body clips; one identity per clip). Protocol as GUAVA Table 2: single-image methods use the "
           "clip's first frame; per-subject methods train on the first half; all are scored on the latter half (512 person crop, GT x "
           "StyleMatte mask on black; GUAVA's metric code, frame-weighted). Driving poses: ground-truth SMPL-X for every method except "
           "GUAVA-FB [SPRINT], which uses SPRINT's per-frame estimate (no GT anywhere); SMPL for GART / GaussianAvatar fitted to the GT "
           "SMPL-X mesh; ExAvatar's own face fitting allowed. No test-time pose fitting for any method. UBody clips are upper body only, "
           "so full-body baselines may struggle there (per-dataset columns). UBody GT is pseudo-GT (UBody's SMPL-X fits); released GUAVA "
           "may have trained on UBody clips; in the full split, 38 of the 40 BEDLAM2 identities appear in other BEDLAM2 clips SPRINT trained on. "
           f"Recon. time: avatar creation from the source (GUAVA-family: model only for GT rows, + SPRINT's source fit for the SPRINT row) / "
           f"per-clip training incl. preprocessing (baselines), measured on {', '.join(sorted(g for g in gpus if g))}."
           + "".join(f" {NAME.get(m, m)}: {agg['all']['clips']} of {n['all']} clips scored (missing: {', '.join(sorted(mc))}); its columns cover only those clips."
                     for m, agg, _, _, mc in rows if agg and mc))]
(E / f"table_{a.tag}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
