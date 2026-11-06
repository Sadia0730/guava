"""FinalBench final comparison tables (PREREG "FinalBench: final comparison tables"; amendment 2026-11-05: main table = rows 1-3).
Reads the score files O/eval/<E>/scores/<method>_full.json written by score.py (E = avatar_test_v1, publicbench_v1, publicbench_v2;
clip pools = the registered splits next to this script) and writes phase3_dnt/RESULTS_FINAL.md + results_final.csv: main table
(single-image: GUAVA released, GUAVA-FB [SMPL-X GT], GUAVA-FB [SPRINT, k = 0]), reference table (per-person, video-trained: GART,
GaussianAvatar, ExAvatar GT-face variant only), the BEDLAM2 synthetic table, paired differences vs GUAVA (released), L1, excluded
clips, timing, captions. Pools: FinalBench = UBody + People-Snapshot + NeuMan + X-Humans (41 clips); BEDLAM2 = 40 clips.
Rules: per clip = mean over its test frames (score.py); per dataset / overall = mean over clips, each clip weighs the same
(frame-weighted means in the CSV only). One clip set per pool: a method takes part if it has a scored clip in every dataset of the
pool (otherwise its row is "—" and it does not constrain the set, e.g. ExAvatar on BEDLAM2 or a score file not written yet); the set
= clips scored by every participating method of both tables; every other pool clip is listed with the reason. 95 % CIs = percentile
bootstrap over clips (--boot resamples; numpy default_rng(--seed), one generator per pool, index draws per dataset in pool order,
the same indices for every method and metric): per dataset = resample its clips with replacement; overall = stratified by dataset
(clips resampled within each dataset independently, overall = mean over the union of the resampled clips). Paired differences =
per-clip (row - GUAVA released), mean + the same bootstrap. Timing from per_clip["run"]: creation = recon_ms / 1000 (GUAVA rows; the
SPRINT row's includes its source fit, recon_ms_model = model only) or train_s (baselines); render = render_ms_per_frame or
1000 x render_s_per_frame. Missing score files are reported and skipped; the script is simply rerun when they appear.
Usage: python final_tables.py [--O <root>] [--out-md phase3_dnt/RESULTS_FINAL.md] [--out-csv phase3_dnt/results_final.csv] [--boot 10000]"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; P3 = HERE.parent
ap = argparse.ArgumentParser(); ap.add_argument("--O", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
ap.add_argument("--out-md", default=str(P3 / "RESULTS_FINAL.md")); ap.add_argument("--out-csv", default=str(P3 / "results_final.csv"))
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261108); a = ap.parse_args()
O = Path(a.O)
EVALS = {"avatar_test_v1": "avatar_test_v1.json", "publicbench_v1": "publicbench_v1.json", "publicbench_v2": "publicbench_v2_xhumans.json"}
MAIN = ["guava", "fb_gt", "fb_sprint"]; REF = ["gart", "gaussianavatar", "exavatar_gtface"]; METHODS = MAIN + REF
NAME = {"guava": "GUAVA (released)", "fb_gt": "GUAVA-FB [SMPL-X GT]", "fb_sprint": "GUAVA-FB [SPRINT, k = 0]", "gart": "GART",
        "gaussianavatar": "GaussianAvatar", "exavatar_gtface": "ExAvatar [GT face]"}
DS = {"ubody": "UBody", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "bedlam2": "BEDLAM2"}
POOLS = {"FinalBench": ["ubody", "peoplesnapshot", "neuman", "xhumans"], "BEDLAM2": ["bedlam2"]}
MET = ("psnr", "ssim", "lpips", "l1"); DEC = {"psnr": 2, "ssim": 3, "lpips": 4, "l1": 4}; ARW = {"psnr": "↑", "ssim": "↑", "lpips": "↓", "l1": "↓"}
DIV = "| *per-person, video-trained (reference)* |"

# ---- load: registered pools, score files, per-clip records, reasons for missing clips, format warnings
pool_ds = {}; split_of = {}
for E, sj in EVALS.items():
    for c in json.loads((HERE / sj).read_text())["clips"]: pool_ds[c["id"]] = c["dataset"]; split_of[c["id"]] = E
per = {m: {} for m in METHODS}; reason = {m: {} for m in METHODS}; files, warn = [], []
for E in EVALS:
    for m in METHODS:
        p = O / "eval" / E / "scores" / f"{m}_full.json"
        if not p.exists():
            files.append((p, None)); warn.append(f"score file missing: {p} ({NAME[m]} on {E}); tables built without it, rerun when it exists")
            for c in [c for c, e in split_of.items() if e == E]: reason[m][c] = f"{NAME[m]}: no score file for {E}"
            continue
        s = json.loads(p.read_text()); files.append((p, p.stat().st_mtime))
        if s.get("tag") != "full" or s.get("subset"): warn.append(f"{p}: tag {s.get('tag')!r} / subset {s.get('subset')!r} (expected full / None)")
        for c, n in (s.get("missing_clips") or {}).items(): reason[m][c] = f"{NAME[m]}: not scored ({n} test renders missing)"
        for c, v in s["per_clip"].items():
            if c not in pool_ds: warn.append(f"{p}: clip {c} is not in the registered splits (ignored)"); continue
            if v.get("dataset") != pool_ds[c] or not all(np.isfinite(v.get(k, np.nan)) for k in MET) or not v.get("n"):
                warn.append(f"{p}: clip {c} malformed ({ {k: v.get(k) for k in ('dataset', 'n') + MET} }); skipped"); reason[m][c] = f"{NAME[m]}: malformed score entry"; continue
            per[m][c] = v
        for c in [c for c, e in split_of.items() if e == E and c not in per[m] and c not in reason[m]]: reason[m][c] = f"{NAME[m]}: not in score file"
for c in pool_ds:
    ns = {m: per[m][c]["n"] for m in METHODS if c in per[m]}
    if len(set(ns.values())) > 1: warn.append(f"clip {c}: test-frame count differs between methods {ns}")


def timing(v):
    r = v.get("run") or {}
    cs = r["recon_ms"] / 1000 if "recon_ms" in r else r.get("train_s"); cm = r["recon_ms_model"] / 1000 if "recon_ms_model" in r else None
    rm = r["render_ms_per_frame"] if "render_ms_per_frame" in r else (1000 * r["render_s_per_frame"] if "render_s_per_frame" in r else None)
    return cs, cm, rm, r.get("gpu")


def fmt_time(s): return "—" if s is None else (f"{s * 1000:.0f} ms" if s < 10 else f"{s / 60:.1f} min" if s < 3600 else f"{s / 3600:.1f} h")


def ci(b): lo, hi = np.percentile(b, [2.5, 97.5]); return float(lo), float(hi)


def agg(vd, idx):
    """vd: {dataset: per-clip array}; returns {dataset | 'all': (mean over clips, ci_lo, ci_hi)}; overall stratified by dataset."""
    out, tot, N = {}, 0.0, 0
    for d, v in vd.items(): b = v[idx[d]].sum(1); out[d] = (float(v.mean()), *ci(b / len(v))); tot = tot + b; N += len(v)
    out["all"] = (float(np.concatenate(list(vd.values())).mean()), *ci(tot / N)); return out


def fw(vd, nd):
    out = {d: float((vd[d] * nd[d]).sum() / nd[d].sum()) for d in vd}; V, Nn = np.concatenate(list(vd.values())), np.concatenate(list(nd.values()))
    out["all"] = float((V * Nn).sum() / Nn.sum()); return out


# ---- per pool: participating methods, common clip set, statistics
R = {}
for pool, dss in POOLS.items():
    clips = [c for c, d in pool_ds.items() if d in dss]
    part = [m for m in METHODS if all(any(c in per[m] for c in clips if pool_ds[c] == d) for d in dss)]
    common = [c for c in clips if all(c in per[m] for m in part)]
    groups = {d: [c for c in common if pool_ds[c] == d] for d in dss}
    excl = [(c, "; ".join(reason[m].get(c, f"{NAME[m]}: not in score file") for m in part if c not in per[m])) for c in clips if c not in common]
    skipped = [(m, ", ".join(DS[d] for d in dss if not any(c in per[m] for c in clips if pool_ds[c] == d))) for m in METHODS if m not in part]
    rng = np.random.default_rng(a.seed); idx = {d: rng.integers(0, len(g), size=(a.boot, len(g))) for d, g in groups.items() if g}
    nd = {d: np.array([per[part[0]][c]["n"] for c in g], float) for d, g in groups.items() if g} if part else {}
    val = {m: {k: {d: np.array([per[m][c][k] for c in g]) for d, g in groups.items() if g} for k in MET} for m in part}
    stat = {m: {k: agg(val[m][k], idx) for k in MET} for m in part} if common else {}
    fwm = {m: {k: fw(val[m][k], nd) for k in MET} for m in part} if common else {}
    diff = {m: {k: agg({d: val[m][k][d] - val["guava"][k][d] for d in val[m][k]}, idx) for k in MET} for m in part if m != "guava" and "guava" in part} if common else {}
    tim = {m: {d: [timing(per[m][c]) for c in g] for d, g in list(groups.items()) + [("all", common)]} for m in part}
    R[pool] = dict(dss=dss, clips=clips, part=part, common=common, groups=groups, excl=excl, skipped=skipped, nd=nd, stat=stat, fwm=fwm, diff=diff, tim=tim)

# ---- markdown
def cell(st, d, keys, sign=False): return " / ".join(f"{st[k][d][0]:{'+' if sign else ''}.{DEC[k]}f}" for k in keys)


def cell_ci(st, d, keys, sign=False): return " / ".join(f"[{st[k][d][1]:{'+' if sign else ''}.{DEC[k]}f}, {st[k][d][2]:{'+' if sign else ''}.{DEC[k]}f}]" for k in keys)


def table(pool, rows, keys, src="stat", sign=False):
    """rows: method ids, or None for the divider between the single-image and the per-person block."""
    r = R[pool]; cols = [d for d in r["dss"] if r["groups"][d]]; cols += ["all"] if len(cols) > 1 else []; src = r[src]
    hdr = "| Method | " + " | ".join(f"All ({len(r['common'])} clips)" if d == "all" else f"{DS[d]} ({len(r['groups'][d])})" for d in cols) + " |"
    L = [hdr, "|---|" + "---|" * len(cols)]
    for m in rows:
        if m is None: L.append(DIV + " |" * len(cols)); continue
        if m not in src: L.append(f"| {NAME[m]} | " + " | ".join(["—"] * len(cols)) + " |"); continue
        L.append(f"| {NAME[m]} | " + " | ".join(cell(src[m], d, keys, sign) for d in cols) + " |")
        L.append("| ↳ 95 % CI | " + " | ".join(cell_ci(src[m], d, keys, sign) for d in cols) + " |")
    return L


def frames(pool, d=None): r = R[pool]; return int(sum(n.sum() for dd, n in r["nd"].items() if d is None or dd == d))


Fb, Bd = R["FinalBench"], R["BEDLAM2"]
nf = lambda pool, d: f"{DS[d]} {len(R[pool]['groups'][d])}"  # noqa: E731
md = [f"# FinalBench: final comparison tables", "",
      f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')} by `eval_split/final_tables.py` (`{' '.join(sys.argv[1:]) or 'defaults'}`); "
      f"percentile bootstrap over clips, {a.boot:,} resamples, `numpy.random.default_rng({a.seed})` (one generator per clip pool; overall "
      f"stratified by dataset). PREREG section \"FinalBench: final comparison tables\" (amendment 2026-11-05: main table = rows 1-3). "
      f"Per-clip rows, frame-weighted means and timing fields: `results_final.csv`.", ""]
notes = [f"WARNING: {w}" for w in warn] + [f"**{pool}: {NAME[m]} row shown as — (no scored clips on {miss}); it does not constrain the clip set.**"
                                           for pool in POOLS for m, miss in R[pool]["skipped"]]
if notes: md += ["## Warnings / notes", ""] + [f"- {n}" for n in notes] + [""]
cells = "cells: PSNR ↑ / SSIM ↑ / LPIPS ↓ (LPIPS-VGG), second line = 95 % CI; L1 ↓ in its own table below"
md += [f"## Main table: single-image methods, FinalBench ({len(Fb['common'])} clips / {frames('FinalBench'):,} test frames)", "",
       *table("FinalBench", MAIN, ("psnr", "ssim", "lpips")), "",
       f"Caption. Self-reenactment on FinalBench ({', '.join(nf('FinalBench', d) for d in Fb['dss'])} clips; one identity per clip; {cells}). Single-image "
       "methods use the clip's FIRST training frame as the only source image and are driven by the GT body model of each test frame (converted to "
       "each method's body model); GUAVA-FB [SPRINT, k = 0] uses no GT at all (own detector, crop, StyleMatte matte and SPRINT pose estimate; k = 0, "
       "largest box). No test-time pose, face or expression fitting in any row. Same 512 person crop, masks and metrics for every row "
       "(`eval_split/score.py`); per clip = mean over its test frames; per dataset and overall = mean over clips (each clip weighs the same). "
       f"Clip set = the clips on which all {len(Fb['part'])} methods of the main and the reference table finished (excluded clips listed below). "
       "Disclosures: the AvatarTest-v1 and PublicBench-v1 numbers of these three rows and of the per-person methods were seen before the FinalBench "
       "registration (2026-11-02); UBody clips are upper body only (UBody GT is pseudo-GT, UBody's SMPL-X fits); NeuMan test sets are 3-10 frames; "
       "released GUAVA may have trained on UBody; X-Humans images are renders of textured scans (mask = rendered depth > 0).", "",
       f"## Reference table: per-person, video-trained methods, FinalBench (same {len(Fb['common'])} clips)", "",
       *table("FinalBench", REF, ("psnr", "ssim", "lpips")), "",
       "Caption. Trained on each person's video (training frames); not directly comparable with single-image methods. GART and GaussianAvatar are "
       "driven by SMPL fitted to the GT SMPL-X mesh of each test frame; ExAvatar [GT face] is the conforming variant (test-frame face = the GT's "
       "converted jaw / expression, eyes 0, neutral where the GT is SMPL; no DECA, keypoint or face fitting on test frames; its training-frame "
       f"refinement kept). Same clip set, crop, masks, metrics and CIs as the main table; {cells}.", "",
       f"## BEDLAM2 synthetic table ({len(Bd['common'])} clips / {frames('BEDLAM2'):,} test frames)", "",
       *table("BEDLAM2", MAIN + [None] + REF, ("psnr", "ssim", "lpips")), "",
       "Caption. AvatarTest-v1 BEDLAM2 (40 synthetic full-body clips), same rules, source, driving, crop, masks, metrics and CIs as above; "
       f"{cells}. ExAvatar [GT face] has no BEDLAM2 runs (—; only the non-conforming subset24 runs exist and are not shown). 38 of the 40 BEDLAM2 "
       "identities appear in other BEDLAM2 clips that SPRINT trained on.", "",
       "## Paired differences vs GUAVA (released)", "",
       "Per clip row − GUAVA (released), mean over clips with the same bootstrap CI (cells: ΔPSNR / ΔSSIM / ΔLPIPS / ΔL1; higher is better for "
       "PSNR / SSIM, lower for LPIPS / L1; a CI that excludes 0 is a significant paired difference at the 5 % level, uncorrected for the number of "
       "comparisons).", "", f"### FinalBench ({len(Fb['common'])} clips)", "",
       *table("FinalBench", MAIN[1:] + [None] + REF, MET, "diff", True), "", f"### BEDLAM2 ({len(Bd['common'])} clips)", "",
       *table("BEDLAM2", MAIN[1:] + [None] + REF, MET, "diff", True), "",
       "## L1 ↓ (mean absolute error on the 512 crop, [0, 1] scale)", "", f"### FinalBench ({len(Fb['common'])} clips)", "",
       *table("FinalBench", MAIN + [None] + REF, ("l1",)), "", f"### BEDLAM2 ({len(Bd['common'])} clips)", "",
       *table("BEDLAM2", MAIN + [None] + REF, ("l1",)), "", "## Excluded clips", ""]
for pool in POOLS:
    r = R[pool]
    md.append(f"- {pool}: {len(r['clips'])} registered clips, {len(r['common'])} in the common set, {len(r['excl'])} excluded.")
    md += [f"  - `{c}`: {why}" for c, why in r["excl"]]
md += ["", "## Timing (median over the FinalBench common clips; not comparable across methods)", "",
       "| Method | Avatar creation / training per clip | Render ms / frame | GPU(s) | Clips with timing |", "|---|---|---|---|---|"]
for m in METHODS:
    if m not in Fb["tim"]: md.append(f"| {NAME[m]} | — | — | — | 0 |"); continue
    t = Fb["tim"][m]["all"]; cs = [x[0] for x in t if x[0] is not None]; cm = [x[1] for x in t if x[1] is not None]; rm = [x[2] for x in t if x[2] is not None]
    gp = sorted({x[3] for x in t if x[3]})
    md.append(f"| {NAME[m]} | {fmt_time(float(np.median(cs))) if cs else '—'}{' (model only ' + fmt_time(float(np.median(cm))) + ')' if cm else ''} | "
              f"{np.median(rm):.1f} | {', '.join(gp) or '—'} | {len(cs)} |" if rm else f"| {NAME[m]} | {fmt_time(float(np.median(cs))) if cs else '—'} | — | {', '.join(gp) or '—'} | {len(cs)} |")
md += ["", "Note. GUAVA-family rows: avatar creation from the source frame (`recon_ms`; the SPRINT row includes its source fit: DWPose on CPU, "
       "matting, SPRINT, refinement) and `render_ms_per_frame`, measured inside the evaluation runs; per-person rows: `train_s` (per-clip training "
       "incl. preprocessing) and `render_s_per_frame` from their `run.json`. The baselines (and parts of the evaluation runs) ran on GPUs shared "
       "with other jobs, so these timings are indicative only and not comparable across methods.", "", "## Input score files", ""]
md += [f"- `{p}`: {'MISSING' if t is None else time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))}" for p, t in files]
Path(a.out_md).write_text("\n".join(md) + "\n")

# ---- csv: one row per method x clip (all scored pool clips, `included` = in the common set), plus dataset / overall aggregates
F = ["level", "pool", "method", "dataset", "clip", "included", "n_frames", "n_clips"] + [f"{k}{s}" for k in MET for s in ("", "_lo", "_hi")] + \
    [f"{k}_fw" for k in MET] + [f"d_{k}{s}" for k in MET for s in ("", "_lo", "_hi")] + ["create_s", "create_model_s", "render_ms_per_frame", "gpu"]
num = lambda x: "" if x is None else f"{x:.6f}"  # noqa: E731
med = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
with open(a.out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=F); w.writeheader()
    for pool, r in R.items():
        cs = set(r["common"])
        for m in METHODS:
            for c in r["clips"]:
                if c not in per[m]: continue
                v = per[m][c]; g = per["guava"].get(c); t = timing(v)
                w.writerow({"level": "clip", "pool": pool, "method": m, "dataset": pool_ds[c], "clip": c, "included": int(m in r["part"] and c in cs), "n_frames": v["n"],
                            **{k: num(v[k]) for k in MET}, **({f"d_{k}": num(v[k] - g[k]) for k in MET} if g and m != "guava" else {}),
                            "create_s": num(t[0]), "create_model_s": num(t[1]), "render_ms_per_frame": num(t[2]), "gpu": t[3] or ""})
        for m in r["part"]:
            for d in [d for d in r["dss"] if r["groups"][d]] + ["all"]:
                t = r["tim"][m][d]; st, dfm = r["stat"][m], r["diff"].get(m)
                w.writerow({"level": "overall" if d == "all" else "dataset", "pool": pool, "method": m, "dataset": d, "included": 1,
                            "n_frames": frames(pool, None if d == "all" else d), "n_clips": len(r["common"]) if d == "all" else len(r["groups"][d]),
                            **{f"{k}{s}": num(st[k][d][i]) for k in MET for i, s in enumerate(("", "_lo", "_hi"))}, **{f"{k}_fw": num(r["fwm"][m][k][d]) for k in MET},
                            **({f"d_{k}{s}": num(dfm[k][d][i]) for k in MET for i, s in enumerate(("", "_lo", "_hi"))} if dfm else {}),
                            "create_s": num(med([x[0] for x in t if x[0] is not None])), "create_model_s": num(med([x[1] for x in t if x[1] is not None])),
                            "render_ms_per_frame": num(med([x[2] for x in t if x[2] is not None])), "gpu": ", ".join(sorted({x[3] for x in t if x[3]}))})
for wn in warn: print("WARNING:", wn)
for pool, r in R.items():
    print(f"{pool}: {len(r['common'])} / {len(r['clips'])} clips (excluded {len(r['excl'])}); methods: {', '.join(NAME[m] for m in r['part'])}")
    for m in r["part"]: print(f"  {NAME[m]:28s} " + cell(r["stat"][m], "all", MET) + ("   Δ " + cell(r["diff"][m], "all", MET, True) if m in r["diff"] else ""))
print("wrote", a.out_md, a.out_csv)
