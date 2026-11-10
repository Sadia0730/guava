"""Extended-benchmark tables (PREREG "Extended benchmark", 2026-11-09). Reads score files written by score.py:
O/eval/<E>/scores/<method>_full.json for E = avatar_test_v1 (UBody clips only), publicbench_v1 (People-Snapshot / NeuMan standard 4),
publicbench_v2 (X-Humans) and finalbench_ext (new People-Snapshot / NeuMan sequences, UBC-Fashion). Writes phase3_dnt/RESULTS_EXT.md
and results_ext.csv.
Rules (registered): per clip = mean over its test frames; per dataset = mean over clips; overall = mean over all clips AND mean of the
five dataset means; People-Snapshot and NeuMan also as "standard 4" columns. One clip set for all rows: the clips scored by every row
(others listed with the reason). 95 % CIs: percentile bootstrap, 10,000 resamples, numpy default_rng(20261109), stratified by dataset,
resampling PEOPLE within each dataset (People-Snapshot: subject; UBC-Fashion: the split's ArcFace cluster "person"; other datasets:
one person per clip); the same draws for every row and metric; paired differences per clip (row - reference) with the same draws.
Usage: python ext_tables.py [--boot 10000] [--out-md ...] [--out-csv ...]"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; P3 = HERE.parent
ap = argparse.ArgumentParser(); ap.add_argument("--O", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261109)
ap.add_argument("--out-md", default=str(P3 / "RESULTS_EXT.md")); ap.add_argument("--out-csv", default=str(P3 / "results_ext.csv")); a = ap.parse_args()
O = Path(a.O)
ROOTS = [("avatar_test_v1", "avatar_test_v1.json", {"ubody"}), ("publicbench_v1", "publicbench_v1.json", None),
         ("publicbench_v2", "publicbench_v2_xhumans.json", None), ("finalbench_ext", "finalbench_ext.json", None)]
SINGLE = ["guava", "fb_gt", "fb_sprint", "fb_update25", "fb_update50", "fb_update100", "lhm_facefix", "lhmpp", "idol"]
REF = ["gart", "gaussianavatar", "exavatar_gtface"]; ROWS = SINGLE + REF
NAME = {"guava": "GUAVA (released)", "fb_gt": "Ours [SMPL-X GT]", "fb_sprint": "Ours [SPRINT pose]",
        "fb_update25": "Ours + update, 25 % of video", "fb_update50": "Ours + update, 50 % of video", "fb_update100": "Ours + update, 100 % of video",
        "lhm_facefix": "LHM", "lhmpp": "LHM++", "idol": "IDOL", "gart": "GART", "gaussianavatar": "GaussianAvatar", "exavatar_gtface": "ExAvatar [GT face]"}
DS = ["ubody", "peoplesnapshot", "neuman", "xhumans", "ubc"]
DSN = {"ubody": "UBody", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "ubc": "UBC-Fashion"}
MET = ("psnr", "ssim", "lpips", "l1"); DEC = {"psnr": 2, "ssim": 3, "lpips": 3, "l1": 4}

# ---- pool: clips, dataset, person, standard-4 flag
pool = {}
for E, sj, only in ROOTS:
    p = HERE / sj
    if not p.exists(): continue
    for c in json.loads(p.read_text())["clips"]:
        if c["dataset"] not in DS or (only and c["dataset"] not in only): continue
        ds = c["dataset"]; name = c["clip"]
        person = (name.rsplit("-", 1)[0] if ds == "peoplesnapshot" else c.get("person") or c["id"]) if ds != "ubc" else f"ubc:{c['person']}"
        pool[c["id"]] = {"E": E, "ds": ds, "person": person, "std4": E == "publicbench_v1"}
# ---- scores
S, missing = {}, {}
for m in ROWS:
    S[m] = {}
    for E in {v["E"] for v in pool.values()}:
        f = O / E / "scores" / f"{m}_full.json"
        if f.exists():
            for cid, r in json.loads(f.read_text())["per_clip"].items():
                if cid in pool and pool[cid]["E"] == E: S[m][cid] = r
    missing[m] = sorted(c for c in pool if c not in S[m])
clips = sorted(c for c in pool if all(c in S[m] for m in ROWS))
excl = {c: [m for m in ROWS if c not in S[m]] for c in pool if c not in clips}
ids = {d: [c for c in clips if pool[c]["ds"] == d] for d in DS}
# ---- bootstrap weights: per dataset W[d] (boot x clips), resampling people
rng = np.random.default_rng(a.seed); W = {}
for d in DS:
    cs = ids[d]
    if not cs: continue
    people = sorted({pool[c]["person"] for c in cs}); pc = {p: [i for i, c in enumerate(cs) if pool[c]["person"] == p] for p in people}
    draw = rng.integers(0, len(people), size=(a.boot, len(people))); w = np.zeros((a.boot, len(cs)))
    for j, p in enumerate(people):
        cnt = (draw == j).sum(1)
        for i in pc[p]: w[:, i] += cnt
    W[d] = w
DSP = [d for d in DS if ids[d]]


def stats(vals):
    """vals: clip id -> value. Returns {column: (mean, lo, hi)} for datasets, standard-4 columns, overall clip-mean, mean of dataset means."""
    out, bm, bnum, bden = {}, [], 0, 0
    for d in DSP:
        x = np.array([vals[c] for c in ids[d]]); bd = (W[d] @ x) / np.maximum(W[d].sum(1), 1e-9)
        out[d] = (float(x.mean()), *np.percentile(bd, [2.5, 97.5])); bm.append(bd); bnum = bnum + W[d] @ x; bden = bden + W[d].sum(1)
        if d in ("peoplesnapshot", "neuman"):
            s = [i for i, c in enumerate(ids[d]) if pool[c]["std4"]]
            if s: out[d + "_std4"] = (float(x[s].mean()), np.nan, np.nan)
    xa = np.array([vals[c] for c in clips])
    out["all_clips"] = (float(xa.mean()), *np.percentile(bnum / bden, [2.5, 97.5]))
    out["mean_of_ds"] = (float(np.mean([out[d][0] for d in DSP])), *np.percentile(np.mean(bm, 0), [2.5, 97.5]))
    return out


R = {m: {k: stats({c: S[m][c][k] for c in clips}) for k in MET} for m in ROWS} if clips else {}
D = {}
if clips:
    for ref, rows in (("guava", [m for m in ROWS if m != "guava"]), ("fb_gt", ["fb_update25", "fb_update50", "fb_update100"])):
        for m in rows: D[(m, ref)] = {k: stats({c: S[m][c][k] - S[ref][c][k] for c in clips}) for k in MET}
# ---- write
COLS = [d for d in DSP] + [c for c in ("peoplesnapshot_std4", "neuman_std4") if clips and c in R["fb_gt"]["psnr"]] + ["all_clips", "mean_of_ds"]
CN = {**DSN, "peoplesnapshot_std4": "P-S std 4", "neuman_std4": "NeuMan std 4", "all_clips": "All clips", "mean_of_ds": "Mean of datasets"}
L = [f"# Extended benchmark results (PREREG \"Extended benchmark\"; written {time.strftime('%F %T')} by eval_split/ext_tables.py)", ""]
L += [f"Clips in the table: {len(clips)} of {len(pool)} (" + ", ".join(f"{DSN[d]} {len(ids[d])}" for d in DS) + f"); people: " +
      ", ".join(f"{DSN[d]} {len({pool[c]['person'] for c in ids[d]})}" for d in DSP) + ". Each clip weighs the same within its dataset.", ""]
if excl:
    L += [f"**Not yet in the table ({len(excl)} clips):** missing rows " + "; ".join(f"{NAME[m]} {sum(m in v for v in excl.values())}" for m in ROWS if any(m in v for v in excl.values())), ""]
if clips:
    for k, lab in (("psnr", "PSNR ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓")):
        L += [f"## {lab}", "", "| Method | " + " | ".join(CN[c] for c in COLS) + " | 95 % CI (all clips) |", "|---|" + "---|" * (len(COLS) + 1)]
        for m in ROWS:
            if m == REF[0]: L.append("| *per-person, trained on the video (reference)* |" + " |" * (len(COLS) + 1))
            st = R[m][k]; L.append(f"| {NAME[m]} | " + " | ".join(f"{st[c][0]:.{DEC[k]}f}" for c in COLS) + f" | [{st['all_clips'][1]:.{DEC[k]}f}, {st['all_clips'][2]:.{DEC[k]}f}] |")
        L.append("")
    L += ["## Paired differences (per clip, same bootstrap draws): mean [95 % CI], all clips / mean of datasets", "",
          "| Row − reference | PSNR | SSIM | LPIPS |", "|---|---|---|---|"]
    for (m, ref), st in D.items():
        L.append(f"| {NAME[m]} − {NAME[ref]} | " + " | ".join(f"{st[k]['all_clips'][0]:+.{DEC[k]}f} [{st[k]['all_clips'][1]:+.{DEC[k]}f}, {st[k]['all_clips'][2]:+.{DEC[k]}f}] / {st[k]['mean_of_ds'][0]:+.{DEC[k]}f}" for k in ("psnr", "ssim", "lpips")) + " |")
    L.append("")
if excl:
    L += ["## Clips not in the table", ""] + [f"- {c}: missing {', '.join(v)}" for c, v in sorted(excl.items())] + [""]
Path(a.out_md).write_text("\n".join(L))
with open(a.out_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "column", "metric", "mean", "lo", "hi"])
    for m in R:
        for k in MET:
            for c, v in R[m][k].items(): w.writerow([m, c, k, *[f"{x:.6f}" for x in v]])
print(len(clips), "clips in table;", len(excl), "not yet;", "missing per row:", {m: len(v) for m, v in missing.items()})
