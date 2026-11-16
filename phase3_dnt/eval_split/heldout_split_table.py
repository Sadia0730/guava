"""Held-out averages and the far-view split, from the per-clip scores (PREREG 2026-11-15 "Held-out averages and the far-view
split, from the per-clip scores"; descriptive). (a) all test frames: O/eval/<root>/scores/<row>_full.json, clips and people as
ext_tables.py; (b) far-view frames: per_clip of O/reports/far_view/far_view_scores.json. Groups: held out = People-Snapshot +
NeuMan + X-Humans, in domain = UBody + UBC-Fashion, each dataset, all. Clip mean -> mean over the clips of a group; 95 %
percentile bootstrap (people resampled within datasets, the same draws for every row and metric; a group pools its datasets).
Reads only; writes --out/heldout_split.{md,json} (never overwrites).
Usage: python heldout_split_table.py [--out O/reports/heldout_split]"""
import argparse, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--O", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261109); ap.add_argument("--out", default="")
a = ap.parse_args(); OR = Path(a.O); O = OR / "eval"; OUT = Path(a.out) if a.out else OR / "reports" / "heldout_split"
for f in ("heldout_split.md", "heldout_split.json"): assert not (OUT / f).exists(), f"{OUT / f} exists (never overwrite)"
ROOTS = [("avatar_test_v1", "avatar_test_v1.json", {"ubody"}), ("publicbench_v1", "publicbench_v1.json", None),
         ("publicbench_v2", "publicbench_v2_xhumans.json", None), ("finalbench_ext", "finalbench_ext.json", None)]
ROWS = ["guava", "lhm_facefix", "lhmpp", "idol", "fb_gt", "fb_update100", "gart", "gaussianavatar", "exavatar_gtface", "ft_gt"]
NAME = {"guava": "GUAVA", "lhm_facefix": "LHM", "lhmpp": "LHM++", "idol": "IDOL", "fb_gt": "RAGA, photo", "fb_update100": "RAGA, photo + update",
        "gart": "GART", "gaussianavatar": "GaussianAvatar", "exavatar_gtface": "ExAvatar", "ft_gt": "RAGA, fine tuned"}
PAIRS = [("fb_gt", r) for r in ("lhm_facefix", "lhmpp", "idol", "guava")] + [("fb_update100", "fb_gt")] + \
        [(m, r) for m in ("fb_update100", "ft_gt") for r in ("gart", "gaussianavatar", "exavatar_gtface")]
DS = ["ubody", "peoplesnapshot", "neuman", "xhumans", "ubc"]
DSN = {"ubody": "UBody", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "ubc": "UBC-Fashion"}
GROUPS = {"heldout": ["peoplesnapshot", "neuman", "xhumans"], "indomain": ["ubody", "ubc"], **{d: [d] for d in DS}, "all": DS}
GN = {"heldout": "Held out", "indomain": "In domain", **DSN, "all": "All"}
MET = ("psnr", "ssim", "lpips"); DEC = {"psnr": 2, "ssim": 3, "lpips": 3}

pool = {}
for E, sj, only in ROOTS:
    for c in json.loads((HERE / sj).read_text())["clips"]:
        if c["dataset"] not in DS or (only and c["dataset"] not in only): continue
        ds = c["dataset"]; name = c["clip"]
        person = (name.rsplit("-", 1)[0] if ds == "peoplesnapshot" else c.get("person") or c["id"]) if ds != "ubc" else f"ubc:{c['person']}"
        pool[c["id"]] = {"E": E, "ds": ds, "clip": name, "person": person}
# (a) all test frames
SA = {m: {} for m in ROWS}
for m in ROWS:
    for E in {v["E"] for v in pool.values()}:
        f = O / E / "scores" / f"{m}_full.json"
        if f.exists():
            for cid, r in json.loads(f.read_text())["per_clip"].items():
                if cid in pool and pool[cid]["E"] == E: SA[m][cid] = r
# (b) far-view frames (keys "<dataset>/<clip>")
fv = json.loads((OR / "reports" / "far_view" / "far_view_scores.json").read_text()); by = {(v["ds"], v["clip"]): c for c, v in pool.items()}
SB = {m: {} for m in ROWS}; nfr = {}
for key, r in fv["per_clip"].items():
    ds, name = key.split("/", 1); cid = by[(ds, name)]; nfr[cid] = r["n"]
    for m in ROWS: SB[m][cid] = r[m]


def table(S, tag):
    clips = sorted(c for c in pool if all(c in S[m] for m in ROWS)); ids = {d: [c for c in clips if pool[c]["ds"] == d] for d in DS}
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
    groups = {g: [d for d in dl if d in W] for g, dl in GROUPS.items()}; groups = {g: dl for g, dl in groups.items() if dl}

    def stats(vals):
        out = {}
        for g, dl in groups.items():
            num = sum(W[d] @ np.array([vals[c] for c in ids[d]]) for d in dl); den = sum(W[d].sum(1) for d in dl)
            x = np.array([vals[c] for d in dl for c in ids[d]]); lo, hi = np.percentile(num / np.maximum(den, 1e-9), [2.5, 97.5])
            out[g] = [float(x.mean()), float(lo), float(hi)]
        return out
    R = {m: {k: stats({c: S[m][c][k] for c in clips}) for k in MET} for m in ROWS}
    D = {f"{m} - {r}": {k: stats({c: S[m][c][k] - S[r][c][k] for c in clips}) for k in MET} for m, r in PAIRS}
    n = {g: sum(len(ids[d]) for d in dl) for g, dl in groups.items()}
    better = {f"{m} - {r}": {g: {k: int(sum((S[m][c][k] - S[r][c][k]) * (-1 if k == "lpips" else 1) > 0 for d in dl for c in ids[d])) for k in MET}
                             for g, dl in groups.items()} for m, r in PAIRS}
    people_n = {g: sum(len({pool[c]["person"] for c in ids[d]}) for d in dl) for g, dl in groups.items()}
    cols = list(groups); L = [f"## {tag} ({len(clips)} clips)", "", "Mean over a clip's frames, then over the clips of the group. PSNR / SSIM / LPIPS.", "",
                              "| Method | " + " | ".join(f"{GN[g]} ({n[g]})" for g in cols) + " |", "|---|" + "---|" * len(cols)]
    for m in ROWS: L.append(f"| {NAME[m]} | " + " | ".join(" / ".join(f"{R[m][k][g][0]:.{DEC[k]}f}" for k in MET) for g in cols) + " |")
    L += ["", "95 % bootstrap intervals of the group means (PSNR):", ""]
    for m in ROWS: L.append(f"- {NAME[m]}: " + "; ".join(f"{GN[g]} {R[m]['psnr'][g][0]:.2f} [{R[m]['psnr'][g][1]:.2f}, {R[m]['psnr'][g][2]:.2f}]" for g in ("heldout", "indomain", "all") if g in groups))
    L += ["", "Paired differences, mean [95 % bootstrap interval], clips better / clips:", ""]
    for m, r in PAIRS:
        k0 = f"{m} - {r}"; L.append(f"- **{NAME[m]} - {NAME[r]}**")
        for g in cols:
            L.append(f"  - {GN[g]} ({n[g]} clips, {people_n[g]} people): " + "; ".join(
                f"{k.upper()} {D[k0][k][g][0]:+.{DEC[k]}f} [{D[k0][k][g][1]:+.{DEC[k]}f}, {D[k0][k][g][2]:+.{DEC[k]}f}] {better[k0][g][k]}/{n[g]}" for k in MET))
    return L, {"clips": len(clips), "n": n, "people": people_n, "missing": {m: sorted(c for c in pool if c not in S[m]) for m in ROWS} if S is SA else None,
               "rows": R, "paired": D, "clips_better": better}


LA, JA = table(SA, "All test frames"); LB, JB = table(SB, f"Far-view frames (turned >= 135 deg from the source; {sum(nfr.values())} frames)")
OUT.mkdir(parents=True, exist_ok=True)
L = [f"# Held-out averages and the far-view split, from the per-clip scores (written {time.strftime('%F %T')})", "",
     "Held out = People-Snapshot, NeuMan, X-Humans (never in training). In domain = UBody, UBC-Fashion. Bootstrap: 10,000 draws, "
     f"seed {a.seed}, people resampled within datasets.", ""] + LA + [""] + LB
(OUT / "heldout_split.md").write_text("\n".join(L) + "\n")
(OUT / "heldout_split.json").write_text(json.dumps({"boot": a.boot, "seed": a.seed, "groups": GROUPS, "all_frames": JA, "far_view": JB,
                                                    "far_view_frames": int(sum(nfr.values()))}, indent=1) + "\n")
print("\n".join(L))
