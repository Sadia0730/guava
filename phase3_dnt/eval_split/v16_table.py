"""LHM++ with 16 images vs our update on the same 16 frames, all 163 clips (PREREG 2026-11-15 "LHM++ with 16 images on all 163
clips, and our update on the same 16 frames"). Reads score.py's O/eval/<root>/scores/<row>_full.json and the run.json of both
16-frame rows (frame lists must match per clip). Clip mean -> dataset mean / mean over all clips; paired differences with
ext_tables.py's bootstrap (people resampled within datasets, same draws for every row and metric).
Usage: python v16_table.py [--out O/reports/v16]"""
import argparse, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--O", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
ap.add_argument("--boot", type=int, default=10000); ap.add_argument("--seed", type=int, default=20261109); ap.add_argument("--out", default="")
a = ap.parse_args(); O = Path(a.O) / "eval"; OUT = Path(a.out) if a.out else Path(a.O) / "reports" / "v16"
ROOTS = [("avatar_test_v1", "avatar_test_v1.json", {"ubody"}), ("publicbench_v1", "publicbench_v1.json", None),
         ("publicbench_v2", "publicbench_v2_xhumans.json", None), ("finalbench_ext", "finalbench_ext.json", None)]
ROWS = ["lhmpp", "lhmpp_v16", "fb_gt", "fb_update_v16", "fb_update100"]
NAME = {"lhmpp": "LHM++, 1 image", "lhmpp_v16": "LHM++, 16 images", "fb_gt": "RAGA, photo", "fb_update_v16": "RAGA, photo + update, same 16 images",
        "fb_update100": "RAGA, photo + update, all reconstruction frames"}
PAIRS = [("lhmpp_v16", "lhmpp"), ("fb_update_v16", "fb_gt"), ("fb_update_v16", "lhmpp_v16"), ("fb_update100", "fb_update_v16"), ("fb_gt", "lhmpp_v16")]
DS = ["ubody", "peoplesnapshot", "neuman", "xhumans", "ubc"]
DSN = {"ubody": "UBody", "peoplesnapshot": "People-Snapshot", "neuman": "NeuMan", "xhumans": "X-Humans", "ubc": "UBC-Fashion"}
MET = ("psnr", "ssim", "lpips"); DEC = {"psnr": 2, "ssim": 3, "lpips": 3}

pool = {}
for E, sj, only in ROOTS:
    for c in json.loads((HERE / sj).read_text())["clips"]:
        if c["dataset"] not in DS or (only and c["dataset"] not in only): continue
        ds = c["dataset"]; name = c["clip"]
        person = (name.rsplit("-", 1)[0] if ds == "peoplesnapshot" else c.get("person") or c["id"]) if ds != "ubc" else f"ubc:{c['person']}"
        pool[c["id"]] = {"E": E, "ds": ds, "clip": name, "person": person}
S = {m: {} for m in ROWS}
for m in ROWS:
    for E in {v["E"] for v in pool.values()}:
        f = O / E / "scores" / f"{m}_full.json"
        if f.exists():
            for cid, r in json.loads(f.read_text())["per_clip"].items():
                if cid in pool and pool[cid]["E"] == E: S[m][cid] = r
# frame lists of the two 16-frame rows must be identical per clip
mismatch, nref = [], {}
for cid, v in pool.items():
    rj = [O / v["E"] / "renders" / m / v["ds"] / v["clip"] / "run.json" for m in ("lhmpp_v16", "fb_update_v16")]
    if not all(p.exists() for p in rj): continue
    ka, kb = (json.loads(p.read_text())["ref_keys"] for p in rj); nref[cid] = len(ka)
    if list(ka) != list(kb): mismatch.append(cid)
missing = {m: sorted(c for c in pool if c not in S[m]) for m in ROWS}
clips = sorted(c for c in pool if all(c in S[m] for m in ROWS) and c not in mismatch)
ids = {d: [c for c in clips if pool[c]["ds"] == d] for d in DS}; DSP = [d for d in DS if ids[d]]
rng = np.random.default_rng(a.seed); W = {}
for d in DSP:
    cs = ids[d]; people = sorted({pool[c]["person"] for c in cs}); pc = {p: [i for i, c in enumerate(cs) if pool[c]["person"] == p] for p in people}
    draw = rng.integers(0, len(people), size=(a.boot, len(people))); w = np.zeros((a.boot, len(cs)))
    for j, p in enumerate(people):
        cnt = (draw == j).sum(1)
        for i in pc[p]: w[:, i] += cnt
    W[d] = w


def stats(vals):
    out, bnum, bden = {}, 0, 0
    for d in DSP:
        x = np.array([vals[c] for c in ids[d]]); bd = (W[d] @ x) / np.maximum(W[d].sum(1), 1e-9)
        out[d] = [float(x.mean()), *map(float, np.percentile(bd, [2.5, 97.5]))]; bnum = bnum + W[d] @ x; bden = bden + W[d].sum(1)
    xa = np.array([vals[c] for c in clips]); out["all"] = [float(xa.mean()), *map(float, np.percentile(bnum / bden, [2.5, 97.5]))]
    return out


R = {m: {k: stats({c: S[m][c][k] for c in clips}) for k in MET} for m in ROWS}
D = {f"{m} - {r}": {k: stats({c: S[m][c][k] - S[r][c][k] for c in clips}) for k in MET} for m, r in PAIRS}
better = {f"{m} - {r}": {k: int(sum((S[m][c][k] - S[r][c][k]) * (-1 if k == "lpips" else 1) > 0 for c in clips)) for k in MET} for m, r in PAIRS}
COLS = DSP + ["all"]; CN = {**DSN, "all": "All"}
L = [f"# LHM++ with 16 images and RAGA's update on the same 16 images ({len(clips)} of {len(pool)} clips; written {time.strftime('%F %T')})", "",
     "Mean over a clip's test frames, then over clips. PSNR / SSIM / LPIPS. Images per clip (source included): "
     f"min {min(nref.values()) if nref else 0}, max {max(nref.values()) if nref else 0}; clips with fewer than 16: {sum(v < 16 for v in nref.values())}.", "",
     "| Method | " + " | ".join(f"{CN[c]} ({len(ids[c]) if c != 'all' else len(clips)})" for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
for m in ROWS: L.append(f"| {NAME[m]} | " + " | ".join(" / ".join(f"{R[m][k][c][0]:.{DEC[k]}f}" for k in MET) for c in COLS) + " |")
L += ["", "Paired differences, mean [95 % bootstrap interval] (clips better), all clips, then PSNR per dataset:", ""]
for (m, r) in PAIRS:
    k0 = f"{m} - {r}"; st = D[k0]
    L.append(f"- {NAME[m]} - {NAME[r]}: " + "; ".join(f"{k.upper()} {st[k]['all'][0]:+.{DEC[k] + 1}f} [{st[k]['all'][1]:+.{DEC[k] + 1}f}, {st[k]['all'][2]:+.{DEC[k] + 1}f}], {better[k0][k]}/{len(clips)}" for k in MET)
             + ". PSNR per dataset: " + ", ".join(f"{DSN[d]} {st['psnr'][d][0]:+.2f} [{st['psnr'][d][1]:+.2f}, {st['psnr'][d][2]:+.2f}]" for d in DSP))
L += ["", "Missing score rows: " + ("; ".join(f"{m}: {len(v)}" for m, v in missing.items() if v) or "none") + f". Frame lists differ: {len(mismatch)} clips {mismatch[:10]}."]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "v16_table.md").write_text("\n".join(L) + "\n")
(OUT / "v16_table.json").write_text(json.dumps({"clips": clips, "n_pool": len(pool), "rows": R, "paired": D, "clips_better": better, "missing": missing,
                                                "frame_list_mismatch": mismatch, "n_images": nref, "boot": a.boot, "seed": a.seed}, indent=1))
print("\n".join(L))
