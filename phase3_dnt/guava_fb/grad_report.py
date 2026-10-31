"""Amendment 6 (B): summary of guava_fb_train.py --grad-diag output. Every quantity is computed from the per-batch, per-group
Gram matrices of the term gradients (G raw, Gp Adam-preconditioned = the step's update): a term group's gradient is the sum
of its terms, dot products add over parameter groups. Per checkpoint (label=grad_diag dir):
  - per term: weighted gradient norm and share of the update <u_i, u_obj> / ||u_obj||^2 (u_obj = the pilot's own objective;
    shares over the objective's terms sum to 1), on the shared layers and per module;
  - per term group: update share; cosine image vs geometry (Sapiens, scan), raw vs refined image, image vs Gram; fraction of
    batches with cosine < 0 (only batches where both gradients are non-zero).
Shared layers = parameter groups where both the image and the geometry gradient are non-zero.
Usage: python grad_report.py --runs B=<dir> C=<dir> --out <dir>"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

CAT = {"image (raw)": ["raw_full", "raw_crops"], "image (refined)": ["ref_full", "ref_crops", "cloth", "arcface", "edge"], "silhouette": ["sil"],
       "regularisers": ["reg"], "geometry (Sapiens)": ["flat", "sap_normal", "sap_dnormal"],
       "geometry (scan)": ["scan_normal", "scan_dnormal", "scan_depth", "scan_chamfer"], "texture (Gram)": ["gram"]}
IMG = CAT["image (raw)"] + CAT["image (refined)"] + CAT["silhouette"]
GEO = CAT["geometry (Sapiens)"] + CAT["geometry (scan)"]
OBJ = {"B": IMG + CAT["regularisers"], "C": IMG + CAT["regularisers"] + CAT["geometry (Sapiens)"],
       "D": IMG + CAT["regularisers"] + GEO + CAT["texture (Gram)"]}
SAP_SRC = {"D": {"bedlam2", "motionx", "ubc"}}     # pilot D: flattening + Sapiens only on these sources' targets (diagnostic measures them everywhere)
PAIRS = [("image", IMG, "geometry (Sapiens)", CAT["geometry (Sapiens)"]), ("image", IMG, "geometry (scan)", CAT["geometry (scan)"]),
         ("image (raw)", CAT["image (raw)"], "image (refined)", CAT["image (refined)"]), ("image", IMG, "texture (Gram)", CAT["texture (Gram)"]),
         ("geometry (Sapiens)", CAT["geometry (Sapiens)"], "geometry (scan)", CAT["geometry (scan)"])]


def vec(terms, sel):
    return np.array([1.0 if t in sel else 0.0 for t in terms])


def load(d):
    recs = [json.loads(l) for l in open(Path(d) / "grad_diag.jsonl")]
    info = json.loads((Path(d) / "grad_diag_info.json").read_text())
    return recs, info


def module(g): return ".".join(g.split(".")[:2])


def objective(label, src):                         # the pilot's objective on a batch from this source
    o = OBJ[label[0]]
    if label[0] in SAP_SRC and src not in SAP_SRC[label[0]]: o = [t for t in o if t not in CAT["geometry (Sapiens)"]]
    return o


report, md = {}, []
for spec in a.runs:
    label, d = spec.split("=", 1); recs, info = load(d); obj = OBJ[label[0]]
    groups = sorted(recs[0]["groups"])
    # shared layers: image and geometry gradients both non-zero (in any batch)
    def nz(g, sel, key="G"):
        return any(float(vec(r["terms"], sel) @ np.array(r["groups"][g][key]) @ vec(r["terms"], sel)) > 0 for r in recs)
    shared = [g for g in groups if nz(g, IMG) and nz(g, GEO)]
    sets = {"shared layers": shared, "all layers": groups}
    for m in sorted({module(g) for g in groups}): sets[m] = [g for g in groups if module(g) == m]

    def gram(r, gs, key):
        return sum(np.array(r["groups"][g][key]) for g in gs)

    res = {"checkpoint": info["checkpoint"], "step": info["step"], "batches": len(recs), "shared_layers": shared,
           "shared_params": int(sum(info["groups"][g] for g in shared)), "objective": obj, "sets": {}}
    for sname, gs in sets.items():
        S = {}
        for key, kname in (("Gp", "update"), ("G", "gradient")):
            term_norm, term_share, cat_share, pair_cos = {}, {}, {}, {}
            for r in recs:
                Gm = gram(r, gs, key); T = r["terms"]; ob = objective(label, r["source"]); wo = vec(T, ob); uu = float(wo @ Gm @ wo)
                for i, t in enumerate(T): term_norm.setdefault(t, []).append(float(np.sqrt(max(Gm[i, i], 0))))   # norm: batches where the term is non-zero
                if uu > 0:                                                    # shares: every batch (0 where a term is absent / not in this batch's objective)
                    for t in obj: term_share.setdefault(t, []).append(float(Gm[T.index(t)] @ wo) / uu if t in T and t in ob else 0.0)
                    for c, sel in CAT.items():
                        if any(t in obj for t in sel): cat_share.setdefault(c, []).append(float(vec(T, [t for t in sel if t in ob]) @ Gm @ wo) / uu)
                for n1, s1, n2, s2 in PAIRS:
                    w1, w2 = vec(T, s1), vec(T, s2); a1, a2 = float(w1 @ Gm @ w1), float(w2 @ Gm @ w2)
                    if a1 > 0 and a2 > 0: pair_cos.setdefault(f"{n1} vs {n2}", []).append(float(w1 @ Gm @ w2) / np.sqrt(a1 * a2))
            S[kname] = {"term_norm": {t: float(np.mean(v)) for t, v in term_norm.items()},
                        "term_share": {t: float(np.mean(v)) for t, v in term_share.items()},
                        "category_share": {c: float(np.mean(v)) for c, v in cat_share.items()},
                        "cosine": {p: {"mean": float(np.mean(v)), "frac_negative": float(np.mean(np.array(v) < 0)), "n": len(v)} for p, v in pair_cos.items()}}
        res["sets"][sname] = S
    # per training source, shared layers, update space
    per_src = {}
    for s in sorted({r["source"] for r in recs}):
        rr = [r for r in recs if r["source"] == s]; d_ = {}
        for r in rr:
            Gm = gram(r, shared, "Gp"); T = r["terms"]; ob = objective(label, s); wo = vec(T, ob); uu = float(wo @ Gm @ wo)
            for c, sel in CAT.items():
                if uu > 0 and any(t in obj for t in sel): d_.setdefault("share " + c, []).append(float(vec(T, [t for t in sel if t in ob]) @ Gm @ wo) / uu)
            for n1, s1, n2, s2 in PAIRS[:3]:
                w1, w2 = vec(T, s1), vec(T, s2); a1, a2 = float(w1 @ Gm @ w1), float(w2 @ Gm @ w2)
                if a1 > 0 and a2 > 0: d_.setdefault(f"cos {n1} vs {n2}", []).append(float(w1 @ Gm @ w2) / np.sqrt(a1 * a2))
        per_src[s] = {k: float(np.mean(v)) for k, v in d_.items()}
    res["per_source_shared_update"] = per_src
    report[label] = res

(out / "grad_report.json").write_text(json.dumps(report, indent=1))
labels = list(report)
md.append("## Update share on the shared layers (Adam-preconditioned; each pilot's own objective; shares sum to 1)\n")
md.append("| term group | " + " | ".join(labels) + " |"); md.append("|---|" + "---|" * len(labels))
for c in CAT:
    md.append(f"| {c} | " + " | ".join(f"{report[l]['sets']['shared layers']['update']['category_share'].get(c, float('nan')):.3f}" for l in labels) + " |")
md.append("\n## Cosine between term groups (shared layers; mean over batches, fraction of batches < 0 in brackets)\n")
for space in ("update", "gradient"):
    md.append(f"\n{space} space\n"); md.append("| pair | " + " | ".join(labels) + " |"); md.append("|---|" + "---|" * len(labels))
    for n1, _, n2, _ in PAIRS:
        p = f"{n1} vs {n2}"; cells = []
        for l in labels:
            c = report[l]["sets"]["shared layers"][space]["cosine"].get(p)
            cells.append(f"{c['mean']:+.3f} ({c['frac_negative']:.2f}, n={c['n']})" if c else "-")
        md.append(f"| {p} | " + " | ".join(cells) + " |")
md.append("\n## Per term (shared layers): weighted gradient norm / update norm / update share\n")
md.append("| term | " + " | ".join(labels) + " |"); md.append("|---|" + "---|" * len(labels))
for t in sum(CAT.values(), []):
    cells = []
    for l in labels:
        S = report[l]["sets"]["shared layers"]
        if t in S["gradient"]["term_norm"]:
            sh = S["update"]["term_share"].get(t)
            cells.append(f"{S['gradient']['term_norm'][t]:.2e} / {S['update']['term_norm'][t]:.2e} / " + (f"{sh:+.3f}" if sh is not None else "(not in objective)"))
        else: cells.append("-")
    md.append(f"| {t} | " + " | ".join(cells) + " |")
md.append("\n## Per module (update space): share image / Sapiens geometry / regularisers; cosine image vs Sapiens geometry\n")
mods = [m for m in report[labels[0]]["sets"] if m not in ("shared layers", "all layers")]
md.append("| module | " + " | ".join(labels) + " |"); md.append("|---|" + "---|" * len(labels))
for m in mods:
    cells = []
    for l in labels:
        S = report[l]["sets"].get(m, {}).get("update")
        if not S: cells.append("-"); continue
        cs = S["category_share"]; img = sum(cs.get(c, 0) for c in ("image (raw)", "image (refined)", "silhouette"))
        cc = S["cosine"].get("image vs geometry (Sapiens)")
        cells.append(f"{img:.2f} / {cs.get('geometry (Sapiens)', float('nan')):.2f} / {cs.get('regularisers', 0):.2f}; cos " + (f"{cc['mean']:+.3f}" if cc else "-"))
    md.append(f"| {m} | " + " | ".join(cells) + " |")
md.append("\n## Per training source (shared layers, update space)\n")
for l in labels:
    md.append(f"\n{l}\n"); ks = sorted({k for v in report[l]["per_source_shared_update"].values() for k in v})
    md.append("| source | " + " | ".join(ks) + " |"); md.append("|---|" + "---|" * len(ks))
    for s, v in report[l]["per_source_shared_update"].items(): md.append(f"| {s} | " + " | ".join(f"{v.get(k, float('nan')):+.3f}" for k in ks) + " |")
(out / "grad_report.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
