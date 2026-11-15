"""Pose table in the format of PEAR's Table 3 (RESULTS_POSE.md) -> table_pose.{tex,md,pdf,png}.
Measured rows are read from the stored evaluation summaries; † rows are copied from PEAR's paper; the AGORA test-server value
(RUNLOG 149) and the L40S speed (RUNLOG 148) are the logged results. Best value per column in bold. CPU only."""
import argparse, json
from pathlib import Path

EP = Path("/raid/ubx858/outputs/dnt/eval_programme"); PE = Path("/raid/ubx858/outputs/phase2/author/pear")
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports")
ALL_COLS = ["coco05", "coco10", "lsp05", "lsp10", "pt05", "pt10", "mpjpe", "pa", "agora"]
COLS = list(ALL_COLS)                                                   # set by --drop
GROUPS = [("COCO", ["coco05", "coco10"]), ("LSP-Extended", ["lsp05", "lsp10"]), ("PoseTrack", ["pt05", "pt10"]), ("3DPW", ["mpjpe", "pa"]),
          ("AGORA", ["agora"])]
SUB = {"coco05": "@0.05", "coco10": "@0.1", "lsp05": "@0.05", "lsp10": "@0.1", "pt05": "@0.05", "pt10": "@0.1", "mpjpe": "MPJPE", "pa": "PA-MPJPE",
       "agora": "Body MVE"}
MDH = {"coco": "COCO", "lsp": "LSP-Ext", "pt": "PoseTrack", "mpjpe": "3DPW", "pa": "3DPW", "agora": "AGORA"}
CW = {"mpjpe": 0.40, "pa": 0.52, "agora": 0.48}                            # figure column widths (in), PCK columns 0.34


def groups(): return [(g, [k for k in ks if k in COLS]) for g, ks in GROUPS if any(k in COLS for k in ks)]


def arrow(k): return "↓" if k in LOWER else "↑"
LOWER = {"mpjpe", "pa", "agora"}


def measured(root, cam_root=None):
    j = lambda p: json.loads(p.read_text())
    c, l = j(root / "COCO-VAL/summary.json")["pck"], j(root / "LSP-EXTENDED/summary.json")["pck"]
    w = j((cam_root or root) / "3DPW-TEST/summary.json")
    return {"coco05": c["0.05"], "coco10": c["0.1"], "lsp05": l["0.05"], "lsp10": l["0.1"], "mpjpe": w["mpjpe_mm"], "pa": w["pa_mpjpe_mm"]}


FPS_SOURCE = "scratch"                                                   # set by --fps-source


def rows():
    if FPS_SOURCE == "harness":                                          # phase3_dnt/timing/pose_fps.py on the idle L40S (RUNLOG 167)
        h = json.loads((O / "pose_table3/fps_harness/pose_fps_nvidia_l40s.json").read_text())["models"]
        fps, fps_pear = h["sprint"]["network_fp32"]["fps"], h["pear"]["network_fp32"]["fps"]
    else:                                                                # scratch measurement of RUNLOG 148
        fps, fps_pear = json.loads((O / "pose_table3/fps_sprint_L40S.json").read_text())["network_fp32"]["fps"], None
    paper = [("OSX$^\\dagger$", "OSX †", "SMPL-X", 20, [0.70, 0.87, 0.42, 0.73, 0.82, 0.90, 74.7, 45.1, 80.2]),
             ("Multi-HMR$^\\dagger$", "Multi-HMR †", "SMPL-X", 10, [0.65, 0.82, 0.43, 0.70, None, None, 78.0, 45.9, 59.6]),
             ("SMPLest-X$^\\dagger$", "SMPLest-X †", "SMPL-X", 20, [0.71, 0.91, 0.40, 0.74, 0.82, 0.91, 74.8, 45.8, 63.5]),
             ("PEAR$^\\dagger$", "PEAR †", "EHM-s", 100, [0.81, 0.94, 0.55, 0.81, 0.87, 0.97, 71.3, 45.3, 59.3])]
    out = [{"tex": t, "txt": n, "model": m, "fps": f, **dict(zip(ALL_COLS, v)), "key": k}
           for (t, n, m, f, v), k in zip(paper, ("osx", "multihmr", "smplestx", "pear"))]
    pear = measured(PE); ours = measured(EP / "sprint_scratch_camwidth/author")
    cal = measured(EP / "sprint_scratch_camwidth/author", EP / "sprint_scratch_camknown/author")
    out.append({"tex": "PEAR (released, measured here)", "txt": "PEAR (released, measured here)", "model": "EHM-s", "fps": None if fps_pear is None else round(fps_pear),
                **pear, "pt05": None, "pt10": None, "agora": None, "key": "pear_rerun"})
    out.append({"tex": "\\textbf{Ours}", "txt": "Ours", "model": "SMPL-X", "fps": round(fps), **ours, "pt05": None, "pt10": None,
                "agora": 126.1, "ours": True, "key": "ours"})                                              # AGORA test server, RUNLOG 149
    out.append({"tex": "Ours, calibrated camera$^\\ddagger$", "txt": "Ours, calibrated camera ‡", "model": "SMPL-X", "fps": round(fps),
                **cal, "pt05": None, "pt10": None, "agora": None, "na_agora": True, "key": "ours_cal"})
    return out


def fmt(k, v):
    return "–" if v is None else (f"{v:.2f}" if k.startswith(("coco", "lsp", "pt")) else (f"{v:.1f}" if k != "fps" else f"{v:d}"))


def best(R):
    b = {}
    for k in COLS + ["fps"]:
        vals = [float(fmt(k, r[k])) for r in R if r[k] is not None]
        b[k] = (min if k in LOWER else max)(vals)
    return b


def notes():
    return ("† Copied from PEAR's paper (Table 3), not re-run here. ‡ 3DPW's calibrated intrinsics as camera input (the other rows "
            "use none); COCO and LSP-Extended have no calibration, so those columns equal the main row. Measured rows: our port of "
            "the PEAR author's evaluation script (reproduces PEAR's row within about 2 PCK points and 2.5 mm). "
            + ("AGORA: test server, no camera input. " if "agora" in COLS else "")
            + "FPS: ours on an idle L40S, batch 1, network only (with the SMPL-X layer 91); others as reported by PEAR. "
            + ("PoseTrack: validation images not available. " if "pt05" in COLS else "") + "Ours is never trained on 3DPW.")


def caption_head():
    pck = [g for g, ks in groups() if ks[0].endswith("05")]
    s = "Human pose and shape. PCK@0.05 / @0.1 on " + (", ".join(pck[:-1]) + " and " + pck[-1] if len(pck) > 1 else pck[0])
    s += "; MPJPE and PA-MPJPE (mm) on 3DPW"
    return s + ("; Body MVE (mm) on AGORA. " if "agora" in COLS else ". ")


def tex(R, B):
    L = ["% Generated by phase3_dnt/pose_table_render.py (RESULTS_POSE.md); needs booktabs, graphicx", "\\begin{table*}[t]", "\\centering",
         "\\caption{" + caption_head() + "Best per column in bold. " + notes().replace("†", "$^\\dagger$").replace("‡", "$^\\ddagger$") + "}",
         "\\label{tab:pose}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{l|l|l|r|" + "|".join("c" * len(ks) for _, ks in groups()) + "}", "\\toprule",
         "Method & Model & Predict parts & FPS & " + " & ".join(
             (f"\\multicolumn{{{len(ks)}}}{{c|}}{{{g}}}" if len(ks) > 1 and n < len(groups()) - 1 else
              f"\\multicolumn{{{len(ks)}}}{{c}}{{{g}}}" if len(ks) > 1 else g) for n, (g, ks) in enumerate(groups())) + " \\\\",
         " & & & & " + " & ".join(SUB[k] + ("$\\downarrow$" if k in LOWER else "$\\uparrow$") for k in COLS) + " \\\\", "\\midrule"]
    for r in R:
        if r.get("ours"): L.append("\\midrule")
        cells = [r["tex"], r["model"], "Body, face, hand"]
        for k in ["fps"] + COLS:
            s = "n/a" if (k == "agora" and r.get("na_agora")) else fmt(k, r[k])
            cells.append(f"\\textbf{{{s}}}" if r[k] is not None and float(s) == B[k] else s)
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(L) + "\n"


STYLE = "notes"                                                          # set by --style
RANK = False                                                             # set by --rank: shade best / 2nd / 3rd per column
RANK_RGB = {1: (0.64, 0.77, 0.88), 2: (0.80, 0.87, 0.94), 3: (0.92, 0.95, 0.97)}   # one hue (cvprblue mixed with white 45/25/10 %)


def ranks(R):
    """{(row, column): 1 | 2 | 3}: dense rank of the displayed value among rows without extra camera input (the calibrated
    row is not ranked); ties share a rank."""
    out = {}
    for k in ["fps"] + COLS:
        vals = sorted({float(fmt(k, r[k])) for r in R if r[k] is not None and r["key"] != "ours_cal"}, reverse=k not in LOWER)
        for i, r in enumerate(R):
            if r[k] is not None and r["key"] != "ours_cal" and float(fmt(k, r[k])) in vals[:3]:
                out[(i, k)] = vals.index(float(fmt(k, r[k]))) + 1
    return out


def pear_fn():
    """footnotes of the PEAR-style table: [(key, text)], numbered in order."""
    fn = []
    if "pt05" in COLS: fn.append(("multihmr", "Multi-HMR always detects additional persons in the PoseTrack dataset; therefore, its "
                                              "PoseTrack results are not reported (as in PEAR)."))
    if FPS_SOURCE == "harness":
        fn.append(("fps", "L40S. Rows above the line as reported by PEAR; rows below measured with one harness: batch 1, FP32, "
                          "network only, median over 3 rounds of 400 passes (with the body model: ours 87, PEAR 51 FPS)."))
    else:
        fn.append(("fps", "Tested on L40S; ours: network only, batch 1 (91 FPS with the SMPL-X layer)."))
    fn.append(("cal", "3DPW's calibrated intrinsics as extra input; COCO and LSP-Extended have no calibration, so those columns "
                      "equal the row above."))
    return fn


def fn_no(key): return 1 + [k for k, _ in pear_fn()].index(key)


def pear_caption(tex_=True):
    pck = [g for g, ks in groups() if ks[0].endswith("05")]; d3 = [g for g, ks in groups() if not ks[0].endswith("05")]
    met = "MPJPE, PA-MPJPE and Body MVE" if "agora" in COLS else "MPJPE and PA-MPJPE"
    j = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]
    cite = "~\\cite{pear}" if tex_ else ""
    s = (f"Quantitative comparison of human body pose. All methods estimate body, face and hand parameters. We report PCK@0.05 "
         f"and @0.1 for the 2D dataset{'s' if len(pck) > 1 else ''} ({j(pck)}) and {met} (mm) for the 3D dataset{'s' if len(d3) > 1 else ''} ({j(d3)}). Rows above the line are copied "
         f"from PEAR{cite}; rows below are measured with our port of PEAR's evaluation script (it reproduces PEAR's published row "
         f"within about 2 PCK points and 2.5 mm), with focal length equal to image width unless stated. ")
    if "pt05" in COLS: s += "PoseTrack validation images were not available to us. "
    if "agora" in COLS: s += "AGORA: evaluation server. "
    s += "Ours is never trained on 3DPW. " + ("Shading marks the best (darkest, also bold), second and third value per column; the "
                                              "calibrated row, which uses extra camera input, is not ranked." if RANK else "Best per column in bold.")
    sup = (lambda n: f" $^{n}$ ") if tex_ else (lambda n: " " + "¹²³⁴"[n - 1] + " ")
    return s + "".join(sup(n) + t for n, (_, t) in enumerate(pear_fn(), 1))


def pear_label(r, tex_=True):
    sup = (lambda n: f"$^{n}$") if tex_ else (lambda n: "¹²³⁴"[n - 1])
    c = (lambda k: f"~\\cite{{{k}}}") if tex_ else (lambda k: "")
    k = r["key"]
    return {"osx": "OSX" + c("ubody"), "smplestx": "SMPLest-X" + c("smplestx"), "pear": "PEAR" + c("pear"),
            "multihmr": "Multi-HMR" + (sup(fn_no("multihmr")) if "pt05" in COLS else "") + c("multihmr"),
            "pear_rerun": "PEAR, rerun", "ours": "Ours", "ours_cal": "Ours, calibrated" + sup(fn_no("cal"))}[k]


def tex_pear(R, B):
    lift = lambda t: f"\\raisebox{{-1.4ex}}[0pt][0pt]{{{t}}}"
    G = groups(); L = ["% Generated by phase3_dnt/pose_table_render.py --style pear (layout of PEAR's Table 3); needs booktabs and graphicx "
                       "(both loaded by cvpr.sty) and bib keys ubody (OSX paper), multihmr, smplestx, pear",
                       "\\begin{table*}[t]", "\\centering", "\\caption{" + pear_caption() + "}", "\\label{tab:pose}"]
    if RANK:
        L[0] += "; shading needs \\usepackage{colortbl}"
        L += [f"\\definecolor{{{n}}}{{rgb}}{{{','.join(f'{c:.2f}' for c in RANK_RGB[i])}}}" for i, n in ((1, "rankfirst"), (2, "ranksecond"), (3, "rankthird"))]
    L += [
                       "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{l|c|c|c|" + "|".join("c" * len(ks) for _, ks in G) + "}", "\\toprule"]
    h1 = [lift("Methods"), lift("Model"), lift("Predict parts"), lift(f"FPS$^{fn_no('fps')}$")]
    rules, c0 = [], 5
    for n, (g, ks) in enumerate(G):
        h1.append(f"\\multicolumn{{{len(ks)}}}{{c{'' if n == len(G) - 1 else '|'}}}{{{g}}}"); rules.append(f"\\cmidrule(lr){{{c0}-{c0 + len(ks) - 1}}}"); c0 += len(ks)
    L += [" & ".join(h1) + " \\\\", "".join(rules),
          " & & & & " + " & ".join(SUB[k] + ("$\\downarrow$" if k in LOWER else "$\\uparrow$") for k in COLS) + " \\\\", "\\midrule"]
    RK = ranks(R) if RANK else {}
    for i, r in enumerate(R):
        if r["key"] == "pear_rerun": L.append("\\midrule")
        cells = [pear_label(r), r["model"], "Body, face, hand"]
        for k in ["fps"] + COLS:
            v = "n/a" if (k == "agora" and r.get("na_agora")) else fmt(k, r[k])
            if RANK:
                rk = RK.get((i, k)); cells.append({1: f"\\cellcolor{{rankfirst}}\\textbf{{{v}}}", 2: f"\\cellcolor{{ranksecond}}{v}",
                                                   3: f"\\cellcolor{{rankthird}}{v}"}.get(rk, v.replace("–", "--")))
            else:
                cells.append(f"\\textbf{{{v}}}" if r[k] is not None and float(v) == B[k] else v.replace("–", "--"))
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def md(R, B):
    h = "| Method | Model | Predict parts | FPS | " + " | ".join(f"{MDH[k.rstrip('015')]} {SUB[k]} {arrow(k)}" for k in COLS) + " |"
    L = [h, "|" + "---|" * (4 + len(COLS))]
    for r in R:
        c = [pear_label(r, False) if STYLE == "pear" else r["txt"], r["model"], "Body, face, hand"]
        for k in ["fps"] + COLS:
            s = "n/a" if (k == "agora" and r.get("na_agora")) else fmt(k, r[k])
            if RANK and STYLE == "pear":
                rk = ranks(R).get((R.index(r), k)); c.append({1: f"**{s}** (1st)", 2: f"{s} (2nd)", 3: f"{s} (3rd)"}.get(rk, s))
            else:
                c.append(f"**{s}**" if r[k] is not None and float(s) == B[k] else s)
        L.append("| " + " | ".join(c) + " |")
    return "\n".join(L) + "\n\n" + (pear_caption(False) if STYLE == "pear" else notes()) + "\n"


def render(R, B, out, name="table_pose"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, textwrap
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    W = 6.88; widths = [1.32, 0.42, 0.68, 0.30] + [CW.get(k, 0.34) for k in COLS]
    sc = W / sum(widths); widths = [w * sc for w in widths]; xs = [sum(widths[:i]) for i in range(len(widths) + 1)]
    pear = STYLE == "pear"; rh = 0.17; RK = ranks(R) if RANK else {}
    notes = textwrap.wrap(pear_caption(False) if pear else globals()["notes"](), 175)
    H = rh * (len(R) + 2) + 0.16 + 0.12 * len(notes) + 0.12 + (0.04 if pear else 0)
    f = plt.figure(figsize=(W, H)); fs = 6.6
    Y = lambda y: 1 - y / H
    def hline(y, lw, x0=0, x1=W): f.add_artist(plt.Line2D([x0 / W, x1 / W], [Y(y)] * 2, transform=f.transFigure, color="k", lw=lw))
    def txt(x, y, s, ha="center", bold=False): f.text(x / W, Y(y), s, ha=ha, va="center", fontsize=fs, fontweight="bold" if bold else "normal")
    top = 0.04 + (0.12 * len(notes) + 0.10 if pear else 0); hline(top, 0.9)
    if pear:
        for n, line in enumerate(notes): f.text(0.0, Y(0.08 + 0.12 * n), line, ha="left", va="center", fontsize=5.9, color="0.15")
    spans, c0 = [], 4
    for g, ks in groups(): spans.append((c0, c0 + len(ks), g)); c0 += len(ks)
    for a_, b_, g in spans:
        txt((xs[a_] + xs[b_]) / 2, top + rh / 2, g)
        if b_ - a_ > 1: hline(top + rh, 0.4, xs[a_] + 0.03, xs[b_] - 0.03)
    for i, s in enumerate(["Methods", "Model", "Predict parts", "FPS" + "¹²³⁴"[fn_no("fps") - 1]] if pear else ["Method", "Model", "Predict parts", "FPS"]):
        txt(xs[i] + (0.03 if i == 0 else widths[i] / 2), top + rh, s, ha="left" if i == 0 else "center")
    for i, s in enumerate(SUB[k] + arrow(k) for k in COLS):
        txt(xs[4 + i] + widths[4 + i] / 2, top + 1.5 * rh, s)
    y = top + 2 * rh + 0.02; hline(y, 0.5)
    for r in R:
        if (r["key"] == "pear_rerun") if pear else r.get("ours"): hline(y, 0.35)
        yc = y + rh / 2
        txt(0.03, yc, pear_label(r, False) if pear else r["txt"], ha="left", bold=bool(r.get("ours")) and not pear); txt(xs[1] + widths[1] / 2, yc, r["model"]); txt(xs[2] + widths[2] / 2, yc, "Body, face, hand")
        for i, k in enumerate(["fps"] + COLS):
            s = "n/a" if (k == "agora" and r.get("na_agora")) else fmt(k, r[k])
            if RANK and pear:
                rk = RK.get((R.index(r), k))
                if rk: f.add_artist(plt.Rectangle((xs[3 + i] / W, Y(y + rh)), widths[3 + i] / W, rh / H, transform=f.transFigure,
                                                  color=RANK_RGB[rk], lw=0, zorder=-1))
                txt(xs[3 + i] + widths[3 + i] / 2, yc, s, bold=rk == 1)
            else:
                txt(xs[3 + i] + widths[3 + i] / 2, yc, s, bold=r[k] is not None and float(s) == B[k])
        y += rh
    hline(y, 0.9)
    for c in [1, 2, 3] + [a_ for a_, _, _ in spans]:                                     # light column separators
        f.add_artist(plt.Line2D([xs[c] / W] * 2, [Y(top), Y(y)], transform=f.transFigure, color="0.6", lw=0.3))
    if not pear:
        for n, line in enumerate(notes): f.text(0.0, Y(y + 0.12 + 0.12 * n), line, ha="left", va="center", fontsize=5.6, color="0.25")
    f.savefig(out / f"{name}.pdf"); f.savefig(out / f"{name}.png", dpi=300); plt.close(f)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", type=Path, default=O / "pose_qual" / "table")
    p.add_argument("--drop", nargs="*", default=[], choices=["posetrack", "agora", "lsp"], help="leave these datasets out")
    p.add_argument("--name", default="table_pose", help="output file stem")
    p.add_argument("--style", default="notes", choices=["notes", "pear"], help="pear: the layout of PEAR's Table 3 (caption above, numbered footnotes, citations)")
    p.add_argument("--rank", action="store_true", help="--style pear: shade the best / second / third value per column")
    p.add_argument("--fps-source", default="scratch", choices=["scratch", "harness"], help="harness: phase3_dnt/timing/pose_fps.py L40S result")
    a = p.parse_args(); global STYLE, RANK, FPS_SOURCE; STYLE, RANK, FPS_SOURCE = a.style, a.rank, a.fps_source
    drop = {"posetrack": ("pt05", "pt10"), "agora": ("agora",), "lsp": ("lsp05", "lsp10")}
    COLS[:] = [k for k in ALL_COLS if not any(k in drop[d] for d in a.drop)]
    a.out.mkdir(parents=True, exist_ok=True); R = rows(); B = best(R)
    (a.out / f"{a.name}.tex").write_text(tex_pear(R, B) if STYLE == "pear" else tex(R, B)); (a.out / f"{a.name}.md").write_text(md(R, B)); render(R, B, a.out, a.name)
    print(md(R, B))


if __name__ == "__main__":
    main()
