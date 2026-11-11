"""Editable PowerPoint version of the architecture figure: one slide built from fig_architecture.py --export (layout.json +
thumbnails). Boxes, bands, trapezoids, circles and the schematic Gaussians are native shapes; arrows are lines with arrowheads;
every label is its own text box; thumbnails are pictures. The slide is `--scale` x the print size (default 2: 13.75 x 7.96 in) so
that fonts are editable sizes; export it to PDF from PowerPoint and include it at \\textwidth. Fonts: Times New Roman (the PDF
uses STIX), symbols in Segoe UI Symbol, the calligraphic G in Cambria Math.
Needs python-pptx (not in the guava env: pip install --target /dev/shm/ubx858_pylib python-pptx XlsxWriter lxml typing_extensions).
Usage: PYTHONPATH=/dev/shm/ubx858_pylib python fig_architecture_pptx.py --export <dir> --out fig_architecture.pptx"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ap = argparse.ArgumentParser(); ap.add_argument("--export", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=2.0); ap.add_argument("--font", default="Times New Roman"); a = ap.parse_args()
EXP = Path(a.export); L = json.loads((EXP / "layout.json").read_text()); W, H, SC = L["W"], L["H"], a.scale
prs = Presentation(); prs.slide_width = Inches(W * SC); prs.slide_height = Inches(H * SC)
slide = prs.slides.add_slide(prs.slide_layouts[6]); sh = slide.shapes
prs.core_properties.title = "Architecture figure (editable)"; prs.core_properties.author = ""


def X(x): return int(Inches(x * SC))
def Y(y): return int(Inches((H - y) * SC))
def D(d): return int(Inches(d * SC))
def rgb(c): c = {"white": "#FFFFFF", "black": "#000000"}.get(c, c); return RGBColor(int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))


def style(shp, fc, ec, lw):
    shp.shadow.inherit = False
    if fc in (None, "none"): shp.fill.background()
    else: shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fc)
    if not lw or ec in (None, "none"): shp.line.fill.background()
    else: shp.line.color.rgb = rgb(ec); shp.line.width = Pt(lw * SC)


def arrowhead(shp):
    ln = shp.line._get_or_add_ln(); ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


MATH = [(r"\mathcal{G}", "\U0001D4A2"), (r"\hat{I}", "Î"), (r"\bar{U}", "Ū"), (r"\theta", "θ"), (r"\Sigma", "Σ"), (r"\approx", "≈"), (r"\rightarrow", "→"), (r"\,", " ")]


def runs(s):
    """mathtext string -> [(text, italic, subscript, font)] ; $...$ segments are converted, the rest is plain."""
    out = []
    for i, seg in enumerate(s.split("$")):
        if i % 2 == 0:
            if seg: out.append((seg, None, False, None))
            continue
        for k, v in MATH: seg = seg.replace(k, v)
        j = 0
        while j < len(seg):
            sub = seg[j] == "_"
            if sub:
                j += 1; tok = seg[j + 1: seg.index("}", j)] if seg[j] == "{" else seg[j]; j += len(tok) + (2 if seg[j] == "{" else 0)
            else: tok = seg[j]; j += 1
            up = tok in ("Σ", "≈", "→", " ") or tok.isdigit()
            out.append((tok, not up, sub, "Cambria Math" if tok == "\U0001D4A2" else None))
    return out


def text(p):
    lines = p["s"].split("\n"); size = p["size"] * SC; plain = [re.sub(r"\\[a-z]+|[${}_]", "", ln) for ln in lines]
    bw = max(0.25 * SC, max(len(x) for x in plain) * size * 0.55 / 72 + 0.06 * SC); bh = len(lines) * size * 1.2 / 72 + 0.02 * SC
    rot = int(p.get("rot") or 0); cx, cy = Inches(p["x"] * SC), Inches((H - p["y"]) * SC); bwE, bhE = int(Inches(bw)), int(Inches(bh))
    left = cx - bwE // 2 if (p["ha"] == "center" or rot) else (cx if p["ha"] == "left" else cx - bwE)
    top = cy - bhE // 2 if (p["va"] == "center" or rot) else (cy if p["va"] == "top" else cy - bhE)
    tb = sh.add_textbox(int(left), int(top), bwE, bhE); tf = tb.text_frame; tf.word_wrap = False; tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "center": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[p["va"]]
    if rot: tb.rotation = (-rot) % 360
    for li, ln in enumerate(lines):
        par = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
        par.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}["center" if rot else p["ha"]]
        for tok, ital, sub, font in runs(ln):
            r = par.add_run(); r.text = tok; f = r.font; f.size = Pt(size); f.bold = bool(p["bold"]); f.color.rgb = rgb(p["color"])
            f.italic = bool(p["italic"]) if ital is None else ital
            f.name = font or ("Segoe UI Symbol" if p.get("family") else a.font)
            if sub: r._r.get_or_add_rPr().set("baseline", "-25000")
    tb.name = "label: " + plain[0][:40]


n = {"rect": 0, "poly": 0, "line": 0, "ell": 0, "img": 0, "text": 0}
for p in sorted(L["prims"], key=lambda q: q["z"]):                                                   # stable: drawing order within a layer
    t = p["t"]; n[t] += 1
    if t == "rect":
        s_ = sh.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, X(p["x"]), Y(p["y"] + p["h"]), D(p["w"]), D(p["h"]))
        s_.adjustments[0] = min(0.5, p["r"] / min(p["w"], p["h"])); style(s_, p["fc"], p["ec"], p["lw"]); s_.name = f"box {n[t]}"
    elif t == "poly":
        pts = [(X(x), Y(y)) for x, y in p["pts"]]; fb = sh.build_freeform(*pts[0]); fb.add_line_segments(pts[1:], close=True)
        s_ = fb.convert_to_shape(); style(s_, p["fc"], p["ec"], p["lw"]); s_.name = f"shape {n[t]}"
    elif t == "line":
        pts = [(X(x), Y(y)) for x, y in p["pts"]]
        if len(pts) == 2: s_ = sh.add_connector(MSO_CONNECTOR.STRAIGHT, *pts[0], *pts[1]); s_.shadow.inherit = False
        else:
            fb = sh.build_freeform(*pts[0]); fb.add_line_segments(pts[1:], close=False); s_ = fb.convert_to_shape(); s_.fill.background(); s_.shadow.inherit = False
        s_.line.color.rgb = rgb(p["color"]); s_.line.width = Pt(p["lw"] * SC); s_.name = ("arrow " if p["arrow"] else "line ") + str(n[t])
        if p["arrow"]: arrowhead(s_)
    elif t == "ell":
        s_ = sh.add_shape(MSO_SHAPE.OVAL, X(p["cx"] - p["w"] / 2), Y(p["cy"] + p["h"] / 2), D(p["w"]), D(p["h"]))
        s_.rotation = (-p["angle"]) % 360; style(s_, p["fc"], p["ec"], p["lw"]); s_.name = f"ellipse {n[t]}"
    elif t == "img":
        s_ = sh.add_picture(str(EXP / p["file"]), X(p["x"]), Y(p["y"] + p["h"]), D(p["w"]), D(p["h"]))
        s_.line.color.rgb = rgb(p["ec"]); s_.line.width = Pt(p["lw"] * SC); s_.name = f"thumbnail {n[t]}"
    elif t == "text": text(p)
prs.save(a.out); print("wrote", a.out, f"{W * SC:.2f} x {H * SC:.2f} in", n)
