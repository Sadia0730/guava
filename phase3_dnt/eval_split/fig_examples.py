"""Fixed-example comparison figures for the GUAVA-FB improvement phase (validation only, ValBench). Rows are chosen by rules fixed in
this file, never by eye: per dataset the clip with the MEDIAN per-clip PSNR change of --pair (typical), the clip with the LARGEST
drop or smallest gain (worst; the regression case), plus any --extra clips (e.g. the H1c wrong-person clips). Frame = the middle test
frame of the clip. Each tile is labelled with the method's per-clip PSNR / LPIPS from E/scores/<method>_<tag>.json.
Usage: python fig_examples.py --E <root> --cols GT,fb_sprint,fb_sprint_k60 --pair fb_sprint,fb_sprint_k60 [--extra id,id] --out fig.jpg"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(); ap.add_argument("--E", required=True); ap.add_argument("--cols", required=True, help="GT or method[:label], ...")
ap.add_argument("--pair", required=True, help="control,arm: per-clip PSNR change used to pick the rows"); ap.add_argument("--extra", default="")
ap.add_argument("--tag", default="vb"); ap.add_argument("--size", type=int, default=256); ap.add_argument("--out", required=True)
a = ap.parse_args(); E = Path(a.E)
S = lambda m: json.loads((E / "scores" / f"{m}_{a.tag}.json").read_text())["per_clip"]
c0, c1 = a.pair.split(","); P0, P1 = S(c0), S(c1)
rows = []
for ds in ("ubody", "bedlam2"):
    d = sorted((P1[k]["psnr"] - P0[k]["psnr"], k) for k in P0 if k.startswith(ds + "/"))
    rows += [(d[len(d) // 2][1], f"typical {ds} (median {d[len(d) // 2][0]:+.2f} dB)"), (d[0][1], f"worst {ds} ({d[0][0]:+.2f} dB)")]
rows += [(k, "extra") for k in a.extra.split(",") if k]
cols = [(c.split(":") + [c.split(":")[0]])[:2] for c in a.cols.split(",")]
sc = {m: S(m) for m, _ in cols if m != "GT"}
try: font = ImageFont.truetype("DejaVuSans.ttf", 14); fsm = ImageFont.truetype("DejaVuSans.ttf", 12)
except OSError: font = fsm = ImageFont.load_default()
z, hd, lw = a.size, 22, 230
canvas = Image.new("RGB", (lw + z * len(cols), hd + z * len(rows)), "white"); dr = ImageDraw.Draw(canvas)
for j, (m, lab) in enumerate(cols): dr.text((lw + j * z + 4, 3), lab, fill="black", font=font)
sel = []
for i, (cid, why) in enumerate(rows):
    frames = sorted(p.stem for p in (E / "gt512" / cid).glob("*.png")); fr = frames[len(frames) // 2]
    y = hd + i * z; dr.text((4, y + 4), cid.split("/")[0], fill="black", font=font)
    for t, line in enumerate([cid.split("/")[1][-28:], fr, why]): dr.text((4, y + 24 + 16 * t), line, fill="black", font=fsm)
    for j, (m, lab) in enumerate(cols):
        p = E / "gt512" / cid / f"{fr}.png" if m == "GT" else E / "renders" / m / cid / f"{fr}.png"
        im = Image.open(p).convert("RGB").resize((z, z), Image.BICUBIC) if p.exists() else Image.new("RGB", (z, z), (90, 0, 0))
        canvas.paste(im, (lw + j * z, y))
        if m != "GT" and cid in sc[m]:
            r = sc[m][cid]; txt = f"{r['psnr']:.2f} dB / {r['lpips']:.3f}"
            dr.rectangle([lw + j * z, y + z - 18, lw + j * z + 8 * len(txt), y + z], fill="black"); dr.text((lw + j * z + 3, y + z - 17), txt, fill="white", font=fsm)
    sel.append({"clip": cid, "frame": fr, "row": why})
canvas.save(a.out, quality=92); Path(a.out).with_suffix(".json").write_text(json.dumps({"pair": a.pair, "cols": a.cols, "rows": sel}, indent=1))
print(json.dumps(sel, indent=1))
