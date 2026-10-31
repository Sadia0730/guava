"""Amendment 5 decision (pilot D vs pilot B), from compare_fb.py outputs + registered 8k validation."""
import json
from pathlib import Path
R = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/pilot_compare_D")
RU = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs")


def summ(l):
    s = json.loads((R / l / "eval.json").read_text())["summary"]
    if (R / l / "geometry.json").exists():
        for t, d in json.loads((R / l / "geometry.json").read_text())["summary"].items(): s.setdefault(t, {}).update(d)
    return s


def val8k(r):
    for line in open(RU / f"guava_fb_pilot{r}" / "train_log.jsonl"):
        j = json.loads(line)
        if j.get("split") == "val" and j.get("iter") == 8000: return j


S = {l: summ(l) for l in "BCD"}; tim = json.loads((R / "timing.json").read_text()); V = {l: val8k(l) for l in "BCD"}
g = lambda l, t, k: S[l][t][k]
crit = [
    ("(2) 4D-DRESS clothing-box LPIPS lower", g("D", "dress4d", "cloth_lpips"), g("B", "dress4d", "cloth_lpips"), lambda d, b: d < b),
    ("(3) zoomed clothing LPIPS lower", g("D", "dress4d", "zoom_lpips"), g("B", "dress4d", "zoom_lpips"), lambda d, b: d < b),
    ("(7) D-Normal cosine vs Sapiens (all) higher", g("D", "all", "geo_cos_dnormal"), g("B", "all", "geo_cos_dnormal"), lambda d, b: d > b),
    ("(8a) scan rendered-normal cosine higher", g("D", "dress4d", "scan_cos_rendered"), g("B", "dress4d", "scan_cos_rendered"), lambda d, b: d > b),
    ("(8b) scan fold detail higher", g("D", "dress4d", "fold_scan_rendered"), g("B", "dress4d", "fold_scan_rendered"), lambda d, b: d > b),
    ("(9) outer-garment miss fraction lower", g("D", "dress4d", "miss_outer"), g("B", "dress4d", "miss_outer"), lambda d, b: d < b),
    ("(1) full-body SSIM not lower by > 0.002", V["D"]["fullbody_mean_ssim"], V["B"]["fullbody_mean_ssim"], lambda d, b: d >= b - 0.002),
    ("(5) flicker not higher by > 10 %", g("D", "flicker", "lpips_consecutive"), g("B", "flicker", "lpips_consecutive"), lambda d, b: d <= 1.10 * b),
    ("(6) frame time not slower by > 2 %", tim["D"]["frame_ms"], tim["B"]["frame_ms"], lambda d, b: d <= 1.02 * b),
]
rows = ["| criterion | B | D | C (ref.) | pass |", "|---|---|---|---|---|"]
cref = [g("C", "dress4d", "cloth_lpips"), g("C", "dress4d", "zoom_lpips"), g("C", "all", "geo_cos_dnormal"), g("C", "dress4d", "scan_cos_rendered"),
        g("C", "dress4d", "fold_scan_rendered"), g("C", "dress4d", "miss_outer"), V["C"]["fullbody_mean_ssim"], g("C", "flicker", "lpips_consecutive"), tim["C"]["frame_ms"]]
ok = True
for (n, d, b, f), c in zip(crit, cref):
    p = f(d, b); ok &= p; rows.append(f"| {n} | {b:.4f} | {d:.4f} | {c:.4f} | {'yes' if p else '**no**'} |")
rows.append(f"\n**Pilot D improves on B (amendment 5): {'yes' if ok else 'no'}**\n")
rows.append("Registered validator, 8k (reported): | | B | C | D |\n|---|---|---|---|")
for t in ("dress4d", "bedlam2", "motionx", "ubody"):
    rows.append(f"| {t} PSNR / SSIM / LPIPS | " + " | ".join(f"{V[l][t]['psnr']:.2f} / {V[l][t]['ssim']:.4f} / {V[l][t]['lpips']:.4f}" for l in "BCD") + " |")
rows.append("| UBody face LPIPS | " + " | ".join(f"{V[l]['ubody']['face_lpips']:.4f}" for l in "BCD") + " |")
rows.append("| eligible (UBody SSIM >= 0.8193, face LPIPS <= 0.1873) | " + " | ".join(str(V[l]["eligible"]) for l in "BCD") + " |")
(R / "decision.md").write_text("\n".join(rows) + "\n"); print("\n".join(rows))
