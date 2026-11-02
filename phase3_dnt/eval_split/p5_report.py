"""PREREG phase 5 report on ValBench-S: per box source (rows p5_yolox / p5_mediapipe / p5_track) PSNR / SSIM / LPIPS, wrong-person frames
(chosen box IoU < 0.3 with the GT subject box), lost / re-detected frames, end-to-end box -> SPRINT -> render ms p50 / p95 (server, host load
recorded), and the registered choice rule. Usage: python p5_report.py [--out JSON]"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--E", default="/dev/shm/ubx858_eval/valbench_s"); ap.add_argument("--out", default="")
a = ap.parse_args(); E = Path(a.E); HERE = Path(__file__).resolve().parent
clips = {c["id"]: c for c in json.load(open(HERE / "valbench_s.json"))["clips"]}


def iou(p, q):
    w, h = max(0, min(p[2], q[2]) - max(p[0], q[0])), max(0, min(p[3], q[3]) - max(p[1], q[1])); i = w * h
    return i / max((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i, 1e-9)


R = {}
for src in ("yolox", "mediapipe", "track"):
    sp = E / "scores" / f"p5_{src}_vbs.json"
    if not sp.exists(): continue
    s = json.loads(sp.read_text())
    if "all" not in s["aggregate"]: continue                                                     # no scored clips (failed run)
    ag = s["aggregate"]["all"]
    wrong = lost = n = 0; e2e, loads, margin = [], [], None
    for cid, c in clips.items():
        r = json.loads((E / "renders" / f"p5_{src}" / c["dataset"] / c["clip"] / "run.json").read_text())
        margin = r.get("box_margin"); loads.append(r.get("load_avg", [None])[0])
        gtb = {f: [b[0], b[1], b[0] + b[2], b[1] + b[3]] for f, b in zip(c["frames"], c["bbox_xywh"])}
        for k, v in r["per_frame"].items():
            f = int(k.split("_")[1]); n += 1; lost += v["lost"]; e2e.append(v["e2e_ms"])
            if f in gtb and iou(v["box"], gtb[f]) < 0.3: wrong += 1
    R[src] = {"psnr": round(ag["psnr"], 3), "ssim": round(ag["ssim"], 4), "lpips": round(ag["lpips"], 4), "frames": n, "wrong_person_frames": wrong,
              "lost_frames": lost, "e2e_ms_p50": round(float(np.percentile(e2e, 50)), 1), "e2e_ms_p95": round(float(np.percentile(e2e, 95)), 1),
              "margin": margin, "host_load_1min_median": round(float(np.median([x for x in loads if x is not None])), 1)}
ref = R.get("yolox")
for src, v in R.items():
    v["acceptable"] = bool(ref and v["psnr"] >= ref["psnr"] - 0.10 and v["lpips"] <= ref["lpips"] + 0.003 and v["wrong_person_frames"] <= ref["wrong_person_frames"])
ok = [s for s, v in R.items() if v["acceptable"]]
R["choice"] = {"acceptable": ok, "chosen_by_server_p50": min(ok, key=lambda s: R[s]["e2e_ms_p50"]) if ok else "yolox",
               "note": "server timing on a CPU-saturated shared host; the registered rule uses laptop p50 when the user supplies it"}
print(json.dumps(R, indent=1))
if a.out: Path(a.out).write_text(json.dumps(R, indent=1))
