"""RAGA checks (see ../SKILL.md). Exit code 0 = pass, 1 = fail.

  speed TIMING.json [--record DIR]   the 30 fps rule on a live_sprint_guavafb.py --timing-json file (worker off); with the
                                     --record folder of a webcam run, also the share of camera frames processed
  same RUN_A RUN_B [--floor RUN_C]   are two replays of one recording (--record folders, replayed with --no-pace) rendering the
                                     same avatar frames? --floor = a second replay of the reference path (run-to-run noise):
                                     B passes when it is identical to A, or its median PSNR to A is within FLOOR_TOL of C's"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FPS_MIN, INT_P50, INT_P95, KEPT_MIN, LAT_WARN = 29.5, 34.0, 40.0, 0.98, 100.0         # SKILL.md rule 1 (ms where timed)
FLOOR_TOL = 0.5                                                                        # dB below the floor's median PSNR


def speed(a):
    t = json.loads(Path(a.timing).read_text()); s = t["stages_ms"]; fi = s["frame_interval"]; cd = s.get("capture_to_display") or {}
    checks = [("observed fps", t.get("fps_observed"), ">=", FPS_MIN), ("frame interval p50 (ms)", fi.get("p50"), "<=", INT_P50),
              ("frame interval p95 (ms)", fi.get("p95"), "<=", INT_P95)]
    if a.record:
        ii = sorted(json.loads(line)["i"] for line in open(Path(a.record) / "frames.jsonl"))
        checks.append(("camera frames processed", len(ii) / (ii[-1] - ii[0] + 1) if ii else None, ">=", KEPT_MIN))
        if t.get("recorder_dropped"): print(f"note: the recorder dropped {t['recorder_dropped']} frames; they count as missed here")
    ok = True
    for name, v, op, thr in checks:
        good = v is not None and (v >= thr if op == ">=" else v <= thr); ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}: {'n/a' if v is None else round(v, 3)} (rule {op} {thr})")
    print("stages p50 / p95 ms: " + ", ".join(f"{k} {v['p50']:.1f} / {v['p95']:.1f}" for k, v in s.items() if v.get("p50") is not None))
    print(f"gpu {t.get('gpu')}, worker {t.get('worker')}, live-loop memory peak {t.get('gpu_mem_peak_gb')} GB")
    if t.get("worker", "off") != "off": print("note: RAGA's speed rule is measured with the worker off")
    if (cd.get("p50") or 0) > LAT_WARN: print(f"warning: capture-to-display latency p50 {cd['p50']:.0f} ms (> {LAT_WARN:.0f} ms is felt)")
    return ok


def diff(A, B):
    import cv2
    ka, kb = {p.name for p in (Path(A) / "avatar").glob("*.jpg")}, {p.name for p in (Path(B) / "avatar").glob("*.jpg")}
    keys = sorted(ka & kb); ident, ps = 0, []
    for k in keys:
        x, y = (Path(A) / "avatar" / k).read_bytes(), (Path(B) / "avatar" / k).read_bytes()
        if x == y: ident += 1; continue
        d = cv2.imread(str(Path(A) / "avatar" / k)).astype(np.float64) - cv2.imread(str(Path(B) / "avatar" / k)).astype(np.float64)
        mse = float((d ** 2).mean()); ps.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    return {"frames_a": len(ka), "frames_b": len(kb), "common": len(keys), "identical": ident,
            "min_psnr": min(ps) if ps else None, "median_psnr": float(np.median(ps)) if ps else None}


def show(name, r):
    s = f"{name}: {r['common']} common frames ({r['frames_a']} / {r['frames_b']}), {r['identical']} byte-identical"
    if r["min_psnr"] is not None: s += f"; the others: PSNR min {r['min_psnr']:.2f} / median {r['median_psnr']:.2f} dB"
    print(s)


def same(a):
    r = diff(a.run_a, a.run_b); show("A vs B", r)
    if not r["common"]: print("FAIL  no common frames (same recording? --record folders with avatar/*.jpg?)"); return False
    if r["frames_a"] != r["frames_b"]: print("note: the runs rendered different numbers of frames (use --no-pace so every frame is processed)")
    if r["identical"] == r["common"]: print("PASS  identical renders"); return True
    if not a.floor: print("DIFFERS  rerun with --floor <second reference replay> to compare with the run-to-run noise"); return False
    f = diff(a.run_a, a.floor); show("A vs floor", f)
    if f["identical"] == f["common"]: print("FAIL  the reference path is deterministic, the changed path is not identical"); return False
    ok = r["median_psnr"] >= f["median_psnr"] - FLOOR_TOL                                            # min is one frame: shown, not decisive
    print(f"{'PASS' if ok else 'FAIL'}  median PSNR {r['median_psnr']:.2f} dB vs the noise floor's {f['median_psnr']:.2f} dB (tolerance {FLOOR_TOL} dB)")
    return ok


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("speed"); p.add_argument("timing"); p.add_argument("--record", default="")
p = sub.add_parser("same"); p.add_argument("run_a"); p.add_argument("run_b"); p.add_argument("--floor", default="")
a = ap.parse_args()
sys.exit(0 if (speed(a) if a.cmd == "speed" else same(a)) else 1)
