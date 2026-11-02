"""PREREG phase 4, H4: apply the registered rule to the pilot's scores and SPRINT validation logs; print and write the report.
E1 (primary) ValBench-S, E2 ValBench-v1 BEDLAM2, E3 ValBench-v1 UBody (SPRINT-seen; reported only); guard = SPRINT's own validation
at step 4,000 (EMA): 3DPW-val and BEDLAM2-val PA-MPJPE, arm <= control + 1.0 mm.
Usage: python h4_report.py --out O/reports/improve/h4_report.json"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--vbs", default="/dev/shm/ubx858_eval/valbench_s"); ap.add_argument("--vb", default="/dev/shm/ubx858_eval/valbench_v1")
ap.add_argument("--runs", default="/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/sprint_h4"); a = ap.parse_args()
ARM, CTRL = "fb_sprint_k0_trkc_h4h1", "fb_sprint_k0_trkc_h4ctrl"


def sc(E, m, tag):
    return json.loads((Path(E) / "scores" / f"{m}_{tag}.json").read_text())


def agg(s, ds):
    x = s["aggregate"][ds]; return {k: round(float(x[k]), 4) for k in ("psnr", "ssim", "lpips")} | {"n_frames": x.get("n_frames")}


def per_clip(sa, sb, ds=None):
    pa, pb = sa["per_clip"], sb["per_clip"]
    return {c: {"psnr": round(pa[c]["psnr"] - pb[c]["psnr"], 3), "lpips": round(pa[c]["lpips"] - pb[c]["lpips"], 4)}
            for c in sorted(pa) if c in pb and (ds is None or pa[c]["dataset"] == ds)}


def val(arm):
    out = {}
    for line in open(Path(a.runs) / f"train_{arm}" / "train_log.jsonl"):
        r = json.loads(line)
        if r.get("step") == 4000 and r.get("split") in ("val3dpw_ema", "val_bedlam2_val_ema", "val_bedlam2_val_tilt90_ema", "val_moyo_val_ema", "val_arctic_s10_ema"):
            out[r["split"]] = round(float(r["pa_mpjpe_mm"]), 2)
    return out


R = {}
s_arm, s_ctrl = sc(a.vbs, ARM, "vbs"), sc(a.vbs, CTRL, "vbs")
s_orig, s_teach, s_gt = sc(a.vbs, "fb_sprint_k0_trkc", "vbs"), sc(a.vbs, "fb_sprint_k60_trkc", "vbs"), sc(a.vbs, "fb_gt", "vbs")
e1 = {m: agg(s, "all") for m, s in (("original_k0_trkc", s_orig), ("control", s_ctrl), ("arm", s_arm), ("teacher_k60_trkc", s_teach), ("gt_driven", s_gt))}
d1 = per_clip(s_arm, s_ctrl); up = sum(v["psnr"] > 0 for v in d1.values())
gain_teacher = e1["teacher_k60_trkc"]["psnr"] - e1["original_k0_trkc"]["psnr"]
R["E1_valbench_s"] = {"aggregate": e1, "arm_minus_control": {"psnr": round(e1["arm"]["psnr"] - e1["control"]["psnr"], 3),
                      "lpips": round(e1["arm"]["lpips"] - e1["control"]["lpips"], 4)}, "clips_up": f"{up} / {len(d1)}", "per_clip_arm_minus_control": d1,
                      "control_minus_original": round(e1["control"]["psnr"] - e1["original_k0_trkc"]["psnr"], 3),
                      "teacher_gain_over_original": round(gain_teacher, 3),
                      "share_of_teacher_gain_recovered": round((e1["arm"]["psnr"] - e1["control"]["psnr"]) / gain_teacher, 3) if gain_teacher else None}
v_arm, v_ctrl = sc(a.vb, ARM, "vb"), sc(a.vb, CTRL, "vb"); v_orig = sc(a.vb, "fb_sprint_k0_trkc", "vb")
for key, ds in (("E2_valbench_bedlam2", "bedlam2"), ("E3_valbench_ubody_sprint_seen", "ubody")):
    e = {m: agg(s, ds) for m, s in (("original_k0_trkc", v_orig), ("control", v_ctrl), ("arm", v_arm))}
    d = per_clip(v_arm, v_ctrl, ds)
    R[key] = {"aggregate": e, "arm_minus_control": {"psnr": round(e["arm"]["psnr"] - e["control"]["psnr"], 3), "lpips": round(e["arm"]["lpips"] - e["control"]["lpips"], 4)},
              "clips_up": f"{sum(v['psnr'] > 0 for v in d.values())} / {len(d)}", "per_clip_arm_minus_control": d}
g_arm, g_ctrl = val("h1"), val("ctrl")
R["guard_sprint_validation_step4000_ema_pa_mpjpe_mm"] = {"control": g_ctrl, "arm": g_arm}
c = {"a_E1_psnr_ge_0.30_and_lpips_not_worse": R["E1_valbench_s"]["arm_minus_control"]["psnr"] >= 0.30 and R["E1_valbench_s"]["arm_minus_control"]["lpips"] <= 0,
     "b_E1_psnr_up_on_ge_2/3_clips": up * 3 >= 2 * len(d1),
     "c_E2_bedlam2_psnr_ge_-0.10": R["E2_valbench_bedlam2"]["arm_minus_control"]["psnr"] >= -0.10,
     "d_guard_3dpw_and_bedlam2val_le_control_plus_1mm": g_arm["val3dpw_ema"] <= g_ctrl["val3dpw_ema"] + 1.0 and g_arm["val_bedlam2_val_ema"] <= g_ctrl["val_bedlam2_val_ema"] + 1.0}
R["rule"] = c; R["verdict"] = "H4 supported" if all(c.values()) else "H4 not supported"
Path(a.out).write_text(json.dumps(R, indent=1))
print(json.dumps({k: v for k, v in R.items() if not k.startswith(("E1", "E2", "E3"))}, indent=1))
for k in ("E1_valbench_s", "E2_valbench_bedlam2", "E3_valbench_ubody_sprint_seen"):
    print(k, json.dumps({kk: vv for kk, vv in R[k].items() if kk != "per_clip_arm_minus_control"}))
