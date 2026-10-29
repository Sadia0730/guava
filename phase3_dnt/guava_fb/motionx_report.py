"""GUAVA-FB / Motion-X, stage 3: index.json (split: 20 val clips, seed 0, from the kept pool; humman excluded from val because
its clips share performers), report.json (candidate pool, pass rates of (a) dataset labels and (b) SPRINT refit, filter
variants, timings, disk), 10 overlays (top: image crop, bottom: the kept label's mesh; 5 dataset-label + 5 refit clips, seed 0),
optional GUAVA renderer check (fb_common.check_render) on dataset-label clips.
Usage: python motionx_report.py --work WORK [--check-render N]"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
DATA, DIAG, REP = O / "data/motionx", O / "runs/motionx/clips", O / "reports/motionx"
LICENCE = ("Motion-X / Motion-X++ (IDEA-Research): access granted for NON-COMMERCIAL research use only (request form, "
           "github.com/IDEA-Research/Motion-X); each sub-dataset keeps its own licence (IDEA400, HAA500, HuMMan, kungfu / music / "
           "perform / animation web videos); no licence file is on disk with the downloaded zips. Research use only; not for "
           "redistribution.")


def src_video(c): return re.sub(r"(_clip\d+)+$", "", c)


def frac(x): return float(np.mean(x)) if len(x) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", type=Path, required=True); ap.add_argument("--check-render", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=20); a = ap.parse_args()
    cand = json.loads((a.work / "candidates.json").read_text())
    fits = {}
    for p in cand["picks"]:
        s, c = p.split("/", 1); f = DIAG / f"{s}__{c}" / "fit.json"
        if f.exists(): fits[p] = json.loads(f.read_text())
    preps = {}
    for p in cand["picks"]:
        s, c = p.split("/", 1); f = a.work / s / f"{c}.prep.json"
        if f.exists(): preps[p] = json.loads(f.read_text())
    ok = {p: f for p, f in fits.items() if "a" in f}
    kept = sorted(p for p, f in ok.items() if f.get("kept"))
    # ---- split
    rng = np.random.default_rng(0); pool = sorted(p for p in kept if not p.startswith("humman/"))
    val = set(rng.choice(pool, min(a.n_val, len(pool)), replace=False).tolist()) if pool else set()
    clips = []
    for p in kept:
        s, c = p.split("/", 1); f = ok[p]
        clips.append({"name": f"{s}__{c}", "subject": f"motionx/{s}/{src_video(c)}", "split": "val" if p in val else "train",
                      "frames": int(f["written_frames"]), "views": ["0"], "label_source": f["label_source"], "subset": s})
    index = {"source": "motionx", "licence": LICENCE, "clips": clips,
             "notes": ("Motion-X++ full-body clips (audit rule: >= 80 % full-body stride-12 rows), frames every ~0.5 s (cap 40), "
                       "registered alignment filter (clip kept iff >= 80 % of its sampled frames pass (fit_ubc.frame_filter, limb-mean reading, as UBC); only "
                       "passing, non-flipped frames written; flipped keys in drop_frames.json). label_source 'dataset' = Motion-X SMPL-X v1.1 labels converted to SMPL-X 2020 (body exact: "
                       "shared template / first 10 shape dirs; one shape per clip = mean label betas; exp50 = linear map of expr10), OSX "
                       "camera; 'sprint_refit' = SPRINT-from-scratch joint-video refit (video_fit.VideoFit). subject = source video (clip "
                       "name minus _clipN; 1 clip per source video, so one shape per subject); performers are not identified in Motion-X "
                       "(HuMMan clips share a few performers -> humman never in val). Monocular: view '0'.")}
    DATA.mkdir(parents=True, exist_ok=True); (DATA / "index.json").write_text(json.dumps(index, indent=1))
    # flipped frames of kept clips (ubc_flip_frames.py rule, computed in motionx_fit.py); motionx_fit already left them out of the
    # written clips, so every key listed here is ABSENT from the clip (listed for the coordinator's uniform drop_frames handling)
    drop = {}
    for p in kept:
        f = ok[p]; fr = f["a" if f["label_source"] == "dataset" else "b"]["frames"]
        ks = [x["key"] if "key" in x else None for x in fr]
        if f["label_source"] == "dataset":
            import numpy as _np
            s_, c_ = p.split("/", 1); vf = _np.load(a.work / s_ / c_ / "prep.npz")["video_frame"]; ks = [f"frame_{int(v):06d}" for v in vf]
        fl = [k for k, x in zip(ks, fr) if x.get("flipped") and x.get("pass")]
        if fl: drop[f"{p.split('/', 1)[0]}__{p.split('/', 1)[1]}"] = fl
    (DATA / "drop_frames.json").write_text(json.dumps(drop, indent=1))
    # ---- pass rates
    by = lambda pred: {s: frac([pred(f) for p, f in ok.items() if p.startswith(s + "/")]) for s in sorted({p.split('/')[0] for p in ok})}
    a_kept = [p for p, f in ok.items() if f["a"]["kept"]]; a_fail = [p for p in ok if p not in a_kept]
    b_ran = [p for p in ok if "frames" in ok[p].get("b", {})]
    vrates = {}
    for v, key in (("limb_mean_registered_reading", "pass_frac"), ("single_kp_max_reading", "pass_frac_single_kp_max")):
        ra = [ok[p]["a"].get(key, 0) >= 0.8 for p in ok]; rb = [ok[p]["b"].get(key, 0) >= 0.8 for p in ok]
        vrates[v] = {"a_clip_pass_rate": frac(ra), "b_clip_pass_rate_all_clips": frac(rb), "kept_rate_a_else_b": frac([x or y for x, y in zip(ra, rb)])}
    flips = {"a_frames_checked": sum(ok[p]["a"].get("flip_checked", 0) for p in ok), "a_frames_flipped": sum(ok[p]["a"].get("flipped", 0) for p in ok),
             "b_frames_checked": sum(ok[p]["b"].get("flip_checked", 0) for p in ok), "b_frames_flipped": sum(ok[p]["b"].get("flipped", 0) for p in ok),
             "kept_clips_passing_frames_dropped_as_flipped": sum(ok[p].get("flipped_passing_frames_dropped", 0) for p in kept),
             "rule": "ubc_flip_frames.py: DWPose shoulder order vs fitted facing, checked when both shoulder scores >= 0.5, x-gap >= 40 px, |facing| >= 0.5"}
    fr_a = [x for f in ok.values() for x in f["a"]["frames"]]; fr_b = [x for f in ok.values() for x in f["b"].get("frames", [])]
    def frame_stats(fr):
        g = [x for x in fr if "mesh_out_frac" in x]
        return {"frames": len(fr), "no_person": len(fr) - len(g), "pass": frac([x["pass"] for x in fr]), "pass_a(mask)": frac([x["pass_a"] for x in fr]),
                "pass_b(kp)": frac([x["pass_b"] for x in fr]), "median_mesh_out_frac": float(np.median([x["mesh_out_frac"] for x in g])) if g else None,
                "median_max_kp_err_rel": float(np.median([x["max_kp_err_rel"] for x in g if np.isfinite(x["max_kp_err_rel"])])) if g else None}
    T = lambda key: [f["times_s"].get(key, 0) for f in ok.values() if "times_s" in f]
    prep_ok = [x for x in preps.values() if x.get("ok")]
    du = lambda p: subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0] if p.exists() else "0"
    rep = {"source": "motionx", "licence": LICENCE, "generated": time.strftime("%Y-%m-%d %H:%M"),
           "smplx": {"dataset_labels": "Motion-X++ local_motion SMPL-X v1.1 NEUTRAL (OSX: 10 betas, 10 expr, absolute 45-D hands, no eyes), OSX crop camera",
                     "conversion_to_2020": "SMPL-X v1.1 and SMPLX_NEUTRAL_2020 share v_template, J_regressor, posedirs, weights and shapedirs[:, :, :10] "
                                           "(checked: identical), so body pose / shape copy exactly (betas10 zero-padded to 200); ONE shape per clip = mean of "
                                           "the clip's labelled betas (mean-shape vs per-frame-shape vertex error reported); exp50 = W expr10 (linear map, "
                                           "build_face_labels_smplx2020 motionx_gt 'linear', 0.56 mm mean head error there); jaw copied; eyes 0",
                     "views": "monocular (view '0')"},
           "candidate_pool": {k: v for k, v in cand.items() if k != "picks" and k != "median_body_h_px"},
           "eval_dataset_exclusion": "name check (People-Snapshot / NeuMan / X-Humans / UBC / 3DPW patterns): 0 hits in all 7 subsets; Motion-X "
                                     "videos come from IDEA400 captures, HuMMan, HAA500 and web videos, none from an evaluation set",
           "prep": {"clips": len(preps), "ok": len(prep_ok), "errors": {e: sum(1 for x in preps.values() if (x.get("error") or "").startswith(e))
                                                                        for e in sorted({(x.get("error") or "")[:30] for x in preps.values() if not x.get("ok")})},
                    "frames": int(sum(x["n_frames"] for x in prep_ok)), "frames_with_person": int(sum(x["det_ok"] for x in prep_ok)),
                    "dwpose_cpu_s_per_frame": float(sum(x["dwpose_s"] for x in prep_ok) / max(1, sum(x["n_frames"] for x in prep_ok))),
                    "mean_shape_conversion_err_mm": float(np.mean([x["conv_err_mean_mm"] for x in prep_ok])) if prep_ok else None},
           "fit": {"clips_processed": len(ok), "errors": [p for p, f in fits.items() if "a" not in f and not f.get("prep_error")][:20],
                   "a_dataset_labels": {"clip_pass_rate": frac([f["a"]["kept"] for f in ok.values()]), "by_subset": by(lambda f: f["a"]["kept"]), "frames": frame_stats(fr_a),
                                        "render_check_record_vs_gt_proj_px512_mean": float(np.mean([f["a"]["check"]["record_vs_gt_proj_px512_mean"] for f in ok.values()]))},
                   "b_sprint_refit": {"clips_refit": len(b_ran), "clip_pass_rate_on_a_failures": frac([ok[p]["b"]["kept"] for p in a_fail]),
                                      "clip_pass_rate_all_clips": frac([ok[p]["b"]["kept"] for p in ok]), "by_subset_all": by(lambda f: f["b"]["kept"]),
                                      "frames": frame_stats(fr_b)},
                   "filter_readings_clip_rates": vrates, "flip_check": flips,
                   "kept": {"clips": len(kept), "dataset": sum(ok[p]["label_source"] == "dataset" for p in kept),
                            "sprint_refit": sum(ok[p]["label_source"] == "sprint_refit" for p in kept),
                            "by_subset": {s: sum(p.startswith(s + "/") for p in kept) for s in sorted({p.split('/')[0] for p in ok})},
                            "frames": int(sum(ok[p]["written_frames"] for p in kept)), "train_clips": sum(c["split"] == "train" for c in clips),
                            "val_clips": sum(c["split"] == "val" for c in clips), "val": sorted(val)},
                   "timing_s_per_clip_mean": {"label_filter": frac(T("label_filter")), "refit_total": frac(T("refit")), "write": frac(T("write")),
                                              "gpu_total": frac([sum(v for k, v in f["times_s"].items() if not k.startswith("refit_")) for f in ok.values() if "times_s" in f])},
                   "peak_gpu_mem_gb_max": max([f.get("peak_mem_gb", 0) for f in ok.values()] or [0])},
           "disk_bytes": {"data_motionx": int(du(DATA)), "runs_motionx": int(du(O / "runs/motionx")), "work_local_frames": int(du(a.work))},
           "filter": "fit_ubc.frame_filter + fit_ubc.silhouettes (via video_fit), identical to UBC: (a) mesh outside matte < 5 %; (b) >= 15 of 21 "
                     "keypoints (OpenPose-134 body incl. hips + 6 foot points, no ears) with score > 0.5 and max over 5 limbs of the limb's mean error "
                     "< 3 % of DWPose box height; clip kept iff >= 80 % of sampled frames pass; only passing, non-flipped frames written. "
                     "First run (own filter implementation) superseded on coordinator instruction; its results in runs/motionx/clips_v1, data_v1."}
    REP.mkdir(parents=True, exist_ok=True); (REP / "overlays").mkdir(exist_ok=True)
    # ---- overlays: rows of the diagnostic overlay (img / dataset mesh / refit mesh) -> top image, bottom kept label's mesh
    rng = np.random.default_rng(0); ds = sorted(p for p in kept if ok[p]["label_source"] == "dataset"); rf = sorted(p for p in kept if ok[p]["label_source"] == "sprint_refit")
    na = min(5, len(ds)); nb = min(10 - na, len(rf)); na = min(10 - nb, len(ds))
    picks = (rng.choice(ds, na, replace=False).tolist() if na else []) + (rng.choice(rf, nb, replace=False).tolist() if nb else [])
    ov = []
    for p in picks:
        s, c = p.split("/", 1); im = cv2.imread(str(DIAG / f"{s}__{c}" / "overlay.jpg")); h = im.shape[0] // 3
        row = 1 if ok[p]["label_source"] == "dataset" else 2
        g = np.concatenate([im[:h], im[row * h:(row + 1) * h]], 0)
        cv2.putText(g, f"{p} [{ok[p]['label_source']}]"[:90], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        out = REP / "overlays" / f"{s}__{c}.jpg"; cv2.imwrite(str(out), g, [cv2.IMWRITE_JPEG_QUALITY, 88]); ov.append(str(out))
    rep["overlays"] = {"files": ov, "note": "top: dataset-label perspective crop; bottom: kept label's mesh (refit rows use the refit's own "
                                            "perspective crop, f = W) with outline; per-frame filter numbers printed"}
    if a.check_render:
        import torch, fb_common as FB
        dev = torch.device("cuda:0"); _, inf, _, _ = FB.setup_a20(dev); res = []
        for p in ds[:a.check_render]:
            s, c = p.split("/", 1); r = FB.check_render(DATA / f"{s}__{c}", n=10, dev=dev, inf=inf, out_jpg=REP / "check_render" / f"{s}__{c}.jpg")
            res.append({"clip": p, "mean_px_err_512": r["mean_px_err_512"], "mean_mask_precision": r["mean_mask_precision"]})
        rep["guava_render_check_dataset_clips"] = {"rows": res, "mean_px_err_512": frac([r["mean_px_err_512"] for r in res]),
                                                   "note": "GUAVA (a20, patched camera) render vs gt_proj (SOURCE v1.1 mesh with per-frame label shape); "
                                                           "residual = one-shape-per-clip conversion"}
    (REP / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep["fit"][k] for k in ("a_dataset_labels", "b_sprint_refit", "kept", "filter_readings_clip_rates", "flip_check")}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
