"""GUAVA-FB, UBC-Fashion: index.json (registered alignment-filter rule: video kept iff >= 80 % of its frames pass) + report
O/reports/ubc/report.json + 10 overlays O/reports/ubc/overlays/ (top: perspective image crop with DWPose keypoints,
bottom: fitted SMPL-X mesh), chosen from different videos, front and back views. Reads fit_ubc.py diagnostics
(O/runs/ubc/clips/<clip>/fit.json, overlay_{front,back}.jpg) and ubc_identity.py's identity.json."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

D = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/ubc_fashion")
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
GART6 = ["91+bCFG1jOS", "91Iegdp9HFS", "91fxYsir49S", "91QFEra7jDS", "91Ile3zLhMS", "91WvLcNpdzS"]
STATS = O / "runs/ubc/video_stats.json"                           # cv2 probe of every video (resolution, fps, frames)


def du(p):
    return int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0]) if Path(p).exists() else 0


def main():
    ident = json.loads((O / "reports/ubc/identity.json").read_text())
    train = sorted(p.stem for p in (D / "train").glob("*.mp4")); test = sorted(p.stem for p in (D / "test").glob("*.mp4"))
    fit_list = [l.strip() for l in (O / "runs/ubc/fit_list.txt").read_text().splitlines() if l.strip()]
    removed = set(ident["removed"]); nearest = ident["per_train"]
    fits = {}
    for n in fit_list:
        f = O / "runs/ubc/clips" / n / "fit.json"
        if f.exists():
            fits[n] = json.loads(f.read_text())
    # within-train identity groups (information; subject = video)
    grp = {}
    for gi, g in enumerate(ident["within_train_identity_groups_size_gt1"]):
        for v in g:
            grp[v] = gi
    kept, excl = [], []
    for n in test:
        excl.append({"name": n, "reason": "official_test_video"})
    for n in GART6:
        excl.append({"name": n, "reason": "GART evaluation video (in the official TRAIN list; never in training)"})
    for n in train:
        if n in GART6:
            continue
        if n in removed:
            excl.append({"name": n, "reason": "identity_match_eval", "max_cos": nearest[n]["max_cos"], "nearest_eval": nearest[n]["nearest_eval"]})
        elif n not in fits:
            excl.append({"name": n, "reason": "not_fitted (missing)"})
        elif not fits[n]["ok"]:
            excl.append({"name": n, "reason": "fit_failed", "error": fits[n]["error"]})
        elif not fits[n]["kept_by_filter"]:
            excl.append({"name": n, "reason": "alignment_filter", "pass_frac": round(fits[n]["pass_frac"], 3),
                         "pass_a_frac": round(fits[n]["pass_a_frac"], 3), "pass_b_frac": round(fits[n]["pass_b_frac"], 3)})
        else:
            kept.append({"name": n, "subject": f"ubc_{n}", "split": "train", "frames": fits[n]["n_frames"], "views": ["0"],
                         "identity_group": grp.get(n), "pass_frac": round(fits[n]["pass_frac"], 3)})
    index = {"source": "ubc", "licence": "UBC-Fashion (Zablotskaia et al., DwNet, BMVC 2019): research use only; cite the paper",
             "clips": kept, "excluded": excl,
             "notes": ("Official TRAIN videos only; every official test video and GART's 6 UBC videos excluded; train videos whose face "
                       "matches an evaluation identity (ArcFace max cosine > 0.5) excluded; SPRINT-from-scratch + joint video refinement "
                       "(one shape per video, temporal smoothing), perspective crop (focal = image width), StyleMatte; registered "
                       "alignment filter (video kept iff >= 80 % of frames pass). subject = the video (no cross-video shape lock); "
                       "identity_group = ArcFace within-train identity component (information).")}
    (O / "data/ubc").mkdir(parents=True, exist_ok=True)
    (O / "data/ubc/index.json").write_text(json.dumps(index, indent=1))
    # ---- report
    ok = [v for v in fits.values() if v["ok"]]
    fr = [f for v in ok for f in v["frames"]]
    tkeys = ["dwpose_cpu", "sprint", "persp_matting", "refine", "filter", "write", "gpu_total"]
    times = {k: {"median": round(float(np.median([v["times_s"][k] for v in ok])), 2), "sum_h": round(float(np.sum([v["times_s"][k] for v in ok])) / 3600, 3)} for k in tkeys} if ok else {}
    kept_groups = {}
    for c in kept:
        if c["identity_group"] is not None:
            kept_groups.setdefault(c["identity_group"], []).append(c["name"])
    vstats = {}
    if STATS.exists():
        from collections import Counter
        for sp, rows in json.loads(STATS.read_text()).items():
            nn = [r["n"] for r in rows]
            vstats[sp] = {"resolution_wxh": {f"{w}x{h}": c for (w, h), c in Counter((r["w"], r["h"]) for r in rows).items()},
                          "fps": sorted({round(r["fps"], 2) for r in rows}), "frames_min_median_max": [min(nn), int(np.median(nn)), max(nn)], "frames_total": int(sum(nn))}
    rep = {"source": "ubc", "dataset": {"video_stats": vstats, "path": str(D), "licence": index["licence"], "train_videos": len(train), "test_videos": len(test),
                                        "size_gb": {"train": round(du(D / "train") / 1e9, 3), "test": round(du(D / "test") / 1e9, 3)}},
           "gart6": {"names": GART6, "found_in": "official TRAIN list (fashion_train.txt), NOT the test list; excluded from training as evaluation videos",
                     "source": "github.com/JiahuiLei/GART script/fit_ubc.sh, eval_ubc.sh"},
           "identity": {k: ident[k] for k in ("rule", "model", "evaluation_identities", "faces_per_video", "n_train_checked", "n_removed", "max_cos_hist", "max_cos_hist_bins",
                                               "removed_if_mean_emb_rule", "within_train_pairs_over_thr", "n_within_train_identities", "n_retried")},
           "fitting": {"script": "phase3_dnt/guava_fb/fit_ubc.py", "ckpt": "/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt",
                       "videos_fitted": len(fit_list), "ok": len(ok), "failed": [v["name"] for v in fits.values() if not v["ok"]],
                       "frames": len(fr), "frames_per_video_median": float(np.median([v["n_frames"] for v in ok])) if ok else 0,
                       "stride_hist": {str(s): sum(1 for v in ok if v["stride"] == s) for s in sorted({v["stride"] for v in ok})},
                       "kp_err_px1024_after_median": round(float(np.median([v["diag"]["kp_err_px1024_after"] for v in ok])), 2) if ok else None,
                       "kp_err_px1024_before_median": round(float(np.median([v["diag"]["kp_err_px1024_before"] for v in ok])), 2) if ok else None,
                       "invtanfov_median": round(float(np.median([f["invtanfov"] for f in fr])), 3) if fr else None,
                       "weights": ok[0]["weights"] if ok else None, "iters": ok[0]["iters"] if ok else None},
           "filter": {"rule": "frame: (a) mesh px outside matte < 5 %; (b) max per-limb (mean of limb keypoints) error < 3 % body height, keypoints score > 0.5, >= 15; video kept iff >= 80 % frames pass",
                      "frame_pass": round(float(np.mean([f["pass"] for f in fr])), 4) if fr else None,
                      "frame_pass_a": round(float(np.mean([f["pass_a"] for f in fr])), 4) if fr else None,
                      "frame_pass_b": round(float(np.mean([f["pass_b"] for f in fr])), 4) if fr else None,
                      "frame_fail_b_lt15kp": round(float(np.mean([f["n_kp05"] < 15 for f in fr])), 4) if fr else None,
                      "frame_pass_if_single_kp_max (information, stricter reading)": round(float(np.mean([f["pass_a"] and f["n_kp05"] >= 15 and f["max_single_kp_err_rel"] < 0.03 for f in fr])), 4) if fr else None,
                      "mesh_out_frac_p50_p90": np.percentile([f["mesh_out_frac"] for f in fr], [50, 90]).round(4).tolist() if fr else None,
                      "limb_err_p50_p90": np.percentile([f["max_kp_err_rel"] for f in fr if np.isfinite(f["max_kp_err_rel"])], [50, 90]).round(4).tolist() if fr else None,
                      "videos_kept": len(kept), "videos_rejected_by_filter": sum(1 for e in excl if e["reason"] == "alignment_filter"),
                      "video_pass_frac_hist": np.histogram([v["pass_frac"] for v in ok], bins=[0, 0.5, 0.8, 0.9, 0.95, 1.0001])[0].tolist() if ok else None},
           "kept": {"videos": len(kept), "frames": int(sum(c["frames"] for c in kept)), "identity_groups_among_kept": len(kept_groups),
                    "kept_videos_sharing_an_identity": sum(len(g) for g in kept_groups.values() if len(g) > 1)},
           "times_s_per_video": times,
           "disk_gb": {"data_ubc": round(du(O / "data/ubc") / 1e9, 3), "runs_ubc": round(du(O / "runs/ubc") / 1e9, 3)},
           "deviations_and_notes": [
               "GART's 6 UBC videos are in the official TRAIN list (not the test list as PREREG states); excluded anyway as evaluation videos and used as evaluation identities in the overlap check.",
               "Filter (b) 'max per-limb error' implemented as max over 5 limbs (head: nose+eyes; arms: shoulder/elbow/wrist; legs: hip/knee/ankle/toes/heel) of the limb's MEAN keypoint error; the stricter max over single keypoints is reported for information (heels / COCO hips are systematically off from SMPL-X by ~3-5 % of body height in this data).",
               "Shape lock is per video (PREREG UBC row), not per subject across videos: identity_group in index.json gives the ArcFace within-train groups.",
               "body_crop M_o2c-hd = identity: the 'original image' of the crop matrices is the 1024 perspective crop (GUAVA's affine box product cannot take a homography); original-frame homography in record['persp_crop'].",
               "No gt_proj.npz (no exact labels)."]}
    # ---- overlays: 10 different videos, 5 back + 5 front, mostly kept, up to 2 rejected
    out = O / "reports/ubc/overlays"; out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.jpg"):
        p.unlink()
    rng = np.random.default_rng(0)
    keptn = [c["name"] for c in kept]; rej = [e["name"] for e in excl if e["reason"] == "alignment_filter"]
    pick_k = list(rng.permutation(keptn))[:8 if rej else 10]; pick_r = list(rng.permutation(rej))[:10 - len(pick_k)]
    chosen = []
    for i, n in enumerate(pick_k + pick_r):
        tag = "back" if i % 2 == 0 else "front"
        src = O / "runs/ubc/clips" / n / f"overlay_{tag}.jpg"
        if src.exists():
            dst = out / f"{i:02d}_{'kept' if n in keptn else 'rejected'}_{tag}_{n}.jpg"; shutil.copy(src, dst); chosen.append(dst.name)
    rep["overlays"] = chosen
    gp = [p_ for p_ in ident["pairs_over_thr"] if p_["eval"].split("/")[-1] in GART6]
    rep["gart6_identity_matches"] = {"n_train_videos_matching_a_gart6_identity": len({p_["train"] for p_ in gp}),
                                     "n_of_those_matching_ONLY_gart6 (not any official test video)": len({p_["train"] for p_ in gp} - {p_["train"] for p_ in ident["pairs_over_thr"] if p_["eval"].startswith("test/")}),
                                     "per_gart_video": {g: sorted({p_["train"] for p_ in gp if p_["eval"].endswith(g)}) for g in GART6},
                                     "all_excluded": all(p_["train"] in removed for p_ in gp),
                                     "gart6_fitted_or_converted": [g for g in GART6 if (O / "data/ubc" / g).exists() or g in fit_list]}
    (O / "reports/ubc/report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("fitting", "filter", "kept", "times_s_per_video", "disk_gb")}, indent=1))


if __name__ == "__main__":
    main()
