"""GUAVA-FB / 4D-DRESS report: O/reports/dress4d/report.json + overlays/*.jpg (FORMAT.md; PREREG GUAVA-FB data checks).

Overlays are rebuilt from the WRITTEN clip only (LMDB crop + mask, optim_tracking_ehm.pkl record, id_share_params shape): the record's
SMPL-X 2020 mesh is posed with smplx lbs (no translation; root about the pelvis as GUAVA's lbs_wobeta), moved by the GUAVA camera
w2c = diag(-1,-1,1,1) [R|T] and projected with the record's invtanfov (principal point at the centre) -> independent check of the
record. Layout (512 px tiles): top row = crop | crop x StyleMatte mask; bottom row = converted mesh silhouette (green, outline) +
head / hand boxes | source mesh vertices (red, gt_proj-style projection) vs the record's (green), per-row error in the title.
Usage: python report_dress4d.py --clips O/data/dress4d --refit DIR --out O/reports/dress4d [--n-overlays 10]"""
import argparse, glob, json, os, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_dress4d import Converter, DATA, CAMS, LICENCE, split_of, CROP


def du(p):
    try:
        return int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True, timeout=7200).stdout.split()[0])
    except Exception as e:  # noqa: BLE001
        return None


def record_projection(cv, rec, shape):
    import torch
    dev = cv.dev; s = rec["smplx_coeffs"]
    pose = np.concatenate([s["global_pose"].reshape(1, 3), s["body_pose"].reshape(21, 3), s["jaw_pose"].reshape(1, 3), s["eye_pose"].reshape(2, 3),
                           s["left_hand_pose"].reshape(15, 3), s["right_hand_pose"].reshape(15, 3)])
    with torch.no_grad():
        V, _ = cv.m20(torch.as_tensor(shape, device=dev).float().reshape(1, 200), torch.as_tensor(s["exp"], device=dev).float()[None],
                      torch.as_tensor(pose, device=dev).float()[None], torch.zeros(1, 3, device=dev))
    V = V[0].cpu().numpy().astype(np.float64)
    RT = np.asarray(s["camera_RT_params"], np.float64); X = (V @ RT[:, :3].T + RT[:, 3]) * np.array([-1.0, -1.0, 1.0])
    f = rec["invtanfov"] * CROP / 2
    return np.stack([CROP / 2 + f * X[:, 0] / X[:, 2], CROP / 2 + f * X[:, 1] / X[:, 2]], 1), X[:, 2]


def overlay(cv, clip, key, path):
    import cv2, lmdb
    recs = pickle.load(open(clip / "optim_tracking_ehm.pkl", "rb")); ids = pickle.load(open(clip / "id_share_params.pkl", "rb"))
    rec = recs[key]
    env = lmdb.open(str(clip / "img_lmdb"), readonly=True, lock=False); txn = env.begin()
    dec = lambda k: cv2.cvtColor(cv2.imdecode(np.frombuffer(txn.get(f"{key}/{k}".encode()), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img, msk = dec("body_image"), dec("body_mask")[..., 0] / 255.0; env.close()
    v2d, z = record_projection(cv, rec, ids["smplx_shape"])
    gp = np.load(clip / "gt_proj.npz"); src = None
    # source mesh for this row (recompute exactly as the converter: gendered v1.1 + original camera + stored H / R_virtual)
    d = rec["dress4d"]; T = DATA / d["subject"] / d["outfit"] / d["take"]
    from fit_dress4d import read_ply_vertices
    Vs = read_ply_vertices(T / "SMPLX" / f"mesh-f{d['frame']}_smplx.ply").astype(np.float64)
    cams = pickle.load(open(T / "Capture" / "cameras.pkl", "rb")); E = cams[rec["view"]]["extrinsics"]
    Xs = (Vs @ E[:, :3].T + E[:, 3]) @ d["R_virtual"].T; f = rec["invtanfov"] * CROP / 2
    src = np.stack([CROP / 2 + f * Xs[:, 0] / Xs[:, 2], CROP / 2 + f * Xs[:, 1] / Xs[:, 2]], 1)
    err = float(np.linalg.norm(src - v2d, axis=1).mean())
    sil = cv.silhouettes(v2d[None], z[None], S=512)[0]
    S = 512; im = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA); mk = cv2.resize(msk, (S, S), interpolation=cv2.INTER_AREA)
    tl = im.copy(); tr = (im * (0.25 + 0.75 * mk[..., None])).astype(np.uint8)
    bl = im.copy(); bl[sil] = (0.5 * bl[sil] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(bl, cnts, -1, (0, 255, 0), 1)
    for part, col in (("head_crop", (255, 255, 0)), ("left_hand_crop", (255, 0, 255)), ("right_hand_crop", (0, 255, 255))):
        Mc2o = np.asarray(rec[part]["M_c2o"], np.float64); c4 = np.array([[0, 0, 1], [512, 512, 1.0]]) @ Mc2o.T * (S / CROP)
        cv2.rectangle(bl, (int(c4[0, 0]), int(c4[0, 1])), (int(c4[1, 0]), int(c4[1, 1])), col, 1)
    br = (im * 0.6).astype(np.uint8)
    for (u, v) in src[::7] * (S / CROP): cv2.circle(br, (int(u), int(v)), 1, (255, 60, 60), -1)
    for (u, v) in v2d[::7] * (S / CROP): cv2.circle(br, (int(u), int(v)), 1, (60, 255, 60), -1)
    canvas = np.concatenate([np.concatenate([tl, tr], 1), np.concatenate([bl, br], 1)], 0)
    txt = f"{clip.name} {key} view {rec['view']} invtanfov {rec['invtanfov']:.2f} rec-vs-src {err:.2f}px@1024"
    cv2.putText(canvas, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2); cv2.putText(canvas, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"clip": clip.name, "key": key, "view": rec["view"], "file": str(path), "rec_vs_src_px1024": err}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True); ap.add_argument("--refit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n-overlays", type=int, default=10)
    ap.add_argument("--no-du", action="store_true"); ap.add_argument("--timing", type=Path, default=None)
    ap.add_argument("--raw-extracted-bytes", type=int, default=None); ap.add_argument("--raw-tar-bytes", type=int, default=None)
    a = ap.parse_args(); (a.out / "overlays").mkdir(parents=True, exist_ok=True)
    import torch
    cv = Converter(a.refit, "cuda:0", matte=False)
    split, gender = split_of()
    refit = {p.stem: json.loads(p.read_text()) for p in sorted(a.refit.glob("[0-9]*.json"))}; model_check = json.loads((a.refit / "model_check.json").read_text())
    clips = sorted(p for p in a.clips.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "stats.json").exists())
    st = {c.name: json.loads((c / "stats.json").read_text()) for c in clips}
    rows = [dict(r, clip=c) for c, s in st.items() for r in s["rows_stats"]]
    agg = lambda k, rr: {"mean": float(np.mean([r[k] for r in rr])), "median": float(np.median([r[k] for r in rr])),
                         "p5": float(np.percentile([r[k] for r in rr], 5)), "min": float(np.min([r[k] for r in rr])),
                         "p95": float(np.percentile([r[k] for r in rr], 95)), "max": float(np.max([r[k] for r in rr]))} if rr else None
    rng = np.random.default_rng(0); r50 = [rows[i] for i in sorted(rng.choice(len(rows), min(50, len(rows)), replace=False))]
    per_split = {}
    for sp in ("train", "val"):
        cs = [s for s in st.values() if split[s["subject"]] == sp]
        per_split[sp] = {"subjects": sorted({s["subject"] for s in cs}), "n_subjects": len({s["subject"] for s in cs}),
                         "genders": {g: len({s["subject"] for s in cs if s["gender"] == g}) for g in ("female", "male")},
                         "outfits": len({(s["subject"], s["outfit"]) for s in cs}), "takes": len(cs),
                         "scan_frames_used": int(sum(s["frames"] for s in cs)), "rows": int(sum(s["rows"] for s in cs)), "views": CAMS}
    # README table vs extracted
    readme_takes = 520; ext_takes = len(glob.glob(str(DATA / "*" / "*" / "Take*")))
    # overlays: 10 rows over different subjects / both splits / all 4 cameras (0028 / 0076 = side / back-side views)
    pick = []
    names = sorted(st); cam_cycle = CAMS * 3
    sel_clips = [names[int(i)] for i in np.linspace(0, len(names) - 1, a.n_overlays).round()]
    for i, cn in enumerate(sel_clips):
        keys = [r["key"] for r in st[cn]["rows_stats"] if r["key"].startswith(cam_cycle[i])]
        pick.append((cn, keys[len(keys) // 2]))
    ovs = [overlay(cv, a.clips / cn, k, a.out / "overlays" / f"{i:02d}_{cn}_{k}.jpg") for i, (cn, k) in enumerate(pick)]
    timing = None
    if a.timing and a.timing.exists(): timing = json.loads(a.timing.read_text())
    tm = [s["timing_s"] for s in st.values()]
    rep = {
        "source": "dress4d", "generated": time.strftime("%Y-%m-%d %H:%M"),
        "licence": LICENCE, "licence_file_in_download": False,
        "size_on_disk_bytes": None if a.no_du else {"raw_extracted": a.raw_extracted_bytes or du(DATA), "raw_tarballs": a.raw_tar_bytes or du(DATA.parent / "4D-DRESS"), "converted": du(a.clips)},
        "dataset": {"subjects": len(gender), "female": sum(v == "female" for v in gender.values()), "male": sum(v == "male" for v in gender.values()),
                    "takes_extracted": ext_takes, "takes_readme": readme_takes, "cameras": CAMS, "image_size_hw": [1280, 940],
                    "frame_stride": 5},
        "split_rule": "val = 2 female + 2 male subjects with the lowest SHA-1(subject id string) within each gender; never in train",
        "split": per_split,
        "smplx_availability": {
            "source_model": "SMPL-X v1.1 gendered (smplx_v1_1 SMPLX_MALE / SMPLX_FEMALE.npz, gender = basic_info.pkl), 10 betas, 10 expression, "
                            "12 hand PCA comps with flat_hand_mean=False (smplx package default hand mean added); all frames of all takes",
            "reproduces_provided_ply_max_abs_m": max(v["max_abs_m"] for v in model_check.values()),
            "frame": "ply / params in the world frame of Capture/cameras.pkl (extrinsics: world -> OpenCV camera); basic_info rotation not applied, offset None",
            "model_check_per_subject": model_check},
        "refit": {"target": "SMPLX_NEUTRAL_2020 (200 shape, 50 exp, full axis-angle hands), one shape per subject, per-frame poses / exp / transl",
                  "check": "per subject median over frames of mean vertex distance < 5 mm (registered)",
                  "per_subject": refit, "dropped": sorted(s for s, v in refit.items() if not v["pass"]),
                  "median_of_subject_medians_mm": float(np.median([v["median_mean_mm"] for v in refit.values()])),
                  "worst_subject_median_mm": float(max(v["median_mean_mm"] for v in refit.values()))},
        "mask_iou_stylematte_vs_provided": {"rows_50_seed0": agg("iou_provided", r50), "all_rows": agg("iou_provided", rows),
                                            "note": "provided masks are renders of the textured scan (incl. clothing / hair) warped with the same homography; StyleMatte alpha > 0.5"},
        "mask_vs_mesh_precision": {"target": 0.95, "all_rows": agg("mesh_prec", rows), "frac_rows_below_0.95": float(np.mean([r["mesh_prec"] < 0.95 for r in rows])),
                                   "pass": bool(np.mean([r["mesh_prec"] for r in rows]) >= 0.95), "recall_all_rows": agg("mesh_rec", rows),
                                   "definition": "refit SMPL-X 2020 silhouette (pytorch3d, 256 px) pixels inside StyleMatte alpha > 0.5 / silhouette pixels, over crop pixels inside the original image"},
        "record_reprojection_px1024": {"record_vs_source_mesh": agg("reproj_rec_vs_src_px1024", rows),
                                       "record_vs_refit_mesh": agg("reproj_rec_vs_refit_px1024", rows),
                                       "note": "GUAVA camera formula (w2c = diag(-1,-1,1,1)[R|T], invtanfov) on the written record; vs source = refit residual in px; "
                                               "the GUAVA-renderer check (< 1 px at 512) is the coordinator's, from gt_proj.npz"},
        "mask_vs_mesh_precision_incl_outside_image": agg("mesh_prec_incl_outside_image", rows),
        "truncation": {"definition": "fraction of source-mesh vertices projecting outside the original 1280 x 940 image (record['dress4d']['trunc_frac'])",
                       "rows_trunc_gt_0": float(np.mean([r["trunc_frac"] > 0 for r in rows])), "rows_trunc_gt_0.05": float(np.mean([r["trunc_frac"] > 0.05 for r in rows])),
                       "rows_trunc_gt_0.2": float(np.mean([r["trunc_frac"] > 0.2 for r in rows])), "stats": agg("trunc_frac", rows)},
        "invtanfov": agg("invtanfov", rows),
        "source_mesh_inside_crop": agg("src_in_crop", rows),
        "overlays": ovs,
        "timing": {"per_take_s_mean": {k: float(np.mean([t[k] for t in tm])) for k in tm[0]} if tm else None, "run": timing},
        "rows_total": len(rows), "clips_total": len(st),
    }
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("split", "mask_iou_stylematte_vs_provided", "mask_vs_mesh_precision", "record_reprojection_px1024")}, indent=1))


if __name__ == "__main__":
    main()
