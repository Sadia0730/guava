"""GUAVA-FB: BEDLAM2 conversion report (FORMAT.md "reports"; PREREG "GUAVA-FB" data checks).

Writes O/reports/bedlam2/report.json and O/reports/bedlam2/overlays/*.jpg:
  - clips / subjects / rows per split, disk size, licence, SMPL-X availability, views, camera motion;
  - conversion residuals (all converted rows): (a) mean vertex distance, SMPL-X 2020 (mapped betas200, same pose,
    pelvis-aligned) vs the BEDLAM2 locked-head label mesh, mm; (b) mean vertex reprojection difference at 512 px between
    the converted mesh with the record camera and the label mesh with the original camera (both in the crop);
  - independent record check on 50 rows (seed 0): the SMPL-X 2020 mesh rebuilt ONLY from the stored records
    (optim_tracking_ehm.pkl + id_share_params.pkl) and projected with an own pinhole camera
    (x_cam = V + t', t' = diag(-1,-1,1) camera_RT_params[:, 3], f = invtanfov * 512, principal point at the centre)
    vs gt_proj-style source projection (label mesh, original K, homography H) -> mean px at 512;
  - mask-vs-mesh precision on the same 50 rows: fraction of the rasterised converted mesh silhouette inside the StyleMatte
    mask (alpha > 0.5); registered target >= 0.95 (AVATAR_B_PLAN §4.1.7);
  - 10 overlays (seed 0 clips; 4 rows each): top = image crop, bottom = crop with the converted mesh (own projection,
    filled + green outline), mask outline (blue) and head / hand boxes (yellow / magenta / cyan) from the record crops.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "/home/ubx858/guava")
import convert_bedlam2 as cb                                          # noqa: E402
from smplx_lean import SMPLXLean, SMPLX2020, LOCKED, rodrigues         # noqa: E402

DATA, REP = cb.DATA, cb.REP


def lmdb_img(clip_dir, key):
    from utils.lmdb import LMDBEngine
    e = LMDBEngine(str(clip_dir / "img_lmdb"), write=False)
    im = e[f"{key}/body_image"].permute(1, 2, 0).numpy(); m = e[f"{key}/body_mask"][0].numpy(); e.close()
    return im, m


def record_mesh(N, rec, shape):
    s = rec["smplx_coeffs"]
    pose = np.concatenate([np.asarray(s["global_pose"]).reshape(1, 3), np.asarray(s["body_pose"]).reshape(21, 3),
                           np.asarray(s["jaw_pose"]).reshape(1, 3), np.asarray(s["eye_pose"]).reshape(2, 3),
                           np.asarray(s["left_hand_pose"]).reshape(15, 3), np.asarray(s["right_hand_pose"]).reshape(15, 3)], 0)
    V, _ = N(torch.tensor(pose.reshape(1, 165), dtype=torch.float64), torch.tensor(np.asarray(shape).reshape(1, 200), dtype=torch.float64),
             torch.tensor(np.asarray(s["exp"]).reshape(1, 50), dtype=torch.float64))
    return V[0].numpy()


def own_projection(V, rec, size=1024):
    RT = np.asarray(rec["smplx_coeffs"]["camera_RT_params"], np.float64)
    tp = np.diag([-1.0, -1.0, 1.0]) @ RT[:, 3]                          # camera frame (OpenCV) = V + t'
    assert np.allclose(RT[:, :3], np.diag([-1.0, -1.0, 1.0]))
    X = V + tp; f = rec["invtanfov"] * size / 2
    return f * X[:, :2] / X[:, 2:] + size / 2


def source_projection(L, clip, row):
    d = cb.job_npz(clip["job"])
    P = torch.tensor(d["pose_cam"][row][None], dtype=torch.float64); S = torch.tensor(d["shape16"][row][None], dtype=torch.float64)
    V, _ = L(P, S); X = V[0].numpy() + d["t_cam"][row]
    return X, d["K"][row].astype(np.float64)


def silhouette(v2d, faces, size=1024):
    m = np.zeros((size, size), np.uint8)
    tri = np.round(v2d[faces] * 4).astype(np.int32)                    # 2 bits subpixel
    for t in tri:                                                      # per triangle (one fillPoly call uses even-odd filling)
        cv2.fillConvexPoly(m, t, 1, lineType=cv2.LINE_8, shift=2)
    return m


def main():
    P = json.loads(cb.PLAN.read_text()); byname = {c["name"]: c for c in P["clips"]}
    metas = {}
    for c in P["clips"]:
        p = DATA / c["name"] / "meta.json"
        if p.exists():
            metas[c["name"]] = json.loads(p.read_text())
    names = sorted(metas)
    N = SMPLXLean(SMPLX2020, 200); L = SMPLXLean(LOCKED, 16); faces = N.faces
    rng = np.random.default_rng(0)
    # ---- 50 check rows (random over all converted rows, seed 0)
    allrows = [(n, k) for n in names for k in metas[n]["frames_keys"]]
    pick = [allrows[i] for i in sorted(rng.choice(len(allrows), min(50, len(allrows)), replace=False))]
    chk = []
    recs_cache = {}
    for n, k in pick:
        cd = DATA / n
        if n not in recs_cache:
            recs_cache[n] = (pickle.load(open(cd / "optim_tracking_ehm.pkl", "rb")), pickle.load(open(cd / "id_share_params.pkl", "rb")))
        recs, ids = recs_cache[n]; rec = recs[k]
        V = record_mesh(N, rec, ids["smplx_shape"][0]); v2d = own_projection(V, rec)
        row = metas[n]["rows"][metas[n]["frames_keys"].index(k)]
        X, K = source_projection(L, byname[n], row)
        # H rebuilt independently is not stored; use the converter's geometry for H (same row) and compare projections
        g = cb.clip_geometry({"rows": [row]}, cb.job_npz(byname[n]["job"]), P["subject_betas200"][byname[n]["subject"]])[0]
        q = X @ K.T; q = q / q[:, 2:]; q = q @ g["H"].T; src = q[:, :2] / q[:, 2:]
        err512 = float(np.linalg.norm(v2d - src, axis=1).mean() * 0.5)
        im, alpha = lmdb_img(cd, k)
        sil = silhouette(v2d, faces)
        msk = alpha > 127
        prec = float((sil & msk).sum() / max(sil.sum(), 1))
        rec_ = float((sil & msk).sum() / max(msk.sum(), 1))
        iou = float((sil & msk).sum() / max((sil | msk).sum(), 1))
        chk.append({"clip": n, "key": k, "reproj_px512_mean": err512, "reproj_px512_max": float(np.linalg.norm(v2d - src, axis=1).max() * 0.5),
                    "mask_precision": prec, "mask_recall": rec_, "mask_iou": iou})
    # ---- overlays: 10 clips, seed 0
    (REP / "overlays").mkdir(parents=True, exist_ok=True)
    ov = [names[i] for i in sorted(np.random.default_rng(0).choice(len(names), min(10, len(names)), replace=False))]
    for n in ov:
        cd = DATA / n; recs = pickle.load(open(cd / "optim_tracking_ehm.pkl", "rb")); ids = pickle.load(open(cd / "id_share_params.pkl", "rb"))
        ks = metas[n]["frames_keys"]; sel = [ks[i] for i in np.unique(np.round(np.linspace(0, len(ks) - 1, 4)).astype(int))]
        top, bot = [], []
        for k in sel:
            rec = recs[k]; im, alpha = lmdb_img(cd, k)
            v2d = own_projection(record_mesh(N, rec, ids["smplx_shape"][0]), rec)
            sil = silhouette(v2d, faces)
            o = im.copy().astype(np.float32)
            o[sil > 0] = 0.55 * o[sil > 0] + 0.45 * np.array([255, 120, 60])
            o = o.astype(np.uint8)
            cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cs, -1, (0, 255, 0), 2)
            cm, _ = cv2.findContours((alpha > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cm, -1, (0, 120, 255), 2)
            for part, col in (("head_crop", (255, 255, 0)), ("left_hand_crop", (255, 0, 255)), ("right_hand_crop", (0, 255, 255))):
                Mc = np.asarray(rec[part]["M_c2o"], np.float64); c4 = np.array([[0, 0, 1], [512, 512, 1.0]]) @ Mc.T
                cv2.rectangle(o, tuple(int(x) for x in c4[0, :2]), tuple(int(x) for x in c4[1, :2]), col, 2)
            top.append(cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)); bot.append(cv2.resize(o, (512, 512), interpolation=cv2.INTER_AREA))
        grid = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
        cv2.putText(grid, n, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(REP / "overlays" / f"{n}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # ---- aggregates
    pc = lambda x: {"mean": float(np.mean(x)), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
                    "p99": float(np.percentile(x, 99)), "max": float(np.max(x))} if len(x) else None
    split = {}
    for s in ("train", "val", "test"):
        ns = [n for n in names if metas[n]["split"] == s]; planned = [c for c in P["clips"] if c["split"] == s]
        split[s] = {"clips": len(ns), "subjects": len({metas[n]["subject"] for n in ns}), "rows": sum(len(metas[n]["frames_keys"]) for n in ns),
                    "planned_clips": len(planned), "planned_rows": sum(len(c["rows"]) for c in planned)}
    split["test"]["note"] = "40 test clips (AVATAR_B_PLAN §3) not converted now"
    res_mm = [x for n in names for x in metas[n]["res_mm"]]; res_px = [x for n in names for x in metas[n]["res_px512"]]
    du = subprocess.run(["du", "-sb", str(DATA)], capture_output=True, text=True).stdout.split()[0]
    mot = {}
    for n in names:
        mot[metas[n]["camera"].get("motion")] = mot.get(metas[n]["camera"].get("motion"), 0) + 1
    prec = [c["mask_precision"] for c in chk]
    vis = {p: float(np.mean([v[p] for n in names for v in metas[n]["part_visible"]])) for p in ("head", "lhand", "rhand")}
    out = {"source": "bedlam2", "licence": cb.LICENCE,
           "smplx_availability": "exact: every row has the BEDLAM2 ground-truth SMPL-X (locked-head model, 16 betas, 165-d pose_cam incl. "
                                 "jaw / eyes / 2 x 15 hand joints with flat hand mean, t_cam) and the exact pinhole intrinsics K; no expression "
                                 "(exp50 = 0). Converted to SMPLX_NEUTRAL_2020 by the linear betas map (no pose refit needed: see residuals).",
           "views": "1 camera per clip (monocular rendered video, view '0'); camera motion by clip: " + json.dumps(mot)
                    + " (translating = camera position range > 1 cm; rotating = fixed position, pan / tilt / roll > 0.5 deg; static otherwise)",
           "camera_motion_clips": mot,
           "split": split, "subjects_total": len({metas[n]["subject"] for n in names}),
           "disk_bytes": int(du), "disk_gb": round(int(du) / 1e9, 2),
           "lmdb_bytes_per_row": {"image_jpg": float(np.mean([metas[n]["bytes_img"] / len(metas[n]["frames_keys"]) for n in names])),
                                  "mask_jpg": float(np.mean([metas[n]["bytes_mask"] / len(metas[n]["frames_keys"]) for n in names]))},
           "conversion_residual": {
               "rows": len(res_mm),
               "vertex_dist_mm_2020_vs_lockedhead": pc(res_mm),
               "vertex_dist_threshold_mm": 5.0, "pose_refit_needed": bool(np.max(res_mm) >= 5.0) if res_mm else None,
               "reproj_px512_converted_vs_label": pc(res_px)},
           "record_check_50_rows": {"rows": len(chk), "reproj_px512_mean": pc([c["reproj_px512_mean"] for c in chk]),
                                    "reproj_px512_max": pc([c["reproj_px512_max"] for c in chk]),
                                    "target": "< 1 px at 512 (mean vertex error)",
                                    "pass": bool(max(c["reproj_px512_mean"] for c in chk) < 1.0) if chk else None},
           "mask_vs_mesh_50_rows": {"precision": pc(prec), "recall": pc([c["mask_recall"] for c in chk]), "iou": pc([c["mask_iou"] for c in chk]),
                                    "target_precision_mean": 0.95, "pass": bool(np.mean(prec) >= 0.95) if prec else None,
                                    "rows_below_0.95": int(sum(p < 0.95 for p in prec))},
           "part_in_crop_fraction": vis,
           "invtanfov": pc([x for n in names for x in metas[n]["invtanfov"]]),
           "crop_side_px": pc([x for n in names for x in metas[n]["crop_side_px"]]),
           "depth_m": pc([x for n in names for x in metas[n]["depth_m"]]),
           "missing_frames_rows": sum(len(metas[n]["missing_frames"]) for n in names),
           "overlays": [f"overlays/{n}.jpg" for n in ov], "check_rows": chk}
    (REP / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("check_rows", "licence", "smplx_availability")}, indent=1))


if __name__ == "__main__":
    main()
