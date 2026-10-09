#!/usr/bin/env python
"""Checkpoint 2, section 3: BEDLAM2 camera -> our crop camera, verified on real frames.

Read-only. For random labelled frames of single-person jobs:
- ground truth mesh in OpenCV camera coordinates: SMPL-X(pose_cam, shape) + trans_cam + cam_ext[:3, 3];
- crop: projected-mesh box, square x1.25, 256x256 (no jitter);
- A: true projection (BEDLAM2 intrinsics) mapped into the crop;
- B: our crop camera (R = diag(-1, -1, 1), focal 24 over [0,1]^2, i.e. 3072 px per 256 px) on the
     body-frame mesh (pose_cam orientation, no translation) with the analytic translation below;
- C: our crop camera with the per-frame least-squares translation (lower bound for this camera model).

Analytic translation: with t = trans_cam + ext_t, a = 256 / box side, k = a * f / t_z,
T_z = 3072 / k, T_x = -t_x - (a (c_x - x0) - 128) / k, T_y = -t_y - (a (c_y - y0) - 128) / k.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

import c2_geometry_conventions as g

FRAMES = Path("/raid/ubx858/datasets/expanded/BEDLAM")
OUT = Path("/raid/ubx858/outputs/phase1_task1/checkpoint2/camera_check")
FOCAL = 24.0
SIZE = 256.0


def our_projection(x_body, t_ours):
    cam = np.stack([-x_body[:, 0] + t_ours[0], -x_body[:, 1] + t_ours[1], x_body[:, 2] + t_ours[2]], -1)
    ndc = FOCAL * cam[:, :2] / cam[:, 2:3]
    return (1.0 - ndc) * 0.5 * SIZE


def analytic_translation(t, K, box):
    x0, y0, side = box
    a = SIZE / side
    k = a * K[0, 0] / t[2]
    return np.array([-t[0] - (a * (K[0, 2] - x0) - 128.0) / k,
                     -t[1] - (a * (K[1, 2] - y0) - 128.0) / k,
                     FOCAL * 128.0 / k])


def least_squares_translation(x_body, target, init):
    t = torch.tensor(init, dtype=torch.float64, requires_grad=True)
    xb = torch.as_tensor(x_body, dtype=torch.float64)
    tgt = torch.as_tensor(target, dtype=torch.float64)
    opt = torch.optim.LBFGS([t], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        cam = torch.stack([-xb[:, 0] + t[0], -xb[:, 1] + t[1], xb[:, 2] + t[2]], -1)
        proj = (1.0 - FOCAL * cam[:, :2] / cam[:, 2:3]) * 0.5 * SIZE
        loss = ((proj - tgt) ** 2).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return t.detach().numpy()


def main():
    rng = np.random.default_rng(7)
    jobs = sorted(p.name for p in FRAMES.iterdir() if (p / "mp4").is_dir())
    model = g.build(True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    while len(rows) < 50:
        job = rng.choice(jobs)
        d = np.load(g.LABELS / f"{job}.npz", allow_pickle=True)
        i = int(rng.integers(len(d["imgname"])))
        seq, name = d["imgname"][i].split("/")
        frame_no = int(name.split("_")[-1].split(".")[0])
        cap = cv2.VideoCapture(str(FRAMES / job / "mp4" / f"{seq}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, image = cap.read()
        cap.release()
        if not ok:
            continue
        K, ext = d["cam_int"][i], d["cam_ext"][i]
        t = d["trans_cam"][i] + ext[:3, 3]
        with torch.no_grad():
            out = g.forward(model, d["pose_cam"][i][None], d["shape"][i][None], np.zeros((1, 3)))
        x_body = out.vertices[0].numpy().astype(np.float64)
        joints_body = out.joints[0, :55].numpy().astype(np.float64)
        uv = g.project((x_body + t)[None], K[None])[0]
        lo, hi = uv.min(0), uv.max(0)
        side = 1.25 * max(hi - lo)
        centre = 0.5 * (lo + hi)
        box = (centre[0] - side / 2, centre[1] - side / 2, side)
        a = SIZE / side
        to_crop = lambda p: (p - [box[0], box[1]]) * a  # noqa: E731
        crop_true = to_crop(uv)
        joints_true = to_crop(g.project((joints_body + t)[None], K[None])[0])
        t_an = analytic_translation(t, K, box)
        crop_an = our_projection(x_body, t_an)
        t_ls = least_squares_translation(x_body, crop_true, t_an)
        crop_ls = our_projection(x_body, t_ls)
        e_an = np.linalg.norm(crop_an - crop_true, axis=-1)
        e_ls = np.linalg.norm(crop_ls - crop_true, axis=-1)
        ej_an = np.linalg.norm(our_projection(joints_body, t_an) - joints_true, axis=-1)
        off_axis_deg = float(np.degrees(np.arctan2(np.hypot(*(centre - K[:2, 2])), K[0, 0])))
        rows.append({"job": job, "imgname": d["imgname"][i], "fx": float(K[0, 0]), "depth_m": float(t[2]),
                     "box_side_px": float(side), "off_axis_deg": off_axis_deg,
                     "analytic_mean_px": float(e_an.mean()), "analytic_max_px": float(e_an.max()),
                     "analytic_joints55_mean_px": float(ej_an.mean()),
                     "lsq_mean_px": float(e_ls.mean()), "lsq_max_px": float(e_ls.max()),
                     "T_analytic": t_an.round(4).tolist(), "T_lsq": t_ls.round(4).tolist()})
        if len(rows) <= 10:
            crop_img = cv2.warpAffine(image, np.float32([[a, 0, -box[0] * a], [0, a, -box[1] * a]]), (256, 256))
            big = cv2.resize(crop_img, (512, 512))
            for p in crop_true[::40]:
                cv2.circle(big, tuple(int(v) for v in p * 2), 2, (0, 255, 0), -1)
            for p in crop_an[::40]:
                cv2.circle(big, tuple(int(v) for v in p * 2), 2, (0, 0, 255), -1)
            cv2.putText(big, f"green=BEDLAM2 cam  red=our crop cam  mean {e_an.mean():.2f}px", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(OUT / f"overlay_{len(rows):02d}_{job}_{seq}_{frame_no:04d}.jpg"), big)

    keys = ["analytic_mean_px", "analytic_max_px", "analytic_joints55_mean_px", "lsq_mean_px", "lsq_max_px",
            "off_axis_deg", "depth_m", "box_side_px"]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows])),
                   "max": float(np.max([r[k] for r in rows]))} for k in keys}
    corr = float(np.corrcoef([r["off_axis_deg"] for r in rows], [r["analytic_mean_px"] for r in rows])[0, 1])
    summary["corr_offaxis_vs_analytic_err"] = corr
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
