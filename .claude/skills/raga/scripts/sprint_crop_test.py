"""Can SPRINT drive from the whole camera frame instead of the detector crop?

Runs SPRINT (k = 0, live settings: the session's reference shape, focal = frame width) on every frame of a recorded
session with three input crops, and scores each one against DWPose:

  yolox  : the live rule, square 1.1 x the recorded YOLOX box (exactly what the live run used)
  full   : the whole 1280x720 frame, padded to a 1280 square (no detector)
  center : a fixed 720x720 square, full frame height, frame centre (no detector)

Accuracy = 2D distance between SPRINT's projected keypoints and DWPose's (DWPose on the YOLOX box, the same reference for
all three), in frame pixels. Also: how far each crop's joint rotations are from the yolox crop's (degrees).

usage: python sprint_crop_test.py REC_DIR OUT_DIR   (guava env python; ~1 min for 333 frames)
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PKG = Path("/data/GUAVA/guava_live"); P3 = PKG / "phase3_dnt"
REC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SPRINT_EVAL_ASSETS", str(PKG / "live_assets" / "eval_assets"))
spec = importlib.util.spec_from_file_location("sprint_avatar_fit", P3 / "sprint_avatar_fit.py")
SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
sess = json.loads((REC / "session.json").read_text())
fitter = SAF.SprintFitter(sess["sprint_ckpt"], "cuda:0", 150, dwpose_threads=16)

import roma   # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(P3 / "worker6"))
import worker_core as WC  # noqa: E402

dev = torch.device("cuda:0"); shp = np.asarray(sess["shape"], np.float32)
R0 = WC.root_cam({k: np.asarray(v) for k, v in sess["source_raw"].items()})
frames = [json.loads(line) for line in open(REC / "frames.jsonl")]
W, H = 1280, 720
MODES = ("yolox", "full", "center")
GROUPS = {"body": np.arange(0, 14), "face": np.arange(41, 92), "hands": np.arange(92, 134)}
dyn = fitter.lean.dyn.cpu().numpy(); wk = SAF.kp_weights()
BODY_EDGES = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0)]


def crop_matrix(mode, box):
    if mode == "yolox":
        return SAF.box_crop_matrix(box, SAF.CROP)
    if mode == "full":
        return SAF.box_crop_matrix([0, 0, W, H], SAF.CROP, scale=1.0)
    return SAF.box_crop_matrix([W / 2 - H / 2, 0, W / 2 + H / 2, H], SAF.CROP, scale=1.0)


def draw(img, box, kp, valid, p, label):
    """Panel: the YOLOX crop region (same zoom for every mode), DWPose white, SPRINT body green / hands yellow / face cyan."""
    Mv = SAF.box_crop_matrix(box, 384)
    pan = cv2.warpAffine(img, Mv[:2], (384, 384), flags=cv2.INTER_AREA)
    tr = lambda q: (np.c_[q, np.ones(len(q))] @ Mv.T)[:, :2]  # noqa: E731
    k2, p2 = tr(kp), tr(p)
    for a, b in BODY_EDGES:
        if valid[a] and valid[b]:
            cv2.line(pan, tuple(map(int, k2[a])), tuple(map(int, k2[b])), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(pan, tuple(map(int, p2[a])), tuple(map(int, p2[b])), (0, 230, 0), 2, cv2.LINE_AA)
    for g, col in (("face", (0, 220, 255)), ("hands", (255, 220, 0))):
        for j in GROUPS[g]:
            cv2.circle(pan, tuple(map(int, p2[j])), 2, col, -1, cv2.LINE_AA)
            if valid[j]:
                cv2.circle(pan, tuple(map(int, k2[j])), 1, (255, 255, 255), -1)
    cv2.rectangle(pan, (0, 0), (384, 26), (0, 0, 0), -1)
    cv2.putText(pan, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return pan


rows, grid = [], []
VIS_AT = set(int(v) for v in np.linspace(0, len(frames) - 1, 6)[1:-1])
t_all = time.perf_counter()
for n, e in enumerate(frames):
    img = cv2.cvtColor(cv2.imread(str(REC / "frames" / f"{e['i']:06d}.jpg")), cv2.COLOR_BGR2RGB)
    assert img.shape[:2] == (H, W), img.shape
    box = np.asarray(e["box"], np.float64); hgt = box[3] - box[1]
    _, kp, sc = fitter.dwpose(img, box)
    valid = (sc >= SAF.KP_THR) & ~dyn & (wk > 0) & (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H)
    view = float(np.degrees(np.linalg.norm(cv2.Rodrigues(WC.root_cam({k: np.asarray(v) for k, v in e["raw"].items()}) @ R0.T)[0])))
    row = {"i": e["i"], "view_deg": view, "person_h_px": hgt}
    poses, panels = {}, []
    for m in MODES:
        M = crop_matrix(m, box)
        crop = cv2.warpAffine(img, M[:2], (SAF.CROP, SAF.CROP), flags=cv2.INTER_LINEAR)
        ct = torch.from_numpy(crop).to(dev).permute(2, 0, 1)[None].float() / 255.0
        cam = fitter.ev.cam_feat((np.diag([256 / SAF.CROP, 256 / SAF.CROP, 1.0]) @ M)[:2], W, H, W, 0.0)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = fitter.sprint(ct, cam, shp)
        torch.cuda.synchronize(); row[f"{m}_sprint_ms"] = (time.perf_counter() - t0) * 1000
        bp = out["body_param"]
        with torch.no_grad():
            _, K = fitter.mesh(bp)
            uv = fitter.project(out["pd_cam"][0, :3, :3], out["pd_cam"][0, :3, 3], K.reshape(-1, 3)).cpu().numpy()
        p = (np.c_[uv * SAF.CROP, np.ones(len(uv))] @ np.linalg.inv(M).T)[:, :2]     # frame px
        err = np.linalg.norm(p - kp, axis=1)
        for g, idx in GROUPS.items():
            sel = idx[valid[idx]]
            row[f"{m}_{g}_px"] = float(err[sel].mean()) if len(sel) else float("nan")
        row[f"{m}_person_px_in_input"] = 256 * hgt * M[0, 0] / SAF.CROP             # person height in SPRINT's 256 input
        poses[m] = ({k: bp[k].reshape(-1, 3, 3) for k in ("body_pose", "left_hand_pose", "right_hand_pose")}, p)
        if n in VIS_AT:
            panels.append(draw(img, box, kp, valid, p, f"{m}: body {row[f'{m}_body_px']:.0f} px, hands {row[f'{m}_hands_px']:.0f} px"))
    for m in MODES[1:]:
        for k in ("body_pose", "left_hand_pose", "right_hand_pose"):
            rel = poses[m][0][k] @ poses["yolox"][0][k].transpose(-1, -2)
            row[f"{m}_vs_yolox_{k}_deg"] = float(torch.rad2deg(roma.rotmat_to_rotvec(rel).norm(dim=-1)).mean())
    rows.append(row)
    if panels:
        grid.append(np.concatenate(panels, 1))
    if n % 50 == 0:
        print(f"{n}/{len(frames)} frames, {time.perf_counter() - t_all:.0f} s", flush=True)

cv2.imwrite(str(OUT / "fullframe_vs_crop.jpg"), cv2.cvtColor(np.concatenate(grid, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
json.dump(rows, open(OUT / "per_frame.json", "w"), indent=0)


def stats(sel, name):
    print(f"\n{name}: {len(sel)} frames")
    print(f"{'crop':<8}{'person px in 256 input':>24}{'body px':>10}{'face px':>10}{'hands px':>10}{'SPRINT ms':>11}"
          f"{'body vs yolox':>15}{'hands vs yolox':>16}")
    out = {}
    for m in MODES:
        g = lambda k: np.nanmean([r[k] for r in sel])  # noqa: E731
        d = (f"{g(f'{m}_vs_yolox_body_pose_deg'):13.1f}°" f"{(g(f'{m}_vs_yolox_left_hand_pose_deg') + g(f'{m}_vs_yolox_right_hand_pose_deg')) / 2:14.1f}°"
             if m != "yolox" else f"{'-':>14}{'-':>16}")
        print(f"{m:<8}{g(f'{m}_person_px_in_input'):>24.0f}{g(f'{m}_body_px'):>10.1f}{g(f'{m}_face_px'):>10.1f}{g(f'{m}_hands_px'):>10.1f}"
              f"{np.median([r[f'{m}_sprint_ms'] for r in sel]):>11.1f}{d}")
        out[m] = {k: float(g(f"{m}_{k}")) for k in ("body_px", "face_px", "hands_px", "person_px_in_input")}
    return out


res = {"all": stats(rows, "all frames"), "front": stats([r for r in rows if r["view_deg"] < 30], "front (< 30 deg from reference)"),
       "person_h_px_median": float(np.median([r["person_h_px"] for r in rows]))}
json.dump(res, open(OUT / "summary.json", "w"), indent=1)
print(f"\nperson height in the frame (median): {res['person_h_px_median']:.0f} px; total {time.perf_counter() - t_all:.0f} s")
print("grid:", OUT / "fullframe_vs_crop.jpg")
