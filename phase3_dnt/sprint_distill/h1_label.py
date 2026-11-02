"""GUAVA-FB improvement phase 4, H4 (PREREG): SPRINT's own UBody training rows fitted offline with H1 (the k = 60 driving
refinement of sprint_avatar_fit: DWPose keypoints + StyleMatte matte + priors) -> distillation labels for the SPRINT trainer
(third_party/PEAR dataset/ubody_label_dataset.py options `rows` / `h1_labels`). No PEAR weights, outputs or features.

  plan   rows to label: SPRINT's `ubody_labels` TRAIN split; identities (ub_identity) of ValBench-v1 / ValBench-S / AvatarTest-v1
         excluded; clips in seeded random order (20261102), up to --per-clip evenly spaced rows each, until --n rows
  label  worker (--shard i --shards n): per planned row, the stored 512 px context crop and the row's UBody box mapped into it,
         SprintFitter.fit(iters=60, cam_outer=(context affine, frame W, H)): H1's objective and weights unchanged, SPRINT's camera
         input as deployed (original frame), the 20 shape offsets free (a training row has no source frame) -> shard npz
  merge  filters F1-F4 and per-joint validity (PREREG H4), 55 kinematic joints of the trainer's own SMPL-X projected with the
         fitted camera into context px -> rows.npy, h1_labels.npz, summary.json
Usage: h1_label.py plan --out DIR | label --out DIR --shard i --shards n [--limit k] | merge --out DIR"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANN = Path("/raid/ubx858/outputs/dnt/data/ubody_labels")
SPRINT = Path("/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
SPLITS = ROOT / "phase3_dnt/eval_split"
CTX, CROP, KP_THR = 512, 1024, 0.3
OP_BODY = np.arange(14)                       # OpenPose 0-13 in the 134 layout (nose ... left ankle); 14-17 eyes / ears
KNEES_ANKLES = [9, 10, 12, 13]
LOWER = [1, 2, 4, 5, 7, 8, 10, 11]            # SMPL-X hips, knees, ankles, feet (rotation and joint indices)


def train_arrays():
    d = np.load(ANN / "annotations.npz", allow_pickle=True); m = d["split"] == "train"
    return {k: d[k][m] for k in ("path", "box", "base_side", "img_wh")}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def plan(a):
    sys.path.insert(0, str(SPLITS)); import build_avatar_test_split as B
    t = train_arrays(); clip = np.array([p.split("/")[3] for p in t["path"]])
    excl = sorted({c["identity"] for f in ("valbench_v1.json", "valbench_s.json", "avatar_test_v1.json")
                   for c in json.load(open(SPLITS / f))["clips"] if c["dataset"] == "ubody"})
    by_clip = collections.defaultdict(list)
    for i, (p, c) in enumerate(zip(t["path"], clip)):
        if B.ub_identity(c) not in excl:
            by_clip[c].append((int(Path(p).stem.split("_")[0]), i))
    keys = sorted(by_clip); rows, used = [], []
    for j in np.random.default_rng(20261102).permutation(len(keys)):
        it = sorted(by_clip[keys[j]]); k = min(len(it), a.per_clip)
        rows += [it[s][1] for s in np.unique(np.round(np.linspace(0, len(it) - 1, k)).astype(int))]; used.append(keys[j])
        if len(rows) >= a.n:
            break
    rows = sorted(rows[:a.n])
    out = {"rows": [int(r) for r in rows], "n_rows": len(rows), "n_clips": len(used), "per_clip": a.per_clip, "seed": 20261102,
           "excluded_identities": excl, "excluded_rows": int(sum(B.ub_identity(c) in excl for c in clip)),
           "annotations_sha256": sha256(ANN / "annotations.npz"), "created": time.strftime("%F %T")}
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "plan.json").write_text(json.dumps(out))
    print({k: v for k, v in out.items() if k not in ("rows", "excluded_identities")}, "excluded identities", len(excl))


FIELDS = {"pose156": (156,), "raw156": (156,), "jaw": (3,), "exp": (50,), "shape": (200,), "t": (3,), "R": (3, 3), "M": (3, 3),
          "kp": (134, 2), "sc": (134,), "uv_after": (134, 2), "kp_err": (2,), "loss_before": (4,), "loss_after": (4,), "sec": (1,)}
LOSSES = ("kp", "mask_in", "mask_cov", "prior")


def label(a):
    P = json.loads((Path(a.out) / "plan.json").read_text())
    rows = P["rows"][a.shard::a.shards][: a.limit or None]
    shard = Path(a.out) / "shards" / f"shard_{a.shard:02d}_of_{a.shards:02d}{'_smoke' if a.limit else ''}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if shard.exists():                                                                    # resume
        z = np.load(shard); done = {int(r): {k: z[k][i] for k in FIELDS} for i, r in enumerate(z["row"])}
    spec = importlib.util.spec_from_file_location("sprint_avatar_fit", ROOT / "phase3_dnt/sprint_avatar_fit.py")
    SAF = importlib.util.module_from_spec(spec); spec.loader.exec_module(SAF)
    fitter = SAF.SprintFitter(SPRINT, "cuda:0", 150, dwpose_threads=a.threads)
    t = train_arrays(); t0 = time.time(); n_new = 0

    def save():
        rr = sorted(done); tmp = shard.with_suffix(".tmp.npz")
        np.savez(tmp, row=np.array(rr, np.int64), **{k: np.stack([done[r][k] for r in rr]).astype(np.float32) for k in FIELDS})
        os.replace(tmp, shard)

    for r in rows:
        if r in done:
            continue
        img = cv2.cvtColor(cv2.imread(str(ANN / t["path"][r])), cv2.COLOR_BGR2RGB)
        bx, by, bw, bh = [float(v) for v in t["box"][r]]; s = CTX / (2.0 * float(t["base_side"][r]))
        C = np.array([[s, 0, CTX / 2 - s * (bx + bw / 2)], [0, s, CTX / 2 - s * (by + bh / 2)], [0, 0, 1]], np.float64)
        box = [CTX / 2 - s * bw / 2, CTX / 2 - s * bh / 2, CTX / 2 + s * bw / 2, CTX / 2 + s * bh / 2]
        W0, H0 = [float(v) for v in t["img_wh"][r]]
        ts = time.perf_counter()
        res = fitter.fit(img, box=box, iters=60, cam_outer=(C, W0, H0))
        sx, rw = res["record"]["smplx_coeffs"], res["raw_smplx"]
        p156 = lambda q: np.concatenate([q["global_pose"].reshape(3), q["body_pose"].reshape(63), q["left_hand_pose"].reshape(45),
                                         q["right_hand_pose"].reshape(45)])
        RT = sx["camera_RT_params"]; dg = res["diag"]
        done[r] = {"pose156": p156(sx), "raw156": p156(rw), "jaw": sx["jaw_pose"].reshape(3), "exp": sx["exp"][:50], "shape": res["shape"][:200],
                   "t": RT[:, 3], "R": RT[:, :3], "M": res["M"], "kp": res["record"]["dwpose_raw"]["keypoints"], "sc": res["kp_score"],
                   "uv_after": res["uv_after"], "kp_err": np.array([dg["kp_err_px1024_before"], dg["kp_err_px1024_after"]]),
                   "loss_before": np.array([dg["loss_before"][k] for k in LOSSES]), "loss_after": np.array([dg["loss_after"][k] for k in LOSSES]),
                   "sec": np.array([time.perf_counter() - ts])}
        n_new += 1
        if n_new % 500 == 0:
            save(); print(f"shard {a.shard}: {len(done)} / {len(rows)} rows, {n_new / (time.time() - t0):.2f} rows/s", flush=True)
    save()
    print(json.dumps({"shard": a.shard, "rows": len(done), "new": n_new, "rows_per_s": n_new / max(time.time() - t0, 1e-9),
                      "median_s_per_row": float(np.median([done[r]["sec"][0] for r in done]))}), flush=True)


def kinematic_joints(pose156, shape, device="cpu"):
    """55 kinematic SMPL-X joints (model frame) of the SPRINT trainer's own model (student_gt_losses.joints55)."""
    import torch
    pear = ROOT / "third_party/PEAR"; cwd = Path.cwd(); os.chdir(pear); sys.path.insert(0, str(pear))
    import student_gt_losses as L
    from dataset.bedlam2_gt_dataset import axis_angle_to_matrix
    model = L.kinematic_smplx(str(pear / "assets/SMPLX"), device)
    out = []
    with torch.no_grad():
        for i in range(0, len(pose156), 512):
            rot = torch.from_numpy(np.stack([axis_angle_to_matrix(p.reshape(52, 3)) for p in pose156[i:i + 512]]).astype(np.float32))
            out.append(L.joints55(model, rot.to(device), torch.from_numpy(shape[i:i + 512].astype(np.float32)).to(device)).cpu().numpy())
    os.chdir(cwd)
    return np.concatenate(out)


def merge(a):
    P = json.loads((Path(a.out) / "plan.json").read_text()); rows = np.array(P["rows"])
    Z = {}
    for f in sorted((Path(a.out) / "shards").glob("shard_*_of_*.npz")):
        if "_smoke" in f.name or ".tmp" in f.name:
            continue
        z = np.load(f)
        for i, r in enumerate(z["row"]):
            Z[int(r)] = {k: z[k][i] for k in FIELDS}
    miss = [int(r) for r in rows if int(r) not in Z]
    assert not miss or a.allow_missing, f"{len(miss)} planned rows not labelled"
    rows = np.array([r for r in rows if int(r) in Z]); A = {k: np.stack([Z[int(r)][k] for r in rows]) for k in FIELDS}
    n = len(rows)
    # keypoints inside the 1024 deployment crop (normalised) and confident
    kc = np.einsum("nij,nkj->nki", A["M"][:, :2, :2], A["kp"]) + A["M"][:, None, :2, 2]
    ins = ((kc >= 0) & (kc <= CROP)).all(-1); good = (A["sc"] >= KP_THR) & ins
    f1 = good[:, OP_BODY].sum(1) >= 6
    f2 = A["kp_err"][:, 1] <= A["kp_err"][:, 0]
    q_kp = float(np.percentile(A["kp_err"][:, 1], 90)); f3 = A["kp_err"][:, 1] <= q_kp
    q_in, q_cov = [float(np.percentile(A["loss_after"][:, LOSSES.index(k)], 95)) for k in ("mask_in", "mask_cov")]
    f4 = (A["loss_after"][:, 1] <= q_in) & (A["loss_after"][:, 2] <= q_cov)
    keep = f1 & f2 & f3 & f4
    # per-joint validity (DWPose 92-112 / 113-133 = the person's left / right hand, as SMPL-X 25-39 / 40-54: checked in the smoke test)
    lh, rh = good[:, 92:113].sum(1) >= 15, good[:, 113:134].sum(1) >= 15
    lower = good[:, KNEES_ANKLES].all(1); face = good[:, [0, 14, 15]].all(1)
    jm = np.ones((n, 52), np.float32); jm[:, 22:37] *= lh[:, None]; jm[:, 37:52] *= rh[:, None]; jm[:, LOWER] *= lower[:, None]
    ok = np.ones((n, 55), bool); ok[:, LOWER] &= lower[:, None]; ok[:, 22:25] &= face[:, None]; ok[:, 25:40] &= lh[:, None]; ok[:, 40:55] &= rh[:, None]
    # 2D targets: kinematic joints through the fitted crop camera (focal 24 over the 1024 crop), then crop -> context px
    J = kinematic_joints(A["pose156"], A["shape"])
    cam = np.einsum("nkj,nij->nki", J, A["R"]) + A["t"][:, None]
    uv = (1.0 - 24.0 * cam[..., :2] / np.clip(cam[..., 2:3], 1e-4, None)) * 0.5 * CROP
    Minv = np.linalg.inv(A["M"]); j2d = np.einsum("nij,nkj->nki", Minv[:, :2, :2], uv) + Minv[:, None, :2, 2]
    # sanity: kinematic elbows / wrists vs the fit's own DWPose-layout points (OpenPose 3, 4, 6, 7 = R elbow, R wrist, L elbow, L wrist)
    dwp = A["uv_after"][:, [3, 4, 6, 7]] * CROP; kin = uv[:, [19, 21, 18, 20]]
    sanity = float(np.median(np.linalg.norm(dwp - kin, axis=-1)))
    # how far H1 moved SPRINT's raw prediction (degrees, mean over joints) on kept rows
    from scipy.spatial.transform import Rotation as Rot
    def ang(p, q):
        return np.degrees((Rot.from_rotvec(p.reshape(-1, 3)).inv() * Rot.from_rotvec(q.reshape(-1, 3))).magnitude()).reshape(len(p), -1)
    dr = ang(A["raw156"][keep], A["pose156"][keep])
    idx = rows[keep]
    np.save(Path(a.out) / "rows.npy", idx.astype(np.int64))
    np.savez(Path(a.out) / "h1_labels.npz", idx=idx.astype(np.int64), pose156=A["pose156"][keep].astype(np.float32), jm52=jm[keep],
             j2d_ctx=j2d[keep].astype(np.float32), j2d_ok=ok[keep], shape200=A["shape"][keep].astype(np.float32), t=A["t"][keep], M=A["M"][keep])
    S = {"planned": int(len(P["rows"])), "labelled": n, "missing": len(miss), "kept": int(keep.sum()),
         "dropped": {"F1_body_kp": int((~f1).sum()), "F2_kp_err_up": int((~f2).sum()), "F3_kp_err_p90": int((~f3).sum()), "F4_mask_p95": int((~f4).sum())},
         "thresholds": {"kp_err_p90_px1024": q_kp, "mask_in_p95": q_in, "mask_cov_p95": q_cov},
         "validity_kept": {"left_hand": float(lh[keep].mean()), "right_hand": float(rh[keep].mean()), "lower_body": float(lower[keep].mean()), "face": float(face[keep].mean())},
         "kp_err_px1024_kept": {"before_median": float(np.median(A["kp_err"][keep, 0])), "after_median": float(np.median(A["kp_err"][keep, 1]))},
         "h1_minus_raw_deg_kept": {"root": float(dr[:, 0].mean()), "body": float(dr[:, 1:22].mean()), "hands": float(dr[:, 22:].mean())},
         "sanity_kinematic_vs_dwpose_layout_elbow_wrist_px1024_median": sanity,
         "sec_per_row_median": float(np.median(A["sec"])), "h1_labels_sha256": sha256(Path(a.out) / "h1_labels.npz"),
         "rows_sha256": sha256(Path(a.out) / "rows.npy"), "created": time.strftime("%F %T")}
    (Path(a.out) / "summary.json").write_text(json.dumps(S, indent=1)); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "label", "merge")); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24000); ap.add_argument("--per-clip", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    {"plan": plan, "label": label, "merge": merge}[a.cmd](a)
