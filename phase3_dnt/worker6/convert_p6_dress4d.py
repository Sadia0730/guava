"""PREREG phase 6: 4D-DRESS validation replay sessions in AvatarTest-v1's scoring format.
Session = all val takes of one (subject, outfit) in take order, one static camera (0004 or 0052), raw 30 fps frames.
  frame key  frame_<take * 1000 + n> (n = capture-f<n>.png); keys in session (time) order
  source     the session's first frame; scored ("test") = every 3rd session frame from index 30 on
  GT crop    AvatarTest-v1 rule on the dataset's GT mesh (SMPL-X v1.1 gendered, PCA-12 hands, fit_dress4d.gendered_verts;
             x_cam = R_c V + t_c): perspective crop aimed at / sized by the projected mesh box clipped to the image, 1.1 x, 1024 px
  GT image   the dataset's rendered mask (exact), warped like the image; gt512 = image x mask at 512 on black; masks512 kept
Outputs (E6 = O/eval/phase6_val): base/dress4d/<clip>/{frames.json, crop_h.npz, masks/<key>.png}, gt512/dress4d/<clip>/<key>.png,
and eval_split split file worker6/p6_val.json (4D-DRESS sessions + the ValBench-v1 BEDLAM2 clips with their tuning / verdict part).
Usage (guava env): python convert_p6_dress4d.py split ; CUDA_VISIBLE_DEVICES=.. python convert_p6_dress4d.py run [--clips id,..]"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE.parent / "eval_split"))
import convert_bedlam2 as cb                                                                       # noqa: E402

DATA = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/4d_dress/extracted")
E6 = cb.O / "eval" / "phase6_val"
SPLIT = HERE / "p6_val.json"
VALBENCH = HERE.parent / "eval_split" / "valbench_v1.json"
TMP = Path("/dev/shm/ubx858_eval/p6_tmp")
CAMS = ("0004", "0052")
TUNING = {"dress4d": ("00127", "00134"), "bedlam2": ("it_4102", "it_4590", "it_4591", "it_4773", "it_4781", "nl_5820", "us_0420", "us_1117",
                                                     "us_1143", "us_1479", "us_1735", "us_1863", "us_2447", "us_2495", "us_2901")}
VERDICT = {"dress4d": ("00137", "00151"), "bedlam2": ("it_4224", "it_4548", "it_4551", "it_4634", "nl_5510", "nl_5764", "nl_5955", "nl_6929",
                                                      "us_1152", "us_1494", "us_1675", "us_1765", "us_1978", "us_2343", "us_2660")}
SKIP, EVERY = 30, 3


def sessions():
    idx = json.load(open(cb.O / "data/dress4d/index.json"))["clips"]
    val = [c for c in idx if c["split"] == "val"]; out = {}
    for c in val: out.setdefault((c["subject"], c["outfit"]), []).append(c["take"])
    return {k: sorted(v, key=lambda t: int(t[4:])) for k, v in out.items()}


def split(a):
    clips = []
    for (s, o), takes in sorted(sessions().items()):
        part = "tuning" if s in TUNING["dress4d"] else "verdict" if s in VERDICT["dress4d"] else None
        assert part, s
        frames, src = [], {}
        for t in takes:
            b = pickle.load(open(DATA / s / o / t / "basic_info.pkl", "rb"))
            for n in b["scan_frames"]:
                f = int(t[4:]) * 1000 + int(n); frames.append(f); src[f] = (t, n)
        test = frames[SKIP::EVERY]
        for cam in CAMS:
            clips.append({"id": f"dress4d/{s}_{o}_{cam}", "dataset": "dress4d", "clip": f"{s}_{o}_{cam}", "identity": s, "outfit": o, "camera": cam,
                          "part": part, "takes": takes, "frames": frames, "source_frame": frames[0], "test_frames": test, "fps": 30,
                          "frame_src": {str(f): [t, n] for f, (t, n) in src.items()},
                          "gender": pickle.load(open(DATA / s / o / takes[0] / "basic_info.pkl", "rb"))["gender"]})
    vb = json.load(open(VALBENCH))["clips"]
    for c in vb:
        if c["dataset"] != "bedlam2": continue
        part = "tuning" if c["identity"] in TUNING["bedlam2"] else "verdict" if c["identity"] in VERDICT["bedlam2"] else None
        assert part, c["identity"]
        clips.append({**{k: c[k] for k in ("id", "dataset", "clip", "identity", "job", "seq", "frames", "rows", "source_frame", "test_frames", "fps", "video_frames")},
                      "part": part, "E": str(cb.O / "eval" / "valbench_v1")})
    J = {"name": "phase6 validation (PREREG phase 6)", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
         "rules": {"dress4d": f"session = val takes of (subject, outfit) in take order, cameras {CAMS}; scored = every {EVERY}rd frame from index {SKIP}",
                   "bedlam2": "ValBench-v1 BEDLAM2 clips (crops, masks, gt512, scored rows unchanged); replay = the 30 fps video from the source frame",
                   "split": "by subject / identity, numpy default_rng(20261103) permutation (PREREG phase 6)"},
         "composition": {p: {d: sum(1 for c in clips if c["part"] == p and c["dataset"] == d) for d in ("dress4d", "bedlam2")} for p in ("tuning", "verdict")},
         "clips": clips}
    SPLIT.write_text(json.dumps(J, indent=1)); print(json.dumps(J["composition"]), len(clips))


def convert(c, m_by_gender, dev, log):
    import cv2, torch, fit_dress4d as FD
    t0 = time.time(); s, o, cam = c["identity"], c["outfit"], c["camera"]; name = c["clip"]
    test = c["test_frames"]; keys = [f"frame_{f:06d}" for f in c["frames"]]
    # GT meshes of the scored frames (+ source) per take
    want = {}
    for f in [c["source_frame"]] + test: t, n = c["frame_src"][str(f)]; want.setdefault(t, []).append((f, n))
    cams = {}; V = {}
    for t, lst in want.items():
        P = [pickle.load(open(DATA / s / o / t / "SMPLX" / f"mesh-f{n}_smplx.pkl", "rb")) for _, n in lst]
        p = {k: np.stack([q[k] for q in P]).astype(np.float32) for k in P[0]}
        with torch.no_grad(): v, _ = FD.gendered_verts(m_by_gender[c["gender"]], p, dev)
        for (f, _), vv in zip(lst, v.cpu().numpy().astype(np.float64)): V[f] = vv
        cams[t] = pickle.load(open(DATA / s / o / t / "Capture" / "cameras.pkl", "rb"))[cam]
    LB = TMP / "base" / "dress4d" / name; LG = TMP / "gt512" / "dress4d" / name
    for p_ in (LB, LG):
        if p_.exists(): shutil.rmtree(p_)
    (LB / "masks").mkdir(parents=True); LG.mkdir(parents=True)
    Hs, srcs, clipped, mfrac = {}, {}, 0, []
    for f in [c["source_frame"]] + test:
        t, n = c["frame_src"][str(f)]; K = np.asarray(cams[t]["intrinsics"], np.float64); E = np.asarray(cams[t]["extrinsics"], np.float64)
        img = cv2.cvtColor(cv2.imread(str(DATA / s / o / t / "Capture" / cam / "images" / f"capture-f{n}.png")), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(DATA / s / o / t / "Capture" / cam / "masks" / f"mask-f{n}.png"), cv2.IMREAD_GRAYSCALE)
        H_, W_ = img.shape[:2]; X = V[f] @ E[:, :3].T + E[:, 3]; Ki = np.linalg.inv(K)
        p = X @ K.T; p = p[:, :2] / p[:, 2:]
        lo = np.clip(p.min(0), 0, [W_, H_]); hi = np.clip(p.max(0), 0, [W_, H_]); clipped += int((p.min(0) < 0).any() or (p.max(0) > [W_, H_]).any())
        Rv = cb.rot_onto_z(Ki @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.0]))
        cor = np.array([[lo[0], lo[1], 1], [hi[0], lo[1], 1], [lo[0], hi[1], 1], [hi[0], hi[1], 1.0]]) @ Ki.T @ Rv.T
        fx = float(K[0, 0]); q = fx * cor[:, :2] / cor[:, 2:]
        L = cb.BODY_SCALE * float((q.max(0) - q.min(0)).max()); sc = cb.CROP / L
        Kv = np.array([[fx * sc, 0, cb.CROP / 2], [0, fx * sc, cb.CROP / 2], [0, 0, 1.0]]); Hm = Kv @ Rv @ Ki
        k = f"frame_{f:06d}"; Hs[k] = Hm; srcs[k] = str(DATA / s / o / t / "Capture" / cam / "images" / f"capture-f{n}.png")
        if f == c["source_frame"] and f not in test: continue
        im5 = cv2.warpPerspective(img, Hm, (1024, 1024), flags=cv2.INTER_LINEAR, borderValue=0)
        m5 = cv2.warpPerspective(msk, Hm, (1024, 1024), flags=cv2.INTER_LINEAR, borderValue=0)
        im5 = cv2.resize(im5, (512, 512), interpolation=cv2.INTER_AREA); m5 = cv2.resize(m5, (512, 512), interpolation=cv2.INTER_AREA)
        a_ = m5.astype(np.float32)[..., None] / 255.0; mfrac.append(float(a_.mean()))
        cv2.imwrite(str(LG / f"{k}.png"), cv2.cvtColor(np.round(im5 * a_).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(LB / "masks" / f"{k}.png"), m5)
    np.savez(LB / "crop_h.npz", keys=np.array(list(Hs)), H=np.stack(list(Hs.values())), orig_src=np.array(list(srcs.values())),
             note=np.array("H: original image px -> 1024 metric crop px (AvatarTest-v1 rule on the GT gendered SMPL-X v1.1 mesh)"))
    fj = {"dataset": "dress4d", "clip": name, "id": c["id"], "keys": keys, "source": keys[0], "recon": [], "test": [f"frame_{f:06d}" for f in test],
          "fps": 30, "frame_src": {k: str(DATA / s / o / c["frame_src"][str(f)][0] / "Capture" / cam / "images" / f"capture-f{c['frame_src'][str(f)][1]}.png")
                                   for k, f in zip(keys, c["frames"])},
          "note": "phase 6 replay session; test = scored frames (every 3rd from index 30); recon unused (no per-person training)"}
    (LB / "frames.json").write_text(json.dumps(fj))
    for src, dst in ((LG, E6 / "gt512" / "dress4d" / name), (LB, E6 / "base" / "dress4d" / name)):
        if dst.exists(): shutil.rmtree(dst)                                                        # only our own earlier output
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(src, dst, copy_function=shutil.copyfile); shutil.rmtree(src)
    rec = {"id": c["id"], "part": c["part"], "frames": len(keys), "scored": len(test), "box_clipped": clipped,
           "mask_frac_mean": float(np.mean(mfrac)), "seconds": round(time.time() - t0, 1)}
    log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)


def run(a):
    import torch, fit_dress4d as FD
    J = json.loads(SPLIT.read_text()); clips = [c for c in J["clips"] if c["dataset"] == "dress4d"]
    if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]
    todo = [c for c in clips if not (E6 / "base" / "dress4d" / c["clip"] / "frames.json").exists()]
    dev = torch.device("cuda:0"); m = {g: FD.Model(FD.V11 / f"SMPLX_{g.upper()}.npz", 10, 10, dev) for g in ("male", "female")}
    (E6 / "reports").mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    with open(E6 / "reports" / "convert_log.jsonl", "a") as log:
        for c in todo: convert(c, m, dev, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=("split", "run")); ap.add_argument("--clips", default="")
    a = ap.parse_args(); {"split": split, "run": run}[a.stage](a)
