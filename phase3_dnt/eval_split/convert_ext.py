"""Extended benchmark converter (PREREG "Extended benchmark", registered 2026-11-09 before any conversion).

New clips, in AvatarTest-v1's formats (phase3_dnt/baselines/CONTRACT.md) so every renderer / adapter / scorer runs unchanged:
  X = O/eval/finalbench_ext   data/<ds>/<clip> (GUAVA tracked), gt512/<ds>/<clip>, base/<ds>/<clip> (+ orig/<key>.png for the SPRINT row)
  datasets: peoplesnapshot (the 20 sequences outside the standard four), neuman (lab, parkinglot), ubc (the 100 official test videos)
The standard People-Snapshot four, NeuMan four, UBody 30 and X-Humans 3 are NOT converted again (their E roots are reused).

Stages
  extract_ps  People-Snapshot files of all 24 sequences from the encrypted zip; the password comes ONLY from the environment
              variable PS_ZIP_PASSWORD (never printed / stored). Writes /dev/shm/ubx858_eval/finalbench_ext_tmp/ps_raw and a tar copy on
              /vault (backup_for_desktop/data/people_snapshot_extracted.tar).
  split       -> eval_split/finalbench_ext.json (AvatarTest-v1 split format, plus "person" for the person-level bootstrap).
  run         converts clips: --part ps | neuman | ubc [--clips id,...]. Resumable (a clip with base/.../frames.json is skipped).

Rules as registered
  People-Snapshot (new 20): InstantAvatar frame processing (sequential cv2 read + undistort); every 4th frame from 0; recon = the
    first two thirds (floor), test = the rest; source = first recon frame; GT = reconstructed_poses.hdf5 (pose 72, trans, betas) with
    the gendered SMPL of GART's rule; SMPL-X 2020 fitted exactly as convert_publicbench.py.
  NeuMan (lab, parkinglot): convert_publicbench.py unchanged (official get_data_splits).
  UBC-Fashion: pseudo-GT = fit_ubc.py --split test --stride 4 --cap 100000 --save-params (SPRINT + joint video refinement); frames
    failing its alignment filter dropped; recon = first half of the kept frames (floor), test = the rest; source = first kept frame;
    excluded iff the fit failed or < 10 recon or < 10 test frames. Crop / records as X-Humans' converter (perspective crop from the
    projected pseudo-GT mesh box, 1.1 x, 1024), StyleMatte alpha on the 1024 crop, GT = image x alpha.
Usage (env: source phase3_dnt/guava_fb/env_bedlam2.sh):
  PS_ZIP_PASSWORD=... $PY convert_ext.py extract_ps      (the user runs this; the password is typed, not stored)
  $PY convert_ext.py split
  CUDA_VISIBLE_DEVICES=6 $PY convert_ext.py run --part ubc [--clips ubc/91-3003CN5S]"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import types
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import convert_bedlam2 as cb                                                                       # noqa: E402
import convert_publicbench as CP                                                                   # noqa: E402
import convert_test as CT                                                                          # noqa: E402

XE = cb.O / "eval" / "finalbench_ext"
TMP = Path("/dev/shm/ubx858_eval/finalbench_ext_tmp")
SPLIT = HERE / "finalbench_ext.json"
BACKUP = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/backup_for_desktop")
PS_ALL = ("female-1-casual", "female-3-casual", "female-3-sport", "female-4-casual", "female-4-sport", "female-6-plaza", "female-7-plaza",
          "female-8-plaza", "male-1-casual", "male-1-plaza", "male-1-sport", "male-2-casual", "male-2-outdoor", "male-2-plaza", "male-2-sport",
          "male-3-casual", "male-3-outdoor", "male-3-plaza", "male-3-sport", "male-4-casual", "male-4-sport", "male-5-outdoor", "male-5-sport",
          "male-9-plaza")
PS_STD = ("male-3-casual", "male-4-casual", "female-3-casual", "female-4-casual")
PS_NEW = tuple(s for s in PS_ALL if s not in PS_STD)
NEUMAN_NEW = ("lab", "parkinglot")
UBC_D = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/data/ubc_fashion")
UBC_FITS = XE / "fits" / "ubc"
UBC_ID = cb.O / "reports" / "ubc" / "ubc_identity.npz"
PS_STRIDE, UBC_MIN = 4, 10
CHUNK = 256
UBC_LICENCE = ("UBC-Fashion (Zablotskaia et al., BMVC 2019; videos from Amazon Fashion, list from the authors' repository): research use "
               "only; no redistribution. Pseudo-GT SMPL-X from our own regressor + fitting (fit_ubc.py). Body model: SMPL-X (MPI licence).")


def ps_raw():
    return TMP / "ps_raw" / "people_snapshot_public"


def patch_cp():
    """Point convert_publicbench's module globals at the extended benchmark (its functions read them at call time)."""
    CP.E2 = XE; CP.TMP = TMP; CP.SPLIT = SPLIT; CP.PS_SUBJECTS = PS_NEW; CP.NEUMAN_SEQS = NEUMAN_NEW
    orig = CP.ps_poses

    def ps_poses(s):
        if s in PS_STD: return orig(s)
        J = json.loads(SPLIT.read_text()); c = next(c for c in J["clips"] if c["id"] == f"peoplesnapshot/{s}")
        d = np.load(TMP / "ps_poses" / f"{s}.npz")
        out = {}
        for sp, lst in (("train", c["recon_frames"]), ("test", c["test_frames"])):
            idx = np.array(lst)
            out[sp] = {"global_orient": d["pose"][idx, :3], "body_pose": d["pose"][idx, 3:72], "transl": d["trans"][idx], "betas": d["betas"].reshape(1, -1)[:, :10]}
        return out
    CP.ps_poses = ps_poses


# ------------------------------------------------------------------------------------------------ People-Snapshot extraction
def extract_ps(a):
    pw = os.environ.get("PS_ZIP_PASSWORD")
    if not pw: sys.exit("set PS_ZIP_PASSWORD in the environment (it is never stored)")
    z = zipfile.ZipFile(CP.PS_ZIP); dst = TMP / "ps_raw"; dst.mkdir(parents=True, exist_ok=True); n = 0
    for i in z.infolist():
        p = i.filename.split("/")
        if len(p) == 3 and p[1] in PS_ALL and p[2] in ("camera.pkl", "masks.hdf5", f"{p[1]}.mp4", "reconstructed_poses.hdf5", "consensus.pkl"):
            if not (dst / i.filename).exists(): z.extract(i, dst, pwd=pw.encode()); n += 1
    del pw
    BACKUP.joinpath("data").mkdir(parents=True, exist_ok=True); t = BACKUP / "data" / "people_snapshot_extracted.tar"
    if not t.exists():
        with tarfile.open(str(t) + ".partial", "w") as tf: tf.add(ps_raw(), arcname="people_snapshot_public")
        os.rename(str(t) + ".partial", t)
    print(json.dumps({"extracted_files": n, "sequences": sorted(os.listdir(ps_raw())), "vault_copy": str(t)}), flush=True)


def ps_reconstructed(s):
    """reconstructed_poses.hdf5 -> TMP/ps_poses/<s>.npz (pose (N, 72), trans (N, 3), betas (10,)); read with GART's h5py env."""
    out = TMP / "ps_poses" / f"{s}.npz"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        code = ("import h5py, numpy as np, sys; f = h5py.File(sys.argv[1], 'r'); "
                "np.savez(sys.argv[2], **{k: np.asarray(f[k]) for k in f.keys()})")
        subprocess.run([CP.H5PY, "-I", "-c", code, str(ps_raw() / s / "reconstructed_poses.hdf5"), str(out)], check=True)
    d = np.load(out); return {k: d[k] for k in d.files}


# ------------------------------------------------------------------------------------------------ split
def ubc_people():
    """ArcFace clusters of the test videos (max face-pair cosine > 0.5, connected components) -> {video: person id}."""
    z = np.load(UBC_ID, allow_pickle=True); ks = sorted(k for k in z.files if k.startswith("test/"))
    E = [z[k] / np.linalg.norm(z[k], axis=1, keepdims=True) for k in ks]; n = len(ks); par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if len(E[i]) and len(E[j]) and float((E[i] @ E[j].T).max()) > 0.5: par[f(i)] = f(j)
    roots = {}; out = {}
    for i, k in enumerate(ks): out[k.split("/", 1)[1]] = "ubc_person_%02d" % roots.setdefault(f(i), len(roots))
    return out


def split(a):
    clips, excluded = [], []
    # People-Snapshot, new 20
    root = ps_raw(); have_ps = all((root / s / f"{s}.mp4").exists() and (root / s / "reconstructed_poses.hdf5").exists() for s in PS_NEW)
    if not have_ps:
        print("People-Snapshot sequences not extracted yet: run 'convert_ext.py extract_ps' with PS_ZIP_PASSWORD; PS clips left out of this split", flush=True)
    else:
        for s in PS_NEW:
            n_prop, n_read, _ = CP.ps_frames(root, s, []); K, dist, wh = CP.ps_camera(root, s); P = ps_reconstructed(s)
            assert len(P["pose"]) >= n_read or len(P["pose"]) == n_read, (s, len(P["pose"]), n_read)
            n = min(n_read, len(P["pose"])); fr = list(range(0, n, PS_STRIDE)); k = (2 * len(fr)) // 3; tr, te = fr[:k], fr[k:]
            g = CP.gart_gender(s)
            clips.append({"id": f"peoplesnapshot/{s}", "dataset": "peoplesnapshot", "part": "full body, real video (static camera, subject rotating in A-pose)",
                          "identity": s, "person": s.rsplit("-", 1)[0], "clip": s, "gender": g, "standard": False,
                          "video": f"{CP.PS_ZIP}:people_snapshot_public/{s}/{s}.mp4", "fps": 30, "wh": list(wh), "video_frames": n_read,
                          "pose_rows": int(len(P["pose"])), "camera": {"K": K.tolist(), "dist": dist.tolist()},
                          "frame_index": "0-based cv2.VideoCapture sequential read, cv2.undistort(K, camera_k) (InstantAvatar processing)",
                          "published_split": {"source": "none published; registered rule: every 4th frame from 0, first two thirds train, rest test"},
                          "gt": {"kind": "People-Snapshot reconstructed_poses.hdf5 (pose 72, trans, betas), gendered SMPL (GART rule)",
                                 "file": str(root / s / "reconstructed_poses.hdf5"), "sha256": CP.sha256(root / s / "reconstructed_poses.hdf5"),
                                 "body_model": f"SMPL_{g.upper()}"},
                          "frames": fr, "n_frames": len(fr), "source_frame": tr[0], "n_recon": len(tr), "recon_frames": tr, "test_frames": te})
    # NeuMan lab / parkinglot: convert_publicbench's split code, unchanged rule
    tmp_split = TMP / "neuman_split_tmp.json"; TMP.mkdir(parents=True, exist_ok=True)
    save = (CP.SPLIT, CP.PS_SUBJECTS, CP.NEUMAN_SEQS)
    CP.SPLIT, CP.PS_SUBJECTS, CP.NEUMAN_SEQS = tmp_split, (), NEUMAN_NEW
    CP.split(types.SimpleNamespace(ps_root=str(TMP / "none")))
    CP.SPLIT, CP.PS_SUBJECTS, CP.NEUMAN_SEQS = save
    for c in json.loads(tmp_split.read_text())["clips"]:
        c["person"] = c["identity"]; c["standard"] = False; clips.append(c)
    # UBC-Fashion test videos
    people = ubc_people(); names = [l.strip().rsplit("/", 1)[-1].replace(".mp4", "") for l in (UBC_D / "fashion_test.txt").read_text().splitlines() if l.strip()]
    srt = sorted(names); perm = np.random.default_rng(20261109).permutation(len(srt)); prio = {srt[i]: r for r, i in enumerate(perm)}
    for nm in names:
        fj = UBC_FITS / nm / "fit.json"; fp = UBC_FITS / nm / "fit_params.npz"
        if not (fj.exists() and fp.exists()):
            excluded.append({"id": f"ubc/{nm}", "reason": "fit not finished or failed"}); continue
        info = json.loads(fj.read_text())
        if not info.get("ok"): excluded.append({"id": f"ubc/{nm}", "reason": "fit failed: " + info.get("error", "")[:200]}); continue
        d = np.load(fp); idx = d["idx"]; ok = d["pass_"].astype(bool); kept = [int(i) for i in idx[ok]]
        nr = len(kept) // 2; tr, te = kept[:nr], kept[nr:]
        if len(tr) < UBC_MIN or len(te) < UBC_MIN:
            excluded.append({"id": f"ubc/{nm}", "reason": f"too few frames after the alignment filter ({len(tr)} recon / {len(te)} test)"}); continue
        clips.append({"id": f"ubc/{nm}", "dataset": "ubc", "part": "full body, real video (studio, static camera, subject turning)",
                      "identity": people.get(nm, nm), "person": people.get(nm, nm), "clip": nm, "gender": "neutral", "standard": False,
                      "video": str(UBC_D / "test" / f"{nm}.mp4"), "fps": 30, "wh": [int(d["W"]), int(d["H"])], "video_frames": int(d["n_video"]),
                      "priority": int(prio[nm]),
                      "frame_index": "0-based cv2.VideoCapture sequential read of the mp4",
                      "gt": {"kind": "pseudo-GT: fit_ubc.py (SPRINT scratch EMA 128k + joint video refinement, one shared shape), SMPL-X 2020",
                             "file": str(fp), "sha256": CP.sha256(fp), "frames_fitted": int(len(idx)), "frames_passing_filter": int(ok.sum())},
                      "frames": kept, "n_frames": len(kept), "source_frame": tr[0], "n_recon": len(tr), "recon_frames": tr, "test_frames": te})
    comp = []
    for ds in ("peoplesnapshot", "neuman", "ubc"):
        cs = [c for c in clips if c["dataset"] == ds]
        comp.append({"dataset": ds, "clips": len(cs), "people": len({c["person"] for c in cs}), "recon_frames": sum(c["n_recon"] for c in cs),
                     "test_frames": sum(len(c["test_frames"]) for c in cs)})
    J = {"name": "FinalBench extension (new clips)", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "phase3_dnt/eval_split/convert_ext.py split",
         "git_head_at_build": CP.git_head(), "prereg": "PREREG.md 'Extended benchmark' (2026-11-09)",
         "protocol": {"crop_metrics": "AvatarTest-v1 (as publicbench_v1 / v2)", "people": "person = bootstrap unit (PS subject; UBC ArcFace cluster)",
                      "ubc_priority": "rank in numpy default_rng(20261109).permutation of the sorted test list (per-person work order)"},
         "composition": comp, "excluded_clips": excluded,
         "licences": {**CP.LICENCES, "ubc": UBC_LICENCE}, "clips": clips}
    SPLIT.write_text(json.dumps(J, indent=1))
    print(json.dumps({"clips": len(clips), "composition": comp, "excluded": len(excluded), "sha256": CP.sha256(SPLIT)}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ UBC conversion (X-Humans converter path)
def ubc_frames(c, want):
    import cv2
    cap = cv2.VideoCapture(c["video"]); ws = set(want); out = {}; i = 0
    while len(out) < len(ws):
        ok, fr = cap.read()
        if not ok: break
        if i in ws: out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release(); return out


def convert_ubc(c, X, matter, M, J, log):
    import cv2, torch, torchvision
    t00 = time.time(); ds, name = c["dataset"], c["clip"]; frames = c["frames"]; n = len(frames); dev = X.dev
    keys = [f"frame_{f:06d}" for f in frames]; G = cb.geo(); W, H = c["wh"]
    f32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)                    # noqa: E731
    d = np.load(c["gt"]["file"]); row = {int(i): r for r, i in enumerate(d["idx"])}; ri = np.array([row[f] for f in frames])
    rots = d["rots_aa"][ri].astype(np.float64); Rv0 = d["Rv"][ri]; Tt = d["Tt"][ri].astype(np.float64)
    b200 = d["beta"].astype(np.float64); exp50 = d["exp"][ri].astype(np.float64)
    K = np.array([[float(W), 0, W / 2], [0, float(W), H / 2], [0, 0, 1.0]])                          # fit_ubc: f = W, principal point = centre
    # 1. pseudo-GT in the ORIGINAL camera: virtual camera = R_v (full camera); fold R_v^T into root / translation about the pelvis
    with torch.no_grad(): _, Jr = X.X.shaped(f32(b200)[None]); J0 = Jr[0, 0].cpu().numpy().astype(np.float64)
    pose = rots.reshape(n, 165).copy(); Rroot_full = np.einsum("nji,njk->nik", Rv0, CP.R_np(rots[:, 0]))
    pose[:, :3] = CP.aa_np(Rroot_full)
    t = np.einsum("nji,nj->ni", Rv0, J0[None] + Tt) - J0[None]
    VS = []
    with torch.no_grad():
        for i in range(0, n, CHUNK):
            V, _ = X.X(f32(pose[i:i + CHUNK]), f32(b200)[None].expand(len(pose[i:i + CHUNK]), -1), f32(exp50[i:i + CHUNK])); VS.append(V + f32(t[i:i + CHUNK])[:, None])
        VS0 = torch.cat(VS)
        # check: virtual-camera mesh (as fitted) = R_v x full-camera mesh
        Vv, _ = X.X(f32(rots[:4].reshape(-1, 165)), f32(b200)[None].expand(min(4, n), -1), f32(exp50[:4]))
        fold_mm = float((torch.einsum("nij,nvj->nvi", f32(Rv0[:4]), VS0[:4]) - (Vv + f32(Tt[:4])[:, None])).norm(dim=-1).max() * 1000)
    assert fold_mm < 1.0, fold_mm
    # 2. float64 GUAVA-layer geometry, records (convert_xhumans.convert_clip step 3)
    VN, JN = [], []
    for i in range(0, n, CHUNK):
        a_, b_ = G["N"](torch.tensor(pose[i:i + CHUNK]), torch.tensor(b200)[None].expand(len(pose[i:i + CHUNK]), -1), torch.tensor(exp50[i:i + CHUNK]))
        VN.append(a_.numpy()); JN.append(b_.numpy())
    VN, JN = np.concatenate(VN), np.concatenate(JN)
    VSn = VS0.cpu().numpy().astype(np.float64); geo = []; Ki = np.linalg.inv(K); f = float(K[0, 0])
    for i in range(n):
        XS = VSn[i]; XN = VN[i] + t[i]; res_mm = float(np.linalg.norm(XN - XS, axis=1).mean() * 1000)
        p = XS @ K.T; p = p[:, :2] / p[:, 2:]
        lo = np.clip(p.min(0), 0, [W, H]); hi = np.clip(p.max(0), 0, [W, H])
        Rv = cb.rot_onto_z(Ki @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.0]))
        cor = np.array([[lo[0], lo[1], 1], [hi[0], lo[1], 1], [lo[0], hi[1], 1], [hi[0], hi[1], 1.0]]) @ Ki.T @ Rv.T
        q = f * cor[:, :2] / cor[:, 2:]
        Lc = cb.BODY_SCALE * float((q.max(0) - q.min(0)).max()); sc = CP.CROP / Lc
        Kv = np.array([[f * sc, 0, CP.CROP / 2], [0, f * sc, CP.CROP / 2], [0, 0, 1.0]])
        Hm = Kv @ Rv @ Ki; invtanfov = 2 * f / Lc
        pj = lambda X3: (lambda u: (lambda w: w[:, :2] / w[:, 2:])((u / u[:, 2:]) @ Hm.T))(X3 @ K.T)   # noqa: E731
        v2d_src = pj(XN); v2d_gt = pj(XS)
        Rroot = Rv @ cv2.Rodrigues(pose[i, :3])[0]; J0x = JN[i, 0]; tp = Rv @ (J0x + t[i]) - J0x
        Dm = np.diag([-1.0, -1.0, 1.0]); RT = np.concatenate([Dm, (Dm @ tp)[:, None]], 1).astype(np.float32)
        Vmodel = (VN[i] - J0x) @ Rv.T + J0x
        v2d = cb.project_guava(Vmodel, RT.astype(np.float64), invtanfov)
        fold_px512 = float(np.linalg.norm(v2d - v2d_src, axis=1).mean() * 512 / CP.CROP)
        pc = pose[i].astype(np.float32)
        smplx = {"global_pose": cb.aa_from_R(Rroot).astype(np.float32), "body_pose": pc[3:66].reshape(21, 3).copy(),
                 "jaw_pose": pc[66:69].reshape(1, 3).copy(), "eye_pose": pc[69:75].reshape(2, 3).copy(),
                 "left_hand_pose": pc[75:120].reshape(15, 3).copy(), "right_hand_pose": pc[120:165].reshape(15, 3).copy(),
                 "exp": exp50[i].astype(np.float32), "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}

        def part(ids, scale):
            pts = v2d[ids]; lt, rb = pts.min(0), pts.max(0)
            Mo = cb.box_crop_matrix([lt[0], lt[1], rb[0], rb[1]], CP.PART, scale)
            return {"M_o2c": Mo, "M_c2o": np.linalg.inv(Mo).astype(np.float32)}, bool(((pts >= 0) & (pts < CP.CROP)).all(1).mean() > 0.5)
        hc, hv = part(G["head"], cb.HEAD_SCALE); lc, lv = part(G["lhand"], cb.HAND_SCALE); rc, rv = part(G["rhand"], cb.HAND_SCALE)
        Mlr = np.diag([CP.BODY_LR / CP.CROP, CP.BODY_LR / CP.CROP, 1.0]).astype(np.float32); I3 = np.eye(3, dtype=np.float32)
        rec = {"smplx_coeffs": smplx,
               "flame_coeffs": {"expression_params": exp50[i].astype(np.float32), "jaw_params": pc[66:69].copy(),
                                "pose_params": np.zeros(3, np.float32), "neck_pose_params": np.zeros(3, np.float32),
                                "eye_pose_params": pc[69:75].copy(), "eyelid_params": np.zeros(2, np.float32),
                                "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
               "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3.copy(), "M_c2o-hd": I3.copy()},
               "head_crop": hc, "left_hand_crop": lc, "right_hand_crop": rc, "invtanfov": float(invtanfov), "view": "0"}
        geo.append({"rec": rec, "H": Hm, "Rv": Rv, "v2d_src": v2d_src.astype(np.float32), "v2d": v2d.astype(np.float32), "v2d_gt": v2d_gt.astype(np.float32),
                    "res_mm": res_mm, "res_px512": fold_px512, "fit_px512": 0.0, "invtanfov": float(invtanfov), "crop_side_px": Lc,
                    "depth_m": float(XS[:, 2].mean()), "vis": {"head": hv, "lhand": lv, "rhand": rv},
                    "box_clipped": bool((p.min(0) < 0).any() or (p.max(0) > [W, H]).any())})
    t2 = time.time()
    # 3. original frames -> 1024 crops; StyleMatte on the crop (as convert_publicbench)
    ori = ubc_frames(c, frames); assert len(ori) == n, (len(ori), n)
    crops, jpgs = [], []
    for fr, gi in zip(frames, geo):
        im = ori[fr]; assert im.shape[:2] == (H, W)
        cr = cv2.warpPerspective(im, gi["H"], (CP.CROP, CP.CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        crops.append(cr); jpgs.append(torchvision.io.encode_jpeg(torch.from_numpy(cr).permute(2, 0, 1).contiguous(), quality=95).numpy().tobytes())
    t3 = time.time(); alphas = matter(np.stack(crops)); tm = time.time() - t3; del crops
    fs = M["Wf"] @ b200
    res = {"g": geo, "jpgs": jpgs, "betas200": b200}
    meta = {"id": c["id"], "dataset": ds, "clip": name, "frames_keys": keys, "missing_frames": [], "dropped_zmin": [],
            "res_mm": [x["res_mm"] for x in geo], "res_px512": [x["res_px512"] for x in geo], "fit_px512": [x["fit_px512"] for x in geo],
            "invtanfov": [x["invtanfov"] for x in geo], "crop_side_px": [x["crop_side_px"] for x in geo], "depth_m": [x["depth_m"] for x in geo],
            "part_visible": [x["vis"] for x in geo], "alpha_mean": [float(x.mean() / 255) for x in alphas],
            "shape_info": {"betas200": "fit_ubc shared shape (pseudo-GT)", "betas200_norm": float(np.linalg.norm(b200))},
            "times": {"geom": t2 - t00, "decode_crop": t3 - t2, "matte": tm}, "licence": UBC_LICENCE,
            "note": "pseudo-GT (fit_ubc.py) is the GT here; res_mm = float64 GUAVA layer vs the converter's lean layer (same params); mask = StyleMatte"}
    CT.TMP = TMP
    mjpg = CT.write_tracked(XE / "data" / ds / name, res, keys, alphas, fs, meta)
    # 4. base package (local, then one copy) + gt512 + orig/ (source + test originals for the SPRINT row)
    dec = lambda b: torchvision.io.decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).float() / 255.0    # noqa: E731
    rs = lambda x: torchvision.transforms.functional.resize(x, (512, 512), antialias=True)                           # noqa: E731
    u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8)                                                      # noqa: E731
    LB = TMP / "base" / ds / name; LG = TMP / "gt512" / ds / name
    for p in (LB, LG):
        if p.exists(): shutil.rmtree(p)
    for p in (LB / "images", LB / "masks", LB / "orig", LG): p.mkdir(parents=True, exist_ok=True)
    fset = set(f"frame_{f:06d}" for f in c["test_frames"]); msk512 = {}
    for k, jb, mb in zip(keys, jpgs, mjpg):
        im, ms = rs(dec(jb)), rs(dec(mb))
        torchvision.io.write_png(u8(im), str(LB / "images" / f"{k}.png")); torchvision.io.write_png(u8(ms[:1]), str(LB / "masks" / f"{k}.png"))
        msk512[k] = u8(ms[0]).numpy() > 127
        if k in fset: torchvision.io.write_png(u8(im * ms), str(LG / f"{k}.png"))
    for fr in [c["source_frame"]] + c["test_frames"]:
        cv2.imwrite(str(LB / "orig" / f"frame_{fr:06d}.png"), cv2.cvtColor(ori[fr], cv2.COLOR_RGB2BGR))
    R = [x["rec"]["smplx_coeffs"] for x in geo]; Dm = np.diag([-1.0, -1.0, 1.0])
    itf = np.array([x["invtanfov"] for x in geo])
    K5 = np.zeros((n, 3, 3)); K5[:, 0, 0] = K5[:, 1, 1] = 256 * itf; K5[:, 0, 2] = K5[:, 1, 2] = 256; K5[:, 2, 2] = 1
    sx = {"keys": np.array(keys), "betas": np.asarray(b200, np.float32),
          "global_orient": np.stack([r["global_pose"] for r in R]).astype(np.float32), "body_pose": np.stack([r["body_pose"] for r in R]).astype(np.float32),
          "jaw_pose": np.stack([np.asarray(r["jaw_pose"]).reshape(3) for r in R]).astype(np.float32),
          "leye_pose": np.stack([np.asarray(r["eye_pose"]).reshape(2, 3)[0] for r in R]).astype(np.float32),
          "reye_pose": np.stack([np.asarray(r["eye_pose"]).reshape(2, 3)[1] for r in R]).astype(np.float32),
          "left_hand_pose": np.stack([r["left_hand_pose"] for r in R]).astype(np.float32), "right_hand_pose": np.stack([r["right_hand_pose"] for r in R]).astype(np.float32),
          "expression": np.stack([r["exp"] for r in R]).astype(np.float32),
          "transl": np.stack([Dm @ np.asarray(r["camera_RT_params"], np.float64)[:, 3] for r in R]).astype(np.float32)}
    np.savez(LB / "smplx2020.npz", **sx); np.savez(LB / "cameras.npz", keys=np.array(keys), K=K5.astype(np.float32))
    pz = np.concatenate([sx["global_orient"], sx["body_pose"].reshape(-1, 63), sx["jaw_pose"], sx["leye_pose"], sx["reye_pose"],
                         sx["left_hand_pose"].reshape(-1, 45), sx["right_hand_pose"].reshape(-1, 45)], 1)
    q = []
    with torch.no_grad():
        for i in range(0, n, CHUNK):
            Vx, _ = X.X(torch.tensor(pz[i:i + CHUNK], device=dev), torch.tensor(sx["betas"], device=dev)[None].expand(len(pz[i:i + CHUNK]), -1),
                        torch.tensor(sx["expression"][i:i + CHUNK], device=dev))
            Vx = Vx + torch.tensor(sx["transl"][i:i + CHUNK], device=dev)[:, None]
            q.append((Vx[..., :2] / Vx[..., 2:]).cpu().numpy() * K5[i:i + CHUNK, 0, 0][:, None, None] + 256)
    q = np.concatenate(q)
    chk_px = float(np.abs(q - np.stack([x["v2d"] for x in geo]) / 2).max())
    Rv = np.stack([x["Rv"] for x in geo])
    np.savez(LB / "smplx_gt.npz", keys=np.array(keys), note=np.array("pseudo-GT = smplx2020.npz (fit_ubc.py fit, SMPL-X 2020); source file: " + c["gt"]["file"]))
    VSv = torch.einsum("nij,nvj->nvi", f32(Rv), VS0)
    VSm = X.mapped(VSv)
    with torch.no_grad(): _, jm = X.smpl_fwd("neutral", torch.zeros(1, 3, device=dev), torch.zeros(1, 69, device=dev), torch.zeros(1, 10, device=dev))
    go_v = CP.aa_np(Rv @ CP.R_np(pose[:, :3])); bp0 = np.concatenate([pose[:, 3:66], np.zeros((n, 6))], 1)
    tr_init = VSm.mean(1).cpu().numpy() - jm[0].cpu().numpy()
    sms = [CP.fit_smpl_neutral(X, VSm[i:i + CHUNK], go_v[i:i + CHUNK], bp0[i:i + CHUNK], np.zeros(10), tr_init[i:i + CHUNK], CP.SMPL_ITERS, 0.02)
           for i in range(0, n, CHUNK)]
    bmed = np.median(np.stack([m["betas"] for m in sms]), 0)
    sm_go, sm_bp, sm_tr, sm_err = [], [], [], []
    for i, m in zip(range(0, n, CHUNK), sms):
        go_, bp_, tr_ = (f32(m["go"]).requires_grad_(True), f32(m["bp"]).requires_grad_(True), f32(m["transl"]).requires_grad_(True))
        be_ = f32(bmed).reshape(1, 10); opt = torch.optim.Adam([go_, bp_, tr_], lr=0.005)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            v, _ = X.smpl_fwd("neutral", go_, bp_, be_.expand(len(go_), -1), tr_); (v - VSm[i:i + CHUNK]).norm(dim=-1).mean().backward(); opt.step()
        with torch.no_grad():
            v, _ = X.smpl_fwd("neutral", go_, bp_, be_.expand(len(go_), -1), tr_); sm_err.append(((v - VSm[i:i + CHUNK]).norm(dim=-1).mean(1) * 1000).cpu().numpy())
        sm_go.append(go_.detach().cpu().numpy()); sm_bp.append(bp_.detach().cpu().numpy()); sm_tr.append(tr_.detach().cpu().numpy())
    sm = {"go": np.concatenate(sm_go), "bp": np.concatenate(sm_bp), "transl": np.concatenate(sm_tr), "betas": bmed, "err_mm": np.concatenate(sm_err)}
    np.savez(LB / "smpl.npz", keys=np.array(keys), betas=np.asarray(sm["betas"], np.float32), global_orient=np.asarray(sm["go"], np.float32),
             body_pose=np.asarray(sm["bp"], np.float32).reshape(n, 23, 3), transl=np.asarray(sm["transl"], np.float32), fit_err_mm=np.asarray(sm["err_mm"], np.float32))
    np.savez(LB / "crop_h.npz", keys=np.array(keys), H=np.stack([x["H"] for x in geo]).astype(np.float64), orig_wh=np.array([W, H]),
             orig_src=np.array([f"{c['video']}#frame={fr}" for fr in frames]), K_orig=np.repeat(K[None], n, 0).astype(np.float64),
             Rv=Rv.astype(np.float64), invtanfov=itf, crop_side_px=np.array([x["crop_side_px"] for x in geo]),
             note=np.array("H: original image pixel (x, y, 1) -> 1024 crop pixel (cv2.warpPerspective(orig, H, (1024, 1024)), bilinear, border 0). "
                           "orig/<key>.png: source + test original frames (cv2 sequential read)."))
    with torch.no_grad():
        qn = []
        for i in range(0, n, CHUNK):
            vsm, _ = X.smpl_fwd("neutral", f32(sm["go"][i:i + CHUNK]), f32(sm["bp"][i:i + CHUNK]), f32(sm["betas"]).reshape(1, 10).expand(len(sm["go"][i:i + CHUNK]), -1), f32(sm["transl"][i:i + CHUNK]))
            qn.append((vsm[..., :2] / vsm[..., 2:]).cpu().numpy() * K5[i:i + CHUNK, 0, 0][:, None, None] + 256)
        qn = np.concatenate(qn)
    chk_i = np.unique(np.round(np.linspace(0, n - 1, 40)).astype(int))
    iou_x = [CP.raster_iou(q[i], X.faces_x, msk512[keys[i]])[0] for i in chk_i]
    iou_s = [CP.raster_iou(qn[i], X.faces_s, msk512[keys[i]])[0] for i in chk_i]
    tiles = []
    for i in np.unique(np.round(np.linspace(0, n - 1, 4)).astype(int)):
        im = cv2.cvtColor(cv2.imread(str(LB / "images" / f"{keys[i]}.png")), cv2.COLOR_BGR2RGB)
        for pts, col in ((q[i][::7], (0, 255, 0)), (qn[i][::5], (255, 0, 255))):
            for x_, y_ in pts.astype(int): cv2.circle(im, (int(x_), int(y_)), 1, col, -1)
        cv2.putText(im, keys[i], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA); tiles.append(im)
    cv2.imwrite(str(LB / "overlay.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    tk = [k for k in keys if k in fset]; rk = [f"frame_{f:06d}" for f in c["recon_frames"]]
    fj = {"dataset": ds, "clip": name, "id": c["id"], "keys": keys, "source": f"frame_{c['source_frame']:06d}", "recon": rk, "test": tk,
          "fps": c["fps"], "dropped_from_split": [], "note": "frame_XXXXXX = 0-based video frame; every 4th frame; frames failing fit_ubc's alignment filter dropped"}
    assert fj["source"] == rk[0] and set(tk) == fset and not (set(rk) & set(tk)) and len(rk) == c["n_recon"]
    (LB / "frames.json").write_text(json.dumps(fj, indent=1))
    for src, dst in ((LG, XE / "gt512" / ds / name), (LB, XE / "base" / ds / name)):
        tmp = dst.parent / (dst.name + ".partial")
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for p in sorted(src.rglob("*")):
            if p.is_dir() or p.name == "frames.json": continue
            (tmp / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, tmp / p.relative_to(src))
        if (src / "frames.json").exists(): shutil.copyfile(src / "frames.json", tmp / "frames.json")
        if dst.exists(): shutil.rmtree(dst)                                                        # only our own earlier partial output
        os.rename(tmp, dst); shutil.rmtree(src)
    rec = {"id": c["id"], "frames": n, "recon": len(rk), "test": len(tk), "fold_mm_virtual_vs_full": fold_mm,
           "layer_res_mm_mean": float(np.mean(meta["res_mm"])), "fold_px512_mean": float(np.mean(meta["res_px512"])), "contract_cam_check_px": chk_px,
           "smpl_npz_err_mm_mean": float(np.mean(sm["err_mm"])), "smpl_npz_err_mm_max": float(np.max(sm["err_mm"])),
           "iou_smplx_mean": float(np.mean(iou_x)), "iou_smplx_min": float(np.min(iou_x)), "iou_smpl_mean": float(np.mean(iou_s)),
           "alpha_mean": float(np.mean(meta["alpha_mean"])), "box_clipped_frames": int(sum(x["box_clipped"] for x in geo)),
           "crop_side_px_mean": float(np.mean(meta["crop_side_px"])),
           "part_visible_frac": {p: float(np.mean([v[p] for v in meta["part_visible"]])) for p in ("head", "lhand", "rhand")},
           "overlay": str(XE / "base" / ds / name / "overlay.jpg"), "times_s": {"total": round(time.time() - t00, 1)}}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(json.dumps(rec), flush=True)
    return rec


def run(a):
    import torch
    patch_cp()
    J = json.loads(SPLIT.read_text()); clips = [c for c in J["clips"] if c["dataset"] == {"ps": "peoplesnapshot", "neuman": "neuman", "ubc": "ubc"}[a.part]]
    if a.clips: want = a.clips.split(","); clips = [c for c in clips if c["id"] in want]
    if a.shard: i, k = map(int, a.shard.split("/")); clips = clips[i::k]
    todo = [c for c in clips if not (XE / "base" / c["dataset"] / c["clip"] / "frames.json").exists()]
    print(f"{a.part}: {len(clips)} clips, {len(todo)} to do", flush=True)
    if not todo: return
    if a.part in ("ps", "neuman"):
        CP.run(types.SimpleNamespace(clips=",".join(c["id"] for c in todo), ps_root=str(ps_raw()) if a.part == "ps" else ""))
        return
    M = CT.maps(); (XE / "reports").mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0"); X = CP.Ctx(dev); matter = cb.Matter()
    with open(XE / "reports" / f"convert_log_ubc{('_' + a.shard.replace('/', 'of')) if a.shard else ''}.jsonl", "a") as log:
        for c in todo:
            try:
                convert_ubc(c, X, matter, M, J, log)
            except Exception as e:                                                                  # noqa: BLE001
                import traceback; print(c["id"], "FAILED", repr(e), traceback.format_exc(), flush=True)
                log.write(json.dumps({"id": c["id"], "error": repr(e)}) + "\n"); log.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("extract_ps", "split", "run")); ap.add_argument("--part", default="ubc", choices=("ps", "neuman", "ubc"))
    ap.add_argument("--clips", default=""); ap.add_argument("--shard", default="", help="i/k: every k-th clip from i")
    a = ap.parse_args()
    {"extract_ps": extract_ps, "split": split, "run": run}[a.stage](a)


if __name__ == "__main__":
    main()
