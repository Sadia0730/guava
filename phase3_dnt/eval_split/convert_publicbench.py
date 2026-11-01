"""PublicBench-v1 test data (PREREG "PublicBench-v1": People-Snapshot + NeuMan), in AvatarTest-v1's formats so that the existing
renderers, baseline adapters and scorer run unchanged:
  E2/data/<dataset>/<clip>/    GUAVA tracked format (FORMAT.md): SMPL-X 2020 records FITTED to the GT SMPL mesh + FLAME fields
  E2/gt512/<dataset>/<clip>/   <key>.png for the test keys: image x StyleMatte alpha at 512 on black (8 bit)
  E2/base/<dataset>/<clip>/    baseline package (phase3_dnt/baselines/CONTRACT.md) + crop_h.npz (original px -> 1024 crop H) +
                               smpl_gt.npz (the GT SMPL itself, published body model / gender) [+ orig/<key>.png for People-Snapshot]
  E2/reports/                  conversion log / report
E2 = O/eval/publicbench_v1 (O = the GUAVA-FB output root on /vault). Dataset names: peoplesnapshot, neuman.

Stages
  split  -> eval_split/publicbench_v1.json (AvatarTest-v1 split format; recon = the PUBLISHED train frames, test = the PUBLISHED
            test frames). CPU only; reads split rules / GT files / frame counts, no model output.
  run    -> converts the split's clips (GPU: StyleMatte + fits). Resumable (a clip with base/.../frames.json is skipped).

Protocol as implemented
  People-Snapshot (InstantAvatar protocol, as GART / GaussianAvatar): male-3-casual, male-4-casual, female-3-casual, female-4-casual.
    frames  = InstantAvatar's frame list: cv2.VideoCapture sequential read of <subject>.mp4 (frame i = image_{i:04d}.png), each frame
              cv2.undistort(frame, K, camera_k) (utils/InstantAvatarPeopleProcessing.py); train / test = START_END[subject][split]
              = sorted(images)[start:end + 1:skip], read from GART lib_data/instant_avatar_people_snapshot.py (ast, verbatim).
    GT      = InstantAvatar's Anim-NeRF-refined SMPL (poses/anim_nerf_{train,test}.npz as shipped in GART's repo, "already included
              when you clone this repo"), GENDERED SMPL (GART solver.py: female-* -> SMPL_FEMALE, male-* -> SMPL_MALE; InstantAvatar
              likewise), camera = camera.pkl (K = camera_f / camera_c, undistorted images; extrinsic identity, asserted as GART).
  NeuMan (official split, NeuMan / HUGS get_data_splits, copied verbatim from ml-hugs hugs/datasets/neuman.py via ast):
    captures = COLMAP sparse/images.txt sorted by image name (NeuManReader order='video'); train = all but every k-th capture,
              test = first half of the held-out captures (val = second half, unused).
    GT      = smpl_output_optimized.pkl (neutral SMPL, NeuManReader.read_smpls: SMPL(gender='neutral'); per-frame pose + betas, no
              translation) mapped to the COLMAP world by alignments.npy[name] (4 x 3, x_world = A^T [x; 1], a similarity with a
              per-frame scale s) and to the camera by the COLMAP pose; the camera frame is divided by s (projection unchanged) so the
              body is metric, i.e. x_cam = R_c SMPL(pose, betas) + t_c with R_c = R_wc A33 / s, t_c = (R_wc a + t_wc) / s, folded into
              (global_orient, transl) exactly (SMPL rotates about the rest pelvis). The standard SMPL forward is used (NeuMan's own
              DA-pose path differs only by pose correctives; reported). Camera = COLMAP PINHOLE (one K per sequence).
  SMPL-X 2020 (SMPLX_NEUTRAL_2020, 200 betas, 50 exp, flat hands) FITTED to the GT SMPL mesh through PEAR's SMPL-X -> SMPL vertex
    correspondence (smplx2smpl.pkl): one betas200 per sequence (initialised by a ridge least-squares fit of the rest-pose shapes),
    per-frame global orient + 21 body joints + translation (Adam); jaw / eyes / fingers = 0 (flat hands), exp50 = 0. Residual (mm)
    reported per frame / sequence.
  Crop / records / FLAME fields: convert_test.py's UBody path (ub_geometry): perspective crop aimed at and sized by the projected
    GT SMPL mesh box clipped to the image, 1.1 x, 1024 px, invtanfov = 2 f / L per frame, Rv folded into the root; FLAME shape =
    maps()['Wf'] @ betas200, expression = exp50 = 0, jaw / eyes 0. StyleMatte on the 1024 crop (NOT the datasets' masks).
  smpl.npz (CONTRACT: neutral SMPL v1.1.0, ONE betas10 per clip, crop-camera frame): NeuMan = the GT re-expressed with one betas
    (the GT betas vary slightly per frame: fit initialised at the GT pose / median betas), People-Snapshot = neutral SMPL fitted to
    the gendered GT mesh; fit_err_mm = residual to the GT mesh. smpl_gt.npz = the GT SMPL itself (gender, per-frame betas) in the
    same crop-camera frame.
  gt_proj.npz (as convert_test, readable by fb_common.check_render): v2d_1024 = the fitted SMPL-X mesh projected with the ORIGINAL
    K and mapped by H; v2d_converted_1024 = the record's GUAVA projection; v2d_gt_smpl_1024 = the GT SMPL mesh (6890) by K and H.

Usage (env: source phase3_dnt/guava_fb/env_bedlam2.sh):
  $PY convert_publicbench.py split [--ps-root DIR]
  CUDA_VISIBLE_DEVICES=3 $PY convert_publicbench.py run [--clips neuman/bike,...] [--ps-root DIR]
People-Snapshot input: --ps-root = a directory containing <subject>/{<subject>.mp4, camera.pkl, masks.hdf5}, or the encrypted zip
O/data/people_snapshot_public.zip with its password in the environment variable PS_ZIP_PASSWORD (never printed / stored)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guava_fb")); sys.path.insert(0, str(HERE))
import convert_bedlam2 as cb                                                                       # noqa: E402
import convert_test as CT                                                                          # noqa: E402

SPLIT = HERE / "publicbench_v1.json"
E2 = cb.O / "eval" / "publicbench_v1"
TMP = Path("/dev/shm/ubx858_eval/publicbench_tmp")
CROP, PART, BODY_LR = cb.CROP, cb.PART, cb.BODY_LR
NEUMAN = Path("/home/ubx858/ml-hugs/data/neuman/dataset")
NEUMAN_SEQS = ("bike", "citron", "jogging", "seattle")
HUGS_NEUMAN_PY = Path("/home/ubx858/ml-hugs/hugs/datasets/neuman.py")
NEUMAN_SMPL_PY = Path("/home/ubx858/ml-hugs/hugs/datasets/neuman_utils/smpl.py")
PS_ZIP = cb.O / "data" / "people_snapshot_public.zip"
PS_SUBJECTS = ("male-3-casual", "male-4-casual", "female-3-casual", "female-4-casual")
GART = Path("/dev/shm/ubx858_baselines/gart/code")
GART_LOADER = GART / "lib_data/instant_avatar_people_snapshot.py"
GART_SOLVER = GART / "solver.py"
GART_PREP = GART / "utils/InstantAvatarPeopleProcessing.py"
GART_COMMIT = "16c11f8a5bb3ae249a9d04dc9d98c316e10f1126"
H5PY = "/dev/shm/ubx858_baselines/gart/env/bin/python"                                            # GART's env (has h5py; read only)
SMPL_DIR = CT.SMPL_DIR                                                                              # npz of /home/ubx858/ml-hugs/data/smpl/SMPL_*.pkl
SMPL_PKL = Path("/home/ubx858/ml-hugs/data/smpl")
SMPLX_ITERS, SMPL_ITERS, RIDGE = 1500, 600, 1e-4
LICENCES = {
    "peoplesnapshot": ("People-Snapshot (Alldieck et al., CVPR 2018; TU Braunschweig / MPI): research-only licence agreed at download "
                       "(https://graphics.tu-bs.de/people-snapshot); no redistribution, no commercial use. Refined poses: InstantAvatar / "
                       "Anim-NeRF via GART's repository (their licences). Body model: SMPL (MPI non-commercial licence)."),
    "neuman": ("NeuMan dataset (Jiang et al., ECCV 2022; Apple, github.com/apple/ml-neuman): research use only under the dataset's / "
               "repository's licence; no redistribution, no commercial use. Body model: SMPL (MPI non-commercial licence)."),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def git_head():
    try: return subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: return None                                                                   # noqa: E722


# ------------------------------------------------------------------------------------------------ published split rules (verbatim)
def start_end():
    """InstantAvatar's START_END as in GART lib_data/instant_avatar_people_snapshot.py (literal, read with ast)."""
    t = GART_LOADER.read_text()
    for node in ast.parse(t).body:
        if isinstance(node, ast.Assign) and any(getattr(x, "id", None) == "START_END" for x in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("START_END not found")


def neuman_get_data_splits():
    """get_data_splits copied verbatim (ast) from ml-hugs hugs/datasets/neuman.py (= NeuMan's create_split_files rule)."""
    t = HUGS_NEUMAN_PY.read_text()
    for node in ast.parse(t).body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_data_splits":
            src = ast.get_source_segment(t, node); ns = {}; exec(src, ns); return ns["get_data_splits"], src
    raise RuntimeError("get_data_splits not found")


def gart_gender(subject):
    """GART solver.py: people_snapshot -> SMPL_FEMALE for 'female*', SMPL_MALE for 'male*' (checked in the source)."""
    s = GART_SOLVER.read_text(); assert 'seq_name.startswith("female")' in s and "SMPL_FEMALE.pkl" in s and "SMPL_MALE.pkl" in s
    return "female" if subject.startswith("female") else "male"


# ------------------------------------------------------------------------------------------------ NeuMan readers
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def neuman_colmap(seq):
    """cameras.txt (PINHOLE) + images.txt as NeuMan's ColmapAsciiReader (two lines per image) -> K, wh, {name: (R_wc, t_wc)}."""
    d = NEUMAN / seq / "sparse"; cams = {}
    for ln in (d / "cameras.txt").read_text().splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        e = ln.split(); assert e[1] == "PINHOLE", e[1]
        W, H, fx, fy, cx, cy = map(float, e[2:]); cams[int(e[0])] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), (int(W), int(H)))
    L = (d / "images.txt").read_text().split("\n"); assert L[0].startswith("# Image list") and L[3].startswith("# Number of images")
    n = int(L[3].split(":")[1].split(",")[0]); imgs = {}
    for i in range(n):
        e = L[4 + 2 * i].split(); assert len(e) == 10
        imgs[e[9]] = {"R": qvec2rotmat(np.array(list(map(float, e[1:5])))), "t": np.array(list(map(float, e[5:8]))), "cam": int(e[8])}
    assert len(cams) == 1
    return cams, imgs


def neuman_images(seq, want):
    """Original NeuMan frames (RGB uint8): images/%05d.png as NeuMan's CapturedImage (PIL / cv2 decode of the PNG)."""
    import cv2
    return {f: cv2.cvtColor(cv2.imread(str(NEUMAN / seq / "images" / f"{f:05d}.png")), cv2.COLOR_BGR2RGB) for f in want}


def neuman_raw(seq):
    import joblib
    raw = joblib.load(NEUMAN / seq / "smpl_output_optimized.pkl"); assert len(raw) == 1
    raw = raw[list(raw.keys())[0]]
    al = np.load(NEUMAN / seq / "alignments.npy", allow_pickle=True).item()
    return {"pose": np.asarray(raw["pose"], np.float64), "betas": np.asarray(raw["betas"], np.float64)}, al


# ------------------------------------------------------------------------------------------------ People-Snapshot access
def ps_root(arg):
    """Directory with <subject>/... (extracted). From --ps-root, an earlier extraction in TMP, or the zip + PS_ZIP_PASSWORD."""
    if arg: return Path(arg)
    loc = TMP / "ps_raw" / "people_snapshot_public"
    if all((loc / s / f"{s}.mp4").exists() and (loc / s / "camera.pkl").exists() and (loc / s / "masks.hdf5").exists() for s in PS_SUBJECTS):
        return loc
    pw = os.environ.get("PS_ZIP_PASSWORD")
    if not pw: return None
    z = zipfile.ZipFile(PS_ZIP); (TMP / "ps_raw").mkdir(parents=True, exist_ok=True)
    for i in z.infolist():
        p = i.filename.split("/")
        if len(p) == 3 and p[1] in PS_SUBJECTS and p[2] in ("camera.pkl", "masks.hdf5", f"{p[1]}.mp4"):
            z.extract(i, TMP / "ps_raw", pwd=pw.encode())
    return loc


def ps_camera(root, s):
    cam = pickle.load(open(root / s / "camera.pkl", "rb"), encoding="latin1")
    K = np.eye(3); K[0, 0], K[1, 1] = cam["camera_f"][0], cam["camera_f"][1]; K[:2, 2] = cam["camera_c"]
    w2c = np.eye(4); import cv2
    w2c[:3, :3] = cv2.Rodrigues(np.asarray(cam["camera_rt"], np.float64))[0]; w2c[:3, 3] = cam["camera_t"]
    assert np.allclose(np.linalg.inv(w2c), np.eye(4)), "People-Snapshot extrinsic is not the identity (GART asserts it)"
    return K, np.asarray(cam["camera_k"], np.float64), (int(cam["width"]), int(cam["height"]))


def ps_frames(root, s, want=None):
    """InstantAvatarPeopleProcessing.py: cv2.VideoCapture, frame_cnt = CAP_PROP_FRAME_COUNT, read until failure, cv2.undistort.
    Returns (n_cap_prop, n_read, {i: RGB uint8 undistorted})."""
    import cv2
    K, dist, _ = ps_camera(root, s); cap = cv2.VideoCapture(str(root / s / f"{s}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = {}; nread = 0; ws = set(want or [])
    for i in range(n):
        ok, fr = cap.read()
        if not ok: break
        nread += 1
        if i in ws: out[i] = cv2.cvtColor(cv2.undistort(fr, K, dist), cv2.COLOR_BGR2RGB)
    cap.release()
    return n, nread, out


def ps_mask_count(root, s):
    r = subprocess.run([H5PY, "-I", "-c", "import h5py,sys; print(h5py.File(sys.argv[1],'r')['masks'].shape)", str(root / s / "masks.hdf5")],
                       capture_output=True, text=True, check=True)
    return list(ast.literal_eval(r.stdout.strip()))


def ps_poses(s):
    d = {sp: np.load(GART / "data/people_snapshot" / s / "poses" / f"anim_nerf_{sp}.npz") for sp in ("train", "test")}
    return d


# ------------------------------------------------------------------------------------------------ split
def split(a):
    SE = start_end(); gds, gds_src = neuman_get_data_splits(); root = ps_root(a.ps_root); clips = []; excluded = []
    for s in PS_SUBJECTS:
        se = SE[s]; tr = list(range(se["train"][0], se["train"][1] + 1, se["train"][2])); te = list(range(se["test"][0], se["test"][1] + 1, se["test"][2]))
        P = ps_poses(s)
        assert len(P["train"]["global_orient"]) == len(tr) and len(P["test"]["global_orient"]) == len(te), s
        bdev = float(np.abs(P["train"]["betas"] - P["test"]["betas"]).max())
        frames = sorted(tr + te); assert not set(tr) & set(te)
        c = {"id": f"peoplesnapshot/{s}", "dataset": "peoplesnapshot", "part": "full body, real video (static camera, subject rotating in A-pose)",
             "identity": s, "clip": s, "gender": gart_gender(s),
             "video": f"{PS_ZIP}:people_snapshot_public/{s}/{s}.mp4", "fps": 30,
             "frame_index": "0-based cv2.VideoCapture sequential read of the mp4 = InstantAvatar image_{i:04d}.png (undistorted with camera.pkl); "
                            "train / test index the SORTED image list (GART loader: sorted(glob(images/*.png))[start:end + 1:skip])",
             "published_split": {"source": "InstantAvatar START_END (GART lib_data/instant_avatar_people_snapshot.py @ " + GART_COMMIT[:8] + ")",
                                 "train": se["train"], "test": se["test"], "val_unused": se["val"]},
             "gt": {"kind": "InstantAvatar's Anim-NeRF-refined SMPL (train and test frames), gendered SMPL (" + gart_gender(s) + ")",
                    "file": {sp: str(GART / "data/people_snapshot" / s / "poses" / f"anim_nerf_{sp}.npz") for sp in ("train", "test")},
                    "sha256": {sp: sha256(GART / "data/people_snapshot" / s / "poses" / f"anim_nerf_{sp}.npz") for sp in ("train", "test")},
                    "key": "row r of anim_nerf_<split>.npz = <split> frame list [r]; betas (1, 10) per file (train / test identical: max abs dev "
                           f"{bdev:.2e}); global_orient / body_pose (69) / transl in the camera frame (extrinsic identity)",
                    "body_model": f"/home/ubx858/ml-hugs/data/smpl/SMPL_{gart_gender(s).upper()}.pkl (npz copy: {SMPL_DIR}/SMPL_{gart_gender(s).upper()}.npz)",
                    "camera": "camera.pkl: K = [[f0, 0, c0], [0, f1, c1]], images undistorted with camera_k (InstantAvatar)"},
             "frames": frames, "n_frames": len(frames), "source_frame": tr[0], "n_recon": len(tr), "recon_frames": tr, "test_frames": te}
        chk = {"anim_nerf_rows_match_split": True, "train_test_disjoint": True, "betas_train_vs_test_max_abs_dev": bdev}
        if root is not None:
            n_prop, n_read, _ = ps_frames(root, s, []); mc = ps_mask_count(root, s); K, dist, wh = ps_camera(root, s)
            c.update({"wh": list(wh), "video_frames": n_read, "video_frames_cap_prop": n_prop, "masks_hdf5_frames": mc[0], "video_sha256": sha256(root / s / f"{s}.mp4"),
                      "camera": {"K": K.tolist(), "dist": dist.tolist()}})
            chk.update({"video_frames_eq_masks": n_read == mc[0], "split_within_video": max(frames) < n_read, "data": "available"})
            assert max(frames) < n_read, (s, max(frames), n_read)
        else:
            c.update({"wh": None, "video_frames": None, "masks_hdf5_frames": None})
            chk.update({"data": "BLOCKED: people_snapshot_public.zip is password-protected (all 192 members ZipCrypto-encrypted); the "
                                "password comes with the dataset licence agreement. Frame counts / camera filled when the data is readable."})
        c["checks"] = chk; clips.append(c)
    for seq in NEUMAN_SEQS:
        cams, imgs = neuman_colmap(seq); raw, al = neuman_raw(seq)
        names = sorted(imgs)                                                                         # NeuManReader order='video'
        files = sorted(os.listdir(NEUMAN / seq / "images"))
        cap = types.SimpleNamespace(captures=names)
        tr, va, te = gds(cap); tr, va, te = sorted(tr), sorted(va), sorted(te)
        fid = [int(n[:-4]) for n in names]
        K, wh = list(cams.values())[0]
        sc = np.array([np.sqrt(np.mean(np.diag(al[n][:3].T @ al[n][:3]))) for n in names])
        iso = max(float(np.abs(al[n][:3].T @ al[n][:3] / sc[i] ** 2 - np.eye(3)).max()) for i, n in enumerate(names))
        chk = {"images_eq_colmap_eq_alignments": files == names == sorted(al), "names_contiguous": names == [f"{i:05d}.png" for i in range(len(names))],
               "smpl_rows_eq_images": len(raw["pose"]) == len(names), "alignment_isotropic_max_dev": iso,
               "alignment_scale_range": [float(sc.min()), float(sc.max())], "betas_per_frame_max_abs_dev_from_median": float(np.abs(raw["betas"] - np.median(raw["betas"], 0)).max()),
               "hand_joints_22_23_max_abs": float(np.abs(raw["pose"][:, 66:72]).max()), "camera_model": "PINHOLE (one camera)"}
        assert chk["images_eq_colmap_eq_alignments"] and chk["names_contiguous"] and chk["smpl_rows_eq_images"] and iso < 1e-4
        frames = sorted(fid[i] for i in tr + te)
        c = {"id": f"neuman/{seq}", "dataset": "neuman", "part": "full body, real video (hand-held moving camera, outdoor)", "identity": f"neuman_{seq}",
             "clip": seq, "gender": "neutral", "video": str(NEUMAN / seq / "images"), "image_pattern": "%05d.png", "wh": list(wh),
             "video_frames": len(names), "fps": None,
             "frame_index": "0-based NeuMan capture index = image file number (images/%05d.png; COLMAP images sorted by name, all registered)",
             "published_split": {"source": "NeuMan / HUGS get_data_splits (ml-hugs hugs/datasets/neuman.py, verbatim)", "rule": gds_src,
                                 "n_captures": len(names), "train": len(tr), "val_unused": [fid[i] for i in va], "test": len(te)},
             "gt": {"kind": "NeuMan smpl_output_optimized (neutral SMPL, per-frame pose + betas) + alignments.npy (per-frame similarity to the COLMAP "
                            "world) + COLMAP camera pose; camera frame divided by the alignment scale (metric body)",
                    "file": {"smpl": str(NEUMAN / seq / "smpl_output_optimized.pkl"), "alignments": str(NEUMAN / seq / "alignments.npy"),
                             "colmap": str(NEUMAN / seq / "sparse")},
                    "sha256": {"smpl": sha256(NEUMAN / seq / "smpl_output_optimized.pkl"), "alignments": sha256(NEUMAN / seq / "alignments.npy"),
                               "images_txt": sha256(NEUMAN / seq / "sparse/images.txt"), "cameras_txt": sha256(NEUMAN / seq / "sparse/cameras.txt")},
                    "key": "row = image file number", "body_model": "/home/ubx858/ml-hugs/data/smpl/SMPL_NEUTRAL.pkl (NeuManReader.read_smpls gender='neutral')",
                    "camera": {"K": K.tolist()}},
             "frames": frames, "n_frames": len(frames), "source_frame": fid[tr[0]], "n_recon": len(tr), "recon_frames": [fid[i] for i in tr],
             "test_frames": [fid[i] for i in te], "checks": chk}
        clips.append(c)
    comp = []
    for ds, pool in (("peoplesnapshot", "InstantAvatar / GART People-Snapshot subjects (4 casual)"), ("neuman", "NeuMan / HUGS sequences used by the GUAVA-FB plan")):
        cs = [c for c in clips if c["dataset"] == ds]; nt = sum(len(c["test_frames"]) for c in clips)
        comp.append({"dataset": ds, "clips": len(cs), "identities": len({c["identity"] for c in cs}), "frames": sum(c["n_frames"] for c in cs),
                     "test_frames": sum(len(c["test_frames"]) for c in cs), "recon_frames": sum(c["n_recon"] for c in cs),
                     "share_of_clips_pct": round(100 * len(cs) / len(clips), 1),
                     "share_of_test_frames_pct": round(100 * sum(len(c["test_frames"]) for c in cs) / nt, 1), "pool": pool})
    J = {"name": "PublicBench-v1", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "phase3_dnt/eval_split/convert_publicbench.py split",
         "git_head_at_build": git_head(),
         "why": ("Second GUAVA-FB evaluation on public avatar benchmarks with their PUBLISHED train / test splits and pose sources "
                 "(PREREG 'PublicBench-v1'); rendering, crops and metrics are AvatarTest-v1's, so absolute numbers are not comparable with "
                 "the papers' tables (other resolutions / crops / backgrounds)."),
         "protocol": {"frames": "per clip, the ordered list 'frames' = published train frames + published test frames (val frames unused)",
                      "source": "source_frame = the first training frame; single-image methods (GUAVA, GUAVA-FB) see only this frame",
                      "recon": "recon_frames = the PUBLISHED TRAIN frames (explicit list; NOT the first half); per-subject methods train on these only",
                      "test": "test_frames = the PUBLISHED TEST frames; every method is scored on exactly these",
                      "driving": "GT rows: the dataset's published SMPL of each test frame (SMPL-X 2020 fitted to it for SMPL-X methods); "
                                 "estimator rows (SPRINT): the estimator's per-frame output",
                      "gt_body_model": "SMPL; People-Snapshot gendered (InstantAvatar / GART), NeuMan neutral (NeuMan)",
                      "smplx_conversion": "SMPL-X 2020 fitted to the GT SMPL mesh via PEAR's smplx2smpl correspondence: one betas200 per clip, "
                                          "per-frame root + 21 body joints + translation; jaw / eyes / fingers 0 (flat hands), exp 0; residual reported",
                      "metrics": "AvatarTest-v1 (GUAVA main/metrics.py PSNR, L1, SSIM, LPIPS-VGG on the 512 perspective person crop; GT = image x "
                                 "StyleMatte alpha on black; mean per clip, frame-weighted; per dataset and overall)",
                      "mask": "StyleMatte (EHM-Tracker stylematte_synth.pt) on the 1024 crop (the datasets' own masks are not used)",
                      "crop": "AvatarTest-v1 perspective crop: aimed at / sized by the projected GT SMPL mesh box clipped to the image, 1.1 x, 1024 px"},
         "composition": comp, "excluded_clips": [], "not_used": [
             {"dataset": "ZJU-MoCap", "reason": "only 1 of its 6 subjects on disk (PREREG PublicBench-v1)"},
             {"dataset": "NeuMan lab / parkinglot", "reason": "not in the registered 4 sequences (bike, citron, jogging, seattle)"},
             {"dataset": "People-Snapshot other subjects (sport / plaza / 1-2-casual)", "reason": "not in the InstantAvatar / GART 4-subject protocol"},
             {"dataset": "NeuMan val frames (second half of the held-out frames)", "reason": "published split's val part; not train, not test"}],
         "caveats": ["People-Snapshot GT poses are Anim-NeRF refinements (fitted on train AND test frames), NeuMan GT is its optimised ROMP fit: "
                     "pseudo-GT, not motion capture.",
                     "GT is SMPL; SMPL-X methods get an SMPL-X 2020 fit to the GT mesh (residual reported, flat hands, neutral face).",
                     "NeuMan test sets are small (3-10 frames per sequence, 27 in total); People-Snapshot subjects rotate in an A-pose in front of a static camera.",
                     "Released GUAVA's training data are undisclosed (overlap with these public sets cannot be excluded); GUAVA-FB never used them (FORMAT.md)."],
         "licences": LICENCES, "clips": clips}
    SPLIT.write_text(json.dumps(J, indent=1))
    print(json.dumps({"clips": len(clips), "composition": comp, "ps_data": root is not None, "sha256": sha256(SPLIT)}, indent=1))


# ------------------------------------------------------------------------------------------------ geometry helpers (torch)
def rotmat(aa):
    """Rodrigues, safe at 0 (gradient-wise)."""
    import torch
    th = torch.sqrt((aa * aa).sum(-1, keepdim=True) + 1e-16); k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, dtype=aa.dtype, device=aa.device) + s * K + (1 - c) * (K @ K)


def aa_np(R):
    import cv2
    return np.stack([cv2.Rodrigues(np.asarray(r, np.float64))[0].reshape(3) for r in np.asarray(R).reshape(-1, 3, 3)])


def R_np(aa):
    import cv2
    return np.stack([cv2.Rodrigues(np.asarray(v, np.float64).reshape(3))[0] for v in np.asarray(aa).reshape(-1, 3)])


class Ctx:
    """GPU models shared by all clips."""
    def __init__(self, dev):
        import torch
        sys.path.insert(0, str(cb.GUAVA)); from tools import eval_3dpw_standard as ev
        from smplx_lean import SMPLXLean
        self.torch, self.dev = torch, dev
        self.M = ev.load_smplx_to_smpl(CT.SMPLX2SMPL).to(dev)                                          # (6890, 10475) sparse
        self.smpl = {g: ev.build_smpl(SMPL_DIR, g, dev) for g in ("neutral", "male", "female")}
        self.X = SMPLXLean(str(CT.X2020), 200, dtype=torch.float32).to(dev)
        Ms = self.M.coalesce(); import scipy.sparse as sp
        self.Msp = sp.csr_matrix((Ms.values().cpu().numpy().astype(np.float64), (Ms.indices()[0].cpu().numpy(), Ms.indices()[1].cpu().numpy())), shape=tuple(Ms.shape))
        d20 = np.load(CT.X2020, allow_pickle=True)
        self.B20 = d20["shapedirs"][:, :, :200].astype(np.float64); self.T20 = d20["v_template"].astype(np.float64)
        self.faces_x = self.X.faces; self.faces_s = np.asarray(self.smpl["neutral"].faces, np.int64)

    def mapped(self, V):                                                                            # (n, 10475, 3) -> (n, 6890, 3)
        n = V.shape[0]; X = V.permute(1, 0, 2).reshape(V.shape[1], n * 3)
        return self.torch.sparse.mm(self.M, X).reshape(-1, n, 3).permute(1, 0, 2)

    def smpl_fwd(self, g, go, bp, be, tr=None):
        out = self.smpl[g](global_orient=go, body_pose=bp, betas=be, transl=tr)
        return out.vertices, out.joints[:, 0]

    def smplx_fwd(self, go, body, be):
        torch = self.torch; n = go.shape[0]
        R = torch.eye(3, device=go.device, dtype=go.dtype).repeat(n, 55, 1, 1)
        R = torch.cat([rotmat(go)[:, None], rotmat(body.reshape(n, 21, 3)), R[:, 22:]], 1)
        return self.X(None, be.expand(n, -1), None, rot=R)


def rest_shape_betas(X, Vs_rest):
    """Ridge least-squares betas200 (+ xyz offset) s.t. M (T20 + B20 b) + c = SMPL rest mesh (6890, 3)."""
    MB = X.Msp @ X.B20.reshape(10475, 600); MB = MB.reshape(6890, 3, 200)
    A = np.concatenate([MB.reshape(-1, 200), np.tile(np.eye(3), (6890, 1))], 1)
    y = (Vs_rest - X.Msp @ X.T20).reshape(-1)
    reg = np.sqrt(RIDGE) * np.concatenate([np.eye(200), np.zeros((200, 3))], 1)
    sol = np.linalg.lstsq(np.concatenate([A, reg]), np.concatenate([y, np.zeros(200)]), rcond=None)[0]
    err = np.linalg.norm((A @ sol - y).reshape(-1, 3), axis=1)
    return sol[:200], {"rest_fit_mm_mean": float(err.mean() * 1000), "rest_fit_mm_max": float(err.max() * 1000), "ridge": RIDGE}


def fit_smplx(X, VS, go0, body0, b0, iters):
    """SMPL-X 2020 fit to GT SMPL meshes VS (n, 6890, 3) (one frame of reference). -> params + per-frame residual (mm)."""
    torch = X.torch; dev = X.dev
    go = torch.tensor(go0, dtype=torch.float32, device=dev).requires_grad_(True)
    body = torch.tensor(body0, dtype=torch.float32, device=dev).requires_grad_(True)
    be = torch.tensor(b0, dtype=torch.float32, device=dev)[None].requires_grad_(True); bref = be.detach().clone()
    with torch.no_grad(): tr = (VS - X.mapped(X.smplx_fwd(go, body, be)[0])).mean(1)
    tr.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [go, body, tr], "lr": 0.01}, {"params": [be], "lr": 0.01}])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters); hist = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        V, _ = X.smplx_fwd(go, body, be); d = (X.mapped(V) + tr[:, None] - VS).norm(dim=-1)
        loss = d.mean() + 1e-5 * (be - bref).square().sum()
        loss.backward(); opt.step(); sch.step()
        if it % 250 == 0 or it == iters - 1: hist.append([it, float(d.mean() * 1000)])
    with torch.no_grad():
        V, J = X.smplx_fwd(go, body, be); d = (X.mapped(V) + tr[:, None] - VS).norm(dim=-1) * 1000
    return {"go": go.detach().cpu().numpy().astype(np.float64), "body": body.detach().cpu().numpy().astype(np.float64),
            "betas": be[0].detach().cpu().numpy().astype(np.float64), "transl": tr.detach().cpu().numpy().astype(np.float64),
            "res_mm": d.mean(1).cpu().numpy(), "res_mm_p95": torch.quantile(d, 0.95, dim=1).cpu().numpy(), "res_mm_max": d.max(1).values.cpu().numpy(), "hist": hist}


def fit_smpl_neutral(X, VS, go0, bp0, be0, tr0, iters, lr):
    """Neutral SMPL (one betas10) fitted to meshes VS (n, 6890, 3); returns the better of init / fit (by mean residual)."""
    torch = X.torch; dev = X.dev; f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)   # noqa: E731
    go, bp, be, tr = f(go0), f(bp0), f(be0).reshape(1, 10), f(tr0)
    with torch.no_grad():
        v, _ = X.smpl_fwd("neutral", go, bp, be.expand(len(go), -1), tr); e0 = (v - VS).norm(dim=-1).mean(1) * 1000
    init = [x.clone() for x in (go, bp, be, tr)]
    for x in (go, bp, be, tr): x.requires_grad_(True)
    opt = torch.optim.Adam([go, bp, be, tr], lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        v, _ = X.smpl_fwd("neutral", go, bp, be.expand(len(go), -1), tr)
        ((v - VS).norm(dim=-1).mean() + 1e-4 * be.square().mean()).backward(); opt.step(); sch.step()
    with torch.no_grad():
        v, _ = X.smpl_fwd("neutral", go, bp, be.expand(len(go), -1), tr); e1 = (v - VS).norm(dim=-1).mean(1) * 1000
    use_fit = float(e1.mean()) <= float(e0.mean())
    P = [x.detach() for x in (go, bp, be, tr)] if use_fit else init
    return {"go": P[0].cpu().numpy(), "bp": P[1].cpu().numpy(), "betas": P[2][0].cpu().numpy(), "transl": P[3].cpu().numpy(),
            "err_mm": (e1 if use_fit else e0).cpu().numpy(), "init_err_mm_mean": float(e0.mean()), "fit_err_mm_mean": float(e1.mean()), "used": "fit" if use_fit else "init"}


def raster_iou(v2d, faces, mask):
    """Silhouette of a projected mesh (cv2.fillConvexPoly per triangle, 512 px) vs mask (bool); -> IoU, precision, recall, sil."""
    import cv2
    sil = np.zeros(mask.shape, np.uint8); P = np.round(v2d * 16).astype(np.int32)                 # 4 fractional bits
    for tri in P[faces]: cv2.fillConvexPoly(sil, tri, 1, lineType=cv2.LINE_8, shift=4)
    s = sil.astype(bool); i = (s & mask).sum(); u = (s | mask).sum()
    return float(i / max(u, 1)), float(i / max(s.sum(), 1)), float(i / max(mask.sum(), 1)), s


# ------------------------------------------------------------------------------------------------ GT per clip (camera frame of the original image)
def load_gt(c, root):
    """-> dict gender, go (n,3), bp (n,69), betas (n,10), transl (n,3) [original camera frame, metric], K (n,3,3), wh, images loader."""
    frames = c["frames"]; n = len(frames)
    if c["dataset"] == "neuman":
        seq = c["clip"]; cams, imgs = neuman_colmap(seq); raw, al = neuman_raw(seq); K0, wh = list(cams.values())[0]
        go, bp, be, Rc, tc, sc = [], [], [], [], [], []
        for f in frames:
            nm = f"{f:05d}.png"; A4 = np.eye(4); A4[:, :3] = al[nm]; T = A4.T                       # x_world = T [x; 1] (read_smpls)
            S = T[:3, :3]; s = float(np.sqrt(np.mean(np.diag(S @ S.T)))); Rw, tw = imgs[nm]["R"], imgs[nm]["t"]
            Rc.append(Rw @ S / s); tc.append((Rw @ T[:3, 3] + tw) / s); sc.append(s)
            go.append(raw["pose"][f, :3]); bp.append(raw["pose"][f, 3:]); be.append(raw["betas"][f])
        return {"gender": "neutral", "go": np.array(go), "bp": np.array(bp), "betas": np.array(be), "Rc": np.array(Rc), "tc": np.array(tc), "scale": np.array(sc),
                "K": np.repeat(K0[None], n, 0), "wh": wh, "orig_src": [str(NEUMAN / seq / "images" / f"{f:05d}.png") for f in frames],
                "load": lambda want: neuman_images(seq, want)}
    s = c["clip"]; P = ps_poses(s); rows = {}
    for sp, lst in (("train", c["recon_frames"]), ("test", c["test_frames"])):
        for r, f in enumerate(lst): rows[f] = (sp, r)
    go = np.stack([P[rows[f][0]]["global_orient"][rows[f][1]] for f in frames]).astype(np.float64)
    bp = np.stack([P[rows[f][0]]["body_pose"][rows[f][1]] for f in frames]).astype(np.float64)
    tr = np.stack([P[rows[f][0]]["transl"][rows[f][1]] for f in frames]).astype(np.float64)
    be = np.repeat(P["train"]["betas"].reshape(1, 10).astype(np.float64), n, 0)                   # GART: smpl_params["betas"][0]
    K, dist, wh = ps_camera(root, s)
    return {"gender": c["gender"], "go": go, "bp": bp, "betas": be, "Rc": np.repeat(np.eye(3)[None], n, 0), "tc": tr, "scale": np.ones(n),
            "K": np.repeat(K[None], n, 0), "wh": wh, "orig_src": [f"{PS_ZIP}:people_snapshot_public/{s}/{s}.mp4#frame={f} (cv2 sequential read, cv2.undistort(K, camera_k))" for f in frames],
            "load": lambda want: ps_frames(root, s, want)[2], "transl_given": True}


def neuman_dapose_check(c, gt, X):
    """NeuMan's own read_smpls path (DA-pose LBS + alignment) for 2 frames vs our standard-forward camera mesh (mm, metric)."""
    import importlib.util, torch
    for nm_ in ("bool", "int", "float", "complex", "object", "str"):                                 # chumpy shim (as convert_test.maps)
        if nm_ not in np.__dict__: setattr(np, nm_, getattr(__import__("builtins"), nm_))
    if "unicode" not in np.__dict__: np.unicode = str
    spec = importlib.util.spec_from_file_location("neuman_smpl", NEUMAN_SMPL_PY); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    bm = m.SMPL(str(SMPL_PKL), gender="neutral", device=torch.device("cpu")); out = []
    cams, imgs = neuman_colmap(c["clip"]); raw, al = neuman_raw(c["clip"])
    for i in (0, len(c["frames"]) - 1):
        f = c["frames"][i]; nm = f"{f:05d}.png"; pose = raw["pose"][f][None].astype(np.float32)
        betas = np.zeros((1, bm.shapedirs.shape[-1]), np.float32); betas[0, :10] = raw["betas"][f]          # pkl has 300 PCs; first 10 used
        A4 = np.eye(4); A4[:, :3] = al[nm]
        da = np.zeros_like(pose).reshape(-1, 3); da[1] = [0, 0, 1.0]; da[2] = [0, 0, -1.0]; da = da.reshape(1, -1)
        _, T_t2pose = bm.verts_transformations(return_tensor=False, poses=pose, betas=betas, concat_joints=True)
        _, T_t2da = bm.verts_transformations(return_tensor=False, poses=da, betas=betas, concat_joints=True)
        T = A4.T @ (T_t2pose @ np.linalg.inv(T_t2da))
        dv, dj = bm(return_tensor=False, return_joints=True, poses=da, betas=betas)
        h = np.concatenate([np.concatenate([dv, dj], 0), np.ones((len(dv) + len(dj), 1))], 1)
        world = np.einsum("BNi,Bi->BN", T, h)[:6890, :3]
        cam = (world @ imgs[nm]["R"].T + imgs[nm]["t"]) / gt["scale"][i]
        with torch.no_grad():
            v, _ = X.smpl_fwd("neutral", torch.tensor(gt["go"][i:i + 1], dtype=torch.float32, device=X.dev), torch.tensor(gt["bp"][i:i + 1], dtype=torch.float32, device=X.dev),
                              torch.tensor(gt["betas"][i:i + 1], dtype=torch.float32, device=X.dev))
        ours = v[0].cpu().numpy().astype(np.float64) @ gt["Rc"][i].T + gt["tc"][i]
        out.append(float(np.linalg.norm(ours - cam, axis=1).mean() * 1000))
    return out


# ------------------------------------------------------------------------------------------------ one clip
def convert_clip(c, X, matter, M, root, J, log):
    import cv2, torch, torchvision
    t00 = time.time(); ds, name = c["dataset"], c["clip"]; frames = c["frames"]; n = len(frames); dev = X.dev
    keys = [f"frame_{f:06d}" for f in frames]; G = cb.geo()
    gt = load_gt(c, root); g = gt["gender"]; f32 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)   # noqa: E731
    # 1. GT SMPL mesh in the ORIGINAL camera frame (metric): x = Rc SMPL(go, bp, betas) + tc (PS: Rc = I, tc = transl)
    with torch.no_grad():
        v0, j0 = X.smpl_fwd(g, f32(gt["go"]), f32(gt["bp"]), f32(gt["betas"]))
    Rc, tc = gt["Rc"], gt["tc"]
    VS0 = torch.einsum("nij,nvj->nvi", f32(Rc), v0) + f32(tc)[:, None]
    # GT params in the camera frame: go_c = Rc R(go); transl_c = Rc J0 + tc - J0  (exact fold; checked)
    J0 = j0.cpu().numpy().astype(np.float64)                                                        # rest pelvis (root rotates about it)
    go_c = aa_np(Rc @ R_np(gt["go"])); tr_c = np.einsum("nij,nj->ni", Rc, J0) + tc - J0
    with torch.no_grad():
        vchk, _ = X.smpl_fwd(g, f32(go_c), f32(gt["bp"]), f32(gt["betas"]), f32(tr_c)); fold_gt_mm = float((vchk - VS0).norm(dim=-1).max() * 1000)
    dap = neuman_dapose_check(c, gt, X) if ds == "neuman" else None
    # 2. SMPL-X 2020 fit (camera frame)
    t1 = time.time()
    with torch.no_grad():
        vr, _ = X.smpl_fwd(g, torch.zeros(1, 3, device=dev), torch.zeros(1, 69, device=dev), f32(np.median(gt["betas"], 0)).reshape(1, 10))
    b0, rest = rest_shape_betas(X, vr[0].cpu().numpy().astype(np.float64))
    fx = fit_smplx(X, VS0, go_c, gt["bp"][:, :63], b0, SMPLX_ITERS); t_fit = time.time() - t1
    b200 = fx["betas"]
    # 3. float64 SMPL-X 2020 (GUAVA layer math) of the fitted params, no translation; exact like ub_geometry's VN / JN
    pose = np.zeros((n, 165)); pose[:, :3] = fx["go"]; pose[:, 3:66] = fx["body"]
    VN, JN = G["N"](torch.tensor(pose), torch.tensor(b200)[None].expand(n, -1)); VN, JN = VN.numpy(), JN.numpy()
    t = fx["transl"]
    VSn = VS0.cpu().numpy().astype(np.float64); Msp = X.Msp
    W, H = gt["wh"]; geo = []
    for i in range(n):
        K = gt["K"][i]; fxx, fyy = K[0, 0], K[1, 1]; f = float(fxx); Ki = np.linalg.inv(K)
        XS = VSn[i]; XN = VN[i] + t[i]
        res_mm = float(np.linalg.norm(Msp @ XN - XS, axis=1).mean() * 1000)
        p = XS @ K.T; p = p[:, :2] / p[:, 2:]
        lo = np.clip(p.min(0), 0, [W, H]); hi = np.clip(p.max(0), 0, [W, H])                       # GT-mesh box clipped to the image
        Rv = cb.rot_onto_z(Ki @ np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.0]))
        cor = np.array([[lo[0], lo[1], 1], [hi[0], lo[1], 1], [lo[0], hi[1], 1], [hi[0], hi[1], 1.0]]) @ Ki.T @ Rv.T
        q = f * cor[:, :2] / cor[:, 2:]
        L = cb.BODY_SCALE * float((q.max(0) - q.min(0)).max()); s = CROP / L
        Kv = np.array([[f * s, 0, CROP / 2], [0, f * s, CROP / 2], [0, 0, 1.0]])
        Hm = Kv @ Rv @ Ki; invtanfov = 2 * f / L
        pj = lambda X3: (lambda u: (lambda w: w[:, :2] / w[:, 2:])((u / u[:, 2:]) @ Hm.T))(X3 @ K.T)   # noqa: E731  pinhole K, then H
        v2d_src = pj(XN); v2d_gt = pj(XS); v2d_gtmap = pj(Msp @ XN)
        Rroot = Rv @ cv2.Rodrigues(pose[i, :3])[0]; J0x = JN[i, 0]; tp = Rv @ (J0x + t[i]) - J0x
        Dm = np.diag([-1.0, -1.0, 1.0]); RT = np.concatenate([Dm, (Dm @ tp)[:, None]], 1).astype(np.float32)
        Vmodel = (VN[i] - J0x) @ Rv.T + J0x
        v2d = cb.project_guava(Vmodel, RT.astype(np.float64), invtanfov)
        fold_px512 = float(np.linalg.norm(v2d - v2d_src, axis=1).mean() * 512 / CROP)                # same mesh: camera fold exactness
        fit_px512 = float(np.linalg.norm(v2d_gtmap - v2d_gt, axis=1).mean() * 512 / CROP)            # SMPL-X fit error seen in the crop
        pc = pose[i].astype(np.float32)
        smplx = {"global_pose": cb.aa_from_R(Rroot).astype(np.float32), "body_pose": pc[3:66].reshape(21, 3).copy(),
                 "jaw_pose": np.zeros((1, 3), np.float32), "eye_pose": np.zeros((2, 3), np.float32),
                 "left_hand_pose": np.zeros((15, 3), np.float32), "right_hand_pose": np.zeros((15, 3), np.float32),
                 "exp": np.zeros(50, np.float32), "camera_RT_params": RT, "body_cam": np.zeros(3, np.float32)}

        def part(ids, scale):
            pts = v2d[ids]; lt, rb = pts.min(0), pts.max(0)
            Mo = cb.box_crop_matrix([lt[0], lt[1], rb[0], rb[1]], PART, scale)
            return {"M_o2c": Mo, "M_c2o": np.linalg.inv(Mo).astype(np.float32)}, bool(((pts >= 0) & (pts < CROP)).all(1).mean() > 0.5)
        hc, hv = part(G["head"], cb.HEAD_SCALE); lc, lv = part(G["lhand"], cb.HAND_SCALE); rc, rv = part(G["rhand"], cb.HAND_SCALE)
        Mlr = np.diag([BODY_LR / CROP, BODY_LR / CROP, 1.0]).astype(np.float32); I3 = np.eye(3, dtype=np.float32)
        rec = {"smplx_coeffs": smplx,
               "flame_coeffs": {"expression_params": np.zeros(50, np.float32), "jaw_params": np.zeros(3, np.float32),
                                "pose_params": np.zeros(3, np.float32), "neck_pose_params": np.zeros(3, np.float32),
                                "eye_pose_params": np.zeros(6, np.float32), "eyelid_params": np.zeros(2, np.float32),
                                "camera_RT_params": RT.copy(), "cam": np.zeros(3, np.float32)},
               "body_crop": {"M_o2c": Mlr, "M_c2o": np.linalg.inv(Mlr).astype(np.float32), "M_o2c-hd": I3.copy(), "M_c2o-hd": I3.copy()},
               "head_crop": hc, "left_hand_crop": lc, "right_hand_crop": rc, "invtanfov": float(invtanfov), "view": "0"}
        geo.append({"rec": rec, "H": Hm, "Rv": Rv, "v2d_src": v2d_src.astype(np.float32), "v2d": v2d.astype(np.float32), "v2d_gt": v2d_gt.astype(np.float32),
                    "res_mm": res_mm, "res_px512": fold_px512, "fit_px512": fit_px512, "invtanfov": float(invtanfov), "crop_side_px": L,
                    "depth_m": float(XS[:, 2].mean()), "zmin_m": float(XS[:, 2].min()), "vis": {"head": hv, "lhand": lv, "rhand": rv},
                    "focal_xy_rel": abs(fxx - fyy) / fxx, "box_clipped": bool((p.min(0) < 0).any() or (p.max(0) > [W, H]).any())})
    t2 = time.time()
    # 4. original frames -> 1024 crops (+ JPEG q95) ; PS originals of source + test kept for the SPRINT row
    ori = gt["load"](frames); crops, jpgs = [], []
    for fr, gi in zip(frames, geo):
        im = ori[fr]; assert im.shape[1] == W and im.shape[0] == H, (im.shape, W, H)
        cr = cv2.warpPerspective(im, gi["H"], (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        crops.append(cr); jpgs.append(torchvision.io.encode_jpeg(torch.from_numpy(cr).permute(2, 0, 1).contiguous(), quality=95).numpy().tobytes())
    t3 = time.time(); alphas = matter(np.stack(crops)); tm = time.time() - t3; del crops
    fs = M["Wf"] @ b200
    res = {"g": geo, "jpgs": jpgs, "betas200": b200}
    meta = {"id": c["id"], "dataset": ds, "clip": name, "frames_keys": keys, "missing_frames": [], "dropped_zmin": [],
            "res_mm": [x["res_mm"] for x in geo], "res_px512": [x["res_px512"] for x in geo], "fit_px512": [x["fit_px512"] for x in geo],
            "invtanfov": [x["invtanfov"] for x in geo], "crop_side_px": [x["crop_side_px"] for x in geo], "depth_m": [x["depth_m"] for x in geo],
            "part_visible": [x["vis"] for x in geo], "alpha_mean": [float(x.mean() / 255) for x in alphas],
            "shape_info": {"betas200_init": "ridge lstsq rest-pose fit", **rest, "betas200_norm": float(np.linalg.norm(b200))},
            "times": {"fit": t_fit, "geom": t2 - t1 - t_fit, "decode_crop": t3 - t2, "matte": tm}, "licence": J["licences"][ds],
            "note": ("res_mm: SMPL-X 2020 fit vs GT SMPL mesh through smplx2smpl (mm); res_px512: camera-fold check (same mesh); fit_px512: fit "
                     "error projected (px at 512); gt_proj v2d_1024 = fitted SMPL-X by the original K + H, v2d_gt_smpl_1024 = GT SMPL by K + H")}
    CT.TMP = TMP
    mjpg = CT.write_tracked(E2 / "data" / ds / name, res, keys, alphas, fs, meta)
    # gt_proj extra key: GT SMPL (6890) projection (write_tracked wrote the SMPL-X ones); rewrite the file in place on /vault
    gp = dict(np.load(E2 / "data" / ds / name / "gt_proj.npz"))
    gi = [keys.index(str(k)) for k in gp["frame_key"]]
    gp["v2d_gt_smpl_1024"] = np.stack([geo[i]["v2d_gt"] for i in gi])
    gp["note"] = np.array("v2d_1024: SMPL-X 2020 fitted to the GT SMPL (camera frame) by the original K, mapped by H; v2d_converted_1024: the "
                          "record's GUAVA projection; v2d_gt_smpl_1024: GT SMPL mesh (6890 vertices) by K and H")
    loc = TMP / "gp.npz"; np.savez_compressed(loc, **gp); shutil.copyfile(loc, E2 / "data" / ds / name / "gt_proj.npz"); loc.unlink()
    # 5. base package (local, then one copy) + gt512
    dec = lambda b: torchvision.io.decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8)).float() / 255.0    # noqa: E731
    rs = lambda x: torchvision.transforms.functional.resize(x, (512, 512), antialias=True)                           # noqa: E731
    u8 = lambda x: (x.clamp(0, 1) * 255).round().to(torch.uint8)                                                      # noqa: E731
    LB = TMP / "base" / ds / name; LG = TMP / "gt512" / ds / name
    for p in (LB, LG):
        if p.exists(): shutil.rmtree(p)
    for p in (LB / "images", LB / "masks", LG): p.mkdir(parents=True, exist_ok=True)
    fset = set(f"frame_{f:06d}" for f in c["test_frames"]); msk512 = {}
    for k, jb, mb in zip(keys, jpgs, mjpg):
        im, ms = rs(dec(jb)), rs(dec(mb))
        torchvision.io.write_png(u8(im), str(LB / "images" / f"{k}.png")); torchvision.io.write_png(u8(ms[:1]), str(LB / "masks" / f"{k}.png"))
        msk512[k] = u8(ms[0]).numpy() > 127
        if k in fset: torchvision.io.write_png(u8(im * ms), str(LG / f"{k}.png"))
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
    with torch.no_grad():
        Vx, _ = X.X(torch.tensor(pz, device=dev), torch.tensor(sx["betas"], device=dev)[None].expand(n, -1), torch.tensor(sx["expression"], device=dev))
        Vx = Vx + torch.tensor(sx["transl"], device=dev)[:, None]
    q = (Vx[..., :2] / Vx[..., 2:]).cpu().numpy() * K5[:, 0, 0][:, None, None] + 256
    chk_px = float(np.abs(q - np.stack([x["v2d"] for x in geo]) / 2).max())                       # contract camera vs GUAVA projection
    # GT SMPL in the CROP camera frame (rotate by Rv about the camera centre): smpl_gt.npz (exact) and smpl.npz (contract: neutral, one betas)
    Rv = np.stack([x["Rv"] for x in geo])
    go_v = aa_np(Rv @ R_np(go_c)); tr_v = np.einsum("nij,nj->ni", Rv, J0 + tr_c) - J0
    VSv = torch.einsum("nij,nvj->nvi", f32(Rv), VS0)
    with torch.no_grad():
        vg, _ = X.smpl_fwd(g, f32(go_v), f32(gt["bp"]), f32(gt["betas"]), f32(tr_v)); gt_crop_mm = float((vg - VSv).norm(dim=-1).max() * 1000)
    np.savez(LB / "smpl_gt.npz", keys=np.array(keys), gender=np.array(g), model=np.array(f"/home/ubx858/ml-hugs/data/smpl/SMPL_{g.upper()}.pkl"),
             betas=gt["betas"].astype(np.float32), global_orient=go_v.astype(np.float32), body_pose=gt["bp"].reshape(n, 23, 3).astype(np.float32),
             transl=tr_v.astype(np.float32), fit_err_mm=np.zeros(n, np.float32),
             note=np.array("the dataset's GT SMPL (published body model / gender, per-frame betas) in the crop camera frame of cameras.npz "
                           "(OpenCV axes): verts = smpl_<gender>(betas[i], global_orient[i], body_pose[i], transl[i]).vertices"))
    bmed = np.median(gt["betas"], 0)
    if ds == "neuman":
        with torch.no_grad(): _, jm = X.smpl_fwd("neutral", torch.zeros(1, 3, device=dev), torch.zeros(1, 69, device=dev), f32(bmed).reshape(1, 10))
        jm = jm[0].cpu().numpy().astype(np.float64); tr_init = np.einsum("nij,nj->ni", Rv, J0 + tr_c) - jm
        sm = fit_smpl_neutral(X, VSv, go_v, gt["bp"], bmed, tr_init, SMPL_ITERS, 0.002)
    else:
        with torch.no_grad(): _, jm = X.smpl_fwd("neutral", torch.zeros(1, 3, device=dev), torch.zeros(1, 69, device=dev), torch.zeros(1, 10, device=dev))
        jm = jm[0].cpu().numpy().astype(np.float64); tr_init = np.einsum("nij,nj->ni", Rv, J0 + tr_c) - jm
        sm = fit_smpl_neutral(X, VSv, go_v, gt["bp"], np.zeros(10), tr_init, SMPL_ITERS, 0.02)
    np.savez(LB / "smpl.npz", keys=np.array(keys), betas=np.asarray(sm["betas"], np.float32), global_orient=np.asarray(sm["go"], np.float32),
             body_pose=np.asarray(sm["bp"], np.float32).reshape(n, 23, 3), transl=np.asarray(sm["transl"], np.float32), fit_err_mm=np.asarray(sm["err_mm"], np.float32))
    # crop_h.npz: original image -> 1024 crop (cv2.warpPerspective matrix, pixel indices), how to load the original
    np.savez(LB / "crop_h.npz", keys=np.array(keys), H=np.stack([x["H"] for x in geo]).astype(np.float64), orig_wh=np.array([W, H]),
             orig_src=np.array(gt["orig_src"]), K_orig=gt["K"].astype(np.float64), Rv=Rv.astype(np.float64), invtanfov=itf,
             crop_side_px=np.array([x["crop_side_px"] for x in geo]),
             note=np.array("H: original image pixel (x, y, 1) -> 1024 crop pixel (cv2.warpPerspective(orig, H, (1024, 1024)), bilinear, "
                           "border 0; same pixel-index convention as the converter); 512 crop = S512 @ H with S512 = [[.5,0,-.25],[0,.5,-.25],[0,0,1]] "
                           "for pixel centres at integers. orig_src: NeuMan = image path (read with cv2.imread, BGR->RGB); People-Snapshot = mp4 "
                           "frame (cv2 sequential read, cv2.undistort(K, camera_k)), saved as orig/<key>.png for the source + test keys. "
                           "K_orig: pinhole K of the original (undistorted) image; Rv: virtual crop-camera rotation (x_crop = Rv x_cam)."))
    if ds == "peoplesnapshot":
        (LB / "orig").mkdir(exist_ok=True)
        for f in [c["source_frame"]] + c["test_frames"]:
            cv2.imwrite(str(LB / "orig" / f"frame_{f:06d}.png"), cv2.cvtColor(ori[f], cv2.COLOR_RGB2BGR))
    # 6. checks: silhouette IoU (SMPL-X of smplx2020.npz, and GT SMPL) vs the StyleMatte mask at 512; overlays
    qs = (vg[..., :2] / vg[..., 2:]).cpu().numpy() * K5[:, 0, 0][:, None, None] + 256
    iou_x, iou_s, prec_x, rec_x = [], [], [], []
    for i, k in enumerate(keys):
        a_, p_, r_, _ = raster_iou(q[i], X.faces_x, msk512[k]); iou_x.append(a_); prec_x.append(p_); rec_x.append(r_)
        iou_s.append(raster_iou(qs[i], X.faces_s, msk512[k])[0])
    with torch.no_grad():
        vsm, _ = X.smpl_fwd("neutral", f32(sm["go"]), f32(sm["bp"]), f32(sm["betas"]).reshape(1, 10).expand(n, -1), f32(sm["transl"]))
    qn = (vsm[..., :2] / vsm[..., 2:]).cpu().numpy() * K5[:, 0, 0][:, None, None] + 256
    tiles = []
    for i in np.unique(np.round(np.linspace(0, n - 1, 4)).astype(int)):
        im = cv2.cvtColor(cv2.imread(str(LB / "images" / f"{keys[i]}.png")), cv2.COLOR_BGR2RGB)
        for pts, col in ((q[i][::7], (0, 255, 0)), (qn[i][::5], (255, 0, 255))):
            for x_, y_ in pts.astype(int): cv2.circle(im, (int(x_), int(y_)), 1, col, -1)
        cv2.putText(im, f"{keys[i]} IoU {iou_x[i]:.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(im)
    cv2.imwrite(str(LB / "overlay.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    # original-image overlay (source + first test frame): GT SMPL by the original K (red) and the crop square (yellow)
    to = []
    for f in [c["source_frame"], c["test_frames"][0]]:
        i = frames.index(f); im = ori[f].copy(); pp = VSn[i] @ gt["K"][i].T; pp = pp[:, :2] / pp[:, 2:]
        for x_, y_ in pp[::6].astype(int): cv2.circle(im, (int(x_), int(y_)), 1, (255, 0, 0), -1)
        Hi = np.linalg.inv(geo[i]["H"]); sq = np.array([[0, 0, 1], [CROP, 0, 1], [CROP, CROP, 1], [0, CROP, 1.0]]) @ Hi.T
        cv2.polylines(im, [np.round(sq[:, :2] / sq[:, 2:]).astype(np.int32)], True, (255, 255, 0), 2)
        sc_ = 540 / im.shape[0]; to.append(cv2.resize(im, (int(im.shape[1] * sc_), 540), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(LB / "overlay_orig.jpg"), cv2.cvtColor(np.concatenate(to, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    tk = [k for k in keys if k in fset]; rk = [f"frame_{f:06d}" for f in c["recon_frames"]]
    fj = {"dataset": ds, "clip": name, "id": c["id"], "keys": keys, "source": f"frame_{c['source_frame']:06d}", "recon": rk, "test": tk,
          "fps": c["fps"], "dropped_from_split": [],
          "note": ("frame_XXXXXX = People-Snapshot video frame (InstantAvatar image index, 30 fps); recon = InstantAvatar train frames, test = "
                   "InstantAvatar test frames" if ds == "peoplesnapshot" else
                   "frame_XXXXXX = NeuMan image number (images/%05d.png); recon = NeuMan train split, test = NeuMan test split (val unused)")}
    assert fj["source"] in keys and set(tk) == fset and not (set(rk) & set(tk)) and len(rk) == c["n_recon"]
    (LB / "frames.json").write_text(json.dumps(fj, indent=1))
    # 7. copy finished folders to /vault (frames.json last = done marker), remove local copies
    for src, dst in ((LG, E2 / "gt512" / ds / name), (LB, E2 / "base" / ds / name)):
        tmp = dst.parent / (dst.name + ".partial")
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for p in sorted(src.rglob("*")):
            if p.is_dir() or p.name == "frames.json": continue
            (tmp / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, tmp / p.relative_to(src))
        if (src / "frames.json").exists(): shutil.copyfile(src / "frames.json", tmp / "frames.json")
        if dst.exists(): shutil.rmtree(dst)                                                        # only our own earlier output
        os.rename(tmp, dst); shutil.rmtree(src)
    rec = {"id": c["id"], "frames": n, "recon": len(rk), "test": len(tk), "gender": g,
           "smplx_fit_mm_mean": float(np.mean(meta["res_mm"])), "smplx_fit_mm_max_frame": float(np.max(meta["res_mm"])),
           "smplx_fit_mm_p95_vertex_mean": float(np.mean(fx["res_mm_p95"])), "smplx_fit_mm_max_vertex": float(np.max(fx["res_mm_max"])),
           "smplx_fit_px512_mean": float(np.mean(meta["fit_px512"])), "smplx_fit_hist": fx["hist"], "betas200_norm": float(np.linalg.norm(b200)), **rest,
           "fold_px512_mean": float(np.mean(meta["res_px512"])), "contract_cam_check_px": chk_px, "gt_fold_check_mm": fold_gt_mm, "gt_crop_frame_check_mm": gt_crop_mm,
           "neuman_dapose_vs_standard_mm": dap,
           "smpl_npz": {"used": sm["used"], "init_err_mm_mean": sm["init_err_mm_mean"], "fit_err_mm_mean": sm["fit_err_mm_mean"],
                        "fit_err_mm_max": float(np.max(sm["err_mm"]))},
           "iou_smplx_mean": float(np.mean(iou_x)), "iou_smplx_min": float(np.min(iou_x)), "iou_gt_smpl_mean": float(np.mean(iou_s)),
           "iou_gt_smpl_min": float(np.min(iou_s)), "sil_precision_smplx_mean": float(np.mean(prec_x)), "sil_recall_smplx_mean": float(np.mean(rec_x)),
           "alpha_mean": float(np.mean(meta["alpha_mean"])), "box_clipped_frames": int(sum(x["box_clipped"] for x in geo)),
           "crop_side_px_mean": float(np.mean(meta["crop_side_px"])), "invtanfov_range": [float(itf.min()), float(itf.max())],
           "part_visible_frac": {p: float(np.mean([v[p] for v in meta["part_visible"]])) for p in ("head", "lhand", "rhand")},
           "overlay": str(E2 / "base" / ds / name / "overlay.jpg"), "overlay_orig": str(E2 / "base" / ds / name / "overlay_orig.jpg"),
           "per_frame": {"keys": keys, "smplx_fit_mm": meta["res_mm"], "iou_smplx": iou_x, "iou_gt_smpl": iou_s, "smpl_npz_err_mm": np.asarray(sm["err_mm"]).tolist()},
           "times_s": {"fit": round(t_fit, 1), "total": round(time.time() - t00, 1), "matte": round(tm, 1)}}
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_frame", "smplx_fit_hist")}), flush=True)
    return rec


def run(a):
    import torch
    J = json.loads(SPLIT.read_text()); clips = J["clips"]
    if a.clips: want = a.clips.split(","); clips = sorted([c for c in clips if c["id"] in want], key=lambda c: want.index(c["id"]))
    root = ps_root(a.ps_root)
    todo = [c for c in clips if not (E2 / "base" / c["dataset"] / c["clip"] / "frames.json").exists()]
    blocked = [c["id"] for c in todo if c["dataset"] == "peoplesnapshot" and (root is None or c.get("wh") is None)]
    todo = [c for c in todo if c["id"] not in blocked]
    print(f"{len(clips)} clips, {len(todo)} to do, blocked: {blocked}", flush=True)
    M = CT.maps(); (E2 / "reports").mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    (E2 / "reports" / "conversion_maps.json").write_text(json.dumps(M["report"], indent=1))
    dev = torch.device("cuda:0"); X = Ctx(dev); matter = cb.Matter()
    with open(E2 / "reports" / "convert_log.jsonl", "a") as log:
        for c in todo:
            convert_clip(c, X, matter, M, root, J, log)
    if blocked:
        (E2 / "reports" / "blocked.json").write_text(json.dumps({"blocked": blocked, "reason": "People-Snapshot zip password-protected; "
                                                                 "rerun with PS_ZIP_PASSWORD or --ps-root (split stage first)"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("split", "run")); ap.add_argument("--clips", default=""); ap.add_argument("--ps-root", default="")
    a = ap.parse_args()
    {"split": split, "run": run}[a.stage](a)


if __name__ == "__main__":
    main()
