"""ValBench-S (UBody only): validation clips that SPRINT never trained on, for SPRINT-side interventions (PREREG phase 4, H4).
All 20 ValBench-v1 UBody clips are in SPRINT's UBody training rows, so SPRINT changes cannot be judged on them. Candidates here =
clips of SPRINT's `ubody_labels` VAL split (2 % of clips held out when SPRINT's UBody rows were built), never tracked for
GUAVA-FB / a20 (c2t), whose identity has no SPRINT training row and is disjoint from ValBench-v1 and AvatarTest-v1; then the
video rules of ValBench-v1's UBody extension (720p, 150-450 frames on the 30 fps timeline) and AvatarTest-v1's subject-track /
>= 60 GT-frame rules (build_avatar_test_split.ubody). Same frame protocol (first frame source, first half recon, latter half test).
Validation only, never a test set.
Usage: python build_valbench_s.py --out valbench_s.json"""
import argparse, json, re, time
from pathlib import Path
import cv2
import numpy as np
import build_avatar_test_split as B

HERE = Path(__file__).resolve().parent
TMP = Path("/dev/shm/ubx858_eval/valbench_s_build"); TMP.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "valbench_s.json")); a = ap.parse_args()

d = np.load("/raid/ubx858/outputs/dnt/data/ubody_labels/annotations.npz", allow_pickle=True)
clip = np.array([p.split("/")[3] for p in d["path"]]); split = d["split"]
vid = {c: p.split("/")[1:4] for c, p in zip(clip, d["path"])}                                   # scene, video, clip
tr_ids = {B.ub_identity(c) for c in clip[split == "train"]}
c2t = {c["clip"] for c in json.load(open("/raid/ubx858/outputs/dnt/c2t/split.json"))["clips"]}
eval_ids = {c["identity"] for f in ("valbench_v1.json", "avatar_test_v1.json") for c in json.load(open(HERE / f))["clips"] if c["dataset"] == "ubody"}
banned = {B.ub_identity(n) for n in c2t} | eval_ids | tr_ids
cands = sorted(c for c in set(clip[split == "val"]) if c not in c2t and B.ub_identity(c) not in banned)
clips, rej = [], {}
for c in cands:
    v = Path("/raid/ubx858/datasets/expanded/UBody/videos") / vid[c][0] / vid[c][1] / f"{c}.mp4"
    cap = cv2.VideoCapture(str(v)); n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
    if not (fps > 0 and 150 <= n * 30 / fps <= 450 and h == 720):
        rej[c] = {"frames_30fps": round(n * 30 / fps) if fps > 0 else None, "h": h}; continue
    clips.append({"scene": vid[c][0], "identity": B.ub_identity(c), "clip": c, "path": str(v), "frames": round(n * 30 / fps),
                  "fps": round(fps, 3), "wh": [w, h], "sha256": None})
print("candidates", len(cands), "video rule kept", len(clips), "rejected", rej, flush=True)
(TMP / "c1_like.json").write_text(json.dumps({"clips": clips})); B.C1 = TMP / "c1_like.json"
res = B.ubody()
kept = [c for c in res if c["n_frames"] >= B.MIN_GT]
for c in kept: c["checks"]["valbench_s"] = {"sprint_split": "val", "identity_sprint_train_rows": 0}
excl = [{"id": c["id"], "gt_frames": c["n_frames"]} for c in res if c not in kept] + [{"id": f"ubody/{k}", "video_rule": v} for k, v in rej.items()]
J = {"name": "ValBench-S", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "phase3_dnt/eval_split/build_valbench_s.py",
     "purpose": "validation-only UBody clips unseen by SPRINT (SPRINT-side interventions); never a test set",
     "protocol": "as avatar_test_v1.json", "composition": B.summarise(kept), "excluded_clips": excl,
     "licences": {"ubody": "UBody (IDEA Research, OSX): research use only"}, "clips": kept}
Path(a.out).write_text(json.dumps(J, indent=1))
print(json.dumps(J["composition"], indent=1), "excluded", excl)
