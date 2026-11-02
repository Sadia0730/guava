"""ValBench-S (UBody only): validation clips whose frames SPRINT never trained on, for SPRINT-side interventions (PREREG phase 4,
H4 + amendment 1). All 20 ValBench-v1 UBody clips are in SPRINT's UBody training rows, so SPRINT changes cannot be judged on them.
Pool = clips of SPRINT's `ubody_labels` VAL split (2 % of clips held out when SPRINT's UBody rows were built), not c2t clips (never
used to train GUAVA-FB / a20), identity (ub_identity, OSX rule) not in ValBench-v1 / AvatarTest-v1; other clips of the same
identity may be in SPRINT's / GUAVA-FB's training (recorded per clip). Video: readable, 720p or 1080p, <= 600 frames at 30 fps;
AvatarTest-v1's subject-track / >= 60 GT-frame rules (build_avatar_test_split.ubody). Seeded order (20261102), at most 2 clips per
identity, up to 20 clips. Same frame protocol (first frame source, first half recon, latter half test). Validation only.
Usage: python build_valbench_s.py --out valbench_s.json"""
import argparse, collections, json, time
from pathlib import Path
import cv2
import numpy as np
import build_avatar_test_split as B

HERE = Path(__file__).resolve().parent
TMP = Path("/dev/shm/ubx858_eval/valbench_s_build"); TMP.mkdir(parents=True, exist_ok=True)
VID = Path("/raid/ubx858/datasets/expanded/UBody/videos")
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "valbench_s.json")); ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()

d = np.load("/raid/ubx858/outputs/dnt/data/ubody_labels/annotations.npz", allow_pickle=True)
clip = np.array([p.split("/")[3] for p in d["path"]]); split = d["split"]
vid = {c: p.split("/")[1:3] for c, p in zip(clip, d["path"])}                                     # scene, video folder
tr_rows = collections.Counter(B.ub_identity(c) for c in clip[split == "train"])
c2t = {c["clip"] for c in json.load(open("/raid/ubx858/outputs/dnt/c2t/split.json"))["clips"]}
c2t_ids = {B.ub_identity(c) for c in c2t}
eval_ids = {c["identity"] for f in ("valbench_v1.json", "avatar_test_v1.json") for c in json.load(open(HERE / f))["clips"] if c["dataset"] == "ubody"}
pool = sorted(c for c in set(clip[split == "val"]) if c not in c2t and B.ub_identity(c) not in eval_ids)
clips, rej = [], {}
for c in pool:
    v = VID / vid[c][0] / vid[c][1] / f"{c}.mp4"
    cap = cv2.VideoCapture(str(v)); n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
    if not (fps > 0 and h in (720, 1080) and n * 30 / fps <= 600):
        rej[c] = {"frames_30fps": round(n * 30 / fps) if fps > 0 else None, "wh": [w, h]}; continue
    clips.append({"scene": vid[c][0], "identity": B.ub_identity(c), "clip": c, "path": str(v), "frames": round(n * 30 / fps),
                  "fps": round(fps, 3), "wh": [w, h], "sha256": None})
print("pool", len(pool), "video rule kept", len(clips), "rejected", len(rej), flush=True)


def run(cl):
    (TMP / "c1_like.json").write_text(json.dumps({"clips": cl})); B.C1 = TMP / "c1_like.json"
    return B.ubody()


res = []
for scene in sorted({c["scene"] for c in clips}):                                                # one annotation load per scene
    sc = [c for c in clips if c["scene"] == scene]
    try:
        res += run(sc)
    except AssertionError as e:                                                                  # a clip whose annotation timeline disagrees
        print("scene batch failed:", e, "-> per clip", flush=True)
        for c in sc:
            try:
                res += run([c])
            except AssertionError as e2:
                rej[c["clip"]] = {"annotation_timeline": str(e2)}
order = np.random.default_rng(20261102).permutation(len(res))
kept, per_id = [], collections.Counter()
for i in order:
    c = res[int(i)]
    if c["n_frames"] < B.MIN_GT or per_id[c["identity"]] >= 2 or len(kept) >= a.n:
        continue
    c["checks"]["valbench_s"] = {"sprint_split": "val", "identity_sprint_train_rows": int(tr_rows[c["identity"]]),
                                 "identity_in_c2t": c["identity"] in c2t_ids}
    kept.append(c); per_id[c["identity"]] += 1
kept.sort(key=lambda c: c["id"])
excl = [{"id": c["id"], "gt_frames": c["n_frames"]} for c in res if c not in kept] + [{"id": f"ubody/{k}", **v} for k, v in rej.items()]
J = {"name": "ValBench-S", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "phase3_dnt/eval_split/build_valbench_s.py",
     "purpose": "validation-only UBody clips whose frames SPRINT never trained on (SPRINT-side interventions); never a test set",
     "protocol": "as avatar_test_v1.json", "composition": B.summarise(kept), "excluded_clips": excl,
     "licences": {"ubody": "UBody (IDEA Research, OSX): research use only"}, "clips": kept}
Path(a.out).write_text(json.dumps(J, indent=1))
print(json.dumps(J["composition"], indent=1), "kept", len(kept), "identities", len(per_id))
