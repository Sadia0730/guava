"""ValBench-v1: development replica of the AvatarTest-v1 protocol on VALIDATION data only (PREREG "GUAVA-FB improvement phase 1").
UBody = a20's 10 validation clips (UBody SMPL-X annotations, AvatarTest-v1 subject-track / >= 60 GT-frame rules); BEDLAM2 =
GUAVA-FB's 10 validation clips (all full-body labelled rows). Same frame protocol (first frame source, first half recon, latter half
test). Reuses build_avatar_test_split.py (its C1 / BEDLAM2-proposal inputs are swapped for the validation lists).
Usage: python build_valbench.py --out valbench_v1.json"""
import argparse, json, time
from pathlib import Path
import cv2
import build_avatar_test_split as B

HERE = Path(__file__).resolve().parent
TMP = Path("/dev/shm/ubx858_eval/valbench_build"); TMP.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "valbench_v1.json")); a = ap.parse_args()

val = json.load(open(B.A20_INFO))["val_clips"]
c2t = {c["clip"]: c for c in json.load(open("/raid/ubx858/outputs/dnt/c2t/split.json"))["clips"]}
clips = []
for name in val:
    c = c2t[name]; cap = cv2.VideoCapture(c["path"])
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    wh = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]; cap.release()
    clips.append({"scene": c["scene"], "identity": B.ub_identity(name), "clip": name, "path": c["path"], "frames": round(n * 30 / fps),  # GT timeline = ffmpeg -r 30
                  "fps": round(fps, 3), "wh": wh, "sha256": None})
(TMP / "c1_like.json").write_text(json.dumps({"clips": clips})); B.C1 = TMP / "c1_like.json"
plan = json.load(open(B.cb.REP / "plan.json"))
b2 = [[c["job"], c["seq"], c["subject"], c["n_full"]] for c in plan["clips"] if c["split"] == "val"]
# BEDLAM2 extension (PREREG 11:05): subjects 11-30 of convert_bedlam2.plan's val rule
R = json.load(open(B.cb.AB / "bedlam_census.json")); sp = json.load(open(B.cb.AB / "bedlam_split_proposal.json"))
S = [r for r in R if not r["multi"] and r["n_full"] >= 30 and r["body_h_px_full_med"] >= 350]
for r in S: r["subject"] = "moyo" if r["body"].startswith("moyo") else r["subject"]
test_sub = {c[2] for c in sp["test"]["clips"]}
pool_sub = {r["subject"] for r in S if r["subject"] not in test_sub and r["job"] not in B.cb.VAL_JOBS}
vc = [r for r in S if r["job"] in B.cb.VAL_JOBS and r["subject"] not in test_sub and r["subject"] not in pool_sub and r["subject"] != "moyo"]
subs = sorted({r["subject"] for r in vc}, key=B.cb.h)
assert [x[2] for x in sorted(b2, key=lambda x: B.cb.h(x[2]))] == subs[:10], "val rule not reproduced"
for sname in subs[10:30]:
    c = sorted([r for r in vc if r["subject"] == sname], key=lambda r: (r["job"] != "20240808_1_250_ai1105_vcam", -r["n_full"], r["job"], r["seq"]))[0]
    b2.append([c["job"], c["seq"], sname, c["n_full"]])
(TMP / "b2_like.json").write_text(json.dumps({"test": {"clips": b2}})); B.B2_PROP = TMP / "b2_like.json"

allc = B.ubody() + B.bedlam2()
kept = [c for c in allc if c["n_frames"] >= B.MIN_GT or c["dataset"] == "bedlam2"]
# UBody extension (PREREG 10:55): training-split videos never tracked for GUAVA-FB / a20, identity-disjoint, seeded round-robin
import numpy as np
banned = {B.ub_identity(n) for n in c2t} | {str(x) for f in ("intra_scene_test_list.npy", "inter_scene_test_list.npy") for x in np.load(B.UB_SPLITS / f, allow_pickle=True)}
vids = sorted(Path("/raid/ubx858/datasets/expanded/UBody/videos").rglob("*.mp4"))
by_scene = {}
for v in vids:
    if B.ub_identity(v.stem) not in banned: by_scene.setdefault(v.relative_to("/raid/ubx858/datasets/expanded/UBody/videos").parts[0], []).append(v)
rng = np.random.default_rng(20261101)
for k in by_scene: by_scene[k] = [by_scene[k][i] for i in rng.permutation(len(by_scene[k]))]
order, seen = [], set()
while any(by_scene.values()):
    for k in sorted(by_scene):
        if by_scene[k]:
            v = by_scene[k].pop(0)
            if B.ub_identity(v.stem) not in seen: order.append((k, v))
ext, used_ids, pos = [], set(), 0
while len(ext) < 15 and pos < len(order):
    batch = []
    while len(batch) < 24 and pos < len(order):
        k, v = order[pos]; pos += 1
        if B.ub_identity(v.stem) in used_ids: continue
        cap = cv2.VideoCapture(str(v)); n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
        if not (fps > 0 and 150 <= n * 30 / fps <= 450 and h == 720): continue
        batch.append({"scene": k, "identity": B.ub_identity(v.stem), "clip": v.stem, "path": str(v), "frames": round(n * 30 / fps),
                      "fps": round(fps, 3), "wh": [w, h], "sha256": None}); used_ids.add(B.ub_identity(v.stem))
    (TMP / "c1_ext.json").write_text(json.dumps({"clips": batch})); B.C1 = TMP / "c1_ext.json"
    try:
        res = B.ubody()
    except AssertionError as e:                                                           # a clip whose annotation timeline disagrees
        print("batch skipped:", e); continue
    for c in res:
        if c["n_frames"] >= B.MIN_GT and len(ext) < 15: c["checks"]["valbench_extension"] = True; ext.append(c)
    print("extension so far", len(ext), "after", pos, "candidates", flush=True)
kept = kept[:5] + ext + [c for c in kept if c["dataset"] == "bedlam2"]
excl = [{"id": c["id"], "gt_frames": c["n_frames"]} for c in allc if c not in kept]
J = {"name": "ValBench-v1", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "phase3_dnt/eval_split/build_valbench.py",
     "purpose": "validation-only replica of the AvatarTest-v1 protocol for development / selection (never a test set)",
     "protocol": "as avatar_test_v1.json", "composition": B.summarise(kept), "excluded_clips": excl,
     "licences": {"ubody": "UBody (IDEA Research, OSX): research use only", "bedlam2": B.cb.LICENCE}, "clips": kept}
Path(a.out).write_text(json.dumps(J, indent=1))
print(json.dumps(J["composition"], indent=1), "excluded", excl)
