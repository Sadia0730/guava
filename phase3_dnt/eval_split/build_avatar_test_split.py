"""AvatarTest-v1: our own held-out avatar test split (GUAVA's 58-ID test set is not released). Every frame has ground-truth
SMPL-X, so models can be driven by GT parameters; the same clips / frames are used for every model (GUAVA, GUAVA-FB, ExAvatar,
GaussianAvatar, GART, ...) and for other projects. CPU only, no model is run.

Protocol (GUAVA Table 2): per clip the ordered GT frame list is split in halves; the first frame is the source of single-image
methods, the first half is the only footage per-subject methods may train on, every method is scored on the latter half.

Parts (each a split fixed earlier and never used for training or checkpoint selection of GUAVA-FB / SPRINT / a20):
  ubody    upper body, real video. The C1 split (/raid/ubx858/outputs/dnt/c1/split.json: 36 identities from UBody's official
           intra-scene test list, one clip each). GT = UBody's released SMPL-X annotations (pseudo-GT; valid_label != 0, as
           OSX). The subject is chosen without any model: annotated boxes are linked into person tracks (greedy, larger boxes
           first, IoU >= 0.3 to the track's last box, gaps <= 30 frames); subject = the track with the most valid GT frames
           (ties: larger median box). Frames where the subject has no valid annotation are dropped; a clip is kept iff >= 60
           GT frames remain (>= 30 test frames).
  bedlam2  full body, synthetic video, exact GT SMPL-X. The 40-clip identity-disjoint test split of
           /raid/ubx858/outputs/dnt/avatar_B/bedlam_split_proposal.json (one clip per subject, SPRINT val jobs; GUAVA-FB's
           training pool excludes these subjects and their outfits). Frames = every labelled (6 fps) row of the clip in
           which the whole body is in view (convert_bedlam2.full_body_mask, the training data's rule), without a cap.

Usage: python build_avatar_test_split.py --out avatar_test_v1.json   (also writes <out>.md, the readable list)"""
from __future__ import annotations

import argparse
import collections as C
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "guava_fb"))
import convert_bedlam2 as cb                                                                       # noqa: E402  (full_body_mask, camera_motion)

C1 = Path("/raid/ubx858/outputs/dnt/c1/split.json")
UB_ZIP = Path("/raid/ubx858/datasets/raw/UBody/annotations.zip")
UB_SPLITS = Path("/raid/ubx858/datasets/expanded/UBody/splits")
A20_INFO = Path("/raid/ubx858/outputs/dnt/guava_a20/run/run_info.json")
B2_PROP = Path("/raid/ubx858/outputs/dnt/avatar_B/bedlam_split_proposal.json")
B2_CENSUS = Path("/raid/ubx858/outputs/dnt/avatar_B/bedlam_census.json")
FB_B2_INDEX = cb.O / "data" / "bedlam2" / "index.json"
IOU_MIN = 0.3
GAP = 30            # a person track survives up to 30 unannotated frames (1 s)
MIN_GT = 60         # a clip is kept iff it has >= 60 GT frames (>= 30 test frames); fixed from GT coverage only


def ub_identity(name):
    return re.split(r"_Trim|_scene", name)[0]                                                    # OSX rule (+ _sceneNNN)


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w, h = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - w * h
    return w * h / u if u > 0 else 0.0


def halves(clip):
    n = len(clip["frames"]); k = n // 2
    if k == 0:
        clip.update(n_frames=n, n_recon=0, test_frames=[])
        return clip
    clip.update(n_frames=n, source_frame=clip["frames"][0], n_recon=k, recon_frames=[clip["frames"][0], clip["frames"][k - 1]],
                test_frames=clip["frames"][k:])
    return clip


def ubody():
    s = json.load(open(C1))
    test_ids = {str(x) for f in ("intra_scene_test_list.npy", "inter_scene_test_list.npy") for x in np.load(UB_SPLITS / f, allow_pickle=True)}
    a20 = json.load(open(A20_INFO)); a20_ids = {ub_identity(c) for c in a20["train_clips"] + a20["val_clips"]}
    z = zipfile.ZipFile(UB_ZIP)
    by_scene = C.defaultdict(list)
    for c in s["clips"]:
        by_scene[c["scene"]].append(c)
    out = []
    for scene in sorted(by_scene):
        t0 = time.time()
        kp = json.load(z.open(f"annotations/{scene}/keypoint_annotation.json"))
        sx = json.load(z.open(f"annotations/{scene}/smplx_annotation.json"))
        want = {c["clip"]: c for c in by_scene[scene]}
        imgs = C.defaultdict(dict)                                                             # clip -> frame -> image id
        for im in kp["images"]:
            cl = im["file_name"].split("/")[-2]
            if cl in want:
                imgs[cl][int(Path(im["file_name"]).stem) - 1] = im["id"]
        img2clip = {iid: (cl, f) for cl, d in imgs.items() for f, iid in d.items()}
        anns = C.defaultdict(list)                                                             # (clip, frame) -> person annotations
        for a in kp["annotations"]:
            if a["image_id"] in img2clip and not a.get("iscrowd", 0):
                anns[img2clip[a["image_id"]]].append(a)
        del kp
        ok = lambda a: a.get("valid_label", 1) != 0 and str(a["id"]) in sx                     # GT = valid SMPL-X annotation (OSX rule)
        for cl in sorted(want):
            c = want[cl]; fr = sorted(imgs[cl])
            assert fr == list(range(len(fr))) and abs(len(fr) - c["frames"]) <= 3, (cl, len(fr), c["frames"])   # ffmpeg -r 30 vs cv2 count
            tracks = []                                                                        # greedy IoU linking, larger boxes first
            for f in fr:
                used = set()
                for a in sorted(anns.get((cl, f), []), key=lambda a: -a["bbox"][2] * a["bbox"][3]):
                    best, bv = None, IOU_MIN
                    for i, t in enumerate(tracks):
                        if i not in used and f - t["last_f"] <= GAP and iou(a["bbox"], t["last"]) >= bv:
                            best, bv = i, iou(a["bbox"], t["last"])
                    if best is None:
                        tracks.append({"items": {}}); best = len(tracks) - 1
                    tracks[best].update(last=a["bbox"], last_f=f); tracks[best]["items"][f] = a; used.add(best)
            score = lambda t: (sum(ok(a) for a in t["items"].values()), float(np.median([a["bbox"][2] * a["bbox"][3] for a in t["items"].values()])))
            subj = max(tracks, key=score) if tracks else {"items": {}}
            frames, ids, boxes, drop = [], [], [], C.Counter()
            for f in fr:
                a = subj["items"].get(f)
                if a is None or not ok(a):
                    drop["subject not annotated" if a is None else "subject annotation invalid (valid_label 0 or no SMPL-X)"] += 1
                    continue
                frames.append(f); ids.append(a["id"]); boxes.append([round(v, 1) for v in a["bbox"]])
            people = max((len(v) for (k, _), v in anns.items() if k == cl), default=0)
            ident = ub_identity(cl)
            out.append(halves({
                "id": f"ubody/{cl}", "dataset": "ubody", "part": "upper body, real video", "scene": scene, "identity": ident,
                "clip": cl, "video": c["path"], "video_sha256": c["sha256"], "fps": c["fps"], "wh": c["wh"], "video_frames": len(fr), "video_frames_cv2": c["frames"],
                "frame_index": "0-based on the 30 fps timeline (UBody PNG 000001 = frame 0; ffmpeg -r 30 as OSX)",
                "gt": {"kind": "pseudo-GT SMPL-X (UBody release)", "file": f"{UB_ZIP}:annotations/{scene}/smplx_annotation.json",
                       "key": "str(ann_id) -> smplx_param (root / body / hands / jaw / shape / expr / trans) + cam_param (focal, princpt)"},
                "frames": frames, "ann_ids": ids, "bbox_xywh": boxes, "dropped": dict(drop), "max_people_per_frame": people,
                "person_tracks": len(tracks),
                "checks": {"in_official_test_list": ident in test_ids, "identity_in_a20_or_guava_fb_ubody_clips": ident in a20_ids}}))
        del sx
        print(f"ubody {scene}: {len(want)} clips, {time.time() - t0:.0f} s", flush=True)
    return out


def bedlam2():
    sp = json.load(open(B2_PROP)); census = json.load(open(B2_CENSUS))
    test = sp["test"]["clips"]
    fb = json.load(open(FB_B2_INDEX)); fb_sub = {c["subject"] for c in fb["clips"]}; fb_clips = {c["name"] for c in fb["clips"]}
    other_job_sub = {("moyo" if r["body"].startswith("moyo") else r["subject"]) for r in census if r["job"] not in cb.VAL_JOBS}
    by_job = C.defaultdict(list)
    for job, seq, subj, n in test:
        by_job[job].append((seq, subj, n))
    out = []
    for job in sorted(by_job):
        d = np.load(cb.PROC / f"{job}.npz", allow_pickle=True)
        full = cb.full_body_mask(d); cams = cb.camera_motion(job, [x[0] for x in by_job[job]])
        cen = {r["seq"]: r for r in census if r["job"] == job}
        for seq, subj, n in by_job[job]:
            sel = np.flatnonzero((d["seq"] == seq) & full)
            sel = sel[np.argsort(d["frame"][sel], kind="stable")]
            assert len(sel) == n, (job, seq, len(sel), n)
            r = cen[seq]
            out.append(halves({
                "id": f"bedlam2/{cb.clip_name(job, seq)}", "dataset": "bedlam2", "part": "full body, synthetic video", "job": job,
                "seq": seq, "identity": subj, "clip": cb.clip_name(job, seq), "video": f"{cb.RAW}/{job}_mp4.tar:{seq}.mp4",
                "fps": 30, "wh": [int(r["img_w"]), int(r["img_h"])], "video_frames": int(r["video_frames"]),
                "frame_index": "0-based video frame (labels every 5th frame, 6 fps)",
                "gt": {"kind": "exact GT SMPL-X (BEDLAM2 render labels, locked-head model; pose_cam / shape16 / t_cam, pinhole K)",
                       "file": f"{cb.PROC}/{job}.npz", "key": "row index (rows below, same order as frames)"},
                "frames": [int(f) for f in d["frame"][sel]], "rows": [int(i) for i in sel],
                "camera": cams[seq], "outfit": r["texture_clothing"].split("_texture")[0] if r["texture_clothing"] else "",
                "body_h_px_full_med": round(float(r["body_h_px_full_med"]), 1),
                "checks": {"subject_in_guava_fb_bedlam2": subj in fb_sub, "clip_in_guava_fb_bedlam2": cb.clip_name(job, seq) in fb_clips,
                           "subject_in_sprint_training_jobs": subj in other_job_sub}}))
    return out


def summarise(clips):
    rows = []
    for ds in ("ubody", "bedlam2"):
        cs = [c for c in clips if c["dataset"] == ds]
        rows.append({"dataset": ds, "clips": len(cs), "identities": len({c["identity"] for c in cs}),
                     "frames": sum(c["n_frames"] for c in cs), "test_frames": sum(len(c["test_frames"]) for c in cs),
                     "recon_frames": sum(c["n_recon"] for c in cs)})
    tc, tt = sum(r["clips"] for r in rows), sum(r["test_frames"] for r in rows)
    for r in rows:
        r["share_of_clips_pct"] = round(100 * r["clips"] / tc, 1); r["share_of_test_frames_pct"] = round(100 * r["test_frames"] / tt, 1)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "avatar_test_v1.json"))
    ap.add_argument("--stats-only", action="store_true"); a = ap.parse_args()
    clips = ubody() + bedlam2()
    if a.stats_only:
        print(json.dumps([[c["id"], c["video_frames"], c["n_frames"], c.get("dropped", {})] for c in clips])); return
    excluded = [{"id": c["id"], "identity": c["identity"], "gt_frames": c["n_frames"], "video_frames": c["video_frames"],
                 "dropped": c.get("dropped", {}), "reason": f"fewer than {MIN_GT} GT frames"}
                for c in clips if c["dataset"] == "ubody" and c["n_frames"] < MIN_GT]
    clips = [c for c in clips if not (c["dataset"] == "ubody" and c["n_frames"] < MIN_GT)]
    comp = summarise(clips)
    pool = {"ubody": {"pool": "UBody official intra-scene test list", "pool_identities": 135,
                      "selection": "C1 split (seed 20261015, round-robin over scenes; clips 150-450 frames at 30 fps, 720 px high; one clip per "
                                   f"identity), minus clips with fewer than {MIN_GT} GT frames of the subject"},
            "bedlam2": {"pool": "BEDLAM2 SPRINT / PEAR-student val jobs (20240731_1_1827_hdri static HDRI, 20240808_1_250_ai1105_vcam moving camera)",
                        "pool_identities": None,
                        "selection": "bedlam_split_proposal.json test: subjects by SHA-1 hash, one clip per subject (most full-body rows), MOYO excluded"}}
    for r in comp:
        r.update(pool[r["dataset"]])
        if r["dataset"] == "ubody":
            r["selected_of_pool_pct"] = round(100 * r["identities"] / 135, 1); r["c1_clips_excluded"] = len(excluded)
    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    J = {"name": "AvatarTest-v1", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "builder": "phase3_dnt/eval_split/build_avatar_test_split.py",
         "git_head_at_build": sha,
         "why": "GUAVA's 58-ID test set is not released; every model (GUAVA, GUAVA-FB, ExAvatar, GaussianAvatar, GART, ...) is run on "
                "these clips and frames instead of copying numbers from a different split.",
         "protocol": {
             "frames": "per clip, the ordered list 'frames' (GT available on every one); n_recon = floor(n / 2)",
             "source": "source_frame = frames[0]; single-image methods (GUAVA, GUAVA-FB) see only this frame",
             "recon": "frames[:n_recon] (recon_frames = first / last); per-subject methods (ExAvatar, GaussianAvatar, GART) may train on "
                      "these frames only (GUAVA Table 2: 'first half of each video for their training')",
             "test": "test_frames = frames[n_recon:]; every method is scored on exactly these frames",
             "driving": "GT rows: the GT SMPL-X of each test frame (+ GT camera); estimator rows (e.g. SPRINT): the estimator's "
                        "per-frame output on the same frames; the source avatar of an estimator row is built from the estimator's "
                        "fit of the source frame",
             "metrics": "GUAVA main/metrics.py: PSNR, L1 (MAE), SSIM, LPIPS-VGG on the 512 x 512 person crop; GT = image x person "
                        "mask on black; mean per clip, then frame-weighted over clips; per dataset and overall",
             "mask": "StyleMatte (EHM-Tracker pretrained/matting/stylematte_synth.pt) on the 1024 crop, as GUAVA's data",
             "crop": "GUAVA-FB perspective body crop (FORMAT.md): square, 1.1 x the long side of the GT-mesh box, 1024 px",
             "recon_time": "avatar creation from the source (ms, CUDA-synchronised), plus the pose fit of the source frame for "
                           "estimator rows; per-subject methods: their training time on the recon frames"},
         "composition": comp,
         "excluded_clips": excluded,
         "not_used": [
             {"dataset": "4D-DRESS", "reason": "all 32 subjects are in GUAVA-FB training (28) or validation / checkpoint selection (4)"},
             {"dataset": "Motion-X", "reason": "dataset SMPL-X labels are poorly aligned (27 of 1,500 audited clips pass the "
                                               "alignment filter); the usable labels are SPRINT refits, which would favour the SPRINT row"},
             {"dataset": "UBC-Fashion", "reason": "no ground-truth SMPL-X (only fits)"},
             {"dataset": "ARCTIC", "reason": "SPRINT trained on 9 of 10 subjects and selects on s10; hand-object scenes"},
             {"dataset": "MOYO", "reason": "single performer, also a BEDLAM2 training body; val takes used for SPRINT selection"},
             {"dataset": "BEDLAM2 validation clips", "reason": "used for GUAVA-FB checkpoint selection"},
             {"dataset": "a20 / GUAVA-FB UBody validation clips", "reason": "used for checkpoint selection"}],
         "caveats": [
             "UBody GT is pseudo-GT (UBody's own SMPL-X fits), not motion capture.",
             "Released GUAVA was trained on undisclosed UBody clips: the UBody part is held out for our models, not provably for GUAVA.",
             "BEDLAM2 test subjects never appear in GUAVA-FB data; SPRINT trained on other BEDLAM2 jobs (see per-clip "
             "checks.subject_in_sprint_training_jobs) and selected checkpoints on the val jobs these clips come from.",
             "No real full-body footage with true GT SMPL-X is included (candidate for v2: X-Humans, needs registration and a download)."],
         "licences": {"ubody": "UBody (IDEA Research, OSX): research use only", "bedlam2": cb.LICENCE},
         "clips": clips}
    Path(a.out).write_text(json.dumps(J, indent=1))
    md = [f"# AvatarTest-v1 (built {J['created']})", "", "| dataset | part | clips | identities | GT frames | recon frames | test frames | "
          "share of clips | share of test frames | pool |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp:
        part = next(c["part"] for c in clips if c["dataset"] == r["dataset"])
        md.append(f"| {r['dataset']} | {part} | {r['clips']} | {r['identities']} | {r['frames']:,} | {r['recon_frames']:,} | "
                  f"{r['test_frames']:,} | {r['share_of_clips_pct']} % | {r['share_of_test_frames_pct']} % | {r['pool']} |")
    md += ["", "| clip | dataset | identity | GT frames | source frame | recon frames | test frames | dropped |", "|---|---|---|---|---|---|---|---|"]
    for c in clips:
        md.append(f"| {c['clip']} | {c['dataset']} | {c['identity']} | {c['n_frames']} | {c['source_frame']} | {c['recon_frames'][0]}-"
                  f"{c['recon_frames'][1]} ({c['n_recon']}) | {c['test_frames'][0]}-{c['test_frames'][-1]} ({len(c['test_frames'])}) | "
                  f"{sum(c.get('dropped', {}).values())} |")
    md += ["", f"Excluded UBody clips (fewer than {MIN_GT} GT frames): " + ", ".join(f"{e['id'].split('/')[1]} ({e['gt_frames']})" for e in excluded)]
    Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(comp, indent=1))
    bad = [c["id"] for c in clips if c["dataset"] == "ubody" and (not c["checks"]["in_official_test_list"] or c["checks"]["identity_in_a20_or_guava_fb_ubody_clips"])]
    bad += [c["id"] for c in clips if c["dataset"] == "bedlam2" and (c["checks"]["subject_in_guava_fb_bedlam2"] or c["checks"]["clip_in_guava_fb_bedlam2"])]
    print("held-out check failures:", bad)


if __name__ == "__main__":
    main()
