"""Phase 2: build the UBody "intra-scene" test subset as in OSX (data/UBody/UBody.py, test mode).

OSX test selection, per scene (one COCO db per scene; `i` counts annotations in file order):
  every `test_sample_interval` (=100, OSX main/config.py) annotation; video in splits/intra_scene_test_list.npy;
  skip crowd / num_keypoints == 0 / valid_label == 0 / no SMPL-X annotation; OSX process_bbox must succeed.
Video matching exactly as OSX: video_name = file_name.split('/')[-2], with '_Trim...' stripped, must be in the
test list. (Clips stored as '..._sceneNNN' therefore never match, in OSX and here; e.g. LiveVlog's
'LiveVlog_S1/LiveVlog_S1_scene001/...' frames are excluded, as OSX's loader excludes them.)
Frames: OSX extracts PNGs with `ffmpeg -i <clip>.mp4 -f image2 -r 30` (tool/UBody/video2image.py), so frame k is on
a 30 fps timeline even for 25 fps clips. The same ffmpeg command is run per needed clip into a temporary folder and
only the selected frames are kept (JPEG q95 is NOT used: PNG kept, as in OSX).
Output: <out>/ubody_intra_test.json (selected annotations + SMPL-X params) and <out>/images/<scene>/<file_name>.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

SCENES = ["ConductMusic", "Entertainment", "Fitness", "Interview", "LiveVlog", "Magic_show", "Movie", "Olympic",
          "Online_class", "SignLanguage", "Singing", "Speech", "TVShow", "TalkShow", "VideoConference"]


def osx_process_bbox(bbox, img_w, img_h, aspect=192 / 256):
    """OSX common/utils/preprocessing.py process_bbox (input_img_shape (256, 192)): sanitize, aspect, x1.25."""
    x, y, w, h = bbox
    x1, y1 = np.max((0, x)), np.max((0, y))
    x2, y2 = np.min((img_w - 1, x1 + np.max((0, w - 1)))), np.min((img_h - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    else:
        return None
    w, h = bbox[2], bbox[3]
    c_x, c_y = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    bbox = np.array([c_x - w * 1.25 / 2.0, c_y - h * 1.25 / 2.0, w * 1.25, h * 1.25], dtype=np.float32)
    return bbox


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations-zip", type=Path, default=Path("/raid/ubx858/datasets/raw/UBody/annotations.zip"))
    p.add_argument("--videos", type=Path, default=Path("/raid/ubx858/datasets/expanded/UBody/videos"))
    p.add_argument("--test-list", type=Path, default=Path("/raid/ubx858/datasets/expanded/UBody/splits/intra_scene_test_list.npy"))
    p.add_argument("--interval", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("/raid/ubx858/datasets/eval_suite/ubody"))
    p.add_argument("--ffmpeg", default="/home/ubx858/miniconda3/envs/guava/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
    p.add_argument("--skip-frames", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    test_videos = set(np.load(args.test_list, allow_pickle=True).tolist())
    z = zipfile.ZipFile(args.annotations_zip)
    items, stats = [], {}
    for scene in SCENES:
        kp = json.load(z.open(f"annotations/{scene}/keypoint_annotation.json"))
        sx = json.load(z.open(f"annotations/{scene}/smplx_annotation.json"))
        imgs = {im["id"]: im for im in kp["images"]}
        n_sel = n_kept = 0
        for i, ann in enumerate(kp["annotations"], start=1):
            if i % args.interval != 0:
                continue
            n_sel += 1
            img = imgs[ann["image_id"]]
            fn = img["file_name"].lstrip("/")
            video = fn.split("/")[-2]
            if "Trim" in video:
                video = video.split("_Trim")[0]
            if video not in test_videos:
                continue
            if ann["iscrowd"] or ann["num_keypoints"] == 0 or ann["valid_label"] == 0 or str(ann["id"]) not in sx:
                continue
            if osx_process_bbox(ann["bbox"], img["width"], img["height"]) is None:
                continue
            n_kept += 1
            items.append({"scene": scene, "file_name": fn, "aid": ann["id"], "img_wh": [img["width"], img["height"]],
                          "bbox": ann["bbox"], "lefthand_valid": bool(ann["lefthand_valid"]),
                          "righthand_valid": bool(ann["righthand_valid"]), "face_valid": bool(ann["face_valid"]),
                          "smplx_param": sx[str(ann["id"])]["smplx_param"], "cam_param": sx[str(ann["id"])]["cam_param"]})
        stats[scene] = {"annotations": len(kp["annotations"]), "every_100th": n_sel, "test_kept": n_kept}
        print(scene, stats[scene], flush=True)
        del kp, sx
    (args.out / "ubody_intra_test.json").write_text(json.dumps({"interval": args.interval, "stats": stats, "items": items}) + "\n")
    print("total test items", len(items), flush=True)
    if args.skip_frames:
        return
    by_clip = defaultdict(set)
    for it in items:
        fn = Path(it["file_name"])
        by_clip[(it["scene"], str(fn.parent))].add(int(fn.stem))
    for n, ((scene, clip_dir), frames) in enumerate(sorted(by_clip.items())):
        dst = args.out / "images" / scene / clip_dir
        if all((dst / f"{k:06d}.png").exists() for k in frames):
            continue
        src = args.videos / scene / f"{clip_dir}.mp4"
        with tempfile.TemporaryDirectory(dir=args.out) as tmp:
            subprocess.run([args.ffmpeg, "-loglevel", "error", "-i", str(src), "-f", "image2", "-r", "30", "-b:v", "5626k",
                            f"{tmp}/%06d.png"], check=True)
            dst.mkdir(parents=True, exist_ok=True)
            for k in frames:
                shutil.copy2(f"{tmp}/{k:06d}.png", dst / f"{k:06d}.png")
        if n % 20 == 0:
            print(f"clip {n}/{len(by_clip)}", flush=True)
    print("frames done", flush=True)


if __name__ == "__main__":
    main()
