"""AvatarTest-v1 input package (E/base/<dataset>/<clip>/) -> GaussianAvatar data folder.

GaussianAvatar's loader (scene/dataset_mono.py) expects
  <data>/{train,test}/images/<name>.png, masks/<name>.png, cam_parms/<name>.npz (per-frame camera, --cam_static 0),
  smpl_parms.pth = {"beta": (1,10), "body_pose": (N,72) [global_orient | body_pose], "trans": (N,3)}
with frames indexed by the sorted image-name order (same layout as its scripts/sample_romp2gsavatar.py writes).

train = the clip's recon keys ONLY, test = its test keys. Names = the package keys (frame_%06d, so sorted = time order).
Images are composited on black with the training mask (alpha > 127) here, and the loader is run with --no_mask 1, so
the official loader reads them unchanged (its own mask path composites on white, see NOTES.md).
Per-frame camera: extrinsic = identity (world = camera, OpenCV axes), intrinsic = K[key] of the 512 crop.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()

    fj = json.load(open(os.path.join(a.clip_dir, "frames.json")))
    cam = np.load(os.path.join(a.clip_dir, "cameras.npz"))
    smpl = np.load(os.path.join(a.clip_dir, "smpl.npz"))
    ck = [str(k) for k in cam["keys"]]
    sk = [str(k) for k in smpl["keys"]]
    for split, keys in (("train", fj["recon"]), ("test", fj["test"])):
        if split not in a.splits.split(","):
            continue
        keys = list(keys)
        assert keys == sorted(keys), "keys must sort in time order"
        assert not (set(fj["recon"]) & set(fj["test"]))
        d = os.path.join(a.out, split)
        for sub in ("images", "masks", "cam_parms"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        for k in keys:
            img = np.array(Image.open(os.path.join(a.clip_dir, "images", f"{k}.png")).convert("RGB"))
            m = np.array(Image.open(os.path.join(a.clip_dir, "masks", f"{k}.png")))
            if m.ndim == 3:
                m = m[..., 0]
            assert img.shape == (512, 512, 3) and m.shape == (512, 512), (k, img.shape, m.shape)
            mb = (m > 127).astype(np.uint8)
            Image.fromarray(img * mb[..., None]).save(os.path.join(d, "images", f"{k}.png"))
            shutil.copyfile(os.path.join(a.clip_dir, "masks", f"{k}.png"), os.path.join(d, "masks", f"{k}.png"))
            np.savez(os.path.join(d, "cam_parms", f"{k}.npz"), extrinsic=np.eye(4, dtype=np.float32),
                     intrinsic=cam["K"][ck.index(k)].astype(np.float32))
        idx = [sk.index(k) for k in keys]
        pose = np.concatenate([smpl["global_orient"][idx].reshape(-1, 3), smpl["body_pose"][idx].reshape(-1, 69)], 1)
        torch.save({"beta": torch.from_numpy(smpl["betas"].reshape(1, 10).astype(np.float32)),
                    "body_pose": torch.from_numpy(pose.astype(np.float32)),
                    "trans": torch.from_numpy(smpl["transl"][idx].astype(np.float32))},
                   os.path.join(d, "smpl_parms.pth"))
        json.dump(keys, open(os.path.join(d, "keys.json"), "w"))
        print(split, len(keys), "frames ->", d, flush=True)


if __name__ == "__main__":
    main()
