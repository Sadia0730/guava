"""AvatarTest-v1 input package -> GART dataset interface (phase3_dnt/baselines/CONTRACT.md).

Mirrors GART's lib_data/instant_avatar_people_snapshot.Dataset (the loader behind the people_* profiles): images cached
as float RGB in [0, 1] composited on WHITE with the mask, mask as float, one dict per frame with keys
rgb / mask / K / smpl_beta / smpl_pose (24, 3) / smpl_trans / idx, and (ret, meta_info) returned by __getitem__.
Differences (the only loader changes; GART's own files are not modified):
  1. file layout: E/base/<dataset>/<clip>/{frames.json, images/<key>.png, masks/<key>.png, cameras.npz, smpl.npz}
     instead of InstantAvatar's images/*.png, masks/*.npy, cameras.npz, poses/anim_nerf_<split>.npz;
  2. frame selection by key list (frames.json "recon" for training, "test" for rendering) instead of START_END slices;
  3. per-frame intrinsics K[key] (the crop follows the person) instead of one K per video; identity extrinsics as in GART;
  4. mask = StyleMatte alpha > 127 (binary, as the contract's training masks) instead of InstantAvatar's .npy masks;
  5. no resize: the crops are already 512 px (GART's IMAGE_ZOOM_RATIO 0.5 exists to bring PeopleSnapshot's 1080 px
     frames to 540 px; applying it here would train at 256 px instead of the registered 512 crops);
  6. neutral SMPL params from smpl.npz (one betas10 per clip, per-frame global_orient / body_pose (23, 3) / transl in the
     camera frame) instead of InstantAvatar's refined gendered-model poses.
"""
from __future__ import annotations

import json
import os.path as osp

import cv2
import numpy as np
from torch.utils.data import Dataset


def read_clip(clip_dir: str):
    fj = json.load(open(osp.join(clip_dir, "frames.json")))
    cam = np.load(osp.join(clip_dir, "cameras.npz"))
    smpl = np.load(osp.join(clip_dir, "smpl.npz"))
    ck = [str(k) for k in cam["keys"]]
    sk = [str(k) for k in smpl["keys"]]
    return fj, {k: cam["K"][i].astype(np.float32) for i, k in enumerate(ck)}, smpl, {k: i for i, k in enumerate(sk)}


class AvatarTestV1Dataset(Dataset):
    def __init__(self, clip_dir: str, split: str = "recon", load_images: bool = True) -> None:
        super().__init__()
        assert split in ("recon", "test")
        self.clip_dir = clip_dir
        self.fj, self.K, smpl, sidx = read_clip(clip_dir)
        self.keys = list(self.fj[split])
        self.video_name = self.fj["clip"]
        self.betas = smpl["betas"].astype(np.float32).reshape(10)
        go = smpl["global_orient"].astype(np.float32)
        bp = smpl["body_pose"].astype(np.float32).reshape(-1, 23, 3)
        tr = smpl["transl"].astype(np.float32)
        self.pose = [np.concatenate([go[sidx[k]][None], bp[sidx[k]]], 0) for k in self.keys]  # (24, 3) axis-angle
        self.trans = [tr[sidx[k]] for k in self.keys]
        self.img_buffer, self.msk_buffer = [], []
        if load_images:
            for k in self.keys:
                img = cv2.imread(osp.join(clip_dir, "images", f"{k}.png"))[..., ::-1]
                a = cv2.imread(osp.join(clip_dir, "masks", f"{k}.png"), cv2.IMREAD_GRAYSCALE)
                assert img.shape[:2] == (512, 512) and a.shape == (512, 512), (k, img.shape, a.shape)
                img = (img[..., :3] / 255).astype(np.float32)
                msk = (a > 127).astype(np.float32)
                img = img * msk[..., None] + (1 - msk[..., None])  # always white, as GART's loaders
                self.img_buffer.append(img)
                self.msk_buffer.append(msk)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        ret = {
            "rgb": self.img_buffer[idx].astype(np.float32) if self.img_buffer else None,
            "mask": self.msk_buffer[idx] if self.msk_buffer else None,
            "K": self.K[self.keys[idx]].copy(),
            "smpl_beta": self.betas.copy(),
            "smpl_pose": self.pose[idx].copy(),
            "smpl_trans": self.trans[idx].copy(),
            "idx": idx,
        }
        meta_info = {"video": self.video_name, "key": self.keys[idx], "viz_id": f"video{self.video_name}_{self.keys[idx]}"}
        return ret, meta_info
