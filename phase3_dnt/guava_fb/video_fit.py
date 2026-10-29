"""GUAVA-FB shared joint-video refiner + registered alignment filter: ONE import point for the real-video sources (UBC-Fashion,
Motion-X). Everything is fit_ubc.py's implementation, re-exported unchanged (coordinator: every source judged the same way):
- VideoFit: SPRINT from scratch (camera input focal = W), shape locked to the median, perspective crop, StyleMatte, joint
  per-clip Adam refinement (one shared shape, temporal smoothing) on the GPU; its .fit() returns per-frame frame_filter results.
- silhouettes(vimg, depth, faces): pytorch3d mesh silhouettes (512 px).
- frame_filter(sil, alpha, uv134, tgt, sc, box, Hm): registered filter, (a) mesh outside matte < 5 %; (b) >= 15 of 21 keypoints
  with score > 0.5 and max over 5 limbs of the limb's MEAN keypoint error < 3 % body height (also returns the single-keypoint
  max, max_single_kp_err_rel).
- flip_check(): the UBC front/back flip rule (ubc_flip_frames.py) for one frame.
NOTE (reported to the coordinator, not changed here to keep sources identical): the DWPose-134 mapping sends the DWPose heels
(rows 20 / 23) to SMPL-X joints 10 / 11 (ball of the foot); measured 7-8 % of body height off on well-aligned labels."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_ubc import (VideoFit, lean_batch, persp_crop, hom, proj, rot_to_z, DIAG, CKPT, WTS, CROP, CLIP_PASS,   # noqa: E402,F401
                     silhouettes, frame_filter, KP_ERR_MAX, KP_MIN)


def facing_of(Rg):
    """root rotation in the (virtual) camera frame -> facing (> 0: body faces the camera), as fit_ubc."""
    return float(-(np.asarray(Rg) @ np.array([0, 0, 1.0]))[2])


def flip_check(kp1024, sc134, facing):
    """ubc_flip_frames.py rule: None = not checked (shoulder score < 0.5, shoulder x-gap < 40 px or |facing| < 0.5);
    True = flipped (DWPose right shoulder (row 2) left of left shoulder (row 5) disagrees with facing > 0)."""
    if sc134[2] < 0.5 or sc134[5] < 0.5 or abs(kp1024[2, 0] - kp1024[5, 0]) < 40 or abs(facing) < 0.5:
        return None
    return bool((kp1024[2, 0] < kp1024[5, 0]) != (facing > 0))
