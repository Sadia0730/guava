"""After ExAvatar's face fitting (recon + test stages) and its texture unwrap: assemble the avatar subject folder.

Run with cwd = ExAvatar fitting/tools (same import setup as ExAvatar's tools/smooth_smplx_params.py).

Per frame: body (root_pose, body_pose, lhand_pose, rhand_pose, trans) = GT conversion (NOT smoothed: it is the GT);
face (jaw_pose, leye_pose, reye_pose, expr) = ExAvatar's face fit, smoothed with ExAvatar's own smoother
(smooth_smplx_params.py: quaternion Savitzky-Golay, window 9, order 2 for poses; Savitzky-Golay on expr), applied to the
recon sequence and to the test sequence separately (no smoothing across the recon / test boundary).
ID: shape_param = GT least-squares betas100; face_offset = ExAvatar's face fit (recon frames); joint_offset = GT rest joints
minus ExAvatar rest joints with that face_offset (root 0); locator_offset = 0; face texture = ExAvatar's unwrap (recon).
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys

import numpy as np
import torch
from scipy.signal import savgol_filter
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.io import save_ply

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))
from utils.smpl_x import smpl_x  # noqa: E402  (ExAvatar fitting's SMPL-X layer, male v1.1)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from exa_common import EXA_ROOT, GTConverter, dump, key2idx, load_clip  # noqa: E402


# --- verbatim from ExAvatar fitting/tools/smooth_smplx_params.py ---
def fix_quaternions(quats):
    assert len(quats.shape) == 3
    assert quats.shape[-1] == 4
    result = quats.copy()
    dot_products = np.sum(quats[1:] * quats[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def smoothen_poses(poses, window_length):
    n_joints = poses.shape[1] // 3
    qs = matrix_to_quaternion(axis_angle_to_matrix(torch.FloatTensor(poses).view(-1, 3))).numpy()
    qs = qs.reshape((-1, n_joints, 4))
    qs = fix_quaternions(qs)
    qs_smooth = []
    for j in range(n_joints):
        qss = savgol_filter(qs[:, j], window_length=window_length, polyorder=2, axis=0)
        qs_smooth.append(qss[:, np.newaxis])
    qs_clean = np.concatenate(qs_smooth, axis=1)
    qs_clean = qs_clean / np.linalg.norm(qs_clean, axis=-1, keepdims=True)
    ps_clean = matrix_to_axis_angle(quaternion_to_matrix(torch.FloatTensor(qs_clean).view(-1, 4))).numpy()
    ps_clean = np.reshape(ps_clean, [-1, n_joints * 3])
    return ps_clean
# --- end verbatim ---


FACE_KEYS = ['jaw_pose', 'leye_pose', 'reye_pose', 'expr']


def face_params(fit_dir, keys, smooth_length=9):
    raw = {}
    for k in keys:
        with open(osp.join(fit_dir, 'smplx_params', '%d.json' % key2idx(k))) as f:
            raw[k] = {kk: np.array(v, dtype=np.float32) for kk, v in json.load(f).items()}
    out = {k: {} for k in keys}
    wl = smooth_length if len(keys) >= smooth_length else (len(keys) // 2) * 2 - 1
    for name in FACE_KEYS:
        arr = np.stack([raw[k][name].reshape(-1) for k in keys])
        if wl >= 3:
            arr = smoothen_poses(arr, wl) if 'pose' in name else savgol_filter(arr, window_length=wl, polyorder=2, axis=0)
        for i, k in enumerate(keys):
            out[k][name] = arr[i]
    return out, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--sid', required=True)
    args = ap.parse_args()
    work = osp.join(EXA_ROOT, 'work', args.sid)
    fr, K, gt = load_clip(args.clip_dir)
    conv = GTConverter(gt['betas'])
    rec_dir = osp.join(work, 'fit_recon', 'smplx_optimized')
    tst_dir = osp.join(work, 'fit_test', 'smplx_optimized')
    av = osp.join(work, 'avatar')
    so = osp.join(av, 'smplx_optimized')
    os.makedirs(osp.join(so, 'smplx_params_smoothed'), exist_ok=True)
    os.makedirs(osp.join(so, 'meshes_smoothed'), exist_ok=True)

    with open(osp.join(rec_dir, 'face_offset.json')) as f:
        face_offset = np.array(json.load(f), dtype=np.float64)  # (10475, 3), zero outside the face
    jo = conv.joint_offset(face_offset)
    dump(conv.beta_ex, osp.join(so, 'shape_param.json'))
    dump(face_offset, osp.join(so, 'face_offset.json'))
    dump(jo, osp.join(so, 'joint_offset.json'))
    dump(np.zeros((55, 3)), osp.join(so, 'locator_offset.json'))
    for name in ['face_texture.png', 'face_texture_mask.png']:
        shutil.copyfile(osp.join(rec_dir, name), osp.join(so, name))
    kd = osp.join(av, 'keypoints_whole_body')
    if osp.exists(kd):
        shutil.rmtree(kd)
    shutil.copytree(osp.join(work, 'fit_recon', 'keypoints_whole_body'), kd)

    stats = {}
    for split, fit_dir, out_dir in [('recon', rec_dir, osp.join(so, 'smplx_params_smoothed')),
                                    ('test', tst_dir, osp.join(work, 'test_params'))]:
        os.makedirs(out_dir, exist_ok=True)
        face, raw = face_params(fit_dir, fr[split])
        jaw = []
        for k in fr[split]:
            p = conv.frame_params(gt, k)
            prm = {'root_pose': p['root_pose'], 'body_pose': p['body_pose'], 'lhand_pose': p['lhand_pose'],
                   'rhand_pose': p['rhand_pose'], 'trans': p['trans'],
                   'jaw_pose': face[k]['jaw_pose'], 'leye_pose': face[k]['leye_pose'], 'reye_pose': face[k]['reye_pose'],
                   'expr': face[k]['expr']}
            dump(prm, osp.join(out_dir, '%d.json' % key2idx(k)))
            jaw.append(np.linalg.norm(face[k]['jaw_pose']))
            if split == 'recon':
                with torch.no_grad():
                    t = lambda x: torch.FloatTensor(np.asarray(x, dtype=np.float32)).view(1, -1)
                    out = smpl_x.layer(global_orient=t(prm['root_pose']), body_pose=t(prm['body_pose']),
                                       jaw_pose=t(prm['jaw_pose']), leye_pose=t(prm['leye_pose']), reye_pose=t(prm['reye_pose']),
                                       left_hand_pose=t(prm['lhand_pose']), right_hand_pose=t(prm['rhand_pose']),
                                       expression=t(prm['expr']), transl=t(prm['trans']), betas=t(conv.beta_ex),
                                       joint_offset=torch.FloatTensor(jo).view(-1, 3), face_offset=torch.FloatTensor(face_offset).view(-1, 3))
                save_ply(osp.join(so, 'meshes_smoothed', '%d_smplx.ply' % key2idx(k)), out.vertices[0], torch.LongTensor(smpl_x.face))
        stats[split] = {'frames': len(fr[split]), 'jaw_angle_mean_rad': float(np.mean(jaw)),
                        'expr_abs_mean': float(np.mean([np.abs(face[k]['expr']).mean() for k in fr[split]]))}
    stats['face_offset_mm_mean'] = float(np.linalg.norm(face_offset[np.abs(face_offset).sum(1) > 0], axis=1).mean() * 1000)
    stats['joint_offset_mm_mean'] = float(np.linalg.norm(jo[1:], axis=1).mean() * 1000)
    stats['joint_offset_mm_max'] = float(np.linalg.norm(jo[1:], axis=1).max() * 1000)
    stats['conversion'] = conv.stats
    dump(stats, osp.join(work, 'finalize_stats.json'))
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
