"""Shared helpers for the ExAvatar baseline adapter (AvatarTest-v1).

GT conversion SMPL-X 2020 (neutral, 200 betas, flat-hand / absolute hand poses) -> ExAvatar's own SMPL-X layer
(SMPL-X v1.1 MALE, 100 betas, FLAME-2020 expression dirs on the face, hand poses relative to the v1.1 hand mean,
`smplx_gender = 'male'` is hard-coded in ExAvatar's config / fitting code and is kept).

* poses: global_orient / body_pose copied; hands: absolute - hands_mean (ExAvatar's layer is flat_hand_mean=False);
* betas100: least squares of the v1.1-male shape basis onto the GT rest shape (all 10,475 vertices, zero expression);
* joint_offset (ExAvatar ID parameter, 55x3): GT rest joints (2020 regressor on the GT rest shape) minus ExAvatar's rest
  joints (male regressor on template + betas100 [+ face_offset]) -> ExAvatar's kinematic chain = the GT one;
* trans: ExAvatar's `trans` is the SMPL-X layer transl -> transl_ex = transl_gt + pelvis_gt_rest - pelvis_ex_rest.
"""
import json
import os
import os.path as osp

import numpy as np

EXA_ROOT = os.environ.get('EXA_ROOT', '/dev/shm/ubx858_baselines/exavatar')
HMF = osp.join(EXA_ROOT, 'human_model_files')
SMPLX2020 = '/home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz'
N_BETAS_EX = 100


def key2idx(key):
    return int(key.split('_')[-1])


def load_clip(clip_dir):
    with open(osp.join(clip_dir, 'frames.json')) as f:
        fr = json.load(f)
    cams = np.load(osp.join(clip_dir, 'cameras.npz'))
    K = {str(k): cams['K'][i].astype(np.float64) for i, k in enumerate(cams['keys'])}
    s = np.load(osp.join(clip_dir, 'smplx2020.npz'))
    keys = [str(k) for k in s['keys']]
    gt = {'betas': s['betas'].astype(np.float64)}
    for k in ['global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose',
              'expression', 'transl']:
        gt[k] = {key: s[k][i].astype(np.float64) for i, key in enumerate(keys)}
    idx = [key2idx(k) for k in fr['keys']]
    assert len(set(idx)) == len(idx), 'frame keys do not map to unique integer indices'
    return fr, K, gt


class GTConverter:
    def __init__(self, betas2020):
        g = np.load(SMPLX2020, allow_pickle=True)
        m = np.load(osp.join(HMF, 'smplx', 'SMPLX_MALE.npz'), allow_pickle=True)
        nb = betas2020.shape[0]
        self.v_shaped_gt = g['v_template'] + np.einsum('vck,k->vc', g['shapedirs'][:, :, :nb], betas2020)
        self.J_gt = g['J_regressor'] @ self.v_shaped_gt  # (55, 3) rest joints of the GT body
        self.Tm = m['v_template'].astype(np.float64)
        self.Sm = m['shapedirs'][:, :, :N_BETAS_EX].astype(np.float64)
        self.Jm = m['J_regressor'].astype(np.float64)
        A = self.Sm.reshape(-1, N_BETAS_EX)
        b = (self.v_shaped_gt - self.Tm).reshape(-1)
        self.beta_ex, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.v_shaped_ex = self.Tm + np.einsum('vck,k->vc', self.Sm, self.beta_ex)
        self.hands_meanl = m['hands_meanl'].reshape(15, 3).astype(np.float64)
        self.hands_meanr = m['hands_meanr'].reshape(15, 3).astype(np.float64)
        v_err = np.linalg.norm(self.v_shaped_ex - self.v_shaped_gt, axis=1) * 1000
        j_err = np.linalg.norm(self.Jm @ self.v_shaped_ex - self.J_gt, axis=1) * 1000
        self.stats = {'rest_vertex_err_mm_mean': float(v_err.mean()), 'rest_vertex_err_mm_max': float(v_err.max()),
                      'rest_joint_err_mm_mean_before_joint_offset': float(j_err.mean()),
                      'rest_joint_err_mm_max_before_joint_offset': float(j_err.max())}

    def joint_offset(self, face_offset_full=None):
        """(55,3) ExAvatar joint_offset so that its rest joints equal the GT rest joints (root entry 0: masked by ExAvatar)."""
        v = self.v_shaped_ex if face_offset_full is None else self.v_shaped_ex + face_offset_full
        jo = self.J_gt - self.Jm @ v
        jo[0] = 0.0
        return jo

    def pelvis_ex(self, shape_param=None):
        v = self.v_shaped_ex if shape_param is None else self.Tm + np.einsum('vck,k->vc', self.Sm, shape_param)
        return self.Jm[0] @ v

    def frame_params(self, gt, key, pelvis_ex=None):
        """ExAvatar-convention body params of one frame (axis-angle). Face params are filled by ExAvatar's face fit."""
        if pelvis_ex is None:
            pelvis_ex = self.pelvis_ex()
        return {
            'root_pose': gt['global_orient'][key].reshape(3),
            'body_pose': gt['body_pose'][key].reshape(21, 3),
            'lhand_pose': gt['left_hand_pose'][key].reshape(15, 3) - self.hands_meanl,
            'rhand_pose': gt['right_hand_pose'][key].reshape(15, 3) - self.hands_meanr,
            'trans': gt['transl'][key].reshape(3) + self.J_gt[0] - pelvis_ex,  # SMPL-X layer transl
            'pelvis_cam': gt['transl'][key].reshape(3) + self.J_gt[0],  # fitting code's `trans` (root joint in camera)
        }


def cam_json(K):
    return {'R': np.eye(3).tolist(), 't': [0.0, 0.0, 0.0], 'focal': [float(K[0, 0]), float(K[1, 1])],
            'princpt': [float(K[0, 2]), float(K[1, 2])]}


def dump(obj, path):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    with open(path, 'w') as f:
        json.dump(conv(obj), f)
