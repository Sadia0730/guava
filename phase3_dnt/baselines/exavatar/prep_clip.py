"""Step 0 of the ExAvatar adapter: turn one AvatarTest-v1 clip package into ExAvatar 'Custom' subject folders.

Creates (all under /dev/shm):
  work/<sid>/fit_recon  (= fitting/data/Custom/data/<sid>_recon): recon frames only, raw 512 crops (background kept, as
                         ExAvatar's DECA / mmpose expect natural images), per-frame cam_params from our K,
                         smplx_init from the GT (replaces Hand4Whole), frame lists.
  work/<sid>/fit_test   (= fitting/data/Custom/data/<sid>_test): the same for the test frames (face fitting only).
  work/<sid>/avatar     (= avatar/data/Custom/data/<sid>): recon frames composited on black with the given alpha,
                         masks (alpha), cam_params of all frames, frame lists.
  work/<sid>/gt_conversion.json: betas100 / joint offsets / per-frame trans of the GT conversion + residuals.
"""
import argparse
import json
import os
import os.path as osp
import shutil

import cv2
import numpy as np

from exa_common import EXA_ROOT, GTConverter, cam_json, dump, key2idx, load_clip

EXA_SRC = osp.join(EXA_ROOT, 'src', 'ExAvatar_RELEASE')


def link(src, dst):
    if osp.islink(dst) or osp.exists(dst):
        os.remove(dst) if osp.islink(dst) else shutil.rmtree(dst)
    os.symlink(src, dst)


def write_lists(root, keys):
    txt = ''.join('%d\n' % key2idx(k) for k in keys)
    for name in ['frame_list_all.txt', 'frame_list_train.txt', 'frame_list_test.txt']:
        with open(osp.join(root, name), 'w') as f:
            f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--sid', required=True)
    args = ap.parse_args()

    fr, K, gt = load_clip(args.clip_dir)
    conv = GTConverter(gt['betas'])
    work = osp.join(EXA_ROOT, 'work', args.sid)
    os.makedirs(work, exist_ok=True)

    for split in ['recon', 'test']:
        root = osp.join(work, 'fit_' + split)
        if osp.exists(root):
            shutil.rmtree(root)
        for d in ['frames', 'cam_params', 'smplx_init']:
            os.makedirs(osp.join(root, d))
        for key in fr[split]:
            i = key2idx(key)
            shutil.copyfile(osp.join(args.clip_dir, 'images', key + '.png'), osp.join(root, 'frames', '%d.png' % i))
            dump(cam_json(K[key]), osp.join(root, 'cam_params', '%d.json' % i))
            p = conv.frame_params(gt, key)
            # smplx_init = what Hand4Whole would provide, here from the GT (jaw / expr: GT; eyes 0 as Hand4Whole writes)
            dump({'root_pose': p['root_pose'], 'body_pose': p['body_pose'], 'lhand_pose': p['lhand_pose'],
                  'rhand_pose': p['rhand_pose'], 'leye_pose': [0, 0, 0], 'reye_pose': [0, 0, 0],
                  'jaw_pose': gt['jaw_pose'][key].reshape(3), 'shape': conv.beta_ex,
                  'expr': gt['expression'][key].reshape(-1),
                  # GT root-joint position in the camera frame (= the fitting code's `trans`); fit_face_gt.py disables
                  # the bbox-based translation guess (Custom.get_smplx_trans_init) so this GT value is used
                  'trans': p['pelvis_cam']}, osp.join(root, 'smplx_init', '%d.json' % i))
        write_lists(root, fr[split])
        link(root, osp.join(EXA_SRC, 'fitting', 'data', 'Custom', 'data', args.sid + '_' + split))

    # avatar subject folder (training images: recon frames on black, soft alpha = the scoring GT construction)
    root = osp.join(work, 'avatar')
    if osp.exists(root):
        shutil.rmtree(root)
    for d in ['frames', 'masks', 'cam_params']:
        os.makedirs(osp.join(root, d))
    for key in fr['recon']:
        i = key2idx(key)
        img = cv2.imread(osp.join(args.clip_dir, 'images', key + '.png'), cv2.IMREAD_COLOR)
        a = cv2.imread(osp.join(args.clip_dir, 'masks', key + '.png'), cv2.IMREAD_GRAYSCALE)
        assert img.shape[:2] == (512, 512) and a.shape == (512, 512)
        comp = np.round(img.astype(np.float32) * (a.astype(np.float32)[:, :, None] / 255.0)).astype(np.uint8)
        cv2.imwrite(osp.join(root, 'frames', '%d.png' % i), comp)
        shutil.copyfile(osp.join(args.clip_dir, 'masks', key + '.png'), osp.join(root, 'masks', '%d.png' % i))
    for key in fr['keys']:
        dump(cam_json(K[key]), osp.join(root, 'cam_params', '%d.json' % key2idx(key)))
    for name, ks in [('frame_list_all.txt', fr['recon']), ('frame_list_train.txt', fr['recon']),
                     ('frame_list_test.txt', fr['test'])]:
        with open(osp.join(root, name), 'w') as f:
            f.write(''.join('%d\n' % key2idx(k) for k in ks))
    link(root, osp.join(EXA_SRC, 'avatar', 'data', 'Custom', 'data', args.sid))

    dump({'beta_ex': conv.beta_ex, 'J_gt_rest': conv.J_gt, 'joint_offset_no_face_offset': conv.joint_offset(),
          'pelvis_ex_rest': conv.pelvis_ex(), 'stats': conv.stats,
          'keys': fr['keys'], 'recon': fr['recon'], 'test': fr['test']}, osp.join(work, 'gt_conversion.json'))
    print(json.dumps(conv.stats))


if __name__ == '__main__':
    main()
