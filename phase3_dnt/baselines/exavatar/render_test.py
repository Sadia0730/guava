"""Render the test frames of one clip with a trained ExAvatar (run with cwd = ExAvatar avatar/main).

Same path as ExAvatar's main/animate.py (Tester checkpoint loading, human_gaussian(smplx_param, cam_param), GaussianRenderer)
with: refined Gaussians (`human_asset_refined`, what test.py / eval_neuman.py score), black background (bg = 0 instead of
white), our per-frame K (identity extrinsics), 512 x 512, 8-bit PNG named <key>.png.
Driving params of a test frame: GT body (converted to ExAvatar's layer) + ExAvatar's own face fit of that frame.
`trans` is re-anchored to the trained shape: transl = transl_gt + pelvis_gt_rest - pelvis_ex_rest(trained shape_param)
(the root joint's rest position depends only on the shape; ExAvatar refines shape_param during training).
No test-time optimisation.
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # avatar/main
from config import cfg  # noqa: E402
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from exa_common import EXA_ROOT, GTConverter, key2idx, load_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--sid', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--test_epoch', default='4')
    args = ap.parse_args()
    assert cfg.dataset == 'Custom'
    cfg.set_args(args.sid)
    from base import Tester
    from utils.smpl_x import smpl_x

    fr, K, gt = load_clip(args.clip_dir)
    conv = GTConverter(gt['betas'])
    work = osp.join(EXA_ROOT, 'work', args.sid)
    so = osp.join(work, 'avatar', 'smplx_optimized')
    ld = lambda n: torch.FloatTensor(json.load(open(osp.join(so, n))))
    smpl_x.set_id_info(ld('shape_param.json'), ld('face_offset.json'), ld('joint_offset.json'), ld('locator_offset.json'))

    t0 = time.time()
    tester = Tester(args.test_epoch)
    tester.smplx_params = None
    tester._make_model()
    hg = tester.model.module.human_gaussian
    renderer = tester.model.module.gaussian_renderer
    torch.cuda.synchronize()
    load_s = time.time() - t0

    shape_trained = hg.shape_param.detach().cpu().numpy().astype(np.float64)
    pelvis_trained = conv.pelvis_ex(shape_trained)
    shift_mm = float(np.linalg.norm(pelvis_trained - conv.pelvis_ex()) * 1000)

    os.makedirs(args.out_dir, exist_ok=True)
    bg = torch.zeros((3)).float().cuda()
    times = []
    for key in fr['test']:
        i = key2idx(key)
        with open(osp.join(work, 'test_params', '%d.json' % i)) as f:
            prm = json.load(f)
        prm['trans'] = (gt['transl'][key].reshape(3) + conv.J_gt[0] - pelvis_trained).tolist()
        smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in prm.items()}
        with open(osp.join(work, 'avatar', 'cam_params', '%d.json' % i)) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()}
        torch.cuda.synchronize()
        ts = time.time()
        with torch.no_grad():
            _, human_asset_refined, _, _ = hg(smplx_param, cam_param)
            out = renderer(human_asset_refined, (512, 512), cam_param, bg)
            img = out['img']
        torch.cuda.synchronize()
        times.append(time.time() - ts)
        img = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(osp.join(args.out_dir, key + '.png'), img[:, :, ::-1])
    res = {'n_test': len(fr['test']), 'render_s_per_frame': float(np.mean(times)), 'render_s_first': float(times[0]),
           'render_s_per_frame_excl_first': float(np.mean(times[1:])) if len(times) > 1 else None,
           'model_load_s': load_s, 'pelvis_reanchor_mm': shift_mm}
    with open(osp.join(work, 'render_stats.json'), 'w') as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
