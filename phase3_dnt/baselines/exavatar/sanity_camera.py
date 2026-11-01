"""Camera / pose-convention sanity check through ExAvatar's own avatar code (run with cwd = ExAvatar avatar/main).

For every frame of a clip: GT SMPL-X 2020 mesh (contract formula) vs. ExAvatar's posed template (HumanGaussian.forward with
the Gaussian offset networks zeroed -> Gaussian means = ExAvatar's posed, upsampled SMPL-X mesh, its own LBS / hand-mean /
joint-offset handling) driven by the converted GT params (GT jaw / eyes / expr here). Reports
  * vertex error (mm) between the two meshes (first 10,475 upsampled vertices = the SMPL-X vertices),
  * IoU(silhouette, masks/<key>.png > 127) where the silhouette is rendered with ExAvatar's GaussianRenderer
    (get_fov / get_view_matrix / get_proj_matrix + diff-gaussian-rasterization-depth) using our K and identity extrinsics,
  * IoU of the GT mesh rasterised with our K (pytorch3d, ExAvatar's utils/vis.py convention) vs the mask (reference),
  * IoU between the two silhouettes.
"""
import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # avatar/main
from config import cfg  # noqa: E402
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from exa_common import EXA_ROOT, SMPLX2020, GTConverter, key2idx, load_clip  # noqa: E402


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def raster_mask(verts, faces, K, H=512, W=512):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
    v = torch.as_tensor(verts, dtype=torch.float32).cuda()[None]
    v = torch.stack((-v[:, :, 0], -v[:, :, 1], v[:, :, 2]), 2)  # as ExAvatar utils/vis.py render_mesh
    f = torch.as_tensor(faces.astype(np.int64)).cuda()[None]
    cams = PerspectiveCameras(focal_length=torch.tensor([[K[0, 0], K[1, 1]]], dtype=torch.float32).cuda(),
                              principal_point=torch.tensor([[K[0, 2], K[1, 2]]], dtype=torch.float32).cuda(),
                              in_ndc=False, image_size=torch.tensor([[H, W]]).cuda(), device='cuda')
    rs = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    frag = MeshRasterizer(cameras=cams, raster_settings=rs)(Meshes(v, f))
    return (frag.pix_to_face[0, :, :, 0] >= 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hand_convention', default='minus_mean', choices=['minus_mean', 'absolute'])
    ap.add_argument('--gauss_scale_px', type=float, default=1.5, help='isotropic Gaussian std in pixels at the median depth')
    ap.add_argument('--vis', default=None)
    args = ap.parse_args()

    import smplx as smplx_pip  # pip smplx 0.1.28 for the GT 2020 model (contract)
    from nets.module import HumanGaussian, GaussianRenderer
    from utils.smpl_x import smpl_x

    fr, K, gt = load_clip(args.clip_dir)
    conv = GTConverter(gt['betas'])
    keys = fr['keys']
    gl = smplx_pip.SMPLX(SMPLX2020, num_betas=200, num_expression_coeffs=50, use_pca=False, flat_hand_mean=True).cuda()
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32).cuda()
    with torch.no_grad():
        g = gl(betas=t(gt['betas'])[None].repeat(len(keys), 1),
               global_orient=t([gt['global_orient'][k] for k in keys]), body_pose=t([gt['body_pose'][k].reshape(-1) for k in keys]),
               jaw_pose=t([gt['jaw_pose'][k] for k in keys]), leye_pose=t([gt['leye_pose'][k] for k in keys]),
               reye_pose=t([gt['reye_pose'][k] for k in keys]),
               left_hand_pose=t([gt['left_hand_pose'][k].reshape(-1) for k in keys]),
               right_hand_pose=t([gt['right_hand_pose'][k].reshape(-1) for k in keys]),
               expression=t([gt['expression'][k] for k in keys]), transl=t([gt['transl'][k] for k in keys]))
    verts_gt = g.vertices.cpu().numpy()

    smpl_x.set_id_info(torch.FloatTensor(conv.beta_ex), torch.zeros((smpl_x.vertex_num, 3)),
                       torch.FloatTensor(conv.joint_offset()), torch.zeros((55, 3)))
    hg = HumanGaussian().cuda()
    with torch.no_grad():
        hg.init()
        for net in [hg.mean_offset_net, hg.mean_offset_offset_net]:
            for p in net[-1].parameters() if isinstance(net, torch.nn.Sequential) else net.parameters():
                p.zero_()
    renderer = GaussianRenderer()
    hand_v = np.concatenate([smpl_x.lhand_vertex_idx, smpl_x.rhand_vertex_idx])
    rows, vis = [], []
    for n, key in enumerate(keys):
        p = conv.frame_params(gt, key)
        if args.hand_convention == 'absolute':
            p['lhand_pose'] = gt['left_hand_pose'][key].reshape(15, 3)
            p['rhand_pose'] = gt['right_hand_pose'][key].reshape(15, 3)
        prm = {'root_pose': p['root_pose'], 'body_pose': p['body_pose'], 'lhand_pose': p['lhand_pose'],
               'rhand_pose': p['rhand_pose'], 'jaw_pose': gt['jaw_pose'][key], 'leye_pose': gt['leye_pose'][key],
               'reye_pose': gt['reye_pose'][key], 'expr': gt['expression'][key], 'trans': p['trans']}
        prm = {k: t(v).view(-1) for k, v in prm.items()}
        cam = {'R': torch.eye(3).cuda(), 't': torch.zeros(3).cuda(), 'focal': t([K[key][0, 0], K[key][1, 1]]),
               'princpt': t([K[key][0, 2], K[key][1, 2]])}
        with torch.no_grad():
            asset, _, _, _ = hg(prm, cam)
            mean3d = asset['mean_3d']
            ga = {'mean_3d': mean3d, 'opacity': torch.ones_like(mean3d[:, :1]),
                  'scale': torch.full_like(mean3d, args.gauss_scale_px * float(mean3d[:, 2].median()) / float(K[key][0, 0])),
                  'rotation': asset['rotation'],
                  'rgb': torch.ones_like(mean3d)}
            r = renderer(ga, (512, 512), cam, torch.zeros(3).cuda())
        sil_g = r['mask'].squeeze().cpu().numpy() > 0.5  # (H, W) alpha
        v_ex = mean3d[:smpl_x.vertex_num].cpu().numpy()
        err = np.linalg.norm(v_ex - verts_gt[n], axis=1) * 1000
        sil_m = raster_mask(verts_gt[n], gl.faces, K[key])
        mask = cv2.imread(osp.join(args.clip_dir, 'masks', key + '.png'), cv2.IMREAD_GRAYSCALE) > 127
        rows.append({'key': key, 'v_err_mm_mean': float(err.mean()), 'v_err_mm_max': float(err.max()),
                     'hand_v_err_mm_mean': float(err[hand_v].mean()),
                     'iou_exavatar_gauss_vs_mask': iou(sil_g, mask), 'iou_gt_mesh_vs_mask': iou(sil_m, mask),
                     'iou_exavatar_gauss_vs_gt_mesh': iou(sil_g, sil_m)})
        if args.vis and n in (0, len(keys) // 2, len(keys) - 1):
            img = cv2.imread(osp.join(args.clip_dir, 'images', key + '.png'))
            ov = img.copy()
            ov[sil_g] = (0.5 * ov[sil_g] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnt, -1, (0, 0, 255), 1)
            vis.append(ov)
    summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'key'}
    summ['hand_convention'] = args.hand_convention
    summ['n_frames'] = len(rows)
    summ['conversion'] = conv.stats
    with open(args.out, 'w') as f:
        json.dump({'summary': summ, 'frames': rows}, f, indent=1)
    if vis:
        cv2.imwrite(args.vis, np.concatenate(vis, 1))
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
