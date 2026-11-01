"""ExAvatar's SMPL-X/FLAME fitting (fitting/main/fit.py) with the BODY replaced by the GT; run from fitting/main.

Identical to fit.py (same Model / losses / Trainer / data loader, same epochs (3), batch size (64), iterations (500/250/250),
stage schedule, learning rates) EXCEPT:
  * per-frame SMPL-X root_pose, body_pose, lhand_pose, rhand_pose and trans are the GT (from smplx_init, written by
    prep_clip.py) and are NOT optimised (fit.py optimises them from 2D keypoints);
  * the shared body-ID parameters are the GT conversion and are NOT optimised: smplx_shape = least-squares betas100,
    joint_offset = GT rest joints - ExAvatar rest joints (recomputed each iteration from the current face_offset, detached),
    locator_offset = 0;
  * the bbox-based translation guess (Custom.get_smplx_trans_init) is disabled; smplx_init['trans'] holds the GT root joint;
  * stage 'recon': per-frame FLAME (root, neck, jaw, eyes, expr, trans) + shared flame_shape and face_offset are optimised
    exactly as in fit.py (ExAvatar's own face fitting, ID from the recon frames only);
  * stage 'test': per-frame FLAME params of the test frames are optimised, flame_shape / face_offset are frozen at the
    recon result (no test-frame ID information);
  * visualisation renders / videos of fit.py are skipped; flame_shape and the raw face_offset are additionally saved.
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.getcwd())  # ExAvatar's fitting/main (this script is run with cwd = fitting/main)
from config import cfg  # fitting/main/config.py (run with cwd = fitting/main)
from pytorch3d.io import save_ply
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
from utils.flame import flame
from utils.smpl_x import smpl_x
import Custom as CustomModule  # data/Custom/Custom.py (added to sys.path by config.py)


def rotation_6d_to_axis_angle(x):
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject_id', required=True)
    ap.add_argument('--gt_conv', required=True)
    ap.add_argument('--stage', choices=['recon', 'test'], required=True)
    ap.add_argument('--id_dir', default=None, help="stage 'test': smplx_optimized folder of the recon stage")
    args = ap.parse_args()
    t0 = time.time()

    # GT translation instead of ExAvatar's bbox-based guess (must be patched before the dataset is built)
    CustomModule.Custom.get_smplx_trans_init = lambda self: None
    from base import Trainer

    cfg.set_args(args.subject_id)
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    with open(args.gt_conv) as f:
        gtc = json.load(f)
    J_gt = torch.tensor(gtc['J_gt_rest'], dtype=torch.float32).cuda()
    beta_ex = torch.tensor(gtc['beta_ex'], dtype=torch.float32).cuda()
    layer = smpl_x.layer
    v_shaped_ex = (layer.v_template + torch.einsum('vck,k->vc', layer.shapedirs[:, :, :smpl_x.shape_param_dim],
                                                   beta_ex.cpu())).cuda()
    J_reg = layer.J_regressor.cuda()

    def gt_joint_offset(face_offset):
        with torch.no_grad():
            v = v_shaped_ex + smpl_x.get_face_offset(face_offset[None])[0]
            jo = J_gt - J_reg @ v
            jo[0] = 0
        return jo

    # per-frame FLAME parameters (initialised from DECA as in fit.py)
    flame_params = {}
    for frame_idx in trainer.flame_params.keys():
        flame_params[frame_idx] = {}
        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
            flame_params[frame_idx][key] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(trainer.flame_params[frame_idx][key].cuda())))
        flame_params[frame_idx]['expr'] = nn.Parameter(trainer.flame_params[frame_idx]['expr'].cuda())
        flame_params[frame_idx]['trans'] = nn.Parameter(trainer.flame_params[frame_idx]['trans'].cuda())

    # shared ID parameters
    if args.stage == 'recon':
        flame_shape = nn.Parameter(trainer.flame_shape_param.float().cuda())
        face_offset = nn.Parameter(torch.zeros((flame.vertex_num, 3)).float().cuda())
    else:
        with open(osp.join(args.id_dir, 'flame_shape_param.json')) as f:
            flame_shape = torch.FloatTensor(json.load(f)).cuda()
        with open(osp.join(args.id_dir, 'face_offset_flame.json')) as f:
            face_offset = torch.FloatTensor(json.load(f)).cuda()
        # the FLAME shape prior must not come from test frames either: use the recon-stage shape
        trainer.trainset_loader.flame_shape_param = flame_shape.detach().cpu()
    smplx_shape = beta_ex.clone()  # GT (least squares), frozen
    locator_offset = torch.zeros((smpl_x.joint['num'], 3)).float().cuda()  # frozen

    # per-frame SMPL-X parameters: body = GT (frozen); jaw / eyes / expr shared with FLAME (optimised by the face fit)
    smplx_params = {}
    for frame_idx in trainer.smplx_params.keys():
        smplx_params[frame_idx] = {}
        for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose']:
            smplx_params[frame_idx][key] = matrix_to_rotation_6d(axis_angle_to_matrix(trainer.smplx_params[frame_idx][key].cuda()))
        smplx_params[frame_idx]['jaw_pose'] = flame_params[frame_idx]['jaw_pose']
        smplx_params[frame_idx]['leye_pose'] = flame_params[frame_idx]['leye_pose']
        smplx_params[frame_idx]['reye_pose'] = flame_params[frame_idx]['reye_pose']
        smplx_params[frame_idx]['expr'] = flame_params[frame_idx]['expr']
        smplx_params[frame_idx]['trans'] = trainer.smplx_params[frame_idx]['trans'].cuda()

    n_opt_steps = 0
    for epoch in range(cfg.end_epoch):
        cfg.set_itr_opt_num(epoch)
        for itr_data, data in enumerate(trainer.batch_generator):
            batch_size = data['img_orig'].shape[0]
            for itr_opt in range(cfg.itr_opt_num):
                cfg.set_stage(epoch, itr_opt)
                if (epoch == 0) and (itr_opt == 0):
                    optimizable_params = []  # fit.py: smplx + flame root pose / trans; smplx ones are GT here
                    for frame_idx in data['frame_idx']:
                        for key in ['root_pose', 'trans']:
                            optimizable_params.append(flame_params[int(frame_idx)][key])
                    trainer.get_optimizer(optimizable_params)
                elif ((epoch == 0) and (itr_opt == cfg.stage_itr[0])) or ((epoch > 0) and (itr_opt == 0)):
                    if (epoch == (cfg.end_epoch - 1)) or (args.stage == 'test'):
                        optimizable_params = []
                    else:
                        optimizable_params = [flame_shape, face_offset]  # fit.py also has smplx_shape, joint/locator offsets (GT here)
                    for frame_idx in data['frame_idx']:
                        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr', 'trans']:
                            optimizable_params.append(flame_params[int(frame_idx)][key])
                    trainer.get_optimizer(optimizable_params)

                joint_offset = gt_joint_offset(face_offset)
                smplx_inputs = {'shape': [smplx_shape for _ in range(batch_size)], 'face_offset': [face_offset for _ in range(batch_size)],
                                'joint_offset': [joint_offset for _ in range(batch_size)], 'locator_offset': [locator_offset for _ in range(batch_size)]}
                flame_inputs = {'shape': [flame_shape for _ in range(batch_size)]}
                for frame_idx in data['frame_idx']:
                    for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                        smplx_inputs.setdefault(key, []).append(smplx_params[int(frame_idx)][key])
                    for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr', 'trans']:
                        flame_inputs.setdefault(key, []).append(flame_params[int(frame_idx)][key])
                smplx_inputs = {k: torch.stack(v) for k, v in smplx_inputs.items()}
                flame_inputs = {k: torch.stack(v) for k, v in flame_inputs.items()}

                trainer.set_lr(itr_opt)
                trainer.optimizer.zero_grad()
                loss, out = trainer.model(smplx_inputs, flame_inputs, data, return_output=((epoch == cfg.end_epoch - 1) and (itr_opt == cfg.itr_opt_num - 1)))
                loss = {k: loss[k].mean() for k in loss}
                sum(loss[k] for k in loss).backward()
                trainer.optimizer.step()
                n_opt_steps += 1
                if itr_opt % 50 == 0 or itr_opt == cfg.itr_opt_num - 1:
                    print('[%s] epoch %d/%d itr_data %d/%d itr_opt %d/%d lr %g ' % (args.stage, epoch, cfg.end_epoch, itr_data, trainer.itr_per_epoch, itr_opt, cfg.itr_opt_num, trainer.get_lr())
                          + ' '.join('%s %.4f' % (k, float(v)) for k, v in loss.items()), flush=True)

            if epoch != (cfg.end_epoch - 1):
                continue
            # save (as fit.py)
            save_root_path = osp.join(cfg.result_dir, 'smplx_optimized')
            os.makedirs(osp.join(save_root_path, 'meshes'), exist_ok=True)
            os.makedirs(osp.join(save_root_path, 'smplx_params'), exist_ok=True)
            smplx_mesh_cam = out['smplx_mesh_cam'].detach().cpu()
            smplx_trans = out['smplx_trans'].detach().cpu().numpy()
            flame_mesh_cam = out['flame_mesh_cam'].detach().cpu()
            for i in range(batch_size):
                frame_idx = int(data['frame_idx'][i])
                if (itr_data == 0) and (i == 0):
                    save_ply(osp.join(save_root_path, 'smplx_wo_pose_wo_expr.ply'), torch.FloatTensor(out['smplx_mesh_wo_pose_wo_expr'][i].detach().cpu()).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                    save_ply(osp.join(save_root_path, 'flame_wo_pose_wo_expr.ply'), torch.FloatTensor(out['flame_mesh_wo_pose_wo_expr'][i].detach().cpu()).contiguous(), torch.IntTensor(flame.face).contiguous())
                save_ply(osp.join(save_root_path, 'meshes', str(frame_idx) + '_smplx.ply'), torch.FloatTensor(smplx_mesh_cam[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                save_ply(osp.join(save_root_path, 'meshes', str(frame_idx) + '_flame.ply'), torch.FloatTensor(flame_mesh_cam[i]).contiguous(), torch.IntTensor(flame.face).contiguous())
                with open(osp.join(save_root_path, 'smplx_params', str(frame_idx) + '.json'), 'w') as f:
                    json.dump({k: rotation_6d_to_axis_angle(smplx_params[frame_idx][k].detach().cpu()).numpy().tolist()
                               for k in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']}
                              | {'expr': smplx_params[frame_idx]['expr'].detach().cpu().numpy().tolist(), 'trans': smplx_trans[i].tolist()}, f)
                if (itr_data == 0) and (i == 0):
                    with open(osp.join(save_root_path, 'shape_param.json'), 'w') as f:
                        json.dump(smplx_shape.detach().cpu().numpy().tolist(), f)
                    with open(osp.join(save_root_path, 'face_offset.json'), 'w') as f:
                        json.dump(smpl_x.get_face_offset(face_offset[None, :, :])[0].detach().cpu().numpy().tolist(), f)
                    with open(osp.join(save_root_path, 'joint_offset.json'), 'w') as f:
                        json.dump(smpl_x.get_joint_offset(joint_offset[None, :, :])[0].detach().cpu().numpy().tolist(), f)
                    with open(osp.join(save_root_path, 'locator_offset.json'), 'w') as f:
                        json.dump(smpl_x.get_locator_offset(locator_offset[None, :, :])[0].detach().cpu().numpy().tolist(), f)
                    with open(osp.join(save_root_path, 'flame_shape_param.json'), 'w') as f:
                        json.dump(flame_shape.detach().cpu().numpy().tolist(), f)
                    with open(osp.join(save_root_path, 'face_offset_flame.json'), 'w') as f:
                        json.dump(face_offset.detach().cpu().numpy().tolist(), f)
    with open(osp.join(cfg.result_dir, 'fit_face_gt_log.json'), 'w') as f:
        json.dump({'stage': args.stage, 'frames': len(trainer.smplx_params), 'opt_steps': n_opt_steps,
                   'wall_s': time.time() - t0,
                   'flame_valid_frames': int(sum(bool(v['is_valid']) for v in trainer.trainset_loader.flame_params.values()))}, f)


if __name__ == '__main__':
    main()
