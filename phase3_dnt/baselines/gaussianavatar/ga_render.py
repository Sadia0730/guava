"""Render the test frames of one clip with a trained GaussianAvatar (stage 2 = the full model).

Same setup as the official eval.py (AvatarModel(train=False), training_setup, checkpoint load, MonoDataset_test,
batch 1), with two differences required by the protocol:
  1. the official AvatarModel.render_free_stage2 takes the SMPL pose from the TRAINING pose embedding indexed by the
     test frame index (self.pose(idx)), which is only meaningful for re-rendering training frames; here the pose and
     translation come from the test frame's own GT SMPL (batch['pose_data'], batch['transl_data']), exactly as the
     official render_free_stage1 does. Every other line of render_free_stage2 is copied verbatim below.
  2. the stage-2 checkpoint is loaded with AvatarModel.stage2_load (eval.py's load() only reads stage-1 net.pth).
Module modes as in eval.py (train mode; BatchNorm uses per-batch statistics). ga_compat.apply() as in training.
No test-time optimisation of anything. Output: <out_dir>/<name>.png, 8-bit RGB, background = model background (black).

  python ga_render.py <repo> -m <stage2 model_path> --epoch 200 --out_dir <dir> [--times_json <file>]
"""
import json
import os
import sys
import time


def render_test_stage2(self, batch_data):
    """Verbatim copy of AvatarModel.render_free_stage2 except the two lines marked CHANGED."""
    import torch
    from gaussian_renderer import render_batch

    rendered_images = []
    inp_posmap = batch_data['inp_pos_map']

    pose_batch = batch_data['pose_data']        # CHANGED: was self.pose(idx)   (idx = batch_data['pose_idx'])
    transl_batch = batch_data['transl_data']    # CHANGED: was self.transl(idx)

    if self.model_parms.smpl_type == 'smplx':
        rest_pose = batch_data['rest_pose']
        live_smpl = self.smpl_model.forward(betas=self.betas,
                                            global_orient=pose_batch[:, :3],
                                            transl=transl_batch,
                                            body_pose=pose_batch[:, 3:66],
                                            jaw_pose=rest_pose[:, :3],
                                            leye_pose=rest_pose[:, 3:6],
                                            reye_pose=rest_pose[:, 6:9],
                                            left_hand_pose=rest_pose[:, 9:54],
                                            right_hand_pose=rest_pose[:, 54:])
    else:
        live_smpl = self.smpl_model.forward(betas=self.betas,
                                            global_orient=pose_batch[:, :3],
                                            transl=transl_batch,
                                            body_pose=pose_batch[:, 3:])

    cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)

    uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()

    geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()

    pose_featmap = self.pose_encoder(inp_posmap)

    pred_res, pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                        geom_featmap=geom_featmap,
                                                        uv_loc=uv_coord_map)

    pred_res = pred_res.permute([0, 2, 1]) * 0.02  # (B, H, W ,3)
    pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()

    cano_deform_point = pred_point_res + self.query_points

    pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
    full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]

    pred_scales = pred_scales.permute([0, 2, 1])

    pred_shs = pred_shs.permute([0, 2, 1])

    pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
    pred_scales = pred_scales.repeat(1, 1, 3)

    pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()
    for batch_index in range(self.batch_size):
        FovX = batch_data['FovX'][batch_index]
        FovY = batch_data['FovY'][batch_index]
        height = batch_data['height'][batch_index]
        width = batch_data['width'][batch_index]
        world_view_transform = batch_data['world_view_transform'][batch_index]
        full_proj_transform = batch_data['full_proj_transform'][batch_index]
        camera_center = batch_data['camera_center'][batch_index]

        points = full_pred[batch_index]
        colors = pred_shs[batch_index]
        scales = pred_scales[batch_index]

        rendered_images.append(
            render_batch(
                points=points,
                shs=None,
                colors_precomp=colors,
                rotations=self.fix_rotation,
                scales=scales,
                opacity=self.fix_opacity,
                FovX=FovX,
                FovY=FovY,
                height=height,
                width=width,
                bg_color=self.background,
                world_view_transform=world_view_transform,
                full_proj_transform=full_proj_transform,
                active_sh_degree=0,
                camera_center=camera_center
            )
        )

    return torch.stack(rendered_images, dim=0)


def main():
    repo = os.path.abspath(sys.argv[1])
    os.chdir(repo)
    sys.path.insert(0, repo)
    import numpy as np
    import torch
    from argparse import ArgumentParser
    from PIL import Image
    from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
    from model.avatar_model import AvatarModel
    from utils.general_utils import safe_state, to_cuda
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    import ga_compat
    ga_compat.apply()

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--times_json", type=str, default=None)
    args = get_combined_args(parser)
    assert args.train_stage == 2, "renders the full (stage-2) model"
    safe_state(args.quiet)
    mp, npar, opar = model.extract(args), network.extract(args), op.extract(args)
    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        t0 = time.time()
        am = AvatarModel(mp, npar, opar, train=False)
        am.training_setup()
        am.stage2_load(args.epoch)
        # Module modes left as constructed (train mode), exactly as the official eval.py / render_novel_pose.py do:
        # the BatchNorm layers of the decoder / pose encoder therefore use per-batch statistics at test time too.
        print('background', am.background.tolist(), flush=True)
        test_dataset = am.getTestDataset()
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        torch.cuda.synchronize(); t_load = time.time() - t0
        t1 = time.time(); gpu_s = 0.0
        for idx, batch_data in enumerate(loader):
            name = test_dataset.name_list[idx][1]
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            torch.cuda.synchronize(); tg = time.time()
            image = render_test_stage2(am, batch_data)[0]
            torch.cuda.synchronize(); gpu_s += time.time() - tg
            arr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()  # = torchvision save_image
            assert arr.shape == (512, 512, 3), arr.shape
            Image.fromarray(arr).save(os.path.join(args.out_dir, f"{name}.png"))
        t_loop = time.time() - t1
    n = len(test_dataset)
    rec = {"n": n, "load_s": t_load, "loop_s": t_loop, "loop_s_per_frame": t_loop / n, "gpu_s_per_frame": gpu_s / n}
    print(json.dumps(rec), flush=True)
    if args.times_json:
        json.dump(rec, open(args.times_json, "w"))


if __name__ == "__main__":
    main()
