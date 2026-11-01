"""Camera-convention sanity check for the GaussianAvatar adapter (run after ga_convert + the canonical posmap).

(a) method path: GaussianAvatar's own initial template -- its canonical query points (posmap 512, no offsets), posed by
    its own LBS (query_lbs, inv_mats, live SMPL A) with the loader's per-frame camera (cam_static 0) and rendered by its
    3DGS rasterizer (fixed isotropic scale 4 mm, white, opacity 1, black bg) -> silhouette = rendered > 0.5.
(b) reference: the posed SMPL mesh (same SMPL params) projected with K (continuous pixel convention) and filled with
    cv2.fillPoly at pixel centres.
IoU of each with masks/<key>.png (alpha > 127), on the train (recon) frames. Writes a json + an overlay jpg.

  python cam_sanity.py <repo> <data_dir> <out_prefix> [max_frames]
"""
import json
import os
import sys


def main():
    repo, data_dir, out_prefix = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    os.chdir(repo)
    sys.path.insert(0, repo)
    import cv2
    import numpy as np
    import torch
    from argparse import ArgumentParser
    from arguments import ModelParams, NetworkParams, OptimizationParams
    from model.avatar_model import AvatarModel
    from gaussian_renderer import render_batch
    from utils.general_utils import to_cuda
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    import ga_compat
    ga_compat.apply()

    parser = ArgumentParser()
    lp, npp, op = ModelParams(parser), NetworkParams(parser), OptimizationParams(parser)
    args = parser.parse_args(["-s", data_dir, "-m", out_prefix + "_model", "--train_stage", "1", "--cam_static", "0",
                              "--no_mask", "1"])
    mp = lp.extract(args); mp.white_background = False
    torch.manual_seed(0)
    with torch.no_grad():
        am = AvatarModel(mp, npp.extract(args), op.extract(args), train=True)
        ds = am.train_dataset
        faces = am.smpl_model.faces.astype(np.int64)
        ious_a, ious_b, ious_ab, tiles = [], [], [], []
        n = min(len(ds), max_frames)
        sel = list(range(0, n - (n % 2)))
        for i0 in range(0, len(sel), 2):
            items = [ds[sel[i0]], ds[sel[i0 + 1]]]
            batch = {k: (torch.stack([torch.as_tensor(it[k]) for it in items]) if not isinstance(items[0][k], (int, float))
                         else torch.tensor([it[k] for it in items])) for k in items[0]}
            batch = to_cuda(batch, device=torch.device("cuda:0"))
            idx = batch["pose_idx"]
            pose, transl = am.pose(idx), am.transl(idx)
            live = am.smpl_model.forward(betas=am.betas, global_orient=pose[:, :3], transl=transl, body_pose=pose[:, 3:])
            mats = torch.matmul(live.A, am.inv_mats)
            pt = torch.einsum('bnj,bjxy->bnxy', am.query_lbs, mats)
            pts = torch.einsum('bnxy,bny->bnx', pt[..., :3, :3], am.query_points) + pt[..., :3, 3]
            for b in range(2):
                img = render_batch(points=pts[b], shs=None, colors_precomp=torch.ones_like(pts[b]), rotations=am.fix_rotation,
                                   scales=torch.full_like(pts[b], 0.004), opacity=am.fix_opacity, FovX=batch["FovX"][b],
                                   FovY=batch["FovY"][b], height=batch["height"][b], width=batch["width"][b],
                                   bg_color=torch.zeros(3, device="cuda"), world_view_transform=batch["world_view_transform"][b],
                                   full_proj_transform=batch["full_proj_transform"][b], active_sh_degree=0,
                                   camera_center=batch["camera_center"][b])
                sil_a = (img.mean(0) > 0.5).cpu().numpy()
                name = ds.name_list[int(idx[b])][1]
                m = cv2.imread(os.path.join(data_dir, "train", "masks", name + ".png"), cv2.IMREAD_UNCHANGED)
                m = (m if m.ndim == 2 else m[..., 0]) > 127
                K = np.load(os.path.join(data_dir, "train", "cam_parms", name + ".npz"))["intrinsic"]
                v = live.vertices[b].cpu().numpy().astype(np.float64)
                uv = v[:, :2] / v[:, 2:] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
                sil_b = np.zeros((512, 512), np.uint8)
                tri = np.round((uv[faces] - 0.5) * 16).astype(np.int32)       # pixel centre (j+0.5, i+0.5); 4 fractional bits
                for t in tri:                                                  # one call per triangle (fillPoly on a
                    cv2.fillConvexPoly(sil_b, t, 1, lineType=cv2.LINE_8, shift=4)  # list would XOR overlaps)
                sil_b = sil_b.astype(bool)
                iou = lambda x, y: float((x & y).sum() / max((x | y).sum(), 1))   # noqa: E731
                ious_a.append(iou(sil_a, m)); ious_b.append(iou(sil_b, m)); ious_ab.append(iou(sil_a, sil_b))
                if len(tiles) < 4 and int(idx[b]) % max(1, n // 4) == 0:
                    rgb = cv2.imread(os.path.join(data_dir, "train", "images", name + ".png"))
                    ov = rgb.copy(); ov[sil_a] = (0.5 * ov[sil_a] + [0, 0, 127]).astype(np.uint8)
                    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(ov, cnt, -1, (0, 255, 0), 1)
                    tiles.append(ov)
    rec = {"frames": len(ious_a), "iou_template_render_vs_mask": float(np.mean(ious_a)),
           "iou_template_render_vs_mask_min": float(np.min(ious_a)), "iou_mesh_fill_vs_mask": float(np.mean(ious_b)),
           "iou_template_render_vs_mesh_fill": float(np.mean(ious_ab))}
    print(json.dumps(rec), flush=True)
    json.dump(rec, open(out_prefix + ".json", "w"), indent=1)
    if tiles:
        cv2.imwrite(out_prefix + ".jpg", np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
