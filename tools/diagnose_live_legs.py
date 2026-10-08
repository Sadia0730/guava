"""Isolate GUAVA leg geometry and camera framing without a pose estimator."""
import json
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main.live_pear_guava import (
    FLAME_KEYS, TargetBuilder, draw_pose_skeleton, initialize_guava, parse_args,
)


def main():
    args = parse_args()
    args.compile_targets = []
    args.warmup = 1
    args.smooth = False
    args.face_mode = "frozen"
    args.rest_lower_body = False
    args.avatar_view = "original"
    device = args.device
    rot = torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
    params = {key: rot.repeat(n, 1) for key, n in (
        ("global_pose", 1), ("body_pose", 21),
        ("left_hand_pose", 15), ("right_hand_pose", 15))}
    params["exp"] = torch.zeros(50, device=device)
    for key, size in zip(FLAME_KEYS, (50, 3, 3, 6, 2)):
        params[key] = torch.zeros(size, device=device)
    out = Path("outputs/debug/live_leg_geometry")
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        renderer, identity, dataset = initialize_guava(args, params)
        target = TargetBuilder(identity, args)(params, 0.)
        # EHM's tracked global orientation turns its native template upright.
        source = dataset._load_source_info(next(iter(dataset.videos_info)))
        target["smplx_coeffs"]["global_pose"] = source["smplx_coeffs"]["global_pose"].clone()
        from utils.graphics_utils import get_full_proj_matrix
        records = {}
        for framing in ("original", "full"):
            if framing == "full":
                cam = torch.eye(4, device=device)
                cam[2, 3] = 32.
                view, proj = get_full_proj_matrix(cam, dataset.tanfov)
                renderer.camera["world_view_transform"] = view[None]
                renderer.camera["full_proj_transform"] = proj[None]
                renderer.camera["camera_center"] = torch.linalg.inv(cam)[None, :3, 3]
            probes = {"rest": None, "left_knee": (3, 0, 1.2),
                      "right_knee": (4, 0, 1.2), "left_hip_side": (0, 2, 0.6),
                      "right_hip_side": (1, 2, -0.6), "left_hip_lift": (0, 0, -0.8)}
            for pose, rotation in probes.items():
                target["smplx_coeffs"]["body_pose"].zero_()
                if rotation is not None:
                    joint, axis, angle = rotation
                    target["smplx_coeffs"]["body_pose"][:, joint, axis] = angle
                image = renderer.render(target).cpu().numpy()
                joints = renderer.last_joints
                skel = draw_pose_skeleton(joints, renderer.camera, image.shape[0], image.shape[1])
                cv2.imwrite(str(out / f"{framing}_{pose}.png"), cv2.hconcat([image, skel]))
                records[f"{framing}_{pose}"] = joints[0, :22].cpu().tolist()
        (out / "joints.json").write_text(json.dumps(records, indent=2))
        dataset._lmdb_engine.close()
    print(out.resolve())


if __name__ == "__main__":
    main()
