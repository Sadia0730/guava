#!/usr/bin/env python
"""Track a driving video with PEAR and write it in EHM-Tracker's output layout.

The animation stage of `main/test.py` (cross-reenactment) only consumes the
per-frame EHM parameters of the driving video: identity parameters are taken
from the source avatar by `change_id_info`, and the driving images/masks are not
rendered. PEAR predicts those per-frame parameters feed-forward, so it can
replace EHM-Tracker's per-video optimization for the driving video while the
source avatar keeps being tracked by EHM-Tracker.

Frames are cropped to one union person box, the way EHM-Tracker crops them, so
that EHM-Tracker's identity and PEAR's driving motion agree on how large the
person is - PEAR reads the camera distance off that size.

The output directory is a drop-in `--data_path` for `main/test.py`:

    python main/pear_tracking.py --in_root driving.mp4 --output_dir tracked/
    python main/test.py -m assets/GUAVA -s outputs/example \
        --data_path tracked/driving \
        --source_data_path assets/example/tracked_image/... \
        --skip_self_act --render_cross_act
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

PEAR_ROOT = Path(__file__).resolve().parents[1] / 'third_party' / 'PEAR'
EHM_ROOT = Path(__file__).resolve().parents[1] / 'EHM-Tracker'
PERSON_DETECTOR = EHM_ROOT / 'pretrained' / 'dwpose' / 'yolox_l.onnx'
PEAR_CHECKPOINT = ('BestWJH/PEAR_models', 'ehm_model_stage1.pt')
MODEL_INPUT_SIZE = 256
BODY_IMAGE_SIZE = 1024  # EHM-Tracker body_hd_size / GUAVA DATASET.origin_image_size


def pad_and_resize(image, target_size):
    """PEAR's letterbox (app.py); a plain resize for the square body crop."""
    height, width = image.shape[:2]
    scale = min(target_size / height, target_size / width)
    resized_width, resized_height = int(width * scale), int(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - resized_width) // 2
    y_offset = (target_size - resized_height) // 2
    padded[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return padded


def detect_union_box(video_path, device, interval=1):
    """EHM-Tracker's union person box (data_prepare_pipeline.py:71-90).

    PEAR infers camera distance from how large the person is in its input, so
    feeding it whole frames places the avatar further away than the EHM-tracked
    source it is driving. One box for the whole video keeps the crop - and with
    it the camera - from jittering frame to frame.
    """
    assert PERSON_DETECTOR.exists(), (
        f'Person detector not found at {PERSON_DETECTOR}, run this in EHM-Tracker: '
        'bash assets/Docs/run_download_pretrained.sh')
    sys.path.insert(0, str(EHM_ROOT))
    import onnxruntime
    from src.modules.dwpose.onnxdet import inference_detector

    providers = ['CUDAExecutionProvider'] if device.startswith('cuda') else []
    session = onnxruntime.InferenceSession(str(PERSON_DETECTOR),
                                           providers=providers + ['CPUExecutionProvider'])
    boxes = []
    reader = imageio.get_reader(str(video_path))
    for index, frame_rgb in enumerate(reader):
        if index % interval:
            continue
        detected = inference_detector(session, frame_rgb)
        if len(detected):
            boxes.append(detected[0])
    reader.close()
    assert boxes, f'No person detected in {video_path}.'

    boxes = np.array(boxes)
    left, top = boxes[:, 0].min(), boxes[:, 1].min()
    right, bottom = boxes[:, 2].max(), boxes[:, 3].max()
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    size = max(right - left, bottom - top)
    return [center_x - size / 2, center_y - size / 2, center_x + size / 2, center_y + size / 2]


def convert_frame(outputs, crop_info, matrix_to_axis_angle):
    """Convert one PEAR prediction into an EHM-Tracker per-frame record."""

    def to_axis_angle(rotmat):  # (1, N, 3, 3) rotations -> (N, 3) axis-angle
        return matrix_to_axis_angle(rotmat)[0].cpu().numpy()

    body_param, flame_param = outputs['body_param'], outputs['flame_param']
    body_pose = to_axis_angle(body_param['body_pose'])
    smplx_coeffs = {
        'exp': body_param['exp'][0].cpu().numpy(),
        'global_pose': to_axis_angle(body_param['global_pose'])[0],
        'body_pose': body_pose,
        'left_hand_pose': to_axis_angle(body_param['left_hand_pose']),
        'right_hand_pose': to_axis_angle(body_param['right_hand_pose']),
        # PEAR predicts the same PyTorch3D camera EHM-Tracker stores here, at the
        # same focal length (24) that GUAVA renders with.
        'camera_RT_params': outputs['pd_cam'][0, :3, :4].cpu().numpy(),
    }
    flame_coeffs = {
        key: flame_param[key][0].cpu().numpy()
        for key in ('expression_params', 'jaw_params', 'pose_params',
                    'eye_pose_params', 'eyelid_params')
    }
    flame_coeffs['neck_pose_params'] = np.zeros(3, dtype=np.float32)  # zeroed by EHM anyway
    identity = {
        'smplx_shape': body_param['shape'].cpu().numpy(),
        'flame_shape': flame_param['shape_params'].cpu().numpy(),
        'head_scale': body_param['head_scale'].cpu().numpy(),
        'hand_scale': body_param['hand_scale'].cpu().numpy(),
    }
    body_affine, inverse_affine = crop_info['M_o2c'], crop_info['M_c2o']
    identity_affine = np.eye(3, dtype=np.float32)
    frame_record = {
        'smplx_coeffs': smplx_coeffs,
        'flame_coeffs': flame_coeffs,
        # One crop is taken, so the hd and non-hd transforms are the same.
        'body_crop': {'M_o2c': body_affine, 'M_c2o': inverse_affine,
                      'M_o2c-hd': body_affine, 'M_c2o-hd': inverse_affine},
        # PEAR crops no head or hands, so those stay identities. They only
        # reach dataset._load_box, whose boxes are unused during inference.
        'head_crop': {'M_o2c': identity_affine, 'M_c2o': identity_affine},
        'left_hand_crop': {'M_o2c': identity_affine, 'M_c2o': identity_affine},
        'right_hand_crop': {'M_o2c': identity_affine, 'M_c2o': identity_affine},
    }
    return frame_record, identity


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in_root', type=Path, required=True, help='driving video')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--detect_interval', type=int, default=1,
                        help='detect the person box every Nth frame; the box is a union over '
                             'the video, so sampling mostly trades exactness for speed')
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.in_root.resolve()
    video_name = video_path.stem
    saving_root = (args.output_dir / video_name).resolve()
    saving_root.mkdir(parents=True, exist_ok=True)

    union_box = detect_union_box(video_path, args.device, args.detect_interval)
    from src.utils.crop import crop_image_by_bbox

    # PEAR reads its config and assets relative to its own root, and its
    # `models`/`utils` packages shadow GUAVA's, so it runs from there.
    assert PEAR_ROOT.exists(), \
        f'PEAR not found at {PEAR_ROOT}, clone https://github.com/Pixel-Talk/PEAR there.'
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    from huggingface_hub import hf_hub_download
    from pytorch3d.transforms import matrix_to_axis_angle
    from models.pipeline.ehm_pipeline import Ehm_Pipeline
    from utils.general_utils import ConfigDict, add_extra_cfgs
    from utils.lmdb import LMDBEngine

    config = add_extra_cfgs(ConfigDict(model_config_path='configs/infer.yaml'))
    try:  # skip the network round-trip (and its retries) once the model is cached
        checkpoint_path = hf_hub_download(*PEAR_CHECKPOINT, repo_type='model', local_files_only=True)
    except FileNotFoundError:
        checkpoint_path = hf_hub_download(*PEAR_CHECKPOINT, repo_type='model')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = Ehm_Pipeline(config)
    model.backbone.load_state_dict(checkpoint['backbone'], strict=False)
    model.head.load_state_dict(checkpoint['head'], strict=False)
    model = model.to(args.device).eval()
    del checkpoint

    lmdb_engine = LMDBEngine(str(saving_root / 'img_lmdb'), write=True)
    tracking_results, identity_results = {}, []
    reader = imageio.get_reader(str(video_path))
    with torch.inference_mode():
        for index, frame_rgb in enumerate(reader):
            frame_key = f'frame_{index:06d}'
            crop_info = crop_image_by_bbox(frame_rgb, union_box, dsize=BODY_IMAGE_SIZE)
            body_image = crop_info['img_crop']
            model_image = pad_and_resize(body_image, MODEL_INPUT_SIZE)
            image_tensor = torch.as_tensor(model_image, device=args.device)
            image_tensor = torch.permute(image_tensor / 255.0, (2, 0, 1)).unsqueeze(0)

            outputs = model(image_tensor)
            frame_record, identity = convert_frame(outputs, crop_info, matrix_to_axis_angle)
            tracking_results[frame_key] = frame_record
            identity_results.append(identity)

            lmdb_engine.dump(f'{frame_key}/body_image', type='image',
                             payload=torch.from_numpy(body_image).permute(2, 0, 1))
            # PEAR does not matte; the driving mask is unused by cross-reenactment.
            lmdb_engine.dump(f'{frame_key}/body_mask', type='image',
                             payload=torch.full((3, BODY_IMAGE_SIZE, BODY_IMAGE_SIZE), 255,
                                                dtype=torch.uint8))
    reader.close()
    lmdb_engine.close()
    assert tracking_results, f'No frames read from {video_path}.'

    id_share_params = {key: np.mean([identity[key] for identity in identity_results], axis=0)
                       for key in identity_results[0]}
    # PEAR does not predict joint offsets; cross-reenactment takes them from the source.
    id_share_params['joints_offset'] = np.zeros((1, 55, 3), dtype=np.float32)

    frames_keys = list(tracking_results.keys())
    with open(saving_root / 'optim_tracking_ehm.pkl', 'wb') as pkl_file:
        pickle.dump(tracking_results, pkl_file)
    with open(saving_root / 'id_share_params.pkl', 'wb') as pkl_file:
        pickle.dump(id_share_params, pkl_file)
    with open(saving_root / 'videos_info.json', 'w', encoding='utf-8') as json_file:
        json.dump({video_name: {'frames_num': len(frames_keys), 'frames_keys': frames_keys}},
                  json_file, ensure_ascii=False, indent=4)
    print(f'PEAR tracking of {len(frames_keys)} frames saved to: {saving_root}')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()
