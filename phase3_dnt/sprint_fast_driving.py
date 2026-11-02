"""Opt-in SPRINT driving backend; source fitting and model weights stay unchanged.

Raw driving needs only a person box, crop, camera features, and SPRINT forward.
The baseline fitter also runs keypoints, matting, a mesh, and fit diagnostics at
k=0. Those outputs are unused by the avatar's driving/rendering path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

_spec = importlib.util.spec_from_file_location('_sprint_fitter_baseline', Path(__file__).with_name('sprint_avatar_fit.py'))
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
SprintFitterBase = _base.SprintFitter


class SprintFitter(SprintFitterBase):
    def __init__(self, *args, dwpose_provider='CPUExecutionProvider', **kwargs):
        super().__init__(*args, **kwargs)
        self.dwpose_provider = dwpose_provider
        self.fast_dwpose_threads = kwargs.get('dwpose_threads', 16)
        if dwpose_provider != 'CPUExecutionProvider':
            self.set_dwpose_provider(dwpose_provider)

    def set_dwpose_provider(self, provider):
        import onnxruntime as ort
        if provider not in ort.get_available_providers():
            raise RuntimeError(f'Unavailable provider: {provider}')
        # The host already has CUDA 11 libraries in the isolated baseline env.
        # Load only its required NVIDIA libraries, not its Python/BLAS runtime.
        # PyTorch's CUDA 12 libraries have different sonames and remain intact.
        if provider == 'CUDAExecutionProvider':
            import ctypes
            libroot = Path('/dev/shm/ubx858_baselines/gart/env/lib')
            for name in ('libcudart.so.11.0', 'libcublasLt.so.11', 'libcublas.so.11', 'libcufft.so.10'):
                p = libroot / name
                if p.exists():
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            cudnn = Path(sys.prefix) / 'lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8'
            if cudnn.exists():
                ctypes.CDLL(str(cudnn), mode=ctypes.RTLD_GLOBAL)
        so = ort.SessionOptions(); so.intra_op_num_threads = self.fast_dwpose_threads
        providers = [provider] if provider == 'CPUExecutionProvider' else [provider, 'CPUExecutionProvider']
        det = ort.InferenceSession(str(_base.EHMT / 'pretrained/dwpose/yolox_l.onnx'), so, providers=providers)
        pose = ort.InferenceSession(str(_base.EHMT / 'pretrained/dwpose/dw-ll_ucoco_384.onnx'), so, providers=providers)
        if any(s.get_providers()[0] != provider for s in (det, pose)):
            raise RuntimeError(f'{provider} fell back; refusing misleading timing')
        self.s_det, self.s_pose = det, pose
        self.dwpose_provider = provider

    def fit(self, img_rgb, box=None, iters=None, shape_lock=None, **kwargs):
        if iters == 0 and not kwargs:
            return self.drive_raw(img_rgb, box=box, shape_lock=shape_lock)
        return super().fit(img_rgb, box=box, iters=iters, shape_lock=shape_lock, **kwargs)

    @torch.no_grad()
    def drive_raw(self, img_rgb, box=None, shape_lock=None):
        import roma
        T = {}; H, W = img_rgb.shape[:2]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if box is None:
            boxes = self.det.inference_detector(self.s_det, img_rgb)
            if len(boxes) == 0:
                raise RuntimeError('no person detected')
            box = boxes[int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))]
        box = np.asarray(box, np.float64)
        torch.cuda.synchronize(); T['detector'] = time.perf_counter() - t0
        M = _base.box_crop_matrix(box, _base.CROP)
        crop = cv2.warpAffine(img_rgb, M[:2], (_base.CROP, _base.CROP), flags=cv2.INTER_LINEAR)
        crop_t = torch.from_numpy(crop).to(self.dev).permute(2, 0, 1)[None].float() / 255.0
        A256 = (np.diag([256 / _base.CROP, 256 / _base.CROP, 1.0]) @ M)[:2]
        cam = self.ev.cam_feat(A256, W, H, W, 0.0)
        out = self.sprint(crop_t, cam, shape_lock)
        torch.cuda.synchronize(); T['sprint_and_crop'] = time.perf_counter() - t0 - T['detector']
        aa = lambda r: roma.rotmat_to_rotvec(r.reshape(-1, 3, 3)).cpu().numpy().astype(np.float32)
        bp = out['body_param']
        RT = out['pd_cam'][0, :3, :4].cpu().numpy().astype(np.float32)
        raw = {'global_pose': aa(bp['global_pose'])[0], 'body_pose': aa(bp['body_pose']),
               'left_hand_pose': aa(bp['left_hand_pose']), 'right_hand_pose': aa(bp['right_hand_pose']),
               'jaw_pose': aa(bp['jaw_pose']).reshape(1, 3), 'eye_pose': np.zeros((2, 3), np.float32),
               'exp': bp['exp'][0, :50].cpu().numpy().astype(np.float32),
               'camera_RT_params': RT, 'body_cam': np.zeros(3, np.float32)}
        Mlr = _base.box_crop_matrix(box, _base.BODY_LR)
        Mp = _base.box_crop_matrix(box, _base.PART)
        # Part crops are interface placeholders, unused by driving rendering.
        # drive_raw must not be used to create a source avatar.
        rec = {'smplx_coeffs': raw,
               'body_crop': {'M_o2c': Mlr, 'M_c2o': np.linalg.inv(Mlr).astype(np.float32),
                             'M_o2c-hd': M, 'M_c2o-hd': np.linalg.inv(M).astype(np.float32)},
               **{p: {'M_o2c': Mp, 'M_c2o': np.linalg.inv(Mp).astype(np.float32)}
                  for p in ('head_crop', 'left_hand_crop', 'right_hand_crop')}}
        shape = bp['shape'][0].cpu().numpy()
        torch.cuda.synchronize(); T['total'] = time.perf_counter() - t0
        return {'record': rec, 'raw_smplx': raw, 'shape': shape, 'raw_shape': shape,
                'box': box, 'M': M, 'times': T, 'cam_feat': cam}


# Preserve source-writing helpers expected by the existing evaluator.
write_tracked = _base.write_tracked
