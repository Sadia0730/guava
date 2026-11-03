"""Opt-in SPRINT driving backend; source fitting and model weights stay unchanged.

Raw driving needs only a person box, crop, camera features, and SPRINT forward.
The baseline fitter also runs keypoints, matting, a mesh, and fit diagnostics at
k=0. Those outputs are unused by the avatar's driving/rendering path.
"""
from __future__ import annotations

import importlib.util
import os
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
    def __init__(self, *args, dwpose_provider='CPUExecutionProvider', driving_matte_size=1024,
                 detector_fp16=False, sprint_graph=False, **kwargs):
        self.driving_matte_size = driving_matte_size
        self._in_driving_refinement = False
        self.detector_fp16 = detector_fp16
        self.use_sprint_graph = sprint_graph
        self._sprint_graph = None
        self.sprint_graph_error = None
        super().__init__(*args, **kwargs)
        self.dwpose_provider = dwpose_provider
        self.fast_dwpose_threads = kwargs.get('dwpose_threads', 16)
        if dwpose_provider != 'CPUExecutionProvider':
            self.set_dwpose_provider(dwpose_provider)
        self._source_s_det = self.s_det
        self._source_detector_module = self.det
        self._driving_s_det = self._make_fp16_detector() if detector_fp16 else self.s_det

    def _make_fp16_detector(self):
        """New cached ONNX artifact; original ONNX/checkpoints remain untouched."""
        import hashlib
        import onnx
        from onnx import helper, numpy_helper
        from onnxruntime.transformers.float16 import convert_float_to_float16
        from onnxruntime.transformers.onnx_model import OnnxModel
        import onnxruntime as ort
        source = _base.EHMT / 'pretrained/dwpose/yolox_l.onnx'
        target = Path('/dev/shm/ubx858_no_latency_20261102/models/yolox_l_person_fp16.onnx')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            model = convert_float_to_float16(onnx.load(str(source)), keep_io_types=True)
            old_output = model.graph.output[0].name
            constants = (('starts', [0]), ('ends', [6]), ('axes', [2]), ('steps', [1]))
            for name, values in constants:
                model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, np.int64), '_fast_person_' + name))
            model.graph.node.append(helper.make_node('Slice', [old_output] + ['_fast_person_' + n for n, _ in constants],
                                                      ['person_predictions'], name='FastPersonOnlyOutput'))
            model.graph.output[0].name = 'person_predictions'
            model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 6
            OnnxModel(model).topological_sort(is_deterministic=True)
            onnx.checker.check_model(model)
            temporary = target.with_name(f'{target.stem}.{os.getpid()}.onnx')
            onnx.save(model, str(temporary)); temporary.replace(target)
        if self.dwpose_provider != 'CUDAExecutionProvider':
            raise ValueError('FP16 detector requires CUDAExecutionProvider')
        so = ort.SessionOptions(); so.intra_op_num_threads = self.fast_dwpose_threads
        session = ort.InferenceSession(str(target), so, providers=[('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider'])
        if session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('FP16 detector fell back to CPU')
        self.detector_artifact = str(target)
        self.detector_artifact_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
        return session

    def set_source_mode(self):
        self.s_det = self._source_s_det
        self.det = self._source_detector_module

    def set_driving_mode(self, shape=None):
        from types import SimpleNamespace
        self.s_det = self._driving_s_det
        if self.detector_fp16:
            self.det = SimpleNamespace(inference_detector=self._person_detector)
        if self.use_sprint_graph and shape is not None and self._sprint_graph is None:
            self._graph_sprint_forward(torch.zeros(1, 3, 1024, 1024, device=self.dev), np.zeros(5, np.float32), shape)

    def _person_detector(self, session, img):
        d = self._source_detector_module
        x, ratio = d.preprocess(img, (640, 640))
        predictions = d.demo_postprocess(session.run(None, {session.get_inputs()[0].name: x[None]})[0], (640, 640))[0]
        boxes = predictions[:, :4]
        xyxy = np.empty_like(boxes)
        xyxy[:, :2] = boxes[:, :2] - boxes[:, 2:4] / 2
        xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:4] / 2
        xyxy /= ratio
        scores = predictions[:, 4] * predictions[:, 5]
        valid = scores > .1
        if not valid.any():
            return np.empty((0, 4), np.float32)
        xyxy, scores = xyxy[valid], scores[valid]
        keep = d.nms(xyxy, scores, .45)
        return xyxy[keep][scores[keep] > .3]

    @torch.no_grad()
    def _graph_sprint_forward(self, crop_t, cam, shape):
        x = torch.nn.functional.interpolate(crop_t, size=(256, 256), mode='bilinear', antialias=True, align_corners=False)
        # torchvision normalisation creates/checks host constants, which cannot
        # execute during CUDA capture. Keep that identical preprocessing outside.
        x = self.model.prepare_input(x)
        if self._sprint_graph is None and self.sprint_graph_error is None:
            self._sg_x = torch.empty_strided(x.size(), x.stride(), device=x.device, dtype=x.dtype).copy_(x)
            self._sg_cam = torch.as_tensor(cam, device=self.dev).float().reshape(1, 5).clone()
            self._sg_shape = torch.as_tensor(np.asarray(shape, np.float32)[:200], device=self.dev).reshape(1, 200).clone()
            self._sg_cond = {'cam': self._sg_cam, 'shape': self._sg_shape, 'shape_on': torch.ones(1, device=self.dev)}
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3): self._graph_token()
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._sg_token = self._graph_token()
                self._sprint_graph = graph
            except Exception as exc:
                self.sprint_graph_error = repr(exc)
                raise RuntimeError(f'SPRINT graph capture failed: {exc}') from exc
        self._sg_x.copy_(x)
        self._sg_cam.copy_(torch.as_tensor(cam).reshape(1, 5))
        self._sg_shape.copy_(torch.as_tensor(np.asarray(shape, np.float32)[:200]).reshape(1, 200))
        self._sprint_graph.replay()
        # Decode with the original head outside capture: its camera helpers
        # construct CPU constants. This preserves the deployed raw-head maths.
        head = self.model.head
        out = head.decode_token(self._sg_token)
        bp = out['body_param']
        jaw = type(head).forward.__globals__['_rodrigues']
        bp['jaw_pose'] = jaw(out['flame_param']['jaw_params'].float()).to(self._sg_token.dtype)[:, None]
        on = self._sg_cond['shape_on'].to(bp['shape'].dtype)[:, None]
        bp['shape'] = on * self._sg_shape.to(bp['shape'].dtype)[:, :200] + (1 - on) * bp['shape']
        out['orient_logits'] = head.orient_cls(self._sg_token)
        out['global_pose_raw'] = bp['global_pose'][:, 0]
        return out

    def _graph_token(self):
        head = self.model.head
        features = self.model.backbone(self._sg_x)
        context = features.flatten(2).transpose(1, 2)
        context = torch.cat([context, head.context_tokens(features.shape[0], features.device, features.dtype, self._sg_cond)], 1)
        return head.transformer(features.new_zeros(features.shape[0], 1, 1), context=context).squeeze(1)

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
        providers = [provider] if provider == 'CPUExecutionProvider' else [
            (provider, {'use_tf32': 0}), 'CPUExecutionProvider']
        det = ort.InferenceSession(str(_base.EHMT / 'pretrained/dwpose/yolox_l.onnx'), so, providers=providers)
        pose = ort.InferenceSession(str(_base.EHMT / 'pretrained/dwpose/dw-ll_ucoco_384.onnx'), so, providers=providers)
        if any(s.get_providers()[0] != provider for s in (det, pose)):
            raise RuntimeError(f'{provider} fell back; refusing misleading timing')
        if provider == 'CUDAExecutionProvider' and any(
                s.get_provider_options().get(provider, {}).get('use_tf32') != '0' for s in (det, pose)):
            raise RuntimeError('CUDA provider did not confirm use_tf32=0')
        self.s_det, self.s_pose = det, pose
        self.dwpose_provider = provider

    def fit(self, img_rgb, box=None, iters=None, shape_lock=None, **kwargs):
        if iters == 0 and not kwargs:
            return self.drive_raw(img_rgb, box=box, shape_lock=shape_lock)
        previous = self._in_driving_refinement
        self._in_driving_refinement = iters is not None and iters > 0 and not kwargs.get('_warm', False)
        try:
            return super().fit(img_rgb, box=box, iters=iters, shape_lock=shape_lock, **kwargs)
        finally:
            self._in_driving_refinement = previous

    @torch.no_grad()
    def matting(self, crop_t):
        if self._in_driving_refinement and self.driving_matte_size != 1024:
            crop_t = torch.nn.functional.interpolate(crop_t, size=(self.driving_matte_size,) * 2,
                                                     mode='bilinear', antialias=True, align_corners=False)
        return super().matting(crop_t)

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
        out = self._graph_sprint_forward(crop_t, cam, shape_lock) if self.use_sprint_graph and shape_lock is not None else self.sprint(crop_t, cam, shape_lock)
        torch.cuda.synchronize(); T['sprint_and_crop'] = time.perf_counter() - t0 - T['detector']
        bp = out['body_param']
        rotations = torch.cat([bp[k].reshape(-1, 3, 3) for k in (
            'global_pose', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose')])
        aa = roma.rotmat_to_rotvec(rotations).reshape(-1)
        nsh = bp['shape'].shape[-1]
        packed = torch.cat([aa, bp['exp'][0, :50], bp['shape'][0], out['pd_cam'][0, :3, :4].reshape(-1)]).cpu().numpy().astype(np.float32)
        pose = packed[:159].reshape(53, 3)
        RT = packed[209 + nsh:].reshape(3, 4)
        raw = {'global_pose': pose[0], 'body_pose': pose[1:22],
               'left_hand_pose': pose[22:37], 'right_hand_pose': pose[37:52],
               'jaw_pose': pose[52:53], 'eye_pose': np.zeros((2, 3), np.float32),
               'exp': packed[159:209],
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
        shape = packed[209:209 + nsh]
        torch.cuda.synchronize(); T['total'] = time.perf_counter() - t0
        return {'record': rec, 'raw_smplx': raw, 'shape': shape, 'raw_shape': shape,
                'box': box, 'M': M, 'times': T, 'cam_feat': cam}


# Preserve source-writing helpers expected by the existing evaluator.
write_tracked = _base.write_tracked
