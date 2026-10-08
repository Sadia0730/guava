#!/usr/bin/env python
"""Drive a cached GUAVA avatar from a live stream using PEAR parameters.

Every displayed frame costs one PEAR forward (image -> EHM parameters) and one
GUAVA forward (parameters -> rendered avatar).  Four independent optimizations
turn that serial pair into a real-time loop.  Each sits behind its own flag so
its contribution can be measured in isolation:

  --precision fp16      cast the weights of PEAR's ViT and of GUAVA's neural
                        refiner once, up front.  The Gaussian rasterizer stays
                        fp32 because its CUDA kernels are fp32-only, and so does
                        the mesh deformation, which is memory bound and feeds
                        those kernels.
  --compile_targets     torch.compile the static-shape graphs.  `pear` and
                        `refiner` pay for themselves; `deform` is offered but
                        off by default, see below.
  --pipeline async      run PEAR on frame t on its own CUDA stream while GUAVA
                        renders frame t-1, so throughput approaches
                        max(PEAR, GUAVA) instead of PEAR + GUAVA.
  --pear_stride N       run PEAR on one frame in N.  A One-Euro filter resamples
                        the parameter stream at render rate, which fills the
                        skipped frames and also removes PEAR's per-frame jitter.

Measured on an RTX 3080 Laptop at 512x512, 150 frames from a video file, each
run started below 60 C because this GPU throttles hard enough to invalidate a
careless comparison:

    fp32, serial, no compile                14.9 FPS   (baseline)
    + fp16 weights                          19.3 FPS
    + torch.compile pear,refiner            23.2 FPS
    + async pipeline                        31.5 FPS
    + pear_stride 2                         31.0 FPS

So the baseline command is

    --precision fp32 --compile_targets --pipeline serial --pear_stride 1 --no-smooth

Two findings worth keeping in mind before changing any of this:

  * `deform` is a poor compile target.  roma's quaternion helpers retrace on
    shapes that vary between call sites, and in async mode that is fatal rather
    than merely slow: FX patches nn.Module.__call__ process-wide, so a retrace
    on the render thread swallows the PEAR thread's forward pass.  GUAVA costs
    the same with and without it.
  * --compile_mode reduce-overhead measured no faster than `default` here.
    Inductor declines to capture most of these graphs ("mutated inputs"), so
    the CUDA-graph machinery buys nothing and only adds aliasing hazards.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from roma import rotmat_to_rotvec

ROOT = Path(__file__).resolve().parents[1]
EHM_TRACKER_ROOT = ROOT / "EHM-Tracker"
PEAR_ROOT = ROOT / "third_party" / "PEAR"
PEAR_CHECKPOINT = ("BestWJH/PEAR_models", "pear_model.pt")
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = (
    ROOT / "assets" / "example" / "tracked_image" / "random_google_pic" / "blue_shirt"
)

PEAR_INPUT_SIZE = 256
# PEAR predicts these as rotation matrices; GUAVA consumes them as axis-angle.
ROTATION_KEYS = ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose")
FLAME_KEYS = (
    "expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params",
)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
COMPILE_TARGETS = ("pear", "deform", "refiner")
BODY_JOINT_NAMES = (
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
LOWER_BODY_JOINTS = (0, 1, 3, 4, 6, 7, 9, 10)
SKELETON_EDGES = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20),
    (12, 14), (14, 17), (17, 19), (19, 21),
)


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_axis_angle(matrix):
    return rotmat_to_rotvec(matrix)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live_config", type=Path, help="Live runtime YAML; command-line options override it.")
    parser.add_argument("--input", default="0", help="webcam index, RTSP URL, HTTP URL, or video")
    parser.add_argument("--source_data_path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--capture_source",
        "--capture_source_from_stream",
        dest="capture_source_from_stream",
        action="store_true",
        help="Capture one frame from a stream and EHM-track it as the source avatar.",
    )
    parser.add_argument(
        "--source_capture_input",
        default=None,
        help="Optional webcam/URL/video for the source frame. Defaults to --input.",
    )
    parser.add_argument(
        "--source_capture_output_dir",
        type=Path,
        default=ROOT / "outputs" / "live_source_captures",
    )
    parser.add_argument(
        "--source_tracking_output_dir",
        type=Path,
        default=ROOT / "outputs" / "live_source_tracking",
    )
    parser.add_argument(
        "--source_capture_skip_frames",
        type=int,
        default=0,
        help="Drop this many frames before saving the source frame.",
    )
    parser.add_argument(
        "--source_capture_delay",
        type=float,
        default=0.0,
        help="Seconds to wait after pressing s before capturing the source frame.",
    )
    parser.add_argument(
        "--source_capture_max_attempts",
        type=int,
        default=60,
        help="Maximum failed stream reads while trying to capture the source frame.",
    )
    parser.add_argument(
        "--source_capture_auto",
        action="store_true",
        help="Capture the first readable source frame without opening a preview window.",
    )
    parser.add_argument(
        "--source_tracking_python",
        default=sys.executable,
        help="Python executable/environment used for EHM-Tracker source tracking.",
    )
    parser.add_argument(
        "--force_source_tracking",
        action="store_true",
        help="Re-run source tracking even when cached tracking exists.",
    )
    parser.add_argument("--model_path", type=Path, default=ROOT / "assets" / "GUAVA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render_size", type=int, choices=(256, 512), default=512)
    parser.add_argument(
        "--avatar_view", choices=("full", "original"), default="full",
        help="Full-body render camera, or the original tighter canonical camera.",
    )
    parser.add_argument(
        "--input_crop_scale",
        type=float,
        default=1.0,
        help="Center-crop the live frame by this zoom factor before PEAR. "
             "Use 1.5-2.0 when the webcam shows too much room.",
    )
    parser.add_argument(
        "--input_framing", choices=("whole", "centered"), default="whole",
        help="Centered removes excess landscape width, preserves full height, and previews model input. No detector.",
    )
    parser.add_argument("--window", type=int, default=30, help="rolling FPS window")
    parser.add_argument("--max_frames", type=int, default=0, help="0 runs until q or stream end")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument(
        "--pear_backend",
        choices=("teacher", "student"),
        default="teacher",
        help="Use the original PEAR teacher or a trained PEAR student checkpoint.",
    )
    parser.add_argument(
        "--student_config",
        type=Path,
        default=Path("configs/student_l70.yaml"),
        help="PEAR student config path, relative to third_party/PEAR unless absolute.",
    )
    parser.add_argument(
        "--student_ckpt",
        type=Path,
        default=None,
        help="Checkpoint from train_pear_student_distill.py, required for --pear_backend student.",
    )
    parser.add_argument(
        "--freeze_face",
        action="store_true",
        help="Keep the source avatar's face completely static (equivalent to --face_mode frozen).",
    )
    parser.add_argument(
        "--face_mode",
        choices=("live", "frozen", "source-delta"),
        default="live",
        help="Use live face coefficients, freeze the source face, or apply live motion around the source face.",
    )
    parser.add_argument(
        "--rest_lower_body",
        action="store_true",
        help="Keep hips, knees, ankles, and feet in the source avatar pose.",
    )
    parser.add_argument(
        "--show_pose_skeleton",
        action="store_true",
        help="Display the posed SMPL-X skeleton before Gaussian rendering.",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=1900,
        help="Maximum preview width in pixels; all panels are scaled to fit.",
    )
    parser.add_argument(
        "--record_dir",
        type=Path,
        default=None,
        help="Save synchronized live_rgb.mp4, animated_avatar.mp4, and comparison.mp4 here.",
    )
    parser.add_argument(
        "--record_fps",
        type=float,
        default=30.0,
        help="Playback FPS written to recording files (default: 30).",
    )

    speed = parser.add_argument_group("throughput")
    speed.add_argument(
        "--precision",
        choices=tuple(DTYPES),
        default="fp16",
        help="Weight dtype for PEAR's ViT and the neural refiner (default: fp16).",
    )
    speed.add_argument(
        "--compile_targets",
        nargs="*",
        choices=COMPILE_TARGETS,
        default=["pear", "refiner"],
        metavar="TARGET",
        help=f"Graphs to torch.compile, any of {COMPILE_TARGETS}. Pass with no "
             f"values to disable compilation entirely.",
    )
    speed.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode; reduce-overhead replays the graphs as CUDA graphs.",
    )
    speed.add_argument(
        "--pipeline",
        choices=("async", "serial"),
        default="async",
        help="async overlaps PEAR(t) with GUAVA(t-1) on two CUDA streams.",
    )
    speed.add_argument(
        "--pear_stride",
        type=int,
        default=1,
        help="Run PEAR on one frame in N and resample the parameters in between.",
    )
    speed.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="One-Euro filter over the predicted parameters (default: on).",
    )
    speed.add_argument("--smooth_min_cutoff", type=float, default=2.0, help="One-Euro fc_min [Hz]")
    speed.add_argument("--smooth_beta", type=float, default=0.3, help="One-Euro speed coefficient")
    speed.add_argument("--smooth_d_cutoff", type=float, default=1.0, help="One-Euro fc_d [Hz]")
    speed.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warm-up iterations per model. CUDA graphs need one replay after capture.",
    )
    argv = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--live_config", type=Path)
    selected, _ = selector.parse_known_args(argv)
    config_argv = []
    if selected.live_config is not None:
        import yaml

        try:
            with selected.live_config.open() as handle:
                settings = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as error:
            parser.error(f"Could not read live config: {error}")
        if not isinstance(settings, dict):
            parser.error("Live config must be a mapping of CLI option names to values")
        actions = {action.dest: action for action in parser._actions
                   if action.dest not in ("help", "live_config")}
        for name, value in settings.items():
            if name not in actions:
                parser.error(f"Unknown live config option: {name}")
            action = actions[name]
            if action.nargs == 0:
                if not isinstance(value, bool):
                    parser.error(f"Live config {name} must be true or false")
                parser.set_defaults(**{name: value})
                continue
            values = value if isinstance(value, list) else [value]
            if isinstance(value, list) and action.nargs not in ("*", "+"):
                parser.error(f"Live config {name} must be a scalar")
            if any(item is None or isinstance(item, (dict, list, bool)) for item in values):
                parser.error(f"Invalid live config value for {name}")
            config_argv.append(action.option_strings[0])
            config_argv.extend(str(item) for item in values)
    return parser.parse_args(config_argv + list(argv))


def format_duration(seconds):
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remainder:.1f}s"


def format_timing_summary(timings):
    return " | ".join(
        f"{name}={format_duration(seconds)}"
        for name, seconds in timings.items()
    )


def center_crop_scale(image, scale):
    if scale <= 1.0:
        return image
    height, width = image.shape[:2]
    crop_width = max(1, int(round(width / scale)))
    crop_height = max(1, int(round(height / scale)))
    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    return image[y0:y0 + crop_height, x0:x0 + crop_width]


def pad_and_resize(image, target_size=PEAR_INPUT_SIZE, crop_scale=1.0, framing="whole"):
    if framing == "centered":
        height, width = image.shape[:2]
        crop_width = min(width, height)
        left = (width - crop_width) // 2
        image = image[:, left:left + crop_width]
    image = center_crop_scale(image, crop_scale)
    height, width = image.shape[:2]
    scale = min(target_size / height, target_size / width)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - resized_width) // 2
    y_offset = (target_size - resized_height) // 2
    padded[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return padded




class ReducedPrecision(torch.nn.Module):
    """Run a submodule in fp16/bf16 and hand fp32 back to its fp32-only caller.

    Casting the weights once rather than using torch.autocast: an autocast
    region caches its weight casts but drops them on exit, so a region entered
    per frame re-casts every parameter every frame. Measured on PEAR's ViT that
    is slower than staying in fp32 (43 ms against 30 ms per frame).
    """

    def __init__(self, module, dtype):
        super().__init__()
        self.module = module.to(dtype)
        self.dtype = dtype

    def forward(self, x):
        return self.module(x.to(self.dtype)).float()


def maybe_reduce_precision(module, dtype):
    return module if dtype == torch.float32 else ReducedPrecision(module, dtype)


def maybe_compile(module, target, args):
    if target not in args.compile_targets:
        return module
    return torch.compile(module, mode=args.compile_mode)


class OneEuroFilter:
    """One-Euro filter (Casiez et al., CHI 2012) applied elementwise to a tensor.

    The cutoff frequency rises with the observed speed of the signal, so the
    filter removes PEAR's per-frame jitter while it is still and stops lagging
    once the subject moves.
    """

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = None
        self._timestamp = None

    @staticmethod
    def _alpha(cutoff, dt):
        # tau = 1 / (2 pi fc); alpha = dt / (dt + tau)
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def __call__(self, x, timestamp):
        if self._x is None:
            self._x = x
            self._dx = torch.zeros_like(x)
            self._timestamp = timestamp
            return x

        dt = max(timestamp - self._timestamp, 1e-3)
        self._timestamp = timestamp

        dx = (x - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * self._dx.abs()
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        return self._x


class TargetBuilder:
    """Turn a PEAR parameter record into the GUAVA target batch for one frame.

    Rotations are smoothed in the continuous 6D representation and
    re-orthonormalized on the way out: filtering rotation matrices entrywise
    leaves the rotation manifold, and filtering axis-angle vectors wraps at pi.
    """

    def __init__(self, identity, args):
        self.identity = identity
        self.face_mode = (
            "frozen" if bool(getattr(args, "freeze_face", False))
            else getattr(args, "face_mode", "live")
        )
        self.rest_lower_body = bool(getattr(args, "rest_lower_body", False))
        self.face_baseline = None
        self.filters = None
        if args.smooth:
            self.filters = {
                key: OneEuroFilter(args.smooth_min_cutoff, args.smooth_beta, args.smooth_d_cutoff)
                for key in ROTATION_KEYS + ("exp",) + FLAME_KEYS
            }

    def __call__(self, params, timestamp):
        if self.filters is not None:
            params = {key: self.filters[key](value, timestamp) for key, value in params.items()}

        live_face = {"exp": params["exp"][None]}
        live_face.update({key: params[key][None] for key in FLAME_KEYS})
        if self.face_mode == "live":
            face = live_face
        elif self.face_mode == "frozen":
            face = {"exp": self.identity["source_exp"], **self.identity["source_flame"]}
        else:
            if self.face_baseline is None:
                self.face_baseline = {
                    key: value.detach().clone() for key, value in live_face.items()
                }
            source_face = {"exp": self.identity["source_exp"], **self.identity["source_flame"]}
            face = {
                key: source_face[key] + live_face[key] - self.face_baseline[key]
                for key in live_face
            }

        rotations = {
            key: matrix_to_axis_angle(rotation_6d_to_matrix(params[key])) for key in ROTATION_KEYS
        }
        body_pose = rotations["body_pose"][None]
        if self.rest_lower_body:
            body_pose = body_pose.clone()
            body_pose[:, LOWER_BODY_JOINTS] = self.identity["source_body_pose"][:, LOWER_BODY_JOINTS]
        smplx_coeffs = {
            "exp": face["exp"],
            "global_pose": rotations["global_pose"].reshape(1, 3),
            "body_pose": body_pose,
            "left_hand_pose": rotations["left_hand_pose"][None],
            "right_hand_pose": rotations["right_hand_pose"][None],
            "shape": self.identity["shape"],
            "joints_offset": self.identity["joints_offset"],
            "head_scale": self.identity["head_scale"],
            "hand_scale": self.identity["hand_scale"],
        }
        flame_coeffs = {key: face[key] for key in FLAME_KEYS}
        flame_coeffs["shape_params"] = self.identity["flame_shape"]
        return {"smplx_coeffs": smplx_coeffs, "flame_coeffs": flame_coeffs}


class PearRunner:
    """PEAR inference on a dedicated CUDA stream, on one frame in `stride`."""

    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.dtype = DTYPES[args.precision]
        self.stride = max(1, args.pear_stride)
        self.input_crop_scale = max(1.0, float(args.input_crop_scale))
        self.input_framing = getattr(args, "input_framing", "whole")
        self.last_input_bgr = None
        self.stream = torch.cuda.Stream(device=args.device)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        # Counted here rather than at the render loop: the mailbox drops frames,
        # and a stride-skipped frame is published microseconds after the one
        # that carried a fresh forward pass, so the renderer rarely observes it.
        self.calls = 0
        self.durations = deque(maxlen=max(1, args.window))
        self._params = None
        self._index = 0

    def _infer(self, frame_bgr):
        image = pad_and_resize(frame_bgr, crop_scale=self.input_crop_scale, framing=self.input_framing)
        self.last_input_bgr = image
        # Letterbox first, swap channels second: the two commute, and this way
        # the colour conversion runs on 256x256 instead of the full frame.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.cuda.stream(self.stream):
            tensor = torch.as_tensor(image, device=self.device)
            tensor = (tensor.permute(2, 0, 1).float() / 255.0).unsqueeze(0)
            self.start_event.record()
            outputs = self.model(tensor)
            self.end_event.record()

            body, flame = outputs["body_param"], outputs["flame_param"]
            params = {key: matrix_to_rotation_6d(body[key].float())[0] for key in ROTATION_KEYS}
            params["exp"] = body["exp"][0].float()
            params.update({key: flame[key][0].float() for key in FLAME_KEYS})
            # Own the parameters outright: this frees the model's output graph
            # and survives the buffer reuse that --compile_mode reduce-overhead
            # would otherwise do underneath the renderer.
            params = {key: value.clone() for key, value in params.items()}

        # Blocks this thread only. It keeps PEAR from running ahead of the
        # renderer and makes the event pair readable without a device sync.
        self.stream.synchronize()
        self.calls += 1
        self.durations.append(self.start_event.elapsed_time(self.end_event))
        self._params = params
        return params

    def step(self, frame_bgr):
        """Returns the newest parameters, re-running PEAR on one frame in `stride`."""
        index = self._index
        self._index += 1
        if self._params is not None and index % self.stride != 0:
            return self._params
        return self._infer(frame_bgr)

    def warmup(self, iterations):
        """Compiles PEAR and returns one parameter record to warm GUAVA with."""
        blank = np.zeros((PEAR_INPUT_SIZE, PEAR_INPUT_SIZE, 3), dtype=np.uint8)
        for _ in range(max(1, iterations)):
            params = self._infer(blank)
        self.calls = 0
        self.durations.clear()
        self._params = None
        self._index = 0
        return params


class AvatarRenderer:
    """GUAVA deformation, rasterization and neural refinement, on the default stream.

    diff_gaussian_rasterization_32 launches its kernels without a stream
    argument, i.e. on the legacy default stream, and PyTorch's side streams are
    created non-blocking, so they do not implicitly synchronize with it. GUAVA
    therefore has to stay on the default stream; only PEAR runs on a side one.
    """

    def __init__(self, avatar, render_model, camera):
        self.avatar = avatar
        self.render_model = render_model
        self.camera = camera
        self.last_joints = None
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def render(self, target):
        """Enqueues one frame and returns it as an HWC uint8 BGR tensor."""
        self.start_event.record()
        assets = self.avatar(target)
        self.last_joints = assets.get("joints")
        rendered = self.render_model(assets, self.camera, bg=0.0)["renders"][0]
        # Scale and swap to BGR on the GPU so the readback is one contiguous
        # uint8 copy instead of three float32 planes.
        image = (rendered.clamp(0.0, 1.0).flip(0) * 255.0).permute(1, 2, 0).to(torch.uint8)
        image = image.contiguous()
        self.end_event.record()
        return image

    def warmup(self, target, iterations):
        for _ in range(iterations):
            self.render(target)
        torch.cuda.synchronize()


def draw_pose_skeleton(joints, camera, height, width):
    """Draw GUAVA's pre-rasterization SMPL-X body joints in render-camera space."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        canvas, "PEAR/EHM pose (orange=legs, cyan=upper body)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.46, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if joints is None:
        return canvas

    body = joints[0, :22].detach().float()
    homogeneous = torch.cat((body, torch.ones_like(body[:, :1])), dim=-1)
    clip = homogeneous @ camera["full_proj_transform"][0]
    w = clip[:, 3]
    ndc = clip[:, :2] / w[:, None].clamp_min(1e-6)
    points = torch.stack(
        (((ndc[:, 0] + 1.0) * width - 1.0) * 0.5,
         ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5),
        dim=-1,
    ).cpu().numpy()
    valid = (
        torch.isfinite(ndc).all(dim=-1)
        & (w > 0.0)
        & (ndc.abs() < 1.25).all(dim=-1)
    ).cpu().numpy()

    for start, end in SKELETON_EDGES:
        if valid[start] and valid[end]:
            lower_body = start <= 11 and end <= 11 and start != 3 and end != 3
            color = (0, 165, 255) if lower_body else (255, 190, 0)
            cv2.line(
                canvas, tuple(points[start].astype(int)), tuple(points[end].astype(int)),
                color, 3, cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if valid[index]:
            color = (0, 210, 255) if index in (1, 2, 4, 5, 7, 8, 10, 11) else (255, 255, 255)
            cv2.circle(canvas, tuple(point.astype(int)), 4, color, -1, cv2.LINE_AA)
            if index in (0, 1, 2, 4, 5, 7, 8, 10, 11, 15):
                cv2.putText(
                    canvas, (("pelvis",) + BODY_JOINT_NAMES)[index], tuple((point + 6).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA,
                )
    return canvas


@dataclass
class Frame:
    image_bgr: Any
    raw_image_bgr: Any
    params: Dict[str, torch.Tensor]


class LiveRecorder:
    """Write paired camera/avatar frames without blocking GPU rendering on encode."""

    def __init__(self, output_dir, fps):
        if fps <= 0.0:
            raise ValueError("--record_fps must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self._queue = queue.Queue(maxsize=32)
        self._error = None
        self._writers = None
        self._frames = 0
        self._thread = threading.Thread(target=self._write_loop, name="live-recorder", daemon=True)
        self._thread.start()

    @staticmethod
    def _open_writer(path, width, height, fps):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open MP4 writer: {path}")
        return writer

    @staticmethod
    def _comparison_frame(live_bgr, avatar_bgr):
        avatar_height = avatar_bgr.shape[0]
        live_width = max(1, int(round(live_bgr.shape[1] * avatar_height / live_bgr.shape[0])))
        live_resized = cv2.resize(live_bgr, (live_width, avatar_height), interpolation=cv2.INTER_AREA)
        return np.concatenate((live_resized, avatar_bgr), axis=1)

    def _write_loop(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                live_bgr, avatar_bgr = item
                if self._writers is None:
                    self._writers = (
                        self._open_writer(
                            self.output_dir / "live_rgb.mp4",
                            live_bgr.shape[1], live_bgr.shape[0], self.fps,
                        ),
                        self._open_writer(
                            self.output_dir / "animated_avatar.mp4",
                            avatar_bgr.shape[1], avatar_bgr.shape[0], self.fps,
                        ),
                        self._open_writer(
                            self.output_dir / "comparison.mp4",
                            self._comparison_frame(live_bgr, avatar_bgr).shape[1],
                            avatar_bgr.shape[0], self.fps,
                        ),
                    )
                live_writer, avatar_writer, comparison_writer = self._writers
                live_writer.write(live_bgr)
                avatar_writer.write(avatar_bgr)
                comparison_writer.write(self._comparison_frame(live_bgr, avatar_bgr))
                self._frames += 1
        except BaseException as error:
            self._error = error

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError(f"Live recording failed: {self._error}") from self._error

    def write(self, live_bgr, avatar_bgr):
        self._raise_error()
        if live_bgr is None or avatar_bgr is None:
            raise ValueError("LiveRecorder requires both a camera frame and an avatar frame")
        if live_bgr.ndim != 3 or avatar_bgr.ndim != 3:
            raise ValueError("LiveRecorder expects HWC BGR images")
        # The producer may reuse capture/readback buffers after this call.
        item = (np.ascontiguousarray(live_bgr.copy()), np.ascontiguousarray(avatar_bgr.copy()))
        while True:
            try:
                self._queue.put(item, timeout=0.1)
                break
            except queue.Full:
                self._raise_error()
        self._raise_error()

    def close(self):
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        if self._writers is not None:
            for writer in self._writers:
                writer.release()
        self._raise_error()
        metadata = {
            "fps": self.fps,
            "frames": self._frames,
            "frame_alignment": "frame N in each file comes from the same selected live frame",
            "live_rgb": "live_rgb.mp4",
            "animated_avatar": "animated_avatar.mp4",
            "comparison": "comparison.mp4",
            "note": "Async live mode may skip camera frames when rendering falls behind; paired files remain aligned.",
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


class LatestFrame:
    """One-slot mailbox between the PEAR thread and the render loop.

    Overwriting instead of queueing is the right policy for a live stream: when
    the renderer falls behind it should skip to the newest parameters rather
    than work through a backlog and drift away from real time.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._closed = False

    def put(self, frame):
        with self._condition:
            self._frame = frame
            self._condition.notify()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify()

    def __iter__(self):
        while True:
            with self._condition:
                while self._frame is None and not self._closed:
                    self._condition.wait()
                if self._frame is None:
                    return
                frame, self._frame = self._frame, None
            yield frame


def serial_frames(capture, pear, stop):
    while not stop.is_set():
        ok, frame_bgr = capture.read()
        if not ok:
            return
        params = pear.step(frame_bgr)
        preview = frame_bgr
        if pear.input_framing == "centered":
            # Show the actual 192-column backbone input, including its side boundaries.
            preview = pad_and_resize(pear.last_input_bgr[:, 32:-32])
        yield Frame(preview, frame_bgr, params)


def async_frames(capture, pear, stop, device):
    """Same stream of frames, produced by a thread so PEAR overlaps the renderer."""
    mailbox = LatestFrame()

    def produce():
        torch.cuda.set_device(device)
        try:
            with torch.inference_mode():
                for frame in serial_frames(capture, pear, stop):
                    mailbox.put(frame)
        finally:
            mailbox.close()

    thread = threading.Thread(target=produce, name="pear", daemon=True)
    thread.start()
    try:
        yield from mailbox
    finally:
        stop.set()
        thread.join(timeout=5.0)


def initialize_pear(args):
    """Load, compile and warm up PEAR, then release PEAR's generic module names."""
    original_directory = Path.cwd()
    student_ckpt = args.student_ckpt.resolve() if args.student_ckpt is not None else None
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from utils.general_utils import ConfigDict, add_extra_cfgs

        if args.pear_backend == "teacher":
            from huggingface_hub import hf_hub_download
            from models.pipeline.ehm_pipeline import Ehm_Pipeline

            config = add_extra_cfgs(ConfigDict(model_config_path="configs/infer.yaml"))
            try:
                checkpoint_path = hf_hub_download(
                    *PEAR_CHECKPOINT, repo_type="model", local_files_only=True
                )
            except FileNotFoundError:
                checkpoint_path = hf_hub_download(*PEAR_CHECKPOINT, repo_type="model")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = Ehm_Pipeline(config)
            model.backbone.load_state_dict(checkpoint["backbone"], strict=False)
            model.head.load_state_dict(checkpoint["head"], strict=False)
            del checkpoint
        else:
            from models.pipeline.student_pipeline import PearStudentPipeline

            if student_ckpt is None:
                raise ValueError("--student_ckpt is required with --pear_backend student")
            config_path = args.student_config
            if not config_path.is_absolute():
                config_path = PEAR_ROOT / config_path
            config = add_extra_cfgs(ConfigDict(model_config_path=str(config_path)))
            try:
                checkpoint = torch.load(student_ckpt, map_location="cpu", weights_only=True)
            except pickle.UnpicklingError as exc:
                print(
                    "Safe student checkpoint load failed; retrying trusted local "
                    f"checkpoint with weights_only=False: {exc}"
                )
                checkpoint = torch.load(student_ckpt, map_location="cpu", weights_only=False)
            model = PearStudentPipeline(config)
            model.load_state_dict(checkpoint["student"], strict=True)
            print(f"Loaded PEAR student step {checkpoint.get('step', 'unknown')}: {student_ckpt}")
            del checkpoint
        model = model.to(args.device).eval()

        # The ViT backbone is essentially all of PEAR's compute; the decoder
        # head is one token and stays fp32, which also keeps its fp32 camera
        # constants from mixing dtypes.
        model.backbone = maybe_reduce_precision(model.backbone, DTYPES[args.precision])

        # Compile and warm up here: dynamo traces PEAR's own `models`/`utils`
        # packages, which are only importable while its root is on sys.path.
        pear = PearRunner(maybe_compile(model, "pear", args), args)
        with torch.inference_mode():
            warmup_params = pear.warmup(args.warmup)
    finally:
        os.chdir(original_directory)
        sys.path.remove(str(PEAR_ROOT))

    # Both projects use top-level packages named `models` and `utils`. The
    # instantiated PEAR model keeps references to its classes, while removing
    # these import-cache names lets GUAVA load its own packages next.
    for module_name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if module_file and str(module_file).startswith(str(PEAR_ROOT)):
            if module_name == "models" or module_name.startswith("models."):
                del sys.modules[module_name]
            elif module_name == "utils" or module_name.startswith("utils."):
                del sys.modules[module_name]
    return pear, warmup_params


def initialize_guava(args, warmup_params):
    from dataset import TrackedData_infer, load_canonical_render_prams
    from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
    from utils.general_utils import ConfigDict, add_extra_cfgs, find_pt_file

    config_path = args.model_path / "config.yaml"
    meta_cfg = add_extra_cfgs(ConfigDict(model_config_path=str(config_path)))
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL).to(args.device).eval()
    render_model = GaussianRenderer(meta_cfg.MODEL).to(args.device).eval()

    checkpoint_path = find_pt_file(str(args.model_path / "checkpoints"), "best")
    if checkpoint_path is None:
        checkpoint_path = find_pt_file(str(args.model_path / "checkpoints"), "latest")
    assert checkpoint_path is not None, f"No GUAVA checkpoint found under {args.model_path}"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    infer_model.load_state_dict(checkpoint["model"], strict=False)
    render_model.load_state_dict(checkpoint["render_model"], strict=False)
    del checkpoint

    dataset_config = dict(meta_cfg["DATASET"])
    dataset_config["data_path"] = str(args.source_data_path)
    meta_cfg.update("DATASET", dataset_config)
    source_dataset = TrackedData_infer(
        cfg=meta_cfg, split="test", device=args.device, test_full=True
    )
    source_id = next(iter(source_dataset.videos_info))
    source_info = source_dataset._load_source_info(source_id)

    with torch.inference_mode():
        vertex_gs, uv_gs, _ = infer_model(source_info)
        avatar = Ubody_Gaussian(meta_cfg.MODEL, vertex_gs, uv_gs, pruning=True)
        avatar.init_ehm(infer_model.ehm)
        avatar.eval()
    camera = load_canonical_render_prams(
        device=args.device, image_size=args.render_size, tanfov=source_dataset.tanfov
    )
    if getattr(args, "avatar_view", "full") == "full":
        from utils.graphics_utils import get_full_proj_matrix

        # The original camera's y=0.6, z=22 framing clips standing lower legs.
        world_to_camera = torch.eye(4, device=args.device)
        world_to_camera[2, 3] = 32.0
        view, projection = get_full_proj_matrix(world_to_camera, source_dataset.tanfov)
        camera["world_view_transform"] = view[None]
        camera["full_proj_transform"] = projection[None]
        camera["camera_center"] = torch.linalg.inv(world_to_camera)[None, :3, 3]

    # Only the refiner drops precision: the rasterizer feeding it is an
    # fp32-only CUDA kernel, and the deformation is memory bound.
    render_model.nerual_refiner = maybe_reduce_precision(
        render_model.nerual_refiner, DTYPES[args.precision]
    )
    render_model.nerual_refiner = maybe_compile(render_model.nerual_refiner, "refiner", args)
    avatar.forward = maybe_compile(avatar.forward, "deform", args)

    identity = {
        "shape": source_info["smplx_coeffs"]["shape"],
        "joints_offset": source_info["smplx_coeffs"]["joints_offset"],
        "head_scale": source_info["smplx_coeffs"]["head_scale"],
        "hand_scale": source_info["smplx_coeffs"]["hand_scale"],
        "flame_shape": source_info["flame_coeffs"]["shape_params"],
        "source_exp": source_info["smplx_coeffs"]["exp"],
        "source_body_pose": source_info["smplx_coeffs"]["body_pose"],
        "source_flame": {key: source_info["flame_coeffs"][key] for key in FLAME_KEYS},
    }
    # Warm up on a real PEAR parameter record rather than on the source's own
    # coefficients. The two differ in stride, and a guard failure would put
    # dynamo back into tracing mid-run - which is fatal in async mode, where FX
    # patches nn.Module.__call__ process-wide and would swallow the PEAR
    # thread's forward pass.
    renderer = AvatarRenderer(avatar, render_model, camera)
    with torch.inference_mode():
        warmup_target = TargetBuilder(identity, args)(warmup_params, time.perf_counter())
        renderer.warmup(warmup_target, args.warmup)
    return renderer, identity, source_dataset


def mean(samples):
    return sum(samples) / len(samples) if samples else 0.0


def fps(milliseconds):
    return 1000.0 / milliseconds if milliseconds > 0.0 else 0.0


class RollingStats:
    def __init__(self, window):
        self._window = window
        self._samples = {}

    def add(self, name, value):
        self._samples.setdefault(name, deque(maxlen=self._window)).append(value)

    def mean(self, name):
        return mean(self._samples.get(name, ()))

    def fps(self, name):
        return fps(self.mean(name))


def open_capture(value):
    source = int(value) if value.isdigit() else value
    capture = cv2.VideoCapture(source)
    assert capture.isOpened(), f"Could not open stream: {value}"
    return capture


def add_capture_prompt(frame_bgr, text):
    display = frame_bgr.copy()
    cv2.rectangle(display, (0, 0), (display.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(
        display,
        text,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def read_source_frame_auto(capture, source, args):
    skip_frames = max(0, args.source_capture_skip_frames)
    max_attempts = max(1, args.source_capture_max_attempts)
    failed_reads = 0

    while failed_reads < max_attempts:
        ok, frame_bgr = capture.read()
        if not ok:
            failed_reads += 1
            time.sleep(0.02)
            continue
        if skip_frames > 0:
            skip_frames -= 1
            continue
        return frame_bgr

    raise RuntimeError(
        f"Could not capture a source frame from {source} after "
        f"{max_attempts} failed reads."
    )


def read_source_frame_interactive(capture, source, args):
    window_name = "GUAVA source capture"
    skip_frames = max(0, args.source_capture_skip_frames)
    capture_delay = max(0.0, float(args.source_capture_delay))
    capture_deadline = None
    max_attempts = max(1, args.source_capture_max_attempts)
    failed_reads = 0

    print(
        "Source capture preview opened. Press 's' to start the capture countdown, "
        "'q' or Esc to cancel."
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while failed_reads < max_attempts:
            ok, frame_bgr = capture.read()
            if not ok:
                failed_reads += 1
                time.sleep(0.02)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise RuntimeError("Source capture cancelled.")
                continue

            failed_reads = 0
            if skip_frames > 0:
                skip_frames -= 1
                prompt = f"Warming stream... {skip_frames} frames"
                can_save = False
            elif capture_deadline is not None:
                remaining = capture_deadline - time.monotonic()
                if remaining <= 0.0:
                    return frame_bgr
                prompt = f"Capturing in {math.ceil(remaining)} s | q/Esc to cancel"
                can_save = False
            else:
                prompt = "Press s to start capture countdown | q/Esc to cancel"
                can_save = True

            cv2.imshow(window_name, add_capture_prompt(frame_bgr, prompt))
            key = cv2.waitKey(1) & 0xFF
            if can_save and key == ord("s"):
                if capture_delay <= 0.0:
                    return frame_bgr
                capture_deadline = time.monotonic() + capture_delay
                print(f"Source capture countdown started: {capture_delay:g} seconds")
            if key in (27, ord("q")):
                raise RuntimeError("Source capture cancelled.")
    finally:
        cv2.destroyWindow(window_name)

    raise RuntimeError(
        f"Could not capture a source frame from {source} after "
        f"{max_attempts} failed reads."
    )


def capture_source_frame(args):
    source = args.source_capture_input or args.input
    output_dir = args.source_capture_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capturing source frame from stream: {source}")
    start = time.perf_counter()
    capture = open_capture(source)
    try:
        if args.source_capture_auto or args.no_display:
            if args.no_display and not args.source_capture_auto:
                print("--no_display is set, so source capture is automatic.")
            frame_bgr = read_source_frame_auto(capture, source, args)
        else:
            frame_bgr = read_source_frame_interactive(capture, source, args)
    finally:
        capture.release()

    capture_seconds = time.perf_counter() - start
    digest = hashlib.sha1(frame_bgr.tobytes()).hexdigest()[:10]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"stream_source_{timestamp}_{digest}.jpg"
    if not cv2.imwrite(str(image_path), frame_bgr):
        raise RuntimeError(f"Could not save captured source frame: {image_path}")

    print(f"Saved captured source frame: {image_path}")
    return image_path, capture_seconds


def track_source_image(image_path, args):
    output_dir = args.source_tracking_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tracked_source_path = output_dir / image_path.stem
    optim_tracking_path = tracked_source_path / "optim_tracking_ehm.pkl"
    if optim_tracking_path.exists() and not args.force_source_tracking:
        print(f"Using cached EHM source tracking: {tracked_source_path}")
        return tracked_source_path, 0.0

    command = [
        args.source_tracking_python,
        "-m",
        "src.tracking_single_image",
        "-i",
        str(image_path.resolve()),
        "-o",
        str(output_dir.resolve()),
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "." if not existing_pythonpath else f".{os.pathsep}{existing_pythonpath}"

    print("Tracking captured source frame with EHM-Tracker...")
    start = time.perf_counter()
    subprocess.run(command, cwd=EHM_TRACKER_ROOT, env=env, check=True)
    tracking_seconds = time.perf_counter() - start

    if not optim_tracking_path.exists():
        raise RuntimeError(
            "EHM-Tracker finished but the expected source tracking file was not "
            f"created: {optim_tracking_path}"
        )
    print(f"Tracked source avatar: {tracked_source_path}")
    return tracked_source_path, tracking_seconds


def prepare_stream_source(args):
    image_path, capture_seconds = capture_source_frame(args)
    source_path, tracking_seconds = track_source_image(image_path, args)
    args.source_data_path = source_path
    return {
        "source_capture": capture_seconds,
        "source_tracking": tracking_seconds,
    }


def describe(args):
    targets = "+".join(args.compile_targets) or "off"
    compile_state = f"{targets}" if targets == "off" else f"{targets}/{args.compile_mode}"
    return (
        f"precision={args.precision} compile={compile_state} pipeline={args.pipeline} "
        f"pear_stride={args.pear_stride} smooth={'on' if args.smooth else 'off'} "
        f" avatar_view={args.avatar_view} rest_lower_body={args.rest_lower_body}"
        f" input_framing={args.input_framing}"
    )


def report(frame_count, pear, stats):
    pear_ms = mean(pear.durations)
    summary = (
        f"frames={frame_count} pear_calls={pear.calls} | "
        f"PEAR {fps(pear_ms):.1f} FPS ({pear_ms:.1f} ms) | "
        f"GUAVA {stats.fps('guava'):.1f} FPS ({stats.mean('guava'):.1f} ms) | "
        f"end-to-end {stats.fps('frame'):.1f} FPS | wait {stats.mean('wait'):.1f} ms"
    )
    return summary


def run_live(args):
    # Static shapes throughout: let cuDNN pick algorithms once, and use the
    # tensor cores for the matmuls that stay in fp32.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    setup_times = {}
    setup_start = time.perf_counter()
    if args.capture_source_from_stream:
        setup_times.update(prepare_stream_source(args))

    print(f"Configuration: {describe(args)}")
    print("Loading PEAR once...")
    pear_start = time.perf_counter()
    pear, warmup_params = initialize_pear(args)
    setup_times["pear_init"] = time.perf_counter() - pear_start


    print("Loading GUAVA and creating the source avatar once...")
    guava_start = time.perf_counter()
    renderer, identity, source_dataset = initialize_guava(args, warmup_params)
    setup_times["guava_avatar_init"] = time.perf_counter() - guava_start
    setup_times["setup_total"] = time.perf_counter() - setup_start
    print(f"Setup time: {format_timing_summary(setup_times)}")
    print("Live pipeline ready. Press q in the display window to stop.")

    build_target = TargetBuilder(identity, args)
    stats = RollingStats(max(1, args.window))
    capture = open_capture(args.input)
    stop = threading.Event()
    if args.pipeline == "async":
        frames = async_frames(capture, pear, stop, args.device)
    else:
        frames = serial_frames(capture, pear, stop)
    recorder = LiveRecorder(args.record_dir, args.record_fps) if args.record_dir is not None else None
    if not args.no_display:
        cv2.namedWindow("PEAR live motion -> GUAVA avatar", cv2.WINDOW_NORMAL)

    frame_count = 0
    previous_frame_start = None

    try:
        with torch.inference_mode():
            while args.max_frames == 0 or frame_count < args.max_frames:
                wait_start = time.perf_counter()
                frame = next(frames, None)
                frame_start = time.perf_counter()
                if frame is None:
                    break

                # The parameters were allocated on PEAR's stream; hold its
                # allocator pool off until this stream is done reading them.
                render_stream = torch.cuda.current_stream()
                for tensor in frame.params.values():
                    tensor.record_stream(render_stream)

                target = build_target(frame.params, frame_start)
                image = renderer.render(target)
                if args.no_display and recorder is None:
                    render_bgr = None
                    renderer.end_event.synchronize()
                else:
                    render_bgr = image.cpu().numpy()

                if recorder is not None:
                    recorder.write(frame.raw_image_bgr, render_bgr)

                stats.add("wait", (frame_start - wait_start) * 1000.0)
                if pear.durations:
                    stats.add("pear", pear.durations[-1])
                stats.add("guava", renderer.start_event.elapsed_time(renderer.end_event))
                if previous_frame_start is not None:
                    stats.add("frame", (frame_start - previous_frame_start) * 1000.0)
                previous_frame_start = frame_start
                frame_count += 1

                if frame_count == 1 or frame_count % args.window == 0:
                    print(report(frame_count, pear, stats))

                if not args.no_display:
                    preview = cv2.resize(
                        frame.image_bgr, (render_bgr.shape[1], render_bgr.shape[0])
                    )
                    panels = [preview, render_bgr]
                    if args.show_pose_skeleton:
                        panels.append(
                            draw_pose_skeleton(
                                renderer.last_joints,
                                renderer.camera,
                                render_bgr.shape[0],
                                render_bgr.shape[1],
                            )
                        )
                    combined = np.concatenate(panels, axis=1)
                    label = (
                        f"PEAR {stats.fps('pear'):.1f} | GUAVA {stats.fps('guava'):.1f} | "
                        f"live {stats.fps('frame'):.1f} FPS"
                    )
                    cv2.putText(
                        combined, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    display = combined
                    max_width = max(1, int(args.display_width))
                    if display.shape[1] > max_width:
                        scale = max_width / display.shape[1]
                        display = cv2.resize(
                            display,
                            (max_width, max(1, int(round(display.shape[0] * scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    cv2.imshow("PEAR live motion -> GUAVA avatar", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        stop.set()
        frames.close()
        capture.release()
        if recorder is not None:
            recorder.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        source_dataset._lmdb_engine.close()

    print(f"Final [{describe(args)}]: {report(frame_count, pear, stats)}")


if __name__ == "__main__":
    run_live(parse_args())
