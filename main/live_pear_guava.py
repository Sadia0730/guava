#!/usr/bin/env python
"""Drive a cached GUAVA avatar from a live stream using PEAR parameters."""
import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PEAR_ROOT = ROOT / "third_party" / "PEAR"
PEAR_CHECKPOINT = ("BestWJH/PEAR_models", "ehm_model_stage1.pt")
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = (
    ROOT / "assets" / "example" / "tracked_image" / "random_google_pic" / "blue_shirt"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="0", help="webcam index, RTSP URL, HTTP URL, or video")
    parser.add_argument("--source_data_path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model_path", type=Path, default=ROOT / "assets" / "GUAVA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render_size", type=int, choices=(256, 512), default=512)
    parser.add_argument("--window", type=int, default=30, help="rolling FPS window")
    parser.add_argument("--max_frames", type=int, default=0, help="0 runs until q or stream end")
    parser.add_argument("--no_display", action="store_true")
    return parser.parse_args()


def pad_and_resize(image, target_size=256):
    import cv2
    import numpy as np

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


def initialize_pear(args):
    """Load PEAR before GUAVA, then release PEAR's generic module names."""
    import torch
    from huggingface_hub import hf_hub_download
    from pytorch3d.transforms import matrix_to_axis_angle

    original_directory = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from models.pipeline.ehm_pipeline import Ehm_Pipeline
        from utils.general_utils import ConfigDict, add_extra_cfgs

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
        model = model.to(args.device).eval()
        del checkpoint
        with torch.inference_mode():
            model(torch.zeros((1, 3, 256, 256), device=args.device))
        torch.cuda.synchronize()
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
    return model, matrix_to_axis_angle


def infer_pear(model, matrix_to_axis_angle, frame_rgb, device):
    import torch

    preprocess_start = time.perf_counter()
    model_image = pad_and_resize(frame_rgb)
    image_tensor = torch.as_tensor(model_image, device=device)
    image_tensor = (image_tensor.permute(2, 0, 1).float() / 255.0).unsqueeze(0)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    inference_start = torch.cuda.Event(enable_timing=True)
    inference_end = torch.cuda.Event(enable_timing=True)
    inference_start.record()
    with torch.inference_mode():
        outputs = model(image_tensor)
    inference_end.record()

    def axis_angle(rotations):
        return matrix_to_axis_angle(rotations)[0]

    body = outputs["body_param"]
    flame = outputs["flame_param"]
    return {
        "smplx_coeffs": {
            "exp": body["exp"][0],
            "global_pose": axis_angle(body["global_pose"])[0],
            "body_pose": axis_angle(body["body_pose"]),
            "left_hand_pose": axis_angle(body["left_hand_pose"]),
            "right_hand_pose": axis_angle(body["right_hand_pose"]),
        },
        "flame_coeffs": {
            key: flame[key][0]
            for key in (
                "expression_params", "jaw_params", "pose_params",
                "eye_pose_params", "eyelid_params",
            )
        },
        "preprocess_ms": preprocess_ms,
        "pear_timing_events": (inference_start, inference_end),
    }


def initialize_guava(args):
    import copy

    import torch
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
    canonical_camera = load_canonical_render_prams(
        device=args.device,
        image_size=args.render_size,
        tanfov=source_dataset.tanfov,
    )

    identity = {
        "shape": source_info["smplx_coeffs"]["shape"],
        "joints_offset": source_info["smplx_coeffs"]["joints_offset"],
        "head_scale": source_info["smplx_coeffs"]["head_scale"],
        "hand_scale": source_info["smplx_coeffs"]["hand_scale"],
        "flame_shape": source_info["flame_coeffs"]["shape_params"],
    }

    # Warm up deformation, rasterization, and neural refinement once.
    warmup_target = {
        "smplx_coeffs": copy.copy(source_info["smplx_coeffs"]),
        "flame_coeffs": copy.copy(source_info["flame_coeffs"]),
    }
    with torch.inference_mode():
        render_model(avatar(warmup_target), canonical_camera, bg=0.0)
    torch.cuda.synchronize()
    return avatar, render_model, canonical_camera, identity, source_dataset


def prediction_to_target(prediction, identity, device):
    import torch

    def batched(values):
        result = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.unsqueeze(0)
            else:
                result[key] = torch.as_tensor(
                    value, dtype=torch.float32, device=device
                ).unsqueeze(0)
        return result

    smplx = batched(prediction["smplx_coeffs"])
    smplx.update(
        shape=identity["shape"],
        joints_offset=identity["joints_offset"],
        head_scale=identity["head_scale"],
        hand_scale=identity["hand_scale"],
    )
    flame = batched(prediction["flame_coeffs"])
    flame["shape_params"] = identity["flame_shape"]
    return {"smplx_coeffs": smplx, "flame_coeffs": flame}


def mean_fps(samples):
    return 1000.0 / (sum(samples) / len(samples)) if samples else 0.0


def open_capture(value):
    import cv2

    source = int(value) if value.isdigit() else value
    capture = cv2.VideoCapture(source)
    assert capture.isOpened(), f"Could not open stream: {value}"
    return capture


def run_live(args):
    import cv2
    import numpy as np
    import torch

    print("Loading PEAR once...")
    pear_model, matrix_to_axis_angle = initialize_pear(args)
    print("Loading GUAVA and creating the source avatar once...")
    avatar, render_model, camera, identity, source_dataset = initialize_guava(args)
    print("Live pipeline ready. Press q in the display window to stop.")

    capture = open_capture(args.input)
    window = max(1, args.window)
    pear_times = deque(maxlen=window)
    guava_times = deque(maxlen=window)
    pipeline_times = deque(maxlen=window)
    live_times = deque(maxlen=window)
    frame_count = 0
    previous_loop_start = None

    try:
        while args.max_frames == 0 or frame_count < args.max_frames:
            loop_start = time.perf_counter()
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            pipeline_start = time.perf_counter()
            prediction = infer_pear(pear_model, matrix_to_axis_angle, frame_rgb, args.device)
            target = prediction_to_target(prediction, identity, args.device)

            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            render_start.record()
            with torch.inference_mode():
                rendered = render_model(avatar(target), camera, bg=0.0)["renders"][0]
            render_end.record()
            render_bgr = (
                rendered.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
            ).astype(np.uint8)[:, :, ::-1]
            pear_start, pear_end = prediction["pear_timing_events"]
            pear_ms = pear_start.elapsed_time(pear_end)
            guava_ms = render_start.elapsed_time(render_end)
            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0

            pear_times.append(pear_ms)
            guava_times.append(guava_ms)
            pipeline_times.append(pipeline_ms)
            if previous_loop_start is not None:
                live_times.append((loop_start - previous_loop_start) * 1000.0)
            previous_loop_start = loop_start
            frame_count += 1

            pear_fps = mean_fps(pear_times)
            guava_fps = mean_fps(guava_times)
            pipeline_fps = mean_fps(pipeline_times)
            live_fps = mean_fps(live_times)
            if frame_count == 1 or frame_count % window == 0:
                print(
                    f"frames={frame_count} | PEAR={pear_fps:.2f} FPS | "
                    f"GUAVA={guava_fps:.2f} FPS | pipeline={pipeline_fps:.2f} FPS | "
                    f"observed_live={live_fps:.2f} FPS"
                )

            if not args.no_display:
                preview = cv2.resize(frame_bgr, (render_bgr.shape[1], render_bgr.shape[0]))
                combined = np.concatenate((preview, render_bgr), axis=1)
                label = (
                    f"PEAR {pear_fps:.1f} | GUAVA {guava_fps:.1f} | "
                    f"pipeline {pipeline_fps:.1f} FPS"
                )
                cv2.putText(
                    combined, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.imshow("PEAR live motion -> GUAVA avatar", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        source_dataset._lmdb_engine.close()

    print(
        f"Final ({frame_count} frames): PEAR={mean_fps(pear_times):.2f} FPS, "
        f"GUAVA={mean_fps(guava_times):.2f} FPS, "
        f"pipeline={mean_fps(pipeline_times):.2f} FPS, "
        f"observed_live={mean_fps(live_times):.2f} FPS"
    )


if __name__ == "__main__":
    arguments = parse_args()
    run_live(arguments)
