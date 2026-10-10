#!/usr/bin/env python
"""Standalone backbone+head latency benchmark (Checkpoint 5, Step 2).

Times the current student backbone and the ViTPose-S candidate backbone, each feeding the same
PEAR head, at batch 1, in fp32 and fp16, using CUDA events. No training code, no dataset, no
teacher model is imported; only `models/backbones/*`, `models/smplx/*` and the head config are
needed. Weights are random by default (--vitpose-weights optionally loads the real pretrained
backbone, e.g. for a VRAM sanity check, but it has no effect on speed).

Run from the PEAR checkout, or pass --pear-root:
    python c5_latency_benchmark.py --pear-root /path/to/third_party/PEAR --device cuda:0
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pear-root", type=Path, default=Path(__file__).resolve().parents[3] / "third_party" / "PEAR",
                   help="path to the PEAR checkout (needs models/, configs/student_l70_v2.yaml, "
                        "assets/SMPLX/smpl_mean_params.npz)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default="current,vitpose", help="comma-separated: current,vitpose")
    p.add_argument("--vitpose-weights", default=None,
                   help="optional: local safetensors path or a HF repo id (e.g. "
                        "usyd-community/vitpose-plus-small) to load real pretrained weights "
                        "instead of random init. Needs `transformers`, `safetensors` and, for a "
                        "repo id, `huggingface_hub` and internet access.")
    return p.parse_args()


def build_current_backbone(pear_root: Path):
    sys.path.insert(0, str(pear_root))
    from models.backbones.student_backbone import PearStudentBackbone
    return PearStudentBackbone(embed_dim=1280, token_dim=512, widths=(96, 192, 384, 512),
                               depths=(2, 3, 6, 3), transformer_depth=4, transformer_heads=8,
                               norm="group", use_pos_embed=True)


def build_vitpose_backbone(pear_root: Path, weights):
    sys.path.insert(0, str(pear_root))
    from models.backbones.vitpose_small_backbone import ViTPoseSBackbone
    path = weights
    if weights and not Path(weights).is_file():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(weights, "model.safetensors")
    return ViTPoseSBackbone(embed_dim=1280, pretrained_path=path)


def build_head(pear_root: Path, batch_size: int):
    sys.path.insert(0, str(pear_root))
    import os
    os.chdir(pear_root)  # SMPLXTransformerDecoderHead loads "assets/SMPLX/smpl_mean_params.npz" (relative path)
    from omegaconf import OmegaConf
    from models.smplx.smplx_head import SMPLXTransformerDecoderHead
    cfg = OmegaConf.load(pear_root / "configs" / "student_l70_v2.yaml")
    return SMPLXTransformerDecoderHead(cfg.HEAD, batch_size)


class BackboneHead(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def time_model(model, input_shape, device, dtype, warmup, iters):
    model = model.to(device=device, dtype=dtype).eval()
    x = torch.randn(*input_shape, device=device, dtype=dtype)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times_ms = []
    for _ in range(iters):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))
        else:
            import time
            t0 = time.perf_counter()
            model(x)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else float("nan")
    times_ms.sort()
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": times_ms[int(0.95 * len(times_ms)) - 1],
        "peak_vram_mb": peak_vram_mb,
    }


def main():
    args = parse_args()
    args.pear_root = args.pear_root.resolve()
    if not (args.pear_root / "models" / "backbones" / "student_backbone.py").is_file():
        raise SystemExit(f"{args.pear_root} does not look like a PEAR checkout "
                          f"(expected models/backbones/student_backbone.py). Pass --pear-root.")
    device = torch.device(args.device)
    models_requested = args.models.split(",")
    input_shape = (args.batch_size, 3, 256, 192)  # PEAR's post-crop input

    builders = {}
    if "current" in models_requested:
        builders["current_backbone"] = lambda: build_current_backbone(args.pear_root)
    if "vitpose" in models_requested:
        builders["vitpose_s_backbone"] = lambda: build_vitpose_backbone(args.pear_root, args.vitpose_weights)

    print(f"device={device} batch_size={args.batch_size} warmup={args.warmup} iters={args.iters} "
          f"input_shape={input_shape}")
    for name, build_backbone in builders.items():
        for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            backbone = build_backbone()
            head = build_head(args.pear_root, args.batch_size)
            model = BackboneHead(backbone, head)
            try:
                stats = time_model(model, input_shape, device, dtype, args.warmup, args.iters)
            except RuntimeError as exc:
                print(f"{name:20s} {dtype_name:4s}  FAILED: {exc}")
                continue
            print(f"{name:20s} {dtype_name:4s}  mean {stats['mean_ms']:7.3f} ms  "
                  f"median {stats['median_ms']:7.3f} ms  p95 {stats['p95_ms']:7.3f} ms  "
                  f"peak VRAM {stats['peak_vram_mb']:8.1f} MB")
            del backbone, head, model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
