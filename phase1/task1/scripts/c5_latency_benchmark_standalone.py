#!/usr/bin/env python
"""Standalone backbone+head latency benchmark (Checkpoint 5, Step 2) -- single file, no other
files needed from this repository or from PEAR.

Only dependency: torch (and torchvision is NOT needed; this script never imports it). No other
package, no network access, no licensed asset (SMPL-X / FLAME / MANO) and no checkpoint download
is used or needed: every weight is randomly initialised, which is correct for a timing script
since none of this depends on the actual weight values.

Contains, inlined verbatim or re-derived to be architecture-for-architecture identical to the
training code (see the docstring of each section for its source):
  1. `PearStudentBackbone`       -- the current student backbone (24.52 M parameters).
  2. `ViTPoseSBackbone`          -- a pure-torch re-implementation of Hugging Face
                                    `transformers.VitPoseBackbone` (the "+" / mixture-of-experts
                                    variant used for the `c-vitpose` candidate), plus the same
                                    random 1x1 projection to 1280 channels used there
                                    (30.52 M + 0.49 M = 31.01 M parameters).
  3. `SMPLXTransformerDecoderHead` -- the PEAR head (40.51 M parameters), with the SMPL-X mean-pose
     initialisation (normally read from `assets/SMPLX/smpl_mean_params.npz`, a file derived from
     the licensed SMPL-X release) replaced by zero buffers of the same shape. This changes the
     *initial values* of a few bias-like buffers only, not the architecture, parameter count, or
     the cost of any forward pass, so it has no effect on latency.

Usage (from any directory, needs only this one file):
    python c5_latency_benchmark_standalone.py --device cuda:0
    python c5_latency_benchmark_standalone.py --device cpu          # no GPU available

Prints, for each of {current, vitpose} x {fp32, fp16}: mean/median/p95 latency in ms over 500
timed iterations (after 50 untimed warm-up iterations), and peak VRAM in MB (CUDA only). Also
prints each component's parameter count so you can confirm the architecture matches the training
models (expected: current backbone 24.52 M, head 40.51 M, ViTPose-S backbone+projection 31.01 M).
"""
from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================================
# 1. Current student backbone. Verbatim copy of
#    third_party/PEAR/models/backbones/student_backbone.py (pure torch; nothing to change).
# =============================================================================================


def make_norm(channels: int, norm: str = "batch") -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        groups = 32
        while groups > 1 and channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm: {norm}")


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, norm="batch"):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            make_norm(out_channels, norm),
            nn.SiLU(inplace=True),
        )


class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, norm="batch"):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size=1, norm=norm),
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, stride=stride,
                        groups=hidden_channels, norm=norm),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            make_norm(out_channels, norm),
        )

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            y = y + x
        return y


class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=3.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed_tokens = self.norm1(tokens)
        attended, _ = self.attn(normed_tokens, normed_tokens, normed_tokens)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class PearStudentBackbone(nn.Module):
    def __init__(self, embed_dim=384, widths=(48, 96, 192, 384), depths=(2, 3, 4, 2),
                 transformer_depth=2, transformer_heads=6, token_dim=None, norm="batch",
                 use_pos_embed=False, pos_embed_grid=(16, 12)):
        super().__init__()
        token_dim = embed_dim if token_dim is None else int(token_dim)
        if token_dim % transformer_heads != 0:
            raise ValueError("token_dim must be divisible by transformer_heads")
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        self.stem = ConvNormAct(3, widths[0], kernel_size=3, stride=2, norm=norm)
        self.stage1 = self._make_stage(widths[0], widths[0], depths[0], stride=1, norm=norm)
        self.stage2 = self._make_stage(widths[0], widths[1], depths[1], stride=2, norm=norm)
        self.stage3 = self._make_stage(widths[1], widths[2], depths[2], stride=2, norm=norm)
        self.stage4 = self._make_stage(widths[2], widths[3], depths[3], stride=2, norm=norm)
        self.proj = ConvNormAct(widths[3], token_dim, kernel_size=1, norm=norm)
        self.pos_embed = None
        if use_pos_embed:
            height, width = pos_embed_grid
            self.pos_embed = nn.Parameter(torch.zeros(1, token_dim, height, width))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = nn.Sequential(
            *[SpatialTransformerBlock(token_dim, transformer_heads) for _ in range(transformer_depth)]
        )
        self.expand = nn.Identity() if token_dim == embed_dim else nn.Conv2d(token_dim, embed_dim, 1)
        self.out_norm = make_norm(embed_dim, norm)

    @staticmethod
    def _make_stage(in_channels, out_channels, depth, stride, norm="batch"):
        blocks = [DepthwiseBlock(in_channels, out_channels, stride=stride, norm=norm)]
        blocks.extend(DepthwiseBlock(out_channels, out_channels, norm=norm) for _ in range(depth - 1))
        return nn.Sequential(*blocks)

    def forward(self, x, return_stages=False):
        stages = {}
        x = self.stem(x)
        stages["stem"] = x
        for name in ("stage1", "stage2", "stage3", "stage4"):
            x = getattr(self, name)(x)
            stages[name] = x
        x = self.proj(x)
        if self.pos_embed is not None:
            if self.pos_embed.shape[-2:] != x.shape[-2:]:
                pos = F.interpolate(self.pos_embed, size=x.shape[-2:], mode="bilinear", align_corners=False)
            else:
                pos = self.pos_embed
            x = x + pos
        stages["proj"] = x
        x = self.transformer(x)
        stages["transformer"] = x
        x = self.out_norm(self.expand(x))
        stages["out"] = x
        if return_stages:
            return x, stages
        return x


# =============================================================================================
# 2. ViTPose-S backbone. Pure-torch re-implementation of Hugging Face's
#    `transformers.models.vitpose_backbone.modeling_vitpose_backbone.VitPoseBackbone` (Apache-2.0,
#    (c) 2024 University of Sydney and the HuggingFace team), configured exactly as
#    `usyd-community/vitpose-plus-small` (hidden_size=384, 12 layers, 12 heads, mlp_ratio=4,
#    patch 16x16, image 256x192, 6-expert mixture-of-experts MLP with 96 "part" channels of 384).
#    Reproduced here (not imported from `transformers`) so this file needs no extra dependency
#    and never downloads the real pretrained weights -- random init only, which does not change
#    latency. Includes the same per-expert loop as the original (computing every expert's linear
#    layer for every token and masking afterwards), since that is what the real model costs.
# =============================================================================================


class ViTPoseBackbonePatchEmbeddings(nn.Module):
    def __init__(self, hidden_size=384, patch_size=16, num_channels=3):
        super().__init__()
        # padding=2 is a deliberate ViTPose quirk (not standard ViT): with image (256, 192) and
        # patch 16 it gives exactly 16x12 patches, like the standard ViT would with padding=0.
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size,
                                     stride=patch_size, padding=2)

    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)  # (B, num_patches, hidden)


class ViTPoseBackboneEmbeddings(nn.Module):
    def __init__(self, hidden_size=384, num_patches=192):
        super().__init__()
        self.patch_embeddings = ViTPoseBackbonePatchEmbeddings(hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))

    def forward(self, x):
        x = self.patch_embeddings(x)
        return x + self.position_embeddings[:, 1:] + self.position_embeddings[:, :1]


class ViTPoseBackboneSelfAttention(nn.Module):
    def __init__(self, hidden_size=384, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        b, n, c = x.shape
        shape = (b, n, self.num_heads, self.head_dim)
        q = self.query(x).view(*shape).transpose(1, 2)
        k = self.key(x).view(*shape).transpose(1, 2)
        v = self.value(x).view(*shape).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.output(out)


class ViTPoseBackboneMoeMLP(nn.Module):
    def __init__(self, hidden_size=384, mlp_ratio=4, part_features=96, num_experts=6):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.part_features = part_features
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, hidden_size - part_features)
        self.experts = nn.ModuleList([nn.Linear(hidden_features, part_features) for _ in range(num_experts)])

    def forward(self, x, indices):
        hidden = self.act(self.fc1(x))
        shared = self.fc2(hidden)
        expert_out = torch.zeros_like(x[:, :, -self.part_features:])
        indices = indices.view(-1, 1, 1)
        for i, expert in enumerate(self.experts):
            expert_out = expert_out + expert(hidden) * (indices == i)
        return torch.cat([shared, expert_out], dim=-1)


class ViTPoseBackboneLayer(nn.Module):
    def __init__(self, hidden_size=384, num_heads=12, mlp_ratio=4, part_features=96, num_experts=6):
        super().__init__()
        self.attention = ViTPoseBackboneSelfAttention(hidden_size, num_heads)
        self.mlp = ViTPoseBackboneMoeMLP(hidden_size, mlp_ratio, part_features, num_experts)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, dataset_index):
        x = self.attention(self.layernorm_before(x)) + x
        x = self.mlp(self.layernorm_after(x), dataset_index) + x
        return x


class ViTPoseBackboneViT(nn.Module):
    """The ViT-S encoder itself (matches `transformers.VitPoseBackbone`, before our own
    projection to 1280 channels is applied)."""

    def __init__(self, hidden_size=384, num_layers=12, num_heads=12, mlp_ratio=4,
                 image_size=(256, 192), patch_size=16, part_features=96, num_experts=6):
        super().__init__()
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.embeddings = ViTPoseBackboneEmbeddings(hidden_size, num_patches)
        self.layer = nn.ModuleList([
            ViTPoseBackboneLayer(hidden_size, num_heads, mlp_ratio, part_features, num_experts)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.grid = (image_size[0] // patch_size, image_size[1] // patch_size)

    def forward(self, x, dataset_index):
        x = self.embeddings(x)
        for layer in self.layer:
            x = layer(x, dataset_index)
        return self.layernorm(x)  # (B, num_patches, hidden_size)


class ViTPoseSBackbone(nn.Module):
    """ViT-S encoder + a random 1x1 projection to 1280 channels, exactly as
    `third_party/PEAR/models/backbones/vitpose_small_backbone.py` does for the `c-vitpose`
    candidate, so that it attaches to the (unchanged) PEAR head. Random init only."""

    def __init__(self, embed_dim=1280, dataset_index=0):
        super().__init__()
        self.vit = ViTPoseBackboneViT()
        self.dataset_index = dataset_index
        self.grid = self.vit.grid
        self.expand = nn.Conv2d(384, embed_dim, 1)

    def forward(self, x):
        idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device) + self.dataset_index
        tokens = self.vit(x, idx)                                      # (B, 192, 384)
        h, w = self.grid
        spatial = tokens.transpose(1, 2).reshape(x.shape[0], 384, h, w)  # (B, 384, 16, 12)
        return self.expand(spatial)


# =============================================================================================
# 3. PEAR SMPL-X transformer decoder head. Re-derived from
#    third_party/PEAR/models/smplx/smplx_head.py and models/smplx/pose_transformer.py, with:
#      - the SMPL-X mean-pose buffers (normally `assets/SMPLX/smpl_mean_params.npz`, derived from
#        the licensed SMPL-X release) replaced by zero buffers of the same shape -- this changes
#        a few additive constants only, not the architecture or the cost of any forward pass;
#      - `einops.rearrange` replaced by equivalent `reshape`/`transpose` calls;
#      - the dead `norm="ada"` / frequency-embedding code paths dropped (unreachable here: the
#        head always uses `norm="layer"`), which removes the `omegaconf` and `einops` imports.
#    Architecture (num_tokens=1, token_dim=1, dim=1024, depth=6, heads=8, dim_head=64,
#    mlp_dim=1024, context_dim=1280, norm="layer") is exactly `configs/student_l70_v2.yaml`'s
#    HEAD section, hardcoded below rather than read from that file.
# =============================================================================================


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim))

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim))

    def forward(self, x, context):
        b, n, _ = x.shape
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q = q.view(b, n, self.heads, -1).transpose(1, 2)
        k = k.view(b, context.shape[1], self.heads, -1).transpose(1, 2)
        v = v.view(b, context.shape[1], self.heads, -1).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, context_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Sequential(nn.LayerNorm(dim), SelfAttention(dim, heads, dim_head)),
                nn.LayerNorm(dim), CrossAttention(dim, context_dim, heads, dim_head),
                nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, mlp_dim)),
            ]))

    def forward(self, x, context):
        for sa, norm_ca, ca, ff in self.layers:
            x = sa(x) + x
            x = ca(norm_ca(x), context) + x
            x = ff(x) + x
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens, token_dim, dim, depth, heads, dim_head, mlp_dim, context_dim):
        super().__init__()
        self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.transformer = TransformerCrossAttn(dim, depth, heads, dim_head, mlp_dim, context_dim)

    def forward(self, token, context):
        x = self.to_token_embedding(token)
        x = x + self.pos_embedding[:, :x.shape[1]]
        return self.transformer(x, context)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """(B, N, 6) -> (B, N, 3, 3), Zhou et al. CVPR 2019."""
    b, n = x.shape[:2]
    x = x.view(b, n, 2, 3)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _projection_matrix(focal: float, z_near=0.01, z_far=100.0) -> torch.Tensor:
    tan_half_fov = 1.0 / focal
    right = tan_half_fov * z_near
    mat = torch.zeros(4, 4)
    mat[0, 0] = z_near / right
    mat[1, 1] = z_near / right
    mat[3, 2] = 1.0
    mat[2, 2] = z_far / (z_far - z_near)
    mat[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return mat


class SMPLXTransformerDecoderHead(nn.Module):
    def __init__(self, batch_size: int, dim: int = 1024, depth: int = 6, heads: int = 8,
                 dim_head: int = 64, mlp_dim: int = 1024, context_dim: int = 1280):
        super().__init__()
        self.transformer = TransformerDecoder(num_tokens=1, token_dim=1, dim=dim, depth=depth,
                                              heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                              context_dim=context_dim)
        self.smplx_poses_decoder = nn.Linear(dim, 312)       # root + body(21) + 2x15 hands, 6D each
        self.smplx_scale_decoder = nn.Linear(dim, 6)
        self.smplx_shape_decoder = nn.Linear(dim, 200)
        self.smplx_expression_decoder = nn.Linear(dim, 50)
        self.smplx_joint_decoder = nn.Linear(dim, 165)       # built but unused downstream, as in PEAR
        self.flame_poses_decoder = nn.Linear(dim, 14)
        self.flame_shape_decoder = nn.Linear(dim, 300)
        self.flame_expression_decoder = nn.Linear(dim, 50)
        self.cam_decoder = nn.Linear(dim, 3)
        self.register_buffer("init_body_pose", torch.zeros(1, 318))  # normally the SMPL-X mean pose
        self.register_buffer("proj_mat", _projection_matrix(focal=24.0))
        self.batch_size = batch_size

    def forward(self, x):
        # x: (B, C, H, W) backbone feature map -> (B, H*W, C) token-first, as PEAR's head expects.
        b, c, h, w = x.shape
        context = x.flatten(2).transpose(1, 2)
        token = context.new_zeros(b, 1, 1)
        token_out = self.transformer(token, context).squeeze(1)

        flame_pose = self.flame_poses_decoder(token_out)
        flame_param = {
            "eye_pose_params": flame_pose[:, :6], "pose_params": flame_pose[:, 6:9],
            "jaw_params": flame_pose[:, 9:12], "eyelid_params": flame_pose[:, 12:14],
            "expression_params": self.flame_expression_decoder(token_out),
            "shape_params": self.flame_shape_decoder(token_out),
        }

        smplx_pose = self.smplx_poses_decoder(token_out)
        smplx_pose = smplx_pose + F.pad(self.init_body_pose, (0, smplx_pose.shape[1] - 318))
        body_param = {
            "global_pose": rot6d_to_rotmat(smplx_pose[:, :6].reshape(-1, 1, 6)),
            "body_pose": rot6d_to_rotmat(smplx_pose[:, 6:132].reshape(-1, 21, 6)),
            "left_hand_pose": rot6d_to_rotmat(smplx_pose[:, 132:222].reshape(-1, 15, 6)),
            "right_hand_pose": rot6d_to_rotmat(smplx_pose[:, 222:312].reshape(-1, 15, 6)),
        }
        smplx_scale = self.smplx_scale_decoder(token_out)
        body_param["hand_scale"], body_param["head_scale"] = smplx_scale[:, :3], smplx_scale[:, 3:]
        body_param["exp"] = self.smplx_expression_decoder(token_out)
        body_param["shape"] = self.smplx_shape_decoder(token_out)
        body_param["joints_offset"] = self.smplx_joint_decoder(token_out).reshape(-1, 55, 3)

        bias = torch.tensor([0.0, 0.0, 1.5], device=token_out.device, dtype=token_out.dtype)
        pd_cam = self.cam_decoder(token_out) + bias
        pd_cam = torch.cat([pd_cam[:, :2], 24.0 / (pd_cam[:, 2:] + 1e-9)], dim=1)
        rt = torch.eye(4, device=pd_cam.device, dtype=pd_cam.dtype)[None].repeat(b, 1, 1)
        rt[:, 0, 0], rt[:, 1, 1] = -1.0, -1.0
        rt[:, :3, 3] = pd_cam
        full_proj = self.proj_mat.to(device=pd_cam.device, dtype=pd_cam.dtype)[None].repeat(b, 1, 1) @ rt

        return {"pd_cam": rt, "body_param": body_param, "flame_param": flame_param, "full_proj": full_proj}


# =============================================================================================
# Timing harness.
# =============================================================================================

REFERENCE_PARAMS_M = {
    "current_backbone": 24.520032,
    "head": 40.513612,
    "vitpose_backbone_and_projection": 31.009664,
}


class BackboneHead(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone(x))


def count_params(module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


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
    return {"mean_ms": statistics.mean(times_ms), "median_ms": statistics.median(times_ms),
            "p95_ms": times_ms[max(0, math.ceil(0.95 * len(times_ms)) - 1)], "peak_vram_mb": peak_vram_mb}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--models", default="current,vitpose", help="comma-separated: current,vitpose")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    input_shape = (args.batch_size, 3, 256, 192)

    current_backbone = lambda: PearStudentBackbone(  # noqa: E731
        embed_dim=1280, token_dim=512, widths=(96, 192, 384, 512), depths=(2, 3, 6, 3),
        transformer_depth=4, transformer_heads=8, norm="group", use_pos_embed=True)
    vitpose_backbone = lambda: ViTPoseSBackbone(embed_dim=1280)  # noqa: E731

    print("--- parameter counts (M), vs the training models' measured values ---")
    for name, build in (("current_backbone", current_backbone), ("vitpose_backbone_and_projection", vitpose_backbone)):
        n = count_params(build())
        ref = REFERENCE_PARAMS_M[name]
        status = "OK" if abs(n - ref) < 1e-3 else f"MISMATCH (expected {ref:.6f})"
        print(f"  {name:32s} {n:.6f} M  [{status}]")
    n = count_params(SMPLXTransformerDecoderHead(args.batch_size))
    ref = REFERENCE_PARAMS_M["head"]
    status = "OK" if abs(n - ref) < 1e-3 else f"MISMATCH (expected {ref:.6f})"
    print(f"  {'head':32s} {n:.6f} M  [{status}]")

    print(f"\ndevice={device} batch_size={args.batch_size} warmup={args.warmup} iters={args.iters} "
          f"input_shape={input_shape}")
    builders = {}
    if "current" in args.models.split(","):
        builders["current_backbone"] = current_backbone
    if "vitpose" in args.models.split(","):
        builders["vitpose_s_backbone"] = vitpose_backbone
    for name, build_backbone in builders.items():
        for dtype_name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            model = BackboneHead(build_backbone(), SMPLXTransformerDecoderHead(args.batch_size))
            try:
                stats = time_model(model, input_shape, device, dtype, args.warmup, args.iters)
            except RuntimeError as exc:
                print(f"{name:20s} {dtype_name:4s}  FAILED: {exc}")
                continue
            print(f"{name:20s} {dtype_name:4s}  mean {stats['mean_ms']:7.3f} ms  "
                  f"median {stats['median_ms']:7.3f} ms  p95 {stats['p95_ms']:7.3f} ms  "
                  f"peak VRAM {stats['peak_vram_mb']:8.1f} MB")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
