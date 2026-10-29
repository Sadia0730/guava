"""GUAVA-FB pilot C: normal supervision after VCR-GauS (Chen et al., NeurIPS 2024, "View Consistent Depth-Normal Regularizer
for Gaussian Surface Reconstruction"). Pseudo normals from Sapiens-normal-1B (Meta, CC-BY-NC 4.0), run frozen on the
target image (training only). Gaussian normal = shortest scale axis (NeuSG / VCR-GauS Eq. 5); rendered normal map (Eq. 9);
depth map rendered from Gaussian centre depths (VCR-GauS uses the ray-plane intersection depth via a custom CUDA kernel:
not reproduced here); D-Normal from the depth map's finite differences (Eq. 11); confidence w (Eq. 13).
All camera-frame vectors are OpenCV (x right, y down, z forward), as GUAVA's w2c / rasterizer."""
import torch
import torch.nn.functional as Fn

SAPIENS_MEAN = torch.tensor([123.675, 116.28, 103.53]) / 255.0
SAPIENS_STD = torch.tensor([58.395, 57.12, 57.375]) / 255.0


class SapiensNormal:
    """frozen Sapiens normal estimator. Square image -> resized to 768, zero-padded (after normalisation) to 1024 x 768
    (no aspect distortion) -> normals cropped back to the input size. `sign` maps Sapiens' axes to OpenCV camera axes
    (calibrated once with normal_check.py against SMPL-X mesh normals)."""

    def __init__(self, path, device, sign=(1.0, 1.0, 1.0), dtype=torch.bfloat16):
        self.m = torch.jit.load(path, map_location="cpu").eval().to(device).to(dtype)
        self.dev, self.dtype = device, dtype
        self.sign = torch.tensor(sign, device=device).view(1, 3, 1, 1)
        self.mean = SAPIENS_MEAN.to(device).view(1, 3, 1, 1); self.std = SAPIENS_STD.to(device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img, chunk=3):
        """img: B x 3 x S x S in [0, 1] (RGB). Returns unit normals B x 3 x S x S."""
        S = img.shape[-1]; out = []
        for x in img.split(chunk):
            x = Fn.interpolate(x.float(), (768, 768), mode="bilinear", antialias=True, align_corners=False)
            x = Fn.pad((x - self.mean) / self.std, (0, 0, 128, 128))
            y = self.m(x.to(self.dtype)).float()
            y = Fn.interpolate(y, (1024, 768), mode="bilinear", align_corners=False)[:, :, 128:896]
            out.append(Fn.interpolate(y, (S, S), mode="bilinear", antialias=True, align_corners=False))
        return Fn.normalize(torch.cat(out) * self.sign, dim=1)


def quat_wxyz_to_rotmat(q):
    q = Fn.normalize(q, dim=-1); w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def gaussian_normals_cam(assets, w2c):
    """per-Gaussian unit normal (axis of the smallest scale, 3DGS covariance R S S^T R^T -> column of R) and centre, in the
    camera frame; normals oriented towards the camera. Returns (n_c, p_c), each B x N x 3."""
    R = quat_wxyz_to_rotmat(assets["rotation"]); k = assets["scaling"].argmin(-1)
    n = torch.gather(R, 3, k[..., None, None].expand(*k.shape, 3, 1))[..., 0]
    p = assets["xyz"]; Rc, tc = w2c[:, None, :3, :3], w2c[:, None, :3, 3]
    p_c = (Rc @ p[..., None])[..., 0] + tc; n_c = (Rc @ n[..., None])[..., 0]
    n_c = torch.where(((n_c * p_c).sum(-1, keepdim=True) > 0), -n_c, n_c)
    return n_c, p_c


def render_geometry(ren, assets, cam_params, w2c):
    """one extra 32-channel rasterisation: alpha-normalised normal map N^ (B x 3 x S x S, unit), depth map D^ (B x 1 x S x S)
    and alpha (B x 1 x S x S). Training / evaluation only."""
    n_c, p_c = gaussian_normals_cam(assets, w2c)
    f = torch.zeros_like(assets["features_color"]); f[..., 0:3] = n_c; f[..., 3] = p_c[..., 2]; f[..., 4] = 1.0
    g = dict(assets); g["features_color"] = f
    raw = ren.forward_raw(g, cam_params, bg=0.0)["raw_features"]
    alpha = raw[:, 4:5]; a = alpha.clamp(min=1e-4)
    return Fn.normalize(raw[:, 0:3] / a, dim=1), raw[:, 3:4] / a, alpha


def depth_to_normal(D, itf):
    """D-Normal (Eq. 11): back-project the depth map with GUAVA's projection (ndc = itf * x / z, pixel centres at
    (i + 0.5) / S * 2 - 1), cross product of the vertical and horizontal central differences, oriented to the camera.
    Returns B x 3 x S x S (border pixels zero)."""
    B, _, S, _ = D.shape
    u = (torch.arange(S, device=D.device, dtype=D.dtype) + 0.5) / S * 2 - 1
    itf = itf.view(B, 1, 1, 1)
    X = torch.cat([u.view(1, 1, 1, S).expand(B, 1, S, S) * D / itf, u.view(1, 1, S, 1).expand(B, 1, S, S) * D / itf, D], 1)
    dh = torch.zeros_like(X); dv = torch.zeros_like(X)
    dh[..., :, 1:-1] = X[..., :, 2:] - X[..., :, :-2]; dv[..., 1:-1, :] = X[..., 2:, :] - X[..., :-2, :]
    n = Fn.normalize(torch.linalg.cross(dv, dh, dim=1), dim=1)
    return torch.where((n * X).sum(1, keepdim=True) > 0, -n, n)


def erode(m, r):
    return 1.0 - Fn.max_pool2d(1.0 - m, 2 * r + 1, stride=1, padding=r) if r > 0 else m


def normal_terms(Nr, Nd, Np, valid):
    """L1 + (1 - cos) per pixel for the rendered (Eq. 8) and D-Normal (Eq. 12) maps vs the pseudo normals; returns
    (per-pixel Ln, per-pixel Ldn without confidence), both B x 1 x S x S, masked by `valid`."""
    ln = (Nr - Np).abs().sum(1, keepdim=True) + (1 - (Nr * Np).sum(1, keepdim=True))
    ldn = (Nd - Np).abs().sum(1, keepdim=True) + (1 - (Nd * Np).sum(1, keepdim=True))
    return ln * valid, ldn * valid
