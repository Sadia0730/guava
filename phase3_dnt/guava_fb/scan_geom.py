"""GUAVA-FB pilot D: 4D-DRESS scan geometry (Wang et al., CVPR 2024: 80k-face textured scans with per-vertex garment labels,
non-commercial research licence). Cache layout (written by dress4d_scans.py, one dir per converted clip):
  verts.npy (F, VMAX, 3) float32 scan vertices in the 4D-DRESS world frame (zero padded), nverts.npy (F,) int32,
  faces.npy (F, 80000, 3) uint16, labels.npy (F, VMAX) uint8, meta.json {"frames": [...], "A": {frame_key: 3 x 4}}.
A maps the 4D-DRESS world into the GUAVA world of that row (the frame where its record's SMPL-X lives and w2c = [I | t']):
  X_guava = R_v R_c X_world + R_v t_c - t'   (convert_dress4d.py step 4: X_cv = R_v (R_c X_world + t_c) = X_model + t').
Zoomed targets keep the row's GUAVA world (only w2c is rotated), so the same A applies.
Labels (4D-DRESS order, checked geometrically in scan_check.py): 0 skin, 1 hair, 2 shoes, 3 upper, 4 lower, 5 outer."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn

LABELS = {0: "skin", 1: "hair", 2: "shoes", 3: "upper", 4: "lower", 5: "outer"}


class ScanStore:
    """memory-mapped access to the per-clip scan cache; get(clip, key) -> (V (n, 3) GUAVA world of the row, F (80000, 3), L (n,))."""
    def __init__(self, root):
        self.root = Path(root); self.c = {}

    def _clip(self, clip):
        if clip not in self.c:
            d = self.root / clip
            if not (d / "meta.json").exists(): self.c[clip] = None
            else:
                m = json.loads((d / "meta.json").read_text())
                self.c[clip] = {"fi": {f: i for i, f in enumerate(m["frames"])}, "A": {k: np.asarray(v, np.float64) for k, v in m["A"].items()},
                                **{n: np.load(d / f"{n}.npy", mmap_mode="r") for n in ("verts", "nverts", "faces", "labels")}}
        return self.c[clip]

    def get(self, clip, key):
        c = self._clip(clip)
        if c is None or key not in c["A"]: return None
        i = c["fi"][key.split("_", 1)[-1]]; n = int(c["nverts"][i]); A = c["A"][key]
        V = np.asarray(c["verts"][i, :n], np.float64) @ A[:, :3].T + A[:, 3]
        return V.astype(np.float32), np.asarray(c["faces"][i], np.int64), np.asarray(c["labels"][i, :n], np.int64)


def to_cam(V, w2c):
    return V @ w2c[:3, :3].T + w2c[:3, 3]


@torch.no_grad()
def render_scans(scans, w2c, itf, S=512):
    """scans: list of (V, F, L) torch tensors on the device (GUAVA world) or None; w2c (B, 4, 4); itf (B,).
    Returns normal map (B, 3, S, S; unit, OpenCV camera frame, towards the camera; 0 off the scan), depth (B, 1, S, S; camera z,
    0 off the scan), label map (B, 1, S, S; long, -1 off the scan). Same pixel convention as GUAVA's rasteriser
    (continuous px = (itf x / z + 1) S / 2); pytorch3d rasteriser, interpolated area-weighted vertex normals."""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.ops import interpolate_face_attributes
    B = w2c.shape[0]; dev = w2c.device
    N = torch.zeros(B, 3, S, S, device=dev); D = torch.zeros(B, 1, S, S, device=dev); Lm = torch.full((B, 1, S, S), -1, dtype=torch.long, device=dev)
    idx = [b for b in range(B) if scans[b] is not None]
    if not idx: return N, D, Lm
    verts, faces, Xc, lab = [], [], [], []
    for b in idx:
        V, F_, L = scans[b]; X = to_cam(V, w2c[b]); z = X[:, 2:].clamp(min=1e-4)
        verts.append(torch.cat([-itf[b] * X[:, :1] / z, -itf[b] * X[:, 1:2] / z, X[:, 2:]], 1)); faces.append(F_); Xc.append(X); lab.append(L)
    mesh = Meshes(verts=verts, faces=faces)
    p2f, zbuf, bary, _ = rasterize_meshes(mesh, image_size=S, blur_radius=0.0, faces_per_pixel=1, bin_size=None, max_faces_per_bin=60000,
                                          perspective_correct=False, cull_backfaces=False)
    camm = Meshes(verts=Xc, faces=faces); vn = camm.verts_normals_packed(); fpk = camm.faces_packed()
    nmap = interpolate_face_attributes(p2f, bary, vn[fpk])[..., 0, :]                       # (b, S, S, 3)
    hit = p2f[..., 0] >= 0
    fl = torch.cat(lab)[fpk[:, 0]]
    for j, b in enumerate(idx):
        n = Fn.normalize(nmap[j], dim=-1); h = hit[j]
        ray = torch.stack(torch.meshgrid(((torch.arange(S, device=dev) + 0.5) / S * 2 - 1) / itf[b], ((torch.arange(S, device=dev) + 0.5) / S * 2 - 1) / itf[b],
                                         indexing="xy") + (torch.ones(S, S, device=dev),), -1)
        n = torch.where(((n * ray).sum(-1, keepdim=True) > 0), -n, n) * h[..., None]
        N[b] = n.permute(2, 0, 1); D[b, 0] = zbuf[j, ..., 0] * h
        Lm[b, 0] = torch.where(h, fl[p2f[j, ..., 0].clamp(min=0)], torch.full_like(p2f[j, ..., 0], -1))
    return N, D, Lm


def blur_masked(x, m, sigma):
    """normalised Gaussian blur of x inside mask m (both B x C x S x S)."""
    r = int(3 * sigma); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2); k = k / k.sum()
    def sep(y):
        C = y.shape[1]; y = Fn.conv2d(y, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), padding=(0, r), groups=C)
        return Fn.conv2d(y, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), padding=(r, 0), groups=C)
    return sep(x * m) / sep(m).clamp(min=1e-6)


def fold_detail(Na, Nb, valid, sigma=3.0):
    """registered fold-detail measure (amendment 5): correlation of the high-pass normal components (normal minus its masked
    Gaussian blur, sigma 3 px at 512) of two normal maps over valid pixels; per image (B,). Smooth or random surfaces -> ~0."""
    ha = Na - blur_masked(Na, valid, sigma); hb = Nb - blur_masked(Nb, valid, sigma)
    num = (ha * hb * valid).sum((1, 2, 3)); den = ((ha ** 2 * valid).sum((1, 2, 3)) * (hb ** 2 * valid).sum((1, 2, 3))).sqrt().clamp(min=1e-12)
    return num / den
