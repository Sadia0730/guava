"""Pure SMPL-X LBS (numpy assets, torch CPU/GPU) for the GUAVA-FB converters: the same math as GUAVA (a20)'s
smplx_only_forward (v_template + shapedirs[:n_shape] / expression dirs, posedirs, LBS; no hand PCA, flat hand mean,
no FLAME insertion). Axis-angle input. Used for SMPLX_NEUTRAL_2020 (200 shape + 50 exp) and the locked-head model (16 betas)."""
import numpy as np
import torch

GUAVA = "/home/ubx858/guava"
SMPLX2020 = f"{GUAVA}/assets/SMPLX/SMPLX_NEUTRAL_2020.npz"
LOCKED = f"{GUAVA}/assets/SMPLX/lockedhead/models_lockedhead/smplx/SMPLX_NEUTRAL.npz"


def rodrigues(aa):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, dtype=aa.dtype, device=aa.device) + s * K + (1 - c) * (K @ K)


class SMPLXLean(torch.nn.Module):
    def __init__(self, path, n_shape, n_exp=50, dtype=torch.float64):
        super().__init__()
        d = np.load(path, allow_pickle=True)
        sd = d["shapedirs"]
        dirs = np.concatenate([sd[:, :, :n_shape], sd[:, :, 300:300 + n_exp]], 2)
        self.n_shape, self.n_exp = n_shape, n_exp
        self.register_buffer("v_template", torch.tensor(d["v_template"], dtype=dtype))
        self.register_buffer("shapedirs", torch.tensor(dirs, dtype=dtype))
        self.register_buffer("posedirs", torch.tensor(d["posedirs"].reshape(-1, 486).T, dtype=dtype))   # (486, V*3)
        self.register_buffer("J_regressor", torch.tensor(np.asarray(d["J_regressor"]), dtype=dtype))
        self.register_buffer("weights", torch.tensor(d["weights"], dtype=dtype))
        par = d["kintree_table"][0].astype(np.int64); par[0] = -1
        self.parents = par
        self.faces = d["f"].astype(np.int64)

    def shaped(self, betas, exp=None):
        B = betas.shape[0]
        exp = betas.new_zeros(B, self.n_exp) if exp is None else exp
        b = torch.cat([betas[:, :self.n_shape], exp], 1)
        v = self.v_template[None] + torch.einsum("vcs,bs->bvc", self.shapedirs, b)
        return v, torch.einsum("jv,bvc->bjc", self.J_regressor, v)

    def forward(self, pose_aa, betas, exp=None, rot=None):
        """pose_aa (B, 165) [root, body 21, jaw, leye, reye, lhand 15, rhand 15] or rot (B, 55, 3, 3). -> verts (B, V, 3), joints (B, 55, 3)."""
        v, J = self.shaped(betas, exp)
        B = v.shape[0]
        R = rodrigues(pose_aa.reshape(B, 55, 3)) if rot is None else rot
        eye = torch.eye(3, dtype=v.dtype, device=v.device)
        vp = v + ((R[:, 1:] - eye).reshape(B, -1) @ self.posedirs).reshape(B, -1, 3)
        par = self.parents
        rel = J.clone(); rel[:, 1:] = J[:, 1:] - J[:, par[1:]]
        T = [None] * 55
        Tl = torch.zeros(B, 55, 4, 4, dtype=v.dtype, device=v.device)
        Tl[:, :, :3, :3] = R; Tl[:, :, :3, 3] = rel; Tl[:, :, 3, 3] = 1
        T[0] = Tl[:, 0]
        for j in range(1, 55):
            T[j] = T[par[j]] @ Tl[:, j]
        Tw = torch.stack(T, 1)
        Jt = Tw[:, :, :3, 3].clone()
        A = Tw.clone(); A[:, :, :3, 3] = Jt - (Tw[:, :, :3, :3] @ J[..., None])[..., 0]
        Tv = torch.einsum("vj,bjkl->bvkl", self.weights, A)
        verts = (Tv[:, :, :3, :3] @ vp[..., None])[..., 0] + Tv[:, :, :3, 3]
        return verts, Jt
