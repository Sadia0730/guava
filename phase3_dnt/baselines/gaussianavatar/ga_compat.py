"""Run-time compatibility shim for GaussianAvatar @ d981c62 under its own environment.yml (pytorch 2.0.1).

AvatarModel.__init__ indexes CPU tensors (the canonical query posmap and the LBS map, both loaded with np.load) with the
CUDA boolean mask `valid_idx` returned by utils.general_utils.load_masks:
    query_points = query_map[valid_idx, :].cuda()      -> RuntimeError: indices should be either on cpu or on the same
    query_lbs[valid_idx, :]                                device as the indexed tensor (cpu)
The shim hands __init__ the same mask on the CPU and moves it back to CUDA when __init__ returns, so every later use
(pred_res[:, self.valid_idx] etc.) sees exactly the tensor the original code intends. No numerical change.
"""


def apply():
    import model.avatar_model as am

    if getattr(am, "_ga_compat_applied", False):
        return
    orig_load_masks = am.load_masks

    def load_masks_cpu_idx(*args, **kwargs):
        flist_uv, valid_idx, uv_coord_map = orig_load_masks(*args, **kwargs)
        return flist_uv, valid_idx.cpu(), uv_coord_map

    am.load_masks = load_masks_cpu_idx
    orig_init = am.AvatarModel.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.valid_idx = self.valid_idx.cuda()

    am.AvatarModel.__init__ = init
    am._ga_compat_applied = True
