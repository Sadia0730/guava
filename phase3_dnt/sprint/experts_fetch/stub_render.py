"""Import before hamer / wilor: replaces their visualisation-only renderer modules (pyrender needs OpenGL / GLU, absent
on this server) with inert stubs. Inference never renders."""
import sys, types
for name in ("hamer.utils.mesh_renderer", "hamer.utils.renderer", "wilor.utils.mesh_renderer", "wilor.utils.renderer"):
    m = types.ModuleType(name)
    class _R:
        def __init__(self, *a, **k): pass
    m.MeshRenderer = _R; m.Renderer = _R; m.cam_crop_to_full = None
    sys.modules[name] = m
