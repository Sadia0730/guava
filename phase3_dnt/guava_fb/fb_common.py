"""GUAVA-FB common code (GUAVA extended to the full body; all credit for the architecture / losses / recipe goes to GUAVA,
Zhang et al.). Spec: phase3_dnt/PREREG.md "GUAVA-FB", phase3_dnt/guava_fb/FORMAT.md, AVATAR_B_PLAN.md section 1 (#3 camera, #6 guard).

GUAVA files are NOT modified; everything is a runtime patch (as guava_finetune.py's smplx_only_forward).

(a) Per-sample perspective camera. GUAVA hard-codes invtanfov = 24 (configs/train/ubody_512.yaml:10) in:
    1. dataset/data_loader.py:24, 91-97      target render camera (get_full_proj_matrix tanfov + tanfovx/y)   -> FBData.target_info()
       (utils/graphics_utils.py:44-50 get_full_proj_matrix already takes tanfov as an argument; principal point centred)
    2. models/UbodyAvatar/ubody_gaussian.py:60-61 (BaseMeshRenderer focal_length=cfg.invtanfov, used by render_fragments
       utils/graphics_utils.py:472-492 for the SOURCE visibility mask)                                    -> render_fragments_fb()
    3. ubody_gaussian.py:80  sample_prj_feature (SOURCE vertex -> DINO feature map projection)             -> sample_prj_feature_fb()
    4. ubody_gaussian.py:99  convert_pixel_feature_to_uv (SOURCE UV texel -> image feature projection)     -> convert_pixel_feature_to_uv_fb()
    The source image's camera (batch['source']['w2c_cam'] + batch['source']['invtanfov']) drives 2-4 (projecting the source image
    features onto the UV map / vertices); the target's camera (target w2c + target invtanfov) drives 1 (Gaussian rasteriser,
    models/UbodyAvatar/gaussian_render.py:44-52 already reads tanfov / projmatrix per sample). Ubody_Gaussian.forward (deformation)
    and get_cam_dirs (rotation only) do not depend on the focal. ubody_gaussian.py:29 intri_cam is unused.
    Per-sample value = record["invtanfov"] (float), default 24 when absent.
    GUAVA bug fixed on the way: GS_Camera.get_projection_transform per-sample branch (graphics_utils.py:174-177) crashes for a
    batch with different focals -> get_projection_transform_fb (single-focal path unchanged).
(b) a20 model setup: a20 best.pt, EHM FLAME-template rows reset to the SMPL-X 2020 template, smplx_only_forward (taken verbatim
    from guava_finetune.py via ast, as avatar_a_compare.py does).
(c) empty-box guard for Optimization_Loss.cal_box_loss (utils/loss_utils.py:140-159; torch.cat([]) at :154 when every crop is empty).
(d) check_render(clip_dir, n): GUAVA camera vs <clip>/gt_proj.npz, mask precision, overlay jpg.

CLI:  python fb_common.py regression      (registered regression check + synthetic projection checks -> O/reports/camera_patch/)
      python fb_common.py selftest        (export one a20 UBody clip to O/data/_selftest in the new format + render equality)
      python fb_common.py check_render <clip_dir> [n]
"""
from __future__ import annotations
import ast, json, os, pickle, random, sys, time, types
from copy import deepcopy
from pathlib import Path
import numpy as np, torch, torchvision
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "phase3_dnt")):
    if p not in sys.path: sys.path.insert(0, p)
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
A20_CKPT = Path("/raid/ubx858/outputs/dnt/guava_a20/run/checkpoints/best.pt")
A20_RUN_INFO = Path("/raid/ubx858/outputs/dnt/guava_a20/run/run_info.json")
TRACKED = Path("/raid/ubx858/outputs/dnt/c2t/tracked")
EXCL = Path("/raid/ubx858/outputs/dnt/sprint/phase1/ubody_ehm_gap/filter_keep/dropped_clips.json")
FACE_CONV = Path("/raid/ubx858/outputs/dnt/guava_a20/conversion/face_a20.npz")
DEFAULT_INVTANFOV = 24.0
NV = 10475
CROP_KEYS = ("body_crop", "head_crop", "left_hand_crop", "right_hand_crop")

os.chdir(ROOT)   # GUAVA builds its models from relative asset paths
from dataset.data_loader import TrackedData, data_to_tensor, squeeze_params
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
from models.UbodyAvatar.ubody_gaussian import get_cam_dirs
from models.modules.ehm import EHM as _EHM  # noqa: F401  (registers the module)
from utils.general_utils import ConfigDict, add_extra_cfgs
from utils.graphics_utils import get_full_proj_matrix, get_proj_matrix, GS_MeshRasterizer, GS_Camera
from utils.lmdb import LMDBEngine
from utils.loss_utils import Optimization_Loss, cal_ssim, cal_psnr
from pytorch3d.structures import Meshes
ehm_mod = sys.modules["models.modules.ehm.EHM"]


def get_cfg():
    return add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "configs/train/ubody_512.yaml")))


def load_pkl(p):
    with open(p, "rb") as f: return pickle.load(f)


# ------------------------------------------------------------------------------------------------- (b) a20 model
def _smplx_only_forward():
    src = (ROOT / "phase3_dnt/guava_finetune.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "smplx_only_forward")
    ns = {"ehm_mod": ehm_mod, "np": np, "torch": torch}; exec(compile(ast.Module([fn], []), "guava_finetune.py", "exec"), ns)
    return ns["smplx_only_forward"]


smplx_only_forward = _smplx_only_forward()


def setup_a20(dev, ckpt=A20_CKPT, camera_patch=True):
    """GUAVA (a20): inferer + renderer from a20 best.pt, SMPL-X 2020 template reset, smplx_only_forward. camera_patch=True
    installs the per-sample camera forward (identical to GUAVA's forward when every invtanfov is 24)."""
    cfg = get_cfg()
    inf = Ubody_Gaussian_inferer(cfg.MODEL).to(dev); ren = GaussianRenderer(cfg.MODEL).to(dev)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    r1 = inf.load_state_dict(ck["model"], strict=False); r2 = ren.load_state_dict(ck["render_model"], strict=False)
    meta = {"ckpt": str(ckpt), "global_iter": ck.get("global_iter"), "variant": ck.get("variant"), "val": ck.get("val"),
            "missing": [list(r1.missing_keys), list(r2.missing_keys)], "unexpected": [list(r1.unexpected_keys), list(r2.unexpected_keys)]}
    assert ck.get("variant") in ("a20", "guava_fb"), ck.get("variant")
    del ck
    with torch.no_grad():                                          # a20: EHM's FLAME-template insertion undone
        s = inf.ehm.smplx; inf.ehm.v_template[:NV] = s.v_template[:NV]; inf.v_template.data[:NV] = s.v_template[:NV]
    inf.ehm.forward = types.MethodType(smplx_only_forward, inf.ehm)
    if camera_patch: patch_inferer(inf)
    return cfg, inf, ren, meta


# ------------------------------------------------------------------------------------------------- (a) camera patch
def batch_invtanfov(batch, B, device):
    v = batch.get("invtanfov") if isinstance(batch, dict) else None
    if v is None: return torch.full((B,), DEFAULT_INVTANFOV, dtype=torch.float32, device=device)
    return torch.as_tensor(v, dtype=torch.float32, device=device).reshape(B)


def _bview(itf, ndim):
    return itf.view((itf.shape[0],) + (1,) * (ndim - 1))


def sample_prj_feature_fb(self, vertices, feature_map, w2c_cam, vertices_img=None, invtanfov=None):
    """ubody_gaussian.py:75-83 with per-sample invtanfov (B,)."""
    if vertices_img is None:
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[:, :, :1])], dim=-1)
        vertices_cam = torch.einsum('bij,bnj->bni', w2c_cam, vertices_homo)[:, :, :3]
        vertices_img = vertices_cam * _bview(invtanfov, 3) / (vertices_cam[:, :, 2:] + 1e-7)
    sampled = nn.functional.grid_sample(feature_map, vertices_img[:, None, :, :2], mode='bilinear', padding_mode='border', align_corners=False)
    return sampled.squeeze(-2).permute(0, 2, 1).contiguous(), vertices_img


def convert_pixel_feature_to_uv_fb(self, img_features, deformed_vertices, w2c_cam, visble_faces=None, uv_features=None, invtanfov=None):
    """ubody_gaussian.py:85-114 with per-sample invtanfov (B,)."""
    batch_size = img_features.shape[0]
    uvmap_f_idx = self.smplx.uvmap_f_idx
    uvmap_f_bary = self.smplx.uvmap_f_bary[None].expand(batch_size, -1, -1, -1)
    faces = self.smplx.faces_tensor
    uv_vertex_id = faces[uvmap_f_idx]
    uv_vertex = deformed_vertices.permute(1, 2, 0).contiguous()[uv_vertex_id]
    uv_vertex = uv_vertex.permute(4, 0, 1, 2, 3).contiguous()
    uv_vertex = torch.einsum('bhwk,bhwkn->bhwn', uvmap_f_bary, uv_vertex)
    uv_vertex_homo = torch.cat([uv_vertex, torch.ones_like(uv_vertex[:, :, :, :1])], dim=-1)
    uv_vertex_cam = torch.einsum('bij,bhwj->bhwi', w2c_cam, uv_vertex_homo)[:, :, :, :3]
    vertices_img = uv_vertex_cam * _bview(invtanfov, 4) / (uv_vertex_cam[..., 2:] + 1e-7)
    uv_features = nn.functional.grid_sample(img_features, vertices_img[:, :, :, :2], mode='bilinear', padding_mode='zeros', align_corners=False)
    mask = self.smplx.uvmap_mask.clone()[None].repeat(batch_size, 1, 1)
    if visble_faces is not None:
        num_faces = self.smplx.faces_tensor.shape[0]
        f_offset = torch.arange(batch_size, device=uvmap_f_idx.device, dtype=torch.int32) * num_faces
        all_faces = torch.arange(0, faces.shape[0], device=uvmap_f_idx.device, dtype=torch.int32)[None].repeat(batch_size, 1) + f_offset[:, None]
        visble_mask = torch.isin(all_faces, torch.unique(visble_faces))[:, uvmap_f_idx]
        mask = mask * visble_mask
    return uv_features * mask[:, None]


def build_mesh_cameras(mr, transform_matrix, invtanfov, reverse_camera=True):
    """graphics_utils.py BaseMeshRenderer camera (render_fragments :479-485) with per-sample focal (GS_Camera supports (B,))."""
    B = transform_matrix.shape[0]; transform_matrix = transform_matrix.clone()
    if reverse_camera:
        tf = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device=transform_matrix.device)
        transform_matrix = torch.bmm(tf[None].expand(B, -1, -1), transform_matrix)
    return mr._build_cameras(transform_matrix, invtanfov.float())


_orig_get_projection_transform = GS_Camera.get_projection_transform


def get_projection_transform_fb(self, device):
    """utils/graphics_utils.py:167-183 GS_Camera.get_projection_transform: the per-sample branch (:174-177) indexes the
    focal twice (`invtanfov=invtanfov[0]; assert invtanfov[0]==invtanfov[1]` -> IndexError on a 0-dim tensor), so a batch with
    different focals crashes. Fixed branch; the single-focal branch is GUAVA's own (bit-identical)."""
    if self.proj_mats is None and torch.unique(self.focal_length).numel() > 1:
        assert bool((self.focal_length[:, 0] == self.focal_length[:, 1]).all())
        self.proj_mats = torch.cat([get_proj_matrix(1 / f[0], device)[None] for f in self.focal_length], 0)
    return _orig_get_projection_transform(self, device)


def install_camera_class_patch():
    GS_Camera.get_projection_transform = get_projection_transform_fb


def render_fragments_fb(mr, vertices, transform_matrix, invtanfov):
    B = vertices.shape[0]
    cameras = build_mesh_cameras(mr, transform_matrix, invtanfov)
    mesh = Meshes(verts=vertices, faces=mr.faces[None].repeat(B, 1, 1))
    fragments = GS_MeshRasterizer(cameras=cameras, raster_settings=mr.raster_settings)(mesh)
    return fragments.pix_to_face, fragments


def inferer_forward_fb(self, batch):
    """Ubody_Gaussian_inferer.forward (ubody_gaussian.py:116-159) with the SOURCE frame's per-sample invtanfov in the
    visibility mesh render, sample_prj_feature and convert_pixel_feature_to_uv."""
    batch_size = batch['image'].shape[0]
    itf = batch_invtanfov(batch, batch_size, batch['image'].device)
    extra_dict = {}
    dino = self.dino_encoder(batch['image'], output_size=self.cfg.image_size)
    img_feature, img_feature_2, global_faeature = dino['f_map1'], dino['f_map2'], dino['f_global']
    vertex_global_feature = self.global_feature_mapping(global_faeature)
    cam_dirs = self.harmo_encoder(get_cam_dirs(batch["w2c_cam"]))
    vertex_base_feature = self.vertex_base_feature[None].expand(batch_size, -1, -1)
    self.smplx_deform_res = self.ehm(batch['smplx_coeffs'], batch['flame_coeffs'])
    vertex_sample_feature, vertex_prj = sample_prj_feature_fb(self, self.smplx_deform_res['vertices'], img_feature_2, batch['w2c_cam'], invtanfov=itf)
    vertex_global_feature = vertex_global_feature[:, None, :].expand(-1, vertex_sample_feature.shape[-2], -1)
    vertex_sample_feature = torch.cat([vertex_sample_feature, vertex_base_feature, vertex_global_feature], dim=-1)
    vertex_gs_dict = self.vertex_gs_decoder(vertex_sample_feature, cam_dirs)
    vertex_gs_dict["positions"] = self.v_template.clone()[None].expand(batch_size, -1, -1)
    image_rgb = nn.functional.interpolate(batch['image'], (self.cfg.image_size, self.cfg.image_size), mode='bilinear', align_corners=False)
    img_feature = torch.cat([image_rgb, img_feature], dim=1)
    with torch.no_grad():
        visble_faces, fragments = render_fragments_fb(self.mesh_renderer, self.smplx_deform_res['vertices'], batch['w2c_cam'], itf)
    uvmap_features = convert_pixel_feature_to_uv_fb(self, img_feature, self.smplx_deform_res['vertices'], batch['w2c_cam'],
                                                    visble_faces=visble_faces, invtanfov=itf)
    extra_style = self.uv_style_mapping(global_faeature)
    uvmap_features = self.uv_feature_decoder(uvmap_features, extra_style=extra_style)
    uvmap_features = torch.cat([uvmap_features, self.uv_base_feature[None].expand(batch_size, -1, -1, -1)], dim=1)
    uv_point_gs_dict = self.uv_point_decoder(uvmap_features, cam_dirs)
    for key in uv_point_gs_dict.keys():
        uv_point_gs_dict[key] = uv_point_gs_dict[key].reshape(batch_size, self.uvmap_size * self.uvmap_size, -1)[:, self.uv_mask_flat, :]
    uv_point_gs_dict['binding_face'] = self.smplx.uvmap_f_idx.clone().reshape(1, self.uvmap_size * self.uvmap_size, -1)[:, self.uv_mask_flat, :]
    uv_point_gs_dict['face_bary'] = self.smplx.uvmap_f_bary.clone().reshape(1, self.uvmap_size * self.uvmap_size, -1)[:, self.uv_mask_flat, :]
    extra_dict['uvmap_texture'] = torch.sigmoid(uvmap_features[:, :3].permute(0, 2, 3, 1).contiguous())
    return vertex_gs_dict, uv_point_gs_dict, extra_dict


def patch_inferer(inf):
    install_camera_class_patch(); inf.forward = types.MethodType(inferer_forward_fb, inf); inf._fb_patched = True


def unpatch_inferer(inf):
    if "forward" in inf.__dict__: del inf.__dict__["forward"]
    inf._fb_patched = False


def render_cam_params(w2c, c2w, invtanfov, image_size):
    """dataset/data_loader.py:91-97 with the record's own invtanfov (python float -> bit-identical to GUAVA at 24)."""
    tanfov = 1.0 / float(invtanfov)
    view, full = get_full_proj_matrix(w2c, tanfov)
    return {"world_view_transform": view, "full_proj_transform": full, "tanfovx": tanfov, "tanfovy": tanfov,
            "image_height": image_size, "image_width": image_size, "camera_center": c2w[:3, 3]}


# ------------------------------------------------------------------------------------------------- (c) empty-box guard
_orig_cal_box_loss = Optimization_Loss.cal_box_loss


def cal_box_loss_guarded(self, render_images, gt_images, box, loss_funs, loss_lambdas):
    """utils/loss_utils.py:140-159 unchanged except: 0 when no crop in the batch is non-empty (else torch.cat([]) crashes)."""
    h = box[:, 3].clamp(max=gt_images.shape[-2]) - box[:, 2]; w = box[:, 1].clamp(max=gt_images.shape[-1]) - box[:, 0]
    if not bool(((h >= 1) & (w >= 1)).any()):
        return render_images.sum() * 0.0
    return _orig_cal_box_loss(self, render_images, gt_images, box, loss_funs, loss_lambdas)


def install_box_guard():
    Optimization_Loss.cal_box_loss = cal_box_loss_guarded


# ------------------------------------------------------------------------------------------------- projection helpers
def project_px(verts, w2c, itf, size=512):
    """GUAVA's projection (sample_prj_feature): ndc = itf * x_cam / z_cam (grid_sample, align_corners=False) -> pixel
    index coordinates (pixel centres at integers) in a size x size image."""
    vh = torch.cat([verts, torch.ones_like(verts[..., :1])], -1)
    vc = torch.einsum('bij,bnj->bni', w2c, vh)[..., :3]
    ndc = vc[..., :2] * _bview(itf, 3) / (vc[..., 2:] + 1e-7)
    return (ndc + 1) * size / 2 - 0.5, vc


class FaceGeom:
    """SMPL-X 2020 face landmarks (51 static landmark embedding of SMPLX_NEUTRAL_2020.npz = iBUG 68 points 17..67) for the
    ArcFace 5-point alignment: [subject's right eye centre (68-pt 36-41), left eye centre (42-47), nose tip (30), right mouth
    corner (48), left mouth corner (54)] = insightface order (image-left first for a frontal face). Head vertices = EHM head
    (smplx2flame_ind[head_index], indices into the FULL EHM vertex array incl. teeth) for face boxes."""
    def __init__(self, inf, dev):
        d = np.load(ROOT / "assets/SMPLX/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)
        f = torch.as_tensor(d["f"].astype(np.int64), device=dev); li = torch.as_tensor(d["lmk_faces_idx"].astype(np.int64), device=dev)
        self.lmk_vid = f[li]; self.lmk_bary = torch.as_tensor(d["lmk_bary_coords"], dtype=torch.float32, device=dev)
        hi = np.unique(inf.ehm.flame.head_index)
        s2f = inf.ehm.smplx.smplx2flame_ind; s2f = s2f.cpu().numpy() if torch.is_tensor(s2f) else np.asarray(s2f)
        self.head_vid = torch.as_tensor(s2f[hi], device=dev).long()

    def lmk51(self, verts):
        return torch.einsum('lk,blkc->blc', self.lmk_bary, verts[:, self.lmk_vid])

    def kps5(self, verts):
        l = self.lmk51(verts)
        return torch.stack([l[:, 19:25].mean(1), l[:, 25:31].mean(1), l[:, 13], l[:, 31], l[:, 37]], 1)


def face_boxes(head_px, size=512):
    """square box (x0, y0, x1, y1) around the projected head vertices (side = max(w, h)) + fraction inside the image."""
    mn, mx = head_px.min(1).values, head_px.max(1).values; c = (mn + mx) / 2; s = (mx - mn).max(-1).values
    b = torch.stack([c[:, 0] - s / 2, c[:, 1] - s / 2, c[:, 0] + s / 2, c[:, 1] + s / 2], -1)
    bi = b.clamp(0, size); area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inside = ((bi[:, 2] - bi[:, 0]) * (bi[:, 3] - bi[:, 1]) / area.clamp(min=1e-6))
    return b, s, inside


def head_forward_cam(jtm, w2c):
    """head (SMPL-X joint 15) forward axis (+z of the canonical face) in the camera frame."""
    f = jtm[:, 15, :3, 2]
    return torch.einsum('bij,bj->bi', w2c[:, :3, :3], f)


# ------------------------------------------------------------------------------------------------- data
def finite_crops(r):
    try:
        return all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for k in CROP_KEYS for v in r[k].values())
    except Exception:
        return False


def _f32(x, shape):
    return np.asarray(x, np.float32).reshape(shape)


def compact_record(r, shape200, face=None, eye_copy=True):
    """record -> the fields GUAVA-FB needs. face=(exp50, jaw) applies a20's UBody conversion (guava_finetune.py variant a20);
    otherwise exp / jaw_pose / eye_pose come from smplx_coeffs (new format). Identity extras off (as a20)."""
    sc = r["smplx_coeffs"]
    out = {"global_pose": _f32(sc["global_pose"], 3), "body_pose": _f32(sc["body_pose"], (21, 3)),
           "left_hand_pose": _f32(sc["left_hand_pose"], (15, 3)), "right_hand_pose": _f32(sc["right_hand_pose"], (15, 3)),
           "camera_RT_params": _f32(sc["camera_RT_params"], (3, 4)), "shape": _f32(shape200, 200)}
    if face is not None:
        out["exp"] = _f32(face[0], 50); out["jaw_pose"] = _f32(face[1], (1, 3))
        out["eye_pose"] = _f32(r["flame_coeffs"]["eye_pose_params"], (2, 3)) if eye_copy else np.zeros((2, 3), np.float32)
    else:
        out["exp"] = _f32(sc["exp"], 50)
        out["jaw_pose"] = _f32(sc.get("jaw_pose", np.zeros(3)), (1, 3)); out["eye_pose"] = _f32(sc.get("eye_pose", np.zeros(6)), (2, 3))
    crops = {k: {kk: np.asarray(vv, np.float32) for kk, vv in r[k].items() if vv is not None} for k in CROP_KEYS}
    iv = r.get("invtanfov"); iv = DEFAULT_INVTANFOV if iv is None else float(iv)
    return {"smplx": out, "crops": crops, "invtanfov": iv, "view": str(r.get("view", "0"))}


def load_clip(args):
    """(clip_dir, face_dict or None, eye_copy) -> dict; run in a process pool."""
    cdir, face, eye_copy = args; cdir = Path(cdir)
    try:
        if not ((cdir / "optim_tracking_ehm.pkl").exists() and (cdir / "id_share_params.pkl").exists() and (cdir / "img_lmdb").exists()):
            return {"dir": str(cdir), "error": "incomplete"}
        d = load_pkl(cdir / "optim_tracking_ehm.pkl"); ids = load_pkl(cdir / "id_share_params.pkl")
        shape = np.asarray(ids["smplx_shape"], np.float32).reshape(-1)[:200]
        recs = {}
        for k in sorted(d):
            if not finite_crops(d[k]): continue
            if face is not None and k not in face: continue
            recs[k] = compact_record(d[k], shape, face[k] if face is not None else None, eye_copy)
        return {"dir": str(cdir), "name": cdir.name, "recs": recs, "keys": sorted(recs), "n_raw": len(d)}
    except Exception as e:
        return {"dir": str(cdir), "error": repr(e)}


class LmdbPool:
    def __init__(self, maxn=32): self.maxn, self.pool, self.pid = maxn, {}, os.getpid()

    def get(self, path):
        if os.getpid() != self.pid:                       # forked: drop the parent's handles without closing them
            for e in self.pool.values(): e._manual_close = True
            self.pool, self.pid = {}, os.getpid()
        if path in self.pool:
            e = self.pool.pop(path); self.pool[path] = e; return e
        if len(self.pool) >= self.maxn:
            k0 = next(iter(self.pool)); e0 = self.pool.pop(k0); e0.close()
        e = LMDBEngine(os.path.join(path, "img_lmdb"), write=False); self.pool[path] = e; return e


class FBData(TrackedData):
    """Record / image access over many clips in the GUAVA-FB format (flat single-clip tracked dirs)."""
    def __init__(self, cfg, clips):
        torch.utils.data.Dataset.__init__(self)
        self.cfg = cfg; self.feature_img_size = cfg.MODEL.feature_img_size; self.image_size = cfg.MODEL.image_size
        self.clips = clips; self.lmdb = LmdbPool()

    def info(self, ci, k):
        c = self.clips[ci]; rec = c["recs"][k]; e = self.lmdb.get(c["dir"])
        img = e[f"{k}/body_image"].float() / 255.0; msk = e[f"{k}/body_mask"].float() / 255.0
        if img.shape[-1] != self.cfg.DATASET.origin_image_size:
            raise ValueError(f"{c['dir']}:{k} image {tuple(img.shape)} != origin_image_size {self.cfg.DATASET.origin_image_size}")
        sm = {kk: torch.from_numpy(v.copy()) for kk, v in rec["smplx"].items()}
        sm.update({"joints_offset": torch.zeros(55, 3), "head_scale": torch.ones(3), "hand_scale": torch.ones(3)})
        r = {"smplx_coeffs": sm, "flame_coeffs": {"expression_params": torch.zeros(50)},
             **{kk: {a: torch.from_numpy(b) for a, b in v.items()} for kk, v in rec["crops"].items()}}
        RT = sm["camera_RT_params"]; c2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])); M4 = torch.eye(4); M4[:3, :4] = RT
        w2c = c2c @ M4; r["w2c_cam"], r["c2w_cam"] = w2c, torch.linalg.inv(w2c)
        r["head_box"], r["left_hand_box"], r["right_hand_box"] = self._load_box(r)
        for kk in CROP_KEYS: r.pop(kk)
        r["invtanfov"] = float(rec["invtanfov"])
        return r, img, msk

    def source_info(self, ci, k):
        st, si, sm = self.info(ci, k); si = si * sm
        st["image"] = torchvision.transforms.functional.resize(si, (self.feature_img_size, self.feature_img_size), antialias=True)
        return st

    def target_info(self, ci, k):
        tt, ti, tm = self.info(ci, k)
        tt["image"] = torchvision.transforms.functional.resize(ti, (self.image_size, self.image_size), antialias=True)
        tt["mask"] = torchvision.transforms.functional.resize(tm, (self.image_size, self.image_size), antialias=True)
        tt["render_cam_params"] = render_cam_params(tt["w2c_cam"], tt["c2w_cam"], tt["invtanfov"], self.image_size)
        return tt


def load_face_conv(path=FACE_CONV):
    fcv = np.load(path); face = {}
    for c, k, e, j in zip(fcv["clip"], fcv["frame_key"], fcv["exp50"], fcv["jaw"]):
        face.setdefault(str(c), {})[str(k)] = (e, j)
    return face, bool(fcv["eye_copy"])


def a20_split(tracked=TRACKED, exclude=EXCL, seed=0, n_val=10):
    """guava_finetune.py's seed-0 clip draw with --exclude-clips (exact)."""
    clips = sorted(p.parent for p in Path(tracked).glob("*/optim_tracking_ehm.pkl"))
    rng = random.Random(seed); val = sorted(rng.sample(clips, min(n_val, max(1, len(clips) // 10))))
    excl = set(json.loads(Path(exclude).read_text()))
    v_drop = [c for c in val if c.name in excl]; keep = [c for c in val if c.name not in excl]
    pool = [c for c in clips if c.name not in excl and c not in val]
    repl = rng.sample(pool, min(len(v_drop), len(pool))) if v_drop else []
    val = sorted(keep + repl); trn = [c for c in clips if c not in val and c.name not in excl]
    ri = json.loads(A20_RUN_INFO.read_text())
    assert [c.name for c in val] == ri["val_clips"] and [c.name for c in trn] == ri["train_clips"], "a20 draw mismatch"
    return trn, val


def load_clips_parallel(jobs, workers=16):
    from concurrent.futures import ProcessPoolExecutor
    if len(jobs) <= 2: return [load_clip(j) for j in jobs]
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex: return list(ex.map(load_clip, jobs, chunksize=1))


# ------------------------------------------------------------------------------------------------- a20's own loader (reference)
class A20MultiClip(TrackedData):
    """guava_finetune.py MultiClip, variant a20, verbatim logic (reference for the regression check)."""
    def __init__(self, cfg, clip_dirs, split, conv):
        torch.utils.data.Dataset.__init__(self)
        self.cfg = cfg; self.bg_color = 0.0; self.feature_img_size = cfg.MODEL.feature_img_size; self.feature_part_size = cfg.MODEL.feature_part_size
        self.image_size = cfg.MODEL.image_size; self.tanfov, self.focal = 1 / cfg.MODEL.invtanfov, cfg.MODEL.invtanfov; self.split = split
        self.conv = conv
        self.clips = [str(c) for c in clip_dirs]; self.tracked, self.ids, self.keys, self.frames = [], [], [], []
        for ci, c in enumerate(self.clips):
            d = load_pkl(Path(c) / "optim_tracking_ehm.pkl"); ks = [k for k in sorted(d) if finite_crops(d[k])]
            self.tracked.append(d); self.ids.append(load_pkl(Path(c) / "id_share_params.pkl")); self.keys.append(ks)
            sel = ks if split == "train" else ks[1::4]
            self.frames += [(ci, k) for k in sel]
        self._engines = {}

    def _eng(self, ci):
        if ci not in self._engines: self._engines[ci] = LMDBEngine(os.path.join(self.clips[ci], "img_lmdb"), write=False)
        return self._engines[ci]

    def __len__(self): return len(self.frames)

    def __getitem__(self, i):
        ci, k = self.frames[i]
        src = self.keys[ci][0]
        st, si, sm = self._info(ci, src); si = si * sm
        si = torchvision.transforms.functional.resize(si, (self.feature_img_size, self.feature_img_size), antialias=True)
        tt, ti, tm = self._info(ci, k)
        ti = torchvision.transforms.functional.resize(ti, (self.image_size, self.image_size), antialias=True)
        tm = torchvision.transforms.functional.resize(tm, (self.image_size, self.image_size), antialias=True)
        view, full = get_full_proj_matrix(tt["w2c_cam"], self.tanfov)
        tt["render_cam_params"] = {"world_view_transform": view, "full_proj_transform": full, "tanfovx": self.tanfov, "tanfovy": self.tanfov,
                                   "image_height": self.image_size, "image_width": self.image_size, "camera_center": tt["c2w_cam"][:3, 3]}
        rec = {"source": {"image": si}, "target": {"image": ti, "mask": tm}}; rec["source"].update(st); rec["target"].update(tt); return rec

    def _info(self, ci, k):
        e = self._eng(ci); img = e[f"{k}/body_image"].float() / 255.0; msk = e[f"{k}/body_mask"].float() / 255.0
        r = deepcopy(self.tracked[ci][k]); ids = self.ids[ci]
        r["smplx_coeffs"].update({"shape": ids["smplx_shape"][0], "joints_offset": np.zeros_like(ids["joints_offset"][0]),
                                  "head_scale": np.ones(3, np.float32), "hand_scale": np.ones(3, np.float32)})
        r["flame_coeffs"].update({"shape_params": ids["flame_shape"][0]})
        e50, jaw = self.conv["face"][(Path(self.clips[ci]).name, k)]
        r["smplx_coeffs"]["exp"] = np.asarray(e50, np.float32).reshape(50)
        r["smplx_coeffs"]["jaw_pose"] = np.asarray(jaw, np.float32).reshape(1, 3)
        r["smplx_coeffs"]["eye_pose"] = (np.asarray(r["flame_coeffs"]["eye_pose_params"], np.float32).reshape(2, 3) if self.conv["eye_copy"]
                                         else np.zeros((2, 3), np.float32))
        r = squeeze_params(data_to_tensor(r))
        RT = r["smplx_coeffs"]["camera_RT_params"]; c2c = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0])); M4 = torch.eye(4); M4[:3, :4] = RT
        w2c = c2c @ M4; r["w2c_cam"], r["c2w_cam"] = w2c, torch.linalg.inv(w2c)
        r["head_box"], r["left_hand_box"], r["right_hand_box"] = self._load_box(r)
        for drop in ("dwpose_raw", "dwpose_rlt", "head_lmk_203", "head_lmk_70", "head_lmk_mp", "left_mano_coeffs", "right_mano_coeffs"):
            r.pop(drop, None)
        return r, img, msk


def to_dev(x, dev):
    if torch.is_tensor(x): return x.to(dev, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v, dev) for k, v in x.items()}
    return x


def collate1(x):
    return torch.utils.data.default_collate([x])


# ------------------------------------------------------------------------------------------------- (d) check_render
@torch.no_grad()
def check_render(clip_dir, n=10, dev=None, inf=None, out_jpg=None, size=512):
    """For rows of <clip>/gt_proj.npz: record -> patched EHM forward (smplx_only_forward, SMPL-X 2020) -> GUAVA's camera
    (per-sample invtanfov) -> 512 px; mean px error vs gt_proj (v2d_1024 converted with pixel centres at integers:
    u512 = (u1024 + 0.5) / 2 - 0.5), mask precision = fraction of rasterised mesh pixels (GUAVA's mesh rasteriser, per-sample
    focal) inside the matte (> 0.5), overlay jpg (image | mesh fill + gt_proj points)."""
    clip_dir = Path(clip_dir); dev = dev or torch.device("cuda:0")
    if inf is None: _, inf, _, _ = setup_a20(dev)
    g = np.load(clip_dir / "gt_proj.npz", allow_pickle=True)
    c = load_clip((str(clip_dir), None, True)); assert "error" not in c, c
    ds = FBData(get_cfg(), [c]); rows = []; tiles = []
    for i in range(min(n, len(g["frame_key"]))):
        k = str(g["frame_key"][i])
        if k not in c["recs"]: rows.append({"frame_key": k, "error": "not in records"}); continue
        t = to_dev(collate1(ds.target_info(0, k)), dev)
        itf = batch_invtanfov(t, 1, dev)
        out = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"]); v = out["vertices"][:, :NV]
        px, vc = project_px(v, t["w2c_cam"], itf, size)
        gt = torch.as_tensor(np.asarray(g["v2d_1024"][i], np.float32), device=dev)[None]
        gt512 = (gt + 0.5) * size / 1024 - 0.5
        err = (px - gt512).norm(dim=-1)
        p2f, _ = render_fragments_fb(inf.mesh_renderer, out["vertices"], t["w2c_cam"], itf)
        sil = (p2f[0, ..., 0] >= 0)
        matte = (t["mask"][0, 0] > 0.5)
        prec = float((sil & matte).sum() / sil.sum().clamp(min=1)); rec_ = float((sil & matte).sum() / matte.sum().clamp(min=1))
        # silhouette vs projected vertices consistency (mesh rasteriser vs projection formula): bbox difference in px
        ys, xs = torch.nonzero(sil, as_tuple=True)
        inside = (px[0, :, 0] >= 0) & (px[0, :, 0] <= size - 1) & (px[0, :, 1] >= 0) & (px[0, :, 1] <= size - 1)
        bbox_d = None
        if len(xs) and bool(inside.all()):
            bbox_d = float(torch.stack([xs.min() - px[0, :, 0].min(), xs.max() - px[0, :, 0].max(), ys.min() - px[0, :, 1].min(), ys.max() - px[0, :, 1].max()]).abs().max())
        d = (px - gt512)[0]
        rows.append({"frame_key": k, "invtanfov": float(itf[0]), "mean_px_err_512": float(err.mean()), "p90_px_err_512": float(err.quantile(0.9)),
                     "max_px_err_512": float(err.max()), "mean_dx": float(d[:, 0].mean()), "mean_dy": float(d[:, 1].mean()),
                     "mask_precision": prec, "mask_recall": rec_, "sil_vs_proj_bbox_px": bbox_d})
        if len(tiles) < 4:
            img = (t["image"][0] * t["mask"][0]).clone()
            ov = img.clone(); ov[1][sil] = ov[1][sil] * 0.4 + 0.6; ov[0][sil] *= 0.6
            q = gt512[0].round().long().clamp(0, size - 1); ov[:, q[::7, 1], q[::7, 0]] = torch.tensor([1.0, 0.0, 0.0], device=dev)[:, None]
            q = px[0].round().long().clamp(0, size - 1); ov[:, q[::7, 1], q[::7, 0]] = torch.tensor([0.0, 0.0, 1.0], device=dev)[:, None]
            tiles.append(torch.cat([img, ov], 2))
    ok = [r for r in rows if "error" not in r]
    summ = {"clip": str(clip_dir), "rows": rows, "n": len(ok),
            "mean_px_err_512": float(np.mean([r["mean_px_err_512"] for r in ok])) if ok else None,
            "mean_mask_precision": float(np.mean([r["mask_precision"] for r in ok])) if ok else None,
            "pass_lt_1px": bool(ok) and float(np.mean([r["mean_px_err_512"] for r in ok])) < 1.0}
    if tiles:
        out_jpg = Path(out_jpg) if out_jpg else clip_dir / "check_render.jpg"
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(torch.cat(tiles, 1), out_jpg); summ["overlay"] = str(out_jpg)
    return summ


# ------------------------------------------------------------------------------------------------- regression check
def _metrics(r, g, lp):
    return [float(cal_psnr(r, g)), float(cal_ssim(r, g)), float(lp(r, g)), float((r - g).abs().mean())]


@torch.no_grad()
def regression(dev):
    from submodules.lpipsPyTorch import LPIPS
    t0 = time.time()
    cfg, inf, ren, meta = setup_a20(dev, camera_patch=False); inf.eval(); ren.eval()
    lp = LPIPS("vgg", "0.1").to(dev)
    trn, val = a20_split()
    face, eye_copy = load_face_conv()
    conv = {"face": {(c, k): v for c, d in face.items() for k, v in d.items()}, "eye_copy": eye_copy}
    ref = A20MultiClip(cfg, val, "valid", conv)
    fb_clips = load_clips_parallel([(str(c), face[c.name], eye_copy) for c in val])
    fbd = FBData(cfg, fb_clips)
    res = {"U": [], "P": [], "P24": []}; maxdiff = {"P": 0.0, "P24": 0.0}; nf = 0
    for ci in range(len(ref.clips)):
        idx = [i for i, (c, _) in enumerate(ref.frames) if c == ci]
        assert fb_clips[ci]["keys"] == ref.keys[ci]
        b0 = to_dev(collate1(ref[idx[0]]), dev)
        unpatch_inferer(inf); torch.manual_seed(0); vg, ug, _ = inf(b0["source"]); avU = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avU.init_ehm(inf.ehm); avU.eval()
        patch_inferer(inf)
        s0 = to_dev(collate1(fbd.source_info(ci, fb_clips[ci]["keys"][0])), dev)
        s0nd = {k: v for k, v in s0.items() if k != "invtanfov"}                  # "absent" -> default 24 path
        torch.manual_seed(0); vg, ug, _ = inf(s0nd); avP = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avP.init_ehm(inf.ehm); avP.eval()
        torch.manual_seed(0); vg, ug, _ = inf(s0); avP24 = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avP24.init_ehm(inf.ehm); avP24.eval()
        for i in idx:
            b = to_dev(collate1(ref[i]), dev); t = b["target"]
            torch.manual_seed(0); rU = ren(avU(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)
            g = t["image"] * t["mask"]; res["U"].append(_metrics(rU, g, lp))
            k = ref.frames[i][1]; tf = to_dev(collate1(fbd.target_info(ci, k)), dev)
            for name, av in (("P", avP), ("P24", avP24)):
                torch.manual_seed(0); rP = ren(av(tf), tf["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)
                gP = tf["image"] * tf["mask"]; res[name].append(_metrics(rP, gP, lp)); maxdiff[name] = max(maxdiff[name], float((rP - rU).abs().max()))
            nf += 1
    summ = {}
    for name, v in res.items():
        a = np.asarray(v); summ[name] = dict(zip(("psnr", "ssim", "lpips", "l1"), a.mean(0).tolist()))
    d = {n: {m: summ[n][m] - summ["U"][m] for m in summ["U"]} for n in ("P", "P24")}
    passed = all(abs(d[n]["psnr"]) <= 0.01 and abs(d[n]["ssim"]) <= 1e-4 and abs(d[n]["lpips"]) <= 1e-4 for n in d)
    a20_logged = {"psnr": 24.694196731281554, "ssim": 0.8213276935242095, "lpips": 0.1566265010374252, "l1": 0.037904362998495984}
    out = {"note": "StyleUNet UV decoder injects random noise: every inf() call is preceded by torch.manual_seed(0) in all arms "
                   "(a20's validate() left it unseeded, so its logged numbers carry that noise)",
           "what": "a20 best.pt on a20's 10 UBody val clips: U = unpatched GUAVA (a20 loader, fixed invtanfov 24); P = camera patch, "
                   "new loader, invtanfov absent (default 24); P24 = camera patch, record invtanfov = 24.0 per sample",
           "frames": nf, "val_clips": [c.name for c in val], "unpatched": summ["U"], "patched_default": summ["P"], "patched_24": summ["P24"],
           "delta": d, "max_abs_pixel_diff_vs_unpatched": maxdiff, "a20_logged_val_iter2000": a20_logged,
           "unpatched_minus_logged": {m: summ["U"][m] - a20_logged[m] for m in a20_logged},
           "criterion": "|dPSNR| <= 0.01 dB, |dSSIM| <= 1e-4, |dLPIPS| <= 1e-4", "pass": bool(passed), "seconds": time.time() - t0}
    out["model"] = {k: v for k, v in meta.items() if k not in ("missing", "unexpected")} | {"missing_n": [len(x) for x in meta["missing"]], "unexpected_n": [len(x) for x in meta["unexpected"]]}
    out["synthetic"] = synthetic_checks(dev, cfg, inf, ren, fbd, fb_clips)
    return out


@torch.no_grad()
def synthetic_checks(dev, cfg, inf, ren, fbd, clips):
    """Projection math under non-default invtanfov: independent float64 pinhole vs GUAVA's 4 code paths, and avatar renders at
    invtanfov 24 vs perspective-equivalent cameras (camera moved along its axis so the pelvis keeps its pixel and scale)."""
    patch_inferer(inf)
    ci, k = 0, clips[0]["keys"][len(clips[0]["keys"]) // 2]
    s0 = to_dev(collate1(fbd.source_info(0, clips[0]["keys"][0])), dev)
    torch.manual_seed(0); vg, ug, _ = inf(s0); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    t = to_dev(collate1(fbd.target_info(ci, k)), dev)
    out = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"]); V = out["vertices"]; pelvis = out["joints"][:, 0]
    w2c0 = t["w2c_cam"]; z0 = float((w2c0[0, :3, :3] @ pelvis[0] + w2c0[0, :3, 3])[2])
    S = 512; res = {"frame": f"{clips[0]['name']}/{k}", "pelvis_depth_m": z0, "cases": []}
    r24 = None; sil24 = None
    for itf in (24.0, 12.0, 6.0, 2.5):
        w2c = w2c0.clone(); w2c[0, 2, 3] += z0 * itf / 24.0 - z0                  # move along the optical axis
        itf_t = torch.tensor([itf], device=dev)
        # independent pinhole (float64): u = f x / z + c, f = itf * S / 2, c = (S - 1) / 2 (pixel centres at integers)
        Vh = torch.cat([V[0], torch.ones_like(V[0, :, :1])], -1).double(); vc = (w2c[0].double() @ Vh.T).T[:, :3]
        uv_ref = itf * S / 2 * vc[:, :2] / vc[:, 2:] + (S - 1) / 2
        # GUAVA path 3 (sample_prj_feature formula) and 4 (convert_pixel_feature_to_uv, same formula on UV texels)
        _, vimg = sample_prj_feature_fb(inf, V, torch.zeros(1, 1, 8, 8, device=dev), w2c, invtanfov=itf_t)
        uv3 = (vimg[0, :, :2].double() + 1) * S / 2 - 0.5
        # GUAVA path 2: mesh renderer camera (GS_Camera, per-sample focal), screen convention (pixel edges) -> -0.5
        cams = build_mesh_cameras(inf.mesh_renderer, w2c, itf_t)
        scr = cams.transform_points_screen(V)[0, :, :2].double() - 0.5
        # GUAVA path 1: Gaussian rasteriser with the record camera: one small opaque Gaussian per probe vertex, intensity centroid
        rcp = render_cam_params(w2c[0].cpu(), torch.linalg.inv(w2c[0]).cpu(), itf, S); rcp = to_dev(collate1(rcp), dev)
        inimg = torch.nonzero(((uv_ref >= 8) & (uv_ref <= S - 9)).all(-1)).flatten()       # probes / bbox: in-image vertices only
        probe = inimg[torch.linspace(0, len(inimg) - 1, 24, device=dev).long()]; cen = []
        for vi in probe:
            ga = {"xyz": V[:, vi:vi + 1], "opacity": torch.ones(1, 1, 1, device=dev), "scaling": torch.full((1, 1, 3), 2.0 * vc[vi, 2].item() / (itf * S / 2), device=dev),   # sigma ~ 2 px
                  "rotation": torch.tensor([[[1.0, 0, 0, 0]]], device=dev), "features_color": torch.ones(1, 1, 32, device=dev)}
            img = ren.forward_raw(ga, rcp, bg=0.0)["raw_renders"][0, 0]
            w = img.double(); ys, xs = torch.meshgrid(torch.arange(S, device=dev).double(), torch.arange(S, device=dev).double(), indexing="ij")
            cen.append(torch.stack([(w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()]) if w.sum() > 0 else torch.full((2,), float("nan"), device=dev).double())
        cen = torch.stack(cen)
        # mesh silhouette (path 2 rasteriser) vs projected bbox
        p2f, _ = render_fragments_fb(inf.mesh_renderer, V, w2c, itf_t); sil = p2f[0, ..., 0] >= 0
        ys_, xs_ = torch.nonzero(sil, as_tuple=True)
        q = uv_ref[inimg].round().long(); sild = torch.nn.functional.max_pool2d(sil[None, None].float(), 3, 1, 1)[0, 0] > 0
        on_sil = float(sild[q[:, 1], q[:, 0]].float().mean())   # projected vertices on the (1 px dilated) mesh silhouette
        bb = 1.0 - on_sil
        # avatar render
        torch.manual_seed(0); rr = ren(av(t), rcp, bg=0.0)["renders"].clamp(0, 1)
        case = {"invtanfov": itf, "cam_depth_shift_m": z0 * itf / 24.0 - z0,
                "max_px_formula_vs_pinhole": float((uv3 - uv_ref).norm(dim=-1).max()),
                "max_px_meshcam_vs_pinhole": float((scr - uv_ref).norm(dim=-1).max()),
                "mean_px_gauss_centroid_vs_pinhole": float((cen - uv_ref[probe]).norm(dim=-1).nanmean().item()),
                "gauss_centroid_err_max": float((cen - uv_ref[probe]).norm(dim=-1).max()),
                "frac_inimage_vertices_off_mesh_silhouette_1px": float(bb)}
        if itf == 24.0:
            r24, sil24, uv24 = rr, sil, uv_ref
        else:
            case["pelvis_px_shift_vs_24"] = 0.0
            case["mean_vertex_px_diff_vs_24 (perspective)"] = float((uv_ref - uv24).norm(dim=-1).mean())
            case["render_psnr_vs_24"] = float(cal_psnr(rr, r24)); case["silhouette_iou_vs_24"] = float((sil & sil24).sum() / (sil | sil24).sum())
        res["cases"].append(case)
        torchvision.utils.save_image(rr, O / "reports/camera_patch" / f"synthetic_itf{itf:g}.jpg")
    # mixed batch (per-sample loop path of GS_Camera.get_projection_transform) == individual
    W = torch.cat([w2c0, w2c0.clone()]); W[1, 2, 3] += z0 * 2.5 / 24 - z0; itf2 = torch.tensor([24.0, 2.5], device=dev)
    pm, _ = render_fragments_fb(inf.mesh_renderer, V.expand(2, -1, -1).contiguous(), W, itf2)
    pa, _ = render_fragments_fb(inf.mesh_renderer, V, W[:1], itf2[:1]); pb, _ = render_fragments_fb(inf.mesh_renderer, V, W[1:], itf2[1:])
    nf = inf.mesh_renderer.faces.shape[0]
    res["mixed_batch_mesh_equal"] = bool(torch.equal(pm[0] >= 0, pa[0] >= 0) and torch.equal(pm[1] >= 0, pb[0] >= 0)
                                         and torch.equal(torch.where(pm[1] >= 0, pm[1] - nf, pm[1]), pb[0]))
    tol = 0.05
    res["pass"] = bool(all(c["max_px_formula_vs_pinhole"] < tol and c["max_px_meshcam_vs_pinhole"] < tol and c["gauss_centroid_err_max"] < 0.1
                           and c["frac_inimage_vertices_off_mesh_silhouette_1px"] < 0.01 for c in res["cases"]) and res["mixed_batch_mesh_equal"])
    return res


# ------------------------------------------------------------------------------------------------- self-test clip
def export_selftest(clip_name="SignLanguage_S62", root=O / "data/_selftest", dev=None):
    """Re-export one a20 UBody val clip in the GUAVA-FB format (a20 face conversion folded in, invtanfov 24, view "0"),
    image / mask LMDB payloads copied byte for byte, gt_proj.npz = independent pinhole projection of the a20 mesh."""
    import lmdb
    src = TRACKED / clip_name; dst = root / clip_name; dst.mkdir(parents=True, exist_ok=True)
    face, eye_copy = load_face_conv(); fc = face[clip_name]
    d = load_pkl(src / "optim_tracking_ehm.pkl"); ids = load_pkl(src / "id_share_params.pkl")
    recs = {}
    for k in sorted(d):
        r = d[k]; sc = r["smplx_coeffs"]
        if k not in fc: continue
        e50, jaw = fc[k]
        eye = np.asarray(r["flame_coeffs"]["eye_pose_params"], np.float32).reshape(2, 3) if eye_copy else np.zeros((2, 3), np.float32)
        smplx = {"global_pose": np.asarray(sc["global_pose"], np.float32), "body_pose": np.asarray(sc["body_pose"], np.float32),
                 "left_hand_pose": np.asarray(sc["left_hand_pose"], np.float32), "right_hand_pose": np.asarray(sc["right_hand_pose"], np.float32),
                 "exp": np.asarray(e50, np.float32).reshape(50), "jaw_pose": np.asarray(jaw, np.float32).reshape(1, 3), "eye_pose": eye,
                 "camera_RT_params": np.asarray(sc["camera_RT_params"], np.float32)}
        z50 = np.zeros(50, np.float32)
        recs[k] = {"smplx_coeffs": smplx,
                   "flame_coeffs": {"expression_params": z50, "jaw_params": np.zeros(3, np.float32), "pose_params": np.zeros(3, np.float32),
                                    "neck_pose_params": np.zeros(3, np.float32), "eye_pose_params": np.zeros(6, np.float32),
                                    "eyelid_params": np.zeros(2, np.float32), "camera_RT_params": smplx["camera_RT_params"].copy(), "cam": np.zeros(3, np.float32)},
                   **{c: deepcopy(r[c]) for c in CROP_KEYS}, "invtanfov": 24.0, "view": "0"}
    with open(dst / "optim_tracking_ehm.pkl", "wb") as f: pickle.dump(recs, f)
    with open(dst / "id_share_params.pkl", "wb") as f:
        pickle.dump({"smplx_shape": np.asarray(ids["smplx_shape"], np.float32).reshape(1, 200), "flame_shape": np.zeros((1, 300), np.float32),
                     "left_mano_shape": np.zeros((1, 10), np.float32), "right_mano_shape": np.zeros((1, 10), np.float32),
                     "head_scale": np.ones((1, 3), np.float32), "hand_scale": np.ones((1, 3), np.float32),
                     "joints_offset": np.zeros((1, 55, 3), np.float32)}, f)
    ks = sorted(recs)
    (dst / "videos_info.json").write_text(json.dumps({clip_name: {"frames_num": len(ks), "frames_keys": ks}}, indent=1))
    se = lmdb.open(str(src / "img_lmdb"), readonly=True, lock=False); de = lmdb.open(str(dst / "img_lmdb"), map_size=1 << 32)
    with se.begin() as st, de.begin(write=True) as dt:
        for k in ks:
            for part in ("body_image", "body_mask"):
                dt.put(f"{k}/{part}".encode(), st.get(f"{k}/{part}".encode()))
    se.close(); de.close()
    # gt_proj: independent numpy pinhole of the a20 mesh, 1024 crop (f = invtanfov * L / 2, c = (L - 1) / 2)
    dev = dev or torch.device("cuda:0")
    _, inf, _, _ = setup_a20(dev)
    c = load_clip((str(dst), None, True)); ds = FBData(get_cfg(), [c]); sel = ks[::max(1, len(ks) // 10)][:10]; v2d = []
    with torch.no_grad():
        for k in sel:
            t = to_dev(collate1(ds.target_info(0, k)), dev); V = inf.ehm(t["smplx_coeffs"], t["flame_coeffs"])["vertices"][0, :NV].double().cpu().numpy()
            w2c = t["w2c_cam"][0].double().cpu().numpy(); vc = V @ w2c[:3, :3].T + w2c[:3, 3]
            v2d.append((24.0 * 1024 / 2 * vc[:, :2] / vc[:, 2:] + 1023 / 2).astype(np.float32))
    np.savez(dst / "gt_proj.npz", frame_key=np.array(sel), v2d_1024=np.stack(v2d))
    (root / "index.json").write_text(json.dumps({"source": "_selftest", "licence": "UBody (research use; a20 data re-exported)",
        "clips": [{"name": clip_name, "subject": clip_name, "split": "train", "frames": len(ks), "views": ["0"]}],
        "notes": "self-test: a20 UBody val clip re-exported in the GUAVA-FB format (fb_common.export_selftest); not training data"}, indent=1))
    del inf; torch.cuda.empty_cache()
    return dst


@torch.no_grad()
def selftest(dev, clip_name="SignLanguage_S62"):
    dst = O / "data/_selftest" / clip_name
    if not (dst / "gt_proj.npz").exists(): export_selftest(clip_name, dev=dev)
    cfg, inf, ren, _ = setup_a20(dev, camera_patch=False); inf.eval(); ren.eval()
    face, eye_copy = load_face_conv()
    ref = A20MultiClip(cfg, [TRACKED / clip_name], "valid", {"face": {(clip_name, k): v for k, v in face[clip_name].items()}, "eye_copy": eye_copy})
    c = load_clip((str(dst), None, True)); fbd = FBData(cfg, [c])
    assert c["keys"] == ref.keys[0], "key sets differ"
    b0 = to_dev(collate1(ref[0]), dev)
    unpatch_inferer(inf); torch.manual_seed(0); vg, ug, _ = inf(b0["source"]); avU = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avU.init_ehm(inf.ehm); avU.eval()
    patch_inferer(inf); s0 = to_dev(collate1(fbd.source_info(0, c["keys"][0])), dev)
    torch.manual_seed(0); vg, ug, _ = inf(s0); avP = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avP.init_ehm(inf.ehm); avP.eval()
    md, n = 0.0, 0; inputs_equal = True
    for i in range(len(ref)):
        b = to_dev(collate1(ref[i]), dev); t = b["target"]; k = ref.frames[i][1]
        tf = to_dev(collate1(fbd.target_info(0, k)), dev)
        inputs_equal &= bool(torch.equal(t["image"], tf["image"]) and torch.equal(t["mask"], tf["mask"]) and torch.equal(t["head_box"], tf["head_box"])
                             and torch.equal(t["render_cam_params"]["full_proj_transform"], tf["render_cam_params"]["full_proj_transform"]))
        torch.manual_seed(0); rU = ren(avU(t), t["render_cam_params"], bg=0.0)["renders"]
        torch.manual_seed(0); rP = ren(avP(tf), tf["render_cam_params"], bg=0.0)["renders"]
        md = max(md, float((rU - rP).abs().max())); n += 1
    cr = check_render(dst, 10, dev, inf, out_jpg=O / "reports/camera_patch/selftest_check_render.jpg")
    out = {"clip": clip_name, "frames_compared": n, "max_abs_render_diff_new_loader_vs_a20_loader": md, "inputs_bit_equal": inputs_equal,
           "pass": md < 1e-5 and inputs_equal, "check_render": cr}
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("regression", "selftest", "check_render", "synthetic"))
    ap.add_argument("clip", nargs="?"); ap.add_argument("n", nargs="?", type=int, default=10)
    a = ap.parse_args(); dev = torch.device("cuda:0"); (O / "reports/camera_patch").mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    if a.cmd == "regression":
        r = regression(dev); p = O / "reports/camera_patch/regression.json"
    elif a.cmd == "selftest":
        r = selftest(dev); p = O / "reports/camera_patch/selftest.json"
    elif a.cmd == "synthetic":                       # re-run only the synthetic part, merged into regression.json
        p = O / "reports/camera_patch/regression.json"; r = json.loads(p.read_text())
        cfg, inf, ren, _ = setup_a20(dev); inf.eval(); ren.eval(); trn, val = a20_split(); face, eye_copy = load_face_conv()
        cl = load_clips_parallel([(str(c), face[c.name], eye_copy) for c in val]); r["synthetic"] = synthetic_checks(dev, cfg, inf, ren, FBData(cfg, cl), cl)
    else:
        r = check_render(a.clip, a.n, dev); p = Path(a.clip) / "check_render.json"
    p.write_text(json.dumps(r, indent=1, default=float)); print(json.dumps(r, indent=1, default=float)[:6000]); print("wrote", p)
