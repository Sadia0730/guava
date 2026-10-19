"""GUAVA per-component latency for ONE image (batch 1, fp32 as GUAVA runs, 512 px render), CUDA events, mean / p50 / p95
over --iters after --warmup. Two phases:
  A. avatar creation (once per source image): DINO encoder; global feature + cam dirs; EHM (source); vertex branch
     (projection sampling + vertex Gaussian decoder); visibility rasterization; UV inverse texture mapping; UV StyleUNet
     decoder; UV point Gaussian decoder (+ gather); Ubody_Gaussian assembly (+ pruning, EHM init).
  B. per driving frame: EHM (target params); Gaussian deformation (rest of Ubody_Gaussian.forward); Gaussian
     rasterization (forward_raw); neural refiner StyleUNet (refine_raw).
Input: frame 0 (source) and frame 60 (target) of one C1 test clip (EHM-tracked)."""
import argparse, json, os, sys, statistics
from pathlib import Path
import torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
ap = argparse.ArgumentParser(); ap.add_argument("--clip", default="/raid/ubx858/outputs/dnt/c1/tracked/ConductMusic_S21_Trim8"); ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--out", default=None); a = ap.parse_args(); dev = torch.device("cuda:0")
from omegaconf import OmegaConf
from dataset import TrackedData_infer
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian, Ubody_Gaussian_inferer
from models.UbodyAvatar.ubody_gaussian import get_cam_dirs
from utils.general_utils import ConfigDict, add_extra_cfgs
cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "assets/GUAVA/config.yaml")))
inf = Ubody_Gaussian_inferer(cfg.MODEL).to(dev).eval(); ren = GaussianRenderer(cfg.MODEL).to(dev).eval()
ck = torch.load(ROOT / "assets/GUAVA/checkpoints/best_160000.pt", map_location="cpu", weights_only=True)
inf.load_state_dict(ck["model"], strict=False); ren.load_state_dict(ck["render_model"], strict=False); del ck
OmegaConf.set_readonly(cfg.DATASET, False); cfg.DATASET.data_path = a.clip
ds = TrackedData_infer(cfg=cfg, split="test", device=dev, test_full=True); vid = list(ds.videos_info.keys())[0]
keys = ds.videos_info[vid]["frames_keys"]; src = ds._load_source_info(vid, 0); tgt = ds._load_target_info(vid, keys[min(60, len(keys) - 1)])
def timeit(fn):
    ts = []
    for i in range(a.warmup + a.iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); r = fn(); e1.record(); torch.cuda.synchronize()
        if i >= a.warmup: ts.append(e0.elapsed_time(e1))
    ts.sort(); return {"mean_ms": statistics.fmean(ts), "p50_ms": ts[len(ts) // 2], "p95_ms": ts[int(len(ts) * 0.95)]}, r
R = {}
with torch.no_grad():
    R["A1_dino_encoder"], d = timeit(lambda: inf.dino_encoder(src["image"], output_size=inf.cfg.image_size))
    R["A2_global_feature_and_cam_dirs"], _ = timeit(lambda: (inf.global_feature_mapping(d["f_global"]), inf.harmo_encoder(get_cam_dirs(src["w2c_cam"]))))
    gfe = inf.global_feature_mapping(d["f_global"]); cam = inf.harmo_encoder(get_cam_dirs(src["w2c_cam"]))
    R["A3_ehm_source"], deform = timeit(lambda: inf.ehm(src["smplx_coeffs"], src["flame_coeffs"]))
    V = deform["vertices"]
    def vbranch():
        vsf, _ = inf.sample_prj_feature(V, d["f_map2"], src["w2c_cam"])
        vsf = torch.cat([vsf, inf.vertex_base_feature[None], gfe[:, None].expand(-1, vsf.shape[1], -1)], -1)
        vg = inf.vertex_gs_decoder(vsf, cam); vg["positions"] = inf.v_template.clone()[None]; return vg
    R["A4_vertex_branch"], vg = timeit(vbranch)
    R["A5_visibility_raster"], (vis, _) = timeit(lambda: inf.mesh_renderer.render_fragments(V, transform_matrix=src["w2c_cam"]))
    feat = torch.cat([F.interpolate(src["image"], (inf.cfg.image_size,) * 2, mode="bilinear", align_corners=False), d["f_map1"]], 1)
    R["A6_uv_inverse_texture_mapping"], uv = timeit(lambda: inf.convert_pixel_feature_to_uv(feat, V, src["w2c_cam"], visble_faces=vis))
    R["A7_uv_stylenet_decoder"], x = timeit(lambda: inf.uv_feature_decoder(uv, extra_style=inf.uv_style_mapping(d["f_global"])))
    def uvpoint():
        y = torch.cat([x, inf.uv_base_feature[None]], 1); ug = inf.uv_point_decoder(y, cam)
        for k in list(ug.keys()): ug[k] = ug[k].reshape(1, inf.uvmap_size * inf.uvmap_size, -1)[:, inf.uv_mask_flat, :]
        ug["binding_face"] = inf.smplx.uvmap_f_idx.clone().reshape(1, -1, 1)[:, inf.uv_mask_flat, :]
        ug["face_bary"] = inf.smplx.uvmap_f_bary.clone().reshape(1, inf.uvmap_size * inf.uvmap_size, -1)[:, inf.uv_mask_flat, :]; return ug
    R["A8_uv_point_decoder"], ug = timeit(uvpoint)
    def assemble():
        av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); return av
    R["A9_avatar_assembly_pruning"], av = timeit(assemble)
    R["A_total_inferer_forward_reference"], _ = timeit(lambda: inf(src))
    R["B1_ehm_target"], _ = timeit(lambda: inf.ehm(tgt["smplx_coeffs"], tgt["flame_coeffs"]))
    R["B12_avatar_deform_total(incl. EHM)"], ga = timeit(lambda: av(tgt))
    R["B3_gaussian_rasterization"], raw = timeit(lambda: ren.forward_raw(ga, tgt["render_cam_params"], bg=0.0))
    R["B4_neural_refiner_stylenet"], _ = timeit(lambda: ren.refine_raw(raw))
    R["B_total_per_frame_reference"], _ = timeit(lambda: ren(av(tgt), tgt["render_cam_params"], bg=0.0))
n_g = int(ga["xyz"].shape[1]) if ga["xyz"].dim() == 3 else int(ga["xyz"].shape[0])
res = {"gpu": torch.cuda.get_device_name(0), "precision": "fp32", "batch": 1, "render_px": 512, "gaussians": n_g, "iters": a.iters, "stages": R}
print(json.dumps(res, indent=1))
if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
