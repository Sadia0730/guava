"""Per-frame GUAVA path profiling (server): GPU-busy vs wall time per stage (launch-bound?), top CUDA ops, and the
refiner in fp16 autocast / channels_last with the output difference vs fp32 (same input, same noise seed)."""
import os, sys, json, statistics
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT); dev = torch.device("cuda:0")
from omegaconf import OmegaConf
from dataset import TrackedData_infer
from models.UbodyAvatar import GaussianRenderer, Ubody_Gaussian_inferer, Ubody_Gaussian
from utils.general_utils import ConfigDict, add_extra_cfgs
cfg = add_extra_cfgs(ConfigDict(model_config_path=str(ROOT / "assets/GUAVA/config.yaml")))
inf = Ubody_Gaussian_inferer(cfg.MODEL).to(dev).eval(); ren = GaussianRenderer(cfg.MODEL).to(dev).eval()
ck = torch.load(ROOT / "assets/GUAVA/checkpoints/best_160000.pt", map_location="cpu", weights_only=True)
inf.load_state_dict(ck["model"], strict=False); ren.load_state_dict(ck["render_model"], strict=False); del ck
OmegaConf.set_readonly(cfg.DATASET, False); cfg.DATASET.data_path = "/raid/ubx858/outputs/dnt/c1/tracked/ConductMusic_S21_Trim8"
ds = TrackedData_infer(cfg=cfg, split="test", device=dev, test_full=True); vid = list(ds.videos_info.keys())[0]; keys = ds.videos_info[vid]["frames_keys"]
src = ds._load_source_info(vid, 0); tgt = ds._load_target_info(vid, keys[60])
with torch.no_grad():
    vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
    ga = av(tgt); raw = ren.forward_raw(ga, tgt["render_cam_params"], bg=0.0)
def wall(fn, n=50, w=10):
    ts = []
    for i in range(n + w):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        if i >= w: ts.append(e0.elapsed_time(e1))
    return statistics.fmean(ts)
from torch.profiler import profile, ProfilerActivity
def gpu_busy(fn, n=20):
    with torch.no_grad():
        for _ in range(5): fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
            for _ in range(n): fn()
            torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0) > 0]
    tot = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in p.key_averages()) / 1000 / n
    top = sorted(p.key_averages(), key=lambda e: -getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)))[:8]
    nk = sum(e.count for e in p.key_averages() if getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) > 0) / n
    return tot, nk, [(e.key[:60], round(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) / 1000 / n, 3)) for e in top]
R = {}
with torch.no_grad():
    for name, fn in (("deform", lambda: av(tgt)), ("raster", lambda: ren.forward_raw(ga, tgt["render_cam_params"], bg=0.0)), ("refiner", lambda: ren.refine_raw(raw)),
                     ("frame", lambda: ren(av(tgt), tgt["render_cam_params"], bg=0.0))):
        wt = wall(fn); busy, nk, top = gpu_busy(fn)
        R[name] = {"wall_ms": wt, "gpu_kernel_ms": busy, "kernels_per_call": nk, "top_ops": top}
    # refiner precision variants (same input, same noise seed)
    def ref(dtype=None, cl=False):
        torch.manual_seed(0)
        if dtype is None: return ren.refine_raw(raw)
        with torch.autocast("cuda", dtype=dtype): return ren.refine_raw(raw).float()
    base = ref()
    for nm, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        o = ref(dt); d = (o - base)
        psnr = float(10 * torch.log10(1 / (d.float() ** 2).mean().clamp_min(1e-12)))
        R[f"refiner_{nm}"] = {"wall_ms": wall(lambda: ref(dt)), "max_abs_diff": float(d.abs().max()), "psnr_vs_fp32_db": psnr}
print(json.dumps(R, indent=1))
Path("/raid/ubx858/outputs/dnt/e2/guava_profile_frame.json").write_text(json.dumps(R, indent=1))
