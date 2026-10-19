"""Per-frame GUAVA path: eager vs torch.compile(mode='reduce-overhead') (CUDA graphs) for the deformation and the refiner.
Equality check: compiled output vs eager output on the same input (refiner: same noise seed)."""
import os, sys, json, statistics, time
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
src = ds._load_source_info(vid, 0); tgts = [ds._load_target_info(vid, keys[i]) for i in (60, 61, 62)]
with torch.no_grad():
    vg, ug, _ = inf(src); av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
def wall(fn, n=50, w=10):
    ts = []
    for i in range(n + w):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        if i >= w: ts.append(e0.elapsed_time(e1))
    return statistics.fmean(ts)
R = {}
with torch.no_grad():
    t = tgts[0]
    deform_e = lambda tt: av(tt)["xyz"]
    ga = av(t); raw = ren.forward_raw(ga, t["render_cam_params"], bg=0.0)
    def ref_e(r):
        torch.manual_seed(0); return ren.refine_raw(r)
    R["eager"] = {"deform_ms": wall(lambda: av(t)), "refiner_ms": wall(lambda: ref_e(raw)), "frame_ms": wall(lambda: ren(av(t), t["render_cam_params"], bg=0.0))}
    t0 = time.time()
    av_c = torch.compile(av, mode="reduce-overhead", dynamic=False)
    ref_c = torch.compile(ren.refine_raw, mode="reduce-overhead", dynamic=False)
    for tt in tgts: av_c(tt)
    for _ in range(3): ref_c(raw)
    torch.cuda.synchronize(); R["compile_seconds"] = time.time() - t0
    dmax = max(float((av_c(tt)["xyz"] - av(tt)["xyz"]).abs().max()) for tt in tgts)
    torch.manual_seed(0); a1 = ref_c(raw).clone(); torch.manual_seed(0); b1 = ren.refine_raw(raw)
    R["compiled"] = {"deform_ms": wall(lambda: av_c(t)), "refiner_ms": wall(lambda: ref_c(raw)),
                     "frame_ms": wall(lambda: ren.refine_raw.__self__ and ref_c(ren.forward_raw(av_c(t), t["render_cam_params"], bg=0.0))),
                     "deform_max_abs_diff_m": dmax, "refiner_max_abs_diff": float((a1 - b1).abs().max()),
                     "refiner_psnr_vs_eager_db": float(10 * torch.log10(1 / ((a1 - b1) ** 2).mean().clamp_min(1e-12)))}
print(json.dumps(R, indent=1)); Path("/raid/ubx858/outputs/dnt/e2/guava_compile_test.json").write_text(json.dumps(R, indent=1))
