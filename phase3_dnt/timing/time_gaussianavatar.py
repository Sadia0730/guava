"""Timing of GaussianAvatar (Hu et al., CVPR 2024; per-subject, trained on the person's video) for the paper's runtime table.

LAPTOP NOTE (RTX 3080 Laptop, 16 GB)
Reuse the matching evaluated environment, code and assets; set the paths in
laptop.env.example.sh. Laptop GPU memory and speed remain unmeasured. Server
CUDA extensions may lack sm_86 kernels; use the method's pinned installation
procedure to build compatible extensions in a fresh build directory. The
wrappers do not rebuild dependencies automatically or change precision.
See TIMING_ON_LAPTOP.md for the short protocol, thermal guard and boundaries.

What is timed (official code d981c62, run exactly as phase3_dnt/baselines/gaussianavatar/run_clip.py ran it; that folder's
helper scripts are imported or launched unchanged):
  reconstruction (one end-to-end run, n = 1; steps as subprocesses like run_clip.py):
    preprocess    = ga_convert.py (package -> GaussianAvatar data folder; masks = the package's StyleMatte masks)
                    + gen_pose_map_cano_smpl.py (canonical 512 query posmap + joint mats, GLUT under Xvfb)
                    + export_stage_1_smpl.py (refined training poses) + gen_pose_map_our_smpl.py (128 pose posmaps of the
                    refined training poses = stage-2 inputs, GLUT under Xvfb). Each also as preprocess_<step>.
    segmentation  = null (package masks used), pose = null (GT SMPL given), network = null (no feed-forward predictor).
    training      = optimisation loops of train.py stage 1 (with training-pose refinement from epoch 11) + stage 2
                    (pose encoder), 200 epochs x 41 steps each (batch 2, 83 recon frames); loop = from the creation of
                    train.py's progress bar (after model / data / LPIPS / checkpoint setup) to the end of the loop incl.
                    the final checkpoint save; also training_stage1 / training_stage2; training_overhead = the rest of
                    the two train.py processes (python start, imports, AvatarModel + dataset + LPIPS construction).
    total         = wall time from the start of the conversion to the end of stage 2 (= run.json "train_s").
  animation (per driving frame, trained stage-2 model, GT SMPL + K of the test frame, batch 1, fp32, 512 px, no_grad,
             modules in train mode exactly as ga_render.py / the official eval.py):
    preprocess    = preprocess_posmap: posed SMPL on the CPU + 128 x 128 UV position map with the official
                    gen_pose_map_our_smpl.render_posmap (GLUT/OpenGL under Xvfb; the stage-2 pose encoder's input),
                    + preprocess_camera: K -> FoV / view / projection matrices as MonoDataset_test.getitem, collate, to_cuda.
                    (The offline script's .obj / .npz file round trip is replaced by the same arrays in memory; checked
                    against the script's own files every run: details.animation_checks.)
    deformation   = ga_render.render_test_stage2 (the evaluated test-render function, called unchanged) up to its call
                    of gaussian_renderer.render_batch: GPU SMPL -> joint transforms, pose-encoder UNet, POP decoder
                    (offsets / scales / colours on the 512 UV grid), LBS of ~200k points.
    rendering     = render_batch (3DGS rasteriser) + stack; boundary found by wrapping gaussian_renderer.render_batch.
    total         = preprocess + deformation + rendering end to end.

  python time_gaussianavatar.py --root SUBSET_ROOT --out OUT_DIR [--clip neuman/bike] [--warmup 10 --iters 50]
                                [--max-steps N (test only: stop each training stage after N steps)] [--keep-work]
"""
import sys

sys.dont_write_bytecode = True                         # nothing written next to the sources (repo, install tree)

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                                   # phase3_dnt/timing
GA_DIR = HERE.parent / "baselines" / "gaussianavatar"                     # evaluated helpers (ga_*.py, run_clip.py)
BASE = os.environ.get("GA_BASE", "/dev/shm/ubx858_baselines/gaussianavatar")
REPO = f"{BASE}/src/GaussianAvatar"
PY = f"{BASE}/env/bin/python"
COMMIT = "d981c62238ef64e89dcc04719d2ebbb4758b080a"
EPOCHS = 200                                                              # official default, both stages
XVFB = ["xvfb-run", "-a", "-s", "-screen 0 640x480x24"]                   # as run_clip.py
# evaluated run (publicbench_v1/renders/gaussianavatar/neuman/bike/run.json, L40S, 2026-11-02), for comparison only
EVAL_REF = {"gpu": "NVIDIA L40S", "train_s": 2927.1, "iterations": 16400, "convert": 16.47, "cano": 4.6, "stage1": 1433.99,
            "export": 2.05, "inpmap": 9.91, "stage2": 1460.08, "render_gpu_per_frame_s": 0.0664,
            "note": "subprocess wall times; render_gpu_per_frame = render_test_stage2 only, posmaps made offline"}
REBUILD_HINT = ("The rasteriser may lack kernels for this GPU. Rebuild its pinned extension for sm_86 "
                "in a fresh build directory using the method's installation instructions; see TIMING_ON_LAPTOP.md.")


# ----------------------------------------------------------------------------------------------------------------------
# worker: one train.py stage (ga_train.py unchanged) with loop timing, peak memory and the --max-steps early stop
# ----------------------------------------------------------------------------------------------------------------------
def worker_train(argv):
    i = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--rec", required=True)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--save-epoch", type=int, default=EPOCHS)
    wa, train_args = p.parse_args(argv[:i]), argv[i + 1:]
    t_start = time.perf_counter()
    os.chdir(REPO)
    sys.path[:0] = [REPO, str(GA_DIR)]
    import torch
    import tqdm as tqdm_mod
    import ga_compat
    ga_compat.apply()                                   # as ga_train.py does (it calls apply() again: no-op)
    import model.avatar_model as amm

    rec = {"steps": 0, "early_stop": False}
    holder = {}

    orig_init = amm.AvatarModel.__init__

    def init(self, *x, **k):                            # keep a handle on train.py's model (for the early-stop save)
        orig_init(self, *x, **k)
        holder["m"] = self
    amm.AvatarModel.__init__ = init

    class LoopStartTqdm(tqdm_mod.tqdm):                 # train.py creates its progress bar right before the epoch loop
        def __init__(self, *x, **k):
            if k.get("desc") == "Training progress" and "t_loop0" not in rec:
                torch.cuda.synchronize()
                rec["t_loop0"] = time.perf_counter()
                try:
                    rec["steps_scheduled"] = len(x[0])
                except Exception:                       # noqa: BLE001
                    pass
            super().__init__(*x, **k)
    tqdm_mod.tqdm = LoopStartTqdm

    class _Stop(Exception):
        pass

    orig_step = amm.AvatarModel.step

    def step(self, epoch):                              # called once per optimisation step by train.py
        orig_step(self, epoch)
        rec["steps"] += 1
        rec["last_epoch"] = epoch
        if wa.max_steps and rec["steps"] >= wa.max_steps:
            raise _Stop()
    amm.AvatarModel.step = step

    sys.argv = [str(GA_DIR / "ga_train.py"), REPO] + train_args
    import ga_train
    try:
        ga_train.main()
    except _Stop:
        rec["early_stop"] = True
        holder["m"].save(wa.save_epoch)                 # the save train.py does after its last epoch
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    rec.update(setup_in_process_s=rec["t_loop0"] - t_start, loop_s=t_end - rec["t_loop0"],
               peak_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
               peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    Path(wa.rec).write_text(json.dumps(rec, indent=1))
    print("TIMING", json.dumps(rec), flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# worker: per-frame animation timing (runs under xvfb-run because the posmap renderer is GLUT)
# ----------------------------------------------------------------------------------------------------------------------
def worker_anim(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--epoch", type=int, default=EPOCHS)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rec", required=True)
    p.add_argument("--png-dir", default="")
    p.add_argument("--png-n", type=int, default=0)
    wa = p.parse_args(argv)
    os.chdir(REPO)
    sys.path[:0] = [REPO, str(GA_DIR), f"{REPO}/scripts", str(HERE)]
    import numpy as np
    import torch
    import trimesh
    from argparse import ArgumentParser
    from PIL import Image
    from torch.utils.data import default_collate
    import ga_compat
    ga_compat.apply()
    import ga_render                                     # evaluated test-render function (render_test_stage2)
    import gaussian_renderer
    from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
    from model.avatar_model import AvatarModel
    from utils.general_utils import safe_state, to_cuda
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
    import smplx as smplx_pip                            # the package gen_pose_map_our_smpl.py imports (pip smplx 0.1.28)
    import gen_pose_map_our_smpl as gpm                  # official posmap script (its __main__ block is not run)
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    from phase_timer import Phases

    # ---- model setup exactly as ga_render.main ----
    sys.argv = ["ga_render.py", "-m", wa.model, "--epoch", str(wa.epoch)]
    parser = ArgumentParser(description="Testing script parameters")
    model, network, op = ModelParams(parser, sentinel=True), NetworkParams(parser), OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    assert args.train_stage == 2
    safe_state(args.quiet)
    mp, npar, opar = model.extract(args), network.extract(args), op.extract(args)
    dev = torch.device("cuda:0")
    rec = {}
    torch.set_grad_enabled(False)                        # ga_render.py runs everything under torch.no_grad()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    am = AvatarModel(mp, npar, opar, train=False)
    am.training_setup()
    am.stage2_load(args.epoch)
    ds = am.getTestDataset()
    torch.cuda.synchronize(); rec["model_load_s"] = time.perf_counter() - t0
    names = [n for _, n in ds.name_list]
    n = len(names)

    # ---- driving inputs held in memory (the "given" GT SMPL + camera of each test frame) ----
    test_dir = Path(wa.data) / "test"
    smpl_data = torch.load(test_dir / "smpl_parms_pred.pth")          # = GT test SMPL (run_clip.py copies smpl_parms.pth)
    cams = []
    for k in names:
        c = np.load(test_dir / "cam_parms" / f"{k}.npz")
        cams.append((np.array(c["extrinsic"]), np.array(c["intrinsic"])))
    W_img, H_img = Image.open(test_dir / "images" / f"{names[0]}.png").size
    assert (W_img, H_img) == (512, 512), (W_img, H_img)
    # per-subject constants of gen_pose_map_our_smpl.py (built once, as the script does before its frame loop)
    smpl_cpu = smplx_pip.SMPL(model_path=f"{REPO}/assets/smpl_files/smpl", batch_size=1)
    _, _, uvs, faces_uvs = load_obj_mesh(f"{REPO}/assets/template_mesh_smpl_uv.obj", with_texture=True)

    def make_posmap(i):
        """gen_pose_map_our_smpl.save_obj + save_npz for frame i (same calls; arrays in memory instead of .obj/.npz)."""
        out = smpl_cpu.forward(betas=smpl_data["beta"], global_orient=smpl_data["body_pose"][i, :3][None],
                               transl=smpl_data["trans"][i, :][None], body_pose=smpl_data["body_pose"][i, 3:][None])
        mesh = trimesh.Trimesh(out.vertices.detach().cpu().numpy().squeeze(), smpl_cpu.faces, process=False)
        posmap128, _, _ = gpm.render_posmap(mesh.vertices, mesh.faces, uvs, faces_uvs, img_size=128)
        return posmap128

    def make_item(i, posmap):
        """MonoDataset_test.getitem without the GT-image / file reads (driving needs pose + camera only)."""
        extr, intr = cams[i]
        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)
        K = np.array(intr, np.float32).reshape(3, 3)
        FovY, FovX = focal2fov(K[1, 1], H_img), focal2fov(K[0, 0], W_img)
        wvt = torch.tensor(getWorld2View2(R, T, ds.trans, ds.scale)).transpose(0, 1)
        proj = getProjectionMatrix(znear=ds.znear, zfar=ds.zfar, fovX=FovX, fovY=FovY, K=K, h=H_img, w=W_img).transpose(0, 1)
        return {"inp_pos_map": posmap.transpose(2, 0, 1), "FovX": FovX, "FovY": FovY, "width": W_img, "height": H_img,
                "pose_idx": i, "pose_data": ds.pose_data[i], "transl_data": ds.transl_data[i],
                "world_view_transform": wvt, "projection_matrix": proj,
                "full_proj_transform": (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0),
                "camera_center": wvt.inverse()[3, :3]}

    def model_input(i, posmap):
        return to_cuda(default_collate([make_item(i, posmap)]), device=dev)

    # ---- checks (untimed): in-memory path == official files / loader ----
    used = ["inp_pos_map", "FovX", "FovY", "width", "height", "pose_data", "transl_data", "world_view_transform",
            "full_proj_transform", "camera_center"]
    pm_diff, cam_diff, img_diff = 0.0, 0.0, 0.0
    for i, k in enumerate(names):
        pm = make_posmap(i)
        off = np.load(test_dir / "inp_map" / f"inp_posemap_128_{str(i).zfill(8)}.npz")["posmap128"]
        pm_diff = max(pm_diff, float(np.abs(pm - off).max()))
        b_off = to_cuda(default_collate([ds[i]]), device=dev)               # official loader (reads the script's npz)
        b_new = model_input(i, pm)
        for key in used[1:]:
            cam_diff = max(cam_diff, float((b_off[key].double() - b_new[key].double()).abs().max()))
        if i < 3:
            img_diff = max(img_diff, float((ga_render.render_test_stage2(am, b_off)
                                            - ga_render.render_test_stage2(am, b_new)).abs().max()))
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VERSION
    rec["checks"] = {"posmap_max_abs_diff_vs_official_script_npz": pm_diff,
                     "camera_pose_fields_max_abs_diff_vs_official_loader": cam_diff,
                     "render_max_abs_diff_official_loader_vs_timed_path_first3": img_diff,
                     "gl_renderer": glGetString(GL_RENDERER).decode(), "gl_version": glGetString(GL_VERSION).decode()}
    print("CHECKS", json.dumps(rec["checks"]), flush=True)

    # ---- timed loop ----
    st = {"t_r": None}
    orig_rb = gaussian_renderer.render_batch

    def render_batch_marked(*x, **k):                   # render_test_stage2 imports render_batch at call time
        if st["t_r"] is None:
            torch.cuda.synchronize()
            st["t_r"] = time.perf_counter()
        return orig_rb(*x, **k)
    gaussian_renderer.render_batch = render_batch_marked

    torch.cuda.reset_peak_memory_stats()
    P = Phases()
    for it in range(wa.warmup + wa.iters):
        i = it % n
        P.start_run(record=it >= wa.warmup)
        with P("preprocess"):
            with P("preprocess_posmap"):
                pm = make_posmap(i)
            with P("preprocess_camera"):
                batch = model_input(i, pm)
        torch.cuda.synchronize(); t0 = time.perf_counter(); st["t_r"] = None
        image = ga_render.render_test_stage2(am, batch)[0]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        assert st["t_r"] is not None and tuple(image.shape) == (3, 512, 512), image.shape
        P.add("deformation", (st["t_r"] - t0) * 1000)
        P.add("rendering", (t1 - st["t_r"]) * 1000)
        P.end_run()
    rec["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
    rec["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    rec["summary"] = P.summary()
    rec["n_test_keys"] = n
    gaussian_renderer.render_batch = orig_rb

    if wa.png_dir and wa.png_n > 0:                       # a few sanity renders (same 8-bit conversion as ga_render.py)
        os.makedirs(wa.png_dir, exist_ok=True)
        for i in range(min(wa.png_n, n)):
            img = ga_render.render_test_stage2(am, model_input(i, make_posmap(i)))[0]
            arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(arr).save(os.path.join(wa.png_dir, f"{names[i]}.png"))
    Path(wa.rec).write_text(json.dumps(rec, indent=1))
    print("TIMING", json.dumps(rec["summary"]), flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# worker: preflight (rasteriser kernels load on this GPU)
# ----------------------------------------------------------------------------------------------------------------------
def worker_preflight(argv):
    sys.path.insert(0, REPO)
    import math
    import torch
    rec = {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
           "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        from gaussian_renderer import render_batch
        from utils.graphics_utils import getProjectionMatrix, focal2fov
        K = torch.tensor([[500.0, 0, 256], [0, 500.0, 256], [0, 0, 1]])
        fov = focal2fov(500.0, 512)
        wvt = torch.eye(4)
        proj = getProjectionMatrix(0.01, 100.0, fov, fov, K.numpy(), 512, 512).transpose(0, 1)
        n = 64
        img = render_batch(points=torch.randn(n, 3, device="cuda") * 0.2 + torch.tensor([0, 0, 3.0], device="cuda"),
                           shs=None, colors_precomp=torch.rand(n, 3, device="cuda"),
                           rotations=torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(n, 1),
                           scales=torch.full((n, 3), 0.02, device="cuda"), opacity=torch.ones(n, 1, device="cuda"),
                           FovX=fov, FovY=fov, height=512, width=512, bg_color=torch.zeros(3, device="cuda"),
                           world_view_transform=wvt.cuda(), full_proj_transform=(wvt @ proj).cuda(),
                           active_sh_degree=0, camera_center=torch.zeros(3, device="cuda"))
        torch.cuda.synchronize()
        rec["raster_ok"] = bool(img.shape == (3, 512, 512) and float(img.sum()) > 0)
        rec["raster_sum"] = float(img.sum())
    except Exception as e:                                # noqa: BLE001
        rec["raster_ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    print("PREFLIGHT", json.dumps(rec), flush=True)
    Path(argv[0]).write_text(json.dumps(rec))


# ----------------------------------------------------------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------------------------------------------------------
class GpuMemPoller(threading.Thread):
    """Peak nvidia-smi memory (MiB) of this process tree, tagged by the step that is running."""
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval, self.step, self.peak, self.ok = interval, None, {}, None
        self._halt = threading.Event()

    def _tree(self):
        me, kids = os.getpid(), {}
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
                except Exception:                         # noqa: BLE001
                    continue
                kids.setdefault(ppid, []).append(int(d))
        out, todo = set(), [me]
        while todo:
            p = todo.pop()
            for c in kids.get(p, []):
                if c not in out:
                    out.add(c); todo.append(c)
        return out

    def run(self):
        while not self._halt.is_set():
            step = self.step
            if step:
                try:
                    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout
                    tree = self._tree()
                    used = 0
                    for line in q.strip().splitlines():
                        pid, mem = [s.strip() for s in line.split(",")[:2]]
                        if pid.isdigit() and int(pid) in tree and mem.isdigit():
                            used += int(mem)
                    if used:
                        self.ok = True
                        self.peak[step] = max(self.peak.get(step, 0), used)
                except Exception:                         # noqa: BLE001
                    pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="timing subset root (has manifest.json)")
    ap.add_argument("--clip", default="neuman/bike")
    ap.add_argument("--out", required=True, help="output dir (-> <out>/gaussianavatar.json) or a .json path")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="TEST ONLY: stop each training stage after N optimisation steps (default 0 = evaluated schedule, "
                         "200 epochs per stage)")
    ap.add_argument("--png", type=int, default=3, help="sanity renders written under <out>/gaussianavatar_png (0 = none)")
    ap.add_argument("--work-root", default="/dev/shm", help="work dir parent (deleted at the end)")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE))
    from phase_timer import PROTOCOL, stats, write

    out = Path(a.out)
    out_json = out if out.suffix == ".json" else out / "gaussianavatar.json"
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    man = json.loads((root / "manifest.json").read_text())
    ent = [c for c in man["clips"] if c["id"] == a.clip]
    assert ent, f"{a.clip} not in {root}/manifest.json"
    ds_name, clip = a.clip.split("/")
    IN = root / ent[0]["E"] / "base" / ds_name / clip
    fj = json.loads((IN / "frames.json").read_text())
    n_recon, n_test = len(fj["recon"]), len(fj["test"])
    full = fj.get("timing_subset_full", {})
    assert not full or full.get("recon") == n_recon, f"subset trimmed the recon frames ({n_recon} of {full.get('recon')})"
    for exe in ("xvfb-run", "nvidia-smi"):
        assert shutil.which(exe), f"{exe} not found (xvfb-run: apt package xvfb)"
    assert os.path.exists(PY) and os.path.isdir(REPO), f"GaussianAvatar install not found under {BASE}"

    W = Path(a.work_root) / f"ubx858_timing_work_gaussianavatar_{os.getpid()}"
    if W.exists():
        raise FileExistsError(W)
    D, S1, S2, LOG = W / "data", W / "out" / "stage1", W / "out" / "stage2", W / "logs"
    for d in (LOG, W / "tmp", W / "xdg"):
        d.mkdir(parents=True, exist_ok=True)
    mesa = Path(BASE) / "cache" / "xdg" / "mesa_shader_cache"            # warm Mesa shader cache, as in the evaluated run
    if mesa.is_dir():
        shutil.copytree(mesa, W / "xdg" / "mesa_shader_cache")
    # environment as run_clip.py gives its steps; TMPDIR / XDG cache point into W (copies) so nothing is written into
    # the install tree; no .pyc files are written anywhere.
    env = dict(os.environ, PIP_CACHE_DIR=f"{BASE}/cache/pip", TORCH_HOME=f"{BASE}/cache/torch", HF_HOME=f"{BASE}/cache/hf",
               XDG_CACHE_HOME=str(W / "xdg"), TMPDIR=str(W / "tmp"), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", LD_LIBRARY_PATH=f"{BASE}/env/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    env.pop("DISPLAY", None)
    me = str(Path(__file__).resolve())
    T = {}
    mon = GpuMemPoller()
    mon.start()

    def run(name, cmd, xvfb=False):
        if xvfb:
            cmd = XVFB + cmd
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        mon.step = name
        t = time.perf_counter()
        with open(LOG / f"{name}.log", "w") as lf:
            r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        T[name] = time.perf_counter() - t
        mon.step = None
        if r.returncode != 0:
            print((LOG / f"{name}.log").read_text()[-4000:])
            raise SystemExit(f"step {name} failed ({r.returncode}); log: {LOG}/{name}.log")
        print(f"   {name}: {T[name]:.1f} s", flush=True)

    try:
        # ---- preflight ----
        pf = W / "preflight.json"
        run("preflight", [PY, me, "--_worker", "preflight", str(pf)])
        pre = json.loads(pf.read_text())
        if not pre.get("raster_ok"):
            raise SystemExit(f"preflight failed: {pre}\n{REBUILD_HINT}")

        cano = f"{REPO}/scripts/gen_pose_map_cano_smpl.py"
        ours = f"{REPO}/scripts/gen_pose_map_our_smpl.py"
        e_last = EPOCHS
        tw = ["--max-steps", str(a.max_steps), "--save-epoch", str(e_last)]
        ep = ["--save_epoch", str(e_last)]                                  # as run_clip.py (only the final model saved)

        # ---- reconstruction: one end-to-end run, same steps / arguments as run_clip.py ----
        t0 = time.perf_counter()
        run("convert", [PY, str(GA_DIR / "ga_convert.py"), "--clip_dir", str(IN), "--out", str(D), "--splits", "train"])
        run("cano", [PY, str(GA_DIR / "ga_official.py"), cano, "smplx_parm_path = ''", f"smplx_parm_path = '{D}/train'"], xvfb=True)
        run("stage1", [PY, me, "--_worker", "train", "--rec", str(W / "stage1.json")] + tw + ["--", "-s", str(D), "-m", str(S1),
                       "--train_stage", "1", "--pose_op_start_iter", "10", "--cam_static", "0", "--no_mask", "1"] + ep)
        run("export", [PY, str(GA_DIR / "ga_official.py"), f"{REPO}/scripts/export_stage_1_smpl.py",
                       "net_save_path = ''", f"net_save_path = '{S1}/net/iteration_{e_last}'",
                       "smpl_parms_path = ''", f"smpl_parms_path = '{D}/train'"])
        run("inpmap", [PY, str(GA_DIR / "ga_official.py"), ours, "smpl_parm_path = 'path to you data folder'",
                       f"smpl_parm_path = '{D}/train'", "parms_name = 'smpl_parms.pth'", "parms_name = 'smpl_parms_pred.pth'"],
            xvfb=True)
        run("stage2", [PY, me, "--_worker", "train", "--rec", str(W / "stage2.json")] + tw + ["--", "-s", str(D), "-m", str(S2),
                       "--train_stage", "2", "--stage1_out_path", f"{S1}/net/iteration_{e_last}", "--cam_static", "0",
                       "--no_mask", "1"] + ep)
        recon_total = time.perf_counter() - t0
        r1, r2 = (json.loads((W / f"stage{s}.json").read_text()) for s in (1, 2))

        # ---- animation setup (untimed, per subject; as run_clip.py's test preparation) ----
        run("test_convert", [PY, str(GA_DIR / "ga_convert.py"), "--clip_dir", str(IN), "--out", str(D), "--splits", "test"])
        run("test_cano", [PY, str(GA_DIR / "ga_official.py"), cano, "smplx_parm_path = ''", f"smplx_parm_path = '{D}/test'"],
            xvfb=True)
        shutil.copyfile(D / "test" / "smpl_parms.pth", D / "test" / "smpl_parms_pred.pth")   # GT test poses
        run("test_inpmap", [PY, str(GA_DIR / "ga_official.py"), ours, "smpl_parm_path = 'path to you data folder'",
                            f"smpl_parm_path = '{D}/test'", "parms_name = 'smpl_parms.pth'",
                            "parms_name = 'smpl_parms_pred.pth'"], xvfb=True)   # only for the in-memory posmap check
        # ---- animation: per-frame timing ----
        png_dir = out_dir / "gaussianavatar_png" if a.png > 0 else ""
        run("animation", [PY, me, "--_worker", "anim", "--data", str(D), "--model", str(S2), "--epoch", str(e_last),
                          "--warmup", str(a.warmup), "--iters", str(a.iters), "--rec", str(W / "anim.json"),
                          "--png-dir", str(png_dir), "--png-n", str(a.png)], xvfb=True)
        an = json.loads((W / "anim.json").read_text())
    finally:
        mon.stop()
        logs_out = out_dir / "gaussianavatar_logs"
        if LOG.exists():
            shutil.rmtree(logs_out, ignore_errors=True)
            shutil.copytree(LOG, logs_out)
        if not a.keep_work:
            shutil.rmtree(W, ignore_errors=True)
            print("deleted", W, flush=True)

    one = lambda s: stats([s * 1000.0])                                    # noqa: E731  (n = 1 run, ms)
    sched = (n_recon // 2) * EPOCHS
    pre_keys = {"preprocess_convert": "convert", "preprocess_cano_posmap": "cano", "preprocess_export_poses": "export",
                "preprocess_pose_posmaps": "inpmap"}
    recon = {"preprocess": one(sum(T[v] for v in pre_keys.values())), **{k: one(T[v]) for k, v in pre_keys.items()},
             "segmentation": None, "pose": None, "network": None,
             "training": one(r1["loop_s"] + r2["loop_s"]), "training_stage1": one(r1["loop_s"]),
             "training_stage2": one(r2["loop_s"]),
             "training_overhead": one(T["stage1"] - r1["loop_s"] + T["stage2"] - r2["loop_s"]),
             "total": one(recon_total)}
    s = an["summary"]
    anim = {"pose": None, "segmentation": None, "preprocess": s["preprocess"], "preprocess_posmap": s["preprocess_posmap"],
            "preprocess_camera": s["preprocess_camera"], "deformation": s["deformation"], "rendering": s["rendering"],
            "rendering_raster": None, "rendering_refiner": None, "update": None, "total": s["total"]}
    full_sched = not a.max_steps or a.max_steps >= sched
    mib = lambda x: None if x is None else round(x)                         # noqa: E731
    details = {
        "schedule": {"epochs_per_stage": EPOCHS, "train_batch": 2, "recon_frames": n_recon, "test_frames": n_test,
                     "steps_per_stage_scheduled": sched, "steps_run": {"stage1": r1["steps"], "stage2": r2["steps"]},
                     "max_steps_arg": a.max_steps or None, "full_schedule": bool(full_sched),
                     "early_stop": {"stage1": r1["early_stop"], "stage2": r2["early_stop"]}},
        "subprocess_s": {k: round(v, 2) for k, v in T.items()},
        "train_setup_in_process_s": {"stage1": round(r1["setup_in_process_s"], 2), "stage2": round(r2["setup_in_process_s"], 2)},
        "train_s_per_step": {"stage1": round(r1["loop_s"] / max(r1["steps"], 1), 4),
                             "stage2": round(r2["loop_s"] / max(r2["steps"], 1), 4)},
        "peak_gpu_mem_mib": {
            "stage1": {"torch_allocated": mib(r1["peak_alloc_mib"]), "torch_reserved": mib(r1["peak_reserved_mib"]),
                       "nvidia_smi_process": mon.peak.get("stage1")},
            "stage2": {"torch_allocated": mib(r2["peak_alloc_mib"]), "torch_reserved": mib(r2["peak_reserved_mib"]),
                       "nvidia_smi_process": mon.peak.get("stage2")},
            "animation": {"torch_allocated": mib(an["peak_alloc_mib"]), "torch_reserved": mib(an["peak_reserved_mib"]),
                          "nvidia_smi_process": mon.peak.get("animation")},
            "posmap_steps_nvidia_smi": {k: mon.peak.get(k) for k in ("cano", "inpmap")},
            "note": "nvidia-smi = whole process (incl. CUDA context), sampled every 1 s; null if nvidia-smi lists no PIDs"},
        "animation_setup": {"model_load_s": round(an["model_load_s"], 2), **an["checks"]},
        "preflight": pre,
        "evaluated_run_reference": EVAL_REF,
    }
    clip_notes = (
        f"GaussianAvatar @ {COMMIT[:7]}, official own-video recipe exactly as evaluated (run_clip.py): 2 stages x {EPOCHS} "
        f"epochs, batch 2, {n_recon} recon frames -> {sched} steps per stage"
        + ("" if full_sched else f"; TEST RUN: stopped after {r1['steps']} + {r2['steps']} steps (--max-steps {a.max_steps}), "
           "training numbers are NOT the evaluated schedule") +
        f". Animation: {a.warmup} warm-up + {a.iters} timed frames cycling over the {an['n_test_keys']} test keys, GT SMPL + K "
        f"of each test frame, batch 1, fp32, 512 px. GL renderer for the pose posmap: {an['checks']['gl_renderer']}.")
    notes = (
        "Reconstruction = per-subject training on the recon frames (one end-to-end run, n = 1). preprocess = ga_convert.py "
        "(package -> method folders) + official gen_pose_map_cano_smpl.py (canonical 512 posmap) + export_stage_1_smpl.py "
        "(refined training poses) + gen_pose_map_our_smpl.py (128 pose posmaps for stage 2); GLUT scripts under Xvfb as "
        "evaluated. segmentation null (the package's StyleMatte masks are used), pose null (GT SMPL given; the training-pose "
        "refinement runs inside stage 1 and is part of training), network null. training = the two train.py optimisation "
        "loops (progress-bar creation to end of loop incl. final checkpoint save); training_overhead = rest of the two "
        "train.py processes (start, imports, model / data / LPIPS setup, stage-1 checkpoint load). total = conversion start "
        "to end of stage 2 (= run.json train_s). Animation per frame: preprocess = CPU SMPL + official render_posmap "
        "(128 UV position map of the posed body, GLUT/OpenGL under Xvfb; input of the stage-2 pose encoder) + camera "
        "matrices / collate / to_cuda; deformation = ga_render.render_test_stage2 up to gaussian_renderer.render_batch "
        "(GPU SMPL joint transforms, pose-encoder UNet, POP decoder on the 512 UV grid, LBS); rendering = render_batch "
        "(3DGS rasteriser, no refiner; rendering_raster / rendering_refiner null). No pose estimation, segmentation or "
        "appearance update per frame (null).")
    protocol = dict(PROTOCOL, warmup=a.warmup, iters=a.iters, recon_warmup=0, recon_reps=1, precision="fp32", render_px=512,
                    train_batch=2, train_steps_per_stage=sched, max_steps=a.max_steps or None)
    write(out_json, method="gaussianavatar", row="GaussianAvatar",
          clips={a.clip: {"reconstruction": recon, "animation": anim, "notes": clip_notes, "details": details}},
          notes=notes, protocol=protocol)
    print(json.dumps({"recon_total_s": round(recon_total, 1), "steps": details["schedule"]["steps_run"],
                      "anim_total_median_ms": anim["total"]["median_ms"], "peak_mem_mib": details["peak_gpu_mem_mib"]},
                     indent=1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--_worker":
        {"train": worker_train, "anim": worker_anim, "preflight": worker_preflight}[sys.argv[2]](sys.argv[3:])
    else:
        main()
