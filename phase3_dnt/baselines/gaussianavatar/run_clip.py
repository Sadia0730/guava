"""Train + test-render GaussianAvatar on one AvatarTest-v1 clip (official custom-video recipe, two stages).

  /dev/shm/ubx858_baselines/gaussianavatar/env/bin/python run_clip.py <dataset> <clip> [--gpu N] [--keep_work]

Steps (all under W = /dev/shm/ubx858_baselines/gaussianavatar/work/<dataset>/<clip>/):
  convert   ga_convert.py: recon keys -> data/train, test keys -> data/test (per-frame K, GT SMPL)
  cano      scripts/gen_pose_map_cano_smpl.py on data/train (canonical query posmap 512 + joint mats)
  stage1    train.py --train_stage 1 --pose_op_start_iter 10        (README "Training for Stage 1", custom video)
  export    scripts/export_stage_1_smpl.py                              (optimised training poses -> smpl_parms_pred.pth)
  inpmap    scripts/gen_pose_map_our_smpl.py on data/train, smpl_parms_pred.pth, 128   (stage-2 pose-encoder inputs)
  stage2    train.py --train_stage 2 --stage1_out_path <stage1>/net/iteration_200
  -- train_s = wall time of all steps above --
  test_prep data/test: canonical posmap (same script) + GT test SMPL -> smpl_parms_pred.pth + inp posmaps (same script)
  render    ga_render.py (GT test pose, test-frame K, black bg) -> E/renders/gaussianavatar/<dataset>/<clip>/<key>.png
Then run.json + small logs are copied to /vault and W is deleted (checkpoints never leave /dev/shm).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

BASE = "/dev/shm/ubx858_baselines/gaussianavatar"
REPO = f"{BASE}/src/GaussianAvatar"
PY = f"{BASE}/env/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get("EVAL_ROOT", "/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval/avatar_test_v1")   # EVAL_ROOT: other splits
COMMIT = "d981c62238ef64e89dcc04719d2ebbb4758b080a"
EPOCHS = 200   # OptimizationParams.epochs default (both stages)

LOADER_CHANGES = [
    "Per-frame intrinsics: the official loader's own per-frame camera path (--cam_static 0: cam_parms/<name>.npz with "
    "extrinsic = identity, intrinsic = K[key] of the 512 crop); no code change.",
    "Background black instead of white: images pre-composited on black with the mask (alpha > 127) in ga_convert.py and "
    "read with --no_mask 1 (the official loader's mask path composites on white with the same >= 128 threshold); "
    "white_background=False (renderer background black) set by ga_train.py because the CLI flag is store_true with "
    "default True. Training and test background = the registered black-background GT.",
    "Test rendering: ga_render.py = official render_free_stage2 with the pose/translation taken from the test frame's GT "
    "SMPL (as render_free_stage1 does) instead of the training pose embedding indexed by the test index; stage-2 "
    "checkpoint loaded with stage2_load. Test-frame posmaps made by the official gen_pose_map_our_smpl.py.",
    "Checkpointing: --save_epoch 200 so that the final epoch-200 model is written (official save rule saves only epochs "
    "> 100 divisible by save_epoch; default 30 never saves epoch 200, while the released People-Snapshot model is "
    "iteration_200). No effect on training.",
    "Run-time compatibility shims, no numerical effect (ga_compat.py, ga_train.py): AvatarModel.__init__ indexes CPU "
    "tensors (np.load'ed query posmap / LBS map) with the CUDA mask valid_idx, which pytorch 2.0.1 (environment.yml) "
    "rejects -> the mask is handed to __init__ on the CPU and moved back to CUDA afterwards; stage 2 runs with "
    "TensorBoard logging off (train.py logs scale_loss, defined only in stage 1 -> UnboundLocalError), via train.py's "
    "own TENSORBOARD_FOUND=False fallback.",
    "Official scripts with hard-coded paths (gen_pose_map_cano_smpl.py, export_stage_1_smpl.py, gen_pose_map_our_smpl.py) "
    "run unchanged except their path lines (ga_official.py); gen_pose_map_our_smpl.py parms_name = smpl_parms_pred.pth "
    "(README: 'generate the predicted position map'). GLUT position-map renderer run under Xvfb (xvfb-run).",
]


def gpu_free(i):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    used = {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.strip().splitlines()}
    return used[i] < 2048, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("clip")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--keep_work", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only (default: official 200)")
    ap.add_argument("--out_root", default=f"{E}/renders/gaussianavatar")
    a = ap.parse_args()

    ok, used = gpu_free(a.gpu)
    if not ok:
        cand = [g for g in (2, 0, 1, 3) if used[g] < 2048]
        assert cand, f"no free GPU among 0-3: {used}"
        print(f"GPU {a.gpu} busy ({used[a.gpu]} MiB); using GPU {cand[0]}", flush=True)
        a.gpu = cand[0]
    gpu_name = subprocess.check_output(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=name", "--format=csv,noheader"], text=True).strip()

    IN = f"{E}/base/{a.dataset}/{a.clip}"
    W = f"{BASE}/work/{a.dataset}/{a.clip}"
    OUT = f"{a.out_root}/{a.dataset}/{a.clip}"
    D, S1, S2 = f"{W}/data", f"{W}/out/stage1", f"{W}/out/stage2"
    if os.path.exists(W):
        shutil.rmtree(W)
    os.makedirs(f"{W}/logs", exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PIP_CACHE_DIR=f"{BASE}/cache/pip", TORCH_HOME=f"{BASE}/cache/torch",
               HF_HOME=f"{BASE}/cache/hf", XDG_CACHE_HOME=f"{BASE}/cache/xdg", TMPDIR=f"{BASE}/tmp", PYTHONNOUSERSITE="1",
               PYTHONUNBUFFERED="1", LD_LIBRARY_PATH=f"{BASE}/env/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    env.pop("DISPLAY", None)
    e_last = a.epochs or EPOCHS
    # save only the final model: official rule saves epochs > save_epochs[0] (=100) with epoch % save_epoch == 0
    ep = ["--save_epoch", str(e_last)] + ([] if a.epochs is None else ["--epochs", str(a.epochs), "--save_epochs", "0"])
    t_steps = {}

    def run(name, cmd, xvfb=False):
        if xvfb:
            cmd = ["xvfb-run", "-a", "-s", "-screen 0 640x480x24"] + cmd
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(cmd)}", flush=True)
        t = time.time()
        with open(f"{W}/logs/{name}.log", "w") as lf:
            r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        t_steps[name] = round(time.time() - t, 2)
        if r.returncode != 0:
            print(open(f"{W}/logs/{name}.log").read()[-4000:])
            raise SystemExit(f"step {name} failed ({r.returncode})")
        print(f"   {name}: {t_steps[name]:.1f} s", flush=True)

    cano = f"{REPO}/scripts/gen_pose_map_cano_smpl.py"
    ours = f"{REPO}/scripts/gen_pose_map_our_smpl.py"
    t0 = time.time()
    run("convert", [PY, f"{HERE}/ga_convert.py", "--clip_dir", IN, "--out", D, "--splits", "train"])
    run("cano", [PY, f"{HERE}/ga_official.py", cano, "smplx_parm_path = ''", f"smplx_parm_path = '{D}/train'"], xvfb=True)
    run("stage1", [PY, f"{HERE}/ga_train.py", REPO, "-s", D, "-m", S1, "--train_stage", "1", "--pose_op_start_iter", "10",
                   "--cam_static", "0", "--no_mask", "1"] + ep)
    run("export", [PY, f"{HERE}/ga_official.py", f"{REPO}/scripts/export_stage_1_smpl.py",
                   "net_save_path = ''", f"net_save_path = '{S1}/net/iteration_{e_last}'",
                   "smpl_parms_path = ''", f"smpl_parms_path = '{D}/train'"])
    run("inpmap", [PY, f"{HERE}/ga_official.py", ours, "smpl_parm_path = 'path to you data folder'", f"smpl_parm_path = '{D}/train'",
                   "parms_name = 'smpl_parms.pth'", "parms_name = 'smpl_parms_pred.pth'"], xvfb=True)
    run("stage2", [PY, f"{HERE}/ga_train.py", REPO, "-s", D, "-m", S2, "--train_stage", "2", "--stage1_out_path",
                   f"{S1}/net/iteration_{e_last}", "--cam_static", "0", "--no_mask", "1"] + ep)
    train_s = time.time() - t0

    t1 = time.time()
    run("test_convert", [PY, f"{HERE}/ga_convert.py", "--clip_dir", IN, "--out", D, "--splits", "test"])
    run("test_cano", [PY, f"{HERE}/ga_official.py", cano, "smplx_parm_path = ''", f"smplx_parm_path = '{D}/test'"], xvfb=True)
    shutil.copyfile(f"{D}/test/smpl_parms.pth", f"{D}/test/smpl_parms_pred.pth")   # GT test poses (no test-time optimisation)
    run("test_inpmap", [PY, f"{HERE}/ga_official.py", ours, "smpl_parm_path = 'path to you data folder'", f"smpl_parm_path = '{D}/test'",
                        "parms_name = 'smpl_parms.pth'", "parms_name = 'smpl_parms_pred.pth'"], xvfb=True)
    rdir = f"{W}/renders"
    run("render", [PY, f"{HERE}/ga_render.py", REPO, "-m", S2, "--epoch", str(e_last), "--out_dir", rdir,
                   "--times_json", f"{W}/logs/render_times.json"])
    test_s = time.time() - t1

    keys_test = json.load(open(f"{D}/test/keys.json"))
    keys_train = json.load(open(f"{D}/train/keys.json"))
    got = sorted(f[:-4] for f in os.listdir(rdir) if f.endswith(".png"))
    assert got == sorted(keys_test), (len(got), len(keys_test))
    rt = json.load(open(f"{W}/logs/render_times.json"))
    n_tr = len(keys_train)
    it_per_stage = (n_tr // 2) * e_last      # batch 2, drop_last
    run_json = {
        "method": "gaussianavatar", "commit": COMMIT,
        "train_s": round(train_s, 1),
        "render_s_per_frame": round((t_steps["test_cano"] + t_steps["test_inpmap"] + rt["loop_s"]) / len(keys_test), 4),
        "gpu": gpu_name, "iterations": 2 * it_per_stage,
        "notes": (f"Official custom-video recipe, 2 stages x {e_last} epochs, batch 2, default hyper-parameters; stage 1 with "
                  f"training-pose refinement from epoch 11 (--pose_op_start_iter 10, README); stage 2 = pose-dependent "
                  f"appearance (pose encoder). {n_tr} recon frames -> {it_per_stage} iterations per stage. Driving = GT "
                  f"SMPL of each test frame, no test-time pose optimisation. render_s_per_frame = (test canonical posmap + "
                  f"test pose posmaps + render loop incl. PNG write) / {len(keys_test)} test frames; model load excluded "
                  f"({rt['load_s']:.1f} s); pure GPU render {rt['gpu_s_per_frame']*1000:.1f} ms/frame. "
                  f"CUDA_VISIBLE_DEVICES={a.gpu}." + (f" SMOKE TEST epochs={a.epochs}." if a.epochs else "")),
        "loader_changes": LOADER_CHANGES,
        "timing_s": {**t_steps, "train_total": round(train_s, 1), "test_total": round(test_s, 1),
                     "render_model_load": round(rt["load_s"], 2), "render_loop": round(rt["loop_s"], 2),
                     "render_gpu_per_frame": round(rt["gpu_s_per_frame"], 5)},
        "frames": {"recon": n_tr, "test": len(keys_test)},
        "epochs_per_stage": e_last, "gpu_index": a.gpu,
    }
    os.makedirs(OUT, exist_ok=True)
    for k in keys_test:
        shutil.copyfile(f"{rdir}/{k}.png", f"{OUT}/{k}.png")
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    for f in sorted(os.listdir(f"{W}/logs")):
        shutil.copyfile(f"{W}/logs/{f}", f"{OUT}/logs/{f}")
    for s, sd in (("stage1", S1), ("stage2", S2)):
        if os.path.exists(f"{sd}/cfg_args"):
            shutil.copyfile(f"{sd}/cfg_args", f"{OUT}/logs/{s}_cfg_args.txt")
    with open(f"{OUT}/run.json", "w") as f:
        json.dump(run_json, f, indent=1)
    print(json.dumps({k: run_json[k] for k in ("train_s", "render_s_per_frame", "gpu", "iterations")}), flush=True)
    if not a.keep_work:
        shutil.rmtree(W)
        print("deleted", W, flush=True)


if __name__ == "__main__":
    main()
