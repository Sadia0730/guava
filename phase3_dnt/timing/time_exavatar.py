"""Laptop timing of ExAvatar (per-subject 3DGS avatar trained on the person's video): paper row "ExAvatar".

The paper row is the fairness-rule variant `exavatar_gtface` (results_table_tex.py reads renders/exavatar_gtface;
pipeline = phase3_dnt/baselines/exavatar/run_clip_gtface.sh): GT body, ExAvatar's own face fit on the recon frames only,
test-frame face = GT jaw / expression, no test-time optimisation. This driver re-runs exactly those stages (same scripts,
same official code at mks0601/ExAvatar_RELEASE@518f5647, same schedules) on the timing subset's neuman/bike clip
(all 83 recon frames, 10 test frames), times each stage as a subprocess, then times per-frame animation of the trained
avatar in-process (the render_test.py / animate.py path).

LAPTOP NOTE (RTX 3080 Laptop, 16 GB)
Reuse the matching evaluated environment, code and assets; set the paths in
laptop.env.example.sh. Laptop GPU memory and speed remain unmeasured. Server
CUDA extensions may lack sm_86 kernels; use the method's pinned installation
procedure to build compatible extensions in a fresh build directory. The
wrappers do not rebuild dependencies automatically or change precision.
See TIMING_ON_LAPTOP.md for the short protocol, thermal guard and boundaries.

Usage (normally via run_exavatar.sh):
  python time_exavatar.py --root SUBSET_ROOT --out OUT_DIR [--clip neuman/bike] [--warmup 10 --iters 50]
         [--max-steps N] [--max-epochs E] [--fit-max-iters N]   # test only: shortened schedules, recorded in the JSON
Nothing under the install (/dev/shm/ubx858_baselines/exavatar) is written: the ExAvatar source tree is mirrored into a
work dir under /dev/shm (real dirs, symlinked files, fresh output/data dirs), all caches and outputs go there, and it
is deleted at the end. Output: OUT_DIR/exavatar.json (phase_timer schema), OUT_DIR/exavatar_details.json (stage
seconds, memory, schedule), OUT_DIR/exavatar_png/<key>.png (a few test renders for a sanity look).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))
BASE = osp.join(REPO, 'phase3_dnt', 'baselines', 'exavatar')      # adapter scripts of the evaluated run
EXA_ROOT_DEFAULT = '/dev/shm/ubx858_baselines/exavatar'

# Written into the work dir and put on PYTHONPATH of every stage process (including the `python ...` processes that
# ExAvatar's tools spawn with os.system): records the torch allocator peak of each process that used CUDA.
SITECUSTOMIZE = r'''
import atexit, json, os, sys
def _exa_timing_mem():
    d, t = os.environ.get("EXA_TIMING_MEMDIR"), sys.modules.get("torch")
    if not d or t is None:
        return
    try:
        if not t.cuda.is_initialized() or t.cuda._is_in_bad_fork():
            return
        rec = {"pid": os.getpid(), "stage": os.environ.get("EXA_TIMING_STAGE", ""), "argv": sys.argv[:2],
               "max_allocated_mib": t.cuda.max_memory_allocated() / 2**20,
               "max_reserved_mib": t.cuda.max_memory_reserved() / 2**20}
        with open(os.path.join(d, "%d.json" % os.getpid()), "w") as f:
            json.dump(rec, f)
    except Exception:
        pass
atexit.register(_exa_timing_mem)
'''

# source-tree entries that runs write into: not mirrored (fresh empty dirs instead, or created by the scripts)
FRESH_DIRS = {'fitting/output', 'avatar/output', 'fitting/data/Custom/data', 'avatar/data/Custom/data'}
SKIP_DIRS = {'fitting/tools/DECA/flame_parmas_out', 'fitting/tools/mmpose/vis_results'}


# ----------------------------------------------------------------------------------------------------------------------
# helpers running inside the stage processes (exavatar env)
# ----------------------------------------------------------------------------------------------------------------------
class _Budget:
    """Wraps train.py's DataLoader: stops after max_steps optimisation steps in total (the rest of the epochs then run
    0 iterations and only save their snapshot); len() stays the full epoch length, so cur_itr / lr schedule / warm-up /
    densification of the steps that do run are exactly those of the full schedule. Also counts steps."""

    def __init__(self, dl, max_steps, st):
        self.dl, self.max, self.st = dl, max_steps, st

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        if self.max is not None and self.st['steps'] >= self.max:
            return
        for data in self.dl:
            now = time.perf_counter()
            self.st.setdefault('t_first', now)
            self.st['t_last'] = now
            self.st['steps'] += 1
            yield data
            if self.max is not None and self.st['steps'] >= self.max:
                break


def wrap_main(argv):
    """`python time_exavatar.py --_wrap SCRIPT ARGS...`: run SCRIPT as __main__ (as `python SCRIPT ARGS` would) after
    runtime patches selected by environment variables. Patches are no-ops for the default (evaluated) schedule."""
    import runpy
    script = osp.abspath(argv[0])
    sys.argv = [script] + argv[1:]
    sys.path[0] = osp.dirname(script)
    name = osp.basename(script)
    smplx2020 = os.environ.get('EXA_TIMING_SMPLX2020')
    if smplx2020 and name in ('prep_clip.py', 'finalize_params.py'):
        import exa_common                                     # GT SMPL-X 2020 model path (hard-coded in exa_common)
        exa_common.SMPLX2020 = smplx2020
    if name == 'fit_face_gt.py' and os.environ.get('EXA_TIMING_FIT_MAX_ITERS'):
        n = int(os.environ['EXA_TIMING_FIT_MAX_ITERS'])
        sys.path.insert(0, os.getcwd())                       # fitting/main, as fit_face_gt.py does itself
        import config
        cfg, orig = config.cfg, config.cfg.set_itr_opt_num

        def capped(epoch):
            orig(epoch)
            cfg.itr_opt_num = min(cfg.itr_opt_num, n)
        cfg.set_itr_opt_num = capped
    if name == 'train.py':
        import config                                         # avatar/main (= sys.path[0])
        cfg = config.cfg
        if os.environ.get('EXA_TIMING_MAX_EPOCHS'):
            cfg.end_epoch = min(cfg.end_epoch, int(os.environ['EXA_TIMING_MAX_EPOCHS']))
        ms = int(os.environ['EXA_TIMING_MAX_STEPS']) if os.environ.get('EXA_TIMING_MAX_STEPS') else None
        st = {'steps': 0, 'max_steps': ms, 'end_epoch': cfg.end_epoch}
        import base
        orig_mbg = base.Trainer._make_batch_generator

        def mbg(self):
            orig_mbg(self)
            st['itr_per_epoch'] = len(self.batch_generator)
            self.batch_generator = _Budget(self.batch_generator, ms, st)
        base.Trainer._make_batch_generator = mbg

        def dump():
            if os.environ.get('EXA_TIMING_TRAIN_STATS'):
                with open(os.environ['EXA_TIMING_TRAIN_STATS'], 'w') as f:
                    json.dump(st, f)
        atexit.register(dump)
    runpy.run_path(script, run_name='__main__')


def animate_main(a):
    """`--_animate`: per-frame animation timing of the trained avatar (cwd = avatar/main of the mirrored tree).
    Same calls as render_test.py (the evaluated renderer): HumanGaussian.forward -> refined Gaussians ->
    GaussianRenderer.forward, black background, the package's K (identity extrinsics), 512 x 512, fp32, no_grad."""
    import glob
    import numpy as np
    sys.path.insert(0, os.getcwd())
    from config import cfg
    cfg.set_args(a.sid)
    sys.path.insert(0, BASE)
    import exa_common
    if a.smplx2020:
        exa_common.SMPLX2020 = a.smplx2020
    from exa_common import EXA_ROOT, GTConverter, cam_json, key2idx, load_clip
    from base import Tester
    from utils.smpl_x import smpl_x
    import torch
    sys.path.insert(0, HERE)
    from phase_timer import Phases, sync

    fr, K, gt = load_clip(a.clip_dir)
    conv = GTConverter(gt['betas'])                            # per-subject setup, not per frame
    work = osp.join(EXA_ROOT, 'work', a.sid)
    so = osp.join(work, 'avatar', 'smplx_optimized')
    ld = lambda n: torch.FloatTensor(json.load(open(osp.join(so, n))))
    smpl_x.set_id_info(ld('shape_param.json'), ld('face_offset.json'), ld('joint_offset.json'), ld('locator_offset.json'))
    snaps = glob.glob(osp.join(cfg.model_dir, 'snapshot_*.pth'))
    epoch = max(int(osp.basename(s)[len('snapshot_'):-len('.pth')]) for s in snaps)

    sync(); t0 = time.perf_counter()
    tester = Tester(str(epoch))
    tester.smplx_params = None
    tester._make_model()
    hg = tester.model.module.human_gaussian
    renderer = tester.model.module.gaussian_renderer
    sync(); load_s = time.perf_counter() - t0
    pelvis_trained = conv.pelvis_ex(hg.shape_param.detach().cpu().numpy().astype(np.float64))
    scene_points = int(tester.model.module.scene_gaussian.mean.shape[0]) if hasattr(tester.model.module.scene_gaussian, 'mean') else None

    def frame_params(key):
        """GT pose of a test frame -> ExAvatar's SMPL-X layer (= finalize_params.py --gt_face_test + render_test.py):
        body converted from SMPL-X 2020, face = GT jaw / expression, eyes 0, trans re-anchored to the trained shape."""
        p = conv.frame_params(gt, key, pelvis_ex=pelvis_trained)
        return {'root_pose': p['root_pose'], 'body_pose': p['body_pose'], 'lhand_pose': p['lhand_pose'],
                'rhand_pose': p['rhand_pose'], 'trans': p['trans'], 'jaw_pose': gt['jaw_pose'][key].reshape(3),
                'leye_pose': np.zeros(3), 'reye_pose': np.zeros(3), 'expr': gt['expression'][key].reshape(-1)}

    # correctness check (untimed): in-memory conversion == the test_params/<idx>.json the evaluated pipeline wrote
    diff = 0.0
    for key in fr['test']:
        ref = json.load(open(osp.join(work, 'test_params', '%d.json' % key2idx(key))))
        mine = frame_params(key)
        for k, v in mine.items():
            if k != 'trans':                                   # trans: render_test.py re-anchors it the same way
                diff = max(diff, float(np.abs(np.asarray(v, np.float64).reshape(-1) - np.asarray(ref[k]).reshape(-1)).max()))

    keys = fr['test']
    bg = torch.zeros(3).float().cuda()
    torch.cuda.reset_peak_memory_stats()
    P = Phases()
    with torch.no_grad():
        for i in range(a.warmup + a.iters):
            key = keys[i % len(keys)]
            P.start_run(record=i >= a.warmup)
            with P('preprocess'):
                prm = frame_params(key)
                smplx_param = {k: torch.tensor(np.asarray(v, np.float64).reshape(-1), dtype=torch.float32).cuda()
                               for k, v in prm.items()}
                cam_param = {k: torch.tensor(v, dtype=torch.float32).cuda() for k, v in cam_json(K[key]).items()}
            with P('deformation'):
                _, asset, _, _ = hg(smplx_param, cam_param)
            with P('rendering'):
                img = renderer(asset, (512, 512), cam_param, bg)['img']
            P.end_run()
        loop_peak = {'max_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                     'max_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
        if a.png_dir and a.n_png > 0:
            import cv2
            os.makedirs(a.png_dir, exist_ok=True)
            for key in keys[:a.n_png]:
                prm = frame_params(key)
                smplx_param = {k: torch.tensor(np.asarray(v, np.float64).reshape(-1), dtype=torch.float32).cuda()
                               for k, v in prm.items()}
                cam_param = {k: torch.tensor(v, dtype=torch.float32).cuda() for k, v in cam_json(K[key]).items()}
                img = renderer(hg(smplx_param, cam_param)[1], (512, 512), cam_param, bg)['img']
                img = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
                cv2.imwrite(osp.join(a.png_dir, key + '.png'), img[:, :, ::-1])
    res = {'animation': P.summary(), 'model_load_s': load_s, 'test_epoch': epoch, 'test_keys': keys,
           'param_check_max_abs_diff': diff, 'human_gaussians': int(smpl_x.vertex_num_upsampled),
           'scene_gaussians_in_checkpoint': scene_points, 'loop_peak_mem': loop_peak,
           'gpu': torch.cuda.get_device_name(0)}
    with open(a.anim_json, 'w') as f:
        json.dump(res, f, indent=1)


# ----------------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------------
def build_mirror(exa_root, root):
    """root/ = a private EXA_ROOT: src/ExAvatar_RELEASE mirrored with real directories and symlinked files (weights and
    code are read through the links, every write lands in root/), fresh output/data dirs, __pycache__ copied,
    human_model_files linked. Returns the number of mirrored files."""
    src, dst = osp.join(exa_root, 'src', 'ExAvatar_RELEASE'), osp.join(root, 'src', 'ExAvatar_RELEASE')
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        rel = osp.relpath(dirpath, src)
        os.makedirs(osp.join(dst, rel), exist_ok=True)
        keep = []
        for d in dirnames:
            r = osp.normpath(osp.join(rel, d))
            s, t = osp.join(dirpath, d), osp.join(dst, r)
            if d == '.git' or r in SKIP_DIRS:
                continue
            if r in FRESH_DIRS:
                os.makedirs(t, exist_ok=True)
            elif osp.islink(s):
                os.symlink(os.readlink(s), t)
            elif d == '__pycache__':
                shutil.copytree(s, t)                         # real copies: warm byte-code, never written through
            else:
                keep.append(d)
        dirnames[:] = keep
        for fn in filenames:
            s, t = osp.join(dirpath, fn), osp.join(dst, rel, fn)
            if fn.endswith('.py'):                           # real copies: Python resolves a symlinked script to the install's
                shutil.copy2(osp.realpath(s), t)             # directory, so its config / output paths would point there
            else:
                os.symlink(os.readlink(s) if osp.islink(s) else s, t)
            n += 1
    os.symlink(osp.join(exa_root, 'human_model_files'), osp.join(root, 'human_model_files'))
    os.makedirs(osp.join(root, 'work'))
    return n


class GpuPoll:
    """Optional (--mem-poll): samples nvidia-smi every `dt` s and keeps, per stage, the max summed GPU memory of this
    driver's descendant processes (includes the CUDA context; misses peaks shorter than dt)."""

    def __init__(self, dt=0.5):
        self.dt, self.stage, self.peak, self._stop = dt, None, {}, threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _ancestor_is_me(self, pid):
        me = os.getpid()
        for _ in range(64):
            if pid == me:
                return True
            try:
                with open('/proc/%d/stat' % pid) as f:
                    pid = int(f.read().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                return False
            if pid <= 1:
                return False
        return False

    def _run(self):
        while not self._stop.is_set():
            try:
                q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=10).stdout
                tot = sum(int(m) for p, m in (l.split(',') for l in q.strip().splitlines() if ',' in l)
                          if m.strip().isdigit() and self._ancestor_is_me(int(p)))
                if self.stage:
                    self.peak[self.stage] = max(self.peak.get(self.stage, 0), tot)
            except Exception:                                                             # noqa: BLE001
                pass
            self._stop.wait(self.dt)

    def start(self):
        self.th.start()

    def stop(self):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', help='timing subset root (contains manifest.json)')
    ap.add_argument('--clip', default='neuman/bike')
    ap.add_argument('--out', help='output dir -> OUT/exavatar.json')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--max-steps', type=int, default=None, help='TEST ONLY: cap avatar-training steps (default: evaluated 5 x 15 x n_recon)')
    ap.add_argument('--max-epochs', type=int, default=None, help='TEST ONLY: cap avatar-training epochs (default 5; changes the lr schedule)')
    ap.add_argument('--fit-max-iters', type=int, default=None, help='TEST ONLY: cap face-fit iterations per batch per epoch (default 500/250/250)')
    ap.add_argument('--exa-root', default=os.environ.get('EXA_ROOT_ORIG', EXA_ROOT_DEFAULT), help='ExAvatar install (read only)')
    ap.add_argument('--work-root', default='/dev/shm', help='work dir parent (deleted at the end)')
    ap.add_argument('--smplx2020', default=None, help='SMPLX_NEUTRAL_2020.npz (default: exa_common path, else REPO/assets/SMPLX)')
    ap.add_argument('--png', type=int, default=2, help='test renders saved to OUT/exavatar_png (0 = none)')
    ap.add_argument('--mem-poll', action='store_true', help='also sample nvidia-smi per process (server memory measurement)')
    ap.add_argument('--keep-work', action='store_true', help='debug: keep the work dir')
    ap.add_argument('--power-note', default='')
    # internal modes (`--_wrap SCRIPT ARGS...` is handled before argparse)
    ap.add_argument('--_animate', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--sid'); ap.add_argument('--clip_dir'); ap.add_argument('--anim_json')
    ap.add_argument('--png_dir'); ap.add_argument('--n_png', type=int, default=0)
    if len(sys.argv) > 2 and sys.argv[1] == '--_wrap':
        return wrap_main(sys.argv[2:])
    a = ap.parse_args()
    if a._animate:
        return animate_main(a)
    assert a.root and a.out, '--root and --out are required'
    sys.path.insert(0, HERE)
    from phase_timer import PROTOCOL, stats, write

    # ---- inputs
    man = json.load(open(osp.join(a.root, 'manifest.json')))
    ent = [c for c in man['clips'] if c['id'] == a.clip]
    assert ent, '%s not in manifest' % a.clip
    ds, clip = a.clip.split('/')
    clip_dir = osp.join(a.root, ent[0]['E'], 'base', ds, clip)
    fr = json.load(open(osp.join(clip_dir, 'frames.json')))
    n_recon, n_test = len(fr['recon']), len(fr['test'])
    smplx2020 = a.smplx2020 or next((p for p in ['/home/ubx858/guava/assets/SMPLX/SMPLX_NEUTRAL_2020.npz',
                                                 osp.join(REPO, 'assets', 'SMPLX', 'SMPLX_NEUTRAL_2020.npz')] if osp.exists(p)), None)
    assert smplx2020 and osp.exists(smplx2020), 'SMPLX_NEUTRAL_2020.npz not found: pass --smplx2020'
    exa_env, exa_env_pose = osp.join(a.exa_root, 'envs', 'exavatar'), osp.join(a.exa_root, 'envs', 'exavatar_pose')
    py, py_pose = osp.join(exa_env, 'bin', 'python'), osp.join(exa_env_pose, 'bin', 'python')
    for p in [py, py_pose, osp.join(a.exa_root, 'src', 'ExAvatar_RELEASE')]:
        assert osp.exists(p), 'missing ' + p
    out_dir = osp.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)

    # ---- private EXA_ROOT under /dev/shm
    work = tempfile.mkdtemp(prefix='exavatar_timing_', dir=a.work_root)
    root = osp.join(work, 'exa_root')
    print('work dir', work, flush=True)
    t_m = time.perf_counter()
    n_files = build_mirror(a.exa_root, root)
    print('mirrored %d files of the ExAvatar tree in %.1f s' % (n_files, time.perf_counter() - t_m), flush=True)
    hook, memdir, logd = osp.join(work, 'pyhook'), osp.join(work, 'mem'), osp.join(work, 'logs')
    for d in [hook, memdir, logd] + [osp.join(work, 'cache', c) for c in ['hf', 'xdg', 'mpl', 'nv', 'torch_ext', 'pip', 'conda_pkgs']] + [osp.join(work, 'tmp')]:
        os.makedirs(d, exist_ok=True)
    with open(osp.join(hook, 'sitecustomize.py'), 'w') as f:
        f.write(SITECUSTOMIZE)

    # env.sh / run_clip_gtface.sh, with every writable location moved into the work dir
    SRC = osp.join(root, 'src', 'ExAvatar_RELEASE')
    FIT, AV = osp.join(SRC, 'fitting'), osp.join(SRC, 'avatar')
    sid = 'gtf__%s__%s' % (ds, clip)
    W = osp.join(root, 'work', sid)
    path0 = os.environ.get('PATH', '')
    env = dict(os.environ)
    env.update(EXA_ROOT=root, EXA_SRC=SRC, EXA_ENV=exa_env, EXA_ENV_POSE=exa_env_pose,
               TORCH_HOME=osp.join(a.exa_root, 'cache', 'torch'),     # read only: FAN / VGG16 weights downloaded at setup
               PIP_CACHE_DIR=osp.join(work, 'cache', 'pip'), CONDA_PKGS_DIRS=osp.join(work, 'cache', 'conda_pkgs'),
               HF_HOME=osp.join(work, 'cache', 'hf'), XDG_CACHE_HOME=osp.join(work, 'cache', 'xdg'),
               MPLCONFIGDIR=osp.join(work, 'cache', 'mpl'), CUDA_CACHE_PATH=osp.join(work, 'cache', 'nv'),
               TORCH_EXTENSIONS_DIR=osp.join(work, 'cache', 'torch_ext'), TMPDIR=osp.join(work, 'tmp'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               PYTHONPATH=hook + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),
               EXA_TIMING_MEMDIR=memdir, EXA_TIMING_SMPLX2020=smplx2020,
               EXA_TIMING_TRAIN_STATS=osp.join(work, 'train_stats.json'),
               PATH=osp.join(exa_env, 'bin') + os.pathsep + path0)
    for k, v in [('EXA_TIMING_MAX_STEPS', a.max_steps), ('EXA_TIMING_MAX_EPOCHS', a.max_epochs), ('EXA_TIMING_FIT_MAX_ITERS', a.fit_max_iters)]:
        env.pop(k, None)
        if v is not None:
            env[k] = str(v)
    env_pose = dict(env, PATH=osp.join(exa_env_pose, 'bin') + os.pathsep + path0)
    me = osp.abspath(__file__)

    poll = GpuPoll() if a.mem_poll else None
    if poll:
        poll.start()
    T, failed = {}, None

    def stage(name, cmd, cwd, e=None, check=None):
        if poll:
            poll.stage = name
        log = osp.join(logd, name + '.log')
        print('[%s] %s ...' % (time.strftime('%H:%M:%S'), name), flush=True)
        t0 = time.perf_counter()
        with open(log, 'w') as f:
            r = subprocess.run(cmd, cwd=cwd, env=dict(e or env, EXA_TIMING_STAGE=name), stdout=f, stderr=subprocess.STDOUT)
        T[name] = time.perf_counter() - t0
        print('[%s] %s %.1f s (exit %d)' % (time.strftime('%H:%M:%S'), name, T[name], r.returncode), flush=True)
        if r.returncode != 0 or (check and not check()):
            raise RuntimeError('stage %s failed (exit %d); log %s' % (name, r.returncode, log))

    def mv_children(src, dst):
        for x in os.listdir(src):
            shutil.move(osp.join(src, x), osp.join(dst, x))

    try:
        # ================= reconstruction: run_clip_gtface.sh steps 0-7 (one end-to-end run) =================
        t_rec = time.perf_counter()
        # 0. ExAvatar subject folders + GT conversion (CPU)
        stage('prep_clip', [py, me, '--_wrap', osp.join(BASE, 'prep_clip.py'), '--clip_dir', clip_dir, '--sid', sid], cwd=work)
        # 1. DECA (initial FLAME), recon frames only
        stage('deca', [py, 'run_deca.py', '--root_path', osp.join(W, 'fit_recon')], cwd=osp.join(FIT, 'tools', 'DECA'),
              check=lambda: osp.isfile(osp.join(W, 'fit_recon', 'flame_init', 'shape_param.json')))
        # 2. mmpose whole-body 2D keypoints (RTMDet + DWPose), recon frames only, pose env
        stage('mmpose', [py_pose, 'run_mmpose.py', '--root_path', osp.join(W, 'fit_recon')], cwd=osp.join(FIT, 'tools', 'mmpose'), e=env_pose,
              check=lambda: len(os.listdir(osp.join(W, 'fit_recon', 'keypoints_whole_body'))) == len(os.listdir(osp.join(W, 'fit_recon', 'frames'))))
        # 3. ExAvatar face fitting on the recon frames (body = GT), then its texture unwrap
        stage('face_fit', [py, me, '--_wrap', osp.join(BASE, 'fit_face_gt.py'), '--subject_id', sid + '_recon',
                           '--gt_conv', osp.join(W, 'gt_conversion.json'), '--stage', 'recon'], cwd=osp.join(FIT, 'main'))
        shutil.rmtree(osp.join(W, 'fit_recon', 'smplx_optimized'), ignore_errors=True)
        mv_children(osp.join(FIT, 'output', 'result', sid + '_recon'), osp.join(W, 'fit_recon'))
        stage('face_unwrap', [py, 'unwrap.py', '--subject_id', sid + '_recon'], cwd=osp.join(FIT, 'main'))
        mv_children(osp.join(FIT, 'output', 'result', sid + '_recon', 'unwrapped_textures'), osp.join(W, 'fit_recon', 'smplx_optimized'))
        # 4. (gt-face variant) no test-frame face fit
        # 5. avatar folder: GT body + smoothed face params, ID files, keypoints, meshes; test params = GT body + GT face
        stage('finalize', [py, me, '--_wrap', osp.join(BASE, 'finalize_params.py'), '--clip_dir', clip_dir, '--sid', sid, '--gt_face_test'],
              cwd=osp.join(FIT, 'tools'))
        # 6. background point cloud (Depth-Anything-V2 static-background route)
        stage('depth_anything', [py, 'run_depth_anything.py', '--root_path', osp.join(W, 'avatar')], cwd=osp.join(FIT, 'tools', 'Depth-Anything-V2'),
              check=lambda: osp.isfile(osp.join(W, 'avatar', 'bkg_point_cloud.txt')))
        # 7. avatar training (ExAvatar main/train.py, default config)
        stage('train', [py, me, '--_wrap', osp.join(AV, 'main', 'train.py'), '--subject_id', sid], cwd=osp.join(AV, 'main'),
              check=lambda: any(f.startswith('snapshot_') for f in os.listdir(osp.join(AV, 'output', 'model_dump', sid))))
        rec_total = time.perf_counter() - t_rec

        # ================= animation (separate process, as render_test.py) =================
        anim_json = osp.join(work, 'anim.json')
        stage('animate', [py, me, '--_animate', '--sid', sid, '--clip_dir', clip_dir, '--anim_json', anim_json,
                          '--warmup', str(a.warmup), '--iters', str(a.iters), '--smplx2020', smplx2020,
                          '--png_dir', osp.join(out_dir, 'exavatar_png') if a.png > 0 else '', '--n_png', str(a.png)],
              cwd=osp.join(AV, 'main'), e=dict(env, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'))
        anim = json.load(open(anim_json))
    except BaseException as ex:                                                           # noqa: BLE001
        failed = ex
    finally:
        if poll:
            poll.stop()

    if failed is not None:
        dst = osp.join(out_dir, 'exavatar_failed_logs')
        shutil.copytree(logd, dst, dirs_exist_ok=True)
        print('FAILED:', failed, '\nlogs copied to', dst, flush=True)
        last = max((osp.join(logd, f) for f in os.listdir(logd)), key=osp.getmtime, default=None)
        if last:
            print('---- tail of', last); print(''.join(open(last, errors='replace').readlines()[-30:]))
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        sys.exit(1)

    # ---- gather
    fit_log = json.load(open(osp.join(W, 'fit_recon', 'fit_face_gt_log.json')))
    with open(osp.join(W, 'avatar', 'bkg_point_cloud.txt')) as f:
        bkg_points = sum(1 for _ in f)
    tr = json.load(open(osp.join(work, 'train_stats.json')))
    mem = {}
    for fn in os.listdir(memdir):
        r = json.load(open(osp.join(memdir, fn)))
        m = mem.setdefault(r['stage'], {'max_allocated_mib': 0.0, 'max_reserved_mib': 0.0})
        m['max_allocated_mib'] = round(max(m['max_allocated_mib'], r['max_allocated_mib']), 1)
        m['max_reserved_mib'] = round(max(m['max_reserved_mib'], r['max_reserved_mib']), 1)
    full_steps = 5 * 15 * n_recon
    shortened = any(v is not None for v in (a.max_steps, a.max_epochs, a.fit_max_iters))
    step_s = ((tr['t_last'] - tr['t_first']) / (tr['steps'] - 1)) if tr.get('steps', 0) > 1 else None
    S1 = lambda s: stats([s * 1000.0])
    pre = T['prep_clip'] + T['face_unwrap'] + T['depth_anything']
    pose = T['deca'] + T['mmpose'] + T['face_fit'] + T['finalize']
    rec = {'preprocess': S1(pre), 'segmentation': None, 'pose': S1(pose), 'network': None, 'training': S1(T['train']),
           'total': S1(rec_total),
           'preprocess_prep_clip': S1(T['prep_clip']), 'preprocess_face_unwrap': S1(T['face_unwrap']),
           'preprocess_depth_anything': S1(T['depth_anything']),
           'pose_deca': S1(T['deca']), 'pose_mmpose': S1(T['mmpose']), 'pose_face_fit': S1(T['face_fit']),
           'pose_finalize': S1(T['finalize'])}
    am = anim['animation']
    ani = {'pose': None, 'segmentation': None, 'preprocess': am['preprocess'], 'deformation': am['deformation'],
           'rendering': am['rendering'], 'rendering_raster': None, 'rendering_refiner': None, 'update': None,
           'total': am['total']}
    sched = ('face fit %d opt steps (evaluated: 2000 for 83 frames%s); avatar training %d steps of %d x %d '
             '(evaluated: 5 x 15 x %d = %d)' % (fit_log['opt_steps'], '; capped at %d iters/batch/epoch' % a.fit_max_iters if a.fit_max_iters else '',
                                                tr['steps'], tr['end_epoch'], tr['itr_per_epoch'], n_recon, full_steps))
    peak = max((m['max_reserved_mib'] for m in mem.values()), default=None)
    clip_notes = ('%s. %s. One reconstruction run (n = 1 per phase); %d recon / %d test frames. Training: %.1f s wall '
                  'incl. setup and 1 snapshot per epoch; mean step %.3f s over %d steps. Animation: %d warm-up + %d timed '
                  'frames cycling over the %d test keys, checkpoint snapshot_%d (model load %.2f s, not in any phase); '
                  'driving params check vs the pipeline\'s test_params max |diff| %.2g. Peak torch-reserved GPU memory '
                  '%.0f MiB (stage %s).' % (
                      'SHORTENED TEST RUN, NOT A RESULT' if shortened else 'Evaluated schedule', sched, n_recon, n_test,
                      T['train'], step_s or float('nan'), tr['steps'], a.warmup, a.iters, len(anim['test_keys']),
                      anim['test_epoch'], anim['model_load_s'], anim['param_check_max_abs_diff'], peak or float('nan'),
                      max(mem, key=lambda k: mem[k]['max_reserved_mib']) if mem else '?'))
    notes = (
        'ExAvatar (mks0601/ExAvatar_RELEASE@518f5647), paper row = fairness-rule variant exavatar_gtface '
        '(run_clip_gtface.sh): GT body converted to ExAvatar\'s SMPL-X v1.1 male layer, ExAvatar face fit on the recon '
        'frames, test face = GT jaw/expression. Reconstruction (each stage a subprocess as in run_clip_gtface.sh; total = '
        'prep_clip start to train.py end, incl. file moves between stages, = run.json train_s): '
        'preprocess = prep_clip.py (GT conversion, subject folders) + unwrap.py (ExAvatar face-texture unwrap) + '
        'Depth-Anything-V2-L background point cloud (run_depth_anything.py); '
        'segmentation = null (the package\'s StyleMatte alpha is used; SAM not run); '
        'pose = DECA initial FLAME (run_deca.py) + RTMDet/DWPose whole-body keypoints (run_mmpose.py, pose env) + '
        'fit_face_gt.py (FLAME/face fit, body frozen at GT) + finalize_params.py (face smoothing, avatar params); '
        'network = null (no feed-forward avatar network); training = ExAvatar main/train.py wall time (dataset + model '
        'setup, 5 epochs x 15 repeats x recon frames, batch 1, scene + human Gaussians, densification, per-epoch '
        'snapshots). Animation per frame (fp32, batch 1, 512 px, package K, black bg, torch.no_grad): '
        'preprocess = GT SMPL-X 2020 params -> ExAvatar layer (GTConverter.frame_params, GT face, trans re-anchor) + '
        'smplx_param / cam_param CUDA tensors; deformation = HumanGaussian.forward (SMPL-X neutral/zero-pose layer '
        'calls, triplane features + geometry/colour MLPs, pose-dependent offsets, kNN skinning, LBS, normals) -> refined '
        'Gaussians; rendering = GaussianRenderer.forward (camera matrices + diff-gaussian-rasterization-depth); image '
        'stays on the GPU (as render_test.py\'s timed region); total = end to end. pose / segmentation / update / '
        'rendering_refiner = null (GT pose given, no masks at test time, no refiner, no update).')
    protocol = dict(PROTOCOL, warmup=a.warmup, iters=a.iters, recon_warmup=0, recon_reps=1,
                    train_steps=tr['steps'], train_steps_full=full_steps, face_fit_steps=fit_log['opt_steps'],
                    shortened=shortened)
    write(osp.join(out_dir, 'exavatar.json'), method='exavatar', row='ExAvatar',
          clips={a.clip: {'reconstruction': rec, 'animation': ani, 'notes': clip_notes}}, notes=notes,
          protocol=protocol, power_note=a.power_note)
    details = {'stage_s': {k: round(v, 3) for k, v in T.items()}, 'reconstruction_total_s': rec_total,
               'train_stats': tr, 'train_mean_step_s': step_s, 'face_fit_log': fit_log,
               'scene_gaussians_init_bkg_points': bkg_points,
               'torch_peak_mem_mib_per_stage': mem, 'nvidia_smi_peak_mib_per_stage': poll.peak if poll else None,
               'animation': anim, 'shortened': shortened, 'args': {k: v for k, v in vars(a).items() if not k.startswith('_')},
               'clip_dir': clip_dir, 'work_dir': work, 'mirrored_files': n_files}
    with open(osp.join(out_dir, 'exavatar_details.json'), 'w') as f:
        json.dump(details, f, indent=1)
    if a.keep_work:
        print('work dir kept:', work)
    else:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
