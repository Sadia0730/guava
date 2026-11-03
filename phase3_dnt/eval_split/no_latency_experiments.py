"""Validation-only experiments; inference architecture and Gaussian capacity fixed.

All generated data goes to a new /dev/shm cache or a new persistent report/run.
Existing checkpoints, renders, and data are never overwritten by this driver.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
O = Path('/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb')
CACHE = Path('/dev/shm/ubx858_no_latency_20261102')
CKPT = Path('/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt')


def load_fast():
    spec = importlib.util.spec_from_file_location('sprint_fast_driving', HERE.parent / 'sprint_fast_driving.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def benchmark(args):
    import torch
    import numpy as np
    import cv2
    cv2.setNumThreads(2)
    fast = load_fast()
    print('LOADING', 'checkpoint', str(CKPT), 'threads', args.threads, flush=True)
    fitter = fast.SprintFitter(CKPT, 'cuda:0', 150, dwpose_threads=args.threads)
    print('LOADED', flush=True)
    sys.path.insert(0, str(HERE.parent / 'guava_fb'))
    import convert_bedlam2 as cb
    cb.LOCAL_TMP = CACHE / 'decode'; cb.LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    clips = [c for c in json.loads((HERE / 'valbench_v1.json').read_text())['clips'] if c['dataset'] == 'bedlam2'][:5]
    frame_cache = CACHE / f'timing_frames_{args.n}.npz'
    if frame_cache.exists():
        with np.load(frame_cache, allow_pickle=False) as z:
            frames = [(str(c), int(f), z[f'image_{i}'].copy()) for i, (c, f) in enumerate(zip(z['clip_ids'], z['frame_ids']))]
        print('CACHED_FRAMES', len(frames), flush=True)
    else:
        frames = []
        for c in clips:
            want = c['test_frames'][:max(1, (args.n + 4) // 5)]
            print('DECODING', c['id'], len(want), flush=True)
            got = cb.decode_frames(c['job'], c['seq'], want)
            frames.extend((c['id'], f, got[f]) for f in want if f in got)
        frames = frames[:args.n]
        np.savez(frame_cache, clip_ids=np.asarray([c for c, _, _ in frames]), frame_ids=np.asarray([f for _, f, _ in frames]),
                 **{f'image_{i}': im for i, (_, _, im) in enumerate(frames)})
    if len(frames) < 5:
        raise RuntimeError('Too few decoded frames')
    shp = fitter.fit(frames[0][2])['shape']
    out = {'gpu': torch.cuda.get_device_name(), 'n_frames': len(frames), 'frames': [(c, f) for c, f, _ in frames],
           'threads': args.threads, 'host_load_at_start': os.getloadavg(),
           'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'arms': {}, 'equivalence': {}}
    ref = []; ref_h1 = []
    arms = ['cpu_raw_full', 'cpu_raw_lean', 'cpu_h1_60', 'cuda_raw_lean', 'cuda_h1_60']
    for name in arms:
        if name == 'cuda_raw_lean':
            try:
                fitter.set_dwpose_provider('CUDAExecutionProvider')
            except Exception as exc:
                out['cuda_error'] = repr(exc)
                print('CUDA_ERROR', repr(exc), flush=True)
                break
        if name.endswith('raw_full'):
            fn = lambda im: fast.SprintFitterBase.fit(fitter, im, iters=0, shape_lock=shp)
        elif name.endswith('raw_lean'):
            fn = lambda im: fitter.drive_raw(im, shape_lock=shp)
        else:
            fn = lambda im: fitter.fit(im, iters=60, shape_lock=shp)
        for _, _, im in frames[:3]:
            fn(im)
        T = []; results = []; torch.cuda.reset_peak_memory_stats()
        for _, _, im in frames:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = fn(im)
            torch.cuda.synchronize(); elapsed = (time.perf_counter() - t0) * 1000
            T.append({'wall_ms': elapsed, **{k: v * 1000 for k, v in r['times'].items()}})
            results.append(r)
        keys = sorted(set().union(*(t.keys() for t in T)))
        out['arms'][name] = {'median_ms': {k: float(np.median([t[k] for t in T if k in t])) for k in keys},
                             'p90_wall_ms': float(np.percentile([t['wall_ms'] for t in T], 90)),
                             'peak_gb': torch.cuda.max_memory_allocated() / 2**30}
        if name == 'cpu_raw_full':
            ref = results
        elif name == 'cpu_h1_60':
            ref_h1 = results
        else:
            compare = ref_h1 if name == 'cuda_h1_60' else ref
            field = 'smplx_coeffs' if name == 'cuda_h1_60' else None
            deltas = {}
            for k in ref[0]['raw_smplx']:
                d = []
                for r, r0 in zip(results, compare):
                    x = r['record'][field][k] if field else r['raw_smplx'][k]
                    y = r0['record'][field][k] if field else r0['raw_smplx'][k]
                    d.append(float(np.abs(np.asarray(x) - np.asarray(y)).max()))
                deltas[k] = max(d)
            out['equivalence'][name] = deltas
        print('ARM', name, json.dumps(out['arms'][name]), 'DELTAS', out['equivalence'].get(name), flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f'{args.name}.json').write_text(json.dumps(out, indent=2))
    report = O / 'reports/no_latency_20261102'; report.mkdir(parents=True, exist_ok=True)
    (report / f'{args.name}.json').write_text(json.dumps(out, indent=2))
    print('REPORT', str(report / f'{args.name}.json'), flush=True)


def inventory(args):
    import torch
    import onnxruntime
    print('gpu', torch.cuda.get_device_name(), 'ort', onnxruntime.get_available_providers(), flush=True)
    for source in ('bedlam2', 'ubc'):
        index = json.loads((O / 'data' / source / 'index.json').read_text())
        print(source, 'index keys', list(index), 'first', index['clips'][:1], flush=True)
        entry = next(c for c in index['clips'] if c.get('split') == 'train')
        meta = O / 'data' / source / entry['name'] / 'meta.json'
        if meta.exists():
            print(source, 'meta', meta.read_text()[:2500], flush=True)
    plan = json.loads((O / 'reports/bedlam2/plan.json').read_text())
    print('bedlam plan keys', list(plan), flush=True)
    for key, value in plan.items():
        if isinstance(value, list) and value:
            print('plan entry', key, value[:1], flush=True)


def probe(args):
    import glob
    import subprocess
    for pattern in ('/dev/shm/ubx858_baselines/*/env/lib/libcublasLt.so*',
                    '/dev/shm/ubx858_baselines/*/env/lib/python*/site-packages/nvidia/*/lib/lib*.so*',
                    '/usr/local/cuda*/lib64/libcublasLt.so*'):
        paths = glob.glob(pattern)
        print('LIBRARIES', pattern, paths, flush=True)
    import onnxruntime
    lib = Path(onnxruntime.__file__).parent / 'capi/libonnxruntime_providers_cuda.so'
    p = subprocess.run(['ldd', str(lib)], capture_output=True, text=True)
    print(p.stdout, flush=True)


def verify(args):
    import hashlib
    import numpy as np
    import cv2
    root = Path(args.E)
    split = json.loads((HERE / 'valbench_v1.json').read_text())['clips']
    u = [c['id'] for c in split if c['dataset'] == 'ubody']
    b = [c['id'] for c in split if c['dataset'] == 'bedlam2']
    keep = list(dict.fromkeys(u[:2] + ['ubody/Olympic_S6_Trim23_scene001', 'ubody/ConductMusic_S29_Trim13'] + b[:2]))
    out = {'method': args.method, 'control': args.control, 'clips': {}, 'missing': [], 'png_identical': 0, 'n_frames': 0}
    for cid in keep:
        frame_info = json.loads((root / 'base' / cid / 'frames.json').read_text())
        d = {'frames': len(frame_info['test']), 'png_identical': 0, 'max_pixel_delta': 0, 'mean_pixel_delta': []}
        for key in frame_info['test']:
            p = root / 'renders' / args.method / cid / f'{key}.png'
            q = root / 'renders' / args.control / cid / f'{key}.png'
            if not p.exists() or not q.exists():
                out['missing'].append(f'{cid}/{key}'); continue
            same = hashlib.md5(p.read_bytes()).digest() == hashlib.md5(q.read_bytes()).digest()
            delta = np.abs(cv2.imread(str(p)).astype(np.int16) - cv2.imread(str(q)).astype(np.int16))
            d['png_identical'] += int(same); d['max_pixel_delta'] = max(d['max_pixel_delta'], int(delta.max()))
            d['mean_pixel_delta'].append(float(delta.mean()))
            out['n_frames'] += 1; out['png_identical'] += int(same)
        d['mean_pixel_delta'] = float(np.mean(d['mean_pixel_delta'])) if d['mean_pixel_delta'] else None
        out['clips'][cid] = d
    target = O / 'reports/no_latency_20261102' / f'{args.method}_equivalence.json'
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


PILOT_ARMS = {
    'fastcpu': ('fb_sprint_k0_trkc_fastcpu', 'CPUExecutionProvider', 150, 0),
    'fastcuda': ('fb_sprint_k0_trkc_fastcuda_fp32', 'CUDAExecutionProvider', 150, 0),
    'src500': ('fb_sprint_k0_trkc_fastcuda_src500', 'CUDAExecutionProvider', 500, 0),
    'cuda_h1_60': ('fb_sprint_k60_trkc_fastcuda_fp32', 'CUDAExecutionProvider', 150, 60),
    'cuda_h1_60_m512': ('fb_sprint_k60_trkc_fastcuda_m512', 'CUDAExecutionProvider', 150, 60),
    'detector_fp16': ('fb_sprint_k0_trkc_detfp16', 'CUDAExecutionProvider', 150, 0),
    'sprint_graph': ('fb_sprint_k0_trkc_graph', 'CUDAExecutionProvider', 150, 0),
    'both': ('fb_sprint_k0_trkc_detfp16_graph', 'CUDAExecutionProvider', 150, 0),
}


def pilot(args):
    """Two independent processes at most; new rows only, fixed validation subset."""
    import shutil
    report_dir = O / 'reports/no_latency_20261102'
    report_dir.mkdir(parents=True, exist_ok=True)
    ids = json.loads((HERE / 'no_latency_subset5.json').read_text())['clips']
    names = args.arms.split(',')
    if any(n not in PILOT_ARMS for n in names):
        raise ValueError(f'Unknown arms: {names}')
    common = ['--row', 'fb_sprint', '--track-subject', '--track-nearest', '--fast-backend',
              '--dwpose-threads', '4', '--split', str(HERE / 'valbench_v1.json'), '--E', args.E,
              '--clips', ','.join(ids)]
    started = []
    for first in range(0, len(names), 2):
        running = []
        for arm in names[first:first + 2]:
            row, provider, source_iters, drive_iters = PILOT_ARMS[arm]
            command = [sys.executable, str(HERE / 'eval_guava_rows.py'), *common,
                       '--name', row, '--dwpose-provider', provider, '--sprint-iters', str(source_iters),
                       '--drive-iters', str(drive_iters)]
            if arm == 'cuda_h1_60_m512':
                command += ['--driving-matte-size', '512']
            if arm in ('detector_fp16', 'both'): command += ['--detector-fp16']
            if arm in ('sprint_graph', 'both'): command += ['--sprint-graph']
            log = (report_dir / f'{arm}.log').open('a')
            p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((arm, row, p, log))
            print('START', arm, 'pid', p.pid, flush=True)
        for arm, row, p, log in running:
            code = p.wait(); log.close()
            print('RENDER_DONE', arm, code, flush=True)
            if code:
                # The sibling is independent; let it finish rather than killing it.
                for _, _, sibling, stream in running:
                    if sibling.poll() is None: sibling.wait()
                    if not stream.closed: stream.close()
                raise RuntimeError(f'{arm} failed; inspect {report_dir / (arm + ".log")}')
            started.append(row)
    cmd = [sys.executable, str(HERE / 'score.py'), '--methods', ','.join(started),
           '--split', str(HERE / 'valbench_v1.json'), '--subset', str(HERE / 'no_latency_subset5.json'),
           '--E', args.E, '--tag', 'nl5']
    with (report_dir / 'score_nl5.log').open('a') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for row in started:
        shutil.copy2(Path(args.E) / 'scores' / f'{row}_nl5.json', report_dir / f'{row}_nl5.json')
    print('SCORED', started, flush=True)
    report(args)


def report(args):
    import numpy as np
    rdir = O / 'reports/no_latency_20261102'
    rows = {}
    for arm, (row, _, _, _) in PILOT_ARMS.items():
        p = rdir / f'{row}_nl5.json'
        if p.exists():
            data = json.loads(p.read_text())
            if data['missing_clips'] or len(data['per_clip']) != 5:
                raise RuntimeError(f'Incomplete pilot scores: {p}')
            rows[arm] = data
    old_root = O / 'reports/improve/valbench_scores'
    ids = json.loads((HERE / 'no_latency_subset5.json').read_text())['clips']
    references = {}
    for name in ('fb_sprint', 'fb_sprint_k0_trkc', 'fb_sprint_k60_trkc', 'fb_gt', 'gart'):
        data = json.loads((old_root / f'{name}_vb.json').read_text())
        agg = {}
        for ds in ('ubody', 'bedlam2', 'all'):
            per = [data['per_clip'][c] for c in ids if ds == 'all' or c.startswith(ds + '/')]
            n = sum(c['n'] for c in per)
            agg[ds] = {'clips': len(per), 'frames': n, **{k: sum(c[k] * c['n'] for c in per) / n
                                                       for k in ('psnr', 'ssim', 'lpips', 'l1')}}
        references[name] = agg
    out = {'validation_only': True, 'n_clips': 5, 'n_frames': 426, 'references_on_same_subset': references,
           'arms': {k: v['aggregate'] for k, v in rows.items()}, 'decisions': {}}
    if 'fastcpu' in rows:
        args.method = PILOT_ARMS['fastcpu'][0]
        verify(args)
    if 'fastcuda' in rows:
        baseline = references['fb_sprint_k0_trkc']
        out['decisions']['N2_gpu_k0_quality'] = all(
            rows['fastcuda']['aggregate'][ds]['psnr'] >= baseline[ds]['psnr'] - .1
            and rows['fastcuda']['aggregate'][ds]['lpips'] <= baseline[ds]['lpips'] + .003 for ds in ('ubody', 'bedlam2'))
    if 'src500' in rows and 'fastcuda' in rows:
        a = rows['src500']; b = rows['fastcuda']
        delta = a['aggregate']['all']['psnr'] - b['aggregate']['all']['psnr']
        up = sum(a['per_clip'][c]['psnr'] > b['per_clip'][c]['psnr'] for c in ids)
        accepted = (delta >= .2 and up >= 4 and a['aggregate']['all']['lpips'] <= b['aggregate']['all']['lpips']
                    and all(a['aggregate'][ds]['psnr'] >= b['aggregate'][ds]['psnr'] - .1
                            and a['aggregate'][ds]['lpips'] <= b['aggregate'][ds]['lpips'] + .003 for ds in ('ubody', 'bedlam2')))
        out['decisions']['N3_source500'] = {'psnr_delta': delta, 'clips_up': up, 'quality_gate': accepted}
    if 'cuda_h1_60_m512' in rows and 'cuda_h1_60' in rows:
        a = rows['cuda_h1_60_m512']['aggregate']; b = rows['cuda_h1_60']['aggregate']
        out['decisions']['N4_matte512_quality'] = all(a[ds]['psnr'] >= b[ds]['psnr'] - .1
                                                    and a[ds]['lpips'] <= b[ds]['lpips'] + .003 for ds in ('ubody', 'bedlam2'))
    if 'fastcuda' in rows:
        b = rows['fastcuda']['aggregate']
        for name in ('detector_fp16', 'sprint_graph', 'both'):
            if name in rows:
                a = rows[name]['aggregate']
                out['decisions']['N5_' + name + '_quality'] = all(a[ds]['psnr'] >= b[ds]['psnr'] - .1
                                                               and a[ds]['lpips'] <= b[ds]['lpips'] + .003 for ds in ('ubody', 'bedlam2'))
    out['timing_note'] = 'Shared GPU / CPU contention: these are not idle absolute latency measurements.'
    (rdir / 'pilot_summary.json').write_text(json.dumps(out, indent=2))
    print('PILOT_SUMMARY', json.dumps(out), flush=True)


def matte_benchmark(args):
    import torch
    import numpy as np
    import cv2
    cv2.setNumThreads(2)
    fast = load_fast()
    fitter = fast.SprintFitter(CKPT, 'cuda:0', 150, dwpose_threads=4, dwpose_provider='CUDAExecutionProvider')
    with np.load(CACHE / 'timing_frames_32.npz', allow_pickle=False) as z:
        frames = [z[f'image_{i}'].copy() for i in range(min(args.n, len(z['frame_ids'])))]
    shape = fitter.fit(frames[0])['shape']
    out = {'gpu': torch.cuda.get_device_name(), 'n_frames': len(frames), 'host_load': os.getloadavg(), 'arms': {}}
    for size in (1024, 512):
        fitter.driving_matte_size = size
        for im in frames[:2]: fitter.fit(im, iters=60, shape_lock=shape)
        times = []; kp = []; finite = []; coverage = []
        for im in frames:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            r = fitter.fit(im, iters=60, shape_lock=shape)
            torch.cuda.synchronize()
            times.append({'wall': (time.perf_counter() - t0) * 1000, **{k: v * 1000 for k, v in r['times'].items()}})
            kp.append(r['diag']['kp_err_px1024_after'])
            finite.append(bool(all(np.isfinite(x).all() for x in r['record']['smplx_coeffs'].values())))
            coverage.append(float((r['alpha'] > .5).mean()))
        out['arms'][size] = {'median_ms': {k: float(np.median([t[k] for t in times])) for k in times[0]},
                             'p90_wall_ms': float(np.percentile([t['wall'] for t in times], 90)),
                             'finite': sum(finite), 'median_kp_px1024': float(np.median(kp)),
                             'median_matte_coverage': float(np.median(coverage))}
        print('MATTE', size, json.dumps(out['arms'][size]), flush=True)
    target = O / 'reports/no_latency_20261102/matte_benchmark.json'
    target.write_text(json.dumps(out, indent=2))


def live_benchmark(args):
    """Complete live path, one fixed source avatar; no video decode in timed loop."""
    import torch
    import numpy as np
    import cv2
    import tempfile
    from types import SimpleNamespace
    cv2.setNumThreads(2)
    fast = load_fast()
    fitter = fast.SprintFitter(CKPT, 'cuda:0', 150, dwpose_threads=4, dwpose_provider='CUDAExecutionProvider',
                               detector_fp16=True, sprint_graph=True)
    with np.load(CACHE / 'timing_frames_32.npz', allow_pickle=False) as z:
        frames = [z[f'image_{i}'].copy() for i in range(len(z['frame_ids']))]
    fitter.set_source_mode()
    rs = fitter.fit(frames[0]); shape = rs['shape']
    sys.path.insert(0, str(HERE.parent / 'guava_fb')); sys.path.insert(0, str(HERE))
    import fb_common as F
    import box_sources as BS
    os.chdir(ROOT)
    from models.UbodyAvatar import Ubody_Gaussian
    cfg, inf, ren, _ = F.setup_a20(torch.device('cuda:0'))
    state = torch.load(O / 'runs/guava_fb_B30k/checkpoints/best.pt', map_location='cpu', weights_only=False)
    inf.load_state_dict(state['model'], strict=True); ren.load_state_dict(state['render_model'], strict=True)
    inf.eval(); ren.eval()
    with tempfile.TemporaryDirectory(dir=CACHE) as td:
        fast.write_tracked(Path(td), 'timing', 'frame_000000', rs)
        clip = F.load_clips_parallel([(td, None, True)], workers=1)
        data = F.FBData(cfg, clip)
        source = F.to_dev(F.collate1(data.source_info(0, 'frame_000000')), torch.device('cuda:0'))
    with torch.no_grad():
        torch.manual_seed(0)
        vg, ug, _ = inf(source)
        avatar = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); avatar.init_ehm(inf.ehm); avatar.eval()
    fitter.set_driving_mode(shape)
    out = {'gpu': torch.cuda.get_device_name(), 'n_unique_frames': len(frames), 'n_timed_per_arm': args.n,
           'warmup_per_arm': 10, 'source_fixed_across_arms': True,
           'gaussians': int(avatar._smplx_xyz.shape[1] + avatar._uv_local_xyz.shape[1]),
           'host_load_start': os.getloadavg(), 'arms': {},
           'detector_artifact': getattr(fitter, 'detector_artifact', None),
           'detector_sha256': getattr(fitter, 'detector_artifact_sha256', None)}

    @torch.no_grad()
    def step(image, previous_box):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        boxes = fitter.det.inference_detector(fitter.s_det, image)
        if len(boxes):
            overlap = [BS.iou(b, previous_box) for b in boxes]; index = int(np.argmax(overlap))
            if overlap[index] < .1:
                c = (previous_box[:2] + previous_box[2:]) / 2
                index = int(np.argmin([np.linalg.norm((b[:2] + b[2:]) / 2 - c) for b in boxes]))
            box = boxes[index]
        else: box = previous_box
        torch.cuda.synchronize(); t1 = time.perf_counter()
        r = fitter.fit(image, box=box, iters=0, shape_lock=shape)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        comp = F.compact_record(dict(r['record']), shape)
        sm = {k: torch.as_tensor(v.copy()) for k, v in comp['smplx'].items()}
        sm.update(joints_offset=torch.zeros(55, 3), head_scale=torch.ones(3), hand_scale=torch.ones(3))
        matrix = torch.eye(4); matrix[:3, :4] = sm['camera_RT_params']
        w2c = torch.diag(torch.tensor([-1., -1., 1., 1.])) @ matrix
        target = {'smplx_coeffs': sm, 'flame_coeffs': {'expression_params': torch.zeros(50)},
                  'w2c_cam': w2c, 'c2w_cam': torch.linalg.inv(w2c), 'invtanfov': comp['invtanfov']}
        target['render_cam_params'] = F.render_cam_params(w2c, target['c2w_cam'], comp['invtanfov'], 512)
        target = F.to_dev(F.collate1(target), torch.device('cuda:0'))
        torch.cuda.synchronize(); t3 = time.perf_counter()
        torch.manual_seed(0)
        render = ren(avatar(target), target['render_cam_params'], bg=0.)['renders']
        torch.cuda.synchronize(); t4 = time.perf_counter()
        return np.asarray(box), {'total': (t4-t0)*1000, 'detector': (t1-t0)*1000,
                                 'crop_sprint_export': (t2-t1)*1000, 'render_prep': (t3-t2)*1000,
                                 'deform_raster_refiner': (t4-t3)*1000}, r['raw_smplx']

    reference_poses = {}
    for name, half, graph in (('fp32', False, False), ('detector_fp16', True, False),
                              ('sprint_graph', False, True), ('both', True, True)):
        fitter.s_det = fitter._driving_s_det if half else fitter._source_s_det
        fitter.det = SimpleNamespace(inference_detector=fitter._person_detector) if half else fitter._source_detector_module
        fitter.use_sprint_graph = graph
        box = np.asarray(rs['box']); times = []; poses = []; numeric = {}
        for i in range(args.n + 10):
            box, t, raw = step(frames[i % len(frames)], box)
            if i >= 10:
                times.append(t); poses.append(raw)
        if not graph:
            reference_poses[name] = poses
        else:
            reference = reference_poses['detector_fp16' if half else 'fp32']
            numeric = {k: max(float(np.abs(a[k] - b[k]).max()) for a, b in zip(poses, reference)) for k in poses[0]}
        total = np.asarray([t['total'] for t in times])
        out['arms'][name] = {'p50_ms': float(np.percentile(total, 50)), 'p95_ms': float(np.percentile(total, 95)),
                             'max_ms': float(total.max()), 'over_33ms_fraction': float((total > 33).mean()),
                             'median_stages_ms': {k: float(np.median([t[k] for t in times])) for k in times[0]},
                             'passes_33ms': bool(np.percentile(total, 50) <= 33 and np.percentile(total, 95) <= 33),
                             'host_load': os.getloadavg(), 'graph_raw_numeric_max_error': numeric}
        print('LIVE', name, json.dumps(out['arms'][name]), flush=True)
        filename = 'live_benchmark' if args.name == 'benchmark' else args.name
        (O / f'reports/no_latency_20261102/{filename}.json').write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('inventory', 'probe', 'benchmark', 'verify', 'pilot', 'report', 'matte_benchmark', 'live_benchmark'))
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--name', default='benchmark')
    ap.add_argument('--method', default='fb_sprint_k0_trkc_fastcpu')
    ap.add_argument('--control', default='fb_sprint_k0_trkc')
    ap.add_argument('--E', default='/dev/shm/ubx858_eval/valbench_v1')
    ap.add_argument('--arms', default='fastcpu,fastcuda,src500,cuda_h1_60')
    args = ap.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
