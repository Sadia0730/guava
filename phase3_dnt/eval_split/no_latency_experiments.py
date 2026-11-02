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
    import hashlib
    fast = load_fast()
    fitter = fast.SprintFitter(CKPT, 'cuda:0', 150)
    sys.path.insert(0, str(HERE.parent / 'guava_fb'))
    import convert_bedlam2 as cb
    cb.LOCAL_TMP = CACHE / 'decode'; cb.LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    clips = [c for c in json.loads((HERE / 'valbench_v1.json').read_text())['clips'] if c['dataset'] == 'bedlam2'][:5]
    frames = []
    for c in clips:
        want = c['test_frames'][:max(1, (args.n + 4) // 5)]
        got = cb.decode_frames(c['job'], c['seq'], want)
        frames.extend((c['id'], f, got[f]) for f in want if f in got)
    frames = frames[:args.n]
    if len(frames) < 5:
        raise RuntimeError('Too few decoded frames')
    shp = fitter.fit(frames[0][2])['shape']
    out = {'gpu': torch.cuda.get_device_name(), 'n_frames': len(frames), 'frames': [(c, f) for c, f, _ in frames],
           'git': os.popen('git rev-parse HEAD').read().strip(), 'arms': {}, 'equivalence': {}}
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
        (CACHE / 'benchmark.json').write_text(json.dumps(out, indent=2))
    report = O / 'reports/no_latency_20261102'; report.mkdir(parents=True, exist_ok=True)
    (report / 'benchmark.json').write_text(json.dumps(out, indent=2))
    print('REPORT', str(report / 'benchmark.json'), flush=True)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('inventory', 'benchmark', 'teacher', 'train'))
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()
    globals()[args.action](args)


if __name__ == '__main__':
    main()
