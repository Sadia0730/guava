"""CPU-only timing JSON -> per-clip CSV, phase CSV, Markdown and LaTeX tables.

Only reads measurements. Never fills absent cells with historical/script timings.
A frame-rate reciprocal is labeled an instrumented rate, not live application FPS.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def value(d,k):
    v=d.get(k)
    return v.get('median_ms') if isinstance(v,dict) else None


def read_results(roots):
    results=[]
    for root in roots:
        p=Path(root);files=[p] if p.is_file() else sorted(p.rglob('*.json'))
        for f in files:
            if p.is_dir() and any(x.startswith('_') for x in f.relative_to(p).parts):continue
            try:r=json.loads(f.read_text())
            except (ValueError,UnicodeError):continue
            if not isinstance(r,dict) or not isinstance(r.get('clips'),dict) or 'method' not in r:continue
            proto=r.get('protocol',{})
            for key in ['max_steps','max_epochs','fit_max_iters']:
                if proto.get(key) not in (None,0):raise ValueError(f'{f}: capped test run ({key}); do not use for the paper table')
            if proto.get('test_only') or proto.get('shortened'):raise ValueError(f'{f}: test-only measurement')
            for c in r['clips'].values():
                if c.get('details',{}).get('reconstruction',{}).get('training_complete') is False:
                    raise ValueError(f'{f}: incomplete training schedule')
            results.append((f,r))
    if not results:raise ValueError('no measurement JSON files found')
    gpus={r.get('env',{}).get('gpu') for _,r in results}-{None,''}
    if len(gpus)>1:raise ValueError(f'mixed GPU measurements: {gpus}; collect each hardware configuration separately')
    return results


def tables(results):
    summary=[];phases=[]
    for file,r in results:
        for clip,c in r['clips'].items():
            rec=c.get('reconstruction',{});anim=c.get('animation',{})
            serial=value(anim,'serial_total');diag=value(anim,'total')
            summary.append({'method':r.get('row',r['method']),'clip':clip,'reconstruction_ms':value(rec,'total'),
              'update_initialization_ms':value(c.get('update_initialization',{}),'total'),
              'update_per_observation_ms':value(c.get('update',{}),'total'),
              'update_attempted':c.get('update_counts',{}).get('attempted'),
              'update_accepted':c.get('update_counts',{}).get('accepted'),
              'phase_timed_frame_ms':diag,'serial_frame_ms':serial,
              'serial_rate_fps':1000/serial if serial and serial>0 else None,
              'instrumented_rate_fps':1000/diag if diag and diag>0 else None,
              'source_json':str(file),'gpu':r.get('env',{}).get('gpu','')})
            for scope in ['reconstruction','animation','update_initialization','update']:
                for name,s in c.get(scope,{}).items():
                    if s is not None and not isinstance(s,dict):continue
                    phases.append({'method':r.get('row',r['method']),'clip':clip,'scope':scope,'phase':name,
                       'median_ms':None if s is None else s.get('median_ms'),'p95_ms':None if s is None else s.get('p95_ms'),
                       'n':None if s is None else s.get('n'),'source_json':str(file)})
    return summary,phases


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['no_measurements']);w.writeheader();w.writerows(rows)


def fmt(v):return '—' if v is None else f'{v:.2f}'
def tex(s):
    return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',nargs='+');p.add_argument('--out-dir',required=True)
    a=p.parse_args();out=Path(a.out_dir).resolve()
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to('/raid') or out.is_relative_to('/vault'):raise ValueError('use a new local report directory')
    results=read_results(a.inputs);summary,phases=tables(results);out.mkdir(parents=True)
    write_csv(out/'summary.csv',summary);write_csv(out/'phases.csv',phases)
    note=('Times are measured per clip, batch one, with model settings recorded in each JSON. '
          'Phase-timed frame totals include instrumentation barriers; their reciprocals are diagnostics. '
          'Serial ms/rate come only from a separate uninstrumented pass, where available. '
          'Neither rate is observed live FPS. No unmeasured stage is filled with zero. '
          'Parent and child stages overlap; do not add pose_fit to its keypoint/matting/regression/refinement children, '
          'or rendering to raster/refiner children. Updated-avatar rendering excludes background-worker contention; '
          'use the replay report to measure that cost.')
    md=['# Measured laptop timing','',note,'','| Method | Clip | Reconstruction ms | Update/observation ms | Phase-timed frame ms | Serial frame ms | Serial rate FPS |',
        '|---|---|---:|---:|---:|---:|---:|']
    tx=[r'\begin{table*}[t]',r'\centering',r'\caption{Per-clip laptop timing. Frame totals include phase synchronization; the timed rate is their reciprocal, not observed live FPS. Build excludes model loading and appearance updates; per-person rows include optimization. Update reports median cost per attempted support observation, including rejections, separately from source accumulator initialization. Updated rows animate the refreshed avatar. See CSV for initialization and independent serial measurements where available.}',r'\label{tab:laptop_runtime}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Build (ms) & Update/obs. (ms) & Frame (ms) & Timed rate (frame/s) \\',r'\midrule']
    previous=None
    for r in summary:
        vals=[r['method'],r['clip']]+[fmt(r[k]) for k in ['reconstruction_ms','update_per_observation_ms','phase_timed_frame_ms','serial_frame_ms','serial_rate_fps']]
        md.append('| '+' | '.join(vals)+' |')
        if r['clip']!=previous:
            tx.append(r'\multicolumn{5}{l}{'+tex(r['clip'])+r'} \\');previous=r['clip']
        tv=[r['method']]+[fmt(r[k]) for k in ['reconstruction_ms','update_per_observation_ms','phase_timed_frame_ms','instrumented_rate_fps']]
        tx.append(' & '.join(tex(v).replace('—','--') for v in tv)+r' \\')
    tx += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (out/'runtime.md').write_text('\n'.join(md)+'\n');(out/'runtime.tex').write_text('\n'.join(tx)+'\n')
    (out/'measurement_sources.json').write_text(json.dumps({'files':[str(f) for f,_ in results],'note':note},indent=2)+'\n')
    print('wrote',out,'rows',len(summary),'phase entries',len(phases))
if __name__=='__main__':main()
