"""Measure actual existing RAGA application replay throughput, worker off versus UV.
Builds a short cyclic replay from the small input subset. No quality scoring or model
settings are changed. Synthetic repetition is a throughput diagnostic, not evidence
for temporal quality, camera latency, or sustained live adaptation.
"""
from __future__ import annotations
import argparse,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path


def stop_job(proc):
    """Stop only this timing job's new process group, including worker children."""
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait()


def run_monitored(cmd,log,thermal_path,env=None):
    trace=[]
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    try:
        while proc.poll() is None:
            state=snapshot();trace.append(state)
            if (state['gpu_c'] or 0)>=86 or (state['cpu_c'] or 0)>=95:
                raise RuntimeError('thermal guard stopped the job; retain logs and do not report partial results')
            time.sleep(1)
        if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,cmd)
    except BaseException:
        stop_job(proc);raise
    finally:
        thermal_path.write_text(json.dumps(trace,indent=2))


def snapshot():
    info={'time':time.time(),'gpu_c':None,'cpu_c':None}
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits','--id=0'],capture_output=True,text=True,timeout=5)
        parts=r.stdout.strip().split(',');info.update(gpu_c=float(parts[0]),gpu_telemetry=r.stdout.strip())
    except (OSError,ValueError,subprocess.TimeoutExpired):pass
    temps=[]
    for d in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (d/'name').read_text().strip() not in {'coretemp','k10temp','zenpower'}:continue
            for p in d.glob('temp*_input'):temps.append(float(p.read_text())/1000)
        except (OSError,ValueError):pass
    if temps:info['cpu_c']=max(temps)
    return info


def cool(max_gpu=75,max_cpu=80,timeout=300):
    start=time.monotonic()
    while True:
        s=snapshot()
        if (s['gpu_c'] is None or s['gpu_c']<=max_gpu) and (s['cpu_c'] is None or s['cpu_c']<=max_cpu):return s
        if time.monotonic()-start>timeout:raise RuntimeError(f'laptop did not cool within {timeout}s: {s}')
        print('Waiting for laptop to cool:',s,flush=True);time.sleep(5)


def prepare(root,clip,out,frames):
    import cv2
    m=json.loads((root/'manifest.json').read_text());c=next(c for c in m['clips'] if c['id']==clip)
    if frames<30:raise ValueError('use at least 30 replay frames')
    origin=root/'orig'/clip;out.mkdir(parents=True)
    shutil.copyfile(origin/f'{c["source"]}.png',out/'reference.png')
    unique=out/'unique';unique.mkdir();(out/'frames').mkdir()
    keys=c.get('support',[])+c['test']
    for k in keys:
        im=cv2.imread(str(origin/f'{k}.png'))
        if im is None or not cv2.imwrite(str(unique/f'{k}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,95]):raise ValueError(k)
    for i in range(frames):os.link(unique/f'{keys[i%len(keys)]}.jpg',out/'frames'/f'{i:06d}.jpg')
    (out/'replay_provenance.json').write_text(json.dumps({'clip':clip,'cyclic_keys':keys,'frames':frames,'jpeg_quality':95,
        'purpose':'timing only; repeated frames, no camera/display or temporal-quality evidence'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--clip',default='neuman/bike');p.add_argument('--frames',type=int,default=120)
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--python',default=sys.executable);p.add_argument('--assets',default='')
    p.add_argument('--workers',default='off,uv');p.add_argument('--dry-run',action='store_true');p.add_argument('--cooldown',type=int,default=45)
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out_dir).resolve();repo=Path(__file__).resolve().parents[2]
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(root) or out.is_relative_to('/raid') or out.is_relative_to('/vault'):p.error('use a new local output directory')
    if a.frames<=a.warmup+1:p.error('frames must exceed warmup+1')
    modes=a.workers.split(',')
    if set(modes)-{'off','uv'}:p.error('workers must be off and/or uv')
    commands=[]
    for mode in modes:
        cmd=[a.python,str(repo/'phase3_dnt/live/live_sprint_guavafb.py'),'--input',str(out/'replay'),'--no-pace','--no-display',
             '--crop','center','--parallel','--frames',str(a.frames),'--warmup',str(a.warmup),'--worker',mode,
             '--timing-json',str(out/f'live_{mode}.json'),'--dwpose-threads','4']
        if a.assets:cmd+=['--assets',a.assets]
        if os.environ.get('RAGA_FB_CKPT'):cmd+=['--fb-ckpt',os.environ['RAGA_FB_CKPT']]
        if os.environ.get('POSE_CKPT'):cmd+=['--sprint-ckpt',os.environ['POSE_CKPT']]
        commands.append((mode,cmd))
    if a.dry_run:
        print(json.dumps(commands,indent=2));return
    out.mkdir(parents=True);prepare(root,a.clip,out/'replay',a.frames);report=[]
    for index,(mode,cmd) in enumerate(commands):
        if index and a.cooldown:time.sleep(a.cooldown)
        before=cool()
        with (out/f'live_{mode}.log').open('x') as log:
            run_monitored(cmd,log,out/f'thermal_{mode}.json')
        r=json.loads((out/f'live_{mode}.json').read_text())
        report.append({'worker':mode,'fps_observed':r.get('fps_observed'),'before':before,'after':snapshot(),
                       'timing_file':str(out/f'live_{mode}.json'),'stages_ms':r.get('stages_ms'),
                       'command':cmd,'interpretation':'cyclic headless replay; no capture/display; not a webcam or sustained-adaptation claim'})
    (out/'live_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
