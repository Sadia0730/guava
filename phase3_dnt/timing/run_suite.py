"""Run selected laptop timers sequentially, with cooling pauses; never change model settings.
Default is the one-clip RAGA profile. --dry-run prints commands without GPU/model execution.
Per-person methods train on one full reconstruction partition; request them explicitly.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from run_live_replay import cool,snapshot,run_monitored


def commands(root,out,methods,clip,py):
    here=Path(__file__).resolve().parent;result=[]
    allowed={'raga','guava','lhm','lhmpp','idol','gart','gaussianavatar','exavatar','ft'}
    if set(methods)-allowed:raise ValueError('unknown methods: '+str(set(methods)-allowed))
    for method in methods:
        if method in ['raga','guava']:
            rows=['guava'] if method=='guava' else ['fb_gt','fb_sprint','fb_update','fb_sprint_update']
            for row in rows:
                label=method if method=='guava' else 'raga_'+row
                result.append((label,['bash',str(here/'run_ours.sh'),str(root),str(out/label),'--rows',row,'--clips',clip]))
            continue
        elif method=='ft':
            cmd=[py,str(here/'time_finetune.py'),'--root',str(root),'--out-dir',str(out/method),'--clip',clip]
        else:
            clipflag='--clips' if method in ['lhm','lhmpp','idol'] else '--clip'
            cmd=['bash',str(here/f'run_{method}.sh'),str(root),str(out/method),clipflag,clip]
            if method in ['lhm','lhmpp','idol']:cmd+=['--recon-warmup','1','--recon-reps','3']
        result.append((method,cmd))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--methods',default='raga');p.add_argument('--clip',default='neuman/bike');p.add_argument('--python',default=os.environ.get('GUAVA_PY',sys.executable))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--cooldown',type=int,default=45);p.add_argument('--allow-training',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out_dir).resolve();methods=a.methods.split(',')
    if set(methods)&{'gart','gaussianavatar','exavatar','ft'} and not a.allow_training:p.error('per-person reconstruction trains one avatar; explicitly add --allow-training')
    if out.is_relative_to(root) or out.is_relative_to('/raid') or out.is_relative_to('/vault'):p.error('use a new local output directory')
    cmds=commands(root,out,methods,a.clip,a.python)
    if a.dry_run:print(json.dumps(cmds,indent=2));return
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);reports=[]
    for i,(method,cmd) in enumerate(cmds):
        if i and a.cooldown:time.sleep(a.cooldown)
        before=cool()
        with (out/f'{method}_suite.log').open('x') as log:
            run_monitored(cmd,log,out/f'{method}_thermal.json',env=dict(os.environ,GUAVA_PY=a.python))
        reports.append({'method':method,'before':before,'after':snapshot(),'command':cmd})
    (out/'suite_metadata.json').write_text(json.dumps(reports,indent=2));print('Completed:',out)
if __name__=='__main__':main()
