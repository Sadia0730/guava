"""Optional evaluated 500-step subject fine-tuning, then timing of its three rows.

This is a real training job, never part of the default laptop run. It preserves the
historical subject_ft.py recipe, using only the chosen clip's reconstruction split.
Run through run_suite.py --methods ft --allow-training for thermal supervision.
"""
from __future__ import annotations
import argparse,json,os,runpy,subprocess,sys,tempfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--clip',default='neuman/bike');p.add_argument('--assets',default=os.environ.get('RAGA_ASSETS',str(REPO/'live_assets')))
    p.add_argument('--fb-ckpt',default=os.environ.get('RAGA_FB_CKPT',''))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--_train',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--_weights',default='',help=argparse.SUPPRESS);p.add_argument('--_log',default='',help=argparse.SUPPRESS)
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out_dir).resolve();assets=Path(a.assets).resolve()
    ckpt=a.fb_ckpt or str(assets/'checkpoints/guava_fb_B28k.pt')
    if out.is_relative_to(root) or out.is_relative_to('/raid') or out.is_relative_to('/vault'):p.error('use a fresh local output directory')
    manifest=json.loads((root/'manifest.json').read_text());clip=next(c for c in manifest['clips'] if c['id']==a.clip)
    if a.clip!='neuman/bike':p.error('this subset preserves the complete per-person reconstruction budget only for neuman/bike')
    os.environ.setdefault('ARCFACE_ONNX',str(assets/'arcface/glintr100.onnx'))
    if a._train:
        # Keep the evaluated recipe intact; only redirect its default server checkpoint.
        sys.path[:0]=[str(REPO/'phase3_dnt/guava_fb'),str(REPO/'phase3_dnt')]
        import fb_common as F
        original=F.setup_a20
        def setup(dev,**kwargs):
            kwargs.setdefault('ckpt',ckpt);return original(dev,**kwargs)
        F.setup_a20=setup
        sys.argv=['subject_ft.py','--E',str(root/clip['E']),'--clip',a.clip,'--steps','500',
                  '--out',a._weights,'--log',a._log,'--fb-ckpt',ckpt]
        runpy.run_path(str(REPO/'phase3_dnt/eval_split/subject_ft.py'),run_name='__main__');return
    if a.dry_run:
        print(json.dumps({'clip':a.clip,'recon_frames':len(clip['recon']),'steps':500,'checkpoint':ckpt,
            'rows':['ft_gt','ft_update','ft_sprint_update'],'training':True},indent=2));return
    if out.exists():raise FileExistsError(out)
    if not Path(ckpt).is_file():raise FileNotFoundError(ckpt)
    out.mkdir(parents=True);raw=out/'_raw';raw.mkdir()
    work=Path(tempfile.mkdtemp(prefix='raga_timing_ft_',dir='/dev/shm'));weights=work/'model.pt';log=raw/'training.json'
    cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--out-dir',str(out),'--clip',a.clip,
         '--assets',str(assets),'--fb-ckpt',ckpt,'--_train','--_weights',str(weights),'--_log',str(log)]
    start=time.perf_counter()
    with (raw/'training.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    training_wall_ms=(time.perf_counter()-start)*1000
    subprocess.run([sys.executable,str(HERE/'time_ours.py'),'--root',str(root),'--out-dir',str(raw/'animation'),
        '--clips',a.clip,'--assets',str(assets),'--fb-ckpt',str(weights),'--rows','fb_gt,fb_update,fb_sprint_update',
        '--warmup','10','--iters','50','--recon-warmup','1','--recon-reps','3','--tag','ft'],check=True)
    import phase_timer as PT
    training=json.loads(log.read_text())
    names={'fb_gt':'RAGA, fine-tuned (supplied)','fb_update':'RAGA, fine-tuned + update (supplied)',
           'fb_sprint_update':'RAGA, fine-tuned + update (estimated)'}
    for row,label in names.items():
        r=json.loads((raw/'animation'/f'ours_{row}_ft.json').read_text());c=r['clips'][a.clip];rec=c['reconstruction']
        rec['post_training_construction']=rec['total'];rec['training']=PT.stats([training['train_s']*1000])
        rec['training_process']=PT.stats([training_wall_ms])
        rec['total']=PT.stats([training_wall_ms+x for x in rec['total']['samples_ms']])
        c['training_details']={'steps':500,'n_recon':len(clip['recon']),'weights_retained':str(weights),
            'total_note':'one shared training run plus repeated final construction; n is not independent training repeats'}
        PT.write(out/f'{row}_ft.json','ours_'+row+'_ft',label,r['clips'],
            {'training':'evaluated 500-step recipe; supplied reconstruction supervision for all fine-tuned rows',
             'raw_measurement':r,'work_retained':str(work)},protocol=dict(r['protocol'],training_steps=500,training_reps=1))
    print('Fine-tuned weights retained:',weights)
if __name__=='__main__':main()
