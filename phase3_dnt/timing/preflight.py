"""CPU-only input/asset checks. --gpu-info additionally queries the laptop CUDA device.
Does not execute any model, train, render, or benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parent.parent


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def verify_data(root,hashes=True):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());errors=[]
    for c in m['clips']:
        if not c['test'] or set(c['test'])&set(c['recon']):errors.append(f"invalid split: {c['id']}")
        if c['source'] not in c['recon']:errors.append(f"source outside reconstruction: {c['id']}")
        support=c.get('support',[])
        if not support or not set(support)<=set(c['recon']):errors.append(f"invalid support: {c['id']}")
        for k in set([c['source']]+support+c['test']):
            if not (root/'orig'/c['id']/f'{k}.png').is_file():errors.append(f"missing original: {c['id']}/{k}")
        b=root/c['E']/'base'/c['id'];d=root/c['E']/'data'/c['id']
        f=json.loads((b/'frames.json').read_text())
        if f['source']!=c['source'] or f['test']!=c['test'] or f['recon']!=c['recon']:errors.append(f"manifest/split mismatch: {c['id']}")
        for name in ['optim_tracking_ehm.pkl','id_share_params.pkl','img_lmdb/data.mdb']:
            if not (d/name).is_file():errors.append(f"missing tracked input: {c['id']}/{name}")
    for name,record in m.get('files',{}).items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=record['bytes']:errors.append('missing/wrong size: '+name)
        elif hashes and digest(p)!=record['sha256']:errors.append('hash mismatch: '+name)
    return m,errors


def asset_checks(methods,assets):
    checks=[]
    def check(label,p):checks.append({'asset':label,'path':str(p),'exists':Path(p).exists()})
    if 'raga' in methods:
        check('RAGA B28k',os.environ.get('RAGA_FB_CKPT',str(assets/'checkpoints/guava_fb_B28k.pt')))
        check('pose regressor',os.environ.get('POSE_CKPT',str(assets/'checkpoints/sprint_scratch_ema128k.pt')))
        check('StyleMatte',REPO/'EHM-Tracker/pretrained/matting/stylematte_synth.pt')
        check('DWPose keypoints',REPO/'EHM-Tracker/pretrained/dwpose/dw-ll_ucoco_384.onnx')
        check('YOLOX reference detector',REPO/'EHM-Tracker/pretrained/dwpose/yolox_l.onnx')
    if set(methods)&{'raga','guava','lhm','lhmpp','idol','ft'}:check('SMPL-X 2020',REPO/'assets/SMPLX/SMPLX_NEUTRAL_2020.npz')
    if 'guava' in methods:check('released GUAVA',REPO/'assets/GUAVA/checkpoints/best_160000.pt')
    if 'ft' in methods or 'live' in methods:check('ArcFace',os.environ.get('ARCFACE_ONNX',str(assets/'arcface/glintr100.onnx')))
    base=Path(os.environ.get('BASELINE_ROOT','/dev/shm/ubx858_baselines'))
    if 'gart' in methods:check('GART env',Path(os.environ.get('GART_ROOT',str(base/'gart')))/'env/bin/python')
    if 'gaussianavatar' in methods:check('GaussianAvatar env',Path(os.environ.get('GA_BASE',str(base/'gaussianavatar')))/'env/bin/python')
    if 'exavatar' in methods:
        for env in ['exavatar','exavatar_pose']:check('ExAvatar '+env,Path(os.environ.get('EXA_ROOT_ORIG',str(base/'exavatar')))/'envs'/env/'bin/python')
    if 'lhm' in methods:check('LHM code',os.environ.get('LHM_CODE',str(base/'lhm/src')));check('LHM model',os.environ.get('LHM_MODEL',str(base/'weights/lhm_1b_hf')))
    if 'lhmpp' in methods:check('LHM++ code',os.environ.get('LHMPP_CODE',str(Path.home()/'LHM-plusplus')));check('LHM++ weights',os.environ.get('LHMPP_PRETRAINED',str(Path.home()/'LHM-plusplus/pretrained_models')))
    if 'idol' in methods:check('IDOL code',os.environ.get('IDOL_CODE',str(base/'idol/src')))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--methods',default='raga')
    p.add_argument('--assets',default=os.environ.get('RAGA_ASSETS',str(REPO/'live_assets')))
    p.add_argument('--data-only',action='store_true');p.add_argument('--skip-hashes',action='store_true');p.add_argument('--gpu-info',action='store_true')
    p.add_argument('--out',default='');a=p.parse_args()
    m,errors=verify_data(a.root,not a.skip_hashes);checks=[] if a.data_only else asset_checks(a.methods.split(','),Path(a.assets))
    errors += ['missing asset: '+v['asset']+' at '+v['path'] for v in checks if not v['exists']]
    report={'clips':[{'id':c['id'],'support':len(c['support']),'recon':len(c['recon']),'test':len(c['test'])} for c in m['clips']],
            'assets':checks,'errors':errors,'gpu_execution_tested':False,'note':'CUDA extensions must match laptop architecture; this check does not execute their kernels.'}
    if a.gpu_info:
        import torch
        report['gpu']={'available':torch.cuda.is_available()}
        if torch.cuda.is_available():report['gpu'].update(name=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),torch=torch.__version__,cuda=torch.version.cuda)
        else:errors.append('CUDA unavailable in this interpreter')
    if a.out:
        out=Path(a.out).resolve()
        if out.is_relative_to(Path(a.root).resolve()) or out.is_relative_to('/raid') or out.is_relative_to('/vault'):p.error('write the report outside read-only inputs')
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2));return 1 if errors else 0
if __name__=='__main__':sys.exit(main())
