"""Overlay timing code into an existing guava_live package, preserving changed files.
Copies Python/shell/config documentation only; never copies weights, data or outputs.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('package');a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];pkg=Path(a.package).resolve()
    if not (pkg/'MANIFEST.json').is_file():p.error('expected an extracted guava_live package with MANIFEST.json')
    if pkg==repo or pkg.is_relative_to('/raid') or pkg.is_relative_to('/vault'):p.error('use the local laptop package directory')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');backup=pkg/'backups'/('timing_'+stamp)
    roots=['phase3_dnt/timing','phase3_dnt/baselines','phase3_dnt/guava_fb','phase3_dnt/worker6','phase3_dnt/live']
    items=[]
    for d in roots:
        items += [f for f in (repo/d).rglob('*') if f.is_file() and f.suffix in {'.py','.sh','.md','.txt','.yaml','.yml'} and not any(x in f.relative_to(repo/d).parts for x in ['bundles','results','work','__pycache__'])]
    items += [repo/'phase3_dnt'/n for n in ['sprint_avatar_fit.py','sprint_fast_driving.py','guava_finetune.py','onnx_arcface_torch.py']]
    items += [repo/'phase3_dnt/eval_split'/n for n in ['uv_update_fns.py','subject_ft.py']]
    hashes={};saved=[]
    for src in items:
        if not src.is_file():raise FileNotFoundError(src)
        rel=src.relative_to(repo);dst=pkg/rel;data=src.read_bytes();hashes[str(rel)]=hashlib.sha256(data).hexdigest()
        if dst.exists() and dst.read_bytes()==data:continue
        if dst.exists():
            b=backup/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,b);saved.append(str(rel))
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    prov=pkg/'TIMING_CODE_MANIFEST.json'
    if prov.exists():backup.mkdir(parents=True,exist_ok=True);shutil.copy2(prov,backup/prov.name)
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with prov.open('w') as f:json.dump({'repo_sha':sha,'copied_sha256':hashes,'backup':str(backup),'saved_paths':saved},f,indent=2)
    print('Timing code installed:',pkg,'files',len(hashes),'backups',len(saved),'source revision',sha)
if __name__=='__main__':main()
