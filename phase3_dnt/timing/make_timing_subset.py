"""CPU-only compact timing inputs. Reads existing conversions; never runs models.

Keeps the evaluated records, images, masks, cameras, source selection, and split
membership. The NeuMan training partition is retained for per-person construction.
Original-resolution source/support/driving PNGs enable actual estimated-input paths.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import shutil
import zipfile
import tarfile
import tempfile
from pathlib import Path

CASES = [('avatar_test_v1', 'ubody/ConductMusic_S15'), ('finalbench_ext', 'peoplesnapshot/female-1-casual'),
         ('publicbench_v1', 'neuman/bike'), ('publicbench_v2', 'xhumans/00028'), ('finalbench_ext', 'ubc/91-3003CN5S')]
DEFAULT_EVAL = '/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_out(path, source=None):
    p = Path(path).resolve()
    if p.is_relative_to('/raid') or p.is_relative_to('/vault') or (source and p.is_relative_to(Path(source).resolve())):
        raise ValueError('output must be a fresh local directory outside the read-only source, /raid and /vault')
    if p.exists():
        raise FileExistsError(p)
    return p


def original_frames(ev, e, cid, keys, seed):
    import cv2
    import numpy as np
    b = ev / e / 'base' / cid
    remain = []
    for k in keys:
        n = f'timing_subset_v1/orig/{cid}/{k}.png'
        p = b / 'orig' / f'{k}.png'
        if seed and n in seed.namelist():
            yield k, seed.read(n), 'original PNG from existing timing archive'
        elif p.exists():
            yield k, p.read_bytes(), 'converted original PNG'
        else:
            remain.append(k)
    if not remain:
        return
    mapping = {}
    if (b / 'crop_h.npz').exists():
        with np.load(b / 'crop_h.npz', allow_pickle=True) as z:
            if 'orig_src' in z:
                mapping = dict(zip(map(str,z['keys']),map(str,z['orig_src'])))
    if mapping and all(Path(mapping[k]).is_file() for k in remain):
        for k in remain:
            yield k, Path(mapping[k]).read_bytes(), 'original frame named by crop_h.npz'
        return
    video = None; camera = None
    if cid.startswith('peoplesnapshot/'):
        name = cid.split('/')[1]
        tarpath = Path('/vault/research/COS/VIRLab/ubx858/persistent_avatar/backup_for_desktop/data/people_snapshot_extracted.tar')
        scratch = Path(tempfile.mkdtemp(prefix='raga_timing_ps_'))
        with tarfile.open(tarpath) as tf:
            for member in tf:
                if member.name.endswith((f'/{name}/{name}.mp4',f'/{name}/camera.pkl')):
                    dest=scratch/Path(member.name).name
                    with tf.extractfile(member) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
        video=scratch/f'{name}.mp4'
        with (scratch/'camera.pkl').open('rb') as f:cam=pickle.load(f,encoding='latin1')
        camera=(np.array([[cam['camera_f'][0],0,cam['camera_c'][0]],
                          [0,cam['camera_f'][1],cam['camera_c'][1]],[0,0,1.]]),cam['camera_k'])
    elif mapping:
        videos={mapping[k].split('#frame=')[0] for k in remain}
        if len(videos)!=1:raise ValueError(f'cannot resolve original video for {cid}')
        video=next(iter(videos))
    if not video:
        meta = json.loads((ev / e / 'data' / cid / 'meta.json').read_text())
        video = meta.get('video')
    if not video:
        split = Path(__file__).resolve().parents[1] / 'eval_split/avatar_test_v1.json'
        records = json.loads(split.read_text())['clips']
        video = next(c['video'] for c in records if c['id'] == cid)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or (cid.startswith('ubody/') and abs(fps-30)>.05):
        raise ValueError(f'{cid}: expected original 30 Hz video, got {fps}; provide original PNGs under base/orig')
    by_index = {int(k.split('_')[-1]): k for k in remain}
    try:
        for i in range(max(by_index)+1):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f'cannot decode frame {i} of {video}')
            if i in by_index:
                if camera:img=cv2.undistort(img,*camera)
                ok, buf = cv2.imencode('.png', img)
                if not ok:
                    raise ValueError('PNG encoding failed')
                yield by_index[i], buf.tobytes(), 'original video frame, undistorted for People-Snapshot'
    finally:
        cap.release()


def compact_npz(src, dst, keep):
    import numpy as np
    with np.load(src, allow_pickle=True) as z:
        if 'keys' not in z:
            shutil.copyfile(src,dst);return
        keys = list(map(str,z['keys'])); inds = [i for i,k in enumerate(keys) if k in keep]
        vals = {k: (z[k][inds] if z[k].ndim and z[k].shape[0] == len(keys) and k not in {'betas','orig_wh'} else z[k]) for k in z.files}
        np.savez_compressed(dst,**vals)


def build(args):
    import lmdb
    ev = Path(args.eval_root).resolve(); out = safe_out(args.out, ev)
    zpath = safe_out(args.zip,ev) if args.zip else None
    if args.support < 1 or args.test < 1:
        raise ValueError('support and test counts must be positive')
    out.mkdir(parents=True)
    seed = zipfile.ZipFile(args.seed_zip) if args.seed_zip else None
    man = {'version':2,'purpose':'timing only; no quality-score claims', 'selection':'fixed first-sorted clip per dataset',
           'clips':[], 'files':{}, 'source_eval_root':str(ev), 'all_recon_retained_for':['neuman/bike']}
    try:
        for e,cid in CASES:
            if args.clips and cid not in args.clips.split(','):
                continue
            ds,name = cid.split('/',1); sb=ev/e/'base'/cid; sd=ev/e/'data'/cid
            fr=json.loads((sb/'frames.json').read_text());src=fr['source'];support=[k for k in fr['recon'] if k!=src][:args.support]
            rec=fr['recon'] if cid=='neuman/bike' else [src]+support
            test=fr['test'][:args.test];keep=set(rec+test+[src]);ordered=[k for k in fr['keys'] if k in keep]
            assert not set(rec)&set(test) and support and test
            tb=out/e/'base'/cid;td=out/e/'data'/cid;tb.mkdir(parents=True);td.mkdir(parents=True)
            for sub in ['images','masks']:
                (tb/sub).mkdir()
                for k in ordered:
                    shutil.copyfile(sb/sub/f'{k}.png',tb/sub/f'{k}.png')
            for p in sb.glob('*.npz'):
                compact_npz(p,tb/p.name,keep)
            fnew=dict(fr,keys=ordered,recon=rec,test=test,timing_full_counts={k:len(fr[k]) for k in ['keys','recon','test']})
            (tb/'frames.json').write_text(json.dumps(fnew,indent=1)+'\n')
            for fname in ['optim_tracking_ehm.pkl','id_share_params.pkl']:
                with (sd/fname).open('rb') as f: data=pickle.load(f)
                if fname.startswith('optim_'):data={k:v for k,v in data.items() if k in keep}
                with (td/fname).open('wb') as f:pickle.dump(data,f,protocol=4)
            for fname in ['meta.json','videos_info.json']:
                data=json.loads((sd/fname).read_text())
                if fname=='meta.json':data['frames_keys']=ordered
                else:
                    for v in data.values():v.update(frames_keys=ordered,frames_num=len(ordered))
                (td/fname).write_text(json.dumps(data,indent=1)+'\n')
            old=lmdb.open(str(sd/'img_lmdb'),readonly=True,lock=False,readahead=False)
            new=lmdb.open(str(td/'img_lmdb'),map_size=1<<30)
            n=0
            try:
                with old.begin() as r,new.begin(write=True) as w:
                    for k in ordered:
                        prefix=(k+'/').encode();cur=r.cursor()
                        if cur.set_range(prefix):
                            for key,val in cur:
                                if not key.startswith(prefix):break
                                w.put(key,val);n+=1
                new.sync()
            finally:
                old.close();new.close()
            od=out/'orig'/cid;od.mkdir(parents=True);provenance={}
            orig_keys=list(dict.fromkeys([src]+support+test))
            for k,buf,note in original_frames(ev,e,cid,orig_keys,seed):
                (od/f'{k}.png').write_bytes(buf);provenance[k]=note
            man['clips'].append({'id':cid,'dataset':ds,'clip':name,'E':e,'source':src,'recon':rec,'support':support,'test':test,
                                 'orig_keys':orig_keys,'original_provenance':provenance,'full_counts':fnew['timing_full_counts'],
                                 'lmdb_entries':n,'source_parameter_sha256':sha(sd/'optim_tracking_ehm.pkl')})
            print(cid,'support',len(support),'recon',len(rec),'test',len(test),'LMDB entries',n,flush=True)
    finally:
        if seed:seed.close()
    if not man['clips']:
        raise ValueError('no clips selected')
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='lock.mdb':man['files'][str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    (out/'manifest.json').write_text(json.dumps(man,indent=1)+'\n')
    if zpath:
        zpath.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.name!='lock.mdb':z.write(p,f'{out.name}/{p.relative_to(out)}')
        zpath.with_suffix(zpath.suffix+'.sha256').write_text(f'{sha(zpath)}  {zpath.name}\n')
        print('ZIP',zpath,zpath.stat().st_size,'bytes',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-root',default=DEFAULT_EVAL);p.add_argument('--out',required=True);p.add_argument('--zip',default='')
    p.add_argument('--support',type=int,default=6);p.add_argument('--test',type=int,default=12);p.add_argument('--clips',default='')
    p.add_argument('--seed-zip',default='',help='reuse exact original-resolution PNGs from the previous timing bundle')
    build(p.parse_args())
if __name__=='__main__':main()
