"""Offset check (IMPROVE_NOTES, H2 manipulation check): per checkpoint, UV-Gaussian local offset norms (|local_pos|, the quantity
GUAVA's threshold_local_xyz hinges) and world offsets (cm) to the bound SMPL-X face at the source pose, on the 5 diagnosis-A clips.
Usage (from phase3_dnt/guava_fb, GUAVA env): python diag_offsets_h2.py <out.json> [--E <ValBench root>] [--models label=ckpt ...]"""
import argparse, sys, json, os, numpy as np, torch
sys.path.insert(0, '.'); import fb_common as F
from fb_common import to_dev
R = '/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs'
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--E', default='/dev/shm/ubx858_eval/valbench_v1')
ap.add_argument('--models', nargs='+', default=[f'B28k={R}/guava_fb_B30k/checkpoints/best.pt', f'H2ctrl={R}/guava_fb_H2ctrl/checkpoints/latest.pt', f'H2off6={R}/guava_fb_H2off6/checkpoints/latest.pt'])
a = ap.parse_args(); a.out = os.path.abspath(a.out); os.chdir(F.ROOT)
from models.UbodyAvatar import Ubody_Gaussian                                                        # noqa: E402
CK = dict(m.split('=', 1) for m in a.models)
dev = torch.device('cuda:0'); cfg, inf, ren, _ = F.setup_a20(dev)
EV = a.E; out = {}
for name, p in CK.items():
    st = torch.load(p, map_location='cpu', weights_only=False); inf.load_state_dict(st['model']); inf.eval(); step = st.get('step'); del st
    for cid in ('bedlam2/20240808_1_250_ai1105_vcam__seq_000050', 'bedlam2/20240808_1_250_ai1105_vcam__seq_000179', 'bedlam2/20240731_1_1827_hdri__seq_000568', 'ubody/Online_class_S10_Trim11', 'ubody/SignLanguage_S182'):
        ds, cl = cid.split('/'); fj = json.load(open(f'{EV}/base/{ds}/{cl}/frames.json'))
        c = F.load_clips_parallel([(f'{EV}/data/{ds}/{cl}', None, True)], workers=1); fbd = F.FBData(cfg, c)
        with torch.no_grad():
            torch.manual_seed(0); vg, ug, _ = inf(to_dev(F.collate1(fbd.source_info(0, fj['source'])), dev))
            av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
            n = ug['local_pos'][0].norm(dim=-1)
            t = to_dev(F.collate1(fbd.target_info(0, fj['source'])), dev); av(t)
            d = (av._uv_xyz_deform[0] - torch.einsum('nk,nkj->nj', av._uv_face_bary, inf.ehm(t['smplx_coeffs'], t['flame_coeffs'])['vertices'][0][inf.ehm.smplx.faces_tensor][av._uv_binding_face])).norm(dim=-1) * 100
        q = lambda x: [round(float(v), 2) for v in torch.quantile(x.float(), torch.tensor([0.5, 0.9, 0.99, 0.999], device=x.device))]
        r = {'step': step, 'local_q50_90_99_999': q(n), 'frac_gt3': round(float((n > 3).float().mean()), 4), 'frac_gt6': round(float((n > 6).float().mean()), 4), 'world_cm_q50_90_99_999': q(d)}
        out.setdefault(cl, {})[name] = r; print(name, cl[-22:], r, flush=True)
json.dump(out, open(a.out, 'w'), indent=1)
