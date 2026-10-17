"""DNT Track A: one table (author's protocol COCO / LSP-Extended / 3DPW test, standard 3DPW test, EHF) for the PEAR-free
student next to L-real, L-real + refiner v2b and PEAR, plus the PREREG "circularity answered" check. Every number is read
from the evaluators' summary files (paths below); nothing is typed in."""
import json, sys
from pathlib import Path

A = Path('/raid/ubx858/outputs/phase2/author'); E = Path('/raid/ubx858/outputs/phase1_task1/eval_ehf')
S = Path('/raid/ubx858/outputs/phase1_task1/eval_3dpw'); T = Path('/raid/ubx858/outputs/dnt/trackA/eval')
ROWS = [  # name, author dir, 3DPW-standard dir, EHF dir
    ('PEAR (released ViT-H)', A / 'pear', S / 'pear/test_gt_keypoints', E / 'pear'),
    ('L-real (PEAR-distilled)', A / 'Lreal', S / 'L_real_best/test_gt_keypoints_fp32', E / 'L_real_best_fp32'),
    ('L-real + refiner v2b', A / 'Lreal_refinerv2b', None, E / 'Lreal_refinerv2b_fp32'),
    ('**Track A best.pt, live (PREREG-selected)**', T / 'A_best_live/author', T / 'A_best_live/3dpw_std', T / 'A_best_live/ehf'),
    ('Track A best.pt, EMA (step 82k)', T / 'A_best_ema/author', T / 'A_best_ema/3dpw_std', T / 'A_best_ema/ehf'),
    ('Track A latest.pt, EMA (step 128k; descriptive)', T / 'A_latest_ema/author', T / 'A_latest_ema/3dpw_std', T / 'A_latest_ema/ehf'),
]
DIAG = [  # descriptive diagnostic: every output Track A never supervises set to its neutral constant (make_neutral.py)
    ('*diag: L-real, unsupervised outputs neutral*', T / 'diag_Lreal_neutral/author', None, T / 'diag_Lreal_neutral/ehf'),
    ('*diag: Track A best.pt live, unsupervised outputs neutral*', T / 'diag_A_best_live_neutral/author', None, T / 'diag_A_best_live_neutral/ehf'),
]

def js(p):
    p = Path(p) / 'summary.json'
    return json.load(open(p)) if p.exists() else None

def mval(x):
    return x['mean'] if isinstance(x, dict) else x

def row(name, a, s, e):
    r = {'name': name}
    for ds in ('COCO-VAL', 'LSP-EXTENDED'):
        d = js(a / ds); r[ds] = (d['pck']['0.05'], d['pck']['0.1']) if d else None
    d = js(a / '3DPW-TEST'); r['3dpw_author'] = (d['mpjpe_mm'], d['pa_mpjpe_mm']) if d else None
    d = js(s) if s else None; r['3dpw_std'] = (mval(d['metrics']['mpjpe_mm']), mval(d['metrics']['pa_mpjpe_mm'])) if d else None
    d = js(e) if e else None
    r['ehf'] = tuple(d.get('metrics_mm', d)[k] for k in ('pa_mpvpe_all', 'mpvpe_all', 'pa_mpvpe_hand', 'pa_mpvpe_face', 'pa_mpjpe_body')) if d else None
    return r

def f(t, fmt):
    return ' / '.join(fmt.format(v) for v in t) if t else '—'

def main():
    rows = [row(*x) for x in ROWS]
    diag = [row(*x) for x in DIAG]
    out = ['| Model | COCO PCK @0.05 / @0.1 | LSP-Ext PCK @0.05 / @0.1 | 3DPW test MPJPE / PA (author) | 3DPW test MPJPE / PA (standard) | EHF PA-PVE all / PVE all / hands / face / body PA-MPJPE |',
           '|---|---|---|---|---|---|']
    for r in rows + diag:
        out.append(f"| {r['name']} | {f(r['COCO-VAL'], '{:.3f}')} | {f(r['LSP-EXTENDED'], '{:.3f}')} | {f(r['3dpw_author'], '{:.2f}')} | "
                   f"{f(r['3dpw_std'], '{:.2f}')} | {f(r['ehf'], '{:.1f}')} |")
    # PREREG check, against L-real re-run under the SAME evaluator SHA (and the stored row, which must agree)
    lc = {ds: js(T / 'Lreal_current_eval/author' / ds) for ds in ('COCO-VAL', '3DPW-TEST')}
    ref = {'mpjpe': lc['3DPW-TEST']['mpjpe_mm'], 'pa': lc['3DPW-TEST']['pa_mpjpe_mm'], 'coco': lc['COCO-VAL']['pck']['0.05']}
    stored = rows[1]
    same = abs(ref['mpjpe'] - stored['3dpw_author'][0]) < 1e-6 and abs(ref['pa'] - stored['3dpw_author'][1]) < 1e-6 \
        and abs(ref['coco'] - stored['COCO-VAL'][0]) < 1e-9
    out += ['', f"L-real re-evaluated with the current evaluator (SHA {lc['3DPW-TEST']['evaluator_sha256'][:8]}): 3DPW test "
            f"{ref['mpjpe']:.2f} / {ref['pa']:.2f}, COCO @0.05 {ref['coco']:.4f}; identical to the stored row: {same}.", '',
            '| Weights | ΔPA-MPJPE (≤ +1.0) | ΔMPJPE (≤ +2.0) | ΔCOCO PCK@0.05 (≥ −0.01) | Circularity answered |', '|---|---|---|---|---|']
    verdicts = {}
    for r in rows[3:]:
        dpa = r['3dpw_author'][1] - ref['pa']; dmp = r['3dpw_author'][0] - ref['mpjpe']; dco = r['COCO-VAL'][0] - ref['coco']
        ok = dpa <= 1.0 and dmp <= 2.0 and dco >= -0.01
        verdicts[r['name']] = {'d_pa': dpa, 'd_mpjpe': dmp, 'd_coco': dco, 'pass': ok}
        out.append(f"| {r['name']} | {dpa:+.2f} {'✓' if dpa <= 1.0 else '✗'} | {dmp:+.2f} {'✓' if dmp <= 2.0 else '✗'} | "
                   f"{dco:+.4f} {'✓' if dco >= -0.01 else '✗'} | **{'PASS' if ok else 'FAIL'}** |")
    d0, d1 = diag
    dd = {'d_pa': d1['3dpw_author'][1] - d0['3dpw_author'][1], 'd_mpjpe': d1['3dpw_author'][0] - d0['3dpw_author'][0],
          'd_coco': d1['COCO-VAL'][0] - d0['COCO-VAL'][0]}
    dd['pass'] = dd['d_pa'] <= 1.0 and dd['d_mpjpe'] <= 2.0 and dd['d_coco'] >= -0.01
    out += [f"| *diag (not gated): neutral vs neutral* | {dd['d_pa']:+.2f} | {dd['d_mpjpe']:+.2f} | {dd['d_coco']:+.4f} | "
            f"*{'would pass' if dd['pass'] else 'would still fail'}* |"]
    verdicts['diag_neutral_vs_neutral'] = dd
    text = '\n'.join(out)
    print(text)
    Path(T / 'trackA_table.md').write_text(text + '\n')
    json.dump({'rows': rows, 'diag_rows': diag, 'lreal_current_eval': ref, 'lreal_identical_to_stored': same, 'verdicts': verdicts},
              open(T / 'trackA_stats.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
