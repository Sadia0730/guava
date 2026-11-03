"""Write E/renders/exavatar/<dataset>/<clip>/run.json (contract keys + a timing breakdown)."""
import argparse
import json
import os.path as osp
import subprocess

from exa_common import EXA_ROOT

EXA_SRC = osp.join(EXA_ROOT, 'src', 'ExAvatar_RELEASE')

LOADER_CHANGES = [
    "cam_params/<idx>.json per frame from our K (R = I, t = 0, focal = (fx, fy), princpt = (cx, cy)) instead of "
    "make_virtual_cam_params.py's constant focal 2000 (ExAvatar's 'virtual camera' loader path, unchanged code)",
    "frames: our 512 crops; fitting folders (DECA, mmpose, face fit) get the raw crops, the avatar folder gets the recon "
    "crops composited on black with the given alpha (image * alpha/255): the per-frame crop cameras make the background "
    "non-static, which breaks both of ExAvatar's background routes (static camera + Depth-Anything, or COLMAP)",
    "masks/<idx>.png = given StyleMatte alpha (ExAvatar binarises at 0.5 itself) instead of SAM (segment-anything not run)",
    "smplx_init = GT SMPL-X 2020 converted to ExAvatar's layer (SMPL-X v1.1 male, 100 betas): poses copied, hands = "
    "absolute - v1.1 hand mean, betas100 by least squares on the GT rest vertices, joint_offset = GT rest joints - "
    "ExAvatar rest joints, trans re-anchored to the pelvis; replaces Hand4Whole (not run)",
    "fitting: fit_face_gt.py = fit.py with the body frozen at the GT (root/body/hand poses, trans, betas, joint/locator "
    "offsets not optimised; bbox translation guess disabled); FLAME + face_offset fitted as in fit.py on the recon frames, "
    "per-frame FLAME of the test frames fitted with flame_shape / face_offset frozen; renders/videos skipped",
    "smoothing: ExAvatar's smoother (savgol 9) applied to the face params only (jaw, eyes, expr), recon and test "
    "sequences separately; GT body not smoothed",
    "avatar/main/config.py: dataset = 'Custom' (the README's instructed setting); all hyper-parameters unchanged",
    "test render: animate.py's path with refined Gaussians (as test.py / eval_neuman.py), bg = black instead of white, "
    "per-frame K, GT body + ExAvatar face fit of the test frame, trans re-anchored to the trained shape's pelvis; "
    "no test-time pose fitting (NeuMan protocol's fit_pose_to_test NOT run)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', required=True)
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--gpu_name', required=True)
    ap.add_argument('--gpu_index', required=True)
    ap.add_argument('--logd', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t = {}
    for line in open(osp.join(a.logd, 'times.txt')):
        k, v = line.split()
        t[k] = float(v)
    work = osp.join(EXA_ROOT, 'work', a.sid)
    fr = json.load(open(osp.join(a.clip_dir, 'frames.json')))
    rs = json.load(open(osp.join(work, 'render_stats.json')))
    fs = json.load(open(osp.join(work, 'finalize_stats.json')))
    fit = {s: json.load(open(osp.join(work, 'fit_' + s, 'fit_face_gt_log.json'))) for s in ['recon', 'test']
           if osp.exists(osp.join(work, 'fit_' + s, 'fit_face_gt_log.json'))}  # gt-face variant: no test fit
    commit = subprocess.check_output(['git', '-C', EXA_SRC, 'rev-parse', 'HEAD']).decode().strip()
    n_recon = len(fr['recon'])
    iters = 5 * 15 * n_recon  # end_epoch 5 x (frame_list_train repeated 15x, batch 1)
    run = {
        'method': 'ExAvatar',
        'commit': 'mks0601/ExAvatar_RELEASE@' + commit,
        'train_s': t['train_end'] - t['train_start'],
        'render_s_per_frame': rs['render_s_per_frame'],
        'gpu': a.gpu_name,
        'iterations': iters,
        'notes': ('Per-clip ExAvatar trained on the %d recon frames (all of them; the README\'s optional 5-fps subsampling '
                  'not applied), default custom-video config (5 epochs x 15 repeats, lr 1e-3, smplx_param_lr 1e-4 '
                  'training-frame refinement kept). Body = GT SMPL-X 2020 converted to ExAvatar\'s male v1.1 layer; face '
                  '(FLAME expr / jaw / eyes, face_offset, face texture) = ExAvatar\'s own pipeline (DECA + DWPose keypoints '
                  '+ its fitting losses, ID from recon frames only, per-frame face fit on test frames). Hand4Whole, SAM, '
                  'COLMAP not run. train_s includes DECA, mmpose, face fitting of recon AND test frames, unwrap, '
                  'Depth-Anything, training. GPU index %s.' % (n_recon, a.gpu_index)),
        'loader_changes': LOADER_CHANGES,
        'n_recon': n_recon, 'n_test': len(fr['test']),
        'timing_s': {
            'input_copy_from_E': t['copy_end'] - t['copy_start'],
            'prep_gt_conversion': t['prep_end'] - t['train_start'],
            'deca_recon_test': t['deca_end'] - t['prep_end'],
            'mmpose_recon_test': t['mmpose_end'] - t['deca_end'],
            'face_fit_recon_and_unwrap': t['fit_recon_end'] - t['mmpose_end'],
            'face_fit_test': t['fit_test_end'] - t['fit_recon_end'],
            'finalize_and_depth_anything': t['preprocess_end'] - t['fit_test_end'],
            'preprocessing_total': t['preprocess_end'] - t['train_start'],
            'avatar_training': t['train_end'] - t['preprocess_end'],
            'test_render_total_incl_load': t['render_end'] - t['train_end'],
            'render_model_load': rs['model_load_s'],
        },
        'face_fit_opt_steps': {s: fit[s]['opt_steps'] for s in fit},
        'deca_valid_frames': {s: fit[s]['flame_valid_frames'] for s in fit},
        'conversion': fs.get('conversion'),
        'joint_offset_mm_mean': fs.get('joint_offset_mm_mean'),
        'pelvis_reanchor_mm': rs['pelvis_reanchor_mm'],
    }
    with open(a.out, 'w') as f:
        json.dump(run, f, indent=1)
    print(json.dumps(run, indent=1))


if __name__ == '__main__':
    main()
