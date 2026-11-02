# GUAVA-FB improvement phase — working notes (validation data only; PREREG "GUAVA-FB improvement phase 1")

Baseline: `O/runs/guava_fb_B30k/checkpoints/best.pt` (28k; SHA-256 3ee8e012...d0ab). GART trains on the first half of each clip;
GUAVA-FB sees one image. AvatarTest-v1 (RUNLOG 85) is fixed and descriptive only.

## ValBench-v1 (validation replica of the test protocol)
`phase3_dnt/eval_split/valbench_v1.json` (build_valbench.py): UBody 20 clips (a20's 5 val clips with >= 60 GT frames + 15
seeded never-tracked training-split clips, identity-disjoint), BEDLAM2 30 clips (GUAVA-FB's 10 val clips + subjects 11-30 of the
same rule); 3,512 test frames. Converted with convert_test.py into `/dev/shm/ubx858_eval/valbench_v1` (copy to O/eval/valbench_v1).

## A. Quality under GT inputs (diag_regions.py; `/dev/shm/ubx858_eval/diag_regions_v1/`)
Image level (PSNR / SSIM / LPIPS): UBody FB 19.06 / 0.783 / 0.187, GART 20.06 / 0.809 / 0.177, FB raw 18.11 / 0.758 / 0.205;
BEDLAM2 FB 26.13 / 0.918 / 0.070, GART 26.71 / 0.918 / 0.073 (same gap pattern as AvatarTest-v1).
Share of squared error (SSE) by class, FB vs GART:
- off-body (mask pixels no GT body surface covers: hair, loose garments, held objects, StyleMatte extras): UBody 4.4 % of pixels,
  23.8 % of FB's SSE, PSNR 10.9 (GART 12.9); BEDLAM2 2.4 % of pixels, 32.7 % of FB's SSE, PSNR 13.9 (GART 16.2). Largest class.
- visible body: UBody arms 14.0 vs 17.0 dB (21.5 % SSE), head 17.9 vs 19.4, torso 16.6 vs 17.5 (GUAVA-FB better on legs);
  BEDLAM2 seen legs 19.3 vs 21.0, seen head 24.7 vs 26.1, hands FB better.
- source-unseen surface: BEDLAM2 ~13.5 % of FB SSE; GART is NOT better there except via views it saw in its training half.
- spill outside mask: GART worse on BEDLAM2 (25.6 % of its SSE vs FB 9.8 %).
- motion: GART's lead is largest at LOW motion on UBody (+1.6 dB vs +0.65 at high) -> reconstruction fidelity of visible content.
- raw vs refined: refiner +0.9 dB (UBody) / +0.6 (BEDLAM2), cuts spill 10.7 -> 7.7 % (UBody).
Per clip: GART better on 14/20 UBody clips (median +1.23 dB), 19/30 BEDLAM2 (median +0.30 dB; FB has better LPIPS on 21/30).
Largest GART wins = mechanisms (diagA_worst.jpg): held objects present in the source but dropped by FB (Online_class_S10, +4.95 dB);
loose dress narrowed / hole between the legs (vcam_050 +2.0, vcam_179 +2.0); back views GART saw in training (hdri_568 +4.0);
source self-occlusion colouring the head (vcam_105 +5.8).
Gaussian offsets (B28k): dresses push 4-6 % of UV Gaussians past the 3.0 hinge (99.9 % world offset ~12 cm); other clips ~0 %.

## Self-reconstruction (source frame rendered by its own avatar; B28k)
UBody 24.25 dB / SSIM 0.906 (mid recon frame 19.59, first test frame 17.96); BEDLAM2 35.64 / 0.981 (26.41, 25.54). Texture
capacity is not the main limit; quality falls with pose / view change (non-skinned garments, unseen surfaces, occlusion).

## B. SPRINT degradation (diag_sprint.py; renders `renders/dB_*`, scores `scores/dB_*_vb.json`; all GT-assisted rows diagnostic)
PSNR UBody / BEDLAM2 (SSIM, LPIPS in the score files):
| variant | UBody | BEDLAM2 |
|---|---|---|
| GT source x GT driving (= fb_gt) | 19.06 | 26.13 |
| GT source x SPRINT root orientation only | 16.44 | 22.35 |
| GT source x SPRINT root translation only | 16.09 | 21.60 |
| GT source x SPRINT body pose only | 15.66 | 21.60 |
| GT source x SPRINT hands only | 18.98 | 26.00 |
| GT source x SPRINT face only | 19.05 | 26.13 |
| GT source x all SPRINT driving (mapped into the metric camera) | 16.87 | 20.95 |
| SPRINT source shape only, GT driving | 18.38 | 24.04 |
| SPRINT source pose + camera only, GT driving | 17.67 | 23.92 |
| SPRINT source (own crop / matte / fit), GT driving | 17.05 | 22.58 |
| SPRINT source x SPRINT driving, mapped into the metric camera | 16.81 | 21.54 |
| SPRINT source x SPRINT driving, own camera + warp (SPRINT on the metric crop image) | 16.91 | 22.09 |
| deployed row fb_sprint (SPRINT on the original frame) | 16.68 | 21.92 |
Reading: camera / intrinsics assumption costs nothing; GT-framed input +0.2 dB only; 3D components are not additive (SPRINT's root
and body errors cancel in 2D); source-only and driving-only costs overlap on UBody (~2 dB each, ~2.2 together) -> per-frame 2D
alignment + source fit consistency, not any single 3D component. Detector misses (no person) crashed the deployed pipeline on one
BEDLAM2 clip (source frame + 3 frames): tracking fallback added (previous box, else whole image).

## Hypotheses (PREREG phase 2): H1 driving refinement (inference) > H2 offset hinge 3 -> 6 (training) > H3 source fit.

## Pilots (PREREG phase 2; ValBench, 3,512 test frames; scores `O/reports/improve/valbench_scores/<row>_vb.json`)
One factor per arm, against the arm's control; PSNR / SSIM / LPIPS, frame-weighted; per-clip counts vs the control.
| arm | UBody (20 clips) | BEDLAM2 (30) | all | per clip vs control | cost (idle L40S) | decision |
|---|---|---|---|---|---|---|
| SPRINT row, deployed (raw driving k = 0, largest box) | 16.68 / 0.732 / 0.2345 | 21.92 / 0.867 / 0.1111 | 18.10 / 0.769 / 0.2009 | control | driving 239-266 ms / frame | - |
| H1 driving refinement k = 20 | 17.79 / 0.755 / 0.2148 | 23.64 / 0.889 / 0.0921 | 19.39 / 0.792 / 0.1814 | PSNR up 49 / 50 | +35 ms | not chosen (57 % of k = 60's gain) |
| H1 driving refinement k = 60 | 18.93 / 0.774 / 0.1989 | 24.21 / 0.895 / 0.0873 | 20.37 / 0.807 / 0.1685 | PSNR up 50 / 50 | +109-116 ms GPU (refine) | **adopted** |
| H1b + subject tracking, stale-box fallback (vs k = 60) | 19.19 / 0.786 / 0.1934 | 24.01 / 0.892 / 0.0904 | 20.50 / 0.815 / 0.1653 | 2 up (+4.13, +3.24), 3 BEDLAM2 down (-3.24, -2.36, -0.62) | 0 | **rejected** |
| H1c + subject tracking, nearest-centre fallback (vs k = 60) | 19.19 / 0.786 / 0.1934 | 24.21 / 0.895 / 0.0873 | 20.55 / 0.816 / 0.1645 | 2 up (+4.13, +3.24), 48 within 1e-5 dB | 0 | **adopted** |
| H3 (phase 4) H1c tracking at k = 0 (vs k = 0) | 16.83 / 0.741 / 0.2316 | 21.92 / 0.867 / 0.1111 | 18.22 / 0.775 / 0.1988 | 2 up (+3.46, +1.82), 48 identical | 0 | **adopted for k = 0** (4.7 % of k60 + H1c's gain) |
| GT row, B28k (start of H2) | 19.06 / 0.783 / 0.1872 | 26.13 / 0.918 / 0.0700 | 20.98 / 0.820 / 0.1553 | - | build 54 ms, render 12.2 ms | reference |
| H2 control: 4k continuation, hinge 3.0 (vs B28k) | 19.00 / 0.782 / 0.1879 | 26.12 / 0.918 / 0.0702 | 20.94 / 0.819 / 0.1559 | -0.27 .. +0.36 dB | 1.0 h train | control; UBody not eligible at 4k |
| H2 hinge 3.0 -> 6.0 (vs control) | 19.00 / 0.782 / 0.1879 | 26.11 / 0.918 / 0.0702 | 20.94 / 0.819 / 0.1559 | -0.13 .. +0.05 dB | 1.0 h train; same speed | **not supported** |
| GART (trains on the first half of each clip) | 20.06 / 0.809 / 0.1765 | 26.71 / 0.918 / 0.0732 | 21.87 / 0.839 / 0.1484 | - | per-clip training | reference |

H2 details (diag_regions; `O/reports/improve/diag_regions_v1/gt_H2*.json`): BEDLAM2 off-body PSNR 14.06 vs control 14.10 (rule
needs >= +0.3); UBody off-body 10.95 vs 10.96; spill 7.14 vs 7.18 % (BEDLAM2). Validator at 4k: UBody SSIM 0.819254 (control
0.819324, threshold 0.819331) -> neither arm eligible at 4k. Manipulation check (`h2_offsets.json`): share of UV Gaussians beyond
3.0 on the dress clips 3.8 -> 4.1 % (vcam_050) and 5.8 -> 6.2 % (vcam_179); the hinge term was ~1.6e-4 of a ~0.2 total loss. In
a 4k continuation at lr 8e-6 the relaxed hinge barely changed the offsets, so this pilot does not test whether the hinge limits
garments when training from scratch. The control's own 4k drift (-0.27 .. +0.36 dB per clip vs B28k) is larger than the arm
difference: a training intervention needs > ~0.3 dB per-clip effects to be visible at this scale.

H1 where the gain sits (diag_silhouette.py, band 5 px around the StyleMatte matte; `diag_silhouette_h1.json`): UBody 74-76 % of
the SSE reduction is inside the eroded matte (body-part alignment), 8-9 % in the band, 16-17 % outside; BEDLAM2 36 / 34 / 30 %,
the same profile as replacing SPRINT driving by GT driving (34 / 42 / 24 %). Miss / spill of the matte: UBody 14.8 / 7.3 % ->
5.8 / 5.7 % (k = 60 + H1c), BEDLAM2 8.9 / 16.3 % -> 5.1 / 9.5 % (GT-driven 3.1 / 6.8 %). Caveat: the refinement fits each target
frame's own DWPose keypoints and StyleMatte matte (available at inference, no GT) and the metric's mask is the same StyleMatte
matte; on UBody (pseudo-GT poses) the refined row spills less than the GT-driven row (5.7 vs 7.7 %) and has higher PSNR (19.19 vs
19.06) with worse LPIPS (0.193 vs 0.187), so it is NOT evidence that SPRINT driving beats GT driving. H1c's UBody source subject is
the split's subject box (a user's click); BEDLAM2 takes the largest box.

Timing / memory (idle GPU 5, L40S; `O/reports/improve/timing_models/timing.json`, `h1_timing_mem.json`, `h1_timing.json`):
avatar build (creation from one image, model only) B28k 54.5 ms, H2 arms 54.9 / 55.2 ms; render 12.24 ms / frame (p90 14.7) for
all three; 207k Gaussians; peak 4.0 GB. SPRINT per driving frame (median of 178 frames, two sessions): k = 0 239-266 ms, k = 60
324-365 ms; the GPU refinement adds 109-116 ms in both sessions, DWPose's CPU part (130-177 ms) moves with host load; peak SPRINT
memory 1.56 GB for k = 0 and k = 60. Creation with SPRINT (source fit, 150 iterations) is unchanged by H1 / H1c.

Visuals (rows picked by fig_examples.py's fixed rules; middle test frame): `O/reports/improve/figures/sprint_examples.jpg`
(typical = median-change clip, worst = smallest gain, plus the two wrong-person clips; remaining failure: Online_class_S2, where
SPRINT driving crouches the lower body of a standing person and k = 60 does not correct it) and `h2_examples.jpg` (typical, worst = dress vcam_050, dress vcam_179).

## Decision (2026-11-01)
- SPRINT pipeline: adopt k = 60 per-frame driving refinement + H1c subject tracking (inference only; model unchanged = B28k, so the
  registered UBody / face requirements hold as before). ValBench all-frame 18.10 -> 20.55 dB, LPIPS 0.2009 -> 0.1645; cost +109-116
  ms GPU per driving frame, no memory change, creation unchanged.
- Avatar under GT inputs: no change. H2 (the one training intervention tested) failed its rule; the gap to GART on ValBench stays
  UBody -1.00 dB / BEDLAM2 -0.58 dB (off-body pixels remain the largest error class).
- AvatarTest-v1: one final evaluation of the selected SPRINT pipeline as an extra row (PREREG "final test evaluation"), pending the
  user's go-ahead (GPU job > 15 min).

## Reproducible commands (repo root /home/ubx858/guava; O = /vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb)
Environment: `export PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=<gpu>`;
PY = /home/ubx858/miniconda3/envs/guava/bin/python; EV = a local copy of `O/eval/valbench_v1` (random LMDB reads over SMB are slow).
```
cd phase3_dnt/eval_split
$PY eval_guava_rows.py --row fb_sprint --drive-iters 60 --track-subject --track-nearest --name fb_sprint_k60_trkc --split valbench_v1.json --E $EV   # selected pipeline
$PY eval_guava_rows.py --row fb_sprint --drive-iters {0,20,60} [--track-subject] --split valbench_v1.json --E $EV                                       # H1 / H1b arms
$PY score.py --methods fb_sprint,fb_sprint_k60,fb_sprint_k60_trkc --split valbench_v1.json --E $EV --tag vb
$PY time_sprint.py --ks 0,60 --n 200 --out h1_timing_mem.json                                                      # idle GPU
$PY diag_silhouette.py --E $EV --methods fb_sprint,fb_sprint_k60,fb_sprint_k60_trkc,fb_gt,gart --base fb_sprint --out diag_silhouette_h1.json
$PY diag_regions.py --E $EV --split valbench_v1.json --methods gt_H2ctrl,gt_H2off6 --out <dir>; $PY diag_report.py <dir> fb_gt,gt_H2ctrl,gt_H2off6
$PY fig_examples.py --E $EV --cols "GT,fb_sprint:SPRINT k=0 (deployed),fb_sprint_k60:k=60 (H1),fb_sprint_k60_trkc:k=60 + H1c,fb_gt:GT-driven (diag.),gart:GART (half clip)" --pair fb_sprint,fb_sprint_k60_trkc --extra ubody/Olympic_S6_Trim23_scene001,ubody/ConductMusic_S29_Trim13 --out sprint_examples.jpg
$PY fig_examples.py --E $EV --cols "GT,fb_gt:B28k (start),gt_H2ctrl:H2 control 4k,gt_H2off6:H2 hinge 6 4k,gart:GART (half clip)" --pair gt_H2ctrl,gt_H2off6 --extra bedlam2/20240808_1_250_ai1105_vcam__seq_000179 --out h2_examples.jpg
cd ../guava_fb
$PY guava_fb_train.py --out guava_fb_H2off6 --offset-threshold 6.0 <the RUNLOG 89 options>                         # H2 arm (control: 3.0)
(cd ../eval_split && $PY eval_guava_rows.py --row fb_gt --name gt_H2off6 --fb-ckpt $O/runs/guava_fb_H2off6/checkpoints/latest.pt --split valbench_v1.json --E $EV)
$PY compare_fb.py --part timing --models B28k=$O/runs/guava_fb_B30k/checkpoints/best.pt H2ctrl=$O/runs/guava_fb_H2ctrl/checkpoints/latest.pt H2off6=$O/runs/guava_fb_H2off6/checkpoints/latest.pt --out <dir>
$PY diag_offsets_h2.py h2_offsets.json   (from phase3_dnt/guava_fb; B28k / H2ctrl / H2off6 by default; reproduces h2_offsets.json exactly)
```

## Phase 4 (2026-11-02): options that keep the live cost (PREREG phase 4; user request after a reviewer note)
- **H3, H1c tracking alone at k = 0:** ValBench 18.10 -> 18.22 dB, LPIPS 0.2009 -> 0.1988 (UBody 16.68 -> 16.83; BEDLAM2 unchanged);
  only the two wrong-person clips change (+3.46 / +1.82 dB). Adopted for the k = 0 pipeline (no cost). It is 4.7 % of the selected
  pipeline's +2.45 dB: almost all of H1's gain is the per-frame refinement, not the subject choice.
- **ValBench UBody is SPRINT training data.** All 20 ValBench UBody clips are in SPRINT's UBody training rows (1,932 rows of these
  clips, every 3rd annotated frame; 25,701 rows of their identities). Every SPRINT row on ValBench UBody is therefore measured on
  clips SPRINT was fitted to (with UBody's pseudo-GT). ValBench BEDLAM2 and all AvatarTest-v1 clips are unseen by SPRINT at the clip
  level (BEDLAM2 val jobs; UBody official test lists), but BEDLAM2 subjects recur across jobs (38 / 40 AvatarTest BEDLAM2 identities
  appear in sequences SPRINT trained on); AvatarTest UBody identities are not in SPRINT's training. The H1 / H1c
  / H3 decisions compare arms with the same SPRINT and stand; the size of H1's gain on unseen real clips is not measured there.
  For SPRINT-side changes: **ValBench-S** (`eval_split/valbench_s.json`; 13 UBody clips, 10 identities, 1,325 test frames) = clips
  of SPRINT's own UBody val split (frames never used for SPRINT), not GUAVA-FB / a20 training clips, identities disjoint from
  ValBench-v1 / AvatarTest-v1; 4 clips have identities SPRINT never saw, 9 share an identity with other SPRINT / GUAVA-FB clips.
- **Live cost of the SPRINT loop (CPU timing, `sprint_distill/time_detector.py`, 1280x720, 16 threads, host load ~10):** the YOLOX-L
  person detector alone is 154 ms / frame (median; p90 160), the DWPose-L keypoints 39 ms. So of the 239-266 ms of a deployed k = 0
  driving frame, the detector is the largest part; SPRINT's forward is 7 ms; the StyleMatte matte (63 ms) and the keypoints are used
  only by the k = 60 refinement (+115 ms GPU). A one-pass (k = 0) loop that keeps the per-frame CPU detector stays at ~165 ms / frame;
  moving the detector to the GPU or detecting every n-th frame (box from the previous fit in between) is the larger live-cost lever.
- **Latent bug:** an empty StyleMatte matte makes the refinement's mask term NaN (its distance transform has no zero), so the fit
  diverges (9 / 200 rows of the context-crop label smoke, 1 / 200 on original frames). Not fixed yet; not checked on ValBench /
  AvatarTest frames.
- **H4, SPRINT trained on its own offline H1 fits:** registered (PREREG phase 4, amendments 1-2); labels on original frames
  (smoke: 162 / 200 rows kept, keypoint error 23.3 -> 8.4 px at 1024, H1 moves SPRINT's raw root / body / hands by 2.1 / 6.6 / 18.0
  deg, frames match the stored crops within 0.7 / 255); pilot waits for the GPU go-ahead.
- **(iii) avatar training without architecture / Gaussian-count change:** not run. GUAVA-FB's training pairs are already GT
  (BEDLAM2, 4D-DRESS), SPRINT + refinement refits (UBC, Motion-X clips that fail the alignment filter) or EHM tracks (UBody), so
  "better aligned pairs" has little headroom by construction; extra garment / geometry supervision was tested (pilots C, D) and did
  not improve appearance; a 4k continuation drifts +-0.3 dB per clip on its own (H2 control), larger than the effects in question.
  A credible test needs a new mechanism and a from-scratch run (~7.5 h at 1.1 it/s).
