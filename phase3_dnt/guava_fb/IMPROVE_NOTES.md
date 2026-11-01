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
