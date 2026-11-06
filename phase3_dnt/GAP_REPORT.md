# Why GUAVA-FB [GT] trails the per-person methods, and what fixes it (2026-11-05)

Question (user brief 2026-11-05): GUAVA-FB [SMPL-X GT] is behind GART / GaussianAvatar / ExAvatar on People-Snapshot and NeuMan.
Find the causes with evidence, fix what can be fixed under the RAGA live rules, measure it. All numbers below are PSNR unless stated;
FinalBench numbers are means over clips (`RESULTS_FINAL.md`), diagnostics are frame-weighted (`score.py`). Nothing here changes the
registered FinalBench main table (rows 1-3, B28k).

## 1. The gap

| | UBody (30) | People-Snapshot (4) | NeuMan (4) | X-Humans (3) | BEDLAM2 (40, synthetic) |
|---|---|---|---|---|---|
| GUAVA-FB [GT] | 19.62 | 27.83 | 23.42 | 24.90 | 27.19 |
| GART | 19.69 | 31.37 | 25.02 | 26.70 | 28.17 |
| GaussianAvatar | 19.46 | 30.37 | 23.76 | 25.42 | 26.79 |
| ExAvatar (GT face) | 19.99 | 30.67 | 24.39 | 24.30 | — |

By view change between the source photo and the test frame (frames; GUAVA-FB [GT] vs GART):

| | [0,30°) | [30,60°) | [60,90°) | [90,135°) | [135,180°] |
|---|---|---|---|---|---|
| People-Snapshot | 29.64 vs 30.70 (75) | 27.01 vs 30.96 (20) | 27.19 vs 33.11 (23) | 26.72 vs 32.82 (31) | 26.68 vs 31.02 (41) |
| NeuMan | 23.72 vs 24.62 (13) | 22.12 vs 25.44 (6) | 22.44 vs 25.09 (1) | 22.17 vs 25.02 (4) | 23.31 vs 25.29 (3) |
| X-Humans | 26.46 vs 27.34 (175) | 26.02 vs 27.39 (175) | 25.39 vs 26.88 (192) | 24.93 vs 26.69 (303) | 24.31 vs 26.40 (263) |

The gap is ~1 dB where the test view is close to the photo and 4-6 dB once the person has turned. On People-Snapshot 115 of 190
test frames are turned views.

## 2. Causes (evidence -> mechanism -> discriminating experiment -> result -> confidence)

**C1. Missing information about the sides and back (the far-view gap; ~70 % of the PS gap).**
Evidence: the view-bin table; the per-person methods train on the full rotation. Mechanism: texels of surfaces the photo does not
show get zero input features and are hallucinated by the UV decoder. Experiment (D1, `fusion_gt_row.py`): feed-forward fusion of the
clip's training frames into the UV map with GT poses, no optimisation. Result: ValBench BEDLAM2 26.13 -> 27.00 (GART 26.71; far bins
within 0.15 dB of GART), ValBench UBody 19.06 -> 19.57, NeuMan 23.04 -> 24.14; unseen torso / legs / arms on PS 22.4 / 24.5 / 21.5 ->
26.1 / 26.8 / 23.6 (GART 29.1 / 30.8 / 26.6). Confidence: high that it is the dominant far-view cause; the fusion recovers it fully
only on exact-pose data (synthetic), partly on real video (misaligned fitted poses, cloth motion).

**C2. Coverage misses: hair, shoes, hands, clothing outside the body model (the near-view gap).**
Evidence (D2 / D3): on PS the "off-body" class is 2.6 % of the pixels and 43 % of GUAVA-FB's squared error (PSNR 15.3 vs GART 21.8);
3.27 % of the person's pixels are left black (GART 0.61 %). Oracle: filling the missed pixels with the GT lifts GUAVA-FB 27.95 ->
29.14 (+1.19) but GART only +0.34; at [0,30°) 29.64 -> 30.88 vs GART 31.09. Mechanism: Gaussians are bound to the SMPL-X surface
with small offsets and no free Gaussians; H2 (offset hinge 3 -> 6) did not change it. Confidence: high. **Not fixed in this
phase** (needs a geometry change at training time: shell / free Gaussians or a coverage objective; post-freeze).

**C3. The UV decoder is out of distribution on fused maps.**
Evidence (D1b): on PS the naive fusion loses 1.5 dB on the near-view bin (29.64 -> 28.12) and doubles the misses; keeping the
source's own texels untouched and filling only the unseen ones (fill arm) still gives 27.91. So it is not averaging blur: any
non-zero content in texels that were always zero during training changes the decoder's geometry and colours everywhere. Confidence:
high. **Addressed by phase 7** (below).

Ruled out or minor: a global colour / exposure shift (+0.25 dB upper bound, GART +0.10); spill outside the mask (GUAVA-FB spills
less than GART); pose GT quality (3 mm SMPL-X fit; the self-view renders at 31.6 dB); fine detail (high-pass residual gap 0.3 dB at
near views vs 2.2 dB in the low frequencies); baseline unfairness (audit of the three adapters: training-frame pose refinement only,
allowed by rule 3; no test-frame fitting in the conforming rows; our protocol is stricter than all three papers' PS protocols, which
fit test poses).

## 3. The implemented solution: fusion-aware training (phase 7) and its measured result

`guava_fb/fb_fusion.py` + `guava_fb_train.py --fuse-kmax 4 --fuse-p1 0.3 --fuse-mode mean --fuse-prior 2`: during training each
source is fused with 0-4 other frames of the same person (never a target), so the same decoder learns to render from one photo and
from an accumulated map. At inference nothing changes per frame: the fusion is the background worker of the live system (phase 6
step 2), and the benchmark row fuses the clip's training frames once.

**Round 1 (PREREG Phase 7 + amendment 1, RUNLOG 116-117; 4,000 steps from B28k at lr 8e-6, the H2 continuation protocol):** arm A
(fusion), B (control, plain continuation), C (fusion + exposure / colour jitter of the fused frames, from the laptop finding of an
over-exposed reference frame). ValBench-v1 (frame-weighted): single photo B28k 20.98 dB; naive fusion B28k 21.59 (prior 1 or 2
identical); A fused 21.49 / B fused 21.48 / C fused 21.49; A single 20.91 / B single 20.94 / C single 20.90. **Registered rule
fails** (needs 21.89); attribution A - B = +0.01 dB; the only consistent effect of fusion training is LPIPS (fused 0.1523 vs
control 0.1551). Plain continuation itself costs 0.1 dB on the fused UBody row. Exposure-shift probe (fused frames x1.2
brightness, x0.75 saturation): every model loses ~0.6 dB (A 20.88, C 20.94, B28k 20.88), C's gain of +0.06 over A is below the
registered +0.20, so the jitter is not adopted; the probe does show that the laptop exposure mismatch is worth fixing at capture.
Audit (user question "check bug or data range issues", `diag_fuse_signal.py`): fusion happened in 64 % of items (35 % of the map
touched); images in [0, 1]; step-0 validation = B28k; source-only decode = static build bit for bit; the training signal exists
(36 % of a target's visible surface is unseen by the source, 54 % of that covered by the fused frames: ~21 % of each target is
rendered from real fused features). The decoder simply did not move: fused-batch image loss 0.0307 vs control 0.0313. Conclusion:
no bug; the continuation protocol (8e-6, 4,000 steps) is too weak to change the decoder's behaviour, as it was for H2 / H4 / step 1.

**Round 2 (amendment 2, launched 2026-11-06 04:03 after the user's go-ahead):** A2 = fusion at lr 3e-5 for 12,000 steps, B2 = its
control; same rules. Descriptive People-Snapshot / NeuMan fused rows of A / B / B28k (test, no selection) are being computed to
see whether the training touched the near-view loss (C3), which ValBench cannot show (naive fusion does not lose on near views there).

What this can and cannot deliver, stated before the runs: it can make the fused row clearly better than today's naive fusion and
competitive with the per-person methods where poses are exact (BEDLAM2) and on NeuMan; it cannot close the People-Snapshot gap in
the single-image row (C1 is an information bound) and it does not touch C2.

## 4. RAGA

Per-frame path unchanged (same network, Gaussian count, render); fusion runs in the background worker; the laptop checks of the
RAGA skill (30 fps, identical-render / ground-truth checks) apply to the checkpoint swap and must be run on the laptop.

## 5. Publication assessment

Supported contribution if phase 7 passes: a feed-forward, animatable, live full-body avatar whose appearance accumulates from a
monocular stream without optimisation, trained to be consistent between the single-photo and the accumulated state; evaluated against
per-person optimisation with the same frames. Closest prior work: GenFusion (arXiv 2603.28997, online feature averaging, novel view
only, ~1 fps, diffusion renderer), LHM++ (batch multi-image fusion), FastAvatar (heads). Reviewer objections to expect: (i) per-person
methods still lead on People-Snapshot by ~3 dB (answer: C1/C2 analysis, exact-pose results, zero per-person training, live); (ii)
the fusion rule is simple (answer: the contribution is the trained consistency and the measured decoder effect, with the naive rule
as the ablation); (iii) only 4 + 4 real clips in PublicBench (answer: X-Humans 3, BEDLAM2 40, and a larger real set below).
Honest status today: no main-table superiority over GART on People-Snapshot; the gap is explained and two of its three causes have
a measured, registered intervention.

## 6. Datasets (survey, agent report 2026-11-05; nothing downloaded)

Ranked for a single-image / streaming avatar with unseen identities, 360° views, GT poses and monocular video for the baselines:
1. MVHumanNet (4,500 identities, 48 cameras, SMPL-X; one camera = the monocular video); 2. DNA-Rendering (60 cameras, predefined
novel-view / novel-pose / novel-identity splits, Part 1 = 192 GB); 3. X-Humans (already in use; face / hands). Both 1 and 2 are
form-gated and multi-TB, so a fixed published subset is needed; each download is > 20 GB and needs approval. On these the
monocular video given to the baselines shows roughly what a photo shows (people do not turn 360°), so the far-view penalty hits
every method and the comparison is informative rather than a turntable regime that maximally favours per-person fitting.

## 7. Reproduction

Diagnostics: `/dev/shm/ubx858_eval/d1_driver.sh {reg,oracle,fill,wta}`; `diag_oracles.py`, `diag_regions.py`, `view_bins.py`,
`fig_gap.py` (RUNLOG 114; copies under `O/reports/gap_diag/`). Phase 7: `/dev/shm/ubx858_eval/p7_smoke.sh`, `p7_valcheck.sh`;
the registered arms' commands are in PREREG "Phase 7". Checkpoint under test: `O/runs/guava_fb_B30k/checkpoints/best.pt` (B28k,
SHA-256 3ee8e012...d0ab).
