# GUAVA-FB improvement phase 4: same live cost (2026-11-02)

Status: **in progress.** H3 is decided. The H4 pilot finished training and is being evaluated; its verdict is not in this file yet.
Details: `PREREG.md` (phase 4 + amendments 1-2), `RUNLOG.md` 91-99, `guava_fb/IMPROVE_NOTES.md` (phase 4 section).
All choices are made on validation data only. AvatarTest-v1 was not touched in this phase.

## Why
A reviewer's note suggested ways to improve quality without raising live per-frame cost:
1. Test H1c subject tracking alone (no k = 60 refinement).
2. Train SPRINT to predict H1's refined fits ("distillation", as in SPIN), so live inference stays one forward pass.
3. Improve avatar training without changing the architecture or the Gaussian count.

It also corrected a timing point: deployed SPRINT driving costs 239-266 ms/frame; the 12.2 ms figure is the avatar render only.

## Results so far

### H3: H1c tracking at k = 0 (adopted for the k = 0 pipeline)
ValBench, PSNR / SSIM / LPIPS:

| Row | UBody | BEDLAM2 | All |
|---|---|---|---|
| SPRINT k = 0 (deployed) | 16.68 / 0.732 / 0.2345 | 21.92 / 0.867 / 0.1111 | 18.10 / 0.769 / 0.2009 |
| k = 0 + H1c (H3) | 16.83 / 0.741 / 0.2316 | 21.92 / 0.867 / 0.1111 | 18.22 / 0.775 / 0.1988 |
| k = 60 + H1c (selected, phase 2) | 19.19 / 0.786 / 0.1934 | 24.21 / 0.895 / 0.0873 | 20.55 / 0.816 / 0.1645 |

- Only the two wrong-person clips change: +3.46 dB and +1.82 dB. The other 48 clips are identical.
- Tracking alone gives **4.7 %** of the selected pipeline's +2.45 dB gain. Almost all of H1's gain comes from the per-frame refinement.
- Cost: none.

### Where live time goes (CPU, 1280x720, 16 threads)
| Stage | ms / frame |
|---|---|
| YOLOX-L person detector (CPU) | 154 |
| DWPose keypoints (CPU, only k = 60 needs them) | 39 |
| StyleMatte matte (GPU, only k = 60 needs it) | 63 |
| SPRINT forward | 7 |
| k = 60 refinement (GPU) | +109-116 |
| Avatar render | 12.2 |

A one-pass SPRINT that still runs the CPU detector on every frame costs about 165 ms/frame. The larger saving is moving the detector to the GPU, or detecting only every n-th frame. A separate session is testing that (`NO_LATENCY_PREREG.md`).

### Finding: ValBench UBody is SPRINT training data
- All 20 ValBench UBody clips are in SPRINT's UBody training rows: 1,932 rows of those clips, and 25,701 rows of their identities.
- SPRINT rows on ValBench UBody are therefore measured on data SPRINT was fitted to. Earlier H1/H1c/H3 decisions still stand, because both arms of each comparison use the same SPRINT.
- AvatarTest-v1 and ValBench BEDLAM2 clips are unseen by SPRINT at the clip level. BEDLAM2 subjects recur, though: 38/40 AvatarTest BEDLAM2 identities appear in SPRINT's training sequences.
- **New validation set ValBench-S** (`eval_split/valbench_s.json`): 13 UBody clips from SPRINT's own validation split, with 10 identities and 1,325 test frames.
  - SPRINT never trained on these frames, and GUAVA-FB never trained on these clips.
  - The identities are disjoint from ValBench-v1 and AvatarTest-v1.

Reference rows on ValBench-S, all frames:

| Row | PSNR | SSIM | LPIPS |
|---|---|---|---|
| SPRINT k = 0 + H1c | 17.85 | 0.774 | 0.1986 |
| SPRINT k = 60 + H1c (teacher) | 20.99 | 0.826 | 0.1546 |
| GT-driven | 21.39 | 0.835 | 0.1400 |

On unseen real clips, H1 adds **+3.15 dB** and closes 89 % of the gap to GT driving. Every clip improves, by +0.86 to +4.41 dB.

### H4: SPRINT trained on its own offline H1 fits (verdict pending)
- **Labels.** 24,000 rows from SPRINT's UBody training split were sampled from 1,400 clips; eval identities were excluded.
  - Each row was fitted with H1's objective (k = 60) on the original video frame, exactly as deployed.
  - Kept: **17,388 rows (72 %)**.
  - Keypoint error on kept rows fell from 27.6 to 9.9 px (1024 px crop).
  - H1 moved SPRINT's raw root / body / hands by 1.9 / 6.8 / 18.5 degrees.
- **Training.** Two 4k-step fine-tunes from SPRINT (EMA 128k), identical except for the UBody labels:
  - control: UBody's own pseudo-GT labels;
  - arm: H1 rotations plus 2D-joint and camera targets.
- **Guard (passed).** SPRINT's own validation is unchanged within 0.13 mm:

  | Validation set (PA-MPJPE, mm) | Control | Arm |
  |---|---|---|
  | 3DPW-val | 48.98 | 49.05 |
  | BEDLAM2-val | 35.90 | 35.89 |
- **Rule (registered before any run).** On ValBench-S at k = 0 + H1c, the arm must:
  - beat the control by at least **+0.30 dB** with LPIPS no worse;
  - win on at least 2/3 of clips;
  - lose no more than 0.10 dB on ValBench BEDLAM2.
- **Status.** Evaluation of both arms is running. Results go into this file.

### Idea 3 (avatar training, same architecture): not run
- Training pairs are already GT (BEDLAM2, 4D-DRESS), SPRINT + refinement refits (UBC, Motion-X) or EHM tracks (UBody). "Better-aligned pairs" has little headroom by construction.
- Garment / geometry supervision was already tried (pilots C, D) and did not improve appearance.
- A 4k continuation drifts about ±0.3 dB per clip on its own (H2 control). A credible test needs a new mechanism and a from-scratch run of about 7.5 h.

## Baselines on AvatarTest-v1
PSNR / SSIM / LPIPS. Driving poses are SMPL-X GT except the GUAVA-FB [SPRINT] row.

70 clips (all methods except ExAvatar):

| Method | UBody | BEDLAM2 | All |
|---|---|---|---|
| GUAVA (released) | 19.89 / 0.798 / 0.184 | 24.76 / 0.905 / 0.080 | 21.32 / 0.829 / 0.153 |
| GART | 19.89 / 0.806 / 0.182 | 28.23 / 0.936 / 0.059 | 22.33 / 0.844 / 0.146 |
| GaussianAvatar | 19.43 / 0.795 / 0.195 | 26.83 / 0.926 / 0.071 | 21.60 / 0.833 / 0.159 |
| GUAVA-FB [SMPL-X GT] | 19.99 / 0.796 / 0.184 | 27.18 / 0.929 / 0.062 | 22.10 / 0.835 / 0.148 |
| GUAVA-FB [SPRINT] | 17.60 / 0.753 / 0.229 | 22.64 / 0.875 / 0.107 | 19.08 / 0.789 / 0.193 |

On subset24, ExAvatar scores UBody 19.66 and BEDLAM2 28.33 dB (GART: 20.09 / 28.31).
- ExAvatar fails on one BEDLAM2 clip (`bedlam2/20240731_1_1827_hdri__seq_000169`): its DECA face fit finds no face in any test frame, so the official pipeline stops. Its subset columns cover 23 of the 24 clips.
- A like-for-like subset23 table is being scored.
- Tables: `O/eval/avatar_test_v1/table_{full,subset24}.md`.

## Issues found
- **Empty-matte NaN.** If StyleMatte returns an empty matte, the refinement's mask term becomes NaN and the fit diverges.
  - This happened on 1,137 of 24,000 label rows, all small people (crop side about 260 px).
  - It never happened on ValBench: 0 black frames.
  - Proposed fix: skip the mask terms when the matte is empty. Not applied yet; it needs its own check.
- **Shared host.** Another user's jobs kept CPU load at about 500, so every CPU-bound stage ran 3-5x slower. My jobs now run with bounded thread counts.

## Pending
1. H4 verdict, then this file and PREREG / RUNLOG are updated.
2. The single AvatarTest-v1 test run, after H4 decides (user choice).
   - Registered: k = 60 + H1c.
   - If H4 passes, the distilled k = 0 pipeline is added by amendment, before any test run.
3. The empty-matte fix, if wanted.
4. The PublicBench-v1 People-Snapshot zip password.
