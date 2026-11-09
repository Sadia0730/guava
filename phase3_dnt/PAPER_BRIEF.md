# Paper brief: single-image full-body Gaussian avatar (working name GUAVA-FB), for drafting a CVPR submission

Written 2026-11-09 from the project records (`RESULTS_FINAL.md`, `PREREG.md`, `RUNLOG.md`, `GAP_REPORT.md`, `RESULTS_FUSION.md`).
Every number below is copied from those files. Nothing here is a new result.

---

## 0. Instructions for the writing assistant (read first)

You are helping the authors draft a CVPR paper (8 pages + references, anonymous, CVPR LaTeX template). Rules:

1. **Use only the facts in this file.** Do not invent numbers, datasets, baselines, ablations, citations, venues or years.
   If the text needs something that is not here, write `[TODO: ...]`.
2. **Citations:** write `[CITE: name]`; the authors add the BibTeX. Venues are given only where this file states them.
3. **Anonymity:** no author names, lab, institution, repository links, GPU-cluster names or internal codenames. Internal
   labels that must NOT appear in the paper: B28k, recipe B, pilot A/B/C/D, a20, v1b, H1, H1c, H2, H4, phase 6/7, RAGA,
   ValBench, AvatarTest-v1, PublicBench. Use the plain descriptions given here instead.
4. **Claims:** follow section 10 ("may claim" / "must not claim") exactly. Section 11 (limitations / disclosures) must be
   reflected in the paper.
5. **Precision:** PSNR 2 decimals, SSIM 3, LPIPS 4 (3 is fine in tables if applied to every row). Bold the best
   single-image result per column; never bold a per-person method in the single-image block.
6. **Credit:** the architecture is GUAVA's. Say so plainly in the method section; the paper's contribution is the full-body
   extension, the evaluation and the analysis, not the backbone.
7. **Tone:** precise, no hype words ("revolutionary", "unprecedented", "photorealistic" without evidence).

---

## 1. One-paragraph summary

From a single photo of an unseen person, the method builds an animatable full-body 3D Gaussian avatar in a single forward
pass (about 55 ms on an idle NVIDIA L40S), which is then driven by SMPL-X poses and rendered at 512 x 512 (about 12 ms per
frame, about 207k Gaussians). It extends GUAVA [CITE: GUAVA, ICCV 2025], an upper-body method, to the full body: an SMPL-X 2020
body model with its own head, a per-sample perspective camera instead of GUAVA's fixed field of view, a full-body training
mixture of synthetic and real video, and added clothing, edge, silhouette and identity objectives. Under one
pre-registered protocol (same clips, source photo, ground-truth driving poses, crop, mattes and metrics for every method; no
test-time fitting), it beats the single-image methods LHM, LHM++ and IDOL by 2.75-4.12 dB PSNR, and released GUAVA by
0.47 dB. It comes within 0.57 dB of GART overall and matches GaussianAvatar (21.18 vs 21.02 dB), both of which train on each
person's video for minutes to hours. It remains 3.46 dB behind GART on People-Snapshot.

---

## 2. Names to use in the paper

| Internal name | Use in the paper |
|---|---|
| GUAVA-FB | the method's name is the authors' choice; "Ours" in tables |
| GUAVA-FB [SMPL-X GT] | Ours (ground-truth pose) |
| GUAVA-FB [SPRINT, k = 0] | Ours (estimated pose, fully automatic) |
| SPRINT | "a lightweight real-time SMPL-X regressor (ours)"; name optional |
| FinalBench | "our evaluation protocol" / "the benchmark" (authors may name it) |

---

## 3. Method facts

### 3.1 Background: GUAVA (inherited, not our contribution)
- GUAVA = "Generalizable Upper Body 3D Gaussian Avatar" [CITE: GUAVA, ICCV 2025]. One image plus a tracked pose gives a
  feed-forward avatar.
- Its components are kept unchanged in our method:
  - DINOv2 image features at 518 px.
  - A UV feature map at 512 on the body model's UV layout.
  - UV Gaussians: one on every textured UV texel.
  - About 1.4k vertex Gaussians.
  - 32-channel Gaussian colour features.
  - Rendering at 512, followed by an image-space StyleUNet refiner.
  - Linear blend skinning to the driving pose.
- Source features reach the UV map only through texels visible in the source photo (inverse texture mapping). Unseen
  texels come from learned base and vertex features.

### 3.2 Our full-body extension (the method contribution)
1. **Body model.** SMPL-X 2020 with its own head: 200 shape and 50 expression coefficients. GUAVA uses a FLAME-based head;
   ours has no FLAME.
2. **Per-sample perspective camera.**
   - GUAVA renders with one fixed field of view (inverse tangent of the half-FoV = 24).
   - We use a perspective person crop with a per-sample value of 2f / L (f = focal length, L = crop side).
   - Motivation, measured: with GUAVA's fixed camera, exact BEDLAM2 cameras leave a median reprojection error of 7 px at
     512.
   - With 2f / L = 24 the change is bit-exact to GUAVA.
3. **Full-body training data.** Final sampling shares:

   | Dataset | Share | What is used |
   |---|---|---|
   | BEDLAM2 | 26.47 % | 1,500 synthetic clips with exact SMPL-X and cameras |
   | UBody | 25 % | 203 upper-body clips |
   | Motion-X | 17.65 % | 310 clips |
   | 4D-DRESS | 17.65 % | 28 subjects |
   | UBC-Fashion | 13.24 % | 110 training videos |

   - **Never in training:** People-Snapshot, NeuMan, X-Humans, UBC-Fashion test videos, held-out 4D-DRESS subjects, 3DPW.
   - **Face check:** training videos were screened with an ArcFace face-identity check against evaluation identities
     (cosine > 0.5 = match = removed).
4. **Objectives.**
   - GUAVA's losses: L1 1.0, LPIPS 0.05, head crop 0.25 -> 0.5, hand crop 0.1, position and scale regularisers.
   - Added: ArcFace identity loss 0.05 (frontal faces >= 48 px).
   - Added: clothing-box L1 + LPIPS 0.5.
   - Added: Sobel-edge L1 0.5 on self-reconstruction targets.
   - Added: silhouette L1 0.5.
   - Target sampling: self-reconstruction targets (p 0.75), same-moment other-camera targets (p 0.5), true 2x zoom-in
     targets (p 0.33).
   - Batch: 2 sources x 3 targets.
5. **Training.**
   - Initialised from GUAVA's released weights, first re-fine-tuned to the SMPL-X 2020 head on 203 UBody clips for 20k
     iterations.
   - Then full-body training for 30k iterations.
   - Adam (betas 0, 0.99), lr 5e-5 decaying linearly to 0.1x, seed 0.
   - One L40S, 1.11 it/s, peak 17.0 GB.
   - The checkpoint was chosen by a rule fixed before training: UBody SSIM and UBody face LPIPS no worse than the
     upper-body model by more than 0.002, then the highest full-body validation SSIM. This selected iteration 28k, the
     first eligible checkpoint.
6. **Inference cost.**
   - Idle L40S: avatar build 54.5 ms, render 12.24 ms per frame, about 207k Gaussians, peak 4.0 GB.
   - Inside the shared-GPU benchmark runs: 99 ms build and 27.5 ms per frame.

### 3.3 Pose for the fully automatic setting (supporting component)
- A real-time SMPL-X regressor:
  - ViTPose+-S backbone.
  - PEAR-style decoder trained from scratch: no PEAR weights or outputs.
  - 128k steps, EMA.
  - Data: BEDLAM2, COCO + MPII (NeuralAnnot SMPL-X), UBody, Motion-X, AGORA, ARCTIC, MOYO.
  - No 3DPW training data.
- Forward pass 7 ms.
- 3DPW-test without extra information: MPJPE 83.07 mm / PA-MPJPE 52.08 mm. PEAR [CITE] gets 46.73 PA-MPJPE in the same
  setting, so ours is **weaker than PEAR**; present it only as the pose source of the automatic row.
- "Estimated pose" row: the avatar is built from the photo with this regressor's fit, then driven by the regressor's
  per-frame estimate. No per-frame optimisation (k = 0) and no ground truth anywhere.

### 3.4 Live system and streaming appearance update (secondary; keep short or move to the supplement)
- **Pipeline:** webcam (MJPG) -> fixed centre crop -> pose regressor -> avatar render. The two stages run asynchronously.
  The avatar is built once from a reference frame.
- **Streaming appearance update:**
  - Accepted frames are encoded and inverse-texture-mapped into the UV map.
  - Each texel is a weighted running mean with weight = visibility x facing x fit quality x sharpness. The source photo
    is the anchor, with prior weight 2.
  - Outliers are rejected.
  - The decoder is re-run on the fused map in a background worker.
- **Measured:**
  - Live server test: PSNR 20.85 -> 21.10 dB (+0.25; +0.26 on turned-away frames).
  - Cost: loop p50 50.1 -> 73.0 ms (about 14 fps) with the worker on.
  - Offline fusion of all training frames with GT poses, on a validation set: 20.98 -> 21.59 dB (BEDLAM2 26.13 -> 27.00,
    where GART gets 26.71).
  - Test, descriptive only: People-Snapshot 27.83 -> 27.66 (worse), NeuMan 23.42 -> 24.13.
- **Speed:**
  - Laptop RTX 3080: 22.3 fps (43 ms p50) with centre crop + async.
  - Server L40S best configuration: 32.30 / 33.43 ms median / p95, with 15.5 % of frames over 33 ms.
  - **30 fps end to end has not been demonstrated.**

---

## 4. Evaluation protocol (registered 2026-11-02, before the final runs)

### 4.1 Rules
- Same clips and test frames for every method.
- Single-image methods get only the clip's **first frame** as source.
- Per-person methods train on the clip's training frames only:
  - UBody and BEDLAM2: the first half of each clip trains, the second half is tested.
  - People-Snapshot, NeuMan and X-Humans: official splits.
- Every method is driven by the **ground-truth body model** of each test frame, converted to that method's body model.
  There is no test-time pose, face or expression fitting in any row.
- One scorer for every row:
  - 512 person crop at 1.1x the projected GT-mesh box.
  - GT = image x StyleMatte matte on black.
  - Metrics: PSNR, SSIM, LPIPS-VGG and L1 (GUAVA's metric code).
  - Per clip = mean over its test frames; per dataset and overall = **mean over clips**.
  - 95 % percentile bootstrap over clips: 10,000 resamples, stratified by dataset.
  - Paired differences per clip.
- This is stricter than the published People-Snapshot protocols of GART, GaussianAvatar and ExAvatar, which fit test poses.
  **Our numbers are therefore not comparable with their published tables.**

### 4.2 Test data (answers "is this a small subset?")

| Dataset | Clips | Test frames | Notes |
|---|---|---|---|
| UBody | 30 | 3,540 | Official intra-scene test list; identities in none of our training clips; upper body only; pseudo-GT SMPL-X from UBody |
| People-Snapshot | 4 | 190 | InstantAvatar split; GT is SMPL, so SMPL-X 2020 is fitted to it |
| NeuMan | 4 | 27 | Official split (bike, citron, jogging, seattle); 3-10 test frames each |
| X-Humans | 3 | 1,108 | Subjects 00028, 00034, 00087; renders of textured scans; mask = rendered depth > 0 |
| **Real total** | **41** | **4,865** | 0 clips excluded |
| BEDLAM2 (synthetic, separate table) | 40 | 1,470 | Exact GT; identity- and outfit-disjoint from our training data |

- People-Snapshot and NeuMan are the full standard test sets used by per-person avatar papers, but they are small.
- X-Humans has 3 subjects, so its bootstrap intervals are wide and unreliable.
- UBody is 73 % of the real clips, so the overall mean is dominated by upper-body clips.

### 4.3 Baselines (one line each; all under the same contract)
**Single image (feed-forward):**
- **GUAVA (released)** [CITE]: upper-body model.
- **LHM** (LHM-1B-HF) [CITE]: shape = GT projected to its 10 betas; the face-crop bug fix was registered before any score was
  read.
- **LHM++** (LHMPP-700M) [CITE].
- **IDOL** [CITE, CVPR 2025]: trained on HuGe100K synthetic full-body humans; SMPL-X v1.1 with 10 shape and 10 expression
  coefficients fitted to the GT.

**Per-person (reference only):**
- **GART** [CITE], **GaussianAvatar** [CITE]: official code, trained on the training frames, driven by SMPL fitted to the GT
  SMPL-X mesh.
- **ExAvatar** [CITE], conforming variant: the test-frame face is the GT's converted jaw and expression; no face fitting on test
  frames.

---

## 5. Results (copy into the paper; cells PSNR ↑ / SSIM ↑ / LPIPS ↓)

### 5.1 Main table: the 41 real clips

| Method | UBody (30) | People-Snapshot (4) | NeuMan (4) | X-Humans (3) | All (41) |
|---|---|---|---|---|---|
| *Single image* | | | | | |
| GUAVA (released) | 19.48 / 0.788 / 0.1912 | 25.80 / 0.921 / 0.0780 | 23.23 / 0.896 / 0.0788 | 22.75 / 0.906 / 0.0907 | 20.70 / 0.820 / 0.1618 |
| LHM | 16.12 / 0.692 / 0.2631 | 25.61 / 0.920 / 0.0781 | 21.71 / 0.883 / 0.0887 | 24.11 / 0.913 / 0.0911 | 18.18 / 0.749 / 0.2155 |
| LHM++ | 16.72 / 0.724 / 0.2403 | 24.66 / 0.916 / 0.0847 | 21.66 / 0.886 / 0.0868 | 22.93 / 0.913 / 0.0832 | 18.43 / 0.772 / 0.1986 |
| IDOL | 14.74 / 0.636 / 0.3082 | 24.77 / 0.915 / 0.0798 | 21.73 / 0.881 / 0.0939 | 23.76 / 0.915 / 0.0838 | 17.06 / 0.708 / 0.2486 |
| Ours (estimated pose) | 17.22 / 0.744 / 0.2357 | 24.47 / 0.908 / 0.0831 | 22.37 / 0.886 / 0.0840 | 21.31 / 0.890 / 0.1056 | 18.73 / 0.785 / 0.1965 |
| **Ours (GT pose)** | 19.62 / 0.786 / 0.1909 | 27.83 / 0.933 / 0.0646 | 23.42 / 0.898 / 0.0738 | 24.90 / 0.922 / 0.0749 | 21.18 / 0.822 / 0.1587 |
| *Per-person, trained on the person's video (reference)* | | | | | |
| GART | 19.57 / 0.797 / 0.1856 | 31.29 / 0.959 / 0.0357 | 25.02 / 0.913 / 0.0723 | 26.43 / 0.935 / 0.0714 | 21.75 / 0.834 / 0.1516 |
| GaussianAvatar | 19.02 / 0.785 / 0.2036 | 30.37 / 0.951 / 0.0543 | 23.61 / 0.901 / 0.0793 | 25.05 / 0.923 / 0.0715 | 21.02 / 0.823 / 0.1672 |
| ExAvatar | 19.47 / 0.797 / 0.1993 | 30.67 / 0.955 / 0.0409 | 24.39 / 0.908 / 0.0743 | 23.18 / 0.911 / 0.0875 | 21.32 / 0.831 / 0.1635 |

Overall 95 % CIs (PSNR): Ours (GT pose) [19.89, 22.47]; GUAVA [19.49, 21.91]; GART [20.50, 23.06]. The intervals of
absolute means overlap because clip difficulty varies a lot; **use the paired differences (5.3) for comparisons.**
UBody SSIM: GUAVA 0.788 is above ours (0.786); bold accordingly.

### 5.2 BEDLAM2 (40 synthetic full-body clips)

| Method | BEDLAM2 |
|---|---|
| GUAVA (released) | 24.80 / 0.905 / 0.0810 |
| LHM | 24.29 / 0.894 / 0.0922 |
| LHM++ | 24.50 / 0.901 / 0.0869 |
| IDOL | 23.83 / 0.889 / 0.0948 |
| Ours (estimated pose) | 22.67 / 0.874 / 0.1079 |
| **Ours (GT pose)** | **27.19 / 0.928 / 0.0630** |
| GART (per-person) | 28.17 / 0.934 / 0.0612 |
| GaussianAvatar (per-person) | 26.79 / 0.925 / 0.0727 |
| ExAvatar | not run |

Disclosure: 38 of the 40 BEDLAM2 subjects appear in other BEDLAM2 sequences that the pose regressor trained on. This affects
only the estimated-pose row.

### 5.3 Paired PSNR differences, per clip (mean [95 % CI]; a CI excluding 0 = significant at 5 %, uncorrected)

**Baseline minus Ours (GT pose):**

| Baseline | UBody | People-Snapshot | NeuMan | X-Humans | All 41 | BEDLAM2 |
|---|---|---|---|---|---|---|
| LHM | -3.50 [-4.86, -2.36] | -2.23 [-2.56, -2.03] | -1.71 [-2.54, -0.96] | -0.79 [-1.61, +0.23] | -3.00 [-4.00, -2.16] | -2.91 [-3.41, -2.44] |
| LHM++ | -2.90 [-4.22, -1.81] | -3.18 [-3.72, -2.80] | -1.76 [-2.55, -1.02] | -1.96 [-2.82, -0.35] | -2.75 [-3.73, -1.93] | -2.70 [-3.20, -2.17] |
| IDOL | -4.88 [-6.23, -3.68] | -3.06 [-3.47, -2.64] | -1.69 [-2.32, -1.06] | -1.14 [-2.08, -0.62] | -4.12 [-5.10, -3.24] | -3.37 [-3.87, -2.88] |

**Ours (GT pose) minus released GUAVA:**

| | UBody | People-Snapshot | NeuMan | X-Humans | All 41 | BEDLAM2 |
|---|---|---|---|---|---|---|
| ΔPSNR | +0.14 [-0.23, +0.57] | +2.03 [+1.59, +2.35] | +0.19 [-0.83, +0.88] | +2.15 [+1.53, +2.79] | +0.47 [+0.19, +0.81] | +2.39 [+1.99, +2.84] |

Read-out:
- The full-body extension pays off where the body is fully visible (People-Snapshot, X-Humans, BEDLAM2: +2.0 to +2.4 dB).
- On upper-body UBody it matches GUAVA. That is the intended "no regression" result, since GUAVA was built for upper bodies
  and may have trained on UBody.
- NeuMan is not significant (4 clips with 3-10 frames each).

**Per-person methods minus released GUAVA (all 41 clips):** GART +1.04, ExAvatar +0.61, GaussianAvatar +0.31; Ours +0.47.
No paired test between Ours and the per-person methods was run.

### 5.4 Cost (median per clip, measured inside the benchmark runs on shared L40S GPUs; indicative only)

| Method | Avatar creation per person | Render per frame |
|---|---|---|
| Ours | 99 ms (54.5 ms on an idle GPU) | 27.5 ms (12.24 ms idle) |
| GUAVA (released) | 143 ms | 29.5 ms |
| LHM / LHM++ / IDOL | 7.7 s / 2.15 s / 1.66 s | 66.5 / 158.3 / 80.6 ms |
| GART / GaussianAvatar / ExAvatar | 2.0 min / 1.2 h / 1.8 h of training | 8.8 / 258.8 / 96.9 ms |

### 5.5 L1 (all 41 clips; lower is better)
Ours (GT pose) 0.0428, GUAVA 0.0425, LHM 0.0658, LHM++ 0.0586, IDOL 0.0857, GART 0.0392, GaussianAvatar 0.0423, ExAvatar
0.0408. On L1, ours is not better than GUAVA overall, because of UBody (0.0533 vs 0.0519). Put L1 in the supplement.

---

## 6. Analysis that can go in the paper

### 6.1 Error vs view change (People-Snapshot, frames; PSNR ours / GART)

| View change from the photo | [0,30°) | [30,60°) | [60,90°) | [90,135°) | [135,180°] |
|---|---|---|---|---|---|
| Ours / GART | 29.64 / 30.70 (75) | 27.01 / 30.96 (20) | 27.19 / 33.11 (23) | 26.72 / 32.82 (31) | 26.68 / 31.02 (41) |

The gap is about 1 dB near the photo's viewpoint and 4-6 dB once the person has turned. 115 of the 190 People-Snapshot test
frames are turned views. Per-person methods see the whole rotation during training; a single photo does not.

### 6.2 Where the remaining gap comes from (diagnostics; descriptive, not causal proofs)
- **Unseen sides.**
  - Feeding the clip's training frames into the UV map with GT poses (no optimisation) recovers most of the far-view gap on
    exact-pose synthetic data: validation BEDLAM2 26.13 -> 27.00, GART 26.71.
  - On real video it recovers only part: NeuMan 23.04 -> 24.14 (GART 24.96).
- **Coverage misses.**
  - Hair, shoes, hands and loose clothing outside the body model make up 2.6 % of People-Snapshot pixels but 43 % of our
    squared error.
  - 3.27 % of the person's pixels are left black (GART 0.61 %).
  - Oracle: filling only the missed pixels with GT lifts ours from 27.95 to 29.14 dB (+1.19); GART gains only +0.34.
- **The decoder was trained on single-view UV maps.** Fused maps are out of distribution: the People-Snapshot near view
  drops from 29.64 to 28.12 when fused.

### 6.3 Qualitative observations
Measured facts only; describe the figures from the images themselves:
- On UBody, LHM++ completes body parts hidden by desks, podiums or the image border. Its extra silhouette is 33 % of the GT
  area, against 13 % for GUAVA and ours, and the masked GT counts this as error. On full-body sets its silhouettes match
  GUAVA's (IoU 0.85-0.91).
- IDOL is trained on full-body synthetic humans only, and its appearance degrades on upper-body sources (UBody PSNR 14.74).
- Ours leaves 3.27 % of the person's pixels black on People-Snapshot, mostly hair, shoes, hands and loose clothing (6.2).

---

## 7. Ablations: what exists and what is missing

**Exists (validation sets only; not the test benchmark):**

| Variant (vs the base full-body recipe) | Result on validation |
|---|---|
| + clothing / edge / silhouette losses (the adopted recipe) | Clothing-box LPIPS on 4D-DRESS 0.3808 vs 0.3839 at 8k iterations; passed all five registered criteria |
| + normal supervision (VCR-GauS normals) | Geometry better (normal cosine 0.914 vs 0.820), clothing LPIPS worse (0.3814); not adopted |
| + scan geometry + GAN / Gram losses | Clothing LPIPS worse (0.3862); not adopted |
| Larger offset range (3 -> 6) | No effect on off-body regions (14.06 vs 14.10 dB) |
| Fusion-aware fine-tuning (two rounds) | Failed its registered rule; single-image quality dropped 0.29 dB in round 2 |
| Final model (28k) vs adopted recipe at 8k | Validation PSNR 4D-DRESS 23.77 vs 23.74, BEDLAM2 26.07 vs 25.64, Motion-X 24.97 vs 24.73, UBody 23.43 vs 22.87 |

**Missing (a CVPR reviewer will ask):** an ablation of each full-body component on the test benchmark:
- per-sample camera vs fixed field of view;
- each training dataset;
- each added loss;
- the upper-body re-fine-tune.

Only released GUAVA vs ours exists on the test benchmark. Write the ablation section as `[TODO: benchmark ablation]` unless
the authors run it.

---

## 8. Qualitative material (already rendered; examples chosen by a rule fixed before any figure was made)

All files are under `/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/reports/qualitative_finalbench/`:
- `fig_single_image.png`
  - Columns: source photo, ground truth, GUAVA, LHM, LHM++, IDOL, Ours (estimated pose), Ours (GT pose).
  - Rows: per dataset, the clip with the median PSNR of ours, at the test frame nearest to and farthest from the photo's
    viewpoint.
  - Each tile shows that frame's PSNR.
- `fig_per_person.png`: the same frames, with GART, GaussianAvatar and ExAvatar next to ours.
- `fig_hard_cases.png`: per dataset, the clip where ours has the smallest margin over the best single-image baseline (the
  failure figure).
- `tiles/<figure>/<row>/NN_<method>.png`: individual crops for laying out LaTeX figures.
- `selection.json`: which clip and frame was picked, and why.
- `videos/<dataset>_<clip>_{single,perperson}.mp4`: side-by-side videos of all test frames of the five typical clips, for the
  supplementary video.

Suggested captions:
- Single-image figure: "Novel-pose renders from a single photo. Columns show the input, the ground truth, the single-image
  baselines and ours, all driven by the same ground-truth SMPL-X pose. Rows: per dataset, the clip with the median PSNR of
  ours, at the test frame with the smallest and the largest change of body orientation from the photo. Examples are chosen
  by this rule, not by hand."
- Failure figure: "Hardest clips for ours..."

---

## 9. Suggested paper structure (8 pages)

1. **Introduction** (1 p).
   - Problem: an animatable full-body avatar from one photo, fast enough for interactive use.
   - Gap: upper-body generalizable avatars, slow or less faithful full-body single-image methods, and per-person methods
     that need a video and minutes to hours of training.
   - Contributions: see section 10.
2. **Related work** (0.75 p).
   - Per-person avatars from video (GART, GaussianAvatar, ExAvatar, InstantAvatar).
   - Single-image / feed-forward avatars (GUAVA, LHM, LHM++, IDOL).
   - Human pose / shape regression (PEAR, ViTPose).
   - Streaming / feed-forward fusion (GenFusion, arXiv 2603.28997, novel-view only).
   - Verify every reference; add recent 2025-26 work the authors know.
3. **Method** (2 p).
   - GUAVA preliminaries (credit).
   - Full-body extension: body model, camera, data, objectives, training and selection.
   - Optional short subsection on the automatic pose and the live update.
4. **Experiments** (3.5 p).
   - Protocol and why it is fair (4.1).
   - Data (4.2).
   - Baselines (4.3).
   - Main table (5.1) with the paired differences (5.3).
   - BEDLAM2 (5.2).
   - Cost (5.4).
   - View-change analysis (6.1).
   - Qualitative figures (8).
   - Ablations (7, `[TODO]`).
5. **Limitations** (0.5 p): section 11.
6. **Conclusion** (0.25 p).

**Supplement:** L1 tables, per-dataset CIs, baseline adapter details, the live system and its timings, the streaming update
results, the failure figure, the videos.

---

## 10. Claims

**May claim (supported by the numbers above):**
- Ours beats every single-image baseline under the same protocol.
  - Against LHM, LHM++ and IDOL the paired CI is below 0 overall, on BEDLAM2, and on every dataset, with one exception: LHM on
    X-Humans, where the CI includes 0.
  - Against released GUAVA: +0.47 dB overall; significant on People-Snapshot, X-Humans and BEDLAM2; on par on UBody and NeuMan.
- Building an avatar takes about 0.1 s and rendering runs at real-time rates (12 ms per frame on an idle L40S). Per-person
  methods need 2 min (GART) to about 2 h (GaussianAvatar, ExAvatar) of training per person.
- Overall, ours comes within 0.57 dB of GART and is on par with GaussianAvatar and ExAvatar (21.18 vs 21.02 / 21.32). It
  needs one photo instead of a video. Say "on par"; there is no significance test against them.
- Fully automatic (no GT pose at all), ours gets 18.73 dB overall. That is higher than LHM, LHM++ and IDOL even though they
  are given GT poses (18.18-18.43). It is lower on People-Snapshot and X-Humans, and untested for significance: state it with
  these caveats or not at all.
- The protocol is stricter than the published per-person protocols: no test-time pose fitting.

**Must not claim:**
- Better than per-person methods in general. GART is ahead overall and by 3.46 dB on People-Snapshot.
- 30 fps live end to end (not demonstrated; 22.3 fps on a laptop).
- That the streaming update generally improves quality (People-Snapshot got worse: 27.83 -> 27.66).
- That the pose regressor is state of the art (it is behind PEAR).
- Causal statements from the diagnostics in 6.2. Write "is associated with" / "an oracle that ... gains ...".
- Any ablation result on the test benchmark (none exists).
- "State of the art on People-Snapshot / NeuMan" in any form.

---

## 11. Limitations and disclosures (must appear in the paper or supplement)

**Data and ground truth:**
- UBody ground truth is UBody's pseudo-GT SMPL-X, and UBody clips show the upper body only.
- NeuMan test sets have 3-10 frames.
- X-Humans has 3 subjects, and its images are renders of textured scans.
- People-Snapshot GT is SMPL, so SMPL-X was fitted to it.

**Training overlap that cannot be ruled out:**
- Released GUAVA may have trained on UBody clips.
- LHM and LHM++ training data are not public, so overlap with the test identities cannot be checked.
- 38 of 40 BEDLAM2 test subjects appear in sequences the pose regressor trained on.

**Process:**
- Some of the reported numbers were seen before the final protocol was registered (2026-11-02).
- LHM, LHM++ and IDOL were added after registration and do not change the clip set.

**Method limitations:**
- Coverage misses on hair, shoes, hands and loose clothing.
- Hallucinated unseen sides: a 4-6 dB gap at turned views.
- The estimated pose costs 2.45 dB (18.73 vs 21.18).
- Streaming fusion is limited by the decoder being trained on single-view maps.

**Licences:** research and non-commercial use only. ArcFace and NeuralAnnot labels are research-only, and so are X-Humans,
BEDLAM2 and UBody.

---

## 12. Honest assessment against the CVPR bar (for the authors, not for the paper)

**Strengths:**
- A clear and significant margin over every single-image full-body baseline under a strict, pre-registered protocol.
- About 100 ms creation.
- On par with per-person methods overall, from one photo.
- A careful protocol with paired statistics.

**Weaknesses a reviewer will raise:**
1. **The architecture is GUAVA's.** The contribution looks like a body-model / camera / data / loss extension, and reviewers
   may call it incremental.
2. **No component ablations on the test benchmark.**
3. **Clearly behind per-person methods** on People-Snapshot (-3.46 dB vs GART) and X-Humans.
4. **Small public test sets** (People-Snapshot 4, NeuMan 4, X-Humans 3), and UBody pseudo-GT dominates the overall mean.
5. **The live system and streaming update are not clean wins:** 30 fps not shown, fusion mixed.

**Verdict:** as it stands, this is borderline to below the CVPR main-track bar.

**What would move it:**
- (a) Benchmark ablations of each full-body component (GPU time needed).
- (b) A fix for coverage misses at training time. The oracle says up to +1.2 dB on People-Snapshot near views.
- (c) A larger exact-pose real benchmark (subsets of MVHumanNet or DNA-Rendering; downloads > 20 GB need approval).
- (d) A demonstrated 30 fps live loop with a quality check.

A strong systems / benchmark framing ("one photo, 0.1 s, on par with per-person methods overall, under a fair protocol") is
the most defensible story.
