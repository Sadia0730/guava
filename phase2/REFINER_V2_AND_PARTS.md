# Refiner v2 and part-wise queries: prepared and smoke-tested (2026-10-13)

Both were smoke-tested on the **L-data** student, because L-real is still training.
- **Refiner v2** starts automatically after L-real, as approved.
- **The part-wise fine-tuning runs are NOT launched.**

## Refiner v2 (no-render, mixed data)

**Recipe.** A4 (ii) unchanged, plus a fixed share of real images per batch:
- **Batch:** 51 BEDLAM2 rows + 13 real images (COCO train2017 + MPII: the L-real crops, with every exclusion).
- **BEDLAM2 rows:** GT 3D losses as before; start from the student's prediction or from GT + a real student error, 50/50.
- **Real rows:** always start from the student's prediction (they have no GT). They get no GT losses, only the confidence-weighted 2D keypoint loss on the *refined* output (hip weight 0).
- **Validation:** adds the 2D loss on 1,024 real val rows for the student and the refined output, to watch 2D alignment directly.
- **Code:**
  - PEAR `train_render_refiner.py`: optional `data.real2d` section; without it the A4 path is unchanged.
  - Config: `configs/phase1_task3/refiner_v2_Lreal.yaml`.
  - Seed 20261012; 20,000 steps.

**Smoke test** (L-data student and its error bank; 200 steps; 4,000-row subsets; shared GPU 0; measured):

| | Step 0 | Step 200 |
|---|---|---|
| BEDLAM2-val MPJPE, 22 joints (2,048 rows) | 70.44 (refined = student) | **60.47** (81% of rows improved) |
| Real-image 2D val loss (1,024 COCO/MPII rows) | 0.04100 (refined = student) | 0.04113 (unchanged) |

- Batch 51 + 13 rows.
- Peak memory 1.9 GB.
- Throughput 124–150 img/s on the shared GPU. A4 (ii) reached about 520 img/s on a quieter GPU; the extra cost here is the SMPL-X forward for the 2D loss.

**After L-real** (chain `screen/after_lreal.sh`, waiting on GPU 2):
1. Evaluate L-real: Table-3 rows (author's protocol), EHF fp32, 3DPW test standard.
2. Build the error bank from L-real (20,000 BEDLAM2-val frames).
3. Train refiner v2. Estimated 1.5–3 h depending on GPU load.
4. Evaluate L-real + refiner v2: Table-3 rows and EHF.

**EHF evaluator.** `tools/eval_ehf_standard.py` gains an optional `--refiner` (fp32), with the same code
as in `tools/eval_author_protocol.py`.
- **Unchanged path:** L-data `best.pt` re-evaluated without the refiner matches the stored EHF results per frame (max difference 0.0 over 100 frames).
- **First use, L-data + A4 refiner (measured):**

| EHF metric | L-data | L-data + refiner |
|---|---|---|
| PA-MPJPE body | 60.80 | **51.26** |
| PA-PVE all | 51.28 | **46.87** |
| PVE all | 83.88 | 85.53 |
| PA-PVE hands | 15.09 | 15.13 |
| PVE face | 38.84 | 24.60 |

## Part-wise queries (5-query decoder)

**Design.**
- **Queries:** five in PEAR's HMR2-style cross-attention decoder: body, left hand, right hand, face, camera.
- **Output heads:** each reads its own query.

  | Query | Decodes |
  |---|---|
  | Body | global + body pose, shape |
  | Each hand | that hand's pose; hand scale = mean of the two hand queries |
  | Face | FLAME parameters, SMPL-X expression, head scale |
  | Camera | `pd_cam` |
- **Weights:** the decoder and all output layers are the trained single-query weights. Each query starts as a copy of the trained query (the position embedding is tiled).
- **Why init is exact:** self-attention over five identical tokens equals attention to one token, so at initialisation the model reproduces the single-query student.
- **Code:**
  - `models/smplx/smplx_head_multiquery.py`
  - `models/pipeline/student_pipeline_vitpose_mq.py`
  - `train_student_gt.py` options `student_head: multiquery` and `hand_crop_tokens`
- **Optional hand-crop variant.**
  - A first pass gives the hand keypoints (SMPL-X → DWPose, projected with the predicted camera; no gradient through the boxes).
  - A 64×64 crop around each hand of the 256 px input becomes 16 tokens through a small conv stem.
  - The hand query reads them through one gated cross-attention + feed-forward block. The gate is tanh(g) with g = 0 at init, so the output is again exact at init.
  - The hand poses are then decoded again.
  - **Limitation:** the crops come from the 256 px network input, not the full-resolution frame. They give finer tokens, not new pixels.

**Checks (measured; `tools/check_multiquery_head.py`; L-data `best.pt`; 64 real images; L40S shared with another user's job).**

| | Single query (L-data) | 5 queries | 5 queries + hand crops |
|---|---|---|---|
| Max output difference vs single query at init | — | 3.8e-6 (float rounding) | 3.8e-6 |
| Parameters, total / head | 71,523,276 / 40,513,612 | +4,096 (the 4 new query embeddings) | +5,036,674 (stem, projection, cross-attention, feed-forward, gates) |
| Latency, batch 1, fp32: median (p10) | 19.9 (10.3) ms | 20.6 (11.2) ms | 25.6 (20.8) ms |
| Latency, batch 1, bf16: median (p10) | 14.5 (11.6) ms | 21.7 (12.2) ms | 26.6 (22.5) ms |

- The GPU was shared, so the medians are noisy.
- The p10 values suggest five queries cost about +0.6–0.9 ms on this GPU, and hand crops about +10 ms. Most of the hand-crop cost is the extra SMPL-X forward that locates the hands.
- Not measured on the laptop.

**Training smoke tests** (200 steps from L-data `best.pt`; L-real data mix; lr 1e-4; GPU 0; measured):
- Both variants exit 0. They load the single-query checkpoint with no missing or unexpected keys (hand-crop modules apart), and give the identical step-1 loss (2.789).
- **Peak memory:** 8.73 GB (5 queries) and 8.94 GB (+ hand crops).
- **Throughput:** 61–69 and 53–67 img/s on the shared GPU.
- **After 200 steps:** the queries have begun to separate (max |q_i − q_body| 0.003), and the hand-crop gate has moved off 0 (−6e-5 / +2e-4).

**Fine-tuning plan (not launched).**
- **Setup:** three runs from **L-real `best.pt`**, with the identical recipe:
  - L-real data mix and losses;
  - 32,000 steps; lr 1e-4 with warmup 1,000; backbone ×0.1;
  - seed 20261009.
- **Configs** (`configs/phase1_task1/`):
  - `finetune_single_Lreal.yaml`: **control**, single query.
  - `finetune_parts_Lreal.yaml`: 5 queries.
  - `finetune_parts_handcrop_Lreal.yaml`: optional, 5 queries + hand crops.
- **Why a control:** the pre-registered comparison is against the single-query model fine-tuned the same way, so extra training alone can't produce the gain.
- **Estimates:** about 6 h per run on a free GPU (from L-data's rate) or 8–9 h on a shared one; 8.7–8.9 GB.
- **Before evaluating:** the evaluators need a loader for the 5-query student (loader only, no metric change). It will be added when the runs are approved, so the L-real table rows keep the same evaluator hash as the other rows.

**Pre-registered target** (as given): 5-query vs the 1-query control:
- EHF hand PA-PVE better by ≥ 1.5 mm;
- UBody hand PA-PVE better by ≥ 1.5 mm;
- 3DPW not worse by > 0.5 mm.

Open points:
- **UBody** is still blocked on SMPL-X v1.1 (`phase2/EVAL_SUITE.md`). The alternative is the provisional locked-head stand-in, used for both models.
- **3DPW metric:** it's not stated which one. Proposed: test PA-MPJPE **and** MPJPE, standard evaluator, GT-keypoint crop.
