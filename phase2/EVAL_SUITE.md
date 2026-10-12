# Phase 2 — Evaluation suite matching PEAR's Table 3 (and EMDB, UBody)

Started 2026-10-12. Status: inventory and protocol research done. COCO-val and the HMR2 labels are
downloaded. LSP-Extended is downloading. Everything else waits for the decisions listed at the end.

**Rules.**
- Evaluators are additive; the existing 3DPW and EHF evaluators are not changed.
- Each new evaluator gets tests.
- **Gate:** the released PEAR checkpoint must reproduce PEAR's Table 3 within about 2% PCK or about
  2 mm before (L-0.1) and student 235000 are evaluated. If it doesn't, stop and report.
- Downloads over 50 GB need approval. Licensed data is never committed.

## Targets (PEAR paper, Table 3; Table 2 for hands)

| Benchmark | Metric | PEAR (paper) | Our evaluator today |
|---|---|---|---|
| COCO | PCK @0.05 / @0.1 | 0.81 / 0.94 | to build |
| LSP-Extended | PCK @0.05 / @0.1 | 0.55 / 0.81 | to build |
| PoseTrack | PCK @0.05 / @0.1 | 0.87 / 0.97 | to build (images needed) |
| 3DPW (test) | MPJPE / PA-MPJPE | 71.3 / 45.3 | ours: 73.56 / 45.65 (Task 1 gate, passed within ~2 mm) |
| AGORA | Body MVE | 59.3 | to build (data and protocol needed) |
| EHF (Table 2) | hand PA-PVE | 12.8 | ours: 11.0 (Task 1, pass with caveat) |
| UBody-intra (Table 2) | hand PA-PVE | 9.8 | to build |
| EMDB | — | not in the paper | new; no PEAR number to gate on |

## Protocols: confirmed vs assumed

**Source.** PEAR's author confirmed that they used HSMR's dataset preparation (github.com/IsshikiHugh/HSMR,
commit `88486dc1`; a copy is in `/raid/ubx858/datasets/eval_assets/reference_code/HSMR`). HSMR in turn
uses HMR2.0's (4D-Humans) packaged evaluation labels, `hmr2_evaluation_data.tar.gz`, plus the original
images. The author's own evaluation script for COCO, LSP-Extended, PoseTrack and 3DPW is expected
from you. When it arrives, it becomes the reference and every *assumed* item below is checked against
it.

### 2D PCK on COCO, LSP-Extended and PoseTrack

**Confirmed** (from the HSMR/HMR2 code; also the protocol PEAR's author says he followed):

| Item | Value |
|---|---|
| Label files | `coco_val.npz` (4,648 persons in 1,985 COCO val2017 images), `hr-lspet_train.npz` (9,428 persons), `posetrack_2018_val.npz` (12,700 persons) |
| GT keypoints | 44 = 25 OpenPose body + 19 "extra" (SPIN/HMR2 J19) |
| Keypoints present in GT | COCO: the 17 COCO keypoints in OpenPose order (no neck, no mid-hip); LSP-Extended: the 19 extra joints only; PoseTrack: body + extra |
| Distance normalisation | max side of the npz person box (`scale` × 200), i.e. PCK@t means error < t × box size. HMR2 computes this in crop-normalised units × (crop size / box max side) |
| Scoring | per-keypoint accuracy over GT confidence > 0.5, then averaged over keypoints (mmpose's `keypoint_pck_accuracy`) |

**Assumed** (to be checked against the author's script):

| Item | Assumption | Why it matters |
|---|---|---|
| Predicted 44 keypoints | EHM-s → SMPL topology through our fixed SMPLX→SMPL map, then HMR2's SMPL keypoint pipeline: OpenPose joints from SMPL joints + vertex picks, and the 19 extra joints from `SMPL_to_J19.pkl` | A model with SMPL-X output could instead use SMPL-X's own OpenPose joints. Differences are a few pixels, which matters at @0.05 |
| Input crop | square crop from the npz box (center, `scale` × 200), expanded with PEAR's `process_bbox` (×1.25) | HMR2 uses the box at aspect 192:256; PEAR may use its own detector or crop |
| 2D projection | PEAR's crop camera (focal 24 on the normalised crop), mapped back to image pixels | Standard for PEAR |
| Person selection | one crop per npz row; no detector | As in HMR2 |

### 3DPW (test)

Already implemented and gated in Task 1. Our evaluator (`tools/eval_3dpw_standard.py`) uses the
H36M regressor → 14 LSP joints, hip-midpoint centring and similarity Procrustes. HMR2/HSMR uses
`3dpw_test.npz` with their 44-keypoint set (indices 25–37 and 43), and no hip alignment for 3DPW.
**Assumed equivalent within about 1 mm**; to be checked against the author's script. Our PEAR result
(73.56 / 45.65) is 2.3 mm above the paper on MPJPE and 0.35 mm on PA-MPJPE.

### AGORA (Body MVE)

- **Assumed: the AGORA *test* set through the official evaluation server.** PEAR's baseline numbers
  in Table 3 (e.g. OSX 80.2) appear to match those methods' published test-set leaderboard numbers.
- Test ground truth is not public. Reproducing 59.3 needs predictions submitted to the AGORA server
  (account required), or a validation-set proxy (GT available after registration), which would not
  be comparable to 59.3.
- **Needs confirmation from PEAR's author.**

### UBody (OSX, "UBody-intra")

- **Confirmed from OSX's code** (`data/UBody/UBody.py`, IDEA-Research/OSX commit in
  `reference_code/OSX/COMMIT`):
  - intra-scene test list (`splits/intra_scene_test_list.npy`, on disk);
  - SMPL-X pseudo-GT from `annotations/<scene>/smplx_annotation.json`;
  - every `test_sample_interval`-th frame;
  - metrics PA-MPVPE all / hand / face, MPVPE, PA-MPJPE body / hand. These are the same
    Hand4Whole-style metrics as our EHF evaluator.
- **Data on disk:** `/raid/ubx858/datasets/raw/UBody` (videos extracted, 7.4 GB; `annotations.zip`
  not yet extracted).
- **Assumed:** the OSX config's `test_sample_interval`, and PEAR's crop. To check.

### EMDB

- **Official protocol** ("EMDB 1", the 17 test sequences): the authors' `evaluate.py`, giving MPJPE,
  PA-MPJPE, MVE and jitter, pelvis-aligned and Procrustes-aligned.
- Not in PEAR's paper, so there is no gate: report PEAR, (L-0.1) and student 235000 side by side.
  Published numbers of other methods are a sanity check only.

## Datasets: size, licence, who downloads

| Dataset (purpose) | What is needed | Size | Licence / terms | Download |
|---|---|---|---|---|
| HMR2 evaluation labels (all 2D benchmarks) | `hmr2_evaluation_data.tar.gz` | 78 MB (*measured*) | packaged labels from the original datasets (4D-Humans, MIT code) | **done**, from HSMR's Hugging Face mirror |
| COCO val2017 (eval) | images + annotations | 0.82 + 0.25 GB zips (*measured*), 1.6 GB extracted | annotations CC BY 4.0; images under Flickr terms | **done** |
| LSP-Extended / hr-lspet (eval) | `hr-lspet.zip` | 2.86 GB (*measured*) | research use (MPI-INF) | **in progress** (slow server) |
| PoseTrack 2018 val (eval) | val images (`images/val/*`) | not known: posetrack.net does not respond today | PoseTrack terms, non-commercial research | **you**: needs your PoseTrack account or a copy you have access to |
| AGORA (eval) | test or validation images + evaluation-server account | GB-scale images; exact size after login | AGORA licence (MPI-IS), non-commercial research | **you**: registration at agora.is.tue.mpg.de |
| EMDB (eval) | full dataset (images + SMPL GT) | not published; known after approval | EMDB terms (ETH); application form | **you**: application form at emdb.ait.ethz.ch |
| UBody (eval) | test videos + annotations | 14 GB raw on disk (*measured*) | UBody terms (IDEA-Research), research | **already on disk** |
| `SMPL_to_J19.pkl` (2D keypoints) | the J19 regressor | small | derived from SMPL; HSMR distributes it | from HSMR's Hugging Face; kept by path only, never committed |

### Training data (item 5: prepare only; NOT downloaded yet, sizes reported first as asked)

| Dataset | What | Size | Licence | Download | Exclusions |
|---|---|---|---|---|---|
| COCO train2017 | images + person keypoints (`annotations_trainval2017` already downloaded) | 19.3 GB zip (*measured*) | annotations CC BY 4.0, images Flickr | public | train2017 is disjoint from val2017 by construction; verify no image id of `coco_val.npz` appears |
| MPII Human Pose | images + annotations | 12.1 GB + 12 MB (*measured*) | BSD-2 (annotations); images from YouTube | public | **PoseTrack val is built from MPII videos**: exclude every MPII image whose source video belongs to a PoseTrack-val sequence (sequence names carry the MPII video id). Also exclude LSP / LSP-Extended images if any overlap is found |
| AI Challenger keypoints | train images + annotations | about 20 GB (*estimate*, from the published dataset description) | AI Challenger terms, research | the official host is offline; **you** need a mirror or a copy | no known overlap with our benchmarks; check by image hash against all eval images |

- **Crop build** (*estimate*): 512-px context crops as for BEDLAM2 come to about 45 KB per crop.
  That is about 150k (COCO) + 25k (MPII) + 370k (AI Challenger) persons, roughly 25 GB.

### Disk (`/raid`, shared with other users)

- **Now:** 170 GB free (*measured*, 13:47). It was 251 GB this morning; part of the drop is other
  users.
- **Our biggest items:** L-data 48 GB (growing; its periodic checkpoints add about 34 GB more before
  it finishes), (L-0.1) 53 GB, (L-0.05) 39 GB. Most of the last two is periodic `step_*.pt`
  checkpoints, about 0.86 GB each.
- **Proposed** (needs your approval; nothing deleted): delete the `step_*.pt` files of the finished
  (L-0.1) and (L-0.05) runs, keeping `best.pt` and `latest.pt`. That frees about 85 GB (*estimate*).
- **If everything above is downloaded:** about 32 GB of training zips, about 30 GB extracted, about
  25 GB of crops, and the evaluation images (sizes of PoseTrack, AGORA and EMDB unknown). Without the
  cleanup, `/raid` would go below 100 GB free.
- **Watcher threshold:** it stops everything below 50 GB.

## Gate results so far

### COCO-val PCK (released PEAR checkpoint, 4,648 persons, GT box crop)

**Gate NOT passed under the assumed protocol.** Per your rule, (L-0.1) and student 235000 are not
evaluated on the 2D benchmarks until the protocol is settled.

| Keypoint definition | PCK @0.05 | PCK @0.1 | Gap to paper (0.81 / 0.94) |
|---|---|---|---|
| Assumed: HMR2's SMPL mapping (`smpl_hmr2`) | 0.782 | 0.910 | −2.8 / −3.0 points |
| Diagnostic: PEAR's own joints (`pear_native`, SMPLXV2 joint regressor) | 0.793 | 0.912 | −1.7 / −2.8 points |

Both runs: `/raid/ubx858/outputs/phase2/pck/pear/coco{,_pear_native}/`, 111 s each on GPU 0.

- **The gap is the hips.** Per-keypoint PCK@0.05 is 0.36–0.42 at the hips against 0.71–0.93 for every
  other keypoint (PCK@0.1: hips 0.78, the rest 0.91–0.94). The predicted hips sit about 3% of the box
  size below the COCO-annotated hips (mean dy +0.027 of box size). This is a keypoint-definition
  difference (a model's internal hip joint vs COCO's surface hip), not a pose error.
- Face, wrists, knees and ankles are at 0.79–0.93 @0.05.
- **Likely explanation** (*assumption*): PEAR's author either uses a different hip definition or keypoint
  subset, or a different crop. His script will settle this. Evaluator tests pass 5/5, including a
  bit-for-bit match with HSMR's own `keypoint_pck_accuracy`.

## Decisions needed

1. **PoseTrack, AGORA, EMDB:** these need your accounts or applications (see the table). Please start
   the AGORA registration and the EMDB application. For PoseTrack, tell me where to get the 2018 val
   images.
2. **AGORA protocol:** test-server submission (comparable to 59.3) or a validation proxy (not
   comparable)? Ideally confirmed by PEAR's author.
3. **Training data:** approve downloading COCO train2017 (19.3 GB) and MPII (12.1 GB), and say where
   AI Challenger should come from.
4. **Disk:** approve deleting the `step_*.pt` files of (L-0.1) and (L-0.05), keeping `best.pt` and
   `latest.pt`.
5. **PEAR's author's script:** when it arrives, the assumed items above are checked against it
   before any gate is declared final.
