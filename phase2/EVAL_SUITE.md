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

### LSP-Extended PCK (released PEAR checkpoint, 9,428 persons)

**Gate NOT passed.** PEAR scores **0.494 / 0.799** against the paper's 0.55 / 0.81 (−5.6 / −1.1
points). Both keypoint definitions give identical numbers here, because LSP-Extended is scored only on
the 19 extra joints, which both definitions take from `SMPL_to_J19`. Output
`/raid/ubx858/outputs/phase2/pck/pear/lspet_*`, 196 s.

The large @0.05 gap points to a different definition of those 19 joints, or a different crop, in
PEAR's own evaluation. Again, only the author's script can settle this.

### UBody-intra (in progress)

- **Test subset built** (`phase2/build_ubody_intra_test.py`): 2,354 items with OSX's exact selection.
  - Every 100th annotation, intra-scene test videos, video name = `file_name.split('/')[-2]` with
    `_Trim…` stripped.
  - No test items come from Movie or Speech: their clips are stored as `…_sceneNNN` and OSX's rule
    never matches them.
  - Frames are extracted with OSX's own `ffmpeg -r 30` command, only for the needed clips.
- **Evaluator** (`tools/eval_ubody_standard.py`): OSX's metrics; tests pass 4/4.
- **Blocked:** the GT meshes need **SMPL-X v1.1** (`SMPLX_NEUTRAL.npz`), which is not on disk. We have
  SMPL-X 2020 and the BEDLAM locked-head variant, neither of which is the model OSX's pseudo-GT was
  fitted with.
- **Provisional run** (GT meshes from the BEDLAM *locked-head* SMPL-X as a stand-in for v1.1; **not a
  gate result**; output `/raid/ubx858/outputs/phase2/ubody/*_PROVISIONAL_lockedhead/`):

  | | PA-MPVPE all / hand / face | MPVPE all / hand / face | PA-MPJPE body / hand |
  |---|---|---|---|
  | GT as prediction (pipeline check) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
  | PEAR (released) | 76.5 / **9.0** / 9.2 | 134.0 / 30.2 / 33.6 | 132.4 / 9.0 |

  - Counts: 2,354 items; 2,307 with a hand inside the image; 2,305 with the face inside.
  - PEAR's hand PA-PVE of 9.0 mm compares with the paper's 9.8 (Table 2). It is within 1 mm, and better,
    as on EHF (11.0 vs 12.8).
  - The body numbers are large. These are upper-body videos: OSX's protocol rigidly aligns all 14
    joints, including legs that are outside the frame and carry only pseudo-GT, then scores only the
    joints inside the image. That likely explains it, but it is not verified, and these body numbers
    are not used for any decision.
  - Final numbers wait for SMPL-X v1.1.

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
   before any gate is declared final. Both 2D gates (COCO, LSP-Extended) currently fail under the
   assumed protocol, so the 2D benchmarks are on hold until then.
6. **SMPL-X v1.1** (`models_smplx_v1_1.zip` from smpl-x.is.tue.mpg.de, under your existing account):
   needed for the UBody GT meshes. Please download it, or allow me to fetch it with your login.

## PEAR author's evaluation script (received 2026-10-13)

**Source.** `evaluation_kp.zip` (SHA-256 `573b9f5b…`) → `phase2/reference/evaluation_kp/evaluation_kp.py`.
It stays **untracked** (`.gitignore`) because there is no permission to redistribute it.
- It imports the author's private code base (`main.evaluation.hsmr_eval`, his `Ehm_Pipeline` and
  config), so it cannot run as is.
- Its protocol is ported in **`tools/eval_author_protocol.py`** (tests 5/5; each ported function is
  checked against HSMR's own code). The joint-index tables are not copied into tracked code; they are
  parsed with `ast.literal_eval` from the untracked file at run time.

### Every difference between the author's script and our previous evaluators

| # | Item | Author (evaluation_kp.py + HSMR) | Ours before | Effect |
|---|---|---|---|---|
| 1 | Crop, 2D sets | HMR2 `ImageDataset` (eval): `bbox_size = expand_to_aspect_ratio(scale·200, [192,256]).max()`, square crop at the npz centre → 256 px | PEAR `process_bbox` ×1.25 of the npz box (square) | ours is about 1.25× looser → COCO +0.8 pt with his crop |
| 2 | Crop, 3DPW | same HMR2 crop from `3dpw_test.npz` centre/scale | GT-keypoint box ×1.2 → `process_bbox` ×1.25 | different input crops |
| 3 | Input normalisation | HSMR's data module returns a mean/std-normalised patch; PEAR's pipeline normalises internally | [0, 1] RGB | **assumed** [0, 1] in the port (otherwise PEAR's input would be normalised twice) |
| 4 | SMPL-X → SMPL | `smplx2smpl.pkl['matrix']` on `vertices[:, :-120]` | the same file and matrix (`SMPLX2SMPL.zip`), `vertices[:10475]` | identical |
| 5 | SMPL model | `SMPL_NEUTRAL.pkl` (version not stated) J_regressor + smplx `vertex_joint_selector` (45 joints) | our `SMPL_NEUTRAL.npz` J_regressor + the same vertex ids | assumed identical |
| 6 | COCO keypoints | SPIN `JOINT_MAP`: OpenPose joints from the SMPL kinematic joints + vertex joints, **hips = SMPL kinematic hips (1, 2)**, mid-hip = pelvis | HMR2 mapping: the same indices | identical (the hip mismatch remains in both) |
| 7 | LSP-Extended (19 GT joints) | SPIN `JOINT_MAP`: limbs = SMPL **kinematic** joints (ankles, knees, wrists, elbows, shoulders); hips, neck, head top, pelvis, thorax, spine, jaw, head from SPIN's 9-joint `J_regressor_extra.npy` | `SMPL_to_J19` for all 19 | the main LSP difference: +7.2 pt @0.05 with his crop and mapping together (not separated) |
| 8 | PoseTrack keypoints | OpenPose-25 (SMPL) + J19, then his `smpl_to_h36m` reindexing: OpenPose shoulders, hips, mid-hip and right knee taken from J19; the extra right ankle, left knee and left ankle from the OpenPose (kinematic) joints | OpenPose-25 + J19, no reindexing | not measurable yet (no PoseTrack images) |
| 9 | PCK normalisation and scoring | HMR2 `EvaluatorPCK` (crop-normalised × `box_size / max(scale·200)`, conf > 0.5, mmpose per-keypoint mean) | the same, in image pixels / `max(scale·200)` | identical (algebraically equal; tests) |
| 10 | 2D projection | `GS_Camera` focal 24 over a 1024 px screen, /1024 − 0.5 | PEAR crop camera → image pixels | equivalent |
| 11 | 3DPW metric | HMR2 `Evaluator`: GT = J19 3D from `3dpw_test.npz`; prediction = the 44 joints above (reindexed); both centred at joint 39 (J19 MPII pelvis); 14 joints [25..37, 43]; MPJPE and PA-MPJPE (similarity on the 14) | H36M regressor → J14 on predicted and GT meshes (gendered SMPL from the 3DPW pkl), hip-midpoint centring, similarity PA; plus PVE | different joints, centring and GT source |
| 12 | 3DPW joint sources | through `smpl_to_h36m` the prediction is asymmetric: right knee from J19, left knee and both ankles from SMPL kinematic joints | symmetric | part of #11 |
| 13 | Pose input to EHM | `pose_type='aa'` (his pipeline outputs axis-angle) | `pose_type='rotmat'` (the released pipeline outputs rotation matrices) | same representation, different format |
| 14 | Checkpoint | `checkpoints/ehm_model.pt` (his local file) | the released Hugging Face checkpoint | **possibly not the same weights**: unknown |

### Gates with the author's protocol (released PEAR, measured)

| Benchmark | PEAR here | Paper (Table 3) | Difference | Gate (~2 pt / ~2 mm) |
|---|---|---|---|---|
| COCO-val PCK @0.05 / @0.1 | 0.790 / 0.919 | 0.81 / 0.94 | −2.0 / −2.1 pt | **borderline**: at the limit |
| LSP-Extended PCK @0.05 / @0.1 | 0.566 / 0.817 | 0.55 / 0.81 | +1.6 / +0.7 pt | **pass** |
| PoseTrack-val PCK | — | 0.87 / 0.97 | — | not run (no images; needs your PoseTrack access) |
| 3DPW-test MPJPE / PA-MPJPE (author's protocol) | 73.78 / 46.73 | 71.3 / 45.3 | +2.5 / +1.4 mm | MPJPE **just outside**, PA pass |
| 3DPW-test MPJPE / PA-MPJPE (our standard protocol, Task 1) | 73.56 / 45.65 | 71.3 / 45.3 | +2.3 / +0.35 mm | (accepted in Task 1) |

Outputs are in `/raid/ubx858/outputs/phase2/author/pear/`.
- Our previous protocol gave COCO 0.782 / 0.910 and LSP 0.494 / 0.799.
- Adopting his crop and his LSP joint definitions closes most of the LSP gap and part of the COCO gap.
- The COCO hip mismatch is identical in his script, so it is not a protocol difference.
- The remaining ~2 pt (COCO) and ~2.5 mm (3DPW MPJPE) are consistent with row 14 (his local checkpoint
  vs the released one), the SMPL-neutral version (row 5) or the input-normalisation assumption
  (row 3). None of these can be checked without the author.
- **Verdict:** LSP passes, COCO is at the limit, 3DPW MPJPE is 0.5 mm over (PA passes). The Table-3
  comparison below uses the author's protocol for every row and is to be read with these caveats.

## Table 3 format: evaluation splits, author's protocol (2026-10-13, measured)

Every row uses `tools/eval_author_protocol.py` (evaluator SHA-256 `820ced0a…`). Splits are PEAR's Table-3 evaluation sets:
- COCO val2017, 4,648 persons;
- LSP-Extended (hr-lspet), 9,428 persons;
- 3DPW test, 35,515 person-frames.

All rows are at stride 1, with a strict checkpoint load. Outputs are in `/raid/ubx858/outputs/phase2/author/<row>/`.
PoseTrack has no images (it needs your PoseTrack access), and the AGORA column is left empty for now. Read the table
with the gate caveats above: COCO is at the limit and 3DPW MPJPE is 0.5 mm outside.

| Method | COCO @0.05 | COCO @0.1 | LSP-Ext @0.05 | LSP-Ext @0.1 | PoseTrack @0.05 | PoseTrack @0.1 | 3DPW MPJPE ↓ | 3DPW PA-MPJPE ↓ | AGORA Body MVE ↓ | Laptop latency † |
|---|---|---|---|---|---|---|---|---|---|---|
| *PEAR, paper (Table 3)* | *0.81* | *0.94* | *0.55* | *0.81* | *0.87* | *0.97* | *71.3* | *45.3* | | |
| PEAR (released ViT-H) | **0.790** | **0.919** | **0.566** | **0.817** | — | — | 73.78 | 46.73 | | 16.5 ms |
| Student 235000 | 0.405 | 0.660 | 0.213 | 0.431 | — | — | 121.49 | 74.94 | | not known ‡ |
| (L-0.1) | 0.727 | 0.897 | 0.463 | 0.738 | — | — | 74.28 | 48.89 | | 5.3 ms |
| (L-0.1) + refiner (ii) | 0.630 | 0.870 | 0.388 | 0.704 | — | — | 75.65 | 45.62 | | 5.3 ms + refiner (not yet measured on laptop) |
| Control: (L-0.1) head trained on GT, same compute as (ii) | 0.612 | 0.861 | 0.376 | 0.700 | — | — | 81.67 | 48.34 | | 5.3 ms |
| L-data | 0.734 | 0.908 | 0.445 | 0.735 | — | — | **72.83** | 48.43 | | 5.3 ms |
| L-data + refiner | 0.653 | 0.885 | 0.390 | 0.710 | — | — | 75.78 | **44.89** | | 5.3 ms + refiner (not yet measured on laptop) |
| L-real (added 2026-10-14) | 0.772 | **0.921** | 0.484 | 0.755 | — | — | 74.51 | 48.75 | | 5.3 ms |
| L-real + refiner v2 (added 2026-10-14) | 0.760 | 0.918 | 0.482 | 0.759 | — | — | 74.96 | 49.04 | | 5.3 ms + refiner (not yet measured on laptop) |

† **Laptop latency** (RTX 3080 Laptop, batch 1) are your measurements, not re-measured here: PEAR 16.5 ms, student 5.3 ms. 5.3 ms is
the ViTPose-S student (single expert), which every student row shares except student 235000.
‡ Student 235000 has the earlier default backbone, so the ViTPose-S figure does not apply to it.

**Rows.**
- PEAR: the released Hugging Face checkpoint.
- Student 235000: `pear_student_v2_phase2/checkpoints/step_0235000.pt`.
- (L-0.1): `best.pt`, step 102k.
- L-data: `best.pt`, step 94k.
- Refiner (ii): the A4 no-render refiner, applied once.
  - On (L-0.1): `phase1_task3/a4/norender/checkpoints/latest.pt`, step 20,000.
  - On L-data: `phase2/refiner_Ldata/norender/checkpoints/latest.pt`, step 20,000, SHA-256 `d521ddb9…`. Same recipe as A4 (ii) and seed 20261012; error bank rebuilt from L-data (`error_bank_Ldata_bedlam2val.npz`); PEAR `907827d`; 76 min on an L40S.
- Control: `phase2/control_headft_L01/run/checkpoints/latest.pt`, step 20,000, SHA-256 `76c60e71…`.
  - Recipe: (L-0.1)'s head (40.5M parameters) trained with the refiner's GT losses, data, order, seed, steps and batch.
  - Backbone frozen; LR 1e-4; PEAR `907827d`.

### Changes (points / mm; measured)

| Comparison | COCO @0.05 | COCO @0.1 | LSP @0.05 | LSP @0.1 | MPJPE | PA-MPJPE |
|---|---|---|---|---|---|---|
| L-data − (L-0.1) | +0.8 | +1.1 | −1.8 | −0.3 | −1.45 | −0.46 |
| (L-0.1) + refiner − (L-0.1) | −9.7 | −2.7 | −7.5 | −3.5 | +1.37 | **−3.27** |
| L-data + refiner − L-data | −8.2 | −2.2 | −5.6 | −2.6 | +2.95 | **−3.54** |
| Control − (L-0.1) | −11.5 | −3.6 | −8.7 | −3.9 | +7.39 | −0.55 |
| L-data − PEAR (released) | −5.5 | −1.1 | −12.1 | −8.2 | −0.95 | +1.70 |
| L-real − L-data | **+3.8** | +1.4 | **+3.8** | +2.0 | +1.68 | +0.32 |
| L-real + refiner v2 − L-real | −1.3 | −0.3 | −0.2 | +0.4 | +0.44 | +0.29 |
| L-real − PEAR (released) | −1.7 | +0.3 | −8.2 | −6.1 | +0.73 | +2.02 |

**What this shows.**
- **L-data is the best plain student.** Against (L-0.1) it is slightly better on COCO and 3DPW and slightly worse on LSP.
  Its 3DPW MPJPE (72.83 mm) is below the released PEAR measured the same way (73.78 mm). Its PA-MPJPE is 1.7 mm above.
- **The refiner trades 2D alignment and MPJPE for PA-MPJPE**, on both students.
  - Gain: −3.3 / −3.5 mm PA-MPJPE.
  - Cost: −8 to −10 pt COCO PCK@0.05 and +1.4 / +3.0 mm MPJPE.
  - **L-data + refiner (44.89 mm PA-MPJPE)** is below both the released PEAR measured here (46.73) and PEAR's paper value (45.3).
- **The control separates the two effects.**
  - Training only the student's head on the same BEDLAM2 GT, with the same compute, loses as much 2D PCK (−11.5 pt) and even more MPJPE (+7.4 mm). It gains only 0.55 mm PA-MPJPE.
  - So the refiner's PA-MPJPE gain is not just extra GT training.
  - The 2D loss on real images (COCO, LSP) comes from fine-tuning on BEDLAM2 GT alone. This is a domain effect that both share, not a property of the refiner.
- **2D gap to PEAR:** −5.5 pt (COCO) and −12 pt (LSP) at @0.05. This is the gap run L-real (real images with 2D keypoints) targets.

**L-real and refiner v2 (2026-10-14, measured).**
- **L-real** is the L-data recipe plus 20% COCO/MPII images with a 2D keypoint loss (hip weight 0): `best.pt`, step 106k, SHA-256 `ff23da22…`, 3DPW-val PA-MPJPE 49.66; about 15 h on GPU 2. Against L-data:
  - 2D PCK@0.05 improves by +3.8 pt on both COCO and LSP, halving the COCO gap to PEAR. COCO @0.1 now matches PEAR.
  - 3DPW gets slightly worse: MPJPE +1.7 mm, PA-MPJPE +0.3 mm. That is consistent with fewer BEDLAM2 rows per batch (38 instead of 51).
- **Refiner v2** is the no-render refiner trained on the L-real student with BEDLAM2 GT plus the real-image 2D loss (`phase2/REFINER_V2_AND_PARTS.md`):
  - `phase2/refiner_v2_Lreal/checkpoints/latest.pt`, step 20,000, SHA-256 `52955e1e…`; PEAR `b3da15f`; 49 min.
  - **Keeps 2D alignment:** −1.3 pt COCO @0.05, against −8.2 for the old refiner on L-data.
  - **Loses the old refiner's 3DPW gain:** PA-MPJPE +0.3 mm, against −3.5 mm.
  - On BEDLAM2 val it still improves strongly (68.2 → 50.8 mm), and on EHF it gives the largest gain of any row (below). The old refiner's 3DPW gain therefore seems to come with the change in 2D alignment; why is not yet known.

### EHF, all students (standard evaluator, fp32, measured)

| | PA-PVE all | PVE all | PA-PVE hands | PA-PVE face | PA-MPJPE body |
|---|---|---|---|---|---|
| PEAR (released) | 47.9 | 66.6 | 11.0 | 8.7 | 49.2 |
| (L-0.1) | 52.0 | 85.9 | 14.4 | 6.2 | 62.0 |
| L-data | 51.3 | 83.9 | 15.1 | 6.3 | 60.8 |
| L-data + refiner | 46.9 | 85.5 | 15.1 | 6.3 | 51.3 |
| L-real | 50.6 | 83.4 | 14.8 | 6.4 | 60.0 |
| L-real + refiner v2 | **45.8** | **79.3** | 14.8 | 6.3 | 51.6 |

- The refiner rows use the new `--refiner` option of `tools/eval_ehf_standard.py`. Its path without the option is per-frame identical to before (evaluator SHA-256 `d3c6a5e2…` since the option was added).
- L-real on 3DPW test with the standard evaluator (GT-keypoint crop, fp32): MPJPE 75.83, PA-MPJPE 48.46, PVE 86.6, lower body 97.36. L-data: 73.82 / 48.28 / 85.30 / 92.04.

### L-data on the standard evaluators (Task 1 protocol, `best.pt` step 94k, measured)

| | MPJPE | PA-MPJPE | PVE | Lower body |
|---|---|---|---|---|
| 3DPW test, GT-keypoint crop, fp32 | 73.82 | 48.28 | 85.30 | 92.04 |
| 3DPW test, detector crop, fp32 | 74.13 | 48.30 | 85.55 | 92.26 |
| (L-0.1), GT-keypoint crop, fp32 (for reference) | 75.46 | 48.81 | 86.70 | 94.46 |
| PEAR, GT-keypoint crop (for reference) | 73.56 | 45.65 | 82.28 | 96.87 |

fp16 equals fp32 to within 0.01 mm on 3DPW.

**EHF, L-data:**

| Metric | All | Hands | Face |
|---|---|---|---|
| PVE, fp32 | 83.9 | 51.5 | 38.8 |
| PA-PVE, fp32 | 51.3 | 15.1 | 6.3 |

PA-MPJPE: body 60.8, hands 15.1. fp16 matches to within 0.1.

For reference:
- (L-0.1): PVE all 85.9; PA-PVE all 52.0 and hands 14.4; PA-MPJPE body 62.0.
- PEAR: PVE all 66.6; PA-PVE all 47.9; PA-MPJPE body 49.2.
