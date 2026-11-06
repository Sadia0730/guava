# IDOL baseline under the AvatarTest-v1 contract

Registered in PREREG "FinalBench addendum 3: IDOL". Adapter `run_idol.py`, driver `idol_run.sh`. RUNLOG 124.

| Item | Location | Provenance |
|---|---|---|
| IDOL code | `/dev/shm/ubx858_baselines/idol/src` | github.com/yiyuzhuang/IDOL @ 9fd9296 (unchanged), MIT |
| Checkpoint + caches | `src/work_dirs/ckpt/model.ckpt` (hard link), `src/work_dirs/cache_sub2` | HF yiyuzhuang/IDOL @ 3c80ec95; model.ckpt sha256 2294c771... |
| Sapiens-1B | `src/work_dirs/ckpt/` (hard link) | same file as LHM's (sha256 b5094b60...) |
| SMPL-X v1.1 | `src/lib/models/deformers/smplx/SMPLX/` | MPI models_smplx_v1_1.zip via `trackA2/download_mpi.sh` (credentials file, never printed) |
| CUDA pieces | `/dev/shm/ubx858_baselines/idol/pydeps` | graphdeco diff-gaussian-rasterization @ 9c5c202 (antialiasing), IDOL's fast-SNARF fuse / filter / precompute; built with system nvcc 12.8, sm_89, against the lhmpp torch 2.3.0+cu121 (build approved by the user) |

Body model: SMPL-X v1.1 neutral is SMPL-X 2020 except for the expression basis (asserted at start). The posed-mesh error of the
converted parameters (10 betas, 10 expressions, hands minus hands_mean) vs the GT mesh is logged per clip: 0.2-2 mm on UBody /
BEDLAM2 / X-Humans, 13-16 mm on People-Snapshot / NeuMan, where the GT shape needs more than 10 betas (rest-shape residual 7-9 mm,
the same projection LHM and LHM++ get). The demo's camera (scale 0.5, fixed distance) is not used; the GT camera-frame SMPL-X and
the 512 K are, and BEDLAM2 silhouettes land within 0-6 px of the matte. Renderer background set to 0 (pure 3DGS).
IDOL is trained on synthetic full-body humans only; on upper-body UBody sources its appearance degrades.
