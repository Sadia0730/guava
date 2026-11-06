# LHM baseline (LHM-1B-HF) under the AvatarTest-v1 contract

Registered in PREREG "FinalBench addendum 2: LHM". Adapter `run_lhm.py`, driver `lhm_run.sh`.

## What runs where (all under /dev/shm, nothing on /raid or the root disk)

| Item | Location | Provenance |
|---|---|---|
| LHM code | `/dev/shm/ubx858_baselines/lhm/src` | github.com/aigc3d/LHM @ 4f88aae (unchanged) |
| Weights | `/dev/shm/ubx858_baselines/weights/lhm_1b_hf` | HF 3DAIGC/LHM-1B-HF @ 92372582, model.safetensors sha256 59dc2516... |
| Sapiens-1B encoder | `src/pretrained_models/sapiens/.../sapiens_1b_epoch_173_torchscript.pt2` (hard link) | HF facebook/sapiens-pretrain-1b-torchscript, sha256 b5094b60... |
| Priors | `src/pretrained_models/`, `src/gfpgan/weights/` | LHM_prior_model.tar on HF 3DAIGC/LHM, members read by HTTP byte range (`tools/tar_ranges.py`, `tools/tar_extract.py` in `/dev/shm/ubx858_baselines/lhm`): 1_40000.ply, RealESRGAN_x4plus.pth, facexlib detection / parsing; SMPL-X / FLAME / voxel grid / VGGHead / arcface copied from the LHM++ priors (byte-identical by size; three small files compared with `cmp`) |
| GFPGAN v1.3 | `/dev/shm/ubx858_baselines/lhm/pydeps/gfpgan/weights/` | GitHub release v1.3.0, sha256 c953a88f... |
| DINOv2 ViT-L/14-reg init | `/dev/shm/ubx858_baselines/lhm/cache/hub/checkpoints/` | downloaded by LHM's constructor; overwritten by the checkpoint ("All keys matched") |
| Python | `lhmpp` conda env (unchanged) + overlay `PYTHONPATH=/dev/shm/ubx858_baselines/lhm/pydeps` | basicsr @ 8d56e3a (git; the PyPI 1.4.2 breaks on torchvision 0.18), gfpgan 1.3.8, facexlib 0.3.0, GPUtil, ImageHash, filterpy; all `--no-deps` |

The Alibaba OSS host of LHM's prior archive does not answer from this server; the HF mirror holds the same archive (18.8 GB),
of which only ~0.26 GB is needed.

## Adapter decisions

- Source: LHM's own `infer_preprocess_image` (copied; `assert w < h` dropped because 8 UBody source boxes are wider than tall,
  and the function's half-body branch handles them), with the package's StyleMatte alpha instead of SAM2 / rembg.
- Head crop: LHM's VGGHead detector on the raw 512 source crop; zeros when no face is found (LHM's own fallback). `face_found`
  is logged per clip in run.json.
- Body model: LHM uses the SMPL-X 2020 neutral file (same template / regressor / skinning / first 10 shape dirs as the package's),
  10 betas, FLAME 100 expression on the SMPL-X face, flat_hand_mean = True: conversions identical to the LHM++ adapter.
- Render: LHM's 3DGS rasteriser directly at 512 x 512 with the frame's K (render size = 2 x principal point = 512) and a black
  background, i.e. the premultiplied render the contract asks for. No crop / paste (unlike LHM++, whose neural renderer works on
  a crop).
- `OPENBLAS_CORETYPE=SkylakeX` is set before numpy import (lhmpp env bug, see `../lhmpp/NOTES.md`).

## Smoke (ValBench-v1, validation only)

Four clips, 8-12 frames each (`valbench_v1/renders/lhm_smoke/`). Silhouette extents on BEDLAM2 at alpha > 80/255 are within
1-6 px of the StyleMatte box (LHM++: 0-5 px); at alpha > 10/255 LHM is ~8 px wider all round (soft edges). PSNR / IoU in PREREG.

## Licences

LHM code Apache-2.0; LHM-1B-HF card: Apache-2.0; Sapiens CC BY-NC 4.0; GFPGAN / Real-ESRGAN BSD-3 / Apache-2.0. Research use only.

## Amendment 1 (face crop) and results

The first test run (`lhm`) fed LHM's head branch zeros on 23 / 81 sources: a 1.65x head box with a negative top / left start
slices to an empty crop (22), and LHM's no-face branch raises a NameError (1). `--clamp-face` clips the start at 0 (NumPy already
clips the end); `--face-on-matte` detects and crops on the matted source so a bystander's head is not chosen. The reported row
is `lhm_facefix` (heads 81 / 81). FinalBench 18.18 dB (unfixed 18.11), BEDLAM2 24.29 (unfixed 24.24). RUNLOG 123.
