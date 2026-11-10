# Timing on the desktop PC (handoff, 2026-11-09)

Goal: measure **reconstruction time** (photo -> ready avatar, once per person) and **animation time** (one driving frame -> 512 px
render) for our model and the baselines on the desktop GPU. Quality numbers come from the server runs on the full test set; do
**not** recompute them on the PC. Only timings are measured here.

## Where things are (vault)

Vault root: `/vault/research/COS/VIRLab/ubx858/persistent_avatar/` (the mount point on the PC may differ; adjust the prefix).

| What | Path (under the vault root) |
|---|---|
| Our model (GUAVA-FB) | `outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt` |
| Released GUAVA (assets/GUAVA) | `backup_for_desktop/models/guava_released.tar` |
| Pose regressor (SPRINT), for "ours, estimated pose" and the live app | `backup_for_desktop/models/sprint_scratch_best_ema.pt` |
| Test data, ready-made (512 crops, poses, cameras, GT) | `outputs/guava_fb/eval/{avatar_test_v1,publicbench_v1,publicbench_v2,finalbench_ext}/base/` and `gt512/` |
| Baseline installs (code, conda envs, caches) | `backup_for_desktop/baselines/{gart,gaussianavatar,exavatar,idol,lhm,lhmpp}.tar` |
| Baseline weights | `backup_for_desktop/baselines/weights.tar`, `backup_for_desktop/baselines/lhmpp_pretrained.tar` |
| NeuMan raw, SMPL models | `backup_for_desktop/data/neuman_dataset.tar`, `backup_for_desktop/data/smpl_models.tar` |
| Contents list | `backup_for_desktop/README.md` |

The baseline envs contain absolute paths: extract them to `/dev/shm/ubx858_baselines/` (or the same path) so they run unchanged.
Treat the vault `eval/` and `runs/` trees as read-only; write timing outputs to a new directory.

Split files (clip lists) are in this directory: `avatar_test_v1.json` (UBody clips), `publicbench_v1.json` (People-Snapshot,
NeuMan), `publicbench_v2_xhumans.json` (X-Humans). `finalbench_ext.json` (all People-Snapshot / NeuMan sequences + UBC-Fashion)
is pushed once its conversion on the server finishes; it is not needed for timing.

## Timing clips (fixed rule: first clip id in sorted order per dataset)

`ubody/ConductMusic_S15`, `peoplesnapshot/female-3-casual`, `neuman/bike`, `xhumans/00028`.
Same clips for every method. Per-person-trained baselines (GART, GaussianAvatar, ExAvatar): train time on `neuman/bike` only is enough.

## Our model: the four configurations to time

| Configuration | Quality row (server) | Reconstruction | Per frame |
|---|---|---|---|
| Ours, GT SMPL-X pose | `fb_gt` | avatar from photo | render (deformation + rasterization + refiner) |
| Ours, SPRINT pose | `fb_sprint` | avatar from photo + SPRINT fit of the photo | SPRINT pose + render |
| Ours, GT SMPL-X pose + UV appearance update | `fb_update25/50/100` | as `fb_gt` | render + UV update (fuse one new frame into the UV map, decode) |
| Ours, SPRINT pose + UV appearance update | none (live app only; no quality row) | as `fb_sprint` | SPRINT pose + render + UV update |

- The update rows' quality on the test set uses GT SMPL-X poses (`fusion_gt_row.py`, as registered). SPRINT + UV update exists
  only in the live app (`live_sprint_guavafb.py --worker uv`, `phase3_dnt/worker6/uv_fusion.py`); time it there, and report it as
  a speed number only.
- Report the UV update both ways: its own ms per fused frame, and the resulting live loop fps with `--worker uv` (the update runs
  asynchronously, so also report how many updates per second it achieves).

## Protocol

- Idle GPU (nothing else running), batch 1, fp32 / the precision each method is evaluated at, 512 px output.
- 10 warm-up runs, then >= 50 timed runs; CUDA events with `torch.cuda.synchronize()`; report median and p95 in ms.
- Report separately:
  - **Reconstruction**: avatar building from the source photo (network part), plus preprocessing (pose fitting, matting) as its own line.
  - **Animation**: per driving frame, pose given -> rendered 512 image (deformation + rasterization + refiner). fps = 1000 / median ms.
  - **Pose estimation** per frame (SPRINT) for the estimated-pose row.
  - **Appearance update** per frame (UV fusion of one new frame), for the update rows.
- Record GPU model, driver, CUDA, torch versions, git SHA. Do not change any model setting for speed.

## Existing code to start from (branch `driven_not_tracked`)

- `phase3_dnt/guava_timing.py`: per-component GUAVA timing (reconstruction stages A1–A9, per-frame stages B1–B4) with the protocol
  above; switch the checkpoint/model setup to GUAVA-FB the way `eval_guava_rows.py` does (`load_ckpt` / `setup_a20`, `--fb-ckpt`).
- `phase3_dnt/eval_split/eval_guava_rows.py`: how the evaluated rows build and render avatars (`--row fb_gt`, `fb_sprint`, `guava`).
- `phase3_dnt/eval_split/fusion_gt_row.py`: the appearance-update rows (`--mode mean --prior 2 --thr 0 --cap 0`).
- `phase3_dnt/eval_split/time_sprint.py`: SPRINT per-frame timing.
- `phase3_dnt/live/live_sprint_guavafb.py --timing-json`: end-to-end live timing (webcam).
- Baselines: `phase3_dnt/baselines/*/` (run scripts per method); time their reconstruction and per-frame render on the same clips.

## Paper wording

"Quality is measured on the full test set; timings on <desktop GPU>, on four clips, median of N runs after 10 warm-up runs."
Log the timing runs in `phase3_dnt/RUNLOG.md` with the git SHA.
