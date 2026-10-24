# Disk operations (DNT)

## 2026-10-19: expanded/BEDLAM removed

```
/raid/ubx858/datasets/expanded/BEDLAM was REMOVED on 2026-10-19 09:48:05 (user-approved, DNT disk plan).
It was only the extracted copy of the BEDLAM2 video archives in /raid/ubx858/datasets/raw/bedlam2.0 (kept).
Checks before removal (logs: /raid/ubx858/outputs/dnt/disk/):
  - readers: only defaults of tools/build_bedlam2_gt_crops.py and tools/build_bedlam2_hand_patches.py (PEAR), and two
    finished Phase-1 analysis scripts (phase1/task1/scripts/c2_crop_boxes.py, c2_camera_crop_check.py); no config,
    trainer or evaluator; no open file handles.
  - all 240 archives in raw/bedlam2.0 list with tar -t without errors (bedlam2_tar_check/exit_codes.txt).
  - all 20,680 files of expanded/BEDLAM (40 jobs, mp4 only) are present inside the matching <job>_mp4.tar.
To regenerate (identical layout <job>/mp4/seq_NNNNNN.mp4):
  mkdir -p /raid/ubx858/datasets/expanded/BEDLAM && cd /raid/ubx858/datasets/expanded/BEDLAM
  for j in 20240416_1_171_yogastudio_orbit_timeofday 20240423_1_171_yogastudio_staticloc_timeofday 20240424_1_171_citysample_orbit 20240425_1_171_citysample_dolly 20240425_1_171_hdri 20240429_1_171_stadium 20240514_1_1001_citysample_tracking 20240606_1_500_stadium_closeup 20240613_1_200_hdri 20240614_1_300_hdri 20240618_1_500_ai0805_orbit 20240619_1_250_ai1004_tracking 20240625_1_2337_hdri 20240628_1_250_ai1004_tracking 20240701_1_250_ai0901_lookat 20240703_1_250_ai0901_orbit_portrait 20240708_1_250_ai0901_static_portrait 20240710_1_250_ai0805_orbit_portrait 20240725_1_250_ai0805_vcam 20240726_1_250_ai0805_vcam 20240729_1_250_ai1004_vcam 20240730_1_250_ai1101_vcam 20240731_1_1827_hdri 20240806_1_250_ai1101_vcam 20240808_1_250_ai1105_vcam 20240809_1_250_ai1102_vcam 20240813_1_250_ai1004_tracking 20241107_1_250_archmodelsvol8_tracking 20241114_1_4619_hdri 20241204_1_2120_hdri 20241213_1_250_rome_tracking 20241217_1_250_rome_vcam 20250103_1_250_rome_vcam 20250110_1_250_chemicalplant_vcam 20250113_1_250_rome_vcam 20250123_1_250_chemicalplant_vcamego 20250131_1_250_ai1102_vcamego 20250211_1_250_ai1105_upperbody 20250212_1_250_yakohama_upperbody 20250214_1_250_chemicalplant_upperbody ; do tar -xf /raid/ubx858/datasets/raw/bedlam2.0/${j}_mp4.tar; done
(~126 GB; the processed crops in /raid/ubx858/datasets/processed/bedlam2_gt* do not need it.)
```

## 2026-10-23 cleanup and new rules
- Abandoned work deleted (~101 GB): see phase3_dnt/DELETED.md (what, size, how to recreate). The expert models (HaMeR /
  WiLoR / SMIRK) are re-downloadable: clone the three repos at the commits listed there into /raid/ubx858/experts, then
  `bash phase3_dnt/sprint/experts_fetch/fetch_weights.sh`.
- Checkpoint rule (user, 2026-10-23): during a run only latest.pt + best.pt; after a successful run latest.pt is deleted
  automatically (`logging.keep_best_only_at_end: true`), leaving best.pt (live + EMA weights; the selection rule says which).
- MOYO image zips (4 train takes, 18.1 GB): deleted after the crops are built and checked (user-approved). Re-download:
  `python -I phase3_dnt/sprint/moyo_download.py --out /raid/ubx858/datasets/raw/moyo --get /raid/ubx858/datasets/raw/moyo/subset_B_urls.txt`
  (login read from ~/.mpi_credentials; URL list = the 4 option-B lines of moyo_toolkit's images_train.txt).
