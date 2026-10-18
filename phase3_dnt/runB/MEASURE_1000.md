# Run B expert labels — measurement on 1,000 crops (2026-10-18, measured; GPU 5, shared L40S)

Crops: COCO-WholeBody training persons (our COCO rows), boxes from the annotated face / hand keypoints mapped back to the
ORIGINAL images (the 512 px context crops make hands ~25 px; originals ~40 px). Code: PEAR `tools/expert_hands.py`,
`tools/expert_faces.py` (renderers stubbed: visualisation only; chumpy patched in a private site dir; weights / MANO /
FLAME under `/raid/ubx858/experts`, not in git).

## Hands: HaMeR + WiLoR, agreement filter (< 15 mm root-aligned 3D AND < 8 px 2D)
| | value |
|---|---|
| GPU forward | HaMeR 9.4 ms / hand, WiLoR 9.8 ms / hand (batch 32) |
| Disagreement median / p90 | 3D 6.0 / 13.1 mm; 2D 1.0 / 3.0 px |
| **Kept** | **92.2 %** (rejected: 6.9 % 3D only, 0.2 % 2D only, 0.7 % both) |
| 2D error vs COCO-WholeBody GT (median) | HaMeR 2.1 px (7.0 % of box), WiLoR 2.3 px; kept 6.8 % vs rejected 10.5 % of box |
| Kept by hand size | 89 % (< 20 px) … 97 % (> 80 px); median box 31 px |
Note: with hands this small the 8 px 2D limit almost never binds; a size-relative limit (e.g. 0.15 x box) would be more
informative. Not changed (user criterion).

## Faces: SMIRK, keep if landmark reprojection < 2 x median
| | value |
|---|---|
| MediaPipe face found | 79.8 % (63 % for faces < 20 px, 93 % > 80 px; median face 31 px) |
| Reprojection (224 px crop) median / p90 | 3.5 / 8.2 px |
| Rejected by reprojection (of found) | 13.5 % |
| **Kept** | **69 %** of all faces |
| Kept labels | jaw 5.5 deg mean (p90 9.2), FLAME expression norm 9.6, eyelid 0.33 — clearly non-neutral |
| Cost | GPU 1.2 ms / face; CPU (MediaPipe) 33 ms / face |
SMIRK outputs FLAME 2020 expression (50) + jaw + eyelids (+ head pose); **no eye-gaze output** ("eyes" = eyelids only).
Its FLAME is EHM's FLAME 2020 generic: no conversion needed.

## Projected cost for the current image set (UBody + COCO-WholeBody; Motion-X / TalkSHOW / ARCTIC to add)
Faces 192k: ~0.07 GPU-h, ~1.8 CPU-h (MediaPipe; parallel over 16 processes ~7 min). Hands 336k: ~1.8 GPU-h (both experts)
+ crop preparation on CPU (parallel). Expected kept: ~132k faces, ~310k hands. Adding the new datasets roughly doubles this.
