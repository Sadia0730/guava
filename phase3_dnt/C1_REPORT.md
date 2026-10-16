# C1 — live-driving gap on GUAVA (G1), 2026-10-16 (measured)

**Setup** (pre-registered in `PREREG.md`): one GUAVA avatar per identity from the first tracked frame; every later frame
>= 60 frames (>= 2 s) after the source rendered under O (EHM-Tracker, offline oracle), S (L-real), S+R (L-real + refiner
v2), P (PEAR ViT-H) and per-part swaps; masked metrics on the upper-body mask; per-identity means; 95% bootstrap CIs over
identities. Code `phase3_dnt/c1_live_gap.py`, statistics `phase3_dnt/c1_report.py`; outputs
`/raid/ubx858/outputs/dnt/c1/results/` (`table.md`, `summary.json`, `report_stats.json`, per-clip `compare.jpg`).

**Coverage.** 36 identities in the split; **33 scored** (EHM-Tracker produced no track for 3 clips: an internal tensor
error on 2, no valid frames on 1). Validity condition (>= 30) met. 6,973 future frames (80–316 per identity). 20 EHM
frames with NaN crop boxes dropped (2 clips). Pre-registered sensitivity rule (oracle keypoint distance > 0.10 torso)
flags **6** identities as oracle failures (O kp distance 0.12–1.11 vs median 0.034).

## G1 decision (pre-registered: GO if O − S >= 0.5 dB PSNR or S − O >= 0.01 LPIPS, CI excluding 0)

| Identities | O − S PSNR (dB) | S − O LPIPS | G1 |
|---|---|---|---|
| **all 33 (primary)** | **+1.48 [+0.89, +2.08]** | **+0.015 [+0.009, +0.021]** | **GO** |
| 27 with a valid oracle (sensitivity) | +1.91 [+1.35, +2.49] | +0.018 [+0.013, +0.024] | GO |

Both criteria are met on both sets. For reference, O − (S+R) = +1.81 [+1.11, +2.57] dB (the refiner does not help
here) and O − P = +0.91 [+0.29, +1.58] dB (PEAR is closer to the oracle than our student).

## How much of the gap is placement? (descriptive, not gated — for C2's baselines)

| | all 33: O − S PSNR | S − O LPIPS | valid-oracle 27: O − S PSNR | S − O LPIPS |
|---|---|---|---|---|
| Raw | +1.48 [+0.89, +2.08] | +0.015 [+0.009, +0.021] | +1.91 [+1.35, +2.49] | +0.018 [+0.013, +0.024] |
| After per-frame 2D similarity alignment | +0.56 [+0.10, +0.96] | +0.006 [+0.002, +0.010] | +0.85 [+0.44, +1.24] | +0.008 [+0.004, +0.012] |
| Share of the gap removed by alignment | **62%** | **60%** | **56%** | **56%** |

**S with O's camera (S+O:cam)** changes S by **−0.75 [−1.19, −0.31] dB**: it does not close the gap, it widens it
(to ~2.2 dB). The image-plane component alone (S+O:ray) gives −0.83 dB; O's depth alone (S+O:depth) +0.08 [−0.23, +0.40]
dB (n.s.). The reverse swaps lose a lot: O with S's camera −2.65 dB, O with S's body −2.27 dB.

**Plain reading.**
- **Most of the S-vs-O gap is 2D placement:** a per-frame similarity transform (translation, scale, in-plane rotation)
  of the render onto the real frame removes ~60% of it. About 40% (0.56 dB, 0.006 LPIPS; CI excludes 0) remains after
  alignment: pose / expression / hand differences.
- **But the placement error cannot be fixed by swapping in a better camera.** Each driver's camera is consistent only
  with its own body (S puts the body ~0.6 m deeper and compensates); swapping the camera or the body alone breaks that
  pairing and makes things worse. Both the body and the camera of S are worse than O's (large leave-one-out losses),
  and they are coupled.
- **Implication for C2 baselines:** a post-hoc 2D correction of the rendered avatar (similarity alignment to keypoints
  detected on the live frame) is the natural first baseline — it would recover ~60% here and is cheap at runtime; a
  camera-only correction is not. Closing the remaining ~40% needs better body / face / hand driving.

## Per-part attribution (descriptive; neck-pivot swaps)

PSNR change, all 33 identities (mean [95% CI]); "gain" = S with the part from O vs S; "loss" = O with the part from S vs O.

| Part | Gain, raw | Gain, aligned | Loss, raw | Loss, aligned |
|---|---|---|---|---|
| body (root in camera frame + body pose) | −1.08 [−1.36, −0.82] | −0.11 [−0.35, +0.14] | +2.27 [+1.70, +2.77] | see `report_stats.json` |
| hands | +0.012 [+0.001, +0.025] | +0.017 [−0.000, +0.038] | +0.15 [+0.10, +0.21] | |
| face (FLAME, expression, head scale) | −0.12 [−0.19, −0.05] | +0.02 [−0.04, +0.07] | +0.13 [+0.01, +0.25] | |
| camera (neck position) | −0.75 [−1.19, −0.31] | −0.23 [−0.53, +0.04] | +2.65 [+1.91, +3.40] | |
| camera: image-plane ray only | −0.83 [−1.24, −0.45] | −0.16 [−0.42, +0.07] | n/a | n/a |
| camera: depth only | +0.08 [−0.23, +0.40] | +0.05 [−0.04, +0.15] | n/a | n/a |

Chart: `phase3_dnt/attribution_all.png` (and `attribution_oracle_ok.png`). Hands and face matter little for whole-image
PSNR (small areas) but are clearly worse locally: S − O face-crop LPIPS +0.044 [+0.023, +0.063], hand-crop LPIPS
+0.057 [+0.043, +0.071]. Temporal LPIPS and flicker of S are at O's level (differences <= 0.002).

**Driver time** (batch 1, this server's shared GPU, median per frame): S 11.4 ms, S+R 14.0 ms, P 16.4 ms.

**Caveats.** O is offline-smoothed and its identity fitted over the whole clip; GUAVA's weights saw undisclosed UBody
clips; masked PSNR over upper-body framings is dominated by the torso; single-part swaps are not additive because body
and camera are coupled.
