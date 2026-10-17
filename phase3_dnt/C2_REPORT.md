# C2 — baselines for closing the live-driving gap (measured, 2026-10-17)

Registered in `PREREG.md` ("C2") before the full run. Code: `phase3_dnt/c1_live_gap.py --c2` (C1's split, avatars, future
frames >= 60, masks, metrics; C1's cached driver outputs, read-only); statistics `phase3_dnt/c2_report.py`. Run
`/raid/ubx858/outputs/dnt/c2/baselines/` (RUNLOG 19), GPU 6, 33 identities.

C2 baselines, 33 identities (per-identity means; 95 % bootstrap CI over identities)

| Variant | PSNR | LPIPS | kp dist | face LPIPS | hand LPIPS | t-LPIPS | flicker |
|---|---|---|---|---|---|---|---|
| O | 21.49 | 0.1146 | 0.121 | 0.1972 | 0.2383 | 0.0181 | -0.0004 |
| S | 20.01 | 0.1295 | 0.113 | 0.2410 | 0.2950 | 0.0197 | +0.0001 |
| S:B1 | 20.66 | 0.1239 | 0.087 | 0.2188 | 0.2830 | 0.0267 | +0.0029 |
| S:B2 | 20.21 | 0.1279 | 0.079 | 0.2265 | 0.2923 | 0.0225 | +0.0015 |
| S:B2+OE | 20.11 | 0.1290 | 0.092 | 0.2320 | 0.2950 | 0.0193 | +0.0004 |

| Baseline | PSNR gap closed | LPIPS gap closed | B − S PSNR (dB) | O − B PSNR (dB) |
|---|---|---|---|---|
| S:B1 | +44 % [+33, +64] | +38 % [+27, +55] | +0.65 [+0.43, +0.89] | +0.83 [+0.34, +1.28] |
| S:B2 | +14 % [-18, +37] | +11 % [-21, +36] | +0.20 [-0.20, +0.59] | +1.28 [+0.72, +1.81] |
| S:B2+OE | +7 % [-30, +29] | +3 % [-32, +26] | +0.10 [-0.29, +0.49] | +1.38 [+0.85, +1.90] |

## Reading (registered rule: "useful" if the PSNR gap-closed CI excludes 0, "strong" if the point estimate >= 50 %)
- **B1 (render-space alignment) is useful, not strong: 44 % [33, 64] of the PSNR gap, 38 % of the LPIPS gap.** It is
  the deployable version of C1's alignment (keypoints of the *unmasked* live frame); C1's masked-frame version, computed
  from C1's stored data before registration, was 47 %. Its cost is temporal: per-frame warps add jitter (temporal LPIPS
  0.0197 -> 0.0267, flicker +0.0001 -> +0.0029).
- **B2 (driver-side camera re-fit) is not useful: 14 % [−18, 37].** It gives the best keypoint distance of all variants
  (0.079, better than O's 0.121) yet barely changes PSNR / LPIPS: matching 2D keypoints by moving the camera does not
  fix a body that is posed / scaled differently (C1: S's camera and body are coupled; swapping either alone hurts).
  Keypoint distance is therefore a poor proxy for render quality here.
- **B2 + One-Euro is not useful: 7 % [−30, 29].** The live demo's defaults (registered, not tuned) lag the re-fitted
  placement by ~2 frames; it is smoother than B2 (temporal LPIPS 0.0193, at S's level) but loses most of B2's small gain.
- **Remaining gap after the best baseline:** O − B1 = +0.83 dB [0.34, 1.28]: B1 recovers placement, not pose / face / hands
  (face-crop LPIPS 0.241 -> 0.219 vs O 0.197; hand-crop LPIPS 0.295 -> 0.283 vs O 0.238).
- **Not measured:** the added latency of B1 (two ViTPose+ passes + a warp) and B2 (one ViTPose+ pass + an EHM forward + a
  3x3 solve) was not timed in this run.

**Next baselines suggested (not run):** B1 with its similarity parameters temporally filtered (removes B1's jitter; the
One-Euro lesson says: tune on held-out non-test clips, not on these), and B1 on top of a better driver (C2-T below).

## C2 training-data tracking proposal (not started; needs approval)
Goal: give the live student O-quality labels on in-domain video, i.e. train it toward what closes the gap (pose + camera
jointly, face, hands) instead of correcting it afterwards. O = EHM-Tracker, as in C1.
- **Data:** UBody training clips (the student's own training manifest; disjoint from the C1 test identities by the
  intra-scene split): 3,083 clips, 504,402 frames (median 82, p90 311 frames per clip).
- **Measured cost (C1):** 6 pinned workers (37 cores each) tracked ~8,950 frames in 5.9 h, i.e. ~0.42 frames/s in total
  (CPU-bound ONNX; one GPU for the optimisation). All 504k frames would take ~330 h (~14 days): too much.
- **Proposal, phase 1:** 300 clips (stratified over the 15 scenes, <= 300 frames each; ~50k frames) -> ~33 h of tracking
  on 6 workers; disk ~55 GB at C1's 0.18 GB per clip (most of it the frame LMDB, which can be dropped after the fits are
  read). Quality gate per clip as C1's sensitivity rule (O's own keypoint distance <= 0.10 torso; failed clips dropped).
- **Use:** EHM fits as pseudo-GT for root / body / hands / camera / FLAME expression + jaw on those frames ("C2-T"
  fine-tune from the F-a student), evaluated on the C1 split with the C1 / C2 protocol, B1 on top as the reference.
  The FLAME fits are also the real face labels F-a currently lacks (UBody's SMPL-X expressions convert with 37 % loss).
- **Decisions needed:** the clip count / budget (CPU for ~1.5 days), and whether to keep the frame LMDBs.
