# Avatar experiments (definitions from the user, 2026-10-19; recorded verbatim in substance so they are not lost)

Both experiments use the C1 setup (GUAVA, the C1 test split of 36 identities, future frames >= 60, masked PSNR / LPIPS,
the C1 / C2 protocol in `phase3_dnt/c1_live_gap.py`) unless stated otherwise. Each gets its own PREREG entry before it
runs; smoke test first, then ask before the full run.

## E1 — "Avatar trained on live poses" (training-time; GUAVA fine-tune)
- **What is trained:** GUAVA's refiner / StyleUNet and the Gaussian heads (the avatar network), fine-tuned on the 269
  EHM-tracked UBody TRAINING clips of C2-T phase 1 (`/raid/ubx858/outputs/dnt/c2t/tracked/`, incl. their frame LMDBs;
  the C1 test identities are disjoint by the intra-scene split).
- **Driving signal per target frame:** student (live) pose with p 0.5; tracker pose with p 0.3; tracker pose + a student
  error residual (a student-minus-tracker residual sampled from other frames) with p 0.2.
- **Loss:** GUAVA's image losses vs the real frame + L1 to the ORIGINAL GUAVA render under tracker poses (anti-regression).
- **Arms (same everything else):** none (tracker poses only) / i.i.d. noise on tracker poses / student error residuals /
  student poses.
- **Evaluation:** C1 test split, the student driving (S), alone and with the B1 render warp (C2's best baseline).
- **PREREG target:** closes >= 40 % of the raw S-vs-O gap; beats B1; oracle-driven (O) LPIPS worse by <= 0.002 (no
  regression when driven by the tracker).

## E2 — "Avatar improves during a session" (test-time only; gradient-free UV-feature fusion)
- **On:** the C1 test clips, at test time only (no training).
- **Fusion:** accumulate GUAVA's per-frame UV features weighted by visibility x facing x confidence, with frame 0 as the
  prior; outlier rejection; a frame is accepted only if it adds UV coverage or shows a new pose (>= 15 deg from the
  accepted ones). Re-run the UV decoder when coverage grew by >= 2 %. Stop when coverage >= 0.9 AND the held-out LPIPS
  gain is < 0.002 over 3 consecutive refreshes.
- **Evaluation:** future frames never used for fusion, vs the static (frame-0) avatar.
- **PREREG target:** >= 5 % relative LPIPS gain; no identity drift.
- **Smoke test:** 1 clip first.
