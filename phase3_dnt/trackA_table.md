| Model | COCO PCK @0.05 / @0.1 | LSP-Ext PCK @0.05 / @0.1 | 3DPW test MPJPE / PA (author) | 3DPW test MPJPE / PA (standard) | EHF PA-PVE all / PVE all / hands / face / body PA-MPJPE |
|---|---|---|---|---|---|
| PEAR (released ViT-H) | 0.790 / 0.919 | 0.566 / 0.817 | 73.78 / 46.73 | 73.56 / 45.65 | 47.9 / 66.6 / 11.0 / 8.7 / 49.2 |
| L-real (PEAR-distilled) | 0.772 / 0.921 | 0.484 / 0.755 | 74.51 / 48.75 | 75.83 / 48.46 | 50.6 / 83.4 / 14.8 / 6.4 / 60.0 |
| L-real + refiner v2b | 0.759 / 0.919 | 0.476 / 0.757 | 74.24 / 48.24 | — | 45.2 / 81.8 / 14.8 / 6.3 / 50.2 |
| **Track A best.pt, live (PREREG-selected)** | 0.580 / 0.846 | 0.355 / 0.649 | 86.66 / 59.61 | 81.96 / 53.74 | 84.8 / 119.5 / 13.9 / 64.5 / 58.4 |
| Track A best.pt, EMA (step 82k) | 0.587 / 0.851 | 0.373 / 0.663 | 86.51 / 60.12 | 81.65 / 54.20 | 83.7 / 116.1 / 13.8 / 66.1 / 57.4 |
| Track A latest.pt, EMA (step 128k; descriptive) | 0.607 / 0.864 | 0.388 / 0.677 | 86.50 / 60.00 | 82.14 / 54.26 | 78.3 / 108.3 / 13.7 / 70.5 / 55.9 |
| *diag: L-real, unsupervised outputs neutral* | 0.771 / 0.921 | 0.483 / 0.755 | 74.47 / 48.73 | — | 50.3 / 83.5 / 14.8 / 5.4 / 59.8 |
| *diag: Track A best.pt live, unsupervised outputs neutral* | 0.743 / 0.915 | 0.391 / 0.695 | 80.92 / 50.66 | — | 47.5 / 93.7 / 13.9 / 5.5 / 51.6 |

L-real re-evaluated with the current evaluator (SHA 9838287b): 3DPW test 74.51 / 48.75, COCO @0.05 0.7724; identical to the stored row: True.

| Weights | ΔPA-MPJPE (≤ +1.0) | ΔMPJPE (≤ +2.0) | ΔCOCO PCK@0.05 (≥ −0.01) | Circularity answered |
|---|---|---|---|---|
| **Track A best.pt, live (PREREG-selected)** | +10.87 ✗ | +12.15 ✗ | -0.1920 ✗ | **FAIL** |
| Track A best.pt, EMA (step 82k) | +11.38 ✗ | +12.00 ✗ | -0.1858 ✗ | **FAIL** |
| Track A latest.pt, EMA (step 128k; descriptive) | +11.25 ✗ | +11.99 ✗ | -0.1652 ✗ | **FAIL** |
| *diag (not gated): neutral vs neutral* | +1.92 | +6.45 | -0.0282 | *would still fail* |
