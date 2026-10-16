# Part-wise queries — evaluation of the three fine-tuning runs (2026-10-16, measured)

Runs (approved 2026-10-14): fine-tuned from L-real `best.pt`, 32k steps, lr 1e-4 (backbone x0.1), seed 20261009, L-real
data mix. Evaluated: each run's `best.pt` (3DPW-val PA-MPJPE; all three at step 12,000). Output:
`/raid/ubx858/outputs/dnt/parts_eval/`. Evaluator changes are loader-only; regression before use: L-real EHF per frame
identical (max diff 0.0), COCO PCK identical.

| Model | COCO PCK @0.05 / @0.1 | LSP PCK @0.05 / @0.1 | 3DPW test MPJPE / PA (author) | 3DPW test MPJPE / PA (standard) | EHF hand PA-PVE | EHF PA-PVE all | UBody hand PA-PVE* | UBody PA-PVE all* |
|---|---|---|---|---|---|---|---|---|
| L-real (reference) | 0.772 / 0.921 | 0.484 / 0.755 | 74.51 / 48.75 | 75.83 / 48.46 | 14.81 | 50.57 | 10.16 | 71.51 |
| (i) 1-query control | 0.762 / 0.918 | 0.478 / 0.753 | 73.42 / 48.38 | 74.17 / 48.07 | 14.36 | 50.02 | 10.18 | 72.45 |
| (ii) 5 queries | 0.757 / 0.917 | 0.476 / 0.754 | 73.52 / 48.39 | 74.15 / 47.96 | 14.38 | 49.78 | 10.15 | 72.39 |
| (iii) 5 queries + hand crops, full-res crops | 0.756 / 0.917 | 0.478 / 0.755 | 73.32 / 48.20 | 73.91 / 47.74 | 14.35 | 49.91 | 10.14 | 73.17 |
| (iv) = (iii), oracle crops at the true hands | identical to (iii) | | | | 14.35 | 49.91 | 10.14 | 73.17 |
| (iii), crops from the 256 px input | identical to (iii) | | | | 14.35 | 49.91 | 10.14 | 73.17 |

\* UBody: provisional locked-head SMPL-X stand-in for the v1.1 pseudo-GT meshes, the same for every row (approved).

**Pre-registered target** ((ii)/(iii) vs (i): hand PA-PVE better by >= 1.5 mm on EHF AND UBody, 3DPW not worse by
> 0.5 mm on both metrics): **not met.** Hand PA-PVE changes vs (i): EHF +0.02 / −0.01 mm, UBody −0.03 / −0.04 mm.
3DPW is not worse (author's protocol: (ii) +0.10 / +0.01, (iii) −0.10 / −0.18 mm). Fine-tuning itself (i vs L-real)
helps 3DPW by 1.1 mm MPJPE and EHF hands by 0.45 mm.

**Why — diagnosed with (iv) and the attention maps, before any fix:**
1. **The hand-crop branch is switched off.** Full-resolution, oracle and 256 px-input crops give bit-identical metrics,
   so the result does not depend on the crop content at all: the gate is tanh(g) with g = 5e-5 (left) / 6e-4 (right)
   after 12,000 steps. It started at 0 (exact identity at init) and never opened. So (iv) does NOT tell us whether
   better hand pixels would help: it is "bad architecture / optimisation", not "no information".
2. **The five queries never separated.** Per-query cross-attention maps (10 EHF images, last 2 decoder layers;
   `parts_eval/FT_5q*/attention/attention_grid.jpg`) are identical: pairwise cosine 0.999–1.000, all diffuse (normalised
   entropy 0.92–0.93). The query embeddings moved by at most 0.096 (max abs) on a vector of norm 11.1 (< 1%) in 12,000
   steps. With identical initial queries, the shared decoder and self-attention keep them symmetric; the only
   asymmetry is that each output head reads a different query, and at lr 1e-4 with weight decay 0.05 that signal moves
   them negligibly.
Both are consequences of the "exact at initialisation" design choice combined with a fine-tuning learning rate.

**Fix to propose (not run):** (a) break the query symmetry at initialisation (small distinct perturbations or learned
part embeddings, with a 10x learning rate and no weight decay on the query and new hand-crop parameters); (b) open the
hand-crop path: gate initialised at g = 0.1 (or a zero-initialised output projection instead of a tanh gate), no weight
decay on it, 10x learning rate; (c) a short 2k-step diagnostic first, gated on the attention maps actually separating
and the oracle-vs-input crop difference becoming non-zero, before another 32k-step run.
