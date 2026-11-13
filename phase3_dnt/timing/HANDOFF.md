# Laptop timing handoff, 2026-11-13

Start with [TIMING_ON_LAPTOP.md](TIMING_ON_LAPTOP.md). The copy-paste laptop agent
request is [LAPTOP_AGENT_PROMPT.md](LAPTOP_AGENT_PROMPT.md).

Changes are limited to `phase3_dnt/timing/`. Existing manuscript files, saved
experiment outputs, datasets, checkpoints, submodules and unrelated work were not
edited or staged. The initial timing scripts were backed up before modification in
the existing paper backup directory (`before_laptop_timing_20261113T102244392476Z`).
No GPU jobs or experiments were launched here. No cleanup was performed.

The input bundle is `bundles/timing_subset_laptop_v2.zip` (ignored by Git),
128,613,078 bytes, SHA-256
`9a551c8c68e56dd958a187a3974cc1c464479e53dffb37b7ba4a29246f59b38c`.
[DATA_SUBSET.json](DATA_SUBSET.json) records clip selection and verification.
Transfer the ZIP and checksum separately; they contain inputs, no model weights.

CPU validation passed: six timing/protocol tests, syntax and `--help` for all
Python CLIs, shell syntax, ZIP CRC/checksum, all 484 manifest file hashes, disjoint
support/test splits, equality of 95 camera/body parameter arrays and all 338
compact LMDB values against the evaluated inputs. GPU extension/API compatibility
and actual laptop latency remain untested. The first laptop run must retain any
failure logs; do not report placeholders or shorten evaluated model settings.

RAGA changes correct the earlier unfinished timer: support comes from reconstruction
frames; updated rows render the refreshed avatar; supplied inputs avoid unused
segmentation; source fitting, support fitting, and raw target driving stay separate.
Phase synchronization and independent serial timing are distinguished from the
actual application's optional worker-off/UV replay throughput. Offline fusion and
experimental worker policies are explicitly separate. Baseline adapters preserve
their evaluated algorithms and record their differing output boundaries.

Default orchestration runs four RAGA rows on one NeuMan clip in separate processes,
with cooldown and thermal supervision. Other methods are opt-in. Per-person methods
require `--allow-training`; their full reconstruction schedules are not made short
by reducing the animation frame count. Fine-tuning remains the evaluated 500-step
recipe on the full 83-frame NeuMan reconstruction partition. Training checkpoints
and logs are retained by the provided wrappers.

Remaining work on the laptop: resolve local paths/assets, verify CUDA extensions,
run the short profiles, inspect temperatures/clocks and raw JSON, and collect the
measured table. Optional replay is a cyclic headless diagnostic, not established
sustained 30 FPS adaptive operation. No historical unverified FPS claim is inserted
into generated results. Manuscript integration is a separate subsequent task.
