You are working on my Linux laptop with an RTX 3080 Laptop GPU, 16 GB VRAM.
Complete the short RAGA timing run using the kit on branch `driven_not_tracked`
of https://github.com/Sadia0730/guava. Read
`phase3_dnt/timing/TIMING_ON_LAPTOP.md` before acting.

My existing working package is `/data/GUAVA/guava_live`, usually using the Python
environment `/home/virlab/miniconda3/envs/guava`. Discover the actual installed
paths before changing anything. Get the newest timing code into a separate source
checkout and use `install_into_package.py` to overlay it with timestamped backups.
Reuse the working environments/assets; do not upgrade Torch or change models.

The separately transferred dataset ZIP is
`/data/GUAVA/timing_data/timing_subset_laptop_v2.zip`, with its `.sha256` file.
Verify it and extract without overwriting files. The input root must contain
`manifest.json`, normally `/data/GUAVA/timing_data/timing_subset_laptop_v2`.
The ZIP is not in Git. If it is missing, ask me to transfer it from the server:
`/home/ubx858/guava/phase3_dnt/timing/bundles/timing_subset_laptop_v2.zip`.

Check free space and preflight the data, assets and CUDA device. Prepare a local
environment file from `laptop.env.example.sh`, matching the installed checkpoints
and dependencies. Use the evaluated checkpoints, resolution, precision, fitting
iterations and model settings unchanged. Keep input data/old results read-only.

Run `run_suite.py --methods raga` on **neuman/bike only**, using the default 10
warm-up and 50 timed frames. Include all four RAGA initialization/update rows.
Keep the default cooldown and thermal supervision; preserve failed/stopped logs.
Do not run all 163 clips. Do not start per-person training, other baseline methods
or long quality evaluations without a further request. No GPU job should be left
running after a failure. Fix routine environment/path problems and retry only the
affected job into a fresh result directory, with cooldown.

Collect completed timings with `collect_timing.py`. Report source construction
separately from support updates, then the animation phase breakdown: pose,
segmentation where actually used, preprocessing, deformation, rasterization and
refinement. Updated rows must animate the refreshed avatar. Do not add overlapping
parent/child timings or fill absent stages with zero. Distinguish instrumented
latency reciprocals, independent serial timing and actual replay FPS. Report exact
input counts, acceptance counts, hardware/power/configuration, medians/p95, and all
limitations. Do not invent missing measurements or reuse the old 51/38 FPS claims.

After the short component run, run the optional 120-frame headless cyclic replay
once with workers off and UV on, with cooldown, **only if ArcFace and the existing
live application are available**. Label this experimental replay throughput and
report accepted worker updates; do not claim established 30 FPS adaptive operation.

Return links/paths to raw JSON/logs, phase/summary CSVs, generated LaTeX/Markdown
tables and replay reports. Do not edit manuscript scores, commit results, delete
old files, alter model outputs, or conceal failed runs.
