#!/usr/bin/env bash
# Laptop timing: the same video with the background worker off and on (step 1, pose), headless, 600 live frames each.
# usage: bash run_timing.sh VIDEO.mp4 [OUT_DIR]      (run from the extracted guava_live directory, guava env active)
set -euo pipefail
VID=${1:?video path}; OUT=${2:-timing_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"
export TORCH_HOME=${TORCH_HOME:-$(pwd)/live_assets/torch}
for W in off pose; do
  python phase3_dnt/live/live_sprint_guavafb.py --input "$VID" --no-display --frames 600 --worker $W --timing-json "$OUT/timing_$W.json" | tail -n 3
done
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1]); R = {w: json.loads((o / f"timing_{w}.json").read_text()) for w in ("off", "pose")}
for w, r in R.items():
    s = r["stages_ms"]; print(f"worker {w:5s}: total p50 {s['total']['p50']:.1f} / p95 {s['total']['p95']:.1f} ms | det {s['det']['p50']:.1f} "
                              f"sprint {s['sprint']['p50']:.1f} render {s['render']['p50']:.1f} | {r['fps_observed']:.1f} FPS | updates {r['worker_updates']}")
(o / "summary.json").write_text(json.dumps({w: {"total": R[w]["stages_ms"]["total"], "fps": R[w]["fps_observed"], "gpu": R[w]["gpu"],
                                                 "worker_updates": R[w]["worker_updates"]} for w in R}, indent=1)); print("->", o / "summary.json")
PY
