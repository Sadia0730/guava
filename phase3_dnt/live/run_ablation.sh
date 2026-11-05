#!/usr/bin/env bash
# Live appearance update ablation on one recording (README_LIVE.md section 8): replay it with the update off, then on (same
# frames at the recorded capture times, same SPRINT k = 0 driving), then score the frames the update never used.
#   bash phase3_dnt/live/run_ablation.sh REC_DIR [OUT_DIR] [extra live-app args for both replays, e.g. --uv-prior 4]
# Output: OUT_DIR/ablation.md (table), ablation.json, per_frame.csv, side_by_side.mp4 (default OUT_DIR = REC_DIR/ablation).
set -euo pipefail
REC=$(realpath "${1:?usage: run_ablation.sh REC_DIR [OUT_DIR] [extra args]}"); OUT=$(realpath -m "${2:-$REC/ablation}")
shift $(( $# >= 2 ? 2 : 1 ))
cd "$(dirname "$0")/../.."
for W in off uv; do
  python phase3_dnt/live/live_sprint_guavafb.py --input "$REC" --worker $W --no-display --record "$OUT/$W" --timing-json "$OUT/$W/timing.json" "$@"
done
python phase3_dnt/live/score_ablation.py --session "$REC" --runs off="$OUT/off" uv="$OUT/uv" --out "$OUT" --video
echo "results: $OUT/ablation.md, $OUT/side_by_side.mp4"
