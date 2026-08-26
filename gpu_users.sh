#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

trim() {
  sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
while IFS=, read -r gpu used total; do
  gpu=$(printf '%s' "$gpu" | trim)
  used=$(printf '%s' "$used" | trim)
  total=$(printf '%s' "$total" | trim)

  if [[ -z "$total" || "$total" == "0" ]]; then
    pct=0
  else
    pct=$((used * 100 / total))
  fi

  users=$(
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
    while read -r pid; do
      pid=$(printf '%s' "$pid" | trim)
      [[ -n "$pid" ]] && ps -o user= -p "$pid" 2>/dev/null || true
    done |
    sort -u |
    paste -sd ' ' -
  )

  echo "GPU $gpu ${users:-free} ${pct}%"
done
