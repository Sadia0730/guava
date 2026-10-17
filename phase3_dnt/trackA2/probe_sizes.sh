#!/usr/bin/env bash
# DNT Track A2: measure BEDLAM v1 download sizes WITHOUT downloading (HTTP HEAD-equivalent via wget --spider).
# Run it yourself:  bash phase3_dnt/trackA2/probe_sizes.sh
# Credentials are read from the terminal, kept in a mode-600 temp file only for this run, and deleted on exit
# (they never appear on a command line or in a log). File list: bedlam_v1_files.txt (from BEDLAM's fetch_training_data.sh).
set -uo pipefail
cd "$(dirname "$0")"
urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; }
read -p "BEDLAM username: " u; read -s -p "BEDLAM password: " p; echo
PF=$(mktemp); chmod 600 "$PF"; trap 'rm -f "$PF"' EXIT
printf 'username=%s&password=%s' "$(urle "$u")" "$(urle "$p")" > "$PF"; unset p
OUT=sizes_bedlam_v1.tsv; : > $OUT; total=0
while read -r f; do
  len=$(wget --spider --server-response --post-file="$PF" --no-check-certificate \
        "https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=$f" 2>&1 | grep -i "Content-Length" | tail -1 | awk '{print $2}' | tr -d '\r')
  echo -e "$f\t${len:-unknown}" | tee -a $OUT
  [[ "$len" =~ ^[0-9]+$ ]] && total=$(( total + len ))
done < bedlam_v1_files.txt
echo "TOTAL_BYTES $total ($(( total / 1000000000 )) GB)" | tee -a $OUT
