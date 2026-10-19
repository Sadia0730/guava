#!/usr/bin/env bash
# DNT A2: download MPI-IS files (wget --continue) into DEST. Args: DEST LISTFILE (lines "<domain> <file>").
# Credentials from ~/.mpi_credentials (mode 600) -> mode-600 temp POST-data file, deleted on exit; never echoed/logged.
# Stops before any file that would leave /raid with < 30 GB free (size from the HEAD probe in sizes.tsv).
set -uo pipefail
DEST=$1; LIST=$2; HERE=$(cd "$(dirname "$0")" && pwd)
urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~_-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; }
U=$(sed -n 's/^MPI_USER=//p' ~/.mpi_credentials | head -1); P=$(sed -n 's/^MPI_PASS=//p' ~/.mpi_credentials | head -1)
TMP=$(mktemp -d); chmod 700 "$TMP"; trap 'rm -rf "$TMP"' EXIT
printf 'username=%s&password=%s' "$(urle "$U")" "$(urle "$P")" > "$TMP/post"; unset U P
mkdir -p "$DEST"
while read -r dom f; do
  # KEEP_DIRS (env, default 0): keep that many parent folders of the remote path (ARCTIC reuses file names across
  # subjects: s01/box_grab_01.zip vs s02/box_grab_01.zip -- saving by basename appended them onto each other).
  if [ "${KEEP_DIRS:-0}" -gt 0 ]; then out="$DEST/$(echo "$f" | awk -F/ -v k="${KEEP_DIRS}" '{s=$NF; for(i=1;i<=k;i++) s=$(NF-i)"/"s; print s}')"; else out="$DEST/$(basename "$f")"; fi
  mkdir -p "$(dirname "$out")"
  need=$(awk -F'\t' -v d="$dom" -v f="$f" '$1==d && $2==f {print $3}' "$HERE/sizes.tsv")
  have=$( [ -f "$out" ] && stat -c %s "$out" || echo 0 )
  [ -n "$need" ] && [ "$have" = "$need" ] && { echo "$(date '+%T') have $f"; continue; }
  free=$(df -B1 --output=avail /raid | tail -1)
  if [ -n "$need" ] && [ $(( free - (need - have) )) -lt 30000000000 ]; then echo "STOP_DISK before $f (free $((free/1000000000)) GB)"; exit 3; fi
  if [ -z "$need" ] && [ "$free" -lt 40000000000 ]; then echo "STOP_DISK before $f (size unknown, free $((free/1000000000)) GB < 40)"; exit 3; fi
  wget -q --continue --post-file="$TMP/post" -O "$out" "https://download.is.tue.mpg.de/download.php?domain=$dom&resume=1&sfile=$f"
  got=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ -n "$need" ] && [ "$got" != "$need" ]; then echo "SIZE_MISMATCH $f got $got want $need"; else echo "$(date '+%T') ok $f $got"; fi
done < "$LIST"
echo DOWNLOAD_DONE
