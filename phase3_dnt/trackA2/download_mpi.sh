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
  out="$DEST/$(basename "$f")"
  need=$(awk -F'\t' -v d="$dom" -v f="$f" '$1==d && $2==f {print $3}' "$HERE/sizes.tsv")
  have=$( [ -f "$out" ] && stat -c %s "$out" || echo 0 )
  [ -n "$need" ] && [ "$have" = "$need" ] && { echo "$(date '+%T') have $f"; continue; }
  free=$(df -B1 --output=avail /raid | tail -1)
  if [ -n "$need" ] && [ $(( free - (need - have) )) -lt 30000000000 ]; then echo "STOP_DISK before $f (free $((free/1000000000)) GB)"; exit 3; fi
  wget -q --continue --post-file="$TMP/post" -O "$out" "https://download.is.tue.mpg.de/download.php?domain=$dom&resume=1&sfile=$f"
  got=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ -n "$need" ] && [ "$got" != "$need" ]; then echo "SIZE_MISMATCH $f got $got want $need"; else echo "$(date '+%T') ok $f $got"; fi
done < "$LIST"
echo DOWNLOAD_DONE
