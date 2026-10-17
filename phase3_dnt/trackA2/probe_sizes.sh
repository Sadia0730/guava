#!/usr/bin/env bash
# DNT Track A2: exact download sizes WITHOUT downloading: a 1-byte range request per file (server resume support),
# total size read from Content-Range. Credentials: ~/.mpi_credentials (mode 600), written only to a mode-600 temp
# POST-data file (deleted on exit); never echoed, logged or on a command line. List: probe_files.txt (<domain> <file>).
set -uo pipefail
cd "$(dirname "$0")"
urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~_-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; }
U=$(sed -n 's/^MPI_USER=//p' ~/.mpi_credentials | head -1); P=$(sed -n 's/^MPI_PASS=//p' ~/.mpi_credentials | head -1)
TMP=$(mktemp -d); chmod 700 "$TMP"; trap 'rm -rf "$TMP"' EXIT
printf 'username=%s&password=%s' "$(urle "$U")" "$(urle "$P")" > "$TMP/post"; unset U P
OUT=sizes.tsv; printf 'domain\tfile\tbytes\thttp\n' > $OUT
while read -r dom f; do
  # step 1: POST (no redirect following) -> session cookie + Location of download2.php; step 2: HEAD on it
  loc=$(curl -s --max-time 30 -c "$TMP/c" -b "$TMP/c" --data-binary @"$TMP/post" -D - -o /dev/null \
        "https://download.is.tue.mpg.de/download.php?domain=$dom&resume=1&sfile=$f" | grep -i '^location' | awk '{print $2}' | tr -d '\r')
  hdr=$(curl -s -I --max-time 30 -b "$TMP/c" "https://download.is.tue.mpg.de/${loc#/}")
  code=$(echo "$hdr" | grep -m1 -o 'HTTP/[0-9.]* [0-9]*' | awk '{print $2}')
  tot=$(echo "$hdr" | grep -i '^content-length' | tail -1 | awk '{print $2}' | tr -d '\r')
  printf '%s\t%s\t%s\t%s\n' "$dom" "$f" "${tot:-?}" "${code:-?}" >> $OUT
done < probe_files.txt
echo PROBE_DONE
