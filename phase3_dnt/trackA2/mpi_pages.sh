#!/usr/bin/env bash
# Log in to an MPI-IS dataset site and save its download page (for file names / sizes). Credentials come from
# ~/.mpi_credentials (MPI_USER=, MPI_PASS=; mode 600); they are written only to a mode-600 temp file used as POST
# data and deleted on exit, never echoed, logged or put on a command line. Usage: mpi_pages.sh <site> <out.html>
set -uo pipefail
SITE=$1; OUT=$2
urle () { local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~_-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; }
U=$(sed -n 's/^MPI_USER=//p' ~/.mpi_credentials | head -1); P=$(sed -n 's/^MPI_PASS=//p' ~/.mpi_credentials | head -1)
TMP=$(mktemp -d); chmod 700 "$TMP"; trap 'rm -rf "$TMP"' EXIT
printf 'username=%s&password=%s&commit=Log+in' "$(urle "$U")" "$(urle "$P")" > "$TMP/post"; unset U P
wget -q --save-cookies "$TMP/c" --keep-session-cookies --post-file="$TMP/post" -O "$TMP/login.html" "https://$SITE/login.php"
wget -q --load-cookies "$TMP/c" -O "$OUT" "https://$SITE/download.php"
if grep -qi 'name="password"' "$OUT"; then echo "LOGIN_FAILED $SITE"; exit 2; fi
echo "LOGIN_OK $SITE $(grep -o 'sfile=[^"&]*' "$OUT" | wc -l) file links"
