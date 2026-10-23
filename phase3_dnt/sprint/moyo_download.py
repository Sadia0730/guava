"""MOYO downloader (SPRINT Phase 0c). Same endpoint and POST login as the official moyo_toolkit download_data.py, but the
login is read here from ~/.mpi_credentials (MPI_USER / MPI_PASS; never printed, logged or passed on a command line).
  --probe URLFILE...  : POST with stream, read Content-Length only, close (no data); writes <out>/sizes.tsv
  --get   URLFILE...  : download those URLs (resumable skip if the file exists with the expected size)
URL files: the toolkit's moyo/bash/assets/urls/*.txt (or a subset list). Output paths = the sfile= part, under --out."""
import argparse, os, sys, time
from pathlib import Path
import requests

ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--probe", nargs="*", default=[]); ap.add_argument("--get", nargs="*", default=[]); a = ap.parse_args()
cred = {}
for line in Path(os.path.expanduser("~/.mpi_credentials")).read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1); cred[k.strip().replace("export ", "")] = v.strip().strip("'\"")
post = {"username": cred["MPI_USER"], "password": cred["MPI_PASS"]}
s = requests.Session(); a.out.mkdir(parents=True, exist_ok=True)
urls = lambda files: [u.strip() for f in files for u in Path(f).read_text().splitlines() if u.strip()]

def open_stream(u):
    r = s.post(u, data=post, stream=True, allow_redirects=True, timeout=120)
    if r.status_code == 401 or "text/html" in r.headers.get("content-type", ""):
        r.close(); sys.exit(f"login failed or not a file for {u.split('sfile=')[-1]} (status {r.status_code})")
    return r

if a.probe:
    rows = []
    for u in urls(a.probe):
        r = open_stream(u); n = int(r.headers.get("content-length", -1)); r.close()
        rows.append(f"{u.split('sfile=')[-1]}\t{n}"); print(rows[-1], flush=True)
    (a.out / "sizes.tsv").write_text("\n".join(rows) + "\n")
for u in urls(a.get):
    rel = u.split("sfile=")[-1]; dst = a.out / rel; dst.parent.mkdir(parents=True, exist_ok=True)
    r = open_stream(u); n = int(r.headers.get("content-length", -1))
    if dst.exists() and dst.stat().st_size == n:
        r.close(); print("skip", rel, n, flush=True); continue
    t0 = time.time(); tmp = dst.with_suffix(dst.suffix + ".part")
    with open(tmp, "wb") as f:
        for c in r.iter_content(1 << 22):
            f.write(c)
    assert tmp.stat().st_size == n, f"size mismatch {rel}"
    tmp.rename(dst); print("got", rel, n, f"{time.time() - t0:.0f}s", flush=True)
