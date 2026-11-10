"""Exit 0 iff every test render of every listed clip exists: E/renders/<method>/<dataset>/<clip>/<key>.png for the test keys of
E/base/<dataset>/<clip>/frames.json. Used by make_ext_jobs.py so a job that exits 0 without writing all renders (e.g. a wrapper
that swallows a CUDA out-of-memory error) is not counted as done. Usage: python check_renders.py <E> <method> <id,id,...>"""
import json, sys
from pathlib import Path
E, m, ids = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
bad = []
for cid in ids:
    ds, clip = cid.split("/", 1); fj = json.loads((E / "base" / ds / clip / "frames.json").read_text()); rd = E / "renders" / m / ds / clip
    if not all((rd / f"{k}.png").exists() for k in fj["test"]): bad.append(cid)
if bad: print("INCOMPLETE", m, ",".join(bad)); sys.exit(1)
print("COMPLETE", m, len(ids), "clips")
