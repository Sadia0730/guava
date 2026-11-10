"""Multi-GPU job queue for the extended benchmark (PREREG "Extended benchmark", 2026-11-09).

Jobs: a JSONL file, one job per line {"id", "cmd", "done", "heavy" (bool), "after" (optional id)}; "cmd" contains {gpu} (the
physical GPU index); "done" is a path whose existence means the job is finished (skipped, never re-run). The file is re-read every
loop, so jobs can be appended while the queue runs. Jobs start in file order (the file is written in the registered priority order).
GPU rule (baseline contract): a GPU is used only if OTHER users' processes hold < 2 GB on it (checked before every launch); at most
--max-jobs of our jobs per GPU, of which at most --max-heavy per-person trainings. Stop launching: touch <state>/STOP.
Usage: python gpu_queue.py --jobs jobs.jsonl --state /dev/shm/ubx858_eval/ext_queue [--gpus 0,1,...,7]"""
from __future__ import annotations

import argparse
import json
import os
import pwd
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True); ap.add_argument("--state", required=True)
ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7"); ap.add_argument("--max-jobs", type=int, default=3)
ap.add_argument("--max-heavy", type=int, default=2); ap.add_argument("--other-mb", type=int, default=2048)
ap.add_argument("--min-free-mb", type=int, default=12000); ap.add_argument("--poll", type=int, default=60)
a = ap.parse_args()
S = Path(a.state); (S / "logs").mkdir(parents=True, exist_ok=True)
ME = pwd.getpwuid(os.getuid()).pw_name
GPUS = [int(x) for x in a.gpus.split(",")]


def log(*x):
    with open(S / "queue.log", "a") as f: f.write(time.strftime("[%F %T] ") + " ".join(map(str, x)) + "\n")


def gpu_state():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    g = {}
    for line in q.strip().splitlines():
        i, u, used, tot = [x.strip() for x in line.split(",")]; g[u] = {"index": int(i), "used": int(used), "total": int(tot), "other": 0}
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in q.strip().splitlines():
        if not line.strip(): continue
        pid, u, mb = [x.strip() for x in line.split(",")]
        try: owner = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except Exception: owner = "?"                                                               # noqa: E722 (not visible = not ours)
        if owner != ME and u in g: g[u]["other"] += int(mb) if mb.isdigit() else 0
    return {v["index"]: v for v in g.values()}


running = {}                                                                                        # id -> (Popen, gpu, heavy)
done_ids = set()
log("QUEUE_START", vars(a))
while True:
    for jid, (p, g, h) in list(running.items()):
        if p.poll() is not None:
            log("END", jid, "gpu", g, "rc", p.returncode); running.pop(jid); done_ids.add(jid)
    jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    todo = [j for j in jobs if j["id"] not in running and j["id"] not in done_ids and not Path(j["done"]).exists()
            and (not j.get("after") or j["after"] in done_ids or Path(next((x["done"] for x in jobs if x["id"] == j["after"]), "/nonexistent")).exists())]
    (S / "status.json").write_text(json.dumps({"time": time.strftime("%F %T"), "running": {k: v[1] for k, v in running.items()},
                                               "todo": len(todo), "jobs": len(jobs), "finished_this_run": len(done_ids)}, indent=1))
    if not todo and not running: log("QUEUE_EMPTY"); break
    if not (S / "STOP").exists() and todo:
        st = gpu_state()
        for g in GPUS:
            if not todo: break
            s = st.get(g)
            if s is None or s["other"] >= a.other_mb: continue
            mine = [v for v in running.values() if v[1] == g]
            if len(mine) >= a.max_jobs or s["total"] - s["used"] < a.min_free_mb: continue
            nheavy = sum(1 for v in mine if v[2])
            j = next((j for j in todo if not (j.get("heavy") and nheavy >= a.max_heavy)), None)
            if j is None: continue
            todo.remove(j)
            lf = open(S / "logs" / (j["id"].replace("/", "__") + ".log"), "a")
            p = subprocess.Popen(["bash", "-c", j["cmd"].format(gpu=g)], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            running[j["id"]] = (p, g, bool(j.get("heavy"))); log("START", j["id"], "gpu", g, "other_mb", s["other"])
            time.sleep(20); st = gpu_state()                                                          # let it allocate before the next check
    time.sleep(a.poll)
log("QUEUE_END")
