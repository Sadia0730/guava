"""Job list for PREREG extended benchmark amendment 6 -> JSONL for gpu_queue.py (re-run at any time; finished jobs are skipped).
  1. fb_sprint_update100 (base model B28k, SPRINT-pose update): chunks of 10 clips per E root (X-Humans: one clip per job).
  2. per-subject fine-tune, one job per clip: subject_ft.py (--steps N, weights -> /dev/shm, deleted on exit) -> ft_gt
     (eval_guava_rows --row fb_gt) -> ft_update100 (fusion_gt_row, fb_update100's rule) -> ft_sprint_update100
     (eval_guava_rows --row fb_sprint_update). Fine-tune log (time, settings) -> O/reports/subject_ft/<root>/<clip>.json.
A job is done only when check_renders.py finds every test render of its rows (some failures exit 0). Retries: a job that ended
with its output missing is re-queued under "<id>#retryN" (at most 2), replacing the original; settings and outputs unchanged.
Usage: python make_a6_jobs.py --steps N [--out /dev/shm/ubx858_eval/a6_queue/jobs.jsonl]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--out", default="/dev/shm/ubx858_eval/a6_queue/jobs.jsonl"); a = ap.parse_args()
OUT = Path(a.out); STATE = OUT.parent
G = ("export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:"
     "/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
     "CUDA_VISIBLE_DEVICES={gpu}; cd %s; P=/home/ubx858/miniconda3/envs/guava/bin/python; " % HERE)
ROOTS = (("avatar_test_v1", "avatar_test_v1.json", "ubody"), ("publicbench_v1", "publicbench_v1.json", None),
         ("publicbench_v2", "publicbench_v2_xhumans.json", None), ("finalbench_ext", "finalbench_ext.json", None))
CHECK = "/usr/bin/python3 %s" % (HERE / "check_renders.py")
jobs = []


def add(jid, cmd, done, heavy=False):
    jobs.append({"id": jid, "cmd": f"({cmd}); {done[1]} && mkdir -p {STATE / 'done'} && touch {done[0]}", "done": done[0], "heavy": heavy})


def marker(jid): return str(STATE / "done" / (jid.replace("/", "__") + ".ok"))


sp = []; ft = []
for root, sj, only in ROOTS:
    E = O / "eval" / root; S = HERE / sj
    ids = [c["id"] for c in json.loads(S.read_text())["clips"] if only is None or c["dataset"] == only]
    groups = [[c] for c in ids if c.startswith("xhumans/")] + [ch for ch in (lambda xs: [xs[i:i + 10] for i in range(0, len(xs), 10)])([c for c in ids if not c.startswith("xhumans/")]) if ch]
    for g in groups:
        C = ",".join(g); jid = f"fb_sprint_update100/{root}/{g[0]}"
        sp.append((jid, G + f"$P eval_guava_rows.py --row fb_sprint_update --dwpose-threads 4 --split {S} --E {E} --clips {C}",
                   (marker(jid), f"{CHECK} {E} fb_sprint_update100 {C}")))
    for cid in ids:
        n = cid.replace("/", "__"); W = f"/dev/shm/ubx858_eval/ft_w/{root}__{n}.pt"; jid = f"ft/{root}/{cid}"
        steps = (f"trap 'rm -f {W}' EXIT; {G}"
                 f"$P subject_ft.py --E {E} --clip {cid} --steps {a.steps} --out {W} --log {O / 'reports/subject_ft' / root / (n + '.json')} && "
                 f"$P eval_guava_rows.py --row fb_gt --fb-ckpt {W} --name ft_gt --split {S} --E {E} --clips {cid} && "
                 f"$P fusion_gt_row.py --E {E} --split {S} --clips {cid} --name ft_update100 --mode mean --prior 2 --thr 0 --cap 0 --fb-ckpt {W} && "
                 f"$P eval_guava_rows.py --row fb_sprint_update --fb-ckpt {W} --name ft_sprint_update100 --dwpose-threads 4 --split {S} --E {E} --clips {cid}")
        chk = " && ".join(f"{CHECK} {E} {r} {cid}" for r in ("ft_gt", "ft_update100", "ft_sprint_update100"))
        ft.append((jid, steps, (marker(jid), chk)))
for jid, cmd, done in sp: add(jid, cmd, done)
for jid, cmd, done in ft: add(jid, cmd, done, heavy=True)
# retries (as make_ext_jobs.py): count ended runs per base id in the queue log
QL = STATE / "queue.log"; fails = {}
if QL.exists():
    for line in QL.read_text().splitlines():
        w = line.split()
        if len(w) >= 7 and w[2] == "END": b = w[3].split("#")[0]; fails[b] = fails.get(b, 0) + 1
byid = {j["id"]: i for i, j in enumerate(jobs)}
for b, nf in fails.items():
    if b in byid and nf <= 2 and not Path(jobs[byid[b]]["done"]).exists(): jobs[byid[b]] = dict(jobs[byid[b]], id=f"{b}#retry{nf}")
OUT.parent.mkdir(parents=True, exist_ok=True); tmp = OUT.with_suffix(".tmp")
tmp.write_text("".join(json.dumps(j) + "\n" for j in jobs)); tmp.replace(OUT)
print(len(jobs), "jobs:", len(sp), "SPRINT-update chunks,", len(ft), "fine-tune clips;", sum(Path(j["done"]).exists() for j in jobs), "done")
