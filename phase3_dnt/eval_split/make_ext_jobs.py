"""Job list for the extended benchmark (PREREG "Extended benchmark", 2026-11-09) -> JSONL for gpu_queue.py.

Re-run at any time: it lists every job whose inputs exist (converted clips) in the registered order; gpu_queue.py skips jobs whose
"done" path exists. Order: appearance-update rows on the existing clips; single-image rows on new clips (chunks); GART; then the
per-person trainings (GaussianAvatar, ExAvatar [GT face]) on People-Snapshot / NeuMan, then UBC-Fashion in the seeded priority order.
Single-image chunks are formed only when a dataset's split is final and all its clips are converted (fixed membership)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/eval")
XE = O / "finalbench_ext"; B = HERE.parent / "baselines"; STATE = Path(sys.argv[2] if len(sys.argv) > 2 else "/dev/shm/ubx858_eval/ext_queue")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/ubx858_eval/ext_queue/jobs.jsonl")
G = ("export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/pytorch3d-torch22:/tmp/guava-cuda-torch22:"
     "/raid/ubx858/datasets/eval_assets/pylib:/raid/ubx858/outputs/dnt/guava_shims PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
     "CUDA_VISIBLE_DEVICES={gpu}; cd %s; P=/home/ubx858/miniconda3/envs/guava/bin/python; " % HERE)
CLEAN = "env -u PYTHONPATH -u PYTORCH_CUDA_ALLOC_CONF -u PYTHONNOUSERSITE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"
UPD = (("fb_update100", ""), ("fb_update50", " --max-frac 0.5"), ("fb_update25", " --max-frac 0.25"))
EXISTING = (("avatar_test_v1", "avatar_test_v1.json", "ubody"), ("publicbench_v1", "publicbench_v1.json", None),
            ("publicbench_v2", "publicbench_v2_xhumans.json", None))
CH = 10
jobs = []


def marker(jid): return str(STATE / "done" / (jid.replace("/", "__") + ".ok"))


def add(jid, cmd, done=None, heavy=False):
    d = done or marker(jid)
    if done is None: cmd = f"({cmd}) && mkdir -p {STATE / 'done'} && touch {d}"
    jobs.append({"id": jid, "cmd": cmd, "done": d, "heavy": heavy})


# 1. appearance-update rows on the existing clips (UBody 30, People-Snapshot 4, NeuMan 4, X-Humans 3)
for root, sj, only in EXISTING:
    J = json.loads((HERE / sj).read_text()); ids = [c["id"] for c in J["clips"] if only is None or c["dataset"] == only]
    for name, extra in UPD:
        for i in range(0, len(ids), CH):
            ch = ids[i:i + CH]
            add(f"{name}/{root}/{i // CH}", G + f"$P fusion_gt_row.py --E {O / root} --split {HERE / sj} --clips {','.join(ch)} --name {name} "
                f"--mode mean --prior 2 --thr 0 --cap 0{extra}")

# 2. new clips
S = HERE / "finalbench_ext.json"
J = json.loads(S.read_text()) if S.exists() else {"clips": []}
conv = lambda c: (XE / "base" / c["dataset"] / c["clip"] / "frames.json").exists()                  # noqa: E731
final = {"neuman": True, "peoplesnapshot": any(c["dataset"] == "peoplesnapshot" for c in J["clips"]),
         "ubc": (Path("/dev/shm/ubx858_eval/ext_split.log").exists() and "EXT_CONVERT_DONE" in Path("/dev/shm/ubx858_eval/ext_split.log").read_text())}
by = {}
for c in J["clips"]: by.setdefault(c["dataset"], []).append(c)
for ds in by: by[ds].sort(key=lambda c: (c.get("priority", 0), c["id"]))
ready = {ds: [c for c in cs if conv(c)] for ds, cs in by.items()}
SI = (("fb_gt", G + "$P eval_guava_rows.py --row fb_gt --split {S} --E {E} --clips {C}"),
      ("guava", G + "$P eval_guava_rows.py --row guava --split {S} --E {E} --clips {C}"),
      ("fb_sprint", G + "$P eval_guava_rows.py --row fb_sprint --dwpose-threads 4 --split {S} --E {E} --clips {C}"),
      ("lhm_facefix", "LHM_NAME=lhm_facefix LHM_FLAGS='--clamp-face --face-on-matte' bash %s/lhm/lhm_run.sh render {gpu} finalbench_ext finalbench_ext.json {C}" % B),
      ("idol", "bash %s/idol/idol_run.sh render {gpu} finalbench_ext finalbench_ext.json {C}" % B),
      ("lhmpp", "cd /dev/shm/ubx858_baselines/lhmpp; C=/dev/shm/ubx858_baselines/lhmpp/cache; export CUDA_VISIBLE_DEVICES={gpu} PYTHONDONTWRITEBYTECODE=1 "
                "HF_HOME=$C TORCH_HOME=$C XDG_CACHE_HOME=$C OMP_NUM_THREADS=8; /home/ubx858/miniconda3/envs/lhmpp/bin/python %s/lhmpp/run_lhmpp.py "
                "--E {E} --split {S} --name lhmpp --views 1 --clips {C}" % B))
SI += tuple((n, G + "$P fusion_gt_row.py --E {E} --split {S} --clips {C} --name %s --mode mean --prior 2 --thr 0 --cap 0%s" % (n, x)) for n, x in UPD)
for ds in ("neuman", "peoplesnapshot", "ubc"):
    cs = by.get(ds, [])
    if not cs or not final[ds] or len(ready.get(ds, [])) < len(cs): continue
    for name, tpl in SI:
        for i in range(0, len(cs), CH):
            ch = ",".join(c["id"] for c in cs[i:i + CH])
            add(f"{name}/ext/{ds}/{i // CH}", tpl.replace("{S}", str(S)).replace("{E}", str(XE)).replace("{C}", ch))
# 3. GART (minutes per clip), every converted clip
for ds in ("neuman", "peoplesnapshot", "ubc"):
    for c in ready.get(ds, []):
        add(f"gart/{c['id']}", f"{CLEAN} CUDA_VISIBLE_DEVICES={{gpu}} bash {B}/gart/run_clip.sh {ds} {c['clip']} --E {XE}",
            done=str(XE / "renders/gart" / ds / c["clip"] / "run.json"))
# 4. per-person trainings: PS / NeuMan first, then UBC-Fashion by priority; GaussianAvatar and ExAvatar interleaved
for ds in ("neuman", "peoplesnapshot", "ubc"):
    for c in ready.get(ds, []):
        add(f"gaussianavatar/{c['id']}", f"{CLEAN} GA_FORCE_GPU=1 EVAL_ROOT={XE} /dev/shm/ubx858_baselines/gaussianavatar/env/bin/python "
            f"{B}/gaussianavatar/run_clip.py {ds} {c['clip']} --gpu {{gpu}} --out_root {XE}/renders/gaussianavatar",
            done=str(XE / "renders/gaussianavatar" / ds / c["clip"] / "run.json"), heavy=True)
        add(f"exavatar_gtface/{c['id']}", f"{CLEAN} EXA_FORCE_GPU=1 EVAL_ROOT={XE} bash {B}/exavatar/run_clip_gtface.sh {ds} {c['clip']} {{gpu}}",
            done=str(XE / "renders/exavatar_gtface" / ds / c["clip"] / "run.json"), heavy=True)
# 5. retries: a job that ended with a non-zero exit code (e.g. out of GPU memory when another user's job grew) and whose output is
#    still missing is re-queued under a new id, at most twice; the outputs and settings are unchanged.
QL = STATE / "queue.log"
if QL.exists():
    fails = {}
    for line in QL.read_text().splitlines():
        w = line.split()
        if len(w) >= 7 and w[2] == "END" and w[-1] != "0": b = w[3].split("#")[0]; fails[b] = fails.get(b, 0) + 1
    byid = {j["id"]: j for j in jobs}
    for b, n in fails.items():
        if b in byid and n <= 2 and not Path(byid[b]["done"]).exists():
            jobs.append(dict(byid[b], id=f"{b}#retry{n}"))
OUT.parent.mkdir(parents=True, exist_ok=True)
tmp = OUT.with_suffix(".tmp"); tmp.write_text("".join(json.dumps(j) + "\n" for j in jobs)); tmp.replace(OUT)
print(len(jobs), "jobs;", {ds: f"{len(ready.get(ds, []))}/{len(by.get(ds, []))} converted, final={final[ds]}" for ds in final})
