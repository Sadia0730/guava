"""GUAVA-FB validation renders, run beside training (read-only on the run): watches <run>/train_log.jsonl, and for every new
validation copies that step's checkpoints/latest.pt to /dev/shm, renders a fixed subset of the trainer's validation set (same
draw and caps as guava_fb_train.build_val; first --groups groups per track, --targets evenly spaced targets each) and writes
<run>/val_renders/iter_XXXXXX/<track>.jpg (rows = groups; columns = source | [GT | render] per target), plus iter_000000 from
the a20 initialisation. Visual only; it computes no metric and changes nothing in training."""
import argparse, json, math, os, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb")
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="guava_fb_v1"); ap.add_argument("--data-root", default="/dev/shm/ubx858_guava_fb/data")
ap.add_argument("--tracks", default="dress4d,bedlam2,motionx,ubody"); ap.add_argument("--groups", type=int, default=4)
ap.add_argument("--targets", type=int, default=2); ap.add_argument("--poll", type=float, default=30)
ap.add_argument("--val-step", type=int, default=4); ap.add_argument("--val-caps", default="dress4d:30,bedlam2:20,motionx:10")
ap.add_argument("--val-cap", type=int, default=150); ap.add_argument("--once", action="store_true")
ap.add_argument("--cross-view-group", default="subject+outfit")
a = ap.parse_args()
run = O / "runs" / a.run; outd = run / "val_renders"; outd.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE))
import numpy as np, torch, torchvision
import fb_common as F
from fb_common import to_dev
os.chdir(F.ROOT)
dev = torch.device("cuda:0")
from models.UbodyAvatar import Ubody_Gaussian


def log_print(*x): print(time.strftime("%H:%M:%S"), *x, flush=True)


# ----------------------------------------------------------------------------- validation clips only (as the trainer)
tracks = [t for t in a.tracks.split(",") if t]
face_conv, eye_copy = F.load_face_conv(); jobs, meta = [], []
for s in tracks:
    if s == "ubody":
        _, val = F.a20_split()
        for c in val: jobs.append((str(c), face_conv.get(c.name, {}), eye_copy)); meta.append({"source": s, "split": "val", "subject": c.name})
        continue
    for e in json.loads((Path(a.data_root) / s / "index.json").read_text())["clips"]:
        if e.get("split") == "val":
            jobs.append((str(Path(a.data_root) / s / e["name"]), None, True)); meta.append({"source": s, "split": "val", "subject": str(e.get("subject", e["name"])),
                     "pair_group": "/".join(str(e[k]) for k in a.cross_view_group.split("+") if k in e) or str(e.get("subject", e["name"]))})
loaded = F.load_clips_parallel(jobs, workers=16); clips = []
drop = {}
for s in tracks:
    dp = Path(a.data_root) / s / "drop_frames.json"
    if dp.exists(): drop[s] = {k: set(v) for k, v in json.loads(dp.read_text()).items()}
for c, m in zip(loaded, meta):
    dk = drop.get(m["source"], {}).get(Path(c["dir"]).name)
    if dk and "keys" in c: c["keys"] = [k for k in c["keys"] if k not in dk]
    if "error" in c or len(c.get("keys", [])) < 2: continue
    c.update(m); clips.append(c)
cfg = F.get_cfg(); fbd = F.FBData(cfg, clips)

caps = {k: int(v) for k, v in (x.split(":") for x in a.val_caps.split(",") if x)}
VAL = {}
for s in tracks:                                                         # = guava_fb_train.build_val
    cis = [i for i, c in enumerate(clips) if c["source"] == s]; G = []
    if s == "ubody":
        for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::4]]))
    elif s == "dress4d":
        by_subj = {}
        for i in cis: by_subj.setdefault(clips[i].get("pair_group", clips[i]["subject"]), []).append(i)
        for subj, ii in sorted(by_subj.items()):
            ii = sorted(ii, key=lambda i: clips[i]["name"]); take = ii[0]
            v0 = sorted({clips[take]["recs"][k]["view"] for k in clips[take]["keys"]})[0]
            src = (take, next(k for k in clips[take]["keys"] if clips[take]["recs"][k]["view"] == v0))
            tg = [(i, k) for i in ii for k in clips[i]["keys"] if clips[i]["recs"][k]["view"] != v0]
            if not tg: continue
            cap = caps.get(s, a.val_cap); step = max(1, math.ceil(len(tg) / cap)); G.append((src, tg[::step][:cap]))
    else:
        for i in cis: G.append(((i, clips[i]["keys"][0]), [(i, k) for k in clips[i]["keys"][1::a.val_step]][:caps.get(s, a.val_cap)]))
    sub = []
    for src, tg in G[:a.groups * (2 if s == "dress4d" else 1)]:
        j = np.linspace(0, len(tg) - 1, a.targets).round().astype(int) if len(tg) > 1 else [0]
        sub.append((src, [tg[x] for x in sorted(set(j))]))
    VAL[s] = sub
log_print("render subset", {s: [(clips[src[0]]["name"], len(t)) for src, t in G] for s, G in VAL.items()})

_, inf, ren, _ = F.setup_a20(dev); inf.eval(); ren.eval()


@torch.no_grad()
def render_all(tag):
    d = outd / tag; d.mkdir(exist_ok=True)
    for s, G in VAL.items():
        rows = []
        for src, tg in G:
            si = F.collate1(fbd.source_info(*src))
            torch.manual_seed(0); vg, ug, _ = inf(to_dev(si, dev))
            av = Ubody_Gaussian(cfg.MODEL, vg, ug, pruning=True); av.init_ehm(inf.ehm); av.eval()
            tsrc = to_dev(F.collate1(fbd.target_info(*src)), dev); row = [(tsrc["image"] * tsrc["mask"])[0]]
            for ref in tg:
                t = to_dev(F.collate1(fbd.target_info(*ref)), dev)
                torch.manual_seed(0); r = ren(av(t), t["render_cam_params"], bg=0.0)["renders"].clamp(0, 1)
                row += [(t["image"] * t["mask"])[0], r[0]]
            while len(row) < 1 + 2 * a.targets: row.append(torch.zeros_like(row[0]))
            rows += row
        torchvision.utils.save_image(torch.stack(rows), d / f"{s}.jpg", nrow=1 + 2 * a.targets, padding=4, pad_value=1.0)
    (d / "README.txt").write_text("rows = validation groups (source clip / subject); columns = source (masked) | then per target: GT (masked) | render\n")
    log_print("wrote", d)


def val_iters():
    p = run / "train_log.jsonl"
    if not p.exists(): return []
    return sorted({json.loads(l)["iter"] for l in p.read_text().splitlines() if '"split": "val"' in l})


if not (outd / "iter_000000").exists(): render_all("iter_000000")    # a20 initialisation
done = {int(p.name.split("_")[1]) for p in outd.glob("iter_*")}
while True:
    todo = [i for i in val_iters() if i > 0 and i not in done]
    if todo:
        it_ = todo[-1]; ck = run / "checkpoints/latest.pt"
        st = None
        for _ in range(10):                                              # latest.pt is written right after the val log line
            time.sleep(20)
            try: st = torch.load(ck, map_location="cpu", weights_only=False); break
            except Exception as e: log_print("latest.pt not readable yet:", repr(e)[:120])
        if st is None: done.update(todo); continue
        if st["step"] == it_:
            inf.load_state_dict(st["model"]); ren.load_state_dict(st["render_model"]); inf.eval(); ren.eval(); del st
            render_all(f"iter_{it_:06d}")
        else:
            log_print(f"latest.pt is step {st['step']}, not {it_}; skipped"); del st
        done.update(todo)
    if a.once or (run / "checkpoints/latest.pt").exists() is False and (run / "checkpoints/best.pt").exists() and not todo:
        break
    time.sleep(a.poll)
