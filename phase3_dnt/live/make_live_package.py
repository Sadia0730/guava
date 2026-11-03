"""Build the laptop package for live_sprint_guavafb.py: one tar, extracted into a fresh directory and run with the laptop's
existing `guava` conda env (PyTorch 2.2 + pytorch3d + diff-gaussian-rasterization-32, as LIVE_PEAR_SETUP.md).
Contents:
  code     git-tracked source of this repo's runtime directories at HEAD (+ the PEAR fork at its pinned commit)
  assets   every non-code repo file the app opened in the audit run (audit_run.py), plus the DWPose ONNX (opened by
           onnxruntime's C++ loader, invisible to the audit hook)
  live_assets/checkpoints   GUAVA-FB B28k (guava_fb_B30k best.pt) and SPRINT from scratch EMA 128k (renamed, unchanged bytes)
  live_assets/torch/hub     the DINOv2 hub repo + weights GUAVA-FB's encoder loads (use with TORCH_HOME=<pkg>/live_assets/torch)
  README_LIVE.md, requirements_live.txt, run_timing.sh, MANIFEST.json (sha256 + size of every file)
Usage: python make_live_package.py --audit /dev/shm/ubx858_eval/live_audit.json --out DIR"""
import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--audit", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
FB = Path("/vault/research/COS/VIRLab/ubx858/persistent_avatar/outputs/guava_fb/runs/guava_fb_B30k/checkpoints/best.pt")
SPRINT = Path("/raid/ubx858/outputs/dnt/eval_programme/sprint_scratch/best_ema.pt")
HUB = Path.home() / ".cache/torch/hub"
CODE_DIRS = ["models", "utils", "dataset", "configs", "tools", "phase3_dnt/live", "phase3_dnt/worker6", "phase3_dnt/guava_fb", "assets/GUAVA"]
SUBMODULE_DIRS = {"EHM-Tracker": ["src"], "submodules/lpipsPyTorch": ["."]}                            # own git repos inside this one
CODE_FILES = ["phase3_dnt/sprint_avatar_fit.py", "phase3_dnt/sprint_fast_driving.py", "requirements.txt", "LIVE_PEAR_SETUP.md"]
CODE_EXT = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".sh", ".cfg"}


def git_files(repo, dirs):
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "--", *dirs], capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out if Path(f).suffix in CODE_EXT and (repo / f).is_file()]


items = {}                                                                                           # arcname -> source path
for f in git_files(ROOT, CODE_DIRS) + [f for f in CODE_FILES if (ROOT / f).is_file()]: items[f] = ROOT / f
for sm, dirs in SUBMODULE_DIRS.items():
    if (ROOT / sm / ".git").exists():
        for f in git_files(ROOT / sm, dirs): items[f"{sm}/{f}"] = ROOT / sm / f
pear = ROOT / "third_party/PEAR"
for f in subprocess.run(["git", "-C", str(pear), "ls-files"], capture_output=True, text=True, check=True).stdout.split():
    if Path(f).suffix in CODE_EXT and (pear / f).is_file(): items[f"third_party/PEAR/{f}"] = pear / f
audit = json.loads(Path(a.audit).read_text())
for f in audit["files"]:
    p = Path(f)
    if "__pycache__" in p.parts:                                                                   # compiled module -> its source
        p = p.parent.parent / (p.name.split(".")[0] + ".py")
        if not p.is_file(): continue
    if str(p).startswith(str(ROOT) + "/"): items[str(p.relative_to(ROOT))] = p
    elif str(p).startswith(str(HUB) + "/"): items[f"live_assets/torch/hub/{p.relative_to(HUB)}"] = p
for p in (HUB / "facebookresearch_dinov2_main").rglob("*"):                                          # the whole hub repo (code is .pyc-cached)
    if p.is_file() and "__pycache__" not in p.parts: items[f"live_assets/torch/hub/{p.relative_to(HUB)}"] = p
items["EHM-Tracker/pretrained/dwpose/dw-ll_ucoco_384.onnx"] = ROOT / "EHM-Tracker/pretrained/dwpose/dw-ll_ucoco_384.onnx"
items["live_assets/checkpoints/guava_fb_B28k.pt"] = FB
items["live_assets/checkpoints/sprint_scratch_ema128k.pt"] = SPRINT
for f in ("README_LIVE.md", "requirements_live.txt", "run_timing.sh"): items[f] = Path(__file__).with_name(f)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
pear_head = subprocess.run(["git", "-C", str(pear), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
man = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "repo_head": head, "pear_head": pear_head,
       "files": {k: {"bytes": v.stat().st_size, "sha256": sha(v)} for k, v in sorted(items.items())}}
(out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
tar = out / f"guava_live_{time.strftime('%Y%m%d')}.tar"
with tarfile.open(tar, "w") as t:
    for k, v in sorted(items.items()): t.add(str(v), arcname=f"guava_live/{k}")
    t.add(str(out / "MANIFEST.json"), arcname="guava_live/MANIFEST.json")
print(json.dumps({"tar": str(tar), "bytes": tar.stat().st_size, "files": len(items), "repo_head": head, "pear_head": pear_head}))
