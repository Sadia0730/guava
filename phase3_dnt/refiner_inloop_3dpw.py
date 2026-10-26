"""3DPW-val with the in-loop validator (tools/eval_3dpw_inloop.InLoop3DPW, split validation, stride 1, camera input on, pure
SMPL-X head: the setting of SPRINT's checkpoint selection) for a SPRINT checkpoint without and with a no-render refiner
(eval_3dpw_standard.RefinedModel). Usage: python phase3_dnt/refiner_inloop_3dpw.py --checkpoint CK [--refiner R] --out JSON"""
import argparse, json, os, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--refiner", type=Path, default=None)
p.add_argument("--model", default="sprint_raw", choices=("sprint", "sprint_raw")); p.add_argument("--out", type=Path, required=True)
p.add_argument("--no-camera", action="store_true", help="camera input off (default on, as in training)")
a = p.parse_args()
from eval_3dpw_inloop import InLoop3DPW, ev
dev = torch.device("cuda:0"); t0 = time.time()
cwd = Path.cwd(); os.chdir(ev.PEAR_ROOT)
model, rep = ev.load_model(a.model, a.checkpoint.resolve(), "configs/student_l70_v2.yaml", 64, dev)
model = model.to(dev).eval()
from models.modules.ehm import EHM_v2
ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(dev).eval()
os.chdir(cwd)
val = InLoop3DPW("validation", stride=1, device="cuda:0")
res = {"checkpoint": str(a.checkpoint), "model": a.model, "camera": not a.no_camera, "head": "smplx"}
res["student"] = val.evaluate(model, ehm, camera=not a.no_camera, head="smplx")
if a.refiner:
    res["refiner"] = str(a.refiner)
    res["student_refiner"] = val.evaluate(ev.RefinedModel(model, a.refiner.resolve(), dev), ehm, camera=not a.no_camera, head="smplx")
res["seconds"] = time.time() - t0
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))
