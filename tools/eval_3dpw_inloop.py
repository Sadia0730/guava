"""Standard 3DPW evaluation of an in-memory model, for checkpoint selection during training.

Uses the functions of eval_3dpw_standard.py (tag eval-3dpw-standard-v1) in the same order as
its main loop (GT-keypoint crop, fp32, EHM-s -> SMPL via the fixed correspondence, H36M J14,
hip-midpoint centring), so its numbers equal the standalone evaluator's on the same frames.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Loaded by file path: inside the PEAR trainer a different top-level `tools` package (PEAR's) wins.
_name = "guava_eval_3dpw_standard"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, Path(__file__).resolve().with_name("eval_3dpw_standard.py"))
    sys.modules[_name] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules[_name])
ev = sys.modules[_name]


class InLoop3DPW:
    def __init__(self, split: str = "validation", stride: int = 1, device: str = "cuda:0",
                 batch_size: int = 64, num_workers: int = 8):
        self.device = torch.device(device)
        self.samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), split)[::stride]
        self.loader = DataLoader(ev.CropDataset(self.samples, "gt_keypoints", None), batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False, pin_memory=True,
                                 persistent_workers=num_workers > 0)
        self.h36m = torch.from_numpy(np.load(ev.DEFAULT_H36M_REGRESSOR)).float().to(self.device)
        self.mapping = ev.load_smplx_to_smpl(ev.DEFAULT_SMPLX2SMPL).to(self.device)
        self.smpl = {g: ev.build_smpl(ev.DEFAULT_SMPL_DIR, g, self.device) for g in ("male", "female", "neutral")}
        self.smpl_regressor = self.smpl["neutral"].J_regressor.float()

    @torch.inference_mode()
    def evaluate(self, model, ehm) -> dict[str, float]:
        was_training = model.training
        model.eval()
        sums, count = {}, 0
        for batch in self.loader:
            bs = [self.samples[i] for i in batch["index"].tolist()]
            images = batch["image"].to(self.device, non_blocking=True).float().div_(255.0)
            output = model(images)
            output = {k: ({kk: (vv.float() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                          if isinstance(v, dict) else v.float()) for k, v in output.items()}
            pred_x = ev.predicted_smplx_vertices(output, ehm)
            pred = torch.stack([torch.sparse.mm(self.mapping, v) for v in pred_x])
            gt = torch.empty_like(pred)
            pose = torch.from_numpy(np.stack([s.pose for s in bs])).to(self.device)
            betas = torch.from_numpy(np.stack([s.betas for s in bs])).to(self.device)
            rot = torch.from_numpy(np.stack([s.cam_rotation for s in bs])).to(self.device)
            for gender in ("male", "female"):
                sel = [i for i, s in enumerate(bs) if s.gender == gender]
                if sel:
                    v = self.smpl[gender](global_orient=pose[sel, :3], body_pose=pose[sel, 3:], betas=betas[sel]).vertices
                    gt[sel] = torch.einsum("bij,bvj->bvi", rot[sel], v)
            for k, v in ev.frame_metrics(pred, gt, self.h36m, self.smpl_regressor).items():
                sums[k] = sums.get(k, 0.0) + float(v.double().sum())
            count += len(bs)
        if was_training:
            model.train()
        return {k: v / count for k, v in sums.items()} | {"frames": count}
