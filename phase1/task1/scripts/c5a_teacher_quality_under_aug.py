"""Checkpoint 5, Step 1.1: is PEAR (the teacher) a good distillation target under our augmentation?

Evaluates PEAR directly against BEDLAM2 ground truth on BEDLAM2 validation crops (never used for
gradient updates in any screening run), under four augmentation regimes: none, rotation only,
flip only, full (the same distribution the screening runs sample from). All comparisons use the
55 SMPL-X kinematic joints (SMPLXV2, use_joint_regressor=False), root (pelvis)-relative, so both
skeletons are exactly comparable (no SMPL-X/SMPL mismatch as on 3DPW). PA-MPJPE uses the same
similarity Procrustes as the standard 3DPW/EHF evaluators.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

GUAVA_ROOT = Path(__file__).resolve().parents[3]
PEAR_ROOT = GUAVA_ROOT / "third_party" / "PEAR"

# Load guava's standard evaluator by file path first (its `similarity_align`), before PEAR_ROOT
# is added to sys.path: PEAR has its own top-level `tools` package that would shadow it.
_name = "guava_eval_3dpw_standard"
_spec = importlib.util.spec_from_file_location(_name, GUAVA_ROOT / "tools" / "eval_3dpw_standard.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules[_name] = ev
_spec.loader.exec_module(ev)

sys.path.insert(0, str(PEAR_ROOT))
import student_gt_losses as L  # noqa: E402
import train_pear_student_distill as tpd  # noqa: E402
from dataset.bedlam2_gt_dataset import AugmentConfig, Bedlam2GTDataset  # noqa: E402

DATA_ROOT = Path("/raid/ubx858/datasets/processed/bedlam2_gt")
BETA_MAP = Path("/raid/ubx858/outputs/phase1_task1/checkpoint2/betas_locked16_to_2020_200.npz")
TEACHER_CKPT = Path("/home/ubx858/.cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/"
                     "513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt")
SCREEN_AUGMENT = AugmentConfig(scale=0.15, shift=0.10, rotation_deg=30.0, rotation_prob=0.6,
                                flip_prob=0.5, color_scale=0.2, occlusion_prob=0.5)
SCENARIOS = {
    "no_augmentation": (False, AugmentConfig()),
    "rotation_only": (True, AugmentConfig(scale=0, shift=0, rotation_deg=30.0, rotation_prob=1.0,
                                           flip_prob=0.0, color_scale=0, occlusion_prob=0)),
    "flip_only": (True, AugmentConfig(scale=0, shift=0, rotation_deg=0.0, rotation_prob=0.0,
                                       flip_prob=1.0, color_scale=0, occlusion_prob=0)),
    "full_augmentation": (True, SCREEN_AUGMENT),
}
N_ROWS = 3000
SEED = 0
BATCH = 64


def main():
    device = torch.device("cuda:0")
    teacher = tpd.load_teacher("configs/infer.yaml", TEACHER_CKPT, device)
    kinematic = L.kinematic_smplx(str(PEAR_ROOT / "assets/SMPLX"), device)

    results = {}
    for name, (train_flag, aug) in SCENARIOS.items():
        ds = Bedlam2GTDataset(DATA_ROOT, "val", BETA_MAP, train=train_flag, augment=aug,
                               max_rows=N_ROWS, seed=SEED)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=8)
        sq_all, sq_body, pa_all, pa_body, n = [], [], [], [], 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                images = batch["image"].float().div_(255.0)
                out = teacher(images)
                pred_rot = L.predicted_rotmats(out)
                pred_j = L.joints55(kinematic, pred_rot, out["body_param"]["shape"].float(),
                                    out["body_param"]["exp"].float())
                gt_j = L.gt_joints(kinematic, batch)
                pred_r = (pred_j - pred_j[:, :1]).double()
                gt_r = (gt_j - gt_j[:, :1]).double()
                sq_all.append((pred_r - gt_r).norm(dim=-1).mean(1))
                sq_body.append((pred_r[:, :22] - gt_r[:, :22]).norm(dim=-1).mean(1))
                pa_all.append((ev.similarity_align(pred_r, gt_r) - gt_r).norm(dim=-1).mean(1))
                pa_body.append((ev.similarity_align(pred_r[:, :22], gt_r[:, :22]) - gt_r[:, :22]).norm(dim=-1).mean(1))
                n += images.shape[0]
        cat = lambda xs: torch.cat(xs).mean().item() * 1000.0  # noqa: E731
        results[name] = {"frames": n, "mpjpe55_mm": cat(sq_all), "mpjpe_body22_mm": cat(sq_body),
                          "pa_mpjpe55_mm": cat(pa_all), "pa_mpjpe_body22_mm": cat(pa_body)}
        print(name, results[name])

    out_dir = Path("/raid/ubx858/outputs/phase1_task1/checkpoint5")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "teacher_quality_under_aug.json").write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", out_dir / "teacher_quality_under_aug.json")


if __name__ == "__main__":
    main()
