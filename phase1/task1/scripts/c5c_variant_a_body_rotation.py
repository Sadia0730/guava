"""Checkpoint 5 follow-up: PA-MPJPE removes global rotation, so (a)'s stuck root orientation
cannot explain its 196.8 mm PA-MPJPE on its own -- its articulated (body) pose must also be wrong.
Read-only, no training. For (a)'s best checkpoint vs the teacher on 3DPW validation: rotation
error for body joints only (root + 21 body joints, excluding both hands), per-joint for the worst
5, and shape/scale differences.
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

_name = "guava_eval_3dpw_standard"
_spec = importlib.util.spec_from_file_location(_name, GUAVA_ROOT / "tools" / "eval_3dpw_standard.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules[_name] = ev
_spec.loader.exec_module(ev)

sys.path.insert(0, str(PEAR_ROOT))
import student_gt_losses as L  # noqa: E402
import train_pear_student_distill as tpd  # noqa: E402

TEACHER_CKPT = Path("/home/ubx858/.cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/"
                     "513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt")
STUDENT_CKPT = Path("/raid/ubx858/outputs/phase1_task1/screen/a/checkpoints/best.pt")
STUDENT_CONFIG = "configs/student_l70_v2.yaml"
BATCH = 64
BODY_JOINT_NAMES = [
    "pelvis(root)", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]  # SMPL-X joints 0-21: root + 21 body joints (hands excluded)


def main():
    device = torch.device("cuda:3")  # idle GPU, not used by the two running screening runs
    teacher = tpd.load_teacher("configs/infer.yaml", TEACHER_CKPT, device)
    student = tpd.load_student(STUDENT_CONFIG, BATCH, 1, device)
    state = torch.load(STUDENT_CKPT, map_location="cpu", weights_only=False)
    student.load_state_dict(state["student"], strict=True)
    student.eval()
    print(f"student checkpoint step: {state.get('step')}")

    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), "validation")
    loader = DataLoader(ev.CropDataset(samples, "gt_keypoints", None), batch_size=BATCH,
                        num_workers=10, shuffle=False, pin_memory=True)

    per_joint_sum = torch.zeros(22, device=device)
    shape_l1_sum, scale_l1_sum, hand_scale_l1_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
            t_out = teacher(images)
            s_out = student(images)
            tof = lambda v: ({kk: tof(vv) for kk, vv in v.items()} if isinstance(v, dict)  # noqa: E731
                             else (v.float() if torch.is_tensor(v) else v))
            t_out, s_out = tof(t_out), tof(s_out)

            s_rot, t_rot = L.predicted_rotmats(s_out), L.predicted_rotmats(t_out)  # (B, 52, 3, 3)
            ang = L.geodesic(s_rot[:, :22], t_rot[:, :22])  # root + 21 body joints only, no hands
            per_joint_sum += ang.sum(0)

            shape_l1_sum += (s_out["body_param"]["shape"] - t_out["body_param"]["shape"]).abs().mean(1).sum().item()
            scale_l1_sum += (L.camera_params(s_out["pd_cam"][:, :3, 3])[:, 2] -
                              L.camera_params(t_out["pd_cam"][:, :3, 3])[:, 2]).abs().sum().item()
            hand_scale_l1_sum += (s_out["body_param"]["hand_scale"] -
                                  t_out["body_param"]["hand_scale"]).abs().mean(1).sum().item()
            n += images.shape[0]

    per_joint_deg = torch.rad2deg(per_joint_sum / n).cpu().tolist()
    ranked = sorted(zip(BODY_JOINT_NAMES, per_joint_deg), key=lambda x: -x[1])
    result = {
        "frames": n,
        "body_rotation_mean_deg": sum(per_joint_deg) / len(per_joint_deg),
        "body_rotation_root_deg": per_joint_deg[0],
        "body_rotation_excl_root_mean_deg": sum(per_joint_deg[1:]) / 21,
        "worst_5_joints_deg": ranked[:5],
        "best_5_joints_deg": ranked[-5:],
        "all_joints_deg": dict(zip(BODY_JOINT_NAMES, per_joint_deg)),
        "shape_l1_vs_teacher": shape_l1_sum / n,
        "camera_scale_s_l1_vs_teacher": scale_l1_sum / n,
        "hand_scale_l1_vs_teacher": hand_scale_l1_sum / n,
    }
    print(json.dumps(result, indent=2))
    out = Path("/raid/ubx858/outputs/phase1_task1/checkpoint5/variant_a_body_rotation.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
