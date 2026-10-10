"""Checkpoint 5, Step 1.2: how closely do the students track the teacher, on 3DPW validation?

Compares variant (a)'s best checkpoint and the existing distilled student (235000) against PEAR
on the same 3DPW validation crops (standard evaluator's GT-keypoint crop), in both parameter
space (rotation geodesic, camera, FLAME) and joint space (3D and 2D, via the same SMPLXV2
regressed-joint path the distillation joint3d/joint2d terms use).
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
STUDENTS = {
    "a_best_32k": Path("/raid/ubx858/outputs/phase1_task1/screen/a/checkpoints/best.pt"),
    "student_235000": Path("/raid/ubx858/outputs/pear_student_v2_phase2/checkpoints/step_0235000.pt"),
}
STUDENT_CONFIG = "configs/student_l70_v2.yaml"
BATCH = 64


def load_student_checkpoint(ckpt_path: Path, device):
    model = tpd.load_student(STUDENT_CONFIG, BATCH, 1, device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["student"], strict=True)
    return model.eval()


def main():
    device = torch.device("cuda:0")
    teacher = tpd.load_teacher("configs/infer.yaml", TEACHER_CKPT, device)
    from models.smplx.SMPLXV2 import SMPLX
    smplx_regressed = SMPLX(str(PEAR_ROOT / "assets/SMPLX"), n_shape=300, n_exp=50).to(device).eval()

    samples = ev.load_samples(ev.DEFAULT_DATASET.resolve(), "validation")
    loader = DataLoader(ev.CropDataset(samples, "gt_keypoints", None), batch_size=BATCH,
                        num_workers=10, shuffle=False, pin_memory=True)

    results = {}
    for name, ckpt in STUDENTS.items():
        student = load_student_checkpoint(ckpt, device)
        rot_all, cam_all, flame_all, j3d_all, j3d_root_all, j2d_all_px, n = [], [], [], [], [], [], 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
                t_out = teacher(images)
                s_out = student(images)
                tof = lambda v: ({kk: tof(vv) for kk, vv in v.items()} if isinstance(v, dict)  # noqa: E731
                                 else (v.float() if torch.is_tensor(v) else v))
                t_out, s_out = tof(t_out), tof(s_out)
                # Parameter space.
                s_rot, t_rot = L.predicted_rotmats(s_out), L.predicted_rotmats(t_out)
                rot_all.append(L.geodesic(s_rot, t_rot).mean(1))
                cam_all.append((L.camera_params(s_out["pd_cam"][:, :3, 3]) -
                                L.camera_params(t_out["pd_cam"][:, :3, 3])).abs().mean(1))
                flame_all.append(tpd.weighted_l1(s_out["flame_param"], t_out["flame_param"],
                                                 tpd.FLAME_PARAM_WEIGHTS).expand(images.shape[0]))
                # Joint space: same SMPLXV2-regressed 22 joints the distillation joint3d/joint2d terms use.
                s_mesh = smplx_regressed(tpd.add_body_cam(s_out), pose_type="rotmat")
                t_mesh = smplx_regressed(tpd.add_body_cam(t_out), pose_type="rotmat")
                sj, tj = s_mesh["joints"][:, :22], t_mesh["joints"][:, :22]
                j3d_all.append((sj - tj).norm(dim=-1).mean(1))
                j3d_root_all.append(((sj - sj[:, :1]) - (tj - tj[:, :1])).norm(dim=-1).mean(1))
                s2d = tpd.project_joints(s_out["pd_cam"], sj) * 256.0
                t2d = tpd.project_joints(t_out["pd_cam"], tj) * 256.0
                j2d_all_px.append((s2d - t2d).norm(dim=-1).mean(1))
                n += images.shape[0]
        cat_mean = lambda xs: torch.cat(xs).mean().item()  # noqa: E731
        results[name] = {
            "frames": n,
            "rotation_geodesic_deg": torch.rad2deg(torch.tensor(cat_mean(rot_all))).item(),
            "camera_params_l1": cat_mean(cam_all),
            "flame_weighted_l1": cat_mean(flame_all),
            "joint3d_mm_body_frame": cat_mean(j3d_all) * 1000.0,
            "joint3d_mm_root_relative": cat_mean(j3d_root_all) * 1000.0,
            "joint2d_px": cat_mean(j2d_all_px),
        }
        print(name, results[name])
        del student
        torch.cuda.empty_cache()

    out_dir = Path("/raid/ubx858/outputs/phase1_task1/checkpoint5")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "student_teacher_agreement.json").write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", out_dir / "student_teacher_agreement.json")


if __name__ == "__main__":
    main()
