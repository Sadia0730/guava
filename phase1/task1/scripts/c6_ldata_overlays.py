"""L-data smoke check (no training): overlays of augmented multi-person BEDLAM2 and UBody samples.

Multi-person BEDLAM2: each sample is one labelled person; drawn are that person's GT mesh (green)
and 2D targets through our crop camera (cyan) with the true BEDLAM2 2D joints (red), plus the
GT-mesh boxes of the OTHER labelled people in the same frame (yellow), all through the same
augmentation affine, to show each crop is centred on its own person and labels stay attached.
UBody: the augmented image with the PEAR teacher's own projected body joints (magenta), i.e. the
distillation target, since UBody rows have no ground truth here. Also reports the black-border
fraction of UBody augmentation (stored UBody crops have no context margin).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

GUAVA_ROOT = Path(__file__).resolve().parents[3]
PEAR_ROOT = GUAVA_ROOT / "third_party" / "PEAR"
sys.path.insert(0, str(PEAR_ROOT))
import os  # noqa: E402
os.chdir(PEAR_ROOT)
import student_gt_losses as L  # noqa: E402
import train_pear_student_distill as tpd  # noqa: E402
from dataset.bedlam2_gt_dataset import AugmentConfig, Bedlam2GTDataset  # noqa: E402
from dataset.ubody_distill_dataset import UBodyDistillDataset  # noqa: E402

S = Path("/raid/ubx858/outputs/phase1_task1/ldata_smoke")
MULTI = S / "crops_multi"
BETA_MAP = "/raid/ubx858/outputs/phase1_task1/checkpoint2/betas_locked16_to_2020_200.npz"
TEACHER = Path("/home/ubx858/.cache/huggingface/hub/models--BestWJH--PEAR_models/snapshots/"
               "513a74e70a6b4bdecc90ac84ef989c17fe415a9e/pear_model.pt")
AUG = AugmentConfig(scale=0.15, shift=0.10, rotation_deg=30.0, rotation_prob=0.6, flip_prob=0.5,
                    color_scale=0.2, occlusion_prob=0.5)


def tile(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (384, 384))


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(3)
    kin = L.kinematic_smplx(str(PEAR_ROOT / "assets/SMPLX"), device)
    ds = Bedlam2GTDataset(MULTI, "train", BETA_MAP, train=True, augment=AUG)
    # index npz metadata per row (same order as the dataset: jobs sorted, rows in file order)
    summary = json.loads((MULTI / "build_summary.json").read_text())
    jobs = sorted(j for j, v in summary["jobs"].items() if v["split"] == "train" and v["rows"] > 0)
    meta = [np.load(MULTI / f"{j}.npz", allow_pickle=True) for j in jobs]
    rows = [(ji, r) for ji, m in enumerate(meta) for r in range(len(m["path"]))]
    rng = np.random.default_rng(0)
    crowded = [i for i, (ji, r) in enumerate(rows)
               if ((meta[ji]["seq"] == meta[ji]["seq"][r]) & (meta[ji]["frame"] == meta[ji]["frame"][r])).sum() >= 3]
    picks = rng.choice(crowded, 6, replace=False)
    tiles = []
    for idx in picks:
        sample = ds[int(idx)]
        ji, r = rows[idx]
        m = meta[ji]
        u, dx, dy, th, fl = sample["aug"].tolist()
        A = ds.affine(u, dx, dy, th, bool(fl))
        batch = {k: (v[None].to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
        with torch.no_grad():
            verts, _ = L.body_mesh(kin, batch["rotmats_noflip"].float(), batch["betas"].float())
            if fl:
                verts = verts * L.MIRROR_X.to(verts)
            joints = L.gt_joints(kin, batch)
            t, ok = L.fit_crop_translation(joints, batch["joints2d"].float(), batch["joints2d_valid"])
            v_px = L.crop_projection_px(verts, t)[0].cpu().numpy()
            j_px = L.crop_projection_px(joints, t)[0].cpu().numpy()
        img = np.ascontiguousarray(sample["image"].permute(1, 2, 0).numpy())
        for x, y in v_px[::25]:
            cv2.circle(img, (int(x), int(y)), 1, (0, 200, 0), -1)
        for j, ((x, y), (tx, ty)) in enumerate(zip(j_px, sample["joints2d"].numpy())):
            if sample["joints2d_valid"][j]:
                cv2.circle(img, (int(tx), int(ty)), 3, (255, 0, 0), 1)
                cv2.circle(img, (int(x), int(y)), 1, (0, 255, 255), -1)
        same = np.flatnonzero((m["seq"] == m["seq"][r]) & (m["frame"] == m["frame"][r]))
        ctx = m["ctx_affine"][r]
        for o in same:
            if o == r:
                continue
            x0, y0, x1, y1 = m["box_xyxy"][o]
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            c = corners @ ctx[:, :2].T + ctx[:, 2]
            c = c @ A[:, :2].T + A[:, 2]
            cv2.polylines(img, [c.astype(np.int32)], True, (255, 255, 0), 1)
        cv2.putText(img, f"{len(same)} people | rot {th:+.0f} flip {int(fl)}", (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append(tile(img))

    teacher = tpd.load_teacher("configs/infer.yaml", TEACHER, device)
    from models.smplx.SMPLXV2 import SMPLX
    reg = SMPLX(str(PEAR_ROOT / "assets/SMPLX"), n_shape=300, n_exp=50).to(device).eval()
    ub = UBodyDistillDataset("/raid/ubx858/datasets/processed/pear_student/train.jsonl", train=True, augment=AUG)
    sel = np.random.default_rng(1).choice(len(ub), 500, replace=False)
    black = []
    for k, i in enumerate(sel):
        s = ub[int(i)]
        im = s["image"].permute(1, 2, 0).numpy()
        black.append(float((im.max(2) == 0).mean()))
        if k < 6:
            x = s["image"][None].to(device).float() / 255.0
            with torch.no_grad():
                out = teacher(x)
                mesh = reg(tpd.add_body_cam(out), pose_type="rotmat")
                p = (tpd.project_joints(out["pd_cam"], mesh["joints"][:, :22]) * 256.0)[0].cpu().numpy()
            img = np.ascontiguousarray(im)
            for x_, y_ in p:
                cv2.circle(img, (int(x_), int(y_)), 3, (255, 0, 255), -1)
            u, dx, dy, th, fl = s["aug"].tolist()
            cv2.putText(img, f"UBody | scale {u:.2f} rot {th:+.0f} flip {int(fl)}", (4, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tiles.append(tile(img))
    bed_black = [float((ds[int(i)]["image"].permute(1, 2, 0).numpy().max(2) == 0).mean())
                 for i in np.random.default_rng(2).choice(len(ds), 300, replace=False)]
    grid = np.vstack([np.hstack(tiles[0:6]), np.hstack(tiles[6:12])])
    cv2.imwrite(str(S / "overlays_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    stats = {"ubody_black_fraction_mean": float(np.mean(black)), "ubody_black_fraction_p90": float(np.percentile(black, 90)),
             "bedlam_multi_black_fraction_mean": float(np.mean(bed_black)),
             "bedlam_multi_black_fraction_p90": float(np.percentile(bed_black, 90))}
    (S / "overlay_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print("wrote", S / "overlays_grid.jpg")


if __name__ == "__main__":
    main()
