#!/usr/bin/env python
"""No-training, paired-history occlusion diagnostic for the PEAR live pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pear-recovery-matplotlib")
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main.live_pear_guava import OneEuroFilter, matrix_to_rotation_6d, rotation_6d_to_matrix
from tools.evaluate_pear_student_quality import (
    PEAR_ROOT, TEACHER_CHECKPOINT, FrameRef, load_images, read_manifest,
    sha256_file, tree_cat, tree_index, tree_index_range, tree_to_cpu, write_video,
)

ROT_KEYS = ("global_pose", "body_pose", "left_hand_pose", "right_hand_pose")
FLAME_KEYS = ("expression_params", "jaw_params", "pose_params", "eye_pose_params", "eyelid_params")
PARTS = {"lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
         "left_arm": [16, 18, 20], "right_arm": [17, 19, 21]}
EDGES = [(0,1),(0,2),(0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),
         (7,10),(8,11),(9,12),(12,13),(12,14),(12,15),(13,16),(14,17),
         (16,18),(17,19),(18,20),(19,21)]
PROTECTED = [ROOT / "avatarbudget" / f for f in ("scout.py", "temporal.py", "router.py")]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("infer", "analyze", "render"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path(
        "/raid/ubx858/outputs/pear_student_v2_phase2/checkpoints/step_0235000.pt"))
    p.add_argument("--manifest", type=Path, default=Path(
        "/raid/ubx858/datasets/processed/pear_student/val.jsonl"))
    p.add_argument("--student-config", default="configs/student_l70_v2.yaml")
    p.add_argument("--teacher-checkpoint", type=Path, default=TEACHER_CHECKPOINT)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--device", default="cuda:5")
    p.add_argument("--render-clips", type=int, default=3)
    return p.parse_args()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fingerprint():
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in PROTECTED}


def make_plan(args):
    sequences = read_manifest(args.manifest)
    rng = random.Random(args.seed)
    by_source = {s: [i for i, q in enumerate(sequences)
                     if q.source == s and q.num_frames >= 180 and q.frame_step == 1
                     and abs(q.fps - 30) < 0.01] for s in ("bedlam", "ubody")}
    for indices in by_source.values():
        rng.shuffle(indices)
    designs = [(s, d, r) for s in by_source for d in (0.2, 0.5, 1., 2.) for r in PARTS]
    designs *= (args.episodes + len(designs) - 1) // len(designs)
    rng.shuffle(designs)
    used_families, episodes = set(), []
    for i, (source, duration, region) in enumerate(designs[:args.episodes]):
        while by_source[source]:
            index = by_source[source].pop()
            seq = sequences[index]
            family = f"{source}/{seq.name.split('_scene')[0]}"
            if family not in used_families:
                used_families.add(family)
                break
        else:
            raise ValueError("Insufficient distinct sequence families")
        gap = round(duration * seq.fps)
        length = 30 + gap + 45
        start = rng.randint(0, seq.num_frames - length)
        episodes.append(dict(id=i, split="calibration" if i % 4 == 0 else "test",
                             source=source, sequence=seq.name, directory=str(seq.directory),
                             sequence_index=index, start=start, length=length, fps=seq.fps,
                             region=region, onset=30, reveal=30 + gap, duration_seconds=duration))
    return sequences, episodes


def infer_model(model, images, args):
    result = []
    with torch.inference_mode():
        for offset in range(0, len(images), args.batch_size):
            result.append(tree_to_cpu(model(images[offset:offset + args.batch_size].to(args.device))))
    return tree_cat(result)


def pack(output):
    values = [matrix_to_rotation_6d(output["body_param"][k]).flatten(1) for k in ROT_KEYS]
    values.append(output["body_param"]["exp"].flatten(1))
    values.extend(output["flame_param"][k].flatten(1) for k in FLAME_KEYS)
    return torch.cat(values, dim=1)


def unpack(values, template):
    result = copy.deepcopy(template)
    offset = 0
    for key in ROT_KEYS:
        shape = template["body_param"][key].shape[:-2] + (6,)
        count = int(np.prod(shape[1:]))
        result["body_param"][key] = rotation_6d_to_matrix(values[:, offset:offset+count].reshape(shape))
        offset += count
    for group, key in [("body_param", "exp")] + [("flame_param", k) for k in FLAME_KEYS]:
        shape = template[group][key].shape
        count = int(np.prod(shape[1:]))
        result[group][key] = values[:, offset:offset+count].reshape(shape)
        offset += count
    assert offset == values.shape[1]
    return result


def filter_stream(values, fps, cutoff=2., beta=.3, reset_at=None, innovation=None):
    filt = OneEuroFilter(cutoff, beta, 1.)
    outputs, resets = [], []
    for t, value in enumerate(values):
        reset = t == reset_at
        # A deployable, observation-only control: large mean 6D body innovation.
        if innovation is not None and filt._x is not None:
            score = (value[:132] - filt._x[:132]).square().mean().sqrt().item()
            reset |= score > innovation
        if reset:
            filt = OneEuroFilter(cutoff, beta, 1.)
            resets.append(t)
        outputs.append(filt(value.clone(), t / fps).clone())
    return torch.stack(outputs), resets


def body_rotations(output):
    return torch.cat([output["body_param"]["global_pose"], output["body_param"]["body_pose"]], dim=1)


def forward_joints(rotations, rest, parents):
    positions, orientations = [], []
    for j in range(len(parents)):
        parent = int(parents[j])
        if parent < 0:
            positions.append(rest[j].expand(len(rotations), 3))
            orientations.append(rotations[:, j])
        else:
            positions.append(positions[parent] + torch.einsum(
                "bij,j->bi", orientations[parent], rest[j] - rest[parent]))
            orientations.append(orientations[parent] @ rotations[:, j])
    points = torch.stack(positions, dim=1)
    return (points - points[:, :1]) * 1000.


def angle_degrees(first, second):
    rel = first @ second.transpose(-1, -2)
    skew = torch.stack([rel[...,2,1]-rel[...,1,2], rel[...,0,2]-rel[...,2,0],
                        rel[...,1,0]-rel[...,0,1]], dim=-1)
    return torch.atan2(skew.norm(dim=-1) / 2,
                       (rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).abs().rad2deg()


def recovery_time(errors, fps, tolerance=5., consecutive=3):
    for i in range(len(errors) - consecutive + 1):
        if bool(torch.all(errors[i:i+consecutive] <= tolerance)):
            return i / fps
    return None


def motion_metrics(pred, reference, fps):
    pv, rv = torch.diff(pred, dim=0) * fps, torch.diff(reference, dim=0) * fps
    energy = float(rv.square().sum())
    speed = float(rv.norm(dim=-1).mean())
    result = {"velocity_error_mm_s": float((pv-rv).norm(dim=-1).mean()),
              "reference_speed_mm_s": speed, "direction_cosine": None,
              "amplitude_gain": None, "best_lag_ms": None,
              "acceleration_error_mm_s2": float((torch.diff(pv, dim=0)-torch.diff(rv, dim=0)).norm(dim=-1).mean()*fps)}
    stationary = rv.norm(dim=-1) < 20
    result["stationary_joint_samples"] = int(stationary.sum())
    result["stationary_speed_mm_s"] = float(pv.norm(dim=-1)[stationary].mean()) if stationary.any() else None
    if speed >= 20 and energy > 1e-8:
        result["amplitude_gain"] = float((pv.square().sum() / energy).sqrt())
        result["direction_cosine"] = float(torch.nn.functional.cosine_similarity(
            pv.flatten().unsqueeze(0), rv.flatten().unsqueeze(0)).item())
        scores = []
        for lag in range(-5, 6):
            a, b = (pv[lag:], rv[:-lag]) if lag > 0 else ((pv[:lag], rv[-lag:]) if lag < 0 else (pv, rv))
            scores.append((float((a-b).square().mean()), lag))
        result["best_lag_ms"] = min(scores,key=lambda item:(item[0],abs(item[1])))[1] / fps * 1000
    return result


def masks_from_points(images, points, episode):
    result = images.clone()
    mask = np.zeros((len(images), 256, 256), dtype=np.uint8)
    indices = PARTS[episode["region"]]
    if episode["region"] == "left_arm":
        indices = indices + list(range(25,40))
    if episode["region"] == "right_arm":
        indices = indices + list(range(40,55))
    for t in range(episode["onset"], episode["reveal"]):
        xy = points[t, indices].numpy()
        valid = np.isfinite(xy).all(1) & (xy[:,0] >= 0) & (xy[:,0] < 256) & (xy[:,1] >= 0) & (xy[:,1] < 256)
        xy = xy[valid]
        if len(xy):
            lo = np.maximum(np.floor(xy.min(0) - 12).astype(int), 0)
            hi = np.minimum(np.ceil(xy.max(0) + 12).astype(int), 256)
            mask[t, lo[1]:hi[1], lo[0]:hi[0]] = 1
            result[t, :, lo[1]:hi[1], lo[0]:hi[0]] = .5
    return result, mask


def run_inference(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this diagnostic with approved GPU access")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    sequences, episodes = make_plan(args)
    protected = fingerprint()
    provenance = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
                      manifest=str(args.manifest), manifest_sha256=sha256_file(args.manifest),
                      teacher_checkpoint=str(args.teacher_checkpoint),
                      teacher_sha256=sha256_file(args.teacher_checkpoint),
                      config=args.student_config, seed=args.seed, precision="fp32",
                      batch_size=args.batch_size, device=args.device,
                      gpu=torch.cuda.get_device_name(args.device), protected_before=protected,
                      git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                      evaluator_sha256=sha256_file(Path(__file__)),
                      protocol="paired synthetic occlusions; no independent pose ground truth")
    existing = args.output_dir / "plan.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        for key in ("checkpoint_sha256", "manifest_sha256", "teacher_sha256", "config", "seed", "precision", "batch_size"):
            if old["provenance"][key] != provenance[key]:
                raise ValueError(f"Cache provenance mismatch: {key}")
        if old["episodes"] != episodes:
            raise ValueError("Existing episode plan differs")
    write_json(existing, dict(provenance=provenance, episodes=episodes))
    original = Path.cwd()
    sys.path.insert(0, str(PEAR_ROOT))
    os.chdir(PEAR_ROOT)
    try:
        from train_pear_student_distill import load_config, set_config_batch_size, add_body_cam
        from models.pipeline.ehm_pipeline import Ehm_Pipeline
        from models.pipeline.student_pipeline import PearStudentPipeline
        from models.smplx.SMPLXV2 import SMPLX
        cfg = load_config("configs/infer.yaml")
        set_config_batch_size(cfg, args.batch_size, 1)
        teacher = Ehm_Pipeline(cfg)
        checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
        teacher.backbone.load_state_dict(checkpoint["backbone"], strict=True)
        teacher.head.load_state_dict(checkpoint["head"], strict=True)
        del checkpoint
        cfg = load_config(args.student_config)
        set_config_batch_size(cfg, args.batch_size, 1)
        student = PearStudentPipeline(cfg)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["student"], strict=True)
        if int(checkpoint["step"]) != 235000:
            raise ValueError("This diagnostic is explicitly for checkpoint 235000")
        del checkpoint
        teacher = teacher.to(args.device).eval()
        student = student.to(args.device).eval()
        mesh = SMPLX(str(PEAR_ROOT / "assets/SMPLX"), n_shape=300, n_exp=50).to(args.device).eval()
        rest = torch.einsum("jv,vc->jc", mesh.J_regressor, mesh.v_template)[:22].detach().cpu()
        torch.save(dict(rest=rest, parents=mesh.parents[:22].cpu()), args.output_dir / "skeleton.pt")
    finally:
        os.chdir(original)
        sys.path.remove(str(PEAR_ROOT))
    (args.output_dir / "cache").mkdir(exist_ok=True)
    for episode in episodes:
        path = args.output_dir / "cache" / f"episode_{episode['id']:03d}.pt"
        if path.exists():
            print(f"cache exists: {path.name}", flush=True)
            continue
        begin = time.perf_counter()
        refs = [FrameRef(episode["sequence_index"], episode["start"]+i) for i in range(episode["length"])]
        images = load_images(sequences, refs)
        teacher_clean = infer_model(teacher, images, args)
        student_clean = infer_model(student, images, args)
        projections = []
        with torch.inference_mode():
            for i in range(0, len(images), args.batch_size):
                out = tree_index_range(teacher_clean, i, i+args.batch_size)
                out = {k: {j:v.to(args.device) for j,v in x.items()} if isinstance(x,dict) else x.to(args.device)
                       for k,x in out.items()}
                joints = mesh(add_body_cam(out), pose_type="rotmat")["joints"]
                cam = out["pd_cam"]
                xyz = torch.einsum("bij,bkj->bki",cam[:,:3,:3],joints) + cam[:,None,:3,3]
                projections.append(((1 - 24*xyz[...,:2]/xyz[...,2:3].clamp_min(1e-4))*128).cpu())
        points = torch.cat(projections)
        corrupted, mask = masks_from_points(images, points, episode)
        preview=[]
        for t in (episode["onset"],episode["reveal"]-1,episode["reveal"]):
            panel=corrupted[t].permute(1,2,0).mul(255).byte().numpy()[...,::-1].copy()
            for xy in points[t,PARTS[episode["region"]]].numpy():
                if np.isfinite(xy).all() and (xy>=0).all() and (xy<256).all():
                    cv2.circle(panel,tuple(xy.astype(int)),3,(0,0,255),-1)
            preview.append(label(panel,f"{episode['region']} frame {t}"))
        cv2.imwrite(str(args.output_dir/"cache"/f"episode_{episode['id']:03d}_mask.jpg"),np.hstack(preview))
        student_corrupt = infer_model(student, corrupted, args)
        teacher_corrupt = infer_model(teacher, corrupted, args)
        reveal = episode["reveal"]
        assert torch.equal(images[reveal:], corrupted[reveal:])
        raw_difference = float((pack(student_clean)[reveal:] - pack(student_corrupt)[reveal:]).abs().max())
        torch.save(dict(episode=episode, teacher_clean=teacher_clean, student_clean=student_clean,
                        student_corrupt=student_corrupt, teacher_corrupt=teacher_corrupt,
                        masks=torch.from_numpy(mask), projected_teacher=points,
                        raw_suffix_max_abs=raw_difference,
                        actual_device=args.device,
                        input_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest()), path)
        area = mask[episode["onset"]:reveal].mean()
        print(f"episode {episode['id']+1}/{len(episodes)} {episode['source']} {episode['region']} "
              f"gap={episode['duration_seconds']} mask={area:.3f} raw_suffix={raw_difference:.3g} "
              f"seconds={time.perf_counter()-begin:.1f}", flush=True)
    provenance["checkpoint_sha256_after"] = sha256_file(args.checkpoint)
    provenance["protected_after"] = fingerprint()
    provenance["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated(args.device)/1e9
    assert provenance["checkpoint_sha256"] == provenance["checkpoint_sha256_after"]
    assert protected == provenance["protected_after"]
    write_json(args.output_dir / "provenance.json", provenance)


def baseline_specs():
    specs = {"raw": {}, "current": dict(cutoff=2.,beta=.3),
             "oracle_reveal_reset": dict(cutoff=2.,beta=.3,oracle=True),
             "oracle_reset_delay3": dict(cutoff=2.,beta=.3,delay=3)}
    for cutoff in (1.,2.,4.,8.):
        for beta in (.3,1.,3.):
            specs[f"one_euro_fc{cutoff:g}_beta{beta:g}"] = dict(cutoff=cutoff,beta=beta)
    for threshold in (.05,.1,.2):
        specs[f"innovation_reset_{threshold:g}"] = dict(cutoff=2.,beta=.3,innovation=threshold)
    return specs


def apply_baseline(output, episode, name, spec, clean=False):
    if name == "raw":
        return output, []
    options = {k:v for k,v in spec.items() if k not in ("oracle", "delay")}
    reset = None
    # Oracle controls intervene only on the corrupted branch at the known reveal.
    if not clean and (spec.get("oracle") or "delay" in spec):
        reset = episode["reveal"] + spec.get("delay",0)
    values, resets = filter_stream(pack(output), episode["fps"], reset_at=reset, **options)
    return unpack(values, output), resets


def episode_metrics(clean, corrupted, teacher, episode, geometry):
    ids = PARTS[episode["region"]]
    unaffected = [j for j in range(1,22) if j not in ids]
    def joints(out):
        return forward_joints(body_rotations(out), geometry["rest"], geometry["parents"])
    cj, bj, tj = joints(clean), joints(corrupted), joints(teacher)
    t, fps = episode["reveal"], episode["fps"]
    history = (bj-cj).norm(dim=-1)[:,ids].mean(-1)
    absolute = (bj-tj).norm(dim=-1)[:,ids].mean(-1)
    clean_error = (cj-tj).norm(dim=-1)[:,ids].mean(-1)
    post = slice(t,t+round(fps))
    excess = (absolute-clean_error).clamp_min(0)
    result = dict(history_first_mm=float(history[t]), history_mean_1s_mm=float(history[post].mean()),
                  history_at_100ms_mm=float(history[t+round(.1*fps)]),
                  history_at_300ms_mm=float(history[t+round(.3*fps)]),
                  recovery_5mm_s=recovery_time(history[t:],fps),
                  recovery_10mm_s=recovery_time(history[t:],fps,10),
                  recovery_20mm_s=recovery_time(history[t:],fps,20),
                  recovery_to_raw_5mm_s=None,
                  teacher_joint_agreement_clean_mm=float(clean_error.mean()),
                  teacher_joint_agreement_post_mm=float(absolute[post].mean()),
                  positive_excess_teacher_error_1s_mm=float(excess[post].mean()),
                  unaffected_history_1s_mm=float((bj-cj).norm(dim=-1)[post][:,unaffected].mean()),
                  occluded_history_mm=float(history[episode["onset"]:t].mean()),
                  reveal_step_mm=float((bj[t]-bj[t-1]).norm(dim=-1)[ids].mean()),
                  clean_reveal_step_mm=float((cj[t]-cj[t-1]).norm(dim=-1)[ids].mean()))
    result["teacher_rotation_agreement_clean_deg"] = float(angle_degrees(body_rotations(clean),body_rotations(teacher))[:,ids].mean())
    result.update({f"post_{k}":v for k,v in motion_metrics(bj[post][:,ids],tj[post][:,ids],fps).items()})
    result.update({f"clean_{k}":v for k,v in motion_metrics(cj[:,ids],tj[:,ids],fps).items()})
    return result, dict(history=history.numpy(),absolute=absolute.numpy(),clean_error=clean_error.numpy())


def aggregate(rows):
    result = {}
    excluded = {"episode", "split", "source", "region", "method", "model", "effective_occlusion"}
    for key in rows[0]:
        if key in excluded:
            continue
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            result[key] = dict(mean=float(np.mean(values)),median=float(np.median(values)),
                               p90=float(np.quantile(values,.9)),n=len(values))
    result["unrecovered_5mm_count"] = sum(r["recovery_5mm_s"] is None for r in rows)
    result["episodes"] = len(rows)
    return result


def analyze(args):
    torch.set_num_threads(4)
    plan = json.loads((args.output_dir/"plan.json").read_text())
    geometry = torch.load(args.output_dir/"skeleton.pt", weights_only=True)
    specs, rows, curves, raw_checks, masks = baseline_specs(), [], {}, [], []
    for ep in plan["episodes"]:
        data = torch.load(args.output_dir/"cache"/f"episode_{ep['id']:03d}.pt",weights_only=True)
        raw_checks.append(data["raw_suffix_max_abs"])
        masks.append(float(data["masks"][ep["onset"]:ep["reveal"]].float().mean()))
        network_mask = data["masks"][ep["onset"]:ep["reveal"],:,32:224]
        effective = bool(network_mask.any())
        for model in ("student", "teacher"):
            reference = data["teacher_clean"]
            raw_joints = forward_joints(body_rotations(data[model+"_clean"]),geometry["rest"],geometry["parents"])
            for name,spec in specs.items():
                clean, _ = apply_baseline(data[model+"_clean"],ep,name,spec,clean=True)
                corrupt, resets = apply_baseline(data[model+"_corrupt"],ep,name,spec)
                metrics, curve = episode_metrics(clean,corrupt,reference,ep,geometry)
                joints = forward_joints(body_rotations(corrupt),geometry["rest"],geometry["parents"])
                to_raw = (joints-raw_joints).norm(dim=-1)[:,PARTS[ep["region"]]].mean(-1)
                metrics["recovery_to_raw_5mm_s"] = recovery_time(to_raw[ep["reveal"]:],ep["fps"])
                metrics["reset_count"] = len(resets)
                rows.append(dict(episode=ep["id"],split=ep["split"],source=ep["source"],region=ep["region"],
                                 duration_seconds=ep["duration_seconds"],method=name,model=model,
                                 effective_occlusion=effective,network_mask_area=float(network_mask.float().mean()),**metrics))
                curves[(ep["id"],model,name)] = curve
        print(f"analyzed {ep['id']+1}/{len(plan['episodes'])}",flush=True)
    with (args.output_dir/"per_episode.csv").open("w") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calibration = [r for r in rows if r["split"]=="calibration" and r["model"]=="student" and r["effective_occlusion"]]
    scores = {}
    for name in specs:
        if not (name.startswith("one_euro_") or name.startswith("innovation_")):
            continue
        selected = [r for r in calibration if r["method"]==name]
        scores[name] = float(np.mean([r["teacher_joint_agreement_clean_mm"] +
                                     r["positive_excess_teacher_error_1s_mm"] for r in selected]))
    tuned = min((n for n in scores if n.startswith("one_euro_")),key=scores.get)
    innovation = min((n for n in scores if n.startswith("innovation_")),key=scores.get)
    methods = ["raw","current",tuned,innovation,"oracle_reveal_reset","oracle_reset_delay3"]
    test = [r for r in rows if r["split"]=="test" and r["effective_occlusion"]]
    summaries = {model:{n:aggregate([r for r in test if r["method"]==n and r["model"]==model])
                        for n in methods} for model in ("student","teacher")}
    rng = np.random.default_rng(args.seed)
    uncertainty = {}
    for name in methods:
        values = np.array([r["history_mean_1s_mm"] for r in test if r["method"]==name and r["model"]=="student"])
        means = values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
        uncertainty[name] = [float(x) for x in np.quantile(means,[.025,.975])]
    report = dict(protocol=json.loads((args.output_dir/"provenance.json").read_text()),episodes=len(plan["episodes"]),test_episodes=len(test)//(2*len(specs)),
                  effective_episodes=len({r["episode"] for r in rows if r["effective_occlusion"]}),
                  empty_mask_ids=sorted({r["episode"] for r in rows if not r["effective_occlusion"]}),
                  unique_clean_frames=sum(ep["length"] for ep in plan["episodes"]),
                  reference="PEAR clean-frame teacher, NOT ground truth; neutral fixed-shape SMPL-X 22-joint FK",
                  inherited_error_reference="same estimator and same filter on uncorrupted history",
                  tuned_method=tuned,tuned_innovation=innovation,calibration_scores=scores,
                  calibration_objective="mean clean teacher joint disagreement + positive excess teacher disagreement in first second",
                  raw_suffix_max_abs=max(raw_checks),mask_area_mean=float(np.mean(masks)),
                  zero_mask_episodes=sum(v==0 for v in masks),summaries=summaries,
                  history_mean_1s_mm_bootstrap95=uncertainty,
                  per_region={region:{name:aggregate([r for r in test if r["model"]=="student" and r["region"]==region and r["method"]==name]) for name in methods}
                              for region in PARTS if any(r["region"]==region for r in test)},
                  limits=["Synthetic masks, not verified natural occlusion/reappearance",
                          "No independent ground truth or laptop webcam execution",
                          "Oracle reset uses known mask removal time and is not a deployable visibility detector",
                          "Teacher agreement cannot establish physical pose accuracy",
                          "Quantitative kinematics cover body/arms/legs, not finger or facial recovery",
                          "Calibration/test split is by sequence family, not verified person identity"])
    write_json(args.output_dir/"report.json",report)
    plot_results(args,plan,rows,curves,methods)
    write_report(args,report)


def plot_results(args,plan,rows,curves,methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    rows=[r for r in rows if r["effective_occlusion"]]
    valid_ids={r["episode"] for r in rows}
    for name in methods:
        series=[]
        for ep in plan["episodes"]:
            if ep["split"]=="test" and ep["id"] in valid_ids:
                series.append(curves[(ep["id"],"student",name)]["history"][ep["reveal"]:ep["reveal"]+45])
        axes[0].plot(np.arange(45)/30,np.mean(series,0),label=name)
        summary=aggregate([r for r in rows if r["split"]=="test" and r["model"]=="student" and r["method"]==name])
        axes[1].scatter(summary["clean_velocity_error_mm_s"]["mean"],summary["history_mean_1s_mm"]["mean"],label=name)
    axes[0].set(xlabel="Seconds after synthetic reveal",ylabel="Inherited joint difference (mm)",title="Matched clean-history control")
    axes[1].set(xlabel="Clean velocity disagreement with teacher (mm/s)",ylabel="Mean inherited difference in first second (mm)",title="Recovery versus motion agreement")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir/"recovery_curves.png",dpi=160)
    plt.close(fig)
    for region in PARTS:
        test=[r for r in rows if r["split"]=="test" and r["model"]=="student" and r["method"]=="current" and r["region"]==region]
        ep_id=max(test,key=lambda r:r["history_mean_1s_mm"])["episode"] if test else 0
        ep=plan["episodes"][ep_id]
        fig,ax=plt.subplots(figsize=(9,4))
        for method in methods:
            values=curves[(ep_id,"student",method)]["history"]
            ax.plot((np.arange(len(values))-ep["reveal"])/30,values,label=method)
        ax.axvspan(-ep["duration_seconds"],0,color="gray",alpha=.15)
        ax.set(xlabel="Seconds relative to reveal",ylabel="Inherited difference (mm)",title=f"Worst test {region} episode {ep_id} (selected by current-filter error)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir/f"example_{region}.png",dpi=130)
        plt.close(fig)


def write_report(args,r):
    lines=["# PEAR 235000 paired-history recovery diagnostic", "",
           "This is a no-training synthetic-occlusion pilot, not proof of an unsolved tracking problem.", "",
           f"Episodes: {r['episodes']}; held-out test episodes: {r['test_episodes']}. FP32, stride one, recorded 30 Hz timestamps.",
           f"Effective occlusions: {r['effective_episodes']}; empty-mask controls excluded from aggregates: {r['empty_mask_ids']}. Unique clean frames: {r['unique_clean_frames']}.",
           f"Raw student's maximum identical-suffix parameter difference: {r['raw_suffix_max_abs']:.9g}.",
           "", "## Held-out student results", "",
           "All mm values are fixed-neutral-shape joint differences, not ground-truth MPJPE.", "",
           "| Method | Reveal history difference (mm) | Mean history difference, first 1 s (mm) | Median recovery to 5 mm (ms) | Unrecovered | Clean teacher joint disagreement (mm) | Clean velocity disagreement (mm/s) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,s in r["summaries"]["student"].items():
        rt=s.get("recovery_5mm_s",{}).get("median")
        rt="n/a" if rt is None else f"{rt*1000:.1f}"
        lines.append(f"| {name} | {s['history_first_mm']['mean']:.2f} | {s['history_mean_1s_mm']['mean']:.2f} | {rt} | {s['unrecovered_5mm_count']}/{s['episodes']} | {s['teacher_joint_agreement_clean_mm']['mean']:.2f} | {s['clean_velocity_error_mm_s']['mean']:.2f} |")
    lines += ["", "## Interpretation", "",
              "The raw framewise estimator is a negative control. Its suffix is rerun through the network, not copied from the clean predictions.",
              "Inherited error compares each filtered corrupted run with that SAME filter's clean-history run. Recovery to this control does not mean recovery to physical truth or to an unsmoothed signal.",
              "The CSV additionally reports recovery to the raw clean estimator, teacher disagreement, amplitude, direction, lag, threshold sensitivity, unaffected-part differences, and reveal-step size.",
              "Oracle reset and delayed oracle reset use the known synthetic reveal timestamp only on the corrupted branch. They are controls, not deployable visibility estimators.",
              "Tuned One-Euro and innovation-reset parameters were selected on calibration clips only; their selection objective is recorded in report.json.",
              "", "## Limitations", ""] + [f"- {v}" for v in r["limits"]]
    lines += ["", "## Artifacts", "", "- `report.json`: aggregate results and protocol.",
              "- `per_episode.csv`: every episode/method result, including non-recoveries.",
              "- `plan.json`: exact frame ranges, corruption durations, and calibration/test assignments.",
              "- `provenance.json`: checkpoint hashes before/after and protected-module hashes.",
              "- `recovery_curves.png`: test recovery curves and clean-motion comparison.",
              "- `cache/`: raw teacher/student predictions, mask pixels, and projections.",
              "- `visuals/`: diagnostic videos if the render stage was run.", "",
              "No training, checkpoint mutation, or modifications to scout, temporal predictor, or router."]
    (args.output_dir/"SUMMARY.md").write_text("\n".join(lines)+"\n")


def label(frame,text):
    panel=frame.copy()
    cv2.rectangle(panel,(0,0),(panel.shape[1],27),(245,245,245),-1)
    cv2.putText(panel,text,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1,cv2.LINE_AA)
    return panel


def render(args):
    from tools.compare_teacher_student import initialize_guava_renderer, render_output
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    plan=json.loads((args.output_dir/"plan.json").read_text())
    report=json.loads((args.output_dir/"report.json").read_text())
    with (args.output_dir/"per_episode.csv").open() as f:
        rows=list(csv.DictReader(f))
    chosen=[]
    for region in PARTS:
        candidates=[r for r in rows if r["split"]=="test" and r["model"]=="student" and r["method"]=="current" and r["region"]==region]
        if candidates:
            chosen.append(int(max(candidates,key=lambda r:float(r["history_mean_1s_mm"]))["episode"]))
    chosen=chosen[:args.render_clips]
    specs=baseline_specs()
    methods=["raw","current",report["tuned_method"],"oracle_reveal_reset"]
    outdir=args.output_dir/"visuals"
    outdir.mkdir(exist_ok=True)
    geometry=torch.load(args.output_dir/"skeleton.pt",weights_only=True)
    first=torch.load(args.output_dir/"cache"/f"episode_{chosen[0]:03d}.pt",weights_only=True)
    runtime=SimpleNamespace(source_data_path=ROOT/"assets/example/tracked_image/NTFbJBzjlts__047",
                            model_path=ROOT/"assets/GUAVA",device=args.device,render_size=256)
    renderer,builder,dataset=initialize_guava_renderer(runtime,tree_index(first["teacher_clean"],0,args.device))
    selections=[]
    try:
        for ep_id in chosen:
            ep=plan["episodes"][ep_id]
            data=torch.load(args.output_dir/"cache"/f"episode_{ep_id:03d}.pt",weights_only=True)
            outputs={name:apply_baseline(data["student_corrupt"],ep,name,specs[name])[0] for name in methods}
            outputs["current_clean_control"] = apply_baseline(data["student_clean"],ep,"current",specs["current"],clean=True)[0]
            joints={name:forward_joints(body_rotations(out),geometry["rest"],geometry["parents"]).numpy() for name,out in outputs.items()}
            teacher_j=forward_joints(body_rotations(data["teacher_clean"]),geometry["rest"],geometry["parents"]).numpy()
            # One fixed projection/scale for the entire episode and all methods.
            all_xy=np.concatenate([teacher_j[...,:2]]+[x[...,:2] for x in joints.values()],axis=0)
            lo,hi=all_xy.min((0,1)),all_xy.max((0,1))
            scale=205/max(hi-lo)
            center=(lo+hi)/2
            frames=[]
            snapshots=[]
            render_differences=[]
            for t in range(ep["length"]):
                photo=cv2.imread(str(Path(ep["directory"])/f"{ep['start']+t:06d}.jpg"))
                photo[data["masks"][t].numpy().astype(bool)]=128
                state="OCCLUDED" if ep["onset"]<=t<ep["reveal"] else "VISIBLE"
                top=[label(photo,f"Input {state} {(t-ep['reveal'])/ep['fps']:+.2f}s")]
                bottom=[]
                renders={}
                for name in ["teacher","current_clean_control"]+methods:
                    output=data["teacher_clean"] if name=="teacher" else outputs[name]
                    with torch.inference_mode():
                        avatar=render_output(renderer,builder,tree_index(output,t,args.device))
                    renders[name]=avatar
                    top.append(label(cv2.resize(avatar,(256,256),interpolation=cv2.INTER_AREA),"Clean teacher" if name=="teacher" else name))
                    joint=teacher_j[t] if name=="teacher" else joints[name][t]
                    xy=(joint[:,:2]-center)*scale+np.array([128,145])
                    panel=np.full((256,256,3),248,dtype=np.uint8)
                    for a,b in EDGES:
                        cv2.line(panel,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(90,90,90),2,cv2.LINE_AA)
                    for j in PARTS[ep["region"]]:
                        cv2.circle(panel,tuple(xy[j].astype(int)),3,(40,60,220),-1,cv2.LINE_AA)
                    bottom.append(label(panel,"Fixed-shape body: "+name))
                blank=np.full((256,256,3),248,dtype=np.uint8)
                cv2.putText(blank,f"Episode {ep_id}: {ep['region']}",(7,85),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
                cv2.putText(blank,"Skeleton: fixed projection",(7,112),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
                cv2.putText(blank,"NOT input-image overlay",(7,139),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
                panel=np.vstack([np.hstack(top),np.hstack([blank]+bottom)])
                frames.append(panel)
                render_differences.append(float(np.abs(renders["current"].astype(np.float32)-renders["current_clean_control"].astype(np.float32)).mean()/255))
                if t in (ep["reveal"]-1,ep["reveal"],ep["reveal"]+3,ep["reveal"]+9):
                    snapshots.append(panel)
            write_video(outdir/f"episode_{ep_id:03d}_{ep['region']}.mp4",frames,ep["fps"])
            cv2.imwrite(str(outdir/f"episode_{ep_id:03d}_contact_sheet.jpg"),np.vstack(snapshots))
            selections.append(dict(episode=ep_id,region=ep["region"],selection="largest test current-filter inherited error within region",
                                   current_filter_render_l1_at_reveal=render_differences[ep["reveal"]],
                                   current_filter_render_l1_first_second=float(np.mean(render_differences[ep["reveal"]:ep["reveal"]+30])),
                                   current_filter_render_l1_by_frame=render_differences))
            print(f"rendered episode {ep_id}",flush=True)
    finally:
        dataset._lmdb_engine.close()
    write_json(outdir/"selection.json",selections)


def main():
    args=arguments()
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="infer":
        run_inference(args)
    elif args.stage=="analyze":
        analyze(args)
    else:
        render(args)


if __name__=="__main__":
    main()
