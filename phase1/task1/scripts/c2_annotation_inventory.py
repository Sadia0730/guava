#!/usr/bin/env python
"""Checkpoint 2, section 1: inventory of the BEDLAM2 (CameraHMR-processed) labels.

Read-only. For every label file: rows, images, sequences, people per image, frame sampling,
field shapes, testset overlap (from the BEDLAM2 SQLite DB), and coverage of our manifest.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from pathlib import Path

import numpy as np

B = Path("/raid/ubx858/datasets/bedlam2_gt")
LABELS = B / "expanded_labels" / "bedlam2_labels_processed"
MANIFESTS = Path("/raid/ubx858/datasets/processed/pear_student")


def main():
    db = sqlite3.connect(B / "render_db" / "bedlam2.sqlite")
    seq_flags = {(job, sid): (test, aff, nb) for job, sid, test, aff, nb in
                 db.execute("select renderjob, sequence_id, testset, affected, num_bodies from sequence")}
    jobs_db = {r[0]: r for r in db.execute("select name, num_sequences, num_images from renderjob")}

    manifest_jobs = collections.Counter()
    manifest_split = {}
    for split in ("train", "val"):
        for line in open(MANIFESTS / f"{split}.jsonl"):
            item = json.loads(line)
            if item["source"] == "bedlam":
                job, seq = item["sequence"].split("/")
                manifest_jobs[job] += 1
                manifest_split[(job, int(seq.split("_")[-1]))] = split

    shapes, rows_total, per_job = {}, 0, {}
    for path in sorted(LABELS.glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        if not shapes:
            shapes = {k: list(d[k].shape) for k in d.files}
        names = d["imgname"]
        seq_ids = np.array([int(n.split("/")[0].split("_")[-1]) for n in names])
        frames = np.array([int(n.split("_")[-1].split(".")[0]) for n in names])
        per_image = collections.Counter(names)
        job = path.stem
        steps = []
        for sid in np.unique(seq_ids)[:20]:
            f = np.unique(frames[seq_ids == sid])
            if len(f) > 1:
                steps.extend(np.diff(f).tolist())
        flags = [seq_flags.get((job, int(s)), (None, None, None)) for s in np.unique(seq_ids)]
        per_job[job] = {
            "rows": int(len(names)),
            "images": int(len(per_image)),
            "sequences": int(len(np.unique(seq_ids))),
            "max_people_per_image": int(max(per_image.values())),
            "frame_step_mode": int(collections.Counter(steps).most_common(1)[0][0]) if steps else None,
            "testset_sequences_in_labels": int(sum(1 for f in flags if f[0] == 1)),
            "affected_sequences_in_labels": int(sum(1 for f in flags if f[1] == 1)),
            "db_sequences": int(jobs_db[job][1]) if job in jobs_db else None,
            "db_testset_sequences": int(sum(1 for (j, _), f in seq_flags.items() if j == job and f[0] == 1)),
            "in_our_manifest_sequences": int(manifest_jobs.get(job, 0)),
            "genders": sorted(set(map(str, np.unique(d["gender"])))),
            "single_person_job": job.split("_")[1] == "1",
        }
        rows_total += len(names)

    summary = {
        "label_files": len(per_job),
        "rows_total": rows_total,
        "field_shapes_first_file": shapes,
        "jobs_in_db": len(jobs_db),
        "db_jobs_without_labels": sorted(set(jobs_db) - set(per_job)),
        "label_jobs_not_in_db": sorted(set(per_job) - set(jobs_db)),
        "manifest_jobs_without_labels": sorted(set(manifest_jobs) - set(per_job)),
        "single_person": {
            "jobs": sum(v["single_person_job"] for v in per_job.values()),
            "rows": sum(v["rows"] for v in per_job.values() if v["single_person_job"]),
        },
        "multi_person": {
            "jobs": sum(not v["single_person_job"] for v in per_job.values()),
            "rows": sum(v["rows"] for v in per_job.values() if not v["single_person_job"]),
            "images": sum(v["images"] for v in per_job.values() if not v["single_person_job"]),
        },
        "testset_sequences_present_in_labels": sum(v["testset_sequences_in_labels"] for v in per_job.values()),
        "manifest_sequences_flagged_testset": {
            split: sum(1 for (key, s) in manifest_split.items() if s == split and seq_flags.get(key, (0,))[0] == 1)
            for split in ("train", "val")},
        "manifest_sequences_flagged_affected": {
            split: sum(1 for (key, s) in manifest_split.items() if s == split and seq_flags.get(key, (0, 0))[1] == 1)
            for split in ("train", "val")},
        "manifest_sequences": {split: sum(1 for s in manifest_split.values() if s == split) for split in ("train", "val")},
        "per_job": per_job,
    }
    out = Path("/raid/ubx858/outputs/phase1_task1/checkpoint2/annotation_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_job"}, indent=2))


if __name__ == "__main__":
    main()
