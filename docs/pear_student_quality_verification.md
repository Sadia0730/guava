# PEAR Student Quality Verification

Use three evaluation levels.  A student can look good on parameter L1 and still
drive GUAVA badly, so the final decision should be render-aware and temporal.

## 1. Teacher Agreement

Run the frozen PEAR teacher and the student on the same held-out frames.

Core metrics:

| Area | Metric | What Good Looks Like |
|---|---|---:|
| Global pose | geodesic rotation error | < 2-3 degrees |
| Body pose | mean joint geodesic error | < 4-5 degrees |
| Hands | mean hand-joint geodesic error | < 8-10 degrees |
| Jaw | geodesic rotation error | < 3-5 degrees |
| Expression | L1 on expression params | compare against PEAR-to-PEAR noise |
| Camera | translation L1 + crop/scale drift | no visible avatar scale jump |

Rotation geodesic error:

```python
angle = acos(clamp((trace(R_student @ R_teacher.T) - 1) / 2, -1, 1))
```

Do not average everything into one number only.  Report body, hands, face, jaw,
and camera separately.

## 2. GUAVA Render Agreement

This is the most important test for your paper.

For each held-out driving frame:

1. Use the same frozen EHM-tracked source avatar.
2. Render the avatar once with PEAR teacher parameters.
3. Render the avatar once with student parameters.
4. Compare the two renders.

Metrics:

| Area | Metric |
|---|---|
| Full avatar | L1, PSNR, SSIM, LPIPS |
| Head crop | L1 + LPIPS |
| Mouth/jaw crop | L1 + LPIPS |
| Hands crop | L1 + LPIPS |
| Silhouette | alpha IoU |

Good student behavior:

| Metric | Target |
|---|---:|
| Full-render LPIPS vs teacher | < 0.03-0.05 |
| Head crop LPIPS vs teacher | < 0.05-0.07 |
| Silhouette IoU vs teacher | > 0.95 |
| Visible mouth/hand failures | rare, not systematic |

For the paper, show side-by-side videos:

```text
input frame | PEAR teacher driven GUAVA | student driven GUAVA | error heatmap
```

## 3. Temporal Quality

Framewise students often jitter even when single-frame error is low.

Measure on videos:

| Metric | Formula |
|---|---|
| Parameter velocity error | `L1(delta student params, delta teacher params)` |
| Joint velocity error | `L1(delta projected joints)` |
| Render flicker | `LPIPS(render_t, render_t-1)` after motion compensation if possible |
| Acceleration/jitter | `mean(abs(x[t+1] - 2*x[t] + x[t-1]))` |

Good result:

```text
Student should be close to teacher in motion,
and ideally less jittery than teacher after temporal training.
```

## 4. Live Visual Check

Use display mode for visual checking:

```bash
python main/live_pear_guava.py \
  --input 0 \
  --render_size 256 \
  --precision fp16 \
  --compile_targets pear refiner \
  --pipeline async \
  --pear_stride 1 \
  --window 60
```

Use benchmark mode only for speed:

```bash
python main/live_pear_guava.py \
  --input 0 \
  --render_size 256 \
  --precision fp16 \
  --compile_targets pear refiner \
  --pipeline async \
  --pear_stride 1 \
  --window 60 \
  --no_display \
  --max_frames 600
```

`--no_display` is not a quality test.  It only removes preview overhead so the
FPS number is cleaner.

## Final Acceptance Rule

Call the student successful only if it passes all four:

| Requirement | Target |
|---|---:|
| Live throughput | >= 30 FPS on target laptop |
| Full-render quality | visually close to PEAR teacher |
| Head/jaw quality | no obvious expression loss |
| Temporal stability | no more jitter than PEAR, ideally less |
