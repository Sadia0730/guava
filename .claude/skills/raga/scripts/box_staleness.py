"""usage: python box_staleness.py REC_DIR
How far off is a fixed or reused person box, measured on the boxes YOLOX produced in a recorded session.
SPRINT's crop = square, side 1.1 x the box's long side, centred on the box (box_crop_matrix)."""
import json, sys, numpy as np
d = sys.argv[1]
L = [json.loads(l) for l in open(f"{d}/frames.jsonl")]
S = json.load(open(f"{d}/session.json"))
t = np.array([r["t_capture"] for r in L]); B = np.array([r["box"] for r in L], np.float64)

def crop(b):                       # (cx, cy, side)
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.1 * max(b[2] - b[0], b[3] - b[1])])

def sq(c):                         # crop square as l, t, r, b
    return np.array([c[0] - c[2] / 2, c[1] - c[2] / 2, c[0] + c[2] / 2, c[1] + c[2] / 2])

def inter(p, q):
    return max(0., min(p[2], q[2]) - max(p[0], q[0])) * max(0., min(p[3], q[3]) - max(p[1], q[1]))

def area(p): return (p[2] - p[0]) * (p[3] - p[1])

def compare(stale_box, fresh_box):
    cs, cf = crop(stale_box), crop(fresh_box); ss, sf = sq(cs), sq(cf)
    i = inter(ss, sf); iou = i / (area(ss) + area(sf) - i)
    inside = inter(ss, fresh_box) / area(fresh_box)            # share of the person (fresh box) inside the stale crop
    shift = np.hypot(*(cs[:2] - cf[:2])) / cf[2]                # centre offset, fraction of the crop side
    scale = cs[2] / cf[2]
    return iou, inside, shift, scale

def summary(name, rows):
    r = np.array(rows)
    print(f"{name:<34} n={len(r):4d}  crop IoU p50 {np.median(r[:,0]):.3f} p5 {np.percentile(r[:,0],5):.3f}  "
          f"person inside crop: mean {r[:,1].mean()*100:5.1f}%  min {r[:,1].min()*100:5.1f}%  frames <99%: {(r[:,1]<0.99).mean()*100:4.1f}%  "
          f"centre shift p95 {np.percentile(r[:,2],95)*100:4.1f}%  scale p5-p95 {np.percentile(r[:,3],5):.2f}-{np.percentile(r[:,3],95):.2f}")

ref = np.array(S["source_box"], np.float64)
summary("fixed box = reference-frame box", [compare(ref, b) for b in B])
for lo, hi, lab in [(80, 120, "~3"), (180, 220, "~6"), (300, 360, "~10"), (450, 550, "~15"), (900, 1100, "~30")]:
    rows = []
    for a in range(len(L)):
        for b in range(a + 1, len(L)):
            g = (t[b] - t[a]) * 1000
            if g > hi: break
            if g >= lo: rows.append(compare(B[a], B[b]))
    summary(f"box reused {lo}-{hi} ms (= {lab} frames @30fps)", rows)
w, h = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
print(f"\nperson box: width p5-p95 {np.percentile(w,5):.0f}-{np.percentile(w,95):.0f} px, height {np.percentile(h,5):.0f}-{np.percentile(h,95):.0f} px (frame 1280x720)")
cx = (B[:, 0] + B[:, 2]) / 2; print(f"box centre x p5-p95 {np.percentile(cx,5):.0f}-{np.percentile(cx,95):.0f} px; reference box centre x {(ref[0]+ref[2])/2:.0f}")
