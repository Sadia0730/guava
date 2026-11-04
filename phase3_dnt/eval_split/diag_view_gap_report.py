import json, sys
import numpy as np

rows = json.load(open(sys.argv[1]))
BINS = [(0, 30), (30, 60), (60, 90), (90, 135), (135, 181)]
M = ["guava", "fb_gt", "fb_sprint", "gart", "gaussianavatar", "exavatar_gtface"]
for ds in ["peoplesnapshot", "xhumans", "bedlam2", "neuman", "ubody"]:
    R = [r for r in rows if r["ds"] == ds]
    print(f"\n== {ds}: {len(R)} frames, view change median {np.median([r['view'] for r in R]):.0f} deg")
    print("bin(deg)   n   " + "  ".join(f"{m[:10]:>10s}" for m in M) + "   FB[GT]-GART  FB[GT]-GUAVA")
    for lo, hi in BINS:
        B = [r for r in R if lo <= r["view"] < hi]
        if not B: continue
        cells = []
        for m in M:
            v = [r[m] for r in B if m in r]
            cells.append(f"{np.mean(v):10.2f}" if len(v) == len(B) else f"{'-':>10s}")
        both = [r for r in B if "fb_gt" in r and "gart" in r]
        d = np.mean([r["fb_gt"] - r["gart"] for r in both]) if both else np.nan
        dg = np.mean([r["fb_gt"] - r["guava"] for r in B if "fb_gt" in r and "guava" in r])
        print(f"{lo:3d}-{hi - 1:3d} {len(B):5d}  " + "  ".join(cells) + f"   {d:+8.2f}     {dg:+8.2f}")
