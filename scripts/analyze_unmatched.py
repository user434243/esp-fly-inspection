#!/usr/bin/env python3
"""
analyze_unmatched.py - what did the model react to when its detection hit NO defect patch?

For every logged 'unmatched' detection it re-casts the ray from the drone camera through the
box centre and finds the first surface it hits (steel bearing, weathered concrete, girder,
grass, ...). That separates 'the model saw rust on a rusty steel bearing' (real) from
'the model saw a defect on plain concrete' (false alarm or texture stain).

Run:  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/analyze_unmatched.py   (newest pilot run)
      ~/drone_ws/venv/bin/python ~/drone_ws/scripts/analyze_unmatched.py ~/drone_ws/logs/pilot_...
"""
import csv
import glob
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot                                              # same camera model + bridge geometry

SURFACE = {"steel": "steel bearings (rusty steel)", "weathered": "weathered concrete (pier, abutments, culvert)",
           "concrete": "girders / deck / parapets", "grass": "grass (ground, embankments)",
           "gravel": "river-bed gravel", "asphalt": "road", None: "nothing (sky / far away)"}

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(
    p for p in glob.glob(os.path.expanduser("~/drone_ws/logs/pilot_*")) if os.path.isfile(os.path.join(p, "unmatched.csv")))[-1])
rows = list(csv.DictReader(open(run / "unmatched.csv")))
_, parts, _ = pilot.load_world()
solid = [(np.array(p["lo"]), np.array(p["hi"]), p) for p in parts]

table, conf_sum, by_part = defaultdict(Counter), defaultdict(float), Counter()
for r in rows:
    u = (float(r["x1"]) + float(r["x2"])) / 2
    v = (float(r["y1"]) + float(r["y2"])) / 2
    o, d = pilot.pixel_ray(u, v, (float(r["x"]), float(r["y"]), float(r["z"])), float(r["yaw"]))
    best = None
    for lo, hi, p in solid:
        t = pilot.ray_box(o, d, lo, hi)
        if t is not None and t > 0.02 and (best is None or t < best[0]):
            best = (t, p)
    mat = best[1]["mat"] if best and best[0] < 15 else None
    table[mat][r["class"]] += 1
    conf_sum[mat] += float(r["conf"])
    if best:
        by_part[best[1]["name"].split("_")[0]] += 1

classes = ["corrosion_stain", "spallation", "crack", "exposed_bars", "efflorescence"]
print(f"\nRun: {run}\n{len(rows)} detections that hit no defect patch\n")
print(f"{'surface hit':46s}" + "".join(f"{c[:10]:>11s}" for c in classes) + f"{'total':>8s}{'avg conf':>10s}")
for mat in sorted(table, key=lambda m: -sum(table[m].values())):
    n = sum(table[mat].values())
    print(f"{SURFACE.get(mat, mat):46s}" + "".join(f"{table[mat][c]:11d}" for c in classes)
          + f"{n:8d}{conf_sum[mat] / n:10.2f}")
print("\nmost-hit structure parts:", ", ".join(f"{k} {v}" for k, v in by_part.most_common(8)))
