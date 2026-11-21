#!/usr/bin/env python3
"""
simdist.py - how detection depends on the DISTANCE to the defect (uses the survey frames, no flying).

  simdist                                        # models B and E5, clean + A11T typical
  simdist bridge5_y11s.pt --looks typical,bad

For every frame taken while scanning a defect: distance camera -> defect wall, size of the defect
patch in the image (pixels), was the defect detected, and was the class right.
Results: <survey>/simdist_results.csv
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "drone_ws/scripts"))
import pilot
from analog_cam import AnalogFX
from simtest import newest_survey, MODELS

BINS = [0.0, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00, 1.30, 9.0]       # metres from the wall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=["bridge5_y11s.pt", "bridge5_y11s_droneview.pt"])
    ap.add_argument("--survey", type=Path, default=None)
    ap.add_argument("--looks", default="clean,typical")
    a = ap.parse_args()
    survey = a.survey.expanduser() if a.survey else newest_survey()
    rows = [r for r in csv.DictReader(open(survey / "frames.csv")) if r["scan_id"]]
    cam = json.loads((survey / "camera.json").read_text())
    pilot.HFOV = cam["hfov"]
    pilot.FOCAL = (pilot.IMG_W / 2) / math.tan(pilot.HFOV / 2)
    pilot.LENS = None                                              # survey without lens = pinhole frames
    if cam.get("lens"):                                            # survey recorded through the real lens
        from fisheye_cam import Lens
        pilot.LENS = Lens(cam["lens"], cam["render_hfov"])
        pilot.FOCAL = pilot.LENS.fx
    decals, _, boxes = pilot.load_world()
    by_id = {d["id"]: d for d in decals}
    for r in rows:
        dc = by_id[r["scan_id"]]
        pos = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
        r["_d"] = float((pos - dc["_c"]) @ dc["_n"])                # distance to the wall
        r["_px"] = dc["width"] * pilot.FOCAL / max(r["_d"], 0.05)   # patch width in the image
        r["_bin"] = next(i for i in range(len(BINS) - 1) if r["_d"] < BINS[i + 1])
    print(f"Survey: {survey}\n  {len(rows)} scan frames, lens {math.degrees(pilot.HFOV):.0f} deg")

    from ultralytics import YOLO
    looks = [x.strip() for x in a.looks.split(",") if x.strip()]
    results = {}                                                      # (model, look) -> bin -> [n, det, right]
    for mname in a.models:
        mp = Path(mname).expanduser()
        mp = mp if mp.exists() else MODELS / mname
        if not mp.exists():
            print(f"!! model not found: {mname} - skipped")
            continue
        model = YOLO(str(mp))
        names = {i: n.lower() for i, n in model.names.items()}
        for look in looks:
            print(f"  testing {mp.name:28s} camera: {'clean' if look == 'clean' else 'A11T ' + look} ...")
            fx = None if look == "clean" else AnalogFX(look, seed=0)
            st = defaultdict(lambda: [0, 0, 0])
            for r in rows:
                img = cv2.imread(str(survey / r["file"]))
                if fx:
                    img = fx(img)
                res = model.predict(img, imgsz=640, conf=pilot.CONF, device=0, verbose=False)[0]
                pos, yaw = np.array([float(r["x"]), float(r["y"]), float(r["z"])]), float(r["yaw"])
                best = None
                for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                                       res.boxes.cls.tolist()):
                    o, d = pilot.pixel_ray((x1 + x2) / 2, (y1 + y2) / 2, pos, yaw)
                    hit = pilot.find_decal(o, d, decals, boxes)
                    if hit is not None and hit["id"] == r["scan_id"] and (best is None or conf > best[1]):
                        best = (names[int(cls)], conf)
                s = st[r["_bin"]]
                s[0] += 1
                if best:
                    s[1] += 1
                    s[2] += best[0] == by_id[r["scan_id"]]["class"]
            results[(mp.name, look)] = st

    keys = list(results)
    print("\n================ DETECTION vs DISTANCE  (detected % | right class %) ================")
    print(f"{'distance':12s}{'frames':>7s}{'patch px':>10s}" +
          "".join(f"{(m.replace('bridge5_', '').replace('.pt', '') + '/' + l)[:21]:>23s}" for m, l in keys))
    out = []
    for b in range(len(BINS) - 1):
        fr = [r for r in rows if r["_bin"] == b]
        if not fr:
            continue
        label = f"{BINS[b]:.2f}-{BINS[b + 1]:.2f} m" if BINS[b + 1] < 9 else f">{BINS[b]:.2f} m"
        px = int(np.median([r["_px"] for r in fr]))
        cells = []
        for k in keys:
            n, det, right = results[k][b]
            cells.append(f"{100 * det / n:5.0f}% | {100 * right / max(det, 1):3.0f}%" if n >= 10 else "few frames")
            out.append({"model": k[0], "camera": k[1], "distance": label, "frames": n, "patch_px": px,
                        "detected_%": round(100 * det / max(n, 1), 1),
                        "right_class_%": round(100 * right / max(det, 1), 1)})
        print(f"{label:12s}{len(fr):7d}{px:10d}" + "".join(f"{c:>23s}" for c in cells))
    with open(survey / "simdist_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)
    print(f"\nSaved: {survey}/simdist_results.csv")


if __name__ == "__main__":
    main()
