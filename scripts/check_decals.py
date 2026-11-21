#!/usr/bin/env python3
"""
check_decals.py - can the model see each defect patch at all, and at what size does it lose it?

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/check_decals.py

For every patch of the ACTIVE world (simworld) it runs the world's YOLO model on the patch photo:
  full   the photo alone, full size (like the training images)
  70% / 50% / 40%   the photo shrunk so its side is that share of 480 px, on a grey road-like
                    background (40% = about what the camera sees from the 1.4 m stop)
Shows the best detection of the TRUE class (and the best other class if it is stronger).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "drone_ws/scripts"))
sys.path.insert(0, str(HOME / "drone_ws/sim/worlds"))
import pilot
from ultralytics import YOLO

SIZES = (0.7, 0.5, 0.4)


def best(model, names, img, truth):
    r = model.predict(img, conf=0.05, device=0, verbose=False)[0]
    t, o = 0.0, (0.0, "")
    for c, k in zip(r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        n = names[int(k)]
        if n == truth:
            t = max(t, c)
        elif c > o[0]:
            o = (c, n)
    s = f"{t:.2f}" if t else " -- "
    if o[0] > t:
        s += f" ({o[1][:5]} {o[0]:.2f})"
    return s


def main():
    decals = json.loads(pilot.DECALS.read_text())["decals"]
    model = YOLO(str(pilot.MODEL))
    names = {i: n.lower() for i, n in model.names.items()}
    print(f"world {pilot.WORLD_NAME}   model {pilot.MODEL.name}   (conf of the TRUE class, '--' = not seen)\n")
    print(f"{'id':4s} {'truth':19s} {'full':>22s}" + "".join(f"{f'{int(s * 100)}%':>22s}" for s in SIZES))
    for dc in decals:
        img = cv2.imread(str(pilot.WORLD_DIR / "decals" / f"{dc['id']}.jpg"))
        row = [best(model, names, img, dc["class"])]
        bg = np.full((480, 640, 3), np.median(img.reshape(-1, 3), axis=0), np.uint8)
        for s in SIZES:
            side = int(480 * s)
            canvas = bg.copy()
            y0, x0 = (480 - side) // 2, (640 - side) // 2
            canvas[y0:y0 + side, x0:x0 + side] = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
            row.append(best(model, names, canvas, dc["class"]))
        print(f"{dc['id']:4s} {dc['class']:19s}" + "".join(f"{x:>22s}" for x in row))


if __name__ == "__main__":
    main()
