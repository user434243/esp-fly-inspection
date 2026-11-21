#!/usr/bin/env python3
"""
simtest.py - judge YOLO models on the SAME drone-view frames (no flying, same views for every model).

  simtest                                   # newest survey, the 3 bridge models, clean + A11T look
  simtest bridge5_y11s.pt bridge5_y11s_analog.pt --looks typical,bad
  simtest --survey ~/drone_ws/logs/auto_20261021_190000/survey

Frames come from a survey flight (SURVEY=1 autopilot): clean camera images + drone position/heading.
For the "A11T" looks the analog effect is added here, with the same random seed for every model,
so every model sees exactly the same pictures. Same matching rules as the mission report:
a detection counts for a defect when its 3D ray hits that defect patch; class = confidence vote.
Results: <survey>/simtest_results.csv  and  <survey>/simtest_per_defect.csv
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "drone_ws/scripts"))
import pilot                                    # world + camera geometry, same as the missions
from analog_cam import AnalogFX

MODELS = HOME / "drone_ws/yolo/models"
DEFAULT = ["bridge5_y11s.pt", "bridge5_y26s.pt", "bridge5_y11s_analog.pt"]
PASSING_WEIGHT = 0.25                           # same as autopilot.py
FOUND_HITS = 3                                  # matched detections needed to call a defect "found"


def newest_survey():
    found = sorted((HOME / "drone_ws/logs").glob("auto_*/survey/frames.csv"))
    if not found:
        sys.exit("No survey found - fly one first:   SURVEY=1 autopilot")
    return found[-1].parent


def run(model, rows, survey, look, decals, boxes):
    names = {i: n.lower() for i, n in model.names.items()}
    fx = None if look == "clean" else AnalogFX(look, seed=0)       # same noise for every model
    votes, hits = defaultdict(Counter), Counter()
    frames_target, frames_det, frames_right, off_high = Counter(), Counter(), Counter(), 0
    truth = {d["id"]: d["class"] for d in decals}
    for r in rows:
        img = cv2.imread(str(survey / r["file"]))
        if fx:
            img = fx(img)
        res = model.predict(img, imgsz=640, conf=pilot.CONF, device=0, verbose=False)[0]
        pos, yaw, sid = np.array([float(r["x"]), float(r["y"]), float(r["z"])]), float(r["yaw"]), r["scan_id"]
        best = None                                                # best detection on the scanned defect
        for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                               res.boxes.cls.tolist()):
            o, d = pilot.pixel_ray((x1 + x2) / 2, (y1 + y2) / 2, pos, yaw)
            hit = pilot.find_decal(o, d, decals, boxes)
            if hit is None:
                off_high += conf >= pilot.ALARM_CONF
                continue
            name = names[int(cls)]
            votes[hit["id"]][name] += conf * (1.0 if hit["id"] == sid else PASSING_WEIGHT) * (1.0 if conf >= pilot.CLASS_CONF else pilot.WEAK_VOTE)
            hits[hit["id"]] += 1
            if hit["id"] == sid and (best is None or conf > best[1]):
                best = (name, conf)
        if sid:
            frames_target[sid] += 1
            if best:
                frames_det[sid] += 1
                frames_right[sid] += best[0] == truth[sid]
    per = []
    for dc in decals:
        did = dc["id"]
        pred = votes[did].most_common(1)[0][0] if hits[did] >= FOUND_HITS else "none"
        result = ("MISSED" if pred == "none" else "correct" if pred == dc["class"]
                  else "partly correct" if pred in dc.get("also_present", []) else "WRONG CLASS")
        per.append({"id": did, "truth": dc["class"], "predicted": pred, "result": result,
                    "scan_frames": frames_target[did], "frames_detected": frames_det[did],
                    "frames_right_class": frames_right[did]})
    n_t, n_d, n_r = sum(frames_target.values()), sum(frames_det.values()), sum(frames_right.values())
    summary = {"found": sum(p["result"] != "MISSED" for p in per),
               "correct": sum(p["result"] == "correct" for p in per),
               "partly": sum(p["result"] == "partly correct" for p in per),
               "frames_detected_%": round(100 * n_d / max(n_t, 1), 1),
               "frames_right_class_%": round(100 * n_r / max(n_d, 1), 1),
               "false_alarms_per_100_frames": round(100 * off_high / max(len(rows), 1), 1)}
    return summary, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=DEFAULT)
    ap.add_argument("--survey", type=Path, default=None)
    ap.add_argument("--looks", default="clean,typical", help="clean,good,typical,bad")
    ap.add_argument("--conf", type=float, default=None, help="detection threshold (default: pilot.CONF)")
    a = ap.parse_args()
    if a.conf is not None:
        pilot.CONF = a.conf
    print(f"detection threshold: {pilot.CONF}")
    survey = (a.survey.expanduser() if a.survey else newest_survey())
    rows = list(csv.DictReader(open(survey / "frames.csv")))
    cam = json.loads((survey / "camera.json").read_text())
    pilot.HFOV = cam["hfov"]                                       # lens used during the survey flight
    pilot.FOCAL = (pilot.IMG_W / 2) / math.tan(pilot.HFOV / 2)
    pilot.LENS = None                                              # survey without lens = pinhole frames
    if cam.get("lens"):                                            # survey recorded through the real lens
        from fisheye_cam import Lens
        pilot.LENS = Lens(cam["lens"], cam["render_hfov"])
        pilot.FOCAL = pilot.LENS.fx
    decals, _, boxes = pilot.load_world()
    n_scan = sum(bool(r["scan_id"]) for r in rows)
    print(f"Survey: {survey}\n  {len(rows)} frames ({n_scan} while scanning a defect), "
          f"lens {math.degrees(pilot.HFOV):.0f} deg")

    from ultralytics import YOLO
    table, per_rows = [], []
    looks = [x.strip() for x in a.looks.split(",") if x.strip()]
    for mname in a.models:
        mp = Path(mname).expanduser()
        mp = mp if mp.exists() else MODELS / mname
        if not mp.exists():
            print(f"!! model not found: {mname} - skipped")
            continue
        model = YOLO(str(mp))
        for look in looks:
            print(f"  testing {mp.name:28s} camera: {'clean' if look == 'clean' else 'A11T ' + look} ...")
            s, per = run(model, rows, survey, look, decals, boxes)
            table.append({"model": mp.name, "camera": look, **s})
            per_rows += [{"model": mp.name, "camera": look, **p} for p in per]

    print("\n=========== DRONE-VIEW TEST (same frames for every model) ===========")
    print(f"{'model':28s}{'camera':9s}{'found':>6s}{'correct':>9s}{'partly':>8s}"
          f"{'det.frames':>12s}{'right class':>13s}{'FA/100fr':>10s}")
    for t in table:
        print(f"{t['model']:28s}{t['camera']:9s}{t['found']:>3d}/{len(decals)}{t['correct']:>7d}{t['partly']:>8d}"
              f"{t['frames_detected_%']:>11.1f}%{t['frames_right_class_%']:>12.1f}%"
              f"{t['false_alarms_per_100_frames']:>10.1f}")
    print("\nper defect (predicted class):")
    keys = [(t["model"], t["camera"]) for t in table]
    print(f"{'id':5s}{'truth':17s}" + "".join(f"{(m.replace('bridge5_', '').replace('.pt', '') + '/' + c)[:22]:>23s}"
                                           for m, c in keys))
    for dc in decals:
        cells = []
        for m, c in keys:
            p = next(x for x in per_rows if x["model"] == m and x["camera"] == c and x["id"] == dc["id"])
            mark = {"correct": "+", "partly correct": "~", "WRONG CLASS": "x", "MISSED": "-"}[p["result"]]
            cells.append(f"{mark} {p['predicted']}"[:22])
        print(f"{dc['id']:5s}{dc['class']:17s}" + "".join(f"{c:>23s}" for c in cells))
    print("   + correct   ~ partly correct   x wrong class   - missed")

    for fname, data in (("simtest_results.csv", table), ("simtest_per_defect.csv", per_rows)):
        with open(survey / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)
    print(f"\nSaved: {survey}/simtest_results.csv  and  simtest_per_defect.csv")


if __name__ == "__main__":
    main()
