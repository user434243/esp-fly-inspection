# inspect_wall.py - full inspection mission:
#   fly the zigzag scan, detect defects live with YOLO, locate every detection on the wall,
#   compare with the ground truth, and save map + tables + video in logs/inspection_<date>/
import csv
import math
import os
import threading
import time
from collections import Counter, defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")                     # draw the map to a file (no extra window)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from flight_utils import Drone
from gz_camera import GazeboCamera
from scan_wall import run_scan

# ---------------- settings (must match make_wall_world.py and add_camera.py) ----------------
MODEL = os.path.expanduser("~/drone_ws/yolo/models/best.pt")
GROUND_TRUTH = os.path.expanduser("~/drone_ws/sim/worlds/inspection_wall/ground_truth.csv")
CONF = 0.35                               # minimum YOLO confidence
IMG_W, IMG_H, HFOV = 640, 480, 1.57       # camera resolution and horizontal field of view (rad)
FOCAL_PX = (IMG_W / 2) / math.tan(HFOV / 2)
CAM_OFFSET = (0.03, 0.0, 0.015)           # camera position on the drone (m)
WALL_X = 2.5                              # wall face (m)
PANEL_W, PANEL_H = 1.2, 0.9
COLS_Y = [2.25, 0.75, -0.75, -2.25]
ROWS_Z = [2.45, 1.50, 0.55]
MIN_SCAN_X = 1.3                          # record detections only while on the scan line
COLORS = {"crack": "red", "leakage": "blue", "abscission": "orange",
          "corrosion": "brown", "bulge": "purple"}

OUT = os.path.expanduser(time.strftime("~/drone_ws/logs/inspection_%Y%m%d_%H%M%S"))
os.makedirs(OUT, exist_ok=True)
state = {"drone": None, "done": False}


def flight():
    """Runs in the background: the same zigzag scan as scan_wall.py."""
    try:
        with Drone(name="inspect") as d:
            state["drone"] = d
            run_scan(d)
    finally:
        state["done"] = True


def pixel_to_wall(u, v, drone_xyz):
    """Where on the wall (y, z in metres) is image pixel (u, v)? Pinhole camera, facing +x."""
    x, y, z = drone_xyz
    cx, cy, cz = x + CAM_OFFSET[0], y + CAM_OFFSET[1], z + CAM_OFFSET[2]
    dist = WALL_X - cx
    return cy - (u - IMG_W / 2) * dist / FOCAL_PX, cz - (v - IMG_H / 2) * dist / FOCAL_PX


# ---------------- 1) fly + detect ----------------
model = YOLO(MODEL)
names = {i: n.lower() for i, n in model.names.items()}
cap = GazeboCamera()
threading.Thread(target=flight, daemon=True).start()

detections, writer = [], None
frames, t0 = 0, time.time()
print("Mission started - watch the 'Inspection' window")
while not state["done"]:
    ok, frame = cap.read()
    if not ok:
        if not cap.isOpened():
            print("Camera stopped - is simwall running?")
            break
        continue
    res = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]

    d = state["drone"]
    if d is not None and d.real[0] > MIN_SCAN_X:
        pos = d.real
        for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.tolist(),
                                               res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            wy, wz = pixel_to_wall((x1 + x2) / 2, (y1 + y2) / 2, pos)
            detections.append({"t_s": round(time.time() - t0, 2), "class": names[int(cls)],
                               "conf": round(conf, 3), "drone_x": round(pos[0], 3),
                               "drone_y": round(pos[1], 3), "drone_z": round(pos[2], 3),
                               "wall_y": round(wy, 3), "wall_z": round(wz, 3)})

    shown = res.plot()
    frames += 1
    cv2.putText(shown, f"{frames / (time.time() - t0):4.1f} fps | detections logged: {len(detections)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if writer is None:
        h, w = shown.shape[:2]
        writer = cv2.VideoWriter(os.path.join(OUT, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    writer.write(shown)
    cv2.imshow("Inspection", shown)
    cv2.waitKey(1)

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

with open(os.path.join(OUT, "detections.csv"), "w", newline="") as f:
    if detections:
        w = csv.DictWriter(f, fieldnames=detections[0].keys())
        w.writeheader()
        w.writerows(detections)

# ---------------- 2) compare with ground truth ----------------
truth = {(float(r["y_m"]), float(r["z_m"])): r["class"].lower() for r in csv.DictReader(open(GROUND_TRUTH))}
slots = [(y, z) for z in ROWS_Z for y in COLS_Y]
score = defaultdict(Counter)     # confidence-weighted votes per panel
count = defaultdict(Counter)     # number of frames per class per panel
off_panel = 0
for det in detections:
    hit = next(((y, z) for (y, z) in slots
                if abs(det["wall_y"] - y) <= PANEL_W / 2 and abs(det["wall_z"] - z) <= PANEL_H / 2), None)
    if hit is None:
        off_panel += 1
        continue
    score[hit][det["class"]] += det["conf"]
    count[hit][det["class"]] += 1

summary = []
for (y, z) in slots:
    true_cls = truth.get((y, z), "blank")
    pred = score[(y, z)].most_common(1)[0][0] if score[(y, z)] else "none"
    n = count[(y, z)][pred] if pred != "none" else 0
    avg_conf = round(score[(y, z)][pred] / n, 2) if n else 0
    if true_cls == "blank":
        result = "correct (clean)" if pred == "none" else "FALSE ALARM"
    elif pred == "none":
        result = "MISSED"
    else:
        result = "correct" if pred == true_cls else "WRONG CLASS"
    summary.append({"y_m": y, "z_m": z, "truth": true_cls, "predicted": pred,
                    "frames": n, "avg_conf": avg_conf, "result": result})

with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys())
    w.writeheader()
    w.writerows(summary)

n_def = sum(1 for s in summary if s["truth"] != "blank")
n_ok = sum(1 for s in summary if s["result"] == "correct")
n_fa = sum(1 for s in summary if s["result"] == "FALSE ALARM")
print("\n=========== INSPECTION REPORT ===========")
print(f"{'panel (y, z)':<16}{'truth':<12}{'predicted':<12}{'frames':>7}{'conf':>7}   result")
for s in summary:
    print(f"({s['y_m']:+.2f}, {s['z_m']:.2f})  {s['truth']:<12}{s['predicted']:<12}"
          f"{s['frames']:>7}{s['avg_conf']:>7}   {s['result']}")
print(f"\nCorrect defects: {n_ok}/{n_def}   False alarms on blank panels: {n_fa}   "
      f"Detections off any panel: {off_panel}")

# ---------------- 3) defect map ----------------
fig, ax = plt.subplots(figsize=(11, 6))
ax.add_patch(Rectangle((-3.2, 0), 6.4, 3.2, color="0.85"))
for s in summary:
    edge = "green" if s["result"].startswith("correct") else "red"
    ax.add_patch(Rectangle((s["y_m"] - PANEL_W / 2, s["z_m"] - PANEL_H / 2), PANEL_W, PANEL_H,
                           fill=False, lw=2.5, ec=edge, ls="--" if s["truth"] == "blank" else "-"))
    ax.text(s["y_m"], s["z_m"] + PANEL_H / 2 - 0.12, f"truth: {s['truth']}", ha="center", fontsize=9)
    ax.text(s["y_m"], s["z_m"] - PANEL_H / 2 + 0.06, f"pred: {s['predicted']}", ha="center",
            fontsize=9, color=edge, weight="bold")
for cls, col in COLORS.items():
    pts = [d for d in detections if d["class"] == cls]
    if pts:
        ax.scatter([p["wall_y"] for p in pts], [p["wall_z"] for p in pts], s=8, c=col, alpha=0.35, label=cls)
ax.set_xlim(3.3, -3.3)                    # flipped so the map looks like the view from the drone
ax.set_ylim(0, 3.3)
ax.set_aspect("equal")
ax.set_xlabel("wall position y (m)  - as seen from the drone")
ax.set_ylabel("height z (m)")
ax.set_title(f"Defect map: {n_ok}/{n_def} defects correct, {n_fa} false alarms "
             f"(green = correct, red = error)")
ax.legend(loc="upper right", fontsize=8, markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "map.png"), dpi=150)
print(f"\nSaved in {OUT}:  map.png  summary.csv  detections.csv  video.mp4")