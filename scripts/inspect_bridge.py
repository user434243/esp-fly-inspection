# inspect_bridge.py - bridge pier inspection mission:
#   fly the zigzag scan in front of the pier, detect defects live with model B (5 bridge
#   defect classes), locate every detection on the pier face, compare with the ground truth
#   and save map + tables + video in logs/bridge_inspection_<date>/
#
#   start the world first:  simbridge      then:  python scripts/inspect_bridge.py
import csv
import math
import os
import sys
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
from scan_wall import run_scan            # same zigzag: the pier face is where the wall was

# ---------------- settings (must match make_bridge_world.py and add_camera.py) ----------------
MODEL = os.path.expanduser("~/drone_ws/yolo/models/bridge5_y11s.pt")
GROUND_TRUTH = os.path.expanduser("~/drone_ws/sim/worlds/bridge_pier/ground_truth.csv")
CONF = 0.30                               # minimum YOLO confidence
IMG_W, IMG_H, HFOV = 640, 480, 1.57       # camera resolution and horizontal field of view (rad)
FOCAL_PX = (IMG_W / 2) / math.tan(HFOV / 2)
CAM_OFFSET = (0.03, 0.0, 0.015)           # camera position on the drone (m)
FACE_X = 2.5                              # pier face (m)
PANEL_W, PANEL_H = 1.2, 0.9
COLS_Y = [2.25, 0.75, -0.75, -2.25]
ROWS_Z = [2.45, 1.50, 0.55]
MIN_SCAN_X = 1.3                          # record detections only while on the scan line
COLORS = {"crack": "red", "spallation": "orange", "efflorescence": "deepskyblue",
          "exposed_bars": "purple", "corrosion_stain": "saddlebrown"}

OUT = os.path.expanduser(time.strftime("~/drone_ws/logs/bridge_inspection_%Y%m%d_%H%M%S"))
state = {"drone": None, "done": False, "connected": False, "error": None}


def flight():
    """Runs in the background: the zigzag scan from scan_wall.py."""
    try:
        with Drone(name="bridge_inspect") as d:
            state["drone"] = d
            state["connected"] = True
            run_scan(d)
    except Exception as e:
        state["error"] = e
    finally:
        state["done"] = True


def pixel_to_face(u, v, drone_xyz):
    """Where on the pier face (y, z in metres) is image pixel (u, v)? Pinhole camera, facing +x."""
    x, y, z = drone_xyz
    cx, cy, cz = x + CAM_OFFSET[0], y + CAM_OFFSET[1], z + CAM_OFFSET[2]
    dist = FACE_X - cx
    return cy - (u - IMG_W / 2) * dist / FOCAL_PX, cz - (v - IMG_H / 2) * dist / FOCAL_PX


# ---------------- 1) fly + detect ----------------
model = YOLO(MODEL)
names = {i: n.lower() for i, n in model.names.items()}
cap = GazeboCamera()
threading.Thread(target=flight, daemon=True).start()

detections, writer = [], None
frames, t0 = 0, time.time()
print("Mission started - watch the 'Bridge inspection' window")
while not state["done"]:
    ok, frame = cap.read()
    if not ok:
        if not cap.isOpened():
            print("Camera stopped - is simbridge running?")
            break
        continue
    res = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]

    d = state["drone"]
    if d is not None and d.real[0] > MIN_SCAN_X:
        pos = d.real
        for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.tolist(),
                                               res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            fy, fz = pixel_to_face((x1 + x2) / 2, (y1 + y2) / 2, pos)
            detections.append({"t_s": round(time.time() - t0, 2), "class": names[int(cls)],
                               "conf": round(conf, 3), "drone_x": round(pos[0], 3),
                               "drone_y": round(pos[1], 3), "drone_z": round(pos[2], 3),
                               "face_y": round(fy, 3), "face_z": round(fz, 3)})

    shown = res.plot()
    frames += 1
    cv2.putText(shown, f"{frames / (time.time() - t0):4.1f} fps | detections logged: {len(detections)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if writer is None:
        os.makedirs(OUT, exist_ok=True)
        h, w = shown.shape[:2]
        writer = cv2.VideoWriter(os.path.join(OUT, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    writer.write(shown)
    cv2.imshow("Bridge inspection", shown)
    cv2.waitKey(1)

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

if not state["connected"]:
    print(f"\n!! Could not connect to the drone: {state['error']}")
    print("   Wait until simbridge has fully started (drone visible, terminal quiet), then try again.")
    if os.path.isdir(OUT):
        for f in os.listdir(OUT):
            os.remove(os.path.join(OUT, f))
        os.rmdir(OUT)
    sys.exit(1)
if state["error"]:
    print(f"\n!! Flight stopped with an error: {state['error']}")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "detections.csv"), "w", newline="") as f:
    if detections:
        w = csv.DictWriter(f, fieldnames=detections[0].keys())
        w.writeheader()
        w.writerows(detections)

# ---------------- 2) compare with ground truth ----------------
truth, also = {}, {}
for r in csv.DictReader(open(GROUND_TRUTH)):
    key = (float(r["y_m"]), float(r["z_m"]))
    truth[key] = r["class"].lower()
    also[key] = [c for c in r.get("also_present", "").lower().split(";") if c]

slots = [(y, z) for z in ROWS_Z for y in COLS_Y]
score = defaultdict(Counter)     # confidence-weighted votes per panel
count = defaultdict(Counter)     # number of frames per class per panel
off_panel = 0
for det in detections:
    hit = next(((y, z) for (y, z) in slots
                if abs(det["face_y"] - y) <= PANEL_W / 2 and abs(det["face_z"] - z) <= PANEL_H / 2), None)
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
    others = also.get((y, z), [])
    if true_cls == "blank":
        result = "correct (clean)" if pred == "none" else "FALSE ALARM"
    elif pred == "none":
        result = "MISSED"
    elif pred == true_cls:
        result = "correct"
    elif pred in others:
        result = "partly correct"        # a defect that IS on the panel, but not the main one
    else:
        result = "WRONG CLASS"
    summary.append({"y_m": y, "z_m": z, "truth": true_cls, "also_present": ";".join(others),
                    "predicted": pred, "frames": n, "avg_conf": avg_conf, "result": result})

with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys())
    w.writeheader()
    w.writerows(summary)

defects = [s for s in summary if s["truth"] != "blank"]
n_def = len(defects)
n_found = sum(1 for s in defects if s["predicted"] != "none")
n_ok = sum(1 for s in defects if s["result"] == "correct")
n_part = sum(1 for s in defects if s["result"] == "partly correct")
n_blank = len(summary) - n_def
n_fa = sum(1 for s in summary if s["result"] == "FALSE ALARM")

print("\n=========== BRIDGE INSPECTION REPORT ===========")
print(f"{'panel (y, z)':<16}{'truth':<17}{'predicted':<17}{'frames':>7}{'conf':>7}   result")
for s in summary:
    extra = f"  (also: {s['also_present'].replace(';', ', ')})" if s["also_present"] else ""
    print(f"({s['y_m']:+.2f}, {s['z_m']:.2f})  {s['truth']:<17}{s['predicted']:<17}"
          f"{s['frames']:>7}{s['avg_conf']:>7}   {s['result']}{extra}")
print(f"\nDefect found (any class):      {n_found}/{n_def}")
print(f"Correct main defect:           {n_ok}/{n_def}")
print(f"Correct or partly correct:     {n_ok + n_part}/{n_def}")
print(f"False alarms on clean concrete: {n_fa}/{n_blank}")
print(f"Detections off any panel:      {off_panel}")

# ---------------- 3) defect map ----------------
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.add_patch(Rectangle((-3.2, 0), 6.4, 3.2, color="0.85"))                 # pier face
ax.add_patch(Rectangle((-3.6, 3.2), 7.2, 0.35, color="0.7"))               # pier cap
for s in summary:
    edge = {"correct": "green", "correct (clean)": "green", "partly correct": "orange"}.get(s["result"], "red")
    ax.add_patch(Rectangle((s["y_m"] - PANEL_W / 2, s["z_m"] - PANEL_H / 2), PANEL_W, PANEL_H,
                           fill=False, lw=2.5, ec=edge, ls="--" if s["truth"] == "blank" else "-"))
    ax.text(s["y_m"], s["z_m"] + PANEL_H / 2 - 0.12, f"truth: {s['truth']}", ha="center", fontsize=8)
    ax.text(s["y_m"], s["z_m"] - PANEL_H / 2 + 0.06, f"pred: {s['predicted']}", ha="center",
            fontsize=8, color=edge, weight="bold")
for cls, col in COLORS.items():
    pts = [d for d in detections if d["class"] == cls]
    if pts:
        ax.scatter([p["face_y"] for p in pts], [p["face_z"] for p in pts], s=8, c=col, alpha=0.35, label=cls)
ax.set_xlim(3.8, -3.8)                    # flipped so the map looks like the view from the drone
ax.set_ylim(0, 3.7)
ax.set_aspect("equal")
ax.set_xlabel("pier position y (m)  - as seen from the drone")
ax.set_ylabel("height z (m)")
ax.set_title(f"Bridge pier: {n_found}/{n_def} defects found, {n_ok}/{n_def} correct, "
             f"{n_part} partly, {n_fa} false alarms  (green = correct, orange = partly, red = error)",
             fontsize=10)
if detections:
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "map.png"), dpi=150)
print(f"\nSaved in {OUT}:  map.png  summary.csv  detections.csv  video.mp4")