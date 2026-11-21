# make_map.py - redraw the defect map of an inspection run (report quality), no flying needed
#   python scripts/make_map.py                      -> newest logs/bridge_inspection_* folder
#   python scripts/make_map.py logs/bridge_inspection_20261010_114834
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PANEL_W, PANEL_H = 1.2, 0.9
COLORS = {"crack": "red", "spallation": "orange", "efflorescence": "deepskyblue",
          "exposed_bars": "purple", "corrosion_stain": "saddlebrown"}
SHORT = {"crack": "crack", "spallation": "spalling", "efflorescence": "efflorescence",
         "exposed_bars": "rebar", "corrosion_stain": "rust"}
EDGE = {"correct": "green", "correct (clean)": "green", "partly correct": "orange"}

if len(sys.argv) > 1:
    run = os.path.expanduser(sys.argv[1])
else:
    runs = sorted(glob.glob(os.path.expanduser("~/drone_ws/logs/bridge_inspection_*")))
    if not runs:
        sys.exit("No logs/bridge_inspection_* folder found")
    run = runs[-1]

summary = list(csv.DictReader(open(os.path.join(run, "summary.csv"))))
det_path = os.path.join(run, "detections.csv")
detections = (list(csv.DictReader(open(det_path)))
              if os.path.exists(det_path) and os.path.getsize(det_path) else [])

defects = [s for s in summary if s["truth"] != "blank"]
n_found = sum(s["predicted"] != "none" for s in defects)
n_ok = sum(s["result"] == "correct" for s in defects)
n_part = sum(s["result"] == "partly correct" for s in defects)
n_blank = len(summary) - len(defects)
n_fa = sum(s["result"] == "FALSE ALARM" for s in summary)

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.add_patch(Rectangle((-3.2, 0), 6.4, 3.2, color="0.88", zorder=0))          # pier face
ax.add_patch(Rectangle((-3.6, 3.2), 7.2, 0.35, color="0.72", zorder=0))       # pier cap
ax.text(0, 3.375, "pier cap / deck", ha="center", va="center", fontsize=9, color="0.3")

# 1) detections first (underneath), small and transparent
for cls, col in COLORS.items():
    pts = [d for d in detections if d["class"] == cls]
    if pts:
        ax.scatter([float(p["face_y"]) for p in pts], [float(p["face_z"]) for p in pts],
                   s=10, c=col, alpha=0.4, label=f"{cls} ({len(pts)})", zorder=2)

# 2) panels and labels on top, labels on white so dots never hide them
box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
for s in summary:
    y, z = float(s["y_m"]), float(s["z_m"])
    edge = EDGE.get(s["result"], "red")
    ax.add_patch(Rectangle((y - PANEL_W / 2, z - PANEL_H / 2), PANEL_W, PANEL_H, fill=False,
                           lw=2.8, ec=edge, ls="--" if s["truth"] == "blank" else "-", zorder=3))
    truth = f"truth: {s['truth']}"
    if s.get("also_present"):
        truth += "\nalso: " + ", ".join(SHORT[c] for c in s["also_present"].split(";"))
    ax.text(y, z + PANEL_H / 2 - 0.06, truth, ha="center", va="top", fontsize=8.5, zorder=4, bbox=box)
    ax.text(y, z - PANEL_H / 2 + 0.11, f"pred: {s['predicted']}", ha="center", fontsize=9.5,
            color=edge, weight="bold", zorder=4, bbox=box)

ax.set_xlim(3.8, -3.8)                    # flipped: the map looks like the view from the drone
ax.set_ylim(0, 3.7)
ax.set_aspect("equal")
ax.set_xlabel("pier position y (m) - as seen from the drone")
ax.set_ylabel("height z (m)")
ax.set_title(f"Simulated bridge pier inspection - defects found {n_found}/{len(defects)}, "
             f"correct {n_ok}/{len(defects)} (+{n_part} partly), false alarms {n_fa}/{n_blank}",
             fontsize=12, weight="bold")
ax.legend(title="detections (frames)", loc="upper left", bbox_to_anchor=(1.01, 1.0),
          fontsize=9, markerscale=2.5, frameon=False)
ax.text(1.01, 0.35, "panel border:\n  green = correct\n  orange = partly correct\n"
        "  red = missed / wrong\n  dashed = clean concrete",
        transform=ax.transAxes, fontsize=9, va="top")
plt.tight_layout()
out = os.path.join(run, "map_report.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("Saved", out)