#!/usr/bin/env python3
"""
make_figures.py - the report figures, made from the results in this repository.

  python3 scripts/make_figures.py          # -> results/figures/*.png  (300 dpi, for Word)

  fig_architecture.png      system overview (drone, video link, ground station, simulator)
  fig_bridge_models.png     bridge world: models B / E4 / E6 (found, correct, false alarms)
  fig_road_progress.png     road world: the 4 steps from 11/20 to 20/20 found
  fig_city_runs.png         city world: the 4 full missions
  fig_wind_path.png         calm vs light wind: path deviation and distance to the nearest wall
  fig_training_map50.png    validation mAP50 per epoch of the main training runs
Flight numbers come from each run's findings.csv / tracking.csv; the bridge model comparison
(runs made before the log folders were named) uses the numbers recorded in docs/06_results.md.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
RUNS, OUT = ROOT / "results/runs", ROOT / "results/figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"           # validated categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK})


def findings(run):
    rows = list(csv.DictReader(open(RUNS / run / "findings.csv")))
    n = len(rows)
    found = sum(r["result"] != "MISSED" for r in rows)
    ok = sum(r["result"] == "correct" for r in rows)
    part = sum(r["result"] == "partly correct" for r in rows)
    return n, found, ok, part


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def grouped(ax, labels, series, total, colors):
    """series: [(name, values)]; bars with a 2 px gap and the value on top (direct labels)."""
    k, w = len(series), 0.8 / len(series)
    for i, ((name, vals), c) in enumerate(zip(series, colors)):
        xs = [j + (i - (k - 1) / 2) * w for j in range(len(labels))]
        bars = ax.bar(xs, vals, w * 0.94, color=c, label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + total * 0.015, f"{v}", ha="center", va="bottom",
                    fontsize=8.5, color=INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, total * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_ylabel(f"defects (of {total})")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=k)


# ---------------- 1. architecture ----------------
def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="white", ec=edge, lw=1.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.0, "\n".join(lines), ha="center", va="top", fontsize=8, color=INK2,
                linespacing=1.45)

    def arrow(x0, y0, x1, y1, text, dy=1.2):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=7.5, color=INK2)

    box(2, 26, 26, 20, "ESP-FLY micro drone", ["XIAO ESP32-S3 + MPU6050", "ESP-Drone firmware (PID)",
                                                "4 x SI2300 motor drivers", "~10 cm, no GPS"], BLUE)
    box(2, 2, 26, 17, "AKK A11T camera", ["fisheye 122.5 deg (calibrated)", "analog video", "5.8 GHz, 25 mW"], BLUE)
    box(37, 2, 22, 17, "Skydroid receiver", ["5731 MHz", "USB video (UVC)", "-> ground station"], BLUE)
    box(68, 2, 30, 44, "Ground station PC", ["YOLO11s defect detection", "(bridge E6 / road E6)", "",
                                             "box stabilizer + class vote", "pixel -> 3D ray -> defect patch", "",
                                             "3D A* path planner", "mission: stops, scans, reports", "",
                                             "phone app / pilot.py / autopilot.py"], ORANGE)
    box(37, 22, 22, 17, "Simulation", ["Gazebo + CrazySim", "(same firmware in the loop)",
                                       "real lens + analog look", "light, wind, 3 worlds"], AQUA)
    arrow(15, 26, 15, 19.6, "", 0)
    arrow(28, 10, 36.6, 10, "5.8 GHz", 1.0)
    arrow(59, 10, 67.6, 10, "USB", 1.0)
    arrow(67.6, 43, 28.4, 43, "Wi-Fi: CRTP position setpoints (UDP 2390)  /  phone app", 0.8)
    arrow(59, 30, 67.6, 30, "camera,\npose", 1.0)
    ax.text(50, 50.5, "System architecture", ha="center", fontsize=12, weight="bold", color=INK)
    save(fig, "fig_architecture.png")


# ---------------- 2. bridge models ----------------
def bridge_models():
    models = ["B", "E4 (analog)", "E6 + stabilizer"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    grouped(a, models, [("found", [18, 19, 17]), ("correct class", [6, 3, 7]),
                        ("correct + partly", [8, 8, 11])], 20, [BLUE, ORANGE, AQUA])
    a.set_title("Bridge world, 20 defects, real A11T camera")
    bars = b.bar(models, [3.6, 1.2, 0.5], 0.55, color=BLUE, zorder=3)
    for bar, v in zip(bars, [3.6, 1.2, 0.5]):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v}", ha="center", fontsize=8.5, color=INK)
    b.set_title("False alarms per flying minute")
    b.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    b.tick_params(axis="x", labelsize=8)
    save(fig, "fig_bridge_models.png")


# ---------------- 3. road progress ----------------
def road_progress():
    steps = [("auto_20261119_201929", "road model\nwhole photos"), ("auto_20261119_205013", "road model\ntight crops"),
             ("auto_20261119_222843", "road E6"), ("auto_20261119_225040", "road E6\nclean world")]
    data = [findings(r) for r, _ in steps]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    grouped(ax, [l for _, l in steps], [("found", [d[1] for d in data]), ("correct class", [d[2] for d in data])],
            20, [BLUE, ORANGE])
    ax.set_title("Road world, 20 road-damage patches, camera 55 deg down")
    save(fig, "fig_road_progress.png")


# ---------------- 4. city runs ----------------
def city_runs():
    runs = [("auto_20261121_140746", "calm 1"), ("auto_20261121_143901", "calm 2"),
            ("auto_20261121_150210", "calm 3"), ("auto_20261121_154949", "light wind")]
    data = [findings(r) for r, _ in runs]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    grouped(ax, [l for _, l in runs], [("found", [d[1] for d in data]), ("correct class", [d[2] for d in data]),
                                       ("correct + partly", [d[2] + d[3] for d in data])], 30,
            [BLUE, ORANGE, AQUA])
    ax.set_title("City world, 30 defects: buildings, box-girder interior, tunnel")
    save(fig, "fig_city_runs.png")


# ---------------- 5. wind ----------------
def wind_path():
    sets = [("auto_20261121_150210", "calm", BLUE), ("auto_20261121_154949", "light wind 1.5 m/s", ORANGE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for run, name, c in sets:
        rows = list(csv.DictReader(open(RUNS / run / "tracking.csv")))
        dev = sorted(float(r["path_dev_m"]) * 100 for r in rows)
        a.plot(dev, [i / len(dev) * 100 for i in range(len(dev))], color=c, lw=2, label=name)
        w = sorted(float(r["wall_dist_m"]) * 100 for r in rows)
        b.plot(w, [i / len(w) * 100 for i in range(len(w))], color=c, lw=2, label=f"{name} (closest {w[0]:.0f} cm)")
    a.set_xlim(0, 30)
    a.set_xlabel("deviation from the planned path (cm)")
    a.set_ylabel("% of flight time below")
    a.set_title("Path holding")
    a.yaxis.grid(True, color=GRID, lw=0.8)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    b.axvline(5, color=INK2, lw=1, ls="--")
    b.text(5.6, 12, "5 cm:\nan arm\ntouches", fontsize=7.5, color=INK2)
    b.axvline(14, color=INK2, lw=1, ls=":")
    b.text(14.6, 16, "14 cm\nplanner\nmargin", fontsize=7.5, color=INK2)
    b.set_xlim(0, 50)
    b.set_ylim(0, 20)
    b.set_xlabel("distance to the nearest surface, drone centre (cm)")
    b.set_ylabel("% of flight time closer than this")
    b.set_title("Closeness to walls")
    b.yaxis.grid(True, color=GRID, lw=0.8)
    b.legend(frameon=False, fontsize=8.5, loc="center right")
    save(fig, "fig_wind_path.png")


# ---------------- 6. training ----------------
def training():
    groups = [("Bridge models (own val sets)", [("bridge5_y11s", "B", BLUE), ("bridge5_y11s_analog", "E4", ORANGE),
                                                ("bridge5_y11s_e6", "E6", AQUA)]),
              ("Road models (own val sets)", [("rdd4_y11s", "road", BLUE), ("rdd4_y11s_e6", "road E6", ORANGE)])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (title, runs) in zip(axes, groups):
        for run, name, c in runs:
            f = ROOT / "results/training" / run / "results.csv"
            if not f.exists():
                continue
            rows = list(csv.DictReader(open(f)))
            key = [k for k in rows[0] if "mAP50(B)" in k][0]
            ax.plot([int(r["epoch"]) for r in rows], [float(r[key]) for r in rows], color=c, lw=2, label=name)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation mAP50")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=8.5)
    save(fig, "fig_training_map50.png")


if __name__ == "__main__":
    architecture()
    bridge_models()
    road_progress()
    city_runs()
    wind_path()
    training()
