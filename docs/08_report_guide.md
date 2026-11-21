# 8. Report guide: where every part of Chapters 3–5 comes from

Graduation report 2 keeps Chapters 1–2 (references corrected) and replaces Chapter 3, then
adds Chapters 4 and 5. This page maps each section to the evidence in this repository, so
every number and figure in the report can be traced back to a file.

Status: ✅ in the repo · 📷 to add (photos / screenshots) · 🔧 depends on the real hardware

## Chapter 3: Methodology

| Section | What to write | Evidence in the repo |
|---|---|---|
| 3.1 Introduction | The 4 needs (no GPS, small, real-time detection, low cost); why the GP1 design (250 mm, Pi 4, LiDAR, 350–400 g) was replaced by a 10 cm drone with detection on the ground | docs/02_hardware.md ✅ |
| 3.2 System architecture | Drone, video link, ground station, simulator | `results/figures/fig_architecture.png` ✅ |
| 3.3 Hardware platform | ESP-FLY, XIAO ESP32-S3, MPU6050, SI2300; motors / battery / weight / flight time | docs/02_hardware.md ✅, photos 📷 `results/hardware/` |
| 3.4 Firmware and control | ESP-Drone PID, phone app, CRTP over UDP 2390 | docs/02_hardware.md ✅ |
| 3.5 Camera and video link | A11T, 5731 MHz, Skydroid UVC, why analog | docs/02_hardware.md ✅ |
| 3.6 Defect detection | Datasets, YOLO11s, training versions, stabilizer, class vote, pixel → 3D matching | docs/04_models.md ✅, `scripts/pilot.py`, `scripts/box_stabilizer.py` |
| 3.7 Navigation | 3D A* on a 10 cm grid, string pulling, stops and scans, wind safety margin | docs/05_simulation.md ✅, `scripts/planner.py`, `scripts/route.py`, `scripts/autopilot.py` |
| 3.8 Simulation method | CrazySim (real firmware in the loop), calibrated lens + analog look, worlds, light, wind drag model | docs/03, docs/05 ✅ |
| 3.9 Evaluation method | Found / correct / partly / false alarms per min; path deviation; distance to walls; test-split-only rule | docs/06_results.md ✅ |
| 3.10 Ground station | pilot / autopilot screens, mini-map, reports | 📷 screenshot of the autopilot window → `results/screenshots/` |

## Chapter 4: Implementation

| Section | Evidence |
|---|---|
| 4.1 Hardware build and bench tests | 📷 camera wiring, receiver, phone picture on 5731 MHz → `results/hardware/` |
| 4.2 Software environment | `setup/environment.txt`, `setup/requirements.txt`, docs/01_setup.md ✅ |
| 4.3 Lens calibration | docs/03 ✅, `results/calibration/*.jpg` ✅, `sim/camera/a11t_calib.json` ✅ |
| 4.4 Datasets | `results/datasets/*_data.yaml` + examples ✅, `scripts/rdd2yolo.py`, notebooks ✅ |
| 4.5 Model training | `results/training/*/results.png`, confusion matrices ✅, `results/figures/fig_training_map50.png` ✅, `scripts/make_road_e6.py` |
| 4.6 Simulated worlds | `sim/worlds/*/site_map.png`, `decals_overview.jpg` ✅; 📷 Gazebo screenshots of each world → `results/screenshots/` |
| 4.7 Autonomous mission | `scripts/autopilot.py`, `planner.py`, `route.py` ✅ |
| 4.8 Lighting and wind | `scripts/simworld.py`, `scripts/wind_gust.py` ✅ |
| 4.9 Real flight tests | 🔧 if the hardware arrives: hover drift (30 s), manual pass of a real wall and a real road crack at 0.5–1 m, A11T video recorded through the receiver and run through the models, real vs simulated picture side by side, flight time. If not: write it as the test plan and state that validation is in simulation |

## Chapter 5: Results and discussion

| Section | Evidence |
|---|---|
| 5.1 Bridge world: models B / E4 / E6 | `results/figures/fig_bridge_models.png` ✅, docs/06 table ✅ |
| 5.2 Road world: 11 → 20/20 found | `results/figures/fig_road_progress.png` ✅, runs `auto_20261119_*` ✅ |
| 5.3 City world: tight spaces, 30/30 stops | `results/figures/fig_city_runs.png` ✅, `results/highlights/city_report.png` ✅ |
| 5.4 Wind and path holding | `results/figures/fig_wind_path.png` ✅, docs/06 wind table ✅, `tracking.csv` of each run ✅ |
| 5.5 Real hardware results | 🔧 |
| 5.6 Discussion | analog video limits the class (≈55 % correct with clean video vs 35–50 % with the A11T); the domain gap and its fix (E6); wind limits a 27 g drone to calm/light air; sim-to-real gap |

## Still to add

1. 📷 `results/hardware/`: photos of the drone, the A11T wiring, the receiver, the phone picture.
2. 📷 `results/screenshots/`: Gazebo views of the 3 worlds, the autopilot window, the drone camera inside the box girder or tunnel.
3. Models: upload the `.pt` files to a GitHub release (see `models/README.md`).
4. Optional: 2–3 short videos (one per world) as release files; they are too big for git.

All figures can be rebuilt with `python3 scripts/make_figures.py`.
