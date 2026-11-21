# 1. Setup

## Machine used

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS (installed on an external HDD) |
| GPU | NVIDIA RTX 4060, 8 GB |
| Python | 3.10, virtual environment `~/drone_ws/venv` |
| Deep learning | Ultralytics 8.4.174 (YOLO11), PyTorch 2.14.1 + CUDA 13 |
| Simulator | Gazebo Sim 7.9 (Garden) + CrazySim (Crazyflie firmware in the loop) |
| Datasets | 1 TB HDD at `/media/hudaifa/Ventoy/datasets` |

The exact Python packages are in [`setup/requirements.txt`](../setup/requirements.txt), the
machine details in [`setup/environment.txt`](../setup/environment.txt).

## Folders

```
~/drone_ws/
  venv/                 Python environment
  scripts/              everything in scripts/ of this repo
  sim/
    CrazySim/           CrazySim (git clone) - the drone firmware + Gazebo plugins
    worlds/             world generators (sim/worlds/ of this repo) and the generated worlds
    assets/textures/    Poly Haven PBR textures (concrete, asphalt, grass, gravel, rust metal)
    a11t_calib.json     the measured A11T lens
    camera_profile.json which camera the simulation uses (camprofile.py writes it)
    active_world        which world simfull loads (simworld.py writes it)
  yolo/models/          trained models (*.pt)
  logs/  -> HDD         every flight: report.png, findings.csv, video.mp4 ...
```

## Install

```bash
# 1. Python environment
python3 -m venv ~/drone_ws/venv
~/drone_ws/venv/bin/pip install -r setup/requirements.txt

# 2. Gazebo Sim (Garden) + CrazySim  - follow https://github.com/gtfactslab/CrazySim
#    clone it to ~/drone_ws/sim/CrazySim and build the SITL firmware

# 3. this repo's code
cp scripts/*.py ~/drone_ws/scripts/
cp sim/worlds/*.py ~/drone_ws/sim/worlds/
cp sim/camera/a11t_calib.json ~/drone_ws/sim/
python3 sim/patches/add_camera.py              # FPV camera on the CrazySim drone

# 4. shell shortcuts
cat setup/aliases.sh >> ~/.bashrc && source ~/.bashrc

# 5. build the worlds (needs the datasets on the HDD and the textures)
~/drone_ws/venv/bin/python ~/drone_ws/sim/worlds/make_bridge_full.py
~/drone_ws/venv/bin/python ~/drone_ws/sim/worlds/make_road_world.py
~/drone_ws/venv/bin/python ~/drone_ws/sim/worlds/make_city_world.py

# 6. real A11T camera in the simulation
camprofile real                                 # (camprofile clean = perfect camera)
```

The scripts in `scripts/` are the final versions; the `scripts/patches/*.py` files are the
upgrade steps that were applied to them during the project, kept as a record.

## Shortcuts (aliases)

| alias | does |
|---|---|
| `simworld [name] [--light ...] [--wind ...]` | choose world / light / wind, show status |
| `simfull` | start Gazebo + drone firmware with the active world |
| `simreset` | stop everything (before every new start) |
| `pilot` | fly yourself, live detection, report at the end |
| `autopilot` | autonomous inspection mission (T = start, ESC = fly home) |
| `camprofile real / clean / status` | real A11T lens + analog video, or a perfect camera |
| `lab` | JupyterLab for the training notebooks |
