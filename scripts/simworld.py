#!/usr/bin/env python3
"""
simworld.py - choose which world 'simfull' loads, and set everything that goes with it.

  simworld                 # show the active world + all worlds you have
  simworld road_full       # switch to the road world
  simworld bridge_full     # back to the bridge
  simworld --install       # one time: teach 'simfull' to load the active world, add the 'simworld' alias
  simworld city_full --light evening    # switch + set the light (noon morning evening overcast dusk)
  simworld --light overcast             # change the light of the active world
  simworld --wind gusty                 # wind: calm light moderate gusty strong   (real physics in Gazebo)
  simworld --wind moderate --wind-dir 90   # direction the wind blows TO, deg (0 = +x east, 90 = +y north)

Each world lives in ~/drone_ws/sim/worlds/<name>/ :
  <name>.sdf     the Gazebo world
  decals.json    the defect patches (ground truth for pilot / autopilot / simtest)
  world.json     settings that go with the world, for example
                   {"kind": "road", "model": "rdd4_y11s.pt", "conf": 0.2,
                    "camera_pitch_deg": 55, "map_x": [-4, 10], "map_y": [-12, 18]}
                 model             YOLO model for this world (in ~/drone_ws/yolo/models), default: pilot.py's MODEL
                 camera_pitch_deg  camera tilted down (0 = straight ahead, like the bridge)

Switching does:
  1. writes the name to ~/drone_ws/sim/active_world  (read by simfull, pilot.py, autopilot.py, route.py)
  2. tilts the FPV camera in the Gazebo drone model to camera_pitch_deg
  3. makes sure CrazySim's worlds folder links to the world
After switching: simreset, then simfull.
"""
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HOME = Path.home()
SIM = HOME / "drone_ws/sim"
WORLDS = SIM / "worlds"
ACTIVE = SIM / "active_world"
CRAZYSIM_WORLDS = SIM / "CrazySim/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/gazebo/worlds"
TEMPLATE = SIM / "CrazySim/crazyflie-firmware/tools/crazyflie-simulation/simulator_files/gazebo/models/crazyflie/model.sdf.jinja"
MODELS = HOME / "drone_ws/yolo/models"
BASHRC = HOME / ".bashrc"
DEFAULT = "bridge_full"
DEFAULT_SETTINGS = {"bridge_full": {"name": "bridge_full", "kind": "bridge", "camera_pitch_deg": 0,
                                    "map_x": [-4, 10], "map_y": [-12, 18]}}


# sun direction (towards the ground), sun colour, ambient, sky colour, shadows
LIGHTS = {
    "noon":     ((-0.30, 0.20, -0.93), (1.00, 0.98, 0.92), 0.55, (0.62, 0.75, 0.92), True),
    "morning":  ((-0.75, -0.35, -0.45), (1.00, 0.86, 0.70), 0.40, (0.78, 0.80, 0.86), True),
    "evening":  ((0.80, 0.30, -0.30), (1.00, 0.60, 0.36), 0.30, (0.86, 0.56, 0.40), True),
    "overcast": ((-0.20, 0.10, -0.97), (0.45, 0.45, 0.47), 0.75, (0.70, 0.72, 0.75), False),
    "dusk":     ((0.60, -0.30, -0.20), (0.18, 0.16, 0.22), 0.16, (0.10, 0.12, 0.24), False),
}


def set_light(name, preset):
    """Change the sun + sky of a world's .sdf (inspection lamps stay as they are)."""
    sdf = WORLDS / name / f"{name}.sdf"
    d, col, amb, sky, shadows = LIGHTS[preset]
    tree = ET.parse(sdf)
    world = tree.getroot().find("world")
    sun = next((l for l in world.findall("light") if l.get("type") == "directional"), None)
    if sun is None:
        print(f"!! {name} has no sun (directional light) - light not changed")
        return
    n = math.sqrt(sum(v * v for v in d))

    def put(parent, tag, text):
        e = parent.find(tag)
        (e if e is not None else ET.SubElement(parent, tag)).text = text

    put(sun, "direction", " ".join(f"{v / n:.3f}" for v in d))
    put(sun, "diffuse", f"{col[0]} {col[1]} {col[2]} 1")
    put(sun, "specular", f"{col[0] * 0.3:.2f} {col[1] * 0.3:.2f} {col[2] * 0.3:.2f} 1")
    put(sun, "cast_shadows", "true" if shadows else "false")
    scene = world.find("scene")
    if scene is None:
        scene = ET.SubElement(world, "scene")
    put(scene, "ambient", f"{amb} {amb} {amb} 1")
    put(scene, "background", f"{sky[0]} {sky[1]} {sky[2]} 1")
    tree.write(sdf, encoding="utf-8", xml_declaration=True)
    f = WORLDS / name / "world.json"
    s = settings(name)
    s["light"] = preset
    f.write_text(json.dumps(s, indent=2))
    print(f"light of {name}: {preset}")


# wind presets live in wind_gust.py (same folder); the wind is a force autopilot pushes on the drone
WIND_NAMES = ("calm", "light", "moderate", "gusty", "strong")


def drone_wind_off():
    """Undo the first wind try: <enable_wind> on the drone made Gazebo's wind flip it."""
    t = TEMPLATE.read_text()
    if "<enable_wind>true</enable_wind>" in t:
        TEMPLATE.write_text(re.sub(r"\n\s*<enable_wind>true</enable_wind>", "", t))
        print("drone model: Gazebo wind switched off again")


def set_wind(name, preset, direction=None):
    """Wind = a drag force from wind_gust.py, applied through Gazebo's ApplyLinkWrench system.
    Here: add that system to the world (+ remove Gazebo's own wind), save the preset in world.json."""
    sdf = WORLDS / name / f"{name}.sdf"
    s = settings(name)
    direction = s.get("wind_dir", 45.0) if direction is None else direction
    drone_wind_off()
    tree = ET.parse(sdf)
    world = tree.getroot().find("world")
    for e in list(world.findall("wind")):
        world.remove(e)
    for e in list(world.findall("plugin")):
        fn = (e.get("filename") or "") + (e.get("name") or "")
        if "wind-effects" in fn or "WindEffects" in fn or "apply-link-wrench" in fn:
            world.remove(e)
    if preset != "calm":
        ET.SubElement(world, "plugin", filename="gz-sim-apply-link-wrench-system",
                      name="gz::sim::systems::ApplyLinkWrench")
    ET.indent(tree, space="  ")
    tree.write(sdf, encoding="utf-8", xml_declaration=True)
    s["wind"], s["wind_dir"] = preset, direction
    (WORLDS / name / "world.json").write_text(json.dumps(s, indent=2))
    print(f"wind in {name}: {preset}" + (f", blowing towards {direction:.0f} deg" if preset != "calm" else " (off)"))


def active():
    return ACTIVE.read_text().strip() if ACTIVE.exists() else DEFAULT


def settings(name):
    f = WORLDS / name / "world.json"
    if f.exists():
        return json.loads(f.read_text())
    if name in DEFAULT_SETTINGS:                      # first time: write the default for the bridge
        f.write_text(json.dumps(DEFAULT_SETTINGS[name], indent=2))
        return DEFAULT_SETTINGS[name]
    return {"name": name, "camera_pitch_deg": 0}


def all_worlds():
    return sorted(d.name for d in WORLDS.iterdir() if d.is_dir() and (d / f"{d.name}.sdf").exists())


def sensor_block(t):
    m = re.search(r'<sensor name="fpv_camera".*?</sensor>', t, re.S)
    if not m:
        sys.exit("!! no FPV camera in the drone model - run add_camera.py first")
    return m


def camera_pitch():
    blk = sensor_block(TEMPLATE.read_text()).group(0)
    p = re.search(r"<pose>([^<]*)</pose>", blk).group(1).split()
    return math.degrees(float(p[4])) if len(p) == 6 else 0.0


def set_camera_pitch(deg):
    t = TEMPLATE.read_text()
    m = sensor_block(t)
    blk = m.group(0)
    pm = re.search(r"<pose>([^<]*)</pose>", blk)
    if not pm:
        sys.exit("!! the FPV camera has no <pose> - not the expected drone model")
    xyz = pm.group(1).split()[:3]                      # keep the camera position, change only the tilt
    pose = f"<pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 {math.radians(deg):.4f} 0</pose>"
    blk = blk[:pm.start()] + pose + blk[pm.end():]
    TEMPLATE.write_text(t[:m.start()] + blk + t[m.end():])


def link_world(name):
    sdf = WORLDS / name / f"{name}.sdf"
    link = CRAZYSIM_WORLDS / f"{name}.sdf"
    if not CRAZYSIM_WORLDS.exists():
        print(f"!! CrazySim worlds folder not found: {CRAZYSIM_WORLDS}")
        return
    if link.is_symlink() and link.resolve() == sdf.resolve():
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(sdf)
    print(f"linked {link.name} into CrazySim")


def install():
    lines = BASHRC.read_text().splitlines(keepends=True)
    want = "-w $(cat $HOME/drone_ws/sim/active_world 2>/dev/null || echo bridge_full)"
    changed = False
    for i, line in enumerate(lines):
        if line.startswith("alias simfull=") and "active_world" not in line:
            new, n = re.subn(r"-w\s+[A-Za-z0-9_]+", lambda _: want, line, count=1)
            if n != 1:
                sys.exit("!! could not find '-w <world>' in the simfull alias - nothing changed")
            lines[i], changed = new, True
    if not any(l.startswith("alias simfull=") for l in lines):
        sys.exit("!! no 'alias simfull=' line in ~/.bashrc - nothing changed")
    if not any(l.startswith("alias simworld=") for l in lines):
        lines.append("alias simworld='~/drone_ws/venv/bin/python ~/drone_ws/scripts/simworld.py'\n")
        changed = True
    if changed:
        (HOME / ".bashrc.preworld").write_text(BASHRC.read_text())
        BASHRC.write_text("".join(lines))
        print("~/.bashrc updated (backup ~/.bashrc.preworld): simfull now loads the active world, "
              "'simworld' alias added.\nRun:  source ~/.bashrc")
    else:
        print("~/.bashrc already set up")


def status():
    cur = active()
    print(f"active world : {cur}")
    print(f"camera tilt  : {camera_pitch():.0f} deg down (Gazebo drone model)")
    print("\nworlds:")
    for name in all_worlds():
        s = settings(name)
        n = len(json.loads((WORLDS / name / "decals.json").read_text())["decals"]) \
            if (WORLDS / name / "decals.json").exists() else 0
        print(f"  {'*' if name == cur else ' '} {name:16s} {s.get('kind', '?'):8s} {n:3d} defects   "
              f"model {s.get('model', '(pilot.py default)'):22s} camera {s.get('camera_pitch_deg', 0):.0f} deg down  "
              f"light {s.get('light', 'noon')}  wind {s.get('wind', 'calm')}")
    want = settings(cur).get("camera_pitch_deg", 0)
    if abs(camera_pitch() - want) > 0.5:
        print(f"\n!! camera tilt should be {want:.0f} deg for {cur} - run:  simworld {cur}")


def switch(name):
    if not (WORLDS / name / f"{name}.sdf").exists():
        sys.exit(f"!! no world '{name}'. You have: {', '.join(all_worlds())}")
    s = settings(name)
    if not (WORLDS / name / "decals.json").exists():
        print(f"!! {name} has no decals.json - pilot/autopilot cannot score it")
    if s.get("model") and not (MODELS / s["model"]).exists():
        print(f"!! model {s['model']} not found in {MODELS}")
    ACTIVE.write_text(name + "\n")
    set_camera_pitch(s.get("camera_pitch_deg", 0))
    link_world(name)
    if s.get("light") in LIGHTS:                      # a rebuilt world starts at noon: re-apply its light
        set_light(name, s["light"])
    if s.get("wind") in WIND_NAMES:                   # ... and calm: re-apply its wind
        set_wind(name, s["wind"])
    print(f"active world: {name}  ({s.get('kind', '?')})")
    print(f"  camera    : {s.get('camera_pitch_deg', 0):.0f} deg down")
    print(f"  model     : {s.get('model', '(pilot.py default)')}   conf {s.get('conf', '(pilot.py default)')}")
    print("now:  simreset ; simfull      (then pilot or autopilot)")


if __name__ == "__main__":
    args = sys.argv[1:]
    light = None
    if "--light" in args:
        i = args.index("--light")
        if i + 1 >= len(args) or args[i + 1] not in LIGHTS:
            sys.exit(f"--light needs one of: {', '.join(LIGHTS)}")
        light = args[i + 1]
        del args[i:i + 2]
    wind, wdir = None, None
    if "--wind-dir" in args:
        i = args.index("--wind-dir")
        wdir = float(args[i + 1])
        del args[i:i + 2]
    if "--wind" in args:
        i = args.index("--wind")
        if i + 1 >= len(args) or args[i + 1] not in WIND_NAMES:
            sys.exit(f"--wind needs one of: {', '.join(WIND_NAMES)}")
        wind = args[i + 1]
        del args[i:i + 2]
    if wdir is not None and wind is None:
        wind = settings(active()).get("wind", "moderate")
        wind = "moderate" if wind == "calm" else wind
    if args and args[0] == "--install":
        install()
    elif args and args[0] not in ("status", "list"):
        switch(args[0])
        if light:
            set_light(args[0], light)
        if wind:
            set_wind(args[0], wind, wdir)
    elif light or wind:
        if light:
            set_light(active(), light)
        if wind:
            set_wind(active(), wind, wdir)
        print("now:  simreset ; simfull")
    else:
        status()
