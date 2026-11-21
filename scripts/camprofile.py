#!/usr/bin/env python3
"""
camprofile.py - switch the simulated drone camera between two profiles:

  clean : the original simulator camera (90 deg lens, clean digital image)
  real  : the real kit camera AKK A11T -> 5.8 GHz -> receiver
          real lens (measured fisheye bending + dark corners, from ~/drone_ws/sim/a11t_calib.json)
          + analog NTSC look (soft, noisy, colour bleed, radio interference)

  camprofile real                   # A11T: measured lens + 'typical' radio link
  camprofile real --link bad        # weak radio link (good / typical / bad)
  camprofile real --nolens          # simple 122 deg lens without fisheye bending (old behaviour)
  camprofile real --nolens --fov 140
  camprofile clean                  # back to the clean camera
  camprofile status

It changes 2 things, which must always match:
  1. the camera in the Gazebo drone model: angle + resolution (needs a simulator restart: simreset + simfull)
     with the real lens Gazebo renders 960x720 at 136 deg, then fisheye_cam.py bends it to 640x480
  2. ~/drone_ws/sim/camera_profile.json, read by pilot.py / autopilot.py / simtest.py
The first time, it also teaches pilot.py and autopilot.py to read that file (backups: *.py.cam).
"""
import json
import math
import re
import shutil
import sys
from pathlib import Path

HOME = Path.home()
TEMPLATE = HOME / ("drone_ws/sim/CrazySim/crazyflie-firmware/tools/crazyflie-simulation/"
                   "simulator_files/gazebo/models/crazyflie/model.sdf.jinja")
PROFILE = HOME / "drone_ws/sim/camera_profile.json"
CALIB = HOME / "drone_ws/sim/a11t_calib.json"
SCRIPTS = HOME / "drone_ws/scripts"
LENS_RENDER_DEG, LENS_RENDER_SIZE = 136.0, (960, 720)     # pinhole render that covers the fisheye corners

PILOT_PATCH = [
    ("IMG_W, IMG_H, HFOV = 640, 480, 1.57\n",
     "_CAMF = HOME / \"drone_ws/sim/camera_profile.json\"          # set by camprofile.py\n"
     "CAM_PROFILE = (json.loads(_CAMF.read_text()) if _CAMF.exists()\n"
     "               else {\"name\": \"clean\", \"hfov\": 1.57, \"analog\": None})\n"
     "IMG_W, IMG_H, HFOV = 640, 480, CAM_PROFILE[\"hfov\"]\n"),
    ("    cam = GazeboCamera()\n",
     "    cam = GazeboCamera()\n"
     "    if CAM_PROFILE.get(\"analog\"):                              # real A11T analog video look\n"
     "        from analog_cam import AnalogCamera\n"
     "        cam = AnalogCamera(cam, CAM_PROFILE[\"analog\"])\n"
     "    print(f\"Camera: {CAM_PROFILE['name']}  lens {math.degrees(HFOV):.0f} deg  \"\n"
     "          f\"analog effect: {CAM_PROFILE.get('analog') or 'off'}\")\n"),
]
AUTO_PATCH = [
    ("    cam = GazeboCamera()\n",
     "    cam = GazeboCamera()\n"
     "    if pilot.CAM_PROFILE.get(\"analog\"):                        # real A11T analog video look\n"
     "        from analog_cam import AnalogCamera\n"
     "        cam = AnalogCamera(cam, pilot.CAM_PROFILE[\"analog\"])\n"
     "    print(f\"Camera: {pilot.CAM_PROFILE['name']}  lens {math.degrees(pilot.HFOV):.0f} deg  \"\n"
     "          f\"analog effect: {pilot.CAM_PROFILE.get('analog') or 'off'}\")\n"),
    ("    print(f\"\\nAUTONOMOUS MISSION: {len(m.log)}/{n_stops} stops scanned, \"\n",
     "    print(f\"\\nCAMERA: {pilot.CAM_PROFILE['name']}  lens {math.degrees(pilot.HFOV):.0f} deg  \"\n"
     "          f\"analog effect: {pilot.CAM_PROFILE.get('analog') or 'off'}\")\n"
     "    print(f\"AUTONOMOUS MISSION: {len(m.log)}/{n_stops} stops scanned, \"\n"),
]


def patch_script(path, patches, marker):
    s = path.read_text()
    if marker in s:
        return "already reads the camera profile"
    missing = [old[:60] for old, _ in patches if s.count(old) != 1]
    if missing:
        sys.exit(f"!! {path.name} is not the expected version, nothing changed. Not found:\n   "
                 + "\n   ".join(missing))
    shutil.copy(path, path.with_suffix(".py.cam"))
    for old, new in patches:
        s = s.replace(old, new)
    path.write_text(s)
    return "now reads the camera profile (backup: " + path.with_suffix(".py.cam").name + ")"


def sensor_block(t):
    m = re.search(r'<sensor name="fpv_camera".*?</sensor>', t, re.S)
    if not m:
        sys.exit("!! no FPV camera in the drone model - run:  python ~/drone_ws/sim/patches/add_camera.py")
    return m


def set_camera(hfov_rad, size):
    t = TEMPLATE.read_text()
    m = sensor_block(t)
    blk = m.group(0)
    blk, n1 = re.subn(r"(<horizontal_fov>)[^<]*(</horizontal_fov>)", rf"\g<1>{hfov_rad:.4f}\g<2>", blk, count=1)
    blk, n2 = re.subn(r"(<width>)[^<]*(</width>)", rf"\g<1>{size[0]}\g<2>", blk, count=1)
    blk, n3 = re.subn(r"(<height>)[^<]*(</height>)", rf"\g<1>{size[1]}\g<2>", blk, count=1)
    if (n1, n2, n3) != (1, 1, 1):
        sys.exit("!! could not find <horizontal_fov>, <width> and <height> of the FPV camera")
    TEMPLATE.write_text(t[:m.start()] + blk + t[m.end():])


def status():
    blk = sensor_block(TEMPLATE.read_text()).group(0)
    gz = float(re.search(r"<horizontal_fov>([^<]*)</horizontal_fov>", blk).group(1))
    gw = int(re.search(r"<width>([^<]*)</width>", blk).group(1))
    gh = int(re.search(r"<height>([^<]*)</height>", blk).group(1))
    p = json.loads(PROFILE.read_text()) if PROFILE.exists() else {"name": "clean", "hfov": 1.57, "analog": None}
    want_fov = p.get("render_hfov", p["hfov"])
    want_size = tuple(p.get("render_size", (640, 480)))
    lens = "real A11T fisheye (" + Path(p["lens"]).name + ")" if p.get("lens") else "simple (no fisheye)"
    print(f"profile file : {p['name']}  lens {math.degrees(p['hfov']):.0f} deg  {lens}  analog: {p.get('analog') or 'off'}")
    print(f"gazebo model : renders {gw}x{gh} at {math.degrees(gz):.0f} deg")
    ok = abs(gz - want_fov) < 1e-3 and (gw, gh) == want_size
    print("OK - both match" if ok else "!! MISMATCH - run camprofile again")


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("clean", "real", "status"):
        print(__doc__)
        return
    if args[0] == "status":
        status()
        return
    link = args[args.index("--link") + 1] if "--link" in args else "typical"
    if link not in ("good", "typical", "bad"):
        sys.exit("--link must be good, typical or bad")
    use_lens = args[0] == "real" and "--nolens" not in args
    if use_lens and not CALIB.exists():
        sys.exit(f"!! {CALIB} not found - create it first, or use:  camprofile real --nolens")
    if use_lens:
        c = json.loads(CALIB.read_text())
        if "--fov" in args:
            print("note: --fov is ignored with the real lens (the angle comes from the calibration)")
        prof = {"name": "real A11T", "hfov": round(math.radians(c["hfov_deg"]), 4), "analog": link,
                "lens": str(CALIB), "render_hfov": round(math.radians(LENS_RENDER_DEG), 4),
                "render_size": list(LENS_RENDER_SIZE)}
        cam_fov, cam_size = prof["render_hfov"], LENS_RENDER_SIZE
    else:
        fov = float(args[args.index("--fov") + 1]) if "--fov" in args else (122.0 if args[0] == "real" else 90.0)
        hfov = 1.57 if args[0] == "clean" and "--fov" not in args else math.radians(fov)
        prof = {"name": "real A11T" if args[0] == "real" else "clean", "hfov": round(hfov, 4),
                "analog": link if args[0] == "real" else None}
        cam_fov, cam_size = prof["hfov"], (640, 480)

    print("pilot.py    :", patch_script(SCRIPTS / "pilot.py", PILOT_PATCH, "CAM_PROFILE"))
    if (SCRIPTS / "autopilot.py").exists():
        print("autopilot.py:", patch_script(SCRIPTS / "autopilot.py", AUTO_PATCH, "pilot.CAM_PROFILE"))
    if not (SCRIPTS / "analog_cam.py").exists():
        sys.exit("!! analog_cam.py missing in ~/drone_ws/scripts")
    if use_lens and not (SCRIPTS / "fisheye_cam.py").exists():
        sys.exit("!! fisheye_cam.py missing in ~/drone_ws/scripts")
    set_camera(cam_fov, cam_size)
    PROFILE.write_text(json.dumps(prof, indent=2))
    lens = "real A11T fisheye + dark corners" if use_lens else "simple"
    print(f"\nCamera profile -> {prof['name']}: lens {math.degrees(prof['hfov']):.0f} deg ({lens}), "
          f"analog effect: {prof['analog'] or 'off'}")
    print(f"Gazebo renders {cam_size[0]}x{cam_size[1]} at {math.degrees(cam_fov):.0f} deg")
    print("Restart the simulator for the new camera:   simreset   then   simfull")


if __name__ == "__main__":
    main()
