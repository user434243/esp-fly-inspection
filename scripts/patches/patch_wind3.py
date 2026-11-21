#!/usr/bin/env python3
"""
patch_wind3.py - autopilot blows the wind itself (run once, after patch_wind.py + patch_wind2.py).

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/patch_wind3.py

When the active world has wind (simworld --wind light|moderate|gusty|strong), autopilot starts
wind_gust.py with the drone: a gusty air-drag force pushed on the drone through Gazebo's
ApplyLinkWrench system (only once it is off the ground, weaker inside sheltered spaces), and
stops + clears it at the end. The report gets a line like
  WIND FORCE: gusty: wind at the drone mean 2.9 m/s, max 5.6 m/s; push mean 52 mN, max 190 mN
Backup: autopilot.py.prewind3
"""
import shutil
import sys
from pathlib import Path

SCRIPTS = Path.home() / "drone_ws/scripts"
MARK = "# --- wind force (patch_wind3) ---"

AUTO = [
    ("        threading.Thread(target=m.stream, daemon=True).start()\n",
     "        threading.Thread(target=m.stream, daemon=True).start()\n"
     "        from wind_gust import Gusts                 " + MARK + "\n"
     "        gust = Gusts(pilot.WORLD, pilot.WORLD_NAME, m.pos).start()\n"),
    ("        m.running = False\n",
     "        gust.stop()                                 # clear the wind force\n"
     "        m.running = False\n"),
    ("    print(f\"CAMERA: {pilot.CAM_PROFILE['name']}  lens",
     "    print(f\"WIND FORCE: {gust.summary()}\")\n"
     "    print(f\"CAMERA: {pilot.CAM_PROFILE['name']}  lens"),
]


def main():
    path = SCRIPTS / "autopilot.py"
    s = path.read_text()
    if MARK in s:
        print("autopilot.py already patched")
        return
    if not (SCRIPTS / "wind_gust.py").exists():
        sys.exit("!! copy wind_gust.py into ~/drone_ws/scripts first")
    bad = [f"{old[:70]!r} (found {s.count(old)}x)" for old, _ in AUTO if s.count(old) != 1]
    if bad:
        sys.exit("!! autopilot.py is not the expected version - NOTHING changed:\n   " + "\n   ".join(bad))
    for old, new in AUTO:
        s = s.replace(old, new)
    shutil.copy(path, path.with_suffix(".py.prewind3"))
    path.write_text(s)
    print("autopilot.py patched (backup autopilot.py.prewind3)")


if __name__ == "__main__":
    main()
