#!/usr/bin/env python3
"""
patch_wind.py - autopilot measures how well the drone holds its path (one time).

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/patch_wind.py

While flying, 20 times a second, autopilot.py now records
  * tracking error: distance between where the drone IS and where it was TOLD to be
    (calm air ~ a few cm; wind pushes it off)
  * closest distance to any wall of the world (the safety margin that is left)
and prints them at the end with the world's wind and light:
  WIND  : gusty (towards 45 deg)   LIGHT: noon
  PATH HOLDING: tracking error mean 4 cm, 95% under 9 cm, max 21 cm   closest to a wall 11 cm
They are also saved in mission_log.csv's folder as tracking.csv.
Backup: autopilot.py.prewind
"""
import shutil
import sys
from pathlib import Path

SCRIPTS = Path.home() / "drone_ws/scripts"
MARK = "# --- path holding (patch_wind) ---"

AUTO = [
    ("from planner import Planner\n",
     "from planner import Planner\n" + MARK + "\nWALLS = None                                   # (lo, hi) arrays of every obstacle box\n"),
    ("        self.running = True\n",
     "        self.running = True\n"
     "        self.track = []                      # (t, tracking error m, wall distance m) while flying\n"),
    ("            raise Abort()\n        time.sleep(1.0 / RATE)\n",
     "            raise Abort()\n"
     "        if self.flying and self.sp is not None:\n"
     "            p = self.pos()\n"
     "            err = float(np.linalg.norm(p - np.asarray(self.sp[:3], float)))\n"
     "            wall = 9.9\n"
     "            if WALLS is not None:\n"
     "                lo, hi = WALLS\n"
     "                wall = float(np.sqrt((np.maximum(np.maximum(lo - p, p - hi), 0) ** 2).sum(1)).min())\n"
     "            self.track.append((time.time(), err, wall))\n"
     "        time.sleep(1.0 / RATE)\n"),
    ("    decals, parts, boxes = load_world()\n",
     "    decals, parts, boxes = load_world()\n"
     "    globals()[\"WALLS\"] = (np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes]))\n"),
    ("    print(f\"MODEL : {pilot.MODEL.name}\")\n",
     "    print(f\"MODEL : {pilot.MODEL.name}\")\n"
     "    print(f\"WIND  : {pilot.WORLD.get('wind', 'calm')}\"\n"
     "          + (f\" (towards {pilot.WORLD.get('wind_dir', 45):.0f} deg)\" if pilot.WORLD.get('wind', 'calm') != 'calm' else '')\n"
     "          + f\"   LIGHT: {pilot.WORLD.get('light', 'noon')}\")\n"
     "    if m.track:\n"
     "        e = np.array([r[1] for r in m.track]) * 100\n"
     "        w = np.array([r[2] for r in m.track]) * 100\n"
     "        print(f\"PATH HOLDING: tracking error mean {e.mean():.0f} cm, 95% under {np.percentile(e, 95):.0f} cm, \"\n"
     "              f\"max {e.max():.0f} cm   closest to a wall {w.min():.0f} cm\")\n"
     "        with open(out / \"tracking.csv\", \"w\", newline=\"\") as f:\n"
     "            csv.writer(f).writerows([(\"t_s\", \"tracking_err_m\", \"wall_dist_m\")] +\n"
     "                                    [(round(t - m.track[0][0], 2), round(a, 3), round(b, 3)) for t, a, b in m.track])\n"),
]


def main():
    path = SCRIPTS / "autopilot.py"
    s = path.read_text()
    if MARK in s:
        print("autopilot.py already patched")
        return
    bad = [f"{old[:70]!r} (found {s.count(old)}x)" for old, _ in AUTO if s.count(old) != 1]
    if bad:
        sys.exit("!! autopilot.py is not the expected version - NOTHING changed. Not found once:\n   "
                 + "\n   ".join(bad))
    for old, new in AUTO:
        s = s.replace(old, new)
    shutil.copy(path, path.with_suffix(".py.prewind"))
    path.write_text(s)
    print("autopilot.py patched (backup autopilot.py.prewind)")


if __name__ == "__main__":
    main()
