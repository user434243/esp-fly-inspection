#!/usr/bin/env python3
"""
patch_margin.py - the safety margin to walls grows with the wind (run once, after patch_wind3.py).

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/patch_margin.py

Calm air: the drone centre keeps 14 cm from every surface (as before). In light wind the test
showed the drone drifting to 4 cm, so the planner now keeps more room when there is wind:
      calm 14 cm   light 20 cm   moderate 24 cm   gusty 26 cm   strong 28 cm
(world.json "clear": 0.18 overrides it). Stops in front of patches keep 2 cm more than that.
The narrowest places still fit: 0.7 m manhole -> 0.30 m of free path at 20 cm.
autopilot prints:  SAFETY MARGIN: 20 cm (light wind)
Backups: *.py.premargin
"""
import shutil
import sys
from pathlib import Path

SCRIPTS = Path.home() / "drone_ws/scripts"
MARK = "# --- wind margin (patch_margin) ---"
BY_WIND = '{"calm": 0.14, "light": 0.20, "moderate": 0.24, "gusty": 0.26, "strong": 0.28}'

PLANNER = [
    ("    def __init__(self, boxes, lo=None, hi=None):",
     "    def __init__(self, boxes, lo=None, hi=None, clear=None):"),
    ("        z_max = Z_MAX if hi is None else self.hi[2] - 0.2\n",
     "        z_max = Z_MAX if hi is None else self.hi[2] - 0.2\n"
     "        CLEAR = globals()[\"CLEAR\"] if clear is None else clear   " + MARK + "\n"
     "        self.clear = CLEAR\n"),
]

AUTO = [
    ("    planner = Planner(boxes, pilot.WORLD.get(\"grid_lo\"), pilot.WORLD.get(\"grid_hi\"))",
     "    margin = pilot.WORLD.get(\"clear\", " + BY_WIND + ".get(pilot.WORLD.get(\"wind\", \"calm\"), 0.14))   " + MARK + "\n"
     "    print(f\"SAFETY MARGIN: {margin * 100:.0f} cm ({pilot.WORLD.get('wind', 'calm')} wind)\")\n"
     "    planner = Planner(boxes, pilot.WORLD.get(\"grid_lo\"), pilot.WORLD.get(\"grid_hi\"), margin)"),
    ("    print(f\"WIND FORCE: {gust.summary()}\")\n",
     "    print(f\"WIND FORCE: {gust.summary()}\")\n"
     "    print(f\"SAFETY MARGIN: {planner.clear * 100:.0f} cm from every surface (drone centre)\")\n"),
]

ROUTE = [
    ("def ordered_stops(decals, boxes, clear=0.16):\n",
     "def ordered_stops(decals, boxes, clear=None):   " + MARK + "\n"),
    ("    as far back as fits (1.0 .. 0.45 m) with 'clear' m to every wall and a clear view.\"\"\"\n",
     "    as far back as fits (1.0 .. 0.45 m) with 'clear' m to every wall and a clear view.\"\"\"\n"
     "    if clear is None:                          # 2 cm more than the planner's wind margin\n"
     "        _, w = world_settings()\n"
     "        clear = w.get(\"clear\", " + BY_WIND + ".get(w.get(\"wind\", \"calm\"), 0.14)) + 0.02\n"),
]


def main():
    todo = {}
    for name, patches in (("planner.py", PLANNER), ("autopilot.py", AUTO), ("route.py", ROUTE)):
        s = (SCRIPTS / name).read_text()
        if MARK in s:
            print(f"{name:13s} already patched")
            continue
        bad = [f"{old[:70]!r} (found {s.count(old)}x)" for old, _ in patches if s.count(old) != 1]
        if bad:
            sys.exit(f"!! {name} is not the expected version - NOTHING changed:\n   " + "\n   ".join(bad))
        for old, new in patches:
            s = s.replace(old, new)
        todo[name] = s
    for name, s in todo.items():
        path = SCRIPTS / name
        shutil.copy(path, path.with_suffix(".py.premargin"))
        path.write_text(s)
        print(f"{name:13s} patched (backup {name}.premargin)")


if __name__ == "__main__":
    main()
