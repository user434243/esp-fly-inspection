#!/usr/bin/env python3
"""patch_stab.py - use box_stabilizer.py (steady boxes) in pilot.py and autopilot.py.
Turn it off any time in pilot.py:  STABILIZE = False      Backups: *.py.prestab   Safe to run twice."""
import re
import shutil
import sys
from pathlib import Path

S = Path.home() / "drone_ws/scripts"
ZIP = re.compile(r"zip\(res\.boxes\.xyxy\.tolist\(\), res\.boxes\.conf\.tolist\(\),\s*res\.boxes\.cls\.tolist\(\)\)")
if not (S / "box_stabilizer.py").exists():
    sys.exit("!! box_stabilizer.py missing in ~/drone_ws/scripts - create it first")
todo = []
for name, call in (("pilot.py", "STAB.update(res)"), ("autopilot.py", "pilot.STAB.update(res)")):
    p = S / name
    s = p.read_text()
    if "STAB" in s:
        print(f"{name:13s}: already uses the box stabilizer")
        continue
    s, n = ZIP.subn(call, s)
    if n != 1:
        sys.exit(f"!! {name}: detection loop not found exactly once ({n}) - nothing changed")
    if name == "pilot.py":
        s, k = re.subn(r"(\nWEAK_VOTE = [^\n]*\n)",
                       r"\1STABILIZE = True        # steady boxes: shown after 3 hits in 5 frames, no blinking\n"
                       r"from box_stabilizer import BoxStabilizer\n"
                       r"STAB = BoxStabilizer(STABILIZE)\n", s, count=1)
        if k != 1:
            sys.exit("!! pilot.py: WEAK_VOTE line not found - run the two-threshold vote patch first; nothing changed")
    if ".plot(" in s:
        print(f"note: {name} also draws with res.plot() - those drawings are not stabilized")
    todo.append((p, s))
for p, s in todo:
    shutil.copy(p, p.with_suffix(".py.prestab"))
    p.write_text(s)
    print(f"{p.name:13s}: steady boxes on (backup {p.stem}.py.prestab)")
