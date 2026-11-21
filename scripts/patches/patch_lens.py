#!/usr/bin/env python3
"""
patch_lens.py - teach pilot.py, autopilot.py, simtest.py and simdist.py the real A11T lens.

  pixel <-> 3D direction now goes through the measured fisheye model when the camera profile has a lens,
  the camera image is bent through the lens (+ dark corners) before the analog effect,
  survey frames record that lens, and simtest/simdist read it back.
Backups: *.py.prelens      Safe to run twice.
"""
import shutil
import sys
from pathlib import Path

S = Path.home() / "drone_ws/scripts"

PILOT = [
    ("\nFOCAL = (IMG_W / 2) / math.tan(HFOV / 2)\n",
     "\nFOCAL = (IMG_W / 2) / math.tan(HFOV / 2)\n"
     "LENS = None                                                    # real-lens model (fisheye_cam.Lens) or pinhole\n"
     "if CAM_PROFILE.get(\"lens\"):\n"
     "    from fisheye_cam import Lens\n"
     "    LENS = Lens(CAM_PROFILE[\"lens\"], CAM_PROFILE[\"render_hfov\"])\n"
     "\n\n"
     "def pix_to_norm(u, v):\n"
     "    \"\"\"Image pixel -> direction at depth 1 (x right, y down).\"\"\"\n"
     "    if LENS is not None:\n"
     "        return LENS.pix_to_norm(u, v)\n"
     "    return (u - IMG_W / 2) / FOCAL, (v - IMG_H / 2) / FOCAL\n"
     "\n\n"
     "def norm_to_pix(x, y):\n"
     "    \"\"\"Direction at depth 1 (x right, y down) -> image pixel.\"\"\"\n"
     "    if LENS is not None:\n"
     "        return LENS.norm_to_pix(x, y)\n"
     "    return IMG_W / 2 + x * FOCAL, IMG_H / 2 + y * FOCAL\n"),
    ("    d = np.array([1.0, -(u - IMG_W / 2) / FOCAL, -(v - IMG_H / 2) / FOCAL])\n",
     "    x, y = pix_to_norm(u, v)\n"
     "    d = np.array([1.0, -x, -y])\n"),
    ("    u, v = IMG_W / 2 - b[1] / b[0] * FOCAL, IMG_H / 2 - b[2] / b[0] * FOCAL\n",
     "    u, v = norm_to_pix(-b[1] / b[0], -b[2] / b[0])\n"),
    ("    cam = GazeboCamera()\n",
     "    cam = GazeboCamera()\n"
     "    if LENS is not None:                                       # real A11T fisheye lens + dark corners\n"
     "        cam = LENS.wrap(cam)\n"),
]
AUTO = [
    ("    return (int(max(-5000, min(5000, IMG_W / 2 - b[1] / b[0] * FOCAL))),\n"
     "            int(max(-5000, min(5000, IMG_H / 2 - b[2] / b[0] * FOCAL))))\n",
     "    u, v = pilot.norm_to_pix(-b[1] / b[0], -b[2] / b[0])\n"
     "    return (int(max(-5000, min(5000, u))),\n"
     "            int(max(-5000, min(5000, v))))\n"),
    ("    cam = GazeboCamera()\n",
     "    cam = GazeboCamera()\n"
     "    if pilot.LENS is not None:                                 # real A11T fisheye lens + dark corners\n"
     "        cam = pilot.LENS.wrap(cam)\n"),
]
SIM = [
    ("    pilot.FOCAL = (pilot.IMG_W / 2) / math.tan(pilot.HFOV / 2)\n",
     "    pilot.FOCAL = (pilot.IMG_W / 2) / math.tan(pilot.HFOV / 2)\n"
     "    pilot.LENS = None                                              # survey without lens = pinhole frames\n"
     "    if cam.get(\"lens\"):                                            # survey recorded through the real lens\n"
     "        from fisheye_cam import Lens\n"
     "        pilot.LENS = Lens(cam[\"lens\"], cam[\"render_hfov\"])\n"
     "        pilot.FOCAL = pilot.LENS.fx\n"),
]
FILES = [("pilot.py", PILOT), ("autopilot.py", AUTO), ("simtest.py", SIM), ("simdist.py", SIM)]

if not (S / "fisheye_cam.py").exists():
    sys.exit("!! fisheye_cam.py missing in ~/drone_ws/scripts - create it first")
plan = []
for name, patches in FILES:
    p = S / name
    if not p.exists():
        print(f"{name:13s}: not found - skipped")
        continue
    s = p.read_text()
    if "LENS" in s:
        print(f"{name:13s}: already knows the lens")
        continue
    if name == "autopilot.py" and "import pilot" not in s:
        sys.exit("!! autopilot.py does not import pilot - nothing changed")
    missing = [old.strip()[:70] for old, _ in patches if s.count(old) != 1]
    if missing:
        sys.exit(f"!! {name} is not the expected version, nothing changed. Not found:\n   " + "\n   ".join(missing))
    for old, new in patches:
        s = s.replace(old, new)
    plan.append((p, s))
for p, s in plan:
    shutil.copy(p, p.with_suffix(".py.prelens"))
    p.write_text(s)
    print(f"{p.name:13s}: now uses the real lens model (backup: {p.stem}.py.prelens)")
