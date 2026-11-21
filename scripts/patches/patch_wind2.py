#!/usr/bin/env python3
"""
patch_wind2.py - fairer path-holding numbers in autopilot (run once, after patch_wind.py).

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/patch_wind2.py

  * PATH DEVIATION replaces 'tracking error': the distance from the drone to the planned straight
    leg it is flying (or to the stop it is holding). The old number also counted the 30 cm the
    target point runs AHEAD of the drone on purpose, so it was too big even in calm air.
  * closest to a wall now also says WHERE (phase + position), and the 5 closest moments are listed
  * tracking.csv: t, path deviation, wall distance, x, y, z, phase
Backup: autopilot.py.prewind2
"""
import shutil
import sys
from pathlib import Path

SCRIPTS = Path.home() / "drone_ws/scripts"
MARK = "# --- path deviation (patch_wind2) ---"

AUTO = [
    ("            s = 0.0\n            while s < L:\n",
     "            self.leg = (np.asarray(a, float), np.asarray(b, float))   " + MARK + "\n"
     "            s = 0.0\n            while s < L:\n"),
    ("            err = float(np.linalg.norm(p - np.asarray(self.sp[:3], float)))\n",
     "            a, b = getattr(self, \"leg\", (p, p))         # distance to the planned leg a -> b\n"
     "            ab = b - a\n"
     "            t_ = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1))\n"
     "            err = float(np.linalg.norm(p - (a + t_ * ab)))\n"),
    ("            self.track.append((time.time(), err, wall))\n",
     "            self.track.append((time.time(), err, wall, *p, self.phase))\n"),
    ("    if m.track:\n"
     "        e = np.array([r[1] for r in m.track]) * 100\n"
     "        w = np.array([r[2] for r in m.track]) * 100\n"
     "        print(f\"PATH HOLDING: tracking error mean {e.mean():.0f} cm, 95% under {np.percentile(e, 95):.0f} cm, \"\n"
     "              f\"max {e.max():.0f} cm   closest to a wall {w.min():.0f} cm\")\n"
     "        with open(out / \"tracking.csv\", \"w\", newline=\"\") as f:\n"
     "            csv.writer(f).writerows([(\"t_s\", \"tracking_err_m\", \"wall_dist_m\")] +\n"
     "                                    [(round(t - m.track[0][0], 2), round(a, 3), round(b, 3)) for t, a, b in m.track])\n",
     "    if m.track:\n"
     "        e = np.array([r[1] for r in m.track]) * 100\n"
     "        w = np.array([r[2] for r in m.track]) * 100\n"
     "        print(f\"PATH HOLDING: deviation from the planned path mean {e.mean():.1f} cm, 95% under \"\n"
     "              f\"{np.percentile(e, 95):.0f} cm, max {e.max():.0f} cm\")\n"
     "        shown = []\n"
     "        for i in np.argsort(w):                     # the 5 closest moments, at different places\n"
     "            r = m.track[i]\n"
     "            if all(np.linalg.norm(np.array(r[3:6]) - np.array(s[3:6])) > 0.5 for s in shown):\n"
     "                shown.append(r)\n"
     "            if len(shown) == 5:\n"
     "                break\n"
     "        print(\"CLOSEST TO A WALL (drone centre; the drone is ~10 cm across):\")\n"
     "        for r in shown:\n"
     "            print(f\"   {r[2] * 100:4.0f} cm  at x {r[3]:5.2f} y {r[4]:6.2f} z {r[5]:4.2f}   {r[6]}\")\n"
     "        with open(out / \"tracking.csv\", \"w\", newline=\"\") as f:\n"
     "            csv.writer(f).writerows([(\"t_s\", \"path_dev_m\", \"wall_dist_m\", \"x\", \"y\", \"z\", \"phase\")] +\n"
     "                                    [(round(r[0] - m.track[0][0], 2), round(r[1], 3), round(r[2], 3),\n"
     "                                      round(r[3], 2), round(r[4], 2), round(r[5], 2), r[6]) for r in m.track])\n"),
]


def main():
    path = SCRIPTS / "autopilot.py"
    s = path.read_text()
    if MARK in s:
        print("autopilot.py already patched")
        return
    bad = [f"{old[:70]!r} (found {s.count(old)}x)" for old, _ in AUTO if s.count(old) != 1]
    if bad:
        sys.exit("!! autopilot.py is not the expected version (run patch_wind.py first) - NOTHING changed:\n   "
                 + "\n   ".join(bad))
    for old, new in AUTO:
        s = s.replace(old, new)
    shutil.copy(path, path.with_suffix(".py.prewind2"))
    path.write_text(s)
    print("autopilot.py patched (backup autopilot.py.prewind2)")


if __name__ == "__main__":
    main()
