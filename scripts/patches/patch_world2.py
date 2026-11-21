#!/usr/bin/env python3
"""
patch_world2.py - second world patch (one time, after patch_world.py): BIG worlds of any shape.

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/patch_world2.py

  planner.py   the 3D grid takes its size from the world (world.json grid_lo / grid_hi) instead of
               the fixed bridge box, and is built much faster (each wall only touches its own cells)
  autopilot.py gives the planner the world's size
  route.py     any world whose patches have an 'order' gets its stops automatically (in that order,
               in front of each patch, as far back as fits, 16 cm from every wall);
               route.png uses the world's map size
bridge_full and road_full work exactly as before. Backups: *.py.preworld2
"""
import shutil
import sys
from pathlib import Path

SCRIPTS = Path.home() / "drone_ws/scripts"
MARK = "# --- big worlds (patch_world2) ---"

PLANNER = [
    ("    def __init__(self, boxes):\n"
     "        self.shape = tuple(int(round(v)) for v in (HI - LO) / RES)\n",
     "    def __init__(self, boxes, lo=None, hi=None):   " + MARK + "\n"
     "        self.lo = np.array(LO if lo is None else lo, float)        # grid corners: the world's size\n"
     "        self.hi = np.array(HI if hi is None else hi, float)\n"
     "        z_max = Z_MAX if hi is None else self.hi[2] - 0.2\n"
     "        self.shape = tuple(int(round(v)) for v in (self.hi - self.lo) / RES)\n"),
    ("        cx = LO[0] + (np.arange(nx) + 0.5) * RES\n"
     "        cy = LO[1] + (np.arange(ny) + 0.5) * RES\n"
     "        cz = LO[2] + (np.arange(nz) + 0.5) * RES\n",
     "        cx = self.lo[0] + (np.arange(nx) + 0.5) * RES\n"
     "        cy = self.lo[1] + (np.arange(ny) + 0.5) * RES\n"
     "        cz = self.lo[2] + (np.arange(nz) + 0.5) * RES\n"),
    ("        for lo, hi in boxes:                   # distance from every cell centre to the box\n"
     "            dx = np.maximum(np.maximum(lo[0] - cx, cx - hi[0]), 0)[:, None, None]\n"
     "            dy = np.maximum(np.maximum(lo[1] - cy, cy - hi[1]), 0)[None, :, None]\n"
     "            dz = np.maximum(np.maximum(lo[2] - cz, cz - hi[2]), 0)[None, None, :]\n"
     "            blocked |= (dx ** 2 + dy ** 2 + dz ** 2) < CLEAR ** 2\n"
     "        blocked |= (cz < Z_MIN)[None, None, :] | (cz > Z_MAX)[None, None, :]\n",
     "        for lo, hi in boxes:                   # distance from the cells NEAR the box to the box\n"
     "            i0, i1 = np.searchsorted(cx, lo[0] - CLEAR), np.searchsorted(cx, hi[0] + CLEAR)\n"
     "            j0, j1 = np.searchsorted(cy, lo[1] - CLEAR), np.searchsorted(cy, hi[1] + CLEAR)\n"
     "            k0, k1 = np.searchsorted(cz, lo[2] - CLEAR), np.searchsorted(cz, hi[2] + CLEAR)\n"
     "            if i0 >= i1 or j0 >= j1 or k0 >= k1:\n"
     "                continue\n"
     "            sx, sy, sz = cx[i0:i1], cy[j0:j1], cz[k0:k1]\n"
     "            dx = np.maximum(np.maximum(lo[0] - sx, sx - hi[0]), 0)[:, None, None]\n"
     "            dy = np.maximum(np.maximum(lo[1] - sy, sy - hi[1]), 0)[None, :, None]\n"
     "            dz = np.maximum(np.maximum(lo[2] - sz, sz - hi[2]), 0)[None, None, :]\n"
     "            blocked[i0:i1, j0:j1, k0:k1] |= (dx ** 2 + dy ** 2 + dz ** 2) < CLEAR ** 2\n"
     "        blocked |= (cz < Z_MIN)[None, None, :] | (cz > z_max)[None, None, :]\n"),
    ("        return tuple(int(np.clip((p[k] - LO[k]) / RES, 1, self.shape[k] - 2)) for k in range(3))\n",
     "        return tuple(int(np.clip((p[k] - self.lo[k]) / RES, 1, self.shape[k] - 2)) for k in range(3))\n"),
]

AUTO = [
    ("    planner = Planner(boxes)\n",
     "    planner = Planner(boxes, pilot.WORLD.get(\"grid_lo\"), pilot.WORLD.get(\"grid_hi\"))   " + MARK + "\n"),
]

ROUTE = [
    ("    if all(abs(dc[\"normal\"][2]) > 0.9 for dc in decals):\n"
     "        return ground_stops(decals, boxes)\n",
     "    if all(abs(dc[\"normal\"][2]) > 0.9 for dc in decals):\n"
     "        return ground_stops(decals, boxes)\n"
     "    if all(\"order\" in dc for dc in decals):\n"
     "        return ordered_stops(decals, boxes)\n"),
    ("def compute_stops(decals, boxes):\n",
     MARK + "\n"
     "def ordered_stops(decals, boxes, clear=0.16):\n"
     "    \"\"\"Any world: one stop per wall patch, in the patch's 'order', straight in front of it,\n"
     "    as far back as fits (1.0 .. 0.45 m) with 'clear' m to every wall and a clear view.\"\"\"\n"
     "    def free(p):\n"
     "        return all(np.linalg.norm(np.maximum(lo - p, 0) + np.maximum(p - hi, 0)) >= clear for lo, hi in boxes)\n"
     "    stops = []\n"
     "    for i, dc in enumerate(sorted(decals, key=lambda d: d[\"order\"]), 1):\n"
     "        c, n = np.array(dc[\"centre\"], float), np.array(dc[\"normal\"], float)\n"
     "        for v in dc.get(\"via\", []):\n"
     "            stops.append({\"kind\": \"via\", \"n\": i, \"id\": dc[\"id\"], \"zone\": dc[\"zone\"],\n"
     "                          \"tip\": dc.get(\"tip\", \"\"), \"pos\": np.array(v, float), \"yaw\": None, \"decal\": dc})\n"
     "        h = np.array([-n[1], n[0], 0.0])\n"
     "        corners = [c + 0.4 * (sx * dc[\"width\"] * h + sz * dc[\"height\"] * np.array([0, 0, 1.0]))\n"
     "                   for sx in (-1, 1) for sz in (-1, 1)]\n"
     "        for s in STANDOFFS:\n"
     "            p = c + s * n\n"
     "            if free(p) and all(_sees(p, q, boxes) for q in [c, *corners]):\n"
     "                break\n"
     "        else:\n"
     "            print(f\"!! {dc['id']}: no free stop in front of it\")\n"
     "            s, p = STANDOFFS[-1], c + STANDOFFS[-1] * n\n"
     "        stops.append({\"kind\": \"stop\", \"n\": i, \"id\": dc[\"id\"], \"zone\": dc[\"zone\"],\n"
     "                      \"tip\": dc.get(\"tip\", dc[\"zone\"]), \"pos\": p, \"standoff\": round(s, 2),\n"
     "                      \"yaw\": math.degrees(math.atan2(-n[1] + 0.0, -n[0] + 0.0)), \"decal\": dc})\n"
     "    return stops\n"
     "\n"
     "\n"
     "def compute_stops(decals, boxes):\n"),
    ("    ax.set_xlim(-2.5, 9.5)\n    ax.set_ylim(-11, 16)\n",
     "    ax.set_xlim(*pilot.MAP_X)\n    ax.set_ylim(*pilot.MAP_Y)\n"),
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
            sys.exit(f"!! {name} is not the expected version - NOTHING changed "
                     "(run patch_world.py first). Not found once:\n   " + "\n   ".join(bad))
        for old, new in patches:
            s = s.replace(old, new)
        todo[name] = s
    for name, s in todo.items():
        path = SCRIPTS / name
        shutil.copy(path, path.with_suffix(".py.preworld2"))
        path.write_text(s)
        print(f"{name:13s} patched (backup {name}.preworld2)")


if __name__ == "__main__":
    main()
