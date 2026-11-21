#!/usr/bin/env python3
"""
patch_world.py - teach pilot.py, autopilot.py and route.py to work in ANY world (one time).

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/patch_world.py

After this they read the active world (~/drone_ws/sim/active_world, set by 'simworld'):
  * decals.json, parts.json and world.json of that world (bridge_full keeps working as before)
  * the YOLO model / conf / map size of the world (world.json)
  * a camera tilted down (camera_pitch_deg) in all the 3D math: pixel -> ray, 3D -> pixel
  * defect patches on the GROUND (roads), not only on walls
  * route.py: on a ground world the stops are above + behind each patch, the camera looking down at it
Backups: *.py.preworld
"""
import shutil
import sys
from pathlib import Path

SCRIPTS = Path.home() / "drone_ws/scripts"
MARK = "# --- active world (simworld) ---"

PILOT = [
    # world block: after the thresholds, so the world can override MODEL and CONF
    ("WEAK_VOTE = 0.01        # weaker ones (CONF..CLASS_CONF) only count when nothing sure was seen\n",
     "WEAK_VOTE = 0.01        # weaker ones (CONF..CLASS_CONF) only count when nothing sure was seen\n"
     + MARK + "\n"
     "_ACTIVE = HOME / \"drone_ws/sim/active_world\"\n"
     "WORLD_NAME = _ACTIVE.read_text().strip() if _ACTIVE.exists() else \"bridge_full\"\n"
     "WORLD_DIR = HOME / \"drone_ws/sim/worlds\" / WORLD_NAME\n"
     "WORLD = json.loads((WORLD_DIR / \"world.json\").read_text()) if (WORLD_DIR / \"world.json\").exists() else {}\n"
     "DECALS = WORLD_DIR / \"decals.json\"\n"
     "if WORLD.get(\"model\"):\n"
     "    MODEL = HOME / \"drone_ws/yolo/models\" / WORLD[\"model\"]\n"
     "CONF = WORLD.get(\"conf\", CONF)\n"),
    ("CAM_OFFSET = np.array([0.03, 0.0, 0.015])\n",
     "CAM_OFFSET = np.array([0.03, 0.0, 0.015])\n"
     "CAM_PITCH = math.radians(WORLD.get(\"camera_pitch_deg\", 0.0))      # camera tilted down (roads)\n"
     "CAM_ROT = np.array([[math.cos(CAM_PITCH), 0, math.sin(CAM_PITCH)], [0, 1, 0],\n"
     "                    [-math.sin(CAM_PITCH), 0, math.cos(CAM_PITCH)]])   # camera -> body\n"),
    ("MAP_X, MAP_Y = (-4.0, 10.0), (-12.0, 18.0)\n",
     "MAP_X, MAP_Y = tuple(WORLD.get(\"map_x\", (-4.0, 10.0))), tuple(WORLD.get(\"map_y\", (-12.0, 18.0)))\n"),
    ("    d = rot @ (d / np.linalg.norm(d))\n",
     "    d = rot @ (CAM_ROT @ (d / np.linalg.norm(d)))\n"),
    ("    b = rot.T @ (np.asarray(P, float) - np.asarray(pos, float) - rot @ CAM_OFFSET)\n",
     "    b = CAM_ROT.T @ (rot.T @ (np.asarray(P, float) - np.asarray(pos, float) - rot @ CAM_OFFSET))\n"),
    ("        if abs(off @ dc[\"_h\"]) > dc[\"width\"] / 2 + margin or abs(off[2]) > dc[\"height\"] / 2 + margin:\n",
     "        if abs(off @ dc[\"_u\"]) > dc[\"width\"] / 2 + margin or abs(off @ dc[\"_v\"]) > dc[\"height\"] / 2 + margin:\n"),
    ("        dc[\"_h\"] = np.array([-dc[\"_n\"][1], dc[\"_n\"][0], 0.0])\n"
     "    import make_bridge_full as mbf\n"
     "    parts = mbf.build_parts()\n"
     "    boxes = [(np.array(p[\"lo\"]), np.array(p[\"hi\"])) for p in parts if p[\"name\"] not in (\"ground\", \"river_bed\")]\n",
     "        if abs(dc[\"_n\"][2]) > 0.9:                 # patch on the ground: 'yaw' turns its width axis\n"
     "            a = math.radians(dc.get(\"yaw\", 0.0))\n"
     "            dc[\"_u\"] = np.array([math.cos(a), math.sin(a), 0.0])\n"
     "        else:                                     # patch on a wall: width horizontal\n"
     "            dc[\"_u\"] = np.array([-dc[\"_n\"][1], dc[\"_n\"][0], 0.0])\n"
     "        dc[\"_v\"] = np.cross(dc[\"_n\"], dc[\"_u\"])        # height axis (up on walls)\n"
     "        dc[\"_h\"] = dc[\"_u\"]\n"
     "    if (WORLD_DIR / \"parts.json\").exists():\n"
     "        parts = json.loads((WORLD_DIR / \"parts.json\").read_text())\n"
     "    else:\n"
     "        import make_bridge_full as mbf\n"
     "        parts = mbf.build_parts()\n"
     "    boxes = [(np.array(p[\"lo\"]), np.array(p[\"hi\"])) for p in parts\n"
     "             if p[\"name\"] not in (\"ground\", \"river_bed\") and not p.get(\"flat\")]\n"),
    ("                                       b * dc[\"height\"] / 2 * np.array([0, 0, 1.0]), pos, yaw)\n",
     "                                       b * dc[\"height\"] / 2 * dc[\"_v\"], pos, yaw)\n"),
    ("    for mat in (\"grass\", \"weathered\", \"concrete\", \"steel\"):\n",
     "    for mat in (\"grass\", \"asphalt\", \"weathered\", \"concrete\", \"steel\"):\n"),
    ("    for mat in (\"grass\", \"gravel\", \"weathered\", \"concrete\", \"steel\"):\n",
     "    for mat in (\"grass\", \"gravel\", \"asphalt\", \"weathered\", \"concrete\", \"steel\"):\n"),
    ("    ax.set_xlim(-4, 10)\n    ax.set_ylim(-12, 18)\n",
     "    ax.set_xlim(*MAP_X)\n    ax.set_ylim(*MAP_Y)\n"),
    ("    ax.set_title(f\"Free-flight bridge inspection:",
     "    ax.set_title(f\"Free-flight {WORLD.get('kind', 'bridge')} inspection ({WORLD_NAME}):"),
]

AUTO = [
    ("CLOSER = (0.6, 0.45, 0.35)                        # extra views closer to the defect (m from surface)\n",
     "CLOSER = tuple(pilot.WORLD.get(\"closer\", (0.6, 0.45, 0.35)))   # extra views closer (m from the patch)\n"
     + MARK + "\n"),
    ("    b = rot.T @ (np.asarray(P, float) - np.asarray(pos, float) - rot @ CAM_OFFSET)\n",
     "    b = pilot.CAM_ROT.T @ (rot.T @ (np.asarray(P, float) - np.asarray(pos, float) - rot @ CAM_OFFSET))\n"),
    ("            q = c + n * d\n",
     "            q = it[\"view_at\"](d) if \"view_at\" in it else c + n * d\n"),
    ("                                   b * dc[\"height\"] / 2 * np.array([0, 0, 1.0]), pos, yaw)\n",
     "                                   b * dc[\"height\"] / 2 * dc[\"_v\"], pos, yaw)\n"),
    ("    colours = {\"grass\": (70, 95, 50), \"weathered\": (120, 120, 120), \"concrete\": (150, 150, 150),\n",
     "    colours = {\"grass\": (70, 95, 50), \"asphalt\": (60, 60, 65), \"weathered\": (120, 120, 120), \"concrete\": (150, 150, 150),\n"),
    ("    for mat in (\"grass\", \"weathered\", \"concrete\", \"steel\"):\n",
     "    for mat in (\"grass\", \"asphalt\", \"weathered\", \"concrete\", \"steel\"):\n"),
    ("        (survey_dir / \"camera.json\").write_text(json.dumps(pilot.CAM_PROFILE))\n",
     "        (survey_dir / \"camera.json\").write_text(json.dumps({**pilot.CAM_PROFILE, \"world\": pilot.WORLD_NAME}))\n"),
    ("    print(f\"\\nMODEL : {pilot.MODEL.name}\")\n",
     "    print(f\"\\nWORLD : {pilot.WORLD_NAME}\")\n    print(f\"MODEL : {pilot.MODEL.name}\")\n"),
]

ROUTE = [
    ("def compute_stops(decals, boxes):\n"
     "    \"\"\"decals/boxes as returned by pilot.load_world(). Returns the route as a list of dicts.\"\"\"\n",
     MARK + "\n"
     "def world_settings():\n"
     "    import json\n"
     "    act = Path.home() / \"drone_ws/sim/active_world\"\n"
     "    name = act.read_text().strip() if act.exists() else \"bridge_full\"\n"
     "    f = Path.home() / \"drone_ws/sim/worlds\" / name / \"world.json\"\n"
     "    return name, (json.loads(f.read_text()) if f.exists() else {})\n"
     "\n"
     "\n"
     "def ground_stops(decals, boxes):\n"
     "    \"\"\"Patches on the ground (roads): stop ABOVE and BEHIND each patch, facing along the road,\n"
     "    so the tilted camera (camera_pitch_deg) looks straight at the patch centre from 'standoff' m.\"\"\"\n"
     "    _, w = world_settings()\n"
     "    P = math.radians(w.get(\"camera_pitch_deg\", 55.0))\n"
     "    standoff = w.get(\"standoff\", 1.4)\n"
     "    stops = []\n"
     "    for i, dc in enumerate(sorted(decals, key=lambda d: d.get(\"order\", 0)), 1):\n"
     "        c = np.array(dc[\"centre\"], float)\n"
     "        yaw = float(dc.get(\"approach_yaw\", 90.0))\n"
     "        f = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])\n"
     "\n"
     "        def view_at(d, c=c, f=f):\n"
     "            return c - f * d * math.cos(P) + np.array([0.0, 0.0, d * math.sin(P)])\n"
     "\n"
     "        s = standoff\n"
     "        while s > 0.6 and not _free(view_at(s), boxes):\n"
     "            s -= 0.05\n"
     "        p = view_at(s)\n"
     "        stops.append({\"kind\": \"stop\", \"n\": i, \"id\": dc[\"id\"], \"zone\": dc[\"zone\"],\n"
     "                      \"tip\": f\"fly {dc.get('lane', 'along the road')}, {p[2]:.2f} m high, \"\n"
     "                             f\"{s * math.cos(P):.2f} m before the patch\",\n"
     "                      \"pos\": p, \"standoff\": round(s, 2), \"yaw\": yaw, \"decal\": dc, \"view_at\": view_at})\n"
     "    return stops\n"
     "\n"
     "\n"
     "def compute_stops(decals, boxes):\n"
     "    \"\"\"decals/boxes as returned by pilot.load_world(). Returns the route as a list of dicts.\"\"\"\n"
     "    if all(abs(dc[\"normal\"][2]) > 0.9 for dc in decals):\n"
     "        return ground_stops(decals, boxes)\n"),
    ("    colours = {\"concrete\": \"0.75\", \"weathered\": \"0.55\", \"grass\": \"#b5c99a\", \"steel\": \"#8a5a3a\"}\n",
     "    colours = {\"concrete\": \"0.75\", \"weathered\": \"0.55\", \"grass\": \"#b5c99a\", \"steel\": \"#8a5a3a\",\n"
     "               \"asphalt\": \"0.35\"}\n"),
    ("    for mat in (\"grass\", \"weathered\", \"concrete\", \"steel\"):\n",
     "    for mat in (\"grass\", \"asphalt\", \"weathered\", \"concrete\", \"steel\"):\n"),
    ("            if p[\"mat\"] == mat and p[\"name\"] not in (\"ground\", \"deck_slab\"):\n",
     "            if p[\"mat\"] == mat and p[\"name\"] not in (\"ground\", \"deck_slab\") and not p[\"name\"].startswith(\"road\"):\n"),
    ("    out = Path.home() / \"drone_ws/sim/worlds/bridge_full/route.png\"\n",
     "    out = pilot.WORLD_DIR / \"route.png\"\n"),
]


def patch(name, patches):
    path = SCRIPTS / name
    s = path.read_text()
    if MARK in s:
        return f"{name:13s} already patched"
    bad = [f"{old[:70]!r} (found {s.count(old)}x)" for old, _ in patches if s.count(old) != 1]
    if bad:
        sys.exit(f"!! {name} is not the expected version - NOTHING changed. Not found once:\n   " + "\n   ".join(bad))
    return s


def main():
    todo = {}
    for name, patches in (("pilot.py", PILOT), ("autopilot.py", AUTO), ("route.py", ROUTE)):
        s = patch(name, patches)
        if s.endswith("already patched"):
            print(s)
            continue
        for old, new in patches:
            s = s.replace(old, new)
        todo[name] = s
    for name, s in todo.items():                       # all checked first -> all or nothing
        path = SCRIPTS / name
        shutil.copy(path, path.with_suffix(".py.preworld"))
        path.write_text(s)
        print(f"{name:13s} patched (backup {name}.preworld)")


if __name__ == "__main__":
    main()
