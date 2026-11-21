#!/usr/bin/env python3
"""
route.py - the fixed demo route through all 20 defect patches of bridge_full.

For every patch it computes a STOP POINT: straight in front of the patch, at the patch height,
facing it, as far back as the space allows (1.0 m, less inside the narrow girder bays and the
culvert), with 12 cm clearance from every wall and a clear line of sight.
pilot.py shows the next stop on the camera view, the HUD and the mini-map.

Run alone to print the route and save a printable map:
    ~/drone_ws/venv/bin/python ~/drone_ws/scripts/route.py
    -> ~/drone_ws/sim/worlds/bridge_full/route.png
"""
import math
from pathlib import Path

import numpy as np

CLEAR = 0.12                       # min distance drone centre <-> any wall (m)
STANDOFFS = [1.0 - 0.05 * i for i in range(12)]    # 1.0 ... 0.45 m, first one that fits

ROUTE = [   # (patch, how to get there, optional waypoints (x, y) to fly through first)
    ("D01", "take off (T), climb to 1.0 m, fly straight to column W", []),
    ("D02", "go round the south side of column W, low (0.6 m)", []),
    ("D06", "climb to 2.0 m, slide east, face the pier cap", []),
    ("D04", "descend to 0.8 m, go round the east side of column E", []),
    ("D03", "north side of column E, 1.2 m", []),
    ("D07", "climb to 2.0 m, slide west along the pier cap", []),
    ("D05", "descend to 1.4 m FIRST, then enter the gap between the columns from the north", []),
    ("D08", "TAB precision: go south under bay 2, climb up through the flange gap", []),
    ("D09", "drop below the girders (2.3 m), move to bay 1, climb up through the gap", []),
    ("D20", "drop below the girders, fly out west, climb to 3.85 m beside the parapet", []),
    ("D15", "descend to 1.2 m, fly south to the abutment", []),
    ("D11", "climb up into bay 3 through the flange gap", []),
    ("D14", "drop below the girders, fly out east past the deck edge", []),
    ("D12", "fly north along the east side, climb into bay 3 at the diaphragm", []),
    ("D10", "drop below the girders, move to bay 2, climb up", []),
    ("D13", "drop below the girders, fly out west past the deck edge", []),
    ("D17", "back under the deck, 1.8 m, face the north abutment", []),
    ("D16", "descend to 0.8 m, slide east along the abutment", []),
    ("D19", "round the embankment on the EAST side at 0.55 m, then into the culvert",
     [(7.7, 8.0), (7.7, 14.0)]),
    ("D18", "TAB precision: continue west through the culvert", []),
]


def ray_box(origin, d, lo, hi):
    t0, t1 = 0.0, 1e9
    for k in range(3):
        if abs(d[k]) < 1e-9:
            if origin[k] < lo[k] or origin[k] > hi[k]:
                return None
            continue
        a, b = (lo[k] - origin[k]) / d[k], (hi[k] - origin[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return None
    return t0


def _free(p, boxes):
    for lo, hi in boxes:
        if np.linalg.norm(np.maximum(lo - p, 0) + np.maximum(p - hi, 0)) < CLEAR:
            return False
    return True


def _sees(p, target, boxes):
    d = target - p
    dist = np.linalg.norm(d)
    d = d / dist
    return all(not (t is not None and t < dist - 0.03) for t in (ray_box(p, d, lo, hi) for lo, hi in boxes))


# --- active world (simworld) ---
def world_settings():
    import json
    act = Path.home() / "drone_ws/sim/active_world"
    name = act.read_text().strip() if act.exists() else "bridge_full"
    f = Path.home() / "drone_ws/sim/worlds" / name / "world.json"
    return name, (json.loads(f.read_text()) if f.exists() else {})


def ground_stops(decals, boxes):
    """Patches on the ground (roads): stop ABOVE and BEHIND each patch, facing along the road,
    so the tilted camera (camera_pitch_deg) looks straight at the patch centre from 'standoff' m."""
    _, w = world_settings()
    P = math.radians(w.get("camera_pitch_deg", 55.0))
    standoff = w.get("standoff", 1.4)
    stops = []
    for i, dc in enumerate(sorted(decals, key=lambda d: d.get("order", 0)), 1):
        c = np.array(dc["centre"], float)
        yaw = float(dc.get("approach_yaw", 90.0))
        f = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])

        def view_at(d, c=c, f=f):
            return c - f * d * math.cos(P) + np.array([0.0, 0.0, d * math.sin(P)])

        s = standoff
        while s > 0.6 and not _free(view_at(s), boxes):
            s -= 0.05
        p = view_at(s)
        stops.append({"kind": "stop", "n": i, "id": dc["id"], "zone": dc["zone"],
                      "tip": f"fly {dc.get('lane', 'along the road')}, {p[2]:.2f} m high, "
                             f"{s * math.cos(P):.2f} m before the patch",
                      "pos": p, "standoff": round(s, 2), "yaw": yaw, "decal": dc, "view_at": view_at})
    return stops


# --- big worlds (patch_world2) ---
def ordered_stops(decals, boxes, clear=None):   # --- wind margin (patch_margin) ---
    """Any world: one stop per wall patch, in the patch's 'order', straight in front of it,
    as far back as fits (1.0 .. 0.45 m) with 'clear' m to every wall and a clear view."""
    if clear is None:                          # 2 cm more than the planner's wind margin
        _, w = world_settings()
        clear = w.get("clear", {"calm": 0.14, "light": 0.20, "moderate": 0.24, "gusty": 0.26, "strong": 0.28}.get(w.get("wind", "calm"), 0.14)) + 0.02
    def free(p):
        return all(np.linalg.norm(np.maximum(lo - p, 0) + np.maximum(p - hi, 0)) >= clear for lo, hi in boxes)
    stops = []
    for i, dc in enumerate(sorted(decals, key=lambda d: d["order"]), 1):
        c, n = np.array(dc["centre"], float), np.array(dc["normal"], float)
        for v in dc.get("via", []):
            stops.append({"kind": "via", "n": i, "id": dc["id"], "zone": dc["zone"],
                          "tip": dc.get("tip", ""), "pos": np.array(v, float), "yaw": None, "decal": dc})
        h = np.array([-n[1], n[0], 0.0])
        corners = [c + 0.4 * (sx * dc["width"] * h + sz * dc["height"] * np.array([0, 0, 1.0]))
                   for sx in (-1, 1) for sz in (-1, 1)]
        for s in STANDOFFS:
            p = c + s * n
            if free(p) and all(_sees(p, q, boxes) for q in [c, *corners]):
                break
        else:
            print(f"!! {dc['id']}: no free stop in front of it")
            s, p = STANDOFFS[-1], c + STANDOFFS[-1] * n
        stops.append({"kind": "stop", "n": i, "id": dc["id"], "zone": dc["zone"],
                      "tip": dc.get("tip", dc["zone"]), "pos": p, "standoff": round(s, 2),
                      "yaw": math.degrees(math.atan2(-n[1] + 0.0, -n[0] + 0.0)), "decal": dc})
    return stops


def compute_stops(decals, boxes):
    """decals/boxes as returned by pilot.load_world(). Returns the route as a list of dicts."""
    if all(abs(dc["normal"][2]) > 0.9 for dc in decals):
        return ground_stops(decals, boxes)
    if all("order" in dc for dc in decals):
        return ordered_stops(decals, boxes)
    by_id = {dc["id"]: dc for dc in decals}
    stops = []
    for i, (did, tip, vias) in enumerate(ROUTE, 1):
        dc = by_id[did]
        for vx, vy in vias:             # waypoints: same height as the stop they lead to
            stops.append({"kind": "via", "n": i, "id": did, "zone": dc["zone"], "tip": tip,
                          "pos": np.array([vx, vy, dc["centre"][2]]), "yaw": None, "decal": dc})
        c, n = np.array(dc["centre"], float), np.array(dc["normal"], float)
        h = np.array([-n[1], n[0], 0.0])
        corners = [c + 0.4 * (sx * dc["width"] * h + sz * dc["height"] * np.array([0, 0, 1.0]))
                   for sx in (-1, 1) for sz in (-1, 1)]
        for s in STANDOFFS:
            p = c + s * n
            if _free(p, boxes) and all(_sees(p, q, boxes) for q in [c, *corners]):
                break
        else:
            s, p = STANDOFFS[-1], c + STANDOFFS[-1] * n
        stops.append({"kind": "stop", "n": i, "id": did, "zone": dc["zone"], "tip": tip, "pos": p,
                      "standoff": round(s, 2),
                      "yaw": math.degrees(math.atan2(-n[1] + 0.0, -n[0] + 0.0)), "decal": dc})
    return stops


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import pilot

    decals, parts, boxes = pilot.load_world()
    route = compute_stops(decals, boxes)
    stops = [st for st in route if st["kind"] == "stop"]
    print(f"{'#':>2}  {'id':4} {'x':>5} {'y':>6} {'z':>5} {'face':>5} {'dist':>5}  zone / how to get there")
    for st in stops:
        x, y, z = st["pos"]
        print(f"{st['n']:2d}  {st['id']:4} {x:5.2f} {y:6.2f} {z:5.2f} {st['yaw']:5.0f} {st['standoff']:5.2f}"
              f"  {st['zone']}\n{'':38}-> {st['tip']}")
    length = sum(np.linalg.norm(b["pos"] - a["pos"]) for a, b in zip(route, route[1:]))
    print(f"\nstraight-line length {length:.0f} m (real path longer: around columns, under girders)")

    colours = {"concrete": "0.75", "weathered": "0.55", "grass": "#b5c99a", "steel": "#8a5a3a",
               "asphalt": "0.35"}
    fig, ax = plt.subplots(figsize=(9, 12))
    for mat in ("grass", "asphalt", "weathered", "concrete", "steel"):
        for p in parts:
            if p["mat"] == mat and p["name"] not in ("ground", "deck_slab") and not p["name"].startswith("road"):
                (x0, y0, _), (x1, y1, _) = p["lo"], p["hi"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=colours[mat], ec="k", lw=0.3))
    xs = [0.0] + [st["pos"][0] for st in route]
    ys = [0.0] + [st["pos"][1] for st in route]
    ax.plot(xs, ys, "--", color="royalblue", lw=1.2)
    for st in route:
        if st["kind"] == "via":
            ax.plot(st["pos"][0], st["pos"][1], "D", color="gold", mec="k", ms=6)
    ax.plot(0, 0, "s", color="k", ms=8)
    ax.annotate("START", (0, 0), xytext=(-40, -4), textcoords="offset points", fontsize=9, weight="bold")
    for st in stops:
        x, y, z = st["pos"]
        a = math.radians(st["yaw"])
        ax.annotate("", (x + 0.45 * math.cos(a), y + 0.45 * math.sin(a)), (x, y),
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
        ax.plot(x, y, "o", color="gold", mec="k", ms=13)
        ax.text(x, y, str(st["n"]), ha="center", va="center", fontsize=7.5, weight="bold")
        ax.annotate(f"{st['id']}  z {z:.2f}", (x, y), xytext=(8, -12), textcoords="offset points", fontsize=7)
    ax.set_xlim(*pilot.MAP_X)
    ax.set_ylim(*pilot.MAP_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Demo route - 20 stops (red arrow = camera direction, z = height, \u25c6 = waypoint)", fontsize=11)
    plt.tight_layout()
    out = pilot.WORLD_DIR / "route.png"
    plt.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
