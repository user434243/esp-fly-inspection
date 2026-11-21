#!/usr/bin/env python3
"""
make_city_world.py - a big city inspection site (world 'city_full', 36 m x 37 m, up to 7.4 m high).

Four zones, from wide open to very tight:
  BUILDINGS (south)  3-storey block with a LOW ARCADE (2.6 m ceiling, 2 rows of columns),
                     balconies, windows, and a 1.0 m wide, 9 m deep ALLEY to the 2-storey block
                     next door (a steel beam crosses the alley at 4 m)
  BRIDGE (middle)    concrete BOX-GIRDER overpass on 2 tall piers, 4.4 m clearance.
                     INSIDE the box: two cells 1.65 m wide x 1.3 m high, entered through a
                     0.8 m HATCH in the bottom slab; 0.7 m x 0.9 m MANHOLES in the diaphragms
                     and a cross-over hole between the cells. Lit by small inspection lamps.
  TUNNEL (north)     L-shaped service tunnel 0.8 m wide x 0.9 m high under an earth berm, dead end
  PLAZA / ROAD       paving, a road under the bridge, lamp posts, parked cars, trees (scenery)
30 defect patches (6 per class: crack, spallation, efflorescence, exposed bars, corrosion) from the
bridge5 TEST photos, on piers, girders, inside the box cells, facades, columns, alley and tunnel.
Camera straight ahead (like the bridge), bridge model E6.

Output  ~/drone_ws/sim/worlds/city_full/   city_full.sdf  parts.json  decals.json  world.json
        decals_overview.jpg  site_map.png    + link in CrazySim's worlds folder
Then:   simworld city_full ; simreset ; simfull ; autopilot
Run:    ~/drone_ws/venv/bin/python ~/drone_ws/sim/worlds/make_city_world.py
"""
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

import make_bridge_full as mbf             # textures, meshes, SDF helpers (same folder)
import make_bridge_world as mbw            # bridge5 TEST photo selection

WORLD_NAME = "city_full"
OUT = mbf.SIM_WORLDS / WORLD_NAME
CLASSES = mbw.CLASSES
SEED = 21
WORLD_SETTINGS = {"name": WORLD_NAME, "kind": "city", "camera_pitch_deg": 0,
                  "map_x": [-3, 33], "map_y": [-14, 23],
                  "grid_lo": [-3.0, -14.0, 0.0], "grid_hi": [33.0, 23.0, 8.6]}

EXTRA_MATERIALS = {   # key -> (texture folder, metres per tile, colour if no texture)
    "glass":  ("glass_dark", 1.0, (0.10, 0.13, 0.17)),
    "paint":  ("road_paint_white", 1.0, (0.86, 0.86, 0.82)),
    "paint_y": ("road_paint_yellow", 1.0, (0.85, 0.68, 0.15)),
    "pole":   ("pole_paint", 1.0, (0.40, 0.42, 0.44)),
    "paving": ("concrete_bridge", 1.5, (0.62, 0.61, 0.58)),
}


def box(name, x0, x1, y0, y1, z0, z1, mat, collide=True, flat=False):
    p = mbf.box(name, x0, x1, y0, y1, z0, z1, mat, collide)
    if flat:
        p["flat"] = True
    return p


# ---------------- 1. geometry ----------------
def buildings():
    p = []
    # B1: 3 storeys, ground floor = open arcade on the north side (y -7 .. -4)
    p.append(box("b1_core", 6.0, 17.5, -13.0, -7.0, 0.0, 2.6, "weathered"))
    p.append(box("b1_upper", 6.0, 17.5, -13.0, -4.0, 2.6, 9.0, "concrete"))
    for x in (7.0, 9.4, 11.8, 14.2, 16.6):                         # 2 rows of arcade columns
        p.append(box(f"b1_col_front_{x}", x - 0.2, x + 0.2, -4.6, -4.2, 0.0, 2.6, "weathered"))
        p.append(box(f"b1_col_mid_{x}", x - 0.2, x + 0.2, -5.9, -5.5, 0.0, 2.6, "weathered"))
    for i, (x0, x1) in enumerate(((7.5, 10.5), (12.5, 15.5))):     # balconies (2nd floor)
        p.append(box(f"b1_balcony_{i}", x0, x1, -4.0, -3.0, 5.65, 5.8, "concrete"))
        p.append(box(f"b1_balc_wall_{i}", x0, x1, -3.1, -3.0, 5.8, 6.7, "concrete"))
    p.append(box("b1_parapet_n", 6.0, 17.5, -4.15, -4.0, 9.0, 9.9, "concrete"))
    p.append(box("b1_parapet_s", 6.0, 17.5, -13.0, -12.85, 9.0, 9.9, "concrete"))
    p.append(box("b1_roof_unit", 9.0, 11.0, -11.0, -9.0, 9.0, 10.2, "steel"))
    p.append(box("b1_stair_tower", 14.0, 16.0, -12.0, -10.0, 9.0, 11.0, "weathered"))
    # B2: 2 storeys, west wall = the other side of the 1.0 m alley
    p.append(box("b2", 18.5, 30.0, -13.0, -4.0, 0.0, 6.4, "weathered"))
    p.append(box("b2_parapet", 18.5, 30.0, -4.15, -4.0, 6.4, 7.1, "weathered"))
    p.append(box("alley_beam", 17.5, 18.5, -8.15, -7.85, 4.0, 4.3, "steel"))  # steel beam across the alley
    # windows (dark glass, 2 cm proud of the wall)
    for z0, z1 in ((3.4, 4.9), (6.6, 8.1)):
        for k in range(5):
            x = 7.2 + 2.4 * k
            p.append(box(f"b1_win_{z0}_{k}", x - 0.7, x + 0.7, -4.0, -3.98, z0, z1, "glass", collide=False))
    for x0, x1 in ((6.8, 8.6), (12.6, 14.0), (15.0, 16.8)):        # shop windows in the arcade
        p.append(box(f"b1_shop_{x0}", x0, x1, -7.0, -6.98, 0.3, 2.2, "glass", collide=False))
    for z0, z1 in ((0.8, 2.3), (4.0, 5.5)):
        for k in range(4):
            x = 20.2 + 2.4 * k
            p.append(box(f"b2_win_{z0}_{k}", x - 0.7, x + 0.7, -4.0, -3.98, z0, z1, "glass", collide=False))
    for k in range(3):                                             # B1 west facade windows
        y = -11.5 + 3.0 * k
        for z0, z1 in ((3.4, 4.9), (6.6, 8.1)):
            p.append(box(f"b1_wwin_{k}_{z0}", 5.98, 6.0, y - 0.7, y + 0.7, z0, z1, "glass", collide=False))
    return p


def bridge():
    p = []
    zb0, zb1, zw1, zt1 = 4.6, 4.85, 6.15, 6.4        # bottom slab, webs top, top slab top
    p.append(box("abut_w", 5.0, 6.0, 7.5, 13.5, 0.0, zt1, "weathered"))
    p.append(box("abut_e", 30.0, 31.0, 7.5, 13.5, 0.0, zt1, "weathered"))
    # bottom slab with the access HATCH (x 8.6..9.4, y 9.0..9.8)
    p.append(box("bslab_w", 6.0, 8.6, 8.5, 12.5, zb0, zb1, "concrete"))
    p.append(box("bslab_e", 9.4, 30.0, 8.5, 12.5, zb0, zb1, "concrete"))
    p.append(box("bslab_hs", 8.6, 9.4, 8.5, 9.0, zb0, zb1, "concrete"))
    p.append(box("bslab_hn", 8.6, 9.4, 9.8, 12.5, zb0, zb1, "concrete"))
    p.append(box("web_s", 6.0, 30.0, 8.5, 8.75, zb1, zw1, "concrete"))
    p.append(box("web_n", 6.0, 30.0, 12.25, 12.5, zb1, zw1, "concrete"))
    # middle web with a cross-over hole (x 18.5..19.2, z 5.1..5.95)
    p.append(box("web_m_w", 6.0, 18.5, 10.4, 10.6, zb1, zw1, "concrete"))
    p.append(box("web_m_e", 19.2, 30.0, 10.4, 10.6, zb1, zw1, "concrete"))
    p.append(box("web_m_lo", 18.5, 19.2, 10.4, 10.6, zb1, 5.1, "concrete"))
    p.append(box("web_m_hi", 18.5, 19.2, 10.4, 10.6, 5.95, zw1, "concrete"))
    p.append(box("top_slab", 6.0, 30.0, 7.5, 13.5, zw1, zt1, "concrete"))
    p.append(box("road_deck", 6.0, 30.0, 7.7, 13.3, zt1, zt1 + 0.02, "asphalt", collide=False))
    p.append(box("deck_parapet_s", 5.0, 31.0, 7.5, 7.7, zt1, zt1 + 0.9, "concrete"))
    p.append(box("deck_parapet_n", 5.0, 31.0, 13.3, 13.5, zt1, zt1 + 0.9, "concrete"))
    # end diaphragms (closed) + diaphragms over the piers with MANHOLES (0.7 x 0.9 m)
    p.append(box("diaph_end_w", 6.0, 6.3, 8.75, 12.25, zb1, zw1, "concrete"))
    p.append(box("diaph_end_e", 29.7, 30.0, 8.75, 12.25, zb1, zw1, "concrete"))
    for x in (14.0, 23.0):
        x0, x1 = x - 0.15, x + 0.15
        for cell, (ya, yb, h0, h1) in enumerate(((8.75, 10.4, 9.2, 9.9), (10.6, 12.25, 11.1, 11.8))):
            p.append(box(f"diaph_{x}_{cell}_a", x0, x1, ya, h0, zb1, zw1, "concrete"))
            p.append(box(f"diaph_{x}_{cell}_b", x0, x1, h1, yb, zb1, zw1, "concrete"))
            p.append(box(f"diaph_{x}_{cell}_lo", x0, x1, h0, h1, zb1, 5.1, "concrete"))
            p.append(box(f"diaph_{x}_{cell}_hi", x0, x1, h0, h1, 6.0, zw1, "concrete"))
        # tall wall pier + steel bearings
        p.append(box(f"pier_{x}", x - 0.6, x + 0.6, 9.0, 12.0, 0.0, 4.4, "weathered"))
        for yb in (9.4, 11.6):
            p.append(box(f"bearing_{x}_{yb}", x - 0.2, x + 0.2, yb - 0.2, yb + 0.2, 4.4, zb0, "steel"))
    return p


def tunnel():
    p = []
    z_f, z_r, z_t, z_b = 0.15, 1.05, 1.25, 1.6       # floor top, ceiling, roof top, berm top
    # segment 1 along x (inside x 4.0..16.0, y 17.6..18.4), segment 2 north (x 15.2..16.0, y 18.4..21.6)
    p.append(box("tun_floor_1", 3.8, 16.2, 17.4, 18.6, 0.0, z_f, "weathered"))
    p.append(box("tun_floor_2", 15.0, 16.2, 18.6, 21.8, 0.0, z_f, "weathered"))
    p.append(box("tun_wall_s", 3.8, 16.2, 17.4, 17.6, z_f, z_r, "weathered"))
    p.append(box("tun_wall_n", 3.8, 15.2, 18.4, 18.6, z_f, z_r, "weathered"))
    p.append(box("tun_wall_e", 16.0, 16.2, 17.6, 21.8, z_f, z_r, "weathered"))
    p.append(box("tun_wall_w2", 15.0, 15.2, 18.6, 21.8, z_f, z_r, "weathered"))
    p.append(box("tun_wall_end", 15.2, 16.0, 21.6, 21.8, z_f, z_r, "weathered"))
    p.append(box("tun_roof_1", 3.8, 16.2, 17.4, 18.6, z_r, z_t, "weathered"))
    p.append(box("tun_roof_2", 15.0, 16.2, 18.6, 21.8, z_r, z_t, "weathered"))
    p.append(box("tun_headwall_s", 3.6, 3.8, 16.4, 17.6, 0.0, z_b, "weathered"))      # entrance face
    p.append(box("tun_headwall_n", 3.6, 3.8, 18.4, 19.6, 0.0, z_b, "weathered"))
    p.append(box("tun_headwall_top", 3.6, 3.8, 17.6, 18.4, z_r, z_b, "weathered"))
    # earth berm around + on top (grass)
    p.append(box("berm_s", 3.8, 17.0, 16.4, 17.4, 0.0, z_t, "grass"))
    p.append(box("berm_n", 3.8, 15.0, 18.6, 22.8, 0.0, z_t, "grass"))
    p.append(box("berm_e", 16.2, 17.0, 17.4, 22.8, 0.0, z_t, "grass"))
    p.append(box("berm_end", 15.0, 16.2, 21.8, 22.8, 0.0, z_t, "grass"))
    p.append(box("berm_top", 3.8, 17.0, 16.4, 22.8, z_t, z_b, "grass"))
    return p


def ground():
    p = [box("ground", -40, 70, -50, 60, -0.05, 0.0, "grass", collide=False)]
    p.append(box("plaza", -3.0, 33.0, -14.0, 6.5, 0.0, 0.012, "paving", False, True))
    p.append(box("street", -3.0, 33.0, 6.5, 15.5, 0.0, 0.015, "asphalt", False, True))
    p.append(box("kerb_s", -3.0, 33.0, 6.35, 6.5, 0.0, 0.12, "concrete", False, True))
    p.append(box("kerb_n", -3.0, 33.0, 15.5, 15.65, 0.0, 0.12, "concrete", False, True))
    x = -3.0
    while x < 33:
        p.append(box(f"street_dash_{x:.0f}", x, x + 2.0, 10.94, 11.06, 0.015, 0.02, "paint_y", False, True))
        x += 4.0
    for y in (6.8, 15.2):
        p.append(box(f"street_edge_{y}", -3.0, 33.0, y - 0.05, y + 0.05, 0.015, 0.02, "paint", False, True))
    # lamp posts (obstacles the planner knows)
    for i, (x, y) in enumerate([(3.0, -2.5), (20.0, -2.5), (3.0, 5.8), (27.0, 5.8), (32.0, 16.5)]):
        p.append(box(f"lamp_pole_{i}", x - 0.07, x + 0.07, y - 0.07, y + 0.07, 0.0, 4.0, "pole"))
        p.append(box(f"lamp_head_{i}", x - 0.25, x + 0.25, y - 0.12, y + 0.12, 3.85, 4.0, "pole"))
    # parked cars: the Fuel models are only looks, so keep-out boxes tell the planner they are there
    for i, (x, y, yaw) in enumerate(CARS):
        dx, dy = (2.4, 1.0) if abs(math.cos(yaw)) > 0.5 else (1.0, 2.4)
        p.append(box(f"car_keepout_{i}", x - dx, x + dx, y - dy, y + dy, 0.0, 1.9, "none", collide=False))
    return p


CARS = [(24.0, 2.0, 0.0), (28.0, 2.0, 0.0), (-1.0, 9.0, 0.0)]


def build_parts():
    return ground() + buildings() + bridge() + tunnel()


# ---------------- 2. defect patches (route order) ----------------
#  id     centre (x, y, z)          normal      width  zone / how to get there
DECALS = [
    ("F05", (9.4, -4.19, 1.3), (0, 1, 0), 0.36, "arcade - front column", "fly south to the low arcade"),
    ("F07", (10.6, -6.99, 1.4), (0, 1, 0), 0.6, "arcade - back wall", "into the arcade between the columns (2.6 m ceiling)"),
    ("F06", (14.0 - 0.01, -5.7, 1.0), (-1, 0, 0), 0.36, "arcade - middle column", "east inside the arcade, face the column"),
    ("F02", (15.6, -3.99, 4.2), (0, 1, 0), 0.6, "B1 facade - 1st floor", "out of the arcade, climb to 4.2 m"),
    ("F04", (13.2, -3.99, 4.3), (0, 1, 0), 0.6, "B1 facade - under balcony", "slide west under the balcony"),
    ("F03", (9.0, -2.99, 6.25), (0, 1, 0), 0.6, "B1 balcony front", "climb over to the west balcony, 6.25 m"),
    ("F01", (10.8, -3.99, 7.4), (0, 1, 0), 0.6, "B1 facade - 2nd floor", "climb to 7.4 m between the windows"),
    ("F08", (17.51, -9.0, 1.5), (1, 0, 0), 0.6, "alley - B1 wall low", "down into the 1.0 m alley"),
    ("F10", (17.51, -11.0, 6.0), (1, 0, 0), 0.6, "alley - B1 wall high", "deep in the alley, climb past the beam"),
    ("F09", (18.49, -10.0, 3.5), (-1, 0, 0), 0.6, "alley - B2 wall", "turn round, face the other wall"),
    ("F11", (26.2, -3.99, 4.8), (0, 1, 0), 0.6, "B2 facade", "out of the alley, east along B2"),
    ("B04", (23.61, 10.0, 3.0), (1, 0, 0), 0.6, "bridge pier 2 - east face", "north to the bridge, under the deck"),
    ("B03", (23.0, 12.01, 1.5), (0, 1, 0), 0.6, "bridge pier 2 - north face", "round the pier, low"),
    ("B07", (29.99, 10.5, 2.0), (-1, 0, 0), 0.8, "east abutment", "east under the bridge end"),
    ("B06", (26.0, 12.51, 5.5), (0, 1, 0), 0.6, "box girder - north web", "climb beside the girder, 5.5 m"),
    ("B02", (14.0, 8.99, 2.5), (0, -1, 0), 0.6, "bridge pier 1 - south face", "west along the south side"),
    ("B01", (13.39, 10.5, 1.2), (-1, 0, 0), 0.6, "bridge pier 1 - west face", "round the pier, low"),
    ("B05", (11.0, 8.49, 5.5), (0, -1, 0), 0.6, "box girder - south web, under the cantilever", "up under the deck overhang"),
    ("C01", (11.0, 8.76, 5.5), (0, 1, 0), 0.48, "INSIDE cell 1 - south web", "up through the 0.8 m HATCH into the box"),
    ("C02", (13.84, 10.15, 5.5), (-1, 0, 0), 0.4, "INSIDE cell 1 - diaphragm by the manhole", "east in cell 1 to the pier diaphragm"),
    ("C03", (17.0, 10.39, 5.5), (0, -1, 0), 0.48, "INSIDE cell 1 - middle web", "through the MANHOLE"),
    ("C04", (21.0, 10.61, 5.5), (0, 1, 0), 0.48, "INSIDE cell 2 - middle web", "through the cross-over hole into cell 2"),
    ("C05", (26.0, 12.24, 5.5), (0, -1, 0), 0.48, "INSIDE cell 2 - north web", "through the 2nd manhole"),
    ("C06", (29.69, 11.4, 5.5), (-1, 0, 0), 0.48, "INSIDE cell 2 - end diaphragm", "to the dead end, then all the way back to the hatch"),
    ("T01", (6.0, 17.61, 0.6), (0, 1, 0), 0.48, "TUNNEL - south wall", "north to the berm, into the 0.8 m tunnel"),
    ("T02", (9.0, 18.39, 0.6), (0, -1, 0), 0.48, "TUNNEL - north wall", "deeper, other wall"),
    ("T03", (12.5, 17.61, 0.6), (0, 1, 0), 0.48, "TUNNEL - south wall", "deeper"),
    ("T04", (15.99, 18.0, 0.6), (-1, 0, 0), 0.48, "TUNNEL - corner wall", "to the corner"),
    ("T05", (15.21, 19.8, 0.6), (1, 0, 0), 0.48, "TUNNEL - branch west wall", "turn north into the branch"),
    ("T06", (15.6, 21.59, 0.6), (0, -1, 0), 0.48, "TUNNEL - dead end", "to the dead end, then back out and home"),
]
LIGHTS = [(10.0, 9.6, 6.0), (17.0, 9.6, 6.0), (21.5, 11.4, 6.0), (27.5, 11.4, 6.0),   # box cells
          (6.5, 18.0, 0.95), (10.5, 18.0, 0.95), (14.5, 18.0, 0.95), (15.6, 20.2, 0.95),  # tunnel
          (11.8, -5.0, 2.45), (15.4, -6.3, 2.45)]                                          # arcade


# ---------------- 3. photos ----------------
def choose_photos(n_per_class):
    """n TEST photos per class (cleanest first, dacl10k and CODEBRIM mixed). If a class has fewer
    usable photos, the missing ones re-use a photo MIRRORED (said in the output)."""
    found = mbw.candidates()
    rng = random.Random(SEED)
    used, chosen = set(), []
    for cid in sorted(range(len(CLASSES)), key=lambda c: len(found[c]["dacl"]) + len(found[c]["cb"])):
        pool = [c for c in found[cid]["dacl"] + found[cid]["cb"] if c[0] not in used]
        rng.shuffle(pool)
        pool.sort(key=lambda c: -round(c[2], 1))
        picks = [(p, a, False) for p, a, _ in pool[:n_per_class]]
        k = 0
        while len(picks) < n_per_class and pool:
            p, a, _ = pool[k % len(pool)]
            picks.append((p, a, True))
            k += 1
        if not picks:
            sys.exit(f"!! no test photos for {CLASSES[cid]}")
        for p, a, flip in picks:
            used.add(p)
            chosen.append((cid, p, a, flip))
        n_flip = sum(f for _, _, f in picks)
        print(f"  {CLASSES[cid]:16s} {len(pool):4d} usable photos -> {n_per_class} patches"
              + (f" ({n_flip} mirrored re-use)" if n_flip else ""))
    rng.shuffle(chosen)
    return chosen


# ---------------- 4. build ----------------
def point_light(world, name, x, y, z):
    sub = mbf.sub
    li = sub(world, "light", name=name, type="point")
    sub(li, "pose", f"{x} {y} {z} 0 0 0")
    sub(li, "diffuse", "1.0 0.93 0.8 1")
    sub(li, "specular", "0.2 0.2 0.2 1")
    att = sub(li, "attenuation")
    for tag, val in (("range", "5"), ("constant", "0.25"), ("linear", "0.25"), ("quadratic", "0.08")):
        sub(att, tag, val)
    sub(li, "cast_shadows", "false")


def check_layout(parts):
    """Every patch sits on a real surface (1 cm in front of a solid box, its whole area)."""
    solid = [(np.array(p["lo"]), np.array(p["hi"])) for p in parts if p["collide"]]
    bad = []
    for did, c, n, w, *_ in DECALS:
        c, n = np.array(c), np.array(n, float)
        u = np.array([-n[1], n[0], 0.0])
        for a in (-0.45, 0.45):
            for b in (-0.45, 0.45):
                q = c + a * w * u + b * 0.75 * w * np.array([0, 0, 1.0]) - 0.03 * n
                if not any((lo <= q).all() and (q <= hi).all() for lo, hi in solid):
                    bad.append(did)
    return sorted(set(bad))


def draw_site_map(parts, truth, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colours = {"grass": "#9bbf7a", "paving": "#cfcac0", "asphalt": "0.35", "weathered": "0.55",
               "concrete": "0.72", "steel": "#8a5a3a", "glass": "#2b3b4f", "pole": "0.4", "paint": "white",
               "paint_y": "#e0b030", "none": None}
    fig, ax = plt.subplots(figsize=(10, 10.5))
    for mat in ("grass", "paving", "asphalt", "paint", "paint_y", "weathered", "concrete", "steel", "pole"):
        for p in parts:
            if p["mat"] == mat and p["name"] != "ground" and not p["name"].startswith(("top_slab", "road_deck", "deck_", "berm_top", "tun_roof", "b1_upper", "b1_parapet", "b1_roof", "b1_stair")):
                (x0, y0, _), (x1, y1, _) = p["lo"], p["hi"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=colours[mat], ec="k", lw=0.2))
    for x0, x1, y0, y1, label in ((6, 17.5, -13, -4, "B1 (3 floors, arcade)"), (18.5, 30, -13, -4, "B2"),
                                  (5, 31, 7.5, 13.5, "box-girder bridge (deck 6.4 m)"),
                                  (3.6, 17, 16.4, 22.8, "tunnel under berm")):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="navy", lw=1.2, ls="--"))
        ax.text(x0 + 0.2, y1 - 0.6, label, fontsize=8, color="navy")
    col = dict(zip(CLASSES, ("tab:red", "tab:orange", "tab:cyan", "tab:purple", "tab:brown")))
    for t in truth:
        x, y, z = t["centre"]
        ax.plot(x, y, "o", color=col[t["class"]], ms=7, mec="k")
        ax.annotate(f"{t['id']} {z:.1f}m", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    for name, c in col.items():
        ax.plot([], [], "o", color=c, mec="k", label=name)
    ax.plot(0, 0, marker="^", color="blue", ms=12, label="drone start")
    ax.set_xlim(*WORLD_SETTINGS["map_x"])
    ax.set_ylim(*WORLD_SETTINGS["map_y"])
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("City site (top view) - 30 defect patches, label = id + height", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    if not mbw.DATASET.exists():
        sys.exit(f"Dataset not found: {mbw.DATASET}  (is the HDD plugged in?)")
    mbf.OUT, mbf.WORLD_NAME = OUT, WORLD_NAME           # the bridge helpers now write this world
    mbf.MATERIALS.update(EXTRA_MATERIALS)
    if (mbf.TEXTURES / "asphalt_clean" / "albedo.jpg").exists():   # made by make_road_world.py
        mbf.MATERIALS["asphalt"] = ("asphalt_clean", 4.0, (0.20, 0.20, 0.21))
    for d in ("meshes", "decals"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    parts = build_parts()
    bad = check_layout(parts)
    if bad:
        sys.exit(f"!! patches not on a surface: {bad}")

    print("Choosing unseen bridge5 TEST photos for the 30 defect patches ...")
    chosen = choose_photos(len(DECALS) // len(CLASSES))

    tree, world = mbf.load_template()
    sub = mbf.sub
    model = sub(world, "model", name="city")
    sub(model, "static", "true")
    link = sub(model, "link", name="link")
    for mat, (_, tile, _) in mbf.MATERIALS.items():
        group = [p for p in parts if p["mat"] == mat]
        if not group:
            continue
        obj = OUT / "meshes" / f"{mat}.obj"
        mbf.write_obj(group, obj, tile)
        vis = sub(link, "visual", name=f"vis_{mat}")
        sub(sub(sub(vis, "geometry"), "mesh"), "uri", "file://" + str(obj))
        m = mbf.pbr_material(vis, mat)
        if mat == "glass":                                   # shiny windows
            m.find("specular").text = "0.7 0.7 0.75 1"
    for p in parts:
        if not p["collide"]:
            continue
        lo, hi = np.array(p["lo"]), np.array(p["hi"])
        col = sub(link, "collision", name=f"col_{p['name']}")
        sub(col, "pose", "%.3f %.3f %.3f 0 0 0" % tuple((lo + hi) / 2))
        sub(sub(sub(col, "geometry"), "box"), "size", "%.3f %.3f %.3f" % tuple(hi - lo))
    for i, (x, y, z) in enumerate(LIGHTS):
        point_light(world, f"inspection_lamp_{i}", x, y, z)

    truth, thumbs = [], []
    for k, ((did, centre, normal, width, zone, tip), (cid, src, also, flip)) in enumerate(zip(DECALS, chosen), 1):
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        x0, y0, x1, y1 = mbw.crop_43(w, h)
        tex = cv2.resize(img[y0:y1, x0:x1], (640, 480), interpolation=cv2.INTER_CUBIC)
        if flip:
            tex = cv2.flip(tex, 1)
        tex_path = OUT / "decals" / f"{did}.jpg"
        cv2.imwrite(str(tex_path), tex, [cv2.IMWRITE_JPEG_QUALITY, 95])
        height = width * 0.75
        m = sub(world, "model", name=f"defect_{did}")
        sub(m, "static", "true")
        sub(m, "pose", mbf.decal_pose(centre, normal))
        vis = sub(sub(m, "link", name="link"), "visual", name="visual")
        plane = sub(sub(vis, "geometry"), "plane")
        sub(plane, "normal", "0 0 1")
        sub(plane, "size", f"{width} {height:.3f}")
        sub(vis, "cast_shadows", "false")
        mbw.material(vis, (1, 1, 1), pbr_albedo=str(tex_path))
        truth.append({"id": did, "class": CLASSES[cid], "also_present": also, "zone": zone, "tip": tip,
                      "order": k, "centre": list(centre), "normal": list(normal), "width": width,
                      "height": round(height, 3), "dataset": "CODEBRIM" if src.name.startswith("cb_") else "dacl10k",
                      "source_image": src.name, "mirrored": flip})
        t = cv2.resize(tex, (256, 192))
        cv2.rectangle(t, (0, 0), (256, 24), (0, 0, 0), -1)
        cv2.putText(t, f"{did} {CLASSES[cid]}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        thumbs.append(t)

    # scenery from Gazebo Fuel - trees outside the flying area, cars on their keep-out boxes
    for i, (x, y) in enumerate([(-7, -10), (-8, 2), (-7, 14), (-6, 26), (37, -12), (38, 0), (37, 12),
                                (36, 26), (12, -18), (26, -19), (4, 28), (22, 28)], 1):
        mbf.fuel(world, f"tree_{i}", "Oak tree" if i % 2 else "Pine Tree", x, y)
    for i, ((x, y, yaw), model_name) in enumerate(zip(CARS, ("Hatchback", "SUV", "Pickup"))):
        mbf.fuel(world, f"car_{i}", model_name, x, y, 0.0, yaw)
    for i, x in enumerate((1.5, 2.2, 2.9)):
        mbf.fuel(world, f"cone_{i}", "Construction Cone", x, 7.4, 0.015)

    import xml.etree.ElementTree as ET
    ET.indent(tree, space="  ")
    sdf_path = OUT / f"{WORLD_NAME}.sdf"
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
    (OUT / "decals.json").write_text(json.dumps({"world": WORLD_NAME, "decals": truth}, indent=2))
    (OUT / "parts.json").write_text(json.dumps(parts))
    (OUT / "world.json").write_text(json.dumps(
        {**WORLD_SETTINGS, **({k: v for k, v in json.loads((OUT / "world.json").read_text()).items()
                               if k in ("model", "conf", "light")} if (OUT / "world.json").exists() else {})},
        indent=2))
    sheet = np.vstack([np.hstack(thumbs[k:k + 6]) for k in range(0, len(thumbs), 6)])
    cv2.imwrite(str(OUT / "decals_overview.jpg"), sheet)
    draw_site_map(parts, truth, OUT / "site_map.png")

    link_path = mbw.CRAZYSIM_WORLDS / f"{WORLD_NAME}.sdf"
    if mbw.CRAZYSIM_WORLDS.exists():
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(sdf_path)
        print(f"Linked into CrazySim: {link_path}")
    print(f"\nWorld: {sdf_path}\n{len(parts)} parts, {len(truth)} defect patches, {len(LIGHTS)} inspection lamps\n")
    print(f"{'id':5s}{'class':17s}{'z':>5s}  zone")
    for t in truth:
        print(f"{t['id']:5s}{t['class']:17s}{t['centre'][2]:5.1f}  {t['zone']}")
    print("\nNext:  simworld city_full ; simreset ; simfull     then  autopilot  (or pilot)")


if __name__ == "__main__":
    main()
