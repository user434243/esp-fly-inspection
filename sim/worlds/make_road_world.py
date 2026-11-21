#!/usr/bin/env python3
"""
make_road_world.py - a 2-lane asphalt road for road-damage inspection (world 'road_full').

  * Road 7 m wide (x 1..8) along y, kerbs, sidewalks, grass verges, lamp posts, trees, parked cars.
  * Markings: dashed centre line, edge lines, a stop line and a zebra crossing.
  * 20 road-damage patches ON the asphalt, 5 per class (longitudinal / transverse / alligator crack,
    pothole), from RDD2022 TEST photos the road model never saw, cropped tight around the damage. Top-down drone photos
    (China_Drone) first, because the drone looks down at the road like that camera did.
  * Distractors that are NOT damage: repair patches, manhole covers, the zebra crossing and stop
    line - a good model must ignore them (they show up as false alarms if it does not).
  * The drone starts at (0, 0) on the west sidewalk, facing +x (east, towards the road).
    Route: NORTH along the west lane (x ~ 2.75), then back SOUTH along the east lane (x ~ 6.25),
    stopping above + behind each patch with the camera tilted 55 deg down (world.json).

Uses the same building blocks as make_bridge_full.py (textures, mesh writer, world template).
Output  ~/drone_ws/sim/worlds/road_full/
  road_full.sdf  meshes/  decals/  decals.json  parts.json  world.json  decals_overview.jpg  site_map.png
and a link in CrazySim's worlds folder. Then:  simworld road_full ; simreset ; simfull

Run:  ~/drone_ws/venv/bin/python ~/drone_ws/sim/worlds/make_road_world.py
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import make_bridge_full as mbf             # textures, meshes, SDF helpers (same folder)
import make_bridge_world as mbw

HOME = Path.home()
WORLD_NAME = "road_full"
OUT = mbf.SIM_WORLDS / WORLD_NAME
RDD = Path("/media/hudaifa/Ventoy/datasets/rdd4_yolo")
NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
PER_CLASS = 5
SEED = 3
WORLD_SETTINGS = {"name": WORLD_NAME, "kind": "road", "model": "rdd4_y11s.pt", "conf": 0.2,
                  "camera_pitch_deg": 55, "standoff": 1.4, "closer": [1.0, 0.75],
                  "map_x": [-4, 10], "map_y": [-12, 18]}

# ---------------- road layout (m) ----------------
ROAD_X = (1.0, 8.0)
ROAD_Y = (-40.0, 40.0)            # visual length (the drone uses y -11 .. 17)
CENTRE_X = 4.5
LANE_NB, LANE_SB = 2.75, 6.25     # lane centres: north-bound (west lane), south-bound (east lane)
Z_ROAD, Z_MARK, Z_DECAL = 0.020, 0.024, 0.030
CROSSING_Y = (4.6, 7.0)           # zebra crossing (stripes along y)
STOP_LINE_Y = 4.0                 # north-bound stop line before the crossing

# extra materials (plain colour if no texture folder): key -> (texture folder, tile m, colour)
EXTRA_MATERIALS = {
    "paint":   ("road_paint_white", 1.0, (0.86, 0.86, 0.82)),
    "paint_y": ("road_paint_yellow", 1.0, (0.85, 0.68, 0.15)),
    "asphalt": ("asphalt_clean", 4.0, (0.20, 0.20, 0.21)),     # road surface WITHOUT the texture's own cracks
    "patch":   ("asphalt_patch", 4.0, (0.12, 0.12, 0.13)),     # darker fresh asphalt (repairs)
    "pole":    ("pole_paint", 1.0, (0.45, 0.47, 0.48)),
}

#      id     centre x, y      lane (NB = north-bound west lane, SB = south-bound east lane), zone
DECALS = [
    ("R01", (2.30, -9.0), "NB", "west lane - right wheel track"),
    ("R02", (3.25, -6.4), "NB", "west lane - centre"),
    ("R03", (2.55, -3.7), "NB", "west lane - left wheel track"),
    ("R04", (3.45, -1.1), "NB", "west lane - near centre line"),
    ("R05", (2.15, 1.5), "NB", "west lane - near kerb"),
    ("R06", (2.80, 6.3), "NB", "west lane - on the zebra crossing"),
    ("R07", (2.95, 9.5), "NB", "west lane - after the crossing"),
    ("R08", (2.30, 12.3), "NB", "west lane - right wheel track"),
    ("R09", (3.30, 15.3), "NB", "west lane - centre"),
    ("R10", (6.00, 15.6), "SB", "east lane - centre"),
    ("R11", (6.85, 12.9), "SB", "east lane - near kerb"),
    ("R12", (5.55, 10.3), "SB", "east lane - near centre line"),
    ("R13", (6.40, 7.9), "SB", "east lane - left wheel track"),
    ("R14", (6.25, 5.3), "SB", "east lane - on the zebra crossing"),
    ("R15", (6.70, 2.7), "SB", "east lane - right wheel track"),
    ("R16", (5.70, 0.1), "SB", "east lane - centre"),
    ("R17", (6.55, -2.5), "SB", "east lane - after the manhole"),
    ("R18", (5.65, -5.1), "SB", "east lane - near centre line"),
    ("R19", (6.95, -7.7), "SB", "east lane - near kerb"),
    ("R20", (6.10, -10.3), "SB", "east lane - centre"),
]
PATCH_SIZE = 1.2                          # patch size (m): a tight crop around the damage

# distractors (NOT damage): (kind, centre x, y, size x, size y)
DISTRACTORS = [
    ("manhole", 2.40, -7.7, 0.7, 0.7), ("manhole", 6.40, -1.2, 0.7, 0.7), ("manhole", 3.10, 13.8, 0.7, 0.7),
    ("repair", 2.60, -5.05, 2.0, 0.7), ("repair", 6.20, -8.95, 1.6, 0.6), ("repair", 2.35, 10.9, 1.2, 0.8),
    ("repair", 6.10, 1.4, 2.2, 0.7), ("repair", 6.30, 14.25, 1.4, 0.8), ("repair", 3.30, 3.0, 1.0, 0.9),
]


# ---------------- 1. geometry ----------------
def box(name, x0, x1, y0, y1, z0, z1, mat, collide=True, flat=False):
    p = mbf.box(name, x0, x1, y0, y1, z0, z1, mat, collide)
    if flat:
        p["flat"] = True                    # on the ground: not an obstacle for the planner
    return p


def build_parts():
    p = []
    y0, y1 = ROAD_Y
    p.append(box("ground", -35, 45, -40, 40, -0.05, 0.0, "grass", collide=False))
    p.append(box("carriageway", *ROAD_X, y0, y1, 0.0, Z_ROAD, "asphalt", collide=False, flat=True))
    p.append(box("kerb_w", 0.85, ROAD_X[0], y0, y1, 0.0, 0.15, "concrete"))
    p.append(box("kerb_e", ROAD_X[1], 8.15, y0, y1, 0.0, 0.15, "concrete"))
    p.append(box("sidewalk_w", -1.6, 0.85, y0, y1, 0.0, 0.012, "concrete", collide=False, flat=True))
    p.append(box("sidewalk_e", 8.15, 10.6, y0, y1, 0.0, 0.012, "concrete", collide=False, flat=True))
    # markings (top at Z_MARK)
    for x in (1.15, 7.75):
        p.append(box(f"edge_line_{x}", x, x + 0.10, y0, y1, Z_ROAD, Z_MARK, "paint", False, True))
    y, k = y0, 0
    while y < y1:                            # dashed centre line: 3 m paint, 3 m gap, not across the crossing
        if not (CROSSING_Y[0] - 1.0 < y + 3 and y < CROSSING_Y[1] + 0.5):
            p.append(box(f"centre_dash_{k}", CENTRE_X - 0.06, CENTRE_X + 0.06, y, y + 3.0, Z_ROAD, Z_MARK,
                         "paint_y", False, True))
        y, k = y + 6.0, k + 1
    p.append(box("stop_line", 1.25, CENTRE_X - 0.1, STOP_LINE_Y - 0.15, STOP_LINE_Y + 0.15, Z_ROAD, Z_MARK,
                 "paint", False, True))
    x = 1.45
    while x + 0.5 < 7.6:                     # zebra stripes, 0.5 m wide, 0.5 m apart
        p.append(box(f"zebra_{x:.2f}", x, x + 0.5, *CROSSING_Y, Z_ROAD, Z_MARK, "paint", False, True))
        x += 1.0
    for i, (kind, cx, cy, sx, sy) in enumerate(DISTRACTORS):
        p.append(box(f"{kind}_{i}", cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, Z_ROAD,
                     Z_MARK, "steel" if kind == "manhole" else "patch", False, True))
    # lamp posts: pole + arm over the road + lamp (real obstacles for the planner)
    for i, (x, y, s) in enumerate([(-0.5, -8.0, 1), (-0.5, 2.5, 1), (-0.5, 13.0, 1), (9.5, -3.0, -1),
                                   (9.5, 8.0, -1)]):
        p.append(box(f"lamp_pole_{i}", x - 0.07, x + 0.07, y - 0.07, y + 0.07, 0.0, 4.0, "pole"))
        ax0, ax1 = sorted((x, x + s * 1.6))
        p.append(box(f"lamp_arm_{i}", ax0, ax1, y - 0.04, y + 0.04, 3.9, 3.98, "pole"))
        lx = x + s * 1.6
        p.append(box(f"lamp_head_{i}", lx - 0.25, lx + 0.25, y - 0.12, y + 0.12, 3.8, 3.9, "pole"))
    return p


def check_layout():
    """Every patch inside its lane, clear of markings and distractors (overlaps confuse the score)."""
    rects = [(k, cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2) for k, cx, cy, sx, sy in DISTRACTORS]
    rects += [("stop line", 1.25, CENTRE_X - 0.1, STOP_LINE_Y - 0.15, STOP_LINE_Y + 0.15),
              ("centre line", CENTRE_X - 0.06, CENTRE_X + 0.06, -40, 40),
              ("edge line", 1.15, 1.25, -40, 40), ("edge line", 7.75, 7.85, -40, 40)]
    size = PATCH_SIZE / 2
    sq = [(d, x - size, x + size, y - size, y + size) for d, (x, y), _, _ in DECALS]
    problems = []
    for d, a0, a1, b0, b1 in sq:
        for k, c0, c1, e0, e1 in rects + [r for r in sq if r[0] != d]:
            gap = max(c0 - a1, a0 - c1, e0 - b1, b0 - e1)
            if gap < 0.2:
                problems.append(f"{d} too close to {k} ({gap:.2f} m)")
    return problems


# ---------------- 2. choose test photos ----------------
def read_labels(txt):
    out = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            q = line.split()
            if len(q) == 5:
                out.append((int(q[0]), *map(float, q[1:])))
    return out


MIN_SIDE_DRONE, MIN_SIDE_STREET = 200, 160   # smallest crop (source px) -> cracks keep a real-world width
HOOD = 0.88                                    # street photos: ignore the bottom 12% (car bonnet)
SKY = 0.42                                     # ... and the top 42% (sky, trees, buildings)
MAX_COLOUR = 0.15                              # skip crops with > 15% strongly coloured pixels (trees, bricks, cars)


def square_crop(box_, side, x_lim, y_lim):
    """Square crop (x0, y0, x1, y1) centred on the box (cx, cy, bw, bh in px), kept inside the limits."""
    cx, cy, _, _ = box_
    x0 = int(min(max(cx - side / 2, x_lim[0]), x_lim[1] - side))
    y0 = int(min(max(cy - side / 2, y_lim[0]), y_lim[1] - side))
    return x0, y0, x0 + side, y0 + side


def candidates(rdd):
    """One entry per (test photo, class in it): a square crop TIGHT around the class's biggest box, so
    the damage fills the patch (like a close drone view) and trees / kerbs / cars mostly fall outside.
    Street (dashcam) photos: only damage in the lower road area, never the car bonnet."""
    found = {c: [] for c in range(len(NAMES))}
    lists = sorted(rdd.glob("test_*.txt"))
    if not lists:
        sys.exit(f"!! no test_<country>.txt in {rdd} - run rdd2yolo.py first")
    for lst in lists:
        country = lst.stem[5:]
        drone = "drone" in country.lower()
        for line in lst.read_text().split():
            img_p = Path(line)
            labels = read_labels(rdd / "labels/test" / f"{img_p.stem}.txt")
            if not labels or len(labels) > 8 or not img_p.exists():
                continue
            try:
                with Image.open(img_p) as im:      # reads only the header: fast
                    w, h = im.size
            except OSError:
                continue
            y_max = h if drone else int(h * HOOD)
            y_min = 0 if drone else int(h * SKY)     # street photos: stay below the horizon (no sky, trees)
            for cid in {l[0] for l in labels}:
                big = max((l for l in labels if l[0] == cid), key=lambda l: l[3] * l[4])
                bx = (big[1] * w, big[2] * h, big[3] * w, big[4] * h)
                long_side, short_side = max(bx[2], bx[3]), min(bx[2], bx[3])
                if long_side < (120 if cid < 3 else 60) or short_side < (8 if cid < 3 else 40):
                    continue                 # too small to see anything
                if not drone and (bx[1] < 0.45 * h or bx[1] + bx[3] / 2 > y_max):
                    continue                 # far away near the horizon, or on the bonnet
                lo = MIN_SIDE_DRONE if drone else MIN_SIDE_STREET
                side = int(min(min(w, y_max - y_min), max(1.35 * long_side, lo)))
                x0, y0, x1, y1 = crop = square_crop(bx, side, (0, w), (y_min, y_max))
                area = Counter()
                for l in labels:            # how much of each class is inside the crop
                    lx0, lx1 = (l[1] - l[3] / 2) * w, (l[1] + l[3] / 2) * w
                    ly0, ly1 = (l[2] - l[4] / 2) * h, (l[2] + l[4] / 2) * h
                    area[l[0]] += max(0, min(lx1, x1) - max(lx0, x0)) * max(0, min(ly1, y1) - max(ly0, y0))
                tot = side * side
                if area[cid] / tot < 0.10:
                    continue                 # damage too small on the patch
                share = area[cid] / sum(area.values())
                also = sorted(NAMES[c] for c, a in area.items() if c != cid and a / tot > 0.02)
                found[cid].append(dict(path=img_p, crop=crop, also=also, share=share, drone=drone,
                                       country=country, px=long_side))
    return found


def colourful(c):
    """Share of strongly coloured pixels in the crop: road (asphalt / concrete) is grey."""
    img = cv2.imread(str(c["path"]))
    x0, y0, x1, y1 = c["crop"]
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return float((hsv[..., 1] > 90).mean())


def choose(found):
    """Per class: top-down drone photos first, then the cleanest (only that damage), then the most
    detail (biggest damage in pixels). Crops that are not mostly road are skipped. Never uses the
    model itself to pick - the test stays honest."""
    rng = random.Random(SEED)
    used, chosen = set(), []
    order = sorted(range(len(NAMES)), key=lambda c: len(found[c]))       # rarest class first
    for cid in order:
        pool = [c for c in found[cid] if c["path"] not in used]
        rng.shuffle(pool)
        pool.sort(key=lambda c: (not c["drone"], -round(c["share"], 1), -c["px"] // 50))
        picks, skipped = [], 0
        for c in pool:
            if len(picks) == PER_CLASS:
                break
            if colourful(c) > MAX_COLOUR:
                skipped += 1
                continue
            picks.append(c)
        if len(picks) < PER_CLASS:
            sys.exit(f"!! only {len(picks)} usable test photos for {NAMES[cid]}")
        for c in picks:
            used.add(c["path"])
            chosen.append((cid, c))
        n_drone = sum(c["drone"] for c in found[cid])
        print(f"  {NAMES[cid]:19s} candidates {len(found[cid]):5d} ({n_drone} top-down, {skipped} not road skipped)"
              f"  -> " + ", ".join(f"{c['path'].stem}" for c in picks))
    rng.shuffle(chosen)
    return chosen


# ---------------- 3. build ----------------
def make_road_textures():
    """asphalt_clean: the asphalt texture with its crack-like lines removed - otherwise the model
    rightly 'finds' cracks in the road texture that are not in the answer key (false alarms).
    Clearly dark lines are found (darker than a grey-scale closing) and painted over (inpaint),
    the normal asphalt grain stays.
    asphalt_patch: repairs = the clean asphalt, darker (fresh)."""
    src = mbf.TEXTURES / "asphalt"
    if not (src / "albedo.jpg").exists():
        return
    img = cv2.imread(str(src / "albedo.jpg"))
    k = max(9, img.shape[1] // 64) | 1                      # ~ 6 cm on the road (4 m tile)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dark = (closed.astype(np.int16) - img.astype(np.int16)).max(axis=2)       # how much darker than around
    thr = max(30.0, float(np.percentile(dark, 98)))                          # only the clearly dark pixels
    raw = (dark > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(raw, np.ones((3, 3), np.uint8)), 8)
    big = np.zeros(n, bool)
    big[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= 2 * k   # long = a line
    mask = cv2.dilate((big[lab]).astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=2)
    clean = cv2.inpaint(img, mask, max(3, k // 2), cv2.INPAINT_TELEA).astype(np.float32)
    print(f"road texture: removed crack-like lines from {mask.mean() / 255:.1%} of the asphalt")
    OUT.mkdir(parents=True, exist_ok=True)
    s = 900 / img.shape[1]
    cv2.imwrite(str(OUT / "asphalt_clean_check.jpg"),                     # before | after
                cv2.resize(np.hstack([img, clean.astype(np.uint8)]), None, fx=s, fy=s))
    for name, im in (("asphalt_clean", clean), ("asphalt_patch", clean * 0.6)):
        dst = mbf.TEXTURES / name
        dst.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst / "albedo.jpg"), np.clip(im, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 93])
        if (src / "roughness.jpg").exists():                 # no normal map: it holds the cracks' relief
            (dst / "roughness.jpg").write_bytes((src / "roughness.jpg").read_bytes())
        if (dst / "normal.jpg").exists():
            (dst / "normal.jpg").unlink()


def draw_site_map(parts, truth, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colours = {"grass": "#9bbf7a", "asphalt": "0.3", "concrete": "0.7", "paint": "white",
               "paint_y": "#e0b030", "patch": "0.12", "steel": "#8a5a3a", "pole": "0.5"}
    fig, ax = plt.subplots(figsize=(7, 12))
    for mat in ("grass", "asphalt", "concrete", "patch", "steel", "paint", "paint_y", "pole"):
        for p in parts:
            if p["mat"] == mat and p["name"] != "ground":
                (x0, y0, _), (x1, y1, _) = p["lo"], p["hi"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=colours[mat], ec="none"))
    col = dict(zip(NAMES, ("tab:red", "tab:orange", "tab:purple", "tab:cyan")))
    for t in truth:
        x, y, _ = t["centre"]
        ax.add_patch(Rectangle((x - t["width"] / 2, y - t["height"] / 2), t["width"], t["height"],
                               fc="none", ec=col[t["class"]], lw=2))
        a = math.radians(t["approach_yaw"])
        ax.annotate(t["id"], (x, y), ha="center", va="center", fontsize=7, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6))
        ax.annotate("", (x, y - 0.5 * math.sin(a)), (x - 1.3 * math.cos(a), y - 1.3 * math.sin(a)),
                    arrowprops=dict(arrowstyle="->", color="gold", lw=1.2))
    for name, c in col.items():
        ax.plot([], [], "s", mfc="none", mec=c, mew=2, label=name)
    ax.plot(0, 0, marker="^", color="blue", ms=12, label="drone start")
    ax.set_xlim(*WORLD_SETTINGS["map_x"])
    ax.set_ylim(*WORLD_SETTINGS["map_y"])
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("x (m)  east ->")
    ax.set_ylabel("y (m)  north ->")
    ax.set_title("Road site (top view) - 20 damage patches, gold arrow = flight direction", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd", type=Path, default=RDD, help="rdd4_yolo dataset folder")
    a = ap.parse_args()
    if not a.rdd.exists():
        sys.exit(f"Dataset not found: {a.rdd}  (is the HDD plugged in?)")
    for msg in check_layout():
        print("!! layout:", msg)
    mbf.OUT, mbf.WORLD_NAME = OUT, WORLD_NAME           # the bridge helpers now write the road world
    mbf.MATERIALS.update(EXTRA_MATERIALS)
    make_road_textures()
    for d in ("meshes", "decals"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    print("Choosing unseen RDD2022 TEST photos for the 20 damage patches ...")
    chosen = choose(candidates(a.rdd))

    tree, world = mbf.load_template()
    parts = build_parts()
    sub = mbf.sub
    model = sub(world, "model", name="road")
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
        mbf.pbr_material(vis, mat)
    for p in parts:
        if not p["collide"]:
            continue
        lo, hi = np.array(p["lo"]), np.array(p["hi"])
        col = sub(link, "collision", name=f"col_{p['name']}")
        sub(col, "pose", "%.3f %.3f %.3f 0 0 0" % tuple((lo + hi) / 2))
        sub(sub(sub(col, "geometry"), "box"), "size", "%.3f %.3f %.3f" % tuple(hi - lo))

    # damage patches: flat on the asphalt, image 'up' along the road (+y)
    truth, thumbs = [], []
    for k, ((did, (x, y), lane, zone), (cid, c)) in enumerate(zip(DECALS, chosen), 1):
        img = cv2.imread(str(c["path"]))
        x0, y0, x1, y1 = c["crop"]
        tex = cv2.resize(img[y0:y1, x0:x1], (768, 768), interpolation=cv2.INTER_AREA)
        tex_path = OUT / "decals" / f"{did}.jpg"
        cv2.imwrite(str(tex_path), tex, [cv2.IMWRITE_JPEG_QUALITY, 95])
        size = PATCH_SIZE
        m = sub(world, "model", name=f"defect_{did}")
        sub(m, "static", "true")
        sub(m, "pose", f"{x} {y} {Z_DECAL} 0 0 0")
        vis = sub(sub(m, "link", name="link"), "visual", name="visual")
        plane = sub(sub(vis, "geometry"), "plane")
        sub(plane, "normal", "0 0 1")
        sub(plane, "size", f"{size} {size}")
        sub(vis, "cast_shadows", "false")
        mbw.material(vis, (1, 1, 1), pbr_albedo=str(tex_path))
        nb = lane == "NB"
        truth.append({"id": did, "class": NAMES[cid], "also_present": c["also"], "zone": zone,
                      "centre": [x, y, Z_DECAL], "normal": [0, 0, 1], "width": size, "height": size,
                      "yaw": 0.0, "order": k, "approach_yaw": 90.0 if nb else -90.0,
                      "lane": "north in the west lane" if nb else "south in the east lane",
                      "dataset": "RDD2022 " + c["country"], "source_image": c["path"].name,
                      "crop": list(c["crop"])})
        t = cv2.resize(tex, (240, 240))
        cv2.rectangle(t, (0, 0), (240, 26), (0, 0, 0), -1)
        cv2.putText(t, f"{did} {NAMES[cid]}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        thumbs.append(t)

    # surroundings (Gazebo Fuel) - away from the flight paths
    for i, (x, y) in enumerate([(-6, -14), (-7, -4), (-6, 6), (-7, 16), (13, -12), (14, -1), (13, 9),
                                (14, 20), (-5, 26), (12, -25)], 1):
        mbf.fuel(world, f"tree_{i}", "Oak tree" if i % 2 else "Pine Tree", x, y)
    mbf.fuel(world, "car_1", "Hatchback", 12.0, -6.0, 0.0, math.pi / 2)
    mbf.fuel(world, "car_2", "SUV", 12.0, 4.0, 0.0, math.pi / 2)
    mbf.fuel(world, "car_3", "Pickup", 4.0, 24.0, Z_ROAD, -math.pi / 2)
    for i, x in enumerate((2.0, 3.0, 4.0), 1):
        mbf.fuel(world, f"cone_{i}", "Construction Cone", x, -14.5, Z_ROAD)

    import xml.etree.ElementTree as ET
    ET.indent(tree, space="  ")
    sdf_path = OUT / f"{WORLD_NAME}.sdf"
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
    (OUT / "decals.json").write_text(json.dumps({"world": WORLD_NAME, "decals": truth}, indent=2))
    (OUT / "parts.json").write_text(json.dumps(parts))
    if not (OUT / "world.json").exists():                 # keep your own edits on a re-run
        (OUT / "world.json").write_text(json.dumps(WORLD_SETTINGS, indent=2))
    sheet = np.vstack([np.hstack(thumbs[k:k + 5]) for k in range(0, len(thumbs), 5)])
    cv2.imwrite(str(OUT / "decals_overview.jpg"), sheet)
    draw_site_map(parts, truth, OUT / "site_map.png")

    link_path = mbw.CRAZYSIM_WORLDS / f"{WORLD_NAME}.sdf"
    if mbw.CRAZYSIM_WORLDS.exists():
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(sdf_path)
        print(f"Linked into CrazySim: {link_path}")

    print(f"\nWorld: {sdf_path}\n{len(parts)} road parts, {len(truth)} damage patches\n")
    print(f"{'id':5s}{'class':20s}{'size':>5s}  {'source':24s}zone")
    for t in truth:
        print(f"{t['id']:5s}{t['class']:20s}{t['width']:5.1f}  {t['dataset']:24s}{t['zone']}")
    print("\nNext:  simworld road_full ; simreset ; simfull     then  autopilot  (or pilot)")


if __name__ == "__main__":
    main()
