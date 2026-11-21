#!/usr/bin/env python3
"""
make_bridge_full.py - a realistic two-span concrete bridge for free-flight inspection.

Built for a ~10 cm micro drone: the interesting places are TIGHT.
  * Pier: two square columns (1.4 m gap) under a cap beam, steel bearings on top.
  * Superstructure: 4 concrete I-girders with 0.64 m gaps between their bottom flanges,
    closed by cross-beams (diaphragms) -> enclosed bays you enter from below.
  * Abutments with bearing seats, deck with road, parapets, approach embankments.
  * Box culvert (1.0 m wide x 0.7 m high, 6.8 m long) through the north embankment.
  * River bed with gravel and water, grass, trees, cars, cones, barriers, lamp posts.
  * 20 defect patches (4 per class) placed in all zones, using TEST photos the model
    never saw. Their exact 3D position is saved as ground truth for pilot.py.

Realism: every surface is a real mesh with real-scale tiled PBR textures (Poly Haven,
CC0: colour + normal + roughness) - run sim/tools/fetch_assets.py first.

Output  ~/drone_ws/sim/worlds/bridge_full/
  bridge_full.sdf  meshes/*.obj  decals/*.jpg  decals.json  decals_overview.jpg  site_map.png
and a link in CrazySim's worlds folder so 'simfull' can load it.

Coordinates: the drone starts at (0, 0) on the river bed, facing +x (east), looking
straight at the pier. The road runs along y (south -> north).

Run:  ~/drone_ws/venv/bin/python ~/drone_ws/sim/worlds/make_bridge_full.py
"""
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

import make_bridge_world as mbw            # photo selection + template helpers (same folder)

HOME = Path.home()
SIM_WORLDS = HOME / "drone_ws/sim/worlds"
TEXTURES = HOME / "drone_ws/sim/assets/textures"
OUT = SIM_WORLDS / "bridge_full"
WORLD_NAME = "bridge_full"
CLASSES = mbw.CLASSES

# material -> (texture folder, metres per texture tile, fallback colour)
MATERIALS = {
    "concrete":  ("concrete_bridge", 2.0, (0.62, 0.61, 0.58)),
    "weathered": ("concrete_weathered", 2.0, (0.55, 0.53, 0.50)),
    "asphalt":   ("asphalt", 4.0, (0.20, 0.20, 0.21)),
    "grass":     ("ground", 6.0, (0.35, 0.42, 0.25)),
    "gravel":    ("gravel", 3.0, (0.50, 0.47, 0.42)),
    "steel":     ("rust_metal", 1.0, (0.35, 0.30, 0.25)),
}

# ---------------- bridge dimensions (m) ----------------
DECK_X = (1.8, 6.2)              # deck width (road runs along y)
DECK_Y = 9.8                     # deck from -9.8 to +9.8
SLAB_Z = (3.15, 3.40)
GIRDERS_X = [2.5, 3.5, 4.5, 5.5]
GIRDER_Y = 9.3
FLANGE_W, FLANGE_T, WEB_T = 0.36, 0.12, 0.12
GIRDER_BOTTOM = 2.45
DIAPHRAGMS_Y = [-9.2, -4.5, 0.0, 4.5, 9.2]
CAP = dict(x=(2.0, 6.0), y=(-0.4, 0.4), z=(1.75, 2.30))
COLUMNS_X, COLUMN_W = [3.0, 5.0], 0.6
ABUT_Y = 9.0                     # abutment front faces at y = +-9.0
CULVERT_Y, CULVERT_IN_W, CULVERT_IN_H, CULVERT_T = 14.0, 1.0, 0.7, 0.2
EMBANK_X, EMBANK_END = (1.0, 7.0), 22.0


# ---------------- 1. geometry as a list of boxes ----------------
def box(name, x0, x1, y0, y1, z0, z1, mat, collide=True):
    return dict(name=name, lo=(x0, y0, z0), hi=(x1, y1, z1), mat=mat, collide=collide)


def build_parts():
    p = []
    zt = SLAB_Z[1]
    # deck, road, parapets
    p.append(box("deck_slab", *DECK_X, -DECK_Y, DECK_Y, *SLAB_Z, "concrete"))
    p.append(box("road", 2.1, 5.9, -DECK_Y, DECK_Y, zt, zt + 0.02, "asphalt", collide=False))
    p.append(box("parapet_w", 1.8, 2.1, -DECK_Y, DECK_Y, zt, zt + 0.9, "concrete"))
    p.append(box("parapet_e", 5.9, 6.2, -DECK_Y, DECK_Y, zt, zt + 0.9, "concrete"))
    # girders: bottom flange + web (the slab is the top flange)
    web_z0 = GIRDER_BOTTOM + FLANGE_T
    for i, xg in enumerate(GIRDERS_X, 1):
        p.append(box(f"girder{i}_flange", xg - FLANGE_W / 2, xg + FLANGE_W / 2, -GIRDER_Y, GIRDER_Y,
                     GIRDER_BOTTOM, web_z0, "concrete"))
        p.append(box(f"girder{i}_web", xg - WEB_T / 2, xg + WEB_T / 2, -GIRDER_Y, GIRDER_Y,
                     web_z0, SLAB_Z[0], "concrete"))
    # diaphragms between the webs -> closed bays
    for j, yd in enumerate(DIAPHRAGMS_Y):
        for k in range(len(GIRDERS_X) - 1):
            p.append(box(f"diaphragm{j}_{k}", GIRDERS_X[k] + WEB_T / 2, GIRDERS_X[k + 1] - WEB_T / 2,
                         yd - 0.1, yd + 0.1, web_z0, SLAB_Z[0], "concrete"))
    # pier: cap beam, two columns, bearings
    p.append(box("pier_cap", *CAP["x"], *CAP["y"], *CAP["z"], "weathered"))
    for i, xc in enumerate(COLUMNS_X, 1):
        p.append(box(f"column{i}", xc - COLUMN_W / 2, xc + COLUMN_W / 2, -COLUMN_W / 2, COLUMN_W / 2,
                     0, CAP["z"][0], "weathered"))
    for xg in GIRDERS_X:
        for yb in (0.0, -9.15, 9.15):
            p.append(box(f"bearing_{xg}_{yb}", xg - 0.15, xg + 0.15, yb - 0.15, yb + 0.15,
                         CAP["z"][1], GIRDER_BOTTOM, "steel"))
    # abutments: front wall with bearing seat + back wall up to the slab
    for side, s in (("s", -1), ("n", 1)):
        y_front, y_back = s * ABUT_Y, s * DECK_Y
        y_mid = s * 9.4
        p.append(box(f"abutment_{side}", 1.5, 6.5, min(y_front, y_back), max(y_front, y_back),
                     0, CAP["z"][1], "weathered"))
        p.append(box(f"backwall_{side}", 1.5, 6.5, min(y_mid, y_back), max(y_mid, y_back),
                     CAP["z"][1], SLAB_Z[0], "weathered"))
    # approach embankments (north one has a culvert through it)
    yc0, yc1 = CULVERT_Y - CULVERT_IN_W / 2 - CULVERT_T, CULVERT_Y + CULVERT_IN_W / 2 + CULVERT_T
    top_c = CULVERT_IN_H + 2 * CULVERT_T
    p.append(box("embank_s", *EMBANK_X, -EMBANK_END, -DECK_Y, 0, zt, "grass"))
    p.append(box("embank_n1", *EMBANK_X, DECK_Y, yc0, 0, zt, "grass"))
    p.append(box("embank_n2", *EMBANK_X, yc1, EMBANK_END, 0, zt, "grass"))
    p.append(box("embank_n3", *EMBANK_X, yc0, yc1, top_c, zt, "grass"))
    for side, y0, y1 in (("s", -EMBANK_END, -DECK_Y), ("n", DECK_Y, EMBANK_END)):
        p.append(box(f"road_{side}", 2.1, 5.9, y0, y1, zt, zt + 0.02, "asphalt", collide=False))
    # box culvert (floor, walls, roof), a little longer than the embankment
    cx0, cx1 = EMBANK_X[0] - 0.4, EMBANK_X[1] + 0.4
    p.append(box("culvert_floor", cx0, cx1, yc0, yc1, 0, CULVERT_T, "weathered"))
    p.append(box("culvert_wall_s", cx0, cx1, yc0, yc0 + CULVERT_T, CULVERT_T, CULVERT_T + CULVERT_IN_H,
                 "weathered"))
    p.append(box("culvert_wall_n", cx0, cx1, yc1 - CULVERT_T, yc1, CULVERT_T, CULVERT_T + CULVERT_IN_H,
                 "weathered"))
    p.append(box("culvert_roof", cx0, cx1, yc0, yc1, CULVERT_T + CULVERT_IN_H, top_c, "weathered"))
    # landscape (visual only - the template ground plane does the physics)
    p.append(box("ground", -35, 45, -40, 40, -0.05, 0.0, "grass", collide=False))
    p.append(box("river_bed", -35, 45, -ABUT_Y, ABUT_Y, -0.05, 0.006, "gravel", collide=False))
    return p


# ---------------- 2. defect patches (decals) ----------------
#        id    centre (x, y, z)       outward normal  width  zone
DECALS = [
    ("D01", (2.69, 0.00, 1.00), (-1, 0, 0), 0.48, "pier column W - west face"),
    ("D02", (3.00, -0.31, 0.60), (0, -1, 0), 0.48, "pier column W - south face"),
    ("D03", (5.00, 0.31, 1.20), (0, 1, 0), 0.48, "pier column E - north face"),
    ("D04", (5.31, 0.00, 0.80), (1, 0, 0), 0.48, "pier column E - east face"),
    ("D05", (3.31, 0.00, 1.40), (1, 0, 0), 0.48, "between columns - column W inner face"),
    ("D06", (4.00, -0.41, 2.02), (0, -1, 0), 0.60, "pier cap - south face"),
    ("D07", (2.80, 0.41, 2.02), (0, 1, 0), 0.60, "pier cap - north face"),
    ("D08", (4.00, -0.11, 2.86), (0, -1, 0), 0.48, "bay 2 - diaphragm over pier"),
    ("D09", (2.57, -2.50, 2.86), (1, 0, 0), 0.48, "bay 1 - girder web"),
    ("D10", (4.43, 3.00, 2.86), (-1, 0, 0), 0.48, "bay 2 - girder web"),
    ("D11", (4.57, -6.00, 2.86), (1, 0, 0), 0.48, "bay 3 - girder web"),
    ("D12", (5.00, 4.39, 2.86), (0, -1, 0), 0.48, "bay 3 - midspan diaphragm"),
    ("D13", (2.43, 6.00, 2.86), (-1, 0, 0), 0.48, "outer girder W - side face"),
    ("D14", (5.57, -3.00, 2.86), (1, 0, 0), 0.48, "outer girder E - side face"),
    ("D15", (3.00, -8.99, 1.20), (0, 1, 0), 0.80, "south abutment"),
    ("D16", (5.00, 8.99, 0.80), (0, -1, 0), 0.80, "north abutment"),
    ("D17", (2.50, 8.99, 1.80), (0, -1, 0), 0.60, "north abutment - near bearing seat"),
    ("D18", (2.50, 13.51, 0.55), (0, 1, 0), 0.48, "culvert - south wall"),
    ("D19", (5.50, 14.49, 0.55), (0, -1, 0), 0.48, "culvert - north wall"),
    ("D20", (1.79, -5.00, 3.85), (-1, 0, 0), 0.60, "parapet - outer face"),
]


# ---------------- 3. mesh writing (tiled UVs in real metres) ----------------
FACES = [  # normal, U axis, V axis   (U x V = normal, so faces point outwards)
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 1, 0), (-1, 0, 0), (0, 0, 1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
]


def write_obj(parts, path, tile):
    v_lines, vt_lines, vn_lines, f_lines = [], [], [], []
    for part in parts:
        lo, hi = np.array(part["lo"]), np.array(part["hi"])
        c, half = (lo + hi) / 2, (hi - lo) / 2
        for n, u, v in FACES:
            n, u, v = map(np.array, (n, u, v))
            a, b = abs(half @ u), abs(half @ v)
            centre = c + n * abs(half @ n)
            base = len(v_lines) + 1
            vn_lines.append("vn %d %d %d" % tuple(n))
            ni = len(vn_lines)
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = centre + su * a * u + sv * b * v
                v_lines.append("v %.4f %.4f %.4f" % tuple(p))
                vt_lines.append("vt %.4f %.4f" % ((p @ u) / tile, (p @ v) / tile))
            i = [base + k for k in range(4)]
            f_lines.append(f"f {i[0]}/{i[0]}/{ni} {i[1]}/{i[1]}/{ni} {i[2]}/{i[2]}/{ni}")
            f_lines.append(f"f {i[0]}/{i[0]}/{ni} {i[2]}/{i[2]}/{ni} {i[3]}/{i[3]}/{ni}")
    path.write_text("\n".join(["# generated by make_bridge_full.py"] + v_lines + vt_lines +
                              vn_lines + f_lines) + "\n")


# ---------------- 4. SDF helpers ----------------
sub = mbw.sub


def pbr_material(visual, mat_key):
    folder, _, rgb = MATERIALS[mat_key]
    tex = TEXTURES / folder
    m = sub(visual, "material")
    sub(m, "ambient", "0.8 0.8 0.8 1" if (tex / "albedo.jpg").exists() else f"{rgb[0]} {rgb[1]} {rgb[2]} 1")
    sub(m, "diffuse", "1 1 1 1" if (tex / "albedo.jpg").exists() else f"{rgb[0]} {rgb[1]} {rgb[2]} 1")
    sub(m, "specular", "0.05 0.05 0.05 1")
    if (tex / "albedo.jpg").exists():
        # Gazebo caches textures by FILE NAME, so every map gets a unique name
        # (otherwise all materials would show the first 'albedo.jpg' it loaded)
        def unique(kind):
            src = tex / f"{kind}.jpg"
            dst = OUT / "textures" / f"{folder}_{kind}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copyfile(src, dst)
                return str(dst)
            return None
        metal = sub(sub(m, "pbr"), "metal")
        sub(metal, "albedo_map", unique("albedo"))
        if unique("normal"):
            sub(metal, "normal_map", unique("normal"))
        if unique("roughness"):
            sub(metal, "roughness_map", unique("roughness"))
        sub(metal, "roughness", "1.0")
        sub(metal, "metalness", "0.0")
    return m


def decal_pose(centre, normal):
    yaw = math.atan2(normal[1], normal[0]) + math.pi / 2
    return f"{centre[0]} {centre[1]} {centre[2]} 1.5708 0 {yaw:.4f}"


def fuel(world, name, model, x, y, z=0.0, yaw=0.0):
    inc = sub(world, "include")
    sub(inc, "uri", f"https://fuel.gazebosim.org/1.0/OpenRobotics/models/{model}")
    sub(inc, "name", name)
    sub(inc, "pose", f"{x} {y} {z} 0 0 {yaw}")
    sub(inc, "static", "true")


def load_template():
    hits = sorted(SIM_WORLDS.rglob("inspection_wall.sdf"))
    if not hits:
        sys.exit(f"inspection_wall.sdf not found under {SIM_WORLDS}")
    tree = ET.parse(hits[0])
    world = tree.getroot().find("world")
    if world.get("name") == "inspection_wall":
        world.set("name", WORLD_NAME)
    for model in list(world.findall("model")):
        if "ground" not in (model.get("name") or "").lower():
            world.remove(model)
        else:                               # keep its physics, hide its plain grey look
            for link in model.findall("link"):
                for vis in link.findall("visual"):
                    link.remove(vis)
    scene = world.find("scene")
    if scene is None:
        scene = sub(world, "scene")
    for tag, val in (("ambient", "0.55 0.55 0.55 1"), ("background", "0.62 0.75 0.92 1"),
                     ("shadows", "true")):
        e = scene.find(tag)
        (e if e is not None else sub(scene, tag)).text = val
    if scene.find("sky") is None:
        sub(scene, "sky")
    return tree, world


# ---------------- 5. build everything ----------------
def main():
    missing = [MATERIALS[m][0] for m in MATERIALS if not (TEXTURES / MATERIALS[m][0] / "albedo.jpg").exists()]
    if missing:
        print("!! textures missing (plain colours used instead):", ", ".join(missing))
    if not mbw.DATASET.exists():
        sys.exit(f"Dataset not found: {mbw.DATASET}  (is the HDD plugged in?)")
    for d in ("meshes", "decals"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # defect photos: 4 per class from the unseen TEST split
    print("Choosing unseen TEST photos for the 20 defect patches ...")
    mbw.PER_CLASS = len(DECALS) // len(CLASSES)
    chosen = mbw.choose(mbw.candidates())

    tree, world = load_template()
    parts = build_parts()

    # bridge = one static model: one mesh per material (looks), one box per part (physics)
    model = sub(world, "model", name="bridge")
    sub(model, "static", "true")
    link = sub(model, "link", name="link")
    for mat, (_, tile, _) in MATERIALS.items():
        group = [p for p in parts if p["mat"] == mat]
        if not group:
            continue
        obj = OUT / "meshes" / f"{mat}.obj"
        write_obj(group, obj, tile)
        vis = sub(link, "visual", name=f"vis_{mat}")
        sub(sub(sub(vis, "geometry"), "mesh"), "uri", "file://" + str(obj))
        pbr_material(vis, mat)
    for p in parts:
        if not p["collide"]:
            continue
        lo, hi = np.array(p["lo"]), np.array(p["hi"])
        col = sub(link, "collision", name=f"col_{p['name']}")
        sub(col, "pose", "%.3f %.3f %.3f 0 0 0" % tuple((lo + hi) / 2))
        sub(sub(sub(col, "geometry"), "box"), "size", "%.3f %.3f %.3f" % tuple(hi - lo))

    # water (visual only, see-through)
    water = sub(world, "model", name="river_water")
    sub(water, "static", "true")
    wv = sub(sub(water, "link", name="link"), "visual", name="visual")
    sub(wv, "pose", "5 4.5 0.03 0 0 0")
    sub(sub(sub(wv, "geometry"), "box"), "size", "80 3 0.02")
    sub(wv, "transparency", "0.35")
    wm = sub(wv, "material")
    for tag, val in (("ambient", "0.10 0.22 0.28 1"), ("diffuse", "0.12 0.28 0.35 1"),
                     ("specular", "0.6 0.6 0.6 1")):
        sub(wm, tag, val)

    # defect patches
    truth, thumbs = [], []
    for (did, centre, normal, width, zone), (cid, src, also) in zip(DECALS, chosen):
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        x0, y0, x1, y1 = mbw.crop_43(w, h)
        tex = cv2.resize(img[y0:y1, x0:x1], (640, 480), interpolation=cv2.INTER_CUBIC)
        tex_path = OUT / "decals" / f"{did}.jpg"
        cv2.imwrite(str(tex_path), tex, [cv2.IMWRITE_JPEG_QUALITY, 95])
        height = width * 0.75
        m = sub(world, "model", name=f"defect_{did}")
        sub(m, "static", "true")
        sub(m, "pose", decal_pose(centre, normal))
        vis = sub(sub(m, "link", name="link"), "visual", name="visual")
        plane = sub(sub(vis, "geometry"), "plane")
        sub(plane, "normal", "0 0 1")
        sub(plane, "size", f"{width} {height:.3f}")
        sub(vis, "cast_shadows", "false")
        mbw.material(vis, (1, 1, 1), pbr_albedo=str(tex_path))
        source = "CODEBRIM" if src.name.startswith("cb_") else "dacl10k"
        truth.append({"id": did, "class": CLASSES[cid], "also_present": also, "zone": zone,
                      "centre": list(centre), "normal": list(normal), "width": width,
                      "height": round(height, 3), "dataset": source, "source_image": src.name})
        t = cv2.resize(tex, (320, 240))
        cv2.rectangle(t, (0, 0), (320, 30), (0, 0, 0), -1)
        cv2.putText(t, f"{did} {CLASSES[cid]}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        thumbs.append(t)

    # surroundings from Gazebo Fuel
    for i, (x, y) in enumerate([(-9, -14), (-12, -6), (-10, 12), (-14, 18), (13, -15), (15, -5),
                                (14, 10), (12, 24), (-6, 26), (20, 2)], 1):
        fuel(world, f"tree_{i}", "Oak tree" if i % 2 else "Pine Tree", x, y)
    zr = SLAB_Z[1] + 0.02
    fuel(world, "car_1", "Hatchback", 3.05, -4.0, zr, math.pi / 2)
    fuel(world, "car_2", "SUV", 4.95, 3.0, zr, -math.pi / 2)
    fuel(world, "car_3", "Pickup", 3.05, 17.0, zr, math.pi / 2)
    for i, y in enumerate((6.4, 7.0, 7.6), 1):
        fuel(world, f"cone_{i}", "Construction Cone", 4.0, y, zr)
    for i, (x, y) in enumerate([(1.3, 12.0), (6.7, 12.0), (1.3, -14.0), (6.7, -14.0)], 1):
        fuel(world, f"barrier_{i}", "Jersey Barrier", x, y, SLAB_Z[1], math.pi / 2)
    for i, y in enumerate((-6.0, 6.0), 1):
        fuel(world, f"lamp_{i}", "Lamp Post", 1.95, y, SLAB_Z[1] + 0.9)

    ET.indent(tree, space="  ")
    sdf_path = OUT / f"{WORLD_NAME}.sdf"
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
    (OUT / "decals.json").write_text(json.dumps({"world": WORLD_NAME, "decals": truth}, indent=2))
    sheet = np.vstack([np.hstack(thumbs[k:k + 5]) for k in range(0, len(thumbs), 5)])
    cv2.imwrite(str(OUT / "decals_overview.jpg"), sheet)
    draw_site_map(parts, truth, OUT / "site_map.png")

    link_path = mbw.CRAZYSIM_WORLDS / f"{WORLD_NAME}.sdf"
    if mbw.CRAZYSIM_WORLDS.exists():
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(sdf_path)
        print(f"Linked into CrazySim: {link_path}")

    print(f"\nWorld: {sdf_path}\n{len(parts)} bridge parts, {len(truth)} defect patches\n")
    print(f"{'id':5s}{'class':17s}{'dataset':10s}zone")
    for t in truth:
        print(f"{t['id']:5s}{t['class']:17s}{t['dataset']:10s}{t['zone']}")


def draw_site_map(parts, truth, path):
    """Top view of the site with every defect patch numbered - for planning the demo flight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colours = {"concrete": "0.65", "weathered": "0.5", "asphalt": "0.25", "grass": "#9bbf7a",
               "gravel": "#cbbf9f", "steel": "#8a5a3a"}
    order = ["grass", "gravel", "weathered", "concrete", "asphalt", "steel"]
    fig, ax = plt.subplots(figsize=(9, 11))
    for mat in order:
        for p in parts:
            if p["mat"] == mat and p["name"] != "ground":
                (x0, y0, _), (x1, y1, _) = p["lo"], p["hi"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=colours[mat], ec="k", lw=0.3,
                                       alpha=0.55 if mat in ("asphalt", "concrete") else 0.9))
    ax.add_patch(Rectangle((-35, 3), 80, 3, fc="#6fa8c9", alpha=0.5))
    for t in truth:
        x, y, _ = t["centre"]
        ax.plot(x, y, "o", color="red", ms=5)
        ax.annotate(t["id"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color="darkred")
    ax.plot(0, 0, marker="^", color="blue", ms=12)
    ax.annotate("drone start (facing +x)", (0, 0), xytext=(-2.5, -1.4), fontsize=9, color="blue")
    ax.set_xlim(-4, 10)
    ax.set_ylim(-12, 18)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)  east ->")
    ax.set_ylabel("y (m)  north ->")
    ax.set_title("Bridge site (top view) - red = defect patches, deck drawn see-through")
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
