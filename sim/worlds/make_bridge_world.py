#!/usr/bin/env python3
"""
make_bridge_world.py - build the Gazebo bridge-inspection world.

What it builds
  * A concrete bridge PIER whose front face is at x = 2.5 m (same place as the old
    wall, so the zigzag scan and the inspection code work unchanged), a DECK on top
    with a road surface and parapets, and the ground.
  * 10 defect PANELS (1.2 m x 0.9 m) on the pier face, 2 per class, in a 4 x 3 grid
    with 2 slots left blank (clean concrete = should give NO detection).
  * The panel photos come ONLY from the TEST split of bridge5_yolo - photos the
    model never saw in training - one from dacl10k and one from CODEBRIM per class
    where possible, so the score is honest. The 'cleanest' photo of each defect
    is chosen (only that defect if possible). Exposed rebar almost always comes
    with rust or spalling, so other defects in a photo are listed in
    ground_truth.csv as 'also_present'.

Output  ~/drone_ws/sim/worlds/bridge_pier/
  bridge_pier.sdf   ground_truth.csv   textures/panel_XX.jpg   panels_overview.jpg
and a link  .../CrazySim/.../gazebo/worlds/bridge_pier.sdf  so 'simbridge' can load it.

The world header (physics, plugins, sun, ground) is copied from the working
inspection_wall.sdf, so the drone, IMU and camera behave exactly as before.

Run:  ~/drone_ws/venv/bin/python ~/drone_ws/sim/worlds/make_bridge_world.py
"""
import csv
import random
import sys
from collections import Counter
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

# ---------------- paths ----------------
HOME = Path.home()
DATASET = Path("/media/hudaifa/Ventoy/datasets/bridge5_yolo")   # model B dataset (HDD)
SIM_WORLDS = HOME / "drone_ws/sim/worlds"
OUT = SIM_WORLDS / "bridge_pier"
CRAZYSIM_WORLDS = (HOME / "drone_ws/sim/CrazySim/crazyflie-firmware/tools/"
                   "crazyflie-simulation/simulator_files/gazebo/worlds")
WORLD_NAME = "bridge_pier"

# ---------------- classes and selection rules ----------------
CLASSES = ["crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"]
PER_CLASS = 2
MIN_LARGEST_BOX = 0.03        # the class's largest box must cover >= 3% of the photo
MAX_BOXES = 8
MIN_INSIDE_CROP = 0.6         # >= 60% of the largest box must survive the 4:3 crop
SEED = 7

# ---------------- geometry (same as the old wall, so scan + inspection still fit) ----------------
FACE_X = 2.5                  # pier front face
PANEL_W, PANEL_H = 1.2, 0.9
COLS_Y = [2.25, 0.75, -0.75, -2.25]
ROWS_Z = [2.45, 1.50, 0.55]
BLANK_SLOTS = {(0, 1), (2, 2)}   # (row, col) left as clean concrete
TEX_W, TEX_H = 800, 600


# ---------------- 1. choose panel photos ----------------
def read_labels(txt):
    rows = []
    for line in txt.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            rows.append((int(p[0]), *map(float, p[1:])))
    return rows


def crop_43(w, h):
    """Centre crop box (x0, y0, x1, y1) in pixels with a 4:3 aspect ratio."""
    if w / h > 4 / 3:
        cw, ch = round(h * 4 / 3), h
    else:
        cw, ch = w, round(w * 3 / 4)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return x0, y0, x0 + cw, y0 + ch


def candidates():
    """For every TEST photo and every defect class in it, one candidate entry.
    share = how much of the photo's boxed area belongs to that class (1.0 = only that
    defect in the photo). The other classes in the photo are kept as 'also present'."""
    found = {c: {"dacl": [], "cb": []} for c in range(len(CLASSES))}
    for img_path in sorted((DATASET / "images/test").glob("*.jpg")):
        labels = read_labels(DATASET / "labels/test" / f"{img_path.stem}.txt")
        if not labels or len(labels) > MAX_BOXES:
            continue
        area = Counter()
        for l in labels:
            area[l[0]] += l[3] * l[4]
        h, w = cv2.imread(str(img_path)).shape[:2]
        x0, y0, x1, y1 = crop_43(w, h)
        source = "cb" if img_path.name.startswith("cb_") else "dacl"
        for cid in area:
            big = max((l for l in labels if l[0] == cid), key=lambda l: l[3] * l[4])
            if big[3] * big[4] < MIN_LARGEST_BOX:
                continue
            bx0, bx1 = (big[1] - big[3] / 2) * w, (big[1] + big[3] / 2) * w
            by0, by1 = (big[2] - big[4] / 2) * h, (big[2] + big[4] / 2) * h
            inside = max(0, min(bx1, x1) - max(bx0, x0)) * max(0, min(by1, y1) - max(by0, y0))
            if inside / ((bx1 - bx0) * (by1 - by0)) < MIN_INSIDE_CROP:
                continue
            share = area[cid] / sum(area.values())
            also = sorted(CLASSES[c] for c in area if c != cid)
            found[cid][source].append((img_path, also, share))
    return found


def choose(found):
    """Per class: the 'cleanest' photos first (highest share, random among equals),
    one from dacl10k and one from CODEBRIM when possible. A photo is used only once."""
    rng = random.Random(SEED)
    chosen, used = [], set()

    def ranked(pool):
        pool = [c for c in pool if c[0] not in used]
        rng.shuffle(pool)
        return sorted(pool, key=lambda c: -round(c[2], 1))      # share in 0.1 steps

    # rarest classes first, so they get their best photos before others take them
    order = sorted(range(len(CLASSES)), key=lambda c: len(found[c]["dacl"]) + len(found[c]["cb"]))
    for cid in order:
        name = CLASSES[cid]
        picks = []
        for src in ("dacl", "cb"):
            pool = ranked(found[cid][src])
            if pool and len(picks) < PER_CLASS:
                picks.append(pool[0])
                used.add(pool[0][0])
        rest = ranked(found[cid]["dacl"] + found[cid]["cb"])
        for cand in rest[:PER_CLASS - len(picks)]:
            picks.append(cand)
            used.add(cand[0])
        if len(picks) < PER_CLASS:
            sys.exit(f"Not enough test photos for class '{name}' - lower MIN_LARGEST_BOX")
        chosen += [(cid, path, also) for path, also, _ in picks]
        print(f"  {name:16s} candidates dacl10k: {len(found[cid]['dacl']):3d}  "
              f"CODEBRIM: {len(found[cid]['cb']):3d}  -> "
              + ", ".join(f"{p.name} (share {sh:.0%}{', +' + '+'.join(a) if a else ''})"
                          for p, a, sh in picks))
    rng.shuffle(chosen)
    return chosen


# ---------------- 2. SDF helpers ----------------
def sub(parent, tag, text=None, **attrib):
    e = ET.SubElement(parent, tag, attrib)
    if text is not None:
        e.text = str(text)
    return e


def material(visual, rgb, pbr_albedo=None):
    m = sub(visual, "material")
    sub(m, "ambient", f"{rgb[0]} {rgb[1]} {rgb[2]} 1")
    sub(m, "diffuse", f"{rgb[0]} {rgb[1]} {rgb[2]} 1")
    sub(m, "specular", "0.05 0.05 0.05 1")
    if pbr_albedo:
        metal = sub(sub(m, "pbr"), "metal")
        sub(metal, "albedo_map", pbr_albedo)
        sub(metal, "roughness", "0.9")
        sub(metal, "metalness", "0.0")


def box_model(world, name, size, pose, rgb, collide=True, shadows=True):
    m = sub(world, "model", name=name)
    sub(m, "static", "true")
    sub(m, "pose", " ".join(map(str, pose)))
    link = sub(m, "link", name="link")
    if collide:
        geo = sub(sub(link, "collision", name="collision"), "geometry")
        sub(sub(geo, "box"), "size", " ".join(map(str, size)))
    vis = sub(link, "visual", name="visual")
    sub(sub(sub(vis, "geometry"), "box"), "size", " ".join(map(str, size)))
    sub(vis, "cast_shadows", "true" if shadows else "false")
    material(vis, rgb)


def panel_model(world, name, y, z, texture):
    m = sub(world, "model", name=name)
    sub(m, "static", "true")
    sub(m, "pose", f"{FACE_X - 0.01} {y} {z} 1.5708 0 -1.5708")
    vis = sub(sub(m, "link", name="link"), "visual", name="visual")
    plane = sub(sub(vis, "geometry"), "plane")
    sub(plane, "normal", "0 0 1")
    sub(plane, "size", f"{PANEL_W} {PANEL_H}")
    sub(vis, "cast_shadows", "false")
    material(vis, (1, 1, 1), pbr_albedo=str(texture))


def load_template():
    hits = sorted(SIM_WORLDS.rglob("inspection_wall.sdf"))
    if not hits:
        sys.exit(f"inspection_wall.sdf not found under {SIM_WORLDS}")
    tree = ET.parse(hits[0])
    world = tree.getroot().find("world")
    if world.get("name") == "inspection_wall":
        world.set("name", WORLD_NAME)
    for model in list(world.findall("model")):              # keep only the ground
        if "ground" not in (model.get("name") or "").lower():
            world.remove(model)
    has_ground = any("ground" in (m.get("name") or "").lower() for m in world.findall("model"))
    print(f"Template: {hits[0]}  (world name '{world.get('name')}', ground kept: {has_ground})")
    return tree, world, has_ground


# ---------------- 3. build ----------------
def main():
    if not DATASET.exists():
        sys.exit(f"Dataset not found: {DATASET}  (is the HDD plugged in?)")
    (OUT / "textures").mkdir(parents=True, exist_ok=True)

    print("Choosing unseen TEST photos for the panels ...")
    chosen = choose(candidates())

    tree, world, has_ground = load_template()
    if not has_ground:
        box_model(world, "ground", (40, 40, 0.1), (0, 0, -0.05, 0, 0, 0), (0.45, 0.42, 0.36))

    concrete, asphalt = (0.62, 0.61, 0.58), (0.18, 0.18, 0.19)
    box_model(world, "pier", (1.0, 6.4, 3.2), (FACE_X + 0.5, 0, 1.6, 0, 0, 0), concrete)
    box_model(world, "pier_cap", (1.4, 7.2, 0.35), (FACE_X + 0.5, 0, 3.375, 0, 0, 0), concrete,
              shadows=False)
    box_model(world, "deck", (3.0, 12.0, 0.45), (FACE_X + 0.5, 0, 3.775, 0, 0, 0), concrete,
              shadows=False)
    box_model(world, "road", (2.6, 12.0, 0.03), (FACE_X + 0.5, 0, 4.015, 0, 0, 0), asphalt,
              collide=False)
    for side, x in (("near", FACE_X - 0.9), ("far", FACE_X + 1.9)):
        box_model(world, f"parapet_{side}", (0.2, 12.0, 0.8), (x, 0, 4.4, 0, 0, 0), concrete,
                  shadows=False)

    rows, overview = [], []
    slots = [(r, c) for r in range(len(ROWS_Z)) for c in range(len(COLS_Y))
             if (r, c) not in BLANK_SLOTS]
    for i, ((r, c), (cid, src, also)) in enumerate(zip(slots, chosen), start=1):
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        x0, y0, x1, y1 = crop_43(w, h)
        tex = cv2.resize(img[y0:y1, x0:x1], (TEX_W, TEX_H), interpolation=cv2.INTER_CUBIC)
        tex_path = OUT / "textures" / f"panel_{i:02d}.jpg"
        cv2.imwrite(str(tex_path), tex, [cv2.IMWRITE_JPEG_QUALITY, 95])
        panel_model(world, f"panel_{i:02d}", COLS_Y[c], ROWS_Z[r], tex_path)
        source = "CODEBRIM" if src.name.startswith("cb_") else "dacl10k"
        rows.append([f"{i:02d}", r, c, CLASSES[cid], ";".join(also), COLS_Y[c], ROWS_Z[r],
                     source, src.name])
        thumb = cv2.resize(tex, (400, 300))
        cv2.rectangle(thumb, (0, 0), (400, 34), (0, 0, 0), -1)
        extra = f" +{len(also)}" if also else ""
        cv2.putText(thumb, f"{i:02d} {CLASSES[cid]}{extra} ({source})", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        overview.append(thumb)

    ET.indent(tree, space="  ")
    sdf_path = OUT / f"{WORLD_NAME}.sdf"
    tree.write(sdf_path, encoding="utf-8", xml_declaration=True)

    with open(OUT / "ground_truth.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["panel", "row", "col", "class", "also_present", "y_m", "z_m", "dataset",
                     "source_image"])
        wr.writerows(rows)

    while len(overview) % 5:
        overview.append(np.zeros_like(overview[0]))
    sheet = np.vstack([np.hstack(overview[k:k + 5]) for k in range(0, len(overview), 5)])
    cv2.imwrite(str(OUT / "panels_overview.jpg"), sheet)

    link = CRAZYSIM_WORLDS / f"{WORLD_NAME}.sdf"
    if CRAZYSIM_WORLDS.exists():
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(sdf_path)
        print(f"Linked into CrazySim: {link}")
    else:
        print(f"WARNING: CrazySim worlds folder not found: {CRAZYSIM_WORLDS}")

    print(f"\nWorld:        {sdf_path}")
    print(f"Ground truth: {OUT / 'ground_truth.csv'}")
    print(f"Overview:     {OUT / 'panels_overview.jpg'}\n")
    print(f"{'panel':6s}{'class':17s}{'y':>7s}{'z':>7s}  {'dataset':10s}also present")
    for row in rows:
        print(f"{row[0]:6s}{row[3]:17s}{row[5]:7.2f}{row[6]:7.2f}  {row[7]:10s}{row[4] or '-'}")


if __name__ == "__main__":
    main()