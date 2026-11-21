# make_wall_world.py - build a Gazebo world with a defect wall from MBDD2025 photos
import csv
import glob
import os
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps

HOME = os.path.expanduser("~")
DATASET = os.path.join(HOME, "drone_ws/yolo/datasets")
OUT_DIR = os.path.join(HOME, "drone_ws/sim/worlds/inspection_wall")
TEX_DIR = os.path.join(OUT_DIR, "textures")
WORLD_NAME = "inspection_wall"

WALL_X = 2.5                       # front face of the wall (m); drone starts at x=0 facing +x
PANEL_W, PANEL_H = 1.2, 0.9        # photo panel size (m), 4:3 like a camera frame
COLS_Y = [2.25, 0.75, -0.75, -2.25]    # panel columns, left to right as seen from the drone
ROWS_Z = [2.45, 1.50, 0.55]            # panel rows, top to bottom
BLANK_SLOTS = {(0, 1), (2, 2)}         # (row, col) left as plain wall
PER_CLASS = 2                      # photos per defect class
FLIP_VERTICAL = False              # set True if the photos appear upside down in Gazebo
random.seed(7)                     # same wall every time you run the script


def find_candidates():
    """Group dataset photos by defect class; keep clear, single-class photos."""
    ann_dirs = glob.glob(os.path.join(DATASET, "**", "Annotations"), recursive=True)
    if not ann_dirs:
        raise SystemExit(f"No 'Annotations' folder found under {DATASET}")
    ann_dir = ann_dirs[0]
    img_dir = os.path.join(os.path.dirname(ann_dir), "JPEGImages")
    by_class = {}
    for xml_path in glob.glob(os.path.join(ann_dir, "*.xml")):
        root = ET.parse(xml_path).getroot()
        w = float(root.findtext("size/width") or 0)
        h = float(root.findtext("size/height") or 0)
        objs = root.findall("object")
        if not objs or len(objs) > 3 or w == 0 or h == 0:
            continue
        names = {o.findtext("name").strip().lower() for o in objs}
        if len(names) != 1:
            continue
        areas = []
        for o in objs:
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            areas.append((x2 - x1) * (y2 - y1) / (w * h))
        if max(areas) < 0.15:          # defect must fill at least 15% of the photo
            continue
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        img = os.path.join(img_dir, stem + ".jpg")
        if os.path.exists(img):
            by_class.setdefault(names.pop(), []).append(img)
    return by_class


def make_texture(src, dst):
    """Centre-crop the photo to 4:3 and save it as a 1024x768 texture."""
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    W, H = im.size
    target = PANEL_W / PANEL_H
    if W / H > target:
        nw = int(H * target)
        im = im.crop(((W - nw) // 2, 0, (W - nw) // 2 + nw, H))
    else:
        nh = int(W / target)
        im = im.crop((0, (H - nh) // 2, W, (H - nh) // 2 + nh))
    im = im.resize((1024, 768))
    if FLIP_VERTICAL:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im.save(dst, quality=92)


def panel_sdf(name, tex, y, z):
    # plane rotated to stand upright, facing the drone (-x)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{WALL_X - 0.01} {y} {z} 1.5708 0 -1.5708</pose>
      <link name="link">
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{PANEL_W} {PANEL_H}</size></plane></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0.1 0.1 0.1 1</specular>
            <pbr><metal><albedo_map>{tex}</albedo_map><roughness>0.9</roughness><metalness>0</metalness></metal></pbr>
          </material>
        </visual>
      </link>
    </model>"""


def world_sdf(panels_xml):
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{WORLD_NAME}">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <plugin name='gz::sim::systems::Physics' filename='libgz-sim-physics-system.so'/>
    <plugin name='gz::sim::systems::UserCommands' filename='libgz-sim-user-commands-system.so'/>
    <plugin name='gz::sim::systems::SceneBroadcaster' filename='libgz-sim-scene-broadcaster-system.so'/>
    <plugin name='gz::sim::systems::Contact' filename='libgz-sim-contact-system.so'/>
    <plugin name='gz::sim::systems::Imu' filename='libgz-sim-imu-system.so'/>
    <plugin name='gz::sim::systems::AirPressure' filename='libgz-sim-air-pressure-system.so'/>
    <plugin name='gz::sim::systems::Sensors' filename='libgz-sim-sensors-system.so'>
      <render_engine>ogre2</render_engine>
    </plugin>
    <gravity>0 0 -9.80665</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <scene>
      <grid>false</grid>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.7 0.8 0.9 1</background>
      <shadows>true</shadows>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="wall">
      <static>true</static>
      <pose>{WALL_X + 0.1} 0 1.6 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>0.2 6.4 3.2</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>0.2 6.4 3.2</size></box></geometry>
          <material><ambient>0.75 0.73 0.70 1</ambient><diffuse>0.75 0.73 0.70 1</diffuse></material>
        </visual>
      </link>
    </model>{panels_xml}
  </world>
</sdf>
"""


def main():
    os.makedirs(TEX_DIR, exist_ok=True)
    by_class = find_candidates()
    print("Usable photos per class:", {k: len(v) for k, v in sorted(by_class.items())})

    chosen = []
    for cls in sorted(by_class):
        picks = random.sample(by_class[cls], min(PER_CLASS, len(by_class[cls])))
        chosen += [(cls, p) for p in picks]
    random.shuffle(chosen)

    slots = [(r, c) for r in range(len(ROWS_Z)) for c in range(len(COLS_Y))
             if (r, c) not in BLANK_SLOTS]
    panels_xml, rows = "", []
    for i, ((r, c), (cls, src)) in enumerate(zip(slots, chosen), start=1):
        tex = os.path.join(TEX_DIR, f"panel_{i:02d}.jpg")
        make_texture(src, tex)
        y, z = COLS_Y[c], ROWS_Z[r]
        panels_xml += panel_sdf(f"panel_{i:02d}", tex, y, z)
        rows.append([f"panel_{i:02d}", r, c, cls, y, z, os.path.basename(src)])

    world_path = os.path.join(OUT_DIR, f"{WORLD_NAME}.sdf")
    with open(world_path, "w") as f:
        f.write(world_sdf(panels_xml))
    with open(os.path.join(OUT_DIR, "ground_truth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "row", "col", "class", "y_m", "z_m", "source_image"])
        w.writerows(rows)

    print(f"World written: {world_path}")
    print(f"{len(rows)} defect panels + {len(BLANK_SLOTS)} blank. Ground truth: ground_truth.csv")
    for row in rows:
        print("  ", row[0], f"{row[3]:<11}", f"y={row[4]:+.2f} z={row[5]:.2f}")


if __name__ == "__main__":
    main()