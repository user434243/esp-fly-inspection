#!/usr/bin/env python3
"""
make_road_e6.py - road model 'E6': teach rdd4_y11s.pt how the DRONE sees a road.

The road model learned from dashcam / drone photos. In the simulator (and on the real drone) it sees
damage through the A11T: camera tilted ~55 deg down, 0.5-1.5 m above the asphalt, fisheye lens with
dark corners, analog radio video. This script renders the TRAIN / VAL photos exactly like that:

  1. square crop around a damage box (like the sim patches), laid flat on an asphalt ground
     (random size 0.7-2.0 m, position, small turn), sometimes with lane lines (NOT damage)
  2. seen from a camera 0.5-1.5 m high, tilted 45-65 deg down (same 960x720 / 136 deg render as the sim)
  3. real A11T fisheye + dark corners (a11t_calib.json) -> 640x480, light / shadow change
  4. analog look (good / typical / bad) on 85 %
  + 8 % empty road (only asphalt + lines) so the model learns markings are not damage
  + the original photos (half with the analog look) so it does not forget the normal views
The boxes go through the same steps. Only train / val photos are used - the 20 sim patches come
from the TEST split, so the sim score stays honest.

  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/make_road_e6.py            # build the dataset (~10 min)
  ~/drone_ws/venv/bin/python ~/drone_ws/scripts/make_road_e6.py --train    # build (if needed) + train (~1 h)
Output: <HDD>/rdd4_e6_yolo/  (examples.jpg = what it made)   model: ~/drone_ws/yolo/models/rdd4_y11s_e6.pt
"""
import argparse
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

HOME = Path.home()
sys.path.insert(0, str(HOME / "drone_ws/scripts"))
from analog_cam import AnalogFX
from fisheye_cam import Lens

HDD = Path("/media/hudaifa/Ventoy/datasets")
TEXTURES = HOME / "drone_ws/sim/assets/textures"
MODELS = HOME / "drone_ws/yolo/models"
RW, RH, RHFOV = 960, 720, math.radians(136)               # same render as the simulator (camprofile real)
FR = (RW / 2) / math.tan(RHFOV / 2)
W, H = 640, 480
LOOKS = {"good": 0.3, "typical": 0.5, "bad": 0.2}
P_ANALOG, P_EMPTY, P_LINES = 0.85, 0.08, 0.4
DRONE_COPIES = 2                                          # top-down photos: rendered twice (different views)
STREET_PER_SPLIT = {"train": 5000, "val": 600}            # dashcam photos rendered (lower road crops)
ORIG_PER_SPLIT = {"train": 6000, "val": 800}              # original photos kept (not forgetting)
HOOD = 0.88


def ground_to_render(h, pitch):
    """Homography: ground point (X forward, Y left, metres, below the camera) -> render pixel."""
    c, s = math.cos(pitch), math.sin(pitch)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])       # camera -> body (pitched down)
    src = np.float32([[1, -1], [1, 1], [4, -2], [4, 2]])
    dst = []
    for X, Y in src:
        d = R.T @ np.array([X, Y, -h])
        dst.append([RW / 2 - FR * d[1] / d[0], RH / 2 - FR * d[2] / d[0]])
    return cv2.getPerspectiveTransform(src, np.float32(dst))


class Renderer:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.lens = Lens(HOME / "drone_ws/sim/a11t_calib.json", RHFOV)
        self.fx = {name: AnalogFX(name, seed=seed * 10 + i) for i, name in enumerate(LOOKS)}
        tex = [cv2.imread(str(TEXTURES / t / "albedo.jpg")) for t in ("asphalt", "asphalt_patch")]
        self.ground = [t for t in tex if t is not None]
        if not self.ground:                                 # no texture: grey noise asphalt
            g = np.full((1024, 1024, 3), 70, np.uint8) + self.rng.integers(0, 30, (1024, 1024, 1), dtype=np.uint8)
            self.ground = [cv2.GaussianBlur(g, (3, 3), 0)]

    def background(self, Hg, h, pitch):
        """Tiled asphalt seen through the camera; sky above the horizon, haze far away."""
        r = self.rng
        tex = random.choice(self.ground)
        tile = r.uniform(3.0, 5.0)                          # metres per texture tile
        S = np.array([[tile / tex.shape[1], 0, r.uniform(0, tile)],
                      [0, tile / tex.shape[0], r.uniform(0, tile)], [0, 0, 1]])
        img = cv2.warpPerspective(tex, Hg @ S, (RW, RH), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_WRAP).astype(np.float32)
        img *= r.uniform(0.95, 1.05, 3) * r.uniform(0.75, 1.2)                 # asphalt colour / brightness
        c, s = math.cos(pitch), math.sin(pitch)
        yn = (np.arange(RH, dtype=np.float32)[:, None] - RH / 2) / FR          # per image row
        xn = (np.arange(RW, dtype=np.float32)[None, :] - RW / 2) / FR
        dz = -s - c * yn                                     # ray direction (body), image row/col
        dist = h * np.sqrt((c - s * yn) ** 2 + xn ** 2) / np.maximum(-dz, 1e-6)
        far = np.clip((dist - 8) / 20, 0, 1)[..., None] * np.ones((1, RW, 1), np.float32)
        sky = np.array([210, 195, 175], np.float32)
        img = img * (1 - far) + sky * far                    # haze, then pure sky above the horizon
        img[np.broadcast_to(dz >= 0, (RH, RW))] = sky
        return img

    def lines(self, img, Hg):
        """Lane / edge lines (paint) - NOT damage."""
        r = self.rng
        for _ in range(r.integers(1, 3)):
            y0, wl = r.uniform(-2.5, 2.5), r.uniform(0.08, 0.15)
            col = (225, 225, 220) if r.random() < 0.7 else (40, 170, 215)
            a = math.radians(r.uniform(-8, 8))
            L = np.array([[0.2, y0], [25, y0 + 25 * math.tan(a)], [25, y0 + wl + 25 * math.tan(a)], [0.2, y0 + wl]])
            q = cv2.perspectiveTransform(L[None].astype(np.float32), Hg)[0]
            cv2.fillPoly(img, [np.round(q).astype(np.int32)], col)

    def view(self, photo, boxes):
        """photo: square crop (BGR) or None, boxes: [(cls, x0, y0, x1, y1) in crop pixels].
        Returns the 640x480 drone frame and YOLO label lines."""
        r = self.rng
        h, pitch = r.uniform(0.5, 1.5), math.radians(r.uniform(45, 65))
        Hg = ground_to_render(h, pitch)
        img = self.background(Hg, h, pitch)
        if r.random() < P_LINES:
            self.lines(img, Hg)
        out = []
        if photo is not None:
            side = r.uniform(0.7, 2.0)                       # patch size on the road (m)
            d_view = h / math.tan(pitch)                     # where the camera centre hits the ground
            cx, cy = d_view + r.uniform(-0.5, 0.9), r.uniform(-0.9, 0.9)
            a = math.radians(r.choice([0, 180]) + r.uniform(-12, 12))
            ps = photo.shape[0]
            k = side / ps                                    # crop pixel -> metres; photo 'up' = forward
            Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

            def to_ground(px):                               # crop pixels (u right, v down) -> ground X, Y
                g = np.stack([(ps / 2 - px[:, 1]) * k, (ps / 2 - px[:, 0]) * k], 1) @ Rm.T
                return g + [cx, cy]

            corners = np.float32([[0, 0], [ps, 0], [ps, ps], [0, ps]])
            M = cv2.getPerspectiveTransform(corners, cv2.perspectiveTransform(
                to_ground(corners)[None].astype(np.float32), Hg)[0])
            warped = cv2.warpPerspective(photo, M, (RW, RH), flags=cv2.INTER_AREA).astype(np.float32)
            mask = cv2.warpPerspective(np.ones(photo.shape[:2], np.float32), M, (RW, RH))[..., None]
            img = img * (1 - mask) + warped * mask
            t = np.linspace(0, 1, 9)
            for c, X0, Y0, X1, Y1 in boxes:
                xs = np.r_[X0 + (X1 - X0) * t, np.full(9, X1), X1 - (X1 - X0) * t, np.full(9, X0)]
                ys = np.r_[np.full(9, Y0), Y0 + (Y1 - Y0) * t, np.full(9, Y1), Y1 - (Y1 - Y0) * t]
                q = cv2.perspectiveTransform(np.stack([xs, ys], 1)[None].astype(np.float32), M)[0]
                q = np.array([self.lens.norm_to_pix((u - RW / 2) / FR, (v - RH / 2) / FR) for u, v in q])
                full = (q[:, 0].max() - q[:, 0].min()) * (q[:, 1].max() - q[:, 1].min())
                qx0, qy0 = np.clip(q.min(0), 0, [W, H])
                qx1, qy1 = np.clip(q.max(0), 0, [W, H])
                if qx1 - qx0 >= 4 and qy1 - qy0 >= 4 and (qx1 - qx0) * (qy1 - qy0) >= 0.3 * full:
                    out.append(f"{int(c)} {(qx0 + qx1) / 2 / W:.6f} {(qy0 + qy1) / 2 / H:.6f} "
                               f"{(qx1 - qx0) / W:.6f} {(qy1 - qy0) / H:.6f}")
        frame = np.clip(img * r.uniform(0.7, 1.15), 0, 255).astype(np.uint8)    # sun / shade
        frame = self.lens.apply(frame)
        if r.random() < P_ANALOG:
            frame = self.fx[random.choices(list(LOOKS), list(LOOKS.values()))[0]](frame)
        return frame, out


def read_boxes(p, w, h):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) == 5:
                c, x, y, bw, bh = int(q[0]), *map(float, q[1:])
                out.append((c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return out


def square_crop(img, boxes, drone, rng):
    """Like the sim patches: a square around one damage box (drone photos: sometimes the whole photo)."""
    h, w = img.shape[:2]
    y_max = h if drone else int(h * HOOD)
    if drone and rng.random() < 0.3:
        side, x0, y0 = min(w, h), (w - min(w, h)) // 2, (h - min(w, h)) // 2
    else:
        ok = [b for b in boxes if (drone or (b[2] + b[4]) / 2 > 0.45 * h) and b[4] <= y_max]
        if not ok:
            return None, []
        c, X0, Y0, X1, Y1 = ok[rng.integers(len(ok))]
        lo = 200 if drone else 160
        side = int(min(min(w, y_max), max(rng.uniform(1.2, 1.8) * max(X1 - X0, Y1 - Y0), lo)))
        x0 = int(min(max((X0 + X1) / 2 - side / 2 + rng.uniform(-0.15, 0.15) * side, 0), w - side))
        y0 = int(min(max((Y0 + Y1) / 2 - side / 2 + rng.uniform(-0.15, 0.15) * side, 0), y_max - side))
    keep = []
    for c, X0, Y0, X1, Y1 in boxes:                          # boxes inside the crop (>= 40 % of them)
        a0, b0 = max(X0, x0), max(Y0, y0)
        a1, b1 = min(X1, x0 + side), min(Y1, y0 + side)
        if a1 > a0 and b1 > b0 and (a1 - a0) * (b1 - b0) >= 0.4 * (X1 - X0) * (Y1 - Y0):
            keep.append((c, a0 - x0, b0 - y0, a1 - x0, b1 - y0))
    crop = cv2.resize(img[y0:y0 + side, x0:x0 + side], (640, 640), interpolation=cv2.INTER_AREA)
    s = 640 / side
    return crop, [(c, a * s, b * s, e * s, f * s) for c, a, b, e, f in keep]


def build(rdd, out):
    rng = np.random.default_rng(5)
    random.seed(5)
    ren = Renderer(7)
    plain_fx = AnalogFX("typical", seed=99)
    lists = {}
    for split in ("train", "val"):
        for d in ("images", "labels"):
            (out / d / split).mkdir(parents=True, exist_ok=True)
        imgs = sorted((rdd / "images" / split).glob("*.jpg"))
        drone = [p for p in imgs if "drone" in p.stem.lower()]
        street = [p for p in imgs if "drone" not in p.stem.lower()]
        street = [street[i] for i in rng.permutation(len(street))[:STREET_PER_SPLIT[split]]]
        jobs = [(p, True, k) for p in drone for k in range(DRONE_COPIES)] + [(p, False, 0) for p in street]
        n_empty = int(len(jobs) * P_EMPTY)
        jobs += [(None, False, k) for k in range(n_empty)]
        made = []
        for p, is_drone, k in tqdm(jobs, desc=f"{split} drone views"):
            name = f"e6_{p.stem}_{k}" if p is not None else f"e6_empty_{split}_{k}"
            dst = out / "images" / split / f"{name}.jpg"
            if not dst.exists():
                photo, boxes = None, []
                if p is not None:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    photo, boxes = square_crop(img, read_boxes(rdd / "labels" / split / f"{p.stem}.txt",
                                                               img.shape[1], img.shape[0]), is_drone, rng)
                    if photo is None:
                        continue
                frame, labels = ren.view(photo, boxes)
                cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(labels))
            made.append(str(dst))
        # originals (half with the analog look, no lens) so normal views are not forgotten
        orig = [imgs[i] for i in rng.permutation(len(imgs))[:ORIG_PER_SPLIT[split]]]
        for p in tqdm(orig, desc=f"{split} originals"):
            dst = out / "images" / split / f"o_{p.stem}.jpg"
            if not dst.exists():
                img = cv2.imread(str(p))
                if img is None:
                    continue
                if rng.random() < 0.5:
                    img = plain_fx(img)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                shutil.copy(rdd / "labels" / split / f"{p.stem}.txt", out / "labels" / split / f"o_{p.stem}.txt")
            made.append(str(dst))
        lists[split] = out / f"{split}.txt"
        lists[split].write_text("\n".join(made))
        print(f"{split}: {len(made)} images ({len(drone)} top-down photos x{DRONE_COPIES}, {len(street)} street, "
              f"{n_empty} empty road, {len(orig)} originals)")
    names = yaml.safe_load((rdd / "data.yaml").read_text())["names"]
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out), "train": str(lists["train"]), "val": str(lists["val"]), "names": names}, sort_keys=False))
    examples(out, names)


def examples(out, names):
    ex = [p for p in sorted((out / "images/train").glob("e6_China*.jpg"))
          if (out / "labels/train" / f"{p.stem}.txt").read_text().strip()][:4]
    ex += [p for p in sorted((out / "images/train").glob("e6_*.jpg"))
           if "China" not in p.stem and (out / "labels/train" / f"{p.stem}.txt").read_text().strip()][:4]
    tiles = []
    for p in ex:
        im = cv2.imread(str(p))
        for line in (out / "labels/train" / f"{p.stem}.txt").read_text().splitlines():
            c, x, y, w_, h_ = map(float, line.split())
            a, b = (int((x - w_ / 2) * W), int((y - h_ / 2) * H)), (int((x + w_ / 2) * W), int((y + h_ / 2) * H))
            cv2.rectangle(im, a, b, (0, 255, 0), 2)
            cv2.putText(im, names[int(c)], (a[0], max(14, a[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(cv2.resize(im, (320, 240)))
    tiles += [np.zeros((240, 320, 3), np.uint8)] * (8 - len(tiles))
    cv2.imwrite(str(out / "examples.jpg"), cv2.vconcat([cv2.hconcat(tiles[:4]), cv2.hconcat(tiles[4:8])]))
    print("examples:", out / "examples.jpg")


def train(out):
    from ultralytics import YOLO
    model = YOLO(str(MODELS / "rdd4_y11s.pt"))
    model.train(data=str(out / "data.yaml"), imgsz=640, epochs=30, patience=10, batch=16, cache=False,
                optimizer="AdamW", lr0=0.0005, device=0, project=str(HOME / "drone_ws/yolo/runs"),
                name="rdd4_y11s_e6", exist_ok=True)
    best = HOME / "drone_ws/yolo/runs/rdd4_y11s_e6/weights/best.pt"
    shutil.copy(best, MODELS / "rdd4_y11s_e6.pt")
    print("saved", MODELS / "rdd4_y11s_e6.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdd", type=Path, default=HDD / "rdd4_yolo")
    ap.add_argument("--out", type=Path, default=HDD / "rdd4_e6_yolo")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if not (a.out / "data.yaml").exists():
        build(a.rdd, a.out)
    if a.train:
        train(a.out)
