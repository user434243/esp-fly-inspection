#!/usr/bin/env python3
"""
rdd2yolo.py - convert RDD2022 (Pascal VOC XML, 7 countries) into one YOLO road-damage dataset.

  python rdd2yolo.py                         # defaults below
  python rdd2yolo.py --max-side 1280 --workers 8

Keeps the 4 standard road-damage classes:
  D00 longitudinal_crack   D10 transverse_crack   D20 alligator_crack   D40 pothole
(other RDD labels, e.g. repairs / manholes / blurred markings, are not damage -> ignored, counted in the summary)
Only the labelled 'train' folders are used; each country is split 80 / 10 / 10 (train / val / test), seed 0.
Images are shrunk to --max-side (long side) to save disk and loading time - YOLO trains at 640 anyway.
Safe to run again: finished images are skipped.
Output: <out>/images/{train,val,test}, labels/..., data.yaml, data_test_<country>.yaml
"""
import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

HDD = Path("/media/hudaifa/Ventoy/datasets")
CLASSES = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
SPLIT = (0.8, 0.1, 0.1)


def find_pairs(src):
    pairs = {}
    for train in sorted(src.glob("*_x/*/train")):
        country = train.parent.name
        imgs = {p.stem: p for p in (train / "images").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        xmls = sorted((train / "annotations").rglob("*.xml"))
        pairs[country] = [(imgs[x.stem], x) for x in xmls if x.stem in imgs]
    return pairs


def convert(job):
    img_p, xml_p, out, split, max_side = job
    stem = img_p.stem
    dst_img = out / "images" / split / f"{stem}.jpg"
    dst_lab = out / "labels" / split / f"{stem}.txt"
    root = ET.parse(xml_p).getroot()
    objs = [(o.findtext("name", "").strip(), o.find("bndbox")) for o in root.iter("object")]
    kept, other = Counter(), Counter()
    if dst_img.exists() and dst_lab.exists():                      # already done (re-run)
        for name, _ in objs:
            (kept if name in CLASSES else other)[name] += 1
        return split, kept, other, 0
    img = cv2.imread(str(img_p))
    if img is None:
        return split, kept, other, 1
    h, w = img.shape[:2]                                           # trust the image, not the XML size
    lines = []
    for name, bb in objs:
        if name not in CLASSES or bb is None:
            other[name] += 1
            continue
        x0, y0, x1, y1 = (float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        x0, x1 = sorted((max(0.0, min(w, x0)), max(0.0, min(w, x1))))
        y0, y1 = sorted((max(0.0, min(h, y0)), max(0.0, min(h, y1))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        kept[name] += 1
        lines.append(f"{CLASSES[name]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                     f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    dst_lab.write_text("\n".join(lines))
    return split, kept, other, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=HDD / "rdd2022_raw/RDD2022")
    ap.add_argument("--out", type=Path, default=HDD / "rdd4_yolo")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for d in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (a.out / d / sp).mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(a.src)
    if not pairs:
        raise SystemExit(f"!! no */train folders found under {a.src}")
    jobs, per_country = [], {}
    for country, items in pairs.items():
        random.Random(0).shuffle(items)
        n = len(items)
        n_tr, n_va = round(n * SPLIT[0]), round(n * SPLIT[1])
        parts = {"train": items[:n_tr], "val": items[n_tr:n_tr + n_va], "test": items[n_tr + n_va:]}
        per_country[country] = {k: [p[0].stem for p in v] for k, v in parts.items()}
        jobs += [(img, xml, a.out, sp, a.max_side) for sp, v in parts.items() for img, xml in v]
        print(f"{country:16s} {n:6d} labelled images -> train {len(parts['train'])}, "
              f"val {len(parts['val'])}, test {len(parts['test'])}")

    kept, other, bad, per_split = Counter(), Counter(), 0, Counter()
    with Pool(a.workers) as pool:
        for sp, k, o, b in tqdm(pool.imap_unordered(convert, jobs, chunksize=16), total=len(jobs), desc="converting"):
            kept.update(k); other.update(o); bad += b; per_split[sp] += 1

    base = {"path": str(a.out), "train": "images/train", "val": "images/val", "names": NAMES}
    (a.out / "data.yaml").write_text(yaml.safe_dump({**base, "test": "images/test"}, sort_keys=False))
    for country, parts in per_country.items():                     # per-country test sets (e.g. China_Drone)
        files = [str(a.out / "images/test" / f"{s}.jpg") for s in parts["test"]]
        (a.out / f"test_{country}.txt").write_text("\n".join(files))
        (a.out / f"data_test_{country}.yaml").write_text(
            yaml.safe_dump({**base, "test": f"test_{country}.txt"}, sort_keys=False))

    print("\nimages:", dict(per_split), "  unreadable:", bad)
    print("kept boxes:")
    for name, i in CLASSES.items():
        print(f"   {name} {NAMES[i]:20s} {kept[name]:7d}")
    if other:
        print("ignored labels (not one of the 4 damage classes):", dict(other.most_common()))
    print("\ndataset ready:", a.out / "data.yaml")


if __name__ == "__main__":
    main()
