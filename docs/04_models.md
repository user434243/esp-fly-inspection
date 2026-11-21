# 4. Models and datasets

All models are **YOLO11s** (Ultralytics), 640 px input. Every simulator test uses photos from
the datasets' **test split only**, so the models never saw the defects they are scored on.

## Datasets (not in this repo - download from the sources)

| Dataset | Used for | Classes kept |
|---|---|---|
| dacl10k + CODEBRIM → `bridge5_yolo` | bridge / concrete model | crack, spallation, efflorescence, exposed_bars, corrosion_stain |
| RDD2022 (7 countries) → `rdd4_yolo` (`scripts/rdd2yolo.py`) | road model | D00 longitudinal, D10 transverse, D20 alligator crack, D40 pothole |

`rdd2yolo.py` converts the RDD2022 Pascal-VOC labels to YOLO, splits every country
80 / 10 / 10 (seed 0), shrinks images to 1280 px and writes a test list per country.
Other RDD labels (repairs, manholes, blurred marks ...) are not damage and are dropped.

## Bridge / concrete models

| Model | File | How it was made |
|---|---|---|
| B | `bridge5_y11s.pt` | trained on bridge5_yolo |
| E4 | `bridge5_y11s_analog.pt` | B fine-tuned with the analog-video look added |
| E5 | `bridge5_y11s_droneview.pt` | drone-view pictures, clean digital camera |
| **E6** | `bridge5_y11s_e6.pt` | E4 fine-tuned on train photos **as the drone sees them**: pasted on concrete, seen at an angle, bent by the real A11T lens, dark corners, analog look on 85 %; 8 % empty walls (40 epochs, AdamW, lr 0.0005) — **the default** |

Live tricks in `pilot.py` (used by everything):
* **Box stabilizer** (`box_stabilizer.py`): a box is shown only after 3 hits in 5 frames, kept
  3 frames, classes voted over the track - no blinking boxes, fewer false alarms.
* **Two-threshold class vote**: detections from conf 0.15 are used to *find* a defect, but only
  detections ≥ 0.35 decide its *class* (weak ones count 1 %).

## Road models

| Model | File | Score |
|---|---|---|
| Road | `rdd4_y11s.pt` | 50 epochs from yolo11s. Val mAP50 0.59; test 0.583 (longitudinal 0.585, transverse 0.591, alligator 0.672, pothole 0.482). By country: China_Drone 0.674 (top-down drone photos), Norway 0.277 (small damage in wide photos) |
| **Road E6** | `rdd4_y11s_e6.pt` | `scripts/make_road_e6.py`: train/val photos cropped like the sim patches, laid on asphalt (0.7–2 m), seen from 0.5–1.5 m with the camera 45–65° down through the A11T lens + analog look; empty road with lane lines (not damage); 6000 original photos kept. 30 epochs from Road. Val mAP50 0.394 on this much harder drone-view val set |

Road E6 took the road world from **11/20 found, 6 correct** to **20/20 found, 15 correct, 0
false alarms**.

## Training notebooks

`notebooks/04_model_upgrade.ipynb` (bridge E4 → E6) and `notebooks/05_road_model.ipynb`
(RDD2022 conversion, road model, per-country test). Training curves and confusion matrices of
every run are in `results/training/`, dataset examples in `results/datasets/`.

## The trained weights

The `.pt` files are attached to the GitHub **release** (see `models/README.md` for names, sizes
and checksums), not committed to git.
