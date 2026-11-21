# 6. Results

All simulator results use the **realistic A11T camera**: the measured fisheye lens + dark
corners + the "typical" analog radio look, unless said otherwise. *Found* = the defect was
detected on the right 3D patch; *partly* = the predicted class is also present in that photo
(e.g. rust next to exposed bars). False alarms = strong (conf ≥ 0.6) detections off every
patch that persist ≥ 3 frames. Log folders are in `results/runs/`.

## Camera calibration

A11T fisheye: RMS 0.83 px, 122.5° × 97.6° ([doc 3](03_camera_calibration.md)).

## Bridge world (20 patches)

Live autonomous missions:

| Model | Found | Correct | + partly | False alarms |
|---|---|---|---|---|
| B `bridge5_y11s` | 18/20 | 6 | +2 | 3.6 / min |
| E4 `bridge5_y11s_analog` (conf 0.15) | 19/20 | 3 | +5 | 1.2 / min |
| **E6 + box stabilizer** | 17/20 | **7** | +4 | **0.5 / min** |

Offline on the same recorded survey (`auto_20261118_220947`, conf 0.15, typical analog):

| Model | Found | Correct | + partly | False alarms |
|---|---|---|---|---|
| E6 | 18/20 | 6 | +4 | 0.5 / 100 frames |
| E4 | 18/20 | 5 | +4 | |
| B | 17/20 | 6 | +2 | |
| E5, clean digital video | | ~11 | +3 | |

→ The analog video, not the model, limits the class: the same kind of model gets ~11/20
right with a clean picture and ~6–7 with the A11T picture. Finding/locating stays 85–95 %.

## Road model (RDD2022 test split)

mAP50 0.583 (longitudinal 0.585, transverse 0.591, alligator 0.672, pothole 0.482);
China_Drone 0.674, Norway 0.277.

## Road world (20 road-damage patches, camera 55° down)

| Run | Found | Correct | False alarms | Log |
|---|---|---|---|---|
| Road model, whole drone photos | 12/20 | 7 | | auto_20261119_201929 |
| Road model, tight crops | 11/20 | 6 | 0 | auto_20261119_205013 |
| **Road E6** | 20/20 | 16 | 1.3 / min | auto_20261119_222843 |
| **Road E6, clean asphalt + no-sky crops (final)** | **20/20** | **15** | **0** | auto_20261119_225040 |

What was learned on the way: the first patch photos included car bonnets, sky and trees, the
road texture itself had crack-like lines (the 1.3 false alarms / min were the model finding
those), and the plain road model lost most detections through the tilted fisheye + analog
picture. Training on drone-view renders (E6) fixed the last one: 11 → 20 found.

## City world (30 patches, buildings + box girder interior + tunnel)

| Run | Found | Correct | + partly | False alarms | Log |
|---|---|---|---|---|---|
| noon, calm (1) | 29/30 | 12 | +5 | 0.2 / min | auto_20261121_140746 |
| noon, calm (2) | 28/30 | 15 | +5 | 0.4 / min | auto_20261121_143901 |
| noon, calm (3) | 28/30 | 14 | +5 | 0.3 / min | auto_20261121_150210 |
| light wind | 28/30 | 15 | +4 | 0.6 / min | auto_20261121_154949 |

* All 30 stops flown in every run - through the 0.8 m hatch, both manholes, the cross-over
  hole, the 1 m alley and the 0.8 m tunnel - with stop errors of 1–12 cm. 221 m, ~14.5 min.
* Weak points: the model over-predicts *corrosion_stain*; B05 (deep shadow under the deck
  overhang) is missed in every run.

## Wind (city world)

| Wind | Path deviation mean / 95 % | Closest to a wall (drone centre) | Outcome |
|---|---|---|---|
| calm | 3.2 cm / 9 cm | 10 cm (corners cut while climbing) | full mission |
| light 1.5 m/s | 4.4 cm / 11 cm | **4 cm** (under the girder by pier 2: a 10 cm drone grazes) | full mission, stop errors up to 25 cm on the open facade |
| moderate 3 m/s | ±0.4 m sway | - | hit the ground, recovered, crashed after ~8 m |
| gusty, peaks ~5 m/s | - | - | flipped right after take-off |

→ The micro drone inspects safely in **calm to light wind** only. The planner's safety margin
now grows with the wind (14 → 20 cm in light wind).
Caveat: the wind force was sent with the `gz topic` command (~2 updates / s, short gaps), which
is harsher than smooth real wind.

## Overall

* **Navigation** in very tight spaces works: 3 cm mean path deviation, no collision in calm air.
* **Finding** defects: 85–100 % in all worlds.
* **Classifying**: 40–75 % exactly right; limited by the analog video. The practical workflow:
  the drone finds and photographs every defect with its 3D position, an engineer confirms
  the type from the saved photos.
