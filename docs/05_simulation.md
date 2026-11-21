# 5. Simulation

Gazebo Sim 7 + **CrazySim**: the real Crazyflie firmware runs in the loop, the PC talks to it
over the same cflib link as to the real drone. An FPV camera was added to the drone model
(`sim/patches/add_camera.py`), 3 cm in front of the centre.

## Worlds (`simworld <name>`)

| World | What | Patches | Camera | Model |
|---|---|---|---|---|
| `bridge_full` | two-span concrete bridge: pier with 2 columns, 4 I-girders with closed bays entered from below (0.64 m gaps), abutments, culvert 1.0 × 0.7 m, river | 20 | straight ahead | bridge E6 |
| `road_full` | 7 m two-lane road, kerbs, sidewalks, lamp posts, markings, zebra crossing, stop line, repair patches, manholes (not damage) | 20 road damages | **55° down** | road E6 |
| `city_full` | 36 × 37 m: 3-storey building with a 2.6 m arcade, balconies, **1.0 m alley** with a beam across, box-girder bridge with **interior cells** (0.8 m hatch, 0.7 × 0.9 m manholes, cross-over hole, lamps), **0.8 m L-tunnel** under a berm, street, cars, trees | 30 | straight ahead | bridge E6 |

Every world folder has `world.json` (model, camera tilt, map size, light, wind ...),
`decals.json` (the ground truth: id, class, 3D position, normal, size, source photo),
`parts.json` (every box, for the planner), `site_map.png`, `decals_overview.jpg`.
The generators (`sim/worlds/make_*.py`) rebuild a world from the datasets in a minute.

Realism details: real-scale tiled PBR textures (Poly Haven), one mesh per material, the road
texture cleaned of its own crack-like lines (they caused false alarms), defect photos laid on
the surfaces, Fuel models for trees and cars (with invisible keep-out boxes so the planner
avoids them).

## Flying

* **`pilot.py`** - you fly (Xbox pad Mode 2 or keyboard), live YOLO, a route guide shows the
  next stop, mini-map, report at the end.
* **`autopilot.py`** - autonomous mission: `planner.py` (A* on a 10 cm 3D grid + string
  pulling, slower near walls) plans collision-free legs between the stops of `route.py`;
  at each stop the drone turns to the patch, scans (closer views, look left/right) until the
  defect is found, then moves on; flies home and lands.
* Every detection is turned into a 3D ray (drone pose + lens model) and matched to the patch it
  hits; a patch is FOUND after 3 frames. Strong off-patch detections that persist are counted
  as **false alarms**.
* Output: `logs/auto_<date>/` report.png, findings.csv, mission_log.csv, tracking.csv,
  findings/ (best photo per defect), false_alarms/, video.mp4.
* `SURVEY=1 autopilot` saves clean frames; `simtest.py` replays them offline through any model
  and analog strength (same frames for every model = fair comparison).

## Light (`simworld --light noon|morning|evening|overcast|dusk`)

Changes the sun direction and colour, the sky and the ambient light of the world.

## Wind (`simworld --wind calm|light|moderate|gusty|strong`)

Gazebo's own WindEffects system flipped the CrazySim drone even in light wind, so the wind is
made in `wind_gust.py`: wind speed = mean × (1 + gust swing · sin + random part), and the air
drag

$$\mathbf F = c\,\lvert \mathbf w-\mathbf v\rvert\,(\mathbf w-\mathbf v), \qquad c=\tfrac12\,\rho\,C_d\,A \approx 0.006\ \mathrm{N/(m/s)^2}$$

($\mathbf w$ = wind, $\mathbf v$ = drone velocity) is pushed on the drone through Gazebo's **ApplyLinkWrench**
system, only once it is airborne, and much weaker inside sheltered spaces (girder cells and
tunnel 5 %, arcade 40 %, alley 50 %).

| preset | mean | gusts | push (drone weight ≈ 270 mN) |
|---|---|---|---|
| light | 1.5 m/s | ±15 % | ~16 mN |
| moderate | 3 m/s | ±25 % | ~60 mN |
| gusty | 3 m/s | ±70 %, every 3 s | ~70 mN, peaks ~140 |
| strong | 5 m/s | ±30 % | ~180 mN |

`autopilot` measures **path holding** 20× a second: deviation from the planned leg, distance to
the nearest surface (with the 5 closest places listed), saved in `tracking.csv`.

## Safety margin

The planner keeps the drone centre this far from every surface: **calm 14 cm, light 20 cm,
moderate 24 cm, gusty 26 cm, strong 28 cm** (`world.json "clear"` overrides). Stops keep 2 cm
more. At 26 cm the 1.0 m alley and the 0.8 m tunnel become impossible - another way of seeing
that a micro drone cannot inspect tight spaces in strong wind.
