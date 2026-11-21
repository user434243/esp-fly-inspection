# 7. Logbook

Short entries of completed tasks, in order.

## Hardware

- [x] Chose the ESP-FLY micro drone design with a Seeed XIAO ESP32-S3; ordered 3 spare XIAO ESP32-S3 (regular)
- [x] AKK A11T camera received; video in/out wires bridged; powered from its own battery
- [x] Skydroid 5.8G OTG receiver tested with the phone app: picture on 5731 MHz
- [x] Decided against optical flow + ToF (too heavy); plan a small GPS module for open-sky legs
- [ ] USB data cable for the receiver on the PC; record real defects with the A11T

## Camera

- [x] Checkerboard printed, A11T calibration video recorded
- [x] Fisheye calibration: 122.5° × 97.6°, RMS 0.83 px (`a11t_calib.json`)
- [x] Simulator camera = real lens (LUT remap, dark corners) + analog look (`camprofile real`)

## Models

- [x] Bridge model B on dacl10k + CODEBRIM (5 classes)
- [x] E4 (analog look), E5 (drone view, clean), E6 (drone view through the real lens + analog) - E6 is the default
- [x] Box stabilizer (no blinking boxes) and two-threshold class vote
- [x] RDD2022 downloaded to the HDD, converted to YOLO (4 classes, split per country)
- [x] Road model: test mAP50 0.583
- [x] Road E6 (drone-view renders through the A11T lens): road world 11 → 20/20 found

## Simulation

- [x] CrazySim + Gazebo, FPV camera added to the drone
- [x] Bridge world (`bridge_full`) with tight girder bays and a culvert, 20 defects
- [x] pilot.py (manual) and autopilot.py (A* planner, autonomous inspection, reports)
- [x] Survey recording + offline model comparison (simtest)
- [x] World switcher `simworld` (each world with its own model and camera tilt)
- [x] Road world (`road_full`): 2-lane road, markings, distractors, 20 road damages, camera 55° down
- [x] Clean road texture + better patch photos (no sky / bonnet / trees)
- [x] City world (`city_full`): buildings, arcade, 1 m alley, box girder with hatch/manholes, 0.8 m tunnel, 30 defects
- [x] Lighting presets (noon, morning, evening, overcast, dusk)
- [x] Wind as air-drag force (light / moderate / gusty / strong), path-holding measurement
- [x] Wind tests: calm and light OK, moderate crashes, gusty flips
- [x] Safety margin grows with the wind
- [ ] More bridge types
- [ ] Final report
