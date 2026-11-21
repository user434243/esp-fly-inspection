# 2. Hardware

## The drone: ESP-FLY

| Part | Choice |
|---|---|
| Flight controller / "brain" | Seeed Studio **XIAO ESP32-S3** (regular, not Plus) |
| IMU | MPU6050 (I2C) |
| Motor drivers | 4 × SI2300 MOSFET |
| Barometer / flow / GPS | none on board (see plans below) |
| Firmware | ESP-Drone (Crazyflie-compatible) |
| Link | Wi-Fi, CRTP over UDP port 2390; phone app (ESP-Drone) or PC with a cflib fork |
| Size | ~10 cm, ~27 g class |

The same Crazyflie firmware runs in the simulator (CrazySim), so the PC software (cflib
position setpoints, logging) is the same for the simulated and the real drone.

## The camera: AKK A11T

| | |
|---|---|
| Type | micro all-in-one analog FPV camera + 5.8 GHz video transmitter, 25 mW |
| Channel used | **5731 MHz** |
| Lens | fisheye, **122.5° horizontal / 97.6° vertical** (measured, see doc 3) |
| Wiring | the white and yellow wires (video in / out) bridged; the green wire (Smart Audio) unused |
| Power | its own small battery during the bench tests |

## The receiver: Skydroid 5.8G OTG

A USB (UVC) 5.8 GHz receiver: on a PC or phone it is a normal webcam. Tested with the
Skydroid phone app: picture received on 5731 MHz. A micro-USB / USB-C to USB-A data cable is
needed to connect it to the PC (still to do), then the PC software reads it like the
simulator camera.

## Plans for the real drone

* The ESP32 cannot run ArduPilot, and an optical-flow + ToF sensor set was rejected as too
  heavy for the 10 cm frame.
* Plan: a **small GPS module** for the open-sky legs, linked to the PC path planner; near
  structures the drone is flown by hand (or with the PC planner on a map) - GPS does not
  work under a bridge deck or inside a box girder.
* 3 more XIAO ESP32-S3 boards were ordered as spares.
* Next real-world step: record real concrete / road defects with the A11T through the
  Skydroid receiver and test the models on that video.
