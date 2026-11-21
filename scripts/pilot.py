#!/usr/bin/env python3
"""
pilot.py - fly the micro drone yourself through the bridge while the AI inspects.

  start the world first:   simfull        then:   python scripts/pilot.py

XBOX CONTROLLER (Mode 2, like a real drone remote)
  left stick   up/down = climb/descend      left/right = turn (yaw)
  right stick  up/down = forward/back       left/right = fly sideways
  A take off   B land   X precision mode   Y snapshot   VIEW help   MENU land + finish

KEYBOARD (click the pilot window first) - works at the same time
  T take off   L land   W/S forward/back   A/D left/right
  UP/DOWN climb/descend   LEFT/RIGHT turn   TAB precision mode
  P snapshot   H help   ESC land and finish (the report is written at the end)

Flight feel: speeds ramp up and down smoothly (like a real drone's inertia); sticks are
proportional with a small dead-zone and expo (gentle near the centre, fast at the edge).
Top speed 1.0 m/s - the Crazyflie position controller's own limit.

What it does
  * Live YOLO (model B, 5 bridge defect classes) on the drone camera.
  * Every detection is projected into 3D (drone position + heading + pixel) and matched
    to the defect patches of the world (decals.json). A patch counts as FOUND after
    3 frames; its first good view is saved as a photo.
  * Pilot display: altitude, speed mode, heading, fps, findings, mini-map with your path.
  * At the end: logs/pilot_<date>/  report.png  findings.csv  findings/*.jpg  video.mp4
"""
import csv
import json
import math
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "drone_ws/sim/worlds"))

# ---------------- settings ----------------
MODEL = HOME / "drone_ws/yolo/models/bridge5_y11s_e6.pt"
DECALS = HOME / "drone_ws/sim/worlds/bridge_full/decals.json"
CONF = 0.15
CLASS_CONF = 0.35       # only detections at least this sure decide the defect CLASS
WEAK_VOTE = 0.01        # weaker ones (CONF..CLASS_CONF) only count when nothing sure was seen
# --- active world (simworld) ---
_ACTIVE = HOME / "drone_ws/sim/active_world"
WORLD_NAME = _ACTIVE.read_text().strip() if _ACTIVE.exists() else "bridge_full"
WORLD_DIR = HOME / "drone_ws/sim/worlds" / WORLD_NAME
WORLD = json.loads((WORLD_DIR / "world.json").read_text()) if (WORLD_DIR / "world.json").exists() else {}
DECALS = WORLD_DIR / "decals.json"
if WORLD.get("model"):
    MODEL = HOME / "drone_ws/yolo/models" / WORLD["model"]
CONF = WORLD.get("conf", CONF)
STABILIZE = True        # steady boxes: shown after 3 hits in 5 frames, no blinking
from box_stabilizer import BoxStabilizer
STAB = BoxStabilizer(STABILIZE)
_CAMF = HOME / "drone_ws/sim/camera_profile.json"          # set by camprofile.py
CAM_PROFILE = (json.loads(_CAMF.read_text()) if _CAMF.exists()
               else {"name": "clean", "hfov": 1.57, "analog": None})
IMG_W, IMG_H, HFOV = 640, 480, CAM_PROFILE["hfov"]
FOCAL = (IMG_W / 2) / math.tan(HFOV / 2)
LENS = None                                                    # real-lens model (fisheye_cam.Lens) or pinhole
if CAM_PROFILE.get("lens"):
    from fisheye_cam import Lens
    LENS = Lens(CAM_PROFILE["lens"], CAM_PROFILE["render_hfov"])


def pix_to_norm(u, v):
    """Image pixel -> direction at depth 1 (x right, y down)."""
    if LENS is not None:
        return LENS.pix_to_norm(u, v)
    return (u - IMG_W / 2) / FOCAL, (v - IMG_H / 2) / FOCAL


def norm_to_pix(x, y):
    """Direction at depth 1 (x right, y down) -> image pixel."""
    if LENS is not None:
        return LENS.norm_to_pix(x, y)
    return IMG_W / 2 + x * FOCAL, IMG_H / 2 + y * FOCAL
CAM_OFFSET = np.array([0.03, 0.0, 0.015])
CAM_PITCH = math.radians(WORLD.get("camera_pitch_deg", 0.0))      # camera tilted down (roads)
CAM_ROT = np.array([[math.cos(CAM_PITCH), 0, math.sin(CAM_PITCH)], [0, 1, 0],
                    [-math.sin(CAM_PITCH), 0, math.cos(CAM_PITCH)]])   # camera -> body
RATE = 20                                  # setpoints per second
TAKEOFF_Z, MIN_Z, MAX_Z = 0.5, 0.12, 4.6
LEASH = 0.35                               # how far the target may lead the drone, + speed-dependent
SPEEDS = {"normal": (1.0, 0.7, 90.0), "precision": (0.3, 0.25, 35.0)}   # max m/s, m/s, deg/s
ACCEL = (2.5, 2.0, 300.0)                  # how fast speeds ramp: m/s2, m/s2, deg/s2
DEADZONE, EXPO = 0.12, 0.45                # stick dead-zone and expo curve
FOUND_FRAMES = 3
ALARM_CONF = 0.6                           # off-patch detection counts as a false alarm only above this
ALARM_PX = 60                              # ... and if it stays (same class, centre moves < 60 px)
HOLD_FOUND, HOLD_MAX = 2.5, 6.0            # route: seconds at a stop after FOUND / at most
SCALE = 1.5                                # camera image shown at 960 x 720
MAP_W, MAP_H = 230, 330
MAP_X, MAP_Y = tuple(WORLD.get("map_x", (-4.0, 10.0))), tuple(WORLD.get("map_y", (-12.0, 18.0)))


# ---------------- 3D geometry: pixel -> ray -> which defect patch? ----------------
def pixel_ray(u, v, pos, yaw_deg):
    """Ray (origin, unit direction) in world coordinates for image pixel (u, v)."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    x, y = pix_to_norm(u, v)
    d = np.array([1.0, -x, -y])
    d = rot @ (CAM_ROT @ (d / np.linalg.norm(d)))
    return np.asarray(pos, float) + rot @ CAM_OFFSET, d


def project(P, pos, yaw_deg):
    """World point -> image pixel (u, v), or None if behind the camera."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    b = CAM_ROT.T @ (rot.T @ (np.asarray(P, float) - np.asarray(pos, float) - rot @ CAM_OFFSET))
    if b[0] < 0.1:
        return None
    u, v = norm_to_pix(-b[1] / b[0], -b[2] / b[0])
    return int(max(-5000, min(5000, u))), int(max(-5000, min(5000, v)))


def ray_box(origin, d, lo, hi):
    """Distance along the ray to an axis-aligned box, or None (slab method)."""
    t0, t1 = 0.0, 1e9
    for k in range(3):
        if abs(d[k]) < 1e-9:
            if origin[k] < lo[k] or origin[k] > hi[k]:
                return None
            continue
        a, b = (lo[k] - origin[k]) / d[k], (hi[k] - origin[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return None
    return t0


def find_decal(origin, d, decals, boxes, max_range=4.5, margin=0.08):
    """The defect patch this ray looks at (nearest, facing the camera, not hidden)."""
    best = None
    for dc in decals:
        n, c = dc["_n"], dc["_c"]
        denom = d @ n
        if denom > -0.05:                   # seen from behind or edge-on
            continue
        t = ((c - origin) @ n) / denom
        if t <= 0.05 or t > max_range:
            continue
        off = origin + t * d - c
        if abs(off @ dc["_u"]) > dc["width"] / 2 + margin or abs(off @ dc["_v"]) > dc["height"] / 2 + margin:
            continue
        if best is None or t < best[1]:
            best = (dc, t)
    if best is None:
        return None
    dc, t = best
    for lo, hi in boxes:                    # something solid in between?
        tb = ray_box(origin, d, lo, hi)
        if tb is not None and 0.02 < tb < t - 0.03:
            return None
    return dc


def load_world():
    data = json.loads(DECALS.read_text())["decals"]
    for dc in data:
        dc["_c"] = np.array(dc["centre"], float)
        dc["_n"] = np.array(dc["normal"], float)
        if abs(dc["_n"][2]) > 0.9:                 # patch on the ground: 'yaw' turns its width axis
            a = math.radians(dc.get("yaw", 0.0))
            dc["_u"] = np.array([math.cos(a), math.sin(a), 0.0])
        else:                                     # patch on a wall: width horizontal
            dc["_u"] = np.array([-dc["_n"][1], dc["_n"][0], 0.0])
        dc["_v"] = np.cross(dc["_n"], dc["_u"])        # height axis (up on walls)
        dc["_h"] = dc["_u"]
    if (WORLD_DIR / "parts.json").exists():
        parts = json.loads((WORLD_DIR / "parts.json").read_text())
    else:
        import make_bridge_full as mbf
        parts = mbf.build_parts()
    boxes = [(np.array(p["lo"]), np.array(p["hi"])) for p in parts
             if p["name"] not in ("ground", "river_bed") and not p.get("flat")]
    return data, parts, boxes


# ---------------- flight control (background thread, 20 Hz) ----------------
class FlightControl:
    """Keys set body-frame speeds; this loop turns them into position setpoints.
    Position setpoints are what works reliably in CrazySim (the high-level commander does not)."""

    def __init__(self, drone):
        self.d = drone
        self.lock = threading.Lock()
        self.cmd = {"fwd": 0.0, "left": 0.0, "up": 0.0, "turn": 0.0}      # wanted (from sticks/keys)
        self.vel = {"fwd": 0.0, "left": 0.0, "up": 0.0, "turn": 0.0}      # actual, ramped smoothly
        self.state = "LANDED"               # LANDED -> TAKEOFF -> FLYING -> LANDING -> LANDED
        self.target, self.yaw_t = None, 0.0
        self.yaw = 0.0                      # measured heading (deg), set by the log callback
        self.running = True
        self._stop_count = 0

    def request(self, state):
        with self.lock:
            if state == "TAKEOFF" and self.state == "LANDED":
                x, y, _ = self.d.real
                self.target, self.yaw_t = [x, y, 0.0], self.yaw
                self.state = "TAKEOFF"
            elif state == "LANDING" and self.state in ("TAKEOFF", "FLYING"):
                self.state = "LANDING"

    def loop(self):
        dt = 1.0 / RATE
        cf = self.d.cf
        while self.running:
            with self.lock:
                state, c = self.state, dict(self.cmd)
                x, y, z = self.d.real
                if state == "LANDED":
                    cf.commander.send_stop_setpoint()
                else:
                    tx, ty, tz = self.target
                    if state == "TAKEOFF":
                        tz = min(TAKEOFF_Z, tz + 0.4 * dt)
                        if tz >= TAKEOFF_Z and z > TAKEOFF_Z - 0.12:
                            self.state = "FLYING"
                    elif state == "LANDING":
                        tz = max(0.0, tz - 0.35 * dt)
                        if tz <= 0.0 and z < 0.1:
                            self._stop_count += 1
                    else:
                        v = self.vel                       # ramp towards the wanted speeds (inertia)
                        for k, a in (("fwd", ACCEL[0]), ("left", ACCEL[0]), ("up", ACCEL[1]), ("turn", ACCEL[2])):
                            v[k] += max(-a * dt, min(a * dt, c[k] - v[k]))
                        yaw = math.radians(self.yaw_t)
                        tx += (v["fwd"] * math.cos(yaw) - v["left"] * math.sin(yaw)) * dt
                        ty += (v["fwd"] * math.sin(yaw) + v["left"] * math.cos(yaw)) * dt
                        tz = min(MAX_Z, max(MIN_Z, tz + v["up"] * dt))
                        self.yaw_t += v["turn"] * dt
                    if state != "FLYING":
                        self.vel = dict.fromkeys(self.vel, 0.0)
                    # leash: the target may lead the drone a little more when flying fast,
                    # but never runs away (e.g. when pushing against a wall)
                    leash = LEASH + 0.6 * math.hypot(self.vel["fwd"], self.vel["left"])
                    dx, dy = tx - x, ty - y
                    dist = math.hypot(dx, dy)
                    if dist > leash:
                        tx, ty = x + dx / dist * leash, y + dy / dist * leash
                    tz = min(tz, z + LEASH + 0.5 * abs(self.vel["up"])) if state != "LANDING" else tz
                    self.target = [tx, ty, tz]
                    self.d.pos, self.d.landed = (tx, ty, tz), False
                    if self._stop_count:                     # touched down: motors off safely
                        cf.commander.send_stop_setpoint()
                        if self._stop_count > 3 * RATE:
                            self.state, self._stop_count, self.d.landed = "LANDED", 0, True
                    else:
                        cf.commander.send_position_setpoint(tx, ty, tz, self.yaw_t)
            time.sleep(dt)


# ---------------- pilot display ----------------
def draw_minimap(pg, parts, decals, results, path, pos, yaw, target=None):
    surf = pg.Surface((MAP_W, MAP_H), pg.SRCALPHA)
    surf.fill((20, 24, 28, 210))
    sc = min((MAP_W - 10) / (MAP_X[1] - MAP_X[0]), (MAP_H - 10) / (MAP_Y[1] - MAP_Y[0]))
    ox = (MAP_W - sc * (MAP_X[1] - MAP_X[0])) / 2          # same scale in x and y, centred

    def to_px(x, y):
        return int(ox + (x - MAP_X[0]) * sc), int(MAP_H - 5 - (y - MAP_Y[0]) * sc)

    colours = {"grass": (70, 95, 50), "gravel": (110, 100, 80), "weathered": (120, 120, 120),
               "concrete": (150, 150, 150), "asphalt": (60, 60, 65), "steel": (140, 90, 60)}
    for mat in ("grass", "asphalt", "weathered", "concrete", "steel"):
        for p in parts:
            if p["mat"] != mat or p["name"] in ("ground", "deck_slab") or p["name"].startswith("road"):
                continue
            (x0, y0, _), (x1, y1, _) = p["lo"], p["hi"]
            a, b = to_px(x0, y1), to_px(x1, y0)
            pg.draw.rect(surf, colours[mat], (a[0], a[1], max(1, b[0] - a[0]), max(1, b[1] - a[1])))
    if len(path) > 1:
        pg.draw.lines(surf, (80, 170, 255), False, [to_px(px, py) for px, py in path[-1500:]], 2)
    for dc in decals:
        r = results.get(dc["id"])
        col = (150, 150, 150) if r is None else ((60, 220, 90) if r == "correct" else (255, 170, 40))
        pg.draw.circle(surf, col, to_px(dc["centre"][0], dc["centre"][1]), 4)
    if target is not None:
        tx, ty = to_px(target[0], target[1])
        pg.draw.line(surf, (255, 80, 255), to_px(pos[0], pos[1]), (tx, ty), 1)
        pg.draw.circle(surf, (255, 80, 255), (tx, ty), 6, 2)
    px, py = to_px(pos[0], pos[1])
    a = math.radians(yaw)
    tip = (px + 9 * math.cos(a), py - 9 * math.sin(a))
    l = (px + 6 * math.cos(a + 2.5), py - 6 * math.sin(a + 2.5))
    r_ = (px + 6 * math.cos(a - 2.5), py - 6 * math.sin(a - 2.5))
    pg.draw.polygon(surf, (255, 60, 60), [tip, l, r_])
    return surf


HELP = ["PAD  left stick: climb + turn   right stick: fly",
        "PAD  A take off  B land  X precision  Y photo  MENU finish",
        "KEYS T take off  L land  W/S/A/D fly  arrows climb/turn",
        "KEYS TAB precision  P photo  H help  ESC land + finish",
        "KEYS N next stop  B previous stop  R route guide on/off"]


def shape(x):
    """Stick value -1..1 -> dead-zone + expo curve (fine control near the centre)."""
    if abs(x) < DEADZONE:
        return 0.0
    x = math.copysign((abs(x) - DEADZONE) / (1 - DEADZONE), x)
    return (1 - EXPO) * x + EXPO * x ** 3


class Pad:
    """Xbox (or any) controller. Uses SDL's GameController mapping when available, so the
    sticks/buttons are the same for USB and Bluetooth pads; falls back to raw joystick."""
    ACTIONS = {"a": "TAKEOFF", "b": "LANDING", "x": "PRECISION", "y": "SNAPSHOT",
               "back": "HELP", "start": "FINISH"}

    def __init__(self, pg):
        self.pg, self.kind, self.dev, self.name = pg, None, None, "keyboard only"
        try:
            from pygame._sdl2 import controller as gc
            gc.init()
            for i in range(gc.get_count()):
                if gc.is_controller(i):
                    self.dev, self.kind = gc.Controller(i), "gc"
                    self.name = gc.name_forindex(i) or "game controller"
                    break
        except Exception:
            self.dev = None
        if self.dev is None:
            pg.joystick.init()
            if pg.joystick.get_count():
                self.dev, self.kind = pg.joystick.Joystick(0), "js"
                self.dev.init()
                self.name = self.dev.get_name()
        print("Input:", self.name)

    def sticks(self):
        """(fwd, left, up, turn) each -1..1, Mode 2."""
        pg = self.pg
        if self.kind == "gc":
            g = lambda ax: self.dev.get_axis(ax) / 32767.0
            lx, ly = g(pg.CONTROLLER_AXIS_LEFTX), g(pg.CONTROLLER_AXIS_LEFTY)
            rx, ry = g(pg.CONTROLLER_AXIS_RIGHTX), g(pg.CONTROLLER_AXIS_RIGHTY)
        elif self.kind == "js":
            n = self.dev.get_numaxes()
            g = lambda i: self.dev.get_axis(i) if i < n else 0.0
            lx, ly = g(0), g(1)
            rx, ry = (g(3), g(4)) if n >= 5 else (g(2), g(3))     # xpad: 2 = left trigger
        else:
            return 0.0, 0.0, 0.0, 0.0
        return shape(-ry), shape(-rx), shape(-ly), shape(-lx)

    def action(self, ev):
        pg = self.pg
        if self.kind == "gc" and ev.type == pg.CONTROLLERBUTTONDOWN:
            names = {pg.CONTROLLER_BUTTON_A: "a", pg.CONTROLLER_BUTTON_B: "b", pg.CONTROLLER_BUTTON_X: "x",
                     pg.CONTROLLER_BUTTON_Y: "y", pg.CONTROLLER_BUTTON_BACK: "back",
                     pg.CONTROLLER_BUTTON_START: "start"}
            return self.ACTIONS.get(names.get(ev.button))
        if self.kind == "js" and ev.type == pg.JOYBUTTONDOWN:
            names = {0: "a", 1: "b", 2: "x", 3: "y", 6: "back", 7: "start"}
            return self.ACTIONS.get(names.get(ev.button))
        return None


def main():
    import os
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")   # pad works even if Gazebo has focus
    import pygame as pg
    from ultralytics import YOLO
    from cflib.crazyflie.log import LogConfig
    sys.path.insert(0, str(HOME / "drone_ws/scripts"))
    from flight_utils import Drone
    from gz_camera import GazeboCamera

    decals, parts, boxes = load_world()
    from route import compute_stops
    route = compute_stops(decals, boxes)
    n_stops = sum(st["kind"] == "stop" for st in route)
    route_i, route_on, hold_t, route_t, route_log, guide = 0, True, None, None, [], None
    out = HOME / time.strftime("drone_ws/logs/pilot_%Y%m%d_%H%M%S")
    (out / "findings").mkdir(parents=True, exist_ok=True)
    (out / "unmatched").mkdir(exist_ok=True)
    (out / "false_alarms").mkdir(exist_ok=True)

    print("Loading model ...")
    model = YOLO(str(MODEL))
    names = {i: n.lower() for i, n in model.names.items()}
    model.predict(np.zeros((IMG_H, IMG_W, 3), np.uint8), device=0, verbose=False)   # warm-up
    cam = GazeboCamera()
    if LENS is not None:                                       # real A11T fisheye lens + dark corners
        cam = LENS.wrap(cam)
    if CAM_PROFILE.get("analog"):                              # real A11T analog video look
        from analog_cam import AnalogCamera
        cam = AnalogCamera(cam, CAM_PROFILE["analog"])
    print(f"Camera: {CAM_PROFILE['name']}  lens {math.degrees(HFOV):.0f} deg  "
          f"analog effect: {CAM_PROFILE.get('analog') or 'off'}")

    evidence = defaultdict(Counter)        # decal id -> class -> confidence sum
    frames_on = Counter()                  # decal id -> frames with a detection on it
    first_seen = {}                        # decal id -> time found
    unmatched = 0
    unmatched_log, last_sample = [], 0.0   # detections that hit no defect patch
    alarm_tracks, false_alarms = [], []    # strong off-patch detections followed over frames
    fly_time, last_t = 0.0, time.time()    # seconds actually flying
    path = []

    with Drone(name="pilot") as d:
        fc = FlightControl(d)
        ylog = LogConfig(name="yaw", period_in_ms=50)
        ylog.add_variable("stateEstimate.yaw")
        d.cf.log.add_config(ylog)
        ylog.data_received_cb.add_callback(lambda ts, data, conf: setattr(fc, "yaw", data["stateEstimate.yaw"]))
        ylog.start()
        threading.Thread(target=fc.loop, daemon=True).start()

        pg.init()
        pad = Pad(pg)
        screen = pg.display.set_mode((int(IMG_W * SCALE), int(IMG_H * SCALE)))
        pg.display.set_caption("Bridge inspection pilot - click here, H = help")
        font, big = pg.font.SysFont("dejavusans", 17), pg.font.SysFont("dejavusans", 22, bold=True)
        writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                 (int(IMG_W * SCALE), int(IMG_H * SCALE)))
        mode, show_help, quitting = "normal", True, False
        t0, n_frames, fps, last_rec = time.time(), 0, 0.0, 0.0
        message, message_t = "Press T to take off", time.time()

        keymap = {pg.K_t: "TAKEOFF", pg.K_l: "LANDING", pg.K_TAB: "PRECISION", pg.K_h: "HELP",
                  pg.K_p: "SNAPSHOT", pg.K_ESCAPE: "FINISH", pg.K_n: "NEXT", pg.K_b: "BACK",
                  pg.K_r: "ROUTE"}
        while True:
            for ev in pg.event.get():
                act = "FINISH" if ev.type == pg.QUIT else pad.action(ev)
                if ev.type == pg.KEYDOWN:
                    act = keymap.get(ev.key)
                if act == "TAKEOFF":
                    fc.request("TAKEOFF")
                    message, message_t = "Taking off ...", time.time()
                elif act == "LANDING":
                    fc.request("LANDING")
                    message, message_t = "Landing ...", time.time()
                elif act == "PRECISION":
                    mode = "precision" if mode == "normal" else "normal"
                    message, message_t = f"{mode.upper()} mode", time.time()
                elif act == "HELP":
                    show_help = not show_help
                elif act == "SNAPSHOT":
                    pg.image.save(screen, str(out / f"snapshot_{time.strftime('%H%M%S')}.png"))
                    message, message_t = "Snapshot saved", time.time()
                elif act in ("NEXT", "BACK"):
                    step = 1 if act == "NEXT" else -1
                    route_i, hold_t = max(0, min(len(route), route_i + step)), None
                    while 0 < route_i < len(route) and route[route_i]["kind"] == "via" and step < 0:
                        route_i -= 1
                elif act == "ROUTE":
                    route_on = not route_on
                    message, message_t = f"Route guide {'ON' if route_on else 'OFF'}", time.time()
                elif act == "FINISH":
                    quitting = True
                    fc.request("LANDING")
                    message, message_t = "Landing and finishing ...", time.time()

            keys = pg.key.get_pressed()
            v_xy, v_z, v_yaw = SPEEDS[mode]
            s_fwd, s_left, s_up, s_turn = pad.sticks()
            clip = lambda v: max(-1.0, min(1.0, v))
            with fc.lock:
                fc.cmd["fwd"] = v_xy * clip(s_fwd + keys[pg.K_w] - keys[pg.K_s])
                fc.cmd["left"] = v_xy * clip(s_left + keys[pg.K_a] - keys[pg.K_d])
                fc.cmd["up"] = v_z * clip(s_up + keys[pg.K_UP] - keys[pg.K_DOWN])
                fc.cmd["turn"] = v_yaw * clip(s_turn + keys[pg.K_LEFT] - keys[pg.K_RIGHT])
                state = fc.state
            if quitting and state == "LANDED":
                break

            ok, frame = cam.read(timeout=0.5)
            if not ok:
                if not cam.isOpened():
                    print("Camera stopped - is simfull running?")
                    break
                continue
            pos, yaw = d.real, fc.yaw
            if time.time() - last_rec > 0.2:
                path.append((pos[0], pos[1]))
                last_rec = time.time()

            res = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]
            shown = frame.copy()
            now = time.time()
            if state == "FLYING":
                fly_time += min(0.5, now - last_t)
            last_t = now
            new_tracks = []
            for (x1, y1, x2, y2), conf, cls in STAB.update(res):
                name = names[int(cls)]
                hit = None
                if state == "FLYING":
                    o, dvec = pixel_ray((x1 + x2) / 2, (y1 + y2) / 2, pos, yaw)
                    hit = find_decal(o, dvec, decals, boxes)
                    if hit is None:
                        unmatched += 1
                        unmatched_log.append([round(time.time() - t0, 2), name, round(conf, 3),
                                              *(round(v, 2) for v in pos), round(yaw, 1),
                                              int(x1), int(y1), int(x2), int(y2)])
                        if time.time() - last_sample > 3 and len(list((out / "unmatched").glob("*.jpg"))) < 80:
                            last_sample = time.time()
                            crop = frame.copy()
                            cv2.rectangle(crop, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
                            cv2.putText(crop, f"{name} {conf:.2f}", (int(x1) + 3, max(15, int(y1) - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                            cv2.imwrite(str(out / "unmatched" / f"{time.time() - t0:07.1f}_{name}.jpg"), crop)
                        if conf >= ALARM_CONF:
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            prev = next((t for t in alarm_tracks if t["cls"] == name and
                                         math.hypot(t["cx"] - cx, t["cy"] - cy) < ALARM_PX), None)
                            tr = {"cls": name, "cx": cx, "cy": cy, "n": prev["n"] + 1 if prev else 1,
                                  "counted": prev["counted"] if prev else False}
                            if prev:
                                alarm_tracks.remove(prev)
                            if tr["n"] >= FOUND_FRAMES and not tr["counted"]:
                                tr["counted"] = True
                                false_alarms.append([round(now - t0, 1), name, round(conf, 3),
                                                     *(round(v, 2) for v in pos), round(yaw, 1)])
                                img = frame.copy()
                                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(img, f"{name} {conf:.2f}", (int(x1) + 3, max(15, int(y1) - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.imwrite(str(out / "false_alarms" / f"{len(false_alarms):03d}_{name}.jpg"), img)
                            new_tracks.append(tr)
                    else:
                        evidence[hit["id"]][name] += conf * (1.0 if conf >= CLASS_CONF else WEAK_VOTE)
                        frames_on[hit["id"]] += 1
                weak = hit is None and conf < ALARM_CONF
                col = (60, 220, 90) if hit else (170, 170, 170) if weak else (0, 200, 255)
                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), col, 1 if weak else 2)
                label = f"{name} {conf:.2f}" + (f"  [{hit['id']}]" if hit else "")
                cv2.putText(shown, label, (int(x1) + 3, max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 2)
                if hit and frames_on[hit["id"]] == FOUND_FRAMES and hit["id"] not in first_seen:
                    first_seen[hit["id"]] = time.time() - t0
                    cv2.imwrite(str(out / "findings" / f"{hit['id']}_{name}.jpg"), shown)
                    message, message_t = f"FOUND {hit['id']}: {name} ({hit['zone']})", time.time()

            alarm_tracks = new_tracks

            # ---- demo route guide ----
            guide = None
            if route_on and route_i < len(route):
                st = route[route_i]
                if route_t is None and state == "FLYING":
                    route_t = now
                dx, dy, dz = st["pos"] - np.asarray(pos, float)
                ya = math.radians(yaw)
                fwd, left = dx * math.cos(ya) + dy * math.sin(ya), -dx * math.sin(ya) + dy * math.cos(ya)
                dyaw = None if st["yaw"] is None else (st["yaw"] - yaw + 180) % 360 - 180
                at = False
                if st["kind"] == "via":
                    if math.hypot(dx, dy) < 0.4 and abs(dz) < 0.3:
                        route_i += 1
                    p_ = project(st["pos"], pos, yaw)
                    if p_:
                        cv2.circle(shown, p_, 12, (255, 80, 255), 2)
                        cv2.putText(shown, "waypoint", (p_[0] + 15, p_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 80, 255), 2)
                else:
                    at = math.hypot(dx, dy) < 0.3 and abs(dz) < 0.2 and abs(dyaw) < 15
                    hold_t = (hold_t or now) if at else None
                    held = now - hold_t if hold_t else 0.0
                    found = st["id"] in first_seen
                    dc = st["decal"]
                    corners = [project(dc["_c"] + a * dc["width"] / 2 * dc["_h"] +
                                       b * dc["height"] / 2 * dc["_v"], pos, yaw)
                               for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                    if all(corners):
                        cv2.polylines(shown, [np.array(corners, np.int32)], True, (255, 80, 255), 2)
                        top = min(corners, key=lambda q: q[1])
                        cv2.putText(shown, f"NEXT {st['id']}", (top[0], max(15, top[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 255), 2)
                    if (found and held > HOLD_FOUND) or held > HOLD_MAX:
                        route_log.append((st["id"], found, round(now - (route_t or now), 1)))
                        message = f"Stop {st['n']} {st['id']}: " + ("FOUND" if found else "not detected - next")
                        message_t, route_i, hold_t = now, route_i + 1, None
                        if route_i == len(route):
                            message = f"ROUTE COMPLETE in {(now - (route_t or now)) / 60:.1f} min - ESC = land + report"
                guide = (st, fwd, left, dz, dyaw, at)
            results = {}
            for did in first_seen:
                pred = evidence[did].most_common(1)[0][0]
                dc = next(x for x in decals if x["id"] == did)
                results[did] = "correct" if pred == dc["class"] else "other"

            # ---- compose the pilot view ----
            n_frames += 1
            if n_frames % 10 == 0:
                fps = n_frames / (time.time() - t0)
            rgb = cv2.cvtColor(cv2.resize(shown, (int(IMG_W * SCALE), int(IMG_H * SCALE))), cv2.COLOR_BGR2RGB)
            screen.blit(pg.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
            panel = pg.Surface((330, 156), pg.SRCALPHA)
            panel.fill((0, 0, 0, 150))
            screen.blit(panel, (8, 8))
            st_col = {"FLYING": (90, 230, 110), "LANDED": (200, 200, 200)}.get(state, (255, 200, 60))
            spd = math.hypot(fc.vel["fwd"], fc.vel["left"])
            lines = [(f"{state}   {mode.upper()} mode", st_col),
                     (f"height {pos[2]:4.2f} m   heading {yaw:+5.0f} deg", (255, 255, 255)),
                     (f"speed {spd:3.1f} m/s   {fps:4.1f} fps   {'PAD' if pad.dev else 'KEYS'}", (255, 255, 255)),
                     (f"defects found {len(first_seen)}/{len(decals)}", (90, 230, 110)),
                     (f"false alarms {len(false_alarms)}  ({len(false_alarms) / max(fly_time / 60, 1e-6):.1f}/min)"
                      if fly_time > 5 else f"false alarms {len(false_alarms)}", (255, 140, 90))]
            for i, (txt, col) in enumerate(lines):
                screen.blit((big if i in (0, 3) else font).render(txt, True, col), (16, 14 + i * 28))
            screen.blit(draw_minimap(pg, parts, decals, results, path, pos, yaw,
                                     guide[0]["pos"] if guide else None),
                        (screen.get_width() - MAP_W - 8, 8))
            if show_help:
                hp = pg.Surface((470, 22 * len(HELP) + 12), pg.SRCALPHA)
                hp.fill((0, 0, 0, 150))
                for i, txt in enumerate(HELP):
                    hp.blit(font.render(txt, True, (255, 255, 255)), (8, 6 + 22 * i))
                screen.blit(hp, (8, screen.get_height() - hp.get_height() - (100 if guide else 8)))
            if guide:
                st, fwd, left, dz, dyaw, at = guide
                if st["kind"] == "via":
                    head = f"ROUTE  waypoint on the way to stop {st['n']}/{n_stops}  {st['id']}"
                else:
                    head = f"ROUTE  stop {st['n']}/{n_stops}  {st['id']}  {st['zone']}"
                if at and hold_t:
                    todo, col2 = f"HOLD STILL - scanning  {now - hold_t:3.1f} s", (90, 230, 110)
                else:
                    todo = []
                    if abs(fwd) > 0.1:
                        todo.append(f"{'forward' if fwd > 0 else 'back'} {abs(fwd):.1f} m")
                    if abs(left) > 0.1:
                        todo.append(f"{'left' if left > 0 else 'right'} {abs(left):.1f} m")
                    if abs(dz) > 0.1:
                        todo.append(f"{'up' if dz > 0 else 'down'} {abs(dz):.1f} m")
                    if dyaw is not None and abs(dyaw) > 8:
                        todo.append(f"turn {'left' if dyaw > 0 else 'right'} {abs(dyaw):.0f} deg")
                    todo, col2 = "    ".join(todo) or "almost there - fine-tune", (255, 255, 255)
                rp = pg.Surface((screen.get_width() - 16, 88), pg.SRCALPHA)
                rp.fill((0, 0, 0, 165))
                rp.blit(font.render(head, True, (255, 80, 255)), (10, 6))
                rp.blit(big.render(todo, True, col2), (10, 31))
                rp.blit(font.render("how: " + st["tip"], True, (200, 200, 200)), (10, 62))
                screen.blit(rp, (8, screen.get_height() - 96))
            if time.time() - message_t < 3:
                m = big.render(message, True, (255, 255, 120))
                screen.blit(m, ((screen.get_width() - m.get_width()) // 2, 150))
            if state == "FLYING" and pos[2] < 0.06:
                w = big.render("Drone on the ground / not responding - press L, restart simfull",
                               True, (255, 80, 80))
                screen.blit(w, ((screen.get_width() - w.get_width()) // 2, 190))
            pg.display.flip()
            writer.write(cv2.cvtColor(pg.surfarray.array3d(screen).swapaxes(0, 1), cv2.COLOR_RGB2BGR))

        fc.running = False
        ylog.stop()
        writer.release()
        cam.release()
        pg.quit()

    with open(out / "unmatched.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "class", "conf", "x", "y", "z", "yaw", "x1", "y1", "x2", "y2"])
        w.writerows(unmatched_log)
    by_cls = Counter(r[1] for r in unmatched_log)
    strong = sum(1 for r in unmatched_log if r[2] >= 0.6)
    mins = max(1e-6, (time.time() - t0) / 60)
    print("\nDetections on plain concrete / no defect patch:")
    for c, k in by_cls.most_common():
        print(f"   {c:17s}{k:7d}")
    print(f"   per minute: {len(unmatched_log) / mins:.0f}   with confidence >= 0.6: {strong}"
          f"   samples: {out / 'unmatched'}")
    with open(out / "false_alarms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "class", "conf", "x", "y", "z", "yaw"])
        w.writerows(false_alarms)
    write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, time.time() - t0,
                 false_alarms, fly_time)
    if route_log:
        print(f"\nDEMO ROUTE: {len(route_log)}/{n_stops} stops done, {sum(f for _, f, _ in route_log)} "
              f"detected at the stop, {route_log[-1][2] / 60:.1f} min")
        with open(out / "route_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "detected_at_stop", "t_s"])
            w.writerows(route_log)


# ---------------- report ----------------
def write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, duration,
                 false_alarms=(), fly_time=0.0):
    rows = []
    for dc in decals:
        did = dc["id"]
        pred = evidence[did].most_common(1)[0][0] if did in first_seen else "none"
        if pred == "none":
            result = "MISSED"
        elif pred == dc["class"]:
            result = "correct"
        elif pred in dc.get("also_present", []):
            result = "partly correct"
        else:
            result = "WRONG CLASS"
        rows.append({"id": did, "zone": dc["zone"], "truth": dc["class"], "predicted": pred,
                     "frames": frames_on[did], "found_after_s": round(first_seen.get(did, -1), 1),
                     "result": result})
    with open(out / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    found = sum(r["result"] != "MISSED" for r in rows)
    ok = sum(r["result"] == "correct" for r in rows)
    part = sum(r["result"] == "partly correct" for r in rows)

    print("\n================ FREE-FLIGHT INSPECTION REPORT ================")
    print(f"{'id':5s}{'zone':40s}{'truth':17s}{'predicted':17s}{'frames':>7s}  result")
    for r in rows:
        print(f"{r['id']:5s}{r['zone']:40s}{r['truth']:17s}{r['predicted']:17s}{r['frames']:7d}  {r['result']}")
    print(f"\nflight time {duration / 60:.1f} min")
    print(f"defects found:        {found}/{n}")
    print(f"correct class:        {ok}/{n}   (+{part} partly correct)")
    print(f"all off-patch detections (any conf, frames): {unmatched}")
    fa_rate = len(false_alarms) / max(fly_time / 60, 1e-6)
    print(f"FALSE ALARMS (conf >= {ALARM_CONF}, {FOUND_FRAMES}+ frames): {len(false_alarms)}"
          f"   = {fa_rate:.1f} per flying minute ({fly_time / 60:.1f} min flying)")
    for c, k in Counter(a[1] for a in false_alarms).most_common():
        print(f"   {c:17s}{k:5d}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colours = {"concrete": "0.7", "weathered": "0.55", "asphalt": "0.3", "grass": "#b5c99a",
               "gravel": "#ddd2b5", "steel": "#8a5a3a"}
    fig, ax = plt.subplots(figsize=(9, 11))
    for mat in ("grass", "gravel", "asphalt", "weathered", "concrete", "steel"):
        for p in parts:
            if p["mat"] == mat and p["name"] not in ("ground", "deck_slab") and not p["name"].startswith("road"):
                (x0, y0, _), (x1, y1, _) = p["lo"], p["hi"]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=colours[mat], ec="k", lw=0.3))
    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path], "-", color="royalblue", lw=1.5, label="flight path")
    edge = {"correct": "green", "partly correct": "orange", "WRONG CLASS": "red", "MISSED": "0.4"}
    for dc, r in zip(decals, rows):
        x, y, _ = dc["centre"]
        ax.plot(x, y, "o", ms=9, color=edge[r["result"]])
        ax.annotate(f"{dc['id']} {r['predicted'] if r['predicted'] != 'none' else ''}", (x, y),
                    xytext=(5, 3), textcoords="offset points", fontsize=7.5)
    for res, col in edge.items():
        ax.plot([], [], "o", color=col, label=res)
    ax.set_xlim(*MAP_X)
    ax.set_ylim(*MAP_Y)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Free-flight {WORLD.get('kind', 'bridge')} inspection ({WORLD_NAME}): {found}/{n} defects found, {ok}/{n} correct class "
                 f"(+{part} partly)\nfalse alarms {len(false_alarms)} ({fa_rate:.1f}/min, "
                 f"conf >= {ALARM_CONF})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "report.png", dpi=150)
    print(f"\nSaved in {out}:  report.png  findings.csv  findings/  false_alarms/  video.mp4")


if __name__ == "__main__":
    main()
