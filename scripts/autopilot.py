#!/usr/bin/env python3
"""
autopilot.py - fully autonomous bridge inspection: the drone flies the 20-stop demo route by
itself (take off -> every defect -> back home -> land) while YOLO inspects live.

  start the world first:   simfull        then:   autopilot

  T      start the mission (take off)
  SPACE  pause / continue
  ESC    abort: fly home safely and land       L  land right here (emergency)
  H      help on/off                            (the report is written at the end)

How it flies
  * planner.py plans collision-free 3D paths between the stops (A* on a 10 cm grid of the
    bridge, 14 cm safety margin, cutting corners where the straight line is free).
  * Long legs are flown facing forward (camera looks where the drone goes); slower near walls.
  * At each stop: turn to face the defect, hold still, scan; next stop when the defect is
    found + 2.5 s (or after 6 s).
  * Same detection, 3D matching, false-alarm rule and report as pilot.py.
  Output: logs/auto_<date>/  report.png  findings.csv  mission_log.csv  findings/  video.mp4
"""
import csv
import math
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
sys.path.insert(0, str(HOME / "drone_ws/scripts"))
import pilot                                          # world, camera geometry, report
from pilot import (CONF, IMG_W, IMG_H, FOCAL, CAM_OFFSET, MODEL, FOUND_FRAMES, ALARM_CONF, ALARM_PX,
                   SCALE, RATE, MAP_W, MAP_H, MAP_X, MAP_Y, load_world, pixel_ray, find_decal, write_report)
from route import compute_stops
from planner import Planner
# --- path holding (patch_wind) ---
WALLS = None                                   # (lo, hi) arrays of every obstacle box

# ---------------- mission settings ----------------
TAKEOFF_Z = 0.5
CRUISE, SLOW, CLIMB = 0.45, 0.22, 0.30       # m/s: open space, close to walls, vertical
SLOW_CLEAR = 0.35                            # slow down when the path is closer than this to a wall
YAW_RATE = 60.0                              # deg/s
LEAD = 0.30                                  # the setpoint never runs more than this ahead
FACE_TRAVEL = 0.8                            # legs longer than this (horizontal) are flown nose-first
ARRIVE_POS, ARRIVE_YAW = 0.12, 8.0           # 'at the stop' tolerance (m, deg)
HOLD_FOUND = 2.5                             # seconds to keep scanning after FOUND (class votes)
VIEW_TIME = 2.0                              # seconds per view when nothing is found yet
CLOSER = tuple(pilot.WORLD.get("closer", (0.6, 0.45, 0.35)))   # extra views closer (m from the patch)
# --- active world (simworld) ---
SWEEP = 15.0                                 # look left / right by this much (deg)
SURVEY = os.environ.get("SURVEY") == "1"      # SURVEY=1 autopilot: full scans + save clean frames
PASSING_WEIGHT = 0.25                        # class vote of frames NOT taken during the scan


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def project(P, pos, yaw_deg):
    """World point -> image pixel, or None if behind the camera."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    b = pilot.CAM_ROT.T @ (rot.T @ (np.asarray(P, float) - np.asarray(pos, float) - rot @ CAM_OFFSET))
    if b[0] < 0.1:
        return None
    u, v = pilot.norm_to_pix(-b[1] / b[0], -b[2] / b[0])
    return (int(max(-5000, min(5000, u))),
            int(max(-5000, min(5000, v))))


class Abort(Exception):
    pass


# ---------------- the mission (runs in background threads) ----------------
class Mission:
    def __init__(self, drone, route, planner, first_seen):
        self.d, self.route, self.planner, self.first_seen = drone, route, planner, first_seen
        self.sp = None                       # setpoint [x, y, z, yaw] or None = motors off
        self.yaw = 0.0                       # measured heading (log callback)
        self.phase, self.item, self.hold_t = "READY - press T", None, None
        self.paused, self.abort, self.done, self.flying = False, None, False, False
        self.idx = 0                         # index in route of the current target
        self.plan_pts = []                   # whole planned path (for the mini-map)
        self.log = []                        # (id, detected at stop, t)
        self.t0 = None
        self.scan_id = None                  # defect being scanned right now
        self.best = {}                       # scan: best confidence on the target / elsewhere
        self.running = True
        self.track = []                      # (t, tracking error m, wall distance m) while flying

    # -- 20 Hz setpoint stream --
    def stream(self):
        cf = self.d.cf
        while self.running:
            sp = self.sp
            if sp is None:
                cf.commander.send_stop_setpoint()
            else:
                cf.commander.send_position_setpoint(*sp)
                self.d.pos, self.d.landed = tuple(sp[:3]), False
            time.sleep(1.0 / RATE)

    # -- helpers --
    def pos(self):
        return np.asarray(self.d.real, float)

    def tick(self):
        if (self.abort == "land" and self.phase != "LANDING") or \
                (self.abort == "home" and self.phase not in ("RETURNING HOME", "LANDING")):
            raise Abort()
        if self.flying and self.sp is not None:
            p = self.pos()
            a, b = getattr(self, "leg", (p, p))         # distance to the planned leg a -> b
            ab = b - a
            t_ = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
            err = float(np.linalg.norm(p - (a + t_ * ab)))
            wall = 9.9
            if WALLS is not None:
                lo, hi = WALLS
                wall = float(np.sqrt((np.maximum(np.maximum(lo - p, p - hi), 0) ** 2).sum(1)).min())
            self.track.append((time.time(), err, wall, *p, self.phase))
        time.sleep(1.0 / RATE)

    def wait_paused(self):
        while self.paused and not self.abort:
            time.sleep(0.05)

    def turn_to(self, yaw_goal):
        goal = self.sp[3] + wrap(yaw_goal - self.sp[3])
        while abs(goal - self.sp[3]) > 0.5:
            self.wait_paused()
            step = YAW_RATE / RATE
            self.sp[3] += max(-step, min(step, goal - self.sp[3]))
            self.tick()
        t = time.time()
        while abs(wrap(self.yaw - yaw_goal)) > ARRIVE_YAW and time.time() - t < 3:
            self.tick()

    def fly_legs(self, legs, final_tol=ARRIVE_POS):
        for k, (a, b) in enumerate(zip(legs, legs[1:])):
            v = b - a
            L = float(np.linalg.norm(v))
            if L < 1e-3:
                continue
            if math.hypot(v[0], v[1]) > FACE_TRAVEL and abs(v[2]) < 0.6 * L:
                self.turn_to(math.degrees(math.atan2(v[1], v[0])))
            self.leg = (np.asarray(a, float), np.asarray(b, float))   # --- path deviation (patch_wind2) ---
            s = 0.0
            while s < L:
                self.wait_paused()
                carrot = a + v * (s / L)
                if np.linalg.norm(carrot - self.pos()) < LEAD:      # leash: wait for the drone
                    speed = SLOW if self.planner.clear_at(carrot) < SLOW_CLEAR else CRUISE
                    if abs(v[2]) > 1e-3:
                        speed = min(speed, CLIMB * L / abs(v[2]))
                    s = min(L, s + speed / RATE)
                    carrot = a + v * (s / L)
                self.sp[:3] = list(carrot)
                self.tick()
            tol = final_tol if k == len(legs) - 2 else 0.18
            t = time.time()
            while np.linalg.norm(b - self.pos()) > tol and time.time() - t < 4:
                self.tick()

    def takeoff(self):
        self.phase = "TAKING OFF"
        x, y, _ = self.pos()
        self.sp = [x, y, 0.0, self.yaw]
        while self.sp[2] < TAKEOFF_Z:
            self.sp[2] = min(TAKEOFF_Z, self.sp[2] + 0.4 / RATE)
            self.tick()
        t = time.time()
        while self.pos()[2] < TAKEOFF_Z - 0.1 and time.time() - t < 5:
            self.tick()
        if self.pos()[2] < TAKEOFF_Z - 0.2:
            raise RuntimeError("drone did not take off - restart simfull (simreset) and try again")

    def land(self):
        self.phase, self.flying = "LANDING", False
        while self.sp[2] > 0.0:
            self.sp[2] = max(0.0, self.sp[2] - 0.35 / RATE)
            time.sleep(1.0 / RATE)
        t = time.time()
        while self.pos()[2] > 0.08 and time.time() - t < 4:
            time.sleep(0.05)
        self.sp = None                       # stop setpoints = motors off
        time.sleep(1.0)
        self.d.landed = True
        self.phase = "LANDED"

    def scan(self, it):
        """Active inspection: look from the stop, then closer, then left/right, until FOUND."""
        c, n, yaw0 = it["decal"]["_c"], it["decal"]["_n"], it["yaw"]
        views, close = [(it["pos"], yaw0, "1st view")], it["pos"]
        for d in CLOSER:
            q = it["view_at"](d) if "view_at" in it else c + n * d
            if d < it["standoff"] - 0.08 and self.planner.free(q) and self.planner.line_free(close, q):
                views.append((q, yaw0, f"closer {d:.2f} m"))
                close = q
        views += [(close, yaw0 + SWEEP, "look left"), (close, yaw0 - SWEEP, "look right")]
        self.scan_id, self.best[it["id"]] = it["id"], {}
        t_found, k = None, 0
        for k, (q, yw, label) in enumerate(views):
            self.phase = f"SCANNING {it['id']} - {label}"
            if k:
                if np.linalg.norm(q - self.pos()) > 0.04:
                    self.fly_legs([self.pos(), q])
                self.turn_to(yw)
            self.hold_t = time.time()
            while True:
                self.wait_paused()
                if t_found is None and it["id"] in self.first_seen and not SURVEY:
                    t_found = time.time()
                if t_found and time.time() - t_found >= HOLD_FOUND:
                    break
                if t_found is None and time.time() - self.hold_t >= VIEW_TIME:
                    break
                self.tick()
            if t_found and time.time() - t_found >= HOLD_FOUND and (k >= 1 or len(views) == 1):
                break
        self.hold_t = None
        if np.linalg.norm(self.pos() - it["pos"]) > 0.05:      # back to the stop = back on the plan
            self.phase = f"BACKING OUT from {it['id']}"
            self.fly_legs([self.pos(), it["pos"]])
        self.scan_id = None
        return it["id"] in self.first_seen, k + 1

    # -- the mission --
    def run(self):
        home = None
        try:
            home = self.pos().copy()
            hover = np.array([home[0], home[1], TAKEOFF_Z])
            self.takeoff()
            self.flying, self.t0 = True, time.time()
            prev = hover
            for self.idx, it in enumerate(self.route):
                self.item = it
                self.phase = f"FLYING to {'waypoint' if it['kind'] == 'via' else it['id']}"
                legs = it["legs"]
                legs = [prev] + legs[1:] if np.linalg.norm(legs[0] - prev) < 0.05 else \
                    self.planner.plan(self.pos(), it["pos"])
                self.fly_legs(legs, ARRIVE_POS if it["kind"] == "stop" else 0.25)
                prev = it["pos"]
                if it["kind"] == "via":
                    continue
                self.phase = f"TURNING to {it['id']}"
                self.turn_to(it["yaw"])
                err_p = float(np.linalg.norm(self.pos() - it["pos"]))
                err_y = abs(wrap(self.yaw - it["yaw"]))
                found, n_views = self.scan(it)
                b = self.best.get(it["id"], {})
                self.log.append((it["id"], found, round(time.time() - self.t0, 1), round(err_p, 3),
                                 round(err_y, 1), n_views, round(b.get("on", 0), 2), round(b.get("off", 0), 2)))
            self.idx, self.item = len(self.route), None
            self.phase = "RETURNING HOME"
            self.fly_legs(self.planner.plan(self.pos(), hover), 0.15)
            self.land()
        except Abort:
            self.hold_t, self.item = None, None
            try:
                if self.abort == "home":
                    self.phase = "RETURNING HOME"
                    hover = np.array([home[0], home[1], TAKEOFF_Z]) if home is not None else self.pos()
                    self.fly_legs(self.planner.plan(self.pos(), hover), 0.15)
            except Abort:                    # L pressed while returning: land here
                pass
            self.land()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("!! mission error:", e)
            if self.sp is not None:
                self.land()
        finally:
            self.flying, self.done = False, True


def draw_minimap(pg, parts, decals, results, path, plan_pts, pos, yaw, target):
    surf = pg.Surface((MAP_W, MAP_H), pg.SRCALPHA)
    surf.fill((20, 24, 28, 210))
    sc = min((MAP_W - 10) / (MAP_X[1] - MAP_X[0]), (MAP_H - 10) / (MAP_Y[1] - MAP_Y[0]))
    ox = (MAP_W - sc * (MAP_X[1] - MAP_X[0])) / 2

    def to_px(x, y):
        return int(ox + (x - MAP_X[0]) * sc), int(MAP_H - 5 - (y - MAP_Y[0]) * sc)

    colours = {"grass": (70, 95, 50), "asphalt": (60, 60, 65), "weathered": (120, 120, 120), "concrete": (150, 150, 150),
               "steel": (140, 90, 60)}
    for mat in ("grass", "asphalt", "weathered", "concrete", "steel"):
        for p in parts:
            if p["mat"] != mat or p["name"] in ("ground", "deck_slab") or p["name"].startswith("road"):
                continue
            (x0, y0, _), (x1, y1, _) = p["lo"], p["hi"]
            a, b = to_px(x0, y1), to_px(x1, y0)
            pg.draw.rect(surf, colours[mat], (a[0], a[1], max(1, b[0] - a[0]), max(1, b[1] - a[1])))
    if len(plan_pts) > 1:
        pg.draw.lines(surf, (170, 70, 170), False, [to_px(p[0], p[1]) for p in plan_pts], 1)
    if len(path) > 1:
        pg.draw.lines(surf, (80, 170, 255), False, [to_px(px, py) for px, py in path[-1500:]], 2)
    for dc in decals:
        r = results.get(dc["id"])
        col = (150, 150, 150) if r is None else ((60, 220, 90) if r == "correct" else (255, 170, 40))
        pg.draw.circle(surf, col, to_px(dc["centre"][0], dc["centre"][1]), 4)
    if target is not None:
        pg.draw.circle(surf, (255, 80, 255), to_px(target[0], target[1]), 6, 2)
    px, py = to_px(pos[0], pos[1])
    a = math.radians(yaw)
    pg.draw.polygon(surf, (255, 60, 60), [(px + 9 * math.cos(a), py - 9 * math.sin(a)),
                                          (px + 6 * math.cos(a + 2.5), py - 6 * math.sin(a + 2.5)),
                                          (px + 6 * math.cos(a - 2.5), py - 6 * math.sin(a - 2.5))])
    return surf


HELP = "T start   SPACE pause/continue   ESC abort: fly home + land   L land here   H help"


def main():
    import pygame as pg
    from ultralytics import YOLO
    from cflib.crazyflie.log import LogConfig
    from flight_utils import Drone
    from gz_camera import GazeboCamera

    decals, parts, boxes = load_world()
    globals()["WALLS"] = (np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes]))
    route = compute_stops(decals, boxes)
    n_stops = sum(it["kind"] == "stop" for it in route)
    print("Planning collision-free paths ...")
    t = time.time()
    margin = pilot.WORLD.get("clear", {"calm": 0.14, "light": 0.20, "moderate": 0.24, "gusty": 0.26, "strong": 0.28}.get(pilot.WORLD.get("wind", "calm"), 0.14))   # --- wind margin (patch_margin) ---
    print(f"SAFETY MARGIN: {margin * 100:.0f} cm ({pilot.WORLD.get('wind', 'calm')} wind)")
    planner = Planner(boxes, pilot.WORLD.get("grid_lo"), pilot.WORLD.get("grid_hi"), margin)   # --- big worlds (patch_world2) ---
    prev, plan_pts, length = np.array([0.0, 0.0, TAKEOFF_Z]), [], 0.0
    for it in route + [{"pos": np.array([0.0, 0.0, TAKEOFF_Z])}]:
        it["legs"] = planner.plan(prev, it["pos"])
        plan_pts += [tuple(p) for p in it["legs"]]
        length += sum(np.linalg.norm(b - a) for a, b in zip(it["legs"], it["legs"][1:]))
        prev = it["pos"]
    print(f"  {length:.0f} m in {time.time() - t:.1f} s, about {length / 0.35 / 60 + n_stops * 6 / 60:.0f} min "
          f"of flying incl. scans")

    out = HOME / time.strftime("drone_ws/logs/auto_%Y%m%d_%H%M%S")
    (out / "findings").mkdir(parents=True, exist_ok=True)
    survey_dir, survey_rows = out / "survey", []
    if SURVEY:
        survey_dir.mkdir()
        print("SURVEY MODE: full scans at every stop, clean frames are saved")
    print("Loading model ...")
    model = YOLO(str(MODEL))
    names = {i: n.lower() for i, n in model.names.items()}
    model.predict(np.zeros((IMG_H, IMG_W, 3), np.uint8), device=0, verbose=False)
    cam = GazeboCamera()
    if pilot.LENS is not None:                                 # real A11T fisheye lens + dark corners
        cam = pilot.LENS.wrap(cam)
    if pilot.CAM_PROFILE.get("analog") and not SURVEY:         # real A11T analog video look
        from analog_cam import AnalogCamera
        cam = AnalogCamera(cam, pilot.CAM_PROFILE["analog"])
    print(f"Camera: {pilot.CAM_PROFILE['name']}  lens {math.degrees(pilot.HFOV):.0f} deg  "
          f"analog effect: {pilot.CAM_PROFILE.get('analog') or 'off'}")

    evidence, frames_on, first_seen = defaultdict(Counter), Counter(), {}
    unmatched, alarm_tracks, false_alarms = 0, [], []
    path, fly_time, last_t = [], 0.0, time.time()

    with Drone(name="autopilot") as d:
        m = Mission(d, route, planner, first_seen)
        ylog = LogConfig(name="yaw", period_in_ms=50)
        ylog.add_variable("stateEstimate.yaw")
        d.cf.log.add_config(ylog)
        ylog.data_received_cb.add_callback(lambda ts, data, conf: setattr(m, "yaw", data["stateEstimate.yaw"]))
        ylog.start()
        threading.Thread(target=m.stream, daemon=True).start()
        from wind_gust import Gusts                 # --- wind force (patch_wind3) ---
        gust = Gusts(pilot.WORLD, pilot.WORLD_NAME, m.pos).start()

        pg.init()
        screen = pg.display.set_mode((int(IMG_W * SCALE), int(IMG_H * SCALE)))
        pg.display.set_caption("Autonomous bridge inspection - click here, T = start")
        font, big = pg.font.SysFont("dejavusans", 17), pg.font.SysFont("dejavusans", 22, bold=True)
        writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                 (int(IMG_W * SCALE), int(IMG_H * SCALE)))
        t0, n_frames, fps, last_rec, show_help = time.time(), 0, 0.0, 0.0, True
        message, message_t, started, quit_now = "Press T to start the mission", time.time(), False, False

        while True:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    ev = pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE)
                if ev.type != pg.KEYDOWN:
                    continue
                if ev.key == pg.K_t and not started:
                    started = True
                    threading.Thread(target=m.run, daemon=True).start()
                    message, message_t = "Mission started", time.time()
                elif ev.key == pg.K_SPACE and started and not m.done:
                    m.paused = not m.paused
                    message, message_t = "PAUSED - SPACE to continue" if m.paused else "Continuing", time.time()
                elif ev.key == pg.K_ESCAPE:
                    if not started or m.done:
                        quit_now = True
                    elif not m.abort:
                        m.abort, m.paused = "home", False
                        message, message_t = "ABORT - flying home to land", time.time()
                elif ev.key == pg.K_l and started and not m.done:
                    m.abort, m.paused = "land", False
                    message, message_t = "EMERGENCY LANDING", time.time()
                elif ev.key == pg.K_h:
                    show_help = not show_help
            if quit_now or (started and m.done):
                break

            ok, frame = cam.read(timeout=0.5)
            if not ok:
                if not cam.isOpened():
                    print("Camera stopped - is simfull running?")
                    m.abort = "land"
                    break
                continue
            pos, yaw, now = d.real, m.yaw, time.time()
            if SURVEY and m.flying and n_frames % (4 if m.scan_id else 20) == 0:
                fn = f"{len(survey_rows):05d}.jpg"
                cv2.imwrite(str(survey_dir / fn), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                survey_rows.append([fn, round(now - t0, 2), *(round(float(v), 4) for v in pos),
                                    round(float(yaw), 2), m.scan_id or "", m.phase])
            if m.flying:
                fly_time += min(0.5, now - last_t)
            last_t = now
            if now - last_rec > 0.2 and started:
                path.append((pos[0], pos[1]))
                last_rec = now

            # ---- detect + match to the defect patches (same rules as pilot.py) ----
            res = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]
            shown, new_tracks = frame.copy(), []
            for (x1, y1, x2, y2), conf, cls in pilot.STAB.update(res):
                name, hit = names[int(cls)], None
                if m.flying:
                    o, dvec = pixel_ray((x1 + x2) / 2, (y1 + y2) / 2, pos, yaw)
                    hit = find_decal(o, dvec, decals, boxes)
                    if hit is None:
                        unmatched += 1
                        if conf >= ALARM_CONF:
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            prev = next((t_ for t_ in alarm_tracks if t_["cls"] == name and
                                         math.hypot(t_["cx"] - cx, t_["cy"] - cy) < ALARM_PX), None)
                            tr = {"cls": name, "cx": cx, "cy": cy, "n": prev["n"] + 1 if prev else 1,
                                  "counted": prev["counted"] if prev else False}
                            if prev:
                                alarm_tracks.remove(prev)
                            if tr["n"] >= FOUND_FRAMES and not tr["counted"]:
                                tr["counted"] = True
                                false_alarms.append([round(now - t0, 1), name, round(conf, 3),
                                                     *(round(v, 2) for v in pos), round(yaw, 1)])
                            new_tracks.append(tr)
                    else:
                        scanning = m.scan_id == hit["id"]
                        evidence[hit["id"]][name] += conf * (1.0 if scanning else PASSING_WEIGHT) * (1.0 if conf >= pilot.CLASS_CONF else pilot.WEAK_VOTE)
                        frames_on[hit["id"]] += 1
                        if scanning:
                            b = m.best.setdefault(hit["id"], {})
                            b["on"] = max(b.get("on", 0), conf)
                if m.scan_id and hit is None and m.flying:
                    b = m.best.setdefault(m.scan_id, {})
                    b["off"] = max(b.get("off", 0), conf)
                weak = hit is None and conf < ALARM_CONF
                col = (60, 220, 90) if hit else (170, 170, 170) if weak else (0, 200, 255)
                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), col, 1 if weak else 2)
                cv2.putText(shown, f"{name} {conf:.2f}" + (f"  [{hit['id']}]" if hit else ""),
                            (int(x1) + 3, max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                if hit and frames_on[hit["id"]] == FOUND_FRAMES and hit["id"] not in first_seen:
                    first_seen[hit["id"]] = now - t0
                    cv2.imwrite(str(out / "findings" / f"{hit['id']}_{name}.jpg"), shown)
                    message, message_t = f"FOUND {hit['id']}: {name} ({hit['zone']})", now
            alarm_tracks = new_tracks

            it = m.item
            if it is not None and it["kind"] == "stop":           # outline of the defect being visited
                dc = it["decal"]
                corners = [project(dc["_c"] + a * dc["width"] / 2 * dc["_h"] +
                                   b * dc["height"] / 2 * dc["_v"], pos, yaw)
                           for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                if all(corners):
                    cv2.polylines(shown, [np.array(corners, np.int32)], True, (255, 80, 255), 2)
                    top = min(corners, key=lambda q: q[1])
                    cv2.putText(shown, f"TARGET {it['id']}", (top[0], max(15, top[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 255), 2)
            results = {}
            for did in first_seen:
                pred = evidence[did].most_common(1)[0][0]
                results[did] = "correct" if pred == next(x for x in decals if x["id"] == did)["class"] else "other"

            # ---- display ----
            n_frames += 1
            if n_frames % 10 == 0:
                fps = n_frames / (time.time() - t0)
            rgb = cv2.cvtColor(cv2.resize(shown, (int(IMG_W * SCALE), int(IMG_H * SCALE))), cv2.COLOR_BGR2RGB)
            screen.blit(pg.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
            panel = pg.Surface((400, 156), pg.SRCALPHA)
            panel.fill((0, 0, 0, 150))
            screen.blit(panel, (8, 8))
            done_stops = len(m.log)
            phase = ("PAUSED - " if m.paused else "") + m.phase
            if m.hold_t:
                phase += f"  {time.time() - m.hold_t:3.1f} s"
            lines = [(f"AUTO  {phase}", (255, 80, 255)),
                     (f"stop {min(done_stops + 1, n_stops)}/{n_stops}" +
                      (f"  {it['id']}  {it['zone']}" if it else ""), (255, 255, 255)),
                     (f"height {pos[2]:4.2f} m   heading {yaw:+5.0f} deg   {fps:4.1f} fps", (255, 255, 255)),
                     (f"defects found {len(first_seen)}/{len(decals)}", (90, 230, 110)),
                     (f"false alarms {len(false_alarms)}" +
                      (f"  ({len(false_alarms) / (fly_time / 60):.1f}/min)" if fly_time > 5 else ""),
                      (255, 140, 90))]
            for i, (txt, col) in enumerate(lines):
                screen.blit((big if i in (0, 3) else font).render(txt, True, col), (16, 14 + i * 28))
            screen.blit(draw_minimap(pg, parts, decals, results, path, plan_pts, pos, yaw,
                                     it["pos"] if it else None), (screen.get_width() - MAP_W - 8, 8))
            if show_help:
                hp = font.render(HELP, True, (255, 255, 255))
                bg = pg.Surface((hp.get_width() + 16, 30), pg.SRCALPHA)
                bg.fill((0, 0, 0, 150))
                screen.blit(bg, (8, screen.get_height() - 38))
                screen.blit(hp, (16, screen.get_height() - 33))
            if time.time() - message_t < 3:
                msg = big.render(message, True, (255, 255, 120))
                screen.blit(msg, ((screen.get_width() - msg.get_width()) // 2, 180))
            pg.display.flip()
            writer.write(cv2.cvtColor(pg.surfarray.array3d(screen).swapaxes(0, 1), cv2.COLOR_RGB2BGR))

        if started and not m.done:                 # window closed / camera lost while flying
            t_ = time.time()
            while not m.done and time.time() - t_ < 60:
                time.sleep(0.1)
        gust.stop()                                 # clear the wind force
        m.running = False
        ylog.stop()
        writer.release()
        cam.release()
        pg.quit()

    if not started:
        print("Mission not started - nothing to report.")
        return
    with open(out / "false_alarms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "class", "conf", "x", "y", "z", "yaw"])
        w.writerows(false_alarms)
    with open(out / "mission_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "found", "t_s", "pos_err_m", "yaw_err_deg", "views", "best_conf_on_target",
                    "best_conf_elsewhere"])
        w.writerows(m.log)
    if SURVEY:
        import json
        with open(survey_dir / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "t_s", "x", "y", "z", "yaw", "scan_id", "phase"])
            w.writerows(survey_rows)
        (survey_dir / "camera.json").write_text(json.dumps({**pilot.CAM_PROFILE, "world": pilot.WORLD_NAME}))
        print(f"SURVEY: {len(survey_rows)} clean frames saved in {survey_dir}")
    write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, time.time() - t0,
                 false_alarms, fly_time)
    print("\nper stop:  id   found  pos_err  yaw_err  views  conf_on_target  conf_elsewhere")
    for r in m.log:
        print(f"           {r[0]}  {'yes' if r[1] else 'NO ':5s} {r[3]:6.2f} m {r[4]:6.1f} deg {r[5]:4d}"
              f"  {r[6]:12.2f}  {r[7]:12.2f}")
    print(f"\nWORLD : {pilot.WORLD_NAME}")
    print(f"MODEL : {pilot.MODEL.name}")
    print(f"WIND  : {pilot.WORLD.get('wind', 'calm')}"
          + (f" (towards {pilot.WORLD.get('wind_dir', 45):.0f} deg)" if pilot.WORLD.get('wind', 'calm') != 'calm' else '')
          + f"   LIGHT: {pilot.WORLD.get('light', 'noon')}")
    if m.track:
        e = np.array([r[1] for r in m.track]) * 100
        w = np.array([r[2] for r in m.track]) * 100
        print(f"PATH HOLDING: deviation from the planned path mean {e.mean():.1f} cm, 95% under "
              f"{np.percentile(e, 95):.0f} cm, max {e.max():.0f} cm")
        shown = []
        for i in np.argsort(w):                     # the 5 closest moments, at different places
            r = m.track[i]
            if all(np.linalg.norm(np.array(r[3:6]) - np.array(s[3:6])) > 0.5 for s in shown):
                shown.append(r)
            if len(shown) == 5:
                break
        print("CLOSEST TO A WALL (drone centre; the drone is ~10 cm across):")
        for r in shown:
            print(f"   {r[2] * 100:4.0f} cm  at x {r[3]:5.2f} y {r[4]:6.2f} z {r[5]:4.2f}   {r[6]}")
        with open(out / "tracking.csv", "w", newline="") as f:
            csv.writer(f).writerows([("t_s", "path_dev_m", "wall_dist_m", "x", "y", "z", "phase")] +
                                    [(round(r[0] - m.track[0][0], 2), round(r[1], 3), round(r[2], 3),
                                      round(r[3], 2), round(r[4], 2), round(r[5], 2), r[6]) for r in m.track])
    print(f"WIND FORCE: {gust.summary()}")
    print(f"SAFETY MARGIN: {planner.clear * 100:.0f} cm from every surface (drone centre)")
    print(f"CAMERA: {pilot.CAM_PROFILE['name']}  lens {math.degrees(pilot.HFOV):.0f} deg  "
          f"analog effect: {pilot.CAM_PROFILE.get('analog') or 'off'}")
    print(f"AUTONOMOUS MISSION: {len(m.log)}/{n_stops} stops scanned, "
          f"{sum(r[1] for r in m.log)} found, "
          f"mission time {(m.log[-1][2] if m.log else 0) / 60:.1f} min")


if __name__ == "__main__":
    main()
