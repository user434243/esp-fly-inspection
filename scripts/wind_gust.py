#!/usr/bin/env python3
"""
wind_gust.py - wind for the simulated drone, as a real force (Gazebo ApplyLinkWrench).

Gazebo's own wind system flips the CrazySim drone, so the wind is made here instead:
  wind speed  = mean x (1 + gust swing x sin(2 pi t / period) + random part)   (preset from world.json)
  air drag    F = c x |w - v| x (w - v)      w = wind, v = drone velocity, c = 0.006 N/(m/s)^2
                (0.5 x air density 1.2 x Cd 1.0 x ~0.01 m2 frontal area of a 10 cm drone)
  shelter     inside the box girder, the tunnel, the arcade ... the wind is mostly blocked
The force is pushed to the drone 10x a second on /world/<world>/wrench/persistent
(python gz-transport if installed, else the 'gz topic' command, ~2x a second).

  standalone test (sim running, drone flying with pilot.py):
     ~/drone_ws/venv/bin/python ~/drone_ws/scripts/wind_gust.py
  autopilot.py starts it by itself when the world has wind (simworld --wind ...).
"""
import math
import re
import subprocess
import threading
import time

import numpy as np

# mean m/s, gust swing (fraction), gust period s, random part (fraction)
PRESETS = {
    "calm":     (0.0, 0.0, 1.0, 0.0),
    "light":    (1.5, 0.15, 8.0, 0.05),
    "moderate": (3.0, 0.25, 6.0, 0.10),
    "gusty":    (3.0, 0.70, 3.0, 0.25),
    "strong":   (5.0, 0.30, 5.0, 0.15),
}
DRAG = 0.006
SHELTER = {   # world -> [(lo, hi, wind factor)]  closed or half-closed spaces
    "city_full": [((6.0, 8.75, 4.85), (30.0, 12.25, 6.15), 0.05),      # inside the box girder
                  ((3.8, 17.4, 0.0), (16.2, 21.8, 1.25), 0.05),        # tunnel
                  ((6.0, -7.0, 0.0), (17.5, -4.2, 2.6), 0.4),          # arcade
                  ((17.5, -13.0, 0.0), (18.5, -4.0, 6.4), 0.5)],       # 1 m alley
    "bridge_full": [((2.5, -9.3, 2.45), (5.5, 9.3, 3.15), 0.2),        # girder bays
                    ((1.0, 13.5, 0.0), (7.4, 14.5, 0.9), 0.05)],       # culvert
}


class Gusts:
    def __init__(self, world, world_name, get_pos, rate=10.0):
        self.preset = world.get("wind", "calm")
        self.v, self.swing, self.period, self.rand = PRESETS.get(self.preset, PRESETS["calm"])
        self.dir = math.radians(world.get("wind_dir", 45.0))
        self.drag = world.get("wind_drag", DRAG)
        self.shelter = [(np.array(a), np.array(b), f) for a, b, f in
                        world.get("shelter", SHELTER.get(world_name, []))]
        self.world, self.get_pos, self.rate = world_name, get_pos, rate
        self.on = self.v > 0
        self.running = False
        self.log = []                          # (t, wind speed, |force| N, shelter factor)
        self.entity = None
        self.pub = None

    # ---- sending the force ----
    def _connect(self):
        out = subprocess.run(["gz", "model", "--list"], capture_output=True, text=True, timeout=15).stdout
        names = re.findall(r"^\s*-\s*(\S+)", out, re.M)
        cf = [n for n in names if n.startswith("cf") or "crazyflie" in n.lower()]
        self.entity = cf[0] if cf else "cf_0"
        try:                                    # fast path: python gz-transport (Garden)
            from gz.transport12 import Node
            from gz.msgs9.entity_wrench_pb2 import EntityWrench
            from gz.msgs9.entity_pb2 import Entity
            self._node = Node()
            self._ew, self._ent = EntityWrench, Entity
            self.pub = self._node.advertise(f"/world/{self.world}/wrench/persistent", EntityWrench)
            self.clear_pub = self._node.advertise(f"/world/{self.world}/wrench/clear", Entity)
            time.sleep(0.5)                     # let Gazebo discover the publishers
        except Exception:
            self.pub = None
            self.rate = min(self.rate, 2.0)     # 'gz topic' is slow
        print(f"WIND: {self.preset} on '{self.entity}' via "
              f"{'gz-transport' if self.pub else 'gz topic command'} ({self.rate:.0f}x/s)")

    def _clear(self):
        if self.pub:
            e = self._ent(name=self.entity, type=self._ent.MODEL)
            self.clear_pub.publish(e)
            time.sleep(0.01)                    # let Gazebo apply the clear before the new force
        else:
            subprocess.run(["gz", "topic", "-t", f"/world/{self.world}/wrench/clear", "-m", "gz.msgs.Entity",
                            "-p", f'name: "{self.entity}", type: MODEL'], capture_output=True, timeout=5)

    def _push(self, f):
        self._clear()                           # a persistent wrench stays until cleared
        if self.pub:
            msg = self._ew()
            msg.entity.name, msg.entity.type = self.entity, self._ent.MODEL
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = map(float, f)
            self.pub.publish(msg)
        else:
            subprocess.run(["gz", "topic", "-t", f"/world/{self.world}/wrench/persistent", "-m",
                            "gz.msgs.EntityWrench", "-p",
                            f'entity: {{name: "{self.entity}", type: MODEL}}, '
                            f'wrench: {{force: {{x: {f[0]:.4f}, y: {f[1]:.4f}, z: 0}}}}'],
                           capture_output=True, timeout=5)

    # ---- the wind ----
    def shelter_factor(self, p):
        for lo, hi, k in self.shelter:
            if (lo <= p).all() and (p <= hi).all():
                return k
        return 1.0

    def loop(self):
        rng = np.random.default_rng()
        t0, last_p, last_t, vel, noise = time.time(), None, None, np.zeros(3), 0.0
        while self.running:
            t = time.time()
            p = np.asarray(self.get_pos(), float)
            if last_p is not None:
                dt = max(t - last_t, 1e-3)
                step = (p - last_p) / dt
                if np.linalg.norm(step) < 5.0:          # ignore position jumps (reset / teleport)
                    vel = 0.6 * vel + 0.4 * step
                noise += -noise * dt / 1.5 + rng.standard_normal() * math.sqrt(dt) * 0.8   # slow random gusts
            last_p, last_t = p, t
            s = t - t0
            speed = max(0.0, self.v * (1 + self.swing * math.sin(2 * math.pi * s / self.period) + self.rand * noise))
            a = self.dir + 0.15 * math.sin(2 * math.pi * s / 23.0)                         # direction wanders
            w = speed * np.array([math.cos(a), math.sin(a), 0.0])
            k = self.shelter_factor(p)
            rel = (w * k - vel) * np.array([1.0, 1.0, 0.0])
            f = self.drag * np.linalg.norm(rel) * rel if p[2] > 0.15 else np.zeros(3)      # not on the ground
            if np.linalg.norm(f) > 0.25:                # never more than ~ the drone's own weight
                f *= 0.25 / np.linalg.norm(f)
            self._push(f)
            self.log.append((s, speed * k, float(np.linalg.norm(f)), k))
            time.sleep(max(0.0, 1.0 / self.rate - (time.time() - t)))

    def start(self):
        if not self.on:
            return self
        self._connect()
        self._clear()
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def stop(self):
        if self.on and self.running:
            self.running = False
            time.sleep(0.3)
            self._clear()

    def summary(self):
        if not self.log:
            return "calm"
        sp = np.array([r[1] for r in self.log])
        fo = np.array([r[2] for r in self.log])
        return (f"{self.preset}: wind at the drone mean {sp.mean():.1f} m/s, max {sp.max():.1f} m/s;  "
                f"push mean {fo.mean() * 1000:.0f} mN, max {fo.max() * 1000:.0f} mN "
                f"(drone weight ~270 mN)")


if __name__ == "__main__":                     # standalone: wind on the drone while you fly with pilot.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.home() / "drone_ws/scripts"))
    import json
    home = Path.home() / "drone_ws/sim"
    name = (home / "active_world").read_text().strip()
    world = json.loads((home / "worlds" / name / "world.json").read_text())
    out = subprocess.run(["gz", "model", "--list"], capture_output=True, text=True).stdout
    print(out)
    g = Gusts(world, name, lambda: np.array([0, 0, 1.0])).start()
    try:
        while True:
            time.sleep(2)
            if g.log:
                print(f"wind {g.log[-1][1]:.1f} m/s  push {g.log[-1][2] * 1000:.0f} mN")
    except KeyboardInterrupt:
        g.stop()
