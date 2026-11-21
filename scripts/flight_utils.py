# flight_utils.py - shared helpers for every CrazySim flight
import csv
import math
import os
import time
import warnings

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

warnings.filterwarnings("ignore", category=DeprecationWarning)   # hide "legacy" notices

URI = "udp://127.0.0.1:19850"
RATE = 20                                        # position commands per second
LOG_DIR = os.path.expanduser("~/drone_ws/logs")
CACHE_DIR = os.path.expanduser("~/drone_ws/cache")


class Drone:
    """Use as:  with Drone(name="scan") as d:  d.move(...); d.land()
    - every flight is recorded to ~/drone_ws/logs/<name>_<date>.csv
    - if the script crashes or you press Ctrl+C, the drone lands by itself"""

    def __init__(self, name="flight", uri=URI):
        cflib.crtp.init_drivers()
        self.name = name
        self.scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache=CACHE_DIR))
        self.pos = (0.0, 0.0, 0.0)     # last commanded position (x, y, z)
        self.real = (0.0, 0.0, 0.0)    # last measured position
        self.landed = True

    # ---------- connection and recording ----------
    def __enter__(self):
        self.scf.open_link()
        self.cf = self.scf.cf
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(LOG_DIR, time.strftime(f"{self.name}_%Y%m%d_%H%M%S.csv"))
        self._file = open(self.log_path, "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["t_s", "x_m", "y_m", "z_m"])
        self._t0 = time.time()
        self._last_print = -1.0

        self._log = LogConfig(name="pos", period_in_ms=100)          # 10 samples per second
        for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
            self._log.add_variable(v)
        self.cf.log.add_config(self._log)
        self._log.data_received_cb.add_callback(self._on_position)
        self._log.start()
        print(f"Connected. Recording flight to {self.log_path}")
        return self

    def _on_position(self, ts, d, conf):
        self.real = (d["stateEstimate.x"], d["stateEstimate.y"], d["stateEstimate.z"])
        t = time.time() - self._t0
        self._csv.writerow([f"{t:.2f}"] + [f"{v:.3f}" for v in self.real])
        if t - self._last_print >= 1.0:                               # print once a second
            self._last_print = t
            x, y, z = self.real
            print(f"   real: x={x:+.2f}  y={y:+.2f}  z={z:+.2f}")

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and not self.landed:
            print("!! Script stopped early - landing for safety")
            try:
                self.land(seconds=2)
            except Exception:
                pass
        self._log.stop()
        self._file.close()
        self.scf.close_link()
        return False

    # ---------- flying ----------
    def fly_to(self, x, y, z, seconds):
        """Move smoothly in a straight line to (x, y, z) in the given time."""
        self.landed = False
        start, goal = self.pos, (x, y, z)
        steps = max(1, int(seconds * RATE))
        for i in range(1, steps + 1):
            a = i / steps
            p = tuple(s + (g - s) * a for s, g in zip(start, goal))
            self.cf.commander.send_position_setpoint(*p, 0)
            time.sleep(1 / RATE)
        self.pos = goal

    def move(self, x, y, z, speed=0.3):
        """Fly to (x, y, z) at a given speed in m/s."""
        dist = math.dist(self.pos, (x, y, z))
        self.fly_to(x, y, z, max(1.0, dist / speed))

    def hold(self, seconds):
        """Hover at the current position."""
        for _ in range(int(seconds * RATE)):
            self.cf.commander.send_position_setpoint(*self.pos, 0)
            time.sleep(1 / RATE)

    def land(self, seconds=3):
        """Descend where you are, then switch motors off safely (no lock)."""
        x, y, _ = self.pos
        self.fly_to(x, y, 0.05, seconds)
        for _ in range(3 * RATE):                 # keep talking for 3 s -> no safety lock
            self.cf.commander.send_stop_setpoint()
            time.sleep(1 / RATE)
        self.landed = True
        print("Landed, motors off.")