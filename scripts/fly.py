# fly.py - square flight in CrazySim using position setpoints (x, y, z in metres)
import time
import warnings
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)   # hide "legacy" notices
URI = "udp://127.0.0.1:19850"
RATE = 20   # commands per second

# Mission: (name, x, y, z, seconds to get there)
WAYPOINTS = [
    ("Take off",      0.0,  0.0, 1.0, 3),
    ("Forward 2 m",   2.0,  0.0, 1.0, 4),
    ("Left 2 m",      2.0,  2.0, 1.0, 4),
    ("Back 2 m",      0.0,  2.0, 1.0, 4),
    ("Right 2 m",     0.0,  0.0, 1.0, 4),
    ("Climb to 2 m",  0.0,  0.0, 2.0, 3),
    ("Down to 1 m",   0.0,  0.0, 1.0, 3),
    ("Land",          0.0,  0.0, 0.05, 3),
]


def fly_to(cf, start, goal, seconds):
    """Move smoothly in a straight line from start to goal (x, y, z)."""
    steps = int(seconds * RATE)
    for i in range(1, steps + 1):
        a = i / steps
        x, y, z = (s + (g - s) * a for s, g in zip(start, goal))
        cf.commander.send_position_setpoint(x, y, z, 0)
        time.sleep(1 / RATE)
    return goal


def hold(cf, point, seconds):
    """Stay at one point."""
    for _ in range(int(seconds * RATE)):
        cf.commander.send_position_setpoint(*point, 0)
        time.sleep(1 / RATE)


def finish(cf):
    """End the flight safely: keep sending 'motors off' for 3 s,
    so the safety system sees a normal landing and does NOT lock the drone."""
    for _ in range(int(3 * RATE)):
        cf.commander.send_stop_setpoint()
        time.sleep(1 / RATE)


def show_position(ts, d, conf):
    print(f"   real: x={d['stateEstimate.x']:+.2f}  "
          f"y={d['stateEstimate.y']:+.2f}  z={d['stateEstimate.z']:+.2f}")


cflib.crtp.init_drivers()
with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
    cf = scf.cf
    print("Connected!")

    log = LogConfig(name="pos", period_in_ms=1000)       # print position once a second
    for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
        log.add_variable(v)
    cf.log.add_config(log)
    log.data_received_cb.add_callback(show_position)
    log.start()

    p = (0.0, 0.0, 0.0)
    for name, x, y, z, secs in WAYPOINTS:
        print(f"-> {name}")
        p = fly_to(cf, p, (x, y, z), secs)
        hold(cf, p, 1)                                     # pause 1 s at each point
    finish(cf); print("Motors off")

    log.stop()
    print("Flight finished.")