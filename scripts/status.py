# status.py - print the drone's safety (supervisor) status for 4 seconds
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

URI = "udp://127.0.0.1:19850"
BITS = {0x01: "can arm", 0x02: "armed", 0x04: "auto-arm", 0x08: "CAN FLY",
        0x10: "flying", 0x20: "TUMBLED", 0x40: "LOCKED", 0x80: "CRASHED"}

cflib.crtp.init_drivers()

def show(ts, d, conf):
    info = d["supervisor.info"]
    flags = ", ".join(name for bit, name in BITS.items() if info & bit) or "nothing"
    print(f"info=0x{info:02x} -> {flags} | z={d['stateEstimate.z']:+.2f} "
          f"| roll={d['stabilizer.roll']:+.0f} pitch={d['stabilizer.pitch']:+.0f}")

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
    log = LogConfig(name="status", period_in_ms=500)
    for v in ("supervisor.info", "stateEstimate.z", "stabilizer.roll", "stabilizer.pitch"):
        log.add_variable(v)
    scf.cf.log.add_config(log)
    log.data_received_cb.add_callback(show)
    log.start()
    time.sleep(4)
    log.stop()