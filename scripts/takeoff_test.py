# takeoff_test.py - follow the takeoff command through: target -> motors -> real height
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

URI = "udp://127.0.0.1:19850"
cflib.crtp.init_drivers()

def show(ts, d, conf):
    print(f"target_z={d['ctrltarget.z']:+.2f} | motor1={d['motor.m1']:6d} | "
          f"real_z={d['stateEstimate.z']:+.2f} | info=0x{d['supervisor.info']:02x}")

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
    cf = scf.cf
    log = LogConfig(name="t", period_in_ms=250)
    for v in ("ctrltarget.z", "motor.m1", "stateEstimate.z", "supervisor.info"):
        log.add_variable(v)
    cf.log.add_config(log)
    log.data_received_cb.add_callback(show)
    log.start()

    print("--- before takeoff ---"); time.sleep(1.5)
    print("--- TAKEOFF ---");        cf.high_level_commander.takeoff(0.5, 2.0); time.sleep(4)
    print("--- LAND ---");           cf.high_level_commander.land(0.0, 2.0);    time.sleep(3)
    cf.high_level_commander.stop()
    log.stop()