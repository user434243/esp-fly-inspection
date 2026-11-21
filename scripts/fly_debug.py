# fly_debug.py - take off and print the drone's position twice a second
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

URI = "udp://127.0.0.1:19850"
cflib.crtp.init_drivers()

def show(ts, d, conf):
    print(f"x={d['stateEstimate.x']:+.2f}  y={d['stateEstimate.y']:+.2f}  z={d['stateEstimate.z']:+.2f}")

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
    cf = scf.cf
    cf.param.set_value("commander.enHighLevel", "1")      # make sure takeoff/go_to are enabled

    log = LogConfig(name="pos", period_in_ms=500)
    for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
        log.add_variable(v, "float")
    cf.log.add_config(log)
    log.data_received_cb.add_callback(show)
    log.start()

    print("--- on the ground ---"); time.sleep(2)
    cf.supervisor.send_arming_request(True); time.sleep(1)
    print("--- takeoff ---");  cf.high_level_commander.takeoff(0.5, 2.0); time.sleep(5)
    print("--- land ---");     cf.high_level_commander.land(0.0, 2.0);    time.sleep(3)
    cf.high_level_commander.stop()
    log.stop()