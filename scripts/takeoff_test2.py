# takeoff_test2.py - A) check the reply to takeoff, B) try hover setpoints instead
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crtp.crtpstack import CRTPPort

URI = "udp://127.0.0.1:19850"
cflib.crtp.init_drivers()

def on_hl_reply(pk):
    print(f"    >> drone replied to high-level command {pk.data[0]}: result={pk.data[3]}  (0 = OK)")

def show(ts, d, conf):
    print(f"target_z={d['ctrltarget.z']:+.2f} | motor1={d['motor.m1']:6d} | real_z={d['stateEstimate.z']:+.2f}")

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
    cf = scf.cf
    cf.add_port_callback(CRTPPort.SETPOINT_HL, on_hl_reply)

    log = LogConfig(name="t", period_in_ms=500)
    for v in ("ctrltarget.z", "motor.m1", "stateEstimate.z"):
        log.add_variable(v)
    cf.log.add_config(log)
    log.data_received_cb.add_callback(show)
    log.start()

    print("=== A) high-level takeoff ===")
    cf.high_level_commander.takeoff(0.5, 2.0)
    time.sleep(3)
    cf.high_level_commander.stop()
    time.sleep(1)

    print("=== B) hover setpoints: climb to 0.5 m and hold ===")
    for i in range(40):                                    # 4 s at 10 Hz
        cf.commander.send_hover_setpoint(0, 0, 0, min(0.5, 0.1 + i * 0.02))
        time.sleep(0.1)
    print("=== B) descend ===")
    for i in range(25):
        cf.commander.send_hover_setpoint(0, 0, 0, max(0.0, 0.5 - i * 0.02))
        time.sleep(0.1)
    cf.commander.send_stop_setpoint()
    time.sleep(1)
    log.stop()