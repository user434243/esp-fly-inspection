# add_camera.py - add (or remove) a forward-facing FPV camera on the CrazySim drone model
#   add:     python add_camera.py
#   remove:  python add_camera.py --remove
import os
import shutil
import sys

TEMPLATE = os.path.expanduser(
    "~/drone_ws/sim/CrazySim/crazyflie-firmware/tools/crazyflie-simulation/"
    "simulator_files/gazebo/models/crazyflie/model.sdf.jinja")
BACKUP = TEMPLATE + ".orig"
MARK = "<!-- FPV camera added by add_camera.py -->"

CAMERA = """
        """ + MARK + """
        <sensor name="fpv_camera" type="camera">
            <pose>0.03 0 0.015 0 0 0</pose>
            <topic>/cf_{{ cf_id }}/camera</topic>
            <always_on>1</always_on>
            <update_rate>30</update_rate>
            <visualize>false</visualize>
            <camera>
                <horizontal_fov>1.57</horizontal_fov>
                <image>
                    <width>640</width>
                    <height>480</height>
                    <format>R8G8B8</format>
                </image>
                <clip><near>0.05</near><far>50</far></clip>
                <noise><type>gaussian</type><mean>0</mean><stddev>0.007</stddev></noise>
            </camera>
        </sensor>"""

if "--remove" in sys.argv:
    if os.path.exists(BACKUP):
        shutil.copy(BACKUP, TEMPLATE)
        print("Camera removed (original model restored).")
    else:
        print("No backup found - nothing to remove.")
    sys.exit()

text = open(TEMPLATE).read()
if MARK in text:
    print("Camera is already installed - nothing to do.")
    sys.exit()

if not os.path.exists(BACKUP):
    shutil.copy(TEMPLATE, BACKUP)                    # keep the original once

start = text.index('<sensor name="imu_sensor"')      # put the camera right after the IMU
end = text.index("</sensor>", start) + len("</sensor>")
text = text[:end] + CAMERA + text[end:]
open(TEMPLATE, "w").write(text)
print("Camera added: /cf_0/camera, 640x480 @ 30 fps, 90 deg field of view, facing +x (the wall).")