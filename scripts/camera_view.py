# camera_view.py - show the drone camera live.  Keys: q = quit, s = save a snapshot
import os
import time

import cv2
from gz_camera import GazeboCamera

SNAP_DIR = os.path.expanduser("~/drone_ws/logs/snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

cap = GazeboCamera()            # later for the real drone: cap = cv2.VideoCapture(0)
frames, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        print("No frames - is simwall running?")
        break
    frames += 1
    fps = frames / (time.time() - t0)
    cv2.putText(frame, f"{fps:4.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Drone camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        path = os.path.join(SNAP_DIR, time.strftime("snap_%H%M%S.jpg"))
        cv2.imwrite(path, frame)
        print("Saved", path)
cap.release()
cv2.destroyAllWindows()