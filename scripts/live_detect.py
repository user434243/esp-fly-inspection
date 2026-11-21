# live_detect.py - run your YOLO model on the drone camera live, and record a video
#   Keys: q = quit, s = save snapshot
import os
import time
from collections import Counter

import cv2
from ultralytics import YOLO
from gz_camera import GazeboCamera

MODEL = os.path.expanduser("~/drone_ws/yolo/models/best.pt")
CONF = 0.35                      # minimum confidence to show a detection
VIDEO_DIR = os.path.expanduser("~/drone_ws/logs/videos")
SNAP_DIR = os.path.expanduser("~/drone_ws/logs/snapshots")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

model = YOLO(MODEL)
print("Classes:", model.names)

cap = GazeboCamera()             # real drone later: cap = cv2.VideoCapture(0)
video_path = os.path.join(VIDEO_DIR, time.strftime("detect_%Y%m%d_%H%M%S.mp4"))
writer = None
frames, t0 = 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("No frames - is simwall running?")
        break

    result = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]
    shown = result.plot()        # frame with boxes and labels drawn

    frames += 1
    fps = frames / (time.time() - t0)
    counts = Counter(model.names[int(c)] for c in result.boxes.cls)
    info = "  ".join(f"{k}:{v}" for k, v in counts.items()) or "no defects"
    cv2.putText(shown, f"{fps:4.1f} fps | {info}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if writer is None:           # start the video once we know the frame size
        h, w = shown.shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    writer.write(shown)

    cv2.imshow("YOLO live", shown)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        path = os.path.join(SNAP_DIR, time.strftime("detect_%H%M%S.jpg"))
        cv2.imwrite(path, shown)
        print("Saved", path)

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("Video saved:", video_path)