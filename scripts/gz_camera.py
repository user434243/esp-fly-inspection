# gz_camera.py - read the simulated drone camera exactly like a webcam
#   cap = GazeboCamera()        (simulation)
#   cap = cv2.VideoCapture(0)   (real drone + UVC receiver)  <- same .read() / .release()
import os
import struct
import subprocess
import threading
import time

import numpy as np

PIPE_BIN = os.path.expanduser("~/drone_ws/sim/tools/gz_camera_pipe/build/gz_camera_pipe")


class GazeboCamera:
    def __init__(self, topic="/cf_0/camera"):
        self.proc = subprocess.Popen([PIPE_BIN, topic], stdout=subprocess.PIPE, bufsize=0)
        self.frame = None
        self.frame_id = 0          # increases with every new frame from Gazebo
        self._last_id = 0          # last frame handed out by read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()   # always keep the newest frame

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def _reader(self):
        while self.running:
            header = self._read_exact(8)
            if header is None:
                break
            w, h = struct.unpack("II", header)
            data = self._read_exact(w * h * 3)
            if data is None:
                break
            rgb = np.frombuffer(data, np.uint8).reshape(h, w, 3)
            with self.lock:
                self.frame = rgb[:, :, ::-1].copy()    # RGB -> BGR (OpenCV order)
                self.frame_id += 1
        self.running = False

    def read(self, timeout=2.0):
        """Wait for the next NEW frame and return (ok, frame), like cv2.VideoCapture.read()."""
        t0 = time.time()
        while self.frame_id == self._last_id:
            if not self.running or time.time() - t0 > timeout:
                return False, None
            time.sleep(0.002)
        with self.lock:
            self._last_id = self.frame_id
            return True, self.frame.copy()

    def isOpened(self):
        return self.running

    def release(self):
        self.running = False
        self.proc.terminate()