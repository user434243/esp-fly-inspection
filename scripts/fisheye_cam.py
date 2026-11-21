#!/usr/bin/env python3
"""
fisheye_cam.py - make the simulator camera look through the REAL A11T lens.

Gazebo renders a normal (pinhole) image wider and bigger (960x720, 136 deg);
this module bends it into the measured 640x480 fisheye image of the real AKK A11T (a11t_calib.json,
from the checkerboard calibration) and adds the measured dark corners (vignetting).

  Lens(calib_json, render_hfov).apply(frame)     -> real-lens image
  Lens.pix_to_norm(u, v) / norm_to_pix(x, y)     -> exact pixel <-> 3D direction (for pilot.py)
  Lens.wrap(cam)                                 -> drop-in wrapper for GazeboCamera
Test:  python fisheye_cam.py some_frame.jpg      -> some_frame_lens.jpg  (before | after)
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

# brightness vs distance from the image centre (0 = centre, 1 = corner), measured from the real video
VIGNETTE_R = [0.00, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
VIGNETTE_G = [1.00, 1.00, 0.99, 0.95, 0.90, 0.82, 0.73, 0.62, 0.56, 0.49, 0.38, 0.30]


class Lens:
    """Kannala-Brandt fisheye model (same as OpenCV fisheye), inverted with a lookup table:
    exact out to the image corners, where cv2.fisheye.undistortPoints is unreliable."""

    def __init__(self, calib_path, render_hfov, size=(640, 480), vignette=True):
        c = json.loads(Path(calib_path).expanduser().read_text())
        w, h = size
        cw, ch = c["image_size"]
        K = np.array(c["K"], np.float64)
        self.fx, self.fy = K[0, 0] * w / cw, K[1, 1] * h / ch     # calibration size -> simulator size
        self.cx, self.cy = K[0, 2] * w / cw, K[1, 2] * h / ch
        self.k = np.array(c["D_fisheye"], np.float64).ravel()
        self.th = np.radians(np.linspace(0, 85, 3401))            # angle from the optical axis
        t2 = self.th ** 2
        self.thd = self.th * (1 + self.k[0] * t2 + self.k[1] * t2 ** 2 + self.k[2] * t2 ** 3 + self.k[3] * t2 ** 4)
        if np.any(np.diff(self.thd) <= 0):
            raise ValueError("lens calibration is not monotonic up to 85 deg - recalibrate")
        self.w, self.h = w, h
        self.render_hfov = render_hfov
        self.maps = {}                                            # (render w, h) -> remap tables
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        self.nx, self.ny = self._undistort(xs, ys)                # direction seen by every output pixel
        r = np.hypot(xs - self.cx, ys - self.cy) / math.hypot(w / 2, h / 2)
        self.vig = np.interp(r, VIGNETTE_R, VIGNETTE_G).astype(np.float32)[..., None] if vignette else None

    def _maps(self, rw, rh):
        if (rw, rh) not in self.maps:
            fr = (rw / 2) / math.tan(self.render_hfov / 2)        # focal length of the Gazebo render
            if np.abs(self.nx).max() * fr > rw / 2 or np.abs(self.ny).max() * fr > rh / 2:
                print(f"!! fisheye_cam: render {rw}x{rh} @ {math.degrees(self.render_hfov):.0f} deg "
                      "does not cover the lens corners (edges will be smeared)")
            self.maps[(rw, rh)] = ((rw / 2 + self.nx * fr).astype(np.float32),
                                   (rh / 2 + self.ny * fr).astype(np.float32))
        return self.maps[(rw, rh)]

    def _undistort(self, u, v):
        xd, yd = (np.asarray(u, float) - self.cx) / self.fx, (np.asarray(v, float) - self.cy) / self.fy
        rd = np.hypot(xd, yd)
        theta = np.interp(rd, self.thd, self.th)
        s = np.where(rd > 1e-9, np.tan(theta) / np.maximum(rd, 1e-9), 1.0)
        return xd * s, yd * s

    def apply(self, frame):
        mx, my = self._maps(frame.shape[1], frame.shape[0])
        out = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if self.vig is not None:
            out = np.clip(out.astype(np.float32) * self.vig, 0, 255).astype(np.uint8)
        return out

    def pix_to_norm(self, u, v):
        """Image pixel -> direction at depth 1 (x right, y down)."""
        x, y = self._undistort(u, v)
        return float(x), float(y)

    def norm_to_pix(self, x, y):
        """Direction at depth 1 (x right, y down) -> image pixel."""
        r = math.hypot(x, y)
        if r < 1e-12:
            return float(self.cx), float(self.cy)
        t = math.atan(r)
        t2 = t * t
        td = t * (1 + self.k[0] * t2 + self.k[1] * t2 ** 2 + self.k[2] * t2 ** 3 + self.k[3] * t2 ** 4)
        return float(self.cx + self.fx * td * x / r), float(self.cy + self.fy * td * y / r)

    def wrap(self, cam):
        return LensCamera(cam, self)


class LensCamera:
    """Drop-in wrapper for GazeboCamera: same read() / isOpened() / release()."""

    def __init__(self, cam, lens):
        self.cam, self.lens = cam, lens

    def read(self, *a, **k):
        ok, frame = self.cam.read(*a, **k)
        return (ok, self.lens.apply(frame)) if ok else (ok, frame)

    def isOpened(self):
        return self.cam.isOpened()

    def release(self):
        self.cam.release()


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1])
    calib = Path.home() / "drone_ws/sim/a11t_calib.json"
    img = cv2.imread(str(src))
    lens = Lens(calib, math.radians(136))          # input: a 960x720 pinhole render at 136 deg
    out = src.with_name(src.stem + "_lens.jpg")
    cv2.imwrite(str(out), np.hstack([cv2.resize(img, (640, 480)), lens.apply(img)]))
    print("saved", out)
