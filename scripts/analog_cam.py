#!/usr/bin/env python3
"""
analog_cam.py - make the clean Gazebo camera image look like the real FPV video chain:
  AKK A11T analog camera -> 5.8 GHz transmitter -> RC832 receiver -> USB capture card -> PC

What it imitates (each step can be tuned below):
  1. analog resolution: NTSC 720x480, soft like a 600-800 TVL micro camera
  2. colour bleeding: analog colour has much less detail than brightness
  3. sensor noise (small 1/4" CMOS) and slight over-sharpening halos of cheap cameras
  4. radio link: occasional interference lines (more on a "bad" link),
     rare short 'snow' bursts (lost signal)
  5. capture card: MJPEG compression
The output is resized back to the input size, so the drone's 3D maths are unchanged.

Use:   from analog_cam import AnalogCamera     cam = AnalogCamera(GazeboCamera())
Test:  python analog_cam.py some_frame.jpg     -> some_frame_analog.jpg (before | after)
"""
import cv2
import numpy as np

PROFILES = {
    #                blur  chroma  noise  sharpen  lines  snow     jpeg
    "good":   dict(blur=0.8, chroma=4, noise=4.0, sharpen=0.4, lines=0.01, snow=0.000, jpeg=80),
    "typical": dict(blur=1.1, chroma=6, noise=7.0, sharpen=0.6, lines=0.04, snow=0.004, jpeg=70),
    "bad":    dict(blur=1.5, chroma=8, noise=12.0, sharpen=0.8, lines=0.12, snow=0.02, jpeg=60),
}
NTSC = (720, 480)


class AnalogFX:
    def __init__(self, profile="typical", seed=None):
        self.p = dict(PROFILES[profile])
        self.rng = np.random.default_rng(seed)
        self.snow_left = 0
        # pre-made noise (generating fresh random noise every frame is slow); a random
        # slice of this bank is used each frame, so the noise still changes every frame
        w, h = NTSC
        self.bank = self.rng.standard_normal((h * 2, w * 2, 3)).astype(np.float32)

    def __call__(self, frame):
        p, rng = self.p, self.rng
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, NTSC, interpolation=cv2.INTER_AREA)

        # 1+2) soft luma, very soft + slightly shifted chroma (YCrCb)
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y = cv2.GaussianBlur(ycc[..., 0], (0, 0), sigmaX=p["blur"] * 1.3, sigmaY=p["blur"])
        if p["sharpen"]:                                     # cheap-camera edge enhancement
            y = y + p["sharpen"] * (y - cv2.GaussianBlur(y, (0, 0), 2.0))
        k = max(1, int(p["chroma"]))                          # analog colour: ~1/k of the detail
        c = cv2.resize(ycc[..., 1:], (NTSC[0] // k, NTSC[1]), interpolation=cv2.INTER_AREA)
        c = cv2.resize(c, NTSC, interpolation=cv2.INTER_LINEAR)
        c = np.roll(c, 2, axis=1)                            # colour lags the brightness a bit

        # 3) sensor noise (luma) + a little colour noise
        r0, c0 = rng.integers(0, NTSC[1]), rng.integers(0, NTSC[0])
        n = self.bank[r0:r0 + NTSC[1], c0:c0 + NTSC[0]]
        y += p["noise"] * n[..., 0]
        c += (p["noise"] * 0.5) * n[..., 1:]

        # 4) radio link: interference lines and short snow bursts
        if rng.random() < p["lines"]:
            for _ in range(rng.integers(1, 4)):
                r = rng.integers(0, NTSC[1] - 6)
                hgt = rng.integers(1, 6)
                y[r:r + hgt] += rng.normal(rng.choice([-40, 40]), 25, (hgt, NTSC[0]))
                shift = int(rng.integers(-12, 12))
                y[r:r + hgt] = np.roll(y[r:r + hgt], shift, axis=1)
        if self.snow_left == 0 and rng.random() < p["snow"]:
            self.snow_left = int(rng.integers(2, 6))         # a few frames of lost signal
        if self.snow_left:
            self.snow_left -= 1
            a = rng.uniform(0.4, 0.8)
            y = (1 - a) * y + a * (128 + 60 * self.bank[:NTSC[1], :NTSC[0], 0])
            c = (1 - a) * c + a * 128

        ycc[..., 0], ycc[..., 1:] = y, c
        out = np.clip(ycc, 0, 255, out=ycc).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

        # 5) capture card MJPEG compression, back to the simulator size
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)


class AnalogCamera:
    """Drop-in wrapper for GazeboCamera: same read() / isOpened() / release()."""

    def __init__(self, cam, profile="typical"):
        self.cam, self.fx = cam, AnalogFX(profile)

    def read(self, *a, **k):
        ok, frame = self.cam.read(*a, **k)
        return (ok, self.fx(frame)) if ok else (ok, frame)

    def isOpened(self):
        return self.cam.isOpened()

    def release(self):
        self.cam.release()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    src = Path(sys.argv[1])
    prof = sys.argv[2] if len(sys.argv) > 2 else "typical"
    img = cv2.imread(str(src))
    fx = AnalogFX(prof, seed=1)
    fx.p["lines"] = 1.0                                    # show one interference event in the demo
    side = np.hstack([img, fx(img)])
    out = src.with_name(f"{src.stem}_analog_{prof}.jpg")
    cv2.imwrite(str(out), side)
    print("saved", out)
