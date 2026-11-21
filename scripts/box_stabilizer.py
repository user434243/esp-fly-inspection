#!/usr/bin/env python3
"""
box_stabilizer.py - steady YOLO boxes: no blinking boxes, fewer one-frame false alarms.

A box is SHOWN only when the same object was detected in at least SHOW of the last WINDOW frames.
Once shown, it is kept for HOLD frames even if YOLO misses it (no blinking), its position is smoothed,
and its class is the confidence-weighted vote over its whole life (no class flicker).
Works on any video (simulator or the real A11T receiver) - it only uses the image boxes.

  stab = BoxStabilizer()
  for (x1, y1, x2, y2), conf, cls in stab.update(results):   # results = model.predict(...)[0]
"""
from collections import Counter, deque

SHOW, WINDOW, HOLD = 3, 5, 3          # shown after 3 hits in 5 frames, kept 3 frames after the last hit
MATCH_IOU = 0.25                      # same object if boxes overlap this much (or centres are close)
SMOOTH = 0.5                          # box position: 0 = frozen, 1 = no smoothing


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def close(a, b):
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return abs(ca[0] - cb[0]) < 0.5 * size and abs(ca[1] - cb[1]) < 0.5 * size


class _Track:
    def __init__(self, box, conf, cls):
        self.box, self.conf = list(box), conf
        self.votes = Counter({int(cls): conf})
        self.hits = deque([True], maxlen=WINDOW)
        self.misses = 0
        self.shown = False

    def add(self, box, conf, cls):
        self.box = [SMOOTH * n + (1 - SMOOTH) * o for n, o in zip(box, self.box)]
        self.conf = 0.5 * conf + 0.5 * self.conf
        self.votes[int(cls)] += conf
        self.hits.append(True)
        self.misses = 0

    def miss(self):
        self.hits.append(False)
        self.misses += 1


class BoxStabilizer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, res):
        """res: one ultralytics result. Returns [((x1, y1, x2, y2), conf, cls), ...] of steady boxes."""
        dets = list(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()))
        if not self.enabled:
            return dets
        free = list(self.tracks)
        for box, conf, cls in sorted(dets, key=lambda d: -d[1]):
            best = max(free, key=lambda t: iou(t.box, box), default=None)
            if best is not None and (iou(best.box, box) >= MATCH_IOU or close(best.box, box)):
                best.add(box, conf, cls)
                free.remove(best)
            else:
                self.tracks.append(_Track(box, conf, cls))
        for t in free:
            t.miss()
        self.tracks = [t for t in self.tracks if t.misses <= HOLD and any(t.hits)]
        out = []
        for t in self.tracks:
            if not t.shown and sum(t.hits) >= SHOW:
                t.shown = True
            if t.shown:
                cls = t.votes.most_common(1)[0][0]
                out.append((tuple(t.box), t.conf * (0.85 ** t.misses), float(cls)))
        return out
