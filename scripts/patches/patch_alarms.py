#!/usr/bin/env python3
"""patch_alarms.py - add a strict 'false alarm' count to pilot.py.
A false alarm = a detection NOT on a defect patch, conf >= ALARM_CONF, seen in FOUND_FRAMES
frames in a row (same class, box centre within ALARM_PX). Weak ones are drawn thin and grey."""
import shutil
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "drone_ws/scripts/pilot.py"
s = p.read_text()
if "ALARM_CONF" in s:
    sys.exit("already patched")
shutil.copy(p, p.with_suffix(".py.bak"))

R = [
("FOUND_FRAMES = 3\n",
 "FOUND_FRAMES = 3\n"
 "ALARM_CONF = 0.6                           # off-patch detection counts as a false alarm only above this\n"
 "ALARM_PX = 60                              # ... and if it stays (same class, centre moves < 60 px)\n"),

("    (out / \"unmatched\").mkdir(exist_ok=True)\n",
 "    (out / \"unmatched\").mkdir(exist_ok=True)\n"
 "    (out / \"false_alarms\").mkdir(exist_ok=True)\n"),

("    unmatched_log, last_sample = [], 0.0   # detections that hit no defect patch\n",
 "    unmatched_log, last_sample = [], 0.0   # detections that hit no defect patch\n"
 "    alarm_tracks, false_alarms = [], []    # strong off-patch detections followed over frames\n"
 "    fly_time, last_t = 0.0, time.time()    # seconds actually flying\n"),

("            res = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]\n"
 "            shown = frame.copy()\n",
 "            res = model.predict(frame, imgsz=640, conf=CONF, device=0, verbose=False)[0]\n"
 "            shown = frame.copy()\n"
 "            now = time.time()\n"
 "            if state == \"FLYING\":\n"
 "                fly_time += min(0.5, now - last_t)\n"
 "            last_t = now\n"
 "            new_tracks = []\n"),

("                    else:\n"
 "                        evidence[hit[\"id\"]][name] += conf\n",
 "                        if conf >= ALARM_CONF:\n"
 "                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2\n"
 "                            prev = next((t for t in alarm_tracks if t[\"cls\"] == name and\n"
 "                                         math.hypot(t[\"cx\"] - cx, t[\"cy\"] - cy) < ALARM_PX), None)\n"
 "                            tr = {\"cls\": name, \"cx\": cx, \"cy\": cy, \"n\": prev[\"n\"] + 1 if prev else 1,\n"
 "                                  \"counted\": prev[\"counted\"] if prev else False}\n"
 "                            if prev:\n"
 "                                alarm_tracks.remove(prev)\n"
 "                            if tr[\"n\"] >= FOUND_FRAMES and not tr[\"counted\"]:\n"
 "                                tr[\"counted\"] = True\n"
 "                                false_alarms.append([round(now - t0, 1), name, round(conf, 3),\n"
 "                                                     *(round(v, 2) for v in pos), round(yaw, 1)])\n"
 "                                img = frame.copy()\n"
 "                                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)\n"
 "                                cv2.putText(img, f\"{name} {conf:.2f}\", (int(x1) + 3, max(15, int(y1) - 6)),\n"
 "                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)\n"
 "                                cv2.imwrite(str(out / \"false_alarms\" / f\"{len(false_alarms):03d}_{name}.jpg\"), img)\n"
 "                            new_tracks.append(tr)\n"
 "                    else:\n"
 "                        evidence[hit[\"id\"]][name] += conf\n"),

("                col = (0, 200, 255) if hit is None else (60, 220, 90)\n"
 "                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)\n",
 "                weak = hit is None and conf < ALARM_CONF\n"
 "                col = (60, 220, 90) if hit else (170, 170, 170) if weak else (0, 200, 255)\n"
 "                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), col, 1 if weak else 2)\n"),

("            results = {}\n"
 "            for did in first_seen:\n",
 "            alarm_tracks = new_tracks\n"
 "            results = {}\n"
 "            for did in first_seen:\n"),

("            panel = pg.Surface((330, 128), pg.SRCALPHA)\n",
 "            panel = pg.Surface((330, 156), pg.SRCALPHA)\n"),

("                     (f\"defects found {len(first_seen)}/{len(decals)}\", (90, 230, 110))]\n",
 "                     (f\"defects found {len(first_seen)}/{len(decals)}\", (90, 230, 110)),\n"
 "                     (f\"false alarms {len(false_alarms)}  ({len(false_alarms) / max(fly_time / 60, 1e-6):.1f}/min)\"\n"
 "                      if fly_time > 5 else f\"false alarms {len(false_alarms)}\", (255, 140, 90))]\n"),

("    write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, time.time() - t0)\n",
 "    with open(out / \"false_alarms.csv\", \"w\", newline=\"\") as f:\n"
 "        w = csv.writer(f)\n"
 "        w.writerow([\"t_s\", \"class\", \"conf\", \"x\", \"y\", \"z\", \"yaw\"])\n"
 "        w.writerows(false_alarms)\n"
 "    write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, time.time() - t0,\n"
 "                 false_alarms, fly_time)\n"),

("def write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, duration):\n",
 "def write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, duration,\n"
 "                 false_alarms=(), fly_time=0.0):\n"),

("    print(f\"detections not on any defect patch (possible false alarms): {unmatched} frames\")\n",
 "    print(f\"all off-patch detections (any conf, frames): {unmatched}\")\n"
 "    fa_rate = len(false_alarms) / max(fly_time / 60, 1e-6)\n"
 "    print(f\"FALSE ALARMS (conf >= {ALARM_CONF}, {FOUND_FRAMES}+ frames): {len(false_alarms)}\"\n"
 "          f\"   = {fa_rate:.1f} per flying minute ({fly_time / 60:.1f} min flying)\")\n"
 "    for c, k in Counter(a[1] for a in false_alarms).most_common():\n"
 "        print(f\"   {c:17s}{k:5d}\")\n"),

("                 f\"(+{part} partly)\", fontsize=11)\n",
 "                 f\"(+{part} partly)\\nfalse alarms {len(false_alarms)} ({fa_rate:.1f}/min, \"\n"
 "                 f\"conf >= {ALARM_CONF})\", fontsize=11)\n"),

("    print(f\"\\nSaved in {out}:  report.png  findings.csv  findings/  video.mp4\")\n",
 "    print(f\"\\nSaved in {out}:  report.png  findings.csv  findings/  false_alarms/  video.mp4\")\n"),
]

missing = [old[:60] for old, _ in R if s.count(old) != 1]
if missing:
    sys.exit("pilot.py is not the expected version, nothing changed. Not found:\n  " + "\n  ".join(missing))
for old, new in R:
    s = s.replace(old, new)
p.write_text(s)
print("pilot.py patched (backup: pilot.py.bak)")
