#!/usr/bin/env python3
"""patch_autopilot2.py - autopilot v2: active inspection at every stop.
  * views: stop point (1.0 m) -> closer (0.6 m) -> closest (0.45 m) -> look left/right (+-15 deg)
    stops as soon as the defect is FOUND and held 2.5 s (at least 2 views)
  * the class is decided by the close-up scan frames (frames seen in passing count 25 %)
  * mission_log.csv gets diagnostics per stop: position/heading error, best confidence
"""
import shutil
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "drone_ws/scripts/autopilot.py"
s = p.read_text()
if "VIEW_TIME" in s:
    sys.exit("already patched")
shutil.copy(p, p.with_suffix(".py.v1"))

R = [
("HOLD_FOUND, HOLD_MAX = 2.5, 6.0              # seconds at a stop: after FOUND / at most\n",
 "HOLD_FOUND = 2.5                             # seconds to keep scanning after FOUND (class votes)\n"
 "VIEW_TIME = 2.0                              # seconds per view when nothing is found yet\n"
 "CLOSER = (0.6, 0.45)                         # extra views closer to the defect (m from surface)\n"
 "SWEEP = 15.0                                 # look left / right by this much (deg)\n"
 "PASSING_WEIGHT = 0.25                        # class vote of frames NOT taken during the scan\n"),

("        self.t0 = None\n",
 "        self.t0 = None\n"
 "        self.scan_id = None                  # defect being scanned right now\n"
 "        self.best = {}                       # scan: best confidence on the target / elsewhere\n"),

("                self.phase = f\"TURNING to {it['id']}\"\n"
 "                self.turn_to(it[\"yaw\"])\n"
 "                self.phase = f\"SCANNING {it['id']}\"\n"
 "                self.hold_t = time.time()\n"
 "                while True:\n"
 "                    self.wait_paused()\n"
 "                    held = time.time() - self.hold_t\n"
 "                    found = it[\"id\"] in self.first_seen\n"
 "                    if (found and held > HOLD_FOUND) or held > HOLD_MAX:\n"
 "                        break\n"
 "                    self.tick()\n"
 "                self.log.append((it[\"id\"], found, round(time.time() - self.t0, 1)))\n"
 "                self.hold_t = None\n",
 "                self.phase = f\"TURNING to {it['id']}\"\n"
 "                self.turn_to(it[\"yaw\"])\n"
 "                err_p = float(np.linalg.norm(self.pos() - it[\"pos\"]))\n"
 "                err_y = abs(wrap(self.yaw - it[\"yaw\"]))\n"
 "                found, n_views = self.scan(it)\n"
 "                b = self.best.get(it[\"id\"], {})\n"
 "                self.log.append((it[\"id\"], found, round(time.time() - self.t0, 1), round(err_p, 3),\n"
 "                                 round(err_y, 1), n_views, round(b.get(\"on\", 0), 2), round(b.get(\"off\", 0), 2)))\n"),

("    # -- the mission --\n",
 "    def scan(self, it):\n"
 "        \"\"\"Active inspection: look from the stop, then closer, then left/right, until FOUND.\"\"\"\n"
 "        c, n, yaw0 = it[\"decal\"][\"_c\"], it[\"decal\"][\"_n\"], it[\"yaw\"]\n"
 "        views, close = [(it[\"pos\"], yaw0, \"1st view\")], it[\"pos\"]\n"
 "        for d in CLOSER:\n"
 "            q = c + n * d\n"
 "            if d < it[\"standoff\"] - 0.08 and self.planner.free(q) and self.planner.line_free(close, q):\n"
 "                views.append((q, yaw0, f\"closer {d:.2f} m\"))\n"
 "                close = q\n"
 "        views += [(close, yaw0 + SWEEP, \"look left\"), (close, yaw0 - SWEEP, \"look right\")]\n"
 "        self.scan_id, self.best[it[\"id\"]] = it[\"id\"], {}\n"
 "        t_found, k = None, 0\n"
 "        for k, (q, yw, label) in enumerate(views):\n"
 "            self.phase = f\"SCANNING {it['id']} - {label}\"\n"
 "            if k:\n"
 "                if np.linalg.norm(q - self.pos()) > 0.04:\n"
 "                    self.fly_legs([self.pos(), q])\n"
 "                self.turn_to(yw)\n"
 "            self.hold_t = time.time()\n"
 "            while True:\n"
 "                self.wait_paused()\n"
 "                if t_found is None and it[\"id\"] in self.first_seen:\n"
 "                    t_found = time.time()\n"
 "                if t_found and time.time() - t_found >= HOLD_FOUND:\n"
 "                    break\n"
 "                if t_found is None and time.time() - self.hold_t >= VIEW_TIME:\n"
 "                    break\n"
 "                self.tick()\n"
 "            if t_found and time.time() - t_found >= HOLD_FOUND and (k >= 1 or len(views) == 1):\n"
 "                break\n"
 "        self.hold_t = None\n"
 "        if np.linalg.norm(self.pos() - it[\"pos\"]) > 0.05:      # back to the stop = back on the plan\n"
 "            self.phase = f\"BACKING OUT from {it['id']}\"\n"
 "            self.fly_legs([self.pos(), it[\"pos\"]])\n"
 "        self.scan_id = None\n"
 "        return it[\"id\"] in self.first_seen, k + 1\n"
 "\n"
 "    # -- the mission --\n"),

("            print(\"!! mission error:\", e)\n",
 "            import traceback\n"
 "            traceback.print_exc()\n"
 "            print(\"!! mission error:\", e)\n"),

("                        evidence[hit[\"id\"]][name] += conf\n"
 "                        frames_on[hit[\"id\"]] += 1\n",
 "                        scanning = m.scan_id == hit[\"id\"]\n"
 "                        evidence[hit[\"id\"]][name] += conf * (1.0 if scanning else PASSING_WEIGHT)\n"
 "                        frames_on[hit[\"id\"]] += 1\n"
 "                        if scanning:\n"
 "                            b = m.best.setdefault(hit[\"id\"], {})\n"
 "                            b[\"on\"] = max(b.get(\"on\", 0), conf)\n"
 "                if m.scan_id and hit is None and m.flying:\n"
 "                    b = m.best.setdefault(m.scan_id, {})\n"
 "                    b[\"off\"] = max(b.get(\"off\", 0), conf)\n"),

("        w.writerow([\"id\", \"detected_at_stop\", \"t_s\"])\n",
 "        w.writerow([\"id\", \"found\", \"t_s\", \"pos_err_m\", \"yaw_err_deg\", \"views\", \"best_conf_on_target\",\n"
 "                    \"best_conf_elsewhere\"])\n"),

("    print(f\"\\nAUTONOMOUS MISSION: {len(m.log)}/{n_stops} stops scanned, \"\n"
 "          f\"{sum(f for _, f, _ in m.log)} detected at the stop, \"\n"
 "          f\"mission time {(m.log[-1][2] if m.log else 0) / 60:.1f} min\")\n",
 "    print(\"\\nper stop:  id   found  pos_err  yaw_err  views  conf_on_target  conf_elsewhere\")\n"
 "    for r in m.log:\n"
 "        print(f\"           {r[0]}  {'yes' if r[1] else 'NO ':5s} {r[3]:6.2f} m {r[4]:6.1f} deg {r[5]:4d}\"\n"
 "              f\"  {r[6]:12.2f}  {r[7]:12.2f}\")\n"
 "    print(f\"\\nAUTONOMOUS MISSION: {len(m.log)}/{n_stops} stops scanned, \"\n"
 "          f\"{sum(r[1] for r in m.log)} found, \"\n"
 "          f\"mission time {(m.log[-1][2] if m.log else 0) / 60:.1f} min\")\n"),
]

missing = [old[:70] for old, _ in R if s.count(old) != 1]
if missing:
    sys.exit("autopilot.py is not the expected version, nothing changed. Not found:\n  " + "\n  ".join(missing))
for old, new in R:
    s = s.replace(old, new)
p.write_text(s)
print("autopilot.py updated to v2 (backup: autopilot.py.v1)")
