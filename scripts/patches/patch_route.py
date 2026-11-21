#!/usr/bin/env python3
"""patch_route.py - add the demo-route guide (route.py) to pilot.py.
Shows the next stop: magenta outline on the camera view, directions at the bottom, magenta ring
on the mini-map. At a stop: hold still; it moves on when the patch is found + 2.5 s (or after 6 s).
Keys: N next stop, B previous stop, R route guide on/off."""
import shutil
import sys
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "drone_ws/scripts/pilot.py"
s = p.read_text()
if "compute_stops" in s:
    sys.exit("already patched")
if "ALARM_CONF" not in s:
    sys.exit("run patch_alarms.py first")
shutil.copy(p, p.with_suffix(".py.bak2"))

R = [
("ALARM_PX = 60                              # ... and if it stays (same class, centre moves < 60 px)\n",
 "ALARM_PX = 60                              # ... and if it stays (same class, centre moves < 60 px)\n"
 "HOLD_FOUND, HOLD_MAX = 2.5, 6.0            # route: seconds at a stop after FOUND / at most\n"),

("def ray_box(origin, d, lo, hi):\n",
 "def project(P, pos, yaw_deg):\n"
 "    \"\"\"World point -> image pixel (u, v), or None if behind the camera.\"\"\"\n"
 "    yaw = math.radians(yaw_deg)\n"
 "    c, s = math.cos(yaw), math.sin(yaw)\n"
 "    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])\n"
 "    b = rot.T @ (np.asarray(P, float) - np.asarray(pos, float) - rot @ CAM_OFFSET)\n"
 "    if b[0] < 0.1:\n"
 "        return None\n"
 "    u, v = IMG_W / 2 - b[1] / b[0] * FOCAL, IMG_H / 2 - b[2] / b[0] * FOCAL\n"
 "    return int(max(-5000, min(5000, u))), int(max(-5000, min(5000, v)))\n"
 "\n"
 "\n"
 "def ray_box(origin, d, lo, hi):\n"),

("def draw_minimap(pg, parts, decals, results, path, pos, yaw):\n",
 "def draw_minimap(pg, parts, decals, results, path, pos, yaw, target=None):\n"),

("    px, py = to_px(pos[0], pos[1])\n"
 "    a = math.radians(yaw)\n",
 "    if target is not None:\n"
 "        tx, ty = to_px(target[0], target[1])\n"
 "        pg.draw.line(surf, (255, 80, 255), to_px(pos[0], pos[1]), (tx, ty), 1)\n"
 "        pg.draw.circle(surf, (255, 80, 255), (tx, ty), 6, 2)\n"
 "    px, py = to_px(pos[0], pos[1])\n"
 "    a = math.radians(yaw)\n"),

("        \"KEYS TAB precision  P photo  H help  ESC land + finish\"]\n",
 "        \"KEYS TAB precision  P photo  H help  ESC land + finish\",\n"
 "        \"KEYS N next stop  B previous stop  R route guide on/off\"]\n"),

("    decals, parts, boxes = load_world()\n",
 "    decals, parts, boxes = load_world()\n"
 "    from route import compute_stops\n"
 "    route = compute_stops(decals, boxes)\n"
 "    n_stops = sum(st[\"kind\"] == \"stop\" for st in route)\n"
 "    route_i, route_on, hold_t, route_t, route_log, guide = 0, True, None, None, [], None\n"),

("                  pg.K_p: \"SNAPSHOT\", pg.K_ESCAPE: \"FINISH\"}\n",
 "                  pg.K_p: \"SNAPSHOT\", pg.K_ESCAPE: \"FINISH\", pg.K_n: \"NEXT\", pg.K_b: \"BACK\",\n"
 "                  pg.K_r: \"ROUTE\"}\n"),

("                elif act == \"FINISH\":\n",
 "                elif act in (\"NEXT\", \"BACK\"):\n"
 "                    step = 1 if act == \"NEXT\" else -1\n"
 "                    route_i, hold_t = max(0, min(len(route), route_i + step)), None\n"
 "                    while 0 < route_i < len(route) and route[route_i][\"kind\"] == \"via\" and step < 0:\n"
 "                        route_i -= 1\n"
 "                elif act == \"ROUTE\":\n"
 "                    route_on = not route_on\n"
 "                    message, message_t = f\"Route guide {'ON' if route_on else 'OFF'}\", time.time()\n"
 "                elif act == \"FINISH\":\n"),

("            alarm_tracks = new_tracks\n",
 "            alarm_tracks = new_tracks\n"
 "\n"
 "            # ---- demo route guide ----\n"
 "            guide = None\n"
 "            if route_on and route_i < len(route):\n"
 "                st = route[route_i]\n"
 "                if route_t is None and state == \"FLYING\":\n"
 "                    route_t = now\n"
 "                dx, dy, dz = st[\"pos\"] - np.asarray(pos, float)\n"
 "                ya = math.radians(yaw)\n"
 "                fwd, left = dx * math.cos(ya) + dy * math.sin(ya), -dx * math.sin(ya) + dy * math.cos(ya)\n"
 "                dyaw = None if st[\"yaw\"] is None else (st[\"yaw\"] - yaw + 180) % 360 - 180\n"
 "                at = False\n"
 "                if st[\"kind\"] == \"via\":\n"
 "                    if math.hypot(dx, dy) < 0.4 and abs(dz) < 0.3:\n"
 "                        route_i += 1\n"
 "                    p_ = project(st[\"pos\"], pos, yaw)\n"
 "                    if p_:\n"
 "                        cv2.circle(shown, p_, 12, (255, 80, 255), 2)\n"
 "                        cv2.putText(shown, \"waypoint\", (p_[0] + 15, p_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,\n"
 "                                    (255, 80, 255), 2)\n"
 "                else:\n"
 "                    at = math.hypot(dx, dy) < 0.3 and abs(dz) < 0.2 and abs(dyaw) < 15\n"
 "                    hold_t = (hold_t or now) if at else None\n"
 "                    held = now - hold_t if hold_t else 0.0\n"
 "                    found = st[\"id\"] in first_seen\n"
 "                    dc = st[\"decal\"]\n"
 "                    corners = [project(dc[\"_c\"] + a * dc[\"width\"] / 2 * dc[\"_h\"] +\n"
 "                                       b * dc[\"height\"] / 2 * np.array([0, 0, 1.0]), pos, yaw)\n"
 "                               for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]\n"
 "                    if all(corners):\n"
 "                        cv2.polylines(shown, [np.array(corners, np.int32)], True, (255, 80, 255), 2)\n"
 "                        top = min(corners, key=lambda q: q[1])\n"
 "                        cv2.putText(shown, f\"NEXT {st['id']}\", (top[0], max(15, top[1] - 8)),\n"
 "                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 255), 2)\n"
 "                    if (found and held > HOLD_FOUND) or held > HOLD_MAX:\n"
 "                        route_log.append((st[\"id\"], found, round(now - (route_t or now), 1)))\n"
 "                        message = f\"Stop {st['n']} {st['id']}: \" + (\"FOUND\" if found else \"not detected - next\")\n"
 "                        message_t, route_i, hold_t = now, route_i + 1, None\n"
 "                        if route_i == len(route):\n"
 "                            message = f\"ROUTE COMPLETE in {(now - (route_t or now)) / 60:.1f} min - ESC = land + report\"\n"
 "                guide = (st, fwd, left, dz, dyaw, at)\n"),

("            screen.blit(draw_minimap(pg, parts, decals, results, path, pos, yaw),\n",
 "            screen.blit(draw_minimap(pg, parts, decals, results, path, pos, yaw,\n"
 "                                     guide[0][\"pos\"] if guide else None),\n"),

("                screen.blit(hp, (8, screen.get_height() - hp.get_height() - 8))\n",
 "                screen.blit(hp, (8, screen.get_height() - hp.get_height() - (100 if guide else 8)))\n"
 "            if guide:\n"
 "                st, fwd, left, dz, dyaw, at = guide\n"
 "                if st[\"kind\"] == \"via\":\n"
 "                    head = f\"ROUTE  waypoint on the way to stop {st['n']}/{n_stops}  {st['id']}\"\n"
 "                else:\n"
 "                    head = f\"ROUTE  stop {st['n']}/{n_stops}  {st['id']}  {st['zone']}\"\n"
 "                if at and hold_t:\n"
 "                    todo, col2 = f\"HOLD STILL - scanning  {now - hold_t:3.1f} s\", (90, 230, 110)\n"
 "                else:\n"
 "                    todo = []\n"
 "                    if abs(fwd) > 0.1:\n"
 "                        todo.append(f\"{'forward' if fwd > 0 else 'back'} {abs(fwd):.1f} m\")\n"
 "                    if abs(left) > 0.1:\n"
 "                        todo.append(f\"{'left' if left > 0 else 'right'} {abs(left):.1f} m\")\n"
 "                    if abs(dz) > 0.1:\n"
 "                        todo.append(f\"{'up' if dz > 0 else 'down'} {abs(dz):.1f} m\")\n"
 "                    if dyaw is not None and abs(dyaw) > 8:\n"
 "                        todo.append(f\"turn {'left' if dyaw > 0 else 'right'} {abs(dyaw):.0f} deg\")\n"
 "                    todo, col2 = \"    \".join(todo) or \"almost there - fine-tune\", (255, 255, 255)\n"
 "                rp = pg.Surface((screen.get_width() - 16, 88), pg.SRCALPHA)\n"
 "                rp.fill((0, 0, 0, 165))\n"
 "                rp.blit(font.render(head, True, (255, 80, 255)), (10, 6))\n"
 "                rp.blit(big.render(todo, True, col2), (10, 31))\n"
 "                rp.blit(font.render(\"how: \" + st[\"tip\"], True, (200, 200, 200)), (10, 62))\n"
 "                screen.blit(rp, (8, screen.get_height() - 96))\n"),

("                 false_alarms, fly_time)\n",
 "                 false_alarms, fly_time)\n"
 "    if route_log:\n"
 "        print(f\"\\nDEMO ROUTE: {len(route_log)}/{n_stops} stops done, {sum(f for _, f, _ in route_log)} \"\n"
 "              f\"detected at the stop, {route_log[-1][2] / 60:.1f} min\")\n"
 "        with open(out / \"route_log.csv\", \"w\", newline=\"\") as f:\n"
 "            w = csv.writer(f)\n"
 "            w.writerow([\"id\", \"detected_at_stop\", \"t_s\"])\n"
 "            w.writerows(route_log)\n"),
]

missing = [old[:60] for old, _ in R if s.count(old) != 1]
if missing:
    sys.exit("pilot.py is not the expected version, nothing changed. Not found:\n  " + "\n  ".join(missing))
for old, new in R:
    s = s.replace(old, new)
p.write_text(s)
print("pilot.py patched (backup: pilot.py.bak2)")
