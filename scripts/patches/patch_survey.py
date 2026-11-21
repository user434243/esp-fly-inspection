# patch_survey.py - adds SURVEY mode to autopilot.py:  SURVEY=1 autopilot
#   * every stop is scanned with ALL views (no early exit), so every model is judged on the same views
#   * the CLEAN camera frames + drone position/heading are saved (5 per second while scanning,
#     1 per second while flying) -> logs/auto_<date>/survey/   (analog look is added later, offline)
import shutil, sys
from pathlib import Path
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "drone_ws/scripts/autopilot.py"
s = p.read_text()
if "SURVEY" in s:
    sys.exit("already patched - nothing changed")
patches = [
    ('PASSING_WEIGHT = 0.25 ',
     'SURVEY = os.environ.get("SURVEY") == "1"      # SURVEY=1 autopilot: full scans + save clean frames\n'
     'PASSING_WEIGHT = 0.25 '),
    ('                if t_found is None and it["id"] in self.first_seen:\n',
     '                if t_found is None and it["id"] in self.first_seen and not SURVEY:\n'),
    ('    (out / "findings").mkdir(parents=True, exist_ok=True)\n    print("Loading model ...")\n',
     '    (out / "findings").mkdir(parents=True, exist_ok=True)\n'
     '    survey_dir, survey_rows = out / "survey", []\n'
     '    if SURVEY:\n'
     '        survey_dir.mkdir()\n'
     '        print("SURVEY MODE: full scans at every stop, clean frames are saved")\n'
     '    print("Loading model ...")\n'),
    ('    if pilot.CAM_PROFILE.get("analog"):                        # real A11T analog video look\n',
     '    if pilot.CAM_PROFILE.get("analog") and not SURVEY:         # real A11T analog video look\n'),
    ('            pos, yaw, now = d.real, m.yaw, time.time()\n',
     '            pos, yaw, now = d.real, m.yaw, time.time()\n'
     '            if SURVEY and m.flying and n_frames % (4 if m.scan_id else 20) == 0:\n'
     '                fn = f"{len(survey_rows):05d}.jpg"\n'
     '                cv2.imwrite(str(survey_dir / fn), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])\n'
     '                survey_rows.append([fn, round(now - t0, 2), *(round(float(v), 4) for v in pos),\n'
     '                                    round(float(yaw), 2), m.scan_id or "", m.phase])\n'),
    ('    write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, time.time() - t0,\n',
     '    if SURVEY:\n'
     '        import json\n'
     '        with open(survey_dir / "frames.csv", "w", newline="") as f:\n'
     '            w = csv.writer(f)\n'
     '            w.writerow(["file", "t_s", "x", "y", "z", "yaw", "scan_id", "phase"])\n'
     '            w.writerows(survey_rows)\n'
     '        (survey_dir / "camera.json").write_text(json.dumps(pilot.CAM_PROFILE))\n'
     '        print(f"SURVEY: {len(survey_rows)} clean frames saved in {survey_dir}")\n'
     '    write_report(out, decals, parts, evidence, frames_on, first_seen, unmatched, path, time.time() - t0,\n'),
]
missing = [old[:70] for old, _ in patches if s.count(old) != 1]
if missing:
    sys.exit("!! autopilot.py is not the expected version, nothing changed. Not found:\n   " + "\n   ".join(missing))
shutil.copy(p, p.with_suffix(".py.presurvey"))
for old, new in patches:
    s = s.replace(old, new)
p.write_text(s)
print("autopilot.py: SURVEY mode added (backup: autopilot.py.presurvey)")
