#!/usr/bin/env python3
"""
fetch_assets.py - download free, realistic assets for the bridge simulator.

1) PBR TEXTURES from Poly Haven (CC0 = free for any use, no account needed):
   colour (albedo), surface bumps (normal map) and roughness, 2K resolution.
   One texture set per surface type: bridge concrete, weathered concrete, asphalt,
   grass/ground, river gravel, rusty metal.
2) 3D MODELS from Gazebo Fuel (OpenRobotics, free): trees, cars, cones, barriers.
   They are cached in ~/.gz/fuel so the world also loads offline later.

Output:  ~/drone_ws/sim/assets/textures/<role>/{albedo,normal,roughness}.jpg
         ~/drone_ws/sim/assets/assets.json   (what was downloaded, with licence info)

Run:  ~/drone_ws/venv/bin/python ~/drone_ws/sim/tools/fetch_assets.py
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ASSETS = Path.home() / "drone_ws/sim/assets"
UA = {"User-Agent": "drone-ws-bridge-sim/1.0 (university graduation project)"}
API = "https://api.polyhaven.com"

# role -> keywords to look for in Poly Haven texture names (first match wins)
TEXTURE_ROLES = {
    "concrete_bridge":  ["concrete_wall", "concrete_panels", "concrete"],
    "concrete_weathered": ["dirty_concrete", "worn_concrete", "concrete_floor_worn", "concrete_rough",
                           "concrete"],
    "asphalt":          ["asphalt", "road"],
    "ground":           ["aerial_grass", "forest_ground", "grass", "brown_mud", "ground"],
    "gravel":           ["gravel", "pebble", "rocky", "stones"],
    "rust_metal":       ["rust", "corrugated", "metal"],
}
MAPS = {"albedo": "Diffuse", "normal": "nor_gl", "roughness": "Rough"}

FUEL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/"
FUEL_MODELS = ["Oak tree", "Pine Tree", "Hatchback", "SUV", "Pickup",
               "Construction Cone", "Jersey Barrier", "Lamp Post"]


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        dest.write_bytes(r.read())


def pick_textures(all_textures):
    """Choose one Poly Haven texture id per role, never the same id twice."""
    ids = sorted(all_textures)
    chosen, used = {}, set()
    for role, keywords in TEXTURE_ROLES.items():
        for kw in keywords:
            hit = next((i for i in ids if kw in i and i not in used), None)
            if hit:
                chosen[role] = hit
                used.add(hit)
                break
    return chosen


def fetch_textures(report):
    print("Asking Poly Haven for its texture list ...")
    all_textures = get_json(f"{API}/assets?type=textures")
    print(f"  {len(all_textures)} textures available")
    for role, tid in pick_textures(all_textures).items():
        files = get_json(f"{API}/files/{tid}")
        out = ASSETS / "textures" / role
        got = []
        for name, key in MAPS.items():
            try:
                url = files[key]["2k"]["jpg"]["url"]
            except KeyError:
                continue
            download(url, out / f"{name}.jpg")
            got.append(name)
        report["textures"][role] = {"polyhaven_id": tid, "maps": got, "licence": "CC0",
                                    "page": f"https://polyhaven.com/a/{tid}"}
        print(f"  {role:20s} <- {tid:30s} {', '.join(got)}")
    missing = [r for r in TEXTURE_ROLES if r not in report["textures"]]
    if missing:
        print("  !! no match for:", ", ".join(missing), "(plain colours will be used)")


def fetch_fuel(report):
    print("\nDownloading 3D models from Gazebo Fuel ...")
    for name in FUEL_MODELS:
        url = FUEL + name.replace(" ", "%20")
        r = subprocess.run(["gz", "fuel", "download", "-u", url], capture_output=True, text=True)
        ok = r.returncode == 0
        report["fuel"][name] = {"uri": FUEL + name, "ok": ok}
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    report = {"textures": {}, "fuel": {}}
    try:
        fetch_textures(report)
    except Exception as e:
        print("!! texture download failed:", e)
    try:
        fetch_fuel(report)
    except FileNotFoundError:
        print("!! 'gz' command not found - is Gazebo installed?")
    (ASSETS / "assets.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved list: {ASSETS / 'assets.json'}")
    if not report["textures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
