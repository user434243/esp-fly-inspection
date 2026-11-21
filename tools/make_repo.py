#!/usr/bin/env python3
"""
make_repo.py - collect the whole ESP-FLY inspection project into one tidy git repository.

  python3 ~/esp-fly-inspection/tools/make_repo.py      # update the repo from ~/drone_ws (run any time)
  (README.md and docs/ are written only on the first build - after that, edit them in the repo)
  python3 make_repo.py --out ~/myrepo

It copies from your PC (nothing is moved or changed in ~/drone_ws):
  scripts/            ~/drone_ws/scripts/*.py   (patch_*.py -> scripts/patches/)
  sim/worlds/         world generators + each world's world.json, decals.json, parts.json, maps
  sim/camera/         a11t_calib.json, camera_profile.json
  sim/patches, tools  add_camera.py, fetch_assets.py ...
  notebooks/          every *.ipynb of the project
  results/runs/       EVERY flight in ~/drone_ws/logs: report.png + all .csv + 6 finding photos
                      (no videos - too big for git)
  results/training/   training curves, confusion matrices, results.csv, args.yaml of every YOLO run
  results/datasets/   examples.jpg + data.yaml of every dataset on the HDD
  results/highlights/ the pictures the README shows
  models/README.md    list of the trained .pt files with size + checksum (the files go to a release)
  setup/              requirements.txt (pip freeze), aliases.sh (from ~/.bashrc), environment.txt
  README.md docs/ LICENSE .gitignore   (from this kit)
Then: a secret scan (passwords, tokens ...), a size check, git init + commit.
Plug in the HDD first (logs, runs and datasets live there).
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
DW = HOME / "drone_ws"
HDD = Path("/media/hudaifa/Ventoy")
KIT = Path(__file__).resolve().parent           # the repo_kit folder, or <repo>/tools
SKIP_DIRS = {"venv", ".venv", "site-packages", ".ipynb_checkpoints", "__pycache__", "CrazySim", ".git",
             "images", "labels", "meshes", "textures", "survey", "node_modules"}
CITY_RUN, ROAD_RUN = "auto_20261121_140746", "auto_20261119_225040"
copied = []


def log(msg):
    print(msg, flush=True)


def cp(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(dst)
    return True


def walk(root, name_ok, max_depth=6):
    """Files under root whose name passes name_ok, skipping big / irrelevant folders."""
    root = Path(root)
    if not root.exists():
        return
    base = len(root.parts)
    for d, dirs, files in os.walk(root, followlinks=True):
        if len(Path(d).parts) - base >= max_depth:
            dirs[:] = []
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
        for f in files:
            if name_ok(f):
                yield Path(d) / f


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


# ---------------- code ----------------
def collect_code(R):
    n = 0
    for f in sorted((DW / "scripts").glob("*.py")):
        n += cp(f, R / ("scripts/patches" if f.name.startswith("patch_") else "scripts") / f.name)
    for f in sorted((DW / "scripts").glob("*.sh")):
        n += cp(f, R / "scripts" / f.name)
    log(f"  scripts: {n} files")
    worlds = DW / "sim/worlds"
    for f in sorted(worlds.glob("*.py")):
        cp(f, R / "sim/worlds" / f.name)
    for wd in sorted(p for p in worlds.iterdir() if p.is_dir()) if worlds.exists() else []:
        if not ((wd / "world.json").exists() or (wd / "decals.json").exists()):
            continue
        for f in wd.iterdir():
            if f.is_file() and (f.suffix in (".json", ".png") or f.suffix == ".jpg" and f.stat().st_size < 3e6):
                cp(f, R / "sim/worlds" / wd.name / f.name)
        log(f"  world: {wd.name}")
    for sub in ("tools", "patches"):
        for f in sorted((DW / "sim" / sub).glob("*.py")):
            cp(f, R / "sim" / sub / f.name)
    for f in walk(DW / "sim", lambda n: n == "add_camera.py", 3):
        cp(f, R / "sim/patches/add_camera.py")
    for name in ("a11t_calib.json", "camera_profile.json", "active_world"):
        cp(DW / "sim" / name, R / "sim/camera" / name)
    nb = 0
    for f in walk(DW, lambda n: n.endswith(".ipynb"), 5):
        if f.stat().st_size > 40e6:
            out = R / "notebooks" / f.name
            out.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([str(DW / "venv/bin/jupyter"), "nbconvert", "--clear-output", "--to", "notebook",
                                "--output", str(out), str(f)], capture_output=True)
            log(f"  notebook {f.name}: {f.stat().st_size / 1e6:.0f} MB -> outputs cleared"
                + ("" if r.returncode == 0 else " (FAILED, skipped)"))
        else:
            cp(f, R / "notebooks" / f.name)
        nb += 1
    log(f"  notebooks: {nb}")


# ---------------- results ----------------
def collect_results(R):
    logs = DW / "logs"
    if not logs.exists() or not any(logs.iterdir()):
        log("  !! ~/drone_ws/logs is empty or the HDD is not plugged in - no flight results copied")
    runs = sorted(p for p in logs.iterdir() if p.is_dir()) if logs.exists() else []
    for run in runs:
        for f in run.iterdir():
            if f.is_file() and f.suffix in (".png", ".csv", ".txt", ".json") and f.stat().st_size < 8e6:
                cp(f, R / "results/runs" / run.name / f.name)
        for sub, k in (("findings", 6), ("false_alarms", 3)):
            for f in sorted((run / sub).glob("*.jpg"))[:k] if (run / sub).exists() else []:
                cp(f, R / "results/runs" / run.name / sub / f.name)
    log(f"  flights: {len(runs)} log folders")
    # highlights for the README
    hl = R / "results/highlights"
    cp(logs / CITY_RUN / "report.png", hl / "city_report.png")
    cp(logs / ROAD_RUN / "report.png", hl / "road_report.png")
    bridge = [r for r in runs if (r / "findings.csv").exists() and (r / "report.png").exists()
              and "\nD" in (r / "findings.csv").read_text()[:200] and r.name.startswith("auto_")]
    if bridge:
        cp(bridge[-1] / "report.png", hl / "bridge_report.png")
    # training runs (YOLO): every folder with results.csv + weights/
    seen = set()
    for root in (DW / "yolo", DW, HDD):
        for f in walk(root, lambda n: n == "results.csv", 6):
            run = f.parent
            if not (run / "weights").exists() or run in seen:
                continue
            seen.add(run)
            for name in ("results.png", "results.csv", "args.yaml", "confusion_matrix_normalized.png",
                         "confusion_matrix.png", "BoxPR_curve.png", "PR_curve.png", "labels.jpg"):
                cp(run / name, R / "results/training" / run.name / name)
    log(f"  training runs: {len(seen)}")
    nd = 0
    for d in sorted((HDD / "datasets").iterdir()) if (HDD / "datasets").exists() else []:
        if d.is_dir():
            nd += cp(d / "examples.jpg", R / "results/datasets" / f"{d.name}_examples.jpg")
            cp(d / "data.yaml", R / "results/datasets" / f"{d.name}_data.yaml")
    log(f"  dataset examples: {nd}")
    for name in ("A11T_lens_before_after.jpg", "sim_lens_demo.jpg"):
        if not (R / "results/calibration" / name).exists():
            cp(KIT / "results/calibration" / name, R / "results/calibration" / name) or \
                cp(HOME / "Downloads" / name, R / "results/calibration" / name)
    figs = R / "scripts/make_figures.py"
    if figs.exists():                                        # refresh the report figures
        subprocess.run([sys.executable, str(figs)], capture_output=True)


# ---------------- models ----------------
def collect_models(R):
    rows = []
    for f in sorted((DW / "yolo/models").glob("*.pt")):
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        rows.append(f"| `{f.name}` | {f.stat().st_size / 1e6:.1f} MB | `{h}` |")
    text = ("# Trained models\n\nThe weights are not in git (too big). They are attached to the GitHub release.\n\n"
            "Upload: on the repo page -> Releases -> Draft a new release -> tag `v1.0` -> drag the files from "
            "`~/drone_ws/yolo/models/` in -> Publish.  (or with the GitHub CLI: `gh release create v1.0 "
            "~/drone_ws/yolo/models/*.pt`)\n\nUse: download them into `~/drone_ws/yolo/models/`.\n\n| file | size | sha256 (first 16) |\n|---|---|---|\n"
            + "\n".join(rows) + "\n\nSee docs/04_models.md for what each one is.\n")
    (R / "models").mkdir(parents=True, exist_ok=True)
    (R / "models/README.md").write_text(text)
    log(f"  models listed: {len(rows)}")


# ---------------- setup ----------------
def collect_setup(R):
    S = R / "setup"
    S.mkdir(parents=True, exist_ok=True)
    pip = DW / "venv/bin/pip"
    req = [l for l in sh(f"{pip} freeze").splitlines() if l and not l.startswith("-e") and " @ file" not in l]
    (S / "requirements.txt").write_text("\n".join(req) + "\n")
    keys = ("drone_ws", "DRONE_WS", "CrazySim", "sim", "pilot", "camprofile", "lab", "simworld")
    bad = re.compile(r"pass|token|secret|key|rustdesk", re.I)
    lines = []
    bashrc = HOME / ".bashrc"
    for l in bashrc.read_text().splitlines() if bashrc.exists() else []:
        if (l.startswith("alias ") or l.startswith("export DRONE_WS")) and any(k in l for k in keys) \
                and not bad.search(l):
            lines.append(l)
    (S / "aliases.sh").write_text("# shortcuts used in this project (from ~/.bashrc)\n" + "\n".join(lines) + "\n")
    py = f"{DW}/venv/bin/python"
    torch_v = sh(py + ' -c "import torch; print(torch.__version__)"')
    yolo_v = sh(py + ' -c "import ultralytics; print(ultralytics.__version__)"')
    env = [f"OS:      {sh('lsb_release -ds')}  (kernel {sh('uname -r')})",
           f"GPU:     {sh('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader')}",
           f"Python:  {sh(py + ' --version')}",
           f"Gazebo:  {sh('gz sim --version')}",
           f"Torch:   {torch_v}",
           f"YOLO:    {yolo_v}"]
    (S / "environment.txt").write_text("\n".join(env) + "\n")
    log(f"  setup: {len(req)} packages, {len(lines)} aliases")


# ---------------- checks ----------------
SECRET = re.compile(r"((password|passwd|pwd)\s*[=:]|token\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{12,}|secret\s*[=:]|api[_-]?key\s*[=:]|"
                    r"rustdesk|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_)", re.I)


def secret_scan(R):
    hits = []
    for f in R.rglob("*"):
        if ".git" in f.parts or not f.is_file() or f.suffix not in (".py", ".sh", ".md", ".txt", ".json", ".yaml", ".ipynb", ".csv"):
            continue
        if f.stat().st_size > 5e6:
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if SECRET.search(line):
                hits.append(f"{f.relative_to(R)}:{i}: {line.strip()[:110]}")
    return hits


def size_check(R):
    files = [f for f in R.rglob("*") if f.is_file() and ".git" not in f.parts]
    total = sum(f.stat().st_size for f in files)
    big = sorted(files, key=lambda f: -f.stat().st_size)[:5]
    log(f"\nrepo size {total / 1e6:.0f} MB in {len(files)} files; biggest:")
    for f in big:
        log(f"   {f.stat().st_size / 1e6:6.1f} MB  {f.relative_to(R)}")
    too_big = [f for f in files if f.stat().st_size > 95e6]
    for f in too_big:
        log(f"  !! {f.relative_to(R)} is over GitHub's 100 MB limit - removed")
        f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HOME / "esp-fly-inspection")
    ap.add_argument("--no-git", action="store_true")
    a = ap.parse_args()
    R = a.out.expanduser()
    R.mkdir(parents=True, exist_ok=True)
    log(f"Building {R}")
    for name in ("README.md", "LICENSE", ".gitignore"):   # only on the first build: never overwrite
        if not (R / name).exists():                          # README / docs edited in the repo
            cp(KIT / name, R / name)
    for f in (KIT / "docs").glob("*.md"):
        if not (R / "docs" / f.name).exists():
            cp(f, R / "docs" / f.name)
    log("code ...")
    collect_code(R)
    log("results ...")
    collect_results(R)
    log("models ...")
    collect_models(R)
    log("setup ...")
    collect_setup(R)
    size_check(R)
    hits = secret_scan(R)
    if hits:
        log("\n!! POSSIBLE SECRETS - check these lines, remove them, then run make_repo.py again:")
        for h in hits[:40]:
            log("   " + h)
        log("(git commit NOT done)")
        return
    log("\nsecret scan: clean")
    if a.no_git:
        return
    if not (R / ".git").exists():
        sh(f"git -C {R} init -b main")
    if not sh("git config --global user.name"):
        sh(f"git -C {R} config user.name 'Hudaifa'")
        sh(f"git -C {R} config user.email 'user434243@users.noreply.github.com'")
    sh(f"git -C {R} add -A")
    out = sh(f"git -C {R} commit -m 'ESP-FLY inspection: code, worlds, results, docs'")
    log(out.splitlines()[0] if out else "nothing new to commit")
    log(f"""
DONE. Next (one time) - put it on GitHub:
  1. github.com -> New repository -> name: esp-fly-inspection -> Private or Public -> Create
     (do NOT add a README there)
  2. cd {R}
     git remote add origin https://github.com/user434243/esp-fly-inspection.git
     git push -u origin main
     (user name: user434243, password: a Personal Access Token from
      github.com -> Settings -> Developer settings -> Personal access tokens -> 'repo' scope)
  3. models: see models/README.md (upload the .pt files to a release)
Later updates: cd {R} && git pull ; python3 tools/make_repo.py ; git push""")


if __name__ == "__main__":
    main()
