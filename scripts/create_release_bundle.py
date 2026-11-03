#!/usr/bin/env python3
"""
OpenStat Studio v3.0 — GitHub Release & Distribution Bundler
Generates a clean, self-contained distribution archive excluding local venv and caches.
"""

import os
import sys
import zipfile
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
ZIP_NAME = "OpenStat-Studio-v3.0.0.zip"
ZIP_PATH = os.path.join(DIST_DIR, ZIP_NAME)

EXCLUDE_DIRS = {
    ".venv", "venv", ".pytest_cache", "__pycache__",
    ".git", ".vscode", ".idea", "dist", "build",
}

EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".egg-info"
}

EXCLUDE_FILES = {
    ".DS_Store", "Thumbs.db"
}


def build_bundle():
    os.makedirs(DIST_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"[1/3] Packaging OpenStat Studio into {ZIP_PATH} ...")
    file_count = 0

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]

            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                    continue
                if root.endswith("workspace") and file in ("live_session.json", "live_session.py", "doctor_env_cache.json"):
                    continue
                if root.endswith("reports") and file.endswith((".log", ".pid", ".omv")):
                    continue

                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, PROJECT_ROOT)

                # Store with Unix-style slashes
                arcname = "OpenStat-Studio/" + rel_path.replace(os.sep, "/")
                zf.write(abs_path, arcname)
                file_count += 1

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    with open(ZIP_PATH, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"[2/3] Verification: {file_count} files packaged successfully.")
    print(f"[3/3] Distribution Archive Ready:")
    print(f"      - Path   : {ZIP_PATH}")
    print(f"      - Size   : {size_mb:.2f} MB (Extremely lightweight & GitHub release ready!)")
    print(f"      - SHA-256: {sha256}")


if __name__ == "__main__":
    build_bundle()
