# FieldStitch

Offline orthomosaics from drone mapping photos, built for the DJI Matrice 4E (RTK). Windows, no install, no internet needed to process.

Import a folder of nadir photos, process them into a georeferenced GeoTIFF orthomosaic, then split it into fields, grid tiles or imported boundaries (KML/KMZ/SHP/GeoJSON) and export one GeoTIFF per area.

## Install

Run `FieldStitch-Setup-<version>.exe`. It installs per-user to `%LOCALAPPDATA%\Programs\FieldStitch` (no admin prompt), adds Start-menu and optional desktop shortcuts and an Apps & features uninstaller. FieldStitch opens in its own window (Edge/Chrome app mode) with no console, and closes itself when its window is closed and nothing is processing.

Installed copies update themselves from this repo via **Check for updates** (they read `updates/latest.json` on `main`).

## How it works

- SIFT features on reduced-resolution photos (on NVIDIA GPUs via CuPy - `gsift.py`, identical to OpenCV's SIFT - with CPU workers in parallel), guided matching using GNSS/RTK and gimbal priors
- Sparse bundle adjustment (Schur complement, Levenberg-Marquardt) with RTK position priors and self-calibrated Brown lens model
- Smoothed 2.5D surface from triangulated points, nadir-weighted seamlines with feathered blending and exposure balancing
- Tiled JPEG or lossless GeoTIFF output (WGS84 / UTM); overviews in an external `.ovr` built from a half-resolution copy written during rendering

## Repository layout

- `app/` – the application (Python package `fieldstitch`, web UI in `app/web`)
- `tools/make_release.py` – builds the in-app update assets (`fieldstitch-app-<ver>.zip` + `update.json`)

The Windows package bundles a portable CPython 3.12 (python-build-standalone) with numpy, scipy, opencv-python-headless, rasterio, pyproj, pillow and pyshp in a `python/` folder next to `app/`. Runtime 2 (FieldStitch 1.5+) adds cupy-cuda12x, cuda-pathfinder, nvidia-cuda-runtime-cu12 and nvidia-cuda-nvrtc-cu12 (no cuBLAS needed); it needs an NVIDIA driver for CUDA 12 and falls back to the CPU otherwise.

Updates that change the runtime (RUNTIME in `__init__.py`) can't be installed in-app; installed copies show the update with a note to run the new installer, which is too large to keep in this repo and is distributed separately.

## Building the installer

1. Stage a folder with `python/` (portable CPython 3.12 + libraries, plus `fieldstitch_runtime.txt`), `app/`, and from `installer/`: `FieldStitch.ico`, `README.txt`, `welcome.bmp`, `header.bmp`.
2. `makensis -DVERSION=<ver> -DSTAGE=<stage folder> -DOUTFILE=FieldStitch-Setup-<ver>.exe installer/FieldStitch.nsi` (NSIS 3, builds on Linux or Windows).

## Releasing an update

1. Change code in `app/`, bump `__version__` in `app/fieldstitch/__init__.py`
   (bump `RUNTIME` only if the bundled `python/` folder changes; those updates need the full package).
2. `python tools/make_release.py "what changed"`
3. Commit `app/` and `updates/`, push to `main`. Installed copies see it within a few minutes.
