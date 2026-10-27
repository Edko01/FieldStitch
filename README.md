# FieldStitch

Offline orthomosaics from drone mapping photos, built for the DJI Matrice 4E (RTK). Windows, no install, no internet needed to process.

Import a folder of nadir photos, process them into a georeferenced GeoTIFF orthomosaic, then split it into fields, grid tiles or imported boundaries (KML/KMZ/SHP/GeoJSON) and export one GeoTIFF per area.

## Download

Unzip the FieldStitch Windows package and double-click **Start FieldStitch.bat**. Installed copies update themselves from this repo via the **Check for updates** button (they read `updates/latest.json` on `main`).

## How it works

- SIFT features on reduced-resolution photos, guided matching using GNSS/RTK and gimbal priors
- Sparse bundle adjustment (Schur complement, Levenberg-Marquardt) with RTK position priors and self-calibrated Brown lens model
- Smoothed 2.5D surface from triangulated points, nadir-weighted seamlines with feathered blending and exposure balancing
- Tiled JPEG or lossless GeoTIFF output (WGS84 / UTM) with overviews

## Repository layout

- `app/` – the application (Python package `fieldstitch`, web UI in `app/web`)
- `tools/make_release.py` – builds the in-app update assets (`fieldstitch-app-<ver>.zip` + `update.json`)

The Windows package bundles a portable CPython 3.12 (python-build-standalone) with numpy, scipy, opencv-python-headless, rasterio, pyproj, pillow and pyshp in a `python/` folder next to `app/`.

## Releasing an update

1. Change code in `app/`, bump `__version__` in `app/fieldstitch/__init__.py`
   (bump `RUNTIME` only if the bundled `python/` folder changes; those updates need the full package).
2. `python tools/make_release.py "what changed"`
3. Commit `app/` and `updates/`, push to `main`. Installed copies see it within a few minutes.
