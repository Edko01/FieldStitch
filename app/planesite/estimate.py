"""Before-processing estimates: output resolution, mosaic area, file size and processing time.

Time constants are calibrated on a real 5,429-photo Matrice 4E job on an i9-14900HX (32 threads)
and scaled by worker count. After each job the ratio actual/estimated is remembered per PC
(see record_actual) so later estimates adapt to the machine.
"""
import json
import math
import os

import numpy as np

from .exif import camera_key
from .sysinfo import worker_count

# Costs in "worker-seconds" (seconds of one effective core), measured on a 5,429-photo Matrice 4E
# job on an i9-14900HX (Balanced, full resolution, 38 min in 1.3.0) and scaled by the speed-ups
# in 1.5.0. Rendering has a per-photo part (decoding) and a per-output-pixel part.
FEATURE_WS = {"fast": 1.2, "balanced": 1.6, "accurate": 6.0}       # per photo, CPU SIFT
FEATURE_WS_GPU = 1.3                                               # per photo with an NVIDIA GPU helping the CPU
MATCH_WS = {"fast": 0.3, "balanced": 0.4, "accurate": 0.75}        # per photo
ALIGN_S = {"fast": 0.045, "balanced": 0.065, "accurate": 0.1}      # per photo on the reference PC
RENDER_WS_PER_PHOTO = 0.75
RENDER_WS_PER_MPX = 0.17
RENDER_PARTIAL_DECODE = 0.85     # 1.6.1: only the needed rows of each photo are decoded
RENDER_GPU = 0.5                 # 1.6.1: the card blends; CPU workers only plan and decode
OVERVIEW_S_PER_MPX = 0.004
FIXED_S = 20
REF_WORKERS = 31


_GPU = []


def gpu_likely():
    """True when the GPU libraries are installed and Windows reports an NVIDIA card."""
    if not _GPU:
        ok = False
        try:
            import importlib.util
            if importlib.util.find_spec("cupy") is not None and not os.environ.get("PLANESITE_NO_GPU"):
                from .sysinfo import gpus
                ok = any("nvidia" in (g.get("name") or "").lower() for g in gpus())
        except Exception:
            ok = False
        _GPU.append(ok)
    return _GPU[0]


def effective_workers(w):
    """Hyper-threads and efficiency cores add much less than a full core each."""
    w = max(1, int(w))
    return min(w, 8) + 0.3 * max(0, w - 8)


BYTES_PER_PX = {"jpeg": 0.33, "deflate": 1.9}


def _calib_path():
    from . import migrate_from_fieldstitch
    migrate_from_fieldstitch()
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "PlaneSite")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "timing_calibration.json")


def _calibration():
    try:
        with open(_calib_path()) as f:
            v = float(json.load(f).get("factor", 1.0))
            return min(4.0, max(0.25, v))
    except Exception:
        return 1.0


def _render_scale(use_gpu):
    try:
        from .jpegcrop import present
        f = RENDER_PARTIAL_DECODE if present() else 1.0
    except Exception:
        f = 1.0
    return f * (RENDER_GPU if use_gpu else 1.0)


def record_actual(estimated_s, actual_s):
    """Blend the ratio of a finished job into the per-PC calibration factor."""
    if not estimated_s or not actual_s or estimated_s < 30:
        return
    raw = actual_s / (estimated_s / _calibration())
    old = _calibration()
    new = 0.5 * old + 0.5 * raw if os.path.exists(_calib_path()) else raw
    try:
        with open(_calib_path(), "w") as f:
            json.dump({"factor": round(min(4.0, max(0.25, new)), 3)}, f)
    except Exception:
        pass


def usable_photos(proj):
    photos = proj["photos"]
    sel = set(proj["selection"]) if proj.get("selection") else None
    maxob = proj["settings"].get("max_oblique_deg", 30)
    groups = {}
    for i, m in enumerate(photos):
        if sel is not None and i not in sel:
            continue
        if m.get("lat") is None:
            continue
        if m.get("pitch") is not None and abs(m["pitch"] + 90) > maxob:
            continue
        groups.setdefault(camera_key(m), []).append(m)
    if not groups:
        return []
    biggest = max(len(v) for v in groups.values())
    out = []
    for v in groups.values():
        if len(v) >= max(5, 0.2 * biggest):
            out.extend(v)
    return out


def native_gsd(photos):
    alts = [m["rel_alt"] for m in photos if m.get("rel_alt")]
    agl = float(np.median(alts)) if alts else 100.0
    f = float(np.median([m["f_px"] for m in photos]))
    return agl / f, agl


def covered_area_m2(photos, agl):
    """Approximate ground area covered by the photos (used for pixel count and file size)."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from .geo import utm_epsg
    lon = np.array([m["lon"] for m in photos])
    lat = np.array([m["lat"] for m in photos])
    tr = Transformer.from_crs(4326, utm_epsg(float(np.median(lon)), float(np.median(lat))), always_xy=True)
    E, N = tr.transform(lon, lat)
    m0 = photos[0]
    f = float(np.median([m["f_px"] for m in photos]))
    r = 0.9 * agl * min(m0["w"], m0["h"]) / 2.0 / f
    cell = max(2.0, r / 6.0)
    x0, x1 = E.min() - r, E.max() + r
    y0, y1 = N.min() - r, N.max() + r
    nx = int(math.ceil((x1 - x0) / cell))
    ny = int(math.ceil((y1 - y0) / cell))
    if nx * ny > 4_000_000:
        cell *= math.sqrt(nx * ny / 4_000_000.0)
        nx = int(math.ceil((x1 - x0) / cell))
        ny = int(math.ceil((y1 - y0) / cell))
    gx = x0 + (np.arange(nx) + 0.5) * cell
    gy = y0 + (np.arange(ny) + 0.5) * cell
    tree = cKDTree(np.c_[E, N])
    covered = 0
    for row in range(0, ny, 256):
        GX, GY = np.meshgrid(gx, gy[row:row + 256])
        d, _ = tree.query(np.c_[GX.ravel(), GY.ravel()], k=1, distance_upper_bound=r)
        covered += int(np.isfinite(d).sum())
    return covered * cell * cell


def estimate(proj, settings, cached_features=0, workers=None, area_m2=None):
    photos = usable_photos(proj)
    n = len(photos)
    if n < 3:
        return {"ok": False, "photos": n}
    preset = settings.get("preset", "balanced")
    gsd_n, agl = native_gsd(photos)
    factor = settings.get("res_factor") or 1.0
    gsd = float(settings["resolution_cm"]) / 100.0 if settings.get("resolution_cm") else gsd_n * float(factor)
    area = area_m2 if area_m2 is not None else covered_area_m2(photos, agl)
    mpx = area / (gsd * gsd) / 1e6
    comp = settings.get("compression", "jpeg")
    size = mpx * 1e6 * BYTES_PER_PX.get(comp, 0.33)
    w = workers or worker_count(700 * 1024 ** 2)
    eff = effective_workers(w)
    rel = effective_workers(REF_WORKERS) / eff        # how much slower than the reference PC
    todo = max(0, n - int(cached_features))
    cal = _calibration()
    use_gpu = str(settings.get("gpu", "auto")).lower() != "off" and gpu_likely()
    fws = FEATURE_WS_GPU if use_gpu else FEATURE_WS.get(preset, 1.6)
    stages = {
        "features": todo * fws / eff + 3,
        "matching": n * MATCH_WS.get(preset, 0.4) / eff,
        "alignment": n * ALIGN_S.get(preset, 0.04) * min(rel, 4.0) * 0.8 + FIXED_S * 0.6,
        "surface": n * ALIGN_S.get(preset, 0.04) * min(rel, 4.0) * 0.2 + FIXED_S * 0.2,
        "ortho": (n * RENDER_WS_PER_PHOTO + mpx * RENDER_WS_PER_MPX) / eff * _render_scale(use_gpu)
                 + mpx * OVERVIEW_S_PER_MPX * min(rel, 2.0) + FIXED_S * 0.2,
    }
    stages = {k: v * cal for k, v in stages.items()}
    secs = sum(stages.values())
    return {"ok": True, "photos": n, "gsd_native_cm": round(gsd_n * 100, 2), "gsd_cm": round(gsd * 100, 2),
            "flight_agl_m": round(agl, 1), "area_ha": round(area / 1e4, 1), "mpx": round(mpx, 1),
            "size_bytes": int(size), "seconds": int(secs), "cached_features": int(cached_features),
            "stages": {k: round(v, 1) for k, v in stages.items()}, "gpu": use_gpu,
            "gpu_available": gpu_likely()}
