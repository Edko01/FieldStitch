"""Project files and photo import."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import __version__
from .exif import find_images, read_image_meta, camera_key
from .pipeline import DEFAULT_SETTINGS, save_json


def project_file(d):
    return os.path.join(d, "project.json")


def create_project(directory, name):
    os.makedirs(directory, exist_ok=True)
    pf = project_file(directory)
    if os.path.exists(pf):
        return load(directory)
    proj = {"name": name, "version": __version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "photos": [], "folders": [], "settings": dict(DEFAULT_SETTINGS), "selection": None}
    save_json(pf, proj)
    return proj


def load(directory):
    with open(project_file(directory), "r", encoding="utf-8") as f:
        proj = json.load(f)
    s = dict(DEFAULT_SETTINGS)
    s.update(proj.get("settings") or {})
    proj["settings"] = s
    return proj


def save(directory, proj):
    save_json(project_file(directory), proj)


def scan_folder(folder, recursive=True, progress=None, known=None):
    """Read metadata for all JPGs in folder. Returns (metas, errors)."""
    return scan_paths(find_images(folder, recursive), progress, known)


MIN_PHOTO_PX = 1200 * 900   # smaller JPGs are thumbnails/previews, not survey photos


def scan_paths(paths, progress=None, known=None):
    known = known or set()
    paths = [p for p in paths if os.path.abspath(p) not in known]
    metas = []
    errors = []
    done = [0]

    def one(p):
        try:
            return read_image_meta(p), None
        except Exception as e:  # noqa
            return None, (p, str(e))

    with ThreadPoolExecutor(max_workers=8) as ex:
        for m, err in ex.map(one, paths):
            done[0] += 1
            if m and m.get("w", 0) * m.get("h", 0) < MIN_PHOTO_PX:
                errors.append((m["path"], "too small to be a survey photo (%dx%d)" % (m["w"], m["h"])))
            elif m:
                metas.append(m)
            else:
                errors.append(err)
            if progress and (done[0] % 10 == 0 or done[0] == len(paths)):
                progress(done[0], len(paths))
    metas.sort(key=lambda m: (m.get("time") or 0, m["path"]))
    return metas, errors


def summarize(proj):
    photos = proj["photos"]
    n = len(photos)
    gps = sum(1 for m in photos if m.get("lat") is not None)
    rtk = sum(1 for m in photos if (m.get("rtk_flag") or 0) >= 50)
    cams = {}
    for m in photos:
        cams[camera_key(m)] = cams.get(camera_key(m), 0) + 1
    alts = sorted(m["rel_alt"] for m in photos if m.get("rel_alt") is not None)
    oblique = sum(1 for m in photos if m.get("pitch") is not None
                  and abs(m["pitch"] + 90) > proj["settings"].get("max_oblique_deg", 30))
    est_gsd = None
    if alts:
        agl = alts[len(alts) // 2]
        fs = sorted(m["f_px"] for m in photos if m.get("f_px"))
        if fs:
            est_gsd = agl / fs[len(fs) // 2] * 100
    main = max(cams.values()) if cams else 0
    other = sum(v for v in cams.values() if v < max(5, 0.2 * main))
    return {"count": n, "with_gps": gps, "rtk_fixed": rtk, "cameras": cams, "other_camera": other,
            "median_rel_alt": alts[len(alts) // 2] if alts else None, "oblique": oblique,
            "est_gsd_cm": round(est_gsd, 2) if est_gsd else None,
            "selected": len(proj["selection"]) if proj.get("selection") else n}
