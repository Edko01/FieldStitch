"""Local web server for the PlaneSite user interface (binds to 127.0.0.1 only)."""
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

from . import __version__, project as P, updater
from .pipeline import load_project, save_json, safe_name

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(APP_DIR, "web")
RUN_SCRIPT = os.path.join(APP_DIR, "run_pipeline.py")
R0 = 64.0  # metres per screen pixel at zoom 0
IS_WIN = sys.platform == "win32"


def console_python():
    """The console Python for worker processes, named PlaneSite.exe in the installed app so
    Task Manager shows PlaneSite rather than python."""
    d = os.path.dirname(sys.executable)
    for n in ("PlaneSite.exe", "python.exe"):
        p = os.path.join(d, n)
        if IS_WIN and os.path.isfile(p):
            return p
    return sys.executable


def config_dir():
    from . import migrate_from_fieldstitch
    migrate_from_fieldstitch()
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "PlaneSite")
    os.makedirs(d, exist_ok=True)
    return d


def default_projects_dir():
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    if not os.path.isdir(docs):
        docs = os.path.expanduser("~")
    return os.path.join(docs, "PlaneSite Projects")


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.dir = None
        self.proj = None
        self.proc = None
        self.scan = None       # {"done", "total", "folder", "error"}
        self.export = None     # {"done", "total", "current", "frac", "files", "errors", "out_dir"}
        self.cfg_path = os.path.join(config_dir(), "config.json")
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                self.cfg = json.load(f)
        except Exception:
            self.cfg = {"recent": []}
        self.tile_local = threading.local()
        self.origin = None
        self.last_seen = 0.0

    def save_cfg(self):
        save_json(self.cfg_path, self.cfg)

    def remember(self, d):
        rec = [r for r in self.cfg.get("recent", []) if os.path.normcase(r) != os.path.normcase(d)]
        rec.insert(0, d)
        self.cfg["recent"] = rec[:12]
        self.save_cfg()

    def save(self):
        P.save(self.dir, self.proj)

    # ------------------------------------------------------------ paths & origin
    def outputs(self):
        return os.path.join(self.dir, "outputs")

    def report(self):
        if self.proj and self.proj.get("geotiff"):
            g = self.proj["geotiff"]
            return {"ortho": g["path"], "epsg": g["epsg"], "gsd_cm": g["gsd_cm"], "imported": True,
                    "area_ha": g.get("area_ha")}
        p = os.path.join(self.outputs(), "report.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def ortho_path(self):
        if self.proj and self.proj.get("geotiff"):
            p = self.proj["geotiff"]["path"]
            return p if os.path.exists(p) else None
        r = self.report()
        if r:
            p = os.path.join(self.outputs(), r.get("ortho", ""))
            if os.path.exists(p):
                return p
        return None

    def compute_origin(self):
        """Fixed local origin (UTM metres) for the map frame."""
        from .geo import Geo, utm_epsg
        if self.proj.get("map_origin"):
            o = self.proj["map_origin"]
            self.origin = (o["epsg"], o["e"], o["n"])
            return
        photos = [m for m in self.proj["photos"] if m.get("lat") is not None]
        if not photos:
            self.origin = None
            return
        lon = float(np.median([m["lon"] for m in photos]))
        lat = float(np.median([m["lat"] for m in photos]))
        epsg = utm_epsg(lon, lat)
        e, n = Geo(epsg).to_utm(lon, lat)
        self.origin = (epsg, math.floor(e), math.floor(n))
        self.proj["map_origin"] = {"epsg": epsg, "e": self.origin[1], "n": self.origin[2]}
        self.save()

    def running(self):
        return self.proc is not None and self.proc.poll() is None


S = State()


# ------------------------------------------------------------------- helpers
def json_body(handler):
    n = int(handler.headers.get("Content-Length") or 0)
    if n == 0:
        return {}
    return json.loads(handler.rfile.read(n).decode("utf-8"))


class ApiError(Exception):
    pass


def open_in_explorer(path):
    if IS_WIN:
        os.startfile(path)  # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


BROWSE_CODE = r'''
import sys, tkinter as tk
from tkinter import filedialog
r = tk.Tk(); r.withdraw()
try:
    r.attributes("-topmost", True)
except Exception:
    pass
mode, title, init = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "folder":
    p = filedialog.askdirectory(parent=r, title=title, initialdir=init or None, mustexist=True)
elif mode == "images":
    ps = filedialog.askopenfilenames(parent=r, title=title, initialdir=init or None,
        filetypes=[("Drone photos", "*.jpg *.jpeg *.JPG *.JPEG"), ("All files", "*.*")])
    p = "\n".join(ps or [])
elif mode == "geotiff":
    p = filedialog.askopenfilename(parent=r, title=title, initialdir=init or None,
        filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")])
else:
    p = filedialog.askopenfilename(parent=r, title=title, initialdir=init or None,
        filetypes=[("Field boundaries", "*.kml *.kmz *.shp *.geojson *.json"), ("All files", "*.*")])
sys.stdout.write(p or "")
'''


def browse(mode, title, initial):
    flags = 0x08000000 if IS_WIN else 0
    out = subprocess.run([sys.executable, "-c", BROWSE_CODE, mode, title, initial or ""],
                         capture_output=True, text=True, creationflags=flags, timeout=3600)
    p = (out.stdout or "").strip()
    if mode == "images":
        return [os.path.normpath(x) for x in p.splitlines() if x.strip()]
    return os.path.normpath(p) if p else ""


# ---------------------------------------------------------------- project ops
def project_summary():
    if not S.proj:
        return None
    s = P.summarize(S.proj)
    s.update({"name": S.proj["name"], "dir": S.dir, "folders": S.proj.get("folders", []),
              "settings": S.proj["settings"], "has_result": S.ortho_path() is not None,
              "geotiff": (S.proj.get("geotiff") or {}).get("path")})
    return s


def recent_projects():
    out = []
    for d in S.cfg.get("recent", []):
        pf = P.project_file(d)
        if not os.path.exists(pf):
            continue
        name = os.path.basename(d).replace("_", " ")
        try:
            with open(pf, "r", encoding="utf-8") as f:
                head = f.read(4096)
            m = re.search(r'"name":\s*"((?:[^"\\]|\\.)*)"', head)
            if m:
                name = json.loads('"%s"' % m.group(1))
        except Exception:
            pass
        out.append({"dir": d, "name": name, "modified": os.path.getmtime(pf),
                    "has_result": os.path.exists(os.path.join(d, "outputs", "report.json")),
                    "has_thumb": os.path.exists(os.path.join(d, "outputs", "preview.png"))})
    return out


def open_project(d):
    d = os.path.normpath(d)
    if not os.path.exists(P.project_file(d)):
        raise ApiError("No PlaneSite project in that folder.")
    with S.lock:
        if S.running():
            raise ApiError("Processing is running. Cancel it before switching projects.")
        S.dir = d
        S.proj = P.load(d)
        S.origin = None
        S.compute_origin()
        S.remember(d)
        S.tile_local = threading.local()


def scan_worker(folder, recursive, files=None):
    try:
        known = set(m["path"] for m in S.proj["photos"])

        def prog(done, total):
            S.scan.update(done=done, total=total)

        if files:
            metas, errors = P.scan_paths(files, prog, known)
        else:
            metas, errors = P.scan_folder(folder, recursive, prog, known)
        with S.lock:
            S.proj["photos"].extend(metas)
            S.proj["photos"].sort(key=lambda m: (m.get("time") or 0, m["path"]))
            if folder and folder not in S.proj["folders"]:
                S.proj["folders"].append(folder)
            S.proj["selection"] = None
            S.proj.pop("map_origin", None)
            S.compute_origin()
            S.save()
        small = [e for e in errors if e[1].startswith("too small")]
        bad = [e for e in errors if not e[1].startswith("too small")]
        S.scan.update(finished=True, added=len(metas), errors=len(bad), skipped_small=len(small),
                      error_examples=[os.path.basename(e[0]) + ": " + e[1] for e in bad[:3]])
    except Exception as e:
        S.scan.update(finished=True, error=str(e))


def start_scan(folder=None, files=None, recursive=True):
    if S.running():
        raise ApiError("Processing is running.")
    if S.scan and not S.scan.get("finished"):
        raise ApiError("Already reading photos.")
    if files:
        files = [os.path.normpath(f) for f in files if os.path.isfile(f)]
        if not files:
            raise ApiError("None of those files could be found.")
    elif not folder or not os.path.isdir(folder):
        raise ApiError("Folder not found: %s" % folder)
    S.scan = {"done": 0, "total": len(files) if files else 0, "folder": folder or "", "finished": False}
    threading.Thread(target=scan_worker, args=(None if files else folder, recursive, files),
                     daemon=True).start()


def unique_project_dir(parent, name):
    base = safe_name(name)
    d = os.path.join(parent, base)
    k = 2
    while os.path.exists(d):
        d = os.path.join(parent, "%s_%d" % (base, k))
        k += 1
    return d


def name_from_source(folder=None, files=None):
    src = folder
    if files:
        src = os.path.dirname(files[0]) if len(files) == 1 else os.path.commonpath(files)
        if os.path.isfile(src):
            src = os.path.dirname(src)
    src = os.path.normpath(src or "")
    nm = os.path.basename(src)
    # DJI card folders (DCIM, 100MEDIA, DJI_2024...) say little; use the parent folder instead
    while nm and re.match(r"^(DCIM|\d{3}MEDIA|DJI_\d.*|\d{3}[A-Z_]*)$", nm, re.I):
        src = os.path.dirname(src)
        nm = os.path.basename(src)
    return nm or "Unnamed project"


def quick_start(folder=None, files=None, recursive=True, parent=None, name=None):
    """Create a project (named after the photos unless a name is given) and begin reading them."""
    if S.running():
        raise ApiError("Processing is running. Stop it before starting another project.")
    if files:
        files = [os.path.normpath(f) for f in files if os.path.isfile(f)]
        if not files:
            raise ApiError("None of those files could be found.")
    elif not folder or not os.path.isdir(folder):
        raise ApiError("Folder not found: %s" % folder)
    parent = parent or S.cfg.get("projects_dir") or default_projects_dir()
    name = (name or "").strip() or name_from_source(folder, files)
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        raise ApiError("Can't create the folder %s: %s" % (parent, e.strerror or e))
    d = unique_project_dir(parent, name)
    P.create_project(d, name)
    if os.path.normcase(parent) != os.path.normcase(S.cfg.get("projects_dir") or default_projects_dir()):
        S.cfg["projects_dir"] = parent
        S.save_cfg()
    open_project(d)
    start_scan(folder, files, recursive)
    return d


def open_geotiff(path):
    import rasterio
    if not os.path.isfile(path):
        raise ApiError("File not found: %s" % path)
    with rasterio.open(path) as ds:
        if ds.crs is None or not ds.crs.is_projected:
            raise ApiError("That GeoTIFF isn't in a projected coordinate system (UTM or similar), "
                           "so it can't be measured or split. Reproject it first.")
        epsg = ds.crs.to_epsg()
        if not epsg:
            raise ApiError("PlaneSite couldn't identify the GeoTIFF's coordinate system (needs an EPSG code).")
        if ds.count < 3:
            raise ApiError("The GeoTIFF needs red, green and blue bands.")
        b = ds.bounds
        gsd = ds.res[0]
        try:
            units = float(ds.crs.linear_units_factor[1])
        except Exception:
            units = 1.0
    if abs(units - 1.0) > 1e-6:
        raise ApiError("Only GeoTIFFs in metres are supported (this one uses other units).")
    name = os.path.splitext(os.path.basename(path))[0]
    parent = S.cfg.get("projects_dir") or default_projects_dir()
    d = unique_project_dir(parent, name)
    proj = P.create_project(d, name)
    proj["geotiff"] = {"path": os.path.normpath(path), "epsg": int(epsg), "gsd_cm": round(gsd * 100, 2)}
    proj["map_origin"] = {"epsg": int(epsg), "e": math.floor((b.left + b.right) / 2),
                          "n": math.floor((b.bottom + b.top) / 2)}
    P.save(d, proj)
    open_project(d)
    return d


def cached_feature_count(preset_name):
    from .pipeline import PRESETS
    pr = PRESETS.get(preset_name, PRESETS["balanced"])
    key = "r%d_n%d" % (pr["reduce"], pr["nfeat"])
    kp = os.path.join(S.dir, "work", "features", "key.json")
    try:
        with open(kp) as f:
            old = json.load(f)
    except Exception:
        return 0
    paths = set(m["path"] for m in S.proj["photos"])
    return sum(1 for v in old.values() if len(v) >= 2 and v[1] == key and v[0] in paths)


_EST_CACHE = {}


def project_estimate(settings):
    from . import estimate as E
    st = dict(S.proj["settings"])
    st.update(settings or {})
    photos = E.usable_photos({"photos": S.proj["photos"], "selection": S.proj.get("selection"),
                              "settings": st})
    if len(photos) < 3:
        return {"ok": False, "photos": len(photos)}
    sig = (len(photos), photos[0]["path"], photos[-1]["path"], st.get("max_oblique_deg"))
    with S.lock:
        if _EST_CACHE.get("sig") != sig:
            gsd_n, agl = E.native_gsd(photos)
            _EST_CACHE.clear()
            _EST_CACHE.update(sig=sig, area=E.covered_area_m2(photos, agl))
    return E.estimate({"photos": photos, "selection": None, "settings": st}, st,
                      cached_features=cached_feature_count(st.get("preset", "balanced")),
                      workers=st.get("workers"), area_m2=_EST_CACHE["area"])


def rename_project(name):
    """Rename the project; the folder follows the name when it was auto-named and nothing is busy."""
    if not name:
        raise ApiError("Enter a project name.")
    with S.lock:
        old_name = S.proj["name"]
        S.proj["name"] = name
        S.save()
        moved = False
        busy = S.running() or (S.scan and not S.scan.get("finished")) or (S.export and not S.export.get("finished"))
        base = os.path.basename(S.dir)
        auto = base == safe_name(old_name) or re.match(re.escape(safe_name(old_name)) + r"_\d+$", base)
        if not busy and auto and safe_name(name) != base:
            new_dir = unique_project_dir(os.path.dirname(S.dir), name)
            S.tile_local = threading.local()
            import gc
            gc.collect()
            for k in range(10):
                try:
                    os.rename(S.dir, new_dir)
                    moved = True
                    break
                except OSError:
                    time.sleep(0.2)
            if moved:
                S.cfg["recent"] = [r for r in S.cfg.get("recent", []) if os.path.normcase(r) != os.path.normcase(S.dir)]
                S.dir = new_dir
                S.remember(new_dir)
        return {"name": name, "dir": S.dir, "moved": moved}


def photo_points():
    if not S.proj or not S.origin:
        return {"points": [], "origin": None}
    from .geo import Geo
    epsg, e0, n0 = S.origin
    g = Geo(epsg)
    photos = S.proj["photos"]
    sel = set(S.proj["selection"]) if S.proj.get("selection") else None
    ok = [i for i, m in enumerate(photos) if m.get("lat") is not None]
    if not ok:
        return {"points": [], "origin": [e0, n0, epsg]}
    E, N = g.to_utm(np.array([photos[i]["lon"] for i in ok]), np.array([photos[i]["lat"] for i in ok]))
    pts = []
    maxob = S.proj["settings"].get("max_oblique_deg", 30)
    from .exif import camera_key
    cams = {}
    for m in photos:
        ck = camera_key(m)
        cams[ck] = cams.get(ck, 0) + 1
    main = max(cams.values()) if cams else 0
    for k, i in enumerate(ok):
        m = photos[i]
        obl = m.get("pitch") is not None and abs(m["pitch"] + 90) > maxob
        other = cams[camera_key(m)] < max(5, 0.2 * main)
        pts.append([round(float(E[k] - e0), 2), round(float(N[k] - n0), 2), i,
                    1 if (sel is None or i in sel) else 0, 1 if (m.get("rtk_flag") or 0) >= 50 else 0,
                    1 if obl else (2 if other else 0)])
    return {"points": pts, "origin": [e0, n0, epsg]}


def start_processing(settings):
    with S.lock:
        if S.running():
            raise ApiError("Processing is already running.")
        if not S.proj["photos"]:
            raise ApiError("Add photos first.")
        S.proj["settings"].update(settings or {})
        S.save()
        prog = os.path.join(S.dir, "progress.json")
        save_json(prog, {"status": "starting", "stage": "", "overall": 0, "message": "Starting", "elapsed": 0})
        logf = open(os.path.join(S.dir, "process_console.txt"), "w")
        # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS: the workers inherit below-normal priority,
        # so they still get every idle core but the desktop and the PlaneSite window come first
        flags = (0x08000000 | 0x00004000) if IS_WIN else 0
        kw = {}
        if not IS_WIN:
            kw["start_new_session"] = True
        try:
            S.proc = subprocess.Popen([console_python(), RUN_SCRIPT, S.dir], cwd=APP_DIR, stdout=logf,
                                      stderr=subprocess.STDOUT, creationflags=flags, **kw)
        finally:
            logf.close()  # the child has its own handle; keeping ours would lock the project folder


def cancel_processing():
    with S.lock:
        if not S.running():
            return
        pid = S.proc.pid
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True,
                           creationflags=0x08000000)
        else:
            import signal
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                S.proc.kill()
        try:
            S.proc.wait(10)
        except Exception:
            pass
        p = os.path.join(S.dir, "progress.json")
        try:
            st = json.load(open(p))
        except Exception:
            st = {}
        st.update(status="cancelled", message="Cancelled")
        save_json(p, st)


def progress_state():
    if not S.dir:
        return None
    p = os.path.join(S.dir, "progress.json")
    try:
        with open(p, "r") as f:
            st = json.load(f)
    except Exception:
        return None
    if st.get("status") in ("running", "starting") and not S.running():
        st["status"] = "error"
        st["message"] = st.get("message") or "Processing stopped unexpectedly. See the log."
    return st


# --------------------------------------------------------------------- areas
def areas_path():
    return os.path.join(S.dir, "areas.json")


def load_areas():
    try:
        with open(areas_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_areas(areas):
    save_json(areas_path(), areas)


def areas_for_client(areas):
    from .export import area_stats
    epsg, e0, n0 = S.origin
    out = []
    for a in areas:
        rings = [[[round(p[0] - e0, 3), round(p[1] - n0, 3)] for p in r] for r in a["rings"]]
        st = area_stats(a)
        out.append({"id": a["id"], "name": a["name"], "kind": a.get("kind", ""), "rings": rings,
                    "ha": float(st["ha"]), "ac": float(st["ac"])})
        if a.get("pilot2"):
            out[-1]["pilot2"] = {k: a["pilot2"][k] for k in ("name", "sent") if k in a["pilot2"]}
    return out


def areas_from_client(lst):
    from .export import new_id
    epsg, e0, n0 = S.origin
    old = {a["id"]: a for a in load_areas()}
    out = []
    for a in lst:
        rings = [[[p[0] + e0, p[1] + n0] for p in r] for r in a["rings"]]
        for r in rings:
            if r[0] != r[-1]:
                r.append(list(r[0]))
        aid = a.get("id") or new_id()
        new = {k: v for k, v in old.get(aid, {}).items() if k not in ("id", "name", "kind", "rings")}
        new.update({"id": aid, "name": a.get("name") or "Area", "kind": a.get("kind", "drawn"), "rings": rings})
        out.append(new)
    return out


# ------------------------------------------------------------------- Pilot 2
def pilot2_settings():
    from . import pilot2
    try:
        has_key = bool(pilot2.load_key())
    except Exception:
        has_key = False
    d = dict(pilot2.DEFAULTS)
    d.update(S.cfg.get("pilot2") or {})
    return {"hub_url": S.cfg.get("hub_url") or pilot2.DEFAULT_HUB, "default_hub": pilot2.DEFAULT_HUB,
            "user": S.cfg.get("hub_user") or "", "has_key": has_key, "defaults": d}


def pilot2_set(b):
    from . import pilot2
    if "hub_url" in b:
        u = (b.get("hub_url") or "").strip().rstrip("/")
        if u and not re.match(r"^https?://[^\s/]+", u):
            raise ApiError("The hub address should start with https://")
        S.cfg["hub_url"] = u or pilot2.DEFAULT_HUB
    if "user" in b:
        S.cfg["hub_user"] = (b.get("user") or "").strip()[:60]
    if b.get("key") is not None and (b.get("key") or b.get("clear_key")):
        try:
            pilot2.save_key(b.get("key") or "")
        except OSError as e:
            raise ApiError(str(e))
    S.save_cfg()
    return pilot2_settings()


def _pilot2_area(b):
    from . import pilot2
    if not S.origin:
        raise ApiError("Open a project first.")
    area = next((a for a in load_areas() if a["id"] == b.get("id")), None)
    if area is None:
        raise ApiError("That field no longer exists.")
    given = dict(S.cfg.get("pilot2") or {})
    given.update((area.get("pilot2") or {}).get("settings") or {})
    given.update(b.get("settings") or {})
    try:
        s, ring, ring_ll, est, notes = pilot2.prepare(area, S.origin[0], given)
    except ValueError as e:
        raise ApiError(str(e))
    default_name = (area.get("pilot2") or {}).get("name") or "%s - %s" % (S.proj["name"], area["name"])
    name = pilot2.clean_name(b.get("name") or default_name)
    return area, s, ring_ll, est, notes, name


def pilot2_preview(b):
    area, s, ring_ll, est, notes, name = _pilot2_area(b)
    return {"settings": s, "estimates": est, "notes": notes, "name": name, "corners": len(ring_ll) - 1,
            "last": (area.get("pilot2") or {}).get("sent")}


def pilot2_send(b):
    """Write the plan-only KMZ (kept in the project's exports folder) and, unless only saving,
    upload it to the hub. Same mission name replaces the copy on the hub."""
    from . import pilot2
    area, s, ring_ll, est, notes, name = _pilot2_area(b)
    kmz = pilot2.build_kmz(ring_ll, s)
    out_dir = os.path.join(S.dir, "exports", "Pilot 2")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".kmz")
    with open(path, "wb") as f:
        f.write(kmz)
    res = {"path": path, "name": name, "bytes": len(kmz), "estimates": est, "notes": notes}
    keep = {k: s[k] for k in pilot2.DEFAULTS}
    if b.get("upload", True):
        try:
            hub = pilot2.upload(S.cfg.get("hub_url") or pilot2.DEFAULT_HUB, pilot2.load_key(), kmz, name + ".kmz",
                                name=name, user=S.cfg.get("hub_user") or None)
        except (PermissionError, ConnectionError, RuntimeError) as e:
            raise ApiError(str(e))
        res["hub"] = hub
        sent = {"at": time.strftime("%Y-%m-%d %H:%M"), "hub_id": hub.get("id"), "replaced": bool(hub.get("replaced"))}
    else:
        sent = None
    # remember: these settings for this field, and as the starting point for new fields
    areas = load_areas()
    for a in areas:
        if a["id"] == area["id"]:
            rec = dict(a.get("pilot2") or {})
            rec.update({"settings": keep, "name": name})
            if sent:
                rec["sent"] = sent
            a["pilot2"] = rec
            res["pilot2"] = rec
    save_areas(areas)
    S.cfg["pilot2"] = {k: keep[k] for k in keep if k != "direction" or not keep["auto_direction"]}
    S.save_cfg()
    return res


def check_ortho_crs():
    """Areas are stored in the ortho CRS; the map frame uses the same UTM zone."""
    rep = S.report()
    if rep and S.origin and rep.get("epsg") != S.origin[0]:
        S.proj["map_origin"] = {"epsg": rep["epsg"], "e": S.origin[1], "n": S.origin[2]}
        S.save()
        S.compute_origin()


def save_diagnostics():
    """Zip the log, report (minus coordinates) and settings for troubleshooting / performance tuning."""
    import zipfile
    from .sysinfo import describe
    stamp = time.strftime("%Y%m%d-%H%M")
    path = os.path.join(S.dir, "PlaneSite-diagnostics-%s-%s.zip" % (safe_name(S.proj["name"]), stamp))
    rep = S.report() or {}
    rep = {k: v for k, v in rep.items() if k not in ("cam_positions", "center_lonlat")}
    summary = P.summarize(S.proj)
    info = {"version": __version__, "settings": S.proj.get("settings"), "photos": summary,
            "progress": progress_state(), "system": describe(S.dir, S.proj.get("folders") or [])}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("log.txt", "process_console.txt"):
            fp = os.path.join(S.dir, name)
            if os.path.exists(fp):
                z.write(fp, name)
        try:      # the app's own log (window freezes, server errors), last part only
            from .appwindow import data_dir
            lp = os.path.join(data_dir(), "logs", "planesite.log")
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                z.writestr("planesite.log", "".join(f.readlines()[-1500:]))
        except Exception:
            pass
        z.writestr("report.json", json.dumps(rep, indent=1))
        z.writestr("info.json", json.dumps(info, indent=1, default=str))
    return path


def export_worker(ids, out_dir, compression, scale=1.0, overviews=True, whole=False):
    from .export import export_area, safe_filename
    try:
        areas = [a for a in load_areas() if a["id"] in ids]
        if whole:
            areas = [None] + areas
        ortho = S.ortho_path()
        S.export.update(total=len(areas))
        os.makedirs(out_dir, exist_ok=True)
        pname = safe_name(S.proj["name"])
        used = set()
        for k, a in enumerate(areas):
            label = a["name"] if a else "Whole orthomosaic"
            S.export.update(current=label, done=k, frac=0)
            base = ("%s_%s" % (pname, safe_filename(a["name"]))) if a else "%s_orthomosaic" % pname
            if scale < 0.999:
                base += "_%dpct" % int(round(scale * 100))
            fn = base
            c = 2
            while fn.lower() in used:
                fn = "%s_%d" % (base, c)
                c += 1
            used.add(fn.lower())
            path = os.path.join(out_dir, fn + ".tif")
            try:
                export_area(ortho, a, path, compression=compression,
                            jpeg_quality=S.proj["settings"].get("jpeg_quality", 90),
                            progress=lambda f: S.export.update(frac=f), scale=scale, overviews=overviews)
                S.export["files"].append(path)
            except Exception as e:
                S.export["errors"].append("%s: %s" % (label, e))
        S.export.update(done=len(areas), finished=True, current="")
    except Exception as e:
        S.export.update(finished=True, error=str(e))


def export_estimate(b):
    """Approximate output size for the export dialog."""
    from .export import area_stats
    op = S.ortho_path()
    if not op:
        return {"ok": False}
    import rasterio
    with rasterio.open(op) as ds:
        gsd = ds.res[0]
        src_jpeg = ds.compression is not None and ds.compression.value == "JPEG"
        W, H = ds.width, ds.height
    rep = S.report() or {}
    data_m2 = (rep.get("area_ha") or 0) * 1e4 or W * H * gsd * gsd
    scale = float(b.get("scale") or 1.0)
    comp = b.get("compression", "same")
    if comp == "same":
        comp = "jpeg" if src_jpeg else "deflate"
    bpp = 0.33 if comp == "jpeg" else 1.9
    ov = 1.33 if b.get("overviews", True) else 1.0
    m2 = data_m2 if b.get("whole") else 0.0
    ids = set(b.get("ids") or [])
    for a in load_areas():
        if a["id"] in ids:
            m2 += min(area_stats(a)["ha"] * 1e4, data_m2)
    px = m2 / (gsd / scale) ** 2
    return {"ok": True, "bytes": int(px * bpp * ov), "gsd_cm": round(gsd / scale * 100, 2),
            "megapixels": round(px / 1e6, 1), "native_gsd_cm": round(gsd * 100, 2)}


# ---------------------------------------------------------------------- tiles
os.environ.setdefault("GDAL_CACHEMAX", "512")   # MB of decoded GeoTIFF blocks shared by tile requests
TILE_CACHE_BYTES = 384 * 1024 * 1024


class _TileCache:
    """Finished map tiles (encoded bytes) for the open orthomosaic, least recently used dropped first."""

    def __init__(self):
        from collections import OrderedDict
        self.lock = threading.Lock()
        self.items = OrderedDict()
        self.bytes = 0
        self.key = None          # (path, mtime) the cached tiles belong to

    def get(self, key):
        with self.lock:
            v = self.items.get(key)
            if v is not None:
                self.items.move_to_end(key)
            return v

    def put(self, key, value):
        with self.lock:
            if key in self.items:
                return
            self.items[key] = value
            self.bytes += len(value[0]) + 64
            while self.bytes > TILE_CACHE_BYTES and self.items:
                _, old = self.items.popitem(last=False)
                self.bytes -= len(old[0]) + 64

    def reset(self, key):
        with self.lock:
            if self.key != key:
                self.items.clear()
                self.bytes = 0
                self.key = key


TILES = _TileCache()


def _tile_dataset(path, mt):
    import rasterio
    loc = S.tile_local
    if getattr(loc, "path", None) != path or getattr(loc, "mtime", None) != mt:
        if getattr(loc, "ds", None) is not None:
            loc.ds.close()
        loc.ds = rasterio.open(path)
        loc.path, loc.mtime = path, mt
    return loc.ds


_TILE_BUSY = [0]          # tiles being built for the screen right now
_LAST_TILE = [0.0]        # when the screen last asked for a tile
_TILE_BUSY_LOCK = threading.Lock()


def tile_png(z, x, y, prefetch=False):
    """Encoded map tile (bytes, content type) or None when the tile is empty. Fully covered tiles
    are JPEG (small and quick to make), tiles on the edge of the mosaic PNG with transparency."""
    path = S.ortho_path()
    if not path or not S.origin:
        return None
    mt = os.path.getmtime(path)
    TILES.reset((path, mt))
    key = (path, mt, z, x, y)
    hit = TILES.get(key)
    if hit is not None:
        return hit if hit[0] else None
    if prefetch and _TILE_BUSY[0] > 0:
        return "busy"          # the screen comes first; the browser will ask again when needed
    if not prefetch:
        _LAST_TILE[0] = time.time()
        with _TILE_BUSY_LOCK:
            _TILE_BUSY[0] += 1
    try:
        res = _make_tile(path, mt, z, x, y)
    finally:
        if not prefetch:
            with _TILE_BUSY_LOCK:
                _TILE_BUSY[0] -= 1
    TILES.put(key, res or (b"", ""))
    return res


def _make_tile(path, mt, z, x, y):
    import cv2
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    ds = _tile_dataset(path, mt)
    _, e0, n0 = S.origin
    s = R0 / (2 ** z)
    te0 = e0 + x * 256 * s
    tn1 = n0 - y * 256 * s
    te1 = te0 + 256 * s
    tn0 = tn1 - 256 * s
    b = ds.bounds
    ie0, ie1 = max(te0, b.left), min(te1, b.right)
    in0, in1 = max(tn0, b.bottom), min(tn1, b.top)
    if ie1 <= ie0 or in1 <= in0:
        return None
    # destination rectangle inside the tile
    dx0 = int(round((ie0 - te0) / s))
    dx1 = int(round((ie1 - te0) / s))
    dy0 = int(round((tn1 - in1) / s))
    dy1 = int(round((tn1 - in0) / s))
    w, h = dx1 - dx0, dy1 - dy0
    if w <= 0 or h <= 0:
        return None
    gsd = ds.res[0]
    col0 = (ie0 - b.left) / gsd
    row0 = (b.top - in1) / gsd
    win = Window(col0, row0, (ie1 - ie0) / gsd, (in1 - in0) / gsd)
    rgb = ds.read([1, 2, 3], window=win, out_shape=(3, h, w), resampling=Resampling.bilinear)
    m = ds.read_masks(1, window=win, out_shape=(h, w), resampling=Resampling.nearest)
    if not m.any():
        return None
    if w == 256 and h == 256 and m.min() == 255:
        bgr = cv2.merge([rgb[2], rgb[1], rgb[0]])
        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return enc.tobytes(), "image/jpeg"
    tile = np.zeros((256, 256, 4), np.uint8)
    tile[dy0:dy1, dx0:dx1, 0] = rgb[2]
    tile[dy0:dy1, dx0:dx1, 1] = rgb[1]
    tile[dy0:dy1, dx0:dx1, 2] = rgb[0]
    tile[dy0:dy1, dx0:dx1, 3] = m
    ok, enc = cv2.imencode(".png", tile, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return enc.tobytes(), "image/png"


_WARM = {"key": None}


def warm_tiles(bounds_local, maxz):
    """Build the zoomed-out tile levels in the background as soon as an orthomosaic is shown, so
    zooming and panning at overview scales never waits on the GeoTIFF. Stops if the ortho changes."""
    path = S.ortho_path()
    if not path or not S.origin:
        return
    mt = os.path.getmtime(path)
    key = (path, mt)
    if _WARM["key"] == key:
        return
    _WARM["key"] = key
    l, bot, r, top = bounds_local
    jobs = []
    for z in range(0, maxz + 1):
        s = R0 / (2 ** z) * 256
        xs = range(int(math.floor(l / s)), int(math.floor(r / s)) + 1)
        ys = range(int(math.floor(-top / s)), int(math.floor(-bot / s)) + 1)
        if len(xs) * len(ys) > 2500 or len(jobs) + len(xs) * len(ys) > 4000:
            break
        jobs += [(z, x, y) for y in ys for x in xs]

    def run(part):
        for z, x, y in part:
            if _WARM["key"] != key or S.ortho_path() != path:
                return
            while _TILE_BUSY[0] > 0 or time.time() - _LAST_TILE[0] < 0.4:
                time.sleep(0.05)       # the screen is loading tiles: step aside
            try:
                tile_png(z, x, y, prefetch=True)
            except Exception:
                return
    threading.Thread(target=run, args=(jobs,), daemon=True).start()


def _empty_png():
    import zlib, struct

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00")) + chunk(b"IEND", b"")


EMPTY_PNG = _empty_png()


# -------------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    server_version = "PlaneSite/" + __version__
    protocol_version = "HTTP/1.1"     # keep connections open: the map loads many tiles in a row
    disable_nagle_algorithm = True    # else each kept-alive response waits ~40 ms for a delayed ACK

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        headers = dict(headers or {})
        self.send_header("Cache-Control", headers.pop("Cache-Control", None)
                         or ("no-store" if ctype == "application/json" else "max-age=60"))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            path = u.path
            q = parse_qs(u.query)
            if path.startswith("/api/"):
                S.last_seen = time.time()
                return self._send(200, self.api_get(path, q))
            m = re.match(r"^/tiles/(\d+)/(-?\d+)/(-?\d+)\.png$", path)
            if m:
                pf = "pf" in q
                res = tile_png(int(m.group(1)), int(m.group(2)), int(m.group(3)), prefetch=pf)
                if res == "busy":
                    return self._send(503, b"", "text/plain", {"Cache-Control": "no-store"})
                data, ctype = res or (EMPTY_PNG, "image/png")
                # the URL carries the ortho's modification time, so a tile never changes
                return self._send(200, data, ctype, {"Cache-Control": "max-age=31536000, immutable"})
            if path == "/thumb":
                d = (q.get("dir") or [""])[0]
                ok = any(os.path.normcase(d) == os.path.normcase(r) for r in S.cfg.get("recent", []))
                p = os.path.join(d, "outputs", "preview.png")
                if ok and os.path.exists(p):
                    with open(p, "rb") as f:
                        return self._send(200, f.read(), "image/png")
                return self._send(404, b"", "text/plain")
            if path == "/preview.png":
                p = os.path.join(S.outputs(), "preview.png") if S.dir else None
                if p and os.path.exists(p):
                    with open(p, "rb") as f:
                        return self._send(200, f.read(), "image/png")
                return self._send(404, b"", "text/plain")
            if path in ("/", ""):
                path = "/index.html"
            fp = os.path.normpath(os.path.join(WEB_DIR, path.lstrip("/")))
            if not fp.startswith(WEB_DIR) or not os.path.isfile(fp):
                return self._send(404, "Not found", "text/plain")
            ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
            if fp.endswith(".js"):
                ctype = "text/javascript"
            elif fp.endswith(".webmanifest"):
                ctype = "application/manifest+json"
            with open(fp, "rb") as f:
                data = f.read()
            if fp.endswith(".html"):
                data = data.replace(b"{{VERSION}}", __version__.encode())
            if fp.endswith((".html", ".js", ".css", ".webmanifest")):
                # always fresh after an update; the files are local so this costs nothing
                return self._send(200, data, ctype, {"Cache-Control": "no-cache"})
            return self._send(200, data, ctype)
        except ApiError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    def do_POST(self):
        try:
            u = urlparse(self.path)
            body = json_body(self)
            self._send(200, self.api_post(u.path, body) or {"ok": True})
        except ApiError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    # ------------------------------------------------------------------ GET
    def api_get(self, path, q):
        if path == "/api/status":
            return {"version": __version__, "project": project_summary(),
                    "recent": [r for r in S.cfg.get("recent", []) if os.path.exists(P.project_file(r))],
                    "default_dir": S.cfg.get("projects_dir") or default_projects_dir(),
                    "running": S.running() if S.dir else False, "progress": progress_state(),
                    "scan": S.scan, "export": S.export, "platform": sys.platform,
                    "update": dict(updater.state)}
        if path == "/api/projects":
            return {"projects": recent_projects(), "dir": S.cfg.get("projects_dir") or default_projects_dir()}
        if path == "/api/pilot2/settings":
            return pilot2_settings()
        if path == "/api/update/check":
            info = updater.check()
            info.pop("_download", None)
            return info
        if not S.proj:
            raise ApiError("No project is open.")
        if path == "/api/photos":
            return photo_points()
        if path == "/api/log":
            p = os.path.join(S.dir, "log.txt")
            if not os.path.exists(p):
                return {"lines": []}
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            return {"lines": lines[-int(q.get("tail", ["300"])[0]):]}
        if path == "/api/result":
            rep = S.report()
            op = S.ortho_path()
            if not rep or not op:
                return {"ready": False}
            check_ortho_crs()
            import rasterio
            with rasterio.open(op) as ds:
                b = ds.bounds
                gsd = ds.res[0]
            _, e0, n0 = S.origin
            # finest tile level: no finer than the mosaic itself (allowing ~1.2x); beyond that the
            # browser enlarges the last level instead of the server making upsampled tiles
            maxz = int(math.floor(math.log2(R0 / gsd) + 0.27))
            rep = dict(rep)
            cams = rep.pop("cam_positions", [])
            warm_tiles((b.left - e0, b.bottom - n0, b.right - e0, b.top - n0), maxz)
            return {"ready": True, "report": rep, "ortho_path": op, "outputs_dir": S.outputs(),
                    "bounds": [b.left - e0, b.bottom - n0, b.right - e0, b.top - n0], "max_native_zoom": maxz,
                    "cams": [[round(c[0] - e0, 2), round(c[1] - n0, 2)] for c in cams],
                    "mtime": os.path.getmtime(op), "r0": R0}
        if path == "/api/areas":
            return {"areas": areas_for_client(load_areas())}
        if path == "/api/estimate":
            st = {}
            if q.get("preset"):
                st["preset"] = q["preset"][0]
            if q.get("res_factor"):
                st["res_factor"] = float(q["res_factor"][0])
                st["resolution_cm"] = None
            if q.get("compression"):
                st["compression"] = q["compression"][0]
            if q.get("gpu"):
                st["gpu"] = q["gpu"][0]
            return project_estimate(st)
        raise ApiError("Unknown request")

    # ----------------------------------------------------------------- POST
    def api_post(self, path, b):
        if path == "/api/browse":
            return {"path": browse(b.get("mode", "folder"), b.get("title", "Select"), b.get("initial", ""))}
        if path == "/api/clientlog":
            msg = str(b.get("msg", ""))[:2000]
            print("window: %s" % msg)
            if S.dir and S.running():
                try:
                    with open(os.path.join(S.dir, "log.txt"), "a", encoding="utf-8") as f:
                        f.write("[%s] window: %s\n" % (time.strftime("%H:%M:%S"), msg[:500]))
                except OSError:
                    pass
            return {}
        if path == "/api/quit":
            if S.dir and S.running():
                if not b.get("force"):
                    raise ApiError("Processing is running.")
                cancel_processing()
            threading.Timer(0.3, lambda: os._exit(0)).start()
            return {}
        if path == "/api/update/install":
            busy = None
            if S.dir and S.running():
                busy = "Processing is running. Let it finish or cancel it before updating."
            elif S.export and not S.export.get("finished"):
                busy = "An export is running. Wait for it to finish before updating."
            elif S.scan and not S.scan.get("finished"):
                busy = "Photos are still being read. Wait a moment before updating."
            try:
                info = updater.install(busy)
            except RuntimeError as e:
                raise ApiError(str(e))
            return {"version": info["latest"]}
        if path == "/api/project/new":
            name = (b.get("name") or "").strip()
            if not name:
                raise ApiError("Enter a project name.")
            parent = b.get("parent") or default_projects_dir()
            d = os.path.join(parent, safe_name(name))
            if os.path.exists(P.project_file(d)):
                raise ApiError("A project with that name already exists in %s." % parent)
            P.create_project(d, name)
            S.cfg["projects_dir"] = parent
            open_project(d)
            return {"dir": d}
        if path == "/api/project/start":
            d = quick_start(b.get("folder"), b.get("files") or None, b.get("recursive", True), b.get("parent"),
                            b.get("name"))
            return {"dir": d}
        if path == "/api/project/suggest":
            files = [f for f in (b.get("files") or []) if f]
            folder = b.get("folder") or None
            parent = b.get("parent") or S.cfg.get("projects_dir") or default_projects_dir()
            name = (b.get("name") or "").strip() or name_from_source(folder, files or None)
            n_files = len(files)
            return {"name": name, "parent": parent, "dir": unique_project_dir(parent, name), "files": n_files}
        if path == "/api/geotiff/open":
            return {"dir": open_geotiff(b.get("path") or "")}
        if path == "/api/projects_dir":
            p = b.get("path") or ""
            if not p:
                raise ApiError("Choose a folder.")
            os.makedirs(p, exist_ok=True)
            S.cfg["projects_dir"] = p
            S.save_cfg()
            return {"path": p}
        if path == "/api/recent/forget":
            S.cfg["recent"] = [r for r in S.cfg.get("recent", []) if r != b.get("dir")]
            S.save_cfg()
            return {}
        if path == "/api/project/open":
            open_project(b.get("dir", ""))
            return {"dir": S.dir}
        if path == "/api/project/close":
            if S.running():
                raise ApiError("Processing is running.")
            S.dir = S.proj = None
            S.scan = S.export = None
            return {}
        if path == "/api/open_path":
            p = b.get("path") or (S.dir if S.dir else "")
            if p and os.path.exists(p):
                open_in_explorer(p)
            return {}
        if not S.proj:
            raise ApiError("No project is open.")
        if path == "/api/photos/add":
            if S.proj.get("geotiff"):
                raise ApiError("This project shows an imported GeoTIFF. Start a new project for photos.")
            files = b.get("files") or None
            folder = None if files else os.path.normpath(b.get("folder") or "")
            start_scan(folder, files, b.get("recursive", True))
            return {}
        if path == "/api/photos/remove_folder":
            folder = b.get("folder")
            nf = os.path.normcase(os.path.normpath(folder))
            S.proj["photos"] = [m for m in S.proj["photos"]
                                if not os.path.normcase(os.path.normpath(m["path"])).startswith(nf + os.sep)
                                and os.path.normcase(os.path.dirname(m["path"])) != nf]
            S.proj["folders"] = [f for f in S.proj["folders"] if f != folder]
            S.proj["selection"] = None
            S.proj.pop("map_origin", None)
            S.compute_origin()
            S.save()
            return {}
        if path == "/api/photos/select":
            if b.get("clear"):
                S.proj["selection"] = None
            else:
                ids = b.get("ids") or []
                S.proj["selection"] = sorted(set(int(i) for i in ids)) or None
            S.save()
            return {}
        if path == "/api/diagnostics":
            p = save_diagnostics()
            if IS_WIN:
                subprocess.Popen(["explorer", "/select,", p])
            else:
                open_in_explorer(os.path.dirname(p))
            return {"path": p}
        if path == "/api/settings":
            S.proj["settings"].update(b)
            S.save()
            return {}
        if path == "/api/rename":
            return rename_project((b.get("name") or "").strip())
        if path == "/api/process/start":
            start_processing(b.get("settings"))
            return {}
        if path == "/api/process/cancel":
            cancel_processing()
            return {}
        if path == "/api/pilot2/settings":
            return pilot2_set(b)
        if path == "/api/pilot2/preview":
            return pilot2_preview(b)
        if path == "/api/pilot2/send":
            return pilot2_send(b)
        if path == "/api/areas":
            save_areas(areas_from_client(b.get("areas", [])))
            return {"areas": areas_for_client(load_areas())}
        if path == "/api/areas/import":
            from .export import read_boundaries, to_ortho_crs
            p = b.get("path") or ""
            if not os.path.isfile(p):
                raise ApiError("File not found: %s" % p)
            bnds, crs = read_boundaries(p)
            if not bnds:
                raise ApiError("No polygons found in that file.")
            new = to_ortho_crs(bnds, crs, S.origin[0])
            areas = load_areas() + new
            save_areas(areas)
            return {"areas": areas_for_client(areas), "added": len(new)}
        if path == "/api/areas/grid":
            from .export import grid_areas
            op = S.ortho_path()
            if not op:
                raise ApiError("Process the photos first.")
            new = grid_areas(op, b.get("mode", "size"), float(b.get("size_m") or 250), int(b.get("rows") or 2),
                             int(b.get("cols") or 2), b.get("prefix") or "Grid")
            areas = load_areas() + new
            save_areas(areas)
            return {"areas": areas_for_client(areas), "added": len(new)}
        if path == "/api/areas/kml":
            from .export import areas_to_kml
            areas = [a for a in load_areas() if not b.get("ids") or a["id"] in b["ids"]]
            out_dir = b.get("out_dir") or os.path.join(S.dir, "exports")
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, "%s_areas.kml" % safe_name(S.proj["name"]))
            with open(p, "w", encoding="utf-8") as f:
                f.write(areas_to_kml(areas, S.origin[0]))
            return {"path": p}
        if path == "/api/export/estimate":
            return export_estimate(b)
        if path == "/api/export":
            if not S.ortho_path():
                raise ApiError("Process the photos first.")
            if S.export and not S.export.get("finished"):
                raise ApiError("An export is already running.")
            ids = b.get("ids") or []
            whole = bool(b.get("whole"))
            if not ids and not whole:
                raise ApiError("Choose the whole orthomosaic or at least one area to export.")
            scale = float(b.get("scale") or 1.0)
            if not 0.05 <= scale <= 1.0:
                raise ApiError("Scale must be between 5 and 100%.")
            out_dir = b.get("out_dir") or os.path.join(S.dir, "exports")
            S.export = {"done": 0, "total": len(ids) + (1 if whole else 0), "current": "", "frac": 0,
                        "files": [], "errors": [], "out_dir": out_dir, "finished": False}
            threading.Thread(target=export_worker,
                             args=(ids, out_dir, b.get("compression", "same"), scale,
                                   bool(b.get("overviews", True)), whole), daemon=True).start()
            return {}
        raise ApiError("Unknown request")


def _window_trace(browser, state):
    """Note in the log when an open window stops asking for updates (it polls every 1-3 s) and
    when it starts again, so a frozen window leaves a trace."""
    if not S.last_seen or (browser is not None and browser.poll() is not None):
        return
    idle = time.time() - S.last_seen
    if idle > 15 and not state.get("quiet"):
        state["quiet"] = S.last_seen
        pr = progress_state() or {}
        print("%s window stopped checking in at %s (processing: %s, %s)" % (
            time.strftime("%H:%M:%S"), time.strftime("%H:%M:%S", time.localtime(S.last_seen)),
            bool(S.dir and S.running()), pr.get("message", "")))
    elif idle < 5 and state.get("quiet"):
        print("%s window responding again after %.0f s" % (time.strftime("%H:%M:%S"), time.time() - state["quiet"]))
        state["quiet"] = 0


def lifetime_watch(srv, browser):
    """Quit when the app window is gone and nothing is running."""
    started = time.time()
    trace = {}
    while True:
        time.sleep(5)
        try:
            _window_trace(browser, trace)
        except Exception:
            pass
        busy = (S.dir and S.running()) or (S.scan and not S.scan.get("finished")) \
            or (S.export and not S.export.get("finished")) \
            or updater.state.get("phase") in ("downloading", "installing", "restarting")
        if busy:
            continue
        idle = time.time() - S.last_seen
        if browser is not None and browser.poll() is None:
            continue  # our app window is still open (possibly minimised)
        if S.last_seen == 0:
            if browser is not None and time.time() - started > 120:
                break  # window closed before the page ever loaded
            continue
        limit = 20 if browser is not None else 180
        if idle > limit:
            break
    print("No window open and nothing running - closing PlaneSite.")
    os._exit(0)


def serve(port=8765, open_browser=True):
    srv = None
    for p in range(port, port + 30):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit("Could not find a free local port.")
    srv.daemon_threads = True
    updater.PORT[0] = port
    url = "http://127.0.0.1:%d/" % port
    print("PlaneSite %s is running at %s" % (__version__, url))
    browser = None
    if open_browser:
        from .appwindow import open_window
        browser = open_window(url)
    threading.Thread(target=lifetime_watch, args=(srv, browser), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
