"""Read EXIF + DJI XMP metadata needed for photogrammetry."""
import math
import os
import re
from datetime import datetime

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg"}

_XMP_ATTR = re.compile(rb'([A-Za-z0-9_\-]+):([A-Za-z0-9_]+)="([^"]*)"')
_XMP_ELEM = re.compile(rb'<([A-Za-z0-9_\-]+):([A-Za-z0-9_]+)>([^<]*)</\1:\2>')


def read_xmp(path, max_bytes=1024 * 1024):
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    s = data.find(b"<x:xmpmeta")
    if s < 0:
        s = data.find(b"<rdf:RDF")
    if s < 0:
        return {}
    e = data.find(b"</x:xmpmeta>", s)
    if e < 0:
        e = len(data)
    chunk = data[s:e]
    out = {}
    for rx in (_XMP_ATTR, _XMP_ELEM):
        for m in rx.finditer(chunk):
            ns = m.group(1).decode("ascii", "replace")
            k = m.group(2).decode("ascii", "replace")
            v = m.group(3).decode("utf-8", "replace").strip()
            if ns == "xmlns":
                continue
            out.setdefault(k, v)
            out[ns + ":" + k] = v
    return out


def _f(v, default=None):
    try:
        if v is None:
            return default
        if isinstance(v, (tuple, list)) and len(v) == 2 and not hasattr(v[0], "numerator"):
            return float(v[0]) / float(v[1]) if float(v[1]) else default
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _dms(v):
    try:
        d, m, s = [_f(x, 0.0) for x in v]
        return d + m / 60.0 + s / 3600.0
    except Exception:
        return None


def _parse_time(s, subsec=None):
    if not s:
        return None
    s = str(s).strip().replace("\x00", "")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            t = datetime.strptime(s[:19], fmt).timestamp()
            if subsec:
                try:
                    t += float("0." + str(subsec).strip())
                except Exception:
                    pass
            return t
        except Exception:
            continue
    return None


def read_image_meta(path):
    """Return a dict with geometry-relevant metadata, or raise."""
    with Image.open(path) as im:
        W, H = im.size
        exif = im.getexif()
        try:
            ifd = exif.get_ifd(0x8769)
        except Exception:
            ifd = {}
        try:
            gps = exif.get_ifd(0x8825)
        except Exception:
            gps = {}
    xmp = read_xmp(path)

    make = str(exif.get(271, "") or xmp.get("Make", "")).strip("\x00 ").strip()
    model = str(exif.get(272, "") or xmp.get("Model", "")).strip("\x00 ").strip()
    orientation = int(exif.get(274, 1) or 1)

    lat = lon = alt = None
    if gps:
        lat = _dms(gps.get(2)) if gps.get(2) else None
        lon = _dms(gps.get(4)) if gps.get(4) else None
        if lat is not None and str(gps.get(1, "N")).upper().startswith("S"):
            lat = -lat
        if lon is not None and str(gps.get(3, "E")).upper().startswith("W"):
            lon = -lon
        alt = _f(gps.get(6))
        if alt is not None and gps.get(5) in (1, b"\x01"):
            alt = -alt
    # DJI XMP overrides (higher precision, RTK)
    xlat = _f(xmp.get("GpsLatitude"))
    xlon = _f(xmp.get("GpsLongitude", xmp.get("GpsLongtitude")))
    if xlat is not None and xlon is not None and (abs(xlat) > 1e-9 or abs(xlon) > 1e-9):
        lat, lon = xlat, xlon
    abs_alt = _f(xmp.get("AbsoluteAltitude"))
    if abs_alt is not None:
        alt = abs_alt
    rel_alt = _f(xmp.get("RelativeAltitude"))

    yaw = _f(xmp.get("GimbalYawDegree"))
    pitch = _f(xmp.get("GimbalPitchDegree"))
    roll = _f(xmp.get("GimbalRollDegree"))
    if yaw is None:
        # Some cameras write Camera:Yaw/Pitch/Roll
        yaw = _f(xmp.get("Camera:Yaw"))
        pitch = _f(xmp.get("Camera:Pitch"))
        roll = _f(xmp.get("Camera:Roll"))
        if pitch is not None:
            pitch = pitch - 90.0  # Camera:Pitch 0 == nadir
    rtk_flag = _f(xmp.get("RtkFlag"))
    rtk_std = [_f(xmp.get(k)) for k in ("RtkStdLon", "RtkStdLat", "RtkStdHgt")]

    focal_mm = _f(ifd.get(37386))
    f35 = _f(ifd.get(41989))
    fpx_res = _f(ifd.get(41486))
    fp_unit = ifd.get(41488)
    t = _parse_time(ifd.get(36867) or exif.get(306), ifd.get(37521))

    # ---- intrinsics estimate (pixels) -------------------------------------------------
    diag = math.hypot(W, H)
    f_px = None
    f_src = ""
    f35_px = f35 * diag / 43.2666 if f35 else None
    if not f35_px and focal_mm and fpx_res and fpx_res > 0:
        unit_mm = {2: 25.4, 3: 10.0, 4: 1.0}.get(int(fp_unit or 2), 25.4)
        f35_px = focal_mm * fpx_res / unit_mm
    cx = W / 2.0
    cy = H / 2.0
    dist = [0.0, 0.0, 0.0, 0.0, 0.0]  # k1 k2 p1 p2 k3
    cal = _f(xmp.get("CalibratedFocalLength"))
    if cal and 0.3 * W < cal < 3 * W and (not f35_px or abs(cal / f35_px - 1) < 0.15):
        f_px, f_src = cal, "dji-calibrated"
        ocx = _f(xmp.get("CalibratedOpticalCenterX"))
        ocy = _f(xmp.get("CalibratedOpticalCenterY"))
        if ocx and ocy and abs(ocx - W / 2) < 0.1 * W and abs(ocy - H / 2) < 0.1 * H:
            cx, cy = ocx, ocy
    dewarp = xmp.get("DewarpData")
    dewarp_flag = _f(xmp.get("DewarpFlag"), 0)
    if dewarp and not dewarp_flag:
        try:
            vals = [float(x) for x in dewarp.split(";")[-1].split(",")]
            if len(vals) >= 9:
                fx, fy, dcx, dcy, k1, k2, p1, p2, k3 = vals[:9]
                fd = 0.5 * (fx + fy)
                if 0.3 * W < fd < 3 * W and (not f35_px or abs(fd / f35_px - 1) < 0.15):
                    if not f_px:
                        f_px, f_src = fd, "dji-dewarp"
                    cx, cy = W / 2.0 + dcx, H / 2.0 + dcy
                    dist = [k1, k2, p1, p2, k3]
        except Exception:
            pass
    if not f_px and f35_px:
        f_px, f_src = f35_px, "exif"
    if not f_px:
        f_px, f_src = 0.85 * max(W, H), "default"

    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "w": W,
        "h": H,
        "make": make,
        "model": model,
        "orientation": orientation,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "rel_alt": rel_alt,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "rtk_flag": rtk_flag,
        "rtk_std": rtk_std,
        "focal_mm": focal_mm,
        "f35": f35,
        "f_px": f_px,
        "f_src": f_src,
        "cx": cx,
        "cy": cy,
        "dist": dist,
        "time": t,
        "source": xmp.get("ImageSource", ""),
    }


def camera_key(m):
    return "%s %s %dx%d f%.1f %s" % (m.get("make", ""), m.get("model", ""), m["w"], m["h"],
                                      m.get("focal_mm") or 0, m.get("source", ""))


SKIP_DIRS = {"planesite", "_planesite", "fieldstitch", "_fieldstitch", "work", "thumbs", "outputs", "exports", "$recycle.bin",
             "system volume information"}


def _is_project_dir(path):
    """PlaneSite (or other processing) project folders hold derived images, not flight photos."""
    return (os.path.isfile(os.path.join(path, "project.json"))
            or os.path.isdir(os.path.join(path, "work", "thumbs")))


def find_images(folder, recursive=True):
    out = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            if root != folder and _is_project_dir(root):
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in SKIP_DIRS]
            for fn in files:
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS and not fn.startswith("._"):
                    out.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in IMAGE_EXTS and not fn.startswith("._"):
                out.append(p)
    out.sort()
    return out
