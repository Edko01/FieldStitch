"""Send a field to DJI Pilot 2: plan-only mission file (KMZ) and upload to the PlaneSite hub.

Spec: claude/hub-integration.md in the PlaneSite project. The KMZ holds only wpmz/template.kml
(no waylines.wpml); Pilot 2 builds the flight lines itself when the pilot downloads the mission.
The template is Justin's own Pilot 2 export (SnowFlakeNorth1) with placeholders; everything else,
including DJI's spelling "visable", stays as Pilot 2 wrote it.
"""
import io
import json
import math
import os
import re
import sys
import time
import uuid
import zipfile

import numpy as np

TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:wpml="http://www.dji.com/wpmz/1.0.6">
  <Document>
    <wpml:createTime>{CREATE_MS}</wpml:createTime>
    <wpml:updateTime>{UPDATE_MS}</wpml:updateTime>
    <wpml:missionConfig>
      <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
      <wpml:finishAction>goHome</wpml:finishAction>
      <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>
      <wpml:executeRCLostAction>goBack</wpml:executeRCLostAction>
      <wpml:takeOffSecurityHeight>{TAKEOFF_SECURITY_HEIGHT}</wpml:takeOffSecurityHeight>
      <wpml:globalTransitionalSpeed>{SPEED}</wpml:globalTransitionalSpeed>
      <wpml:droneInfo>
        <wpml:droneEnumValue>99</wpml:droneEnumValue>
        <wpml:droneSubEnumValue>0</wpml:droneSubEnumValue>
      </wpml:droneInfo>
      <wpml:waylineAvoidLimitAreaMode>0</wpml:waylineAvoidLimitAreaMode>
      <wpml:payloadInfo>
        <wpml:payloadEnumValue>88</wpml:payloadEnumValue>
        <wpml:payloadSubEnumValue>0</wpml:payloadSubEnumValue>
        <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
      </wpml:payloadInfo>
    </wpml:missionConfig>
    <Folder>
      <wpml:templateType>mapping2d</wpml:templateType>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineCoordinateSysParam>
        <wpml:coordinateMode>WGS84</wpml:coordinateMode>
        <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
        <wpml:globalShootHeight>{HEIGHT}</wpml:globalShootHeight>
        <wpml:surfaceFollowModeEnable>{TERRAIN}</wpml:surfaceFollowModeEnable>
        <wpml:isRealtimeSurfaceFollow>{TERRAIN}</wpml:isRealtimeSurfaceFollow>
        <wpml:surfaceRelativeHeight>{HEIGHT}</wpml:surfaceRelativeHeight>
      </wpml:waylineCoordinateSysParam>
      <wpml:autoFlightSpeed>{SPEED}</wpml:autoFlightSpeed>
      <Placemark>
        <wpml:caliFlightEnable>0</wpml:caliFlightEnable>
        <wpml:elevationOptimizeEnable>{ELEV_OPT}</wpml:elevationOptimizeEnable>
        <wpml:smartObliqueEnable>0</wpml:smartObliqueEnable>
        <wpml:quickOrthoMappingEnable>0</wpml:quickOrthoMappingEnable>
        <wpml:facadeWaylineEnable>0</wpml:facadeWaylineEnable>
        <wpml:isLookAtSceneSet>0</wpml:isLookAtSceneSet>
        <wpml:smartObliqueGimbalPitch>-45</wpml:smartObliqueGimbalPitch>
        <wpml:shootType>distance</wpml:shootType>
        <wpml:direction>{DIRECTION}</wpml:direction>
        <wpml:margin>{MARGIN}</wpml:margin>
        <wpml:efficiencyFlightModeEnable>0</wpml:efficiencyFlightModeEnable>
        <wpml:doubleGridWaylineEnable>0</wpml:doubleGridWaylineEnable>
        <wpml:overlap>
          <wpml:orthoLidarOverlapH>{OVERLAP_H}</wpml:orthoLidarOverlapH>
          <wpml:orthoLidarOverlapW>{OVERLAP_W}</wpml:orthoLidarOverlapW>
          <wpml:orthoCameraOverlapH>{OVERLAP_H}</wpml:orthoCameraOverlapH>
          <wpml:orthoCameraOverlapW>{OVERLAP_W}</wpml:orthoCameraOverlapW>
        </wpml:overlap>
        <Polygon>
          <outerBoundaryIs>
            <LinearRing>
              <coordinates>
{COORDS}
              </coordinates>
            </LinearRing>
          </outerBoundaryIs>
        </Polygon>
        <wpml:ellipsoidHeight>{HEIGHT}</wpml:ellipsoidHeight>
        <wpml:height>{HEIGHT}</wpml:height>
      </Placemark>
      <wpml:payloadParam>
        <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
        <wpml:imageFormat>visable</wpml:imageFormat>
      </wpml:payloadParam>
    </Folder>
  </Document>
</kml>
"""

# Justin's defaults (hub-integration.md)
DEFAULTS = {"height_m": 111.771522521973, "speed_ms": 15.0, "direction": 90.0, "auto_direction": False,
            "margin_m": 15.0, "overlap_front": 80, "overlap_side": 70, "battery_min": 35.0,
            # Pilot 2's "Real-Time Follow" (terrain follow on the aircraft's own sensors, at the
            # flight height above the ground) and "Elevation Optimization" (a short oblique pass at
            # the end for better elevation accuracy)
            "terrain_follow": True, "elevation_optimize": False}
_BOOLS = ("auto_direction", "terrain_follow", "elevation_optimize")

# M4E wide camera
SENSOR_W_MM, SENSOR_H_MM, FOCAL_MM, IMAGE_W_PX = 17.3, 13.0, 12.29, 5280
DEFAULT_HUB = "https://planesite-hub.justin-anderson.workers.dev"
MAX_KMZ_BYTES = int(1.9 * 1024 * 1024)
BAD_NAME_CHARS = r'\/:*?"<>|'


def _num(v):
    """Number as Pilot 2 writes it: integers without a decimal point, others as given."""
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return repr(v)


def clean_name(name, limit=80):
    s = "".join(c for c in (name or "") if c not in BAD_NAME_CHARS and ord(c) >= 32)
    s = re.sub(r"\s+", " ", s).strip().strip(".")
    if len(s) > limit:
        s = s[:limit].rstrip(" -_.")
    return s or "PlaneSite mission"


# ------------------------------------------------------------------------------ boundary
def _simplify(pts, tol):
    """Douglas-Peucker on an open polyline (n, 2)."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = q - p
        L = float(np.hypot(*seg))
        rel = pts[a + 1:b] - p
        if L < 1e-9:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def mission_ring(area, simplify_m=0.5):
    """Outer ring of a field (metres, closed), cleaned up for a mission: holes dropped, repeated
    points removed, outline simplified (~0.5 m). Returns (ring (n, 2), notes)."""
    notes = []
    rings = area.get("rings") or []
    if not rings or len(rings[0]) < 4:
        raise ValueError("The field has no usable outline.")
    if len(rings) > 1:
        notes.append("Holes in the outline were left out (a mission covers the whole outer boundary).")
    r = np.asarray(rings[0], np.float64)[:, :2]
    if np.allclose(r[0], r[-1]):
        r = r[:-1]
    keep = np.r_[True, np.hypot(*(np.diff(r, axis=0).T)) > 1e-3]
    r = r[keep]
    if len(r) < 3:
        raise ValueError("The field outline has fewer than 3 corners.")
    # simplify as a closed loop: split at the point farthest from the first one
    far = int(np.argmax(np.hypot(*(r - r[0]).T)))
    a = _simplify(r[:far + 1], simplify_m)
    b = _simplify(np.vstack([r[far:], r[:1]]), simplify_m)
    s = np.vstack([a, b[1:-1]]) if len(b) > 2 else a
    if len(s) < 3:
        s = r
    if len(s) < len(r) and len(r) > 20:
        notes.append("Outline simplified from %d to %d corners (within %.1f ft)." % (len(r), len(s), simplify_m / 0.3048))
    return np.vstack([s, s[:1]]), notes


def ring_lonlat(ring_xy, epsg):
    from pyproj import Transformer
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = tr.transform(ring_xy[:, 0], ring_xy[:, 1])
    return np.c_[lon, lat]


def long_axis_heading(ring_xy):
    """Heading (degrees clockwise from north, 0-179) of the longer side of the smallest rectangle
    around the field: flight lines along it need the fewest turns."""
    from scipy.spatial import ConvexHull
    p = ring_xy[:-1] if np.allclose(ring_xy[0], ring_xy[-1]) else ring_xy
    try:
        h = p[ConvexHull(p).vertices]
    except Exception:
        h = p
    best = None
    for k in range(len(h)):
        e = h[(k + 1) % len(h)] - h[k]
        if np.hypot(*e) < 1e-6:
            continue
        ang = math.atan2(e[1], e[0])
        c, s = math.cos(ang), math.sin(ang)
        u = h[:, 0] * c + h[:, 1] * s
        v = -h[:, 0] * s + h[:, 1] * c
        du, dv = np.ptp(u), np.ptp(v)
        area = du * dv
        if best is None or area < best[0]:
            long_ang = ang if du >= dv else ang + math.pi / 2
            best = (area, long_ang)
    ang = best[1] if best else 0.0
    # direction vector (cos ang, sin ang) in east/north -> heading from north, clockwise
    hdg = math.degrees(math.atan2(math.cos(ang), math.sin(ang))) % 180.0
    return round(hdg, 1)


# ------------------------------------------------------------------------------ estimates
def estimates(ring_xy, s):
    """Office estimates (Pilot 2 computes the real path): spacing, line length, photos, time."""
    H = float(s["height_m"])
    across = SENSOR_W_MM * H / FOCAL_MM
    along = SENSOR_H_MM * H / FOCAL_MM
    line_sp = across * (1 - float(s["overlap_side"]) / 100.0)
    photo_sp = along * (1 - float(s["overlap_front"]) / 100.0)
    m = float(s["margin_m"])
    p = ring_xy[:-1]
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.sum(np.hypot(*(np.diff(ring_xy, axis=0).T))))
    covered = area + perim * m + math.pi * m * m          # field plus the edge buffer
    length = covered / max(line_sp, 0.1)
    hdg = math.radians(float(s["direction"]))
    across_dir = np.array([math.cos(hdg), -math.sin(hdg)])  # perpendicular to the flight lines
    width = float(np.ptp(p @ across_dir)) + 2 * m
    lines = max(1, int(math.ceil(width / max(line_sp, 0.1))))
    speed = max(float(s["speed_ms"]), 0.5)
    turn_s = 3.0                                           # slowing, turning, speeding up again
    secs = length / speed + (lines - 1) * (line_sp / speed + turn_s)
    photos = int(math.ceil(length / max(photo_sp, 0.1))) + lines
    batt = max(1, int(math.ceil(secs / 60.0 / max(float(s.get("battery_min", 35)), 5))))
    gsd_cm = 100 * H * SENSOR_W_MM / FOCAL_MM / IMAGE_W_PX
    return {"area_ha": round(float(area) / 1e4, 2), "area_ac": round(float(area) / 4046.8564224, 1),
            "gsd_cm": round(gsd_cm, 2), "line_spacing_m": round(line_sp, 1), "photo_spacing_m": round(photo_sp, 1),
            "lines": lines, "length_km": round(float(length) / 1000, 1), "minutes": int(round(secs / 60)),
            "photos": photos, "batteries": batt,
            "gsd_cm_per_m": 100 * SENSOR_W_MM / FOCAL_MM / IMAGE_W_PX}   # the window links GSD and height


def mission_settings(given=None):
    s = dict(DEFAULTS)
    for k, v in (given or {}).items():
        if k in DEFAULTS and v is not None and v != "":
            s[k] = bool(v) if k in _BOOLS else float(v)
    s["height_m"] = min(max(s["height_m"], 12.0), 500.0)
    s["speed_ms"] = min(max(s["speed_ms"], 1.0), 15.0)
    s["margin_m"] = min(max(s["margin_m"], 0.0), 200.0)
    s["overlap_front"] = int(round(min(max(s["overlap_front"], 10), 95)))
    s["overlap_side"] = int(round(min(max(s["overlap_side"], 10), 95)))
    s["direction"] = float(s["direction"]) % 360.0
    return s


# ------------------------------------------------------------------------------ the file
def template_kml(ring_ll, s, now_ms=None):
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    coords = "\n".join("                %.9f,%.9f,0" % (lon, lat) for lon, lat in ring_ll)
    vals = {"CREATE_MS": str(now_ms), "UPDATE_MS": str(now_ms), "TAKEOFF_SECURITY_HEIGHT": _num(s["height_m"]),
            "HEIGHT": _num(s["height_m"]), "SPEED": _num(s["speed_ms"]), "DIRECTION": _num(round(s["direction"])),
            "MARGIN": _num(s["margin_m"]), "OVERLAP_H": _num(s["overlap_front"]), "OVERLAP_W": _num(s["overlap_side"]),
            "TERRAIN": "1" if s.get("terrain_follow", True) else "0",
            "ELEV_OPT": "1" if s.get("elevation_optimize") else "0",
            "COORDS": coords}
    out = TEMPLATE
    for k, v in vals.items():
        out = out.replace("{%s}" % k, v)
    return out


def build_kmz(ring_ll, s, now_ms=None):
    """KMZ bytes holding exactly wpmz/template.kml."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wpmz/template.kml", template_kml(ring_ll, s, now_ms).encode("utf-8"))
    data = buf.getvalue()
    if len(data) > MAX_KMZ_BYTES:
        raise ValueError("The mission file is too large for the hub (%.1f MB)." % (len(data) / 1e6))
    return data


def prepare(area, epsg, given=None):
    """Everything the dialog shows and the send uses: settings, cleaned ring, estimates, notes."""
    s = mission_settings(given)
    ring, notes = mission_ring(area)
    s["long_axis"] = long_axis_heading(ring)
    if s["auto_direction"]:
        s["direction"] = s["long_axis"]
    est = estimates(ring, s)
    return s, ring, ring_lonlat(ring, epsg), est, notes


# ------------------------------------------------------------------------------ the hub
def upload(base_url, key, kmz, filename, name=None, user=None, timeout=60):
    """POST the KMZ to the hub. Same name replaces the existing mission. Returns the hub's JSON."""
    import urllib.error
    import urllib.request
    if not key:
        raise PermissionError("Enter the hub's office password first.")
    url = (base_url or DEFAULT_HUB).rstrip("/") + "/admin/api/missions"
    bnd = "----PlaneSite" + uuid.uuid4().hex
    parts = []

    def field(n, v):
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n" % (bnd, n)).encode("utf-8")
                     + v.encode("utf-8") + b"\r\n")
    if name:
        field("name", name)
    if user:
        field("user", user)
    fn = filename.replace('"', "")
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
                  "Content-Type: application/vnd.google-earth.kmz\r\n\r\n" % (bnd, fn)).encode("utf-8") + kmz + b"\r\n")
    parts.append(("--%s--\r\n" % bnd).encode("utf-8"))
    body = b"".join(parts)
    from . import __version__
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "multipart/form-data; boundary=%s" % bnd, "x-admin-key": key,
        "User-Agent": "PlaneSite/%s" % __version__})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error")
        except Exception:
            msg = None
        if e.code == 401:
            raise PermissionError("The hub did not accept the office password.")
        raise RuntimeError("The hub refused the mission (HTTP %d)%s" % (e.code, ": " + msg if msg else "."))
    except urllib.error.URLError as e:
        raise ConnectionError("Can't reach the hub (%s). Check the internet connection and the hub address."
                              % getattr(e, "reason", e))


# ------------------------------------------------------------------------------ the password
_CRED_TARGET = "PlaneSite/hub-office-password"


def _cred_file():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "PlaneSite", "hub_key")


def save_key(key):
    """Keep the office password in Windows Credential Manager (a private file elsewhere)."""
    if sys.platform == "win32":
        _win_cred_write(_CRED_TARGET, key or "")
        return
    p = _cred_file()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not key:
        if os.path.exists(p):
            os.remove(p)
        return
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)


def load_key():
    if sys.platform == "win32":
        return _win_cred_read(_CRED_TARGET)
    try:
        with open(_cred_file()) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _win_cred_struct():
    import ctypes
    from ctypes import wintypes

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
                    ("Comment", wintypes.LPWSTR), ("LastWritten", wintypes.FILETIME),
                    ("CredentialBlobSize", wintypes.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                    ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
                    ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]
    return CREDENTIAL


def _win_cred_write(target, secret, comment="PlaneSite: office password for the Pilot 2 hub"):
    import ctypes
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    if not secret:
        adv.CredDeleteW(ctypes.c_wchar_p(target), 1, 0)
        return
    CREDENTIAL = _win_cred_struct()
    blob = secret.encode("utf-16-le")
    buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
    c = CREDENTIAL()
    c.Type = 1                       # CRED_TYPE_GENERIC
    c.TargetName = target
    c.Comment = comment
    c.CredentialBlobSize = len(blob)
    c.CredentialBlob = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
    c.Persist = 2                    # CRED_PERSIST_LOCAL_MACHINE (this Windows user, this PC)
    c.UserName = "PlaneSite"
    if not adv.CredWriteW(ctypes.byref(c), 0):
        raise OSError("Could not save the password in Windows Credential Manager (error %d)." % ctypes.get_last_error())


def _win_cred_read(target):
    import ctypes
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    CREDENTIAL = _win_cred_struct()
    p = ctypes.POINTER(CREDENTIAL)()
    if not adv.CredReadW(ctypes.c_wchar_p(target), 1, 0, ctypes.byref(p)):
        return None
    try:
        c = p.contents
        raw = ctypes.string_at(c.CredentialBlob, c.CredentialBlobSize)
        return raw.decode("utf-16-le") or None
    finally:
        adv.CredFree(p)
