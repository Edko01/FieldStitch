"""Satellite base map and place lookup for planning before there is an orthomosaic.

Base map: USDA NAIP aerial imagery (public domain, about 0.6 m, flown every 2-3 years per state)
from the USGS National Map image service. The service reprojects on request, so each map tile is
asked for directly in the job's UTM frame and lines up with the drawn fields and, later, the
orthomosaic. Tiles are kept in the project folder (basemap/), so a job planned in the office still
shows its imagery offline.

Place lookup: coordinates in most written forms (decimal, degrees-minutes-seconds, Google Maps
links) without internet; street addresses through the US Census Bureau geocoder.
"""
import json
import math
import os
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request

NAIP_EXPORT = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
CENSUS = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
NOMINATIM = "https://nominatim.openstreetmap.org/search"     # towns, counties, roads, landmarks
CREDIT = "Imagery: USDA NAIP via USGS The National Map"
TILE = 256

_slots = threading.BoundedSemaphore(6)      # requests to the image service at once
_UA = {"User-Agent": "PlaneSite/2.0 (drone mapping desktop app; github.com/Edko01/PlaneSite)"}
BLOCK = 4                                    # tiles are fetched 4 x 4 at a time (one 1024 px image)
_block_locks = {}
_block_locks_lock = threading.Lock()
ERROR = "error"


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context())


def tile_bounds(z, x, y, origin, r0):
    """UTM bounds (xmin, ymin, xmax, ymax) of map tile z/x/y. The map's units are metres from the
    origin; one screen pixel at zoom z is r0 / 2^z metres, and y grows downward (north is -y)."""
    epsg, e0, n0 = origin
    m = TILE * r0 / (2.0 ** z)
    return e0 + x * m, n0 - (y + 1) * m, e0 + (x + 1) * m, n0 - y * m


def _cached(d, x, y):
    """(bytes, type) / None (no imagery there) / False (not fetched yet)."""
    for ext, ctype in ((".jpg", "image/jpeg"), (".png", "image/png"), (".none", None)):
        p = os.path.join(d, "%d_%d%s" % (x, y, ext))
        if os.path.exists(p):
            if ctype is None:
                return None
            with open(p, "rb") as f:
                return f.read(), ctype
    return False


def tile(cache_dir, z, x, y, origin, r0):
    """(bytes, content type) of one base-map tile, None where there's no imagery, or ERROR when the
    image service couldn't be reached (the window tries again).

    The image service takes about as long for a 1024 px image as for a 256 px one, so tiles are
    fetched in blocks of 4 x 4: the first tile of a block asked for fetches all 16, and the others
    come from disk."""
    epsg, e0, n0 = origin
    d = os.path.join(cache_dir, "%d_%d_%d" % (epsg, e0, n0), str(z))
    hit = _cached(d, x, y)
    if hit is not False:
        return hit
    bx, by = (x // BLOCK) * BLOCK, (y // BLOCK) * BLOCK
    key = (d, bx, by)
    with _block_locks_lock:
        lk = _block_locks.setdefault(key, threading.Lock())
    with lk:
        hit = _cached(d, x, y)            # another request may have fetched the block meanwhile
        if hit is not False:
            return hit
        if not _fetch_block(d, z, bx, by, origin, r0):
            return ERROR
    with _block_locks_lock:
        _block_locks.pop(key, None)
    hit = _cached(d, x, y)
    return None if hit is False else hit


def _fetch_block(d, z, bx, by, origin, r0):
    import io
    import numpy as np
    from PIL import Image
    epsg = origin[0]
    xmin, _, _, ymax = tile_bounds(z, bx, by, origin, r0)
    _, ymin, xmax, _ = tile_bounds(z, bx + BLOCK - 1, by + BLOCK - 1, origin, r0)
    n = TILE * BLOCK
    q = urllib.parse.urlencode({
        "bbox": "%.3f,%.3f,%.3f,%.3f" % (xmin, ymin, xmax, ymax), "bboxSR": epsg, "imageSR": epsg,
        "size": "%d,%d" % (n, n), "format": "jpgpng", "transparent": "true", "bandIds": "0,1,2",
        "interpolation": "RSP_BilinearInterpolation", "f": "image"})
    data = None
    for attempt in range(2):
        with _slots:
            try:
                with _get(NAIP_EXPORT + "?" + q, timeout=30) as r:
                    data = r.read()
                    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
                break
            except Exception:
                data = None
        time.sleep(0.5)
    if not data or not ctype.startswith("image/"):
        return False
    try:
        im = np.array(Image.open(io.BytesIO(data)).convert("RGBA"))
    except Exception:
        return False
    if im.shape[0] != n or im.shape[1] != n:
        im = np.array(Image.fromarray(im).resize((n, n)))
    os.makedirs(d, exist_ok=True)
    for j in range(BLOCK):
        for i in range(BLOCK):
            t = im[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE]
            a = t[:, :, 3]
            stem = os.path.join(d, "%d_%d" % (bx + i, by + j))
            if a.max() == 0:
                open(stem + ".none", "wb").close()          # outside the imagery
                continue
            b = io.BytesIO()
            if a.min() == 255:
                Image.fromarray(t[:, :, :3]).save(b, "JPEG", quality=85)
                ext = ".jpg"
            else:
                Image.fromarray(t).save(b, "PNG", compress_level=3)
                ext = ".png"
            with open(stem + ext + ".part", "wb") as f:
                f.write(b.getvalue())
            os.replace(stem + ext + ".part", stem + ext)
    return True


def _blank_png(data):
    """True when a transparent PNG from the service holds no imagery at all."""
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        if im.mode in ("RGBA", "LA") or "transparency" in im.info:
            a = im.convert("RGBA").getchannel("A")
            return a.getextrema()[1] == 0
    except Exception:
        pass
    return False


# ------------------------------------------------------------------ places
_DEC = re.compile(r"(-?\d{1,2}\.\d+)\s*°?\s*([NS])?\s*[,;\s]\s*(-?\d{1,3}\.\d+)\s*°?\s*([EW])?", re.I)
_DMS = re.compile(r"(\d{1,2})\s*[°d\s]\s*(\d{1,2})\s*['′m\s]\s*(\d{1,2}(?:\.\d+)?)\s*(?:\"|″|''|s)?\s*([NS])"
                  r"[\s,;]*(\d{1,3})\s*[°d\s]\s*(\d{1,2})\s*['′m\s]\s*(\d{1,2}(?:\.\d+)?)\s*(?:\"|″|''|s)?\s*([EW])", re.I)
_URL_LL = re.compile(r"(?:@|[?&](?:q|ll|query|center|destination)=)(-?\d{1,2}\.\d+),\s*(-?\d{1,3}\.\d+)")


def parse_coords(text):
    """(lat, lon) from typed or pasted coordinates, or None."""
    t = text.strip()
    m = _URL_LL.search(urllib.parse.unquote(t))
    if m:
        return _check(float(m.group(1)), float(m.group(2)))
    m = _DMS.search(t)
    if m:
        g = m.groups()
        lat = int(g[0]) + int(g[1]) / 60.0 + float(g[2]) / 3600.0
        lon = int(g[4]) + int(g[5]) / 60.0 + float(g[6]) / 3600.0
        return _check(-lat if g[3].upper() == "S" else lat, -lon if g[7].upper() == "W" else lon)
    m = _DEC.fullmatch(t) or (_DEC.search(t) if re.fullmatch(r"[\d\s.,;°NSEWnsew+-]+", t) else None)
    if m:
        lat, lon = float(m.group(1)), float(m.group(3))
        if m.group(2) and m.group(2).upper() == "S":
            lat = -abs(lat)
        if m.group(4) and m.group(4).upper() == "W":
            lon = -abs(lon)
        return _check(lat, lon)
    return None


def _check(lat, lon):
    if -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0):
        return lat, lon
    return None


def locate(text):
    """{"lat", "lon", "label"} for coordinates, a map link or a US street address."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter an address, coordinates or a map link.")
    ll = parse_coords(text)
    if ll:
        return {"lat": ll[0], "lon": ll[1], "label": "%.6f, %.6f" % ll}
    if re.match(r"https?://", text):
        # a shortened map link: follow it and read the coordinates from where it lands
        try:
            with _get(text, timeout=10) as r:
                final = r.geturl()
                body = r.read(200000).decode("utf-8", "replace")
            ll = parse_coords(final) or parse_coords(_first_url_ll(body))
        except Exception:
            ll = None
        if ll:
            return {"lat": ll[0], "lon": ll[1], "label": "%.6f, %.6f" % ll}
        raise ValueError("Couldn't read a location from that link. Paste the coordinates instead "
                         "(in Google Maps, right-click the field and click the numbers at the top).")
    # street addresses: US Census Bureau geocoder (exact house locations)
    reached = False
    if re.match(r"^\s*\d+\s+\S", text):
        q = urllib.parse.urlencode({"address": text, "benchmark": "Public_AR_Current", "format": "json"})
        try:
            with _get(CENSUS + "?" + q, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8"))
            reached = True
            hits = (d.get("result") or {}).get("addressMatches") or []
            if hits:
                c = hits[0]["coordinates"]
                return {"lat": float(c["y"]), "lon": float(c["x"]), "label": hits[0].get("matchedAddress") or text}
        except Exception:
            pass
    # towns, counties, roads and places: OpenStreetMap's search (one request per search, as its
    # usage policy asks)
    q = urllib.parse.urlencode({"q": text, "format": "jsonv2", "limit": 1, "countrycodes": "us,ca,mx"})
    try:
        with _get(NOMINATIM + "?" + q, timeout=15) as r:
            hits = json.loads(r.read().decode("utf-8"))
        reached = True
    except Exception:
        hits = []
    if hits:
        h = hits[0]
        return {"lat": float(h["lat"]), "lon": float(h["lon"]),
                "label": (h.get("display_name") or text) + " (search © OpenStreetMap)"}
    if not reached:
        raise ValueError("Couldn't reach the address search. Check the internet connection, or paste "
                         "coordinates instead.")
    raise ValueError("No match for that. Try a town and state (Waco, TX), a full street address, or paste "
                     "coordinates (for example 31.5493, -97.1467) or a Google Maps link.")


def _first_url_ll(body):
    m = _URL_LL.search(body)
    return m.group(0) if m else ""


def lonlat_center(rings_ll):
    """Middle of the bounding box of lon/lat rings."""
    xs = [p[0] for r in rings_ll for p in r]
    ys = [p[1] for r in rings_ll for p in r]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def origin_for(lon, lat):
    from .geo import Geo, utm_epsg
    epsg = utm_epsg(lon, lat)
    e, n = Geo(epsg).to_utm(lon, lat)
    return {"epsg": epsg, "e": int(math.floor(e)), "n": int(math.floor(n))}
