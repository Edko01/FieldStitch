"""Web-map tiles of the orthomosaic for the XAG One app (and any other XYZ tile viewer).

The XAG app loads a base map from https://raw.githubusercontent.com/<owner>/<map>/main/{z}/{x}/{y}.png,
so one map = one GitHub repo = one folder of tiles. This writes that folder:

* standard web-mercator XYZ tiles (EPSG:3857), 256 x 256, {z}/{x}/{y}.png, y counted from the top
  (not TMS);
* PNG for every tile; outside the orthomosaic is transparent (no black fill), so edge tiles keep
  their partial transparency;
* tiles with nothing on them are not written at all (fewer files and fewer downloads in the app).

The map view's own tiles (server._make_tile) are cut in the job's flat UTM frame, so they can't be
reused directly: here each tile's pixels are mapped from web mercator to the orthomosaic's UTM
grid. Reading follows the same approach: only the part of the GeoTIFF under the tile is read, at
about the tile's resolution (so the overviews are used for the lower zooms).
"""
import math
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

WM = 20037508.342789244          # half the width of the web-mercator world, metres
TILE = 256
PNG_LEVEL = 9                    # smallest files; the map view uses 1 for speed
GITHUB_SOFT = 1 << 30            # GitHub recommends repositories under 1 GB
GITHUB_PUSH = 2 << 30            # and refuses a single push over 2 GB
GITHUB_HARD = 5 << 30            # "strongly recommended" under 5 GB

_local = threading.local()


def repo_name(text):
    """A name GitHub accepts for a repository (letters, digits, '.', '-', '_')."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip()).strip("-.")
    return s[:100] or "map"


def tile_url(owner, name):
    return "https://raw.githubusercontent.com/%s/%s/main/{z}/{x}/{y}.png" % (owner or "OWNER", name)


def _tile_bounds(z, x, y):
    size = 2 * WM / (2 ** z)
    x0 = -WM + x * size
    y1 = WM - y * size
    return x0, y1 - size, x0 + size, y1


class _Ortho:
    """What every worker needs about the orthomosaic: its grid, a coarse footprint, transforms."""

    def __init__(self, path):
        import rasterio
        from pyproj import Transformer
        self.path = path
        with rasterio.open(path) as ds:
            self.crs = ds.crs
            self.b = ds.bounds
            self.res = ds.res[0]
            self.w, self.h = ds.width, ds.height
            # footprint: where the ortho has pixels, at most ~2000 px across (from the overviews)
            f = max(1, int(math.ceil(max(ds.width, ds.height) / 2000.0)))
            fw, fh = max(1, ds.width // f), max(1, ds.height // f)
            m = ds.read_masks(1, out_shape=(fh, fw))
        import cv2
        self.foot = cv2.dilate((m > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
        self.fsx = (self.b.right - self.b.left) / fw
        self.fsy = (self.b.top - self.b.bottom) / fh
        self.to_utm = Transformer.from_crs(3857, self.crs, always_xy=True)
        to_wm = Transformer.from_crs(self.crs, 3857, always_xy=True)
        # ortho bounds in web mercator (densified edges)
        t = np.linspace(0, 1, 41)
        ex = np.concatenate([self.b.left + t * (self.b.right - self.b.left), np.full(41, self.b.right),
                             self.b.right - t * (self.b.right - self.b.left), np.full(41, self.b.left)])
        ny = np.concatenate([np.full(41, self.b.bottom), self.b.bottom + t * (self.b.top - self.b.bottom),
                             np.full(41, self.b.top), self.b.top - t * (self.b.top - self.b.bottom)])
        wx, wy = to_wm.transform(ex, ny)
        self.wm = (float(np.min(wx)), float(np.min(wy)), float(np.max(wx)), float(np.max(wy)))

    def tiles(self, z):
        """Tiles of zoom z that touch the ortho's footprint."""
        size = 2 * WM / (2 ** z)
        x0, y0, x1, y1 = self.wm
        tx = range(int(math.floor((x0 + WM) / size)), int(math.floor((x1 + WM) / size)) + 1)
        ty = range(int(math.floor((WM - y1) / size)), int(math.floor((WM - y0) / size)) + 1)
        out = []
        for x in tx:
            for y in ty:
                if self._touches(z, x, y):
                    out.append((z, x, y))
        return out

    def _utm_box(self, z, x, y):
        a, b, c, d = _tile_bounds(z, x, y)
        px = np.array([a, c, c, a, (a + c) / 2, (a + c) / 2, a, c])
        py = np.array([b, b, d, d, b, d, (b + d) / 2, (b + d) / 2])
        ex, ny = self.to_utm.transform(px, py)
        return float(np.min(ex)), float(np.min(ny)), float(np.max(ex)), float(np.max(ny))

    def _touches(self, z, x, y):
        e0, n0, e1, n1 = self._utm_box(z, x, y)
        if e1 <= self.b.left or e0 >= self.b.right or n1 <= self.b.bottom or n0 >= self.b.top:
            return False
        c0 = max(0, int((e0 - self.b.left) / self.fsx))
        c1 = min(self.foot.shape[1], int(math.ceil((e1 - self.b.left) / self.fsx)))
        r0 = max(0, int((self.b.top - n1) / self.fsy))
        r1 = min(self.foot.shape[0], int(math.ceil((self.b.top - n0) / self.fsy)))
        return c1 > c0 and r1 > r0 and bool(self.foot[r0:r1, c0:c1].any())


def _dataset(path):
    import rasterio
    if getattr(_local, "path", None) != path:
        if getattr(_local, "ds", None) is not None:
            _local.ds.close()
        _local.ds = rasterio.open(path)
        _local.path = path
    return _local.ds


def _pixel_grid(o, z, x, y):
    """UTM easting/northing of every tile pixel centre (256 x 256), from a coarse 17 x 17 grid of
    exact transforms; web mercator to UTM is smooth, so the interpolation error is far below a
    millimetre."""
    a, b, c, d = _tile_bounds(z, x, y)
    s = (c - a) / TILE
    k = 17
    u = np.linspace(0.5, TILE - 0.5, k)
    gx, gy = np.meshgrid(a + u * s, d - u * s)
    ce, cn = o.to_utm.transform(gx.ravel(), gy.ravel())
    ce, cn = np.asarray(ce).reshape(k, k), np.asarray(cn).reshape(k, k)
    p = (np.arange(TILE) + 0.5 - 0.5) / ((TILE - 1) / (k - 1))
    i0 = np.clip(np.floor(p).astype(int), 0, k - 2)
    t = (p - i0)[None, :]
    def up(g):
        rows = g[:, i0] * (1 - t) + g[:, i0 + 1] * t                    # (k, 256): along x
        tt = (p - i0)[:, None]
        return rows[i0, :] * (1 - tt) + rows[i0 + 1, :] * tt           # (256, 256)
    return up(ce), up(cn)


def render(o, z, x, y):
    """PNG bytes of one tile, or None when nothing of the ortho is on it."""
    import cv2
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    ds = _dataset(o.path)
    E, N = _pixel_grid(o, z, x, y)
    # read the window at about the tile's own resolution (overviews for the lower zooms), with a
    # margin of two read pixels so interpolation at the tile's edge stays inside what was read
    tile_res = (float(E.max()) - float(E.min())) / TILE
    pad = 2 * max(o.res, tile_res)
    e0, e1 = max(float(E.min()) - pad, o.b.left), min(float(E.max()) + pad, o.b.right)
    n0, n1 = max(float(N.min()) - pad, o.b.bottom), min(float(N.max()) + pad, o.b.top)
    if e1 <= e0 or n1 <= n0:
        return None
    f = max(1.0, tile_res / o.res)
    col0, row0 = (e0 - o.b.left) / o.res, (o.b.top - n1) / o.res
    wcols, wrows = (e1 - e0) / o.res, (n1 - n0) / o.res
    ow, oh = max(1, int(math.ceil(wcols / f))), max(1, int(math.ceil(wrows / f)))
    win = Window(col0, row0, wcols, wrows)
    rs = Resampling.average if f > 1.5 else Resampling.bilinear
    rgb = ds.read([1, 2, 3], window=win, out_shape=(3, oh, ow), resampling=rs)
    m = ds.read_masks(1, window=win, out_shape=(oh, ow), resampling=Resampling.nearest)
    if not m.any():
        return None
    sx, sy = wcols * o.res / ow, wrows * o.res / oh
    mx = ((E - e0) / sx - 0.5).astype(np.float32)
    my = ((n1 - N) / sy - 0.5).astype(np.float32)
    bgr = cv2.merge([rgb[2], rgb[1], rgb[0]])
    img = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.remap(m.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # keep only pixels made entirely of ortho pixels: the outside never bleeds in as a dark fringe
    alpha = np.where(a >= 254.5, 255, 0).astype(np.uint8)
    if not alpha.any():
        return None
    params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL]
    if alpha.min() == 255:
        ok, enc = cv2.imencode(".png", img, params)                       # fully covered: RGB
    else:
        img[alpha == 0] = 0
        ok, enc = cv2.imencode(".png", cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2], alpha]), params)
    return enc.tobytes() if ok else None


def plan(path, zmin, zmax):
    """Tiles to try per zoom (those touching the ortho), without rendering anything."""
    o = _Ortho(path)
    return o, {z: o.tiles(z) for z in range(zmin, zmax + 1)}


def clear_zoom_folders(root):
    """Remove earlier tiles (the numbered zoom folders only): a git checkout, README and anything
    else in the folder stay."""
    if not os.path.isdir(root):
        return 0
    n = 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name.isdigit() and os.path.isdir(p):
            shutil.rmtree(p)
            n += 1
    return n


def export(path, root, zmin, zmax, progress=None, cancel=None, workers=None):
    """Write the tiles into root/{z}/{x}/{y}.png. Returns a summary dict."""
    o, todo = plan(path, zmin, zmax)
    jobs = [t for z in sorted(todo) for t in todo[z]]
    clear_zoom_folders(root)
    os.makedirs(root, exist_ok=True)
    done = [0]
    lock = threading.Lock()
    per_zoom = {z: [0, 0] for z in todo}            # tiles written, bytes
    made = set()

    def one(t):
        if cancel and cancel():
            return
        z, x, y = t
        data = render(o, z, x, y)
        if data:
            d = os.path.join(root, str(z), str(x))
            with lock:
                if d not in made:
                    os.makedirs(d, exist_ok=True)
                    made.add(d)
            with open(os.path.join(d, "%d.png" % y), "wb") as fh:
                fh.write(data)
        with lock:
            if data:
                per_zoom[z][0] += 1
                per_zoom[z][1] += len(data)
            done[0] += 1
            if progress:
                progress(done[0], len(jobs), z)

    n = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    with ThreadPoolExecutor(max_workers=n) as ex:
        for _ in ex.map(one, jobs):
            pass
    tiles = sum(v[0] for v in per_zoom.values())
    size = sum(v[1] for v in per_zoom.values())
    grid = sum(_grid_count(o, z) for z in todo)
    return {"tiles": tiles, "bytes": size, "tried": len(jobs), "skipped_empty": grid - tiles,
            "zooms": {str(z): {"tiles": v[0], "bytes": v[1]} for z, v in sorted(per_zoom.items())},
            "cancelled": bool(cancel and cancel())}


def _grid_count(o, z):
    """All tiles of the ortho's bounding box at zoom z (what a plain tiler would have written)."""
    size = 2 * WM / (2 ** z)
    x0, y0, x1, y1 = o.wm
    return ((int(math.floor((x1 + WM) / size)) - int(math.floor((x0 + WM) / size)) + 1) *
            (int(math.floor((WM - y0) / size)) - int(math.floor((WM - y1) / size)) + 1))


def github_notes(nbytes):
    """What to tell the user about putting this many bytes on GitHub."""
    notes = []
    if nbytes > GITHUB_HARD:
        notes.append("Over 5 GB: GitHub strongly recommends keeping a repository under 5 GB. Consider a lower "
                     "top zoom level, or splitting the map.")
    elif nbytes > GITHUB_SOFT:
        notes.append("Over 1 GB: GitHub recommends keeping repositories under 1 GB (and strongly under 5 GB).")
    if nbytes > GITHUB_PUSH:
        notes.append("GitHub refuses a single push over 2 GB, so push this map in several parts, for example "
                     "one or two zoom folders per commit and push.")
    return notes
