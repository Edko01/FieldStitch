"""Publishing a map's tile folder where the XAG One app can load it.

A destination is somewhere a folder of {z}/{x}/{y}.png tiles can be put so that it's served at a
fixed address: GitHub now (one repository per map, served from raw.githubusercontent.com),
Cloudflare R2 later. Each destination implements the same small interface (Destination below);
the window and the server only talk to that interface.

Shared here: reading the tile folder and splitting it into upload batches. Large maps go up in
several parts, smallest zoom levels first, then the biggest level in ranges of x columns, so no
single upload is too big (GitHub refuses a push over 2 GB).
"""
import json
import os
import sys

BATCH_LIMIT = 1_500_000_000          # bytes of tiles per upload (GitHub caps a push at 2 GB)


def use_vendor():
    """The pure-Python libraries bundled with the app (dulwich, urllib3) live in app/vendor."""
    v = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vendor")
    if os.path.isdir(v) and v not in sys.path:
        sys.path.insert(0, v)


class PublishError(Exception):
    pass


class Destination:
    """Interface every publish destination implements."""
    id = ""
    label = ""

    def status(self):
        """{"ready": bool, "user": str|None, ...}: whether publishing can start (signed in etc.)."""
        raise NotImplementedError

    def targets(self):
        """Places the user can publish into (for GitHub: accounts/organisations), plus how to add
        one: {"targets": [{"id", "label", "ok", "note"}], "add_url": str|None}."""
        raise NotImplementedError

    def check(self, target, name, folder):
        """Before publishing: {"exists": bool, "resumable": bool, "done": int, "total": int}."""
        raise NotImplementedError

    def publish(self, folder, target, name, mode, progress, cancel):
        """Upload the folder. mode: "new" (create), "into" (existing), "resume". progress(dict)
        is called as it goes; cancel() returns True to stop between parts. Returns a summary with
        "url" (the XYZ tile address)."""
        raise NotImplementedError

    def tile_url(self, target, name):
        raise NotImplementedError


_DEST = {}


def get(dest_id, cfg):
    """The destination object for an id ("github")."""
    if dest_id not in _DEST:
        if dest_id == "github":
            from .github import GitHub
            _DEST[dest_id] = GitHub(cfg)
        else:
            raise PublishError("Unknown publish destination: %s" % dest_id)
    d = _DEST[dest_id]
    d.cfg = cfg
    return d


# ------------------------------------------------------------------ the tile folder
def scan_tiles(folder):
    """[(z, x, y, path, size)] of every {z}/{x}/{y}.png in the folder, sorted by z, x, y."""
    out = []
    if not os.path.isdir(folder):
        return out
    for zn in os.listdir(folder):
        zp = os.path.join(folder, zn)
        if not (zn.isdigit() and os.path.isdir(zp)):
            continue
        for xn in os.listdir(zp):
            xp = os.path.join(zp, xn)
            if not (xn.isdigit() and os.path.isdir(xp)):
                continue
            with os.scandir(xp) as it:
                for e in it:
                    n = e.name
                    if n.endswith(".png") and n[:-4].isdigit() and e.is_file():
                        out.append((int(zn), int(xn), int(n[:-4]), e.path, e.stat().st_size))
    out.sort(key=lambda t: (t[0], t[1], t[2]))
    return out


def plan_batches(tiles, limit=None):
    """Split the tiles into uploads of at most `limit` bytes: whole zoom levels while they fit
    (smallest first), then a level that doesn't fit by ranges of x columns (a column bigger than
    the limit by itself is split by y). Returns a list of lists of tiles."""
    from itertools import groupby
    limit = limit or BATCH_LIMIT
    batches, cur, size = [], [], 0

    def flush():
        nonlocal cur, size
        if cur:
            batches.append(cur)
        cur, size = [], 0

    for z, zt in groupby(tiles, key=lambda t: t[0]):
        zt = list(zt)
        zs = sum(t[4] for t in zt)
        if size + zs <= limit:
            cur += zt
            size += zs
            continue
        flush()
        if zs <= limit:
            cur, size = list(zt), zs
            continue
        for x, col in groupby(zt, key=lambda t: t[1]):
            col = list(col)
            cs = sum(t[4] for t in col)
            if size + cs > limit:
                flush()
            if cs <= limit:
                cur += col
                size += cs
            else:
                for t in col:
                    if size + t[4] > limit:
                        flush()
                    cur.append(t)
                    size += t[4]
    flush()
    return batches


def describe_batch(batch):
    """'zoom 14-19' / 'zoom 20, x 283410-283460'."""
    zs = sorted({t[0] for t in batch})
    if len(zs) > 1:
        return "zoom %d-%d" % (zs[0], zs[-1])
    xs = [t[1] for t in batch]
    return "zoom %d, x %d-%d" % (zs[0], min(xs), max(xs))


def folder_summary(folder, limit=None):
    tiles = scan_tiles(folder)
    total = sum(t[4] for t in tiles)
    batches = plan_batches(tiles, limit)
    zs = sorted({t[0] for t in tiles})
    return {"tiles": len(tiles), "bytes": total, "pushes": len(batches),
            "zooms": [zs[0], zs[-1]] if zs else None}


# ------------------------------------------------------------------ resume records
def _state_dir():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "PlaneSite", "publish")
    os.makedirs(d, exist_ok=True)
    return d


def load_state(dest_id, target, name):
    p = os.path.join(_state_dir(), "%s_%s_%s.json" % (dest_id, target, name))
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(dest_id, target, name, state):
    p = os.path.join(_state_dir(), "%s_%s_%s.json" % (dest_id, target, name))
    if state is None:
        if os.path.exists(p):
            os.remove(p)
        return
    with open(p + ".part", "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(p + ".part", p)
