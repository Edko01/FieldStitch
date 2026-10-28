"""Processing pipeline: photos -> features -> matches -> adjustment -> surface -> orthomosaic."""
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
import time
import traceback

import numpy as np

from . import __version__
from .exif import camera_key
from .geo import Geo, utm_epsg, rot_from_ypr
from .sysinfo import worker_count, total_ram_bytes

PRESETS = {
    "fast": dict(reduce=4, nfeat=3000, neighbors=10, per_cell=2, res_factor=2.0, ba_iter=(10, 15, 10)),
    "balanced": dict(reduce=4, nfeat=5000, neighbors=12, per_cell=3, res_factor=1.0, ba_iter=(15, 25, 15)),
    "accurate": dict(reduce=2, nfeat=7000, neighbors=16, per_cell=4, res_factor=1.0, ba_iter=(20, 35, 20)),
}

DEFAULT_SETTINGS = {
    "preset": "balanced",
    "resolution_cm": None,      # None = automatic from preset
    "surface": "fields",        # fields | detailed
    "exposure_balance": True,
    "compression": "jpeg",      # jpeg | deflate
    "jpeg_quality": 90,
    "max_oblique_deg": 30.0,    # skip photos tilted more than this from nadir
    "use_dsm_export": True,
    "trim_edges": True,         # drop ragged edges seen by only one photo
    "workers": None,
}

STAGES = [("features", 0.22), ("matching", 0.26), ("alignment", 0.17), ("surface", 0.05), ("ortho", 0.30)]


def image_heading(R):
    """Grid azimuth (deg) of the image-up direction for a camera-from-world rotation."""
    up = R.T @ np.array([0.0, -1.0, 0.0])
    return math.degrees(math.atan2(up[0], up[1]))


def rotate_heading(Rs, deg, mask):
    """Turn cameras about the vertical axis by deg (clockwise seen from above)."""
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    Wt = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]]).T
    out = Rs.copy()
    out[mask] = Rs[mask] @ Wt
    return out


class Cancelled(Exception):
    pass


class Timer:
    """Wall-clock time per stage plus summed worker CPU time per sub-step."""

    def __init__(self):
        self.t = time.time()
        self.stages = []
        self.workers = {}

    def lap(self, name):
        now = time.time()
        self.stages.append((name, now - self.t))
        self.t = now

    def add(self, group, d):
        g = self.workers.setdefault(group, {})
        for k, v in d.items():
            g[k] = g.get(k, 0) + v

    def summary(self, log):
        total = sum(v for _, v in self.stages) or 1.0
        log("Timing summary (wall clock):")
        for name, v in self.stages:
            log("  %-26s %7.1f s  %4.1f%%" % (name, v, 100 * v / total))
        for group, d in self.workers.items():
            if group == "render" and "total" in d:
                d = dict(d)
                d["blend & other"] = max(0.0, d["total"] - d.get("read", 0) - d.get("decode", 0) - d.get("warp", 0))
            parts = ", ".join("%s %.1f s" % (k, v) for k, v in d.items() if k not in ("bytes", "loads"))
            extra = ""
            if d.get("bytes") and d.get("read"):
                extra = " | read %.0f MB at %.0f MB/s per worker" % (d["bytes"] / 1e6, d["bytes"] / 1e6 / max(d["read"], 1e-6))
            log("  %s worker time: %s%s" % (group, parts, extra))

    def as_dict(self):
        return {"stages": [[n, round(v, 2)] for n, v in self.stages],
                "workers": {g: {k: round(v, 2) for k, v in d.items()} for g, d in self.workers.items()}}


class Progress:
    def __init__(self, project_dir):
        self.path = os.path.join(project_dir, "progress.json")
        self.log_path = os.path.join(project_dir, "log.txt")
        self.t0 = time.time()
        self.stage = ""
        self.last_write = 0
        self.state = {"status": "running", "stage": "", "stage_frac": 0.0, "overall": 0.0, "message": "",
                      "started": self.t0, "elapsed": 0.0, "eta": None}

    def log(self, msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def set(self, stage, frac, message=None, force=False):
        offset = 0.0
        weight = 0.0
        for name, w in STAGES:
            if name == stage:
                weight = w
                break
            offset += w
        overall = min(1.0, offset + weight * max(0.0, min(1.0, frac)))
        self.state.update(stage=stage, stage_frac=frac, overall=overall, elapsed=time.time() - self.t0)
        if message is not None:
            self.state["message"] = message
        if overall > 0.03:
            el = time.time() - self.t0
            self.state["eta"] = el / overall * (1 - overall)
        now = time.time()
        if force or now - self.last_write > 0.5:
            self.last_write = now
            self._write()

    def finish(self, status, message=""):
        self.state.update(status=status, message=message, elapsed=time.time() - self.t0)
        if status == "done":
            self.state["overall"] = 1.0
        self._write()

    def _write(self):
        # progress is informational: a failed write must never stop a job
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state, f)
            replace_retry(tmp, self.path, tries=8)
        except Exception:
            pass


def load_project(project_dir):
    with open(os.path.join(project_dir, "project.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def replace_retry(tmp, path, tries=40):
    """os.replace that waits out Windows sharing violations (another program reading the file,
    antivirus, indexer) instead of failing."""
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(min(0.25, 0.02 * (k + 1)))


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    replace_retry(tmp, path)


def safe_name(s):
    keep = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in s).strip()
    return keep.replace(" ", "_") or "project"


# ------------------------------------------------------------------------------ run
def run(project_dir):
    prog = Progress(project_dir)
    try:
        open(prog.log_path, "w").close()
        _run(project_dir, prog)
        prog.finish("done", "Finished")
    except Cancelled:
        prog.finish("cancelled", "Cancelled")
    except Exception as e:
        prog.log("ERROR: %s\n%s" % (e, traceback.format_exc()))
        prog.finish("error", str(e))
        raise


def _pool(n, init, arg):
    ctx = mp.get_context("spawn")
    return ctx.Pool(n, initializer=init, initargs=(arg,))


def _run(project_dir, prog):
    log = prog.log
    proj = load_project(project_dir)
    settings = dict(DEFAULT_SETTINGS)
    settings.update(proj.get("settings") or {})
    preset = PRESETS[settings["preset"]]
    work = os.path.join(project_dir, "work")
    feat_dir = os.path.join(work, "features")
    thumb_dir = os.path.join(work, "thumbs")
    out_dir = os.path.join(project_dir, "outputs")
    for d in (work, feat_dir, thumb_dir, out_dir):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    T = Timer()
    log("FieldStitch %s - project '%s'" % (__version__, proj.get("name")))
    log("Preset: %s | RAM %.1f GB | CPUs %d" % (settings["preset"], total_ram_bytes() / 1e9, os.cpu_count() or 1))
    from .sysinfo import describe
    sysd = describe(project_dir, proj.get("folders") or [])
    log("System: %s | %s | %s" % (sysd.get("cpu"), sysd.get("os"),
        "; ".join("%s (%s GB)" % (g.get("name"), g.get("memory_gb")) for g in sysd.get("gpus", [])) or "GPU: n/a"))
    if sysd.get("photo_drives"):
        log("Photo drive type: %s | project drive: %s, %s GB free" % (", ".join(sysd["photo_drives"]),
            sysd.get("project_drive"), sysd.get("project_free_gb")))

    # ---------------------------------------------------------------- select photos
    photos = proj["photos"]
    sel = proj.get("selection")
    sel_set = set(sel) if sel else None
    use = []
    skipped = {"no_gps": 0, "oblique": 0, "deselected": 0, "other_camera": 0, "missing": 0}
    groups = {}
    for pi, m in enumerate(photos):
        if sel_set is not None and pi not in sel_set:
            skipped["deselected"] += 1
            continue
        if m.get("lat") is None or m.get("lon") is None:
            skipped["no_gps"] += 1
            continue
        if m.get("pitch") is not None and abs(m["pitch"] + 90.0) > settings["max_oblique_deg"]:
            skipped["oblique"] += 1
            continue
        if not os.path.exists(m["path"]):
            skipped["missing"] += 1
            continue
        groups.setdefault(camera_key(m), []).append(pi)
    if not groups:
        raise RuntimeError("No usable photos (photos need GPS positions and must be roughly nadir).")
    biggest = max(len(v) for v in groups.values())
    gkeys = [k for k, v in groups.items() if len(v) >= max(5, 0.2 * biggest)]
    for k, v in groups.items():
        if k not in gkeys:
            skipped["other_camera"] += len(v)
    for k in gkeys:
        use.extend(groups[k])
    use.sort(key=lambda pi: (photos[pi].get("time") or 0, photos[pi]["path"]))
    n = len(use)
    if n < 3:
        raise RuntimeError("Need at least 3 usable photos, found %d." % n)
    log("Using %d photos (%s)" % (n, ", ".join("%s: %d" % (k, v) for k, v in skipped.items() if v)
                                   or "none skipped"))
    meta = [photos[pi] for pi in use]
    cam_group = np.array([gkeys.index(camera_key(m)) for m in meta], np.int64)
    G = len(gkeys)
    W = np.array([m["w"] for m in meta], np.int64)
    H = np.array([m["h"] for m in meta], np.int64)

    # ---------------------------------------------------------------- coordinates
    lon0 = float(np.median([m["lon"] for m in meta]))
    lat0 = float(np.median([m["lat"] for m in meta]))
    epsg = utm_epsg(lon0, lat0)
    geo = Geo(epsg)
    E, N = geo.to_utm(np.array([m["lon"] for m in meta]), np.array([m["lat"] for m in meta]))
    alt = np.array([m["alt"] if m.get("alt") is not None else 0.0 for m in meta], float)
    origin = np.array([math.floor(np.median(E) / 100) * 100, math.floor(np.median(N) / 100) * 100, 0.0])
    Cgps = np.c_[np.asarray(E) - origin[0], np.asarray(N) - origin[1], alt]
    conv = geo.convergence(lon0, lat0)
    rel = np.array([m["rel_alt"] if m.get("rel_alt") is not None else np.nan for m in meta])
    has_rel = np.isfinite(rel)
    rtk = np.array([(m.get("rtk_flag") or 0) >= 50 for m in meta])
    log("UTM EPSG:%d, %d RTK-fixed photos, grid convergence %.2f deg" % (epsg, rtk.sum(), conv))

    # intrinsics per group (median of members)
    intr0 = np.zeros((G, 8))
    intr_sig = np.zeros((G, 8))
    for g in range(G):
        mem = [meta[i] for i in range(n) if cam_group[i] == g]
        f = float(np.median([m["f_px"] for m in mem]))
        cx = float(np.median([m["cx"] for m in mem]))
        cy = float(np.median([m["cy"] for m in mem]))
        dist = np.median(np.array([m["dist"] for m in mem]), axis=0)
        intr0[g] = [f, cx, cy, dist[0], dist[1], dist[2], dist[3], dist[4]]
        src = mem[0].get("f_src", "")
        Wg, Hg = mem[0]["w"], mem[0]["h"]
        fs = {"dji-calibrated": 0.005, "dji-dewarp": 0.005, "exif": 0.03}.get(src, 0.10)
        ks = 0.02 if src.startswith("dji") and np.any(dist != 0) else 0.3
        intr_sig[g] = [fs * f, 0.01 * Wg, 0.01 * Hg, ks, ks, 0.005, 0.005, ks]
        log("Camera %d: %s | f=%.1fpx (%s) | %d photos" % (g, gkeys[g], f, src or "?", len(mem)))

    def Kmat(i):
        f, cx, cy = intr0[cam_group[i], :3]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

    def dist_of(i):
        k = intr0[cam_group[i]]
        return np.array([k[3], k[4], k[5], k[6], k[7]])

    has_gimbal = np.array([m.get("yaw") is not None and m.get("pitch") is not None for m in meta])
    R_gimbal = np.array([rot_from_ypr((m["yaw"] or 0) + conv, m["pitch"] or -90, m.get("roll") or 0)
                         if has_gimbal[i] else np.eye(3) for i, m in enumerate(meta)])
    f_med = float(np.median(intr0[cam_group, 0]))
    agl_guess = np.where(has_rel, rel, np.nan)
    agl_default = float(np.nanmedian(agl_guess)) if has_rel.any() else 100.0
    zg = Cgps[:, 2] - np.where(has_rel, rel, agl_default)

    # ---------------------------------------------------------------- features
    reduce = preset["reduce"]
    nfeat = preset["nfeat"]
    nworkers = settings.get("workers") or worker_count(700 * 1024 ** 2)
    feat_key = "r%d_n%d" % (reduce, nfeat)
    key_path = os.path.join(feat_dir, "key.json")
    old = {}
    if os.path.exists(key_path):
        try:
            old = json.load(open(key_path))
        except Exception:
            old = {}
    tasks = []
    for i, m in enumerate(meta):
        out = os.path.join(feat_dir, "%d.npz" % i)
        if list(old.get(str(i), []))[:2] == [m["path"], feat_key] and os.path.exists(out):
            continue
        thumb = os.path.join(thumb_dir, "%d.jpg" % i)
        if os.path.exists(thumb) and old.get(str(i), [None])[0] != m["path"]:
            os.remove(thumb)
        tasks.append((i, m["path"], out, thumb, reduce, nfeat))
    nfeats = np.zeros(n, np.int64)
    T.lap("setup")
    prog.set("features", 0, "Detecting features in %d photos" % n, force=True)
    log("Features: %d to compute (%d cached), %d workers" % (len(tasks), n - len(tasks), nworkers))
    from .features import extract
    newkey = {str(i): [m["path"], feat_key] + list(old.get(str(i), []))[2:3] for i, m in enumerate(meta)}
    if tasks:
        done = 0
        with _pool(nworkers, _noop_init, None) as pool:
            for idx, cnt, err, tt in pool.imap_unordered(extract, tasks, chunksize=1):
                done += 1
                T.add("features", tt)
                newkey[str(idx)] = [meta[idx]["path"], feat_key, int(cnt)]
                if err:
                    log("  feature error on %s: %s" % (meta[idx]["name"], err))
                prog.set("features", done / len(tasks), "Features %d / %d" % (done, len(tasks)))
    # feature counts: from the cache index when known, otherwise read once and remember
    missing = [i for i in range(n) if len(newkey[str(i)]) < 3]
    for k, i in enumerate(missing):
        try:
            with np.load(os.path.join(feat_dir, "%d.npz" % i)) as d:
                newkey[str(i)] = [meta[i]["path"], feat_key, int(len(d["xy"]))]
        except Exception:
            newkey[str(i)] = [meta[i]["path"], feat_key, 0]
        if k % 50 == 0:
            prog.set("features", k / max(1, len(missing)), "Checking saved features %d / %d" % (k, len(missing)))
    for i in range(n):
        nfeats[i] = newkey[str(i)][2]
    save_json(key_path, newkey)
    log("Features: median %d per photo" % int(np.median(nfeats)))
    T.lap("features")

    # ---------------------------------------------------------------- pairs
    from scipy.spatial import cKDTree
    xy = Cgps[:, :2]
    tree = cKDTree(xy)
    k = min(n - 1, preset["neighbors"])
    dists, nbrs = tree.query(xy, k=k + 1)
    agl_i = np.where(has_rel, rel, agl_default)
    half_short = agl_i * (np.minimum(W, H) / 2.0) / f_med
    half_long = agl_i * (np.maximum(W, H) / 2.0) / f_med
    pairs = set()
    for i in range(n):
        for d, j in zip(dists[i][1:], nbrs[i][1:]):
            if j >= n:
                continue
            lim = 2 * half_long[i] * 0.95 if has_rel.any() else np.inf
            if d < lim:
                pairs.add((min(i, j), max(i, j)))
    # temporal neighbours
    for i in range(n - 1):
        for off in (1, 2):
            j = i + off
            if j < n and np.hypot(*(xy[i] - xy[j])) < 2 * half_long[i]:
                pairs.add((i, j))
    pairs = sorted(pairs)
    log("Matching: %d candidate pairs" % len(pairs))

    ctx = {"feat_dir": feat_dir, "W": W, "H": H, "K": [Kmat(i) for i in range(n)],
           "dist": [dist_of(i) for i in range(n)], "R": R_gimbal, "C": Cgps, "zg": zg,
           "has_rot": has_gimbal.copy(), "prior_tol": float(os.environ.get("FIELDSTITCH_PRIOR_TOL", "0.18")), "ransac_thr": max(1.0, 0.6 * reduce),
           "min_inliers": 20, "ratio": 0.8, "checks": 48, "use_prior": True,
           "guided": True, "n_coarse": 1500, "guided_radius": 0.03, "guided_k": 24}
    ctx_file = os.path.join(work, "match_ctx.pkl")

    from .matching import init_worker, match_pair
    # quick sanity check of gimbal yaw on consecutive pairs (no prior filter)
    use_gimbal = has_gimbal.copy()
    yaw_offset = 0.0
    if has_gimbal.sum() >= 3:
        ctx["use_prior"] = False
        with open(ctx_file, "wb") as f:
            pickle.dump(ctx, f)
        cons = [p for p in pairs if p[1] - p[0] == 1]
        test_pairs = cons[:: max(1, len(cons) // 40)][:40]
        init_worker(ctx_file)
        sims = []
        for p in test_pairs:
            i, j, q, t, sim, nin = match_pair(p)
            if q is not None and sim and sim[0] != "error":
                sims.append((i, j, sim, nin))
        from .recon import estimate_yaw_from_pairs, angdiff
        cxs = intr0[cam_group, 1]
        cys = intr0[cam_group, 2]
        yaw_est = estimate_yaw_from_pairs(sims, Cgps, cxs, cys, n)
        # effective heading of the image "up" direction from the full gimbal rotation (yaw, pitch AND roll;
        # looking straight down, "yaw X" and "yaw X+180 with roll 180" describe the same camera)
        gy = np.array([image_heading(R_gimbal[i]) for i in range(n)])
        both = np.isfinite(yaw_est) & has_gimbal
        if both.sum() >= 3:
            d = angdiff(yaw_est[both], gy[both])
            rad = np.radians(d)
            mean = math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))
            spread = float(np.median(np.abs(angdiff(d, mean))))
            log("Gimbal heading check: offset %.1f deg, spread %.1f deg over %d photos" % (mean, spread, both.sum()))
            if spread > 10:
                log("WARNING: gimbal headings disagree with image motion - ignoring gimbal angles")
                use_gimbal[:] = False
            elif abs(mean) > 5:
                snap = round(mean / 90.0) * 90.0
                yaw_offset = snap if abs(mean - snap) < 10 else mean
                log("Gimbal headings are consistently %.0f deg off from the photos - correcting" % yaw_offset)
                R_gimbal = rotate_heading(R_gimbal, yaw_offset, has_gimbal)
                ctx["R"] = R_gimbal
    ctx["has_rot"] = use_gimbal
    ctx["use_prior"] = True
    with open(ctx_file, "wb") as f:
        pickle.dump(ctx, f)

    T.lap("pair selection")
    prog.set("matching", 0, "Matching %d image pairs" % len(pairs), force=True)
    match_res = []
    sims = []
    done = 0
    errors = 0
    with _pool(nworkers, init_worker, ctx_file) as pool:
        for i, j, q, t, sim, nin in pool.imap(match_pair, pairs, chunksize=8):
            done += 1
            if isinstance(sim, tuple) and len(sim) == 2 and sim[0] == "error":
                errors += 1
                sim = None
            if q is not None:
                match_res.append((i, j, q, t))
                if sim:
                    sims.append((i, j, sim, nin))
            if done % 20 == 0 or done == len(pairs):
                prog.set("matching", done / len(pairs), "Matched %d / %d pairs" % (done, len(pairs)))
    log("Matching: %d of %d pairs verified%s" % (len(match_res), len(pairs),
                                                ", %d errors" % errors if errors else ""))
    if use_gimbal.any() and len(pairs) > 20 and len(match_res) < 0.6 * len(pairs):
        log("WARNING: only %d of %d pairs matched using the drone's orientation data - matching again without it"
            % (len(match_res), len(pairs)))
        use_gimbal[:] = False
        ctx["has_rot"] = use_gimbal
        ctx["use_prior"] = False
        with open(ctx_file, "wb") as f:
            pickle.dump(ctx, f)
        match_res, sims, done, errors = [], [], 0, 0
        prog.set("matching", 0, "Matching again without orientation data", force=True)
        with _pool(nworkers, init_worker, ctx_file) as pool:
            for i, j, q, t, sim, nin in pool.imap(match_pair, pairs, chunksize=8):
                done += 1
                if isinstance(sim, tuple) and len(sim) == 2 and sim[0] == "error":
                    sim = None
                if q is not None:
                    match_res.append((i, j, q, t))
                    if sim:
                        sims.append((i, j, sim, nin))
                if done % 20 == 0 or done == len(pairs):
                    prog.set("matching", done / len(pairs), "Matched %d / %d pairs" % (done, len(pairs)))
        log("Matching (no orientation data): %d of %d pairs verified" % (len(match_res), len(pairs)))
    T.lap("matching")
    if len(match_res) < max(2, n // 4):
        raise RuntimeError("Too few overlapping photo pairs could be matched (%d). Check overlap / photo set."
                           % len(match_res))

    # ---------------------------------------------------------------- alignment
    from .recon import (build_tracks, select_tracks, estimate_yaw_from_pairs, fill_nan_by_time,
                        Reconstruction, undistort_normalized, triangulate)
    prog.set("alignment", 0.0, "Building tracks", force=True)
    tid, timg, tfeat, ntr = build_tracks(match_res, nfeats, n)
    log("Tracks: %d (%d observations)" % (ntr, len(tid)))
    from .matching import load_features
    from .matching import load_xy
    uv_all = np.zeros((len(tid), 2))
    o_img = np.argsort(timg, kind="stable")
    bounds_i = np.searchsorted(timg[o_img], np.arange(n + 1))
    for i in range(n):
        idx = o_img[bounds_i[i]:bounds_i[i + 1]]
        if len(idx):
            uv_all[idx] = load_xy(feat_dir, i, W[i], H[i])[tfeat[idx]]
    del o_img

    # initial rotations
    R0 = R_gimbal.copy()
    rot_sigma = 5.0
    if not use_gimbal.all():
        cxs = intr0[cam_group, 1]
        cys = intr0[cam_group, 2]
        yaw_est = estimate_yaw_from_pairs(sims, Cgps, cxs, cys, n)
        order = np.argsort([m.get("time") or 0 for m in meta], kind="stable")
        yaw_est = fill_nan_by_time(yaw_est, order)
        for i in range(n):
            if not use_gimbal[i]:
                R0[i] = rot_from_ypr(yaw_est[i], -90.0, 0.0)
        rot_sigma = 5.0 if use_gimbal.any() else 25.0
        log("Estimated headings from image motion for %d photos" % (~use_gimbal).sum())

    sig_h = np.where(rtk, 0.03, 2.0)
    sig_v = np.where(rtk, 0.05, 4.0)
    for i, m in enumerate(meta):
        st = m.get("rtk_std") or [None, None, None]
        if rtk[i] and st[0] is not None and st[1] is not None:
            sig_h[i] = max(0.02, 1.5 * max(st[0] or 0, st[1] or 0))
        if rtk[i] and st[2] is not None:
            sig_v[i] = max(0.03, 1.5 * (st[2] or 0))
    Csig = np.c_[sig_h, sig_h, sig_v]
    sigma_px = max(1.0, 0.5 * reduce)
    rec = Reconstruction(R0.copy(), Cgps.copy(), intr0.copy(), cam_group, W, H, Cgps, Csig,
                         R0.copy(), rot_sigma, intr0.copy(), intr_sig, sigma_px, log)

    sel_mask = select_tracks(tid, timg, uv_all, W, H, ntr, per_cell=preset["per_cell"])
    s_tid, s_img, s_uv = tid[sel_mask], timg[sel_mask], uv_all[sel_mask]
    from .recon import _compact
    s_tid, _ = _compact(s_tid, np.zeros((ntr, 3)))
    ntr_s = int(s_tid.max()) + 1 if len(s_tid) else 0
    log("Adjustment set: %d tracks, %d observations" % (ntr_s, len(s_tid)))
    it1, it2, it3 = preset["ba_iter"]
    rounds = [
        dict(loss=8.0, optimize_intrinsics=False, max_iter=it1, reject=10.0, retriangulate=True),
        dict(loss=3.0, optimize_intrinsics=True, max_iter=it2, reject=4.0, retriangulate=True),
        dict(loss=2.0, optimize_intrinsics=True, max_iter=it3, reject=None),
    ]

    def ba_prog(fr, msg):
        prog.set("alignment", 0.1 + 0.75 * fr, msg, force=True)

    T.lap("tracks")
    X, b_tid, b_img, b_uv, err = rec.adjust(s_tid, s_img, s_uv, ntr_s, rounds, progress=ba_prog)
    T.lap("bundle adjustment")
    obs_per_img = np.bincount(b_img, minlength=n)
    aligned = obs_per_img >= 12
    rms = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")
    log("Aligned %d / %d photos, reprojection RMS %.2f px, median %.2f px" %
        (aligned.sum(), n, rms, float(np.median(err)) if len(err) else float("nan")))
    for g in range(G):
        log("Camera %d refined: f=%.1f cx=%.1f cy=%.1f k1=%.4f k2=%.4f p1=%.5f p2=%.5f k3=%.4f" %
            ((g,) + tuple(rec.intr[g])))
    dC = rec.C - Cgps
    gps_res = np.sqrt(np.mean(dC[aligned] ** 2, axis=0)) if aligned.any() else np.zeros(3)
    log("Camera position adjustment RMS: E %.3f m, N %.3f m, U %.3f m" % tuple(gps_res))

    # dense points from all tracks
    prog.set("alignment", 0.9, "Triangulating surface points", force=True)
    keep_obs = aligned[timg]
    d_tid, d_img, d_uv = tid[keep_obs], timg[keep_obs], uv_all[keep_obs]
    d_tid, _ = _compact(d_tid, np.zeros((ntr, 3)))
    nd = int(d_tid.max()) + 1 if len(d_tid) else 0
    Xd, ang, ok = rec.triangulate_tracks(d_tid, d_img, d_uv, nd)
    e = rec.reproj_errors(Xd, d_tid, d_img, d_uv)
    bad = np.zeros(nd, bool)
    np.logical_or.at(bad, d_tid, e > 3 * sigma_px)
    cnt = np.bincount(d_tid, minlength=nd)
    good = ok & ~bad & (ang > 1.0) & (cnt >= 2)
    pts = Xd[good]
    log("Surface points: %d" % len(pts))
    if len(pts) < 50:
        pts = X
    del uv_all
    T.lap("surface points")

    # ---------------------------------------------------------------- surface
    from .surface import build_dsm
    prog.set("surface", 0.1, "Building surface model", force=True)
    ia = np.where(aligned)[0]
    # robust ground height under each camera from nearby points
    ptree = cKDTree(pts[:, :2])
    agl = np.zeros(n)
    zmed = float(np.median(pts[:, 2]))
    for i in ia:
        idx = ptree.query_ball_point(rec.C[i, :2], r=max(10.0, 0.3 * half_short[i]))
        zz = np.median(pts[idx, 2]) if len(idx) >= 5 else zmed
        agl[i] = rec.C[i, 2] - zz
    agl_med = float(np.median(agl[ia]))
    gsd_native = float(np.median(agl[ia] / rec.intr[cam_group[ia], 0]))
    if settings.get("resolution_cm"):
        gsd = float(settings["resolution_cm"]) / 100.0
    else:
        gsd = gsd_native * preset["res_factor"]
    gsd = max(gsd, gsd_native * 0.5)
    gsd = round(gsd, 4)
    log("Flight height above ground ~%.1f m, native GSD %.2f cm, output GSD %.2f cm" %
        (agl_med, gsd_native * 100, gsd * 100))
    if settings["surface"] == "detailed":
        cell = max(0.5, 0.008 * agl_med, 10 * gsd)
        smooth = 1.0
    else:
        cell = max(1.5, 0.02 * agl_med, 20 * gsd)
        smooth = 2.0
    foot = 2.2 * float(np.max(half_long[ia]))
    cxy = rec.C[ia, :2]
    bounds = (cxy[:, 0].min() - foot, cxy[:, 1].min() - foot, cxy[:, 0].max() + foot, cxy[:, 1].max() + foot)
    dsm = build_dsm(pts, bounds, cell, smooth, log)

    # footprints
    from .ortho import footprint
    cams = []
    allfp = []
    for i in ia:
        k = rec.intr[cam_group[i]]
        fp = footprint(rec.R[i], rec.C[i], k, W[i], H[i], dsm)
        allfp.append(fp)
        cams.append({"path": meta[i]["path"], "name": meta[i]["name"], "R": rec.R[i], "C": rec.C[i], "k": k,
                     "W": int(W[i]), "H": int(H[i]), "agl": float(max(agl[i], 1.0)),
                     "bbox": (fp[:, 0].min(), fp[:, 1].min(), fp[:, 0].max(), fp[:, 1].max()),
                     "gain": (1.0, 1.0, 1.0), "idx": int(i)})
    allfp = np.concatenate(allfp)
    ox0 = math.floor((allfp[:, 0].min()) / gsd) * gsd
    oy1 = math.ceil((allfp[:, 1].max()) / gsd) * gsd
    ox1 = math.ceil((allfp[:, 0].max()) / gsd) * gsd
    oy0 = math.floor((allfp[:, 1].min()) / gsd) * gsd
    out_w = int(round((ox1 - ox0) / gsd))
    out_h = int(round((oy1 - oy0) / gsd))
    log("Orthomosaic size: %d x %d px (%.1f Mpx)" % (out_w, out_h, out_w * out_h / 1e6))

    T.lap("surface model")
    # exposure balance
    if settings.get("exposure_balance", True):
        prog.set("surface", 0.6, "Balancing exposure", force=True)
        import cv2
        from .features import read_file
        thumbs = []
        for c in cams:
            p = os.path.join(thumb_dir, "%d.jpg" % c["idx"])
            try:
                t = cv2.imdecode(read_file(p), cv2.IMREAD_COLOR)[..., ::-1]
                t = cv2.blur(t, (5, 5))
                thumbs.append(np.ascontiguousarray(t))
            except Exception:
                thumbs.append(None)
        from .ortho import compute_gains
        spacing = max(2.0, float(np.median([2 * hs for hs in half_short[ia]])) / 14.0)
        gains = compute_gains(cams, thumbs, dsm, (ox0, oy0, ox1, oy1), spacing, log)
        for c, gn in zip(cams, gains):
            c["gain"] = tuple(float(x) for x in gn)
        del thumbs

    # save reconstruction
    recon_out = {"epsg": epsg, "origin": origin.tolist(), "meta_idx": use, "R": rec.R, "C": rec.C,
                 "intr": rec.intr, "cam_group": cam_group, "aligned": aligned, "gsd": gsd,
                 "gsd_native": gsd_native, "dsm": dsm}
    with open(os.path.join(work, "recon.pkl"), "wb") as f:
        pickle.dump(recon_out, f)
    prog.set("surface", 1.0, "Surface ready", force=True)
    T.lap("exposure balance")

    # ---------------------------------------------------------------- ortho
    name = safe_name(proj.get("name", "project"))
    ortho_path = os.path.join(out_dir, "%s_orthomosaic.tif" % name)
    write_orthomosaic(ortho_path, cams, dsm, origin, epsg, ox0, oy1, gsd, out_w, out_h, settings, prog, log, work, T)

    seam = None
    try:
        from .qa import seam_check
        prog.set("ortho", 1.0, "Checking seams", force=True)
        seam = seam_check(cams, dsm, gsd, n_pairs=30, log=log)
    except Exception as e:
        log("Seam check skipped: %s" % e)
    T.lap("seam check")

    if settings.get("use_dsm_export", True):
        try:
            write_dsm(os.path.join(out_dir, "%s_surface.tif" % name), dsm, origin, epsg)
        except Exception as e:
            log("DSM export failed: %s" % e)
    T.lap("surface export")

    # ---------------------------------------------------------------- report
    lon_c, lat_c = geo.to_ll(origin[0] + (ox0 + ox1) / 2, origin[1] + (oy0 + oy1) / 2)
    report = {
        "version": __version__,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing_seconds": round(time.time() - t_start, 1),
        "preset": settings["preset"],
        "photos_total": len(photos),
        "photos_used": n,
        "photos_aligned": int(aligned.sum()),
        "skipped": skipped,
        "rtk_photos": int(rtk.sum()),
        "gimbal_yaw_offset_deg": round(float(yaw_offset), 1),
        "gimbal_used": bool(use_gimbal.any()),
        "reprojection_rms_px": round(rms, 3),
        "seam_check": seam,
        "sparse_points": int(len(X)),
        "surface_points": int(len(pts)),
        "gsd_native_cm": round(gsd_native * 100, 2),
        "gsd_cm": round(gsd * 100, 2),
        "flight_agl_m": round(agl_med, 1),
        "camera_position_rms_m": [round(float(x), 3) for x in gps_res],
        "epsg": epsg,
        "center_lonlat": [lon_c, lat_c],
        "ortho": os.path.basename(ortho_path),
        "ortho_size_px": [out_w, out_h],
        "area_ha": None,
        "cameras": [gkeys[g] for g in range(G)],
        "unaligned_photos": [meta[i]["name"] for i in range(n) if not aligned[i]][:200],
    }
    try:
        import rasterio
        with rasterio.open(ortho_path) as ds:
            ov = ds.overviews(1)
            fac = ov[-1] if ov else 1
            msk = ds.read_masks(1, out_shape=(max(1, out_h // fac), max(1, out_w // fac)))
            report["area_ha"] = round(float((msk > 0).sum()) * (gsd * fac) ** 2 / 1e4, 2)
    except Exception:
        pass
    report["cam_positions"] = [[float(rec.C[i, 0] + origin[0]), float(rec.C[i, 1] + origin[1])] for i in ia]
    T.lap("report")
    T.summary(log)
    report["timing"] = T.as_dict()
    report["system"] = sysd
    report["workers"] = nworkers
    save_json(os.path.join(out_dir, "report.json"), report)
    log("Done in %.1f min" % ((time.time() - t_start) / 60))


def _noop_init(_):
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


# ------------------------------------------------------------------------ geotiff output
def ortho_profile(width, height, epsg, transform, settings):
    from rasterio.crs import CRS
    prof = dict(driver="GTiff", width=width, height=height, dtype="uint8", crs=CRS.from_epsg(epsg),
                transform=transform, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    if settings.get("compression", "jpeg") == "jpeg":
        prof.update(count=3, compress="JPEG", photometric="YCBCR", jpeg_quality=int(settings.get("jpeg_quality", 90)))
    else:
        prof.update(count=4, compress="DEFLATE", predictor=2, ZLEVEL=6, photometric="RGB", ALPHA="YES")
    return prof


def write_orthomosaic(path, cams, dsm, origin, epsg, ox0, oy1, gsd, out_w, out_h, settings, prog, log, work, T=None):
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from rasterio.enums import Resampling, ColorInterp
    from .ortho import init_render, render_group

    # decode reduction for speed when output is coarser than the photos
    gsd_native = float(np.median([c["agl"] / c["k"][0] for c in cams]))
    ratio = gsd / gsd_native
    reduce = 1
    for r in (2, 4, 8):
        if ratio >= r * 0.95:
            reduce = r
    tile = 2048
    tiles = []
    for ty in range(0, out_h, tile):
        for tx in range(0, out_w, tile):
            tiles.append((tx, ty, min(tile, out_w - tx), min(tile, out_h - ty)))
    # group 2x2 tiles for image cache locality, snake order
    groups = []
    ntx = int(math.ceil(out_w / tile))
    nty = int(math.ceil(out_h / tile))
    for gy in range(0, nty, 2):
        xs = list(range(0, ntx, 2))
        if (gy // 2) % 2:
            xs = xs[::-1]
        for gx in xs:
            grp = [tiles[yy * ntx + xx] for yy in (gy, gy + 1) for xx in (gx, gx + 1) if yy < nty and xx < ntx]
            groups.append(grp)
    img_bytes = float(np.median([c["W"] * c["H"] * 3 for c in cams])) / (reduce * reduce)
    per_worker = 900 * 1024 ** 2 + 6 * img_bytes
    nworkers = settings.get("workers") or worker_count(per_worker)
    ram = total_ram_bytes()
    cache_bytes = int(max(6 * img_bytes, min(0.45 * ram / max(nworkers, 1), 40 * img_bytes)))
    ctx = {"cams": cams, "dsm": dsm, "x0": ox0, "y1": oy1, "gsd": gsd, "reduce": reduce, "coarse_step": 16,
           "tau": max(0.01, 2.5 / float(np.median([c["agl"] for c in cams]))), "edge_fade": 0.03,
           "cache_bytes": cache_bytes,
           "min_views": 2 if settings.get("trim_edges", True) else 1}
    ctx_file = os.path.join(work, "render_ctx.pkl")
    with open(ctx_file, "wb") as f:
        pickle.dump(ctx, f)
    log("Rendering %d tiles with %d workers (photo decode 1/%d)" % (len(tiles), nworkers, reduce))
    transform = Affine(gsd, 0, ox0 + origin[0], 0, -gsd, oy1 + origin[1])
    prof = ortho_profile(out_w, out_h, epsg, transform, settings)
    tmp = path + ".part"
    jpeg = prof["compress"] == "JPEG"
    prog.set("ortho", 0, "Rendering orthomosaic", force=True)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(tmp, "w", **prof) as dst:
            if not jpeg:
                dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
            done = 0
            ctxm = mp.get_context("spawn")
            with ctxm.Pool(nworkers, initializer=init_render, initargs=(ctx_file,)) as pool:
                t_write = 0.0
                for res, tt in pool.imap_unordered(render_group, groups):
                    if T:
                        T.add("render", tt)
                    tw0 = time.time()
                    for (tx, ty, tw, th), rgb, alpha in res:
                        win = Window(tx, ty, tw, th)
                        if jpeg:
                            dst.write(rgb.transpose(2, 0, 1), window=win)
                            dst.write_mask(alpha, window=win)
                        else:
                            dst.write(np.concatenate([rgb, alpha[..., None]], 2).transpose(2, 0, 1), window=win)
                        done += 1
                        prog.set("ortho", 0.85 * done / len(tiles), "Rendered %d / %d tiles" % (done, len(tiles)))
                    t_write += time.time() - tw0
            if T:
                T.lap("render tiles")
                T.add("geotiff", {"write": t_write})
            prog.set("ortho", 0.87, "Building overviews", force=True)
            facs = []
            f = 2
            while max(out_w, out_h) / f >= 256:
                facs.append(f)
                f *= 2
            if facs:
                dst.build_overviews(facs, Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
    if T:
        T.lap("overviews")
    if os.path.exists(path):
        os.remove(path)
    replace_retry(tmp, path)
    # preview image
    try:
        import cv2
        with rasterio.open(path) as ds:
            s = max(1, int(math.ceil(max(out_w, out_h) / 1600)))
            ph, pw = max(1, out_h // s), max(1, out_w // s)
            rgb = ds.read([1, 2, 3], out_shape=(3, ph, pw), resampling=Resampling.average)
            m = ds.read_masks(1, out_shape=(ph, pw))
            img = np.dstack([rgb[2], rgb[1], rgb[0], m])
            cv2.imencode(".png", img)[1].tofile(os.path.join(os.path.dirname(path), "preview.png"))
    except Exception as e:
        log("preview failed: %s" % e)
    prog.set("ortho", 1.0, "Orthomosaic written", force=True)
    log("Orthomosaic: %s" % path)
    if T:
        T.lap("preview")


def write_dsm(path, dsm, origin, epsg):
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    z = dsm["z"]
    tr = Affine(dsm["cell"], 0, dsm["x0"] - dsm["cell"] / 2 + origin[0], 0, -dsm["cell"],
                dsm["y1"] + dsm["cell"] / 2 + origin[1])
    with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
                       crs=CRS.from_epsg(epsg), transform=tr, compress="DEFLATE", predictor=3, tiled=True,
                       nodata=-9999) as dst:
        dst.write(z.astype(np.float32), 1)
