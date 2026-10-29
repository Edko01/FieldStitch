"""Orthorectification, seamline selection, blending and exposure balancing."""
import math
import os
import pickle
import time
from collections import OrderedDict

import cv2
import numpy as np

from .camera import undistort_normalized
from .features import load_image
from .surface import sample_dsm


# ------------------------------------------------------------------------- projection
def project_pts(R, C, k, P):
    """Project world points P (n,3) into one camera. Returns u, v, depth."""
    Xc = (P - C) @ R.T
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = Xc[:, 0] / zs
    y = Xc[:, 1] / zs
    f, cx, cy, k1, k2, p1, p2, k3 = k
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return f * xd + cx, f * yd + cy, z


def pixels_to_ground(R, C, k, uv, dsm, iters=4):
    """Intersect pixel rays with the DSM surface."""
    xy = undistort_normalized(uv, k)
    d = np.c_[xy, np.ones(len(xy))] @ R  # world dirs
    z0 = sample_dsm(dsm, np.array([C[0]]), np.array([C[1]]))[0]
    zg = np.full(len(d), z0)
    G = None
    for _ in range(iters):
        dz = np.where(d[:, 2] > -1e-3, -1e-3, d[:, 2])
        t = (zg - C[2]) / dz
        t = np.clip(t, 0, 20 * max(C[2] - z0, 10))
        G = C[None] + t[:, None] * d
        zg = sample_dsm(dsm, G[:, 0], G[:, 1])
    return G


def footprint(R, C, k, W, H, dsm, n=12):
    xs = np.linspace(0, W - 1, n)
    ys = np.linspace(0, H - 1, n)
    border = np.r_[np.c_[xs, np.zeros(n)], np.c_[np.full(n, W - 1), ys],
                   np.c_[xs[::-1], np.full(n, H - 1)], np.c_[np.zeros(n), ys[::-1]]]
    G = pixels_to_ground(R, C, k, border, dsm)
    return G[:, :2]


# ------------------------------------------------------------------------ exposure gains
def view_features(P, cam, u, v):
    """Viewing-geometry terms for each ground point P seen by cam at pixel (u, v): the horizontal
    offset from the camera relative to its height (world frame, so sun-angle effects keep their
    direction whichever way the drone flies), its squares and product, and the squared distance
    from the image centre (lens vignetting)."""
    agl = max(cam["agl"], 1.0)
    dx = (P[:, 0] - cam["C"][0]) / agl
    dy = (P[:, 1] - cam["C"][1]) / agl
    f, cx, cy = cam["k"][0], cam["k"][1], cam["k"][2]
    r2 = ((u - cx) / f) ** 2 + ((v - cy) / f) ** 2
    return np.stack([dx, dy, dx * dx, dy * dy, dx * dy, r2], 1)


N_VIEW = 6


def view_correction(view, F):
    """Per-channel brightness factors (n, 3) that remove the fitted viewing-angle effect."""
    if view is None:
        return np.ones((F.shape[0], 3))
    return np.clip(np.exp(F @ np.asarray(view).T), 0.7, 1.4)


def flight_segments(times, gap_s=150.0, min_photos=40):
    """Split photos into flights by gaps in capture time (battery swaps). Returns a segment index
    per photo; 0 for all when times are missing. Very short pieces join the previous flight."""
    n = len(times)
    t = np.array([np.nan if v is None else float(v) for v in times], np.float64)
    if n == 0 or np.isnan(t).mean() > 0.2:
        return np.zeros(n, np.int64), 1
    t = np.where(np.isnan(t), np.nanmedian(t), t)
    order = np.argsort(t, kind="stable")
    seg_sorted = np.zeros(n, np.int64)
    cur = 0
    count = 1
    for k in range(1, n):
        if t[order[k]] - t[order[k - 1]] > gap_s and count >= min_photos:
            cur += 1
            count = 0
        seg_sorted[k] = cur
        count += 1
    if cur > 0 and count < min_photos:      # a short last piece joins the previous flight
        seg_sorted[seg_sorted == cur] = cur - 1
        cur -= 1
    seg = np.empty(n, np.int64)
    seg[order] = seg_sorted
    return seg, cur + 1


def _smooth_over_site(g, centres, radius):
    """Gaussian-weighted average of per-photo values g around each photo (sigma = radius), on a
    coarse grid so it stays fast for thousands of photos."""
    if len(g) < 3 or radius <= 0:
        return np.full_like(g, float(np.mean(g)) if len(g) else 0.0)
    cell = radius / 4.0
    x0, y0 = centres.min(0) - 3 * radius
    nx = int((centres[:, 0].max() + 3 * radius - x0) / cell) + 1
    ny = int((centres[:, 1].max() + 3 * radius - y0) / cell) + 1
    ix = ((centres[:, 0] - x0) / cell).astype(int)
    iy = ((centres[:, 1] - y0) / cell).astype(int)
    num = np.zeros((ny, nx), np.float64)
    den = np.zeros((ny, nx), np.float64)
    np.add.at(num, (iy, ix), g)
    np.add.at(den, (iy, ix), 1.0)
    num = cv2.GaussianBlur(num, (0, 0), 4.0)
    den = cv2.GaussianBlur(den, (0, 0), 4.0)
    return num[iy, ix] / np.maximum(den[iy, ix], 1e-9)


def compute_gains(cams, thumbs, dsm, bounds, spacing, log=None, times=None):
    """Per-image RGB gains that equalise overlapping images (log-domain least squares), solved
    together with one shared viewing-angle model per channel (sun angle and vignetting). Without
    that model, the brightness difference between photos that see the ground from opposite sides
    adds up from one flight line to the next and shows as large colour casts across the site.
    Each flight (separated by a battery swap) gets its own viewing-angle model, because the sun
    moves between flights. Returns (gains (n, 3), view coefficients per photo (n, 3, N_VIEW))."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    xmin, ymin, xmax, ymax = bounds
    gx = np.arange(xmin + spacing / 2, xmax, spacing)
    gy = np.arange(ymin + spacing / 2, ymax, spacing)
    if len(gx) * len(gy) > 600000:
        s = math.sqrt(len(gx) * len(gy) / 600000.0)
        spacing *= s
        gx = np.arange(xmin + spacing / 2, xmax, spacing)
        gy = np.arange(ymin + spacing / 2, ymax, spacing)
    EE, NN = np.meshgrid(gx, gy)
    E = EE.ravel()
    N = NN.ravel()
    Z = sample_dsm(dsm, E, N)
    P = np.c_[E, N, Z]
    samp_pt = []
    samp_img = []
    samp_val = []
    samp_f = []
    for ci, cam in enumerate(cams):
        th = thumbs[ci]
        if th is None:
            continue
        bx0, by0, bx1, by1 = cam["bbox"]
        # regular grid: index ranges straight from the footprint box
        c0, c1 = np.searchsorted(gx, bx0), np.searchsorted(gx, bx1, side="right")
        r0, r1 = np.searchsorted(gy, by0), np.searchsorted(gy, by1, side="right")
        if c1 <= c0 or r1 <= r0:
            continue
        idx = (np.arange(r0, r1)[:, None] * len(gx) + np.arange(c0, c1)[None, :]).ravel()
        u, v, z = project_pts(cam["R"], cam["C"], cam["k"], P[idx])
        W, H = cam["W"], cam["H"]
        ok = (z > 0) & (u > 0.03 * W) & (u < 0.97 * W) & (v > 0.03 * H) & (v < 0.97 * H)
        if not ok.any():
            continue
        th_h, th_w = th.shape[:2]
        mx = ((u[ok] + 0.5) * th_w / W - 0.5).astype(np.float32).reshape(1, -1)
        my = ((v[ok] + 0.5) * th_h / H - 0.5).astype(np.float32).reshape(1, -1)
        # cv2.remap needs map width < 32767
        vals = []
        for s0 in range(0, mx.shape[1], 30000):
            vals.append(cv2.remap(th, mx[:, s0:s0 + 30000], my[:, s0:s0 + 30000], cv2.INTER_LINEAR)[0])
        val = np.concatenate(vals, 0).astype(np.float64)
        good = np.all((val > 8) & (val < 247), axis=1)
        samp_pt.append(idx[ok][good])
        samp_img.append(np.full(good.sum(), ci))
        samp_val.append(np.log(val[good]))
        samp_f.append(view_features(P[idx[ok][good]], cam, u[ok][good], v[ok][good]))
    n = len(cams)
    gains = np.ones((n, 3))
    seg, nseg = flight_segments(times if times is not None else [None] * n)
    view = np.zeros((nseg, 3, N_VIEW))
    if not samp_pt:
        return gains, view[seg], None
    sp_pt = np.concatenate(samp_pt)
    sp_img = np.concatenate(samp_img)
    sp_val = np.concatenate(samp_val)
    sp_f = np.concatenate(samp_f)
    o = np.argsort(sp_pt, kind="stable")
    sp_pt, sp_img, sp_val, sp_f = sp_pt[o], sp_img[o], sp_val[o], sp_f[o]
    # build pairs: each sample vs the next few samples of the same point
    ia = []
    ib = []
    for off in (1, 2, 3):
        a = np.arange(len(sp_pt) - off)
        b = a + off
        m = sp_pt[a] == sp_pt[b]
        ia.append(a[m])
        ib.append(b[m])
    ia = np.concatenate(ia)
    ib = np.concatenate(ib)
    if len(ia) < 10:
        return gains, view[seg], None
    per_img = 2.0 * len(ia) / max(n, 1)          # pair constraints touching an average photo
    lam = math.sqrt(0.01 * per_img)              # pull towards no change: damps slow drift only
    lam_v = math.sqrt(1e-3 * len(ia) / nseg)     # light ridge on each flight's view model
    nv = nseg * N_VIEW
    sa, sb = seg[sp_img[ia]], seg[sp_img[ib]]
    corrected = np.empty((len(sp_val), 3))
    centres = np.array([c["C"][:2] for c in cams], np.float64)
    foot = float(np.median([max(c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1]) for c in cams]))
    smooth_r = 2.0 * foot
    for ch in range(3):
        rhs = sp_val[ib, ch] - sp_val[ia, ch]
        keep = np.ones(len(ia), bool)
        for it in range(3):
            ka, kb, r = ia[keep], ib[keep], rhs[keep]
            m = len(ka)
            # unknowns: n photo log-gains, then N_VIEW view terms per flight
            ar = np.arange(m)
            va = n + sa[keep][:, None] * N_VIEW + np.arange(N_VIEW)[None]
            vb = n + sb[keep][:, None] * N_VIEW + np.arange(N_VIEW)[None]
            rows = np.r_[ar, ar, np.repeat(ar, N_VIEW), np.repeat(ar, N_VIEW), m + np.arange(n), m + n + np.arange(nv)]
            cols = np.r_[sp_img[ka], sp_img[kb], va.ravel(), vb.ravel(), np.arange(n), n + np.arange(nv)]
            vals = np.r_[np.ones(m), -np.ones(m), sp_f[ka].ravel(), -sp_f[kb].ravel(), np.full(n, lam), np.full(nv, lam_v)]
            A = sp.csr_matrix((vals, (rows, cols)), shape=(m + n + nv, n + nv))
            b = np.r_[r, np.zeros(n + nv)]
            x = spla.spsolve((A.T @ A).tocsc(), A.T @ b)
            g, vw = x[:n], x[n:].reshape(nseg, N_VIEW)
            corr = g[sp_img] + (sp_f * vw[seg[sp_img]]).sum(1)       # log correction per sample
            res = corr[ia] - corr[ib] - rhs
            keep = np.abs(res) < (0.12 if it else 0.25)
        # only even out differences between nearby photos: the slow part of the correction across
        # the site (wider than a few photo footprints) comes from model error, not from the
        # camera, and shows up as patches
        g = g - _smooth_over_site(g, centres, smooth_r)
        corr = g[sp_img] + (sp_f * vw[seg[sp_img]]).sum(1)
        # keep the mosaic as bright as the photos: the average correction over all samples is zero
        g -= corr.mean()
        corr -= corr.mean()
        gains[:, ch] = np.exp(g)
        view[:, ch] = vw
        corrected[:, ch] = np.exp(sp_val[:, ch] + corr)
    gains = np.clip(gains, 0.6, 1.6)
    if log:
        lg = np.log(gains)
        cast = lg - lg.mean(1, keepdims=True)
        log("Exposure balance: %d overlap samples, %d flight%s, gain range %.2f-%.2f, colour spread %.2f, "
            "sun/vignetting model (R, flight 1) %s"
            % (len(ia), nseg, "" if nseg == 1 else "s", gains.min(), gains.max(), float(np.ptp(cast)),
               " ".join("%+.3f" % x for x in view[0, 0])))
    lum = corrected @ np.array([0.299, 0.587, 0.114])
    stats = {"median_lum": float(np.median(lum)), "p5": float(np.percentile(lum, 5)), "p95": float(np.percentile(lum, 95))}
    return gains, view[seg], stats


# ----------------------------------------------------------------------------- rendering
_RCTX = None
_IMG_CACHE = OrderedDict()
_IMG_BYTES = [0]
_T = {"read": 0.0, "decode": 0.0, "warp": 0.0, "total": 0.0, "bytes": 0, "loads": 0}


def init_render(ctx_file):
    global _RCTX
    from .sysinfo import full_speed
    full_speed()
    with open(ctx_file, "rb") as f:
        _RCTX = pickle.load(f)
    _RCTX["bboxes"] = np.array([cam["bbox"] for cam in _RCTX["cams"]], np.float64).reshape(-1, 4)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _get_window(ci, x0, y0, x1, y1):
    """Pixels [x0, x1) x [y0, y1) (clipped) of photo ci at decode scale, as a view, and the window's
    top-left corner. Only the needed rows are decoded (full width); the strip is kept for the next
    block, which usually needs the same rows of the photo further along."""
    ent = _IMG_CACHE.get(ci)
    if ent is not None and ent[0] <= max(0, y0) and ent[1] >= y1 and ent[2] is not None:
        _IMG_CACHE.move_to_end(ci)
        by0, by1, band = ent
        y1 = min(y1, by1)
    else:
        if ent is not None:
            _IMG_CACHE.pop(ci)
            _IMG_BYTES[0] -= ent[2].nbytes
        cam = _RCTX["cams"][ci]
        from .features import read_file
        from .jpegcrop import decode_rows
        t0 = time.perf_counter()
        buf = read_file(cam["path"])
        t1 = time.perf_counter()
        band, by0 = decode_rows(buf, max(0, y0), y1, _RCTX["reduce"])
        by1 = by0 + band.shape[0]
        _T["read"] += t1 - t0
        _T["decode"] += time.perf_counter() - t1
        _T["bytes"] += int(buf.nbytes)
        _T["loads"] += 1
        _IMG_CACHE[ci] = (by0, by1, band)
        _IMG_BYTES[0] += band.nbytes
        while _IMG_BYTES[0] > _RCTX["cache_bytes"] and len(_IMG_CACHE) > 1:
            _, old = _IMG_CACHE.popitem(last=False)
            _IMG_BYTES[0] -= old[2].nbytes
        y1 = min(y1, by1)
    x0 = max(0, x0)
    y0 = max(by0, y0)
    x1 = min(band.shape[1], x1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0, 3), np.uint8), x0, y0
    return band[y0 - by0:y1 - by0, x0:x1], x0, y0


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def render_tile(tile):
    """Single tile (kept for tests). Returns (tile, rgb uint8, alpha uint8)."""
    return render_region(tile, [tile])[0]


def render_region(region, tiles):
    """Render a block of output pixels photo by photo (each photo decoded once for the block) and
    cut it into the given tiles. region/tiles: (tx, ty, tw, th) in output pixels.
    Returns [(tile, rgb uint8, alpha uint8, half_rgb, half_alpha)] where half_* is the tile at
    half resolution (for the overview file)."""
    plan = region_plan(region, tiles)
    if plan is None:
        return [_with_half(t, np.zeros((t[3], t[2], 3), np.uint8), np.zeros((t[3], t[2]), np.uint8)) for t in tiles]
    c = _RCTX
    tx, ty, tw, th = region
    step = plan["step"]
    ncy, ncx = plan["ncy"], plan["ncx"]
    cand, Wt, U, V, covered = plan["cand"], plan["Wt"], plan["U"], plan["V"], plan["covered"]
    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw), np.float32)
    r = c["reduce"]
    upsample = _upsampler(step)

    for a, ci, r0, r1, c0, c1 in plan["items"]:
        prep = prepare_item(plan, a, ci, r0, r1, c0, c1)
        if prep is None:
            continue
        coarse, gi0, gj0, crop = prep
        h, w = r1 - r0, c1 - c0
        tw0 = time.perf_counter()
        # weight x correction (B, G, R), weight, and photo position: bilinear from the grid
        M = np.array([[1.0 / step, 0, (c0 - gj0 * step) / step], [0, 1.0 / step, (r0 - gi0 * step) / step]], np.float64)
        wg_f = cv2.warpAffine(np.ascontiguousarray(coarse[..., :4]), M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        uv_f = cv2.warpAffine(np.ascontiguousarray(coarse[..., 4:]), M, (w, h),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        w_f = np.ascontiguousarray(wg_f[..., 3])
        w3 = np.ascontiguousarray(wg_f[..., :3])
        m1, m2 = cv2.convertMaps(uv_f, None, cv2.CV_16SC2)
        samp = cv2.remap(crop, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        _T["warp"] += time.perf_counter() - tw0
        av = acc[r0:r1, c0:c1]
        cv2.add(av, cv2.multiply(samp, w3, dtype=cv2.CV_32F), dst=av)
        wv = wsum[r0:r1, c0:c1]
        cv2.add(wv, w_f, dst=wv)
    # smooth coverage edge: interpolate the coarse coverage indicator and cut at 0.5
    cov = covered.reshape(ncy, ncx).astype(np.float32)
    cov_f = upsample(cov, 0, 0, th, tw)
    valid = cv2.bitwise_and((wsum > 1e-4).view(np.uint8), (cov_f >= 0.5).view(np.uint8))
    inv = cv2.divide(1.0, cv2.max(wsum, 1e-6))
    out = cv2.multiply(acc, cv2.merge([inv, inv, inv]))
    look = c.get("look")
    if look:
        out = apply_look(out, look)
    out = highlight_rolloff(out)
    rgb = cv2.cvtColor(cv2.convertScaleAbs(out), cv2.COLOR_BGR2RGB)
    alpha = cv2.multiply(valid, 255)
    rgb = cv2.bitwise_and(rgb, rgb, mask=valid)
    return cut_tiles(region, tiles, rgb, alpha)


def item_weights(plan, a, ci, win):
    """Coarse-grid float32 (gh, gw, 4) over the window win = (gi0, gi1, gj0, gj1): blend weight x
    colour correction for B, G, R, and the blend weight itself. The correction is the photo's
    exposure gain times the shared viewing-angle factor at each grid point."""
    c = _RCTX
    ncy, ncx = plan["ncy"], plan["ncx"]
    gi0, gi1, gj0, gj1 = win
    idx = (np.arange(gi0, gi1)[:, None] * ncx + np.arange(gj0, gj1)[None, :]).ravel()
    wa = plan["Wt"][a][idx]
    cam = c["cams"][ci]
    corr = np.tile(np.asarray(cam["gain"], np.float64), (wa.size, 1))
    view = cam.get("view")
    live = wa > 0
    if view is not None and live.any():
        il = idx[live]
        F = view_features(plan["P"][il], cam, plan["U"][a][il].astype(np.float64), plan["V"][a][il].astype(np.float64))
        corr[live] *= view_correction(view, F)
    out = np.empty((wa.size, 4), np.float32)
    out[:, 0] = wa * corr[:, 2]
    out[:, 1] = wa * corr[:, 1]
    out[:, 2] = wa * corr[:, 0]
    out[:, 3] = wa
    return out.reshape(gi1 - gi0, gj1 - gj0, 4)


def prepare_item(plan, a, ci, r0, r1, c0, c1):
    """Everything needed to draw one photo into one block, shared by the CPU and GPU renderers:
    (coarse, gi0, gj0, crop) where coarse is float32 (gh, gw, 6) on the coarse grid window
    starting at grid point (gi0, gj0): weight x correction for B, G, R, the weight, and the
    photo position u, v relative to crop, the decoded window of the photo (BGR uint8) that holds
    every pixel the block samples. None when the photo adds nothing."""
    step, ncy, ncx = plan["step"], plan["ncy"], plan["ncx"]
    gi0, gj0 = r0 // step, c0 // step
    gi1 = min(ncy, -(-(r1 - 1) // step) + 1)
    gj1 = min(ncx, -(-(c1 - 1) // step) + 1)
    win = (gi0, gi1, gj0, gj1)
    wg = item_weights(plan, a, ci, win)
    if wg[..., 3].max() <= 0:
        return None
    r = _RCTX["reduce"]
    U = plan["U"][a].reshape(ncy, ncx)[gi0:gi1, gj0:gj1]
    V = plan["V"][a].reshape(ncy, ncx)[gi0:gi1, gj0:gj1]
    if r != 1:
        U = (U + 0.5) / r - 0.5
        V = (V + 0.5) / r - 0.5
    # every fine pixel with weight > 0 interpolates u, v between grid points next to a weighted one
    m = cv2.dilate((wg[..., 3] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    us, vs = U[m], V[m]
    x0 = int(math.floor(us.min())) - 2
    x1 = int(math.ceil(us.max())) + 3
    y0 = int(math.floor(vs.min())) - 2
    y1 = int(math.ceil(vs.max())) + 3
    crop, ox, oy = _get_window(ci, x0, y0, x1, y1)
    if crop.size == 0:
        return None
    coarse = np.empty(wg.shape[:2] + (6,), np.float32)
    coarse[..., :4] = wg
    coarse[..., 4] = U - ox
    coarse[..., 5] = V - oy
    return coarse, gi0, gj0, crop


LOOKS = {"vivid": {"contrast": 1.15, "saturation": 1.4}, "natural": None}


def look_params(name, stats):
    """Global colour look for the whole mosaic (same everywhere, so it cannot create seams):
    contrast around the mosaic's median brightness and a saturation factor."""
    lk = LOOKS.get(name or "vivid")
    if not lk:
        return None
    mid = float(stats["median_lum"]) if stats else 118.0
    return {"mid": mid, "contrast": lk["contrast"], "saturation": lk["saturation"]}


def apply_look(out, look):
    """out: float32 BGR (h, w, 3), 0-255. Same arithmetic, in the same order, as the GPU kernel."""
    m = np.float32(look["mid"])
    k = np.float32(look["contrast"])
    sat = np.float32(look["saturation"])
    b, g, r = out[..., 0], out[..., 1], out[..., 2]
    y = (np.float32(0.114) * b + np.float32(0.587) * g) + np.float32(0.299) * r
    y2 = m + (y - m) * k
    # vibrance: pale colours get the full boost, strong colours less (no cartoon greens)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    rel = (mx - mn) / np.maximum(mx, np.float32(1.0))
    se = np.float32(1.0) + (sat - np.float32(1.0)) * (np.float32(1.0) - rel)
    res = np.empty_like(out)
    for c in range(3):
        res[..., c] = y2 + se * (out[..., c] - y)
    return np.clip(res, 0, 255)


KNEE = 220.0


def highlight_rolloff(out):
    """Soft shoulder above KNEE so near-white surfaces (roofs, equipment, concrete) keep their
    shading instead of clipping to flat white after exposure gains and contrast. The brightest
    channel sets one factor for all three, so colours keep their hue. Same arithmetic as the GPU
    kernel."""
    k = np.float32(KNEE)
    span = np.float32(255.0 - KNEE)
    mx = out.max(axis=2)
    hi = mx > k
    if not hi.any():
        return out
    out = out.copy()
    m = mx[hi]
    f = (k + span * np.tanh((m - k) / span)) / m
    out[hi] = out[hi] * f[:, None]
    return out


def _upsampler(step):
    def upsample(coarse, r0, c0, h, w):
        """Fine-grid bilinear interpolation of coarse samples taken every `step` pixels (exact affine)."""
        M = np.array([[1.0 / step, 0, c0 / step], [0, 1.0 / step, r0 / step]], np.float64)
        return cv2.warpAffine(coarse, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    return upsample


def cut_tiles(region, tiles, rgb, alpha):
    tx, ty = region[0], region[1]
    res = []
    for t in tiles:
        ox, oy = t[0] - tx, t[1] - ty
        res.append(_with_half(t, np.ascontiguousarray(rgb[oy:oy + t[3], ox:ox + t[2]]),
                              np.ascontiguousarray(alpha[oy:oy + t[3], ox:ox + t[2]])))
    return res


def region_plan(region, tiles):
    """Geometry for a block: which photos contribute where, their blend weights and the photo
    coordinates on a coarse grid. Shared by the CPU and GPU renderers. None if nothing to draw."""
    c = _RCTX
    tx, ty, tw, th = region
    gsd = c["gsd"]
    step = c["coarse_step"]
    ncx = int(math.ceil(tw / step)) + 1
    ncy = int(math.ceil(th / step)) + 1
    px = np.arange(ncx) * step
    py = np.arange(ncy) * step
    E = c["x0"] + (tx + px + 0.5) * gsd
    N = c["y1"] - (ty + py + 0.5) * gsd
    EE, NN = np.meshgrid(E, N)
    ZZ = sample_dsm(c["dsm"], EE, NN)
    P = np.c_[EE.ravel(), NN.ravel(), ZZ.ravel()]
    e0, e1 = E[0] - gsd, E[-1] + gsd
    n1, n0 = N[0] + gsd, N[-1] - gsd
    bb = c["bboxes"]
    cand = np.nonzero(~((bb[:, 2] < e0) | (bb[:, 0] > e1) | (bb[:, 3] < n0) | (bb[:, 1] > n1)))[0].tolist()
    if not cand:
        return None
    nc = len(cand)
    scores = np.full((nc, P.shape[0]), np.inf)
    U = np.zeros((nc, P.shape[0]), np.float32)
    V = np.zeros((nc, P.shape[0]), np.float32)
    FADE = np.zeros((nc, P.shape[0]))
    gidx = np.arange(P.shape[0]).reshape(ncy, ncx)
    for a, ci in enumerate(cand):
        cam = c["cams"][ci]
        # only grid points inside this photo's footprint box
        bx0, by0, bx1, by1 = bb[ci]
        j0 = max(0, int(np.searchsorted(E, bx0 - gsd * step)))
        j1 = min(ncx, int(np.searchsorted(E, bx1 + gsd * step)) + 1)
        i0 = max(0, int(np.searchsorted(-N, -(by1 + gsd * step))))
        i1 = min(ncy, int(np.searchsorted(-N, -(by0 - gsd * step))) + 1)
        if j1 <= j0 or i1 <= i0:
            continue
        sub = gidx[i0:i1, j0:j1].ravel()
        Ps = P[sub]
        u, v, z = project_pts(cam["R"], cam["C"], cam["k"], Ps)
        W, H = cam["W"], cam["H"]
        margin = 1.0
        ok = (z > 0) & (u > margin) & (u < W - 1 - margin) & (v > margin) & (v < H - 1 - margin)
        edge = np.minimum.reduce([u, W - 1 - u, v, H - 1 - v]) / (c["edge_fade"] * max(W, H))
        hd = np.hypot(Ps[:, 0] - cam["C"][0], Ps[:, 1] - cam["C"][1]) / max(cam["agl"], 1.0)
        sc = np.full(len(sub), np.inf)
        sc[ok] = hd[ok]
        scores[a, sub] = sc
        U[a, sub] = u
        V[a, sub] = v
        FADE[a, sub] = np.where(ok, _smoothstep(edge), 0.0)
    best = scores.min(0)
    nviews = np.isfinite(scores).sum(0)
    best = np.where(nviews >= c.get("min_views", 1), best, np.inf)
    covered = np.isfinite(best)
    if not covered.any():
        return None
    with np.errstate(invalid="ignore"):
        Wt = np.where(np.isfinite(scores) & covered[None], np.exp(-(scores - best[None]) / c["tau"]), 0.0) * FADE
    Wt[Wt < c.get("w_cut", 0.03)] = 0.0
    items = []
    for a, ci in enumerate(cand):
        wa = Wt[a].reshape(ncy, ncx)
        if wa.max() <= 0:
            continue
        rows = np.where(wa.max(1) > 0)[0]
        cols = np.where(wa.max(0) > 0)[0]
        r0 = max(0, (rows[0] - 1) * step)
        r1 = min(th, (rows[-1] + 1) * step + 1)
        c0 = max(0, (cols[0] - 1) * step)
        c1 = min(tw, (cols[-1] + 1) * step + 1)
        if r1 <= r0 or c1 <= c0:
            continue
        items.append((a, ci, int(r0), int(r1), int(c0), int(c1)))
    return {"cand": cand, "Wt": Wt, "U": U, "V": V, "covered": covered, "ncy": ncy, "ncx": ncx,
            "step": step, "items": items, "P": P}


def half_res(rgb, alpha):
    """2x downsample averaging only valid pixels (invalid pixels are already 0);
    valid where at least half of the 2x2 block is."""
    h, w = alpha.shape
    size = ((w + 1) // 2, (h + 1) // 2)
    s = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    out = cv2.multiply(s, cv2.merge([255.0 / np.maximum(a, 1.0)] * 3))
    valid = a >= 127.5
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out, (valid * 255).astype(np.uint8)


def _with_half(tile, rgb, alpha):
    h_rgb, h_a = half_res(rgb, alpha)
    return tile, rgb, alpha, h_rgb, h_a


def render_worker_main(ctx_file, task_q, res_q, link=None, wid=0):
    """Render worker: pull runs of tile groups, stream each group's tiles back. With a GPU link,
    blocks go to the card (this worker only plans them and decodes the photo windows) whenever
    the card has room; otherwise the worker renders them itself."""
    import traceback
    gl = None
    try:
        init_render(ctx_file)
        if link is not None:
            from .gpurender import GpuLink
            gl = GpuLink(wid, link)

        def finish(done):
            for group, res, snap in done:
                if res is None:      # the card failed: render here
                    res, snap2 = render_group(group)
                    for k, v in snap2.items():
                        snap[k] = snap.get(k, 0) + v
                res_q.put(("ok", ship(res, snap), snap))

        while True:
            seg = task_q.get()
            if seg is None:
                break
            for group in seg:
                if gl is not None and gl.usable():
                    r = _gpu_group(gl, group)
                    if r is not None:
                        res_q.put(("ok", ship(r[0], r[1]), r[1]))
                    finish(gl.collect(block=not gl.free))
                else:
                    res, snap = render_group(group)
                    res_q.put(("ok", ship(res, snap), snap))
                    if gl is not None and gl.out:
                        finish(gl.collect(block=False))
        while gl is not None and gl.out:
            finish(gl.collect(block=True))
    except Exception:
        res_q.put(("error", traceback.format_exc()))
    finally:
        if gl is not None:
            gl.close()


def ship(res, snap):
    """Before tiles go to the writer: JPEG-compress the GeoTIFF tiles here, in parallel with the
    other workers, when the main process uses its own TIFF writer."""
    q = _RCTX.get("tiff_jpeg_q") if _RCTX else None
    if q is None:
        return res
    from .tiffwriter import encode_tiles
    t0 = time.perf_counter()
    out = []
    for tile, rgb, alpha, h_rgb, h_alpha in res:
        out.append((tile, encode_tiles(tile, rgb, alpha, q), None, h_rgb, h_alpha))
    snap["encode"] = snap.get("encode", 0.0) + time.perf_counter() - t0
    return out


def _snap():
    snap = dict(_T)
    for k in _T:
        _T[k] = 0 if k in ("bytes", "loads") else 0.0
    return snap


def _gpu_group(gl, group):
    """Plan a block and decode its photo windows here; the card does the rest. Returns
    (tiles, snap) when there was nothing for the card (empty block), else None."""
    from .gpurender import prepare_block
    t0 = time.perf_counter()
    x0 = min(t[0] for t in group)
    y0 = min(t[1] for t in group)
    region = (x0, y0, max(t[0] + t[2] for t in group) - x0, max(t[1] + t[3] for t in group) - y0)
    plan, preps = prepare_block(region, group)
    if not preps:
        res = [_with_half(t, np.zeros((t[3], t[2], 3), np.uint8), np.zeros((t[3], t[2]), np.uint8)) for t in group]
        _T["total"] += time.perf_counter() - t0
        return res, _snap()
    snap = {}
    gl.submit(group, region, plan, preps, snap)
    _T["total"] += time.perf_counter() - t0
    snap.update(_snap())
    snap["gpu_blocks"] = 1
    return None


def render_group(group):
    t0 = time.perf_counter()
    x0 = min(t[0] for t in group)
    y0 = min(t[1] for t in group)
    x1 = max(t[0] + t[2] for t in group)
    y1 = max(t[1] + t[3] for t in group)
    res = render_region((x0, y0, x1 - x0, y1 - y0), group)
    if os.environ.get("PLANESITE_TEST_DIE"):
        os._exit(3)  # test hook: simulate a worker being killed
    _T["total"] += time.perf_counter() - t0
    snap = dict(_T)
    for k in _T:
        _T[k] = 0 if k in ("bytes", "loads") else 0.0
    return res, snap
