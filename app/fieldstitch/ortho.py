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
def compute_gains(cams, thumbs, dsm, bounds, spacing, log=None):
    """Per-image RGB gains that equalise overlapping images (log-domain least squares)."""
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
        ok = (z > 0) & (u > 0.1 * W) & (u < 0.9 * W) & (v > 0.1 * H) & (v < 0.9 * H)
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
    n = len(cams)
    gains = np.ones((n, 3))
    if not samp_pt:
        return gains
    sp_pt = np.concatenate(samp_pt)
    sp_img = np.concatenate(samp_img)
    sp_val = np.concatenate(samp_val)
    o = np.argsort(sp_pt, kind="stable")
    sp_pt, sp_img, sp_val = sp_pt[o], sp_img[o], sp_val[o]
    starts = np.flatnonzero(np.r_[True, sp_pt[1:] != sp_pt[:-1]])
    ends = np.r_[starts[1:], len(sp_pt)]
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
        return gains
    lam = 0.05 * math.sqrt(len(ia) / max(n, 1))
    for ch in range(3):
        rhs = sp_val[ib, ch] - sp_val[ia, ch]
        keep = np.ones(len(ia), bool)
        g = np.zeros(n)
        for it in range(2):
            ka, kb, r = ia[keep], ib[keep], rhs[keep]
            m = len(ka)
            rows = np.r_[np.arange(m), np.arange(m), m + np.arange(n)]
            cols = np.r_[sp_img[ka], sp_img[kb], np.arange(n)]
            vals = np.r_[np.ones(m), -np.ones(m), np.full(n, lam)]
            A = sp.csr_matrix((vals, (rows, cols)), shape=(m + n, n))
            b = np.r_[r, np.zeros(n)]
            # normal equations: small (photos x photos), sparse, well conditioned thanks to lam
            g = spla.spsolve((A.T @ A).tocsc(), A.T @ b)
            res = g[sp_img[ia]] - g[sp_img[ib]] - rhs
            keep = np.abs(res) < 0.12
        gains[:, ch] = np.exp(g)
    # keep the overall brightness unchanged
    gains /= np.exp(np.mean(np.log(gains), axis=0, keepdims=True))
    gains = np.clip(gains, 0.6, 1.6)
    if log:
        log("Exposure balance: %d overlap samples, gain range %.2f-%.2f" % (len(ia), gains.min(), gains.max()))
    return gains


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


def _get_image(ci):
    if ci in _IMG_CACHE:
        _IMG_CACHE.move_to_end(ci)
        return _IMG_CACHE[ci]
    cam = _RCTX["cams"][ci]
    import time
    from .features import read_file, decode
    t0 = time.perf_counter()
    buf = read_file(cam["path"])
    t1 = time.perf_counter()
    img = decode(buf, _RCTX["reduce"], color=True)
    _T["read"] += t1 - t0
    _T["decode"] += time.perf_counter() - t1
    _T["bytes"] += int(buf.nbytes)
    _T["loads"] += 1
    _IMG_CACHE[ci] = img
    _IMG_BYTES[0] += img.nbytes
    while _IMG_BYTES[0] > _RCTX["cache_bytes"] and len(_IMG_CACHE) > 2:
        _, old = _IMG_CACHE.popitem(last=False)
        _IMG_BYTES[0] -= old.nbytes
    return img


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
        h, w = r1 - r0, c1 - c0
        wa = Wt[a].reshape(ncy, ncx)
        w_f = upsample(wa.astype(np.float32), r0, c0, h, w)
        if w_f.max() <= 0:
            continue
        uv = np.dstack([U[a].reshape(ncy, ncx), V[a].reshape(ncy, ncx)]).astype(np.float32)
        if r != 1:
            uv = (uv + 0.5) / r - 0.5
        uv_f = upsample(uv, r0, c0, h, w)
        img = _get_image(ci)
        tw0 = time.perf_counter()
        m1, m2 = cv2.convertMaps(uv_f, None, cv2.CV_16SC2)
        samp = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        _T["warp"] += time.perf_counter() - tw0
        g = c["cams"][ci]["gain"]
        w3 = cv2.merge([w_f * g[2], w_f * g[1], w_f * g[0]])  # BGR gains folded into the weights
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
    rgb = cv2.cvtColor(cv2.convertScaleAbs(out), cv2.COLOR_BGR2RGB)
    alpha = cv2.multiply(valid, 255)
    rgb = cv2.bitwise_and(rgb, rgb, mask=valid)
    return cut_tiles(region, tiles, rgb, alpha)


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
            "step": step, "items": items}


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


def render_worker_main(ctx_file, task_q, res_q):
    """Render worker: pull runs of tile groups, stream each group's tiles back."""
    import traceback
    try:
        init_render(ctx_file)
        while True:
            seg = task_q.get()
            if seg is None:
                break
            for group in seg:
                res, snap = render_group(group)
                res_q.put(("ok", res, snap))
    except Exception:
        res_q.put(("error", traceback.format_exc()))


def render_group(group):
    t0 = time.perf_counter()
    x0 = min(t[0] for t in group)
    y0 = min(t[1] for t in group)
    x1 = max(t[0] + t[2] for t in group)
    y1 = max(t[1] + t[3] for t in group)
    res = render_region((x0, y0, x1 - x0, y1 - y0), group)
    if os.environ.get("FIELDSTITCH_TEST_DIE"):
        os._exit(3)  # test hook: simulate a worker being killed
    _T["total"] += time.perf_counter() - t0
    snap = dict(_T)
    for k in _T:
        _T[k] = 0 if k in ("bytes", "loads") else 0.0
    return res, snap
