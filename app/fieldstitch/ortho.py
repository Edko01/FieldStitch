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
        m = (E >= bx0) & (E <= bx1) & (N >= by0) & (N <= by1)
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
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
            g = spla.lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
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
    with open(ctx_file, "rb") as f:
        _RCTX = pickle.load(f)
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
    """tile: (tx, ty, tw, th) in output pixels. Returns (tile, rgb uint8, alpha uint8)."""
    c = _RCTX
    tx, ty, tw, th = tile
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
    # tile bbox in ground coordinates
    e0, e1 = E[0] - gsd, E[-1] + gsd
    n1, n0 = N[0] + gsd, N[-1] - gsd
    cand = []
    for ci, cam in enumerate(c["cams"]):
        bx0, by0, bx1, by1 = cam["bbox"]
        if bx1 < e0 or bx0 > e1 or by1 < n0 or by0 > n1:
            continue
        cand.append(ci)
    rgb = np.zeros((th, tw, 3), np.uint8)
    alpha = np.zeros((th, tw), np.uint8)
    if not cand:
        return tile, rgb, alpha
    nc = len(cand)
    scores = np.full((nc, P.shape[0]), np.inf)
    U = np.zeros((nc, P.shape[0]), np.float32)
    V = np.zeros((nc, P.shape[0]), np.float32)
    FADE = np.zeros((nc, P.shape[0]))
    for a, ci in enumerate(cand):
        cam = c["cams"][ci]
        u, v, z = project_pts(cam["R"], cam["C"], cam["k"], P)
        W, H = cam["W"], cam["H"]
        margin = 1.0
        ok = (z > 0) & (u > margin) & (u < W - 1 - margin) & (v > margin) & (v < H - 1 - margin)
        edge = np.minimum.reduce([u, W - 1 - u, v, H - 1 - v]) / (c["edge_fade"] * max(W, H))
        hd = np.hypot(P[:, 0] - cam["C"][0], P[:, 1] - cam["C"][1]) / max(cam["agl"], 1.0)
        scores[a, ok] = hd[ok]
        U[a] = u
        V[a] = v
        FADE[a] = np.where(ok, _smoothstep(edge), 0.0)
    best = scores.min(0)
    nviews = np.isfinite(scores).sum(0)
    best = np.where(nviews >= c.get("min_views", 1), best, np.inf)
    covered = np.isfinite(best)
    if not covered.any():
        return tile, rgb, alpha
    with np.errstate(invalid="ignore"):
        Wt = np.where(np.isfinite(scores) & covered[None], np.exp(-(scores - best[None]) / c["tau"]), 0.0) * FADE
    Wt[Wt < 2e-3] = 0.0
    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw), np.float32)
    # fine pixel -> coarse grid coordinates
    fx = (np.arange(tw, dtype=np.float32) / step)
    fy = (np.arange(th, dtype=np.float32) / step)
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
        mxc, myc = np.meshgrid(fx[c0:c1], fy[r0:r1])
        Ua = U[a].reshape(ncy, ncx)
        Va = V[a].reshape(ncy, ncx)
        w_f = cv2.remap(wa.astype(np.float32), mxc, myc, cv2.INTER_LINEAR)
        if w_f.max() <= 0:
            continue
        u_f = cv2.remap(Ua, mxc, myc, cv2.INTER_LINEAR)
        v_f = cv2.remap(Va, mxc, myc, cv2.INTER_LINEAR)
        img = _get_image(ci)
        r = c["reduce"]
        if r != 1:
            u_f = (u_f + 0.5) / r - 0.5
            v_f = (v_f + 0.5) / r - 0.5
        tw0 = time.perf_counter()
        samp = cv2.remap(img, u_f, v_f, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        _T["warp"] += time.perf_counter() - tw0
        g = c["cams"][ci]["gain"]
        samp = samp.astype(np.float32) * np.array(g[::-1], np.float32)  # BGR order
        acc[r0:r1, c0:c1] += samp * w_f[..., None]
        wsum[r0:r1, c0:c1] += w_f
    # smooth coverage edge: interpolate the coarse coverage indicator and cut at 0.5
    cov = covered.reshape(ncy, ncx).astype(np.float32)
    FXg, FYg = np.meshgrid(fx, fy)
    cov_f = cv2.remap(cov, FXg, FYg, cv2.INTER_LINEAR)
    valid = (wsum > 1e-4) & (cov_f >= 0.5)
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    rgb = out[..., ::-1]  # to RGB
    rgb = np.where(valid[..., None], rgb, 0).astype(np.uint8)
    alpha = (valid * 255).astype(np.uint8)
    return tile, np.ascontiguousarray(rgb), alpha


def render_group(group):
    t0 = time.perf_counter()
    res = [render_tile(t) for t in group]
    _T["total"] += time.perf_counter() - t0
    snap = dict(_T)
    for k in _T:
        _T[k] = 0 if k in ("bytes", "loads") else 0.0
    return res, snap
