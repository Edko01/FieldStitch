"""Coarse surface model (2.5D DSM) from triangulated points."""
import math

import cv2
import numpy as np


def fill_nans(a):
    """Fill NaN cells with a pyramid push-pull interpolation."""
    m = np.isfinite(a)
    if m.all():
        return a.copy()
    if not m.any():
        return np.zeros_like(a)
    vals = np.where(m, a, 0.0).astype(np.float64)
    w = m.astype(np.float64)
    pyr = [(vals, w)]
    while min(vals.shape) > 1 and (w <= 1e-9).any():
        h, wd = vals.shape
        nh, nw = max(1, (h + 1) // 2), max(1, (wd + 1) // 2)
        vals = cv2.resize(vals, (nw, nh), interpolation=cv2.INTER_AREA)
        w = cv2.resize(w, (nw, nh), interpolation=cv2.INTER_AREA)
        pyr.append((vals, w))
    v, w = pyr[-1]
    mean = float(np.nanmean(a))
    filled = np.where(w > 1e-9, v / np.maximum(w, 1e-12), mean)
    for v, w in reversed(pyr[:-1]):
        up = cv2.resize(filled, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
        known = v / np.maximum(w, 1e-12)
        alpha = np.clip(w * 4.0, 0, 1)  # partially-known cells blend toward upsampled
        filled = np.where(w > 1e-9, alpha * known + (1 - alpha) * up, up)
    return np.where(m, a, filled)


def build_dsm(points, bounds, cell, smooth_cells=2.0, log=None):
    """points: (n,3) E,N,Z. bounds: (xmin, ymin, xmax, ymax). Returns dict grid."""
    xmin, ymin, xmax, ymax = bounds
    nx = int(math.ceil((xmax - xmin) / cell)) + 1
    ny = int(math.ceil((ymax - ymin) / cell)) + 1
    grid = np.full((ny, nx), np.nan)
    if len(points):
        ix = np.floor((points[:, 0] - xmin) / cell + 0.5).astype(np.int64)
        iy = np.floor((ymax - points[:, 1]) / cell + 0.5).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy, z = ix[ok], iy[ok], points[ok, 2]
        idx = iy * nx + ix
        o = np.lexsort((z, idx))
        idx_s, z_s = idx[o], z[o]
        starts = np.flatnonzero(np.r_[True, idx_s[1:] != idx_s[:-1]])
        ends = np.r_[starts[1:], len(idx_s)]
        med = z_s[(starts + ends - 1) // 2]
        grid.ravel()[idx_s[starts]] = med
        # remove spikes versus local median
        filled = fill_nans(grid).astype(np.float32)
        k = 5
        local = cv2.medianBlur(filled, k)
        dev = np.abs(grid - local)
        valid = np.isfinite(grid)
        if valid.sum() > 20:
            mad = np.nanmedian(dev[valid]) * 1.4826 + 1e-3
            spikes = valid & (dev > max(4 * mad, 1.5))
            grid[spikes] = np.nan
            if log:
                log("DSM: %d cells with data, %d spikes removed" % (valid.sum(), spikes.sum()))
    grid = fill_nans(grid)
    if smooth_cells and smooth_cells > 0:
        grid = cv2.GaussianBlur(grid.astype(np.float64), (0, 0), smooth_cells, borderType=cv2.BORDER_REPLICATE)
    return {"z": grid.astype(np.float32), "x0": float(xmin), "y1": float(ymax), "cell": float(cell)}


def sample_dsm(dsm, E, N):
    """Bilinear sample of the DSM at arrays E, N (same shape)."""
    z = dsm["z"]
    h, w = z.shape
    fx = np.clip((np.asarray(E, float) - dsm["x0"]) / dsm["cell"], 0, w - 1)
    fy = np.clip((dsm["y1"] - np.asarray(N, float)) / dsm["cell"], 0, h - 1)
    x0 = np.clip(np.floor(fx).astype(np.int64), 0, max(w - 2, 0))
    y0 = np.clip(np.floor(fy).astype(np.int64), 0, max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = fx - x0
    ay = fy - y0
    return (z[y0, x0] * (1 - ax) * (1 - ay) + z[y0, x1] * ax * (1 - ay)
            + z[y1, x0] * (1 - ax) * ay + z[y1, x1] * ax * ay).astype(np.float64)
