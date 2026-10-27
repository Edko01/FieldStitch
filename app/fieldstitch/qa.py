"""Quality check: measure how well neighbouring photos line up after orthorectification."""
import math

import cv2
import numpy as np

from .features import load_image
from .ortho import project_pts
from .surface import sample_dsm


def _ortho_patch(cam, img, reduce, E, N, Z):
    P = np.c_[E.ravel(), N.ravel(), Z.ravel()]
    u, v, z = project_pts(cam["R"], cam["C"], cam["k"], P)
    W, H = cam["W"], cam["H"]
    ok = (z > 0) & (u > 2) & (u < W - 3) & (v > 2) & (v < H - 3)
    mx = ((u + 0.5) / reduce - 0.5).astype(np.float32).reshape(E.shape)
    my = ((v + 0.5) / reduce - 0.5).astype(np.float32).reshape(E.shape)
    patch = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
    return patch, ok.reshape(E.shape)


def seam_check(cams, dsm, gsd, n_pairs=30, patch=384, log=None):
    """Returns dict with median / 90th percentile misalignment (pixels at gsd and cm)."""
    n = len(cams)
    if n < 2:
        return None
    C = np.array([c["C"] for c in cams])
    from scipy.spatial import cKDTree
    tree = cKDTree(C[:, :2])
    pairs = []
    for i in range(n):
        d, j = tree.query(C[i, :2], k=min(n, 4))
        for dd, jj in zip(np.atleast_1d(d)[1:], np.atleast_1d(j)[1:]):
            if i < jj and dd > 0.15 * cams[i]["agl"] and dd < 0.8 * cams[i]["agl"]:
                pairs.append((i, int(jj)))
    if not pairs:
        return None
    rng = np.random.default_rng(0)
    if len(pairs) > n_pairs:
        pairs = [pairs[k] for k in rng.choice(len(pairs), n_pairs, replace=False)]
    gsd_native = float(np.median([c["agl"] / c["k"][0] for c in cams]))
    reduce = 1
    for r in (2, 4):
        if gsd >= r * gsd_native * 0.95:
            reduce = r
    cache = {}

    def get(i):
        if i not in cache:
            g = load_image(cams[i]["path"], reduce, color=False)
            cache[i] = g.astype(np.float32)
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[i]

    shifts = []
    win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
    for i, j in pairs:
        mid = 0.5 * (C[i, :2] + C[j, :2])
        off = (np.arange(patch) - patch / 2 + 0.5) * gsd
        E, N = np.meshgrid(mid[0] + off, mid[1] - off)
        Z = sample_dsm(dsm, E, N)
        a, oka = _ortho_patch(cams[i], get(i), reduce, E, N, Z)
        b, okb = _ortho_patch(cams[j], get(j), reduce, E, N, Z)
        ok = oka & okb
        if ok.mean() < 0.9:
            continue
        a = a - a.mean()
        b = b - b.mean()
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        if resp > 0.08 and abs(dx) < patch / 4 and abs(dy) < patch / 4:
            shifts.append(math.hypot(dx, dy))
    if not shifts:
        return None
    s = np.array(shifts)
    out = {"pairs": int(len(s)), "median_px": round(float(np.median(s)), 2),
           "p90_px": round(float(np.percentile(s, 90)), 2),
           "median_cm": round(float(np.median(s)) * gsd * 100, 1),
           "p90_cm": round(float(np.percentile(s, 90)) * gsd * 100, 1)}
    if log:
        log("Seam check (%d photo pairs): median misalignment %.2f px (%.1f cm), 90%% under %.2f px (%.1f cm)"
            % (out["pairs"], out["median_px"], out["median_cm"], out["p90_px"], out["p90_cm"]))
    return out
