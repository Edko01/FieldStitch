"""Pairwise feature matching with geometric verification (worker side)."""
import math
import os
import pickle
from collections import OrderedDict

import cv2
import numpy as np

from .camera import undistort_pixels

_CTX = None
_CACHE = OrderedDict()
_CACHE_MAX = 48


def init_worker(ctx_file):
    global _CTX
    with open(ctx_file, "rb") as f:
        _CTX = pickle.load(f)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def load_features(feat_dir, i, W, H):
    d = np.load(os.path.join(feat_dir, "%d.npz" % i))
    xy = d["xy"].astype(np.float64)
    if len(xy):
        rw, rh = d["red_wh"]
        xy = (xy + 0.5) * np.array([W / float(rw), H / float(rh)]) - 0.5
    return xy, d["desc"]


def load_xy(feat_dir, i, W, H):
    """Keypoint coordinates only (full-resolution pixels), without reading descriptors."""
    with np.load(os.path.join(feat_dir, "%d.npz" % i)) as d:
        xy = d["xy"].astype(np.float64)
        if len(xy):
            rw, rh = d["red_wh"]
            xy = (xy + 0.5) * np.array([W / float(rw), H / float(rh)]) - 0.5
    return xy


def _get(i):
    if i in _CACHE:
        _CACHE.move_to_end(i)
        return _CACHE[i]
    c = _CTX
    xy, desc = load_features(c["feat_dir"], i, c["W"][i], c["H"][i])
    if len(xy):
        K, dd = c["K"][i], c["dist"][i]
        und = undistort_pixels(xy, (K[0, 0], K[0, 2], K[1, 2], dd[0], dd[1], dd[2], dd[3], dd[4]))
    else:
        und = xy
    d = desc.astype(np.float32)
    item = {"xy": xy, "und": und, "desc": d, "norm": (d * d).sum(1), "tree": None, "wh": (c["W"][i], c["H"][i])}
    _CACHE[i] = item
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return item


def _tree(item):
    if item["tree"] is None:
        from scipy.spatial import cKDTree
        item["tree"] = cKDTree(item["und"])
    return item["tree"]


def _ratio_match(da, db, ratio, checks):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=checks))
    knn = flann.knnMatch(da, db, k=2)
    q, t, dd = [], [], []
    for m in knn:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            q.append(m[0].queryIdx)
            t.append(m[0].trainIdx)
            dd.append(m[0].distance)
    return _unique(np.array(q, np.int64), np.array(t, np.int64), np.array(dd, np.float32))


def _ratio_match_bf(da, db, ratio, na=None, nb=None):
    """Exact nearest-neighbour ratio matching with one matrix product (fast for ~1000 features)."""
    if len(da) < 2 or len(db) < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    na = (da * da).sum(1) if na is None else na
    nb = (db * db).sum(1) if nb is None else nb
    d2 = na[:, None] + nb[None, :] - 2.0 * (da @ db.T)
    two = np.argpartition(d2, 1, axis=1)[:, :2]
    r = np.arange(len(da))
    a0, a1 = d2[r, two[:, 0]], d2[r, two[:, 1]]
    first = np.where(a0 <= a1, two[:, 0], two[:, 1])
    b1 = np.maximum(np.minimum(a0, a1), 0)
    b2 = np.maximum(np.maximum(a0, a1), 0)
    ok = np.sqrt(b1) < ratio * np.sqrt(b2)
    return _unique(r[ok].astype(np.int64), first[ok].astype(np.int64), b1[ok].astype(np.float32))


def _unique(q, t, dd):
    if len(q) == 0:
        return q, t
    order = np.lexsort((dd, t))
    t_s = t[order]
    first = np.r_[True, t_s[1:] != t_s[:-1]]
    keep = order[first]
    return q[keep], t[keep]


def _guided(A, B, Hm, radius, kc, ratio):
    """Match all features of A to spatially-predicted candidates in B."""
    pred = cv2.perspectiveTransform(A["und"].reshape(-1, 1, 2).astype(np.float64), Hm).reshape(-1, 2)
    nb = len(B["und"])
    # only features predicted to land inside B (with a margin) can match
    Wb, Hb = B["wh"]
    inside = np.nonzero((pred[:, 0] > -radius) & (pred[:, 0] < Wb + radius)
                        & (pred[:, 1] > -radius) & (pred[:, 1] < Hb + radius))[0]
    if len(inside) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    tree = _tree(B)
    kc = min(kc, nb)
    dist, idx = tree.query(pred[inside], k=kc, distance_upper_bound=radius)
    if kc == 1:
        idx = idx[:, None]
    valid = idx < nb
    has = valid.any(1)
    inside, idx, valid = inside[has], idx[has], valid[has]
    idx = np.where(valid, idx, 0)
    qs, ts, ds = [], [], []
    for s0 in range(0, len(inside), 1024):
        sl = slice(s0, s0 + 1024)
        da = A["desc"][inside[sl]]
        cand = B["desc"][idx[sl]]  # (n,k,128)
        d2 = A["norm"][inside[sl], None] + B["norm"][idx[sl]] - 2 * np.matmul(cand, da[:, :, None])[:, :, 0]
        d2 = np.where(valid[sl], np.maximum(d2, 0), np.inf)
        o = np.argpartition(d2, 1, axis=1)[:, :2] if d2.shape[1] > 2 else np.argsort(d2, axis=1)[:, :2]
        r = np.arange(d2.shape[0])
        a0 = d2[r, o[:, 0]]
        a1 = d2[r, o[:, 1]] if d2.shape[1] > 1 else np.full(len(r), np.inf)
        best = np.where(a0 <= a1, o[:, 0], o[:, 1]) if d2.shape[1] > 1 else o[:, 0]
        b1 = np.minimum(a0, a1)
        b2 = np.maximum(a0, a1)
        ok = np.isfinite(b1) & (np.sqrt(b1) < ratio * np.sqrt(b2))
        qs.append(inside[sl][ok])
        ts.append(idx[sl][r[ok], best[ok]])
        ds.append(b1[ok])
    q = np.concatenate(qs)
    t = np.concatenate(ts)
    return _unique(q, t, np.concatenate(ds))


def _predict(u, i, j):
    """Predict where undistorted pixels u of image i land in image j (ground plane prior)."""
    c = _CTX
    Ki, Kj = c["K"][i], c["K"][j]
    Ri, Rj = c["R"][i], c["R"][j]
    Ci, Cj = c["C"][i], c["C"][j]
    rays = np.c_[u, np.ones(len(u))] @ np.linalg.inv(Ki).T
    d = rays @ Ri  # = (Ri^T rays^T)^T
    zg = c["zg"][i]
    dz = d[:, 2]
    dz = np.where(np.abs(dz) < 1e-6, -1e-6, dz)
    t = (zg - Ci[2]) / dz
    G = Ci[None] + t[:, None] * d
    q = (G - Cj[None]) @ Rj.T @ Kj.T
    z = q[:, 2]
    ok = (t > 0) & (z > 0)
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return q[:, :2] / z[:, None], ok


def _verify(c, i, j, ui, uj, q, t):
    pi = ui[q].astype(np.float32)
    pj = uj[t].astype(np.float32)
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    F, mask = cv2.findFundamentalMat(pi, pj, method, c["ransac_thr"], 0.999, 10000)
    if F is None or mask is None:
        return None, None, 0
    mask = mask.ravel().astype(bool)
    return q[mask], t[mask], int(mask.sum())


def _similarity(c, i, ui, uj, q, t):
    try:
        M, _ = cv2.estimateAffinePartial2D(ui[q].astype(np.float32), uj[t].astype(np.float32), method=cv2.RANSAC,
                                           ransacReprojThreshold=0.02 * max(c["W"][i], c["H"][i]))
        if M is None:
            return None
        return (math.atan2(M[1, 0], M[0, 0]), math.hypot(M[0, 0], M[1, 0]), float(M[0, 2]), float(M[1, 2]))
    except Exception:
        return None


def match_pair(task):
    i, j = task
    c = _CTX
    try:
        A = _get(i)
        B = _get(j)
        ui, uj = A["und"], B["und"]
        if len(ui) < 30 or len(uj) < 30:
            return i, j, None, None, None, 0
        use_prior = c["has_rot"][i] and c["has_rot"][j] and c.get("use_prior", True)
        tol = c["prior_tol"] * max(c["W"][j], c["H"][j])
        result = None
        # ---- stage 1: strongest features -> coarse homography -> guided matching
        n1 = c.get("n_coarse", 1500)
        if c.get("guided", True) and len(ui) > n1 // 2:
            q1, t1 = _ratio_match_bf(A["desc"][:n1], B["desc"][:n1], 0.85, A["norm"][:n1], B["norm"][:n1])
            if use_prior and len(q1):
                pred, ok = _predict(ui[q1], i, j)
                m = ok & (np.linalg.norm(pred - uj[t1], axis=1) < tol)
                q1, t1 = q1[m], t1[m]
            if len(q1) >= 12:
                Hm, hm = cv2.findHomography(ui[q1], uj[t1], cv2.RANSAC, 0.012 * max(c["W"][i], c["H"][i]))
                if Hm is not None and hm is not None and hm.sum() >= 10:
                    radius = c["guided_radius"] * max(c["W"][j], c["H"][j])
                    q, t = _guided(A, B, Hm, radius, c.get("guided_k", 24), 0.85)
                    if len(q) >= c["min_inliers"]:
                        q, t, n_in = _verify(c, i, j, ui, uj, q, t)
                        if q is not None and n_in >= max(c["min_inliers"], 40):
                            result = (q, t, n_in)
        # ---- fallback: exhaustive ratio matching
        if result is None:
            q, t = _ratio_match(A["desc"], B["desc"], c.get("ratio", 0.8), c.get("checks", 48))
            if use_prior and len(q):
                pred, ok = _predict(ui[q], i, j)
                m = ok & (np.linalg.norm(pred - uj[t], axis=1) < tol)
                q, t = q[m], t[m]
            if len(q) < 20:
                return i, j, None, None, None, 0
            q, t, n_in = _verify(c, i, j, ui, uj, q, t)
            if q is None or n_in < c["min_inliers"]:
                return i, j, None, None, None, n_in
            result = (q, t, n_in)
        q, t, n_in = result
        sim = _similarity(c, i, ui, uj, q, t)
        return i, j, q.astype(np.int32), t.astype(np.int32), sim, n_in
    except Exception as e:  # noqa
        return i, j, None, None, ("error", str(e)), 0
