"""Structure-from-motion using GNSS/gimbal priors: tracks, triangulation, bundle adjustment."""
import math

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .ba import BundleAdjuster, project, NI
from .camera import undistort_normalized as undistort_normalized_px


# ------------------------------------------------------------------------------ tracks
def build_tracks(pairs, nfeat, n_img):
    """pairs: list of (i, j, qi, tj). Returns tid, img, feat arrays (observations) and n_tracks."""
    nfeat = np.asarray(nfeat, np.int64)
    offsets = np.r_[0, np.cumsum(nfeat)]
    rows = []
    cols = []
    for i, j, qi, tj in pairs:
        rows.append(offsets[i] + qi.astype(np.int64))
        cols.append(offsets[j] + tj.astype(np.int64))
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), 0
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    T = int(offsets[-1])
    G = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(T, T))
    _, lab = connected_components(G, directed=False)
    nodes = np.unique(np.r_[rows, cols])
    comp = lab[nodes].astype(np.int64)
    img = np.searchsorted(offsets, nodes, side="right") - 1
    feat = nodes - offsets[img]
    key = comp * n_img + img
    uk, cnt = np.unique(key, return_counts=True)
    bad = np.unique(uk[cnt > 1] // n_img)
    m = ~np.isin(comp, bad)
    comp, img, feat = comp[m], img[m], feat[m]
    uc, inv, sizes = np.unique(comp, return_inverse=True, return_counts=True)
    keep = sizes[inv] >= 2
    comp, img, feat = inv[keep], img[keep], feat[keep]
    uc2, tid = np.unique(comp, return_inverse=True)
    order = np.argsort(tid, kind="stable")
    return tid[order], img[order], feat[order], len(uc2)


def select_tracks(tid, img, xy, W, H, n_tracks, per_cell=2, gx=8, gy=6, max_len=8, seed=0):
    """Pick a well-distributed subset of tracks; cap track length. Returns observation mask."""
    tlen = np.bincount(tid, minlength=n_tracks)
    cx = np.clip((xy[:, 0] / W[img] * gx).astype(np.int64), 0, gx - 1)
    cy = np.clip((xy[:, 1] / H[img] * gy).astype(np.int64), 0, gy - 1)
    cell = img * (gx * gy) + cy * gx + cx
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(tid)) * 0.5
    order = np.lexsort((-(tlen[tid] + jitter), cell))
    cs = cell[order]
    first = np.searchsorted(cs, cs, side="left")
    rank = np.arange(len(cs)) - first
    chosen = np.zeros(n_tracks, bool)
    chosen[tid[order[rank < per_cell]]] = True
    m = chosen[tid]
    # cap track length (random subset of observations)
    idx = np.where(m)[0]
    r = rng.random(len(idx))
    o = np.lexsort((r, tid[idx]))
    t_s = tid[idx][o]
    first = np.searchsorted(t_s, t_s, side="left")
    rk = np.arange(len(t_s)) - first
    drop = idx[o[rk >= max_len]]
    m[drop] = False
    return m


# ---------------------------------------------------------------------- geometry helpers
def undistort_normalized(uv, img, intr, cam_group):
    """Pixel observations -> unit ray directions in camera frame."""
    out = np.zeros((len(uv), 3))
    g_of = cam_group[img]
    for g in np.unique(g_of):
        m = g_of == g
        xy = undistort_normalized_px(uv[m], intr[g])
        out[m, 0] = xy[:, 0]
        out[m, 1] = xy[:, 1]
        out[m, 2] = 1.0
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def triangulate(R, C, rays, oc, op, P):
    """Least-squares ray intersection (midpoint). Returns X (P,3), angle_deg (P,), ok (P,)."""
    d = np.einsum("mji,mj->mi", R[oc], rays)  # world ray directions
    A = np.zeros((P, 3, 3))
    b = np.zeros((P, 3))
    Cm = C[oc]
    for a in range(3):
        for c in range(3):
            val = (1.0 if a == c else 0.0) - d[:, a] * d[:, c]
            A[:, a, c] = np.bincount(op, weights=val, minlength=P)
            b[:, a] += np.bincount(op, weights=val * Cm[:, c], minlength=P)
    A[:, [0, 1, 2], [0, 1, 2]] += 1e-9
    X = np.linalg.solve(A, b[..., None])[..., 0]
    # triangulation angle: spread of ray directions
    mean_d = np.zeros((P, 3))
    for a in range(3):
        mean_d[:, a] = np.bincount(op, weights=d[:, a], minlength=P)
    mean_d /= np.linalg.norm(mean_d, axis=1, keepdims=True) + 1e-12
    cosang = np.clip((d * mean_d[op]).sum(1), -1, 1)
    ang = np.zeros(P)
    np.maximum.at(ang, op, np.degrees(np.arccos(cosang)))
    ang *= 2
    depth = ((X[op] - Cm) * d).sum(1)
    neg = np.zeros(P, bool)
    neg[op[depth <= 0]] = True
    ok = ~neg & np.all(np.isfinite(X), 1)
    return X, ang, ok


# ------------------------------------------------------------------------- yaw estimate
def estimate_yaw_from_pairs(sims, C, cx, cy, n, min_baseline=3.0):
    """Estimate grid azimuth of image-up for nadir images from pairwise similarities + GNSS.

    sims: list of (i, j, (ang, s, tx, ty), weight) where similarity maps undistorted px i -> j.
    Returns yaw_deg (n,) with NaN where unknown.
    """
    acc = np.zeros((n, 2))
    for i, j, sim, wgt in sims:
        ang, s, tx, ty = sim
        if s <= 0:
            continue
        dE, dN = C[j, 0] - C[i, 0], C[j, 1] - C[i, 1]
        base = math.hypot(dE, dN)
        if base < min_baseline:
            continue
        ca, sa = math.cos(ang), math.sin(ang)
        A = s * np.array([[ca, -sa], [sa, ca]])
        t = np.array([tx, ty])
        # centre of j seen in i
        pj = np.linalg.solve(A, np.array([cx[j], cy[j]]) - t)
        o = pj - np.array([cx[i], cy[i]])
        # centre of i seen in j
        pi = A @ np.array([cx[i], cy[i]]) + t
        o2 = pi - np.array([cx[j], cy[j]])
        beta = math.atan2(dE, dN)
        w = wgt * min(1.0, base / 10.0)
        if np.hypot(*o) > 5:
            yi = beta - math.atan2(o[0], -o[1])
            acc[i] += w * np.array([math.cos(yi), math.sin(yi)])
        if np.hypot(*o2) > 5:
            yj = (beta + math.pi) - math.atan2(o2[0], -o2[1])
            acc[j] += w * np.array([math.cos(yj), math.sin(yj)])
    yaw = np.full(n, np.nan)
    good = np.linalg.norm(acc, axis=1) > 0
    yaw[good] = np.degrees(np.arctan2(acc[good, 1], acc[good, 0]))
    return yaw


def fill_nan_by_time(vals, order):
    """Fill NaN angles (deg) from temporal neighbours."""
    v = vals.copy()
    seq = v[order]
    idx = np.where(~np.isnan(seq))[0]
    if len(idx) == 0:
        return np.zeros_like(v)
    for k in range(len(seq)):
        if np.isnan(seq[k]):
            nearest = idx[np.argmin(np.abs(idx - k))]
            seq[k] = seq[nearest]
    v[order] = seq
    return v


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- pipeline
class Reconstruction:
    """Holds the camera model and runs the adjustment stages."""

    def __init__(self, R, C, intr, cam_group, W, H, C_prior, C_sigma, R_prior, rot_sigma_deg,
                 intr_prior, intr_sigma, sigma_px, log):
        self.R = R
        self.C = C
        self.intr = intr
        self.cg = cam_group
        self.W = W
        self.H = H
        self.Cp = C_prior
        self.Cs = C_sigma
        self.Rp = R_prior
        self.rot_sigma = rot_sigma_deg
        self.Ip = intr_prior
        self.Is = intr_sigma
        self.sigma = sigma_px
        self.log = log

    def triangulate_tracks(self, tid, img, uv, n_tracks):
        rays = undistort_normalized(uv, img, self.intr, self.cg)
        X, ang, ok = triangulate(self.R, self.C, rays, img, tid, n_tracks)
        return X, ang, ok

    def reproj_errors(self, X, tid, img, uv):
        p, z = project(self.R, self.C, self.intr, self.cg, X, img, tid)
        e = np.linalg.norm(p - uv, axis=1)
        e[z <= 0] = 1e9
        return e

    def adjust(self, tid, img, uv, n_tracks, rounds, progress=None, time_budget=None):
        """rounds: list of dict(loss=, optimize_intrinsics=, max_iter=, reject=)"""
        X, ang, ok = self.triangulate_tracks(tid, img, uv, n_tracks)
        good_t = ok & (ang > 0.5)
        m = good_t[tid]
        tid, img, uv = tid[m], img[m], uv[m]
        tid, X = _compact(tid, X)
        for k, rd in enumerate(rounds):
            if progress:
                progress(k / max(1, len(rounds)), "Bundle adjustment round %d of %d" % (k + 1, len(rounds)))
            ba = BundleAdjuster(self.R, self.C, self.intr, self.cg, X, img, tid, uv,
                                sigma_px=self.sigma, C_prior=self.Cp, C_sigma=self.Cs,
                                R_prior=self.Rp, rot_sigma_deg=self.rot_sigma,
                                intr_prior=self.Ip, intr_sigma=self.Is,
                                optimize_intrinsics=rd.get("optimize_intrinsics", False),
                                loss_scale=rd.get("loss", 3.0), log=self.log)
            ba.run(max_iter=rd.get("max_iter", 30), time_limit=time_budget)
            self.R, self.C, self.intr = ba.R, ba.C, ba.intr
            # ba sorted observations internally; take its ordering
            img, tid, uv, X = ba.oc, ba.op, ba.uv, ba.X
            e = ba.residuals_px()
            rej = rd.get("reject", None)
            if rej:
                thr = rej * self.sigma
                keep = e < thr
                self.log("  rejected %d of %d observations (> %.1f px)" % ((~keep).sum(), len(keep), thr))
                img, tid, uv = img[keep], tid[keep], uv[keep]
                cnt = np.bincount(tid, minlength=len(X))
                m = cnt[tid] >= 2
                img, tid, uv = img[m], tid[m], uv[m]
                tid, X = _compact(tid, X)
                if rd.get("retriangulate", False):
                    Xn, ang, ok = self.triangulate_tracks(tid, img, uv, len(X))
                    X = np.where(ok[:, None], Xn, X)
        e = self.reproj_errors(X, tid, img, uv)
        return X, tid, img, uv, e


def _compact(tid, X):
    used = np.unique(tid)
    remap = -np.ones(len(X), np.int64)
    remap[used] = np.arange(len(used))
    return remap[tid], X[used]
