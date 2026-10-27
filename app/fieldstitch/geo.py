"""Coordinate systems and rotation helpers."""
import math

import numpy as np
from pyproj import Transformer


def utm_epsg(lon, lat):
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


class Geo:
    def __init__(self, epsg):
        self.epsg = int(epsg)
        self.fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self.inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def to_utm(self, lon, lat):
        return self.fwd.transform(lon, lat)

    def to_ll(self, e, n):
        return self.inv.transform(e, n)

    def convergence(self, lon, lat):
        """Azimuth of true north measured in the grid (degrees, clockwise)."""
        e0, n0 = self.fwd.transform(lon, lat)
        e1, n1 = self.fwd.transform(lon, lat + 1e-4)
        return math.degrees(math.atan2(e1 - e0, n1 - n0))


_M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)  # NED -> ENU
_G = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # camera(OpenCV) -> gimbal(fwd,right,down)


def rot_from_ypr(yaw_deg, pitch_deg, roll_deg):
    """Camera-from-world rotation (world = local ENU grid) from gimbal yaw/pitch/roll.

    yaw: grid azimuth of the camera forward axis (deg, clockwise from north)
    pitch: -90 = nadir.
    """
    p, t, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cz, sz = math.cos(p), math.sin(p)
    cy, sy = math.cos(t), math.sin(t)
    cx, sx = math.cos(r), math.sin(r)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rwc = _M @ Rz @ Ry @ Rx @ _G
    return Rwc.T


def rodrigues_batch(w):
    """Rotation matrices from rotation vectors, shape (N,3) -> (N,3,3)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=1)
    n = w.shape[0]
    K = np.zeros((n, 3, 3))
    small = th < 1e-12
    k = np.zeros_like(w)
    k[~small] = w[~small] / th[~small, None]
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    s = np.sin(th)[:, None, None]
    c = (1 - np.cos(th))[:, None, None]
    R = np.eye(3)[None] + s * K + c * (K @ K)
    return R


def rotvec_batch(R):
    """Rotation vectors from rotation matrices (N,3,3) -> (N,3)."""
    R = np.asarray(R, float)
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(th)
    out = np.zeros_like(v)
    ok = s > 1e-8
    out[ok] = v[ok] * (th[ok] / (2 * s[ok]))[:, None]
    small = ~ok & (th < 1e-3)
    out[small] = v[small] / 2
    return out
