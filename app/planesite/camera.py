"""Brown-Conrady camera model helpers (OpenCV convention, fx = fy)."""
import numpy as np


def undistort_normalized(uv, k, iters=25):
    """Pixel coords (n,2) -> undistorted normalized coords (n,2). k = f,cx,cy,k1,k2,p1,p2,k3."""
    f, cx, cy, k1, k2, p1, p2, k3 = k
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    xd = (uv[:, 0] - cx) / f
    yd = (uv[:, 1] - cy) / f
    x = xd.copy()
    y = yd.copy()
    if k1 == 0 and k2 == 0 and k3 == 0 and p1 == 0 and p2 == 0:
        return np.c_[x, y]
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        rad = np.where(np.abs(rad) < 0.2, 0.2, rad)
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad
        y = (yd - dy) / rad
    return np.c_[x, y]


def undistort_pixels(uv, k):
    """Distorted pixels -> ideal pinhole pixels with the same f, cx, cy."""
    n = undistort_normalized(uv, k)
    return np.c_[n[:, 0] * k[0] + k[1], n[:, 1] * k[0] + k[2]]
