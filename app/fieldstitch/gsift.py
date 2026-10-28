"""SIFT keypoints and descriptors written only with array operations, so the same code runs on an
NVIDIA GPU (CuPy) or on the CPU (NumPy). It follows Lowe's SIFT as implemented in OpenCV:
the first octave is the image upsampled 2x, 3 layers per octave, sigma 1.6, sub-pixel refinement,
edge rejection, 36-bin orientation histograms and 4x4x8 descriptors with trilinear binning.

The GPU and CPU versions give the same keypoints (up to float rounding), which is what the GPU
self-test checks before a job uses the GPU.
"""
import math

import numpy as np

N_LAYERS = 3
SIGMA = 1.6
INIT_SIGMA = 0.5
BORDER = 5
MAX_INTERP = 5
ORI_BINS = 36
ORI_RADIUS = 4.5          # 3 * 1.5
ORI_SIG = 1.5
ORI_PEAK = 0.8
D_WIDTH = 4
D_BINS = 8
D_SCL = 3.0
D_MAG_THR = 0.2


def backend(gpu):
    if gpu:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi
        return cp, cndi
    import scipy.ndimage as ndi
    return np, ndi


def to_host(xp, a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _scatter_sum(xp, idx, w, n):
    """Sum w into n bins. On the GPU this uses float32 atomics (double-precision atomics, which
    bincount uses, are very slow on GeForce cards)."""
    if xp is np:
        return np.bincount(idx, weights=w, minlength=n).astype(np.float32)
    import cupyx
    out = xp.zeros(n, xp.float32)
    cupyx.scatter_add(out, idx, w.astype(xp.float32, copy=False))
    return out


def _gauss_kernel(sigma):
    k = int(round(sigma * 8 + 1)) | 1
    x = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    g = np.exp(-x * x / (2 * sigma * sigma))
    return (g / g.sum()).astype(np.float32)


class _Kernels:
    """Gaussian kernels, uploaded once per backend."""

    def __init__(self, xp):
        sig = [0.0] * (N_LAYERS + 3)
        sig[0] = SIGMA
        k = 2.0 ** (1.0 / N_LAYERS)
        for i in range(1, N_LAYERS + 3):
            prev = k ** (i - 1) * SIGMA
            total = prev * k
            sig[i] = math.sqrt(total * total - prev * prev)
        self.layer = [xp.asarray(_gauss_kernel(s)) for s in sig]
        base = math.sqrt(max(SIGMA * SIGMA - (2 * INIT_SIGMA) ** 2, 0.01))
        self.base = xp.asarray(_gauss_kernel(base))


_KCACHE = {}


def _kernels(xp):
    key = xp.__name__
    if key not in _KCACHE:
        _KCACHE[key] = _Kernels(xp)
    return _KCACHE[key]


def _blur(ndi, img, k):
    t = ndi.correlate1d(img, k, axis=0, mode="mirror")
    return ndi.correlate1d(t, k, axis=1, mode="mirror")


def _up2(xp, a):
    """2x bilinear upsample (pixel-centre aligned, edges clamped), like cv2.resize INTER_LINEAR."""
    def along(a, axis):
        n = a.shape[axis]
        prev = xp.take(a, xp.clip(xp.arange(n) - 1, 0, n - 1), axis=axis)
        nxt = xp.take(a, xp.clip(xp.arange(n) + 1, 0, n - 1), axis=axis)
        even = 0.75 * a + 0.25 * prev
        odd = 0.75 * a + 0.25 * nxt
        out = xp.stack([even, odd], axis=axis + 1)
        shape = list(a.shape)
        shape[axis] = 2 * n
        return out.reshape(shape)
    return along(along(a, 0), 1)


def _pyramid(xp, ndi, gray):
    K = _kernels(xp)
    img = xp.asarray(gray, dtype=xp.float32) * (1.0 / 255.0)
    base = _blur(ndi, _up2(xp, img), K.base)
    n_oct = max(1, int(round(math.log2(min(gray.shape)) - 2)) + 1)
    gpyr, dpyr = [], []
    cur = base
    for o in range(n_oct):
        if min(cur.shape) < 2 * BORDER + 4:
            break
        layers = [cur]
        for i in range(1, N_LAYERS + 3):
            layers.append(_blur(ndi, layers[-1], K.layer[i]))
        gpyr.append(layers)
        dpyr.append(xp.stack([layers[i + 1] - layers[i] for i in range(N_LAYERS + 2)]))
        cur = layers[N_LAYERS][::2, ::2]
    return gpyr, dpyr


def _inv3_solve(xp, H, b):
    """Solve batched 3x3 systems H x = b (H: (n,3,3), b: (n,3)). Returns x and a validity mask."""
    a, bb, c = H[:, 0, 0], H[:, 0, 1], H[:, 0, 2]
    d, e, f = H[:, 1, 0], H[:, 1, 1], H[:, 1, 2]
    g, h, i = H[:, 2, 0], H[:, 2, 1], H[:, 2, 2]
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    det = a * A + bb * B + c * C
    ok = xp.abs(det) > 1e-20
    det = xp.where(ok, det, 1.0)
    inv = xp.stack([
        xp.stack([A, -(bb * i - c * h), bb * f - c * e], 1),
        xp.stack([B, a * i - c * g, -(a * f - c * d)], 1),
        xp.stack([C, -(a * h - bb * g), a * e - bb * d], 1)], 1) / det[:, None, None]
    x = (inv * b[:, None, :]).sum(2)   # no BLAS needed on the GPU
    return x, ok


def _derivs(xp, D, l, r, c):
    v = D[l, r, c]
    dx = (D[l, r, c + 1] - D[l, r, c - 1]) * 0.5
    dy = (D[l, r + 1, c] - D[l, r - 1, c]) * 0.5
    ds = (D[l + 1, r, c] - D[l - 1, r, c]) * 0.5
    dxx = D[l, r, c + 1] + D[l, r, c - 1] - 2 * v
    dyy = D[l, r + 1, c] + D[l, r - 1, c] - 2 * v
    dss = D[l + 1, r, c] + D[l - 1, r, c] - 2 * v
    dxy = (D[l, r + 1, c + 1] - D[l, r + 1, c - 1] - D[l, r - 1, c + 1] + D[l, r - 1, c - 1]) * 0.25
    dxs = (D[l + 1, r, c + 1] - D[l + 1, r, c - 1] - D[l - 1, r, c + 1] + D[l - 1, r, c - 1]) * 0.25
    dys = (D[l + 1, r + 1, c] - D[l + 1, r - 1, c] - D[l - 1, r + 1, c] + D[l - 1, r - 1, c]) * 0.25
    g = xp.stack([dx, dy, ds], 1)
    H = xp.stack([xp.stack([dxx, dxy, dxs], 1), xp.stack([dxy, dyy, dys], 1), xp.stack([dxs, dys, dss], 1)], 1)
    return v, g, H, dxx, dyy, dxy


def _detect_octave(xp, ndi, D, contrast, edge):
    """Candidate extrema in one octave's DoG stack, refined. Returns arrays (l, r, c, x, y, s_off, resp)."""
    nl, rows, cols = D.shape
    pthr = 0.8 * 0.5 * contrast / N_LAYERS
    mx = ndi.maximum_filter(D, size=3, mode="nearest")
    mn = ndi.minimum_filter(D, size=3, mode="nearest")
    ext = ((D >= mx) & (D > pthr)) | ((D <= mn) & (D < -pthr))
    ext[0] = False
    ext[-1] = False
    ext[:, :BORDER] = False
    ext[:, rows - BORDER:] = False
    ext[:, :, :BORDER] = False
    ext[:, :, cols - BORDER:] = False
    l, r, c = xp.nonzero(ext)
    if l.size == 0:
        return None
    l = l.astype(xp.int32)
    r = r.astype(xp.int32)
    c = c.astype(xp.int32)
    alive = xp.ones(l.shape, bool)
    done = xp.zeros(l.shape, bool)
    X = xp.zeros((l.size, 3), xp.float32)
    for _ in range(MAX_INTERP):
        _, g, H, _, _, _ = _derivs(xp, D, l, r, c)
        x, ok = _inv3_solve(xp, H, g)
        x = -x
        alive &= ok
        conv = (xp.abs(x) < 0.5).all(1)
        newly = alive & ~done & conv
        X = xp.where(newly[:, None], x, X)
        done |= newly
        move = alive & ~done
        big = (xp.abs(x) > 1e6).any(1)
        alive &= ~(move & big)
        move = alive & ~done
        step = xp.rint(xp.where(move[:, None], x, 0)).astype(xp.int32)
        c = c + step[:, 0]
        r = r + step[:, 1]
        l = l + step[:, 2]
        inside = (l >= 1) & (l <= N_LAYERS) & (c >= BORDER) & (c < cols - BORDER) & (r >= BORDER) & (r < rows - BORDER)
        alive &= inside
        l = xp.clip(l, 1, N_LAYERS)
        r = xp.clip(r, BORDER, rows - BORDER - 1)
        c = xp.clip(c, BORDER, cols - BORDER - 1)
    keep = alive & done
    l, r, c, X = l[keep], r[keep], c[keep], X[keep]
    if l.size == 0:
        return None
    v, g, H, dxx, dyy, dxy = _derivs(xp, D, l, r, c)
    contr = v + 0.5 * (g * X).sum(1)
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    ok = (xp.abs(contr) * N_LAYERS >= contrast) & (det > 0) & (tr * tr * edge < (edge + 1) ** 2 * det)
    return (l[ok], r[ok], c[ok], X[ok, 0], X[ok, 1], X[ok, 2], xp.abs(contr[ok]))


def _grad(xp, img):
    """Gradient magnitude and angle (degrees, 0..360) as OpenCV's SIFT computes them, zero on the border."""
    dx = xp.zeros_like(img)
    dy = xp.zeros_like(img)
    dx[1:-1, 1:-1] = img[1:-1, 2:] - img[1:-1, :-2]
    dy[1:-1, 1:-1] = img[:-2, 1:-1] - img[2:, 1:-1]
    mag = xp.sqrt(dx * dx + dy * dy)
    ang = xp.degrees(xp.arctan2(dy, dx))
    ang = xp.where(ang < 0, ang + 360.0, ang)
    return mag, ang


def _orientations(xp, mag, ang, r, c, scl, rows, cols):
    """Dominant orientations. Returns (kp index, angle in OpenCV convention)."""
    n = r.size
    rad = xp.rint(ORI_RADIUS * scl).astype(xp.int32)
    R = int(to_host(xp, rad.max())) if n else 0
    off = xp.arange(-R, R + 1, dtype=xp.int32)
    oi, oj = xp.meshgrid(off, off, indexing="ij")
    oi = oi.ravel()
    oj = oj.ravel()
    y = r[:, None] + oi[None]
    x = c[:, None] + oj[None]
    inwin = (xp.abs(oi)[None] <= rad[:, None]) & (xp.abs(oj)[None] <= rad[:, None])
    valid = inwin & (y > 0) & (y < rows - 1) & (x > 0) & (x < cols - 1)
    yc = xp.clip(y, 0, rows - 1)
    xc = xp.clip(x, 0, cols - 1)
    sw = ORI_SIG * scl
    w = xp.exp(-(oi[None] ** 2 + oj[None] ** 2).astype(xp.float32) / (2.0 * sw[:, None] ** 2))
    m = mag[yc, xc] * w * valid
    b = xp.rint(ang[yc, xc] * (ORI_BINS / 360.0)).astype(xp.int32) % ORI_BINS
    idx = (xp.arange(n, dtype=xp.int32)[:, None] * ORI_BINS + b).ravel()
    hist = _scatter_sum(xp, idx, m.ravel(), n * ORI_BINS).reshape(n, ORI_BINS)
    h = (xp.roll(hist, 2, 1) + xp.roll(hist, -2, 1)) * (1.0 / 16) + \
        (xp.roll(hist, 1, 1) + xp.roll(hist, -1, 1)) * (4.0 / 16) + hist * (6.0 / 16)
    hmax = h.max(1, keepdims=True)
    left = xp.roll(h, 1, 1)
    right = xp.roll(h, -1, 1)
    peak = (h > left) & (h > right) & (h >= ORI_PEAK * hmax) & (hmax > 0)
    kp, j = xp.nonzero(peak)
    hl, hc, hr = left[kp, j], h[kp, j], right[kp, j]
    den = hl - 2 * hc + hr
    den = xp.where(xp.abs(den) < 1e-12, -1e-12, den)
    bin_ = j + 0.5 * (hl - hr) / den
    bin_ = xp.where(bin_ < 0, bin_ + ORI_BINS, bin_)
    bin_ = xp.where(bin_ >= ORI_BINS, bin_ - ORI_BINS, bin_)
    angle = 360.0 - (360.0 / ORI_BINS) * bin_
    angle = xp.where(xp.abs(angle - 360.0) < 1e-4, 0.0, angle)
    return kp, angle.astype(xp.float32)


def detect(gray, gpu=False, max_kp=12000, contrast=0.015, edge=12.0):
    """Detect oriented keypoints. Returns dict of host arrays: x, y (input-image pixels), size, angle,
    response, plus private fields used by describe() (octave, layer, octave coords)."""
    xp, ndi = backend(gpu)
    gpyr, dpyr = _pyramid(xp, ndi, gray)
    out = []
    grads = {}
    for o, D in enumerate(dpyr):
        det = _detect_octave(xp, ndi, D, contrast, edge)
        if det is None:
            continue
        l, r, c, xc, yc, xi, resp = det
        rows, cols = D.shape[1:]
        scl = SIGMA * 2.0 ** ((l.astype(xp.float32) + xi) / N_LAYERS)    # in octave pixels
        for li in range(1, N_LAYERS + 1):
            sel = xp.nonzero(l == li)[0]
            if sel.size == 0:
                continue
            key = (o, li)
            if key not in grads:
                grads[key] = _grad(xp, gpyr[o][li])
            mag, ang = grads[key]
            kp, angle = _orientations(xp, mag, ang, r[sel], c[sel], scl[sel], rows, cols)
            s = sel[kp]
            octx = c[s].astype(xp.float32) + xc[s]
            octy = r[s].astype(xp.float32) + yc[s]
            f = 2.0 ** o * 0.5
            out.append(dict(x=octx * f, y=octy * f, size=scl[s] * 2 * f, angle=angle, response=resp[s],
                            octave=xp.full(s.shape, o, xp.int32), layer=xp.full(s.shape, li, xp.int32)))
    if not out:
        return {k: np.zeros(0, np.float32) for k in ("x", "y", "size", "angle", "response", "octave", "layer")}, \
            (xp, ndi, gpyr, grads)
    kp = {k: xp.concatenate([d[k] for d in out]) for k in out[0]}
    # strongest first; keep at most max_kp
    order = xp.argsort(-kp["response"], kind="stable") if xp is np else xp.argsort(-kp["response"])
    order = order[:max_kp]
    host = {k: to_host(xp, v[order]) for k, v in kp.items()}
    return host, (xp, ndi, gpyr, grads)


def describe(state, kps, sel, budget=2.5e6):
    """128-d SIFT descriptors (float, OpenCV scaling: L2 norm 512) for keypoints kps[sel]."""
    xp, ndi, gpyr, grads = state
    sel = np.asarray(sel, np.int64)
    n_all = len(sel)
    desc = np.zeros((n_all, D_WIDTH * D_WIDTH * D_BINS), np.float32)
    if n_all == 0:
        return desc
    o_all = kps["octave"][sel]
    l_all = kps["layer"][sel]
    for key in sorted(set(zip(o_all.tolist(), l_all.tolist()))):
        o, li = key
        rows_idx = np.nonzero((o_all == o) & (l_all == li))[0]
        if key not in grads:
            grads[key] = _grad(xp, gpyr[o][li])
        mag, ang = grads[key]
        rows, cols = mag.shape
        f = 1.0 / (2.0 ** o * 0.5)
        smax = float(kps["size"][sel[rows_idx]].max()) * f * 0.5
        rmax = int(round(D_SCL * smax * 1.4142135623730951 * (D_WIDTH + 1) * 0.5))
        chunk = max(32, int(budget / (2 * rmax + 1) ** 2))
        for s0 in range(0, len(rows_idx), chunk):
            part = rows_idx[s0:s0 + chunk]
            k = sel[part]
            px = xp.asarray(kps["x"][k] * f)
            py = xp.asarray(kps["y"][k] * f)
            scl = xp.asarray(kps["size"][k] * f * 0.5)
            ang_kp = 360.0 - xp.asarray(kps["angle"][k])
            ang_kp = xp.where(xp.abs(ang_kp - 360.0) < 1e-4, 0.0, ang_kp)
            desc[part] = to_host(xp, _descr(xp, mag, ang, px, py, scl, ang_kp, rows, cols))
    return desc


def _descr(xp, mag, ang, px, py, scl, ori, rows, cols):
    d, nb = D_WIDTH, D_BINS
    n = px.size
    hist_width = D_SCL * scl
    rad = xp.rint(hist_width * 1.4142135623730951 * (d + 1) * 0.5).astype(xp.int32)
    rad = xp.minimum(rad, int(math.sqrt(rows * rows + cols * cols)))
    R = int(to_host(xp, rad.max()))
    off = xp.arange(-R, R + 1, dtype=xp.float32)
    oi, oj = xp.meshgrid(off, off, indexing="ij")
    oi = oi.ravel()[None]
    oj = oj.ravel()[None]
    t = xp.radians(ori)
    cos_t = (xp.cos(t) / hist_width)[:, None]
    sin_t = (xp.sin(t) / hist_width)[:, None]
    c_rot = oj * cos_t - oi * sin_t
    r_rot = oj * sin_t + oi * cos_t
    rbin = r_rot + d / 2.0 - 0.5
    cbin = c_rot + d / 2.0 - 0.5
    pr = xp.rint(py).astype(xp.int32)[:, None]
    pc = xp.rint(px).astype(xp.int32)[:, None]
    y = pr + oi.astype(xp.int32)
    x = pc + oj.astype(xp.int32)
    inwin = (xp.abs(oi) <= rad[:, None]) & (xp.abs(oj) <= rad[:, None])
    ok = inwin & (rbin > -1) & (rbin < d) & (cbin > -1) & (cbin < d) & (y > 0) & (y < rows - 1) & (x > 0) & (x < cols - 1)
    yc = xp.clip(y, 0, rows - 1)
    xc = xp.clip(x, 0, cols - 1)
    w = xp.exp((c_rot * c_rot + r_rot * r_rot) * (-1.0 / (d * d * 0.5)))
    m = mag[yc, xc] * w * ok
    obin = (ang[yc, xc] - ori[:, None]) * (nb / 360.0)
    r0 = xp.floor(rbin)
    c0 = xp.floor(cbin)
    o0 = xp.floor(obin)
    rb = rbin - r0
    cb = cbin - c0
    ob = obin - o0
    o0 = o0.astype(xp.int32)
    o0 = xp.where(o0 < 0, o0 + nb, o0)
    o0 = xp.where(o0 >= nb, o0 - nb, o0)
    r0 = r0.astype(xp.int32)
    c0 = c0.astype(xp.int32)
    v_r1 = m * rb
    v_r0 = m - v_r1
    v_rc11 = v_r1 * cb
    v_rc10 = v_r1 - v_rc11
    v_rc01 = v_r0 * cb
    v_rc00 = v_r0 - v_rc01
    v111 = v_rc11 * ob
    v110 = v_rc11 - v111
    v101 = v_rc10 * ob
    v100 = v_rc10 - v101
    v011 = v_rc01 * ob
    v010 = v_rc01 - v011
    v001 = v_rc00 * ob
    v000 = v_rc00 - v001
    HS = (d + 2) * (d + 2) * (nb + 2)
    base = xp.arange(n, dtype=xp.int64)[:, None] * HS + ((r0 + 1) * (d + 2) + c0 + 1) * (nb + 2) + o0
    base = xp.where(ok, base, 0)
    s1, s2 = nb + 2, (d + 2) * (nb + 2)
    idx = xp.concatenate([(base).ravel(), (base + 1).ravel(), (base + s1).ravel(), (base + s1 + 1).ravel(),
                          (base + s2).ravel(), (base + s2 + 1).ravel(), (base + s2 + s1).ravel(),
                          (base + s2 + s1 + 1).ravel()])
    val = xp.concatenate([v000.ravel(), v001.ravel(), v010.ravel(), v011.ravel(),
                          v100.ravel(), v101.ravel(), v110.ravel(), v111.ravel()])
    hist = _scatter_sum(xp, idx, val, n * HS).reshape(n, d + 2, d + 2, nb + 2)
    hist[..., 0] += hist[..., nb]
    hist[..., 1] += hist[..., nb + 1]
    dst = hist[:, 1:d + 1, 1:d + 1, :nb].reshape(n, d * d * nb)
    nrm = xp.sqrt((dst * dst).sum(1, keepdims=True))
    dst = xp.minimum(dst, nrm * D_MAG_THR)
    nrm = xp.sqrt((dst * dst).sum(1, keepdims=True))
    return dst * (512.0 / xp.maximum(nrm, 1e-7))
