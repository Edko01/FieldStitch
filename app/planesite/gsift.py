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


_PYR_SRC = r"""
extern "C" __global__ void sift_up2(const float* a, float* out, const int rows, const int cols)
{
    // 2x bilinear upsample, pixel-centre aligned, edges clamped (same arithmetic as _up2)
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 2 * cols || y >= 2 * rows) return;
    const int iy = y >> 1, ix = x >> 1;
    const int ny = (y & 1) ? min(iy + 1, rows - 1) : max(iy - 1, 0);
    const int nx = (x & 1) ? min(ix + 1, cols - 1) : max(ix - 1, 0);
    const float vc = 0.75f * a[iy * cols + ix] + 0.25f * a[ny * cols + ix];
    const float vn = 0.75f * a[iy * cols + nx] + 0.25f * a[ny * cols + nx];
    out[y * 2 * cols + x] = 0.75f * vc + 0.25f * vn;
}

extern "C" __global__ void sift_blur(const float* in, float* out, const int rows, const int cols,
                                     const float* k, const int klen, const int axis)
{
    // 1-D correlation along rows (axis 0) or columns (axis 1), mirror edges (d c b | a b c d | c b a)
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    const int r = klen / 2, n = axis == 0 ? rows : cols, c0 = axis == 0 ? y : x;
    float s = 0.0f;
    for (int j = 0; j < klen; j++) {
        int p = c0 + j - r;
        if (n == 1) p = 0;
        else while (p < 0 || p >= n) p = p < 0 ? -p : 2 * n - 2 - p;
        s += k[j] * (axis == 0 ? in[p * cols + x] : in[y * cols + p]);
    }
    out[y * cols + x] = s;
}
"""


def _blur_cuda(xp, img, k):
    img = xp.ascontiguousarray(img)
    rows, cols = img.shape
    t = xp.empty_like(img)
    out = xp.empty_like(img)
    g = ((cols + 31) // 32, (rows + 7) // 8)
    kern = _kernel(xp, "blur")
    kern(g, (32, 8), (img, t, np.int32(rows), np.int32(cols), k, np.int32(k.size), np.int32(0)))
    kern(g, (32, 8), (t, out, np.int32(rows), np.int32(cols), k, np.int32(k.size), np.int32(1)))
    return out


def _up2_cuda(xp, a):
    a = xp.ascontiguousarray(a)
    rows, cols = a.shape
    out = xp.empty((2 * rows, 2 * cols), xp.float32)
    _kernel(xp, "up2")(((2 * cols + 31) // 32, (2 * rows + 7) // 8), (32, 8), (a, out, np.int32(rows), np.int32(cols)))
    return out


def _pyramid(xp, ndi, gray):
    K = _kernels(xp)
    img = xp.asarray(gray, dtype=xp.float32) * (1.0 / 255.0)
    if xp is not np and USE_CUDA_KERNELS[0]:
        blur = lambda im, k: _blur_cuda(xp, im, k)   # noqa: E731
        base = blur(_up2_cuda(xp, img), K.base)
    else:
        blur = lambda im, k: _blur(ndi, im, k)       # noqa: E731
        base = blur(_up2(xp, img), K.base)
    n_oct = max(1, int(round(math.log2(min(gray.shape)) - 2)) + 1)
    gpyr, dpyr = [], []
    cur = base
    for o in range(n_oct):
        if min(cur.shape) < 2 * BORDER + 4:
            break
        layers = [cur]
        for i in range(1, N_LAYERS + 3):
            layers.append(blur(layers[-1], K.layer[i]))
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


# ---------------------------------------------------------------------------------------------
# Everything after the pyramid works on "atlases": all octaves' DoG stacks (and gradient images)
# packed into one flat array, with per-keypoint offsets. That lets the whole photo be handled in
# a few dozen large array operations instead of thousands of small per-octave ones, which is what
# makes the GPU worthwhile (each operation costs ~10-20 us of Python/launch overhead).

USE_CUDA_KERNELS = [True]   # hand-written CUDA kernels where array operations are slow on GPUs
ORI_R_MAX = int(round(ORI_RADIUS * SIGMA * 2.0 ** ((N_LAYERS + 0.5) / N_LAYERS)))   # 16


class _Atlas:
    def __init__(self, xp, gpyr, dpyr):
        self.xp = xp
        rows = [D.shape[1] for D in dpyr]
        cols = [D.shape[2] for D in dpyr]
        dsz = [D.size for D in dpyr]
        self.rows = xp.asarray(rows, xp.int64)
        self.cols = xp.asarray(cols, xp.int64)
        self.rows_h = np.array(rows, np.int64)
        self.cols_h = np.array(cols, np.int64)
        self.dbase_h = np.r_[0, np.cumsum(dsz)[:-1]].astype(np.int64)
        self.dbase = xp.asarray(self.dbase_h)
        self.DF = xp.concatenate([D.ravel() for D in dpyr])
        # gradient magnitude / angle of gauss layers 1..N_LAYERS per octave
        gsz = [N_LAYERS * r * c for r, c in zip(rows, cols)]
        self.gbase_h = np.r_[0, np.cumsum(gsz)[:-1]].astype(np.int64)
        self.gbase = xp.asarray(self.gbase_h)
        if xp is not np and USE_CUDA_KERNELS[0]:
            self.MAG = xp.empty(int(sum(gsz)), xp.float32)
            self.ANG = xp.empty(int(sum(gsz)), xp.float32)
            k = _kernel(xp, "grad")
            for o, layers in enumerate(gpyr):
                R, C = rows[o], cols[o]
                for li in range(1, N_LAYERS + 1):
                    a = int(self.gbase_h[o]) + (li - 1) * R * C
                    G = xp.ascontiguousarray(layers[li], dtype=xp.float32)
                    n = R * C
                    k(((n + 255) // 256,), (256,), (G, np.int32(R), np.int32(C), self.MAG[a:a + n], self.ANG[a:a + n]))
            return
        mags, angs = [], []
        for o, layers in enumerate(gpyr):
            G = xp.stack(layers[1:N_LAYERS + 1])
            dx = xp.zeros_like(G)
            dy = xp.zeros_like(G)
            dx[:, 1:-1, 1:-1] = G[:, 1:-1, 2:] - G[:, 1:-1, :-2]
            dy[:, 1:-1, 1:-1] = G[:, :-2, 1:-1] - G[:, 2:, 1:-1]
            mags.append(xp.sqrt(dx * dx + dy * dy).ravel())
            a = xp.degrees(xp.arctan2(dy, dx))
            angs.append(xp.where(a < 0, a + 360.0, a).ravel())
        self.MAG = xp.concatenate(mags)
        self.ANG = xp.concatenate(angs)

    def gidx(self, o, li, y, x, C, RC):
        """Flat index into MAG/ANG for octave o, layer li (1-based), pixel (y, x)."""
        return self.gbase[o] + (li - 1) * RC + y * C + x


def _derivs_flat(xp, DF, k, C, RC):
    v = DF[k]
    xp1, xm1 = DF[k + 1], DF[k - 1]
    yp1, ym1 = DF[k + C], DF[k - C]
    sp1, sm1 = DF[k + RC], DF[k - RC]
    dx = (xp1 - xm1) * 0.5
    dy = (yp1 - ym1) * 0.5
    ds = (sp1 - sm1) * 0.5
    dxx = xp1 + xm1 - 2 * v
    dyy = yp1 + ym1 - 2 * v
    dss = sp1 + sm1 - 2 * v
    dxy = (DF[k + C + 1] - DF[k + C - 1] - DF[k - C + 1] + DF[k - C - 1]) * 0.25
    dxs = (DF[k + RC + 1] - DF[k + RC - 1] - DF[k - RC + 1] + DF[k - RC - 1]) * 0.25
    dys = (DF[k + RC + C] - DF[k + RC - C] - DF[k - RC + C] + DF[k - RC - C]) * 0.25
    g = xp.stack([dx, dy, ds], 1)
    H = xp.stack([xp.stack([dxx, dxy, dxs], 1), xp.stack([dxy, dyy, dys], 1), xp.stack([dxs, dys, dss], 1)], 1)
    return v, g, H, dxx, dyy, dxy


def _extrema(xp, ndi, dpyr, contrast, A=None):
    """Flat DoG-atlas indices of 3x3x3 extrema in layers 1..N_LAYERS (one sync for all octaves)."""
    pthr = 0.8 * 0.5 * contrast / N_LAYERS
    if xp is not np and USE_CUDA_KERNELS[0] and A is not None:
        mask = xp.zeros(A.DF.size, xp.uint8)
        k = _kernel(xp, "extrema")
        for o in range(len(A.rows_h)):
            R, C = int(A.rows_h[o]), int(A.cols_h[o])
            n = N_LAYERS * (R - 2 * BORDER) * (C - 2 * BORDER)
            if n > 0:
                k(((n + 255) // 256,), (256,), (A.DF, np.int64(A.dbase_h[o]), np.int32(R), np.int32(C),
                                                 np.float32(pthr), mask, np.int32(n)))
        return xp.nonzero(mask)[0]
    masks = []
    for D in dpyr:
        rows, cols = D.shape[1:]
        mx = ndi.maximum_filter(D, size=3, mode="nearest")
        mn = ndi.minimum_filter(D, size=3, mode="nearest")
        ext = ((D >= mx) & (D > pthr)) | ((D <= mn) & (D < -pthr))
        ext[0] = False
        ext[-1] = False
        ext[:, :BORDER] = False
        ext[:, rows - BORDER:] = False
        ext[:, :, :BORDER] = False
        ext[:, :, cols - BORDER:] = False
        masks.append(ext.ravel())
    return xp.nonzero(xp.concatenate(masks))[0]


def _refine(xp, A, F, contrast, edge):
    """Sub-pixel refinement + contrast and edge tests for all candidates of all octaves at once."""
    if xp is not np and USE_CUDA_KERNELS[0]:
        return _refine_cuda(xp, A, F, contrast, edge)
    o = (xp.searchsorted(A.dbase, F, side="right") - 1).astype(xp.int64)
    R = A.rows[o]
    C = A.cols[o]
    RC = R * C
    loc = F - A.dbase[o]
    l = loc // RC
    rem = loc - l * RC
    r = rem // C
    c = rem - r * C
    alive = xp.ones(F.shape, bool)
    done = xp.zeros(F.shape, bool)
    X = xp.zeros((F.size, 3), xp.float32)
    for _ in range(MAX_INTERP):
        k = A.dbase[o] + l * RC + r * C + c
        _, g, H, _, _, _ = _derivs_flat(xp, A.DF, k, C, RC)
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
        step = xp.rint(xp.where(move[:, None], x, 0)).astype(xp.int64)
        c = c + step[:, 0]
        r = r + step[:, 1]
        l = l + step[:, 2]
        inside = (l >= 1) & (l <= N_LAYERS) & (c >= BORDER) & (c < C - BORDER) & (r >= BORDER) & (r < R - BORDER)
        alive &= inside
        l = xp.clip(l, 1, N_LAYERS)
        r = xp.clip(r, BORDER, R - BORDER - 1)
        c = xp.clip(c, BORDER, C - BORDER - 1)
    keep = alive & done
    k = A.dbase[o] + l * RC + r * C + c
    v, g, H, dxx, dyy, dxy = _derivs_flat(xp, A.DF, k, C, RC)
    contr = v + 0.5 * (g * X).sum(1)
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    ok = keep & (xp.abs(contr) * N_LAYERS >= contrast) & (det > 0) & (tr * tr * edge < (edge + 1) ** 2 * det)
    s = xp.nonzero(ok)[0]
    return o[s], l[s], r[s], c[s], X[s], xp.abs(contr[s])


def _orientations_flat(xp, A, o, li, r, c, scl, chunk=8192):
    """Dominant orientations for all keypoints (any octave/layer). Returns (kp index, angle)."""
    if xp is not np and USE_CUDA_KERNELS[0]:
        n = r.size
        hist = xp.empty((n, ORI_BINS), xp.float32)
        gofs = A.gbase[o] + (li - 1) * A.rows[o] * A.cols[o]
        _kernel(xp, "ori")((n,), (128,), (A.MAG, A.ANG, gofs.astype(xp.int64), A.rows[o].astype(xp.int32),
                                          A.cols[o].astype(xp.int32), r.astype(xp.int32), c.astype(xp.int32),
                                          scl.astype(xp.float32), hist, np.int32(n)))
        return _ori_peaks(xp, hist, 0)
    R = ORI_R_MAX
    off = xp.arange(-R, R + 1, dtype=xp.int64)
    oi, oj = xp.meshgrid(off, off, indexing="ij")
    oi = oi.ravel()[None]
    oj = oj.ravel()[None]
    d2 = (oi * oi + oj * oj).astype(xp.float32)
    kps, angs = [], []
    n_all = r.size
    for s0 in range(0, n_all, chunk):
        sl = slice(s0, s0 + chunk)
        oo, ll, rr, cc, ss = o[sl], li[sl], r[sl], c[sl], scl[sl]
        n = rr.size
        rows = A.rows[oo][:, None]
        cols = A.cols[oo][:, None]
        rad = xp.rint(ORI_RADIUS * ss).astype(xp.int64)[:, None]
        y = rr[:, None] + oi
        x = cc[:, None] + oj
        valid = (xp.abs(oi) <= rad) & (xp.abs(oj) <= rad) & (y > 0) & (y < rows - 1) & (x > 0) & (x < cols - 1)
        y = xp.clip(y, 0, rows - 1)
        x = xp.clip(x, 0, cols - 1)
        gi = A.gbase[oo][:, None] + (ll[:, None] - 1) * rows * cols + y * cols + x
        sw = ORI_SIG * ss
        w = xp.exp(-d2 / (2.0 * sw[:, None] ** 2))
        m = A.MAG[gi] * w * valid
        b = xp.rint(A.ANG[gi] * (ORI_BINS / 360.0)).astype(xp.int64) % ORI_BINS
        idx = (xp.arange(n, dtype=xp.int64)[:, None] * ORI_BINS + b).ravel()
        hist = _scatter_sum(xp, idx, m.ravel(), n * ORI_BINS).reshape(n, ORI_BINS)
        kp, angle = _ori_peaks(xp, hist, s0)
        kps.append(kp)
        angs.append(angle)
    return xp.concatenate(kps), xp.concatenate(angs)


def _ori_peaks(xp, hist, s0):
    """Smooth 36-bin orientation histograms and return (kp index + s0, peak angle) pairs."""
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
    return kp + s0, angle.astype(xp.float32)


def detect(gray, gpu=False, max_kp=12000, contrast=0.015, edge=12.0, timings=None):
    """Detect oriented keypoints. Returns (dict of host arrays: x, y (input-image pixels), size,
    angle, response, octave, layer; state for describe())."""
    import time as _t
    xp, ndi = backend(gpu)

    def lap(name, t0):
        if timings is not None:
            if xp is not np:
                xp.cuda.Stream.null.synchronize()
            timings[name] = timings.get(name, 0.0) + _t.perf_counter() - t0
        return _t.perf_counter()

    t = _t.perf_counter()
    gpyr, dpyr = _pyramid(xp, ndi, gray)
    t = lap("pyramid", t)
    A = _Atlas(xp, gpyr, dpyr)
    t = lap("atlas", t)
    F = _extrema(xp, ndi, dpyr, contrast, A)
    t = lap("extrema", t)
    empty = {k: np.zeros(0, np.float32) for k in ("x", "y", "size", "angle", "response", "octave", "layer")}
    if F.size == 0:
        return empty, (xp, A)
    o, l, r, c, X, resp = _refine(xp, A, F, contrast, edge)
    t = lap("refine", t)
    if o.size == 0:
        return empty, (xp, A)
    scl = SIGMA * 2.0 ** ((l.astype(xp.float32) + X[:, 2]) / N_LAYERS)    # in octave pixels
    kp, angle = _orientations_flat(xp, A, o, l, r, c, scl)
    t = lap("orientation", t)
    f = (2.0 ** o.astype(xp.float32)) * 0.5
    x = (c[kp].astype(xp.float32) + X[kp, 0]) * f[kp]
    y = (r[kp].astype(xp.float32) + X[kp, 1]) * f[kp]
    size = scl[kp] * 2 * f[kp]
    response = resp[kp]
    # strongest first (ties: by octave/layer/position order, as before); keep at most max_kp
    order = xp.argsort(-response, kind="stable") if xp is np else xp.argsort(-response)
    order = order[:max_kp]
    host = {"x": x[order], "y": y[order], "size": size[order], "angle": angle[order], "response": response[order],
            "octave": o[kp][order].astype(xp.int32), "layer": l[kp][order].astype(xp.int32)}
    host = {k: to_host(xp, v) for k, v in host.items()}
    lap("select", t)
    return host, (xp, A)


def describe(state, kps, sel, budget=2.5e6, timings=None):
    """128-d SIFT descriptors (float, OpenCV scaling: L2 norm 512) for keypoints kps[sel]."""
    import time as _t
    xp, A = state
    t0 = _t.perf_counter()
    sel = np.asarray(sel, np.int64)
    n_all = len(sel)
    desc = np.zeros((n_all, D_WIDTH * D_WIDTH * D_BINS), np.float32)
    if n_all == 0:
        return desc
    o = kps["octave"][sel].astype(np.int64)
    li = kps["layer"][sel].astype(np.int64)
    f = 1.0 / (2.0 ** o * 0.5)
    px = kps["x"][sel] * f
    py = kps["y"][sel] * f
    scl = kps["size"][sel] * f * 0.5
    ang = 360.0 - kps["angle"][sel]
    ang = np.where(np.abs(ang - 360.0) < 1e-4, 0.0, ang)
    rows = A.rows_h[o]
    cols = A.cols_h[o]
    rad = np.rint(D_SCL * scl * 1.4142135623730951 * (D_WIDTH + 1) * 0.5).astype(np.int64)
    rad = np.minimum(rad, np.sqrt(rows * rows + cols * cols).astype(np.int64))
    if xp is not np and USE_CUDA_KERNELS[0]:
        desc[:] = to_host(xp, _descr_cuda(xp, A, o, li, px, py, scl, ang, rad))
        if timings is not None:
            xp.cuda.Stream.null.synchronize()
            timings["describe"] = timings.get("describe", 0.0) + _t.perf_counter() - t0
        return desc
    order = np.argsort(rad, kind="stable")
    outs = []
    s0 = 0
    while s0 < n_all:
        rmax = int(rad[order[min(n_all - 1, s0)]])
        n = max(32, int(budget / (2 * rmax + 1) ** 2))
        part = order[s0:s0 + n]
        rmax = int(rad[part].max())
        outs.append((part, _descr_flat(xp, A, xp.asarray(o[part]), xp.asarray(li[part]), xp.asarray(px[part]),
                                       xp.asarray(py[part]), xp.asarray(scl[part]), xp.asarray(ang[part]),
                                       xp.asarray(rad[part]), rmax)))
        s0 += n
    allp = np.concatenate([p for p, _ in outs])
    alld = to_host(xp, xp.concatenate([d for _, d in outs]))
    desc[allp] = alld
    if timings is not None:
        timings["describe"] = timings.get("describe", 0.0) + _t.perf_counter() - t0
    return desc


_DESC_SRC = r"""
extern "C" __global__ void sift_desc(const float* MAG, const float* ANG, const long long* gofs,
        const int* rows, const int* cols, const float* px, const float* py, const float* scl,
        const float* ori, const int* rad, float* out, const int n)
{
    // one block per keypoint; 4x4 spatial x 8 orientation bins (+ borders) histogram in shared memory
    const int k = blockIdx.x;
    if (k >= n) return;
    __shared__ float hist[360];
    for (int i = threadIdx.x; i < 360; i += blockDim.x) hist[i] = 0.f;
    __syncthreads();
    const float hist_width = 3.0f * scl[k];
    const float orik = ori[k];
    const float t = orik * 0.017453292519943295f;
    const float cos_t = cosf(t) / hist_width;
    const float sin_t = sinf(t) / hist_width;
    const int R = rad[k];
    const int W = 2 * R + 1;
    const int pr = (int)rintf(py[k]);
    const int pc = (int)rintf(px[k]);
    const int nr = rows[k];
    const int nc = cols[k];
    const long long base = gofs[k];
    for (int s = threadIdx.x; s < W * W; s += blockDim.x) {
        const int i = s / W - R;
        const int j = s % W - R;
        const float c_rot = j * cos_t - i * sin_t;
        const float r_rot = j * sin_t + i * cos_t;
        const float rbin = r_rot + 1.5f;
        const float cbin = c_rot + 1.5f;
        const int y = pr + i;
        const int x = pc + j;
        if (rbin > -1.f && rbin < 4.f && cbin > -1.f && cbin < 4.f && y > 0 && y < nr - 1 && x > 0 && x < nc - 1) {
            const long long gi = base + (long long)y * nc + x;
            const float w = expf((c_rot * c_rot + r_rot * r_rot) * (-1.0f / 8.0f));
            const float m = MAG[gi] * w;
            const float obin = (ANG[gi] - orik) * (8.0f / 360.0f);
            const float fr0 = floorf(rbin), fc0 = floorf(cbin), fo0 = floorf(obin);
            const float rb = rbin - fr0, cb = cbin - fc0, ob = obin - fo0;
            const int r0 = (int)fr0, c0 = (int)fc0;
            int o0 = (int)fo0;
            if (o0 < 0) o0 += 8;
            if (o0 >= 8) o0 -= 8;
            const float v_r1 = m * rb, v_r0 = m - v_r1;
            const float v_rc11 = v_r1 * cb, v_rc10 = v_r1 - v_rc11;
            const float v_rc01 = v_r0 * cb, v_rc00 = v_r0 - v_rc01;
            const float v111 = v_rc11 * ob, v110 = v_rc11 - v111;
            const float v101 = v_rc10 * ob, v100 = v_rc10 - v101;
            const float v011 = v_rc01 * ob, v010 = v_rc01 - v011;
            const float v001 = v_rc00 * ob, v000 = v_rc00 - v001;
            const int idx = ((r0 + 1) * 6 + c0 + 1) * 10 + o0;
            atomicAdd(&hist[idx], v000);
            atomicAdd(&hist[idx + 1], v001);
            atomicAdd(&hist[idx + 10], v010);
            atomicAdd(&hist[idx + 11], v011);
            atomicAdd(&hist[idx + 60], v100);
            atomicAdd(&hist[idx + 61], v101);
            atomicAdd(&hist[idx + 70], v110);
            atomicAdd(&hist[idx + 71], v111);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 360; i += blockDim.x) out[(long long)k * 360 + i] = hist[i];
}
"""
_KERNELS = {}


def _kernel(xp, name):
    if name not in _KERNELS:
        src, fn = {"desc": (_DESC_SRC, "sift_desc"), "grad": (_GRAD_SRC, "sift_grad"),
                   "blur": (_PYR_SRC, "sift_blur"), "up2": (_PYR_SRC, "sift_up2"),
                   "extrema": (_EXTREMA_SRC, "sift_extrema"), "refine": (_REFINE_SRC, "sift_refine"),
                   "ori": (_ORI_SRC, "sift_ori")}[name]
        # no fused multiply-add, so results follow the array version's rounding
        _KERNELS[name] = xp.RawKernel(src, fn, options=("--fmad=false",))
    return _KERNELS[name]


def _desc_kernel(xp):
    return _kernel(xp, "desc")


_CONSTS = "#define N_LAYERS %d\n#define BORDER %d\n#define MAX_INTERP %d\n#define ORI_BINS %d\n" \
          "#define ORI_RADIUS %.9ff\n#define ORI_SIG %.9ff\n#define ORI_R_MAX %d\n" \
          % (N_LAYERS, BORDER, MAX_INTERP, ORI_BINS, ORI_RADIUS, ORI_SIG, ORI_R_MAX)

_GRAD_SRC = _CONSTS + r"""
extern "C" __global__ void sift_grad(const float* G, const int rows, const int cols, float* MAG, float* ANG)
{
    // gradient magnitude and angle (degrees, 0..360) of one gauss layer; zero on the 1-pixel border
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= rows * cols) return;
    const int r = k / cols, c = k - (k / cols) * cols;
    float dx = 0.f, dy = 0.f;
    if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
        dx = G[k + 1] - G[k - 1];
        dy = G[k - cols] - G[k + cols];
    }
    MAG[k] = sqrtf(dx * dx + dy * dy);
    float a = atan2f(dy, dx) * 57.29577951308232f;
    ANG[k] = a < 0.f ? a + 360.f : a;
}
"""

_EXTREMA_SRC = _CONSTS + r"""
extern "C" __global__ void sift_extrema(const float* DF, const long long dbase, const int rows, const int cols,
        const float pthr, unsigned char* mask, const int n)
{
    // one thread per interior voxel of DoG layers 1..N_LAYERS of one octave; marks 3x3x3 extrema
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int iw = cols - 2 * BORDER, ih = rows - 2 * BORDER;
    const int l = t / (iw * ih) + 1;
    const int rem = t - (l - 1) * (iw * ih);
    const int r = rem / iw + BORDER;
    const int c = rem - (rem / iw) * iw + BORDER;
    const long long RC = (long long)rows * cols;
    const long long k = dbase + l * RC + (long long)r * cols + c;
    const float v = DF[k];
    bool mx = v > pthr, mn = v < -pthr;
    if (!mx && !mn) return;
    for (int dl = -1; dl <= 1; dl++)
        for (int dr = -1; dr <= 1; dr++)
            for (int dc = -1; dc <= 1; dc++) {
                const float u = DF[k + dl * RC + dr * cols + dc];
                if (u > v) mx = false;
                if (u < v) mn = false;
            }
    if (mx || mn) mask[k] = 1;
}
"""

_REFINE_SRC = _CONSTS + r"""
extern "C" __global__ void sift_refine(const float* DF, const long long* dbase, const long long* rows,
        const long long* cols, const int n_oct, const long long* F, const int n, const float contrast,
        const float edge, int* oo, int* ol, int* orr, int* oc, float* oX, float* oresp, unsigned char* ok)
{
    // one thread per candidate: sub-pixel fit (up to MAX_INTERP moves), then contrast and edge tests
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ok[i] = 0;
    const long long f = F[i];
    int o = 0;
    while (o + 1 < n_oct && f >= dbase[o + 1]) o++;
    const long long base = dbase[o];
    const int R = (int)rows[o], C = (int)cols[o];
    const long long RC = (long long)R * C;
    const long long loc = f - base;
    int l = (int)(loc / RC);
    const long long rem = loc - l * RC;
    int r = (int)(rem / C);
    int c = (int)(rem - (long long)r * C);
    bool good = false;
    float X0 = 0.f, X1 = 0.f, X2 = 0.f;
    float v, dx, dy, ds, dxx, dyy, dss, dxy, dxs, dys;
#define DERIVS() { \
        const long long k = base + l * RC + (long long)r * C + c; \
        v = DF[k]; \
        const float xp1 = DF[k + 1], xm1 = DF[k - 1], yp1 = DF[k + C], ym1 = DF[k - C]; \
        const float sp1 = DF[k + RC], sm1 = DF[k - RC]; \
        dx = (xp1 - xm1) * 0.5f; dy = (yp1 - ym1) * 0.5f; ds = (sp1 - sm1) * 0.5f; \
        dxx = xp1 + xm1 - 2.f * v; dyy = yp1 + ym1 - 2.f * v; dss = sp1 + sm1 - 2.f * v; \
        dxy = (DF[k + C + 1] - DF[k + C - 1] - DF[k - C + 1] + DF[k - C - 1]) * 0.25f; \
        dxs = (DF[k + RC + 1] - DF[k + RC - 1] - DF[k - RC + 1] + DF[k - RC - 1]) * 0.25f; \
        dys = (DF[k + RC + C] - DF[k + RC - C] - DF[k - RC + C] + DF[k - RC - C]) * 0.25f; }
    for (int it = 0; it < MAX_INTERP; it++) {
        DERIVS();
        // H = [[dxx dxy dxs] [dxy dyy dys] [dxs dys dss]], solve H x = -g by the adjugate
        const float a = dxx, b = dxy, cc = dxs, d = dxy, e = dyy, ff = dys, g = dxs, h = dys, ii = dss;
        const float A0 = e * ii - ff * h, B0 = -(d * ii - ff * g), C0 = d * h - e * g;
        const float det = a * A0 + b * B0 + cc * C0;
        if (!(fabsf(det) > 1e-20f)) break;
        const float i00 = A0 / det, i01 = -(b * ii - cc * h) / det, i02 = (b * ff - cc * e) / det;
        const float i10 = B0 / det, i11 = (a * ii - cc * g) / det, i12 = -(a * ff - cc * d) / det;
        const float i20 = C0 / det, i21 = -(a * h - b * g) / det, i22 = (a * e - b * d) / det;
        const float x0 = -((i00 * dx + i01 * dy) + i02 * ds);
        const float x1 = -((i10 * dx + i11 * dy) + i12 * ds);
        const float x2 = -((i20 * dx + i21 * dy) + i22 * ds);
        if (fabsf(x0) < 0.5f && fabsf(x1) < 0.5f && fabsf(x2) < 0.5f) {
            good = true; X0 = x0; X1 = x1; X2 = x2;
            break;
        }
        if (!(fabsf(x0) <= 1e6f && fabsf(x1) <= 1e6f && fabsf(x2) <= 1e6f)) break;   // also NaN
        c += (int)rintf(x0);
        r += (int)rintf(x1);
        l += (int)rintf(x2);
        if (l < 1 || l > N_LAYERS || c < BORDER || c >= C - BORDER || r < BORDER || r >= R - BORDER) break;
    }
    if (!good) return;
    DERIVS();
    const float contr = v + 0.5f * ((dx * X0 + dy * X1) + ds * X2);
    const float tr = dxx + dyy;
    const float dt = dxx * dyy - dxy * dxy;
    if (fabsf(contr) * N_LAYERS >= contrast && dt > 0.f && tr * tr * edge < (edge + 1.f) * (edge + 1.f) * dt) {
        ok[i] = 1;
        oo[i] = o; ol[i] = l; orr[i] = r; oc[i] = c;
        oX[3 * i] = X0; oX[3 * i + 1] = X1; oX[3 * i + 2] = X2;
        oresp[i] = fabsf(contr);
    }
#undef DERIVS
}
"""

_ORI_SRC = _CONSTS + r"""
extern "C" __global__ void sift_ori(const float* MAG, const float* ANG, const long long* gofs, const int* rows,
        const int* cols, const int* kr, const int* kc, const float* scl, float* out, const int n)
{
    // one block per keypoint: gaussian-weighted 36-bin gradient orientation histogram (unsmoothed)
    const int k = blockIdx.x;
    if (k >= n) return;
    __shared__ float hist[ORI_BINS];
    for (int i = threadIdx.x; i < ORI_BINS; i += blockDim.x) hist[i] = 0.f;
    __syncthreads();
    const float sw = ORI_SIG * scl[k];
        const int rad = (int)rintf(ORI_RADIUS * scl[k]);
    const int R = rad < ORI_R_MAX ? rad : ORI_R_MAX;
    const int W = 2 * R + 1;
    const int nr = rows[k], nc = cols[k], pr = kr[k], pc = kc[k];
    const long long base = gofs[k];
    for (int s = threadIdx.x; s < W * W; s += blockDim.x) {
        const int i = s / W - R;
        const int j = s - (s / W) * W - R;
        const int y = pr + i, x = pc + j;
        if (y > 0 && y < nr - 1 && x > 0 && x < nc - 1) {
            const long long gi = base + (long long)y * nc + x;
            const float w = expf(-(float)(i * i + j * j) / (2.f * sw * sw));
            int b = (int)rintf(ANG[gi] * (ORI_BINS / 360.f));
            b = ((b % ORI_BINS) + ORI_BINS) % ORI_BINS;
            atomicAdd(&hist[b], MAG[gi] * w);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < ORI_BINS; i += blockDim.x) out[(long long)k * ORI_BINS + i] = hist[i];
}
"""


def _refine_cuda(xp, A, F, contrast, edge):
    n = int(F.size)
    oo, ol, orr, oc = (xp.empty(n, xp.int32) for _ in range(4))
    X = xp.empty((n, 3), xp.float32)
    resp = xp.empty(n, xp.float32)
    ok = xp.empty(n, xp.uint8)
    _kernel(xp, "refine")(((n + 127) // 128,), (128,),
                          (A.DF, A.dbase, A.rows, A.cols, np.int32(len(A.rows_h)), F.astype(xp.int64), np.int32(n),
                           np.float32(contrast), np.float32(edge), oo, ol, orr, oc, X, resp, ok))
    s = xp.nonzero(ok)[0]
    return (oo[s].astype(xp.int64), ol[s].astype(xp.int64), orr[s].astype(xp.int64), oc[s].astype(xp.int64),
            X[s], resp[s])


def _descr_cuda(xp, A, o, li, px, py, scl, ori, rad):
    """Descriptors with one CUDA block per keypoint (shared-memory histogram). Inputs are host arrays."""
    n = len(px)
    gofs = (A.gbase_h[o] + (li - 1) * A.rows_h[o] * A.cols_h[o]).astype(np.int64)
    args = [xp.asarray(gofs), xp.asarray(A.rows_h[o].astype(np.int32)), xp.asarray(A.cols_h[o].astype(np.int32)),
            xp.asarray(px.astype(np.float32)), xp.asarray(py.astype(np.float32)), xp.asarray(scl.astype(np.float32)),
            xp.asarray(ori.astype(np.float32)), xp.asarray(rad.astype(np.int32))]
    out = xp.zeros((n, 360), xp.float32)
    _desc_kernel(xp)((n,), (128,), (A.MAG, A.ANG, *args, out, np.int32(n)))
    d, nb = D_WIDTH, D_BINS
    hist = out.reshape(n, d + 2, d + 2, nb + 2)
    hist[..., 0] += hist[..., nb]
    hist[..., 1] += hist[..., nb + 1]
    dst = hist[:, 1:d + 1, 1:d + 1, :nb].reshape(n, d * d * nb)
    nrm = xp.sqrt((dst * dst).sum(1, keepdims=True))
    dst = xp.minimum(dst, nrm * D_MAG_THR)
    nrm = xp.sqrt((dst * dst).sum(1, keepdims=True))
    return dst * (512.0 / xp.maximum(nrm, 1e-7))


def _descr_flat(xp, A, o, li, px, py, scl, ori, rad, R):
    d, nb = D_WIDTH, D_BINS
    n = px.size
    rows = A.rows[o][:, None]
    cols = A.cols[o][:, None]
    hist_width = D_SCL * scl
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
    pr = xp.rint(py).astype(xp.int64)[:, None]
    pc = xp.rint(px).astype(xp.int64)[:, None]
    y = pr + oi.astype(xp.int64)
    x = pc + oj.astype(xp.int64)
    radc = rad[:, None]
    ok = (xp.abs(oi) <= radc) & (xp.abs(oj) <= radc) & (rbin > -1) & (rbin < d) & (cbin > -1) & (cbin < d) \
        & (y > 0) & (y < rows - 1) & (x > 0) & (x < cols - 1)
    y = xp.clip(y, 0, rows - 1)
    x = xp.clip(x, 0, cols - 1)
    gi = A.gbase[o][:, None] + (li[:, None] - 1) * rows * cols + y * cols + x
    w = xp.exp((c_rot * c_rot + r_rot * r_rot) * (-1.0 / (d * d * 0.5)))
    m = A.MAG[gi] * w * ok
    obin = (A.ANG[gi] - ori[:, None]) * (nb / 360.0)
    r0 = xp.floor(rbin)
    c0 = xp.floor(cbin)
    o0 = xp.floor(obin)
    rb = rbin - r0
    cb = cbin - c0
    ob = obin - o0
    o0 = o0.astype(xp.int64)
    o0 = xp.where(o0 < 0, o0 + nb, o0)
    o0 = xp.where(o0 >= nb, o0 - nb, o0)
    r0 = r0.astype(xp.int64)
    c0 = c0.astype(xp.int64)
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
