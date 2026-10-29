"""Orthomosaic blocks blended on an NVIDIA GPU (CuPy).

The geometry (which photos cover which pixels, blend weights, photo coordinates on a coarse grid)
comes from ortho.region_plan, exactly as for the CPU renderer. Photos are decoded on the CPU by a
few threads; only the part of each photo a block needs is uploaded. Warping, blending and the
final normalisation run on the card. Before trusting the card, the worker renders its first block
on both the CPU and the GPU and compares them.
"""
import math
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import ortho

_T = {"gpu_blocks": 0, "gpu_render": 0.0, "gpu_decode": 0.0, "gpu_wait": 0.0, "gpu_plan": 0.0}


class _PhotoCache:
    """Thread-safe LRU of decoded photos; decoding runs on a thread pool and is started ahead."""

    def __init__(self, threads, max_items):
        self.ex = ThreadPoolExecutor(max_workers=threads)
        self.lock = threading.Lock()
        self.items = OrderedDict()
        self.max_items = max_items

    def _load(self, ci):
        from .features import read_file, decode
        c = ortho._RCTX
        t0 = time.perf_counter()
        img = decode(read_file(c["cams"][ci]["path"]), c["reduce"], color=True)
        _T["gpu_decode"] += time.perf_counter() - t0
        return img

    def get(self, ci):
        with self.lock:
            f = self.items.get(ci)
            if f is not None:
                self.items.move_to_end(ci)
                return f
            f = self.ex.submit(self._load, ci)
            self.items[ci] = f
            while len(self.items) > self.max_items:
                self.items.popitem(last=False)
            return f


def _interp_grid(cp, coarse, ys, xs):
    """Bilinear interpolation of coarse (k, ny, nx) at fractional rows ys and columns xs
    (edges replicated, like cv2.warpAffine with BORDER_REPLICATE). Returns (k, len(ys), len(xs))."""
    ny, nx = coarse.shape[1:]
    ys = cp.clip(ys, 0, ny - 1)
    xs = cp.clip(xs, 0, nx - 1)
    y0 = cp.minimum(cp.floor(ys).astype(cp.int32), max(ny - 2, 0))
    x0 = cp.minimum(cp.floor(xs).astype(cp.int32), max(nx - 2, 0))
    fy = (ys - y0)[None, :, None]
    fx = (xs - x0)[None, None, :]
    y1 = cp.minimum(y0 + 1, ny - 1)
    x1 = cp.minimum(x0 + 1, nx - 1)
    a = coarse[:, y0[:, None], x0[None, :]]
    b = coarse[:, y0[:, None], x1[None, :]]
    c = coarse[:, y1[:, None], x0[None, :]]
    d = coarse[:, y1[:, None], x1[None, :]]
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy


def render_region_gpu(cp, region, tiles, plan, cache):
    c = ortho._RCTX
    tx, ty, tw, th = region
    step = plan["step"]
    ncy, ncx = plan["ncy"], plan["ncx"]
    Wt, U, V = plan["Wt"], plan["U"], plan["V"]
    r = c["reduce"]
    acc = cp.zeros((th, tw, 3), cp.float32)
    wsum = cp.zeros((th, tw), cp.float32)
    futs = [(it, cache.get(it[1])) for it in plan["items"]]
    for (a, ci, r0, r1, c0, c1), fut in futs:
        i0 = r0 // step
        i1 = min(ncy - 1, -(-(r1 - 1) // step))
        j0 = c0 // step
        j1 = min(ncx - 1, -(-(c1 - 1) // step))
        if i1 <= i0:
            i0 = max(0, i1 - 1)
        if j1 <= j0:
            j0 = max(0, j1 - 1)
        wa = Wt[a].reshape(ncy, ncx)[i0:i1 + 1, j0:j1 + 1]
        ua = U[a].reshape(ncy, ncx)[i0:i1 + 1, j0:j1 + 1].astype(np.float64)
        va = V[a].reshape(ncy, ncx)[i0:i1 + 1, j0:j1 + 1].astype(np.float64)
        if r != 1:
            ua = (ua + 0.5) / r - 0.5
            va = (va + 0.5) / r - 0.5
        tw0 = time.perf_counter()
        img = fut.result()
        _T["gpu_wait"] += time.perf_counter() - tw0
        Hi, Wi = img.shape[:2]
        u0 = int(max(0, math.floor(float(ua.min())) - 2))
        u1 = int(min(Wi, math.ceil(float(ua.max())) + 3))
        v0 = int(max(0, math.floor(float(va.min())) - 2))
        v1 = int(min(Hi, math.ceil(float(va.max())) + 3))
        if u1 - u0 < 2 or v1 - v0 < 2:
            continue
        crop = cp.asarray(np.ascontiguousarray(img[v0:v1, u0:u1]))
        sub = cp.asarray(np.stack([wa, ua - u0, va - v0]).astype(np.float32))
        ys = cp.arange(r0, r1, dtype=cp.float32) / step - i0
        xs = cp.arange(c0, c1, dtype=cp.float32) / step - j0
        g3 = _interp_grid(cp, sub, ys, xs)
        w_f, uu, vv = g3[0], g3[1], g3[2]
        Hc, Wc = crop.shape[:2]
        uu = cp.clip(uu, 0, Wc - 1)
        vv = cp.clip(vv, 0, Hc - 1)
        x0 = cp.minimum(cp.floor(uu).astype(cp.int32), Wc - 2)
        y0 = cp.minimum(cp.floor(vv).astype(cp.int32), Hc - 2)
        fx = (uu - x0)[..., None]
        fy = (vv - y0)[..., None]
        p00 = crop[y0, x0].astype(cp.float32)
        p01 = crop[y0, x0 + 1].astype(cp.float32)
        p10 = crop[y0 + 1, x0].astype(cp.float32)
        p11 = crop[y0 + 1, x0 + 1].astype(cp.float32)
        top = p00 + (p01 - p00) * fx
        samp = top + (p10 + (p11 - p10) * fx - top) * fy
        g = c["cams"][ci]["gain"]
        gains = cp.asarray(np.array([g[2], g[1], g[0]], np.float32))     # BGR
        acc[r0:r1, c0:c1] += samp * (w_f[..., None] * gains)
        wsum[r0:r1, c0:c1] += w_f
    cov = cp.asarray(plan["covered"].reshape(1, ncy, ncx).astype(np.float32))
    cov_f = _interp_grid(cp, cov, cp.arange(th, dtype=cp.float32) / step, cp.arange(tw, dtype=cp.float32) / step)[0]
    valid = (wsum > 1e-4) & (cov_f >= 0.5)
    out = acc / cp.maximum(wsum, 1e-6)[..., None]
    out = cp.clip(cp.rint(out), 0, 255).astype(cp.uint8)[..., ::-1]           # to RGB
    out = out * valid[..., None].astype(cp.uint8)
    rgb = cp.asnumpy(cp.ascontiguousarray(out))
    alpha = cp.asnumpy((valid.astype(cp.uint8) * 255))
    return ortho.cut_tiles(region, tiles, rgb, alpha)


def _compare(res_gpu, res_cpu):
    """Mean/99th-percentile difference and alpha agreement between two renders of the same block."""
    diffs = []
    agree = []
    for g, c in zip(res_gpu, res_cpu):
        ag = g[2] > 0
        ac = c[2] > 0
        agree.append((ag == ac).mean())
        both = ag & ac
        if both.any():
            diffs.append(np.abs(g[1].astype(np.int16) - c[1].astype(np.int16))[both].ravel())
    d = np.concatenate(diffs) if diffs else np.zeros(1)
    return float(d.mean()), float(np.percentile(d, 99)), float(min(agree))


def gpu_render_worker_main(ctx_file, task_q, res_q, threads):
    """GPU render worker: pulls runs of tile blocks like the CPU workers."""
    try:
        ortho.init_render(ctx_file)
        cv2.setNumThreads(2)
        use_gpu = False
        cp = None
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            from .sysinfo import raise_priority
            raise_priority()
            use_gpu = True
        except Exception as e:  # noqa
            res_q.put(("gpu_off", "rendering: %s: %s" % (type(e).__name__, str(e)[:200])))
        cache = _PhotoCache(threads, max_items=4 * threads + 16)
        checked = False
        while True:
            seg = task_q.get()
            if seg is None:
                break
            t_plan = time.perf_counter()
            plans = []
            for group in seg:
                x0 = min(t[0] for t in group)
                y0 = min(t[1] for t in group)
                x1 = max(t[0] + t[2] for t in group)
                y1 = max(t[1] + t[3] for t in group)
                region = (x0, y0, x1 - x0, y1 - y0)
                plans.append((group, region, ortho.region_plan(region, group) if use_gpu else None))
            _T["gpu_plan"] += time.perf_counter() - t_plan
            # start decoding every photo of this run, in order
            if use_gpu:
                for _, _, pl in plans:
                    if pl:
                        for it in pl["items"]:
                            cache.get(it[1])
            for group, region, plan in plans:
                t0 = time.perf_counter()
                if not use_gpu:
                    res = ortho.render_region(region, group)
                elif plan is None:
                    res = [ortho._with_half(t, np.zeros((t[3], t[2], 3), np.uint8), np.zeros((t[3], t[2]), np.uint8))
                           for t in group]
                else:
                    try:
                        res = render_region_gpu(cp, region, group, plan, cache)
                        if not checked:
                            checked = True
                            ref = ortho.render_region(region, group)
                            mean_d, p99, agree = _compare(res, ref)
                            if mean_d > 0.75 or p99 > 6 or agree < 0.998:
                                use_gpu = False
                                res = ref
                                res_q.put(("gpu_off", "rendering: GPU result differs from the CPU (mean %.2f, "
                                           "99%% %.1f levels, coverage %.4f); rendering on the CPU" % (mean_d, p99, agree)))
                            else:
                                res_q.put(("gpu", "rendering on the GPU checked against the CPU: mean difference "
                                           "%.2f, 99%% within %.1f levels" % (mean_d, p99)))
                        _T["gpu_blocks"] += 1
                    except Exception as e:  # noqa
                        use_gpu = False
                        res_q.put(("gpu_off", "rendering failed on the GPU, continuing on the CPU: %s: %s"
                                   % (type(e).__name__, str(e)[:200])))
                        res = ortho.render_region(region, group)
                _T["gpu_render"] += time.perf_counter() - t0
                snap = dict(_T)
                for k in _T:
                    _T[k] = 0 if k == "gpu_blocks" else 0.0
                res_q.put(("ok", res, snap))
        cache.ex.shutdown(wait=False, cancel_futures=True)
    except Exception:
        res_q.put(("error", traceback.format_exc()))
