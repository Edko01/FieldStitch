"""Orthomosaic blocks blended on an NVIDIA graphics card.

Division of labour:
  * CPU render workers ("feeders") plan each block (which photos are drawn where, blend weights,
    photo positions on a coarse grid; ortho.region_plan) and decode just the window of each photo
    the block needs (ortho.prepare_item). They pack that into their own shared-memory slot and
    hand the block to the GPU process.
  * One GPU process does the per-pixel work: interpolating weights, colour corrections and photo
    positions from the coarse grid, sampling the photos, blending, the coverage edge and the 8-bit
    result, which it writes back into the feeder's slot. The feeder cuts the block into tiles
    (and their half-resolution copies) and sends them to the GeoTIFF writer as usual.

A feeder only hands a block to the card while the card has fewer than a few blocks waiting;
otherwise it renders the block itself. So the CPU and GPU share the work in whatever proportion
keeps both busy. If the card fails, every block it had is rendered again on the CPU.

The kernels reproduce OpenCV's arithmetic (fixed-point bilinear photo sampling, float bilinear
grid interpolation, round-half-even conversion), so GPU blocks match CPU blocks. The GPU process
checks that on a real block before it takes any work.
"""
import math
import os
import queue
import threading
import time
import traceback

import numpy as np

from . import ortho

MAX_PENDING_PER_THREAD = 3
SLOTS = 2

_SRC = r"""
__device__ __forceinline__ int reflect_idx(int p, int len)
{
    // OpenCV BORDER_REFLECT (fedcba|abcdefgh|hgfedcb)
    if (len == 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p - 1;
        else p = len - 1 - (p - len);
    }
    return p;
}

__device__ __forceinline__ float grid_lerp(const float* g, int gh, int gw, int nch, int ch,
                                           int y0, int x0, float w00, float w01, float w10, float w11)
{
    // bilinear from the coarse grid with BORDER_REPLICATE, OpenCV's summation order
    const int ya = min(max(y0, 0), gh - 1), yb = min(max(y0 + 1, 0), gh - 1);
    const int xa = min(max(x0, 0), gw - 1), xb = min(max(x0 + 1, 0), gw - 1);
    return g[(ya * gw + xa) * nch + ch] * w00 + g[(ya * gw + xb) * nch + ch] * w01
         + g[(yb * gw + xa) * nch + ch] * w10 + g[(yb * gw + xb) * nch + ch] * w11;
}

extern "C" __global__ void ps_acc_item(const float* coarse, const int gh, const int gw,
        const int ystart, const int xstart, const int step,
        const unsigned char* crop, const int ch, const int cw,
        const int r0, const int c0, const int h, const int w, const int tw,
        float* acc, float* wsum)
{
    // one thread per output pixel of the item's rectangle; adds weight x correction x sample
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= h || j >= w) return;
    // grid position of this pixel: (j + xstart) / step, in 1/32 steps like warpAffine
    const int gxq = ((j + xstart) * 32) / step, gyq = ((i + ystart) * 32) / step;
    const int x0 = gxq >> 5, y0 = gyq >> 5;
    const float ax = (float)(gxq & 31) * (1.0f / 32.0f), ay = (float)(gyq & 31) * (1.0f / 32.0f);
    const float w00 = (1.0f - ay) * (1.0f - ax), w01 = (1.0f - ay) * ax, w10 = ay * (1.0f - ax), w11 = ay * ax;
    const float wt = grid_lerp(coarse, gh, gw, 6, 3, y0, x0, w00, w01, w10, w11);
    if (!(wt > 0.0f)) return;
    const float u = grid_lerp(coarse, gh, gw, 6, 4, y0, x0, w00, w01, w10, w11);
    const float v = grid_lerp(coarse, gh, gw, 6, 5, y0, x0, w00, w01, w10, w11);
    // convertMaps(CV_16SC2): 1/32-pixel fixed point, round half to even
    const int iu = __float2int_rn(u * 32.0f), iv = __float2int_rn(v * 32.0f);
    const int sx = iu >> 5, sy = iv >> 5, fx = iu & 31, fy = iv & 31;
    // remap INTER_LINEAR on 8-bit: integer weights summing to 32768
    const int k00 = (32 - fx) * (32 - fy) * 32, k01 = fx * (32 - fy) * 32, k10 = (32 - fx) * fy * 32, k11 = fx * fy * 32;
    const int xa = reflect_idx(sx, cw), xb = reflect_idx(sx + 1, cw);
    const int ya = reflect_idx(sy, ch), yb = reflect_idx(sy + 1, ch);
    const unsigned char* p00 = crop + ((long long)ya * cw + xa) * 3;
    const unsigned char* p01 = crop + ((long long)ya * cw + xb) * 3;
    const unsigned char* p10 = crop + ((long long)yb * cw + xa) * 3;
    const unsigned char* p11 = crop + ((long long)yb * cw + xb) * 3;
    const long long o = (long long)(r0 + i) * tw + (c0 + j);
    for (int c = 0; c < 3; c++) {
        const int s = (p00[c] * k00 + p01[c] * k01 + p10[c] * k10 + p11[c] * k11 + (1 << 14)) >> 15;
        const float wc = grid_lerp(coarse, gh, gw, 6, c, y0, x0, w00, w01, w10, w11);
        acc[o * 3 + c] = acc[o * 3 + c] + (float)s * wc;
    }
    wsum[o] = wsum[o] + wt;
}

extern "C" __global__ void ps_finish(const float* acc, const float* wsum, const float* cov,
        const int ncy, const int ncx, const int step, const int th, const int tw,
        unsigned char* rgb, unsigned char* alpha, const int use_look, const float lmid, const float lk,
        const float lsat)
{
    // normalise, coverage edge, 8-bit RGB and alpha
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= th || j >= tw) return;
    const int gxq = (j * 32) / step, gyq = (i * 32) / step;
    const int x0 = gxq >> 5, y0 = gyq >> 5;
    const float ax = (float)(gxq & 31) * (1.0f / 32.0f), ay = (float)(gyq & 31) * (1.0f / 32.0f);
    const float covf = grid_lerp(cov, ncy, ncx, 1, 0, y0, x0, (1.0f - ay) * (1.0f - ax), (1.0f - ay) * ax,
                                 ay * (1.0f - ax), ay * ax);
    const long long o = (long long)i * tw + j;
    const float ws = wsum[o];
    const bool valid = (ws > 1e-4f) && (covf >= 0.5f);
    alpha[o] = valid ? 255 : 0;
    const float inv = 1.0f / fmaxf(ws, 1e-6f);
    float v[3];
    for (int c = 0; c < 3; c++) v[c] = acc[o * 3 + c] * inv;
    if (use_look) {
        // contrast around the mosaic's median brightness, then saturation (same as ortho.apply_look)
        const float y = (0.114f * v[0] + 0.587f * v[1]) + 0.299f * v[2];
        const float y2 = lmid + (y - lmid) * lk;
        const float mx = fmaxf(fmaxf(v[0], v[1]), v[2]), mn = fminf(fminf(v[0], v[1]), v[2]);
        const float rel = (mx - mn) / fmaxf(mx, 1.0f);
        const float se = 1.0f + (lsat - 1.0f) * (1.0f - rel);    // vibrance
        for (int c = 0; c < 3; c++) v[c] = fminf(fmaxf(y2 + se * (v[c] - y), 0.0f), 255.0f);
    }
    for (int c = 0; c < 3; c++) {
        int q = __float2int_rn(fabsf(v[c]));
        q = min(max(q, 0), 255);
        rgb[o * 3 + (2 - c)] = valid ? (unsigned char)q : 0;   // BGR -> RGB
    }
}
"""

_KNAMES = ("ps_acc_item", "ps_finish")


# ------------------------------------------------------------------------------- devices
class _CupyDev:
    """The real card (CuPy)."""

    def __init__(self):
        import cupy as cp
        self.cp = cp
        mod = cp.RawModule(code=_SRC, options=("--fmad=false",))
        self.k = {n: mod.get_function(n) for n in _KNAMES}
        self.stream = cp.cuda.Stream(non_blocking=True)
        self._pin = {}
        self._bufs = {}

    def name(self):
        cp = self.cp
        p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        n = p["name"]
        return n.decode() if isinstance(n, bytes) else str(n)

    def _pinned(self, key, nbytes):
        cur = self._pin.get(key)
        if cur is None or cur[1] < nbytes:
            size = int(nbytes * 1.25) + (1 << 20)
            mem = self.cp.cuda.alloc_pinned_memory(size)
            cur = (mem, size, np.frombuffer(mem, np.uint8, size))
            self._pin[key] = cur
        return cur[2][:nbytes]

    def buffer(self, key, nbytes):
        cur = self._bufs.get(key)
        if cur is None or cur.size < nbytes:
            cur = self.cp.empty(int(nbytes * 1.25) + 4096, self.cp.uint8)
            self._bufs[key] = cur
        return cur[:nbytes]

    def upload(self, host_u8):
        pin = self._pinned("in", host_u8.nbytes)
        np.copyto(pin, host_u8)
        d = self.buffer("in", host_u8.nbytes)
        d.set(pin, stream=self.stream)
        return d

    def zeros_f32(self, key, n):
        d = self.buffer(key, n * 4).view(self.cp.float32)
        with self.stream:
            d.fill(0)
        return d

    def empty_u8(self, key, n):
        return self.buffer(key, n)

    def view(self, d, off, n, dtype):
        return d[off:off + n * np.dtype(dtype).itemsize].view(dtype)

    def launch(self, name, grid, block, args):
        with self.stream:
            self.k[name](grid, block, args)

    def download(self, d, host_u8):
        pin = self._pinned("out", host_u8.nbytes)
        d.get(out=pin, stream=self.stream)
        self.stream.synchronize()
        np.copyto(host_u8, pin)

    def sync(self):
        self.stream.synchronize()


class _EmuDev:
    """The same kernels compiled for the CPU (tests on machines without a card)."""

    def __init__(self, so_path):
        import ctypes
        self.ct = ctypes
        self.lib = ctypes.CDLL(so_path)
        self._bufs = {}

    def name(self):
        return "CPU emulation"

    def buffer(self, key, nbytes):
        cur = self._bufs.get(key)
        if cur is None or cur.size < nbytes:
            cur = np.empty(int(nbytes * 1.25) + 4096, np.uint8)
            self._bufs[key] = cur
        return cur[:nbytes]

    def upload(self, host_u8):
        d = self.buffer("in", host_u8.nbytes)
        np.copyto(d, host_u8)
        return d

    def zeros_f32(self, key, n):
        d = self.buffer(key, n * 4).view(np.float32)
        d[...] = 0
        return d

    def empty_u8(self, key, n):
        return self.buffer(key, n)

    def view(self, d, off, n, dtype):
        return d[off:off + n * np.dtype(dtype).itemsize].view(dtype)

    def launch(self, name, grid, block, args):
        ct = self.ct
        conv = []
        for a in args:
            if isinstance(a, np.ndarray) and a.ndim > 0:
                conv.append(ct.c_void_p(a.ctypes.data))
            elif isinstance(a, (np.floating, float)):
                conv.append(ct.c_float(float(a)))
            else:
                conv.append(ct.c_int(int(a)))
        getattr(self.lib, "run_" + name)(ct.c_int(grid[0]), ct.c_int(grid[1]), ct.c_int(block[0]),
                                         ct.c_int(block[1]), *conv)

    def download(self, d, host_u8):
        np.copyto(host_u8, d[:host_u8.nbytes])

    def sync(self):
        pass


def make_device():
    emu = os.environ.get("PLANESITE_GPU_EMULATE")
    if emu:
        return _EmuDev(emu)
    return _CupyDev()


# ------------------------------------------------------------------------- block layout
def _align(n, a=64):
    return (n + a - 1) // a * a


def pack_block(buf_for, region, plan, preps):
    """Lay out a block's inputs (coverage grid, then per photo its coarse grid and window) and room
    for the result. buf_for(nbytes) returns a writable uint8 array of at least nbytes.
    Returns (job description, buffer)."""
    tx, ty, tw, th = region
    cov = plan["covered"].reshape(plan["ncy"], plan["ncx"]).astype(np.float32)
    off = 0
    lay = []
    cov_off = off
    off = _align(off + cov.nbytes)
    for (r0, r1, c0, c1), (coarse, gi0, gj0, crop) in preps:
        co = off
        off = _align(off + coarse.nbytes)
        cr = off
        off = _align(off + crop.shape[0] * crop.shape[1] * 3)
        lay.append((co, coarse.shape[0], coarse.shape[1], gi0, gj0, cr, crop.shape[0], crop.shape[1], r0, r1, c0, c1))
    in_bytes = off
    out_off = off
    total = out_off + th * tw * 4
    buf = buf_for(total)
    buf[cov_off:cov_off + cov.nbytes] = cov.view(np.uint8).ravel()
    for (co, gh, gw, gi0, gj0, cr, ch, cw, r0, r1, c0, c1), (_, (coarse, _, _, crop)) in zip(lay, preps):
        buf[co:co + coarse.nbytes] = np.ascontiguousarray(coarse).view(np.uint8).ravel()
        dst = buf[cr:cr + ch * cw * 3].reshape(ch, cw, 3)
        dst[...] = crop
    job = {"region": region, "step": plan["step"], "ncy": plan["ncy"], "ncx": plan["ncx"], "cov_off": cov_off,
           "look": ortho._RCTX.get("look") if ortho._RCTX else None,
           "items": lay, "in_bytes": in_bytes, "out_off": out_off, "total": total}
    return job, buf


def run_block(dev, job, buf):
    """Blend one packed block on the device; the result lands in buf[out_off:]."""
    tx, ty, tw, th = job["region"]
    step = job["step"]
    d_in = dev.upload(buf[:job["in_bytes"]])
    acc = dev.zeros_f32("acc", th * tw * 3)
    wsum = dev.zeros_f32("wsum", th * tw)
    B = (32, 8)
    for co, gh, gw, gi0, gj0, cr, ch, cw, r0, r1, c0, c1 in job["items"]:
        h, w = r1 - r0, c1 - c0
        coarse = dev.view(d_in, co, gh * gw * 6, np.float32)
        crop = dev.view(d_in, cr, ch * cw * 3, np.uint8)
        dev.launch("ps_acc_item", ((w + B[0] - 1) // B[0], (h + B[1] - 1) // B[1]), B,
                   (coarse, np.int32(gh), np.int32(gw), np.int32(r0 - gi0 * step), np.int32(c0 - gj0 * step),
                    np.int32(step), crop, np.int32(ch), np.int32(cw), np.int32(r0), np.int32(c0), np.int32(h),
                    np.int32(w), np.int32(tw), acc, wsum))
    cov = dev.view(d_in, job["cov_off"], job["ncy"] * job["ncx"], np.float32)
    lk = job.get("look")
    out = dev.empty_u8("out", th * tw * 4)
    rgb = out[:th * tw * 3]
    alpha = out[th * tw * 3:]
    dev.launch("ps_finish", ((tw + B[0] - 1) // B[0], (th + B[1] - 1) // B[1]), B,
               (acc, wsum, cov, np.int32(job["ncy"]), np.int32(job["ncx"]), np.int32(step), np.int32(th),
                np.int32(tw), rgb, alpha, np.int32(1 if lk else 0), np.float32(lk["mid"] if lk else 0),
                np.float32(lk["contrast"] if lk else 1), np.float32(lk["saturation"] if lk else 1)))
    dev.download(out, buf[job["out_off"]:job["out_off"] + th * tw * 4])


def block_result(job, buf, tiles):
    """Tiles (with half-resolution copies) from a finished block in buf."""
    tx, ty, tw, th = job["region"]
    o = job["out_off"]
    rgb = buf[o:o + th * tw * 3].reshape(th, tw, 3)
    alpha = buf[o + th * tw * 3:o + th * tw * 4].reshape(th, tw)
    return ortho.cut_tiles(job["region"], tiles, rgb, alpha)


def prepare_block(region, tiles):
    """CPU side of a GPU block: plan and decoded photo windows. None if the block is empty."""
    plan = ortho.region_plan(region, tiles)
    if plan is None:
        return None, None
    preps = []
    for a, ci, r0, r1, c0, c1 in plan["items"]:
        p = ortho.prepare_item(plan, a, ci, r0, r1, c0, c1)
        if p is not None:
            preps.append(((r0, r1, c0, c1), p))
    return plan, preps


# ---------------------------------------------------------------------- feeder side
class GpuLink:
    """A render worker's connection to the GPU process: two shared-memory slots, submit / collect."""

    def __init__(self, wid, link):
        self.wid = wid
        self.job_q = link["job_q"]
        self.reply_q = link["reply_qs"][wid]
        self.state = link["state"]          # [alive (1 ok, 0 starting, -1 off), pending]
        self.max_pending = link["max_pending"]
        self.shm = [None] * SLOTS
        self.free = list(range(SLOTS))
        self.out = {}                       # slot -> (group, job, t0, snap)
        self.stats = {"gpu_blocks": 0, "gpu_wait": 0.0}
        self.seq = 0

    def usable(self):
        if self.state[0] != 1 or not self.free:
            return False
        return self.state[1] < self.max_pending

    def _buf(self, slot, nbytes):
        from multiprocessing import shared_memory
        cur = self.shm[slot]
        if cur is None or cur.size < nbytes:
            if cur is not None:
                cur.close()
                cur.unlink()
            cur = shared_memory.SharedMemory(create=True, size=int(nbytes * 1.3) + (1 << 20))
            self.shm[slot] = cur
        return np.ndarray((nbytes,), np.uint8, cur.buf)

    def submit(self, group, region, plan, preps, snap):
        slot = self.free.pop()
        job, _ = pack_block(lambda n: self._buf(slot, n), region, plan, preps)
        self.seq += 1
        job.update(wid=self.wid, slot=slot, shm=self.shm[slot].name, seq=self.seq)
        with self.state.get_lock():
            self.state[1] += 1
        self.job_q.put(job)
        self.out[slot] = (group, job, time.perf_counter(), snap)

    def collect(self, block=True, timeout=900):
        """Finished blocks as (group, tiles, snap); tiles is None when the GPU failed (the caller
        renders those on the CPU). With block=True waits for at least one."""
        done = []
        t_start = time.perf_counter()
        while self.out:
            try:
                msg = self.reply_q.get(timeout=2.0) if block else self.reply_q.get_nowait()
            except queue.Empty:
                if not block:
                    break
                if self.state[0] == -1 or time.perf_counter() - t_start > timeout:
                    # the GPU stopped (or hangs): take every block back; these slots are not
                    # reused because the card gets no more work
                    self.state[0] = -1
                    for s in list(self.out):
                        group, job, _, snap = self.out.pop(s)
                        done.append((group, None, snap))
                    break
                continue
            kind, slot, seq, info = msg
            ent = self.out.get(slot)
            if ent is None or ent[1]["seq"] != seq:
                continue     # answer for a block already taken back
            group, job, _, snap = self.out.pop(slot)
            if kind == "ok":
                self.free.append(slot)
                buf = np.ndarray((job["total"],), np.uint8, self.shm[slot].buf)
                res = block_result(job, buf, group)
                self.stats["gpu_blocks"] += 1
                done.append((group, res, snap))
            else:
                done.append((group, None, snap))
            block = False   # got one; pick up any others that are already finished
        self.stats["gpu_wait"] += time.perf_counter() - t_start
        return done

    def close(self):
        for s in self.shm:
            if s is not None:
                try:
                    s.close()
                    s.unlink()
                except Exception:
                    pass


# ------------------------------------------------------------------------ GPU process
def _selftest(dev, test_groups, log):
    """Render real blocks on the CPU and the device; they must agree."""
    checked = 0
    for group in test_groups:
        x0 = min(t[0] for t in group)
        y0 = min(t[1] for t in group)
        region = (x0, y0, max(t[0] + t[2] for t in group) - x0, max(t[1] + t[3] for t in group) - y0)
        plan, preps = prepare_block(region, group)
        if not preps:
            continue
        job, buf = pack_block(lambda n: np.empty(n, np.uint8), region, plan, preps)
        run_block(dev, job, buf)
        dev.sync()
        g = block_result(job, buf, group)
        c = ortho.render_region(region, group)
        for (t, rg, ag, _, _), (_, rc, ac, _, _) in zip(g, c):
            d = np.abs(rg.astype(np.int16) - rc.astype(np.int16))
            frac = float((d > 0).any(2).mean())
            if d.max() > 3 or frac > 0.01 or float((ag != ac).mean()) > 0.001:
                raise RuntimeError("GPU block differs from CPU (max %d, %.2f%% of pixels)" % (d.max(), 100 * frac))
        checked += 1
        if checked >= 2:
            break
    return checked


def gpu_render_server_main(ctx_file, job_q, reply_qs, state, res_q, test_groups, threads=2):
    """The GPU process: self-test, then blend blocks from all render workers."""
    from multiprocessing import shared_memory
    try:
        from .sysinfo import full_speed, raise_priority
        full_speed()
        ortho.init_render(ctx_file)
        t0 = time.perf_counter()
        devs = [make_device() for _ in range(threads)]
        n = _selftest(devs[0], test_groups, None)
        ortho._IMG_CACHE.clear()      # photo strips from the self-test are not needed any more
        ortho._IMG_BYTES[0] = 0
        raise_priority()
        res_q.put(("gpu", 0, "rendering on %s: %d threads, ready in %.0f s (checked against the CPU on %d block%s)"
                   % (devs[0].name(), threads, time.perf_counter() - t0, n, "" if n == 1 else "s")))
        state[0] = 1
    except Exception as e:  # noqa
        state[0] = -1
        res_q.put(("gpu_off", 0, "rendering stays on the CPU: %s: %s" % (type(e).__name__, str(e)[:300])))
        return
    segs = {}
    lock = threading.Lock()
    stats = {"blocks": 0, "busy": 0.0}

    def worker(dev):
        while True:
            job = job_q.get()
            if job is None:
                job_q.put(None)
                return
            t1 = time.perf_counter()
            try:
                with lock:
                    shm = segs.get(job["shm"])
                    if shm is None:
                        try:
                            shm = shared_memory.SharedMemory(name=job["shm"], track=False)
                        except TypeError:   # Python < 3.13 (Windows has no tracker anyway)
                            shm = shared_memory.SharedMemory(name=job["shm"])
                        segs[job["shm"]] = shm
                buf = np.ndarray((job["total"],), np.uint8, shm.buf)
                fail = os.environ.get("PLANESITE_GPU_TEST_FAIL_AFTER")      # test hooks
                if fail and stats["blocks"] >= int(fail):
                    if os.environ.get("PLANESITE_GPU_TEST_CRASH"):
                        os._exit(5)
                    raise RuntimeError("test failure")
                run_block(dev, job, buf)
                kind, info = "ok", time.perf_counter() - t1
            except Exception as e:  # noqa
                kind, info = "error", "%s: %s" % (type(e).__name__, e)
                state[0] = -1   # stop sending blocks here; feeders render them on the CPU
            with state.get_lock():
                state[1] -= 1
            with lock:
                stats["blocks"] += kind == "ok"
                stats["busy"] += time.perf_counter() - t1
            reply_qs[job["wid"]].put((kind, job["slot"], job["seq"], info))
    ths = [threading.Thread(target=worker, args=(d,), daemon=True) for d in devs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    res_q.put(("gpu", 0, "rendered %d blocks on the card (%.0f s busy)" % (stats["blocks"], stats["busy"])))
    for s in segs.values():
        try:
            s.close()
        except Exception:
            pass
