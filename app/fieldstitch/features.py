"""Feature extraction (worker side)."""
import os
import time

import cv2
import numpy as np

_REDUCE_GRAY = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
                4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}
_REDUCE_COLOR = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def read_file(path):
    return np.fromfile(path, dtype=np.uint8)  # unicode-safe on Windows


def decode(buf, reduce=1, color=True):
    table = _REDUCE_COLOR if color else _REDUCE_GRAY
    flag = table[reduce] | cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise IOError("cannot decode image")
    return img


def load_image(path, reduce=1, color=True):
    return decode(read_file(path), reduce, color)


def write_jpg(path, img, q=90):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise IOError("encode failed")
    enc.tofile(path)


def _spread(kps, W, H, n, gx=12, gy=9):
    """Keep n keypoints spread over a grid (best responses per cell first)."""
    if len(kps) <= n:
        return list(range(len(kps)))
    resp = np.array([k.response for k in kps], np.float32)
    xy = np.array([k.pt for k in kps], np.float32)
    cx = np.clip((xy[:, 0] / W * gx).astype(int), 0, gx - 1)
    cy = np.clip((xy[:, 1] / H * gy).astype(int), 0, gy - 1)
    cell = cy * gx + cx
    order = np.lexsort((-resp, cell))
    cs = cell[order]
    first = np.searchsorted(cs, cs, side="left")
    rank = np.arange(len(cs)) - first
    cap = int(np.ceil(n / (gx * gy)))
    sel = order[rank < cap]
    if len(sel) < n:
        rest = np.setdiff1d(np.arange(len(kps)), sel)
        rest = rest[np.argsort(-resp[rest])][: n - len(sel)]
        sel = np.concatenate([sel, rest])
    return sel[:n].tolist()


def prepare(buf, reduce, thumb_path):
    """Decode once: grey image (contrast-equalised) for features, plus the colour thumbnail if missing."""
    if thumb_path and not os.path.exists(thumb_path):
        col = decode(buf, reduce, color=True)
        th = cv2.resize(col, ((col.shape[1] + 1) * reduce // 8, (col.shape[0] + 1) * reduce // 8),
                        interpolation=cv2.INTER_AREA) if reduce < 8 else col
        write_jpg(thumb_path, th, 88)
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    else:
        gray = decode(buf, reduce, color=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _save(out_npz, xy, desc, resp, w, h):
    """RootSIFT-encode descriptors and write the feature file (same format for CPU and GPU)."""
    if desc is None or len(xy) < 10:
        np.savez(out_npz, xy=np.zeros((0, 2), np.float32), desc=np.zeros((0, 128), np.uint8),
                 resp=np.zeros(0, np.float32), red_wh=np.array([w, h], np.int32))
        return 0
    desc = desc / (np.abs(desc).sum(1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc)
    desc = np.clip(desc * 512.0, 0, 255).astype(np.uint8)
    # coordinates stay in reduced-image pixels; red_wh lets loaders map to full res
    np.savez(out_npz, xy=np.asarray(xy, np.float32), desc=desc, resp=np.asarray(resp, np.float32),
             red_wh=np.array([w, h], np.int32))
    return len(xy)


def sift_cpu(g2, nfeat):
    sift = cv2.SIFT_create(nfeatures=int(nfeat * 2.2), contrastThreshold=0.015, edgeThreshold=12)
    kps = sift.detect(g2, None)
    h, w = g2.shape[:2]
    if len(kps) < 10:
        return np.zeros((0, 2), np.float32), None, np.zeros(0, np.float32), 0.0
    t = time.perf_counter()
    sel = _spread(kps, w, h, nfeat)
    kps = [kps[i] for i in sel]
    kps = sorted(kps, key=lambda k: -k.response)
    kps, desc = sift.compute(g2, kps)
    if desc is None:
        return np.zeros((0, 2), np.float32), None, np.zeros(0, np.float32), 0.0
    xy = np.array([k.pt for k in kps], np.float32)
    resp = np.array([k.response for k in kps], np.float32)
    return xy, desc, resp, time.perf_counter() - t


class _KP:
    __slots__ = ("pt", "response")

    def __init__(self, x, y, r):
        self.pt = (x, y)
        self.response = r


def sift_gpu(g2, nfeat, gpu=True):
    """Same detector and descriptor as sift_cpu, computed with gsift on the GPU."""
    from . import gsift
    h, w = g2.shape[:2]
    kps, state = gsift.detect(g2, gpu=gpu, max_kp=int(nfeat * 2.2))
    n = len(kps["x"])
    if n < 10:
        return np.zeros((0, 2), np.float32), None, np.zeros(0, np.float32), 0.0
    t = time.perf_counter()
    lst = [_KP(float(x), float(y), float(r)) for x, y, r in zip(kps["x"], kps["y"], kps["response"])]
    sel = np.array(_spread(lst, w, h, nfeat), np.int64)
    sel = sel[np.argsort(-kps["response"][sel], kind="stable")]
    desc = gsift.describe(state, kps, sel)
    xy = np.c_[kps["x"][sel], kps["y"][sel]].astype(np.float32)
    return xy, desc, kps["response"][sel].astype(np.float32), time.perf_counter() - t


def extract(task):
    """CPU worker. task: (index, path, out_npz, thumb_path, reduce, nfeat).

    Returns (index, n_features, error, timings)
    """
    idx, path, out_npz, thumb_path, reduce, nfeat = task
    tt = {"read": 0.0, "decode": 0.0, "detect": 0.0, "describe": 0.0, "bytes": 0, "photos_cpu": 1}
    try:
        t0 = time.perf_counter()
        buf = read_file(path)
        tt["bytes"] = int(buf.nbytes)
        t1 = time.perf_counter()
        tt["read"] = t1 - t0
        g2 = prepare(buf, reduce, thumb_path)
        t2 = time.perf_counter()
        tt["decode"] = t2 - t1
        xy, desc, resp, t_desc = sift_cpu(g2, nfeat)
        tt["detect"] = time.perf_counter() - t2 - t_desc
        tt["describe"] = t_desc
        n = _save(out_npz, xy, desc, resp, g2.shape[1], g2.shape[0])
        return idx, n, None, tt
    except Exception as e:  # noqa
        return idx, 0, "%s: %s" % (type(e).__name__, e), tt


def cpu_worker_main(task_q, res_q):
    """Feature worker process pulling from a shared queue (used next to GPU workers)."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    while True:
        task = task_q.get()
        if task is None:
            break
        res_q.put(("ok", extract(task)))
    res_q.put(("exit", "cpu"))


# --------------------------------------------------------------------------------- GPU
def gpu_selftest():
    """Run gsift on a small test image on the GPU and on the CPU; they must agree."""
    from . import gsift
    import cupy as cp
    rng = np.random.default_rng(7)
    img = cv2.GaussianBlur(rng.integers(0, 255, (240, 320)).astype(np.uint8), (0, 0), 2.0)
    img = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    kg, sg = gsift.detect(img, gpu=True, max_kp=400)
    kc, sc = gsift.detect(img, gpu=False, max_kp=400)
    if len(kc["x"]) < 20:
        raise RuntimeError("self-test image produced too few keypoints")
    if abs(len(kg["x"]) - len(kc["x"])) > 0.03 * len(kc["x"]) + 2:
        raise RuntimeError("GPU found %d keypoints, CPU %d" % (len(kg["x"]), len(kc["x"])))
    from scipy.spatial import cKDTree
    d, j = cKDTree(np.c_[kc["x"], kc["y"]]).query(np.c_[kg["x"], kg["y"]])
    if (d < 0.05).mean() < 0.95:
        raise RuntimeError("GPU keypoint positions differ from CPU")
    m = np.nonzero(d < 0.05)[0][:100]
    dg = gsift.describe(sg, kg, m)
    dc = gsift.describe(sc, kc, j[m])
    cos = (dg * dc).sum(1) / (np.linalg.norm(dg, axis=1) * np.linalg.norm(dc, axis=1) + 1e-9)
    if np.median(cos) < 0.995:
        raise RuntimeError("GPU descriptors differ from CPU")
    cp.cuda.Stream.null.synchronize()
    return True


def gpu_info():
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    if n < 1:
        raise RuntimeError("no CUDA device")
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    free, total = cp.cuda.runtime.memGetInfo()
    return {"name": name, "mem_gb": round(total / 1e9, 1), "free_gb": round(free / 1e9, 1),
            "cc": "%d.%d" % (props["major"], props["minor"])}


def gpu_worker_main(task_q, res_q, n_threads, wid):
    """GPU feature worker: decoder threads (CPU) feed one GPU loop running gsift.
    Falls back to OpenCV SIFT (still threaded) if the GPU is unavailable or fails."""
    import queue
    import threading
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    gpu_ok = False
    try:
        t0 = time.perf_counter()
        info = gpu_info()
        gpu_selftest()
        gpu_ok = True
        res_q.put(("gpu", wid, "%s (%.1f GB, compute %s), ready in %.0f s"
                   % (info["name"], info["mem_gb"], info["cc"], time.perf_counter() - t0)))
    except Exception as e:  # noqa
        res_q.put(("gpu_off", wid, "%s: %s" % (type(e).__name__, str(e)[:300])))

    if not gpu_ok:
        n_threads = 1   # the CPU workers already use every core

        def cpu_thread():
            while True:
                task = task_q.get()
                if task is None:
                    break
                res_q.put(("ok", extract(task)))
        ths = [threading.Thread(target=cpu_thread, daemon=True) for _ in range(n_threads)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        res_q.put(("exit", "gpu%d" % wid))
        return

    ready = queue.Queue(maxsize=max(4, n_threads))

    def decoder():
        while True:
            task = task_q.get()
            if task is None:
                ready.put(None)
                break
            idx, path, out_npz, thumb_path, reduce, nfeat = task
            tt = {"read": 0.0, "decode": 0.0, "detect": 0.0, "describe": 0.0, "bytes": 0, "photos_gpu": 1}
            try:
                t0 = time.perf_counter()
                buf = read_file(path)
                tt["bytes"] = int(buf.nbytes)
                t1 = time.perf_counter()
                tt["read"] = t1 - t0
                g2 = prepare(buf, reduce, thumb_path)
                tt["decode"] = time.perf_counter() - t1
                ready.put((task, g2, tt, None))
            except Exception as e:  # noqa
                ready.put((task, None, tt, "%s: %s" % (type(e).__name__, e)))

    ths = [threading.Thread(target=decoder, daemon=True) for _ in range(n_threads)]
    for t in ths:
        t.start()
    finished = 0
    use_gpu = True
    while finished < n_threads:
        item = ready.get()
        if item is None:
            finished += 1
            continue
        task, g2, tt, err = item
        idx, path, out_npz, thumb_path, reduce, nfeat = task
        if err:
            res_q.put(("ok", (idx, 0, err, tt)))
            continue
        try:
            t2 = time.perf_counter()
            if use_gpu:
                try:
                    xy, desc, resp, t_desc = sift_gpu(g2, nfeat)
                except Exception as e:  # noqa
                    use_gpu = False
                    res_q.put(("gpu_off", wid, "GPU failed during processing, continuing on the CPU: %s: %s"
                               % (type(e).__name__, str(e)[:300])))
                    tt["photos_gpu"] = 0
                    tt["photos_cpu"] = 1
                    xy, desc, resp, t_desc = sift_cpu(g2, nfeat)
            else:
                tt["photos_gpu"] = 0
                tt["photos_cpu"] = 1
                xy, desc, resp, t_desc = sift_cpu(g2, nfeat)
            if tt.get("photos_gpu"):
                tt["gpu_detect"] = time.perf_counter() - t2 - t_desc
                tt["gpu_describe"] = t_desc
            else:
                tt["detect"] = time.perf_counter() - t2 - t_desc
                tt["describe"] = t_desc
            n = _save(out_npz, xy, desc, resp, g2.shape[1], g2.shape[0])
            res_q.put(("ok", (idx, n, None, tt)))
        except Exception as e:  # noqa
            res_q.put(("ok", (idx, 0, "%s: %s" % (type(e).__name__, e), tt)))
    for t in ths:
        t.join()
    res_q.put(("exit", "gpu%d" % wid))
