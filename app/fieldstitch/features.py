"""Feature extraction (worker side)."""
import os

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


def extract(task):
    """task: (index, path, out_npz, thumb_path, reduce, nfeat).

    Returns (index, n_features, error, timings)
    """
    idx, path, out_npz, thumb_path, reduce, nfeat = task
    import time
    tt = {"read": 0.0, "decode": 0.0, "detect": 0.0, "describe": 0.0, "bytes": 0}
    try:
        t0 = time.perf_counter()
        buf = read_file(path)
        tt["bytes"] = int(buf.nbytes)
        t1 = time.perf_counter()
        tt["read"] = t1 - t0
        gray = decode(buf, reduce, color=False)
        h, w = gray.shape[:2]
        if thumb_path and not os.path.exists(thumb_path):
            th = decode(buf, 8, color=True)
            write_jpg(thumb_path, th, 88)
        t2 = time.perf_counter()
        tt["decode"] = t2 - t1
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g2 = clahe.apply(gray)
        sift = cv2.SIFT_create(nfeatures=int(nfeat * 2.2), contrastThreshold=0.015, edgeThreshold=12)
        kps = sift.detect(g2, None)
        if len(kps) < 10:
            np.savez(out_npz, xy=np.zeros((0, 2), np.float32), desc=np.zeros((0, 128), np.uint8),
                     resp=np.zeros(0, np.float32), red_wh=np.array([w, h], np.int32))
            return idx, 0, None, tt
        t3 = time.perf_counter()
        tt["detect"] = t3 - t2
        sel = _spread(kps, w, h, nfeat)
        kps = [kps[i] for i in sel]
        kps = sorted(kps, key=lambda k: -k.response)
        kps, desc = sift.compute(g2, kps)
        tt["describe"] = time.perf_counter() - t3
        if desc is None or len(kps) < 10:
            np.savez(out_npz, xy=np.zeros((0, 2), np.float32), desc=np.zeros((0, 128), np.uint8),
                     resp=np.zeros(0, np.float32), red_wh=np.array([w, h], np.int32))
            return idx, 0, None, tt
        # RootSIFT
        desc = desc / (np.abs(desc).sum(1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc)
        desc = np.clip(desc * 512.0, 0, 255).astype(np.uint8)
        xy = np.array([k.pt for k in kps], np.float32)
        # coordinates stay in reduced-image pixels; red_wh lets loaders map to full res
        resp = np.array([k.response for k in kps], np.float32)
        np.savez(out_npz, xy=xy, desc=desc, resp=resp, red_wh=np.array([w, h], np.int32))
        return idx, len(kps), None, tt
    except Exception as e:  # noqa
        return idx, 0, "%s: %s" % (type(e).__name__, e), tt
