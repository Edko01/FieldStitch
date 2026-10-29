"""Decode only part of a JPEG (libjpeg-turbo's TurboJPEG 3 API through ctypes).

Rendering needs only a small window of each photo for each block of the mosaic (where that photo
is the best view), so decoding the whole 20 MP photo wastes most of the work. TurboJPEG can skip
the rows above the window (only their entropy decoding is needed) and stop after the last row.
Pixels are identical to a full decode by OpenCV (both use
libjpeg-turbo's default accurate IDCT and smooth chroma upsampling).

Falls back to a full OpenCV decode when the library is missing or the file can't be cropped
(e.g. not a JPEG)."""
import ctypes
import os
import sys
import threading

import numpy as np

_TJINIT_DECOMPRESS = 1
_TJPARAM_JPEGWIDTH = 5
_TJPARAM_JPEGHEIGHT = 6
_TJPF_BGR = 1


class _Region(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("w", ctypes.c_int), ("h", ctypes.c_int)]


class _Scale(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int), ("denom", ctypes.c_int)]


_LIB = [None, False]   # library, tried
_LOCAL = threading.local()
STATS = {"crop": 0, "full": 0}


def _load():
    if _LIB[1]:
        return _LIB[0]
    _LIB[1] = True
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
    names = [os.environ.get("PLANESITE_TURBOJPEG", "")]
    if sys.platform == "win32":
        names += [os.path.join(here, "turbojpeg.dll")]
    else:
        names += [os.path.join(here, "libturbojpeg.so"), "libturbojpeg.so.0", "libturbojpeg.so"]
    for n in names:
        if not n:
            continue
        try:
            lib = ctypes.CDLL(n)
            lib.tj3Init.restype = ctypes.c_void_p
            lib.tj3Init.argtypes = [ctypes.c_int]
            lib.tj3DecompressHeader.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            lib.tj3Get.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.tj3SetScalingFactor.argtypes = [ctypes.c_void_p, _Scale]
            lib.tj3SetCroppingRegion.argtypes = [ctypes.c_void_p, _Region]
            lib.tj3Decompress8.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                           ctypes.c_int, ctypes.c_int]
            lib.tj3GetErrorStr.restype = ctypes.c_char_p
            lib.tj3GetErrorStr.argtypes = [ctypes.c_void_p]
            _LIB[0] = lib
            return lib
        except (OSError, AttributeError):
            continue
    return None


def available():
    return _load() is not None


def present():
    """Whether the library looks installed (without loading it)."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
    if os.environ.get("PLANESITE_TURBOJPEG"):
        return True
    return os.path.exists(os.path.join(here, "turbojpeg.dll" if sys.platform == "win32" else "libturbojpeg.so")) \
        or (sys.platform != "win32" and available())


def _handle(lib):
    h = getattr(_LOCAL, "h", None)
    if h is None:
        h = lib.tj3Init(_TJINIT_DECOMPRESS)
        if not h:
            raise RuntimeError("tj3Init failed")
        _LOCAL.h = h
    return h


def decode_rows(buf, y0, y1, reduce=1):
    """Decode rows [y0, y1) (clipped to the image) at full width. Returns (BGR image, first row)."""
    img, _, oy = decode_crop(buf, 0, y0, 10 ** 9, y1, reduce)
    return img, oy


def decode_crop(buf, x0, y0, x1, y1, reduce=1):
    """Decode the window [x0, x1) x [y0, y1) (pixels of the image decoded at 1/reduce scale) of the
    JPEG in buf (uint8 array). Returns (BGR image, ox, oy): the image covers at least the window,
    its top-left pixel is (ox, oy). The window is clipped to the image."""
    lib = _load()
    if lib is not None:
        try:
            return _crop_tj(lib, buf, x0, y0, x1, y1, reduce)
        except Exception:
            pass
    return _crop_cv(buf, x0, y0, x1, y1, reduce)


def _crop_tj(lib, buf, x0, y0, x1, y1, reduce):
    h = _handle(lib)
    ptr = buf.ctypes.data_as(ctypes.c_void_p)
    if lib.tj3DecompressHeader(h, ptr, buf.nbytes) != 0:
        raise IOError(lib.tj3GetErrorStr(h).decode(errors="replace"))
    W = lib.tj3Get(h, _TJPARAM_JPEGWIDTH)
    H = lib.tj3Get(h, _TJPARAM_JPEGHEIGHT)
    if lib.tj3SetScalingFactor(h, _Scale(1, int(reduce))) != 0:
        raise IOError(lib.tj3GetErrorStr(h).decode(errors="replace"))
    sw = (W + reduce - 1) // reduce
    sh = (H + reduce - 1) // reduce
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(sw, int(x1))
    y1 = min(sh, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty window")
    # Only whole rows are cropped: skipping rows gives exactly the pixels of a full decode, but
    # libjpeg-turbo's column cropping upsamples chroma slightly differently (up to ~10 levels).
    # The columns are cut afterwards.
    ax = 0
    if y0 == 0 and y1 == sh:
        region = _Region(0, 0, 0, 0)          # whole image
    else:
        region = _Region(0, y0, sw, y1 - y0)
    if lib.tj3SetCroppingRegion(h, region) != 0:
        raise IOError(lib.tj3GetErrorStr(h).decode(errors="replace"))
    out = np.empty((y1 - y0, sw, 3), np.uint8)
    rc = lib.tj3Decompress8(h, ptr, buf.nbytes, out.ctypes.data_as(ctypes.c_void_p), out.strides[0], _TJPF_BGR)
    # leave the handle without a cropping region for the next call
    lib.tj3SetCroppingRegion(h, _Region(0, 0, 0, 0))
    if rc != 0:
        raise IOError(lib.tj3GetErrorStr(h).decode(errors="replace"))
    STATS["crop"] += 1
    if x0 == 0 and x1 == sw:
        return out, 0, y0
    return np.ascontiguousarray(out[:, x0:x1]), x0, y0


def _crop_cv(buf, x0, y0, x1, y1, reduce):
    from .features import decode
    img = decode(buf, reduce, color=True)
    sh, sw = img.shape[:2]
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(sw, int(x1))
    y1 = min(sh, int(y1))
    STATS["full"] += 1
    return np.ascontiguousarray(img[y0:y1, x0:x1]), x0, y0
