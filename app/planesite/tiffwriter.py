"""Write the JPEG orthomosaic GeoTIFF from tiles compressed in the render workers.

GDAL compresses a GeoTIFF on (nearly) one core while writing, which became the slowest part of
rendering (about 3 minutes for a 16-gigapixel mosaic). Here the render workers JPEG-compress the
512 x 512 TIFF tiles in parallel and the main process only appends the finished bytes.

The file has the same structure GDAL writes with COMPRESS=JPEG, PHOTOMETRIC=YCBCR and an internal
mask: (Big)TIFF, 512 x 512 tiles, YCbCr 4:2:0 JPEG with the quantisation tables in the JPEGTables
tag and per-tile optimised Huffman tables, then a 1-bit deflate mask IFD (NewSubfileType 4,
Photometric 4) that GDAL, QGIS and ArcGIS read as the dataset mask. GeoTIFF keys are copied from a
small file GDAL makes with the same CRS and transform. The IFDs go at the end of the file.
"""
import struct
import zlib

import numpy as np

TILE = 512


# ------------------------------------------------------------------------- encoding (workers)
def _split_jpeg(b):
    """Split a full JPEG into (DQT segments, stream without APP0/DQT): GDAL's abbreviated layout."""
    i = 2
    dqt = []
    keep = [b[:2]]
    while i < len(b):
        if b[i] != 0xFF:
            raise ValueError("bad JPEG marker")
        m = b[i + 1]
        if m == 0xDA:                       # start of scan: the rest is entropy-coded data
            keep.append(b[i:])
            break
        L = struct.unpack(">H", b[i + 2:i + 4])[0]
        seg = b[i:i + 2 + L]
        if m == 0xDB:
            dqt.append(seg)
        elif not (0xE0 <= m <= 0xEF):       # drop APPn (JFIF); keep SOF, DHT, DRI ...
            keep.append(seg)
        i += 2 + L
    return b"".join(dqt), b"".join(keep)


def encode_tiles(tile, rgb, alpha, quality):
    """Compress one output tile (any multiple of 512 at a 512-aligned origin, possibly cut at the
    mosaic edge) into TIFF tiles. Returns (list of (tx, ty, jpeg, mask) with tx, ty in TIFF tile
    units and None for completely empty tiles, DQT bytes)."""
    import cv2
    x0, y0, w, h = tile
    out = []
    dqt = None
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    for oy in range(0, h, TILE):
        for ox in range(0, w, TILE):
            a = alpha[oy:oy + TILE, ox:ox + TILE]
            tx, ty = (x0 + ox) // TILE, (y0 + oy) // TILE
            if not a.any():
                out.append((tx, ty, None, None))
                continue
            sub = rgb[oy:oy + TILE, ox:ox + TILE]
            if sub.shape[0] != TILE or sub.shape[1] != TILE:     # pad tiles on the mosaic edge
                pad = np.zeros((TILE, TILE, 3), np.uint8)
                pad[:sub.shape[0], :sub.shape[1]] = sub
                sub = pad
                pa = np.zeros((TILE, TILE), np.uint8)
                pa[:a.shape[0], :a.shape[1]] = a
                a = pa
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(sub, cv2.COLOR_RGB2BGR), params)
            if not ok:
                raise IOError("JPEG encoding failed")
            q, body = _split_jpeg(enc.tobytes())
            if dqt is None:
                dqt = q
            elif q != dqt:
                raise ValueError("JPEG quantisation tables differ between tiles")
            mask = zlib.compress(np.packbits(a > 127, axis=1).tobytes(), 6)
            out.append((tx, ty, body, mask))
    return out, dqt


def _empty_tiles(quality):
    import cv2
    ok, enc = cv2.imencode(".jpg", np.zeros((TILE, TILE, 3), np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    q, body = _split_jpeg(enc.tobytes())
    return body, zlib.compress(bytes(TILE * TILE // 8), 6), q


# ------------------------------------------------------------------------- GeoTIFF keys
_GEO_TAGS = (33550, 33922, 34264, 34735, 34736, 34737)


def _geo_tags_from_gdal(crs, transform):
    """GeoTIFF tags (code -> (type, count, raw value bytes)) of a 1x1 GeoTIFF made by GDAL."""
    import rasterio
    from rasterio.io import MemoryFile
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", width=1, height=1, count=1, dtype="uint8", crs=crs, transform=transform) as d:
            d.write(np.zeros((1, 1, 1), np.uint8))
        data = mf.read()
    return _read_tags(data, _GEO_TAGS)


_TSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 11: 4, 12: 8, 16: 8}


def _read_tags(data, want):
    bo = "<" if data[:2] == b"II" else ">"
    magic = struct.unpack(bo + "H", data[2:4])[0]
    out = {}
    if magic == 42:
        off = struct.unpack(bo + "I", data[4:8])[0]
        n = struct.unpack(bo + "H", data[off:off + 2])[0]
        for k in range(n):
            e = off + 2 + 12 * k
            code, typ, cnt = struct.unpack(bo + "HHI", data[e:e + 8])
            size = _TSIZE.get(typ, 1) * cnt
            raw = data[e + 8:e + 8 + size] if size <= 4 else \
                data[struct.unpack(bo + "I", data[e + 8:e + 12])[0]:][:size]
            if code in want:
                out[code] = (typ, cnt, raw if bo == "<" else None)
    else:   # BigTIFF
        off = struct.unpack(bo + "Q", data[8:16])[0]
        n = struct.unpack(bo + "Q", data[off:off + 8])[0]
        for k in range(n):
            e = off + 8 + 20 * k
            code, typ, cnt = struct.unpack(bo + "HHQ", data[e:e + 12])
            size = _TSIZE.get(typ, 1) * cnt
            raw = data[e + 12:e + 12 + size] if size <= 8 else \
                data[struct.unpack(bo + "Q", data[e + 12:e + 20])[0]:][:size]
            if code in want:
                out[code] = (typ, cnt, raw if bo == "<" else None)
    return {k: v for k, v in out.items() if v[2] is not None}


# ------------------------------------------------------------------------- the file
class JpegTiffWriter:
    """Append pre-compressed tiles in any order; close() writes the directories."""

    def __init__(self, path, width, height, crs, transform, quality=90):
        self.path = path
        self.w, self.h = int(width), int(height)
        self.quality = quality
        self.ntx = (self.w + TILE - 1) // TILE
        self.nty = (self.h + TILE - 1) // TILE
        n = self.ntx * self.nty
        self.off = np.zeros(n, np.uint64)
        self.cnt = np.zeros(n, np.uint64)
        self.moff = np.zeros(n, np.uint64)
        self.mcnt = np.zeros(n, np.uint64)
        self.dqt = None
        # like GDAL's BIGTIFF=IF_SAFER: classic TIFF only when even uncompressed it would fit
        self.big = self.w * self.h * 3 > 3_900_000_000
        self.geo = _geo_tags_from_gdal(crs, transform)
        self.f = open(path, "wb")
        if self.big:
            self.f.write(b"II" + struct.pack("<HHHQ", 43, 8, 0, 0))
        else:
            self.f.write(b"II" + struct.pack("<HI", 42, 0))
        self.pos = self.f.tell()

    def _append(self, b):
        if self.pos & 1:                     # word-align data (TIFF convention)
            self.f.write(b"\0")
            self.pos += 1
        o = self.pos
        self.f.write(b)
        self.pos += len(b)
        return o

    def write_tiles(self, tiles, dqt):
        if dqt is not None:
            if self.dqt is None:
                self.dqt = dqt
            elif dqt != self.dqt:
                raise ValueError("JPEG quantisation tables differ between blocks")
        for tx, ty, jpeg, mask in tiles:
            if jpeg is None or tx >= self.ntx or ty >= self.nty:
                continue
            k = ty * self.ntx + tx
            self.off[k] = self._append(jpeg)
            self.cnt[k] = len(jpeg)
            self.moff[k] = self._append(mask)
            self.mcnt[k] = len(mask)

    def close(self):
        ej, em, q = _empty_tiles(self.quality)
        if self.dqt is None:
            self.dqt = q
        empty = self.cnt == 0
        if empty.any():                      # every empty tile points at one shared black tile
            o = self._append(ej)
            om = self._append(em)
            self.off[empty] = o
            self.cnt[empty] = len(ej)
            self.moff[empty] = om
            self.mcnt[empty] = len(em)
        tables = b"\xff\xd8" + self.dqt + b"\xff\xd9"
        ifd1 = self._write_ifd(self._mask_entries(), 0)
        ifd0 = self._write_ifd(self._main_entries(tables), ifd1)
        self.f.seek(8 if self.big else 4)
        self.f.write(struct.pack("<Q" if self.big else "<I", ifd0))
        self.f.close()

    # tag entries: (code, type, count, value bytes)
    def _offs(self, arr):
        return (16, len(arr), arr.astype("<u8").tobytes()) if self.big else (4, len(arr), arr.astype("<u4").tobytes())

    def _common(self):
        sh = lambda v: struct.pack("<" + "H" * len(v), *v)   # noqa: E731
        lg = lambda v: struct.pack("<I", v)                  # noqa: E731
        return sh, lg

    def _main_entries(self, tables):
        sh, lg = self._common()
        e = [(256, 4, 1, lg(self.w)), (257, 4, 1, lg(self.h)), (258, 3, 3, sh((8, 8, 8))), (259, 3, 1, sh((7,))),
             (262, 3, 1, sh((6,))), (277, 3, 1, sh((3,))), (284, 3, 1, sh((1,))), (322, 3, 1, sh((TILE,))),
             (323, 3, 1, sh((TILE,)))]
        e.append((324,) + self._offs(self.off))
        e.append((325,) + self._offs(self.cnt))
        e += [(339, 3, 3, sh((1, 1, 1))), (347, 7, len(tables), tables), (530, 3, 2, sh((2, 2))),
              (532, 5, 6, struct.pack("<12I", 0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1))]
        for code, (typ, cnt, raw) in sorted(self.geo.items()):
            e.append((code, typ, cnt, raw))
        return sorted(e)

    def _mask_entries(self):
        sh, lg = self._common()
        e = [(254, 4, 1, lg(4)), (256, 4, 1, lg(self.w)), (257, 4, 1, lg(self.h)), (258, 3, 1, sh((1,))),
             (259, 3, 1, sh((8,))), (262, 3, 1, sh((4,))), (277, 3, 1, sh((1,))), (284, 3, 1, sh((1,))),
             (317, 3, 1, sh((1,))), (322, 3, 1, sh((TILE,))), (323, 3, 1, sh((TILE,)))]
        e.append((324,) + self._offs(self.moff))
        e.append((325,) + self._offs(self.mcnt))
        e.append((339, 3, 1, sh((1,))))
        return sorted(e)

    def _write_ifd(self, entries, next_ifd):
        inline = 8 if self.big else 4
        # out-of-line values first
        vals = []
        for code, typ, cnt, raw in entries:
            if len(raw) > inline:
                vals.append(self._append(raw))
            else:
                vals.append(None)
        if self.pos & 1:
            self.f.write(b"\0")
            self.pos += 1
        start = self.pos
        out = [struct.pack("<Q", len(entries)) if self.big else struct.pack("<H", len(entries))]
        for (code, typ, cnt, raw), vo in zip(entries, vals):
            if self.big:
                v = struct.pack("<Q", vo) if vo is not None else raw.ljust(8, b"\0")
                out.append(struct.pack("<HHQ", code, typ, cnt) + v)
            else:
                v = struct.pack("<I", vo) if vo is not None else raw.ljust(4, b"\0")
                out.append(struct.pack("<HHI", code, typ, cnt) + v)
        out.append(struct.pack("<Q", next_ifd) if self.big else struct.pack("<I", next_ifd))
        blob = b"".join(out)
        self.f.write(blob)
        self.pos += len(blob)
        return start
