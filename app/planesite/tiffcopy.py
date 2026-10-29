"""Export the whole orthomosaic without decoding it.

The rendered orthomosaic is a tiled JPEG GeoTIFF (<name>.tif) with its overviews in a separate
file (<name>.tif.ovr). Exporting it used to read every pixel back, compress it again and rebuild
the overviews: as slow as a second render for a large site, and a second round of JPEG loss.

Here the compressed tiles of both files are copied byte for byte into one GeoTIFF with internal
overviews, and only the TIFF directories are rewritten (BigTIFF when the result needs it). The
copy runs at disk speed and the pixels are identical to the rendered orthomosaic.

Layout of the result: header, the tile data of the main file, the tile data of the overview file,
then all the directories in GDAL's order (image, its mask, each overview and its mask). Overview
directories get NewSubfileType 1 (reduced image) and their masks 5 (reduced image + mask), as GDAL
writes them.
"""
import os
import struct

# TIFF field types: size in bytes
_TSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_OFFSET_TAGS = (273, 324)        # StripOffsets, TileOffsets
_COUNT_TAGS = (279, 325)         # StripByteCounts, TileByteCounts
_NEWSUBFILETYPE = 254
_SUBIFDS = 330
_UNSUPPORTED = (_SUBIFDS, 34665, 34853)   # sub-IFDs, EXIF and GPS directories point to other IFDs


class _Ifd:
    def __init__(self):
        self.entries = {}       # tag -> (type, count, raw value bytes)
        self.offsets = None     # tile/strip offsets (ints)
        self.counts = None      # tile/strip byte counts (ints)
        self.off_tag = None
        self.cnt_tag = None

    def subfile(self):
        e = self.entries.get(_NEWSUBFILETYPE)
        return struct.unpack("<I", e[2][:4])[0] if e else 0

    def set_subfile(self, v):
        self.entries[_NEWSUBFILETYPE] = (4, 1, struct.pack("<I", v))


def tile_index(path):
    """(image IFD, mask IFD or None, tile size) of the full-resolution image."""
    ifds = read_ifds(path)
    full = [d for d in ifds if not d.subfile() & 1]
    img = full[0]
    msk = next((d for d in full[1:] if d.subfile() & 4 and _dims(d) == _dims(img)), None)
    tw = img.entries.get(322)
    th = img.entries.get(323)
    if not tw or not th:
        return img, msk, None
    tw = struct.unpack("<H" if tw[0] == 3 else "<I", tw[2][:2 if tw[0] == 3 else 4])[0]
    th = struct.unpack("<H" if th[0] == 3 else "<I", th[2][:2 if th[0] == 3 else 4])[0]
    return img, msk, (tw if tw == th else None)


def read_ifds(path):
    """All directories of a little-endian (Big)TIFF, with their values loaded."""
    out = []
    with open(path, "rb") as f:
        hdr = f.read(16)
        if hdr[:2] != b"II":
            raise ValueError("not a little-endian TIFF")
        ver = struct.unpack("<H", hdr[2:4])[0]
        if ver == 42:
            big = False
            nxt = struct.unpack("<I", hdr[4:8])[0]
        elif ver == 43:
            big = True
            nxt = struct.unpack("<Q", hdr[8:16])[0]
        else:
            raise ValueError("not a TIFF")
        seen = set()
        while nxt:
            if nxt in seen:
                raise ValueError("TIFF directory loop")
            seen.add(nxt)
            f.seek(nxt)
            if big:
                n = struct.unpack("<Q", f.read(8))[0]
                esz, inl = 20, 8
            else:
                n = struct.unpack("<H", f.read(2))[0]
                esz, inl = 12, 4
            data = f.read(n * esz + (8 if big else 4))
            ifd = _Ifd()
            for i in range(n):
                e = data[i * esz:(i + 1) * esz]
                tag, typ = struct.unpack("<HH", e[:4])
                if big:
                    cnt = struct.unpack("<Q", e[4:12])[0]
                    val = e[12:20]
                else:
                    cnt = struct.unpack("<I", e[4:8])[0]
                    val = e[8:12]
                size = _TSIZE.get(typ, 1) * cnt
                if size <= inl:
                    raw = val[:size]
                else:
                    off = struct.unpack("<Q" if big else "<I", val)[0]
                    pos = f.tell()
                    f.seek(off)
                    raw = f.read(size)
                    f.seek(pos)
                if tag in _UNSUPPORTED:
                    raise ValueError("TIFF tag %d is not supported here" % tag)
                ifd.entries[tag] = (typ, cnt, raw)
            nxt = struct.unpack("<Q" if big else "<I", data[n * esz:])[0]
            for tags, attr in ((_OFFSET_TAGS, "offsets"), (_COUNT_TAGS, "counts")):
                for t in tags:
                    if t in ifd.entries:
                        typ, cnt, raw = ifd.entries.pop(t)
                        fmt = {3: "H", 4: "I", 16: "Q"}[typ]
                        setattr(ifd, attr, list(struct.unpack("<%d%s" % (cnt, fmt), raw)))
                        setattr(ifd, "off_tag" if attr == "offsets" else "cnt_tag", t)
            if ifd.offsets is None or ifd.counts is None or len(ifd.offsets) != len(ifd.counts):
                raise ValueError("TIFF directory without tiles")
            out.append(ifd)
    return out


def _data_span(path, ifds):
    """(start, end) byte range holding every tile of the file."""
    lo, hi = None, 0
    for d in ifds:
        for o, c in zip(d.offsets, d.counts):
            if c:
                lo = o if lo is None else min(lo, o)
                hi = max(hi, o + c)
    if lo is None:
        lo = hi = 0
    return lo, hi


def _pack_ifd(d, pos, delta, next_ifd, big=True):
    """Directory for d placed at pos; tile offsets shifted by delta. Returns bytes."""
    ents = dict(d.entries)
    offs = [o + delta if c else 0 for o, c in zip(d.offsets, d.counts)]
    if big:
        ents[d.off_tag] = (16, len(offs), struct.pack("<%dQ" % len(offs), *offs))
        ents[d.cnt_tag] = (16, len(d.counts), struct.pack("<%dQ" % len(d.counts), *d.counts))
    else:
        ents[d.off_tag] = (4, len(offs), struct.pack("<%dI" % len(offs), *offs))
        ents[d.cnt_tag] = (4, len(d.counts), struct.pack("<%dI" % len(d.counts), *d.counts))
    tags = sorted(ents)
    n = len(tags)
    esz, inl = (20, 8) if big else (12, 4)
    head = (8 + n * 20 + 8) if big else (2 + n * 12 + 4)
    extra = bytearray()
    body = bytearray(struct.pack("<Q" if big else "<H", n))
    for t in tags:
        typ, cnt, raw = ents[t]
        if big:
            body += struct.pack("<HHQ", t, typ, cnt)
        else:
            if typ in (16, 17, 18):
                raise ValueError("64-bit TIFF values need BigTIFF")
            body += struct.pack("<HHI", t, typ, cnt)
        if len(raw) <= inl:
            body += raw + b"\0" * (inl - len(raw))
        else:
            if (pos + head + len(extra)) % 2:
                extra += b"\0"
            body += struct.pack("<Q" if big else "<I", pos + head + len(extra))
            extra += raw
    body += struct.pack("<Q" if big else "<I", next_ifd)
    return bytes(body) + bytes(extra)


def _copy_range(src, dst, start, end, progress, done, total, chunk=64 << 20):
    src.seek(start)
    left = end - start
    while left > 0:
        b = src.read(min(chunk, left))
        if not b:
            raise IOError("unexpected end of file")
        dst.write(b)
        left -= len(b)
        done[0] += len(b)
        if progress:
            progress(done[0] / float(total))


_GEO_TAGS = (33550, 33922, 34264, 34735, 34736, 34737)
CLASSIC_LIMIT = 4_000_000_000


def _dims(d):
    def one(t):
        typ, cnt, raw = d.entries[t]
        return struct.unpack("<H" if typ == 3 else "<I", raw[:2 if typ == 3 else 4])[0]
    return one(256), one(257)


def merge(main_path, level_paths, out_path, progress=None, extra_tags=None, overviews=True):
    """Write the image of main_path plus every image in level_paths (overview files: a .ovr, or
    one file per level) as one GeoTIFF with internal overviews. Tiles are copied unchanged.
    extra_tags: {tag: (type, count, raw)} added to the full-resolution directory."""
    files = [(main_path, read_ifds(main_path))] + [(p, read_ifds(p)) for p in level_paths]
    spans = [_data_span(p, ifds) for p, ifds in files]
    total = sum(b - a for a, b in spans)
    main = files[0][1]
    full = [d for d in main if not d.subfile() & 1]
    if not full or full[0].subfile() & 4:
        raise ValueError("no full-resolution image in %s" % main_path)
    image = full[0]
    if extra_tags:
        image.entries.update(extra_tags)
    owner = {}                                  # id(ifd) -> file index
    for k, (p, ifds) in enumerate(files):
        for d in ifds:
            owner[id(d)] = k
    order = [(image, 0)]
    mk = [d for d in full[1:] if d.subfile() & 4 and _dims(d) == _dims(image)]
    order += [(d, 4) for d in mk[:1]]
    used = set(id(d) for d, _ in order)
    rest = [d for p, ifds in files for d in ifds if id(d) not in used and _dims(d)[0] < _dims(image)[0]]
    if not overviews:
        rest = []
    imgs = sorted([d for d in rest if not d.subfile() & 4], key=lambda d: -_dims(d)[0])
    msks = [d for d in rest if d.subfile() & 4]
    for d in imgs:
        for g in _GEO_TAGS:                     # GDAL keeps the georeferencing on the main image only
            d.entries.pop(g, None)
        order.append((d, 1))
        for m in msks:
            if _dims(m) == _dims(d):
                order.append((m, 5))
                msks.remove(m)
                break
    for d, sub in order:
        d.set_subfile(sub)
    # classic TIFF when everything fits (some farm software doesn't read BigTIFF)
    ifd_guess = sum(16 * len(d.offsets) + 4096 for d, _ in order)
    big = total + ifd_guess + 16 > CLASSIC_LIMIT
    tmp = out_path + ".part"
    with open(tmp, "wb") as out:
        if big:
            out.write(b"II" + struct.pack("<HHHQ", 43, 8, 0, 0))
        else:
            out.write(b"II" + struct.pack("<HI", 42, 0))
        done = [0]
        delta = {}
        for k, (p, ifds) in enumerate(files):
            a, b = spans[k]
            if out.tell() % 2:
                out.write(b"\0")
            delta[k] = out.tell() - a
            with open(p, "rb") as src:
                _copy_range(src, out, a, b, progress, done, max(1, total))
        if out.tell() % 2:
            out.write(b"\0")
        pos = first = out.tell()
        packed = []
        for d, _ in order:
            dl = delta[owner[id(d)]]
            b = _pack_ifd(d, pos, dl, 0, big)
            packed.append([d, dl, pos, b])
            pos += len(b) + (len(b) % 2)
        for i, item in enumerate(packed):
            nxt = packed[i + 1][2] if i + 1 < len(packed) else 0
            if nxt:
                item[3] = _pack_ifd(item[0], item[2], item[1], nxt, big)
            out.write(item[3])
            if len(item[3]) % 2:
                out.write(b"\0")
        if big:
            out.seek(8)
            out.write(struct.pack("<Q", first))
        else:
            if first >= 1 << 32:
                raise ValueError("file too large for classic TIFF")
            out.seek(4)
            out.write(struct.pack("<I", first))
    if os.path.exists(out_path):
        os.remove(out_path)
    os.replace(tmp, out_path)
    if progress:
        progress(1.0)
    return {"path": out_path, "levels": len(imgs), "bytes": os.path.getsize(out_path), "bigtiff": big}


def merge_with_overviews(path, out_path, progress=None, overviews=True, extra_tags=None):
    """Write path (+ path.ovr if present) as one GeoTIFF with internal overviews."""
    ovr = path + ".ovr"
    levels = [ovr] if overviews and os.path.exists(ovr) else []
    return merge(path, levels, out_path, progress, extra_tags, overviews)
