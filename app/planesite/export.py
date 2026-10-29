"""Split the orthomosaic into areas: drawn shapes, grids, or imported boundaries (KML/KMZ/SHP/GeoJSON)."""
import io
import json
import math
import os
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


def new_id():
    return uuid.uuid4().hex[:10]


# ------------------------------------------------------------------ boundary import
def _kml_coords(text):
    pts = []
    for tok in text.replace("\n", " ").replace("\t", " ").split(" "):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts


def _strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def read_kml_bytes(data):
    root = _strip_ns(ET.fromstring(data))
    out = []
    count = 0
    for pm in root.iter("Placemark"):
        name_el = pm.find("name")
        name = (name_el.text or "").strip() if name_el is not None and name_el.text else ""
        polys = list(pm.iter("Polygon"))
        for k, poly in enumerate(polys):
            outer = poly.find("outerBoundaryIs/LinearRing/coordinates")
            if outer is None or not outer.text:
                continue
            rings = [_kml_coords(outer.text)]
            for inner in poly.findall("innerBoundaryIs/LinearRing/coordinates"):
                if inner.text:
                    rings.append(_kml_coords(inner.text))
            count += 1
            nm = name or "Area %d" % count
            if len(polys) > 1:
                nm = "%s (%d)" % (nm, k + 1)
            out.append({"name": nm, "rings": rings})
    return out, "EPSG:4326"


def read_boundaries(path):
    """Return (list of {name, rings[[(x,y),...],...]}, crs_string)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".kml":
        with open(path, "rb") as f:
            return read_kml_bytes(f.read())
    if ext == ".kmz":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not names:
                raise ValueError("KMZ contains no KML file")
            main = "doc.kml" if "doc.kml" in names else names[0]
            return read_kml_bytes(z.read(main))
    if ext in (".geojson", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        crs = "EPSG:4326"
        try:
            nm = gj["crs"]["properties"]["name"]
            m = re.search(r"EPSG::?(\d+)", nm)
            if m:
                crs = "EPSG:" + m.group(1)
        except Exception:
            pass
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        return _from_geo_features(feats), crs
    if ext == ".shp":
        import shapefile
        r = shapefile.Reader(path)
        fields = [f[0] for f in r.fields[1:]]
        name_field = None
        for cand in ("name", "Name", "NAME", "field", "Field", "FIELD", "FIELD_NAME", "FieldName", "label", "id", "ID"):
            if cand in fields:
                name_field = cand
                break
        if name_field is None:
            for f in r.fields[1:]:
                if f[1] == "C":
                    name_field = f[0]
                    break
        feats = []
        for sr in r.iterShapeRecords():
            props = {}
            if name_field:
                props["name"] = str(sr.record[name_field])
            feats.append({"geometry": sr.shape.__geo_interface__, "properties": props})
        prj = os.path.splitext(path)[0] + ".prj"
        crs = None
        if os.path.exists(prj):
            with open(prj, "r", errors="replace") as f:
                crs = f.read()
        out = _from_geo_features(feats)
        if crs is None:
            # guess lon/lat if values are in range
            allpts = [p for a in out for ring in a["rings"] for p in ring]
            if allpts and all(abs(x) <= 180 and abs(y) <= 90 for x, y in allpts):
                crs = "EPSG:4326"
        return out, crs
    raise ValueError("Unsupported boundary file type: %s (use KML, KMZ, SHP or GeoJSON)" % ext)


def _from_geo_features(feats):
    out = []
    count = 0
    for ft in feats:
        geom = ft.get("geometry") or {}
        props = ft.get("properties") or {}
        name = ""
        for k in ("name", "Name", "NAME", "field", "Field", "FIELD", "title", "id"):
            if props.get(k):
                name = str(props[k])
                break
        t = geom.get("type")
        polys = []
        if t == "Polygon":
            polys = [geom["coordinates"]]
        elif t == "MultiPolygon":
            polys = geom["coordinates"]
        for k, poly in enumerate(polys):
            count += 1
            rings = [[(float(p[0]), float(p[1])) for p in ring] for ring in poly]
            nm = name or "Area %d" % count
            if len(polys) > 1:
                nm = "%s (%d)" % (nm, k + 1)
            out.append({"name": nm, "rings": rings})
    return out


def to_ortho_crs(boundaries, src_crs, dst_epsg):
    from pyproj import CRS, Transformer
    if src_crs is None:
        src = CRS.from_epsg(dst_epsg)
    else:
        src = CRS.from_user_input(src_crs)
    tr = Transformer.from_crs(src, CRS.from_epsg(dst_epsg), always_xy=True)
    areas = []
    for b in boundaries:
        rings = []
        for ring in b["rings"]:
            if len(ring) < 3:
                continue
            xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
            rings.append([[float(x), float(y)] for x, y in zip(xs, ys)])
        if rings:
            areas.append({"id": new_id(), "name": b["name"], "kind": "imported", "rings": rings})
    return areas


# ----------------------------------------------------------------------- grid split
def grid_areas(ortho_path, mode="size", size_m=250.0, rows=2, cols=2, prefix="Grid"):
    import rasterio
    with rasterio.open(ortho_path) as ds:
        b = ds.bounds
        ov = ds.overviews(1)
        fac = ov[-1] if ov else 1
        mh, mw = max(1, ds.height // fac), max(1, ds.width // fac)
        mask = ds.read_masks(1, out_shape=(mh, mw)) > 0
        # tighten to data extent
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []
        sx = (b.right - b.left) / mw
        sy = (b.top - b.bottom) / mh
        left = b.left + xs.min() * sx
        right = b.left + (xs.max() + 1) * sx
        top = b.top - ys.min() * sy
        bottom = b.top - (ys.max() + 1) * sy
        if mode == "size":
            ncols = max(1, int(math.ceil((right - left) / size_m)))
            nrows = max(1, int(math.ceil((top - bottom) / size_m)))
            cw = ch = size_m
            # centre the grid on the data
            left = (left + right) / 2 - ncols * cw / 2
            top = (top + bottom) / 2 + nrows * ch / 2
        else:
            ncols, nrows = max(1, int(cols)), max(1, int(rows))
            cw = (right - left) / ncols
            ch = (top - bottom) / nrows
        areas = []
        for r in range(nrows):
            for c in range(ncols):
                x0 = left + c * cw
                x1 = x0 + cw
                y1 = top - r * ch
                y0 = y1 - ch
                # skip empty cells
                c0 = int(max(0, (x0 - b.left) / sx))
                c1 = int(min(mw, math.ceil((x1 - b.left) / sx)))
                r0 = int(max(0, (b.top - y1) / sy))
                r1 = int(min(mh, math.ceil((b.top - y0) / sy)))
                if c1 <= c0 or r1 <= r0 or not mask[r0:r1, c0:c1].any():
                    continue
                areas.append({"id": new_id(), "name": "%s_R%d_C%d" % (prefix, r + 1, c + 1), "kind": "grid",
                              "rings": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]})
        return areas


# --------------------------------------------------------------------------- export
def safe_filename(s):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip().strip(".")
    return s or "area"


def _gdal_metadata(items):
    # GDAL escapes item values twice (its own escaping, then the XML writer's)
    xml = "<GDALMetadata>\n" + "".join('  <Item name="%s">%s</Item>\n' % (k, _xml_escape(_xml_escape(v)))
                                        for k, v in items.items()) + "</GDALMetadata>\n"
    raw = xml.encode("utf-8") + b"\0"
    return {42112: (2, len(raw), raw)}


def _jpeg_quality_of(tables, hint):
    """The cv2 JPEG quality whose quantisation tables are exactly the file's JPEGTables, or None."""
    import cv2
    from .tiffwriter import _split_jpeg
    img = np.zeros((16, 16, 3), np.uint8)
    for q in [int(hint)] + list(range(100, 0, -1)):
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        dqt, _ = _split_jpeg(enc.tobytes())
        if b"\xff\xd8" + dqt + b"\xff\xd9" == tables:
            return q
    return None


def _read_clipped(d, x, y, w, h):
    """RGB and mask of the window, zero outside the image. (A boundless read goes through a VRT
    of the file, which loses the overview level the dataset was opened at.)"""
    from rasterio.windows import Window
    rgb = np.zeros((3, h, w), np.uint8)
    m = np.zeros((h, w), bool)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(d.width, x + w), min(d.height, y + h)
    if x1 > x0 and y1 > y0:
        win = Window(x0, y0, x1 - x0, y1 - y0)
        rgb[:, y0 - y:y1 - y, x0 - x:x1 - x] = d.read([1, 2, 3], window=win)
        m[y0 - y:y1 - y, x0 - x:x1 - x] = d.read_masks(1, window=win) > 127
    return rgb, m


def _threads():
    return max(2, min(16, (os.cpu_count() or 4)))


def _export_area_fast(ortho_path, area, out_path, jpeg_quality, progress, overviews):
    """Area export without recompressing the inside of the field: the output is aligned to the
    orthomosaic's 512-pixel tiles, tiles wholly inside the field are copied byte for byte, only
    tiles on the field edge are decoded, cut and compressed again (in parallel), and overviews
    are cut from the orthomosaic's own overviews. Returns None when the file isn't suitable."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask, rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from . import tiffcopy
    from .tiffwriter import JpegTiffWriter, encode_tiles
    T = 512
    img, msk, tile = tiffcopy.tile_index(ortho_path)
    comp = img.entries.get(259)
    if tile != T or msk is None or comp is None or comp[2][:2] != b"\x07\x00" or 347 not in img.entries:
        return None
    q = _jpeg_quality_of(img.entries[347][2], jpeg_quality)
    if q is None:
        return None
    label = area["name"]
    geom = {"type": "Polygon", "coordinates": area["rings"]}
    with rasterio.open(ortho_path) as src:
        W, H = src.width, src.height
        crs, str_ = src.crs, src.transform
        ovs = src.overviews(1)
    inv = ~str_
    pts = np.array([p for ring in area["rings"] for p in ring], float)
    cols, rows = inv * (pts[:, 0], pts[:, 1])
    c0 = max(0, int(math.floor(min(cols) / T)) * T)
    r0 = max(0, int(math.floor(min(rows) / T)) * T)
    c1 = min(W, int(math.ceil(max(cols) / T)) * T)
    r1 = min(H, int(math.ceil(max(rows) / T)) * T)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("Area '%s' does not overlap the orthomosaic" % label)
    Wo, Ho = c1 - c0, r1 - r0
    tr = str_ * Affine.translation(c0, r0)
    ntx, nty = (Wo + T - 1) // T, (Ho + T - 1) // T
    src_ntx = (W + T - 1) // T
    # which tiles are wholly inside the field: not crossed by its outline, centre inside
    tgrid = tr * Affine.scale(T)
    lines = [{"type": "LineString", "coordinates": ring} for ring in area["rings"]]
    edge = rasterize([(g, 1) for g in lines], out_shape=(nty, ntx), transform=tgrid, all_touched=True,
                     dtype="uint8") > 0
    centre = geometry_mask([geom], out_shape=(nty, ntx), transform=tgrid, invert=True)
    shared = {}
    for o in img.offsets:
        shared[o] = shared.get(o, 0) + 1
    tls = threading.local()

    def ds():
        d = getattr(tls, "ds", None)
        if d is None:
            d = tls.ds = rasterio.open(ortho_path)
        return d

    tmp_files = []
    nthreads = _threads()
    level_list = []
    f = 2
    while overviews and max(Wo, Ho) / f >= 256:
        level_list.append(f)
        f *= 2
    # rough work shares for the progress bar
    w_main = 0.5 if level_list else 0.9
    w_lvl = 0.4 if level_list else 0.0
    any_data = [False]
    try:
        main_tmp = out_path + ".main.part"
        tmp_files.append(main_tmp)
        wr = JpegTiffWriter(main_tmp, Wo, Ho, crs, tr, q)
        copy, cut = [], []
        for ty in range(nty):
            for tx in range(ntx):
                if edge[ty, tx]:
                    cut.append((tx, ty))
                elif centre[ty, tx]:
                    copy.append((tx, ty))
        ntiles = max(1, len(copy) + len(cut))
        done = [0]
        # 1) tiles inside the field: copy the compressed bytes (in file order)
        keys = []
        for tx, ty in copy:
            k = (r0 // T + ty) * src_ntx + (c0 // T + tx)
            if img.counts[k] and shared[img.offsets[k]] == 1:
                keys.append((img.offsets[k], k, tx, ty))
        keys.sort()
        with open(ortho_path, "rb") as fh:
            batch = []
            for o, k, tx, ty in keys:
                fh.seek(o)
                jb = fh.read(img.counts[k])
                fh.seek(msk.offsets[k])
                mb = fh.read(msk.counts[k])
                batch.append((tx, ty, jb, mb))
                if len(batch) >= 64:
                    wr.write_tiles(batch, None)
                    done[0] += len(batch)
                    batch = []
                    if progress:
                        progress(w_main * done[0] / ntiles)
            wr.write_tiles(batch, None)
            done[0] += len(batch)
        any_data[0] = bool(keys)

        # 2) tiles on the field edge: decode, cut along the outline, compress again
        def cut_tile(t):
            tx, ty = t
            d = ds()
            rgb, m = _read_clipped(d, c0 + tx * T, r0 + ty * T, T, T)
            inside = geometry_mask([geom], out_shape=(T, T), transform=tr * Affine.translation(tx * T, ty * T),
                                   invert=True)
            a = m & inside
            if not a.any():
                return [], None
            rgb[:, ~a] = 0
            return encode_tiles((tx * T, ty * T, T, T), rgb.transpose(1, 2, 0), a.astype(np.uint8) * 255, q)

        with ThreadPoolExecutor(nthreads) as ex:
            for tiles, dqt in ex.map(cut_tile, cut):
                if tiles:
                    wr.write_tiles(tiles, dqt)
                    any_data[0] = True
                done[0] += 1
                if progress and done[0] % 16 == 0:
                    progress(w_main * done[0] / ntiles)
        wr.close()
        if not any_data[0]:
            raise ValueError("Area '%s' contains no orthomosaic data" % label)

        # 3) overviews, cut from the orthomosaic's overviews (the output starts on a tile corner,
        #    so its overview pixels are exactly the orthomosaic's)
        level_tmps = []
        for li, f in enumerate(level_list):
            wf, hf = (Wo + f - 1) // f, (Ho + f - 1) // f
            tmp = out_path + ".ov%d.part" % f
            tmp_files.append(tmp)
            level_tmps.append(tmp)
            lw = JpegTiffWriter(tmp, wf, hf, crs, tr * Affine.scale(f), q)
            CH = 2048
            chunks = [(x, y) for y in range(0, hf, CH) for x in range(0, wf, CH)]
            lvl = ovs.index(f) if f in ovs else None

            def cut_chunk(c, f=f, wf=wf, hf=hf, lvl=lvl):
                x, y = c
                cw, chh = min(CH, wf - x), min(CH, hf - y)
                if lvl is not None:
                    d = getattr(tls, "ov%d" % f, None)
                    if d is None:
                        d = rasterio.open(ortho_path, overview_level=lvl)
                        setattr(tls, "ov%d" % f, d)
                    rgb, m = _read_clipped(d, c0 // f + x, r0 // f + y, cw, chh)
                else:
                    d = ds()
                    win = Window(c0 + x * f, r0 + y * f, cw * f, chh * f)
                    rgb = d.read([1, 2, 3], window=win, out_shape=(3, chh, cw), boundless=True,
                                 resampling=Resampling.average)
                    m = d.read_masks(1, window=win, out_shape=(chh, cw), boundless=True) > 127
                inside = geometry_mask([geom], out_shape=(chh, cw),
                                       transform=tr * Affine.scale(f) * Affine.translation(x, y), invert=True)
                a = m & inside
                if not a.any():
                    return [], None
                rgb[:, ~a] = 0
                return encode_tiles((x, y, cw, chh), rgb.transpose(1, 2, 0), a.astype(np.uint8) * 255, q)

            with ThreadPoolExecutor(nthreads) as ex:
                for i, (tiles, dqt) in enumerate(ex.map(cut_chunk, chunks)):
                    if tiles:
                        lw.write_tiles(tiles, dqt)
                    if progress:
                        share = 0.75 if li == 0 else 0.25 / max(1, len(level_list) - 1)
                        base = 0 if li == 0 else 0.75 + 0.25 * (li - 1) / max(1, len(level_list) - 1)
                        progress(w_main + w_lvl * (base + share * (i + 1) / len(chunks)))
            lw.close()

        # 4) one GeoTIFF with internal overviews
        res = tiffcopy.merge(main_tmp, level_tmps, out_path,
                             progress=(lambda p: progress(w_main + w_lvl + (1 - w_main - w_lvl) * p)) if progress else None,
                             extra_tags=_gdal_metadata({"AREA_NAME": area["name"]}))
    finally:
        for p_ in tmp_files:
            try:
                os.remove(p_)
            except OSError:
                pass
        for k in list(vars(tls)):
            try:
                getattr(tls, k).close()
            except Exception:
                pass
    return {"path": out_path, "width": Wo, "height": Ho, "fast": True, "bytes": res["bytes"]}


def export_area(ortho_path, area, out_path, compression="same", jpeg_quality=90, progress=None,
                scale=1.0, overviews=True):
    """Write the part of the orthomosaic inside area's polygon to out_path (GeoTIFF).
    area=None exports the whole orthomosaic. scale < 1 writes a coarser copy (e.g. 0.5 = half the pixels
    on each side).

    At full resolution in the orthomosaic's own compression the tiles are copied without
    recompressing (fast, no extra JPEG loss): the whole orthomosaic as is, an area cut on the
    orthomosaic's 512-pixel tile grid (the extra margin is transparent)."""
    scale = float(scale or 1.0)
    if scale >= 0.999 and _same_codec(ortho_path, compression) and not os.environ.get("PLANESITE_SLOW_EXPORT"):
        from . import tiffcopy
        try:
            if area is None:
                r = tiffcopy.merge_with_overviews(ortho_path, out_path, progress, overviews)
                return {"path": out_path, "fast": True, "bytes": r["bytes"]}
            r = _export_area_fast(ortho_path, area, out_path, jpeg_quality, progress, overviews)
            if r is not None:
                return r
        except ValueError as e:
            if "no orthomosaic data" in str(e) or "does not overlap" in str(e):
                raise
            _log_fast_fail(e)
        except Exception as e:      # anything unexpected: fall back to the plain export
            _log_fast_fail(e)
    return _export_area_gdal(ortho_path, area, out_path, compression, jpeg_quality, progress, scale, overviews)


FAST_FAILURES = []     # reasons the fast export wasn't used (shown in diagnostics)


def _same_codec(ortho_path, compression):
    """Whether the export asks for the orthomosaic's own compression (the export window names it:
    "jpeg" or "deflate"; "same" means keep it)."""
    if compression in ("same", None, ""):
        return True
    try:
        import rasterio
        with rasterio.open(ortho_path) as d:
            c = d.compression.value.lower() if d.compression is not None else ""
    except Exception:
        return False
    return str(compression).lower() == c


def _log_fast_fail(e):
    import sys
    FAST_FAILURES.append("%s: %s" % (type(e).__name__, e))
    print("fast export not used: %s" % e, file=sys.stderr)


def _export_area_gdal(ortho_path, area, out_path, compression="same", jpeg_quality=90, progress=None,
                      scale=1.0, overviews=True):
    import rasterio
    from rasterio.enums import Resampling, ColorInterp
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine
    from rasterio.windows import Window, from_bounds, transform as win_transform
    scale = float(scale or 1.0)
    label = area["name"] if area else "orthomosaic"
    with rasterio.open(ortho_path) as src:
        full = Window(0, 0, src.width, src.height)
        if area is not None:
            geom = {"type": "Polygon", "coordinates": area["rings"]}
            pts = np.array([p for ring in area["rings"] for p in ring])
            win = from_bounds(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), src.transform)
            win = win.round_offsets(op="floor").round_lengths(op="ceil")
            try:
                win = win.intersection(full)
            except Exception:
                raise ValueError("Area '%s' does not overlap the orthomosaic" % label)
        else:
            geom = None
            win = full
        if win.width <= 0 or win.height <= 0:
            raise ValueError("Area '%s' does not overlap the orthomosaic" % label)
        col0, row0 = int(win.col_off), int(win.row_off)
        W, H = int(win.width), int(win.height)
        # keep square pixels with an exact size (scale 0.5 of 1.4 cm -> 2.8 cm)
        Wo, Ho = max(1, int(W * scale + 1e-6)), max(1, int(H * scale + 1e-6))
        sx = sy = 1.0 / scale
        tr = win_transform(win, src.transform) * Affine.scale(sx, sy)
        src_jpeg = (src.compression is not None and src.compression.value == "JPEG")
        comp = compression
        if comp == "same":
            comp = "jpeg" if src_jpeg else "deflate"
        prof = dict(driver="GTiff", width=Wo, height=Ho, dtype="uint8", crs=src.crs, transform=tr, tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        if comp == "jpeg":
            prof.update(count=3, compress="JPEG", photometric="YCBCR", jpeg_quality=int(jpeg_quality))
        else:
            prof.update(count=4, compress="DEFLATE", predictor=2, photometric="RGB", ALPHA="YES")
        tmp = out_path + ".part"
        any_data = False
        rs = Resampling.average if scale < 0.999 else Resampling.nearest
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_NUM_THREADS="ALL_CPUS"):
            with rasterio.open(tmp, "w", **prof) as dst:
                if comp != "jpeg":
                    dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
                step = 2048
                nblocks = int(math.ceil(Ho / step))
                for bi, r in enumerate(range(0, Ho, step)):
                    h = min(step, Ho - r)
                    sr0 = row0 + r * sy
                    sr1 = min(row0 + H, row0 + (r + h) * sy)
                    rw = Window(col0, sr0, min(W, Wo * sx), sr1 - sr0)
                    rgb = src.read([1, 2, 3], window=rw, out_shape=(3, h, Wo), resampling=rs)
                    m = src.read_masks(1, window=rw, out_shape=(h, Wo), resampling=rs) > 127
                    alpha = m
                    if geom is not None:
                        btr = win_transform(Window(0, r, Wo, h), tr)
                        inside = geometry_mask([geom], out_shape=(h, Wo), transform=btr, invert=True,
                                               all_touched=False)
                        alpha = m & inside
                    any_data = any_data or bool(alpha.any())
                    rgb[:, ~alpha] = 0
                    ow = Window(0, r, Wo, h)
                    if comp == "jpeg":
                        dst.write(rgb, window=ow)
                        dst.write_mask((alpha * 255).astype(np.uint8), window=ow)
                    else:
                        dst.write(np.concatenate([rgb, (alpha * 255).astype(np.uint8)[None]], 0), window=ow)
                    if progress:
                        progress((bi + 1) / nblocks * (0.85 if overviews else 0.98))
                facs = []
                f = 2
                while overviews and max(Wo, Ho) / f >= 256:
                    facs.append(f)
                    f *= 2
                if facs:
                    dst.build_overviews(facs, Resampling.average)
                    dst.update_tags(ns="rio_overview", resampling="average")
                if area is not None:
                    dst.update_tags(AREA_NAME=area["name"])
        if not any_data:
            os.remove(tmp)
            raise ValueError("Area '%s' contains no orthomosaic data" % label)
        if os.path.exists(out_path):
            os.remove(out_path)
        from .pipeline import replace_retry
        replace_retry(tmp, out_path)
        if progress:
            progress(1.0)
    return {"path": out_path, "width": Wo, "height": Ho}


def area_stats(area):
    """Area in hectares and acres for a polygon in metres."""
    def ring_area(r):
        p = np.array(r)
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    rings = area["rings"]
    a = ring_area(rings[0]) - sum(ring_area(r) for r in rings[1:])
    return {"ha": round(a / 1e4, 3), "ac": round(a / 4046.8564224, 3)}


def areas_to_kml(areas, epsg):
    from pyproj import Transformer
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    out = io.StringIO()
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
    for a in areas:
        out.write("<Placemark><name>%s</name><Polygon>" % _xml_escape(a["name"]))
        for k, ring in enumerate(a["rings"]):
            xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
            coords = " ".join("%.8f,%.8f,0" % (x, y) for x, y in zip(xs, ys))
            tag = "outerBoundaryIs" if k == 0 else "innerBoundaryIs"
            out.write("<%s><LinearRing><coordinates>%s</coordinates></LinearRing></%s>" % (tag, coords, tag))
        out.write("</Polygon></Placemark>\n")
    out.write("</Document></kml>\n")
    return out.getvalue()


def _xml_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
