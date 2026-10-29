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


def export_area(ortho_path, area, out_path, compression="same", jpeg_quality=90, progress=None,
                scale=1.0, overviews=True):
    """Write the part of the orthomosaic inside area's polygon to out_path (GeoTIFF).
    area=None exports the whole orthomosaic. scale < 1 writes a coarser copy (e.g. 0.5 = half the pixels
    on each side)."""
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
