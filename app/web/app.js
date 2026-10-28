"use strict";
/* FieldStitch front end. Map frame: lat = northing offset (m), lng = easting offset (m) from the project origin. */

const $ = (id) => document.getElementById(id);
const R0 = 64;
const PRESET_HINTS = {
  fast: "Quickest. Output at half the photo resolution, good for scouting and same-day checks.",
  balanced: "Full-resolution output with quick alignment. The everyday choice for field maps.",
  accurate: "Finer feature matching for the tightest seams. Takes roughly twice as long.",
};
const STAGES = ["features", "matching", "alignment", "surface", "ortho"];

let status = null;
let photos = null;
let result = null;
let areas = [];
let checked = new Set();
let map = null;
let layers = {};
let tool = null;
let draw = null;
let pollTimer = null;
let lastProjectDir = null;
let lastResultMtime = null;
let lastProgressStatus = null;
let scanWasRunning = false;
let exportWasRunning = false;

// ------------------------------------------------------------------ utils
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let data = {};
  try { data = await r.json(); } catch (e) { /* empty */ }
  if (!r.ok) throw new Error(data.error || ("Request failed (" + r.status + ")"));
  return data;
}
function toast(msg, ms = 3500) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}
const fmt = (n, d = 0) => Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
function fmtTime(s) {
  if (s == null || !isFinite(s)) return "";
  s = Math.round(s);
  if (s < 60) return s + " s";
  const m = Math.floor(s / 60);
  if (m < 60) return m + " min " + (s % 60) + " s";
  return Math.floor(m / 60) + " h " + (m % 60) + " min";
}
function sep() { return status && status.platform === "win32" ? "\\" : "/"; }
function setErr(id, msg) { $(id).textContent = msg || ""; }
function facts(el, rows) {
  el.innerHTML = "";
  for (const [k, v, cls] of rows) {
    if (v === null || v === undefined || v === "") continue;
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    if (cls) dd.className = cls;
    el.append(dt, dd);
  }
}

// ------------------------------------------------------------------ polling
async function refresh() {
  try {
    status = await api("/api/status");
  } catch (e) {
    if (!updRestarting) toast("Lost contact with FieldStitch. Start it again from the Start menu or desktop icon.");
    schedule(4000);
    return;
  }
  render();
  const busy = status.running || (status.scan && !status.scan.finished) || (status.export && !status.export.finished);
  schedule(busy ? 1000 : 3000);
}
function schedule(ms) { clearTimeout(pollTimer); pollTimer = setTimeout(refresh, ms); }

// ------------------------------------------------------------------ render
function render() {
  const p = status.project;
  $("welcome").hidden = !!p;
  $("rail").hidden = !p;
  $("mapwrap").hidden = !p;
  $("projTitle").hidden = !p;
  $("topActions").hidden = !p;
  if (!p) {
    lastProjectDir = null;
    renderWelcome();
    return;
  }
  if (p.dir !== lastProjectDir) {
    lastProjectDir = p.dir;
    photos = null; result = null; areas = []; checked = new Set(); lastResultMtime = null; lastProgressStatus = null;
    initMap();
    clearLayers();
    loadPhotos(true);
    loadResult(true);
    applySettings(p.settings);
    $("exportDir").value = p.dir + sep() + "exports";
  }
  $("projName").textContent = p.name;
  $("projDir").textContent = p.dir;
  renderPhotos(p);
  renderProcess(p);
  renderExport();
}

function renderWelcome() {
  if (!$("newParent").value) $("newParent").value = status.default_dir;
  const ul = $("recentList");
  ul.innerHTML = "";
  if (!status.recent.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Projects you create or open will be listed here.";
    ul.append(li);
  }
  for (const d of status.recent) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    const name = d.split(/[\\/]/).pop();
    b.innerHTML = "<strong></strong><small></small>";
    b.querySelector("strong").textContent = name.replace(/_/g, " ");
    b.querySelector("small").textContent = d;
    b.onclick = () => openProject(d);
    li.append(b);
    ul.append(li);
  }
}

// ------------------------------------------------------------------ photos
function renderPhotos(p) {
  const ul = $("folderList");
  ul.innerHTML = "";
  for (const f of p.folders) {
    const li = document.createElement("li");
    const s = document.createElement("span");
    const parts = f.split(/[\\/]/).filter(Boolean);
    const bName = document.createElement("b");
    bName.textContent = parts.pop() || f;
    const sm = document.createElement("small");
    const parent = parts.join(sep());
    sm.textContent = parent.length > 42 ? "…" + parent.slice(-41) : parent;
    s.append(bName, sm);
    s.title = f;
    const x = document.createElement("button");
    x.textContent = "×";
    x.title = "Remove these photos from the project";
    x.setAttribute("aria-label", "Remove folder " + f);
    x.onclick = async () => {
      if (status.running) return toast("Cancel processing first.");
      await api("/api/photos/remove_folder", { folder: f });
      await loadPhotos(true);
      refresh();
    };
    li.append(s, x);
    ul.append(li);
  }
  const sc = status.scan;
  const scanning = sc && !sc.finished;
  $("scanBox").hidden = !scanning;
  $("folderAdd").disabled = scanning || status.running;
  if (scanning) {
    $("scanBar").style.width = (sc.total ? (100 * sc.done) / sc.total : 2) + "%";
    $("scanText").textContent = sc.total ? "Reading photo " + fmt(sc.done) + " of " + fmt(sc.total) : "Looking for photos…";
  }
  if (scanWasRunning && !scanning && sc) {
    if (sc.error) setErr("photoErr", sc.error);
    else {
      toast(sc.added ? "Added " + fmt(sc.added) + " photos." : "No new photos found in that folder.");
      if (sc.errors) setErr("photoErr", fmt(sc.errors) + " files could not be read, e.g. " + (sc.error_examples || []).join("; "));
      $("folderPath").value = "";
    }
    loadPhotos(true);
  }
  scanWasRunning = scanning;

  const cams = Object.entries(p.cameras || {});
  const rows = [["Photos", fmt(p.count)]];
  if (p.count) {
    rows.push(["With GPS position", fmt(p.with_gps), p.with_gps < p.count ? "warn" : ""]);
    rows.push(["RTK fixed", fmt(p.rtk_fixed)]);
    if (p.oblique) rows.push(["Tilted (skipped)", fmt(p.oblique), "warn"]);
    if (p.median_rel_alt) rows.push(["Flight height", fmt(p.median_rel_alt, 0) + " m (" + fmt(p.median_rel_alt * 3.28084, 0) + " ft)"]);
    if (p.est_gsd_cm) rows.push(["Photo resolution", "≈ " + fmt(p.est_gsd_cm, 1) + " cm/px"]);
    for (const [k, n] of cams) rows.push([cams.length > 1 ? "Camera (" + fmt(n) + ")" : "Camera", k.replace(/\s+f[\d.]+/, "").trim()]);
  }
  facts($("photoFacts"), rows);
  $("selectionBox").hidden = !p.count;
  const n = p.selected;
  $("selText").textContent = n === p.count ? "All photos will be processed." : fmt(n) + " of " + fmt(p.count) + " photos picked for processing.";
  $("btnSelectAll").hidden = n === p.count;
  $("mapEmpty").hidden = p.count > 0 || !!result;
  $("stepPhotos").classList.toggle("done", p.count > 0);
}

async function loadPhotos(fit) {
  try {
    photos = await api("/api/photos");
  } catch (e) { return; }
  drawPhotos();
  if (fit && !result) fitJob();
}

function drawPhotos() {
  if (!map) return;
  if (layers.cams) layers.cams.remove();
  layers.cams = L.layerGroup();
  if (!photos) return;
  const pts = photos.points;
  const renderer = L.canvas({ padding: 0.3 });
  for (const [x, y, i, sel, rtk, obl] of pts) {
    let style;
    if (obl) style = { radius: 3, color: "#B5432F", weight: 1.5, fillOpacity: 0 };
    else if (!sel) style = { radius: 2.5, color: "#7A857E", weight: 1, fillColor: "#7A857E", fillOpacity: 0.6 };
    else style = { radius: 3.5, color: "#ffffff", weight: 1.2, fillColor: rtk ? "#3E6B51" : "#E9B20F", fillOpacity: 1 };
    L.circleMarker([y, x], Object.assign({ renderer, interactive: false }, style)).addTo(layers.cams);
  }
  if ($("showCams").checked) layers.cams.addTo(map);
  const anyRtk = pts.some((p) => p[4]);
  const anyNon = pts.some((p) => !p[4]);
  const lg = $("legend");
  lg.innerHTML = "";
  if (pts.length) {
    if (anyRtk) lg.insertAdjacentHTML("beforeend", '<span><i style="background:#3E6B51"></i>RTK fixed photo</span>');
    if (anyNon) lg.insertAdjacentHTML("beforeend", '<span><i style="background:#E9B20F"></i>GPS photo</span>');
    if (pts.some((p) => !p[3])) lg.insertAdjacentHTML("beforeend", '<span><i style="background:#7A857E"></i>Not picked</span>');
    if (pts.some((p) => p[5])) lg.insertAdjacentHTML("beforeend", '<span><i style="background:transparent;border-color:#B5432F"></i>Tilted, skipped</span>');
  }
  lg.hidden = !pts.length || !$("showCams").checked;
}

// ------------------------------------------------------------------ process
function applySettings(s) {
  document.querySelectorAll('input[name=preset]').forEach((r) => (r.checked = r.value === s.preset));
  document.querySelectorAll('input[name=surface]').forEach((r) => (r.checked = r.value === s.surface));
  document.querySelectorAll('input[name=compression]').forEach((r) => (r.checked = r.value === s.compression));
  $("optExposure").checked = !!s.exposure_balance;
  $("optTrim").checked = s.trim_edges !== false;
  $("resMode").value = s.resolution_cm ? "custom" : "auto";
  $("resCm").hidden = !s.resolution_cm;
  $("resCm").value = s.resolution_cm || "";
  $("presetHint").textContent = PRESET_HINTS[s.preset] || "";
}
function readSettings() {
  const v = (n) => (document.querySelector('input[name=' + n + ']:checked') || {}).value;
  const custom = $("resMode").value === "custom" && parseFloat($("resCm").value) > 0;
  return {
    preset: v("preset") || "balanced",
    surface: v("surface") || "fields",
    compression: v("compression") || "jpeg",
    exposure_balance: $("optExposure").checked,
    trim_edges: $("optTrim").checked,
    resolution_cm: custom ? parseFloat($("resCm").value) : null,
  };
}
async function saveSettings() {
  const s = readSettings();
  $("presetHint").textContent = PRESET_HINTS[s.preset];
  $("resCm").hidden = $("resMode").value !== "custom";
  try { await api("/api/settings", s); } catch (e) { toast(e.message); }
}

function renderProcess(p) {
  const pr = status.progress;
  const running = status.running;
  $("btnStart").disabled = running || !p.count || (status.scan && !status.scan.finished);
  $("btnStart").textContent = p.has_result ? "Process again" : "Start processing";
  $("btnStart").hidden = running;
  $("runBox").hidden = !(running || (pr && (pr.status === "error" || pr.status === "cancelled") && pr.stage));
  document.querySelectorAll("#stepProcess fieldset input, #stepProcess select, #stepProcess details input").forEach((el) => (el.disabled = running));
  if (pr) {
    const idx = STAGES.indexOf(pr.stage);
    document.querySelectorAll("#stageList li").forEach((li, k) => {
      li.classList.toggle("done", k < idx || (pr.status === "done"));
      li.classList.toggle("active", running && k === idx);
    });
    $("runBar").style.width = Math.round((pr.overall || 0) * 100) + "%";
    $("runMsg").textContent = pr.message || "";
    $("runTime").textContent = running ? fmtTime(pr.elapsed) + (pr.eta ? ", about " + fmtTime(pr.eta) + " left" : "") : "";
    if (pr.status === "error") setErr("procErr", "Processing stopped: " + (pr.message || "unknown error") + "\nOpen the log for details.");
    else if (pr.status === "cancelled" && !running) setErr("procErr", "Processing was cancelled.");
    else setErr("procErr", "");
    $("btnCancel").hidden = !running;
    if (lastProgressStatus && lastProgressStatus !== pr.status && pr.status === "done") {
      toast("Orthomosaic ready.");
      loadResult(true);
    }
    lastProgressStatus = pr.status;
  }
  if (!$("logDrawer").hidden && running) loadLog();
  $("stepProcess").classList.toggle("done", !!p.has_result && !running);
}

async function loadResult(fit) {
  try { result = await api("/api/result"); } catch (e) { result = null; }
  if (!result || !result.ready) {
    result = null;
    $("resultBox").hidden = true;
    $("areasLocked").hidden = false;
    $("areasTools").hidden = true;
    if (layers.ortho) { layers.ortho.remove(); layers.ortho = null; }
    return;
  }
  const r = result.report;
  const rows = [
    ["Photos aligned", fmt(r.photos_aligned) + " of " + fmt(r.photos_used)],
    ["Resolution", fmt(r.gsd_cm, 2) + " cm/px"],
    ["Area covered", r.area_ha ? fmt(r.area_ha * 2.47105, 1) + " ac (" + fmt(r.area_ha, 1) + " ha)" : null],
    ["Alignment error", fmt(r.reprojection_rms_px, 1) + " px (" + fmt(r.reprojection_rms_px * r.gsd_native_cm, 1) + " cm)"],
    ["Seam alignment", r.seam_check ? fmt(r.seam_check.median_cm, 1) + " cm median" : null],
    ["Processing time", fmtTime(r.processing_seconds)],
  ];
  if (r.photos_aligned < r.photos_used) rows[0][2] = "warn";
  facts($("resultFacts"), rows);
  $("resultBox").hidden = false;
  $("areasLocked").hidden = true;
  $("areasTools").hidden = false;
  if (result.mtime !== lastResultMtime) {
    lastResultMtime = result.mtime;
    drawOrtho();
    await loadAreas();
    if (fit) fitJob();
  }
}

function drawOrtho() {
  if (!map || !result) return;
  if (layers.ortho) layers.ortho.remove();
  const b = result.bounds;
  layers.ortho = L.tileLayer("/tiles/{z}/{x}/{y}.png?v=" + result.mtime, {
    tileSize: 256, minNativeZoom: 0, maxNativeZoom: result.max_native_zoom, minZoom: -4, maxZoom: 22,
    noWrap: true, bounds: L.latLngBounds([b[1], b[0]], [b[3], b[2]]), keepBuffer: 4, updateWhenZooming: false,
  }).addTo(map);
  layers.ortho.bringToBack();
  map.setMaxZoom(result.max_native_zoom + 3);
  $("mapEmpty").hidden = true;
}

// ------------------------------------------------------------------ map
function initMap() {
  if (map) return;
  const crs = L.extend({}, L.CRS.Simple, { transformation: new L.Transformation(1 / R0, 0, -1 / R0, 0) });
  map = L.map("map", { crs, minZoom: -4, maxZoom: 15, zoomSnap: 0.25, zoomDelta: 0.5, attributionControl: false, zoomControl: false, wheelPxPerZoomLevel: 90 });
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.control.scale({ position: "bottomright", imperial: true, metric: true }).addTo(map);
  map.setView([0, 0], 2);
  layers.areas = L.layerGroup().addTo(map);
  map.on("mousedown", onMapDown);
  map.on("mousemove", onMapMove);
  map.on("mouseup", onMapUp);
  map.on("click", onMapClick);
  map.on("dblclick", onMapDbl);
}
function clearLayers() {
  for (const k of ["cams", "ortho", "draft"]) if (layers[k]) { layers[k].remove(); layers[k] = null; }
  if (layers.areas) layers.areas.clearLayers();
}
function fitJob() {
  if (!map) return;
  let bb = null;
  if (result) {
    const b = result.bounds;
    bb = L.latLngBounds([b[1], b[0]], [b[3], b[2]]);
  } else if (photos && photos.points.length) {
    bb = L.latLngBounds(photos.points.map((p) => [p[1], p[0]]));
    bb = bb.pad(0.15);
  }
  if (bb && bb.isValid()) map.fitBounds(bb, { padding: [30, 30] });
}

// ------------------------------------------------------------------ tools
const TOOL_HINTS = {
  select: "Drag a box around the photos you want to process.",
  rect: "Drag on the map to draw a rectangle.",
  poly: "Click to place corners. Click the first corner or double-click to finish. Backspace removes the last corner, Esc cancels.",
};
function setTool(t) {
  tool = tool === t ? null : t;
  cancelDraft();
  $("toolRect").setAttribute("aria-pressed", tool === "rect");
  $("toolPoly").setAttribute("aria-pressed", tool === "poly");
  $("btnSelectArea").classList.toggle("secondary", tool === "select");
  $("toolHint").textContent = tool && tool !== "select" ? TOOL_HINTS[tool] : "";
  if (tool === "select") toast(TOOL_HINTS.select);
  const c = map.getContainer();
  c.classList.toggle("cursor-cross", !!tool);
  if (tool) { map.dragging.disable(); map.doubleClickZoom.disable(); }
  else { map.dragging.enable(); map.doubleClickZoom.enable(); }
}
function cancelDraft() {
  draw = null;
  if (layers.draft) { layers.draft.remove(); layers.draft = null; }
}
function onMapDown(e) {
  if (tool !== "rect" && tool !== "select") return;
  draw = { start: e.latlng };
  layers.draft = L.rectangle(L.latLngBounds(e.latlng, e.latlng), { color: "#E9B20F", weight: 2, dashArray: "6 4", fillOpacity: 0.12 }).addTo(map);
}
function onMapMove(e) {
  if (!draw) return;
  if ((tool === "rect" || tool === "select") && draw.start) layers.draft.setBounds(L.latLngBounds(draw.start, e.latlng));
  if (tool === "poly" && draw.pts && draw.pts.length) layers.draft.setLatLngs([...draw.pts, e.latlng]);
}
async function onMapUp(e) {
  if (!draw || !draw.start) return;
  const b = L.latLngBounds(draw.start, e.latlng);
  const t = tool;
  cancelDraft();
  const sz = map.latLngToContainerPoint(b.getNorthEast()).distanceTo(map.latLngToContainerPoint(b.getSouthWest()));
  if (sz < 8) return;
  if (t === "select") {
    const ids = photos.points.filter((p) => b.contains([p[1], p[0]]) && !p[5]).map((p) => p[2]);
    setTool(null);
    if (!ids.length) return toast("No photos inside that box.");
    await api("/api/photos/select", { ids });
    await loadPhotos(false);
    refresh();
    return;
  }
  const w = b.getWest(), s = b.getSouth(), en = b.getEast(), n = b.getNorth();
  addArea({ name: nextName(), kind: "drawn", rings: [[[w, s], [en, s], [en, n], [w, n], [w, s]]] });
}
function onMapClick(e) {
  if (tool !== "poly") return;
  if (!draw) {
    draw = { pts: [e.latlng] };
    layers.draft = L.polyline([e.latlng], { color: "#E9B20F", weight: 2, dashArray: "6 4" }).addTo(map);
    return;
  }
  const first = map.latLngToContainerPoint(draw.pts[0]);
  if (draw.pts.length >= 3 && first.distanceTo(e.containerPoint) < 10) return finishPoly();
  draw.pts.push(e.latlng);
  layers.draft.setLatLngs(draw.pts);
}
function onMapDbl() { if (tool === "poly" && draw && draw.pts.length >= 3) finishPoly(); }
function finishPoly() {
  // a double-click also fires two clicks; drop a duplicate final point
  let pts = draw.pts;
  if (pts.length > 3) {
    const a = map.latLngToContainerPoint(pts[pts.length - 1]), b = map.latLngToContainerPoint(pts[pts.length - 2]);
    if (a.distanceTo(b) < 4) pts = pts.slice(0, -1);
  }
  const ring = pts.map((p) => [p.lng, p.lat]);
  ring.push(ring[0]);
  cancelDraft();
  addArea({ name: nextName(), kind: "drawn", rings: [ring] });
}
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  if (e.key === "Escape") { if (draw) cancelDraft(); else if (tool) setTool(null); }
  if (e.key === "Backspace" && tool === "poly" && draw && draw.pts.length > 1) {
    draw.pts.pop();
    layers.draft.setLatLngs(draw.pts);
    e.preventDefault();
  }
});

// ------------------------------------------------------------------ areas
function nextName() {
  let k = areas.length + 1;
  const names = new Set(areas.map((a) => a.name));
  while (names.has("Area " + k)) k++;
  return "Area " + k;
}
async function loadAreas() {
  try { areas = (await api("/api/areas")).areas; } catch (e) { areas = []; }
  areas.forEach((a) => checked.add(a.id));
  drawAreas();
}
const plain = () => areas.map((a) => ({ id: a.id, name: a.name, kind: a.kind, rings: a.rings }));
async function saveAreas() {
  try {
    areas = (await api("/api/areas", { areas: plain() })).areas;
    drawAreas();
  } catch (e) { toast(e.message); }
}
async function addArea(a) {
  a.id = "";
  areas.push(a);
  const before = new Set(areas.map((x) => x.id));
  areas = (await api("/api/areas", { areas: plain() })).areas;
  areas.forEach((x) => { if (!before.has(x.id)) checked.add(x.id); });
  drawAreas();
  focusArea(areas[areas.length - 1].id, true);
}
function drawAreas() {
  if (!layers.areas) return;
  layers.areas.clearLayers();
  for (const a of areas) {
    const ll = a.rings.map((r) => r.map((p) => [p[1], p[0]]));
    const poly = L.polygon(ll, { color: "#E9B20F", weight: 2.5, fillColor: "#E9B20F", fillOpacity: checked.has(a.id) ? 0.1 : 0.02, dashArray: checked.has(a.id) ? null : "4 4" });
    poly.bindTooltip(a.name, { permanent: true, direction: "center", className: "area-label" });
    poly.on("click", (e) => { if (!tool) { L.DomEvent.stop(e); focusArea(a.id); } });
    poly.addTo(layers.areas);
    a._layer = poly;
  }
  renderAreaList();
}
function renderAreaList() {
  const ul = $("areaList");
  ul.innerHTML = "";
  for (const a of areas) {
    const li = document.createElement("li");
    li.dataset.id = a.id;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checked.has(a.id);
    cb.setAttribute("aria-label", "Export " + a.name);
    cb.onchange = () => { cb.checked ? checked.add(a.id) : checked.delete(a.id); drawAreas(); };
    const nm = document.createElement("input");
    nm.type = "text";
    nm.value = a.name;
    nm.setAttribute("aria-label", "Area name");
    nm.onchange = () => { a.name = nm.value.trim() || a.name; saveAreas(); };
    nm.onfocus = () => focusArea(a.id, false, true);
    const sz = document.createElement("span");
    sz.className = "size";
    sz.textContent = fmt(a.ac, a.ac < 10 ? 2 : 1) + " ac";
    sz.title = fmt(a.ha, 2) + " ha";
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "×";
    del.title = "Remove area";
    del.setAttribute("aria-label", "Remove " + a.name);
    del.onclick = () => { areas = areas.filter((x) => x.id !== a.id); checked.delete(a.id); saveAreas(); };
    li.append(cb, nm, sz, del);
    ul.append(li);
  }
  $("areaHead").hidden = !areas.length;
  $("exportBox").hidden = !areas.length;
  $("areaAll").checked = areas.length && areas.every((a) => checked.has(a.id));
  const n = areas.filter((a) => checked.has(a.id)).length;
  $("btnExport").textContent = n === 1 ? "Export 1 area" : "Export " + n + " areas";
  $("btnExport").disabled = !n || (status && status.export && !status.export.finished);
}
function focusArea(id, zoom, noScroll) {
  document.querySelectorAll("#areaList li").forEach((li) => li.classList.toggle("focus", li.dataset.id === id));
  const a = areas.find((x) => x.id === id);
  if (!a) return;
  if (zoom && a._layer) map.fitBounds(a._layer.getBounds(), { padding: [60, 60], maxZoom: map.getZoom() });
  if (!noScroll) {
    const li = document.querySelector('#areaList li[data-id="' + id + '"]');
    if (li) li.scrollIntoView({ block: "nearest" });
  }
}
function renderExport() {
  const ex = status.export;
  const running = ex && !ex.finished;
  $("exportProg").hidden = !running;
  if (running) {
    const f = ex.total ? (ex.done + (ex.frac || 0)) / ex.total : 0;
    $("exportBar").style.width = Math.round(f * 100) + "%";
    $("exportText").textContent = "Exporting " + (ex.current || "") + " (" + (ex.done + 1) + " of " + ex.total + ")";
  }
  if (exportWasRunning && !running && ex) {
    if (ex.error) setErr("exportErr", ex.error);
    else {
      setErr("exportErr", ex.errors.length ? ex.errors.join("\n") : "");
      toast("Exported " + ex.files.length + (ex.files.length === 1 ? " GeoTIFF." : " GeoTIFFs."), 5000);
      if (ex.files.length) api("/api/open_path", { path: ex.out_dir });
    }
  }
  exportWasRunning = !!running;
  if (areas.length) renderAreaList();
}

// ------------------------------------------------------------------ log
async function loadLog() {
  try {
    const r = await api("/api/log?tail=400");
    const pre = $("logText");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
    pre.textContent = r.lines.join("\n") || "Nothing logged yet.";
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  } catch (e) { /* ignore */ }
}
function showLog() { $("logDrawer").hidden = false; loadLog(); }

// ------------------------------------------------------------------ actions
async function openProject(d) {
  setErr("openErr", "");
  try { await api("/api/project/open", { dir: d }); await refresh(); }
  catch (e) { setErr("openErr", e.message); }
}
async function browse(mode, title, initial) {
  try { return (await api("/api/browse", { mode, title, initial: initial || "" })).path; }
  catch (e) { toast("Could not open the file browser. Paste the path instead."); return ""; }
}

function wire() {
  $("newBrowse").onclick = async () => { const p = await browse("folder", "Where should projects be saved?", $("newParent").value); if (p) $("newParent").value = p; };
  $("newForm").onsubmit = async (e) => {
    e.preventDefault();
    setErr("newErr", "");
    try { await api("/api/project/new", { name: $("newName").value, parent: $("newParent").value }); $("newName").value = ""; await refresh(); }
    catch (err) { setErr("newErr", err.message); }
  };
  $("openOther").onclick = async () => { const p = await browse("folder", "Choose a FieldStitch project folder", status.default_dir); if (p) openProject(p); };
  $("btnClose").onclick = async () => { try { await api("/api/project/close", {}); refresh(); } catch (e) { toast(e.message); } };
  $("btnOpenFolder").onclick = () => api("/api/open_path", {});
  $("projName").onclick = async () => {
    const n = prompt("Project name", status.project.name);
    if (n && n.trim()) { await api("/api/rename", { name: n.trim() }); refresh(); }
  };
  $("folderBrowse").onclick = async () => { const p = await browse("folder", "Choose a folder of drone photos", $("folderPath").value); if (p) { $("folderPath").value = p; } };
  $("folderAdd").onclick = async () => {
    setErr("photoErr", "");
    const f = $("folderPath").value.trim().replace(/^"|"$/g, "");
    if (!f) { const p = await browse("folder", "Choose a folder of drone photos", ""); if (!p) return; $("folderPath").value = p; return $("folderAdd").onclick(); }
    try { await api("/api/photos/add", { folder: f, recursive: $("folderRecursive").checked }); scanWasRunning = true; refresh(); }
    catch (e) { setErr("photoErr", e.message); }
  };
  $("folderPath").addEventListener("keydown", (e) => { if (e.key === "Enter") $("folderAdd").onclick(); });
  $("btnSelectArea").onclick = () => setTool("select");
  $("btnSelectAll").onclick = async () => { await api("/api/photos/select", { clear: true }); await loadPhotos(false); refresh(); };

  document.querySelectorAll("#stepProcess input, #stepProcess select").forEach((el) => el.addEventListener("change", saveSettings));
  $("btnStart").onclick = async () => {
    setErr("procErr", "");
    try { await api("/api/process/start", { settings: readSettings() }); lastProgressStatus = "starting"; refresh(); }
    catch (e) { setErr("procErr", e.message); }
  };
  $("btnCancel").onclick = async () => { if (confirm("Stop processing? Finished steps are kept and reused next time.")) { await api("/api/process/cancel", {}); refresh(); } };
  $("btnLog").onclick = showLog;
  $("btnLog2").onclick = showLog;
  const diag = async () => {
    try { const r = await api("/api/diagnostics", {}); toast("Diagnostics saved to " + r.path, 10000); }
    catch (e) { toast(e.message); }
  };
  $("btnDiag").onclick = diag;
  $("btnDiag2").onclick = diag;
  $("logClose").onclick = () => ($("logDrawer").hidden = true);
  $("btnOutputs").onclick = () => api("/api/open_path", { path: result && result.outputs_dir });

  $("btnFit").onclick = fitJob;
  $("showCams").onchange = () => { if (layers.cams) { $("showCams").checked ? layers.cams.addTo(map) : layers.cams.remove(); } $("legend").hidden = !$("showCams").checked || !photos || !photos.points.length; };

  $("toolRect").onclick = () => { $("gridForm").hidden = true; $("toolGrid").setAttribute("aria-pressed", false); setTool("rect"); };
  $("toolPoly").onclick = () => { $("gridForm").hidden = true; $("toolGrid").setAttribute("aria-pressed", false); setTool("poly"); };
  $("toolGrid").onclick = () => {
    if (tool) setTool(null);
    const show = $("gridForm").hidden;
    $("gridForm").hidden = !show;
    $("toolGrid").setAttribute("aria-pressed", show);
  };
  document.querySelectorAll("input[name=gridmode]").forEach((r) => (r.onchange = () => {
    const sz = document.querySelector("input[name=gridmode]:checked").value === "size";
    document.querySelector(".gridsize").hidden = !sz;
    document.querySelector(".gridcount").hidden = sz;
  }));
  $("gridForm").onsubmit = async (e) => {
    e.preventDefault();
    const mode = document.querySelector("input[name=gridmode]:checked").value;
    let size = parseFloat($("gridSize").value) || 300;
    if ($("gridUnit").value === "ft") size *= 0.3048;
    try {
      const keep = new Set(areas.map((a) => a.id));
      const r = await api("/api/areas/grid", { mode, size_m: size, rows: $("gridRows").value, cols: $("gridCols").value });
      areas = r.areas;
      areas.forEach((a) => { if (!keep.has(a.id)) checked.add(a.id); });
      drawAreas();
      toast("Added " + r.added + " grid areas. Empty cells were skipped.");
      $("gridForm").hidden = true;
      $("toolGrid").setAttribute("aria-pressed", false);
    } catch (err) { toast(err.message); }
  };
  $("toolImport").onclick = async () => {
    const p = await browse("file", "Choose field boundaries (KML, KMZ, SHP, GeoJSON)", "");
    if (!p) return;
    try {
      const keep = new Set(areas.map((a) => a.id));
      const r = await api("/api/areas/import", { path: p });
      areas = r.areas;
      areas.forEach((a) => { if (!keep.has(a.id)) checked.add(a.id); });
      drawAreas();
      toast("Imported " + r.added + (r.added === 1 ? " boundary." : " boundaries."));
    } catch (e) { toast(e.message, 6000); }
  };
  $("areaAll").onchange = () => { areas.forEach((a) => ($("areaAll").checked ? checked.add(a.id) : checked.delete(a.id))); drawAreas(); };
  $("areaClear").onclick = () => { if (confirm("Remove all areas?")) { areas = []; checked.clear(); saveAreas(); } };
  $("exportBrowse").onclick = async () => { const p = await browse("folder", "Save exported GeoTIFFs in", $("exportDir").value); if (p) $("exportDir").value = p; };
  $("btnExport").onclick = async () => {
    setErr("exportErr", "");
    const ids = areas.filter((a) => checked.has(a.id)).map((a) => a.id);
    try { await api("/api/export", { ids, out_dir: $("exportDir").value.trim(), compression: $("exportComp").value }); exportWasRunning = true; refresh(); }
    catch (e) { setErr("exportErr", e.message); }
  };
  $("btnKml").onclick = async () => {
    try { const r = await api("/api/areas/kml", { out_dir: $("exportDir").value.trim() }); toast("Saved " + r.path, 5000); }
    catch (e) { toast(e.message); }
  };
}

// ------------------------------------------------------------------ updates
let updInfo = null;
let updRestarting = false;
async function checkUpdates(silent) {
  if (!silent) {
    $("updDlg").showModal();
    $("updMsg").textContent = "Checking for updates…";
    $("updNotes").hidden = true; $("updInstall").hidden = true; $("updFull").hidden = true; $("updBarWrap").hidden = true;
    setErr("updErr", "");
  }
  try { updInfo = await api("/api/update/check"); }
  catch (e) { updInfo = { ok: false, message: e.message }; }
  $("updDot").hidden = !(updInfo.ok && updInfo.available);
  if (silent) return;
  showUpdateInfo();
}
function showUpdateInfo() {
  const u = updInfo;
  if (!u.ok) { $("updMsg").textContent = u.message; return; }
  if (!u.available) {
    $("updMsg").textContent = u.message || ("You have the latest version (" + u.current + ").");
    return;
  }
  $("updMsg").textContent = "Version " + u.latest + " is available. You have " + u.current + ".";
  if (u.notes) { $("updNotes").textContent = u.notes; $("updNotes").hidden = false; }
  if (u.needs_full) {
    setErr("updErr", "This update changes the bundled libraries, so it needs the full download. Unzip it in place of your current FieldStitch folder; your projects are kept.");
    if (u.full_url || u.page) { $("updFull").href = u.full_url || u.page; $("updFull").hidden = false; }
  } else {
    $("updInstall").hidden = false;
  }
}
async function installUpdate() {
  setErr("updErr", "");
  try { await api("/api/update/install", {}); }
  catch (e) { setErr("updErr", e.message); return; }
  $("updInstall").disabled = true;
  $("updBarWrap").hidden = false;
  updRestarting = true;
  pollUpdate();
}
async function pollUpdate() {
  let st = null;
  try { st = await api("/api/status"); } catch (e) { st = null; }
  if (st && updRestarting) {
    const u = st.update || {};
    if (st.version === (updInfo && updInfo.latest)) {
      $("updMsg").textContent = "Updated to " + st.version + ". Reloading…";
      setTimeout(() => location.reload(), 800);
      return;
    }
    if (u.phase === "error") {
      setErr("updErr", u.message || "The update failed.");
      $("updInstall").disabled = false; $("updBarWrap").hidden = true; updRestarting = false;
      return;
    }
    if (u.phase === "downloading") { $("updMsg").textContent = "Downloading update…"; $("updBar").style.width = Math.round((u.frac || 0) * 100) + "%"; }
    if (u.phase === "installing") { $("updMsg").textContent = "Installing…"; $("updBar").style.width = "100%"; }
    if (u.phase === "restarting") $("updMsg").textContent = "Restarting FieldStitch…";
  } else if (!st) {
    $("updMsg").textContent = "Restarting FieldStitch…";
  }
  setTimeout(pollUpdate, 1000);
}

wire();
$("btnUpdate").onclick = () => checkUpdates(false);
$("updClose").onclick = () => { if (!updRestarting) $("updDlg").close(); };
$("updInstall").onclick = installUpdate;
refresh();
setTimeout(() => checkUpdates(true), 2500);
