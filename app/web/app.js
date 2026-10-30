"use strict";
/* PlaneSite front end. Map frame: lat = northing offset (m), lng = easting offset (m) from the project origin. */

const $ = (id) => document.getElementById(id);
const R0 = 64;
const STAGES = ["features", "matching", "alignment", "surface", "ortho"];
const STAGE_LABEL = { features: "Finding features", matching: "Matching photos", alignment: "Aligning cameras", surface: "Building surface", ortho: "Rendering orthomosaic" };
// resolution slider: output pixel size as a multiple of the photos' own (left = low, right = full)
const RES_STEPS = [4, 3, 2.5, 2, 1.5, 1.25, 1];
const PRESET_RES = { fast: 2, balanced: 1, accurate: 1 };

let status = null;
let photos = null;
let result = null;
let areas = [];
let map = null;
let layers = {};
let tool = null;
let draw = null;
let pollTimer = null;
let lastProjectDir = undefined;
let lastFrame = null;
let lastResultMtime = null;
let lastProgressStatus = null;
let scanWasRunning = false;
let exportWasRunning = false;
let lastScan = null;
let panel = "import";
let nativeGsdCm = null;
let optionsAfterScan = false;   // open Processing options when the current photo import finishes

// ------------------------------------------------------------------ utils
async function api(path, body, timeoutMs) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let timer = 0;
  if (timeoutMs) {
    const ac = new AbortController();
    opt.signal = ac.signal;
    timer = setTimeout(() => ac.abort(), timeoutMs);
  }
  let r;
  try { r = await fetch(path, opt); } finally { clearTimeout(timer); }
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
function fmtDur(s) {
  if (s == null || !isFinite(s)) return "–";
  const m = Math.max(1, Math.round(s / 60));
  if (m < 60) return "≈ " + m + " min";
  return "≈ " + Math.floor(m / 60) + " h " + (m % 60 ? (m % 60) + " min" : "");
}
function fmtBytes(b) {
  if (b == null || !isFinite(b)) return "–";
  if (b >= 1e9) return fmt(b / 1e9, b >= 1e10 ? 0 : 1) + " GB";
  if (b >= 1e6) return fmt(b / 1e6, 0) + " MB";
  return fmt(Math.max(b / 1e3, 1), 0) + " KB";
}
const cmIn = (cm) => fmt(cm, cm < 10 ? 1 : 0) + " cm (" + fmt(cm / 2.54, 2) + " in) per pixel";
const acHa = (ha) => fmt(ha * 2.47105, ha * 2.47105 < 10 ? 1 : 0) + " ac (" + fmt(ha, 1) + " ha)";
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
const busyScan = () => status && status.scan && !status.scan.finished;
const busyExport = () => status && status.export && !status.export.finished;

// ------------------------------------------------------------------ polling
// Errors in the page go to PlaneSite's log (the window has no console to look at).
let clientErrs = 0;
function reportError(msg) {
  if (!String(msg).startsWith("note") && clientErrs++ > 20) return;
  try { fetch("/api/clientlog", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ msg: String(msg).slice(0, 2000) }) }); } catch (e) { /* ignore */ }
}
function errContext() {
  // what the window and map looked like when it happened
  try {
    const sz = map ? map.getSize() : null;
    return " | window " + innerWidth + "x" + innerHeight + " dpr " + devicePixelRatio + " " + document.visibilityState +
      (map ? " | map " + (sz ? sz.x + "x" + sz.y : "?") + " zoom " + map.getZoom() + " layers " +
        ["cams", "ortho", "areas", "draft"].filter((k) => layers[k] && map.hasLayer(layers[k])).join(",") : "") +
      (photos ? " | photos " + photos.points.length : "") + (status && status.progress ? " | stage " + status.progress.stage : "");
  } catch (err) { return " | context failed: " + err.message; }
}
Error.stackTraceLimit = 60;   // a runaway recursion shows its whole cycle, not just 10 frames of it
function stackOf(err) {
  if (!err || !err.stack) return "";
  const lines = err.stack.split("\n").slice(1).map((l) => l.trim());
  const seen = new Set(), uniq = [];
  for (const l of lines) if (!seen.has(l)) { seen.add(l); uniq.push(l); }
  return "\n    " + uniq.slice(0, 25).join("\n    ") + (uniq.length < lines.length ? "\n    (" + lines.length + " frames, " + uniq.length + " different)" : "");
}
window.addEventListener("error", (e) => reportError((e.message || "error") + " at " + (e.filename || "").split("/").pop() + ":" + e.lineno + errContext() + stackOf(e.error)));
window.addEventListener("unhandledrejection", (e) => reportError("unhandled: " + ((e.reason && (e.reason.stack || e.reason.message)) || e.reason) + errContext()));
document.addEventListener("visibilitychange", () => { if (status && status.running) reportError("note: window " + document.visibilityState + errContext()); });

async function refresh() {
  // whatever happens, the next refresh is scheduled: a failed request or a display error must
  // not leave the window showing stale progress
  let next = 3000;
  try {
    try {
      status = await api("/api/status", undefined, 15000);
    } catch (e) {
      if (!updRestarting) toast("Lost contact with PlaneSite. Start it again from the Start menu or desktop icon.");
      next = 4000;
      return;
    }
    try { render(); } catch (e) { reportError("render: " + (e.stack || e.message)); }
    const busy = status.running || busyScan() || busyExport();
    next = busy ? 1000 : 3000;
  } finally {
    schedule(next);
  }
}
function schedule(ms) { clearTimeout(pollTimer); pollTimer = setTimeout(refresh, ms); }

// ------------------------------------------------------------------ render
function render() {
  const p = status.project;
  const dir = p ? p.dir : null;
  if (dir !== lastProjectDir) {
    lastProjectDir = dir;
    photos = null; result = null; areas = []; lastResultMtime = null; lastProgressStatus = null;
    clearLayers();
    lastFrame = null;
    if (p) {
      $("dash").hidden = true;
      loadPhotos(true);
      loadResult(true);
      $("exportDir").value = p.dir + sep() + "exports";
      // a job planned before flying opens on its fields; a flown one on its photos or result
      setPanel(p.has_result ? "layers" : status.running ? "process" : (!p.count && p.has_frame) ? "areas" : "import");
    } else {
      setPanel(null);
      map.setView([0, 0], 2);
      openDash();
    }
  }
  // the map frame is known (photos read, or a planned job); it can also move once (ortho in another UTM zone)
  const frame = p ? p.frame : null;
  if (frame !== lastFrame) {
    const first = lastFrame === null;
    lastFrame = frame;
    drawBase();
    if (p && p.has_frame) loadAreas().then(() => { if (first && !p.count && !p.has_result) fitJob(); });
  }
  updateAreasLock(p);
  if (!p && $("dash").hidden) openDash();
  $("btnProjects").setAttribute("aria-pressed", !$("dash").hidden);
  $("nameWrap").hidden = !p;
  const name = $("projName");
  if (document.activeElement !== name) name.value = p ? p.name : "";
  $("mOpenFolder").disabled = !p;
  $("mClose").disabled = !p;
  $("mDiag").disabled = !p;
  $("mLog").disabled = !p;

  const geotiff = p && p.geotiff;
  $("railProcess").disabled = !p || !p.count || !!geotiff;
  $("railAreas").disabled = !p;
  $("railExport").disabled = !(p && p.has_result);

  renderImport(p);
  renderProcess(p);
  renderStatusBar(p);
  if (!$("dash").hidden) renderDash();
}

function setPanel(name) {
  panel = name;
  $("side").hidden = !name;
  document.querySelectorAll(".pane").forEach((el) => (el.hidden = el.dataset.pane !== name));
  document.querySelectorAll(".railbtn[data-panel]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.panel === name));
  if (name !== "areas" && tool && tool !== "select") setTool(null);
  if (map) setTimeout(() => map.invalidateSize(), 0);
}

// ------------------------------------------------------------------ import
function renderImport(p) {
  const sc = status.scan;
  const scanning = busyScan();
  ["addFolder", "addImages"].forEach((id) => ($(id).disabled = scanning || status.running));
  $("scanBox").hidden = !scanning;
  if (scanning) {
    $("scanBar").style.width = (sc.total ? (100 * sc.done) / sc.total : 2) + "%";
    $("scanText").textContent = sc.total ? "Reading photo " + fmt(sc.done) + " of " + fmt(sc.total) + "…" : "Looking for photos…";
  }
  if (scanWasRunning && !scanning && sc) {
    lastScan = sc;
    loadPhotos(true);
    if (sc.error) { setErr("importErr", sc.error); optionsAfterScan = false; }
    else if (!sc.added) { setErr("importErr", "No drone photos were found there. Choose the folder that holds the flight's JPG photos."); optionsAfterScan = false; }
    else {
      setErr("importErr", "");
      toast("Loaded " + fmt(sc.added) + " photos.");
      if (optionsAfterScan && p && p.count) setTimeout(openOptions, 300);
      optionsAfterScan = false;
    }
  }
  scanWasRunning = scanning;

  $("importDetails").hidden = !p || (!p.count && !p.geotiff && !p.planned) || scanning;
  if (!p) return;
  $("addRow").hidden = !!p.geotiff;
  $("addLabel").textContent = p.count ? "Add more photos to this project:" : "After the flight, add its photos:";
  document.querySelector(".newhint").hidden = !p.count;
  const ul = $("folderList");
  ul.innerHTML = "";
  if (p.geotiff) {
    const li = document.createElement("li");
    li.innerHTML = "<span><b></b><small>GeoTIFF</small></span>";
    li.querySelector("b").textContent = p.geotiff.split(/[\\/]/).pop();
    li.title = p.geotiff;
    ul.append(li);
  }
  for (const f of p.folders) {
    const li = document.createElement("li");
    const s = document.createElement("span");
    const parts = f.split(/[\\/]/).filter(Boolean);
    const bName = document.createElement("b");
    bName.textContent = parts.pop() || f;
    const sm = document.createElement("small");
    const parent = parts.join(sep());
    sm.textContent = parent.length > 40 ? "…" + parent.slice(-39) : parent;
    s.append(bName, sm);
    s.title = f;
    const x = document.createElement("button");
    x.textContent = "×";
    x.title = "Remove these photos from the project";
    x.setAttribute("aria-label", "Remove folder " + f);
    x.onclick = async () => {
      if (status.running) return toast("Stop processing first.");
      await api("/api/photos/remove_folder", { folder: f });
      await loadPhotos(true);
      refresh();
    };
    li.append(s, x);
    ul.append(li);
  }
  if (p.geotiff) {
    facts($("photoFacts"), []);
    $("issues").hidden = true;
    $("selectionBox").hidden = true;
    $("btnToProcess").hidden = true;
    return;
  }
  const cams = Object.entries(p.cameras || {}).sort((a, b) => b[1] - a[1]);
  const rows = p.count ? [["Photos", fmt(p.count)]] : [];
  if (p.count) {
    rows.push(["RTK fixed", fmt(p.rtk_fixed)]);
    if (p.median_rel_alt) rows.push(["Flight height", fmt(p.median_rel_alt, 0) + " m (" + fmt(p.median_rel_alt * 3.28084, 0) + " ft)"]);
    if (p.est_gsd_cm) rows.push(["Photo resolution", fmt(p.est_gsd_cm, 1) + " cm (" + fmt(p.est_gsd_cm / 2.54, 2) + " in)"]);
    if (cams.length) rows.push(["Camera", cams[0][0].replace(/\s+f[\d.]+.*$/, "").replace(/\s+\d+x\d+$/, "").trim()]);
  }
  facts($("photoFacts"), rows);

  showIssues($("issues"), p);

  $("selectionBox").hidden = !p.count;
  const n = p.selected;
  $("selText").textContent = n === p.count ? "All photos will be processed." : fmt(n) + " of " + fmt(p.count) + " photos picked for processing.";
  $("btnSelectAll").hidden = n === p.count;
  $("btnToProcess").hidden = !p.count;
  $("btnToProcess").disabled = status.running || scanning;
  $("btnToProcess").textContent = p.has_result ? "Process again…" : "Processing options…";
}

function photoIssues(p) {
  const issues = [];
  const noGps = p.count - p.with_gps;
  if (noGps) issues.push(fmt(noGps) + (noGps === 1 ? " photo has" : " photos have") + " no GPS position and will be skipped.");
  if (p.oblique) issues.push(fmt(p.oblique) + (p.oblique === 1 ? " photo is" : " photos are") + " tilted away from straight down and will be skipped.");
  if (p.other_camera) issues.push(fmt(p.other_camera) + (p.other_camera === 1 ? " photo comes" : " photos come") + " from another camera and will be skipped.");
  if (lastScan && lastScan.skipped_small) issues.push(fmt(lastScan.skipped_small) + " small images (thumbnails or previews) were not loaded.");
  if (lastScan && lastScan.errors) issues.push(fmt(lastScan.errors) + " files couldn't be read, e.g. " + (lastScan.error_examples || []).join("; "));
  return issues;
}
function showIssues(box, p) {
  const issues = photoIssues(p);
  box.hidden = !issues.length;
  box.innerHTML = "<b>Photo check</b><ul></ul>";
  for (const t of issues) { const li = document.createElement("li"); li.textContent = t; box.querySelector("ul").append(li); }
}

async function addPhotos(kind) {
  setErr("importErr", "");
  let arg;
  if (kind === "images") {
    const files = await browse("images", "Choose drone photos to add", "");
    if (!files || !files.length) return;
    arg = { files, recursive: false };
  } else {
    const folder = await browse("folder", "Choose a folder of drone photos to add", "");
    if (!folder) return;
    arg = { folder, recursive: true };
  }
  try {
    await api("/api/photos/add", arg);
    scanWasRunning = true;
    lastScan = null;
    optionsAfterScan = true;
    setPanel("import");
    refresh();
  } catch (e) { setErr("importErr", e.message); setPanel("import"); }
}

async function openGeotiff() {
  setErr("dErr", "");
  const f = await browse("geotiff", "Choose an orthomosaic GeoTIFF", "");
  if (!f) return;
  try {
    await api("/api/geotiff/open", { path: f });
    lastScan = null;
    $("dash").hidden = true;
    await refresh();
    toast("Opened " + f.split(/[\\/]/).pop());
  } catch (e) { setErr("dErr", e.message); }
}

// ------------------------------------------------------------------ new project window
let np = { folder: null, files: null, nameEdited: false };
let npTimer = null;
async function openNew() {
  np = { folder: null, files: null, nameEdited: false };
  $("npName").value = "";
  setErr("npErr", "");
  $("npPicked").hidden = true;
  $("npWhere").textContent = "";
  $("npCreate").disabled = true;
  ["npFolder", "npImages"].forEach((id) => $(id).setAttribute("aria-pressed", false));
  try { $("npParent").value = (await api("/api/projects")).dir; } catch (e) { $("npParent").value = status.default_dir || ""; }
  $("newDlg").showModal();
}
async function npSuggest() {
  if (!np.folder && !np.files) return;
  try {
    const r = await api("/api/project/suggest", { folder: np.folder, files: np.files, parent: $("npParent").value.trim(),
                                                  name: np.nameEdited ? $("npName").value : "" });
    if (!np.nameEdited) $("npName").value = r.name;
    $("npWhere").textContent = "The project will be saved as " + r.dir;
    $("npCreate").disabled = !$("npName").value.trim();
  } catch (e) { setErr("npErr", e.message); }
}
async function npPick(kind) {
  setErr("npErr", "");
  if (kind === "folder") {
    const f = await browse("folder", "Choose the folder with the flight's photos", "");
    if (!f) return;
    np.folder = f; np.files = null;
    $("npPicked").innerHTML = "Folder: <b></b>";
    $("npPicked").querySelector("b").textContent = f;
  } else {
    const files = await browse("images", "Choose drone photos", "");
    if (!files || !files.length) return;
    np.files = files; np.folder = null;
    const dir = files[0].replace(/[\\/][^\\/]*$/, "");
    $("npPicked").innerHTML = "<b></b> from ";
    $("npPicked").querySelector("b").textContent = fmt(files.length) + (files.length === 1 ? " photo" : " photos");
    $("npPicked").append(dir);
  }
  $("npPicked").hidden = false;
  $("npFolder").setAttribute("aria-pressed", kind === "folder");
  $("npImages").setAttribute("aria-pressed", kind !== "folder");
  npSuggest();
}
async function npCreate() {
  setErr("npErr", "");
  const name = $("npName").value.trim();
  if (!name) return setErr("npErr", "Enter a project name.");
  $("npCreate").disabled = true;
  try {
    await api("/api/project/start", { folder: np.folder, files: np.files, recursive: true, name, parent: $("npParent").value.trim() });
    $("newDlg").close();
    scanWasRunning = true;
    lastScan = null;
    optionsAfterScan = true;
    $("dash").hidden = true;
    await refresh();
    setPanel("import");
  } catch (e) { setErr("npErr", e.message); $("npCreate").disabled = false; }
}

async function loadPhotos(fit) {
  try { photos = await api("/api/photos"); } catch (e) { photos = null; return; }
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
  for (const [x, y, i, sel, rtk, flag] of pts) {
    let style;
    if (flag === 1) style = { radius: 3, color: "#B5432F", weight: 1.5, fillOpacity: 0 };
    else if (flag === 2) style = { radius: 3, color: "#9AA59E", weight: 1.5, fillOpacity: 0 };
    else if (!sel) style = { radius: 2.5, color: "#7A857E", weight: 1, fillColor: "#7A857E", fillOpacity: 0.6 };
    else style = { radius: 3.5, color: "#ffffff", weight: 1.2, fillColor: rtk ? "#3E6B51" : "#E9B20F", fillOpacity: 1 };
    L.circleMarker([y, x], Object.assign({ renderer, interactive: false }, style)).addTo(layers.cams);
  }
  if ($("lyrCams").checked) layers.cams.addTo(map);
  const lg = $("legend");
  lg.innerHTML = "";
  const add = (c, ring, t) => lg.insertAdjacentHTML("beforeend", '<span><i style="background:' + c + (ring ? ";border-color:" + ring : "") + '"></i>' + t + "</span>");
  if (pts.some((p) => p[4] && !p[5])) add("#3E6B51", null, "RTK fixed photo");
  if (pts.some((p) => !p[4] && !p[5])) add("#E9B20F", null, "GPS photo");
  if (pts.some((p) => !p[3])) add("#7A857E", null, "Not picked");
  if (pts.some((p) => p[5] === 1)) add("transparent", "#B5432F", "Tilted, skipped");
  if (pts.some((p) => p[5] === 2)) add("transparent", "#9AA59E", "Other camera, skipped");
}

// ------------------------------------------------------------------ process
function renderProcess(p) {
  const pr = status.progress;
  const running = status.running;
  $("btnOptions").disabled = running || !p || !p.count || busyScan();
  $("btnOptions").textContent = p && p.has_result ? "Process again…" : "Processing options…";
  $("procIdle").hidden = !!(p && p.count) || running;
  const showRun = running || (pr && (pr.status === "error" || pr.status === "cancelled") && pr.stage);
  $("runBox").hidden = !showRun;
  if (pr) {
    const idx = STAGES.indexOf(pr.stage);
    document.querySelectorAll("#stageList li").forEach((li, k) => {
      li.classList.toggle("done", k < idx || pr.status === "done");
      li.classList.toggle("active", running && k === idx);
      let sm = li.querySelector("small");
      if (running && k === idx) {
        if (!sm) { sm = document.createElement("small"); li.append(sm); }
        sm.textContent = pr.message || "";
      } else if (sm) sm.remove();
    });
    if (pr.status === "error") setErr("procErr", "Processing stopped: " + (pr.message || "unknown error") + "\nOpen the log for details, or save diagnostics to send for support.");
    else if (pr.status === "cancelled" && !running) setErr("procErr", "Processing was stopped. Finished steps are kept and reused next time.");
    else setErr("procErr", "");
    if (lastProgressStatus && lastProgressStatus !== pr.status && pr.status === "done") {
      if (p && p.planned && fields().length) {
        // the planned field is already on the new imagery: go straight to refining it for spraying
        toast("Orthomosaic ready. Your field is on it: check the outline, mark obstacles, then Export boundaries for the spray drone.", 9000);
        loadResult(true).then(() => setPanel("areas"));
      } else {
        toast("Orthomosaic ready.", 5000);
        loadResult(true);
      }
    }
    lastProgressStatus = pr.status;
  }
  if (!$("logDrawer").hidden && running) loadLog();
}

function renderStatusBar(p) {
  const pr = status.progress;
  const st = $("stState");
  let frac = null, time = "";
  $("btnCancel").hidden = !status.running;
  if (busyScan()) {
    const sc = status.scan;
    st.innerHTML = "<b>Importing</b> · " + (sc.total ? "reading photo " + fmt(sc.done) + " of " + fmt(sc.total) : "looking for photos…");
    frac = sc.total ? sc.done / sc.total : 0.02;
  } else if (status.running && pr) {
    st.innerHTML = "<b>Processing</b> · ";
    st.append(STAGE_LABEL[pr.stage] || pr.message || "Starting");
    frac = pr.overall || 0;
    const elapsed = pr.started ? Math.max(pr.elapsed || 0, Date.now() / 1000 - pr.started) : pr.elapsed;
    time = fmtTime(elapsed) + (pr.eta ? " · about " + fmtTime(pr.eta) + " left" : "");
  } else if (busyExport()) {
    const ex = status.export;
    st.innerHTML = "<b>Exporting</b> · ";
    st.append((ex.current || "") + " (" + Math.min(ex.done + 1, ex.total) + " of " + ex.total + ")");
    frac = ex.total ? (ex.done + (ex.frac || 0)) / ex.total : 0;
  } else if (pr && pr.status === "error") {
    st.textContent = "Processing stopped: " + (pr.message || "error");
  } else if (!p) {
    st.textContent = "Ready";
  } else if (p.has_result) {
    st.textContent = "Orthomosaic ready";
  } else if (p.count) {
    st.textContent = "Ready to process";
  } else {
    st.textContent = "Ready";
  }
  $("stBarWrap").hidden = frac === null;
  $("stPct").textContent = frac === null ? "" : Math.round(frac * 100) + "%";
  if (frac !== null) $("stBar").style.width = Math.round(frac * 100) + "%";
  $("stTime").textContent = time;
  const info = [];
  if (p && p.count) info.push(fmt(p.count) + " photos");
  if (result && result.report) {
    if (result.report.gsd_cm) info.push(fmt(result.report.gsd_cm, 1) + " cm/px");
    if (result.report.area_ha) info.push(acHa(result.report.area_ha));
  }
  $("stInfo").textContent = info.join(" · ");
  $("btnLog").hidden = !p;
  renderExportProgress();
}

// ---- processing options dialog
let estTimer = null;
let estSeq = 0;
function openOptions() {
  const p = status.project;
  if (!p || !p.count) return;
  const s = p.settings;
  setErr("startErr", "");
  document.querySelectorAll("input[name=preset]").forEach((r) => (r.checked = r.value === s.preset));
  document.querySelectorAll("input[name=surface]").forEach((r) => (r.checked = r.value === s.surface));
  document.querySelectorAll("input[name=compression]").forEach((r) => (r.checked = r.value === s.compression));
  document.querySelectorAll("input[name=look]").forEach((r) => (r.checked = r.value === (s.look || "vivid")));
  $("optExposure").checked = !!s.exposure_balance;
  $("optTrim").checked = s.trim_edges !== false;
  $("optGpu").checked = s.gpu !== "off";
  nativeGsdCm = p.est_gsd_cm;
  let f = s.res_factor || PRESET_RES[s.preset] || 1;
  if (s.resolution_cm && nativeGsdCm) f = s.resolution_cm / nativeGsdCm;
  setSliderFactor(f);
  $("btnStart").textContent = p.has_result ? "Process again" : "Start processing";
  showIssues($("optIssues"), p);
  if ($("procDlg").open) return;
  $("procDlg").showModal();
  updateResLabel();
  requestEstimate();
}
function setSliderFactor(f) {
  let best = 0;
  RES_STEPS.forEach((v, k) => { if (Math.abs(Math.log(v / f)) < Math.abs(Math.log(RES_STEPS[best] / f))) best = k; });
  $("resSlider").value = best;
}
const sliderFactor = () => RES_STEPS[+$("resSlider").value];
function updateResLabel() {
  const f = sliderFactor();
  const el = $("resVal");
  el.innerHTML = "";
  if (nativeGsdCm) {
    const b = document.createElement("b");
    b.textContent = cmIn(nativeGsdCm * f);
    el.append(b, f === 1 ? " · full photo resolution" : " · " + fmt(f, f % 1 ? 2 : 0) + "× coarser than the photos");
  } else {
    el.textContent = f === 1 ? "Full photo resolution" : fmt(f, 2) + "× coarser than the photos";
  }
}
function readOptions() {
  const v = (n) => (document.querySelector("input[name=" + n + "]:checked") || {}).value;
  return {
    preset: v("preset") || "balanced",
    surface: v("surface") || "fields",
    compression: v("compression") || "jpeg",
    look: v("look") || "vivid",
    exposure_balance: $("optExposure").checked,
    trim_edges: $("optTrim").checked,
    gpu: $("optGpu").checked ? "auto" : "off",
    res_factor: sliderFactor(),
    resolution_cm: null,
  };
}
function requestEstimate() {
  clearTimeout(estTimer);
  estTimer = setTimeout(async () => {
    const o = readOptions();
    const seq = ++estSeq;
    $("estBox").style.opacity = 0.55;
    try {
      const e = await api("/api/estimate?preset=" + o.preset + "&res_factor=" + o.res_factor + "&compression=" + o.compression + "&gpu=" + o.gpu);
      if (seq !== estSeq) return;
      $("estBox").style.opacity = 1;
      if (!e.ok) { ["estPhotos", "estArea", "estSize", "estTime"].forEach((id) => ($(id).textContent = "–")); $("estPhotos").textContent = fmt(e.photos || 0); return; }
      if (e.gsd_native_cm) { nativeGsdCm = e.gsd_native_cm; updateResLabel(); }
      $("estPhotos").textContent = fmt(e.photos);
      $("estArea").textContent = fmt(e.area_ha * 2.47105, 0) + " ac";
      $("estArea").title = fmt(e.area_ha, 1) + " ha";
      $("estSize").textContent = fmtBytes(e.size_bytes);
      $("estTime").textContent = fmtDur(e.seconds);
      const parts = [];
      if (e.gpu) parts.push("Feature detection runs on the NVIDIA graphics card.");
      if (e.cached_features) parts.push("Features from an earlier run are reused for " + fmt(e.cached_features) + " photos.");
      parts.push("Estimates adjust to this computer after each job.");
      $("estNote").textContent = parts.join(" ");
      $("gpuRow").hidden = !e.gpu_available;
    } catch (err) { if (seq === estSeq) $("estBox").style.opacity = 1; }
  }, 180);
}
async function startProcessing() {
  setErr("startErr", "");
  try {
    await api("/api/process/start", { settings: readOptions() });
    lastProgressStatus = "starting";
    $("procDlg").close();
    setPanel("process");
    refresh();
  } catch (e) { setErr("startErr", e.message); }
}

// ------------------------------------------------------------------ result
async function loadResult(fit) {
  try { result = await api("/api/result"); } catch (e) { result = null; }
  if (!result || !result.ready) {
    result = null;
    $("resultBox").hidden = true;
    updateAreasLock(status && status.project);
    if (layers.ortho) { layers.ortho.remove(); layers.ortho = null; }
    if (map) map.setMaxZoom(11);
    return;
  }
  const r = result.report;
  let rows;
  if (r.imported) {
    rows = [["Source", "Imported GeoTIFF"], ["Resolution", cmIn(r.gsd_cm)]];
  } else {
    rows = [
      ["Photos aligned", fmt(r.photos_aligned) + " of " + fmt(r.photos_used), r.photos_aligned < r.photos_used ? "warn" : ""],
      ["Resolution", fmt(r.gsd_cm, 2) + " cm (" + fmt(r.gsd_cm / 2.54, 2) + " in)"],
      ["Area covered", r.area_ha ? acHa(r.area_ha) : null],
      ["Alignment error", r.reprojection_rms_px != null ? fmt(r.reprojection_rms_px, 1) + " px (" + fmt(r.reprojection_rms_px * r.gsd_native_cm, 1) + " cm)" : null],
      ["Seam alignment", r.seam_check ? fmt(r.seam_check.median_cm, 1) + " cm median" : null],
      ["Processing time", fmtTime(r.processing_seconds)],
    ];
  }
  facts($("resultFacts"), rows);
  $("resultBox").hidden = false;
  $("btnOutputs").hidden = !!r.imported;
  updateAreasLock(status && status.project);
  if (result.mtime !== lastResultMtime) {
    lastResultMtime = result.mtime;
    drawOrtho();
    await loadAreas();
    if (fit) fitJob();
  }
}

// The ortho is drawn by two tile layers in one pane:
//  * a quick layer (Leaflet's usual nearest zoom level), so panning and zooming respond at once;
//  * a sharp layer on top with tiles at least as detailed as the screen's real pixels (including
//    Windows display scaling). It only switches level once zooming pauses, so it never slows the
//    zoom down; until then the quick layer shows through.
function sharpZoom(layer, zoom) {
  const o = layer.options;
  const dpr = Math.min(Math.max(window.devicePixelRatio || 1, 1), 2);
  // finer tiles only when the stretch would be visible (over ~1.2x): each finer level is 4x the tiles
  // Leaflet re-syncs a tile layer whose level is more than one level away from the map zoom; a
  // finer level than that made it call itself forever (stack overflow, frozen map) at 150%
  // display scaling. So never more than one level finer than the map.
  const tz = Math.min(Math.ceil(zoom + Math.log2(dpr) - 0.27), Math.floor(zoom + 1));
  return (o.maxZoom !== undefined && tz > o.maxZoom) || (o.minZoom !== undefined && tz < o.minZoom) ? undefined : layer._clampZoom(tz);
}
const SharpTileLayer = L.TileLayer.extend({
  _setView(center, zoom, noPrune, noUpdate, now) {
    const tz = sharpZoom(this, zoom);
    if (!now && tz !== this._tileZoom) {
      clearTimeout(this._deferT);
      this._deferT = setTimeout(() => {
        this._deferT = 0;
        if (this._map) this._setView(this._map.getCenter(), this._map.getZoom(), false, false, true);
      }, 280);
      this._setZoomTransforms(center, zoom);
      return;
    }
    if (!noUpdate || now) {
      this._tileZoom = tz;
      if (this._abortLoading) this._abortLoading();
      this._updateLevels();
      this._resetGrid();
      if (tz !== undefined) this._update(center);
      if (!noPrune) this._pruneTiles();
      this._noPrune = !!noPrune;
    }
    this._setZoomTransforms(center, zoom);
  },
  _onMoveEnd() {
    if (this._deferT) return;          // a level change is pending; it loads the new view itself
    L.TileLayer.prototype._onMoveEnd.call(this);
  },
  onRemove(map) {
    clearTimeout(this._deferT);
    this._deferT = 0;
    L.TileLayer.prototype.onRemove.call(this, map);
  },
});

function drawOrtho() {
  if (!map || !result) return;
  if (layers.ortho) layers.ortho.remove();
  const b = result.bounds;
  const url = "/tiles/{z}/{x}/{y}.png?v=" + result.mtime;
  const opts = {
    tileSize: 256, minNativeZoom: 0, maxNativeZoom: result.max_native_zoom, minZoom: -4, maxZoom: 22, pane: "ortho",
    noWrap: true, bounds: L.latLngBounds([b[1], b[0]], [b[3], b[2]]), keepBuffer: 3, updateWhenZooming: false, updateInterval: 60,
  };
  const quick = L.tileLayer(url, Object.assign({ zIndex: 1 }, opts));
  const sharp = new SharpTileLayer(url, Object.assign({ zIndex: 2 }, opts));
  layers.ortho = L.layerGroup([quick, sharp]);
  layers.orthoQuick = quick;
  map.getPane("ortho").style.opacity = +$("lyrOrthoOp").value;
  if ($("lyrOrtho").checked) layers.ortho.addTo(map);
  map.setMaxZoom(result.max_native_zoom + 4);
  prefetch.seen = new Set();
  prefetch.queue = [];
  quick.on("load", () => { clearTimeout(prefetch.timer); prefetch.timer = setTimeout(prefetchTiles, 150); });
}

// While the map is idle, fetch the tiles just outside the view and the next zoom level in, so
// panning and zooming in show finished tiles straight away (tiles are cached by the browser).
const prefetch = { seen: new Set(), queue: [], active: 0, timer: 0 };
function prefetchTiles() {
  const lyr = layers.orthoQuick;
  if (!lyr || !map.hasLayer(lyr) || lyr._tileZoom === undefined) return;
  const z = lyr._tileZoom, maxz = result.max_native_zoom, b = map.getBounds();
  const want = [];
  const range = (zz, pad, frac) => {
    const c = b.getCenter(), h = (b.getNorth() - b.getSouth()) * frac / 2, w = (b.getEast() - b.getWest()) * frac / 2;
    const p0 = map.project(L.latLng(c.lat + h, c.lng - w), zz).divideBy(256).floor();
    const p1 = map.project(L.latLng(c.lat - h, c.lng + w), zz).divideBy(256).floor();
    for (let y = p0.y - pad; y <= p1.y + pad; y++)
      for (let x = p0.x - pad; x <= p1.x + pad; x++) {
        const coords = L.point(x, y); coords.z = zz;
        if (lyr._isValidTile(coords)) want.push("/tiles/" + zz + "/" + x + "/" + y + ".png?v=" + result.mtime);
      }
  };
  range(z, 1, 1);                       // a ring of tiles around the view
  prefetch.queue = want.filter(u => !prefetch.seen.has(u)).slice(0, 60);
  pumpPrefetch();
}
function pumpPrefetch() {
  while (prefetch.active < 2 && prefetch.queue.length) {
    const u = prefetch.queue.shift();
    prefetch.seen.add(u);
    prefetch.active++;
    const img = new Image();
    img.fetchPriority = "low";
    img.onload = img.onerror = () => { prefetch.active--; pumpPrefetch(); };
    img.src = u + "&pf=1";     // warms the app's tile cache; the server skips it when busy
  }
}

// ------------------------------------------------------------------ map
function initMap() {
  const crs = L.extend({}, L.CRS.Simple, { transformation: new L.Transformation(1 / R0, 0, -1 / R0, 0) });
  map = L.map("map", { crs, minZoom: -4, maxZoom: 15, zoomSnap: 0.25, zoomDelta: 0.5, attributionControl: false, zoomControl: false, wheelPxPerZoomLevel: 90 });
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.control.scale({ position: "bottomright", imperial: true, metric: true }).addTo(map);
  map.setView([0, 0], 2);
  map.createPane("base").style.zIndex = 120;    // satellite map, under everything
  map.createPane("ortho").style.zIndex = 150;   // below the areas and markers
  layers.areas = L.layerGroup().addTo(map);
  map.on("mousedown", onMapDown);
  map.on("mousemove", onMapMove);
  map.on("mouseup", onMapUp);
  map.on("click", onMapClick);
  map.on("dblclick", onMapDbl);
  map.on("movestart zoomstart", () => { prefetch.queue = []; clearTimeout(prefetch.timer); });   // the view comes first
}
function clearLayers() {
  if (edit) endEdit();
  for (const k of ["cams", "ortho", "draft"]) if (layers[k]) { layers[k].remove(); layers[k] = null; }
  if (layers.areas) layers.areas.clearLayers();
  renderAreaList();
}
function fitJob() {
  if (!map) return;
  let bb = null;
  if (result) {
    const b = result.bounds;
    bb = L.latLngBounds([b[1], b[0]], [b[3], b[2]]);
  } else if (photos && photos.points.length) {
    bb = L.latLngBounds(photos.points.map((p) => [p[1], p[0]])).pad(0.15);
  } else if (areas.length) {
    bb = L.latLngBounds(areas.flatMap((a) => a.rings[0].map((q) => [q[1], q[0]]))).pad(0.2);
  } else if (status && status.project && status.project.has_frame) {
    map.setView([0, 0], 5);     // a planned job without a field yet: the spot that was found
    return;
  }
  if (bb && bb.isValid()) map.fitBounds(bb, { padding: [30, 30], maxZoom: 8 });
}

// ------------------------------------------------------------------ satellite base map
// USDA NAIP imagery, drawn in the job's own map frame by the PlaneSite server (and kept in the
// project folder), so it lines up with the fields and the orthomosaic.
function drawBase() {
  if (!map) return;
  if (layers.base) { layers.base.remove(); layers.base = null; }
  const p = status && status.project;
  $("mapCredit").hidden = true;
  if (!p || !p.frame) return;
  layers.base = L.tileLayer("/basemap/{z}/{x}/{y}?o=" + p.frame, {
    tileSize: 256, minZoom: -4, maxZoom: 22, minNativeZoom: -4, maxNativeZoom: 7, pane: "base",
    keepBuffer: 2, updateWhenZooming: false, updateInterval: 120 });
  if ($("lyrBase").checked) { layers.base.addTo(map); $("mapCredit").hidden = false; }
}
function updateAreasLock(p) {
  const ok = !!(p && (p.has_frame || result));
  $("areasLocked").hidden = ok;
  $("areasTools").hidden = !ok;
  if (!p) return;
  const planning = !!(p.planned && !result);
  $("planBox").hidden = !planning;
  $("areasIntro").hidden = planning;
  $("areasIntro").textContent = result
    ? (p.planned ? "Adjust the field to the new imagery (pencil), mark obstacles, then Export boundaries for the spray drone."
                 : "Split the orthomosaic into fields or tiles and export each one as its own GeoTIFF. Mark obstacles (trees, poles, buildings) inside a field for spray drones.")
    : "Draw fields on the satellite map now; they stay in place when the orthomosaic is ready.";
  $("toolGrid").disabled = !result;
  $("toolGrid").title = result ? "" : "Splitting into tiles needs the orthomosaic";
  $("btnExportAreas").hidden = !result;
  const f = fields();
  $("pbDraw").classList.toggle("done", f.length > 0);
  $("pbSend").classList.toggle("done", f.some((a) => a.pilot2 && a.pilot2.sent));
  $("pbFly").classList.toggle("done", !!p.count);
}

// ------------------------------------------------------------------ plan a mission
const pl = { kind: null, path: null, lib: null, libSel: null, pt: null, nameEdited: false, last: null, map: null, pin: null };
// Picking the spot: an ordinary web map (USGS imagery with roads and place names), separate from the
// job's own map, which only exists once the job has a location.
const PL_TILES = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryTopo/MapServer/tile/{z}/{y}/{x}";
function plMapInit() {
  if (pl.map) { setTimeout(() => pl.map.invalidateSize(), 0); return; }
  pl.map = L.map("plMap", { zoomSnap: 0.5, attributionControl: true, worldCopyJump: true });
  L.tileLayer(PL_TILES, { maxNativeZoom: 16, maxZoom: 19, attribution: "USGS The National Map" }).addTo(pl.map);
  pl.map.attributionControl.setPrefix(false);
  pl.map.on("click", (e) => plSetPoint(e.latlng.lat, e.latlng.lng, "Picked on the map"));
  if (pl.last) pl.map.setView([pl.last.lat, pl.last.lon], 13); else pl.map.setView([38.5, -96.5], 4);
  setTimeout(() => pl.map.invalidateSize(), 0);
}
function plSetPoint(lat, lon, label, fly) {
  pl.pt = { lat, lon };
  const icon = L.divIcon({ className: "plpin", html: '<svg viewBox="0 0 22 22"><path d="M11 21s7-7.2 7-12a7 7 0 0 0-14 0c0 4.8 7 12 7 12z"/><circle cx="11" cy="9" r="2.6" fill="#2A2100"/></svg>', iconSize: null });
  if (!pl.pin) {
    pl.pin = L.marker([lat, lon], { icon, draggable: true, keyboard: false }).addTo(pl.map);
    pl.pin.on("dragend", () => { const ll = pl.pin.getLatLng(); plSetPoint(ll.lat, ll.lng, "Moved on the map"); });
  } else pl.pin.setLatLng([lat, lon]);
  if (fly) pl.map.flyTo([lat, lon], fly, { duration: 0.8 });
  $("plFound").textContent = label + ": " + lat.toFixed(5) + ", " + lon.toFixed(5) + ". The job's map starts here.";
  $("plFound").hidden = false;
  plSync();
}
async function openPlan() {
  Object.assign(pl, { kind: null, path: null, libSel: null, pt: null, nameEdited: false });
  if (pl.pin) { pl.pin.remove(); pl.pin = null; }
  $("plName").value = "";
  $("plQ").value = "";
  $("plLibQ").value = "";
  $("plFound").hidden = true;
  setErr("plErr", "");
  plMode(null);
  try { const r = await api("/api/plan/start"); $("plParent").value = r.dir; pl.last = r.last; }
  catch (e) { $("plParent").value = (status && status.default_dir) || ""; pl.last = null; }
  $("plLast").hidden = !pl.last;
  if (pl.last) $("plLast").textContent = "Go to the last job" + (pl.last.name ? " (" + pl.last.name + ")" : "");
  $("planDlg").showModal();
}
function plMode(kind) {
  pl.kind = kind;
  for (const [k, id] of [["file", "File"], ["lib", "Lib"], ["find", "Find"]]) {
    $("pl" + id).setAttribute("aria-pressed", kind === k);
    $("pl" + id + "Pane").hidden = kind !== k;
  }
  plSync();
}
function plSuggest(name) { if (!pl.nameEdited && name) { $("plName").value = name; plSync(); } }
function plSync() {
  const ready = (pl.kind === "file" && pl.path) || (pl.kind === "lib" && pl.libSel) || (pl.kind === "find" && pl.pt);
  $("plCreate").disabled = !(ready && $("plName").value.trim());
}
async function plPickFile() {
  const f = await browse("file", "Choose the field boundary (KML, KMZ, Shapefile, GeoJSON)", "");
  if (!f) { if (!pl.path) plMode(null); return; }
  pl.path = f;
  $("plFileName").textContent = f;
  plMode("file");
  plSuggest(f.split(/[\\/]/).pop().replace(/\.[^.]+$/, "").replace(/[_]+/g, " ").replace(/\s*boundary$/i, ""));
}
async function plOpenLib() {
  plMode("lib");
  if (!pl.lib) {
    $("plLibList").innerHTML = '<li class="empty">Looking through your projects…</li>';
    try { pl.lib = (await api("/api/fields/library")).fields; } catch (e) { pl.lib = []; }
  }
  plRenderLib();
  $("plLibQ").focus();
}
function plRenderLib() {
  const ul = $("plLibList");
  const q = $("plLibQ").value.trim().toLowerCase();
  const rows = pl.lib.filter((f) => !q || (f.name + " " + f.project).toLowerCase().includes(q));
  ul.innerHTML = "";
  if (!rows.length) {
    ul.innerHTML = '<li class="empty"></li>';
    ul.firstChild.textContent = pl.lib.length ? "No field matches that search." : "No fields in your projects yet. Use a boundary file or find the field on the map.";
    return;
  }
  for (const f of rows.slice(0, 150)) {
    const li = document.createElement("li");
    li.innerHTML = '<label><input type="radio" name="plfield"><b></b><small></small><span class="ac"></span></label>';
    const r = li.querySelector("input");
    r.checked = !!(pl.libSel && pl.libSel.dir === f.dir && pl.libSel.id === f.id);
    r.onchange = () => { pl.libSel = { dir: f.dir, id: f.id }; plSuggest(f.name); plSync(); };
    li.querySelector("b").textContent = f.name;
    li.querySelector("small").textContent = f.project + " · " + new Date(f.modified * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) +
      (f.obstacles ? " · " + f.obstacles + (f.obstacles === 1 ? " obstacle" : " obstacles") : "");
    li.querySelector(".ac").textContent = fmt(f.ac, f.ac < 10 ? 1 : 0) + " ac";
    ul.append(li);
  }
}
async function plFindGo() {
  setErr("plErr", "");
  const q = $("plQ").value.trim();
  if (!q) return;
  $("plGo").disabled = true;
  try {
    const r = await api("/api/locate", { q }, 30000);
    plMapInit();
    // a street address or coordinates are the spot itself; a town or county is only where to look
    const exact = /^\s*\d+\s+\S/.test(q) || !/OpenStreetMap/.test(r.label);
    plSetPoint(r.lat, r.lon, "Found " + r.label, exact ? 16 : 12);
    if (!exact) $("plFound").textContent += " Zoom in and click the field to start right on it.";
  } catch (e) { setErr("plErr", e.message); }
  finally { $("plGo").disabled = false; plSync(); }
}
async function plCreate() {
  setErr("plErr", "");
  const name = $("plName").value.trim();
  if (!name) return setErr("plErr", "Enter a name for the job.");
  const where = pl.kind === "file" ? { kind: "file", path: pl.path }
    : pl.kind === "lib" ? { kind: "library", dir: pl.libSel.dir, id: pl.libSel.id }
    : { kind: "point", lat: pl.pt.lat, lon: pl.pt.lon };
  $("plCreate").disabled = true;
  try {
    const r = await api("/api/plan/new", { name, parent: $("plParent").value.trim(), where }, 60000);
    $("planDlg").close();
    $("dash").hidden = true;
    lastProjectDir = undefined;
    await refresh();
    setPanel("areas");
    if (!r.fields) {
      setTool("poly");
      toast("Draw the field: click its corners on the satellite map, then click the first corner to finish.", 7000);
    } else {
      toast(r.fields === 1 ? "Field ready. Check the outline, then send it to Pilot 2." : r.fields + " fields ready. Check the outlines, then send them to Pilot 2.", 6000);
    }
  } catch (e) { setErr("plErr", e.message); plSync(); }
}

// ------------------------------------------------------------------ tools
const TOOL_HINTS = {
  select: "Drag a box around the photos you want to process.",
  rect: "Drag on the map to draw a rectangle.",
  poly: "Click to place corners; drag to move the map. Click the first corner or double-click to finish. Backspace removes the last corner, Esc cancels.",
  obst: "Obstacle: click around the tree, pole or building inside a field; click the first corner or double-click to finish. Backspace removes the last corner, Esc cancels.",
};
const isPoly = (t) => t === "poly" || t === "obst";
const OBST = "#D2452F";
function setTool(t) {
  if (edit && t) finishEdit(true);
  tool = tool === t ? null : t;
  cancelDraft();
  $("toolRect").setAttribute("aria-pressed", tool === "rect");
  $("toolPoly").setAttribute("aria-pressed", tool === "poly");
  $("toolObst").setAttribute("aria-pressed", tool === "obst");
  $("btnSelectArea").classList.toggle("secondary", tool === "select");
  $("toolHint").textContent = tool && tool !== "select" ? TOOL_HINTS[tool] : "";
  if (tool === "select") toast(TOOL_HINTS.select);
  map.getContainer().classList.toggle("cursor-cross", !!tool);
  // Rectangles and the photo selection box are drawn by dragging, so the map can't pan then.
  // Polygons are drawn by clicking: dragging still pans the map (Leaflet sends no click after a drag).
  if (tool === "rect" || tool === "select") map.dragging.disable(); else map.dragging.enable();
  if (tool) map.doubleClickZoom.disable(); else if (!edit) map.doubleClickZoom.enable();
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
  if (isPoly(tool) && draw.pts && draw.pts.length) layers.draft.setLatLngs([...draw.pts, e.latlng]);
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
  addArea({ name: nextName("field"), kind: "drawn", rings: [[[w, s], [en, s], [en, n], [w, n], [w, s]]] });
}
function onMapClick(e) {
  if (edit && !tool) { if (edit.sel) { edit.sel = null; renderEdit(); } return; }
  if (!isPoly(tool)) return;
  if (!draw) {
    draw = { pts: [e.latlng] };
    layers.draft = L.polyline([e.latlng], { color: tool === "obst" ? OBST : "#E9B20F", weight: 2, dashArray: "6 4" }).addTo(map);
    return;
  }
  const first = map.latLngToContainerPoint(draw.pts[0]);
  if (draw.pts.length >= 3 && first.distanceTo(e.containerPoint) < 10) return finishPoly();
  draw.pts.push(e.latlng);
  layers.draft.setLatLngs(draw.pts);
}
function onMapDbl() { if (isPoly(tool) && draw && draw.pts.length >= 3) finishPoly(); }
function finishPoly() {
  let pts = draw.pts;
  if (pts.length > 3) {
    const a = map.latLngToContainerPoint(pts[pts.length - 1]), b = map.latLngToContainerPoint(pts[pts.length - 2]);
    if (a.distanceTo(b) < 4) pts = pts.slice(0, -1);
  }
  const ring = pts.map((p) => [p.lng, p.lat]);
  ring.push(ring[0]);
  cancelDraft();
  if (tool === "obst") addArea({ name: nextName("obstacle"), kind: "drawn", role: "obstacle", rings: [ring] });
  else addArea({ name: nextName("field"), kind: "drawn", rings: [ring] });
}
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  if (edit) {
    if (e.key === "Escape") { cancelEdit(); e.preventDefault(); return; }
    if (e.key === "Enter") { finishEdit(true); e.preventDefault(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { undoEdit(); e.preventDefault(); return; }
    if ((e.key === "Delete" || e.key === "Backspace") && edit.sel) { removeCorner(edit.sel[0], edit.sel[1]); e.preventDefault(); return; }
  }
  if (e.key === "Escape") { if (draw) cancelDraft(); else if (tool) setTool(null); else if (!$("dash").hidden && status.project) $("dash").hidden = true; }
  if (e.key === "Backspace" && isPoly(tool) && draw && draw.pts.length > 1) {
    draw.pts.pop();
    layers.draft.setLatLngs(draw.pts);
    e.preventDefault();
  }
});

// ------------------------------------------------------------------ edit an area outline
// Corners are kept per ring without the closing point while editing; the map units are metres.
let edit = null;   // { id, orig, rings, undo, sel, group }
const ringsOpen = (rings) => rings.map((r) => {
  const c = r.map((p) => [p[0], p[1]]);
  if (c.length > 1 && c[0][0] === c[c.length - 1][0] && c[0][1] === c[c.length - 1][1]) c.pop();
  return c;
});
const ringsClosed = (rings) => rings.map((r) => [...r.map((p) => [p[0], p[1]]), [r[0][0], r[0][1]]]);
function ringAreaM2(r) {
  let s = 0;
  for (let i = 0; i < r.length; i++) { const a = r[i], b = r[(i + 1) % r.length]; s += a[0] * b[1] - b[0] * a[1]; }
  return Math.abs(s) / 2;
}
function editAreaHa() {
  if (!edit) return 0;
  return Math.max(0, edit.rings.reduce((s, r, i) => s + (i === 0 ? 1 : -1) * ringAreaM2(r), 0)) / 1e4;
}
function startEdit(id) {
  if (edit && edit.id === id) return;
  if (edit) finishEdit(true);
  if (tool) setTool(null);
  const a = areas.find((x) => x.id === id);
  if (!a || !map) return;
  edit = { id, orig: JSON.parse(JSON.stringify(a.rings)), rings: ringsOpen(a.rings), undo: [], sel: null,
           group: L.layerGroup().addTo(map) };
  map.doubleClickZoom.disable();
  $("editBar").hidden = false;
  $("editLabel").textContent = "Editing " + a.name;
  focusArea(id, true);
  renderAreaList();
  renderEdit();
}
function pushUndo() {
  edit.undo.push(JSON.parse(JSON.stringify(edit.rings)));
  if (edit.undo.length > 100) edit.undo.shift();
  $("editUndo").disabled = false;
}
function undoEdit() {
  if (!edit || !edit.undo.length) return;
  edit.rings = edit.undo.pop();
  edit.sel = null;
  $("editUndo").disabled = !edit.undo.length;
  renderEdit();
}
function removeCorner(ri, k) {
  if (!edit) return;
  const r = edit.rings[ri];
  if (r.length <= 3) {
    if (ri > 0) { pushUndo(); edit.rings.splice(ri, 1); edit.sel = null; renderEdit(); }  // a hole: remove it
    else toast("An area needs at least 3 corners.");
    return;
  }
  pushUndo();
  r.splice(k, 1);
  edit.sel = null;
  renderEdit();
}
function editShape() {
  const a = areas.find((x) => x.id === edit.id);
  if (a && a._layer) a._layer.setLatLngs(edit.rings.map((r) => r.map((p) => [p[1], p[0]])));
  const ha = editAreaHa();
  $("editSize").textContent = a && a.role === "obstacle" ? sizeText({ ha, ac: ha * 2.47105, role: "obstacle" }) :
    fmt(ha * 2.47105, ha * 2.47105 < 10 ? 2 : 1) + " ac (" + fmt(ha, 2) + " ha)";
}
function renderEdit() {
  if (!edit) return;
  edit.group.clearLayers();
  editShape();
  const icon = (cls) => L.divIcon({ className: "vtx" + cls, iconSize: null });
  edit.rings.forEach((r, ri) => {
    r.forEach((p, k) => {
      const sel = edit.sel && edit.sel[0] === ri && edit.sel[1] === k;
      const m = L.marker([p[1], p[0]], { icon: icon(sel ? " sel" : ""), draggable: true, keyboard: false, zIndexOffset: 1000 });
      m.on("dragstart", () => { pushUndo(); edit.sel = [ri, k]; });
      m.on("drag", (e) => { const ll = e.target.getLatLng(); r[k] = [ll.lng, ll.lat]; editShape(); });
      m.on("dragend", () => renderEdit());
      m.on("click", (e) => { L.DomEvent.stop(e); edit.sel = [ri, k]; renderEdit(); });
      m.on("contextmenu", (e) => { L.DomEvent.stop(e); removeCorner(ri, k); });
      m.addTo(edit.group);
      // middle dot to the next corner: drag (or click) to add a corner there
      const q = r[(k + 1) % r.length];
      const mid = L.marker([(p[1] + q[1]) / 2, (p[0] + q[0]) / 2], { icon: icon(" mid"), draggable: true, keyboard: false, zIndexOffset: 900 });
      let added = false;
      const add = () => { if (!added) { pushUndo(); r.splice(k + 1, 0, [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2]); added = true; } };
      mid.on("dragstart", add);
      mid.on("drag", (e) => { const ll = e.target.getLatLng(); r[k + 1] = [ll.lng, ll.lat]; editShape(); });
      mid.on("dragend", () => { edit.sel = [ri, k + 1]; renderEdit(); });
      mid.on("click", (e) => { L.DomEvent.stop(e); add(); edit.sel = [ri, k + 1]; renderEdit(); });
      mid.addTo(edit.group);
    });
  });
}
function endEdit() {
  if (!edit) return;
  edit.group.remove();
  edit = null;
  map.doubleClickZoom.enable();
  $("editBar").hidden = true;
}
async function finishEdit(save) {
  if (!edit) return;
  const a = areas.find((x) => x.id === edit.id);
  const changed = save && a && JSON.stringify(ringsClosed(edit.rings)) !== JSON.stringify(edit.orig);
  if (a) a.rings = save ? ringsClosed(edit.rings) : edit.orig;
  endEdit();
  if (changed) await saveAreas(); else drawAreas();
}
function cancelEdit() { finishEdit(false); }

// ------------------------------------------------------------------ areas
function nextName(role) {
  const word = role === "obstacle" ? "Obstacle " : status && status.project && status.project.planned ? "Field " : "Area ";
  let k = areas.filter((a) => (a.role === "obstacle") === (role === "obstacle")).length + 1;
  const names = new Set(areas.map((a) => a.name));
  while (names.has(word + k)) k++;
  return word + k;
}
const isObst = (a) => a.role === "obstacle";
const fields = () => areas.filter((a) => !isObst(a));
// Obstacles are small: square feet up to a quarter acre, acres above that.
function sizeText(a) {
  if (a.role === "obstacle" && a.ac < 0.25) return fmt(a.ha * 107639.104, 0) + " ft²";
  return fmt(a.ac, a.ac < 10 ? 2 : 1) + " ac";
}
async function loadAreas() {
  try { areas = (await api("/api/areas")).areas; } catch (e) { areas = []; }
  drawAreas();
}
const plain = () => areas.map((a) => ({ id: a.id, name: a.name, kind: a.kind, role: a.role, rings: a.rings }));
async function saveAreas() {
  try { areas = (await api("/api/areas", { areas: plain() })).areas; drawAreas(); }
  catch (e) { toast(e.message); }
}
async function addArea(a) {
  a.id = "";
  areas.push(a);
  try { areas = (await api("/api/areas", { areas: plain() })).areas; } catch (e) { toast(e.message); }
  drawAreas();
  const last = areas[areas.length - 1];
  if (last) {
    focusArea(last.id, !isObst(last));
    if (isObst(last) && !last.parent) toast(last.name + " is not inside a field, so it won't be exported with one. Draw obstacles inside a field outline.", 7000);
  }
}
function drawAreas() {
  if (!layers.areas) return;
  layers.areas.clearLayers();
  for (const a of [...fields(), ...areas.filter(isObst)]) {     // obstacles on top so they can be clicked
    const ll = a.rings.map((r) => r.map((p) => [p[1], p[0]]));
    const ob = isObst(a);
    const poly = L.polygon(ll, ob ? { color: OBST, weight: 2, fillColor: OBST, fillOpacity: 0.25 }
                                  : { color: "#E9B20F", weight: 2.5, fillColor: "#E9B20F", fillOpacity: 0.08 });
    poly.bindTooltip(a.name, { permanent: !ob, direction: "center", className: "area-label" + (ob ? " obst" : "") });
    poly.on("click", (e) => { if (!tool) { L.DomEvent.stop(e); setPanel("areas"); focusArea(a.id); } });
    poly.on("dblclick", (e) => { if (!tool) { L.DomEvent.stop(e); setPanel("areas"); startEdit(a.id); } });
    poly.addTo(layers.areas);
    a._layer = poly;
  }
  renderAreaList();
  if (edit) renderEdit();
}
function areaRow(a) {
  const ob = isObst(a);
  const li = document.createElement("li");
  li.dataset.id = a.id;
  if (ob) li.className = "obs";
  const nm = document.createElement("input");
  nm.type = "text";
  nm.value = a.name;
  nm.setAttribute("aria-label", ob ? "Obstacle name" : "Area name");
  nm.onchange = () => { a.name = nm.value.trim() || a.name; saveAreas(); };
  nm.onfocus = () => focusArea(a.id, false, true);
  const sz = document.createElement("span");
  sz.className = "size";
  sz.textContent = sizeText(a);
  sz.title = fmt(a.ha, 3) + " ha" + (!ob && a.obstacles ? " · " + a.obstacles + (a.obstacles === 1 ? " obstacle" : " obstacles") : "");
  const del = document.createElement("button");
  del.className = "del";
  del.textContent = "×";
  del.title = ob ? "Remove obstacle" : "Remove area";
  del.setAttribute("aria-label", "Remove " + a.name);
  del.onclick = () => { if (edit && edit.id === a.id) cancelEdit(); areas = areas.filter((x) => x.id !== a.id); saveAreas(); };
  const ed = document.createElement("button");
  ed.className = "edit";
  ed.innerHTML = '<svg aria-hidden="true"><use href="#i-pencil"/></svg>';
  ed.title = "Edit outline";
  ed.setAttribute("aria-label", "Edit the outline of " + a.name);
  ed.setAttribute("aria-pressed", !!(edit && edit.id === a.id));
  ed.onclick = () => (edit && edit.id === a.id ? finishEdit(true) : startEdit(a.id));
  let send = document.createElement("span");
  if (!ob) {
    send = document.createElement("button");
    const sentAt = a.pilot2 && a.pilot2.sent && a.pilot2.sent.at;
    send.className = "send" + (sentAt ? " sent" : "");
    send.innerHTML = '<svg aria-hidden="true"><use href="#i-send"/></svg>';
    send.title = sentAt ? "Send to Pilot 2 (last sent " + sentAt + ")" : "Send to Pilot 2";
    send.setAttribute("aria-label", "Send " + a.name + " to Pilot 2");
    send.onclick = () => openPilot2(a.id);
  }
  li.append(nm, sz, ed, send, del);
  return li;
}
function renderAreaList() {
  const ul = $("areaList");
  ul.innerHTML = "";
  const obs = areas.filter(isObst);
  for (const a of fields()) {
    ul.append(areaRow(a));
    for (const o of obs) if (o.parent === a.id) ul.append(areaRow(o));
  }
  const orphans = obs.filter((o) => !o.parent || !areas.some((a) => a.id === o.parent));
  if (orphans.length) {
    const h = document.createElement("li");
    h.className = "orphan-head";
    h.textContent = "Obstacles outside any field";
    ul.append(h);
    for (const o of orphans) ul.append(areaRow(o));
  }
  const f = fields();
  $("areaHead").hidden = !areas.length;
  $("areaActions").hidden = !areas.length;
  $("btnExportAreas").disabled = !f.length;
  $("btnOutlines").disabled = !f.length;
  const tot = f.reduce((s, a) => s + a.ac, 0);
  updateAreasLock(status && status.project);
  $("areaCount").textContent = f.length + (f.length === 1 ? " area · " : " areas · ") + fmt(tot, tot < 10 ? 1 : 0) + " ac" +
    (obs.length ? " · " + obs.length + (obs.length === 1 ? " obstacle" : " obstacles") : "");
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

// ------------------------------------------------------------------ send to Pilot 2
const p2 = { id: null, timer: 0, conn: null, seq: 0 };
const p2Fields = { height_m: "p2Height", speed_ms: "p2Speed", margin_m: "p2Margin", overlap_front: "p2OvH", overlap_side: "p2OvW", direction: "p2Dir" };
async function openPilot2(id) {
  const a = areas.find((x) => x.id === id);
  if (!a) return;
  p2.id = id;
  setErr("p2Err", "");
  $("p2Ok").hidden = true;
  $("p2Field").textContent = a.name;
  $("p2Est").textContent = "Working out the mission…";
  $("p2Notes").innerHTML = "";
  $("p2Name").value = "";
  try {
    p2.conn = await api("/api/pilot2/settings");
    $("p2Url").value = p2.conn.hub_url;
    $("p2User").value = p2.conn.user || "";
    $("p2Key").value = "";
    p2ConnState();
    const r = await api("/api/pilot2/preview", { id });
    p2Fill(r, true);
  } catch (e) { setErr("p2Err", e.message); }
  if (!$("p2Dlg").open) $("p2Dlg").showModal();
}
function p2ConnState() {
  const c = p2.conn || {};
  $("p2ConnState").textContent = c.has_key ? "· connected with the saved office password" : "· enter the office password to send";
  $("p2Conn").open = !c.has_key;
}
function p2Read() {
  const s = { auto_direction: $("p2Auto").checked };
  for (const [k, id] of Object.entries(p2Fields)) { const v = $(id).value; if (v !== "") s[k] = +v; }
  return s;
}
function p2Fill(r, first) {
  const s = r.settings;
  for (const [k, id] of Object.entries(p2Fields)) {
    if (first || k === "direction") $(id).value = k === "height_m" ? +(+s[k]).toFixed(1) : Math.round(s[k] * 10) / 10;
  }
  if (first) { $("p2Auto").checked = !!s.auto_direction; $("p2Name").value = r.name; }
  $("p2Dir").disabled = $("p2Auto").checked;
  $("p2Axis").textContent = "(" + Math.round(s.long_axis) + "°)";
  const e = r.estimates;
  $("p2Gsd").textContent = "≈ " + fmt(e.gsd_cm, 1) + " cm per pixel";
  $("p2Est").innerHTML = "";
  const parts = [
    ["Field", acHa(e.area_ha)],
    ["Flight lines", e.lines + " lines, " + fmt(e.line_spacing_m, 1) + " m apart"],
    ["Photos", "≈ " + fmt(e.photos) + " (every " + fmt(e.photo_spacing_m, 1) + " m)"],
    ["Flight", "≈ " + fmt(e.length_km, 1) + " km, " + fmtDur(e.minutes * 60).replace("≈ ", "") + ", " + e.batteries + (e.batteries === 1 ? " battery" : " batteries")],
  ];
  for (const [k, v] of parts) { const b = document.createElement("b"); b.textContent = k + ": "; $("p2Est").append(b, v, document.createElement("br")); }
  const small = document.createElement("small");
  small.textContent = "Office estimate. Pilot 2 works out the exact path." + (r.last ? " Last sent " + r.last.at + "." : "");
  $("p2Est").append(small);
  const ul = $("p2Notes");
  ul.innerHTML = "";
  for (const n of r.notes || []) { const li = document.createElement("li"); li.textContent = n; ul.append(li); }
}
function p2Changed() {
  clearTimeout(p2.timer);
  $("p2Dir").disabled = $("p2Auto").checked;
  p2.timer = setTimeout(async () => {
    const seq = ++p2.seq;
    try {
      const r = await api("/api/pilot2/preview", { id: p2.id, settings: p2Read() });
      if (seq === p2.seq) { setErr("p2Err", ""); p2Fill(r, false); }
    } catch (e) { if (seq === p2.seq) setErr("p2Err", e.message); }
  }, 250);
}
async function p2SaveConn() {
  const c = p2.conn || {};
  const body = {};
  if ($("p2Url").value.trim() !== (c.hub_url || "")) body.hub_url = $("p2Url").value.trim();
  if ($("p2User").value.trim() !== (c.user || "")) body.user = $("p2User").value.trim();
  if ($("p2Key").value) body.key = $("p2Key").value;
  if (!Object.keys(body).length) return;
  p2.conn = await api("/api/pilot2/settings", body);
  $("p2Key").value = "";
  p2ConnState();
}
async function p2Go(upload) {
  setErr("p2Err", "");
  $("p2Ok").hidden = true;
  const btn = upload ? $("p2Send") : $("p2Save");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = upload ? "Sending…" : "Saving…";
  try {
    await p2SaveConn();
    if (upload && !(p2.conn && p2.conn.has_key)) throw new Error("Enter the hub's office password under Hub connection.");
    const r = await api("/api/pilot2/send", { id: p2.id, settings: p2Read(), name: $("p2Name").value, upload });
    $("p2Name").value = r.name;
    let msg;
    if (upload) {
      msg = "Sent “" + r.name + "”" + (r.hub && r.hub.replaced ? ", replacing the earlier version on the hub" : "") +
        ". In Pilot 2 it is in Flight Route → cloud library: download it to open the boundary and flight lines.";
    } else {
      msg = "Saved " + r.path + ".";
    }
    $("p2Ok").textContent = msg;
    $("p2Ok").hidden = false;
    const a = areas.find((x) => x.id === p2.id);
    if (a && r.pilot2) { a.pilot2 = r.pilot2; renderAreaList(); }
  } catch (e) {
    setErr("p2Err", e.message);
    if (/password/i.test(e.message)) $("p2Conn").open = true;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

// ------------------------------------------------------------------ export boundaries
// Mapping missions (DJI Pilot 2) take the field outline only. Spray operations take the field and
// the obstacles inside it; DJI Agras also takes the rate, height and spray width.
const BND_HOW = {
  dji_mapping: "One KML per field. Copy it to the remote controller and import it in Pilot 2 to plan a mapping mission there. " +
    "To send a ready-made mission with height and overlap instead, use the send button next to the field.",
  dji_agras: "Writes a DJI\\ShapeFile folder. Copy the DJI folder to the root of the remote controller's microSD card or a USB drive, " +
    "then import the field in the Agras app. Each field is one shape with the obstacles cut out.",
  xag: "Writes a folder with a KML and a GeoJSON for each field. Import the KML in XAG's field manager (Field Import) or the XAG One app.",
  ceres: "One KML per field, boundary only. In Applicator View: Manage Fields → Import Fields.",
  generic: "Boundaries and obstacles as separate shapes tagged BOUNDARY and OBSTACLE, as Pix4Dfields writes them.",
};
const BND_OBS = { dji_mapping: false, ceres: false };
const bnd = { dir: "" };
const bndTarget = () => document.querySelector("input[name=bndtarget]:checked").value;
async function openOutlines() {
  setErr("bndErr", "");
  $("bndOk").hidden = true;
  $("bndOpen").hidden = true;
  let st = { last: {}, default_dir: "" };
  try { st = await api("/api/areas/outlines/settings"); } catch (e) { /* defaults */ }
  const last = st.last || {};
  const t = last.target && document.querySelector('input[name=bndtarget][value="' + last.target + '"]');
  if (t) t.checked = true;
  const f = last.format && document.querySelector('input[name=bndfmt][value="' + last.format + '"]');
  if (f) f.checked = true;
  if (last.per_field != null) $("bndPerField").checked = !!last.per_field;
  if (last.obstacles != null) $("bndObs").checked = !!last.obstacles;
  const sp = last.spray || {};
  if (sp.rate_gal_ac != null) $("bndRate").value = sp.rate_gal_ac;
  if (sp.height_ft != null) $("bndHeight").value = sp.height_ft;
  if (sp.width_ft != null) $("bndWidth").value = sp.width_ft;
  $("bndDir").value = last.out_dir || st.default_dir || "";
  const ul = $("bndList");
  ul.innerHTML = "";
  const focus = document.querySelector("#areaList li.focus");
  const focusId = focus && focus.dataset.id;
  for (const a of fields()) {
    const li = document.createElement("li");
    li.innerHTML = '<label class="check"><input type="checkbox"> <span></span><span class="obsn"></span><span class="size"></span></label>';
    const c = li.querySelector("input");
    c.dataset.id = a.id;
    c.checked = true;
    c.onchange = syncOutlines;
    li.querySelector("span").textContent = a.name;
    li.querySelector(".obsn").textContent = a.obstacles ? a.obstacles + (a.obstacles === 1 ? " obstacle" : " obstacles") : "";
    li.querySelector(".size").textContent = sizeText(a);
    ul.append(li);
  }
  // one field picked in the list: start with just that one
  if (focusId && fields().some((a) => a.id === focusId))
    ul.querySelectorAll("input").forEach((c) => (c.checked = c.dataset.id === focusId));
  syncOutlines();
  $("bndDlg").showModal();
}
function syncOutlines() {
  const t = bndTarget();
  $("bndHow").textContent = BND_HOW[t];
  $("bndGeneric").hidden = t !== "generic";
  $("bndSpray").hidden = t !== "dji_agras";
  const obsOk = BND_OBS[t] !== false;
  $("bndObs").disabled = !obsOk;
  $("bndObsNote").textContent = obsOk ? "" : t === "dji_mapping" ? "(not used for mapping missions)" : "(Applicator View takes the boundary only)";
  const boxes = [...document.querySelectorAll("#bndList input")];
  const n = boxes.filter((c) => c.checked).length;
  $("bndAll").checked = n === boxes.length && n > 0;
  $("bndAll").indeterminate = n > 0 && n < boxes.length;
  $("bndAllLabel").textContent = "All (" + n + " of " + boxes.length + " selected)";
  const nObs = fields().filter((a) => boxes.some((c) => c.checked && c.dataset.id === a.id)).reduce((s, a) => s + (a.obstacles || 0), 0);
  $("bndObsRow").hidden = !nObs;
  $("bndGo").disabled = !n;
}
async function runOutlines() {
  setErr("bndErr", "");
  $("bndOk").hidden = true;
  const t = bndTarget();
  const body = {
    target: t,
    ids: [...document.querySelectorAll("#bndList input:checked")].map((c) => c.dataset.id),
    format: document.querySelector("input[name=bndfmt]:checked").value,
    per_field: $("bndPerField").checked,
    obstacles: $("bndObs").checked && BND_OBS[t] !== false,
    out_dir: $("bndDir").value.trim(),
  };
  body.spray = { rate_gal_ac: +$("bndRate").value, height_ft: +$("bndHeight").value, width_ft: +$("bndWidth").value };
  if (t === "dji_agras" && (!(body.spray.width_ft > 0) || !(body.spray.height_ft > 0)))
    return setErr("bndErr", "Enter the spray height and spray width.");
  const btn = $("bndGo");
  btn.disabled = true;
  try {
    const r = await api("/api/areas/outlines", body);
    bnd.dir = r.dir;
    $("bndOk").textContent = "Saved " + r.files.length + (r.files.length === 1 ? " file" : " files") + " in " + r.dir + ".";
    $("bndOk").hidden = false;
    $("bndOpen").hidden = false;
  } catch (e) { setErr("bndErr", e.message); }
  finally { btn.disabled = false; }
}

// ------------------------------------------------------------------ export
function openExport(areasFirst) {
  if (!result) return;
  setErr("exportErr", "");
  const has = fields().length > 0;
  $("expWhole").checked = !has || !areasFirst;
  $("expAreasWrap").hidden = !has;
  $("expNoAreas").hidden = has;
  const ul = $("expAreaList");
  ul.innerHTML = "";
  for (const a of fields()) {
    const li = document.createElement("li");
    li.innerHTML = '<label class="check"><input type="checkbox" checked> <span></span><span class="size"></span></label>';
    li.querySelector("input").dataset.id = a.id;
    li.querySelector("input").onchange = () => { syncAll(); requestExportEstimate(); };
    li.querySelector("span").textContent = a.name;
    li.querySelector(".size").textContent = fmt(a.ac, a.ac < 10 ? 2 : 1) + " ac";
    ul.append(li);
  }
  if (!areasFirst) ul.querySelectorAll("input").forEach((c) => (c.checked = false));
  syncAll();
  const comp = (status.project && status.project.settings.compression) || "jpeg";
  document.querySelectorAll("input[name=expcomp]").forEach((r) => (r.checked = r.value === comp));
  const g = result.report.gsd_cm;
  if (g) {
    $("sc100").innerHTML = "100%<small>" + fmt(g, 1) + " cm</small>";
    $("sc50").innerHTML = "50%<small>" + fmt(g * 2, 1) + " cm</small>";
    $("sc25").innerHTML = "25%<small>" + fmt(g * 4, 1) + " cm</small>";
  }
  $("expDlg").showModal();
  requestExportEstimate();
}
function syncAll() {
  const boxes = [...document.querySelectorAll("#expAreaList input")];
  const n = boxes.filter((c) => c.checked).length;
  $("expAreasAll").checked = n === boxes.length && n > 0;
  $("expAreasAll").indeterminate = n > 0 && n < boxes.length;
  $("expAreasLabel").textContent = "Areas (" + n + " of " + boxes.length + " selected, one GeoTIFF each)";
}
function readExport() {
  return {
    whole: $("expWhole").checked,
    ids: [...document.querySelectorAll("#expAreaList input:checked")].map((c) => c.dataset.id),
    compression: (document.querySelector("input[name=expcomp]:checked") || {}).value || "jpeg",
    scale: parseFloat((document.querySelector("input[name=expscale]:checked") || {}).value || "1"),
    overviews: $("expOverviews").checked,
    out_dir: $("exportDir").value.trim(),
  };
}
let expTimer = null;
function requestExportEstimate() {
  clearTimeout(expTimer);
  expTimer = setTimeout(async () => {
    const o = readExport();
    const n = o.ids.length + (o.whole ? 1 : 0);
    $("btnExport").disabled = !n || busyExport();
    $("btnExport").textContent = n > 1 ? "Export " + n + " files" : "Export";
    if (!n) { $("expEst").textContent = "Choose what to export."; return; }
    try {
      const e = await api("/api/export/estimate", o);
      $("expEst").innerHTML = "";
      const b = document.createElement("b");
      b.textContent = "≈ " + fmtBytes(e.bytes);
      $("expEst").append(b, " in " + n + (n === 1 ? " GeoTIFF" : " GeoTIFFs") + " at " + fmt(e.gsd_cm, 1) + " cm (" + fmt(e.gsd_cm / 2.54, 2) + " in) per pixel.");
    } catch (err) { $("expEst").textContent = ""; }
  }, 120);
}
async function startExport() {
  setErr("exportErr", "");
  try {
    await api("/api/export", readExport());
    exportWasRunning = true;
    $("expDlg").close();
    refresh();
  } catch (e) { setErr("exportErr", e.message); }
}
function renderExportProgress() {
  const ex = status.export;
  const running = busyExport();
  if (exportWasRunning && !running && ex) {
    if (ex.error) toast("Export failed: " + ex.error, 8000);
    else {
      const msg = "Exported " + ex.files.length + (ex.files.length === 1 ? " GeoTIFF." : " GeoTIFFs.");
      toast(ex.errors.length ? msg + " Problems: " + ex.errors.join("; ") : msg, ex.errors.length ? 10000 : 5000);
      if (ex.files.length) api("/api/open_path", { path: ex.out_dir });
    }
  }
  exportWasRunning = !!running;
}

// ------------------------------------------------------------------ projects dashboard
async function openDash() {
  const wasHidden = $("dash").hidden;
  $("dash").hidden = false;
  $("dClose").hidden = !(status && status.project);
  $("btnProjects").setAttribute("aria-pressed", true);
  if (wasHidden) { setErr("dErr", ""); await renderDash(true); }
}
let dashLoaded = 0;
async function renderDash(force) {
  if (!force && Date.now() - dashLoaded < 10000) return;
  dashLoaded = Date.now();
  let d;
  try { d = await api("/api/projects"); } catch (e) { return; }
  $("dLoc").textContent = d.dir;
  const ul = $("dCards");
  ul.innerHTML = "";
  if (!d.projects.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No projects yet. Plan a mission to start a job before flying, or start a project from a flight's photos.";
    ul.append(li);
  }
  const cur = status.project && status.project.dir;
  for (const pr of d.projects) {
    const li = document.createElement("li");
    li.className = "card" + (pr.dir === cur ? " current" : "");
    const b = document.createElement("button");
    b.className = "open";
    b.innerHTML = '<span class="thumb"></span><span class="meta"><strong></strong><small></small><small></small></span>';
    const th = b.querySelector(".thumb");
    if (pr.has_thumb) th.style.backgroundImage = "url('/thumb?dir=" + encodeURIComponent(pr.dir) + "&t=" + Math.round(pr.modified) + "')";
    else if (pr.planned) { th.textContent = "Mission planned · not flown yet"; th.classList.add("planned"); }
    else th.textContent = pr.has_result ? "" : "Not processed yet";
    b.querySelector("strong").textContent = pr.name;
    const sm = b.querySelectorAll("small");
    sm[0].textContent = "Changed " + new Date(pr.modified * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    sm[1].textContent = pr.dir;
    b.title = pr.dir;
    b.onclick = () => openProject(pr.dir);
    const x = document.createElement("button");
    x.className = "forget";
    x.textContent = "×";
    x.title = "Remove from this list (files are kept)";
    x.setAttribute("aria-label", "Remove " + pr.name + " from the list");
    x.onclick = async () => { await api("/api/recent/forget", { dir: pr.dir }); renderDash(true); };
    li.append(b, x);
    ul.append(li);
  }
}
async function openProject(d) {
  setErr("dErr", "");
  try {
    await api("/api/project/open", { dir: d });
    $("dash").hidden = true;
    lastScan = null;
    lastProjectDir = undefined; // pick the default panel for this project
    await refresh();
  } catch (e) { setErr("dErr", e.message); }
}
async function closeProject() {
  try { await api("/api/project/close", {}); lastScan = null; await refresh(); return true; }
  catch (e) { toast(e.message); return false; }
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

// ------------------------------------------------------------------ wiring
async function browse(mode, title, initial) {
  try { return (await api("/api/browse", { mode, title, initial: initial || "" })).path; }
  catch (e) { toast("Could not open the file browser: " + e.message); return ""; }
}
function closeMenu() { $("moreMenu").hidden = true; $("btnMore").setAttribute("aria-expanded", false); }

function wire() {
  document.querySelectorAll(".railbtn[data-panel]").forEach((b) => (b.onclick = () => setPanel(panel === b.dataset.panel ? null : b.dataset.panel)));
  document.querySelectorAll("[data-closepane]").forEach((b) => (b.onclick = () => setPanel(null)));
  $("railExport").onclick = () => openExport(false);

  // top bar
  $("btnProjects").onclick = () => {
    if ($("dash").hidden) openDash();
    else if (status.project) { $("dash").hidden = true; $("btnProjects").setAttribute("aria-pressed", false); }
  };
  $("btnRename").onclick = () => { $("projName").focus(); };
  const nm = $("projName");
  nm.onkeydown = (e) => { if (e.key === "Enter") nm.blur(); if (e.key === "Escape") { nm.value = status.project ? status.project.name : ""; nm.blur(); } };
  nm.onfocus = () => nm.select();
  nm.onblur = async () => {
    const p = status.project;
    const v = nm.value.trim();
    if (!p || !v || v === p.name) { nm.value = p ? p.name : ""; return; }
    try {
      const r = await api("/api/rename", { name: v });
      lastProjectDir = r.dir;
      if (r.moved) $("exportDir").value = r.dir + sep() + "exports";
      toast(r.moved ? "Renamed. The project folder is now " + r.dir : "Renamed.");
      refresh();
    } catch (e) { toast(e.message); nm.value = p.name; }
  };
  $("btnMore").onclick = (e) => { e.stopPropagation(); const h = $("moreMenu").hidden; $("moreMenu").hidden = !h; $("btnMore").setAttribute("aria-expanded", h); };
  document.addEventListener("click", (e) => { if (!e.target.closest(".menuwrap")) closeMenu(); });
  $("mOpenFolder").onclick = () => { closeMenu(); api("/api/open_path", {}); };
  $("mLog").onclick = () => { closeMenu(); showLog(); };
  const diag = async () => {
    closeMenu();
    try { const r = await api("/api/diagnostics", {}); toast("Diagnostics saved to " + r.path, 10000); }
    catch (e) { toast(e.message); }
  };
  $("mDiag").onclick = diag;
  $("btnDiag2").onclick = diag;
  $("mClose").onclick = () => { closeMenu(); closeProject(); };

  // photos
  $("addFolder").onclick = () => addPhotos("folder");
  $("addImages").onclick = () => addPhotos("images");
  $("newFromPanel").onclick = openNew;
  $("btnSelectArea").onclick = () => setTool("select");
  $("btnSelectAll").onclick = async () => { await api("/api/photos/select", { clear: true }); await loadPhotos(false); refresh(); };
  $("btnToProcess").onclick = openOptions;

  // process
  $("btnOptions").onclick = openOptions;
  document.querySelectorAll("input[name=preset]").forEach((r) => (r.onchange = () => { setSliderFactor(PRESET_RES[r.value] || 1); updateResLabel(); requestEstimate(); }));
  $("resSlider").oninput = () => { updateResLabel(); requestEstimate(); };
  $("optGpu").onchange = requestEstimate;
  document.querySelectorAll("input[name=compression]").forEach((r) => (r.onchange = requestEstimate));
  $("procForm").onsubmit = (e) => {
    if (e.submitter && e.submitter.value === "start") { e.preventDefault(); startProcessing(); }
  };
  $("btnCancel").onclick = async () => { if (confirm("Stop processing? Finished steps are kept and reused next time.")) { await api("/api/process/cancel", {}); refresh(); } };
  $("btnLog").onclick = () => ($("logDrawer").hidden ? showLog() : ($("logDrawer").hidden = true));
  $("btnLog2").onclick = showLog;
  $("logClose").onclick = () => ($("logDrawer").hidden = true);
  $("btnOutputs").onclick = () => api("/api/open_path", { path: result && result.outputs_dir });

  // layers
  $("btnFit").onclick = fitJob;
  $("lyrCams").onchange = () => { if (layers.cams) $("lyrCams").checked ? layers.cams.addTo(map) : layers.cams.remove(); };
  $("lyrOrtho").onchange = () => { if (layers.ortho) { $("lyrOrtho").checked ? layers.ortho.addTo(map) : layers.ortho.remove(); } };
  $("lyrOrthoOp").oninput = () => { if (map) map.getPane("ortho").style.opacity = +$("lyrOrthoOp").value; };
  $("lyrAreas").onchange = () => { $("lyrAreas").checked ? layers.areas.addTo(map) : layers.areas.remove(); };

  // areas
  const closeForms = (keep) => {
    for (const [f, b] of [["gridForm", "toolGrid"], ["importForm", "toolImport"]]) {
      if (f === keep) continue;
      $(f).hidden = true;
      $(b).setAttribute("aria-pressed", false);
    }
  };
  $("toolRect").onclick = () => { closeForms(); setTool("rect"); };
  $("toolPoly").onclick = () => { closeForms(); setTool("poly"); };
  $("toolObst").onclick = () => {
    closeForms();
    if (!fields().length && tool !== "obst") toast("Draw or import the field first, then mark the obstacles inside it.", 6000);
    setTool("obst");
  };
  const toggleForm = (f, b) => {
    if (tool) setTool(null);
    closeForms(f);
    const show = $(f).hidden;
    $(f).hidden = !show;
    $(b).setAttribute("aria-pressed", show);
  };
  $("toolGrid").onclick = () => toggleForm("gridForm", "toolGrid");
  $("toolImport").onclick = () => toggleForm("importForm", "toolImport");
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
      const r = await api("/api/areas/grid", { mode, size_m: size, rows: $("gridRows").value, cols: $("gridCols").value });
      areas = r.areas;
      drawAreas();
      toast("Added " + r.added + " grid areas. Empty cells were skipped.");
      $("gridForm").hidden = true;
      $("toolGrid").setAttribute("aria-pressed", false);
    } catch (err) { toast(err.message); }
  };
  $("importPick").onclick = async () => {
    const role = document.querySelector("input[name=importrole]:checked").value;
    const p = await browse("file", role === "obstacle" ? "Choose obstacles (KML, KMZ, SHP, GeoJSON)" :
                                   "Choose field boundaries (KML, KMZ, SHP, GeoJSON)", "");
    if (!p) return;
    try {
      const r = await api("/api/areas/import", { path: p, role });
      areas = r.areas;
      drawAreas();
      const parts = [];
      if (r.added) parts.push(r.added + (r.added === 1 ? " boundary" : " boundaries"));
      if (r.obstacles) parts.push(r.obstacles + (r.obstacles === 1 ? " obstacle" : " obstacles"));
      toast("Imported " + parts.join(" and ") + ".");
      $("importForm").hidden = true;
      $("toolImport").setAttribute("aria-pressed", false);
    } catch (e) { toast(e.message, 6000); }
  };
  $("areaClear").onclick = () => { if (confirm("Remove all areas?")) { if (edit) endEdit(); areas = []; saveAreas(); } };
  $("btnExportAreas").onclick = () => openExport(true);
  $("btnOutlines").onclick = openOutlines;
  // plan a mission
  $("dPlan").onclick = openPlan;
  $("plFile").onclick = plPickFile;
  $("plLib").onclick = plOpenLib;
  $("plFind").onclick = () => { plMode("find"); plMapInit(); $("plQ").focus(); };
  $("plGo").onclick = plFindGo;
  $("plQ").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); plFindGo(); } };
  $("plLibQ").oninput = plRenderLib;
  $("plLibQ").onkeydown = (e) => { if (e.key === "Enter") e.preventDefault(); };
  $("plLast").onclick = () => { plMapInit(); plSetPoint(pl.last.lat, pl.last.lon, "Last job (" + (pl.last.name || "") + ")", 14); };
  $("plName").oninput = () => { pl.nameEdited = !!$("plName").value.trim(); plSync(); };
  $("plName").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); if (!$("plCreate").disabled) plCreate(); } };
  $("plBrowse").onclick = async () => { const p = await browse("folder", "Save the job in", $("plParent").value); if (p) $("plParent").value = p; };
  $("planForm").onsubmit = (e) => { if (e.submitter && e.submitter.value === "create") { e.preventDefault(); plCreate(); } };
  $("pbPhotos").onclick = () => setPanel("import");
  try { $("lyrBase").checked = localStorage.getItem("planesite.base") !== "off"; } catch (e) { /* default on */ }
  $("lyrBase").onchange = () => {
    try { localStorage.setItem("planesite.base", $("lyrBase").checked ? "on" : "off"); } catch (e) { /* ignore */ }
    if (!layers.base) return;
    if ($("lyrBase").checked) { layers.base.addTo(map); $("mapCredit").hidden = false; }
    else { layers.base.remove(); $("mapCredit").hidden = true; }
  };
  document.querySelectorAll("input[name=bndtarget]").forEach((r) => (r.onchange = syncOutlines));
  $("bndAll").onchange = () => { document.querySelectorAll("#bndList input").forEach((c) => (c.checked = $("bndAll").checked)); syncOutlines(); };
  $("bndBrowse").onclick = async () => { const p = await browse("folder", "Save boundary files in", $("bndDir").value); if (p) $("bndDir").value = p; };
  $("bndOpen").onclick = () => api("/api/open_path", { path: bnd.dir });
  $("bndForm").onsubmit = (e) => { if (e.submitter && e.submitter.value === "export") { e.preventDefault(); runOutlines(); } };

  // export dialog
  $("expWhole").onchange = requestExportEstimate;
  $("expAreasAll").onchange = () => { document.querySelectorAll("#expAreaList input").forEach((c) => (c.checked = $("expAreasAll").checked)); syncAll(); requestExportEstimate(); };
  document.querySelectorAll("input[name=expcomp], input[name=expscale]").forEach((r) => (r.onchange = requestExportEstimate));
  $("expOverviews").onchange = requestExportEstimate;
  $("exportBrowse").onclick = async () => { const p = await browse("folder", "Save exported GeoTIFFs in", $("exportDir").value); if (p) $("exportDir").value = p; };
  // send to Pilot 2
  for (const id of Object.values(p2Fields)) $(id).oninput = p2Changed;
  $("p2Auto").onchange = p2Changed;
  $("p2Save").onclick = () => p2Go(false);
  $("p2Form").onsubmit = (e) => { if (e.submitter && e.submitter.value === "send") { e.preventDefault(); p2Go(true); } };
  $("p2Form").onkeydown = (e) => { if (e.key === "Enter" && e.target.tagName === "INPUT") e.preventDefault(); };
  // editing an area outline
  $("editDone").onclick = () => finishEdit(true);
  $("editCancel").onclick = () => cancelEdit();
  $("editUndo").onclick = () => undoEdit();

  $("expForm").onsubmit = (e) => {
    if (e.submitter && e.submitter.value === "export") { e.preventDefault(); startExport(); }
  };

  // dashboard
  $("dClose").onclick = () => { $("dash").hidden = true; $("btnProjects").setAttribute("aria-pressed", false); };
  $("dNew").onclick = openNew;
  $("dTif").onclick = openGeotiff;
  $("npFolder").onclick = () => npPick("folder");
  $("npImages").onclick = () => npPick("images");
  $("npName").oninput = () => { np.nameEdited = true; $("npCreate").disabled = !$("npName").value.trim() || (!np.folder && !np.files); clearTimeout(npTimer); npTimer = setTimeout(npSuggest, 300); };
  $("npParent").onchange = npSuggest;
  $("npBrowse").onclick = async () => { const p = await browse("folder", "Save the new project in", $("npParent").value); if (p) { $("npParent").value = p; npSuggest(); } };
  $("newForm").onsubmit = (e) => { if (e.submitter && e.submitter.value === "create") { e.preventDefault(); npCreate(); } };
  $("dOpen").onclick = async () => { const p = await browse("folder", "Choose a PlaneSite project folder", $("dLoc").textContent); if (p) openProject(p); };
  $("dLocChange").onclick = async () => {
    const p = await browse("folder", "Save new projects in", $("dLoc").textContent);
    if (!p) return;
    try { await api("/api/projects_dir", { path: p }); renderDash(true); } catch (e) { setErr("dErr", e.message); }
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
    setErr("updErr", "This update changes the bundled libraries, so it needs the full installer. Your projects are kept.");
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
    if (u.phase === "restarting") $("updMsg").textContent = "Restarting PlaneSite…";
  } else if (!st) {
    $("updMsg").textContent = "Restarting PlaneSite…";
  }
  setTimeout(pollUpdate, 1000);
}

initMap();
wire();
$("btnUpdate").onclick = () => checkUpdates(false);
$("updClose").onclick = () => { if (!updRestarting) $("updDlg").close(); };
$("updInstall").onclick = installUpdate;
// PlaneSite opens on the Projects page. A project left open in the background (the window was
// closed and opened again) is closed first, unless it's processing, reading photos or exporting.
(async () => {
  try {
    const st = await api("/api/status", undefined, 15000);
    const p = st.project;
    const busy = st.running || (st.scan && !st.scan.finished) || (st.export && !st.export.finished);
    if (p && !busy) await api("/api/project/close", {});
  } catch (e) { /* refresh() reports a lost connection */ }
  refresh();
})();
setTimeout(() => checkUpdates(true), 2500);
