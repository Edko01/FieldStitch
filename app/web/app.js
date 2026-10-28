"use strict";
/* FieldStitch front end. Map frame: lat = northing offset (m), lng = easting offset (m) from the project origin. */

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
let lastResultMtime = null;
let lastProgressStatus = null;
let scanWasRunning = false;
let exportWasRunning = false;
let lastScan = null;
let panel = "import";
let nativeGsdCm = null;
let optionsAfterScan = false;   // open Processing options when the current photo import finishes

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
async function refresh() {
  try {
    status = await api("/api/status");
  } catch (e) {
    if (!updRestarting) toast("Lost contact with FieldStitch. Start it again from the Start menu or desktop icon.");
    schedule(4000);
    return;
  }
  render();
  const busy = status.running || busyScan() || busyExport();
  schedule(busy ? 1000 : 3000);
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
    if (p) {
      $("dash").hidden = true;
      loadPhotos(true);
      loadResult(true);
      $("exportDir").value = p.dir + sep() + "exports";
      setPanel(p.has_result ? "layers" : (status.running ? "process" : "import"));
    } else {
      setPanel(null);
      map.setView([0, 0], 2);
      openDash();
    }
  }
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

  $("importDetails").hidden = !p || (!p.count && !p.geotiff) || scanning;
  if (!p) return;
  $("addRow").hidden = !!p.geotiff;
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
  const rows = [["Photos", fmt(p.count)]];
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
      toast("Orthomosaic ready.", 5000);
      loadResult(true);
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
    $("areasLocked").hidden = false;
    $("areasTools").hidden = true;
    if (layers.ortho) { layers.ortho.remove(); layers.ortho = null; }
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
    opacity: +$("lyrOrthoOp").value,
  });
  if ($("lyrOrtho").checked) layers.ortho.addTo(map);
  layers.ortho.bringToBack();
  map.setMaxZoom(result.max_native_zoom + 3);
}

// ------------------------------------------------------------------ map
function initMap() {
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
  map.getContainer().classList.toggle("cursor-cross", !!tool);
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
  if (e.key === "Escape") { if (draw) cancelDraft(); else if (tool) setTool(null); else if (!$("dash").hidden && status.project) $("dash").hidden = true; }
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
  drawAreas();
}
const plain = () => areas.map((a) => ({ id: a.id, name: a.name, kind: a.kind, rings: a.rings }));
async function saveAreas() {
  try { areas = (await api("/api/areas", { areas: plain() })).areas; drawAreas(); }
  catch (e) { toast(e.message); }
}
async function addArea(a) {
  a.id = "";
  areas.push(a);
  try { areas = (await api("/api/areas", { areas: plain() })).areas; } catch (e) { toast(e.message); }
  drawAreas();
  if (areas.length) focusArea(areas[areas.length - 1].id, true);
}
function drawAreas() {
  if (!layers.areas) return;
  layers.areas.clearLayers();
  for (const a of areas) {
    const ll = a.rings.map((r) => r.map((p) => [p[1], p[0]]));
    const poly = L.polygon(ll, { color: "#E9B20F", weight: 2.5, fillColor: "#E9B20F", fillOpacity: 0.08 });
    poly.bindTooltip(a.name, { permanent: true, direction: "center", className: "area-label" });
    poly.on("click", (e) => { if (!tool) { L.DomEvent.stop(e); setPanel("areas"); focusArea(a.id); } });
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
    del.onclick = () => { areas = areas.filter((x) => x.id !== a.id); saveAreas(); };
    li.append(nm, sz, del);
    ul.append(li);
  }
  $("areaHead").hidden = !areas.length;
  $("areaActions").hidden = !areas.length;
  const tot = areas.reduce((s, a) => s + a.ac, 0);
  $("areaCount").textContent = areas.length + (areas.length === 1 ? " area · " : " areas · ") + fmt(tot, tot < 10 ? 1 : 0) + " ac";
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

// ------------------------------------------------------------------ export
function openExport(areasFirst) {
  if (!result) return;
  setErr("exportErr", "");
  const has = areas.length > 0;
  $("expWhole").checked = !has || !areasFirst;
  $("expAreasWrap").hidden = !has;
  $("expNoAreas").hidden = has;
  const ul = $("expAreaList");
  ul.innerHTML = "";
  for (const a of areas) {
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
    li.textContent = "No projects yet. Click New project and choose the folder with a flight's photos.";
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
  $("lyrOrtho").onchange = () => { if (layers.ortho) { $("lyrOrtho").checked ? layers.ortho.addTo(map) : layers.ortho.remove(); if ($("lyrOrtho").checked) layers.ortho.bringToBack(); } };
  $("lyrOrthoOp").oninput = () => { if (layers.ortho) layers.ortho.setOpacity(+$("lyrOrthoOp").value); };
  $("lyrAreas").onchange = () => { $("lyrAreas").checked ? layers.areas.addTo(map) : layers.areas.remove(); };

  // areas
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
      const r = await api("/api/areas/grid", { mode, size_m: size, rows: $("gridRows").value, cols: $("gridCols").value });
      areas = r.areas;
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
      const r = await api("/api/areas/import", { path: p });
      areas = r.areas;
      drawAreas();
      toast("Imported " + r.added + (r.added === 1 ? " boundary." : " boundaries."));
    } catch (e) { toast(e.message, 6000); }
  };
  $("areaClear").onclick = () => { if (confirm("Remove all areas?")) { areas = []; saveAreas(); } };
  $("btnExportAreas").onclick = () => openExport(true);
  $("btnKml").onclick = async () => {
    try { const r = await api("/api/areas/kml", { out_dir: $("exportDir").value.trim() }); toast("Saved " + r.path, 6000); }
    catch (e) { toast(e.message); }
  };

  // export dialog
  $("expWhole").onchange = requestExportEstimate;
  $("expAreasAll").onchange = () => { document.querySelectorAll("#expAreaList input").forEach((c) => (c.checked = $("expAreasAll").checked)); syncAll(); requestExportEstimate(); };
  document.querySelectorAll("input[name=expcomp], input[name=expscale]").forEach((r) => (r.onchange = requestExportEstimate));
  $("expOverviews").onchange = requestExportEstimate;
  $("exportBrowse").onclick = async () => { const p = await browse("folder", "Save exported GeoTIFFs in", $("exportDir").value); if (p) $("exportDir").value = p; };
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
  $("dOpen").onclick = async () => { const p = await browse("folder", "Choose a FieldStitch project folder", $("dLoc").textContent); if (p) openProject(p); };
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
    if (u.phase === "restarting") $("updMsg").textContent = "Restarting FieldStitch…";
  } else if (!st) {
    $("updMsg").textContent = "Restarting FieldStitch…";
  }
  setTimeout(pollUpdate, 1000);
}

initMap();
wire();
$("btnUpdate").onclick = () => checkUpdates(false);
$("updClose").onclick = () => { if (!updRestarting) $("updDlg").close(); };
$("updInstall").onclick = installUpdate;
refresh();
setTimeout(() => checkUpdates(true), 2500);
