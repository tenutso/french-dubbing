// Vanilla JS, no framework. Single page app for the dubbing web UI.

const TERMINAL = new Set(["completed", "failed", "cancelled"]);
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Per-job EventSource handles so we can close them when a job terminates
const streams = new Map();        // job_id -> EventSource
const logBuffers = new Map();     // job_id -> string[]
let lastJobsHash = "";

// Translation review state
let reviewJobId = null;
let reviewSegments = [];

// ── Options ──────────────────────────────────────────────────────────────────
async function loadOptions() {
  const r = await fetch("/api/options");
  if (!r.ok) return;
  const data = await r.json();
  fillSelect("locale", data.locales, data.defaults.locale);
  $("#volume_boost").value = data.defaults.volume_boost ?? 0;
}

function fillSelect(id, choices, def) {
  const sel = $("#" + id);
  sel.innerHTML = "";
  // Empty value = use config default
  const def0 = document.createElement("option");
  def0.value = "";
  def0.textContent = `(config default: ${def})`;
  sel.appendChild(def0);
  for (const c of choices) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    if (c === def) opt.dataset.isDefault = "1";
    sel.appendChild(opt);
  }
}

// ── Health badge ─────────────────────────────────────────────────────────────
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    if (!r.ok) return;
    const h = await r.json();
    const parts = [];
    if (h.gpu) parts.push(`${h.gpu} (${h.vram_free_gb}/${h.vram_total_gb} GB free)`);
    if (h.disk_free_gb != null) parts.push(`disk ${h.disk_free_gb} GB free`);
    parts.push(`ollama ${h.ollama_up ? "✓" : "✗"}`);
    parts.push(`HF ${h.hf_token_present ? "✓" : "—"}`);
    $("#health").textContent = parts.join(" · ");
  } catch (e) {
    $("#health").textContent = "health check failed";
  }
}

// ── Submit form ──────────────────────────────────────────────────────────────
const fileInput = $("#video");
const dropEl = $("#drop");
const dropText = $("#drop-text");

["dragenter", "dragover"].forEach(ev =>
  dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(ev =>
  dropEl.addEventListener(ev, e => { e.preventDefault(); dropEl.classList.remove("dragover"); })
);
dropEl.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    updateDropText();
  }
});
fileInput.addEventListener("change", updateDropText);
function updateDropText() {
  if (fileInput.files.length) {
    const f = fileInput.files[0];
    dropText.textContent = `${f.name} (${(f.size / 1e6).toFixed(1)} MB)`;
  } else {
    dropText.textContent = "Drop a video file here or click to choose";
  }
}

// ── Source toggle: upload file vs. Vimeo URL vs. on-pod path ──────────────────
const vimeoField = $("#vimeo-field");
const vimeoInput = $("#vimeo_url");
const pathField = $("#path-field");
const pathInput = $("#local_path");
function sourceMode() {
  const checked = document.querySelector('input[name="source"]:checked');
  return checked ? checked.value : "file";
}
function updateSourceMode() {
  const mode = sourceMode();
  dropEl.hidden = mode !== "file";
  vimeoField.hidden = mode !== "vimeo";
  pathField.hidden = mode !== "path";
  // Only the active input is required so the form doesn't block the other modes.
  fileInput.required = mode === "file";
  vimeoInput.required = mode === "vimeo";
  pathInput.required = mode === "path";
}
document.querySelectorAll('input[name="source"]').forEach(r =>
  r.addEventListener("change", updateSourceMode)
);
updateSourceMode();

$("#submit-form").addEventListener("submit", e => {
  e.preventDefault();
  const mode = sourceMode();
  const isUpload = mode === "file";
  const status = $("#upload-status");
  if (mode === "vimeo") {
    if (!vimeoInput.value.trim()) { status.textContent = "enter a Vimeo URL"; return; }
  } else if (mode === "path") {
    if (!pathInput.value.trim()) { status.textContent = "enter a path to a file on the pod"; return; }
  } else if (!fileInput.files.length) {
    return;
  }

  // Send only the fields for the active mode (skip the empty other inputs).
  const fd = new FormData();
  if (mode === "vimeo") fd.append("vimeo_url", vimeoInput.value.trim());
  else if (mode === "path") fd.append("local_path", pathInput.value.trim());
  else fd.append("video", fileInput.files[0]);
  fd.append("locale", $("#locale").value);
  fd.append("speakers", $("#speakers").value);
  fd.append("volume_boost", $("#volume_boost").value);
  if ($("#force").checked) fd.append("force", "on");
  if ($("#review").checked) fd.append("review", "on");

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  const prog = $("#upload-progress");
  prog.hidden = !isUpload;  // progress only meaningful for a browser upload
  prog.value = 0;
  status.textContent = isUpload ? "uploading…" : "submitting…";
  $("#submit-btn").disabled = true;

  xhr.upload.onprogress = e => {
    if (e.lengthComputable) {
      prog.value = Math.round((e.loaded / e.total) * 100);
      status.textContent = `uploading ${prog.value}%`;
    }
  };
  xhr.onload = () => {
    $("#submit-btn").disabled = false;
    prog.hidden = true;
    if (xhr.status === 201) {
      const data = JSON.parse(xhr.responseText);
      status.textContent = `queued as ${data.id} (position ${data.position || 1})`;
      $("#submit-form").reset();
      updateDropText();
      updateSourceMode();
      refreshJobs();
    } else {
      let err = xhr.responseText;
      try { err = JSON.parse(xhr.responseText).detail || err; } catch (_) {}
      status.textContent = `failed: ${err}`;
    }
  };
  xhr.onerror = () => {
    $("#submit-btn").disabled = false;
    prog.hidden = true;
    status.textContent = "network error";
  };
  xhr.send(fd);
});

// ── Jobs list ────────────────────────────────────────────────────────────────
async function refreshJobs() {
  try {
    const r = await fetch("/api/jobs");
    if (!r.ok) return;
    const data = await r.json();
    const hash = JSON.stringify(data.jobs.map(j => [j.id, j.status, j.phase, Object.keys(j.outputs || {}).length]));
    if (hash === lastJobsHash) return;  // skip identical re-render
    lastJobsHash = hash;
    renderJobs(data.jobs);
  } catch (e) {
    /* ignore — periodic poll */
  }
}

function renderJobs(jobs) {
  const active = jobs.filter(j => !TERMINAL.has(j.status));
  const history = jobs.filter(j => TERMINAL.has(j.status));
  renderList("#active-list", active, "No active jobs.");
  renderList("#history-list", history, "No completed jobs yet.");

  // Open SSE for every active job we don't already have a stream for.
  for (const j of active) {
    if (!streams.has(j.id)) {
      openStream(j.id);
    }
  }
  // Close streams for jobs that are no longer active
  for (const id of [...streams.keys()]) {
    if (!active.find(j => j.id === id)) {
      streams.get(id).close();
      streams.delete(id);
    }
  }

  // Show review editor for awaiting_review jobs
  const awaitingJob = active.find(j => j.status === "awaiting_review");
  if (awaitingJob) {
    if (reviewJobId !== awaitingJob.id) {
      loadReviewSegments(awaitingJob.id, awaitingJob);
    }
  } else if (reviewJobId) {
    $("#review-section").hidden = true;
    $("#voice-ref-section").hidden = true;
    reviewJobId = null;
    reviewSegments = [];
  }
}

function renderList(selector, list, emptyMsg) {
  const root = $(selector);
  root.innerHTML = "";
  if (!list.length) {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = emptyMsg;
    root.appendChild(e);
    return;
  }
  for (const job of list) {
    root.appendChild(renderCard(job));
  }
}

function renderCard(job) {
  const tpl = $("#job-card-tpl").content.cloneNode(true);
  const root = tpl.querySelector(".job");
  root.dataset.id = job.id;
  $(".filename", root).textContent = job.video_filename;
  const pill = $(".status-pill", root);
  pill.textContent = job.status;
  pill.dataset.status = job.status;

  const meta = [];
  const o = job.options || {};
  if (o.locale) meta.push(o.locale);
  if (o.speakers) meta.push(`${o.speakers} spk`);
  if (o.volume_boost != null) meta.push(`+${o.volume_boost}%`);
  if (o.force) meta.push("force");
  const dur = job.ended_at ? (job.ended_at - job.started_at)
            : job.started_at ? (Date.now() / 1000 - job.started_at) : 0;
  if (dur > 0) meta.push(`${Math.round(dur)}s`);
  $(".job-meta", root).textContent = meta.join(" · ");

  $(".job-phase", root).textContent = job.phase || "";

  // Progress Bar
  const prog = $(".job-progress", root);
  const progInner = $(".job-progress-inner", root);
  if (job.status === "running" && job.phase) {
    prog.hidden = false;
    const m = job.phase.match(/\[(\d+)\/6\]/);
    if (m) {
      const step = parseInt(m[1]);
      const pct = Math.round((step / 6) * 100);
      progInner.style.width = `${pct}%`;
    }
  } else {
    prog.hidden = true;
  }

  // Downloads
  const dl = $(".job-downloads", root);
  for (const [kind, label] of [
    ["video", "Video (MP4)"], ["audio", "Audio"],
    ["srt", "French SRT"], ["full", "Full mix"], ["english_srt", "English SRT"],
  ]) {
    if (job.outputs && job.outputs[kind]) {
      const a = document.createElement("a");
      a.href = `/api/jobs/${job.id}/download/${kind}`;
      a.textContent = label;
      a.className = "dl";
      a.download = "";
      dl.appendChild(a);
    }
  }
  if (job.error) {
    const e = document.createElement("div");
    e.className = "err";
    e.textContent = job.error;
    dl.appendChild(e);
  }

  // Log panel: reuse any buffered text we already streamed
  const logEl = $(".job-log", root);
  const buf = logBuffers.get(job.id);
  if (buf) {
    logEl.textContent = buf.join("\n");
  }

  const showBtn = $(".show-log", root);
  showBtn.addEventListener("click", () => {
    const visible = !logEl.hidden;
    logEl.hidden = visible;
    showBtn.textContent = visible ? "Show log" : "Hide log";
    if (!visible) logEl.scrollTop = logEl.scrollHeight;
  });

  const cancelBtn = $(".cancel", root);
  if (TERMINAL.has(job.status)) {
    cancelBtn.textContent = "Remove";
  }
  cancelBtn.addEventListener("click", async () => {
    if (!confirm(`${TERMINAL.has(job.status) ? "Remove" : "Cancel"} ${job.video_filename}?`)) return;
    const url = `/api/jobs/${job.id}` + (TERMINAL.has(job.status) ? "?cleanup=true" : "");
    await fetch(url, { method: "DELETE" });
    refreshJobs();
  });

  // "Open review" button for awaiting_review jobs
  if (job.status === "awaiting_review") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Open review";
    btn.addEventListener("click", () =>
      document.getElementById("review-section").scrollIntoView({ behavior: "smooth" })
    );
    $(".job-actions", root).insertBefore(btn, $(".show-log", root));
  }

  // "Re-open review" for finished jobs — step back to the review stage and
  // re-run Phase 2 (TTS) without redoing Phase 1 (transcription/translation).
  if (job.status === "completed" || job.status === "failed") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Re-open review";
    btn.title = "Edit segments / voices and regenerate the dub without re-processing the source video";
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const res = await fetch(`/api/jobs/${job.id}/reopen`, { method: "POST" });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          alert(err.detail || `Could not re-open review (HTTP ${res.status})`);
          btn.disabled = false;
          return;
        }
        await refreshJobs();
        document.getElementById("review-section").scrollIntoView({ behavior: "smooth" });
      } catch (e) {
        alert("Network error re-opening review");
        btn.disabled = false;
      }
    });
    $(".job-actions", root).insertBefore(btn, $(".show-log", root));
  }

  return root;
}

// ── SSE log stream ───────────────────────────────────────────────────────────
function openStream(jobId) {
  const es = new EventSource(`/api/jobs/${jobId}/logs`);
  streams.set(jobId, es);
  const buf = logBuffers.get(jobId) || [];
  logBuffers.set(jobId, buf);

  es.onmessage = ev => {
    buf.push(ev.data);
    if (buf.length > 500) buf.splice(0, buf.length - 500);
    appendLog(jobId, ev.data);
  };
  es.addEventListener("done", _ => {
    es.close();
    streams.delete(jobId);
    refreshJobs();
  });
  es.onerror = _ => {
    // EventSource auto-reconnects; we just refresh state if it happens
  };
}

function appendLog(jobId, line) {
  const card = document.querySelector(`.job[data-id="${jobId}"]`);
  if (!card) return;
  const phaseEl = card.querySelector(".job-phase");
  const phaseMatch = line.match(/\[(\d+)\/6\]\s+(.+)/);
  if (phaseMatch) phaseEl.textContent = `[${phaseMatch[1]}/6] ${phaseMatch[2]}`;
  const logEl = card.querySelector(".job-log");
  const wasScrolled = logEl.scrollTop + logEl.clientHeight + 20 >= logEl.scrollHeight;
  logEl.textContent += (logEl.textContent ? "\n" : "") + line;
  if (wasScrolled) logEl.scrollTop = logEl.scrollHeight;
}

// ── Advanced options (config.yaml editor) ────────────────────────────────────
let configSchema = [];
let configValues = {};   // server-side current values (baseline for "dirty" + revert)
let configDirty = {};    // path -> value (pending changes)

async function loadConfig() {
  try {
    const r = await fetch("/api/config");
    if (!r.ok) return;
    const data = await r.json();
    configSchema = data.schema || [];
    configValues = data.values || {};
    configDirty = {};
    $("#config-path").textContent = data.path || "config.yaml";
    renderConfigFields();
    updateAdvancedSummary();
  } catch (e) {
    $("#config-status").textContent = "failed to load config: " + e;
  }
}

function renderConfigFields() {
  const root = $("#advanced-fields");
  root.innerHTML = "";
  // Group by `group`
  const groups = new Map();
  for (const f of configSchema) {
    if (!groups.has(f.group)) groups.set(f.group, []);
    groups.get(f.group).push(f);
  }
  for (const [name, fields] of groups) {
    const g = document.createElement("div");
    g.className = "cfg-group";
    const h = document.createElement("h3");
    h.textContent = name;
    g.appendChild(h);
    const grid = document.createElement("div");
    grid.className = "cfg-grid";
    g.appendChild(grid);
    for (const f of fields) {
      grid.appendChild(renderField(f));
    }
    root.appendChild(g);
  }
}

function renderField(f) {
  const wrap = document.createElement("label");
  wrap.className = "cfg-field";
  wrap.dataset.path = f.path;
  const head = document.createElement("div");
  head.className = "cfg-label";
  head.textContent = f.label;
  wrap.appendChild(head);

  let input;
  // Show dirty value if there's a pending change, else the server baseline
  const current = (f.path in configDirty) ? configDirty[f.path] : configValues[f.path];
  const isDirty = f.path in configDirty;
  if (isDirty) wrap.classList.add("dirty");

  if (f.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!current;
    wrap.classList.add("cfg-bool");
  } else if (f.choices) {
    input = document.createElement("select");
    for (const c of f.choices) {
      const o = document.createElement("option");
      o.value = String(c);
      o.textContent = String(c);
      if (String(c) === String(current)) o.selected = true;
      input.appendChild(o);
    }
  } else if (f.type === "int" || f.type === "float") {
    input = document.createElement("input");
    input.type = "number";
    if (f.min != null) input.min = f.min;
    if (f.max != null) input.max = f.max;
    if (f.step != null) input.step = f.step;
    input.value = current ?? "";
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = current ?? "";
  }

  input.dataset.path = f.path;
  input.dataset.type = f.type;
  input.addEventListener("input", onFieldChange);
  input.addEventListener("change", onFieldChange);
  wrap.appendChild(input);

  if (f.help) {
    const h = document.createElement("div");
    h.className = "cfg-help muted";
    h.textContent = f.help;
    wrap.appendChild(h);
  }
  return wrap;
}

function onFieldChange(ev) {
  const el = ev.target;
  const path = el.dataset.path;
  const type = el.dataset.type;
  let val;
  if (type === "bool") val = el.checked;
  else if (type === "int") val = el.value === "" ? null : parseInt(el.value, 10);
  else if (type === "float") val = el.value === "" ? null : parseFloat(el.value);
  else val = el.value;

  // Compare against baseline; only track as dirty if it differs
  const baseline = configValues[path];
  const same = type === "bool"
    ? Boolean(baseline) === Boolean(val)
    : String(baseline ?? "") === String(val ?? "");
  if (same) {
    delete configDirty[path];
    el.closest(".cfg-field").classList.remove("dirty");
  } else {
    configDirty[path] = val;
    el.closest(".cfg-field").classList.add("dirty");
  }
  updateAdvancedSummary();
}

function updateAdvancedSummary() {
  const n = Object.keys(configDirty).length;
  $("#advanced-summary").textContent = n ? `· ${n} unsaved change${n === 1 ? "" : "s"}` : "";
  $("#config-save").disabled = n === 0;
  $("#config-revert").disabled = n === 0;
}

async function saveConfig(force = false) {
  const status = $("#config-status");
  if (!Object.keys(configDirty).length) return;
  status.textContent = "saving…";
  $("#config-save").disabled = true;
  try {
    const url = "/api/config" + (force ? "?force=true" : "");
    const r = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: configDirty }),
    });
    const data = await r.json().catch(() => ({}));
    if (r.status === 409 && !force) {
      $("#config-save").disabled = false;
      const msg = (data.detail || "a job is currently running.") +
        "\n\nApply changes anyway? (Running job is unaffected; only the next job will pick them up.)";
      if (confirm(msg)) return saveConfig(true);
      status.textContent = "save cancelled";
      return;
    }
    if (!r.ok) {
      status.textContent = "failed: " + (data.detail || r.statusText);
      $("#config-save").disabled = false;
      return;
    }
    const mirrored = (data.mirrored || []).length;
    status.textContent = `saved ${Object.keys(data.values || {}).length} field(s)` +
      (mirrored ? ` (mirrored to ${mirrored} repo cop${mirrored === 1 ? "y" : "ies"})` : "");
    await loadConfig();
    // Re-sync the simple form defaults too (locale, volume_boost)
    await loadOptions();
  } catch (e) {
    status.textContent = "network error: " + e;
    $("#config-save").disabled = false;
  }
}

function revertConfig() {
  configDirty = {};
  renderConfigFields();
  updateAdvancedSummary();
  $("#config-status").textContent = "reverted";
}

// ── Glossary editor ──────────────────────────────────────────────────────────
let glossaryBaseline = [];     // server-side terms (for revert)
let glossaryTerms = [];        // editable working copy
let glossaryModes = ["always", "suggest"];

async function loadGlossary() {
  try {
    const r = await fetch("/api/glossary");
    if (!r.ok) return;
    const data = await r.json();
    glossaryBaseline = JSON.parse(JSON.stringify(data.terms || []));
    glossaryTerms = JSON.parse(JSON.stringify(data.terms || []));
    glossaryModes = data.modes || glossaryModes;
    $("#glossary-path").textContent = data.path || "canadian_glossary.yaml";
    renderGlossary();
  } catch (e) {
    $("#glossary-status").textContent = "load failed: " + e;
  }
}

function glossaryDirty() {
  return JSON.stringify(glossaryBaseline) !== JSON.stringify(glossaryTerms);
}

function renderGlossary() {
  const tbody = $("#glossary-table tbody");
  tbody.innerHTML = "";
  for (let i = 0; i < glossaryTerms.length; i++) {
    tbody.appendChild(renderGlossaryRow(glossaryTerms[i], i));
  }
  const n = glossaryTerms.length;
  $("#glossary-summary").textContent =
    `· ${n} term${n === 1 ? "" : "s"}` + (glossaryDirty() ? " · unsaved" : "");
}

function renderGlossaryRow(term, idx) {
  const tr = document.createElement("tr");
  if (glossaryDirty()) tr.classList.add("maybe-dirty");
  const fields = [
    { key: "en", placeholder: "speaker" },
    { key: "fr_ca", placeholder: "conférencier·ère" },
  ];
  for (const f of fields) {
    const td = document.createElement("td");
    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = term[f.key] || "";
    inp.placeholder = f.placeholder;
    inp.addEventListener("input", e => {
      glossaryTerms[idx][f.key] = e.target.value;
      // No full re-render — just update the summary
      $("#glossary-summary").textContent =
        `· ${glossaryTerms.length} terms${glossaryDirty() ? " · unsaved" : ""}`;
    });
    td.appendChild(inp);
    tr.appendChild(td);
  }
  // Mode select
  const tdMode = document.createElement("td");
  const sel = document.createElement("select");
  for (const m of glossaryModes) {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    if ((term.mode || "suggest") === m) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener("change", e => { glossaryTerms[idx].mode = e.target.value; renderGlossary(); });
  tdMode.appendChild(sel);
  tr.appendChild(tdMode);
  // Delete
  const tdDel = document.createElement("td");
  const del = document.createElement("button");
  del.type = "button";
  del.className = "cancel";
  del.textContent = "✕";
  del.title = "Remove term";
  del.addEventListener("click", () => {
    glossaryTerms.splice(idx, 1);
    renderGlossary();
  });
  tdDel.appendChild(del);
  tr.appendChild(tdDel);
  return tr;
}

async function saveGlossary() {
  const status = $("#glossary-status");
  // Drop empty rows silently
  const payload = glossaryTerms.filter(t => (t.en || "").trim() && (t.fr_ca || "").trim());
  status.textContent = "saving…";
  try {
    const r = await fetch("/api/glossary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ terms: payload }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) { status.textContent = "failed: " + (data.detail || r.statusText); return; }
    const m = (data.mirrored || []).length;
    status.textContent = `saved ${data.count} term${data.count === 1 ? "" : "s"}` +
      (m ? ` (mirrored to ${m} cop${m === 1 ? "y" : "ies"})` : "");
    await loadGlossary();
  } catch (e) { status.textContent = "network error: " + e; }
}

// ── Translation review ───────────────────────────────────────────────────────
async function loadReviewSegments(jobId, job) {
  const status = $("#review-status");
  status.textContent = "loading segments…";
  try {
    const r = await fetch(`/api/jobs/${jobId}/segments`);
    if (!r.ok) { status.textContent = "failed to load: " + await r.text(); return; }
    reviewJobId = jobId;
    reviewSegments = await r.json();
    $("#review-job-info").textContent =
      `${job ? job.video_filename : jobId} · ${reviewSegments.length} segments`;
    const dlArea = $("#review-downloads");
    dlArea.innerHTML = "";
    if (job?.outputs?.english_srt) {
      const a = document.createElement("a");
      a.href = `/api/jobs/${jobId}/download/english_srt`;
      a.textContent = "Download English SRT";
      a.className = "dl";
      a.download = "";
      dlArea.appendChild(a);
    }
    renderReviewTable();
    await loadVoiceRefs(jobId);
    status.textContent = `${reviewSegments.length} segments loaded`;
    $("#review-section").hidden = false;
    document.getElementById("review-section").scrollIntoView({ behavior: "smooth" });
  } catch (e) { status.textContent = "network error: " + e; }
}

function renderReviewTable() {
  const tbody = $("#review-tbody");
  tbody.innerHTML = "";
  for (let i = 0; i < reviewSegments.length; i++) {
    const seg = reviewSegments[i];
    const tr = document.createElement("tr");
    // English column
    const tdEn = document.createElement("td");
    tdEn.className = "review-en";
    const ts = document.createElement("div");
    ts.className = "review-ts muted";
    ts.textContent = `${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s`;
    const enDiv = document.createElement("div");
    enDiv.textContent = seg.text || "";
    tdEn.append(ts, enDiv);
    tr.appendChild(tdEn);
    // French editable column
    const tdFr = document.createElement("td");
    const ta = document.createElement("textarea");
    ta.className = "review-fr-input";
    ta.value = seg.text_fr || seg.text_fr_natural || "";
    ta.rows = Math.max(2, Math.ceil(ta.value.length / 55));
    ta.addEventListener("input", e => {
      reviewSegments[i].text_fr = e.target.value;
      reviewSegments[i].text_fr_natural = e.target.value;
    });
    tdFr.appendChild(ta);
    tr.appendChild(tdFr);
    tbody.appendChild(tr);
  }
}

async function saveReviewSegments() {
  const status = $("#review-status");
  status.textContent = "saving…";
  try {
    const r = await fetch(`/api/jobs/${reviewJobId}/segments`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reviewSegments),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      const d = data.detail;
      const msg = typeof d === "string" ? d
        : Array.isArray(d) ? d.map(e => e.msg || JSON.stringify(e)).join("; ")
        : r.statusText;
      status.textContent = "save failed: " + msg;
      return false;
    }
    if (!await saveVoiceRefs()) return false;
    status.textContent = `saved ${data.count} segments`;
    return true;
  } catch (e) { status.textContent = "network error: " + e; return false; }
}

async function approveReview() {
  if (!confirm("Approve translations and start TTS synthesis?")) return;
  if (!await saveReviewSegments()) return;
  const status = $("#review-status");
  status.textContent = "approving…";
  try {
    const r = await fetch(`/api/jobs/${reviewJobId}/approve`, { method: "POST" });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) { status.textContent = "failed: " + (data.detail || r.statusText); return; }
    status.textContent = "Phase 2 queued — TTS synthesis starting…";
    $("#review-section").hidden = true;
    $("#voice-ref-section").hidden = true;
    reviewJobId = null;
    reviewSegments = [];
    await refreshJobs();
  } catch (e) { status.textContent = "network error: " + e; }
}

// ── Voice reference selection ────────────────────────────────────────────────
let voiceLibrary = [];
let vocalsDuration = 0;

async function loadVoiceRefs(jobId) {
  const section = $("#voice-ref-section");
  const panels = $("#voice-ref-panels");
  panels.innerHTML = "";
  try {
    const [spkRes, libRes] = await Promise.all([
      fetch(`/api/jobs/${jobId}/voices/speakers`),
      fetch(`/api/voices/library`),
    ]);
    if (!spkRes.ok) { section.hidden = true; return; }
    const data = await spkRes.json();
    voiceLibrary = libRes.ok ? (await libRes.json()).voices || [] : [];
    vocalsDuration = data.vocals_duration || 0;
    if (!data.vocals_available) {
      $("#voice-ref-hint").textContent =
        "Source vocals not available for this job — voice reference is auto-selected.";
      section.hidden = false;
      return;
    }
    $("#voice-ref-hint").textContent =
      `Pick the cleanest sample of each speaker's voice (source is ${vocalsDuration.toFixed(0)}s). Leave on Auto to let the pipeline choose.`;
    for (const spk of data.speakers) {
      panels.appendChild(
        buildVoicePanel(jobId, spk, (data.saved || {})[spk], (data.ranges || {})[spk])
      );
    }
    section.hidden = false;
  } catch (e) { section.hidden = true; }
}

function buildVoicePanel(jobId, speaker, saved, rangeInfo) {
  const wrap = document.createElement("div");
  wrap.className = "voice-panel";
  wrap.dataset.speaker = speaker;

  const mode = saved?.source || "auto";
  // Default this speaker's range to a window where THEY actually speak (their
  // longest turn), not a shared fixed offset — picking a generic time on the
  // mixed vocals can land on another speaker and collapse the voices together.
  const sug = rangeInfo?.suggested || {};
  const defStart = saved?.start ?? sug.start ?? 20;
  const defDur = saved?.duration ?? sug.duration ?? 12;
  const libOpts = voiceLibrary.map(v =>
    `<option value="${v}" ${saved?.source === "library" && (saved.path || "").endsWith(v) ? "selected" : ""}>${v}</option>`
  ).join("");

  // Clickable chips for this speaker's longest turns, so the user can drop the
  // range onto a known-good window in one click.
  const turns = rangeInfo?.turns || [];
  const chips = turns.map(([a, b]) =>
    `<button type="button" class="vr-turn" data-start="${a}" data-dur="${Math.min(Math.round(b - a), 12)}">${a}–${b}s</button>`
  ).join("");
  const turnsRow = turns.length
    ? `<div class="vr-turns">talks at: ${chips}</div>`
    : "";

  wrap.innerHTML = `
    <strong>${speaker}</strong>
    <select class="vr-mode">
      <option value="auto" ${mode === "auto" ? "selected" : ""}>Auto</option>
      <option value="range" ${mode === "range" ? "selected" : ""}>Pick range</option>
      <option value="library" ${mode === "library" ? "selected" : ""} ${voiceLibrary.length ? "" : "disabled"}>Library</option>
    </select>
    <span class="vr-range" ${mode === "range" ? "" : "hidden"}>
      start <input type="number" class="vr-start" min="0" step="1" value="${defStart}"> s,
      length <input type="number" class="vr-dur" min="1" max="30" step="1" value="${defDur}"> s
      ${turnsRow}
    </span>
    <span class="vr-library" ${mode === "library" ? "" : "hidden"}>
      <select class="vr-lib">${libOpts || "<option>(no clips)</option>"}</select>
    </span>
    <button type="button" class="vr-preview">▶ Preview</button>
    <audio class="vr-audio" controls hidden></audio>
  `;

  const modeSel = wrap.querySelector(".vr-mode");
  const rangeBox = wrap.querySelector(".vr-range");
  const libBox = wrap.querySelector(".vr-library");
  const preview = wrap.querySelector(".vr-preview");
  const audio = wrap.querySelector(".vr-audio");
  modeSel.addEventListener("change", () => {
    rangeBox.hidden = modeSel.value !== "range";
    libBox.hidden = modeSel.value !== "library";
    preview.hidden = modeSel.value === "auto";
    audio.hidden = true;
  });
  // Talk-time chip → set the range to that turn and preview it immediately.
  wrap.querySelectorAll(".vr-turn").forEach(chip => {
    chip.addEventListener("click", () => {
      wrap.querySelector(".vr-start").value = chip.dataset.start;
      wrap.querySelector(".vr-dur").value = chip.dataset.dur;
      preview.click();
    });
  });
  preview.hidden = mode === "auto";
  preview.addEventListener("click", () => {
    if (modeSel.value === "range") {
      const s = parseFloat(wrap.querySelector(".vr-start").value) || 0;
      const d = parseFloat(wrap.querySelector(".vr-dur").value) || 12;
      audio.src = `/api/jobs/${jobId}/vocals-clip?start=${s}&dur=${d}`;
    } else if (modeSel.value === "library") {
      const f = wrap.querySelector(".vr-lib").value;
      audio.src = `/api/voices/library/${encodeURIComponent(f)}`;
    } else { return; }
    audio.hidden = false;
    audio.play().catch(() => {});
  });
  return wrap;
}

function collectVoiceRefs() {
  const out = {};
  document.querySelectorAll("#voice-ref-panels .voice-panel").forEach(p => {
    const spk = p.dataset.speaker;
    const mode = p.querySelector(".vr-mode").value;
    if (mode === "range") {
      out[spk] = {
        source: "range",
        start: parseFloat(p.querySelector(".vr-start").value) || 0,
        duration: parseFloat(p.querySelector(".vr-dur").value) || 12,
      };
    } else if (mode === "library") {
      out[spk] = { source: "library", path: p.querySelector(".vr-lib").value };
    }
    // "auto" → omit
  });
  return out;
}

async function saveVoiceRefs() {
  if ($("#voice-ref-section").hidden) return true;
  try {
    const r = await fetch(`/api/jobs/${reviewJobId}/voice-refs`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectVoiceRefs()),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      $("#review-status").textContent = "voice ref save failed: " + (d.detail || r.statusText);
      return false;
    }
    return true;
  } catch (e) { $("#review-status").textContent = "network error: " + e; return false; }
}

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  await loadOptions();
  await loadConfig();
  await loadGlossary();
  await refreshHealth();
  await refreshJobs();
  $("#config-save").addEventListener("click", () => saveConfig(false));
  $("#config-revert").addEventListener("click", revertConfig);
  $("#review-save").addEventListener("click", () => saveReviewSegments());
  $("#review-approve").addEventListener("click", approveReview);
  $("#glossary-add").addEventListener("click", () => {
    glossaryTerms.push({ en: "", fr_ca: "", mode: "suggest" });
    renderGlossary();
    const inputs = $$("#glossary-table tbody tr:last-child input");
    if (inputs.length) inputs[0].focus();
  });
  $("#glossary-save").addEventListener("click", saveGlossary);
  $("#glossary-revert").addEventListener("click", () => {
    glossaryTerms = JSON.parse(JSON.stringify(glossaryBaseline));
    renderGlossary();
    $("#glossary-status").textContent = "reverted";
  });
  setInterval(refreshJobs, 5000);
  setInterval(refreshHealth, 15000);
}
init();
