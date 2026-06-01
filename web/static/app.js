// Vanilla JS, no framework. Single page app for the dubbing web UI.

const TERMINAL = new Set(["completed", "failed", "cancelled"]);
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Per-job EventSource handles so we can close them when a job terminates
const streams = new Map();        // job_id -> EventSource
const logBuffers = new Map();     // job_id -> string[]
let lastJobsHash = "";

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

$("#submit-form").addEventListener("submit", e => {
  e.preventDefault();
  if (!fileInput.files.length) return;
  const fd = new FormData($("#submit-form"));

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  const prog = $("#upload-progress");
  prog.hidden = false;
  prog.value = 0;
  const status = $("#upload-status");
  status.textContent = "uploading…";
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
  for (const [kind, label] of [["audio", "Audio"], ["srt", "SRT"], ["full", "Full mix"]]) {
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

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  await loadOptions();
  await refreshHealth();
  await refreshJobs();
  setInterval(refreshJobs, 5000);
  setInterval(refreshHealth, 15000);
}
init();
