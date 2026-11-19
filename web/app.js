"use strict";

const $ = (id) => document.getElementById(id);
const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
const pct = (value) => value == null ? "—" : percent.format(value);
const safe = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[char]);
const labels = { scam: "Scam", genuine: "Genuine", promotional_spam: "Promotional spam", unsure: "Unsure" };
const labelIds = { scam: 1, genuine: 0, promotional_spam: 2, unsure: null };
const reviewLabel = (label) => [labels.genuine, labels.scam, labels.promotional_spam][label] ?? labels.unsure;
const original = "Aapka bank account block hoga. KYC ke liye OTP share karo abhi.";
const variants = [
  "Aapka bank account block hoga. K.Y.C ke liye O.T.P share kro abhi.",
  "Aapka b a n k account block hoga. K Y C ke liye O T P share karo abhi.",
  "आपका बैंक अकाउंट ब्लॉक होगा। केवाईसी के लिए ओटीपी शेयर करो अभी।"
];
let variantIndex = 0;
let data;
let scannedText;
let scanRevision = 0;
let replayState;
let trainingBusy = false;
let savedDriftReports = [];

async function request(url, payload) {
  const options = payload === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
  };
  options.signal = AbortSignal.timeout(30_000);
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!body && response.ok) throw new Error("The server returned an unexpected response.");
  if (!response.ok) throw new Error(typeof body?.detail === "string" ? body.detail : `Request failed (HTTP ${response.status}).`);
  return body;
}

function notice(message, error = false) {
  const node = $("notice");
  node.hidden = !message;
  node.textContent = message;
  node.className = error ? "error" : "";
}

async function run(button, action) {
  button.disabled = true;
  notice("");
  try {
    await action();
  } catch (error) {
    notice(error.message, true);
  } finally {
    button.disabled = false;
    if (button.id === "train") renderTrainingButton();
  }
}

function selectTab() {
  const known = ["scan", "replay", "models", "drift"];
  const tab = known.includes(location.hash.slice(1)) ? location.hash.slice(1) : "scan";
  for (const name of known) {
    $(name + "-panel").hidden = tab !== name;
    const link = document.querySelector('[data-tab="' + name + '"]');
    if (name === tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  notice("");
  if (tab === "drift") {
    loadDriftHistory().catch((error) => notice(error.message, true));
  }
}

function labelButtons(onClick) {
  const buttons = Object.entries(labels).map(([label, name]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.onclick = async () => {
      buttons.forEach((item) => { item.disabled = true; });
      try {
        await run(button, () => onClick(label));
      } finally {
        buttons.forEach((item) => { item.disabled = false; });
      }
    };
    return button;
  });
  return buttons;
}

function renderTrainingButton() {
  const counts = data?.review_counts;
  $("train").disabled = trainingBusy || !counts?.scam || !counts?.genuine;
  $("train").textContent = trainingBusy ? "Training candidate…" : "Train candidate";
}

function renderStatus() {
  $("live").innerHTML = "<i></i> Active · v" + data.active;
  $("scans").textContent = data.scans.total + " scans";
  $("review-count").textContent = `${data.review_counts.scam} Scam, ${data.review_counts.genuine} Genuine and ${data.review_counts.promotional_spam ?? 0} Promotional spam labels. Save at least one Scam and one Genuine label to train.`;
  renderTrainingButton();
  const selected = $("candidate").value;
  $("candidate").innerHTML = '<option value="">Active model only</option>' + data.models
    .filter((model) => model.id !== data.active)
    .map((model) => '<option value="' + model.id + '">Version ' + model.id + '</option>').join("");
  if ([...$("candidate").options].some((option) => option.value === selected)) $("candidate").value = selected;
  $("models").innerHTML = data.models.map((model) => {
    const active = model.id === data.active;
    const m = model.metrics;
    return '<section class="model-row"><div class="card-head"><div><h3>Version ' + model.id + '</h3>' +
      '<p class="muted">MLflow run ' + safe(model.run_id.slice(0, 12)) + ' · ' + m.reviewed_rows + ' reviewed examples</p></div>' +
      '<span class="tag">' + (active ? "ACTIVE" : model.deployed ? "PREVIOUSLY DEPLOYED" : "CANDIDATE") + '</span></div>' +
      '<p class="muted">' + safe(m.model_type ?? "Logistic regression") + ' · ' +
      safe(new Date(model.created).toLocaleString()) + ' · Data ' + safe(m.dataset_sha256?.slice(0, 12) ?? "unavailable") + '</p>' +
      stats([[pct(m.benchmark?.recall), "Benchmark recall"], [pct(m.benchmark?.false_alarm), "Benchmark false alarms"],
        [pct(m.campaign_audit?.recall), "Campaign recall"], [pct(m.campaign_audit?.false_alarm), "Campaign false alarms"]]) +
      (active ? "" : '<ul class="gates">' + model.gates.map((gate) =>
        '<li>' + (gate.pass ? "Pass · " : "Fail · ") + safe(gate.label) +
        ' <strong>(' + safe(gate.actual) + ')</strong></li>').join("") + '</ul><div class="actions">' +
        '<button data-action="promote" data-version="' + model.id + '"' + (model.gates.every((gate) => gate.pass) ? "" : " disabled") + '>Promote</button>' +
        (model.deployed ? '<button data-action="rollback" data-version="' + model.id + '">Roll back to v' + model.id + '</button>' : "") + '</div>') +
      '</section>';
  }).join("");
  $("history").innerHTML = data.events.map((event) => '<div class="event"><span>' + safe(event.kind) + ' · v' + event.model_id +
    '</span><time datetime="' + safe(event.created) + '">' + safe(new Date(event.created).toLocaleString()) + '</time></div>').join("");
  $("scan-activity").innerHTML = data.recent_scans.map((row) => '<div class="event"><span>v' + row.model_id +
    '</span><span>' + row.scans + ' scans · ' + pct(row.flagged / row.scans) + ' flagged · ' +
    row.latency_ms.toFixed(1) + ' ms mean scoring time</span></div>').join("") ||
    '<p class="muted">No scans recorded in this period.</p>';
}

async function refresh() {
  data = await request("/api/status");
  renderStatus();
}

function stats(entries) {
  return '<div class="stats">' + entries.map(([value, label]) =>
    '<div><strong>' + safe(value) + '</strong><span>' + safe(label) + '</span></div>').join("") + '</div>';
}

function evaluationSlices(title, groups) {
  return '<details class="model-slices"><summary>' + title + '</summary>' +
    Object.entries(groups).map(([name, m]) => '<div class="event"><strong>' + safe(name) +
      '</strong><span>' + m.caught + '/' + m.positives + ' caught · ' +
      m.false_flags + '/' + m.negatives + ' false flags · N=' + m.total + '</span></div>').join("") +
    '</details>';
}

function renderReplay(result, cohort) {
  const review = cohort === "review";
  const showReference = !review || $("show-reference").checked;
  const changes = $("changes").value;
  $("replay-options").hidden = false;
  $("review-options").hidden = !review;
  $("changes-control").hidden = review;
  for (const option of $("changes").options) {
    option.disabled = ["caught", "missed", "false", "fixed"].includes(option.value) && result.comparison.length < 2;
  }
  $("review-progress").textContent = result.rows.filter((row) => row.reviewed && row.review_label != null).length +
    " of " + result.rows.length + " labeled";
  const metricStats = (m) => stats([[m.caught + "/" + (m.caught + m.missed), "Scams caught"],
    [m.false_flags + "/" + (m.false_flags + m.genuine), "False flags"], [pct(m.recall), "Recall"],
    [pct(m.false_alarm), "False-alarm rate"], [pct(m.precision), "Precision"],
    [pct(m.f1), "F1"], [m.missed, "False negatives"], [m.genuine, "True negatives"]]);
  $("comparison").innerHTML = result.comparison.map((model, i) => {
    return '<article class="card"><h3>' + (i ? "Comparison" : "Active") + ' · v' + model.version + '</h3>' +
      (showReference ? metricStats(model.metrics) + evaluationSlices("By campaign", model.by_campaign) +
        evaluationSlices("By writing script", model.by_script) : '') + '</article>';
  }).join("");
  $("campaigns").replaceChildren();
  const groups = new Map();
  for (const [index, row] of result.rows.entries()) {
    if (review && $("pending-only").checked && row.reviewed && row.review_label != null) continue;
    if (!review && ["fp", "fn"].includes(changes)) {
      const flag = result.comparison.at(-1).flags[index];
      if (changes === "fp" ? row.label || !flag : !row.label || flag) continue;
    } else if (!review && changes && result.comparison.length > 1) {
      const [before, after] = result.comparison.map((model) => model.flags[index]);
      const change = before === after ? "" : row.label ? (after ? "caught" : "missed") : (after ? "false" : "fixed");
      if (change !== changes) continue;
    }
    if (!groups.has(row.campaign)) groups.set(row.campaign, []);
    groups.get(row.campaign).push([row, index]);
  }
  if (!groups.size) $("campaigns").textContent = "No messages match the current filter.";
  for (const [name, rows] of groups) {
    const section = document.createElement("details");
    section.open = true;
    section.innerHTML = '<summary>' + safe(name) + ' · ' + rows.length + ' messages</summary><div class="table-scroll">' +
      '<table><thead><tr><th scope="col">Message</th>' + (showReference ? '<th scope="col">Reference label</th>' : '') +
      result.comparison.map((model) => '<th>v' + model.version + '</th>').join("") +
      (cohort === "review" ? "<th>Review label</th>" : "") + '</tr></thead><tbody></tbody></table></div>';
    for (const [row, index] of rows) {
      const tr = section.querySelector("tbody").insertRow();
      tr.insertCell().textContent = row.text;
      if (showReference) tr.insertCell().textContent = row.label ? "Scam" : "Genuine";
      for (const model of result.comparison) {
        const cell = tr.insertCell();
        const flagged = model.flags[index];
        cell.textContent = pct(model.scores[index]) + " · " + (flagged ? "Flagged" : "Below threshold");
        if (showReference && flagged !== Boolean(row.label)) cell.className = "warn";
      }
      if (cohort === "review") {
        const cell = tr.insertCell();
        const saved = document.createElement("div");
        saved.className = "muted";
        const history = document.createElement("details");
        const updateSaved = () => {
          saved.textContent = row.reviewed ? "Saved · " + reviewLabel(row.review_label) : "Pending";
          history.hidden = !row.review_history.length;
          history.innerHTML = "<summary>Label history</summary><ul>" + row.review_history.map((entry) =>
            "<li>" + reviewLabel(entry.label) + " · " + safe(new Date(entry.created).toLocaleString()) + "</li>"
          ).join("") + "</ul>";
        };
        updateSaved();
        const actions = document.createElement("div");
        actions.className = "actions";
        actions.append(...labelButtons(async (label) => {
          const savedReview = await request("/api/reviews", { text: row.text, label });
          row.reviewed = true;
          row.review_label = labelIds[label];
          row.review_history = [{ label: row.review_label, created: savedReview.updated }, ...row.review_history].slice(0, 3);
          updateSaved();
          if (replayState?.result === result) {
            $("review-progress").textContent = result.rows.filter((item) => item.reviewed && item.review_label != null).length +
              " of " + result.rows.length + " labeled";
            if ($("pending-only").checked && row.review_label != null) {
              tr.remove();
              if (!section.querySelector("tbody tr")) section.remove();
              if (!$("campaigns").children.length) $("campaigns").textContent = "No messages match the current filter.";
            }
          }
          await refresh();
        }));
        cell.append(saved, actions, history);
      }
    }
    $("campaigns").append(section);
  }
}

function invalidateScan() {
  scanRevision++;
  scannedText = undefined;
  $("scan-version").textContent = "—";
  $("score-number").textContent = "—";
  $("score-number").classList.toggle("alert", false);
  $("risk").value = 0;
  $("verdict").textContent = "Analyze a message to view its score.";
  $("cue-heading").hidden = true;
  $("cues").replaceChildren();
  $("scan-review").hidden = true;
}

$("scan-form").onsubmit = (event) => {
  event.preventDefault();
  const text = $("message").value;
  invalidateScan();
  const revision = scanRevision;
  run(event.submitter, async () => {
    const result = await request("/api/scan", { text });
    if (revision !== scanRevision || text !== $("message").value) return;
    scannedText = text;
    $("scan-version").textContent = "v" + result.model;
    $("score-number").textContent = pct(result.score);
    $("score-number").classList.toggle("alert", result.flagged);
    $("risk").value = result.score;
    $("verdict").textContent = (result.priority ?? (result.flagged ? "Flagged for review" : "Below the flagging threshold")) +
      " · Threshold " + pct(result.threshold);
    $("cue-heading").hidden = result.cues.length === 0;
    $("cues").innerHTML = result.cues.map((cue) => '<span class="tag">' + safe(cue.kind) + ' · ' + safe(cue.text) + '</span>').join("");
    $("scan-review").hidden = false;
    await refresh();
  });
};
$("scan-labels").append(...labelButtons(async (label) => {
  const text = scannedText;
  if (text === undefined || text !== $("message").value) return;
  await request("/api/reviews", { text, label });
  await refresh();
  notice("Review saved as " + labels[label] + ".");
}));
$("message").addEventListener("input", invalidateScan);
$("sample").onclick = () => {
  $("message").value = original;
  invalidateScan();
};
$("variant").onclick = () => {
  $("message").value = variants[variantIndex % variants.length];
  variantIndex++;
  invalidateScan();
  $("message").focus();
};
function clearReplay() {
  replayState = undefined;
  $("comparison").replaceChildren();
  $("campaigns").replaceChildren();
  $("replay-options").hidden = true;
}

const driftLabels = {
  normal: "Normal", warning: "Warning", significant: "Significant shift", insufficient: "Insufficient data",
};

function renderDrift(report) {
  const output = $("drift-output");
  if (!report) {
    output.innerHTML = '<p class="muted">No saved reports. Analyze drift to compare recent scans.</p>';
    return;
  }
  const count = (state) => report.features.filter((feature) => feature.status === state).length;
  const date = (value) => safe(new Date(value).toLocaleString());
  const action = count("significant") || count("warning")
    ? "Inspect shifted features and review a sample of labeled outcomes."
    : count("insufficient")
      ? "Collect more scans in both windows. Predictions are compared only within the active model version."
      : "Continue monitoring incoming traffic and reviewed outcomes.";
  output.innerHTML = '<article class="card history"><h3>Report ' + report.id + ' · Active v' +
    safe(report.active ?? "unavailable") + '</h3>' +
    '<p class="muted">Reference ' + date(report.reference_start) + ' to ' + date(report.current_start) +
    ' (' + report.reference_count + ') · Current ' + date(report.current_start) + ' to ' + date(report.created) +
    ' (' + report.current_count + ') · Minimum ' + report.minimum + ' per window.</p>' +
    stats([[count("significant"), "Significant"], [count("warning"), "Warnings"], [count("insufficient"), "Insufficient"]]) +
    '<p>' + action + '</p><div class="table-scroll"><table><thead><tr>' +
    '<th scope="col">Feature</th><th scope="col">Type</th><th scope="col">Distance</th>' +
    '<th scope="col">Samples (ref / current)</th><th scope="col">Status</th></tr></thead><tbody>' +
    report.features.map((feature) => '<tr><th scope="row">' + safe(feature.feature) + '</th><td>' +
      safe(feature.kind) + '</td><td>' + (feature.distance == null ? "—" : feature.distance.toFixed(3)) +
      '</td><td>' + feature.reference_count + ' / ' + feature.current_count + '</td><td>' +
      safe(driftLabels[feature.status]) + '</td></tr>').join("") +
    '</tbody></table></div></article>';
}

async function loadDriftHistory(selected) {
  savedDriftReports = (await request("/api/drift")).reports;
  const history = $("drift-history");
  history.innerHTML = savedDriftReports.map((report) => '<option value="' + report.id + '">' +
    safe(new Date(report.created).toLocaleString()) + ' · report ' + report.id + '</option>').join("") ||
    '<option value="">No reports available</option>';
  if (selected != null && savedDriftReports.some((report) => report.id === selected)) history.value = selected;
  renderDrift(savedDriftReports.find((report) => report.id === Number(history.value)));
}

$("drift-history").onchange = (event) => {
  renderDrift(savedDriftReports.find((report) => report.id === Number(event.target.value)));
};

$("drift-form").onsubmit = (event) => {
  event.preventDefault();
  run(event.submitter, async () => {
    const report = await request("/api/drift", {
      days: Number($("drift-days").value), minimum: Number($("drift-minimum").value),
    });
    renderDrift(report);
    await loadDriftHistory(report.id);
  });
};

for (const field of ["cohort", "candidate"]) $(field).onchange = clearReplay;
for (const field of ["changes", "pending-only", "show-reference"]) {
  $(field).onchange = () => {
    if (replayState) renderReplay(replayState.result, replayState.cohort);
  };
}

$("replay-form").onsubmit = (event) => {
  event.preventDefault();
  run(event.submitter, async () => {
    const cohort = $("cohort").value;
    const candidate = $("candidate").value;
    const result = await request("/api/replay", { cohort, candidate: candidate ? Number(candidate) : null });
    if (cohort !== $("cohort").value || candidate !== $("candidate").value) return;
    replayState = { result, cohort };
    $("changes").value = "";
    renderReplay(result, cohort);
  });
};
async function watchTraining(job) {
  trainingBusy = true;
  renderTrainingButton();
  notice("Training candidate. This may take a moment.");
  try {
    const deadline = Date.now() + 180_000;
    while (job.state === "training" && Date.now() < deadline) {
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      job = await request("/api/train/" + job.id);
      if (job.state === "training") notice("Candidate training · " + (job.phase ?? "running"));
    }
    if (job.state === "training") throw new Error("Training is still running. Reload the dashboard to resume tracking.");
    if (job.state !== "finished") throw new Error(job.detail || "Training failed. Check the server logs.");
    await refresh();
    clearReplay();
    $("candidate").value = job.version;
    notice("Candidate v" + job.version + " trained and logged in MLflow.");
  } finally {
    trainingBusy = job.state === "training";
    renderTrainingButton();
  }
}

$("train").onclick = () => run($("train"), async () => {
  const latest = await request("/api/train");
  await watchTraining(latest.state === "training" ? latest : await request("/api/train", {}));
});
$("models").onclick = (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  run(button, async () => {
    const result = await request("/api/models/" + button.dataset.version + "/" + button.dataset.action, {});
    await refresh();
    clearReplay();
    notice("Version " + result.active + " is now active.");
  });
};
window.addEventListener("hashchange", selectTab);
selectTab();
refresh().then(async () => {
  const job = await request("/api/train");
  if (job.state === "training") await watchTraining(job);
  else if (job.state === "interrupted" || job.state === "failed") notice(job.detail, true);
}).catch((error) => {
  if (!data) $("live").textContent = "Disconnected";
  notice(error.message, true);
});
