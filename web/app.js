"use strict";

const $ = (id) => document.getElementById(id);
const pct = (value) => new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 }).format(value);
const safe = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[char]);
const labels = { scam: "Scam", genuine: "Genuine", unsure: "Unsure" };
const original = "Aapka bank account block hoga. KYC ke liye OTP share karo abhi.";
const variants = [
  "Aapka bank account block hoga. K.Y.C ke liye O.T.P share kro abhi.",
  "Aapka b a n k account block hoga. K Y C ke liye O T P share karo abhi.",
  "आपका बैंक अकाउंट ब्लॉक होगा। केवाईसी के लिए ओटीपी शेयर करो अभी।"
];
let variantIndex = 0;
let data;
let scannedText;

async function request(url, payload) {
  const options = payload === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
  };
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Request failed. Check your input and try again.");
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
    if (button.id === "train" && data) button.disabled = data.reviewed === 0;
  }
}

function selectTab() {
  const known = ["scan", "replay", "models"];
  const tab = known.includes(location.hash.slice(1)) ? location.hash.slice(1) : "scan";
  for (const name of known) {
    $(name + "-panel").hidden = tab !== name;
    const link = document.querySelector('[data-tab="' + name + '"]');
    if (name === tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  notice("");
}

function labelButtons(onClick) {
  return Object.entries(labels).map(([label, name]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.onclick = () => run(button, () => onClick(label));
    return button;
  });
}

function renderStatus() {
  $("live").innerHTML = "<i></i> Active · v" + data.active;
  $("scans").textContent = data.scans.total + " scans";
  $("review-count").textContent = data.reviewed + " labeled messages available for retraining.";
  $("train").disabled = data.reviewed === 0;
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
      stats([[pct(m.benchmark.recall), "Benchmark recall"], [pct(m.benchmark.false_alarm), "Benchmark false alarms"],
        [pct(m.campaign_audit.recall), "Campaign recall"], [pct(m.campaign_audit.false_alarm), "Campaign false alarms"]]) +
      (active ? "" : '<ul class="gates">' + model.gates.map((gate) =>
        '<li>' + (gate.pass ? "Pass · " : "Fail · ") + safe(gate.label) + '</li>').join("") + '</ul><div class="actions">' +
        '<button data-action="promote" data-version="' + model.id + '"' + (model.gates.every((gate) => gate.pass) ? "" : " disabled") + '>Promote</button>' +
        (model.deployed ? '<button data-action="rollback" data-version="' + model.id + '">Roll back to v' + model.id + '</button>' : "") + '</div>') +
      '</section>';
  }).join("");
  $("history").innerHTML = data.events.map((event) => '<div class="event"><span>' + safe(event.kind) + ' · v' + event.model_id +
    '</span><time datetime="' + safe(event.created) + '">' + safe(new Date(event.created).toLocaleString()) + '</time></div>').join("");
}

async function refresh() {
  data = await request("/api/status");
  renderStatus();
}

function stats(entries) {
  return '<div class="stats">' + entries.map(([value, label]) =>
    '<div><strong>' + safe(value) + '</strong><span>' + safe(label) + '</span></div>').join("") + '</div>';
}

function renderReplay(result, cohort) {
  $("comparison").innerHTML = result.comparison.map((model, i) => {
    const m = model.metrics;
    return '<article class="card"><h3>' + (i ? "Comparison" : "Active") + ' · v' + model.version + '</h3>' +
      stats([[m.caught, "Scams caught"], [m.missed, "Missed"], [m.false_flags, "False flags"],
        [pct(m.recall), "Recall"]]) + '</article>';
  }).join("");
  $("campaigns").replaceChildren();
  const groups = new Map();
  for (const row of result.rows) {
    if (!groups.has(row.campaign)) groups.set(row.campaign, []);
    groups.get(row.campaign).push(row);
  }
  for (const [name, rows] of groups) {
    const section = document.createElement("details");
    section.open = true;
    section.innerHTML = '<summary>' + safe(name) + ' · ' + rows.length + ' messages</summary><div class="table-scroll">' +
      '<table><thead><tr><th>Message</th><th>Expected</th>' +
      result.comparison.map((model) => '<th>v' + model.version + '</th>').join("") +
      (cohort === "review" ? "<th>Review label</th>" : "") + '</tr></thead><tbody></tbody></table></div>';
    for (const row of rows) {
      const index = result.rows.indexOf(row);
      const tr = section.querySelector("tbody").insertRow();
      tr.insertCell().textContent = row.text;
      tr.insertCell().textContent = row.label ? "Scam" : "Genuine";
      for (const model of result.comparison) {
        const cell = tr.insertCell();
        const flagged = model.flags[index];
        cell.textContent = pct(model.scores[index]) + " · " + (flagged ? "Flagged" : "Below threshold");
        if (flagged !== Boolean(row.label)) cell.className = "warn";
      }
      if (cohort === "review") {
        const cell = tr.insertCell();
        const saved = document.createElement("div");
        saved.className = "muted";
        saved.textContent = row.reviewed ? "Saved · " + (row.review_label === null ? "Unsure" : row.review_label ? "Scam" : "Genuine") : "Pending";
        const actions = document.createElement("div");
        actions.className = "actions";
        actions.append(...labelButtons(async (label) => {
          await request("/api/reviews", { text: row.text, label });
          saved.textContent = "Saved · " + labels[label];
          await refresh();
        }));
        cell.append(saved, actions);
      }
    }
    $("campaigns").append(section);
  }
}

$("scan-form").onsubmit = (event) => {
  event.preventDefault();
  const text = $("message").value;
  run(event.submitter, async () => {
    const result = await request("/api/scan", { text });
    scannedText = text;
    $("scan-version").textContent = "v" + result.model;
    $("score-number").textContent = pct(result.score);
    $("score-number").classList.toggle("alert", result.flagged);
    $("risk").value = result.score;
    $("verdict").textContent = result.flagged ? "Flagged for review." : "Below the flagging threshold.";
    $("cues").innerHTML = result.cues.map((cue) => '<span class="tag">' + safe(cue.kind) + ' · ' + safe(cue.text) + '</span>').join("");
    $("scan-review").hidden = false;
    await refresh();
  });
};
$("scan-labels").append(...labelButtons(async (label) => {
  await request("/api/reviews", { text: scannedText, label });
  await refresh();
  notice("Review saved as " + labels[label] + ".");
}));
$("message").addEventListener("input", () => { $("scan-review").hidden = true; });
$("sample").onclick = () => {
  $("message").value = original;
  $("scan-review").hidden = true;
};
$("variant").onclick = () => {
  $("message").value = variants[variantIndex % variants.length];
  variantIndex++;
  $("scan-review").hidden = true;
  $("message").focus();
};
$("replay-form").onsubmit = (event) => {
  event.preventDefault();
  run(event.submitter, async () => {
    const cohort = $("cohort").value;
    const candidate = $("candidate").value;
    renderReplay(await request("/api/replay", { cohort, candidate: candidate ? Number(candidate) : null }), cohort);
  });
};
$("train").onclick = () => run($("train"), async () => {
  notice("Training candidate. This may take a moment.");
  await request("/api/train", {});
  let result;
  do {
    await new Promise((resolve) => window.setTimeout(resolve, 700));
    result = await request("/api/train");
  } while (result.state === "training");
  if (result.state !== "finished") throw new Error("Training status was lost. Check the model registry.");
  await refresh();
  $("candidate").value = result.version;
  notice("Candidate v" + result.version + " trained and logged in MLflow.");
});
$("models").onclick = (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  run(button, async () => {
    const result = await request("/api/models/" + button.dataset.version + "/" + button.dataset.action, {});
    await refresh();
    $("comparison").replaceChildren();
    $("campaigns").replaceChildren();
    notice("Version " + result.active + " is now active.");
  });
};
window.addEventListener("hashchange", selectTab);
selectTab();
refresh().catch((error) => {
  $("live").textContent = "Disconnected";
  notice(error.message, true);
});
