"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const response = (json) => ({ ok: true, json: async () => json });
const settle = () => new Promise((resolve) => setImmediate(resolve));

function bootDashboard(route, hash = "") {
  const nodes = new Map();
  const makeNode = () => ({
    hidden: true, value: "", textContent: "", options: [],
    classList: { toggle() {} },
    append(...children) { this.children = children; },
    replaceChildren(...children) { this.children = children; },
    addEventListener(name, callback) { this[name] = callback; },
    setAttribute() {},
    removeAttribute() {},
  });
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, makeNode());
    return nodes.get(id);
  }
  const fetch = async (url, options) => {
    if (url === "/api/status") {
      return response({ active: 1, models: [], events: [], scans: { total: 0 },
        review_counts: { scam: 0, genuine: 0 }, recent_scans: [] });
    }
    if (url === "/api/train") return response({ state: "idle" });
    return route(url, options);
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "web/app.js"), "utf8"), {
    document: { getElementById: node, querySelector: node, createElement: makeNode },
    window: { addEventListener() {} }, location: { hash }, fetch, AbortSignal,
  });
  return { node, submit: (form) => node(form).onsubmit({ preventDefault() {}, submitter: makeNode() }) };
}

test("a scan that finishes after an edit cannot assign feedback to the previous message", async () => {
  const reviews = [];
  let finishScan;
  const { node, submit } = bootDashboard(async (url, options) => {
    if (url === "/api/scan") {
      return new Promise((resolve) => {
        finishScan = () => resolve(response({ model: 1, score: .2, threshold: .5, flagged: false, cues: [] }));
      });
    }
    if (url === "/api/reviews") {
      reviews.push(JSON.parse(options.body));
      return response({ saved: true });
    }
    throw new Error("Unexpected request " + url);
  });
  await settle();
  node("message").value = "First message";
  submit("scan-form");
  node("message").value = "Second message";
  node("message").input();
  finishScan();
  await settle();
  assert.equal(node("scan-review").hidden, true);
  await node("scan-labels").children[0].onclick();
  assert.deepEqual(reviews, []);
  submit("scan-form");
  finishScan();
  await settle();
  assert.equal(node("scan-review").hidden, false);
  await node("scan-labels").children[0].onclick();
  assert.deepEqual(reviews, [{ text: "Second message", label: "scam" }]);
});

test("drift form submits its window controls and displays the persisted report", async () => {
  let submitted;
  const saved = [];
  const report = {
    id: 7, active: 1, reference_count: 25, current_count: 29, minimum: 25,
    created: "2026-11-19T12:00:00Z", reference_start: "2026-11-13T12:00:00Z",
    current_start: "2026-11-16T12:00:00Z",
    features: [{ feature: "URL present", kind: "input", distance: .4, status: "significant",
      reference_count: 25, current_count: 29 }],
  };
  const { node, submit } = bootDashboard(async (url, options) => {
    assert.equal(url, "/api/drift");
    if (options.method !== "POST") return response({ reports: saved });
    submitted = JSON.parse(options.body);
    saved.push(report);
    return response(report);
  }, "#drift");
  await settle();
  assert.match(node("drift-output").innerHTML, /No saved reports/);
  node("drift-days").value = "3";
  node("drift-minimum").value = "25";
  submit("drift-form");
  await settle();
  assert.deepEqual(submitted, { days: 3, minimum: 25 });
  assert.match(node("drift-history").innerHTML, /report 7/);
  assert.match(node("drift-output").innerHTML, /URL present/);
  assert.match(node("drift-output").innerHTML, /Significant shift/);
});
