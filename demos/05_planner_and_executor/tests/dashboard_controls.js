// Click the dashboard's own controls and report what each one did.
//
//     node tests/dashboard_controls.js <index.html>
//
// Prints one JSON object on stdout, consumed by tests/test_demo11.py.
//
// Why this exists next to dashboard_client.js: that one loads the page and feeds it snapshots,
// which is the test for "the decision route stops updating". This one is the test for a control
// that no longer does what its label says. The environment configuration moved out of the panel
// and into a window behind a button; a page can look exactly right and still have that button
// wired to nothing, and nothing in Python can see that - the ids are in the markup either way. So
// the page's own script runs here against a DOM that remembers the handlers it was given, and
// this harness does what a person does: open the window, change the options, press start.
//
// Nothing reaches the network: fetch is a stub that records what the page would have sent. What is
// checked, in the order a person does it:
//
//   1. "configurar entorno" is in the environment panel, to the left of "iniciar entorno", and
//      pressing it opens a window.
//   2. The window holds the scene editor and the session options; the panel no longer holds either.
//   3. The line under the buttons states what the next start will use, with the form's defaults,
//      and follows the form when it changes.
//   4. "iniciar entorno" sends what the form holds now - not what it held at some earlier time -
//      and closes the window so the start bar is visible while Isaac boots.
//   5. Escape closes the window, as it does the other two.
//   6. The scene editor still paints its rows, now inside the window.

"use strict";

const fs = require("fs");
const vm = require("vm");

const PAGE = process.argv[2];
const HTML = fs.readFileSync(PAGE, "utf8");
const script = HTML.slice(HTML.indexOf("<script>") + 8, HTML.lastIndexOf("</script>"));

const report = { loadError: "", steps: [], warnings: [] };
function step(name, ok, detail) { report.steps.push({ name, ok: !!ok, detail: detail || "" }); }

// ---- the DOM ----------------------------------------------------------------------------
// Every id the page declares, as the client harness does: the shim only hands out elements that
// really exist, so a typo in the page's script fails here rather than silently in the browser.
const PAGE_IDS = new Set([...HTML.matchAll(/id="([^"]+)"/g)].map(m => m[1]));

function registry(id) {
  if (!PAGE_IDS.has(id)) throw new Error("la pagina no declara ningun id " + id);
  const node = new El("div");
  node.attrs.id = id;
  return node;
}

class El {
  constructor(tag) {
    this.tagName = tag || "div"; this.attrs = {}; this.children = [];
    this._text = ""; this._html = ""; this.dataset = {}; this.style = {}; this.value = "";
    this.hidden = false; this.disabled = false; this.handlers = {};
    this.classList = {
      _s: new Set(),
      add: (...c) => c.forEach(x => this.classList._s.add(x)),
      remove: (...c) => c.forEach(x => this.classList._s.delete(x)),
      toggle: (c, on) => (on ? this.classList._s.add(c) : this.classList._s.delete(c)),
      contains: c => this.classList._s.has(c),
    };
    this.className = "";
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text || this.children.map(c => c.textContent).join(""); }
  set innerHTML(html) { this.children = []; this._html = String(html); this._text = ""; }
  get innerHTML() { return this._html; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  append(...kids) { kids.forEach(k => { this.children.push(k); k.parent = this; }); }
  appendChild(k) { this.append(k); return k; }
  querySelector(sel) { return findById(this, sel.replace(/^#/, "")); }
  // A click is whatever onclick was set to, exactly as in a browser; an element with no handler
  // does nothing, which is itself what "is this button wired" is about.
  click() { if (this.onclick) return this.onclick(); }
  focus() {}
  addEventListener(type, fn) { (this.handlers[type] = this.handlers[type] || []).push(fn); }
  dispatchEvent(type) {
    let result = true;
    for (const fn of this.handlers[type] || []) {
      result = fn({ type, target: this, preventDefault() {} });
    }
    return result;
  }
}

function findById(root, id, seen = new Set()) {
  if (seen.has(root)) return null;
  seen.add(root);
  for (const child of root.children || []) {
    if (child.attrs && child.attrs.id === id) return child;
    const hit = findById(child, id, seen);
    if (hit) return hit;
  }
  return null;
}

const page = new El("body");
for (const id of PAGE_IDS) page.append(registry(id));

// The tree the shim holds is flat - one element per id, all children of body - so "is this inside
// the panel" is answered from the markup, which is where a browser reads the panel's bounds from
// too. Two helpers, used only for the structural steps: the DOM is used for everything a click
// touches.
function markupBetween(fromId, stop) {
  const from = HTML.indexOf(`id="${fromId}"`);
  const to = HTML.indexOf(stop);
  return from < 0 || to < from ? "" : HTML.slice(from, to);
}

// ---- the hidden attributes a browser would start with ------------------------------------
// The modal starts hidden, and the start bar is hidden until the first press. Without this a
// shim element is `hidden: false` from the start, so the first thing a window test would prove
// is that the test harness forgot an attribute.
for (const m of HTML.matchAll(/<([a-zA-Z][a-zA-Z0-9]*)([^>]*)>/g)) {
  const block = m[2];
  if (!/\bhidden\b/.test(block)) continue;
  const id = /\bid="([^"]+)"/.exec(block);
  if (!id) continue;
  const node = page.querySelector("#" + id[1]);
  if (node) node.hidden = true;
}

// ---- the values a browser would start with ----------------------------------------------
// Without this every field reads as an empty string: the summary line would say "rondas  ·" and
// any test of the defaults would be a test of nothing. A select shows its first option, and an
// input shows its value attribute, which is what the page's markup declares.
for (const m of HTML.matchAll(/<(input|select|textarea)\b([^>]*)>/g)) {
  const tag = m[1], block = m[2];
  const id = /\bid="([^"]+)"/.exec(block);
  if (!id) continue;
  const node = page.querySelector("#" + id[1]);
  if (!node) continue;
  if (tag === "select") {
    const open = HTML.indexOf(m[0]);
    const options = [...HTML.slice(open, HTML.indexOf("</select>", open))
      .matchAll(/<option\s+value="([^"]+)"/g)].map(o => o[1]);
    node.value = options[0] || "";
  } else {
    const value = /\bvalue="([^"]*)"/.exec(block);
    node.value = value ? value[1] : "";
  }
}

// ---- the server, in memory --------------------------------------------------------------
const sent = [];
const answers = {
  "/api/state": () => ({ nodes: [], edges: [], state: {}, commands: [], events: [], metrics: {} }),
  "/api/config": () => ({
    ok: true, available_colors: ["red", "green", "blue", "yellow", "purple"],
    config: { version: 1, objects: [{ id: "cube_red", kind: "cube", color: "red", count: 1 }],
              containers: [{ id: "tray_blue", name: "Azul A", color: "blue", size_mm: [70, 70, 30],
                             cavity_mm: [56, 56], wall_mm: 7, floor_mm: 6, aliases: [] }] },
    catalog: [
      { id: "cube_red", kind: "cube", color: "red", container: false, name: "cube_red",
        size: [0.03, 0.03, 0.03], inner: [0, 0], height: 0.03, x: 0.29, y: -0.435 },
      { id: "ball_red", kind: "ball", color: "red", container: false, name: "ball_red",
        size: [0.03, 0.03, 0.03], inner: [0, 0], height: 0.03, x: 0.36, y: -0.435 },
      { id: "tray_blue", kind: "box", color: "blue", container: true, name: "Azul A",
        size: [0.07, 0.07, 0.03], inner: [0.056, 0.056], height: 0.03, x: 0.32, y: -0.5 },
    ],
    layout: { work_x: [0.27, 0.52], work_y: [-0.54, -0.42], robot: [0.35, -0.64], reach: 0.33 },
  }),
  "/api/env": body => { sent.push(body); return { ok: true, note: "simulado" }; },
  "/api/command": () => ({ ok: true, state: "QUEUED" }),
};
const document = {
  body: page,
  createElementNS: (ns, tag) => new El(tag),
  createElement: tag => new El(tag),
  querySelector: sel => page.querySelector(sel),
  querySelectorAll: () => [],
  getElementById: id => page.querySelector("#" + id),
  listeners: {},
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
  dispatchEvent(type, event) { for (const fn of this.listeners[type] || []) fn(event); },
};

const sandbox = {
  console: { log() {}, warn: (...a) => report.warnings.push(a.join(" ")),
             error: (...a) => report.warnings.push("ERROR " + a.join(" ")) },
  document,
  location: { protocol: "http:", host: "127.0.0.1:8765" },
  performance: { now: () => 0 },
  URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
  Blob: class {},
  WebSocket: class { constructor() { this.readyState = 0; } close() {} },
  fetch: (path, options) => {
    const body = options && options.body ? JSON.parse(options.body) : null;
    const answer = answers[path] ? answers[path](body) : { ok: true };
    return Promise.resolve({ json: async () => answer });
  },
  setInterval: () => 0, clearInterval() {},
  setTimeout: () => 0, clearTimeout() {},
  navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } },
  AudioContext: class { constructor() { this.sampleRate = 48000; } close() {} },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

try {
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, { filename: "index.html" });
} catch (error) {
  report.loadError = `${error.constructor.name}: ${error.message}`;
  console.log(JSON.stringify(report));
  process.exit(0);
}

function byId(id) { return page.querySelector("#" + id); }

async function main() {
  // 1. the button is in the environment panel, to the left of start, and it opens the window
  const panel = markupBetween("env-group", 'id="cmd-group"');
  const buttonAt = panel.indexOf('id="env-config"');
  const startAt = panel.indexOf('id="start"');
  step("configurar entorno is in the environment panel, left of iniciar entorno",
       buttonAt >= 0 && startAt > buttonAt, `button ${buttonAt}, start ${startAt}`);
  const modal = byId("env-modal");
  step("the configuration window is closed to begin with", modal.hidden, "");
  byId("env-config").click();
  step("the button opens the configuration window", modal.hidden === false, "");

  // 2. the window holds the scene editor and the session options; the panel holds neither
  const expected = ["scene-objects", "scene-containers", "scene-save", "variant-sel", "camera-sel",
                    "rounds", "seed-in"];
  const popup = markupBetween("env-modal", "<main>");
  const inPanel = expected.filter(id => panel.includes(`id="${id}"`));
  const inPopup = expected.filter(id => popup.includes(`id="${id}"`));
  step("the window holds the scene editor and the session options",
       inPopup.length === expected.length, `missing: ${expected.filter(id => !inPopup.includes(id)).join(", ")}`);
  step("the panel no longer holds any of them", inPanel.length === 0, inPanel.join(", "));

  // 3. the summary line under the buttons states the defaults, and follows the form
  const defaults = byId("env-summary").textContent;
  step("the panel states what the next start will use",
       /recinto/.test(defaults) && /cenital/.test(defaults) && /1 ronda/.test(defaults)
       && /semilla 0/.test(defaults), defaults);
  byId("variant-sel").value = "out_of_reach";
  byId("variant-sel").dispatchEvent("change");
  step("changing the variant changes the line",
       /fuera de alcance/.test(byId("env-summary").textContent), byId("env-summary").textContent);
  byId("rounds").value = "3";
  byId("rounds").dispatchEvent("input");
  step("changing the rounds changes the line",
       /3 ronda/.test(byId("env-summary").textContent), byId("env-summary").textContent);

  // 4. start sends what the form holds, and closes the window so the start bar is visible
  byId("start").click();
  for (let i = 0; i < 8; i++) await Promise.resolve();  // the post round-trips through promises
  const last = sent[sent.length - 1];
  step("start sends the options the form holds",
       !!last && last.action === "start" && last.variant === "out_of_reach" && last.rounds === 3
       && last.camera === "top" && last.seed === 0, JSON.stringify(last || null));
  step("start closes the configuration window", modal.hidden === true, "");

  // 5. Escape closes it again, as it does the other two
  byId("env-config").click();
  step("the button opens it again", modal.hidden === false, "");
  document.dispatchEvent("keydown", { key: "Escape" });
  step("Escape closes the configuration window", modal.hidden === true, "");

  // 6. the scene editor paints its rows, now inside the window
  const rows = byId("scene-objects").children.length + byId("scene-containers").children.length;
  step("the scene editor paints its rows inside the window", rows >= 2, `${rows} row(s)`);

  // 7. the table map draws what the server built, and the summary says the same thing in words
  const map = byId("scene-map").innerHTML;
  step("the table map draws the scene it was given",
       /tray/.test(map) && /m-blue/.test(map) && /brazo/.test(map) && /Azul A/.test(map),
       `map holds ${map.length} characters`);
  step("the map states the arm's reach",
       /class="reach"/.test(map) && /330 mm/.test(map), "reach circle and radius");
  step("the summary states what is on the table",
       /1 cubo rojo en Azul A/.test(byId("scene-summary").textContent)
       && /1 pieza y 1 recipiente/.test(byId("scene-count").textContent),
       `${byId("scene-summary").textContent} (${byId("scene-count").textContent})`);
  // 8. every colour the server offers is offered as a swatch, in each row
  function findByClass(root, name) {
    for (const child of root.children || []) {
      if (child.className && String(child.className).split(" ").includes(name)) return child;
      const hit = findByClass(child, name);
      if (hit) return hit;
    }
    return null;
  }
  const swatchBox = findByClass(byId("scene-objects").children[0], "swatches");
  step("the colours are swatches, one per colour the server offers",
       !!swatchBox && swatchBox.children.length === 5,
       swatchBox ? `${swatchBox.children.length} swatch(es)` : "no swatch box");

  const failed = report.steps.filter(s => !s.ok);
  console.log(JSON.stringify({ ...report, ok: failed.length === 0 }));
}

main();
