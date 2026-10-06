// Run the dashboard's OWN script against real bus snapshots, and report what it painted.
//
//     node tests/dashboard_client.js <snapshots.json>
//
// Prints one JSON object on stdout. Driven by tests/test_demo11.py, which builds the snapshots
// from telemetry.Bus so the data here is the data the robot really publishes.
//
// Why this exists, and why it is a node script and not a Python one: the bug it exists for was
// in the browser's own code, and nothing in Python can see it. ``paintNodes`` wrote the live
// note of every box into ``painted.live``, but ``buildFlow`` only creates that element for the
// boxes with no content of their own - ``input``, ``understand`` and ``plan`` have none - so the
// write hit null on the SECOND box of twelve, every single time. Measured: a TypeError, two of
// twelve boxes repainted, ten frozen, and the throw happened before ``paintState`` and the log
// so nothing else on the page moved either. It looked like a live diagram that had stopped.
//
// A second one lived next to it: ``paintPlan`` wrote to ``#plan-n``, an id the page never
// declared, so the whole plan card threw before it filled the steps table, the note or the list
// of parts - silently, because ``paintState`` swallowed it in a bare catch.
//
// So the page is loaded here for real, into a DOM small enough to run in, and fed a whole run's
// snapshots. What it painted is then counted and returned. The DOM shim is deliberately honest
// about its one hard part: ``innerHTML`` does not parse, but it does register the ids inside it,
// because looking up ``#note-<id>`` after assigning markup is exactly what the real page does.

"use strict";

const fs = require("fs");
const vm = require("vm");

const PAGE = process.argv[2];
const SNAPSHOTS = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
// The idle stroke, read from the page rather than guessed: COLORS.idle is the colour
// buildFlow paints every box with, so it is the only value "this box has not been visited yet"
// can be compared against. (Comparing against the idle *fill* instead would mark all twelve
// boxes as visited before the run started.)
const IDLE_STROKE = process.argv[4] || "#4a5164";

// ---- the DOM ----------------------------------------------------------------------------

// Every id the page declares, so the shim only hands out elements that really exist: a typo in
// the script then fails here, which is the point.
const PAGE_IDS = new Set([...fs.readFileSync(PAGE, "utf8").matchAll(/id="([^"]+)"/g)]
  .map(m => m[1]));

class El {
  constructor(tag, ns) {
    this.tagName = tag; this.ns = ns || ""; this.attrs = {}; this.children = [];
    this._text = ""; this._html = ""; this.dataset = {}; this.style = {};
    this.classList = {
      _s: new Set(),
      add: (...c) => c.forEach(x => this.classList._s.add(x)),
      remove: (...c) => c.forEach(x => this.classList._s.delete(x)),
      toggle: (c, on) => (on ? this.classList._s.add(c) : this.classList._s.delete(c)),
      contains: c => this.classList._s.has(c),
    };
    this.className = "";
  }
  set textContent(v) { this._text = String(v); this.children = []; this._html = ""; }
  get textContent() { return this._text || this.children.map(c => c.textContent).join(""); }
  set innerHTML(html) {
    this.children = []; this._text = ""; this._html = String(html);
    for (const m of this._html.matchAll(/id="([^"]+)"/g)) this.append(registry(m[1]));
  }
  get innerHTML() { return this._html; }
  get tags() {
    return [...String(this._html).matchAll(/<([a-zA-Z][a-zA-Z0-9]*)/g)].map(m => m[1]);
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  append(...kids) { kids.forEach(k => { this.children.push(k); k.parent = this; }); }
  appendChild(k) { this.append(k); return k; }
  addEventListener() {}
  querySelector(sel) { return findById(this, sel.replace(/^#/, "")); }
}

function registry(id) {
  if (!PAGE_IDS.has(id)) throw new Error("la pagina no declara ningun id " + id);
  const node = new El("div");
  node.attrs.id = id;
  return node;
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

const sandbox = {
  console: { log() {}, warn() {}, error() {} },
  document: {
    body: page,
    createElementNS: (ns, tag) => new El(tag, ns),
    createElement: tag => new El(tag),
    querySelector: sel => page.querySelector(sel),
    querySelectorAll: () => [],
    getElementById: id => page.querySelector("#" + id),
  },
  location: { protocol: "http:", host: "127.0.0.1:8765" },
  window: {},
  globalThis: null,
  performance: { now: () => 0 },
  URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
  Blob: class {},
  WebSocket: class { constructor() { this.readyState = 0; } close() {} },
  // Nothing here reaches the network: the transport is under test in test_demo11.py, over a
  // real socket. This only needs the page to load.
  fetch: () => new Promise(() => {}),
  setInterval: () => 0, clearInterval() {},
  setTimeout: () => 0, clearTimeout() {},
  navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [] }) } },
  AudioContext: class { constructor() { this.sampleRate = 48000; } close() {} },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

// ---- load the page's own script ---------------------------------------------------------

const html = fs.readFileSync(PAGE, "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));

let loadError = null;
const warnings = [];
try {
  vm.createContext(sandbox);
  sandbox.console.warn = (...a) => warnings.push(a.join(" "));
  sandbox.console.error = (...a) => warnings.push("ERROR " + a.join(" "));
  vm.runInContext(script, sandbox, { filename: "index.html" });
} catch (error) {
  loadError = error.constructor.name + ": " + error.message;
}

// ---- feed it the run --------------------------------------------------------------------

const flowSvg = page.querySelector("#flow-svg");

function groups() {
  return flowSvg.children.filter(g => g.tagName === "g" && g.dataset.id);
}
function noteOf(group) {
  // A box with content of its own keeps the note inside that content, in `#note-<id>`; a box
  // without one has it as a plain <text> after the detail line. Both are where paintNodes writes.
  const host = findById(group, "note-" + group.dataset.id);
  if (host) return host.textContent;
  const texts = group.children.filter(c => c.tagName === "text");
  return texts.length ? texts[texts.length - 1].textContent : "";
}

const throws = [];
const lit = new Set();
const withNote = new Set();
for (const [index, snap] of SNAPSHOTS.entries()) {
  try {
    sandbox.applySnapshot(snap);
  } catch (error) {
    throws.push(`snapshot ${index}: ${error.message}`);
  }
  for (const group of groups()) {
    const stroke = group.children[0] && group.children[0].attrs.stroke;
    if (stroke && stroke !== IDLE_STROKE) lit.add(group.dataset.id);
    const note = noteOf(group);
    // the title is a static string from telemetry.Node.title; a note is whatever the robot said
    if (note && note !== group.children[2].textContent) withNote.add(group.dataset.id);
  }
}

const all = SNAPSHOTS[0].nodes.map(n => n.id);
const planSteps = page.querySelector("#plan-steps");
const intentRow = page.querySelector("#row-intent");

process.stdout.write(JSON.stringify({
  loadError,
  throws,
  warnings,
  nodes: all.length,
  boxes: groups().length,
  lit: [...lit].sort(),
  withNote: [...withNote].sort(),
  dark: all.filter(id => !lit.has(id)),
  noNote: all.filter(id => !withNote.has(id)),
  colours: Object.fromEntries(groups().map(g => [g.dataset.id, g.children[0].attrs.stroke])),
  notes: Object.fromEntries(groups().map(g => [g.dataset.id, noteOf(g)])),
  paintError: page.querySelector("#paint-error") ? page.querySelector("#paint-error").textContent : null,
  logLines: page.querySelector("#log").children.length,
  planRows: planSteps ? planSteps.tags.filter(t => t === "tr").length : 0,
  planVersion: page.querySelector("#plan-n") ? page.querySelector("#plan-n").textContent : null,
  planParts: page.querySelector("#plan-cmd") ? page.querySelector("#plan-cmd").textContent : null,
  command: page.querySelector("#cmd-text") ? page.querySelector("#cmd-text").textContent : null,
  intentChips: intentRow ? intentRow.tags.length : 0,
  phase: page.querySelector("#phase").textContent,
  verified: page.querySelector("#verified").textContent,
  missed: page.querySelector("#missed").textContent,
}));