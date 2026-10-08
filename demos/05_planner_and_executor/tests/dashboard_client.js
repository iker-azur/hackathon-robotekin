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
// The file holds one snapshot list, or a list of them: a second batch replays into the SAME page
// without a reload, which is both cheaper than a second node process and a slightly better test -
// it proves a page keeps working when more state arrives after it has already painted.
const RAW = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const BATCHES = Array.isArray(RAW[0]) ? RAW : [RAW];
const SNAPSHOTS = BATCHES[0];
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
  // `innerHTML` here registers the ids inside, and ALSO builds the class attribute of the node
  // itself when the markup declares one. That is enough for the option grid, whose whole job is
  // to be told "this element has class `opts`" - the grid is CSS, and a real browser applies it
  // from the class. Without reading it, every grid looked empty to this shim.
  set innerHTML(html) {
    this.children = []; this._text = ""; this._html = String(html);
    // Each id becomes a child, carrying whatever class its own tag declares. The class matters:
    // an option grid IS a class, and a real browser gets it from the markup, so a shim that
    // dropped it would report every grid as unstyled and every layout check against it as failing
    // for a reason that does not exist in a browser.
    const declared = /<(\w+)([^>]*\bid="([^"]+)"[^>]*)>/g;
    for (const m of this._html.matchAll(declared)) {
      const node = registry(m[3]);
      const cls = /class="([^"]*)"/.exec(m[2]);
      if (cls) node.className = cls[1];
      this.append(node);
    }
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
  click() { if (this.onclick) return this.onclick(); }
  querySelector(sel) { return findById(this, sel.replace(/^#/, "")); }
}

// Ids the script creates at runtime, in markup it assigns itself. The strict check still applies
// to everything in the HTML: these have to be EXACTLY the ones the page's own markup declares,
// otherwise a real browser and this shim would disagree about what exists, and the disagreement
// is the bug. `buildFlow` writes `id="live-<node>"` into a div it builds, so that pattern is
// accepted for ids of a known flow node.
const FLOW_NODE_IDS = new Set([
  "setup", "input", "understand", "capture", "perceive", "link",
  "ground", "plan", "execute", "verify", "recover", "report",
]);

function isKnown(id) {
  if (PAGE_IDS.has(id)) return true;
  const runtime = /^live-([a-z]+)$/.exec(id);
  return !!(runtime && FLOW_NODE_IDS.has(runtime[1]));
}

function registry(id) {
  if (!isKnown(id)) throw new Error("la pagina no declara ningun id " + id);
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
    // The page listens for Escape on the whole document; the harness never sends keys.
    addEventListener() {},
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

const HTML = fs.readFileSync(process.argv[2], "utf8");
const script = HTML.slice(HTML.indexOf("<script>") + 8, HTML.lastIndexOf("</script>"));

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
  // Where the live note lives depends on the box, and both are checked here rather than assumed:
  // a box with content of its own keeps it in `#note-<id>`, and one without keeps it in
  // `#live-<id>`, inside the div that also holds the static detail. The title is the *last* <text>
  // in the group, so falling back to it would report "Cámara" as the note.
  const id = group.dataset.id;
  const owned = findById(group, `note-${id}`);
  if (owned && owned.textContent) return owned.textContent;
  const live = findById(group, `live-${id}`);
  if (live && live.textContent) return live.textContent;
  return "";
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
function layoutOf(id) {
  const g = flowSvg.children.find(c => c.tagName === "g" && c.dataset.id === id);
  return g && g.dataset.box ? JSON.parse(g.dataset.box) : null;
}
const planSteps = page.querySelector("#exec-steps");
const intentRow = page.querySelector("#row-intent");

function htmlOf(id) {
  const node = page.querySelector("#" + id);
  return node ? node.innerHTML : "";
}
function textOf(id) {
  const node = page.querySelector("#" + id);
  return node ? node.textContent : "";
}
function tagCount(id, tag) {
  const node = page.querySelector("#" + id);
  return node ? node.tags.filter(t => t === tag).length : 0;
}

// The run's own state, captured BEFORE the stress batch is replayed over it. The stress batch
// deliberately replaces the classification and the plan with enormous strings, so anything read
// after it would answer a different question than the one being asked.
const RUN = {
  intentHtml: htmlOf("row-intent"),
  clsOverall: textOf("cls-overall"),
  tasks: htmlOf("plan-tasks"),
  taskRows: tagCount("plan-tasks", "li"),
  command: textOf("cmd-text"),
  planHeader: textOf("note-plan"),
  planRows: tagCount("exec-steps", "tr"),
  tally: textOf("tally"),
  notes: Object.fromEntries(groups().map(g => [g.dataset.id, noteOf(g)])),
  logLines: page.querySelector("#log").children.length,
  phase: textOf("phase"),
};

// Any further batches, replayed into the page exactly as it stands - no reload. This is what the
// second batch in the file is for: it proves a page that has already painted keeps working when
// more state arrives, which is the situation every long run is in.
for (let b = 1; b < BATCHES.length; b++) {
  for (const [index, snap] of BATCHES[b].entries()) {
    try {
      sandbox.applySnapshot(snap);
    } catch (error) {
      throws.push(`batch ${b} snapshot ${index}: ${error.message}`);
    }
  }
  for (const group of groups()) {
    const stroke = group.children[0] && group.children[0].attrs.stroke;
    if (stroke && stroke !== IDLE_STROKE) lit.add(group.dataset.id);
  }
}

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
  // Where each box ended up, and whether the text inside it fits. A box's own right edge is
// x + w and its bottom is y + h, so anything a child needs more of than that is text that would
// have escaped - the failure this layout exists to prevent. Measured rather than assumed: the
// boxes publish their geometry, and the longest note in the run is checked against the narrowest
// box.
paintError: page.querySelector("#paint-error") ? page.querySelector("#paint-error").textContent : null,
  layout: Object.fromEntries(all.map(id => [id, layoutOf(id)])),
  // The gap the layout was built with, read off row A's own geometry rather than from a constant
  // of its own: the test's job is to check the drawn boxes, so it asks the drawn boxes.
  gutter: (() => {
    const row = ["setup", "input"].map(n => layoutOf(n)).filter(Boolean);
    return row.length === 2 ? Math.round((row[1].x - (row[0].x + row[0].w)) * 10) / 10 : null;
  })(),
  // `run` is the run's own content, read before the stress batch replaced it. `notes` and the
  // counts at the top level are the final state, after everything.
  run: RUN,
  clsOverall: textOf("cls-overall"),
  notes: Object.fromEntries(groups().map(g => [g.dataset.id, noteOf(g)])),
  logLines: page.querySelector("#log").children.length,
  planRows: tagCount("exec-steps", "tr"),
  tally: textOf("tally"),
  planHeader: textOf("note-plan"),
  command: textOf("cmd-text"),
  // La tabla de la intención: una celda por opción, con la elegida resaltada y su probabilidad.
  //
  // The row count comes from the page's own markup rather than from the shim's tree: `table.intent`
  // is drawn by assigning innerHTML to a <table>, and this shim's innerHTML only registers the ids
  // inside it, so the <tr>s never become children for anything here to count. The browser parses
  // them, and the markup is what it parses.
  intentRows: (() => {
    const markup = /<table class="intent">([\s\S]*?)<\/table>/.exec(HTML);
    return markup ? (markup[1].match(/<tr>/g) || []).length : 0;
  })(),
  // Cells per row, and how many are a div rather than a span. The second is the whole point of
  // this change: a matrix cell has to be a block with a width of its own, and a <span> laid out
  // inline is exactly what made the options read as a wrapped list before.
  cells: Object.fromEntries(["row-intent", "row-kind", "row-color", "row-count",
                             "row-place", "row-sort", "row-desk"].map(id => {
    const row = page.querySelector("#" + id);
    const html = row ? String(row.innerHTML) : "";
    return [id, {
      total: (html.match(/class="opt/g) || []).length,
      divs: (html.match(/<div class="opt/g) || []).length,
      spans: (html.match(/<span class="opt/g) || []).length,
      selected: (html.match(/class="opt sel"/g) || []).length,
    }];
  })),
  // the grid the cells sit in, per row
  grids: Object.fromEntries(["row-intent", "row-kind", "row-color", "row-count",
                             "row-place", "row-sort", "row-desk"].map(id => {
    const row = page.querySelector("#" + id);
    return [id, row ? row.className : null];
  })),
  intentHtml: intentRow ? intentRow.innerHTML : "",
  tasks: htmlOf("plan-tasks"),
  taskRows: tagCount("plan-tasks", "li"),
  phase: textOf("phase"),
  verified: textOf("verified"),
  missed: textOf("missed"),
  resultCommand: textOf("result-command"),
  resultSession: textOf("result-session"),
  examples: page.querySelector("#scene-examples").children.map(e => e.textContent),
}));
