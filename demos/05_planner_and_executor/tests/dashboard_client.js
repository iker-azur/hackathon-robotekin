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
//
// What is read out of the page is the shape the panel now has: the classification box as three
// fixed lines (what was understood, what was not sure, the doubt) instead of a grid of options,
// and the "ahora" line above the map. There is no grid left to count, so what is reported is the
// text of each line, which is what a person reads.

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

let htmlWrites = 0;
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
    htmlWrites += 1;
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
const paintStarted = process.hrtime.bigint();
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

// The classification box's option space, field by field.
//
// The box shows every option of every field the command model can choose between, and the whole
// point of it is that a field is never empty: a run with a sort command still shows the sixteen
// intents, and seeing "separar" next to "recoger" and "parar" is what makes the chosen one mean
// anything. So what is counted per row is the options offered, the one chosen, and how many of
// them exist on the table (the ones marked as realistic, with a dot).
const OPTION_ROWS = ["opt-intent", "opt-kind", "opt-color", "opt-count", "opt-place", "opt-sort"];
function optionsReport() {
  return Object.fromEntries(OPTION_ROWS.map(id => {
    const row = page.querySelector("#" + id);
    const html = row ? String(row.innerHTML) : "";
    return [id, {
      total: (html.match(/class="opt/g) || []).length,
      chosen: (html.match(/opt chosen"/g) || []).length,
      here: (html.match(/opt here"/g) || []).length,
      divs: (html.match(/<span class="opt/g) || []).length,
      score: (/<span class="p">(\d+)%</.exec(html) || [])[1] || "",
    }];
  }));
}

// The run's own state, captured BEFORE the stress batch is replayed over it. The stress batch
// deliberately replaces the classification and the plan with enormous strings, so anything read
// after it would answer a different question than the one being asked.
const RUN = {
  // What was understood, in the one element the classification box has for it: the generic
  // machinery writes the stage's live note there, and the classification's own sentence comes
  // after and wins.
  said: textOf("note-understand"),
  clsSure: textOf("cls-sure"),
  clsWarn: textOf("cls-warn"),
  // The option space, field by field: how many options each field offers, which one is the
  // chosen one, and how many exist on the table.
  options: optionsReport(),
  routeNow: textOf("route-now"),
  tasks: htmlOf("hero-tasks-list"),
  taskRows: tagCount("hero-tasks-list", "li"),
  command: textOf("cmd-text"),
  planHeader: textOf("note-plan"),
  planRows: tagCount("exec-steps", "tr"),
  stepColumns: (() => {
    // How many task columns the movement matrix has, read from its numbered header cells. The
    // matrix has no pose or millimetre cells on purpose: those are the session log's job.
    const table = page.querySelector("#exec-steps");
    const markup = table ? String(table.innerHTML) : "";
    return (markup.match(/<th>\d+<\/th>/g) || []).length;
  })(),
  tally: textOf("hero-tally"),
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
  paintMs: Number(process.hrtime.bigint() - paintStarted) / 1e6,
  htmlWrites,
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
  // The classification box, after the run and after everything.
  said: textOf("note-understand"),
  clsSure: textOf("cls-sure"),
  clsWarn: textOf("cls-warn"),
  options: optionsReport(),
  // The raw markup of each row and the scores it carries, so a test can ask "is the chosen one
  // among the offered ones" and "whose score is that" without re-deriving it.
  optionHtml: Object.fromEntries(OPTION_ROWS.map(id => [id, htmlOf(id)])),
  optionScores: Object.fromEntries(OPTION_ROWS.map(id => {
    const html = htmlOf(id);
    return [id, [...String(html).matchAll(/<span class="p">(\d+)%</g)].map(m => m[1])];
  })),
  // What is happening now, in the fixed line above the map.
  routeNow: textOf("route-now"),
  stepColumns: (() => {
    // How many columns the steps table has, from its own header row. The pose in millimetres
    // and the blob code went away on purpose, so this is one.
    const table = page.querySelector("#exec-steps");
    const markup = table ? String(table.innerHTML) : "";
    return (markup.match(/<th>\d+<\/th>/g) || []).length;
  })(),
  notes: Object.fromEntries(groups().map(g => [g.dataset.id, noteOf(g)])),
  logLines: page.querySelector("#log").children.length,
  tasks: htmlOf("hero-tasks-list"),
  taskRows: tagCount("hero-tasks-list", "li"),
  tally: textOf("hero-tally"),
  phase: textOf("phase"),
  verified: textOf("verified"),
  missed: textOf("missed"),
  resultCommand: textOf("result-command"),
  resultSession: textOf("result-session"),
  examples: (page.querySelector("#scene-examples") || {children:[]}).children.map(e => e.textContent),
  exampleCommands: (page.querySelector("#scene-examples") || {children:[]}).children.map(e => (e.dataset || {}).cmd),
  // What the presentation layer paints: the six-stage rail, the objective and tasks, and the
  // one-click demo status line. Read the way the page paints them - the rail from the elements
  // it built (className), the hero from the markup it wrote - so a test can check the
  // presentation is driven by real telemetry and never invents a state.
  present: (() => {
    const rail = page.querySelector("#stage-rail");
    const stages = (rail && rail.children ? rail.children : [])
      .filter(c => typeof c.className === "string" && c.className.split(" ").includes("stage"));
    const heroTasks = String(htmlOf("hero-tasks-list"));
    return {
      stageCount: stages.length,
      stageNames: stages.map(s => (s.children[0] && s.children[0].children[1] ? s.children[0].children[1].textContent : "")),
      stageClasses: stages.map(s => s.className),
      stageNotes: stages.map(s => (s.children[1] ? s.children[1].textContent : "")),
      heroObjective: textOf("hero-objective"),
      heroTaskRows: (heroTasks.match(/<li/g) || []).length,
      heroStates: {
        doing: (heroTasks.match(/class="doing"/g) || []).length,
        done: (heroTasks.match(/class="done"/g) || []).length,
        fail: (heroTasks.match(/class="fail"/g) || []).length,
        todo: (heroTasks.match(/class="todo"/g) || []).length,
      },
      heroTally: textOf("hero-tally"),
      heroRecovering: (heroTasks.match(/RECUPERANDO/g) || []).length,
      demoStatus: (() => { const b = page.querySelector("#demo-status"); return b ? b.className + " | " + b.textContent : ""; })(),
      demoMessage: typeof sandbox._demoMsg === "string" ? sandbox._demoMsg : "",
    };
  })(),
}));
