/**
 * Demo 12 -- the shell.
 *
 * Reads the WebSocket for state and polls `/api/events` for the collaboration log. Nothing
 * here computes a verdict: no task status, no progress, no "done". Every value on screen
 * came from the server, which read it from PhysX. A component that formed its own opinion
 * about whether a pick worked would be the exact thing this demo was rebuilt to stop doing,
 * so the rule holds all the way to the top.
 *
 * Four views, because the four questions are different:
 *
 *   Cell          what is happening right now, and where everything is
 *   Plan          what I want to happen -- assign, type a command, apply
 *   Collaboration who is doing what, why, and what it cost
 *   Editor        the cell's geometry, which costs a restart and says so
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, API, WS, type EventRecord, type Lang, type Snapshot } from "./api";
import { t, num } from "./i18n";
import CellMap from "./CellMap";
import Timeline from "./Timeline";
import Evidence, { CRITERION_LABEL_KEYS } from "./Evidence";
import Planner from "./Planner";
import SceneEditor from "./SceneEditor";

type Tab = "cell" | "plan" | "collab" | "editor";

const TAB_KEY: Record<Tab, string> = {
  cell: "tabCell", plan: "tabPlan", collab: "tabCollab", editor: "tabEditor",
};

function Frame() {
  const [nonce, setNonce] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const id = window.setInterval(() => setNonce((n) => n + 1), 700);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="frame">
      {error ? (
        <div className="frame-error">
          <strong>Sin imagen / No image</strong>
          <code>{error}</code>
        </div>
      ) : (
        <img
          src={`${API}/api/frame.jpg?t=${nonce}`}
          alt="Isaac Sim"
          onError={async () => {
            try {
              const body = await (await fetch(`${API}/api/state`)).json();
              setError(body.frameError ?? "the server returned no image");
            } catch {
              setError("no server at 127.0.0.1:8000");
            }
          }}
        />
      )}
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState<Lang>(() =>
    (localStorage.getItem("demo12-lang") as Lang) || "en");
  const [state, setState] = useState<Snapshot | null>(null);
  const [socketUp, setSocketUp] = useState<"connecting" | "up" | "down">("connecting");
  const [tab, setTab] = useState<Tab>("cell");
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [showPhases, setShowPhases] = useState(false);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [selectedPart, setSelectedPart] = useState<string | null>(null);
  const [hoveredBin, setHoveredBin] = useState<string | null>(null);
  const [plan, setPlan] = useState<Map<string, string>>(new Map());
  const cursor = useRef(0);

  useEffect(() => {
    localStorage.setItem("demo12-lang", lang);
    document.documentElement.lang = lang;
  }, [lang]);

  const tr = useCallback((k: string) => t(lang, k), [lang]);

  // -- live state ---------------------------------------------------------
  useEffect(() => {
    const ws = new WebSocket(WS);
    ws.onopen = () => setSocketUp("up");
    ws.onclose = () => setSocketUp("down");
    ws.onerror = () => setSocketUp("down");
    ws.onmessage = (e) => setState(JSON.parse(e.data));
    return () => ws.close();
  }, []);

  // Events are polled on a cursor rather than pushed whole, so a long demo does not
  // re-transfer the whole log twice a second.
  useEffect(() => {
    let alive = true;
    const pull = async () => {
      try {
        const res = await api.events(cursor.current);
        if (!alive) return;
        cursor.current = res.cursor;
        setEvents((prev) => {
          const merged = [...prev, ...res.events];
          const seen = new Set<number>();
          return merged.filter((e) => !seen.has(e.seq) && seen.add(e.seq)).slice(-600);
        });
      } catch {
        /* the server is down; the socket indicator already says so */
      }
    };
    void pull();
    const id = window.setInterval(pull, 1000);
    return () => { alive = false; window.clearInterval(id); };
  }, []);

  const tasks = state?.tasks ?? [];
  const robots = state?.robots ?? [];
  const scene = state?.scene ?? null;
  const metrics = state?.metrics ?? null;

  // The plan starts from what the cell is already being asked to do, so the panel reflects
  // reality until someone edits it.
  useEffect(() => {
    setPlan(new Map(tasks.map((x) => [x.partId, x.binId])));
  }, [tasks.length]);

  const onApplied = useCallback(() => {
    api.state().then(setState).catch(() => undefined);
  }, []);

  const pickPart = useCallback((id: string) => {
    setSelectedPart((cur) => (cur === id ? null : id));
  }, []);

  const pickBin = useCallback((binId: string) => {
    setSelectedPart((cur) => {
      if (!cur) return cur;
      setPlan((p) => new Map(p).set(cur, binId));
      return null;
    });
    setTab("plan");
  }, []);

  const focus = useMemo(
    () => tasks.find((x) => x.id === selectedTask) ?? tasks[0] ?? null,
    [tasks, selectedTask],
  );

  const progress = metrics && metrics.tasksTotal > 0
    ? (metrics.tasksCompleted / metrics.tasksTotal) * 100 : 0;

  return (
    <div className={`app lang-${lang}`}>
      <header className="head">
        <div className="brand">
          <h1>{tr("appTitle")}</h1>
          <span className="sub">{tr("appSubtitle")}</span>
        </div>

        <nav className="tabs">
          {(["cell", "plan", "collab", "editor"] as Tab[]).map((k) => (
            <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
              {tr(TAB_KEY[k])}
            </button>
          ))}
        </nav>

        <div className="head-right">
          <button className="lang-toggle"
                  onClick={() => setLang(lang === "es" ? "en" : "es")}>
            {lang === "es" ? "ES" : "EN"}
          </button>
          <span className={`conn ${socketUp}`}>
            {socketUp === "up" ? "●" : socketUp === "connecting" ? "◌" : "○"}{" "}
            {tr(socketUp === "up" ? "connected"
               : socketUp === "connecting" ? "connecting" : "disconnected")}
          </span>
        </div>
      </header>

      {state?.lastError && (
        <div className="banner bad">
          <strong>{tr("error")}</strong> ({state.errorCount}×): {state.lastError}
        </div>
      )}

      {/* ---------------------------------------------------------- CELL */}
      {tab === "cell" && (
        <div className="grid-cell">
          <div className="col-main">
            <CellMap
              lang={lang}
              scene={scene}
              robots={robots}
              tasks={tasks}
              parts={state?.parts ?? []}
              bins={scene?.bins ?? state?.bins ?? []}
              selectedPart={selectedPart}
              onPickPart={pickPart}
              onPickBin={pickBin}
              hoveredBin={hoveredBin}
            />
            <p className="hint centre">
              {selectedPart
                ? `${tr("clickForTarget")} (${selectedPart})`
                : tr("plannerHint")}
            </p>
          </div>

          <div className="col-side">
            <Frame />

            <section className="card">
              <h2>{tr("activity")}</h2>
              <div className="bar">
                <div className="bar-fill"
                     style={{ width: `${progress}%`,
                              background: metrics?.tasksFailed ? "#d98c1f" : "#3aa76d" }} />
              </div>
              <dl className="kv">
                <Row k={tr("simTime")} v={`${num(lang, state?.simTimeS ?? 0, 1)} s`} />
                <Row k={tr("controlTicks")} v={num(lang, state?.controlTicks ?? 0, 0)} />
                <Row k={tr("completed")} v={`${metrics?.tasksCompleted ?? 0} / ${metrics?.tasksTotal ?? 0}`} />
              </dl>
              <div className="robots">
                {robots.map((r) => {
                  const task = tasks.find((x) => x.id === r.taskId);
                  const pct = Math.round(r.utilisation * 100);
                  return (
                    <div key={r.id} className={`robot st-${r.state}`}>
                      <div className="robot-head">
                        <strong>{r.id}</strong>
                        <span className={`pill ${r.state}`}>
                          {tr(r.state === "idle" ? "free" : r.state === "working" ? "busy" : "faulted")}
                        </span>
                        <span className="dim">{r.taskId ?? ""}</span>
                        <span className="spacer" />
                        {r.state !== "faulted" ? (
                          <button className="tiny" onClick={() => api.fault(r.id).then(onApplied)}>
                            {tr("fault")}
                          </button>
                        ) : (
                          <button className="tiny" onClick={() => api.repair(r.id).then(onApplied)}>
                            {tr("repair")}
                          </button>
                        )}
                      </div>
                      <div className="robot-bar">
                        <div className="robot-fill"
                             style={{ width: `${pct}%`,
                                      background: r.state === "faulted" ? "#e0483c"
                                        : r.state === "working" ? "#3aa76d" : "#475569" }} />
                        <span>{pct}%</span>
                      </div>
                      <div className="robot-meta dim">
                        {task?.progress && (
                          <span>
                            {tr(task.progress.phase)} ·{" "}
                            {Math.round(task.progress.fraction * 100)}%
                          </span>
                        )}
                        <span>{num(lang, r.gripN, 1)} N</span>
                        <span>{num(lang, r.spanM * 1000, 1)} mm</span>
                        <span>{r.tasksCompleted} {tr("tasksDone")}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="row">
                <button onClick={() => api.faults().then(onApplied)}>{tr("faultAll")}</button>
                <button onClick={() => {
                  robots.filter((r) => r.state === "faulted")
                    .forEach((r) => void api.repair(r.id));
                  onApplied();
                }}>{tr("repair")} all</button>
              </div>
            </section>

            <section className="card">
              <h2>{tr("task")}</h2>
              <div className="task-list">
                {tasks.map((task) => (
                  <button key={task.id}
                          className={`task st-${task.state} ${
                            focus?.id === task.id ? "on" : ""}`}
                          onClick={() => setSelectedTask(task.id)}>
                    <span className="t-id">{task.partId} → {task.binId}</span>
                    <span className={`pill ${task.state}`}>{tr(task.state)}</span>
                    <span className="dim">
                      {task.assignedRobot ?? "—"}
                      {task.attempts > 1 ? ` · ×${task.attempts}` : ""}
                    </span>
                    {task.progress && task.state !== "done" && task.state !== "failed" && (
                      <span className="t-prog">
                        <i style={{ width: `${task.progress.fraction * 100}%` }} />
                      </span>
                    )}
                  </button>
                ))}
                {tasks.length === 0 && <p className="hint">{tr("noMoves")}</p>}
              </div>
              {focus && (
                <>
                  <h3>{tr("evidence")}</h3>
                  <p className="hint small">{tr("evidenceHint")}</p>
                  <Evidence lang={lang} evidence={focus.evidence} note={focus.note} />
                </>
              )}
            </section>
          </div>
        </div>
      )}

      {/* ---------------------------------------------------------- PLAN */}
      {tab === "plan" && (
        <div className="two-col">
          <Planner lang={lang} scene={scene} tasks={tasks}
                    running={state?.running ?? false}
                    plan={plan} setPlan={setPlan} onApplied={onApplied} />
          <div>
            <CellMap lang={lang} scene={scene} robots={robots} tasks={tasks}
                     parts={state?.parts ?? []} bins={scene?.bins ?? state?.bins ?? []}
                     selectedPart={selectedPart}
                     onPickPart={pickPart}
                     onPickBin={pickBin}
                     hoveredBin={hoveredBin} />
          </div>
        </div>
      )}

      {/* ------------------------------------------------------ COLLAB */}
      {tab === "collab" && (
        <div className="two-col">
          <section className="card">
            <h2>{tr("collabTitle")}</h2>
            <p className="hint">{tr("collabHint")}</p>
            <Timeline lang={lang} events={events} counts={state?.eventCounts ?? {}}
                      showPhases={showPhases} onTogglePhases={setShowPhases} />
          </section>

          <div>
            {metrics && (
              <section className="card">
                <h2>{tr("metrics")}</h2>
                <div className="metrics">
                  <Metric label={tr("graspSuccess")}
                          value={`${Math.round(metrics.graspSuccessRate * 100)}%`}
                          sub={`${metrics.tasksCompleted}/${metrics.attempts} ${tr("attempts").toLowerCase()}`} />
                  <Metric label={tr("tasksPerMin")}
                          value={num(lang, metrics.tasksPerSimMinute, 2)} />
                  <Metric label={tr("reassignments")} value={String(metrics.reassignments)} />
                  <Metric label={tr("injectedFaults")} value={String(metrics.failuresInjected)} />
                  <Metric label={tr("meanRise")}
                          value={`${num(lang, metrics.meanPartRiseMm, 1)} mm`} />
                  <Metric label={tr("meanGrip")}
                          value={`${num(lang, metrics.meanGripN, 1)} N`} />
                  <Metric label={tr("meanError")}
                          value={`${num(lang, metrics.meanToolErrorMm, 2)} mm`} />
                  <Metric label={tr("load")}
                          value={`${metrics.tasksCompleted}/${metrics.tasksTotal}`} />
                </div>

                {Object.keys(metrics.criteriaPassed).length > 0 && (
                  <>
                    <h3>{tr("evidence")}</h3>
                    <table className="crit-table">
                      <thead>
                        <tr><th /><th>✓</th><th>✗</th></tr>
                      </thead>
                      <tbody>
                        {Object.entries(metrics.criteriaPassed).map(([k, v]) => (
                          <tr key={k}>
                            <td>{tr(CRITERION_LABEL_KEYS[k] ?? k)}</td>
                            <td className="ok">{v.passed}</td>
                            <td className={v.failed ? "bad" : "dim"}>{v.failed}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}
              </section>
            )}

            <section className="card">
              <h2>{tr("load")}</h2>
              {robots.map((r) => {
                const busy = r.busyTicks;
                const idle = r.idleTicks;
                const total = Math.max(1, busy + idle);
                return (
                  <div key={r.id} className="load-row">
                    <span className="lbl">{r.id}</span>
                    <div className="stacked">
                      <div className="s-busy" style={{ width: `${(busy / total) * 100}%` }} />
                      <div className="s-idle" style={{ width: `${(idle / total) * 100}%` }} />
                    </div>
                    <span className="dim">
                      {Math.round((busy / total) * 100)}% · {r.tasksCompleted} {tr("tasksDone")}
                    </span>
                  </div>
                );
              })}
              <div className="legend small">
                <span><i className="s-busy" /> {tr("busy")}</span>
                <span><i className="s-idle" /> {tr("idle")}</span>
              </div>
            </section>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------ EDITOR */}
      {tab === "editor" && (
        <div className="two-col">
          <section className="card">
            <h2>{tr("editorTitle")}</h2>
            <SceneEditor lang={lang} scene={scene} onApplied={onApplied} />
          </section>
          <div>
            <CellMap lang={lang} scene={scene} robots={robots} tasks={tasks}
                     parts={state?.parts ?? []} bins={scene?.bins ?? state?.bins ?? []}
                     selectedPart={null}
                     onPickPart={() => undefined}
                     onPickBin={() => undefined}
                     hoveredBin={null} />
          </div>
        </div>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="kv-row">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="metric">
      <span className="m-label">{label}</span>
      <span className="m-value">{value}</span>
      {sub && <span className="m-sub dim">{sub}</span>}
    </div>
  );
}