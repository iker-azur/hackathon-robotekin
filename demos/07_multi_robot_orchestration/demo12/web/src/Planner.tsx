/**
 * Choosing what should happen.
 *
 * Three ways in, because people arrive at the same question differently:
 *
 *  * **Click a part, click a tray.** The direct route, and the one the cell map is built
 *    for -- you see the piece and you see the tray.
 *  * **Type a sentence.** "mueve el cubo rojo a la caja azul". Interpreted, previewed, and
 *    only applied on a second click, because someone typing into a box wants to see what
 *    was understood before the cell starts moving.
 *  * **Pick a scenario.** One click to a known-good assignment, which is how a demo gets
 *    from "start it" to "something interesting" in two seconds.
 *
 * The pending plan is held here and never auto-applied. Every path ends at the same Apply
 * button, so what runs is always what the person last looked at.
 */

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Bin, type Part, type Scene, type Task } from "./api";
import { t, num, type Lang } from "./i18n";

type Move = { partId: string; binId: string };

type Props = {
  lang: Lang;
  scene: Scene | null;
  tasks: Task[];
  running: boolean;
  plan: Map<string, string>;
  setPlan: (p: Map<string, string>) => void;
  onApplied: () => void;
};

export default function Planner({
  lang, scene, tasks, running, plan, setPlan, onApplied,
}: Props) {
  const [command, setCommand] = useState("");
  const [parsed, setParsed] = useState<{ moves: Move[]; errors: string[]; understood: string[] } | null>(null);
  const [scenarios, setScenarios] = useState<Record<string, { moves: Move[]; es: string; en: string; errors: string[] }>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    api.scenarios().then(setScenarios).catch(() => setScenarios({}));
  }, []);

  const moves = useMemo<Move[]>(
    () => [...plan.entries()].map(([partId, binId]) => ({ partId, binId })),
    [plan],
  );

  const submit = useCallback(async (list: Move[]) => {
    if (list.length === 0) {
      setMessage({ ok: false, text: lang === "es" ? "El plan está vacío" : "The plan is empty" });
      return;
    }
    setBusy(true);
    try {
      const res = await api.submitTasks(list);
      setMessage(
        res.ok
          ? { ok: true, text: lang === "es"
                ? `${res.tasksSubmitted?.length ?? list.length} tarea(s) en cola`
                : `${res.tasksSubmitted?.length ?? list.length} task(s) queued` }
          : { ok: false, text: (res.errors ?? []).join("; ") },
      );
      onApplied();
    } catch (err) {
      setMessage({ ok: false, text: String(err) });
    } finally {
      setBusy(false);
    }
  }, [lang, onApplied]);

  const runCommand = useCallback(async (apply: boolean) => {
    if (!command.trim()) return;
    setBusy(true);
    try {
      const res = await api.command(command, apply);
      setParsed({ moves: res.moves, errors: res.errors, understood: res.understood });
      if (res.moves.length > 0) {
        setPlan(new Map(res.moves.map((m) => [m.partId, m.binId])));
        if (apply) {
          setMessage({ ok: true, text: lang === "es" ? "Plan aplicado" : "Plan applied" });
          onApplied();
        }
      }
    } catch (err) {
      setMessage({ ok: false, text: String(err) });
    } finally {
      setBusy(false);
    }
  }, [command, lang, onApplied, setPlan]);

  const parts = scene?.parts ?? [];
  const bins = scene?.bins ?? [];

  return (
    <div className="planner">
      <p className="hint">{t(lang, "plannerHint")}</p>

      {/* ---- the queue, as a list you can read at a glance ------------- */}
      <table className="plan-table">
        <thead>
          <tr>
            <th>{t(lang, "piece")}</th>
            <th>{t(lang, "destination")}</th>
            <th>{t(lang, "state")}</th>
            <th>{t(lang, "robot")}</th>
          </tr>
        </thead>
        <tbody>
          {parts.map((p) => {
            const binId = plan.get(p.id);
            const task = tasks.find((x) => x.partId === p.id);
            const live = task && task.state !== "done" && task.state !== "failed";
            return (
              <tr key={p.id} className={binId ? "assigned" : ""}>
                <td>
                  <span className="chip">
                    <i style={{ background: p.color }} />
                    {p.id.replace("part_", "P")}
                  </span>
                  <span className="dim">
                    {" "}{p.kind} {num(lang, p.sizeM[0] * 1000, 0)} mm
                  </span>
                </td>
                <td>
                  {binId ? (
                    <select
                      value={binId}
                      disabled={live}
                      onChange={(e) => {
                        const next = new Map(plan);
                        next.set(p.id, e.target.value);
                        setPlan(next);
                      }}
                    >
                      {bins.map((b) => (
                        <option key={b.id} value={b.id}>{b.label}</option>
                      ))}
                    </select>
                  ) : (
                    <span className="dim">{t(lang, "unassigned")}</span>
                  )}
                </td>
                <td>
                  {task
                    ? <span className={`st st-${task.state}`}>{t(lang, task.state)}</span>
                    : <span className="dim">—</span>}
                </td>
                <td className="dim">{task?.assignedRobot ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="row">
        <button className="primary" disabled={busy || moves.length === 0}
                onClick={() => submit(moves)}>
          {t(lang, "applyPlan")} ({moves.length})
        </button>
        <button disabled={busy || plan.size === 0}
                onClick={() => setPlan(new Map())}>
          {t(lang, "clearPlan")}
        </button>
        <button disabled={busy} onClick={() => api.clearTasks().then(onApplied)}>
          {lang === "es" ? "Vaciar cola" : "Clear queue"}
        </button>
        {running ? (
          <button onClick={() => api.stop().then(onApplied)}>{t(lang, "stop")}</button>
        ) : (
          <button onClick={() => api.run().then(onApplied)}>{t(lang, "run")}</button>
        )}
      </div>

      {message && (
        <p className={`msg ${message.ok ? "ok" : "bad"}`}>{message.text}</p>
      )}

      {/* ---- scenarios ------------------------------------------------- */}
      <div className="scenarios">
        <h3>{t(lang, "scenarios")}</h3>
        <div className="row">
          {Object.entries(scenarios).map(([key, s]) => (
            <button key={key} disabled={busy}
                    onClick={() => setPlan(new Map(s.moves.map((m) => [m.partId, m.binId])))}>
              {t(lang, key === "color" ? "scenarioColor"
                  : key === "mixed" ? "scenarioMixed" : "scenarioBalanced")}
            </button>
          ))}
        </div>
        {Object.entries(scenarios).map(([key, s]) => (
          <p key={`d-${key}`} className="dim small">{s[lang]}</p>
        ))}
      </div>

      {/* ---- command bar ------------------------------------------------ */}
      <div className="command">
        <h3>{t(lang, "commandLabel")}</h3>
        <div className="row">
          <input
            value={command}
            placeholder={t(lang, "commandPlaceholder")}
            onChange={(e) => setCommand(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") runCommand(false); }}
          />
          <button disabled={busy || !command.trim()} onClick={() => runCommand(false)}>
            {t(lang, "commandRun")}
          </button>
          <button className="primary" disabled={busy || !command.trim()}
                  onClick={() => runCommand(true)}>
            {t(lang, "commandApply")}
          </button>
        </div>

        {parsed && (
          <div className={`parse ${parsed.errors.length ? "bad" : "ok"}`}>
            {parsed.understood.length > 0 && (
              <div>
                <b>{t(lang, "understood")}:</b>{" "}
                {parsed.understood.join(" · ")}
              </div>
            )}
            {parsed.moves.length > 0 && (
              <div className="moves">
                {parsed.moves.map((m, i) => (
                  <span key={i} className="chip">
                    {m.partId} → {m.binId}
                  </span>
                ))}
              </div>
            )}
            {parsed.errors.map((e, i) => <div key={i}>✗ {e}</div>)}
          </div>
        )}
      </div>
    </div>
  );
}