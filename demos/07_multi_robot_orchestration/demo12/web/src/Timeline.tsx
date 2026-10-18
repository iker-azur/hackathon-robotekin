/**
 * The collaboration timeline.
 *
 * A multi-robot demo's interesting behaviour happens *between* the task rows: which robot
 * took which job, why that one, what a fault cost, how many attempts a pick needed before
 * it worked. None of that is visible in a table of current states, because by the time a
 * task is done the decisions that produced it are gone.
 *
 * This is that record, live, newest first, coloured by kind. It is a log of decisions the
 * orchestrator made -- not a visualisation of the physics -- so it can explain *why* the
 * cell is doing what it is doing without ever claiming something the simulation did not
 * support.
 */

import React from "react";
import type { EventRecord } from "./api";
import { t, num, type Lang } from "./i18n";

const KIND_COLOUR: Record<string, string> = {
  assign: "#3aa76d",
  reassign: "#d98c1f",
  release: "#94a3b8",
  phase: "#475569",
  plan: "#a855f7",
  fault: "#e0483c",
  repair: "#3aa76d",
  retry: "#d98c1f",
  done: "#22c55e",
  failed: "#e0483c",
  scene: "#0ea5e9",
  note: "#64748b",
};

const KIND_LABEL: Record<string, { es: string; en: string }> = {
  assign: { es: "asignado", en: "assigned" },
  reassign: { es: "reasignado", en: "reassigned" },
  release: { es: "liberado", en: "released" },
  phase: { es: "fase", en: "phase" },
  plan: { es: "plan", en: "plan" },
  fault: { es: "avería", en: "fault" },
  repair: { es: "reparado", en: "repaired" },
  retry: { es: "reintento", en: "retry" },
  done: { es: "completado", en: "completed" },
  failed: { es: "fallido", en: "failed" },
  scene: { es: "escena", en: "scene" },
  note: { es: "nota", en: "note" },
};

/** Phase events are the bulk of the volume and the least interesting on their own. */
const NOISY = new Set(["phase"]);

type Props = {
  lang: Lang;
  events: EventRecord[];
  counts: Record<string, number>;
  showPhases: boolean;
  onTogglePhases: (v: boolean) => void;
};

export default function Timeline({ lang, events, counts, showPhases, onTogglePhases }: Props) {
  const shown = showPhases ? events : events.filter((e) => !NOISY.has(e.kind));

  return (
    <div className="timeline">
      <div className="timeline-head">
        <div className="kinds">
          {Object.entries(counts)
            .filter(([k]) => k !== "phase" || showPhases)
            .sort((a, b) => b[1] - a[1])
            .map(([kind, n]) => (
              <span key={kind} className="kind-chip"
                    style={{ borderColor: KIND_COLOUR[kind] ?? "#64748b" }}>
                <i style={{ background: KIND_COLOUR[kind] ?? "#64748b" }} />
                {KIND_LABEL[kind]?.[lang] ?? kind} <b>{n}</b>
              </span>
            ))}
        </div>
        <label className="chk">
          <input type="checkbox" checked={showPhases}
                 onChange={(e) => onTogglePhases(e.target.checked)} />
          {lang === "es" ? "fases" : "phases"}
        </label>
      </div>

      <ol className="events">
        {shown.length === 0 && (
          <li className="empty">{lang === "es" ? "Sin actividad todavía" : "No activity yet"}</li>
        )}
        {[...shown].reverse().map((e) => {
          const cost = e.data?.cost as
            | { reachM: number; carryM: number; total: number } | undefined;
          return (
            <li key={e.seq} className={`ev ev-${e.kind}`}>
              <span className="ev-t">t={num(lang, e.simTimeS, 1)}s</span>
              <span className="ev-kind"
                    style={{ color: KIND_COLOUR[e.kind] ?? "#64748b" }}>
                {KIND_LABEL[e.kind]?.[lang] ?? e.kind}
              </span>
              {e.robotId && <span className="ev-rob">{e.robotId}</span>}
              {e.taskId && <span className="ev-task">{e.taskId}</span>}
              <span className="ev-detail">{e.detail}</span>
              {cost && (
                <span className="ev-cost" title="reach + 0.6 x carry + 0.15 x retries">
                  {num(lang, cost.total, 3)} m
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

export { KIND_COLOUR, KIND_LABEL };