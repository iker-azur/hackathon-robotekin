/**
 * A top-down map of the cell, drawn from the simulation's own state.
 *
 * **Why this exists.** The camera view shows three arms doing three things at once, which
 * is exactly the kind of thing that is hard to follow. This view answers the questions the
 * camera cannot: which robot is holding which part, where it is going, who is waiting for
 * what, and how far apart the arms are. It is the difference between watching a cell and
 * understanding one.
 *
 * Everything drawn here is a measurement or a decision the orchestrator already made --
 * part positions come from the rigid bodies, reach annuli from the measured envelope, task
 * arrows from the task list. Nothing is interpolated or animated for effect. A part that has
 * not moved is not drawn moving, and an idle robot is drawn idle even while its neighbours
 * are busy.
 */

import React from "react";
import type { Bin, LivePart, Robot, Scene, Task } from "./api";
import { t, num, type Lang } from "./i18n";

const W = 760;
const H = 470;
const PAD = 34;

/** Table metres -> SVG units. Fitted to the configured table, not hard-coded. */
function makeScale(table: { sizeM: number[]; centerXy: number[] }) {
  const [tw, th] = table.sizeM;
  const [cx, cy] = table.centerXy;
  const sx = (W - PAD * 2) / tw;
  const sy = (H - PAD * 2) / th;
  const s = Math.min(sx, sy);
  const offX = (W - tw * s) / 2;
  const offY = (H - th * s) / 2;
  return {
    s,
    X: (x: number) => offX + (x - (cx - tw / 2)) * s,
    // World +y is "away from the viewer", so it is drawn upward on screen.
    Y: (y: number) => offY + ((cy + th / 2) - y) * s,
    invX: (px: number) => (px - offX) / s + (cx - tw / 2),
    invY: (py: number) => (cy + th / 2) - (py - offY) / s,
    cx, cy, tw, th,
  };
}

const STATE_COLOUR: Record<string, string> = {
  queued: "#94a3b8",
  approaching: "#3d7fd6",
  closing: "#d98c1f",
  lifting: "#d98c1f",
  transferring: "#d98c1f",
  releasing: "#a855f7",
  verifying: "#a855f7",
  done: "#3aa76d",
  failed: "#e0483c",
};

function partColour(scene: Scene | null, id: string, fallback: string): string {
  return scene?.parts.find((p) => p.id === id)?.color ?? fallback;
}

function PartShape({ kind, colour }: { kind: string; colour: string }) {
  const common = { fill: colour, stroke: "rgba(0,0,0,.45)", strokeWidth: 1 };
  if (kind === "cylinder") {
    return <ellipse rx={7} ry={4.5} {...common} />;
  }
  if (kind === "ball") {
    return <circle r={6.5} {...common} />;
  }
  return <rect x={-6.5} y={-6.5} width={13} height={13} rx={2} {...common} />;
}

type Props = {
  lang: Lang;
  scene: Scene | null;
  robots: Robot[];
  tasks: Task[];
  parts: LivePart[];
  bins: Bin[];
  selectedPart: string | null;
  onPickPart: (id: string) => void;
  onPickBin: (id: string) => void;
  hoveredBin: string | null;
};

export default function CellMap({
  lang, scene, robots, tasks, parts, bins,
  selectedPart, onPickPart, onPickBin, hoveredBin,
}: Props) {
  const table = scene?.table ?? { sizeM: [0.8, 0.78], centerXy: [0.36, -0.38] };
  const sc = makeScale(table);

  const liveById = new Map(parts.map((p) => [p.id, p]));
  const partById = new Map((scene?.parts ?? []).map((p) => [p.id, p]));
  const binById = new Map(bins.map((b) => [b.id, b]));
  const activeTasks = tasks.filter((x) => x.state !== "done" && x.state !== "failed");

  const trayLoad = new Map<string, number>();
  for (const task of tasks) {
    if (task.state === "done") trayLoad.set(task.binId, (trayLoad.get(task.binId) ?? 0) + 1);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="cellmap" role="img"
         aria-label={t(lang, "map")}>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
        </marker>
        <pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">
          <path d="M 26 0 L 0 0 0 26" fill="none" stroke="rgba(120,140,170,.10)"
                strokeWidth="1" />
        </pattern>
      </defs>

      {/* Table */}
      <rect x={sc.X(table.centerXy[0] - table.sizeM[0] / 2)}
            y={sc.Y(table.centerXy[1] + table.sizeM[1] / 2)}
            width={table.sizeM[0] * sc.s} height={table.sizeM[1] * sc.s}
            rx={8} fill="#1b2432" stroke="#2c3a50" strokeWidth={1.5} />
      <rect x={sc.X(table.centerXy[0] - table.sizeM[0] / 2)}
            y={sc.Y(table.centerXy[1] + table.sizeM[1] / 2)}
            width={table.sizeM[0] * sc.s} height={table.sizeM[1] * sc.s}
            rx={8} fill="url(#grid)" />

      {/* Reach annuli, drawn only for healthy arms so a faulted one visibly stops competing */}
      {(scene?.robots ?? []).filter((r) => r.enabled).map((r) => {
        const [bx, by] = r.baseXy;
        return (
          <g key={`reach-${r.id}`} opacity={0.5}>
            <circle cx={sc.X(bx)} cy={sc.Y(by)} r={r.reachMaxM * sc.s}
                    fill="none" stroke="#3d7fd6" strokeWidth={1}
                    strokeDasharray="3 5" />
            <circle cx={sc.X(bx)} cy={sc.Y(by)} r={r.reachMinM * sc.s}
                    fill="none" stroke="#3d7fd6" strokeWidth={1} opacity={0.4} />
          </g>
        );
      })}

      {/* Planned routes: part -> tray, one per open task */}
      {activeTasks.map((task) => {
        const live = liveById.get(task.partId);
        const tray = binById.get(task.binId);
        const colour = STATE_COLOUR[task.state] ?? "#94a3b8";
        if (!tray) return null;
        const px = live ? live.x : partById.get(task.partId)?.x;
        const py = live ? live.y : partById.get(task.partId)?.y;
        if (px === undefined || py === undefined) return null;
        return (
          <g key={`route-${task.id}`}>
            <path
              d={`M ${sc.X(px)} ${sc.Y(py)} Q ${sc.X((px + tray.x) / 2)} ${sc.Y(py) - 46} ${sc.X(tray.x)} ${sc.Y(tray.y)}`}
              fill="none" stroke={colour} strokeWidth={1.6}
              strokeDasharray={task.state === "queued" ? "4 4" : undefined}
              markerEnd="url(#arrow)" opacity={task.state === "queued" ? 0.5 : 0.95}
            />
          </g>
        );
      })}

      {/* Trays */}
      {bins.map((b) => {
        const fill = trayLoad.get(b.id) ?? 0;
        const active = hoveredBin === b.id;
        const w = (b.outerM?.[0] ?? 0.087) * sc.s;
        const h = (b.outerM?.[1] ?? 0.087) * sc.s;
        return (
          <g key={b.id} className="tray" onClick={() => onPickBin(b.id)}
             style={{ cursor: selectedPart ? "pointer" : "default" }}>
            <rect x={sc.X(b.x) - w / 2} y={sc.Y(b.y) - h / 2} width={w} height={h}
                  rx={3} fill="rgba(255,255,255,.05)"
                  stroke={active ? "#fff" : b.color}
                  strokeWidth={active ? 2.5 : 1.5}
                  opacity={selectedPart ? 1 : 0.85} />
            <rect x={sc.X(b.x) - (b.innerM[0] * sc.s) / 2}
                  y={sc.Y(b.y) - (b.innerM[1] * sc.s) / 2}
                  width={b.innerM[0] * sc.s} height={b.innerM[1] * sc.s}
                  fill="none" stroke={b.color} strokeWidth={1} strokeDasharray="2 3"
                  opacity={0.6} />
            <text x={sc.X(b.x)} y={sc.Y(b.y) + 3} textAnchor="middle"
                  fontSize={11} fill={b.color} fontWeight={700}>
              {fill > 0 ? `${b.label} (${fill})` : b.label}
            </text>
          </g>
        );
      })}

      {/* Robot bases */}
      {(scene?.robots ?? []).filter((r) => r.enabled).map((r) => {
        const live = robots.find((x) => x.id === r.id);
        const faulted = live?.state === "faulted";
        const bx = sc.X(r.baseXy[0]);
        const by = sc.Y(r.baseXy[1]);
        const colour = faulted ? "#e0483c" : live?.state === "working" ? "#3aa76d" : "#64748b";
        return (
          <g key={r.id}>
            <rect x={bx - 11} y={by - 7} width={22} height={14} rx={3}
                  fill="#0f1722" stroke={colour} strokeWidth={1.6}
                  opacity={faulted ? 0.55 : 1} />
            <text x={bx} y={by + 4} textAnchor="middle" fontSize={9.5}
                  fill={colour} fontWeight={700}>
              {r.id.replace("robot_", "R")}
            </text>
            {faulted && (
              <text x={bx + 17} y={by - 6} fontSize={13} fill="#e0483c" fontWeight={700}>✕</text>
            )}
          </g>
        );
      })}

      {/* Live tool positions, with a line back to the owning base */}
      {robots.filter((r) => r.tool.length === 3 && r.state === "working").map((r) => {
        const scene_ = scene?.robots.find((x) => x.id === r.id);
        const task = tasks.find((x) => x.id === r.taskId);
        const colour = STATE_COLOUR[task?.state ?? ""] ?? "#3aa76d";
        const tx = sc.X(r.tool[0]);
        const ty = sc.Y(r.tool[1]);
        const bx = sc.X(scene_?.baseXy[0] ?? 0);
        const by = sc.Y(scene_?.baseXy[1] ?? 0);
        const closing = task?.holding === "closing";
        return (
          <g key={`tool-${r.id}`}>
            <line x1={bx} y1={by} x2={tx} y2={ty} stroke={colour}
                  strokeWidth={1.4} opacity={0.55} strokeDasharray="5 4" />
            {/* The jaws, at their measured span. */}
            <g transform={`translate(${tx} ${ty}) rotate(${r.spanM > 0 ? 0 : 0})`}>
              <line x1={-r.spanM * sc.s / 2} y1={0} x2={r.spanM * sc.s / 2} y2={0}
                    stroke={closing ? "#d98c1f" : "#e2e8f0"} strokeWidth={3.5}
                    strokeLinecap="round" />
            </g>
            <circle cx={tx} cy={ty} r={3} fill={colour} />
          </g>
        );
      })}

      {/* Parts, at their measured positions */}
      {(scene?.parts ?? []).map((p) => {
        const live = liveById.get(p.id);
        const x = live ? live.x : p.x;
        const y = live ? live.y : p.y;
        const selected = selectedPart === p.id;
        const task = tasks.find((t) => t.partId === p.id);
        const reached = task?.state === "done";
        const moving = live ? !live.resting : false;
        return (
          <g key={p.id} className="part" onClick={() => onPickPart(p.id)}
             style={{ cursor: "pointer" }}>
            {selected && (
              <circle cx={sc.X(x)} cy={sc.Y(y)} r={16} fill="none"
                      stroke="#fff" strokeWidth={2} />
            )}
            <g transform={`translate(${sc.X(x)} ${sc.Y(y)})`}
               opacity={task?.state === "failed" ? 0.35 : 1}>
              <PartShape kind={p.kind} colour={p.color} />
              {reached && <circle r={11} fill="none" stroke="#3aa76d" strokeWidth={1.6} />}
              {moving && <circle r={11} fill="none" stroke="#d98c1f" strokeWidth={1.4}
                                  strokeDasharray="2 3" />}
            </g>
            <text x={sc.X(x)} y={sc.Y(y) - 13} textAnchor="middle" fontSize={9}
                  fill="#94a3b8">
              {p.id.replace("part_", "P")}
            </text>
          </g>
        );
      })}

      {/* Scale bar: a metric scale nobody has to guess at */}
      <g transform={`translate(${PAD + 4} ${H - 16})`}>
        <line x1={0} y1={0} x2={0.1 * sc.s} y2={0} stroke="#64748b" strokeWidth={2} />
        <line x1={0} y1={-4} x2={0} y2={4} stroke="#64748b" strokeWidth={2} />
        <line x1={0.1 * sc.s} y1={-4} x2={0.1 * sc.s} y2={4} stroke="#64748b" strokeWidth={2} />
        <text x={0.05 * sc.s} y={-7} textAnchor="middle" fontSize={9.5} fill="#94a3b8">
          100 mm
        </text>
      </g>
      <text x={W - PAD} y={H - 16} textAnchor="end" fontSize={9.5} fill="#64748b">
        {t(lang, "table")} {num(lang, table.sizeM[0] * 1000, 0)}×
        {num(lang, table.sizeM[1] * 1000, 0)} mm
      </text>
    </svg>
  );
}