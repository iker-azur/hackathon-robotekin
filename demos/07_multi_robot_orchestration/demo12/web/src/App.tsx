import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * Demo 12 -- live view of the cell.
 *
 * Reads the WebSocket for state and the MJPEG endpoint for the render. Nothing here
 * computes anything: no task status, no progress, no "done". Every value on screen came
 * from the server, which read it from PhysX. A component that derived its own opinion
 * about whether a pick succeeded is exactly the thing this demo was rebuilt to stop doing.
 *
 * The frame is an <img> pointed at /api/frame.jpg and nudged with a cache-busting query.
 * If there is no frame -- headless Kit with no viewport -- the image errors and the panel
 * says so with the server's own reason, rather than showing a placeholder that would let a
 * dead render look alive.
 */

const API = "http://127.0.0.1:8000";
const WS = "ws://127.0.0.1:8000/ws";

type Health = {
  running: boolean;
  steps: number;
  lastError: string | null;
  errorCount: number;
};

type Robot = { id: string; state: string; taskId: string | null; note: string };
type Task = {
  id: string;
  partId: string;
  binId: string;
  label: string;
  state: string;
  assignedRobot: string | null;
  previousRobots: string[];
  attempts: number;
  note: string;
  holding: string | null;
};
type Part = {
  id: string;
  x: number;
  y: number;
  z: number;
  yawDeg: number;
  linSpeed: number;
  resting: boolean;
};
type Bin = { id: string; label: string; x: number; y: number; innerM: number[] };

type Snapshot = {
  simTimeS: number;
  controlTicks: number;
  allTasksCompleted: boolean;
  tasksCompleted: number;
  tasksTotal: number;
  failuresInjected: number;
  reassignments: number;
  robots: Robot[];
  tasks: Task[];
  health?: Health;
  cell?: { parts: Part[]; bins: Bin[]; tableTopZ: number };
  frameError?: string;
};

const STATE_ES: Record<string, string> = {
  queued: "en cola",
  approaching: "acerccando",
  closing: "cerrando",
  lifting: "levantando",
  transferring: "trasladando",
  releasing: "soltando",
  verifying: "verificando",
  done: "completado",
  failed: "fallido",
};

const STATE_COLOUR: Record<string, string> = {
  queued: "#6b7280",
  approaching: "#3d7fd6",
  closing: "#d98c1f",
  lifting: "#d98c1f",
  transferring: "#d98c1f",
  releasing: "#a855f7",
  verifying: "#a855f7",
  done: "#3aa76d",
  failed: "#e0483c",
};

function Pill({ value, text }: { value: string; text?: string }) {
  return (
    <span
      style={{
        background: STATE_COLOUR[value] ?? "#6b7280",
        color: "#fff",
        borderRadius: 10,
        padding: "2px 9px",
        fontSize: 11,
        fontWeight: 600,
        whiteSpace: "nowrap",
      }}
    >
      {text ?? STATE_ES[value] ?? value}
    </span>
  );
}

function Frame() {
  const [nonce, setNonce] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  // A plain <img> re-requested on an interval. No canvas, no worker, and the server is
  // already doing the expensive part.
  useEffect(() => {
    timer.current = window.setInterval(() => setNonce((n) => n + 1), 700);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, []);

  return (
    <div style={{ border: "1px solid #d8dce3", borderRadius: 8, overflow: "hidden" }}>
      {error ? (
        <div style={{ padding: 20, color: "#8a5a00", background: "#fff8e6", fontSize: 13 }}>
          <strong>Sin imagen.</strong>
          <div style={{ marginTop: 6, fontFamily: "ui-monospace, monospace", fontSize: 11 }}>
            {error}
          </div>
          <div style={{ marginTop: 6, fontSize: 12 }}>
            Arranca el servidor con <code>--gui</code> para tener una ventana de Isaac Sim, o
            acepta que este panel quede vacío en headless.
          </div>
        </div>
      ) : (
        <img
          src={`${API}/api/frame.jpg?t=${nonce}`}
          alt="Isaac Sim"
          onError={async () => {
            try {
              const res = await fetch(`${API}/api/state`);
              const body = await res.json();
              setError(body.frameError ?? "el servidor no devolvio una imagen");
            } catch {
              setError("no hay servidor en 127.0.0.1:8000");
            }
          }}
          style={{ width: "100%", display: "block", background: "#0d1017" }}
        />
      )}
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<Snapshot | null>(null);
  const [socketUp, setSocketUp] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(WS);
    ws.onopen = () => setSocketUp(true);
    ws.onclose = () => setSocketUp(false);
    ws.onmessage = (event) => setState(JSON.parse(event.data));
    return () => ws.close();
  }, []);

  const post = useCallback(async (path: string) => {
    await fetch(`${API}${path}`, { method: "POST" });
  }, []);

  const health = state?.health;
  const parts = state?.cell?.parts ?? [];
  const bins = state?.cell?.bins ?? [];

  const done = state?.allTasksCompleted ?? false;
  const progress = useMemo(() => {
    if (!state || state.tasksTotal === 0) return 0;
    return (state.tasksCompleted / state.tasksTotal) * 100;
  }, [state]);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 1180, margin: "0 auto", padding: 20 }}>
      <header style={{ display: "flex", alignItems: "baseline", gap: 14, marginBottom: 14 }}>
        <h1 style={{ fontSize: 20, margin: 0 }}>Demo 12 — Orquestación multi-robot</h1>
        <span style={{ fontSize: 12, color: socketUp ? "#3aa76d" : "#e0483c" }}>
          {socketUp ? "● WebSocket conectado" : "○ sin conexión"}
        </span>
      </header>

      {/* Health is shown before anything else. A demo that looks fine while the loop is
          throwing has told you nothing useful. */}
      {health?.lastError && (
        <div
          style={{
            background: "#fdecea",
            border: "1px solid #f3b0a8",
            color: "#8a1c10",
            padding: 10,
            borderRadius: 6,
            marginBottom: 14,
            fontSize: 13,
          }}
        >
          <strong>Error del orquestador</strong> ({health.errorCount}×
          ): {health.lastError}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 18 }}>
        <div>
          <Frame />

          <div style={{ display: "flex", gap: 8, margin: "12px 0" }}>
            <button onClick={() => post("/api/run")} style={btn(true)}>
              Ejecutar
            </button>
            <button onClick={() => post("/api/stop")} style={btn(false)}>
              Detener
            </button>
            {(state?.robots ?? []).map((r) => (
              <button key={r.id} onClick={() => post(`/api/fault/${r.id}`)} style={btn(false)}>
                Fallar {r.id}
              </button>
            ))}
            {(state?.robots ?? [])
              .filter((r) => r.state === "faulted")
              .map((r) => (
                <button key={r.id} onClick={() => post(`/api/repair/${r.id}`)} style={btn(false)}>
                  Reparar {r.id}
                </button>
              ))}
          </div>

          <section style={card}>
            <h2 style={h2}>Tareas</h2>
            <div style={{ height: 6, background: "#e5e7eb", borderRadius: 3, marginBottom: 10 }}>
              <div
                style={{
                  height: "100%",
                  width: `${progress}%`,
                  background: done ? "#3aa76d" : "#3d7fd6",
                  borderRadius: 3,
                  transition: "width .3s",
                }}
              />
            </div>
            <table style={table}>
              <thead>
                <tr>
                  <th>Tarea</th>
                  <th>Pieza → destino</th>
                  <th>Estado</th>
                  <th>Fase</th>
                  <th>Robot</th>
                  <th>Intentos</th>
                  <th>Nota</th>
                </tr>
              </thead>
              <tbody>
                {(state?.tasks ?? []).map((t) => (
                  <tr key={t.id}>
                    <td>{t.id}</td>
                    <td>
                      {t.partId} → {t.binId}
                    </td>
                    <td>
                      <Pill value={t.state} />
                    </td>
                    <td style={{ fontSize: 12, color: "#4b5563" }}>
                      {t.holding ? STATE_ES[t.holding] ?? t.holding : "—"}
                    </td>
                    <td>{t.assignedRobot ?? "—"}</td>
                    <td>{t.attempts}</td>
                    <td style={{ fontSize: 11, color: "#6b7280", maxWidth: 260 }}>
                      {t.note || (t.previousRobots.length ? `antes: ${t.previousRobots.join(", ")}` : "")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>

        <div>
          <section style={card}>
            <h2 style={h2}>Resumen</h2>
            <dl style={{ margin: 0, fontSize: 13 }}>
              <Row k="Simulado (s)" v={state?.simTimeS?.toFixed(1) ?? "—"} />
              <Row k="Ticks de control" v={state?.controlTicks ?? "—"} />
              <Row k="Tareas" v={`${state?.tasksCompleted ?? 0} / ${state?.tasksTotal ?? 0}`} />
              <Row k="Reasignaciones" v={state?.reassignments ?? 0} />
              <Row k="Fallos inyectados" v={state?.failuresInjected ?? 0} />
              <Row
                k="Resultado"
                v={done ? "todas completadas" : state ? "en curso" : "—"}
                colour={done ? "#3aa76d" : undefined}
              />
            </dl>
          </section>

          <section style={card}>
            <h2 style={h2}>Robots</h2>
            {(state?.robots ?? []).map((r) => (
              <div key={r.id} style={{ fontSize: 13, marginBottom: 8 }}>
                <strong>{r.id}</strong>{" "}
                <Pill
                  value={r.state === "faulted" ? "failed" : r.state === "working" ? "closing" : "queued"}
                  text={r.state === "idle" ? "libre" : r.state === "working" ? "ocupado" : "averiado"}
                />{" "}
                <span style={{ color: "#6b7280" }}>{r.taskId ?? ""}</span>
              </div>
            ))}
          </section>

          <section style={card}>
            <h2 style={h2}>Piezas</h2>
            <table style={table}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Posición (m)</th>
                  <th>v (m/s)</th>
                  <th>Reposo</th>
                </tr>
              </thead>
              <tbody>
                {parts.map((p) => (
                  <tr key={p.id}>
                    <td>{p.id}</td>
                    <td style={{ fontFamily: "ui-monospace, monospace", fontSize: 11 }}>
                      {p.x.toFixed(3)}, {p.y.toFixed(3)}, {p.z.toFixed(3)}
                    </td>
                    <td>{p.linSpeed.toFixed(4)}</td>
                    <td>{p.resting ? "sí" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: 11, color: "#6b7280", marginTop: 8 }}>
              Una tarea solo se marca completada si la pieza está dentro de la cavidad del
              destino <em>y</em> en reposo. Las dos cosas se leen del cuerpo rígido.
            </div>
          </section>

          <section style={card}>
            <h2 style={h2}>Destinos</h2>
            {bins.map((b) => (
              <div key={b.id} style={{ fontSize: 13 }}>
                <strong>{b.id}</strong> — {b.label} en ({b.x.toFixed(3)}, {b.y.toFixed(3)}),
                interior {(b.innerM[0] * 1000).toFixed(0)}×
                {(b.innerM[1] * 1000).toFixed(0)} mm
              </div>
            ))}
          </section>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v, colour }: { k: string; v: React.ReactNode; colour?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0" }}>
      <dt style={{ color: "#6b7280" }}>{k}</dt>
      <dd style={{ margin: 0, fontWeight: 600, color: colour }}>{v}</dd>
    </div>
  );
}

const card: React.CSSProperties = {
  background: "#fff",
  border: "1px solid #d8dce3",
  borderRadius: 8,
  padding: 14,
  marginBottom: 14,
};

const h2: React.CSSProperties = { fontSize: 14, margin: "0 0 10px" };

const table: React.CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: 12,
};

function btn(primary: boolean): React.CSSProperties {
  return {
    padding: "6px 12px",
    borderRadius: 6,
    border: "1px solid #c9ced8",
    background: primary ? "#3d7fd6" : "#fff",
    color: primary ? "#fff" : "#1f2937",
    cursor: "pointer",
    fontSize: 12,
  };
}