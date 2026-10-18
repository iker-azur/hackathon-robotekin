/**
 * Typed client for the Demo 12 API.
 *
 * Every shape here is the server's, not a guess. The interface never derives its own
 * opinion about whether a pick succeeded -- it renders what it is given, and where a value
 * is missing it renders nothing rather than a placeholder. That is the same rule the
 * backend follows, and it is the reason the screen can be trusted.
 */

export const API = "http://127.0.0.1:8000";
export const WS = "ws://127.0.0.1:8000/ws";

export type Lang = "es" | "en";

export type Health = {
  running: boolean;
  steps: number;
  lastError: string | null;
  errorCount: number;
};

export type Evidence = {
  ok: boolean;
  partId: string;
  binId: string;
  robotId: string;
  criteria: Record<string, boolean>;
  failures: string[];
  measurements: {
    toolErrorM: number;
    spanOnPartM: number;
    partWidthM: number;
    gripForceN: number;
    gripRequiredN: number;
    gripMargin: number | null;
    gripForceDuringLiftN: number;
    partDriftOnCloseM: number;
    partRiseM: number;
    partCarriedToBin: boolean;
    partInBin: boolean;
    partResting: boolean;
  };
};

export type Robot = {
  id: string;
  state: string;
  taskId: string | null;
  note: string;
  busyTicks: number;
  idleTicks: number;
  tasksCompleted: number;
  tasksFailed: number;
  gripN: number;
  spanM: number;
  tool: number[];
  joints: number[];
  utilisation: number;
};

export type TaskProgress = {
  phase: string;
  stage: string;
  fraction: number;
  tick: number;
  spanTicks: number;
};

export type Task = {
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
  progress: TaskProgress | null;
  evidence: Evidence | null;
  history: {
    robotId: string;
    attempt: number;
    ok: boolean;
    criteria: Record<string, boolean>;
    measurements: Evidence["measurements"];
    failures: string[];
  }[];
  startedTick: number;
  finishedTick: number;
};

export type Part = {
  id: string;
  kind: string;
  material: string;
  color: string;
  x: number;
  y: number;
  sizeM: number[];
  massG: number;
  graspable: boolean;
};

export type Bin = {
  id: string;
  label: string;
  color: string;
  x: number;
  y: number;
  innerM: number[];
  outerM: number[];
  heightM: number;
};

export type SceneRobot = {
  id: string;
  label: string;
  baseXy: number[];
  baseYawDeg: number;
  enabled: boolean;
  reachMinM: number;
  reachMaxM: number;
};

export type Scene = {
  name: string;
  table: { sizeM: number[]; centerXy: number[] };
  robots: SceneRobot[];
  parts: Part[];
  bins: Bin[];
  tasks: { id: string; partId: string; binId: string; label: string }[];
  errors: string[];
};

export type EventRecord = {
  seq: number;
  tick: number;
  simTimeS: number;
  kind: string;
  robotId: string | null;
  taskId: string | null;
  partId: string | null;
  binId: string | null;
  detail: string;
  data: Record<string, unknown>;
};

export type Metrics = {
  tasksTotal: number;
  tasksCompleted: number;
  tasksFailed: number;
  attempts: number;
  graspSuccessRate: number;
  reassignments: number;
  failuresInjected: number;
  controlTicks: number;
  simTimeS: number;
  tasksPerSimMinute: number;
  meanPartRiseMm: number;
  meanGripN: number;
  meanToolErrorMm: number;
  robotUtilisation: Record<string, number>;
  criteriaPassed: Record<string, { passed: number; failed: number }>;
};

export type LivePart = { id: string; x: number; y: number; z: number; resting: boolean };

export type Snapshot = {
  simTimeS: number;
  controlTicks: number;
  running: boolean;
  allTasksCompleted: boolean;
  tasksCompleted: number;
  tasksTotal: number;
  failuresInjected: number;
  reassignments: number;
  rosterChanges: number;
  robots: Robot[];
  tasks: Task[];
  parts: LivePart[];
  bins: Bin[];
  tableTopZ: number;
  scene: Scene | null;
  metrics: Metrics | null;
  assignmentCosts: Record<string, { robotId: string; reachM: number; carryM: number; total: number }>;
  eventCursor: number;
  eventCounts: Record<string, number>;
  lastError: string | null;
  errorCount: number;
  frameError: string;
  frameNote: string;
  ledger: { restored: boolean };
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

const post = <T,>(path: string, body?: unknown) =>
  req<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });

export const api = {
  state: () => req<Snapshot>("/api/state"),
  health: () => req<Health & { robots: number; headless: boolean }>("/api/health"),
  scene: () => req<Scene>("/api/scene"),
  metrics: () => req<Metrics>("/api/metrics"),
  events: (since: number) =>
    req<{ events: EventRecord[]; cursor: number; counts: Record<string, number> }>(
      `/api/events?since=${since}&limit=400`),
  scenarios: () => req<Record<string, { moves: { partId: string; binId: string }[]; en: string; es: string; errors: string[] }>>(
    "/api/scenarios"),
  validateScene: (scene: unknown) =>
    post<{ ok: boolean; errors: string[]; scene?: Scene }>("/api/scene/validate", scene),
  applyScene: (scene: unknown) =>
    post<{ ok: boolean; errors?: string[]; restarting?: boolean; reason?: string }>(
      "/api/scene/apply", scene),
  plan: (moves: { partId: string; binId: string }[]) =>
    post<{ ok: boolean; errors: string[]; tasks: { id: string; partId: string; binId: string }[] }>(
      "/api/plan", { moves }),
  command: (text: string, apply: boolean) =>
    post<{
      ok: boolean; text: string;
      moves: { partId: string; binId: string }[];
      errors: string[]; understood: string[];
      applied?: boolean;
    }>("/api/command", { text, apply }),
  submitTasks: (moves: { partId: string; binId: string }[]) =>
    post<{ ok: boolean; errors?: string[]; tasksSubmitted?: string[] }>(
      "/api/tasks/submit", { moves }),
  clearTasks: () => post<{ ok: boolean }>("/api/tasks/clear"),
  run: () => post<{ ok: boolean }>("/api/run"),
  stop: () => post<{ ok: boolean }>("/api/stop"),
  fault: (id: string) => post<{ ok: boolean }>(`/api/fault/${id}`),
  faults: (ids?: string[]) => post<{ ok: boolean; reason?: string }>("/api/faults", { robotIds: ids }),
  repair: (id: string) => post<{ ok: boolean }>(`/api/repair/${id}`),
};