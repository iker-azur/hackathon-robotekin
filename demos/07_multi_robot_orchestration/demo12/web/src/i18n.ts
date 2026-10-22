/**
 * ES/EN strings for the whole interface.
 *
 * One table, both languages, looked up by key. Not a i18n library: the interface has
 * roughly ninety strings and a build step, and a library would be more machinery than the
 * thing it manages.
 *
 * Every string is a key here rather than inline in a component, which means a missing
 * translation shows up as the key rather than as silently-English text in a Spanish
 * screen -- ``t()`` falls back to the English entry and then to the key itself, so the gap
 * is visible instead of invisible.
 */

export type Lang = "es" | "en";

type Table = Record<string, { es: string; en: string }>;

export const STRINGS: Table = {
  // -- header ------------------------------------------------------------
  appTitle: { es: "Orquestación multi-robot", en: "Multi-robot orchestration" },
  appSubtitle: {
    es: "Isaac Lab · PhysX · SO-101",
    en: "Isaac Lab · PhysX · SO-101",
  },
  connected: { es: "En línea", en: "Connected" },
  disconnected: { es: "Sin conexión", en: "Disconnected" },
  connecting: { es: "Conectando…", en: "Connecting…" },
  language: { es: "Language", en: "Language" },

  // -- tabs --------------------------------------------------------------
  tabCell: { es: "Celda", en: "Cell" },
  tabPlan: { es: "Plan", en: "Plan" },
  tabCollab: { es: "Colaboración", en: "Collaboration" },
  tabEditor: { es: "Editor", en: "Editor" },

  // -- run control -------------------------------------------------------
  run: { es: "Ejecutar", en: "Run" },
  stop: { es: "Pausar", en: "Pause" },
  running: { es: "En marcha", en: "Running" },
  paused: { es: "En pausa", en: "Paused" },
  fault: { es: "Averiar", en: "Fault" },
  repair: { es: "Reparar", en: "Repair" },
  faultAll: { es: "Averiar un brazo", en: "Fail an arm" },
  free: { es: "Libre", en: "Idle" },
  busy: { es: "Ocupado", en: "Busy" },
  faulted: { es: "Averiado", en: "Faulted" },
  waitingForWork: { es: "esperando trabajo", en: "waiting for work" },

  // -- task states -------------------------------------------------------
  queued: { es: "En cola", en: "Queued" },
  approaching: { es: "Acercándose", en: "Approaching" },
  closing: { es: "Cerrando", en: "Closing" },
  lifting: { es: "Levantando", en: "Lifting" },
  transferring: { es: "Trasladando", en: "Transferring" },
  releasing: { es: "Soltando", en: "Releasing" },
  verifying: { es: "Verificando", en: "Verifying" },
  done: { es: "Completada", en: "Done" },
  failed: { es: "Fallida", en: "Failed" },

  // -- cell view ---------------------------------------------------------
  camera: { es: "Cámara", en: "Camera" },
  map: { es: "Mapa de la celda", en: "Cell map" },
  noFrame: { es: "Sin imagen", en: "No image" },
  legend: { es: "Leyenda", en: "Legend" },
  reach: { es: "alcance", en: "reach" },
  table: { es: "mesa", en: "table" },
  tray: { es: "bandeja", en: "tray" },

  // -- planner -----------------------------------------------------------
  plannerTitle: { es: "Plan de trabajo", en: "Work plan" },
  plannerHint: {
    es: "Haz clic en una pieza y luego en una bandeja para asignarla.",
    en: "Click a part, then a tray, to assign it.",
  },
  assigned: { es: "Asignado", en: "Assigned" },
  unassigned: { es: "Sin destino", en: "No destination" },
  clearPlan: { es: "Vaciar plan", en: "Clear plan" },
  applyPlan: { es: "Aplicar plan", en: "Apply plan" },
  planning: { es: "Planificando…", en: "Planning…" },
  scenarios: { es: "Escenarios", en: "Scenarios" },
  commandLabel: { es: "Comando", en: "Command" },
  commandPlaceholder: {
    es: "p. ej. “mueve el cubo rojo a la caja azul”",
    en: "e.g. “move the red cube to the blue tray”",
  },
  commandRun: { es: "Interpretar", en: "Interpret" },
  commandApply: { es: "Interpretar y aplicar", en: "Interpret and apply" },
  understood: { es: "Entendido", en: "Understood" },
  noMoves: { es: "Sin movimientos", en: "No moves" },
  scenarioColor: { es: "Por color", en: "By colour" },
  scenarioMixed: { es: "Todo a una bandeja", en: "All to one tray" },
  scenarioBalanced: { es: "Clasificación cruzada", en: "Cross-sort" },

  // -- collaboration -----------------------------------------------------
  collabTitle: { es: "Línea de tiempo", en: "Timeline" },
  collabHint: {
    es: "Cada decisión del orquestador, con su motivo.",
    en: "Every orchestration decision, with its reason.",
  },
  activity: { es: "Actividad", en: "Activity" },
  load: { es: "Carga por robot", en: "Load per robot" },
  utilisation: { es: "Utilización", en: "Utilisation" },
  idle: { es: "libre", en: "idle" },
  tasksDone: { es: "tareas", en: "tasks" },
  metrics: { es: "Métricas", en: "Metrics" },
  simTime: { es: "Tiempo simulado", en: "Simulated time" },
  controlTicks: { es: "Ticks de control", en: "Control ticks" },
  completed: { es: "Completadas", en: "Completed" },
  reassignments: { es: "Reasignaciones", en: "Reassignments" },
  injectedFaults: { es: "Fallos inyectados", en: "Injected faults" },
  graspSuccess: { es: "Éxito de agarre", en: "Grasp success" },
  tasksPerMin: { es: "Tareas por minuto", en: "Tasks per minute" },
  meanRise: { es: "Subida media", en: "Mean rise" },
  meanGrip: { es: "Fuerza media", en: "Mean grip" },
  meanError: { es: "Error medio", en: "Mean tracking error" },

  // -- evidence ----------------------------------------------------------
  evidence: { es: "Prueba del agarre", en: "Grasp evidence" },
  evidenceHint: {
    es: "Una tarea solo se completa si se cumplen las siete. Todas se leen de la simulación.",
    en: "A task completes only if all seven hold. Every one is read from the simulation.",
  },
  evTracking: { es: "Seguimiento", en: "Tracking" },
  evPads: { es: "Dedos en caras", en: "Pads on faces" },
  evGrip: { es: "Fuerza", en: "Grip" },
  evRose: { es: "La pieza subió", en: "Part rose" },
  evCarried: { es: "Transportada", en: "Carried" },
  evInBin: { es: "Dentro de la bandeja", en: "In tray" },
  evResting: { es: "En reposo", en: "At rest" },
  gripMargin: { es: "margen de fricción", en: "friction margin" },
  noEvidence: {
    es: "Aún no hay datos: la tarea no ha terminado un intento.",
    en: "No data yet: the task has not finished an attempt.",
  },

  // -- editor ------------------------------------------------------------
  editorTitle: { es: "Editor de la celda", en: "Cell editor" },
  editorHint: {
    es: "Isaac Lab no puede añadir piezas ni brazos a una escena viva, así que aplicar un cambio reinicia el proceso. El plan de tareas se conserva.",
    en: "Isaac Lab cannot add parts or arms to a live scene, so applying a change restarts the process. The task plan is preserved.",
  },
  editorValidation: { es: "Validación", en: "Validation" },
  valid: { es: "La celda es válida", en: "The cell is valid" },
  invalid: { es: "La celda no se puede construir", en: "The cell cannot be built" },
  applyScene: { es: "Aplicar y reiniciar", en: "Apply and restart" },
  restarting: { es: "Reiniciando…", en: "Restarting…" },
  robots: { es: "Brazos", en: "Arms" },
  parts: { es: "Piezas", en: "Parts" },
  trays: { es: "Bandejas", en: "Trays" },
  noSupervisor: {
    es: "Arrancado sin supervisor: no puede reiniciarse a una escena nueva. Usa demo12.launcher.",
    en: "Started without the supervisor, so it cannot restart onto a new scene. Run demo12.launcher.",
  },

  // -- errors / misc -----------------------------------------------------
  error: { es: "Error del orquestador", en: "Orchestrator error" },
  noServer: { es: "No hay servidor en 127.0.0.1:8000", en: "No server at 127.0.0.1:8000" },
  task: { es: "Tarea", en: "Task" },
  piece: { es: "Pieza", en: "Part" },
  destination: { es: "Destino", en: "Destination" },
  state: { es: "Estado", en: "State" },
  phase: { es: "Fase", en: "Phase" },
  robot: { es: "Robot", en: "Robot" },
  attempts: { es: "Intentos", en: "Attempts" },
  note: { es: "Nota", en: "Note" },
  progress: { es: "Progreso", en: "Progress" },
  pieces: { es: "piezas", en: "parts" },
  trays_filled: { es: " bandejas con piezas", en: " trays filled" },
  clickForTarget: {
    es: "Clic en una bandeja para cambiar el destino",
    en: "Click a tray to change the destination",
  },
};

export function t(lang: Lang, key: string): string {
  const entry = STRINGS[key];
  if (!entry) return key;
  return entry[lang] ?? entry.en ?? key;
}

/** Insert a number in the reader's own convention, not the host machine's. */
export function num(lang: Lang, value: number, digits = 1): string {
  return value.toLocaleString(lang === "es" ? "es-ES" : "en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}