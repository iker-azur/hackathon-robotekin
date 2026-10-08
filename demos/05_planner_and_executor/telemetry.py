"""One bus between the robot and the dashboard: state, events, frames and commands.

Demo 11 runs the simulator on the main thread - Isaac requires it - and the web app on a
background thread, so the two can only meet through something that is safe to touch from
either side. This module is that something, and it deliberately depends on nothing but the
standard library, so importing it can never be the thing that breaks a run.

Three directions of travel:

* **Outbound** (``node``, ``log``, ``set_state``, ``put_frame``) is what the robot publishes.
  Every call takes one lock, updates a plain dict, and pokes a waiter. Nothing here can block
  on the network: a dashboard that has stopped reading must never be able to stall the arm
  between two physics steps.
* **Inbound** (``submit``, ``send_control``) is what a person in the browser presses. It goes
  into ordinary queues, so the demo's own loop decides when to look at it - the browser never
  reaches into the robot.
* **The start gate** (``open_start`` / ``wait_start``) exists because the dashboard is meant to
  be up *before* Isaac, so the environment can be set up from the browser and then started
  from there. Two minutes of simulator start-up is exactly the sort of thing you want to be
  watching rather than staring at a black console.

The node list is the decision path of the whole system, in the order the data flows: the setup
that was chosen, the command, what the model understood, the frame, what was in it, which blob
is which part, what the sentence means, what the arm will do, what the arm did, whether the
part landed, and what happened to the ones that did not. The dashboard draws its diagram from
``FLOW`` and ``EDGES`` rather than from its own markup, so there is one definition of the
pipeline and the picture cannot drift away from the code.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

# Node states, in the order the dashboard should escalate their colour. Kept as a set rather
# than an enum so a stray state from a typo shows up as an unknown colour instead of silently
# looking like the "fine" one.
IDLE = "idle"
ACTIVE = "active"
OK = "ok"
WARN = "warn"
FAIL = "fail"

STATES = (IDLE, ACTIVE, OK, WARN, FAIL)

# Command lifecycle. Every browser or terminal request gets an id and moves through these states,
# so the page always knows whether a request was received, whether the robot accepted it, and how
# it ended. Only the transitions in ALLOWED are legal; anything else is refused.
RECEIVED, QUEUED, RUNNING = "RECEIVED", "QUEUED", "RUNNING"
SUCCESS, FAILED, CANCELLED = "SUCCESS", "FAILED", "CANCELLED"
NEEDS_CLARIFICATION, PARTIAL_SUCCESS = "NEEDS_CLARIFICATION", "PARTIAL_SUCCESS"
ALLOWED = {
    RECEIVED: {QUEUED, FAILED, CANCELLED},
    QUEUED: {RUNNING, CANCELLED, FAILED},
    RUNNING: {SUCCESS, FAILED, CANCELLED, NEEDS_CLARIFICATION, PARTIAL_SUCCESS},
}
# How many sentences and controls may wait. Past these, the request is answered FAILED, never
# silently dropped, and the robot never has to chew through an unbounded backlog.
MAX_QUEUED_COMMANDS = 16
MAX_QUEUED_CONTROLS = 32
# Controls that are harmless to repeat while one is already waiting: a second "relook" adds no
# information, so it is answered CANCELLED with the reason instead of costing another perception.
COALESCE_ACTIONS = frozenset({"relook", "save"})
# How many command-log entries the bus keeps.
COMMAND_LOG_SIZE = 50


@dataclass(frozen=True)
class Node:
    """One box in the flow diagram: a stage, not an event.

    ``row`` is the diagram's row. The spine is one column and the recovery loop is offset to
    the right, because a linear list of boxes cannot show that a failed verification sends the
    system back to the camera and to the planner rather than forwards to the report.
    """

    id: str
    title: str
    detail: str
    row: int


FLOW: tuple[Node, ...] = (
    Node("setup", "Entorno",
         "variante, cámara, pre-vuelo", 0),
    Node("input", "Comando",
         "escrito o hablado", 1),
    Node("understand", "Clasificación",
         "intención, piezas, destino", 2),
    Node("capture", "Cámara",
         "un fotograma y sensores", 3),
    Node("perceive", "Percepción",
         "manchas de color a piezas", 4),
    Node("link", "Enlace",
         "cada mancha, su pieza", 5),
    Node("ground", "Resolución",
         "resuelve, pregunta si duda", 6),
    Node("plan", "Plan",
         "poses y órdenes de pinzas", 7),
    Node("execute", "Ejecución",
         "paso a paso, reapunta", 8),
    Node("verify", "Verificación",
         "¿quedó dentro de la caja?", 9),
    Node("recover", "Reintento",
         "reintenta o declara fallo", 10),
    Node("report", "Informe",
         "resultado de la ronda", 11),
)

# (from, to, label). The spine plus the two edges that make the recovery loop a loop.
EDGES: tuple[tuple[str, str, str], ...] = (
    ("setup", "input", ""),
    ("input", "understand", "frase"),
    ("understand", "capture", "mirar"),
    ("capture", "perceive", "fotograma"),
    ("perceive", "link", "manchas"),
    ("link", "ground", "piezas"),
    ("ground", "plan", "objetivos"),
    ("plan", "execute", "pasos"),
    ("execute", "verify", "movimiento"),
    ("verify", "recover", "no dentro"),
    ("recover", "capture", "re-mirada"),
    ("recover", "plan", "re-plan"),
    ("recover", "report", "rendido"),
    ("verify", "report", "todas dentro"),
)

SPINE = tuple(node.id for node in FLOW)


class Bus:
    """The shared state, and the queue of things a person asked for.

    One lock for everything, held for the length of a dict update and never across a socket
    write. ``snapshot`` copies under the lock and returns plain JSON-ready types, so the
    websocket handler can serialise it without holding anything up.
    """

    def __init__(self, history: int = 400):
        self._lock = threading.RLock()
        self._seq = 0
        # Bumped by ``_wake``, so by every outbound call there is: a state merge and a camera
        # frame move it just as much as a log line does. ``_seq`` cannot serve as the dashboard's
        # "has anything changed" test, because it only counts log lines - two snapshots with the
        # same ``_seq`` can still be different pictures, and the dashboard repainting on a
        # signature that ignores that leaves stale numbers on screen.
        self._rev = 0
        self._events: deque[dict] = deque(maxlen=history)
        self._nodes: dict[str, dict] = {
            node.id: {"id": node.id, "title": node.title, "detail": node.detail,
                      "row": node.row, "state": IDLE, "note": "", "at": None, "hits": 0}
            for node in FLOW
        }
        self._state: dict = {}
        self._hooks: list = []
        self._frame_seq = 0
        self._frame: tuple[int, bytes, dict] | None = None
        # Unbounded on purpose. A command a person typed must never be dropped because the arm
        # is halfway through a pick; the demo drains it when it is ready.
        # Bounded, and full means refused: see MAX_QUEUED_COMMANDS. A STOP never goes in here.
        self._commands: queue.Queue[tuple[str, str, str]] = queue.Queue(
            maxsize=MAX_QUEUED_COMMANDS)
        self._last_command_source = ""
        self._last_command_id = ""
        self._active_command_id = ""
        self._controls: queue.Queue[dict] = queue.Queue(maxsize=MAX_QUEUED_CONTROLS)
        self._pending_actions: set[str] = set()
        self._cmd_seq = 0
        self._cmd_log: dict[str, dict] = {}
        self._stop = threading.Event()
        self._stop_cid = ""
        self._ticks: dict[str, deque] = {}
        self._metrics: dict = {}
        self._gate = threading.Event()

    # -- outbound --------------------------------------------------------------

    def node(self, node_id: str, state: str = ACTIVE, note: str = "", **data) -> None:
        """Set one diagram box's state and note.

        ``state`` is not validated against ``STATES`` on purpose: a typo should show up in the
        dashboard as an unknown colour where someone can see it, not raise inside the middle
        of a pick.
        """
        with self._lock:
            box = self._nodes.get(node_id)
            if box is None:
                return
            box["state"] = state
            box["note"] = note
            box["at"] = time.time()
            box["hits"] += 1
            if data:
                box["data"] = data
        self._wake()

    def log(self, message: str = "", level: str = "info", node: str = "", **data) -> None:
        """One line in the dashboard console.

        ``message`` may be several lines: the robot prints its perception summaries and plan
        summaries as multi-line blocks, and splitting them here would lose the blank lines
        that make those readable in a terminal.
        """
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "t": time.time(), "level": level, "node": node,
                     "message": message}
            if data:
                event["data"] = data
            self._events.append(event)
        self._wake()

    def set_state(self, **fields) -> None:
        """Merge top-level values the dashboard shows as pills and counters."""
        with self._lock:
            self._state.update(fields)
        self._wake()

    def put_frame(self, jpeg: bytes, meta: dict | None = None) -> None:
        """Publish the latest camera frame, replacing whatever was there.

        One slot, never a queue: a dashboard wants the newest picture, and buffering frames
        for a slow reader would show the operator a scene that has already moved on.
        """
        with self._lock:
            self._frame_seq += 1
            self._frame = (self._frame_seq, bytes(jpeg), dict(meta or {}))
        self._wake()

    # -- reading ---------------------------------------------------------------

    def frame(self) -> tuple[int, bytes | None, dict]:
        with self._lock:
            if self._frame is None:
                return 0, None, {}
            return self._frame

    # -- performance instrumentation -------------------------------------------

    def tick(self, name: str) -> None:
        """Count one occurrence of ``name`` (a physics step, a camera frame). Rates come from these."""
        with self._lock:
            self._ticks.setdefault(name, deque(maxlen=600)).append(time.monotonic())

    def rate(self, name: str, window: float = 2.0) -> float:
        """Occurrences of ``name`` per second over the last ``window`` seconds."""
        with self._lock:
            stamps = self._ticks.get(name)
            if not stamps:
                return 0.0
            now = time.monotonic()
            return sum(1 for stamp in stamps if now - stamp <= window) / window

    def set_metric(self, name: str, value) -> None:
        with self._lock:
            self._metrics[name] = value

    def add_to_metric(self, name: str, delta: int) -> None:
        with self._lock:
            self._metrics[name] = self._metrics.get(name, 0) + delta
        self._wake()

    def metrics(self) -> dict:
        """The numbers the optional diagnostics section shows, in one place."""
        with self._lock:
            out = dict(self._metrics)
            out["command_queue_depth"] = self._commands.qsize()
            out["control_queue_depth"] = self._controls.qsize()
            names = list(self._ticks)
        for name in names:
            out[f"{name}_per_s"] = round(self.rate(name), 1)
        return out

    def snapshot(self, events_after: int | None = None) -> dict:
        """Everything a dashboard needs, in one JSON-ready dict.

        ``events_after`` makes the event list a delta: only the log lines whose sequence number is
        greater than it. A socket that has already been sent lines up to ``seq`` asks for the rest,
        so each push carries what is new instead of the whole history again. ``None`` means the
        last 140 lines, which is what a freshly connected page starts from.
        """
        with self._lock:
            events = list(self._events)
            if events_after is None:
                events = events[-140:]
            else:
                events = [event for event in events if event["seq"] > events_after][-140:]
            return {
                "seq": self._seq,
                "rev": self._rev,
                "t": time.time(),
                "nodes": [dict(box) for box in self._nodes.values()],
                "edges": [list(edge) for edge in EDGES],
                "state": dict(self._state),
                "events": events,
                "commands": [dict(cmd) for cmd in list(self._cmd_log.values())[-12:]],
                # Re-entrant lock: safe to read the metrics from inside the snapshot.
                "metrics": self.metrics(),
            }

    # -- inbound ---------------------------------------------------------------

    def submit(self, text: str, source: str = "dashboard") -> str | None:
        """Queue a command, and say where it came from.

        The queued sentence is published as state as well, because the browser has no other way
        to know it was heard: the page sends the text, the HTTP call returns, and from that
        moment the command is invisible until the robot's own loop gets round to it. Measured,
        that wait is the whole of "I typed it and nothing happened". So the panel shows what is
        sitting in the queue from the moment it arrives, which is also what tells somebody that
        a second command typed while the first is running is not lost.

        ``source`` exists because the terminal reads on its own thread (see
        ``MonitoredRobot._terminal_reader``) and lands in this same queue, so the robot can no
        longer tell the two apart by where it read them.
        """
        text = (text or "").strip()
        if not text:
            return None
        cid = self.new_command("command", source, text=text)
        try:
            self._commands.put_nowait((text, source, cid))
        except queue.Full:
            # Refused, and said so: a sentence must never vanish without the person knowing.
            self.set_command_state(cid, FAILED, "hay demasiadas órdenes esperando")
            return cid
        self.set_command_state(cid, QUEUED)
        self.set_state(queued=text, queued_from=source)
        return cid

    def command_source(self) -> str:
        """Where the command ``take_command`` last returned came from."""
        return self._last_command_source

    def control_pending(self) -> bool:
        """Is a control queued? Answers without consuming it.

        ``Queue.empty`` is only a snapshot, which is all that is wanted here: the caller uses
        this to decide whether it is safe to block on something else without swallowing a
        request the browser made.
        """
        return not self._controls.empty()

    def take_command(self, timeout: float | None = None) -> str | None:
        """The next queued command, or None if none arrived in time.

        The origin is remembered rather than returned, because every caller wants the sentence
        and only the log wants to know which of the three inputs it came from.
        """
        try:
            text, source, cid = (self._commands.get(timeout=timeout) if timeout else
                                 self._commands.get_nowait())
        except queue.Empty:
            return None
        self._last_command_source = source
        self._last_command_id = cid
        self._active_command_id = cid
        # Latency from the moment the page's request was received to the moment the robot took it.
        with self._lock:
            cmd = self._cmd_log.get(cid)
            if cmd is not None:
                self._metrics["command_latency_ms"] = round((time.time() - cmd["t"]) * 1000, 1)
        self.set_command_state(cid, RUNNING)
        return text

    def send_control(self, action: str, **params) -> str:
        """Queue a control action, and return its command id.

        ``stop`` is not a control: it is a flag (see ``request_stop``) so it cannot wait behind
        anything queued. A repeat of a coalescable control that is already waiting is CANCELLED
        with the reason, so it costs one perception pass rather than several.
        """
        source = params.pop("source", "dashboard")
        if action == "stop":
            return self.request_stop(source)
        cid = self.new_command(action, source, params=params)
        with self._lock:
            duplicate = action in COALESCE_ACTIONS and action in self._pending_actions
            if action in COALESCE_ACTIONS and not duplicate:
                self._pending_actions.add(action)
        if duplicate:
            self.set_command_state(cid, CANCELLED, f"ya hay una «{action}» esperando")
            return cid
        try:
            self._controls.put_nowait({"action": action, "params": params, "id": cid})
        except queue.Full:
            with self._lock:
                self._pending_actions.discard(action)
            self.set_command_state(cid, FAILED, "hay demasiadas órdenes esperando")
            return cid
        self.set_command_state(cid, QUEUED)
        return cid

    def take_control(self, timeout: float | None = None) -> dict | None:
        """The next queued control, or None. It becomes RUNNING; the robot finishes it."""
        try:
            control = (self._controls.get(timeout=timeout) if timeout else
                       self._controls.get_nowait())
        except queue.Empty:
            return None
        with self._lock:
            self._pending_actions.discard(control.get("action"))
        self._last_command_id = control.get("id", "")
        self.set_command_state(control.get("id", ""), RUNNING)
        return control

    # -- command lifecycle -----------------------------------------------------

    def new_command(self, kind: str, source: str, text: str = "",
                    params: dict | None = None) -> str:
        """Register a request from the browser or the terminal, and return its id. It is RECEIVED.

        Nothing here waits for the robot, so the answer to the person is immediate even while the
        arm is in the middle of a pick. What happens next is reported by the state changes.
        """
        with self._lock:
            self._cmd_seq += 1
            cid = f"cmd_{self._cmd_seq:05d}"
            self._cmd_log[cid] = {"id": cid, "type": kind, "state": RECEIVED, "text": text,
                                  "source": source, "note": "", "params": dict(params or {}),
                                  "t": time.time()}
            while len(self._cmd_log) > COMMAND_LOG_SIZE:
                self._cmd_log.pop(next(iter(self._cmd_log)))
        self._wake()
        return cid

    def set_command_state(self, cid: str, state: str, note: str = "") -> bool:
        """Move a command to ``state`` if the lifecycle allows it. False, with no change, if not."""
        with self._lock:
            cmd = self._cmd_log.get(cid)
            if cmd is None or state not in ALLOWED.get(cmd["state"], ()):
                return False
            cmd["state"] = state
            if note:
                cmd["note"] = note
            cmd["t_state"] = time.time()
        self._wake()
        return True

    def command_state(self, cid: str) -> dict | None:
        with self._lock:
            cmd = self._cmd_log.get(cid)
            return dict(cmd) if cmd is not None else None

    def last_command_id(self) -> str:
        """The id of the command that ``take_command`` or ``take_control`` returned last."""
        return self._last_command_id

    def active_command_id(self) -> str:
        """Text request identity, unaffected by controls consumed during motion."""
        return self._active_command_id

    def fail_active(self, reason: str) -> None:
        with self._lock:
            active = [cid for cid, cmd in self._cmd_log.items()
                      if cmd["state"] in (RECEIVED, QUEUED, RUNNING)]
        for cid in active:
            self.set_command_state(cid, FAILED, reason)

    # -- stop ------------------------------------------------------------------

    def request_stop(self, source: str = "dashboard") -> str:
        """STOP. It is never queued behind anything.

        It sets a flag the robot checks between physics steps and cancels every sentence and
        control still waiting: none of them can run once the run is stopping. The stop itself is
        QUEUED until the robot acknowledges it with ``complete_stop``.
        """
        cid = self.new_command("stop", source)
        self._stop_cid = cid
        self._stop.set()
        self._cancel_pending("parada solicitada")
        self.set_command_state(cid, QUEUED)
        return cid

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def stop_command_id(self) -> str:
        return self._stop_cid

    def complete_stop(self) -> None:
        """The robot has stopped: the stop's own command becomes SUCCESS, if one was requested."""
        if self._stop_cid:
            self.set_command_state(self._stop_cid, RUNNING)
            self.set_command_state(self._stop_cid, SUCCESS, "el robot se ha detenido")

    def _cancel_pending(self, reason: str) -> None:
        while True:
            try:
                _text, _source, cid = self._commands.get_nowait()
            except queue.Empty:
                break
            self.set_command_state(cid, CANCELLED, reason)
        while True:
            try:
                control = self._controls.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._pending_actions.discard(control.get("action"))
            self.set_command_state(control.get("id", ""), CANCELLED, reason)

    # -- the start gate --------------------------------------------------------

    def open_start(self) -> None:
        self._gate.set()

    def wait_start(self, timeout: float | None = None) -> bool:
        """Block until the dashboard says start. False if it timed out.

        A timeout rather than an error on purpose: a person who started the process from a
        terminal and then walked away to the browser should not find the process wedged
        forever, and ``--start-timeout`` is there for anyone who wants the opposite.
        """
        return self._gate.wait(timeout)

    def started(self) -> bool:
        return self._gate.is_set()

    # -- change notification ---------------------------------------------------

    def add_hook(self, hook) -> None:
        """Call ``hook()`` after every change, from whichever thread made it.

        The websocket layer registers a hook that hands the wake-up to its own event loop,
        because calling into an asyncio loop from the thread that is inside ``env.step`` is
        not allowed.
        """
        with self._lock:
            self._hooks.append(hook)

    def drop_hook(self, hook) -> None:
        with self._lock:
            if hook in self._hooks:
                self._hooks.remove(hook)

    def _wake(self) -> None:
        """Notify the listeners that something changed, and stamp the change.

        Every outbound method ends here, so this is the one place that has to know a change
        happened - which is what makes ``_rev`` an honest "is this snapshot different from the
        last one" answer for the dashboard, including for the changes that add no log line.
        """
        with self._lock:
            self._rev += 1
            hooks = list(self._hooks)
        for hook in hooks:
            try:
                hook()
            except Exception:  # noqa: BLE001 - a broken listener must not stop the robot
                pass


def reset_nodes(bus: Bus, states: dict[str, str] | None = None) -> None:
    """Put the diagram back to idle, leaving any state the caller wants to keep.

    Called between commands so the picture does not keep showing "plan: active" from a plan
    that finished thirty seconds ago - a dashboard that always looks busy is a dashboard
    nobody trusts.
    """
    keep = states or {}
    for node in FLOW:
        if node.id not in keep:
            bus.node(node.id, IDLE)
