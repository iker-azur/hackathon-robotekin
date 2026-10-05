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
    Node("setup", "Environment setup",
         "variant, seed, camera, real materials, preflight", 0),
    Node("input", "Command",
         "spoken or typed, from the dashboard or the terminal", 1),
    Node("understand", "Command model",
         "sentence to intent, items and destination", 2),
    Node("capture", "Camera",
         "render one frame and read the sensor buffers", 3),
    Node("perceive", "Perception",
         "colour blobs to kinds, areas and table positions", 4),
    Node("link", "Link",
         "each perceived blob matched to the part it is", 5),
    Node("ground", "Grounding",
         "resolve the reference, ask when it is ambiguous, refuse what is out of reach", 6),
    Node("plan", "Planning",
         "poses and jaw commands, plus what could not be planned", 7),
    Node("execute", "Execution",
         "step by step, re-aiming from the camera while the arm moves", 8),
    Node("verify", "Verification",
         "did the part come to rest inside the tray it was sent to", 9),
    Node("recover", "Retry / re-plan",
         "re-look and try again, or refuse to fake a grasp", 10),
    Node("report", "Report",
         "verified, retried, missed, and what the round cost", 11),
)

# (from, to, label). The spine plus the two edges that make the recovery loop a loop.
EDGES: tuple[tuple[str, str, str], ...] = (
    ("setup", "input", ""),
    ("input", "understand", "sentence"),
    ("understand", "capture", "look"),
    ("capture", "perceive", "frame"),
    ("perceive", "link", "blobs"),
    ("link", "ground", "parts"),
    ("ground", "plan", "targets"),
    ("plan", "execute", "steps"),
    ("execute", "verify", "motion"),
    ("verify", "recover", "not inside"),
    ("recover", "capture", "re-look"),
    ("recover", "plan", "re-plan"),
    ("recover", "report", "gave up"),
    ("verify", "report", "all inside"),
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
        self._commands: queue.Queue[str] = queue.Queue()
        self._controls: queue.Queue[dict] = queue.Queue()
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

    def snapshot(self) -> dict:
        """Everything a freshly connected dashboard needs, in one JSON-ready dict."""
        with self._lock:
            return {
                "seq": self._seq,
                "t": time.time(),
                "nodes": [dict(box) for box in self._nodes.values()],
                "edges": [list(edge) for edge in EDGES],
                "state": dict(self._state),
                "events": list(self._events)[-140:],
            }

    # -- inbound ---------------------------------------------------------------

    def submit(self, text: str) -> None:
        """Queue a command from the dashboard."""
        text = (text or "").strip()
        if text:
            self._commands.put(text)

    def take_command(self, timeout: float | None = None) -> str | None:
        """The next queued command, or None if none arrived in time."""
        try:
            return self._commands.get(timeout=timeout) if timeout else \
                self._commands.get_nowait()
        except queue.Empty:
            return None

    def send_control(self, action: str, **params) -> None:
        """Queue a control action: re-look, next round, save frame, stop."""
        self._controls.put({"action": action, "params": params})

    def take_control(self, timeout: float | None = None) -> dict | None:
        try:
            return self._controls.get(timeout=timeout) if timeout else \
                self._controls.get_nowait()
        except queue.Empty:
            return None

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
        with self._lock:
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