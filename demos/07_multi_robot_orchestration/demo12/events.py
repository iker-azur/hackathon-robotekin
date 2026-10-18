"""Demo 12 -- the collaboration event log.

**Why this module exists.** A multi-robot demo is mostly invisible. Three robots picking
three parts at once produces no obvious story: from outside you see arms move, and from the
task table you see four columns change. The interesting part -- *why* robot_2 ended up
holding task_3, how long robot_1 was idle while robot_2 was blocked on a retry, that the
replan after a fault cost 4 seconds -- happens between the columns, and nothing recorded it.

So every decision the orchestrator makes is recorded here as an event, with the simulated
time, the tick, and who was involved. The interface renders it as a timeline, and
``/api/events`` serves it for anything else that wants it.

This is a log, not a simulation. It cannot change what the physics does and it is not
consulted when judging whether a grasp worked; it exists so a human can see the
orchestration that the numbers only imply.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Event kinds. A closed vocabulary, so the interface can colour and group by it and
# anything that invents a new kind is a visible mistake rather than an unstyled row.
# ---------------------------------------------------------------------------

ASSIGN = "assign"          # a healthy idle robot took a queued task
REASSIGN = "reassign"      # a task moved between robots (fault, retry, or handoff)
RELEASE = "release"        # a task went back on the queue and nobody holds it
PHASE = "phase"            # a task changed phase (approaching -> closing -> ...)
PLAN = "plan"              # the task list changed: submitted, cleared, or edited
FAULT = "fault"            # a robot was taken out of the pool
REPAIR = "repair"          # a robot came back
RETRY = "retry"            # a task failed an attempt and went back on the queue
DONE = "done"              # a task reached DONE, with its evidence attached
FAILED = "failed"          # a task exhausted its attempts
SCENE = "scene"            # the cell or the mission was reconfigured
NOTE = "note"              # anything else worth saying

EVENT_KINDS = (ASSIGN, REASSIGN, RELEASE, PHASE, PLAN, FAULT, REPAIR,
               RETRY, DONE, FAILED, SCENE, NOTE)


@dataclass
class Event:
    """One thing the orchestrator decided, with enough context to read it cold."""

    seq: int
    tick: int
    sim_time_s: float
    kind: str
    robot_id: str | None = None
    task_id: str | None = None
    part_id: str | None = None
    bin_id: str | None = None
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "tick": self.tick,
            "simTimeS": round(self.sim_time_s, 3),
            "kind": self.kind,
            "robotId": self.robot_id,
            "taskId": self.task_id,
            "partId": self.part_id,
            "binId": self.bin_id,
            "detail": self.detail,
            "data": self.data,
        }


class EventLog:
    """A bounded, monotonically numbered log of orchestration decisions.

    Bounded because this runs for as long as the demo does. A ring keeps the recent past
    -- which is what anyone watching wants -- without a long-running demo turning into a
    slow leak. The sequence number is what lets the interface ask for "everything after
    what I already have" over the WebSocket instead of the whole log every half second.
    """

    def __init__(self, capacity: int = 4000) -> None:
        self._events: deque[Event] = deque(maxlen=capacity)
        #: The last sequence number issued. A plain int rather than an ``itertools.count``
        #: so that reading it is free. See ``next_seq`` for why that matters.
        self._seq = 0
        self._by_kind: dict[str, int] = {}

    # -- writing ---------------------------------------------------------

    def emit(self, kind: str, *, tick: int = 0, sim_time_s: float = 0.0,
             robot_id: str | None = None, task_id: str | None = None,
             part_id: str | None = None, bin_id: str | None = None,
             detail: str = "", **data: Any) -> Event:
        self._seq += 1
        event = Event(
            seq=self._seq,
            tick=tick,
            sim_time_s=sim_time_s,
            kind=kind,
            robot_id=robot_id,
            task_id=task_id,
            part_id=part_id,
            bin_id=bin_id,
            detail=detail,
            data=data,
        )
        self._events.append(event)
        self._by_kind[kind] = self._by_kind.get(kind, 0) + 1
        return event

    # -- reading ---------------------------------------------------------

    @property
    def next_seq(self) -> int:
        """The sequence number a caller should send to get everything from here on.

        Read-only, and it stays put until something is emitted.

        This used to be ``next(self._seq)`` on an ``itertools.count``, which consumed a
        number on every read. That sounds harmless and is not: the interface polls for the
        cursor on a snapshot and again on a log fetch, twenty times a second, so the cursor
        advanced continuously with no new events. A client keying its redraw on cursor
        change redrew forever, and "nothing has happened" could not be represented at all --
        which is the one thing a collaboration timeline most needs to be able to say.
        """
        return self._seq

    def since(self, seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        """Events newer than ``seq``, oldest first.

        If the caller has fallen so far behind that its cursor has been evicted from the
        ring, the answer is the oldest thing still held rather than a silent empty list --
        an empty list would read as "nothing happened", which is the one answer that is
        always wrong.
        """
        out = [e.to_dict() for e in self._events if e.seq > seq]
        return out[-limit:] if len(out) > limit else out

    def recent(self, n: int = 200) -> list[dict[str, Any]]:
        items = list(self._events)
        return [e.to_dict() for e in items[-n:]]

    def counts(self) -> dict[str, int]:
        return dict(self._by_kind)

    def clear(self) -> None:
        self._events.clear()
        self._by_kind.clear()

    def __len__(self) -> int:
        return len(self._events)