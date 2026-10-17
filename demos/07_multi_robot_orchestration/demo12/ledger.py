"""Demo 12 -- the task ledger, and what survives a restart.

The roster cannot change inside a running Isaac cell (see ``docs/PHYSICS.md`` section 9 --
the in-process rebuild hangs), so changing it means restarting the process. A restart throws
away the simulation, so this module exists to answer one question precisely: **what carries
over, and what does not.**

Carries over, because it lives in the orchestrator and not in the physics:

* the task queue, and each task's state
* attempt counts and the previous-owner history
* task notes, which are the only record of *why* something failed

Does **not** carry over, because it was never stored anywhere:

* every part's position. The cell is respawned, so a part is back at its configured spot.
  A task that was verified DONE is still marked DONE, and that is now a claim the simulation
  no longer supports -- so :meth:`OrchestratorLedger.reset_verified` demotes them.
* any robot that was mid-grasp, mid-lift, or holding anything.

That asymmetry is the whole reason ``reset_verified`` exists. A restart that silently
preserved the word "done" would produce exactly the kind of claim this demo refuses to make
elsewhere: a task reported complete on the strength of a physical fact that no longer exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import MoveTask

LEDGER_VERSION = 1


@dataclass
class LedgerEntry:
    """One task as it will be after a restart."""

    id: str
    part_id: str
    bin_id: str
    label: str = ""
    state: str = "queued"
    attempts: int = 0
    previous_robots: list[str] = field(default_factory=list)
    note: str = ""
    #: True only if the part is currently verifiably in its bin. Cleared by a restart.
    verified: bool = False

    def to_move_task(self) -> MoveTask:
        return MoveTask(id=self.id, part_id=self.part_id, bin_id=self.bin_id,
                        label=self.label)


@dataclass
class OrchestratorLedger:
    """Everything worth keeping across a process restart."""

    robot_count: int
    entries: list[LedgerEntry]
    completed_before_restart: int = 0
    parts_moved_before_restart: int = 0

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LEDGER_VERSION,
            "robotCount": self.robot_count,
            "completedBeforeRestart": self.completed_before_restart,
            "partsMovedBeforeRestart": self.parts_moved_before_restart,
            "tasks": [
                {
                    "id": e.id,
                    "partId": e.part_id,
                    "binId": e.bin_id,
                    "label": e.label,
                    "state": e.state,
                    "attempts": e.attempts,
                    "previousRobots": list(e.previous_robots),
                    "note": e.note,
                    "verified": e.verified,
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestratorLedger":
        if data.get("version") != LEDGER_VERSION:
            raise ValueError(
                f"ledger version {data.get('version')!r} is not "
                f"{LEDGER_VERSION}; this demo will not guess at a format it cannot read")
        return cls(
            robot_count=int(data.get("robotCount", 1)),
            completed_before_restart=int(data.get("completedBeforeRestart", 0)),
            parts_moved_before_restart=int(data.get("partsMovedBeforeRestart", 0)),
            entries=[
                LedgerEntry(
                    id=str(t["id"]),
                    part_id=str(t["partId"]),
                    bin_id=str(t["binId"]),
                    label=str(t.get("label", "")),
                    state=str(t.get("state", "queued")),
                    attempts=int(t.get("attempts", 0)),
                    previous_robots=list(t.get("previousRobots", [])),
                    note=str(t.get("note", "")),
                    verified=bool(t.get("verified", False)),
                )
                for t in data.get("tasks", [])
            ],
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "OrchestratorLedger | None":
        """Read a ledger, tolerating a BOM.

        ``utf-8-sig`` rather than ``utf-8`` because this file is written and read from
        PowerShell on Windows, and PowerShell 5.1's ``-Encoding UTF8`` writes a BOM. Reading
        it as plain UTF-8 fails, and a restart that silently finds no ledger quietly reverts
        to the default scene -- which is exactly what happened before this was caught by a
        test rather than in production.
        """
        if not path.is_file():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
        except (ValueError, KeyError, json.JSONDecodeError):
            return None

    # -- semantics ------------------------------------------------------

    def reset_verified(self) -> list[str]:
        """Demote every DONE task. Returns the ids that were demoted.

        Called on load. The parts are about to be respawned at their starting positions, so a
        task that was verified done is no longer true -- and leaving it as ``done`` would be
        the one thing in this demo that reports success it has not measured.
        """
        demoted: list[str] = []
        for entry in self.entries:
            if entry.state != "done":
                continue
            entry.state = "queued"
            entry.verified = False
            entry.note = ("requeued after a restart: the part was respawned, so the "
                          "earlier placement is no longer a fact about the simulation")
            demoted.append(entry.id)
        return demoted

    def move_tasks(self) -> list[MoveTask]:
        return [e.to_move_task() for e in self.entries]


def capture(orch) -> OrchestratorLedger:
    """Read the orchestrator's live state into a ledger.

    Whether a task counts as ``verified`` is asked of the physics, not remembered from the
    task's own state: the part has to be inside its bin and at rest. A task the orchestrator
    marked done whose part is not actually in the bin does not get to survive a restart as
    done.
    """
    entries: list[LedgerEntry] = []
    moved = 0

    for task in orch.tasks.values():
        verified = False
        try:
            bin_cfg = orch.cell.bins.get(task.bin_id)
            part_cfg = orch.cell.part_cfgs.get(task.part_id)
            if bin_cfg is not None and part_cfg is not None:
                state = orch.cell.part_state(task.part_id)
                inside = bin_cfg.contains_footprint(
                    state.x, state.y, part_cfg.footprint_radius_m)
                verified = bool(inside and state.resting)
                if verified:
                    moved += 1
        except Exception:                          # noqa: BLE001
            # A ledger that cannot check is a ledger that does not claim.
            verified = False

        entries.append(LedgerEntry(
            id=task.id,
            part_id=task.part_id,
            bin_id=task.bin_id,
            label=task.label,
            state=task.state,
            attempts=task.attempts,
            previous_robots=list(task.previous_robots),
            note=task.note,
            verified=verified,
        ))

    return OrchestratorLedger(
        robot_count=len(orch.robots),
        entries=entries,
        completed_before_restart=len([t for t in orch.tasks.values()
                                      if t.state == "done"]),
        parts_moved_before_restart=moved,
    )