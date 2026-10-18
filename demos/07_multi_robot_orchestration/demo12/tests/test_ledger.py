"""Demo 12 -- the task ledger across a restart.

Fast, no Isaac: this is a decision-logic module, and the decisions it makes are the ones that
would otherwise be invisible. There is no simulation here to check, deliberately -- the ledger
is tested for *what it claims*, and ``tools``/``test_grasp`` is what measures the cell.

The rule these tests exist to defend: **a task verified done must not survive a restart as
done**, because the parts are respawned and the placement it referred to is no longer a fact
about the simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo12.ledger import (LedgerEntry, OrchestratorLedger, capture)

# ``demo12.tests`` is a package, so the scripted cell comes in through it rather than by
# putting this directory on ``sys.path`` -- which is what made the import work only under
# pytest's default import mode, and broke the moment a second test module needed the class.
from demo12.tests.test_orchestrator import (  # noqa: E402
    FakeCell, _make, _scene_with)


def _ledger(**overrides) -> OrchestratorLedger:
    defaults = dict(
        robot_count=2,
        entries=[LedgerEntry(id="task_1", part_id="part_1", bin_id="bin_1")],
    )
    defaults.update(overrides)
    return OrchestratorLedger(**defaults)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_ledger_round_trips_through_json():
    original = _ledger(entries=[
        LedgerEntry(id="task_1", part_id="part_1", bin_id="bin_1", state="done",
                    attempts=2, previous_robots=["robot_1"],
                    note="finished", verified=True),
        LedgerEntry(id="task_2", part_id="part_2", bin_id="bin_2", state="queued",
                    attempts=1),
    ])

    restored = OrchestratorLedger.from_dict(original.to_dict())

    assert restored.robot_count == original.robot_count
    assert [e.id for e in restored.entries] == ["task_1", "task_2"]
    assert restored.entries[0].state == "done"
    assert restored.entries[0].attempts == 2
    assert restored.entries[0].previous_robots == ["robot_1"]
    assert restored.entries[0].verified is True


def test_ledger_saves_and_loads_a_file(tmp_path):
    path = tmp_path / "ledger.json"
    _ledger().save(path)

    loaded = OrchestratorLedger.load(path)

    assert loaded is not None
    assert loaded.entries[0].part_id == "part_1"


def test_missing_ledger_is_none_not_an_error(tmp_path):
    assert OrchestratorLedger.load(tmp_path / "nope.json") is None


def test_corrupt_ledger_is_ignored_rather_than_crashing(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{not json at all", encoding="utf-8")

    assert OrchestratorLedger.load(path) is None


def test_unknown_ledger_version_is_refused(tmp_path):
    """A format this code does not understand must not be guessed at."""
    path = tmp_path / "ledger.json"
    path.write_text('{"version": 999, "robotCount": 2, "tasks": []}', encoding="utf-8")

    assert OrchestratorLedger.load(path) is None


def test_ledger_with_a_bom_is_still_read(tmp_path):
    """PowerShell 5.1 writes a BOM with ``-Encoding UTF8``, and that must not lose a restart."""
    path = tmp_path / "ledger.json"
    path.write_text('{"version":1,"robotCount":2,"tasks":[]}', encoding="utf-8-sig")

    loaded = OrchestratorLedger.load(path)

    assert loaded is not None
    assert loaded.robot_count == 2


# ---------------------------------------------------------------------------
# The honest part
# ---------------------------------------------------------------------------


def test_verified_tasks_are_demoted_on_restart():
    ledger = _ledger(entries=[
        LedgerEntry(id="task_1", part_id="part_1", bin_id="bin_1", state="done",
                    attempts=1, verified=True),
        LedgerEntry(id="task_2", part_id="part_2", bin_id="bin_2", state="queued",
                    attempts=1),
    ])

    demoted = ledger.reset_verified()

    assert demoted == ["task_1"]
    assert ledger.entries[0].state == "queued"
    assert ledger.entries[0].verified is False
    assert "no longer a fact" in ledger.entries[0].note
    # Untouched work is left alone.
    assert ledger.entries[1].state == "queued"
    # History survives, because it is worth knowing how many attempts have been spent.
    assert ledger.entries[0].attempts == 1


def test_demoting_is_idempotent():
    ledger = _ledger(entries=[
        LedgerEntry(id="task_1", part_id="part_1", bin_id="bin_1", state="done",
                    verified=True)])

    assert ledger.reset_verified() == ["task_1"]
    assert ledger.reset_verified() == [], "a second pass should have nothing left to demote"


def test_capture_only_marks_verified_what_the_physics_agrees_with():
    """The ledger re-checks the part; it does not trust the orchestrator's own word.

    This is the whole point. An orchestrator that says ``done`` while the part is still on
    the bench must not be able to write that into a file that survives a restart.
    """
    scene = _scene_with(2, ["part_1"])
    # Scripted so the grasp succeeds: the fake moves part_1 into bin_1 when it verifies.
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)
    for _ in range(40_000):
        if not orch.work_outstanding():
            break
        orch.step()

    assert orch.tasks["task_1"].state == "done", "the fake should have completed it"
    ledger = capture(orch)

    entry = ledger.entries[0]
    assert entry.state == "done"
    assert entry.verified is True, "the part really is in the bin, so this is honest"

    # Now a ledger whose task claims done but whose part has been put back on the table.
    cell.parts["part_1"].x += 0.20
    again = capture(orch)

    assert again.entries[0].state == "done", "the orchestrator still says done"
    assert again.entries[0].verified is False, "but the part is not there any more"
    assert again.reset_verified() == ["task_1"]


def test_capture_counts_parts_actually_moved():
    scene = _scene_with(2, ["part_1", "part_2"])
    cell = FakeCell(scene, {"task_1": True, "task_2": True})
    orch = _make(scene, cell)
    for _ in range(40_000):
        if not orch.work_outstanding():
            break
        orch.step()

    ledger = capture(orch)

    assert ledger.completed_before_restart == 2
    assert ledger.parts_moved_before_restart == 2
    assert ledger.robot_count == 2


def test_capture_survives_a_cell_that_cannot_answer():
    """A ledger that cannot check is a ledger that does not claim.

    The cell is broken *after* the run, so what is under test is ``capture``'s tolerance
    rather than the orchestrator's -- a cell that cannot answer ``part_state`` at all would
    stop the orchestrator long before this point.
    """
    scene = _scene_with(2, ["part_1"])
    cell = FakeCell(scene, {"task_1": True})
    orch = _make(scene, cell)
    for _ in range(40_000):
        if not orch.work_outstanding():
            break
        orch.step()
    assert orch.tasks["task_1"].state == "done"

    def broken(part_id):
        raise RuntimeError("no such part")

    cell.part_state = broken

    ledger = capture(orch)

    assert ledger.entries
    assert ledger.entries[0].state == "done", "the record is still made"
    assert ledger.entries[0].verified is False, "but nothing is claimed for it"


def test_move_tasks_rebuilds_the_queue():
    ledger = _ledger(entries=[
        LedgerEntry(id="task_1", part_id="part_1", bin_id="bin_1", label="cube"),
        LedgerEntry(id="task_2", part_id="part_2", bin_id="bin_2", label="ball"),
    ])

    tasks = ledger.move_tasks()

    assert [t.id for t in tasks] == ["task_1", "task_2"]
    assert tasks[1].part_id == "part_2"
    assert tasks[1].label == "ball"