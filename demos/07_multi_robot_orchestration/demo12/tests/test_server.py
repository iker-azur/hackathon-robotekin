"""Demo 12 -- the HTTP surface.

**These tests exist because 23 routes and a WebSocket had never been run against anything.**
Everything in `server.py` was written, typechecked by the client, documented in the README,
and never executed. A route that raises on its first call is invisible until a judge clicks
the button.

The arrangement that makes this possible is the one the server already had: `build_app`
touches the simulation only through `cell.snapshot()`, and reads planning from `orch.scene`.
So a scripted cell and a real orchestrator are enough to exercise every handler, and the
whole file runs in under a second on a machine with no GPU.

What this does **not** cover, and it is worth being precise about that: the Isaac cell, the
supervisor restart, and the live WebSocket frame timing. Those need a real process. What it
does cover is the half of the surface most likely to be wrong -- the request/response shapes
the interface parses -- which is the half a judge will hit first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo12 import materials as M
from demo12.config import SceneConfig
from demo12.orchestrator import MultiRobotOrchestrator
from demo12.server import Hub, build_app
from demo12.tests.test_orchestrator import FakeCell

ROBOT_COUNT = 2


# ---------------------------------------------------------------------------
# A cell that also answers what the server asks of a cell
# ---------------------------------------------------------------------------


class ServerCell(FakeCell):
    """`FakeCell` plus the two things only the real `IsaacCell` has.

    `snapshot()` is what `/api/state`'s parts, bins and `tableTopZ` come from, and
    `sim_time_s` is what the orchestrator's clock is read from. Without these the app cannot
    be built, which would make "never run" the only available verdict.
    """

    def __init__(self, scene: SceneConfig, outcomes: dict[str, object]):
        super().__init__(scene, outcomes)
        self.sim_time_s = 0.0

    def snapshot(self) -> dict[str, object]:
        return {
            "parts": [
                {"id": p.id, "x": s.x, "y": s.y, "z": s.z, "resting": s.resting}
                for p in self.scene.parts
                for s in (self.parts[p.id],)
            ],
            "bins": [
                {"id": b.id, "x": b.x, "y": b.y, "z": M.TABLE_TOP_Z_M}
                for b in self.scene.bins
            ],
            "tableTopZ": M.TABLE_TOP_Z_M,
            "simTimeS": self.sim_time_s,
        }


def _make(client_outcomes: dict[str, object] | None = None) -> tuple[TestClient, Hub]:
    """A TestClient over the real app, on a scripted cell."""
    from demo12.config import default_scene

    scene = default_scene(ROBOT_COUNT)
    cell = ServerCell(scene, client_outcomes or {})
    orch = MultiRobotOrchestrator(cell, scene)
    cell.bind(orch)
    orch.set_tasks(scene.mission.tasks)

    hub = Hub()

    # ``build_app`` builds its own ``run_loop``, which drives the simulation on the thread
    # that owns Kit and never returns. Nothing here starts it: these tests are about the
    # request/response half, and the snapshot is published by hand where a test needs the
    # hub to hold something.
    app = build_app(cell, orch, hub, ROBOT_COUNT, headless=True,
                    frame_every=20, control_path=None)
    return TestClient(app), hub


@pytest.fixture
def client() -> TestClient:
    c, _ = _make()
    return c


def _publish(client: TestClient, hub: Hub, ticks: int = 1) -> None:
    """Run the snapshot copy the real loop runs, so the hub holds something true."""
    orch = client.app.state.orch
    cell = client.app.state.cell
    snap = orch.snapshot()
    snap["cell"] = cell.snapshot()
    hub.publish(simTimeS=snap["simTimeS"], controlTicks=snap["controlTicks"],
                tasks=snap["tasks"], robots=snap["robots"],
                tasksCompleted=snap["tasksCompleted"], tasksTotal=snap["tasksTotal"],
                parts=snap["cell"]["parts"], bins=snap["cell"]["bins"],
                tableTopZ=snap["cell"]["tableTopZ"], scene=orch.scene.to_dict(),
                metrics=snap["metrics"])


# ---------------------------------------------------------------------------
# Read endpoints: they must answer, and answer with a shape the client can parse
# ---------------------------------------------------------------------------


def test_health_reports_what_the_interface_needs_to_explain_the_cell(client):
    _publish(client, client.app.state.hub)
    body = client.get("/api/health").json()
    assert body["ok"] is True
    # The interface shows these on the header. If armCollision silently empties, the arm
    # drawings disappear and nobody is told why.
    assert body["armCollision"], "no arm bodies declared, the cell map will draw nothing"
    assert body["headless"] is True
    assert body["robots"] == ROBOT_COUNT


def test_state_carries_the_fields_the_cell_map_draws(client):
    hub = client.app.state.hub
    _publish(client, hub)
    body = client.get("/api/state").json()
    for key in ("simTimeS", "controlTicks", "parts", "bins", "tableTopZ",
                "scene", "metrics", "eventCursor"):
        assert key in body, f"/api/state is missing {key!r}, which the interface reads"


def test_scene_describes_everything_the_editor_needs_to_draw_the_cell(client):
    body = client.get("/api/scene").json()
    assert body["errors"] == [], f"the default scene should be valid, got {body['errors']}"

    assert len(body["robots"]) == ROBOT_COUNT
    for robot in body["robots"]:
        assert robot["reachMinM"] < robot["reachMaxM"]
        assert len(robot["baseXy"]) == 2

    # The editor needs each part's mass to show what a grip has to hold, and whether the
    # gripper can take it at all.
    for part in body["parts"]:
        assert part["massG"] > 0, f"{part['id']} has no mass, so the grip floor is undefined"
        assert part["graspable"] is True, f"{part['id']} is in the cell but not graspable"
    assert body["table"]["sizeM"]


def test_catalog_gives_the_command_parser_the_same_words_it_will_be_typed_in(client):
    """The colour vocabulary must cover the colours actually in the cell.

    ``parse_command`` matches typed words against ``planner.COLOR_WORDS``. The catalogue
    ships the hex values, and the interface's ``i18n.ts`` ships the words. If a part's
    colour is ever absent from the vocabulary, a command naming it fails with no clue why.
    """
    from demo12 import planner as P

    body = client.get("/api/catalog").json()
    assert body["colors"], "no colours offered"

    scene = client.get("/api/scene").json()
    for part in scene["parts"]:
        assert part["color"] in body["colors"], (
            f"{part['id']} is {part['color']}, which the catalogue does not offer")
        assert part["color"] in P.COLOR_WORDS, (
            f"{part['color']} has no names, so no command can name it")

    # Materials carry both languages, which is what the ES/EN toggle reads.
    for mat in body["materials"]:
        assert mat["labelEs"] and mat["labelEn"], f"material {mat['id']} is missing a language"


# ---------------------------------------------------------------------------
# Planning: these are pure functions of the scene, so they can be checked exactly
# ---------------------------------------------------------------------------


def test_plan_turns_moves_into_tasks_without_touching_the_queue(client):
    body = client.post("/api/plan", json={"moves": [
        {"partId": "part_1", "binId": "bin_1"},
        {"partId": "part_2", "binId": "bin_2"},
    ]}).json()

    assert body["ok"] is True, body["errors"]
    assert len(body["tasks"]) == 2
    assert {t["partId"] for t in body["tasks"]} == {"part_1", "part_2"}


def test_plan_refuses_an_impossible_move_and_says_why(client):
    """The whole point of the endpoint: grey out the move, explain it, do not accept it.

    A rejected move that only fails three minutes into a pick is the failure mode this
    replaces.
    """
    body = client.post("/api/plan", json={"moves": [
        {"partId": "part_1", "binId": "no_such_bin"},
    ]}).json()

    assert body["ok"] is False
    assert body["errors"], "a refusal with no reason is not actionable"
    assert body["tasks"] == []


def test_a_command_is_previewed_before_it_is_applied(client):
    """Default is preview. Someone typing into a box wants to see what was understood first."""
    body = client.post("/api/command", json={"text": "mueve el cubo rojo a la caja azul"}).json()

    assert body["ok"] is True, body
    assert body["moves"], "the command was understood as nothing"
    assert "applied" not in body or body["applied"] is False, \
        "a preview must not apply anything"
    # Nothing queued: the simulation thread must not have been handed work.
    assert client.app.state.hub.commands.empty(), "preview queued a command"


def test_an_applied_command_is_confirmed_truthfully(client):
    """`Queue.put` returns None. If the flag is taken from it, it is always null.

    The interface uses this to say "done", so a null here reads as a server that ignored
    the command.
    """
    hub = client.app.state.hub
    body = client.post("/api/command", json={
        "text": "part_4 -> bin_2", "apply": True}).json()

    assert body["ok"] is True, body
    assert body["applied"] is True, f"applied was {body['applied']!r}, not True"
    assert not hub.commands.empty(), "applied=True but nothing was queued"

    name, payload = hub.commands.get_nowait()
    assert name == "set_tasks"
    assert payload and payload[0]["partId"] == "part_4"


def test_apply_on_a_command_that_was_not_understood_says_so_instead_of_doing_nothing(client):
    """The awkward case: asked to apply, did not understand. Silence would look like success."""
    hub = client.app.state.hub
    body = client.post("/api/command", json={"text": "flibbertigibbet", "apply": True}).json()

    assert body["ok"] is False
    assert body["applied"] is False
    assert hub.commands.empty(), "queued work for a command it did not understand"


def test_every_scenario_builds_a_plan_that_validates(client):
    """A one-click scenario that does not validate is a button that breaks the demo."""
    body = client.get("/api/scenarios").json()
    assert body, "no scenarios offered"
    for name, scenario in body.items():
        assert scenario["errors"] == [], f"scenario {name!r}: {scenario['errors']}"
        assert scenario["tasks"], f"scenario {name!r} produced no tasks"


def test_rank_explains_the_choice_with_the_arithmetic(client):
    body = client.get("/api/tasks/rank", params={"taskId": "task_1"}).json()
    assert body["ok"] is True
    assert body["candidates"], "a task with no candidates would sit queued forever"

    best = body["candidates"][0]
    # The interface prints these, so a missing term is a number the demo cannot show.
    for key in ("reachM", "carryM", "attempts", "conflictM", "conflicted", "total"):
        assert key in best, f"the cost breakdown is missing {key!r}"

    # And it must be sorted, or "cheapest first" is a claim rather than a fact.
    totals = [c["total"] for c in body["candidates"]]
    assert totals == sorted(totals)


def test_rank_of_an_unknown_task_is_reported_not_raised(client):
    """A 500 here would leave the interface spinning instead of saying "no such task"."""
    body = client.get("/api/tasks/rank", params={"taskId": "task_nope"}).json()
    assert body["ok"] is False
    assert body["errors"]


# ---------------------------------------------------------------------------
# The queue: the HTTP thread only ever *asks*, and the tests check that it asks correctly
# ---------------------------------------------------------------------------


def test_submit_queues_a_validated_plan(client):
    hub = client.app.state.hub
    body = client.post("/api/tasks/submit", json={"moves": [
        {"partId": "part_3", "binId": "bin_2"}]}).json()

    assert body["ok"] is True, body
    assert body["tasksSubmitted"]
    name, payload = hub.commands.get_nowait()
    assert name == "set_tasks"
    assert payload[0]["partId"] == "part_3"


def test_submit_refuses_a_bad_move_and_queues_nothing(client):
    hub = client.app.state.hub
    body = client.post("/api/tasks/submit", json={"moves": [
        {"partId": "part_1", "binId": "nope"}]}).json()

    assert body["ok"] is False
    assert hub.commands.empty(), "queued a plan it had just reported as invalid"


def test_clear_and_run_and_stop_are_commands_not_actions(client):
    """They are queued, because the objects they touch belong to the thread that owns Kit."""
    hub = client.app.state.hub
    for path, expected in (("/api/tasks/clear", "clear_tasks"),
                           ("/api/run", "run"),
                           ("/api/stop", "stop")):
        body = client.post(path).json()
        assert body["ok"] is True, path
        assert hub.commands.get_nowait()[0] == expected, path


def test_fault_and_repair_are_commands(client):
    hub = client.app.state.hub
    client.post("/api/fault/robot_1")
    assert hub.commands.get_nowait() == ("fault", "robot_1")

    client.post("/api/repair/robot_1")
    assert hub.commands.get_nowait() == ("repair", "robot_1")


# ---------------------------------------------------------------------------
# Faults, events, metrics
# ---------------------------------------------------------------------------


def test_events_are_cursor_addressed_so_a_polling_client_sees_each_one_once(client):
    first = client.get("/api/events", params={"since": 0}).json()
    assert "events" in first and "cursor" in first and "counts" in first
    cursor = first["cursor"]
    assert first["events"], "the orchestrator logs its plan on startup; nothing was recorded"

    # Nothing new: the cursor must not move. It used to, on every read, because
    # ``next_seq`` consumed the counter -- so a client keying its redraw on cursor change
    # redrew forever and "nothing has happened" could not be represented.
    second = client.get("/api/events", params={"since": cursor}).json()
    assert second["events"] == []
    assert second["cursor"] == cursor, "the cursor moved with no new event"

    # And an event that arrives afterwards is delivered exactly once.
    orch = client.app.state.orch
    orch.events.emit("test", detail="something happened")
    third = client.get("/api/events", params={"since": cursor}).json()
    assert [e["detail"] for e in third["events"]] == ["something happened"]
    assert third["cursor"] > cursor

    # Re-reading the same cursor must not redeliver it.
    again = client.get("/api/events", params={"since": cursor}).json()
    assert [e["detail"] for e in again["events"]] == ["something happened"]


def test_metrics_reports_the_numbers_the_demo_quotes(client):
    body = client.get("/api/metrics").json()
    for key in ("tasksCompleted", "tasksTotal", "controlTicks", "simTimeS",
                "robotUtilisation", "criteriaPassed"):
        assert key in body, f"/api/metrics is missing {key!r}"


def test_roster_explains_why_robots_cannot_be_added_live(client):
    """This is an interface button that cannot work. It has to say why, not just fail."""
    body = client.get("/api/roster").json()
    assert body["current"] == []
    assert body["inProcessRebuild"] is False
    assert body["reason"], "a refusal with no reason is a dead button"
    assert "control.json" in body["reason"] or "launcher" in body["reason"]


def test_scene_validate_accepts_the_default_scene_and_rejects_a_damaged_one(client):
    scene = client.get("/api/scene").json()
    payload = {k: scene[k] for k in ("name", "table", "robots", "parts", "bins", "tasks")}

    good = client.post("/api/scene/validate", json=payload).json()
    assert good["ok"] is True, good["errors"]

    # Put a part where no arm can reach. The editor calls this on every keystroke, so it has
    # to refuse while the person is still holding the scene.
    broken = json.loads(json.dumps(payload))
    broken["parts"][0]["x"] = 1.5
    bad = client.post("/api/scene/validate", json=broken).json()
    assert bad["ok"] is False
    assert bad["errors"], "refused without saying which part or why"


def test_scene_apply_refuses_honestly_when_there_is_no_supervisor(client):
    """Started without the launcher, the server cannot restart itself. It must say that."""
    body = client.post("/api/scene/apply", json=client.get("/api/scene").json()).json()
    assert body["ok"] is False
    assert "launcher" in body["reason"], body


def test_frame_is_a_503_with_a_reason_rather_than_a_placeholder_image(client):
    """A stub frame would let the interface look healthy while nothing renders."""
    response = client.get("/api/frame.jpg")
    assert response.status_code == 503
    assert response.json()["error"]


# ---------------------------------------------------------------------------
# The WebSocket: the one endpoint whose failure is invisible from outside
# ---------------------------------------------------------------------------


def test_relaunch_plan_writes_the_ledger_now_not_sometimes(tmp_path):
    """The supervisor calls this and then kills the process. So it has to actually write.

    Measured against a live cell: the launcher posts here, the handler answered
    ``{"ok": true, "queued": "relaunch_plan"}``, ``terminate()`` landed before ``run_loop``
    drained the queue, and ``out12/ledger.json`` was never written -- so the restart came back
    with an empty queue, looking exactly like a successful one.

    ``saved`` is therefore only ``True`` when the file exists afterwards, and the launcher
    checks it before killing anything.
    """
    ledger = tmp_path / "ledger.json"
    from demo12.server import build_app as _build_app

    scene_client, hub = _make()
    orch = scene_client.app.state.orch
    cell = scene_client.app.state.cell

    app = _build_app(cell, orch, hub, ROBOT_COUNT, headless=True, frame_every=20,
                     control_path=None, ledger_path=ledger)
    live = TestClient(app)

    body = live.post("/api/relaunch-plan").json()

    assert body["ok"] is True
    assert body["saved"] is True, body
    assert ledger.is_file(), "it reported saved:true and the file is not there"

    # And the file is a ledger, not a stub.
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["tasks"], "an empty ledger would restore an empty queue"
    assert {"partId", "binId"} <= set(saved["tasks"][0]), (
        f"the ledger entry shape is {sorted(saved['tasks'][0])}, and restore reads partId")


def test_relaunch_plan_refuses_honestly_with_no_ledger_path():
    """Started with ``--no-ledger`` there is nothing to save, and saying so beats pretending."""
    from demo12.server import build_app as _build_app

    scene_client, hub = _make()
    app = _build_app(scene_client.app.state.cell, scene_client.app.state.orch, hub,
                     ROBOT_COUNT, headless=True, frame_every=20,
                     control_path=None, ledger_path=None)

    body = TestClient(app).post("/api/relaunch-plan").json()
    assert body["ok"] is False
    assert body["saved"] is False
    assert "no-ledger" in body["reason"] or "nothing to save" in body["reason"]


def test_the_websocket_connects_and_streams_state(client):
    """A failed upgrade and a healthy server look identical from the browser.

    The interface sits on "sin conexion" while REST and the camera image both look fine,
    which is exactly how the FastAPI-route bug presented. Asserting that the socket opens
    and yields parseable state is the only way that failure cannot reach a judge.
    """
    with client.websocket_connect("/ws") as socket:
        message = socket.receive_json()

    assert message["controlTicks"] >= 0
    # The client's ``api.ts`` reads these names; a rename on either side shows up here.
    assert "running" in message
    assert "eventCursor" in message
    assert "scene" in message