"""Demo 12 -- FastAPI over a live Isaac Lab cell.

**Why this file runs the HTTP server on its own thread.** Kit installs and pumps its own
asyncio event loop (``omni.kit.async_engine`` calls ``loop.run_once()`` from Kit's update
events). ``uvicorn.run()`` then tries to run a *second* loop on the same thread and the process
dies with::

    RuntimeError: Cannot run the event loop while another loop is running

That failure is quiet and misleading. Plain REST requests keep working -- Kit's pump happens to
service them -- so ``/api/health`` answers 200 and ``/api/frame.jpg`` serves a JPEG, while
every WebSocket upgrade is rejected with ``403 Forbidden`` and the interface sits there saying
"sin conexión" forever. Measured, and it cost a while to find because everything *except* the
thing that was broken looked fine.

So the two halves are split by thread, and neither touches the other's world:

* **Main thread**: Kit, PhysX, the camera, the orchestration. A plain synchronous loop -- no
  asyncio at all, which is what the probes do and what demonstrably renders.
* **HTTP thread**: uvicorn on its own event loop. It never imports or calls anything in
  Isaac. It reads a snapshot the main thread published and posts commands back.

The only thing crossing between them is a snapshot dict and a command queue, both
thread-safe. A command posted by the UI is *queued* and applied by the main loop, so no HTTP
handler ever mutates the simulation from the wrong thread -- which is the same main-thread rule
that governs rendering, arrived at from the other direction.

What this server will not do, because either would make the demo lie:

- **It will not catch and swallow orchestration errors.** An earlier version wrapped the step
  in a bare ``except Exception: pass``, so a cell broken for an hour still served cheerful
  "everything is fine" state.
- **It will not serve a placeholder frame.** ``/api/frame.jpg`` returns a real render or a
  503 that says why.

Run::

    python -u -m demo12.server --robots 2

Then open the Vite dev server (see ``web/``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
# parents[0] = demo12, [1] = 07_multi_robot_orchestration, [2] = demos
_DEMO_ROOT = _HERE.parents[0]
_DEMOS_ROOT = _HERE.parents[2]

for _p in (str(_HERE.parents[1]), str(_DEMOS_ROOT / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402


class Hub:
    """The only thing shared between the simulation thread and the HTTP thread.

    A plain dict behind a lock, and a queue. Not a framework and not a message bus -- the
    simulation is the authority, the HTTP thread is a reader with a way to ask for things, and
    anything more elaborate would make it look like the two halves share state they do not.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "simTimeS": 0.0,
            "controlTicks": 0,
            "steps": 0,
            "running": True,
            "startedAt": time.time(),
            "lastError": None,
            "errorCount": 0,
            "frameError": "no frame captured yet",
            "frameNote": "",
            "frameEvery": 20,
            "roster": [],
            "tasks": [],
            "robots": [],
            "tasksCompleted": 0,
            "tasksTotal": 0,
            "allTasksCompleted": False,
            "failuresInjected": 0,
            "reassignments": 0,
            "rosterChanges": 0,
            "parts": [],
            "bins": [],
            "tableTopZ": 0.0,
            "scene": None,
            "metrics": {},
            "eventCursor": 0,
            "eventCounts": {},
            "assignmentCosts": {},
            "ledger": {"restored": False},
        }
        self._frame: bytes | None = None
        self.commands: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    def publish(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def set_frame(self, data: bytes | None, note: str, error: str = "") -> None:
        with self._lock:
            if data:
                self._frame = data
            self._state["frameNote"] = note
            self._state["frameError"] = error

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame


def build_app(cell, orch, hub: Hub, robot_count: int, headless: bool,
              frame_every: int, control_path: "Path | None" = None,
              ledger_path: "Path | None" = None):
    """Build the FastAPI app. Every endpoint reads the hub or queues a command.

    Nothing here touches the simulation. ``cell`` is accepted only so the frame endpoint can
    report why it is unavailable; the bytes come from the hub, written by the main thread.

    ``ledger_path`` is where ``/api/relaunch-plan`` writes, and it is passed in rather than
    derived so the endpoint and the shutdown path cannot disagree about the file.
    """
    from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from starlette.routing import WebSocketRoute

    from . import planner as P
    from .config import MoveTask, SceneConfig, SceneError, catalog, default_scene
    from .isaac_env import ARM_BODIES_WITH_COLLISION

    app = FastAPI(title="Demo 12 -- multi-robot orchestration")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                       "http://localhost:4173", "http://127.0.0.1:4173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _command(name: str, payload: Any = None) -> dict[str, Any]:
        """Queue a command for the simulation thread.

        Queued rather than executed, because the simulation object belongs to the thread that
        owns Kit. Executing it here would be the wrong-thread bug this whole arrangement
        exists to avoid.
        """
        hub.commands.put((name, payload))
        return {"ok": True, "queued": name}

    # ---------------------------------------------------------------- the loop

    def publish_snapshot() -> None:
        """Copy the simulation's state into the hub. Main thread only."""
        snap = orch.snapshot()
        snap["cell"] = cell.snapshot()
        hub.publish(
            simTimeS=snap["simTimeS"],
            controlTicks=snap["controlTicks"],
            tasks=snap["tasks"],
            robots=snap["robots"],
            tasksCompleted=snap["tasksCompleted"],
            tasksTotal=snap["tasksTotal"],
            allTasksCompleted=snap["allTasksCompleted"],
            failuresInjected=snap["failuresInjected"],
            reassignments=snap["reassignments"],
            rosterChanges=snap["rosterChanges"],
            parts=snap["cell"].get("parts", []),
            bins=snap["cell"].get("bins", []),
            tableTopZ=snap["cell"].get("tableTopZ"),
            steps=hub.state()["steps"],
            scene=orch.scene.to_dict(),
            metrics=snap["metrics"],
            eventCursor=snap["eventCursor"],
            eventCounts=snap["eventCounts"],
        )

    def drain_commands() -> None:
        """Apply everything the interface asked for. Main thread only."""
        while True:
            try:
                name, payload = hub.commands.get_nowait()
            except queue.Empty:
                return
            try:
                if name == "fault":
                    result = orch.inject_failure(payload)
                elif name == "fault_all":
                    result = orch.fault_all(payload)
                elif name == "repair":
                    result = orch.repair_robot(payload)
                elif name == "run":
                    result = {"ok": True, "running": True}
                    hub.publish(running=True)
                elif name == "stop":
                    result = {"ok": True, "running": False}
                    hub.publish(running=False)
                elif name == "relaunch_plan":
                    result = orch.release_for_relaunch()
                elif name == "set_tasks":
                    # Replaces the queue live. No rebuild: a task is orchestration state, so
                    # it can change while the physics keeps running.
                    result = orch.set_tasks(
                        [MoveTask(id=str(t["id"]), part_id=str(t["partId"]),
                                  bin_id=str(t["binId"]), label=str(t.get("label") or ""))
                         for t in payload])
                elif name == "clear_tasks":
                    result = orch.clear_tasks()
                else:
                    result = {"ok": False, "reason": f"unknown command {name!r}"}
                hub.publish(lastResult=result)
            except Exception as exc:               # noqa: BLE001
                hub.publish(lastError=f"{type(exc).__name__}: {exc}")

    def run_loop() -> None:
        """The whole simulation. Synchronous, on the thread that owns Kit.

        No asyncio here, deliberately: Kit pumps its own loop, and adding a second one is what
        broke the WebSocket. The camera renders through ``cell.step``, which drives rendering
        itself, so nothing is lost by not having an event loop here.
        """
        interval = 1.0 / 20.0
        frame_countdown = frame_every
        started = time.time()
        hub.publish(startedAt=started, running=True)

        while True:
            tick_started = time.time()
            drain_commands()
            state = hub.state()

            try:
                if state["running"] and orch.work_outstanding():
                    orch.step()
                hub.publish(
                    simTimeS=orch.sim_time_s,
                    controlTicks=orch.control_ticks,
                    steps=state["steps"] + 1,
                    lastError=None,
                )
            except Exception as exc:               # noqa: BLE001
                # Recorded and surfaced. The previous version passed here and reported success
                # forever.
                hub.publish(lastError=f"{type(exc).__name__}: {exc}",
                            errorCount=state["errorCount"] + 1)
                print(f"[demo12] orchestration error: {exc}", flush=True)

            publish_snapshot()

            frame_countdown -= 1
            if frame_countdown <= 0:
                frame_countdown = frame_every
                # Encoding a JPEG is milliseconds, but it has to happen here regardless: the
                # camera is a CUDA resource owned by this thread.
                data, note = cell.render_frame_jpeg()
                hub.set_frame(data, note if data else "", "" if data else note)

            elapsed = time.time() - tick_started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    # ------------------------------------------------------------- endpoints

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        state = hub.state()
        return {
            "ok": state["lastError"] is None,
            "running": state["running"],
            "steps": state["steps"],
            "uptimeS": round(time.time() - state["startedAt"], 1),
            "errorCount": state["errorCount"],
            "lastError": state["lastError"],
            "robots": len(state["robots"]),
            "armCollision": list(ARM_BODIES_WITH_COLLISION),
            "headless": headless,
        }

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        return JSONResponse(hub.state())

    @app.get("/api/catalog")
    def get_catalog() -> JSONResponse:
        return JSONResponse(catalog())

    # ---------------------------------------------------------------- planning
    #
    # Everything below is read-only with respect to the physics. The scene editor, the
    # piece-to-box picker and the command bar all live here rather than in the simulation
    # loop, because changing what *should* happen is a decision and changing *what is*
    # happening is PhysX's job.

    @app.get("/api/scene")
    def get_scene() -> JSONResponse:
        """The cell as configured, for the editor.

        Includes everything the interface needs to draw the scene and everything it needs
        to explain a rejected move: each part's measured mass and whether the gripper can
        take it at all, each tray's real cavity size, and each robot's reach annulus.
        """
        scene = orch.scene
        robots = []
        for r in scene.robots:
            envelope = r.reach
            robots.append({
                "id": r.id,
                "label": r.label,
                "baseXy": list(r.base_xy),
                "baseYawDeg": r.base_yaw_deg,
                "enabled": r.enabled,
                # ``ReachEnvelope`` spells these ``min_reach_m`` / ``max_reach_m``. Reading
                # ``min_m`` here made this endpoint raise AttributeError -- a 500 on the scene
                # editor, discovered only by running it.
                "reachMinM": envelope.min_reach_m,
                "reachMaxM": envelope.max_reach_m,
            })
        return JSONResponse({
            "name": scene.name,
            "table": {"sizeM": list(scene.table.size_m),
                      "centerXy": list(scene.table.center_xy)},
            "robots": robots,
            "parts": scene.to_dict()["parts"],
            "bins": scene.to_dict()["bins"],
            "tasks": scene.to_dict()["mission"]["tasks"],
            "errors": scene.collect_errors(),
        })

    @app.post("/api/scene/validate")
    async def validate_scene(payload: dict[str, Any]) -> JSONResponse:
        """Check a proposed scene without building it.

        The editor calls this on every edit, so a layout that cannot be built is rejected
        with the reason while the person is still holding it, rather than three minutes
        later inside a simulator boot. Nothing is spawned and nothing is restarted.
        """
        try:
            proposed = SceneConfig.from_dict(payload)
        except SceneError as exc:
            return JSONResponse({"ok": False, "errors": [str(exc)]}, status_code=200)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]},
                                status_code=200)
        return JSONResponse({"ok": True, "errors": [],
                             "scene": proposed.to_dict()})

    @app.post("/api/scene/apply")
    def apply_scene(payload: dict[str, Any]) -> JSONResponse:
        """Save a scene and ask the supervisor to restart onto it.

        Isaac Lab cannot add a part, a tray or an arm to a live scene -- the in-process
        rebuild was measured to hang at 370 CPU-seconds without returning (PHYSICS.md
        section 9) -- so changing the *geometry* means a new process. The supervisor owns
        that, and it is polled through a file rather than an HTTP call precisely because the
        process being asked to do the work is the one that has to die.

        The task queue survives across that restart via the ledger; part positions do not,
        so anything marked done is re-checked against the physics and requeued if it no
        longer holds.
        """
        if control_path is None:
            return JSONResponse({
                "ok": False,
                "reason": ("this server was started without the supervisor, so it cannot "
                           "restart itself onto a new scene. Run demo12.launcher instead."),
            }, status_code=200)
        try:
            proposed = SceneConfig.from_dict(payload)
        except (SceneError, KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"ok": False, "errors": [str(exc)]}, status_code=200)

        enabled = len(proposed.enabled_robots)
        try:
            control_path.parent.mkdir(parents=True, exist_ok=True)
            control_path.write_text(
                json.dumps({"robots": enabled, "scene": "scene.json"}),
                encoding="utf-8")
        except OSError as exc:
            return JSONResponse({"ok": False, "errors": [f"could not write the control "
                                                         f"request: {exc}"]},
                                status_code=200)

        # Save the scene where the supervisor will read it on the way back up, so the
        # relaunch is not racing this response.
        scene_file = control_path.parent / "scene.json"
        try:
            scene_file.write_text(json.dumps(proposed.to_dict(), indent=2),
                                  encoding="utf-8")
        except OSError as exc:
            return JSONResponse({"ok": False, "errors": [f"could not write the scene: {exc}"]},
                                status_code=200)

        return JSONResponse({
            "ok": True,
            "restarting": True,
            "robots": enabled,
            "parts": len(proposed.parts),
            "bins": len(proposed.bins),
            "note": ("the supervisor is stopping this process and starting a new one with "
                     "this scene. The task queue is saved first; part positions are not."),
        })

    @app.post("/api/plan")
    def post_plan(payload: dict[str, Any]) -> JSONResponse:
        """Turn ``[{partId, binId}]`` into a validated task list, without applying it.

        Returns the tasks *and* the reasons for any that were refused, so the interface can
        grey out an impossible move and explain it instead of accepting it and failing three
        minutes into a pick.
        """
        moves = [(str(m["partId"]), str(m["binId"])) for m in payload.get("moves", [])
                 if m.get("partId") and m.get("binId")]
        built = P.build_plan(orch.scene, moves)
        return JSONResponse({
            "ok": built["ok"],
            "errors": built["errors"],
            "tasks": [t.to_dict() for t in built["tasks"]],
        })

    @app.post("/api/command")
    def post_command(payload: dict[str, Any]) -> JSONResponse:
        """Read one typed line and return the plan it names.

        ``apply: true`` also replaces the queue. The default is preview, because someone
        typing into a box wants to see what was understood before the cell starts moving.
        """
        text = str(payload.get("text") or "")
        parsed = P.parse_command(text, orch.scene)
        body = parsed.to_dict()
        body["ok"] = bool(parsed.moves) and not parsed.errors
        if payload.get("apply"):
            if body["ok"]:
                built = P.build_plan(orch.scene, parsed.moves)
                body["plan"] = {"ok": built["ok"], "errors": built["errors"],
                                "tasks": [t.to_dict() for t in built["tasks"]]}
                if built["ok"]:
                    # ``Queue.put`` returns None, so the truth has to be stated rather than
                    # taken from its return value -- otherwise this field is always ``null``
                    # and the interface can never confirm what it just sent.
                    _command("set_tasks", [t.to_dict() for t in built["tasks"]])
                    body["applied"] = True
            else:
                # Ask for apply, did not understand it: say so, do not silently do nothing.
                body["applied"] = False
                body["errors"] = list(body.get("errors") or []) + [
                    "not applied: the line was not understood. Preview it first."]
        return JSONResponse(body)

    @app.get("/api/scenarios")
    def get_scenarios() -> JSONResponse:
        """One-click assignments, so the demo is one click from something interesting."""
        out = {}
        for name in P.SCENARIOS:
            moves = P.scenario_moves(orch.scene, name)
            built = P.build_plan(orch.scene, moves)
            out[name] = {
                "moves": [{"partId": p, "binId": b} for p, b in moves],
                "tasks": [t.to_dict() for t in built["tasks"]],
                "errors": built["errors"],
                **P.SCENARIOS[name],
            }
        return JSONResponse(out)

    @app.get("/api/tasks/rank")
    def rank_robots(task_id: str = Query(..., alias="taskId")) -> JSONResponse:
        """Which robot would take this task, and why, cheapest first.

        ``alias="taskId"`` is load-bearing. The handler parameter is ``task_id`` because
        that is what Python calls things, but FastAPI binds query parameters *by the
        parameter name*, so the endpoint demanded ``?task_id=``. The client sends
        ``?taskId=``, matching every other name in this API and the README -- so every call
        returned 422 with a FastAPI validation body, and the "why this robot" panel could
        never have opened. The Python name stays idiomatic and the wire name matches the
        rest of the interface.
        """
        task = orch.tasks.get(task_id)
        if task is None:
            return JSONResponse({"ok": False, "errors": [f"no task {task_id!r}"]})
        return JSONResponse({"ok": True, "taskId": task_id,
                             "candidates": orch.rank_robots(task)})

    @app.post("/api/tasks")
    def set_tasks(payload: dict[str, Any]) -> dict[str, Any]:
        """Replace the queue live. See ``_command``: applied by the simulation thread."""
        return _command("set_tasks", payload.get("tasks") or [])

    @app.post("/api/tasks/clear")
    def clear_tasks() -> dict[str, Any]:
        return _command("clear_tasks")

    @app.post("/api/tasks/submit")
    def submit_tasks(payload: dict[str, Any]) -> dict[str, Any]:
        """Build a plan from moves and apply it, in one request."""
        moves = [(str(m["partId"]), str(m["binId"])) for m in payload.get("moves", [])
                 if m.get("partId") and m.get("binId")]
        built = P.build_plan(orch.scene, moves)
        if not built["ok"]:
            return {"ok": False, "errors": built["errors"]}
        _command("set_tasks", [t.to_dict() for t in built["tasks"]])
        return {"ok": True, "tasksSubmitted": [t.id for t in built["tasks"]]}

    @app.get("/api/events")
    def get_events(since: int = 0, limit: int = 300) -> JSONResponse:
        """The collaboration log. Pass the last ``seq`` you saw to get only what is new."""
        return JSONResponse({"events": orch.events.since(since, limit),
                             "cursor": orch.events.next_seq,
                             "counts": orch.events.counts()})

    @app.get("/api/metrics")
    def get_metrics() -> JSONResponse:
        """Headline numbers for the demo, all measured over the run so far."""
        return JSONResponse(orch.metrics())

    @app.get("/api/frame.jpg")
    def get_frame() -> Response:
        data = hub.frame()
        if data is None:
            # 503 with the reason, not a placeholder image. A stub frame would let the
            # interface look healthy while nothing is being rendered.
            return JSONResponse({"error": hub.state()["frameError"]}, status_code=503)
        return Response(content=data, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/fault/{robot_id}")
    def fault(robot_id: str) -> dict[str, Any]:
        return _command("fault", robot_id)

    @app.post("/api/faults")
    def fault_many(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Take several arms out of service at once.

        Refuses to fault every robot: that leaves nothing able to advance the queue, and a
        cell that quietly stops rather than says so is worse than one that explains why the
        button did nothing.
        """
        ids = (payload or {}).get("robotIds")
        return _command("fault_all", list(ids) if ids else None)

    @app.post("/api/repair/{robot_id}")
    def repair(robot_id: str) -> dict[str, Any]:
        return _command("repair", robot_id)

    @app.get("/api/roster")
    def roster() -> dict[str, Any]:
        return {
            "current": [r["id"] for r in hub.state()["robots"]],
            "inProcessRebuild": False,
            "reason": ("Isaac Lab cannot insert an articulation into a live scene, and the "
                       "in-process cell rebuild hangs on this build. Use the launcher: write "
                       "{'robots': N} to out12/control.json."),
            "supportedRoute": "demo12.launcher restarts the process with a new --robots",
        }

    @app.post("/api/run")
    def start() -> dict[str, Any]:
        return _command("run")

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        return _command("stop")

    @app.post("/api/relaunch-plan")
    def relaunch_plan() -> dict[str, Any]:
        """Write the ledger **now**, and say whether it worked.

        This is the one endpoint that does not queue, because it is the one the supervisor
        calls immediately before killing this process.

        It used to queue like everything else, and that was a silent failure. Measured: the
        launcher posts here, the handler answers ``{"ok": true, "queued": "relaunch_plan"}``,
        and ``terminate()`` lands before ``run_loop`` has drained the queue -- so the file was
        never written and the next process came back with ``{"restored": false}``. The whole
        point of asking before killing is defeated by asking in a way that only enqueues.

        So this writes the file on the calling thread and returns ``saved: true`` only if the
        file is on disk afterwards. Mutating the orchestrator is still avoided: the tasks are
        requeued by the restore path, which demotes anything in an active state anyway.

        ``Popen.terminate()`` is ``TerminateProcess``: immediate, and no shutdown hooks run.
        That is why the save has to be an explicit HTTP call and why it has to be synchronous.
        """
        if ledger_path is None:
            return {"ok": False, "saved": False,
                    "reason": "this server was started with --no-ledger, so there is nothing "
                              "to save. The task queue will not survive the restart."}

        result = orch.release_for_relaunch()
        try:
            from .ledger import capture as capture_ledger

            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            capture_ledger(orch).save(ledger_path)
        except Exception as exc:                   # noqa: BLE001
            hub.publish(lastError=f"ledger save failed: {type(exc).__name__}: {exc}")
            return {"ok": False, "saved": False, "released": result.get("released", []),
                    "reason": f"{type(exc).__name__}: {exc}"}

        saved = ledger_path.is_file()
        hub.publish(ledger={"restored": False, "savedAt": time.time(),
                            "path": str(ledger_path)})
        print(f"[demo12] ledger saved to {ledger_path}", flush=True)
        return {"ok": True, "saved": saved, "path": str(ledger_path),
                "tasks": len(orch.tasks),
                "released": result.get("released", [])}

    # Deliberately a Starlette ``WebSocketRoute``, not FastAPI's ``@app.websocket``.
    #
    # In an Isaac-booted process FastAPI's ``APIWebSocketRoute`` matches the request and then
    # answers ``websocket.close`` without ever calling the handler, which uvicorn turns into a
    # bare ``403 Forbidden``. Measured on the same process, the same app, the same endpoint
    # signature: a raw ASGI handler and a Starlette ``WebSocketRoute`` both connect and stream,
    # and only FastAPI's route fails. Importing the web stack before Kit boots -- so that
    # site-packages wins over Isaac's bundled copy in ``omni.kit.pip_archive/*/pip_prebundle``
    # -- does not fix it, so it is not a version-skew problem. See
    # ``demo12/tools/probe_ws_isaac.py``, which reproduces the whole comparison without a cell.
    #
    # The visible symptom was the interface stuck on "sin conexion" while REST and the camera
    # image both looked perfectly healthy, because a failed upgrade and a healthy server look
    # exactly the same from outside.
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_text(json.dumps(hub.state(), default=str))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    app.router.routes.append(WebSocketRoute("/ws", ws))

    app.state.hub = hub
    app.state.cell = cell
    app.state.orch = orch
    # The caller runs this on the thread that owns Kit. It never returns under normal
    # operation; Ctrl-C is the exit.
    app.state.sim_loop = run_loop
    return app


def serve_in_thread(app, host: str, port: int) -> threading.Thread:
    """Run uvicorn on its own event loop, in its own thread.

    This is the whole point of the arrangement. Kit owns the main thread's event loop; uvicorn
    gets a different one on a different thread, and the two never meet. The HTTP thread does
    no physics, so nothing it does can deadlock Kit.
    """
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="demo12-http", daemon=True)
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo 12 server")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frame-every", type=int, default=20,
                        help="capture a frame every N control ticks")
    parser.add_argument("--ledger", default=str(_DEMO_ROOT.parent / "out12" / "ledger.json"),
                        help="where to save/restore the task ledger across restarts")
    parser.add_argument("--no-ledger", action="store_true",
                        help="start with an empty task queue, ignoring any saved ledger")
    parser.add_argument("--gui", dest="headless", action="store_false", default=True,
                        help="show Isaac Sim so the cell can be watched directly")
    parser.add_argument("--scene", default=None,
                        help="a scene JSON written by the interface's editor, built instead "
                             "of the default cell")
    parser.add_argument("--control", default=str(_DEMO_ROOT.parent / "out12" / "control.json"),
                        help="where to ask the supervisor for a restart. Absent means this "
                             "process was not started by one.")
    args = parser.parse_args()

    control_path = Path(args.control)
    env_loader.prepare_runtime(headless=args.headless)

    from .config import SceneConfig, SceneError, default_scene
    from .isaac_env import IsaacCell
    from .ledger import OrchestratorLedger as Ledger, capture as capture_ledger
    from .orchestrator import MultiRobotOrchestrator

    # The scene comes from the editor when the supervisor passed one, and from the defaults
    # otherwise. A bad scene file is a user error, not a crash: it is reported and the
    # default cell is built so the interface still comes up and can say what was wrong.
    scene_errors: list[str] = []
    if args.scene and Path(args.scene).is_file():
        try:
            scene = SceneConfig.from_dict(json.loads(
                Path(args.scene).read_text(encoding="utf-8-sig")))
            print(f"[demo12] scene loaded from {args.scene}: {len(scene.parts)} part(s), "
                  f"{len(scene.bins)} tray(s), {len(scene.enabled_robots)} robot(s)",
                  flush=True)
        except (SceneError, json.JSONDecodeError, OSError) as exc:
            scene_errors = [f"scene {args.scene} rejected: {exc}"]
            print(f"[demo12] {scene_errors[0]}; falling back to the default cell",
                  flush=True)
            scene = default_scene(args.robots)
    else:
        scene = default_scene(args.robots)

    cell = IsaacCell(scene, device="cuda:0", headless=args.headless)
    cell.settle(90)

    # Create and prime the observation camera while the cell is idle. Creating it lazily from
    # inside the running loop does not work: Replicator's annotator asks for zero frames.
    cell.camera()

    orch = MultiRobotOrchestrator(cell, scene, scene.mission.tasks)

    ledger_path = None if args.no_ledger else Path(args.ledger)
    restored: dict[str, Any] = {"restored": False}
    if ledger_path is not None and ledger_path.is_file():
        ledger = Ledger.load(ledger_path)
        if ledger is not None:
            from .orchestrator import ACTIVE_STATES, QUEUED

            demoted = ledger.reset_verified()
            orch = MultiRobotOrchestrator(cell, scene, ledger.move_tasks())
            for entry in ledger.entries:
                task = orch.tasks.get(entry.id)
                if task is None:
                    continue
                task.attempts = entry.attempts
                task.previous_robots = list(entry.previous_robots)
                task.note = entry.note
                if entry.state in ACTIVE_STATES:
                    task.state = QUEUED
                    task.note = (f"requeued after a restart: {entry.note}" if entry.note
                                 else "requeued after a restart: the parts were respawned")
            restored = {
                "restored": True,
                "tasks": len(ledger.entries),
                "completedBeforeRestart": ledger.completed_before_restart,
                "partsMovedBeforeRestart": ledger.parts_moved_before_restart,
                "demotedDoneTasks": demoted,
            }
            print(f"[demo12] ledger restored: {restored}", flush=True)

    hub = Hub()
    app = build_app(cell, orch, hub, args.robots, args.headless, args.frame_every,
                    control_path=control_path, ledger_path=ledger_path)
    if scene_errors:
        hub.publish(lastError=scene_errors[0])

    # Log what actually got registered. A websocket route that fails to register looks
    # exactly like a websocket the server refuses, because Starlette answers an unmatched
    # websocket scope with ``websocket.close`` and uvicorn turns that into a bare 403 with
    # an empty body -- indistinguishable, from the outside, from a crash.
    print(f"[demo12] {len(app.routes)} routes:", flush=True)
    for route in app.routes:
        print(f"[demo12]   {type(route).__name__:<18} {getattr(route, 'path', '?')}",
              flush=True)

    # HTTP first, so the interface has something to talk to while the cell settles.
    serve_in_thread(app, args.host, args.port)
    print(f"[demo12] API on http://{args.host}:{args.port}  "
          f"({args.robots} robot(s), headless={args.headless})", flush=True)

    # Then the simulation, forever, on this thread. ``run_loop`` deliberately never returns:
    # Ctrl-C is the exit, and ``demo12.launcher`` stops the process the other way -- by asking
    # over HTTP first, because terminate() runs no shutdown hooks on Windows.
    try:
        app.state.sim_loop()
    except KeyboardInterrupt:
        print("[demo12] stopping", flush=True)

    if ledger_path is not None:
        try:
            capture_ledger(orch).save(ledger_path)
            print(f"[demo12] ledger saved to {ledger_path}", flush=True)
        except Exception as exc:                   # noqa: BLE001
            print(f"[demo12] could not save the ledger: {exc}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()