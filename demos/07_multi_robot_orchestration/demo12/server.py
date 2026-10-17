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
              frame_every: int):
    """Build the FastAPI app. Every endpoint reads the hub or queues a command.

    Nothing here touches the simulation. ``cell`` is accepted only so the frame endpoint can
    report why it is unavailable; the bytes come from the hub, written by the main thread.
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from starlette.routing import WebSocketRoute

    from .isaac_env import ARM_BODIES_WITH_COLLISION

    app = FastAPI(title="Demo 12 -- multi-robot orchestration")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
        from .config import catalog

        return JSONResponse(catalog())

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
        return _command("relaunch_plan")

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
    args = parser.parse_args()

    env_loader.prepare_runtime(headless=args.headless)

    from .config import default_scene
    from .isaac_env import IsaacCell
    from .ledger import OrchestratorLedger as Ledger, capture as capture_ledger
    from .orchestrator import MultiRobotOrchestrator

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
    app = build_app(cell, orch, hub, args.robots, args.headless, args.frame_every)

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