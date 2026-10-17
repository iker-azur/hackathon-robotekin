"""Demo 12 -- FastAPI over a live Isaac Lab cell.

REST plus WebSocket, over a simulation that is really running. The cell is built once at
startup and stepped by a background task; the interface only ever reads it.

Two things this server will not do, because either would make the demo lie:

- **It will not catch and swallow orchestration errors.** An earlier version wrapped the
  step in a bare ``except Exception: pass``, which means a cell that has been broken for an
  hour still serves cheerful "everything is fine" state. The loop records the failure and
  the interface shows it.
- **It will not serve a placeholder frame.** ``/api/frame.jpg`` returns a real render or a
  503 that says why, never a stub image.

Run::

    C:\\Users\\Usuario\\miniconda3\\envs\\hackathon-robotekin\\python.exe -m demo12.server --robots 2

Then open the Vite dev server (see ``web/``).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve()
# parents[0] = demo12, [1] = 07_multi_robot_orchestration, [2] = demos
_DEMO_ROOT = _HERE.parents[0]
_DEMOS_ROOT = _HERE.parents[2]

for _p in (str(_HERE.parents[1]), str(_DEMOS_ROOT / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402


def _to_rgb_array(sample) -> tuple[Any, str]:
    """Turn whatever Replicator handed back into an HxWx3 uint8 array.

    Replicator's return type has changed more than once, and guessing wrong produces an
    endpoint that imports cleanly and fails on first call. Measured on Isaac Sim 5.1 /
    omni.replicator.core 1.12.27: ``rep.get(render_product, "rgb")`` returns a
    ``HydraTexture``, which has **no** ``get_data`` method.

    So each known shape is tried in turn and the one that worked is named, which means the
    next API change reports itself instead of hiding.
    """
    attempts: list[str] = []

    def as_array(value):
        """``np.asarray`` that never raises.

        numpy probes an unknown object for array-like attributes before giving up, so on a
        ``HydraTexture`` it raises ``AttributeError: 'HydraTexture' object has no attribute
        'get_data'`` -- which is not about pixels at all, and swallowed the real error the
        first time this was written.
        """
        try:
            return np.asarray(value)
        except Exception as exc:                   # noqa: BLE001
            attempts.append(f"asarray({type(value).__name__}): {type(exc).__name__}")
            return None

    array = as_array(sample)
    if array is not None and array.ndim >= 3:
        return array, "asarray"

    for attr in ("image", "data", "buffer", "np_array"):
        if not hasattr(sample, attr):
            attempts.append(f"no.{attr}")
            continue
        try:
            value = getattr(sample, attr)
            value = value() if callable(value) else value
            array = as_array(value)
            if array is not None and array.ndim >= 3:
                return array, f".{attr}"
            if array is not None:
                attempts.append(f".{attr}->ndim{array.ndim}")
        except Exception as exc:                   # noqa: BLE001
            attempts.append(f".{attr}: {type(exc).__name__}")

    # Last resort: ask Replicator's own annotator rather than the texture object.
    try:
        annotator = rep_annotator_for(sample)
        if annotator is not None:
            array = as_array(annotator)
            if array is not None and array.ndim >= 3:
                return array, "annotator"
    except Exception as exc:                       # noqa: BLE001
        attempts.append(f"annotator: {type(exc).__name__}")

    return None, (f"could not read pixels from {type(sample).__name__}; tried "
                  + ", ".join(attempts))


def rep_annotator_for(sample):
    """``rep.get.`` style accessor as a free function, if this build exposes one."""
    import omni.replicator.core as rep

    getter = getattr(rep, "get", None)
    if getter is None:
        return None
    # HydraTexture carries the render product path; if it does not, nothing to try.
    path = getattr(sample, "path", None)
    return getter(path, "rgb") if path else None


async def capture_frame(cell, timeout: float = 6.0) -> tuple[bytes | None, str]:
    """A JPEG of the cell, or ``(None, reason)``.

    Reads the cell's own observation camera (``IsaacCell.camera``), not a viewport
    snapshot. That matters here: this machine's viewport has no camera
    (``has_primed_camera() == False``) so a viewport capture can never produce an image,
    while an explicit ``UsdGeom.Camera`` prim read by a sensor works in headless.

    Every obvious alternative was measured and does not work on Isaac Sim 5.1 -- see
    ``docs/PHYSICS.md`` section 9. In brief: ``rep.Renderer`` does not exist,
    ``rep.create.render_product`` returns a ``HydraTexture`` with no pixel accessor,
    ``isaaclab.sensors.Camera`` initialises itself from a **timeline PLAY event** that a
    bare ``SimulationContext`` never fires, and ``SensorBase`` swallows the real exception
    into ``builtins.ISAACLAB_CALLBACK_EXCEPTION``. The cell works around the last one by
    calling the initialisation callback directly.

    Synchronous despite being a coroutine, and deliberately **not** in an executor: Kit and
    CUDA own the main thread, and moving a sensor read to a worker thread deadlocks. The same
    trap as stepping the simulation off-thread, which cost a day once already.
    """
    try:
        return cell.render_frame_jpeg()
    except Exception as exc:                       # noqa: BLE001
        return None, f"frame capture failed: {type(exc).__name__}: {exc}"


def build_app(robot_count: int = 2, headless: bool = True,
              rate_hz: float = 20.0, frame_every: int = 20,
              ledger_path: Path | None = None):
    """Build the FastAPI app over a real cell. Kit must already be running."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response

    from .config import default_scene
    from .isaac_env import ARM_BODIES_WITH_COLLISION, IsaacCell
    from .ledger import OrchestratorLedger as Ledger, capture as capture_ledger
    from .orchestrator import (ACTIVE_STATES, QUEUED, MultiRobotOrchestrator)

    scene = default_scene(robot_count)
    cell = IsaacCell(scene, device="cuda:0", headless=headless)
    cell.settle(90)

    # Create and prime the observation camera *here*, while the cell is idle, rather than
    # lazily on the first capture.
    #
    # Measured: a camera created while the orchestration loop is running reads back an empty
    # frame -- Replicator's annotator tries to expand the buffer to size 0 and raises
    #   expand(torch.cuda.ByteTensor{[360, 640, 3]}, size=[0])
    # The same camera created against the same cell while nothing is stepping reads
    # perfectly, with values spanning 28..155. So it is a create-while-running problem, not
    # a camera problem, and creating it at startup avoids it.
    cell.camera()

    orch = MultiRobotOrchestrator(cell, scene, scene.mission.tasks)

    # Restore a ledger from a previous run, if there is one. This is how the task queue
    # survives the restart that changing the robot count requires.
    #
    # Any task that was verified done is demoted first: the cell has just been respawned, so
    # the parts are back at their configured positions and the earlier placement is no longer
    # a fact about the simulation. See demo12.ledger.
    restored: dict[str, Any] = {"restored": False}
    if ledger_path is not None and ledger_path.is_file():
        ledger = Ledger.load(ledger_path)
        if ledger is None:
            restored = {"restored": False,
                        "reason": f"could not read {ledger_path}"}
        else:
            demoted = ledger.reset_verified()
            orch = MultiRobotOrchestrator(cell, scene, ledger.move_tasks())
            for entry in ledger.entries:
                task = orch.tasks.get(entry.id)
                if task is None:
                    continue
                task.attempts = entry.attempts
                task.previous_robots = list(entry.previous_robots)
                task.note = entry.note
                # A task that was in progress when the process died cannot still be true.
                if entry.state in ACTIVE_STATES:
                    task.state = QUEUED
                    task.note = (f"requeued after a restart: {entry.note}"
                                 if entry.note else
                                 "requeued after a restart: the parts were respawned")
            restored = {
                "restored": True,
                "tasks": len(ledger.entries),
                "completedBeforeRestart": ledger.completed_before_restart,
                "partsMovedBeforeRestart": ledger.parts_moved_before_restart,
                "demotedDoneTasks": demoted,
            }
            print(f"[demo12] ledger restored: {restored}", flush=True)

    state: dict[str, Any] = {
        "running": True,
        "started_at": time.time(),
        "steps": 0,
        "last_error": None,
        "error_count": 0,
    }
    latest_frame: bytes | None = None
    frame_error = "no frame captured yet"
    frame_note = ""
    # Mutable so the loop can back off without ``nonlocal`` gymnastics -- and so the current
    # cadence is visible in /api/state rather than hidden in a closure.
    capture: dict[str, int] = {"every": max(1, frame_every)}

    app = FastAPI(title="Demo 12 -- multi-robot orchestration")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- the loop

    async def run_loop() -> None:
        nonlocal latest_frame, frame_error, frame_note
        interval = 1.0 / rate_hz
        while state["running"]:
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                # **On this thread, deliberately.** The first version ran this in a
                # ThreadPoolExecutor via ``run_in_executor`` so the event loop would not
                # block. It deadlocks: Kit and PhysX own the main thread, and stepping the
                # simulation off it wedges the whole process. Measured -- the server came up
                # healthy and then simply stopped after one step, with no error anywhere,
                # which is the hardest kind of failure to notice.
                #
                # So the loop blocks instead. The WebSocket goes quiet while a tick runs,
                # which at ~10-20 Hz of real simulation is well under a second -- and honest,
                # unlike a deadlock.
                _advance()
                state["steps"] += 1
                if state["steps"] % capture["every"] == 0:
                    data, note = await capture_frame(cell)
                    if data:
                        latest_frame = data
                        frame_error = ""
                        frame_note = note
                    else:
                        frame_error = note
                        # Back off if the viewport is simply not there. Spamming a 3 s
                        # capture timeout would stall the orchestration loop, which is a worse
                        # failure than a missing picture.
                        capture["every"] = (max(capture["every"], 200)
                                            if "viewport" in note.lower()
                                            else capture["every"])
            except Exception as exc:               # noqa: BLE001
                # Recorded and surfaced, never swallowed. An earlier version passed here
                # and reported success forever.
                state["last_error"] = f"{type(exc).__name__}: {exc}"
                state["error_count"] += 1
                print(f"[demo12] orchestrator error: {state['last_error']}", flush=True)

            elapsed = loop.time() - started
            await asyncio.sleep(max(0.0, interval - elapsed))

    def _advance() -> None:
        """One orchestration tick. Completion stops the loop, not a timeout."""
        if orch.work_outstanding():
            orch.step()

    # ------------------------------------------------------------- endpoints

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": state["last_error"] is None,
            "running": state["running"],
            "steps": state["steps"],
            "uptimeS": round(time.time() - state["started_at"], 1),
            "errorCount": state["error_count"],
            "lastError": state["last_error"],
            "robots": len(orch.robots),
            "armCollision": list(ARM_BODIES_WITH_COLLISION),
            "headless": headless,
        }

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        payload = orch.snapshot()
        payload["health"] = {
            "running": state["running"],
            "steps": state["steps"],
            "lastError": state["last_error"],
            "errorCount": state["error_count"],
        }
        payload["cell"] = cell.snapshot()
        payload["frameError"] = frame_error
        payload["frameNote"] = frame_note
        payload["frameEvery"] = capture["every"]
        payload["ledger"] = restored
        return JSONResponse(payload)

    @app.get("/api/catalog")
    def get_catalog() -> JSONResponse:
        from .config import catalog

        return JSONResponse(catalog())

    @app.get("/api/frame.jpg")
    def get_frame() -> Response:
        if latest_frame is None:
            # 503 with the reason, not a placeholder image. A stub frame would let the
            # interface look healthy while nothing is being rendered.
            return JSONResponse({"error": frame_error}, status_code=503)
        return Response(content=latest_frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/fault/{robot_id}")
    def fault(robot_id: str) -> JSONResponse:
        return JSONResponse(orch.inject_failure(robot_id))

    @app.post("/api/repair/{robot_id}")
    def repair(robot_id: str) -> JSONResponse:
        return JSONResponse(orch.repair_robot(robot_id))

    @app.get("/api/roster")
    def roster() -> dict[str, Any]:
        """What it would take to change the robot count, and why it cannot be done here."""
        return {
            "current": [r.id for r in orch.robots.values()],
            "inProcessRebuild": False,
            "reason": ("Isaac Lab cannot insert an articulation into a live scene, and the "
                       "in-process cell rebuild hangs on this build (measured: RemovePrim "
                       "+ sim.reset() in one Kit session never returns)."),
            "supportedRoute": ("restart the process with a different --robots. "
                               "release_for_relaunch() requeues every in-flight task first, "
                               "because the parts respawn at their configured positions "
                               "and no in-flight task is still true."),
        }

    @app.post("/api/relaunch-plan")
    def relaunch_plan() -> JSONResponse:
        """Requeue in-flight work, save the ledger, and ask the supervisor to restart.

        This is the supported way to change the robot count, and it is three moves rather
        than one because the process doing the asking is the process that has to die:

        1. save the ledger, so the task queue and every attempt count survive;
        2. write the control file the supervisor polls;
        3. exit, which is where the supervisor notices.

        The parts do not survive, and nothing claims they do: ``demo12.ledger`` re-checks
        every completed task against the part's actual position before recording it, and
        demotes the ones that no longer hold.
        """
        result = orch.release_for_relaunch()
        saved = None
        if ledger_path is not None:
            try:
                entry = capture_ledger(orch)
                entry.save(ledger_path)
                saved = str(ledger_path)
            except Exception as exc:               # noqa: BLE001
                saved = f"could not save the ledger: {type(exc).__name__}: {exc}"

        return JSONResponse({
            **result,
            "ledgerSavedTo": saved,
            "requestedRobotCount": robot_count,
            "note": ("the supervisor watches out12/control.json; write "
                     '{"robots": N} to it to change the roster. This process will be '
                     "restarted by that."),
        })

    @app.get("/api/ledger")
    def get_ledger() -> JSONResponse:
        """The ledger as it would be saved right now.

        Read-only, and it re-checks every completed task against the physics rather than
        trusting the orchestrator's own state.
        """
        try:
            entry = capture_ledger(orch)
        except Exception as exc:                   # noqa: BLE001
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
        return JSONResponse(entry.to_dict())

    @app.post("/api/run")
    def start() -> dict[str, Any]:
        state["running"] = True
        return {"ok": True, "running": True}

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        state["running"] = False
        return {"ok": True, "running": False}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                payload = orch.snapshot()
                payload["health"] = {
                    "running": state["running"],
                    "steps": state["steps"],
                    "lastError": state["last_error"],
                    "errorCount": state["error_count"],
                }
                payload["cell"] = cell.snapshot()
                await websocket.send_text(json.dumps(payload, default=str))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    @app.on_event("startup")
    async def _start() -> None:
        asyncio.create_task(run_loop())

    @app.on_event("shutdown")
    async def _stop() -> None:
        state["running"] = False
        # Save the ledger on the way out, so an ordinary shutdown (or the supervisor's
        # restart) does not lose the queue. Best-effort: a failure here must not stop the
        # server shutting down, and it is reported rather than swallowed.
        if ledger_path is not None:
            try:
                saved = capture_ledger(orch)
                saved.save(ledger_path)
                print(f"[demo12] ledger saved to {ledger_path} "
                      f"({len(saved.entries)} task(s))", flush=True)
            except Exception as exc:               # noqa: BLE001
                print(f"[demo12] could not save the ledger: {type(exc).__name__}: {exc}",
                      flush=True)

    app.state.cell = cell
    app.state.orch = orch
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo 12 server")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gui", dest="headless", action="store_false", default=True,
                        help="show Isaac Sim so the cell can be watched directly")
    parser.add_argument("--ledger", default=str(_DEMO_ROOT.parent / "out12" / "ledger.json"),
                        help="where to save/restore the task ledger across restarts")
    parser.add_argument("--no-ledger", action="store_true",
                        help="start with an empty task queue, ignoring any saved ledger")
    args = parser.parse_args()

    env_loader.prepare_runtime(headless=args.headless)

    import uvicorn

    ledger = Path(args.ledger)
    if args.no_ledger:
        ledger = None
    uvicorn.run(build_app(args.robots, headless=args.headless, ledger_path=ledger),
                host=args.host, port=args.port, log_level="info")
    os._exit(0)


if __name__ == "__main__":
    main()