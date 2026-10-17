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


async def capture_frame(cell, width: int = 640, height: int = 360,
                        timeout: float = 3.0) -> tuple[bytes | None, str]:
    """Capture the viewport and JPEG-encode it. Returns ``(data, note_or_reason)``.

    **Every obvious API for this is wrong on Isaac Sim 5.1.** Measured, not assumed --
    ``demo12/tools/probe_frame.py`` is the probe that found all of it:

    * ``omni.replicator.core`` has no ``Renderer``, no ``render`` and no ``pyrep``. It
      exposes ``create.camera``, ``create.render_product`` and ``rep.get``.
    * ``create.render_product`` returns a ``HydraTexture`` whose only members are
      ``destroy``, ``hydra_texture`` and ``path`` -- **no pixel accessor whatsoever**, so
      neither ``.get_data()`` nor ``rep.get`` can read it.
    * ``numpy.asarray`` on that object raises ``AttributeError: ... has no attribute
      'get_data'`` while probing it, which is not about pixels and hides the real error.
    * ``isaaclab.sensors.Camera`` fails at construction on this build
      (``'RigidObject' object has no attribute '_external_force_b'``).

    What does work is Kit's own viewport utility, renamed:

    * ``omni.kit.viewport.utility.capture_viewport_to_image`` -> ``..._to_buffer``
    * signature ``(viewport_api, on_capture_fn, is_hdr=False)`` -- **callback based**
    * ``omni.kit.app.get_default_app_window`` does not exist; use
      ``omni.kit.viewport.utility.get_active_viewport()``

    The callback is asynchronous: it fires on a later app update, not on return. A probe
    that captures and immediately exits sees "NO CALLBACK FIRED", which is why this is a
    coroutine awaiting an event rather than a plain function. Under the server's event loop
    the loop is live and the callback lands.
    """
    try:
        import numpy as np
        from omni.kit.viewport.utility import (capture_viewport_to_buffer,
                                               get_active_viewport)
        from PIL import Image
    except ImportError as exc:
        return None, f"viewport capture unavailable: {exc}"

    # The cell has to actually render. ``IsaacCell.step`` only asks for a render when Kit
    # has a window, so in headless the viewport would hold a stale or black frame -- a
    # valid JPEG of nothing, which is worse than an error because the interface would show
    # it as a live render.
    request = getattr(cell, "render_once", None)
    if callable(request):
        try:
            request()
        except Exception as exc:                   # noqa: BLE001
            return None, f"cell render failed: {exc}"

    try:
        viewport = get_active_viewport()
    except Exception as exc:                       # noqa: BLE001
        return None, f"no active viewport: {type(exc).__name__}: {exc}"
    if viewport is None:
        return None, ("no active viewport. Headless Kit has no viewport to capture; "
                      "start the server with --gui, or accept that /api/frame.jpg "
                      "returns 503")

    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def on_capture(pixels) -> None:
        if done.done():
            return
        try:
            array = np.asarray(pixels)
            if array.ndim == 4:
                array = array[0]
            if array.ndim == 3 and array.shape[-1] >= 4:
                array = array[..., :3]
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=82)
            loop.call_soon_threadsafe(done.set_result, buffer.getvalue())
        except Exception as exc:                   # noqa: BLE001
            loop.call_soon_threadsafe(done.set_exception, exc)

    try:
        capture_viewport_to_buffer(viewport, on_capture)
    except Exception as exc:                       # noqa: BLE001
        return None, f"capture request failed: {type(exc).__name__}: {exc}"

    try:
        data = await asyncio.wait_for(done, timeout=timeout)
    except asyncio.TimeoutError:
        return None, (f"viewport capture did not fire within {timeout:.0f} s. The "
                      f"callback needs Kit's event loop; in headless it may never run. "
                      f"Start the server with --gui.")
    except Exception as exc:                       # noqa: BLE001
        return None, f"capture failed: {type(exc).__name__}: {exc}"

    return data, f"viewport {width}x{height}"


def build_app(robot_count: int = 2, headless: bool = True,
              rate_hz: float = 20.0, frame_every: int = 20):
    """Build the FastAPI app over a real cell. Kit must already be running."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response

    from .config import default_scene
    from .isaac_env import ARM_BODIES_WITH_COLLISION, IsaacCell
    from .orchestrator import MultiRobotOrchestrator

    scene = default_scene(robot_count)
    cell = IsaacCell(scene, device="cuda:0", headless=headless)
    cell.settle(90)

    orch = MultiRobotOrchestrator(cell, scene, scene.mission.tasks)

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
    args = parser.parse_args()

    env_loader.prepare_runtime(headless=args.headless)

    import uvicorn

    uvicorn.run(build_app(args.robots, headless=args.headless),
                host=args.host, port=args.port, log_level="info")
    os._exit(0)


if __name__ == "__main__":
    main()