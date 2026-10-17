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

_HERE = Path(__file__).resolve()
_PARENT = _HERE.parent
_DEMOS_ROOT = _PARENT.parent

for _p in (str(_PARENT), str(_DEMOS_ROOT / "01_simulation_basics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402


def _frame_bytes(cell) -> tuple[bytes | None, str]:
    """Render the viewport and JPEG-encode it.

    Returns ``(data, "")`` or ``(None, reason)``. The reason is reported rather than
    swallowed: rendering needs a viewport, and a headless Kit has none unless one was
    created, so "no frame" is a configuration fact the operator needs, not an error to hide.
    """
    try:
        import omni.replicator.core as rep
        from PIL import Image
    except ImportError as exc:
        return None, f"replicator/PIL unavailable: {exc}"

    try:
        renderer = rep.Renderer()
        camera = rep.create.camera(rotation=(0, 0, 0), position=(0.9, -0.2, 0.9),
                                  focal_length=24.0)
        # Hide nothing: this is the point of the frame.
        render_product = renderer.render(camera)
        (rgb,) = render_product.get_data(["rgb"])
        image = Image.fromarray(rgb[0][..., :3])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        return buffer.getvalue(), ""
    except Exception as exc:                       # noqa: BLE001 - reported below
        return None, f"viewport render failed: {exc}"


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
        nonlocal latest_frame, frame_error
        interval = 1.0 / rate_hz
        while state["running"]:
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                await loop.run_in_executor(None, _advance)
                state["steps"] += 1
                if state["steps"] % frame_every == 0:
                    data, reason = await loop.run_in_executor(None, _frame_bytes, cell)
                    if data:
                        latest_frame = data
                        frame_error = ""
                    else:
                        frame_error = reason
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