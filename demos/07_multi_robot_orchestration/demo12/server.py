"""Demo 12 — FastAPI server (REST + WebSocket) over a real Isaac Lab cell."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "01_simulation_basics"))
import env_loader
env_loader.prepare_runtime(headless=True)

import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .config import default_scene
from .isaac_env import IsaacCell
from .orchestrator import MultiRobotOrchestrator


def build_app(robot_count: int = 2) -> FastAPI:
    app = FastAPI(title="Demo12")
    scene = default_scene(robot_count)
    cell = IsaacCell(scene, device="cuda:0", headless=True)
    cell.settle(10)
    orch = MultiRobotOrchestrator(cell, scene.mission.tasks)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(cell.snapshot())

    @app.get("/api/frame.jpg")
    def frame() -> Any:
        return JSONResponse({"ok": False}, status_code=501)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_text(json.dumps(cell.snapshot()))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass

    import asyncio

    async def _loop() -> None:
        while True:
            try:
                orch.step()
            except Exception:
                pass
            await asyncio.sleep(0.05)

    @app.on_event("startup")
    async def _start_loop() -> None:
        asyncio.create_task(_loop())

    app.state.cell = cell
    app.state.orch = orch
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(2), host="127.0.0.1", port=8000)
