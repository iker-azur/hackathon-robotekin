"""Servidor del panel de RecoveryKit (FastAPI, en un hilo aparte).

Sólo lee instantáneas de ``Telemetry`` y deja órdenes en su cola; nunca toca
Isaac. El bucle de la simulación (``17_recovery_kit.py``) atiende esas órdenes.

    GET  /                 panel (en español)
    GET  /api/state        estado completo (JSON)
    GET  /api/frame.jpg    última imagen de la cámara cenital
    GET  /api/poses        poses para la vista 3D
    POST /api/kit          {"text": "..."}                      pedir un kit
    POST /api/control      {"action": "pause|resume|stop|reset|randomize|demo"}
    POST /api/perturb      {"kind": "empujar|golpe_cierre|tirón_subida|fallo_pinza|sacar_caja"}
    GET  /api/config       configuración de la escena
    POST /api/config       nueva configuración (se aplica reiniciando la simulación)
    WS   /ws               estado ~4 Hz, poses ~15 Hz, imágenes (binario) cuando cambian
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

# a nivel de módulo: con ``from __future__ import annotations`` FastAPI resuelve
# las anotaciones (WebSocket) en los globales del módulo
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).resolve().parent / "static"


def build_app(tele, get_config, perturb_kinds: dict[str, str]):
    from recoverykit import config as config_mod
    from recoverykit.telemetry import Command

    app = FastAPI(title="RecoveryKit")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        return JSONResponse(tele.snapshot())

    @app.get("/api/frame.jpg")
    def frame():
        _, jpg = tele.frame()
        if jpg is None:
            raise HTTPException(404, "todavía no hay imagen")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/poses")
    def poses():
        return JSONResponse(tele.poses() or {})

    @app.post("/api/kit")
    async def kit_request(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "escribe el kit que quieres")
        if tele.snapshot()["status"] == "ejecutando":
            raise HTTPException(409, "el robot ya está preparando un kit; detenlo antes")
        tele.commands.put(Command("kit", {"text": text}))
        return {"ok": True}

    @app.post("/api/control")
    async def control(body: dict):
        action = body.get("action")
        running = tele.snapshot()["status"] == "ejecutando"
        if action == "pause":
            tele.pause()
        elif action == "resume":
            tele.resume()
        elif action == "stop":
            tele.request_stop()
        elif action == "reset":
            if running:
                tele.request_stop()
            tele.commands.put(Command("reset", {}))
        elif action in ("randomize", "demo"):
            if running:
                raise HTTPException(409, "detén la ejecución antes")
            tele.commands.put(Command(action, {}))
        else:
            raise HTTPException(400, f"acción desconocida: {action}")
        return {"ok": True}

    @app.post("/api/perturb")
    async def perturb(body: dict):
        kind = body.get("kind")
        if kind not in perturb_kinds:
            raise HTTPException(400, f"perturbación desconocida: {kind}")
        tele.commands.put(Command("perturb", {"kind": kind}))
        return {"ok": True, "label": perturb_kinds[kind]}

    @app.get("/api/config")
    def get_cfg():
        return JSONResponse({"config": get_config().to_dict(), "presets": ["demo", "simple", "formas"]})

    @app.post("/api/config")
    async def set_cfg(body: dict):
        try:
            cfg = config_mod.preset(body["preset"]) if "preset" in body else config_mod.from_dict(body["config"])
        except Exception as exc:  # noqa: BLE001 — el error vuelve al usuario tal cual
            raise HTTPException(400, f"configuración no válida: {exc}") from exc
        notes = config_mod.validate(cfg)
        problems = [e for e in notes if not e.startswith("aviso")]
        if problems:
            raise HTTPException(400, "; ".join(problems))
        if tele.snapshot()["status"] == "ejecutando":
            raise HTTPException(409, "detén la ejecución antes de cambiar la escena")
        tele.commands.put(Command("config", {"config": cfg.to_dict()}))
        return {"ok": True, "warnings": [e for e in notes if e.startswith("aviso")]}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        last_frame = -1
        last_state = 0.0
        last_pose_t = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                now = loop.time()
                if now - last_state >= 0.25:
                    last_state = now
                    await sock.send_text(json.dumps({"type": "state", "state": tele.snapshot()}, ensure_ascii=False))
                p = tele.poses()
                if p and p.get("t") != last_pose_t:
                    last_pose_t = p.get("t")
                    await sock.send_text(json.dumps({"type": "poses", "poses": p}))
                fid, jpg = tele.frame()
                if jpg is not None and fid != last_frame:
                    last_frame = fid
                    await sock.send_bytes(jpg)
                await asyncio.sleep(0.066)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def start(tele, get_config, perturb_kinds: dict[str, str], host: str = "127.0.0.1", port: int = 8770):
    """Arranca uvicorn en un hilo de fondo y devuelve el hilo."""
    import uvicorn

    app = build_app(tele, get_config, perturb_kinds)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    th = threading.Thread(target=server.run, name="panel", daemon=True)
    th.start()
    return th
