"""Servidor del panel (FastAPI): estado, imágenes, órdenes y control. No toca Isaac: deja órdenes en una cola.

    GET  /                   el panel (static/index.html)
    GET  /api/state          instantánea del estado (JSON)
    GET  /api/frame/{n}.jpg  imagen: cam (cenital), seg (segmentación) o vista (3/4)
    POST /api/orden          {"text": "..."}   orden escrita
    POST /api/voz            graba una frase con el micrófono, la transcribe y la lanza como orden
    POST /api/control        {"action": "detener" | "pausa" | "reanudar" | "reiniciar" | "aleatorizar"}
    WS   /ws                 estado (JSON) y fotogramas (binario, con un byte de cabecera por imagen)
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from . import DEMO_ROOT
from .telemetry import FRAME_NAMES, BinTelemetry

STATIC = DEMO_ROOT / "static"
FRAME_TAG = {"cam": 1, "seg": 2, "vista": 3}


def build_app(tele: BinTelemetry, voice=None):
    app = FastAPI(title="Clasificación de piezas por voz")

    def running() -> bool:
        return tele.snapshot()["status"] in ("ejecutando",)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        return JSONResponse(tele.snapshot())

    @app.get("/api/frame/{name}.jpg")
    def frame(name: str):
        if name not in FRAME_NAMES:
            raise HTTPException(404, "imagen desconocida")
        _, jpg = tele.named(name)
        if jpg is None:
            raise HTTPException(404, "todavía no hay imagen")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/orden")
    async def orden(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "escribe o di qué piezas mover y a qué caja")
        if running():
            raise HTTPException(409, "el robot está trabajando; pulsa Detener antes de dar otra orden")
        if not tele.snapshot().get("ready"):
            raise HTTPException(409, "la simulación todavía está arrancando")
        tele.send("orden", text=text, source=str(body.get("source", "texto")))
        return {"ok": True}

    @app.post("/api/voz")
    async def voz():
        if voice is None or voice.problem():
            raise HTTPException(503, (voice.problem() if voice else "la voz no está activada") or "sin voz")
        if running():
            raise HTTPException(409, "el robot está trabajando; pulsa Detener antes de dar otra orden")
        if getattr(voice, "busy", False):
            raise HTTPException(409, "ya estoy escuchando")

        def work():
            tele.set(voice={"state": "escuchando", "detail": "habla ahora…"})
            try:
                text = voice.listen()
            except Exception as exc:  # noqa: BLE001
                tele.set(voice={"state": "error", "detail": str(exc)})
                return
            if not text:
                tele.set(voice={"state": "inactiva", "detail": "no he oído nada"})
                return
            tele.set(voice={"state": "inactiva", "detail": ""}, heard=text)
            tele.send("orden", text=text, source="voz")

        threading.Thread(target=work, name="voz", daemon=True).start()
        return {"ok": True}

    @app.post("/api/control")
    async def control(body: dict):
        action = body.get("action")
        if action == "detener":
            tele.request_stop()
        elif action == "pausa":
            tele.pause()
        elif action == "reanudar":
            tele.resume()
        elif action in ("reiniciar", "aleatorizar"):
            if running():
                tele.request_stop()
            tele.send(action)
        else:
            raise HTTPException(400, f"acción desconocida: {action}")
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        last = {n: -1 for n in FRAME_NAMES}
        last_state = 0.0
        loop = asyncio.get_running_loop()
        try:
            while True:
                now = loop.time()
                if now - last_state >= 0.25:
                    last_state = now
                    await sock.send_text(json.dumps({"type": "state", "state": tele.snapshot()}, ensure_ascii=False))
                for name in FRAME_NAMES:
                    fid, jpg = tele.named(name)
                    if jpg is not None and fid != last[name]:
                        last[name] = fid
                        await sock.send_bytes(bytes([FRAME_TAG[name]]) + jpg)
                await asyncio.sleep(0.066)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def start(tele: BinTelemetry, voice=None, host: str = "127.0.0.1", port: int = 8780):
    """Arranca uvicorn en un hilo de fondo y devuelve el hilo."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(build_app(tele, voice), host=host, port=port, log_level="warning"))
    th = threading.Thread(target=server.run, name="panel", daemon=True)
    th.start()
    return th
