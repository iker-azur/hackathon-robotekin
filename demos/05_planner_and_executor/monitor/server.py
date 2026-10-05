"""The localhost dashboard: a web app in a thread, next to a simulator that owns the main one.

Why this is a thread and not a second process
---------------------------------------------
Isaac Sim requires the main thread, ``cv2.imshow`` requires the main thread, and the simulator
is two minutes of start-up that nobody wants to pay twice. So the web app is a
``uvicorn.Server`` running on a daemon thread inside the same process, and the two halves meet
only through ``telemetry.Bus``. ``uvicorn`` notices it is not on the main thread and skips
signal handling (its ``capture_signals`` returns early), so nothing about Ctrl-C changes.

What it exposes
---------------
* ``GET  /``              the dashboard, one self-contained HTML file with no CDN, so it works
  with the network off - which is how this repository is run.
* ``GET  /ws``            a websocket carrying JSON snapshots and binary JPEG frames. Every
  change to the bus wakes the handler, so the picture is live without polling.
* ``GET  /api/state``     the same snapshot over plain HTTP, for ``curl`` and for scripts.
* ``GET  /api/frame.jpg`` the newest annotated frame, for a browser that cannot hold a socket.
* ``POST /api/command``   ``{"text": "..."}`` or ``{"action": "start"}``.
* ``POST /api/env``       ``{"action": "relook"|"next"|"save"|"stop"|"voice", ...}``.

Two deliberate omissions. There is no authentication and it binds to 127.0.0.1: this is a
bench tool for the person standing at the machine, and adding a password to a loopback
dashboard would be security theatre. And nothing here can drive the robot directly - every
action becomes a queued request that the robot's own loop consumes, so the ordering guarantees
the simulation already has are not replaced by HTTP ordering.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

import telemetry

HERE = Path(__file__).resolve().parent
INDEX = HERE / "static" / "index.html"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# How long ``serve`` waits for uvicorn to bind before giving up. Binding a loopback socket is
# instant, so anything longer than this means the port is in use and the caller should hear
# about it rather than wait.
READY_TIMEOUT_S = 10.0


def build_app(bus: telemetry.Bus) -> FastAPI:
    """The ASGI app. Separated from ``serve`` so the tests can drive it without a thread."""

    app = FastAPI(title="leisaac demo 11", docs_url=None, redoc_url=None)

    @app.get("/")
    def index() -> FileResponse:
        if not INDEX.exists():
            return JSONResponse({"error": f"{INDEX} is missing"}, status_code=500)
        return FileResponse(INDEX, media_type="text/html")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "started": bus.started(), "seq": bus.snapshot()["seq"]}

    @app.get("/api/state")
    def state() -> dict:
        return bus.snapshot()

    @app.get("/api/frame.jpg")
    def frame() -> Response:
        _seq, jpeg, _meta = bus.frame()
        if jpeg is None:
            return JSONResponse({"error": "no frame yet"}, status_code=404)
        return Response(content=jpeg, media_type="image/jpeg")

    @app.post("/api/command")
    async def command(payload: dict) -> dict:
        """A sentence to run, or the go-ahead to start Isaac.

        ``action`` and ``text`` are both accepted on the same endpoint because they are both
        "something the person wants the robot to do next" and splitting them would mean the
        dashboard picking between two shapes for the same idea.
        """
        action = str(payload.get("action") or "").strip()
        text = str(payload.get("text") or "").strip()
        if action == "start":
            bus.open_start()
            return {"ok": True, "action": "start"}
        if text:
            bus.submit(text)
            return {"ok": True, "queued": text}
        if action:
            bus.send_control(action, **{k: v for k, v in payload.items() if k != "action"})
            return {"ok": True, "queued": action}
        return JSONResponse({"error": "nothing to do: send text or action"}, status_code=400)

    @app.post("/api/env")
    async def env(payload: dict, request: Request) -> dict:
        action = str(payload.get("action") or "").strip()
        if not action:
            return JSONResponse({"error": "no action"}, status_code=400)
        params = {k: v for k, v in payload.items() if k != "action"}
        if action == "start":
            bus.open_start()
            return {"ok": True, "action": "start"}
        bus.send_control(action, **params)
        return {"ok": True, "queued": action, "params": params}

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        """Push snapshots and frames; read commands and controls.

        Two tasks, because this socket has to do both without polling: one waits for the bus
        to change and pushes, the other reads what the browser sent. Polling the bus on a timer
        would either be late or would flood the socket, and the bus knows exactly when
        something changed.
        """
        await websocket.accept()
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()

        def hook() -> None:
            # Called from whichever thread the robot is on - possibly the one inside
            # ``env.step`` - so the hand-off to the loop has to be explicit.
            loop.call_soon_threadsafe(wake.set)

        bus.add_hook(hook)
        reader = asyncio.create_task(_reader(websocket, bus))
        waiter = asyncio.create_task(wake.wait())
        last_frame = 0
        try:
            while True:
                done, _pending = await asyncio.wait({reader, waiter},
                                                    return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    break  # the browser hung up, or sent something malformed
                waiter.cancel()
                waiter = asyncio.create_task(wake.wait())
                await websocket.send_json({"type": "snapshot", **bus.snapshot()})
                seq, jpeg, meta = bus.frame()
                if jpeg is not None and seq > last_frame:
                    await websocket.send_json({"type": "frame", "seq": seq, "meta": meta})
                    await websocket.send_bytes(jpeg)
                    last_frame = seq
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bus.drop_hook(hook)
            for task in (reader, waiter):
                task.cancel()

    return app


async def _reader(websocket: WebSocket, bus: telemetry.Bus) -> None:
    """Read client messages until the socket closes. Never raises out of the handler."""
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except ValueError:
                bus.submit(raw)
                continue
            if not isinstance(message, dict):
                continue
            kind = str(message.get("type") or "")
            if kind == "command" or "text" in message:
                bus.submit(str(message.get("text") or ""))
            elif kind == "env":
                action = str(message.get("action") or "")
                if action:
                    params = {k: v for k, v in message.items()
                              if k not in ("type", "action")}
                    bus.send_control(action, **params)
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - a bad client must not take the socket handler down
        return


class Monitor:
    """A running dashboard: ``start`` it, read ``url``, ``stop`` it."""

    def __init__(self, bus: telemetry.Bus, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        import uvicorn

        self.bus = bus
        self.host = host
        self.port = port
        self.app = build_app(bus)
        self.config = uvicorn.Config(self.app, host=host, port=port, log_level="warning",
                                     access_log=False, lifespan="off")
        self.server = uvicorn.Server(self.config)
        self._thread = threading.Thread(target=self.server.run, name="dashboard",
                                        daemon=True)

    def start(self) -> "Monitor":
        self._thread.start()
        return self

    def wait_ready(self, timeout: float = READY_TIMEOUT_S) -> bool:
        """Block until the socket is bound, and resolve port 0 to the real port.

        ``--port 0`` is worth keeping: two copies of the demo on one machine, or a test, should
        not have to agree on a free port in advance.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(self.server, "started", False):
                for bound in self.server.servers or []:
                    for sock in bound.sockets or []:
                        self.port = sock.getsockname()[1]
                        self.config.port = self.port
                        return True
                return self.port > 0
            if not self._thread.is_alive():
                return False
            time.sleep(0.02)
        return False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def open_browser(self) -> None:
        try:
            webbrowser.open(self.url)
        except Exception:  # noqa: BLE001 - no browser on a headless box is not an error
            pass

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=timeout)


def serve(bus: telemetry.Bus, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          open_browser: bool = False, announce=lambda _line: None) -> Monitor:
    """Start the dashboard and return it once it is listening."""
    monitor = Monitor(bus, host, port).start()
    if not monitor.wait_ready():
        announce(f"  the dashboard did not come up on {host}:{port}; is the port in use?")
        return monitor
    announce(f"  dashboard: {monitor.url}")
    if open_browser:
        monitor.open_browser()
    return monitor