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
import io
import json
import threading
import time
import wave
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

# The floor between two pushes on one socket.
#
# **This exists because of a measurement, and the number it replaces is the interesting part.**
# The handler used to wake on an ``asyncio.Event`` that nothing ever cleared, so once the event
# was set it stayed set: every new waiter returned immediately and the handler re-sent a whole
# snapshot as fast as the socket would take it. ONE bus change - a person clicking *iniciar
# entorno*, a single ``bus.node`` call - was enough to start it, and it never stopped. Measured
# with a client connected and nothing else happening: **2171 snapshots per second**, each one a
# full JSON copy of the 12-node diagram plus the last 140 log lines.
#
# That is not a cosmetic problem, it is the reason the dashboard looked dead and the simulator
# looked slow, and it was one line of bookkeeping:
#
# * the page could not keep up. ``applySnapshot`` repaints the whole SVG diagram and rebuilds
#   the detection, plan and log tables on every message, so a tab receiving thousands of
#   identical snapshots a second stops responding - and because the socket backlog grows faster
#   than the browser drains it, new frames stop arriving too. The dashboard is blank *and* the
#   machine is at 100% of a core, which reads as "localhost does not work".
# * the simulator is in the same process. A Python thread spinning on ``json.dumps`` holds the
#   GIL against the main thread that is between two ``env.step`` calls: measured, a stand-in
#   physics step went from 40822 to 17259 steps/s, and sustained over six seconds the process
#   burned 5.97 CPU-seconds in 6.02 seconds of wall clock with a client connected and nothing
#   being published.
#
# So pushes are keyed on a *version counter* rather than on an event that can stay set, and a
# burst of a hundred bus writes becomes one push rather than a hundred. 20 Hz is far above what
# the picture changes at - the fastest thing the robot publishes is a camera frame, and the
# frame pump is 3 - so nothing visible is lost.
PUSH_MIN_INTERVAL_S = 0.05

# The dashboard's own speech-to-text, built the same way demo 10's terminal listener is: one
# faster-whisper model, loaded lazily and only once, because two copies of a 464 MB model in
# one process is a way to run out of memory rather than a way to be fast. Recording happens in
# the browser (it owns the microphone naturally), the server only ever decodes a small WAV.
STT_MODEL = "small"
STT_LOCK = threading.Lock()
STT_CACHE: dict = {}


def decode_wav(payload: bytes) -> tuple:
    """A 16-bit PCM WAV blob -> (float32 samples, sample rate), mono or first channel."""
    import numpy as np

    with wave.open(io.BytesIO(payload)) as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        if not 8000 <= rate <= 96000 or wav.getnframes()/rate > 15 or channels not in (1, 2):
            raise ValueError("Audio no válido: utiliza mono o estéreo y un máximo de 15 segundos.")
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        return None, rate
    samples = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32767.0
    if channels > 1:
        samples = samples[::channels]
    return samples, rate


def transcribe_wav(payload: bytes, model_name: str = STT_MODEL, language: str | None = None) -> str:
    """One WAV blob -> one sentence. The model cache is on the machine already."""
    samples, rate = decode_wav(payload)
    if samples is None:
        raise ValueError("El audio debe ser WAV PCM de 16 bits.")
    if rate != 16000:
        import numpy as np
        samples = np.interp(np.arange(round(len(samples)*16000/rate))*rate/16000,
                            np.arange(len(samples)), samples).astype("float32")
    key = (model_name, language)
    with STT_LOCK:
        model = STT_CACHE.get(key)
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            STT_CACHE.clear()
            STT_CACHE[key] = model
    segments, _info = model.transcribe(samples, language=language, beam_size=1)
    return " ".join(segment.text for segment in segments).strip()


def _ack(bus: telemetry.Bus, cid: str, **extra) -> dict:
    """The answer to a request: its command id, and the state it is in when the answer goes out."""
    state = bus.command_state(cid) or {}
    return {"ok": True, "id": cid, "state": state.get("state", ""), **extra}


def build_app(bus: telemetry.Bus, stt=None) -> FastAPI:
    """The ASGI app. Separated from ``serve`` so the tests can drive it without a thread.

    ``stt`` replaces the faster-whisper transcriber (browser WAV bytes -> text): a test should
    not have to load a real speech model, and a bench that wants a different engine does not
    have to fork the app.
    """
    app = FastAPI(title="leisaac demo 11", docs_url=None, redoc_url=None)

    @app.get("/api/config")
    def config():
        import scene_config
        state = bus.snapshot()["state"]
        saved = state.get("scene_config", scene_config.default_config())
        catalog = state.get("catalog")
        if catalog is None:
            # No session has built one yet (a fresh supervisor, or a page opened before starting):
            # the same items the robot would build, so the table map is never empty on purpose.
            try:
                _, items = scene_config.build(saved)
                catalog = scene_config.catalog(saved, items)
            except (ValueError, TypeError):
                catalog = []
        return {"config": saved, "available_colors": list(scene_config.AVAILABLE_COLORS),
                "catalog": catalog, "layout": scene_config.layout(),
                # The presets are served rather than written in the page, so the
                # button and the tests read the same scene. ``demo`` is the one a
                # live presentation should start from; ``simple`` is the two-cube
                # minimal one; ``default`` is the full four-part scene.
                "presets": {name: scene_config.PRESETS[name]()
                            for name in scene_config.PRESET_NAMES}}

    @app.get("/api/config/preset/{name}")
    def preset(name: str):
        import scene_config
        if name not in scene_config.PRESETS:
            return JSONResponse({"error": f"Escena predefinida desconocida: {name}."},
                                status_code=404)
        try:
            canonical, items = scene_config.build(scene_config.PRESETS[name]())
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        # Saved as well as returned: the supervisor reads its scene from the bus
        # state and from disk, so a preset chosen here has to be both.
        from supervisor import atomic_json
        atomic_json(HERE.parent / "out11" / "scene_config.json", canonical)
        catalog = scene_config.catalog(canonical, items)
        bus.set_state(scene_config=canonical, catalog=catalog, error="")
        return {"ok": True, "config": canonical, "catalog": catalog,
                "layout": scene_config.layout()}

    @app.post("/api/config")
    def save_config(payload: dict):
        import scene_config
        from supervisor import atomic_json
        if bus.snapshot()["state"].get("sim") == "running" or bus.snapshot()["state"].get("phase") in ("loading", "starting", "stopping"):
            return JSONResponse({"error": "Detén el entorno antes de cambiar la escena."}, status_code=409)
        try:
            canonical, items = scene_config.build(payload)
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        atomic_json(HERE.parent / "out11" / "scene_config.json", canonical)
        catalog = scene_config.catalog(canonical, items)
        bus.set_state(scene_config=canonical, catalog=catalog, error="")
        # The catalog comes back with the answer because building it is what proves the scene is
        # buildable: the page draws the table it just saved, so a scene that cannot be placed is
        # reported before a two-minute start rather than during it.
        return {"ok": True, "config": canonical, "catalog": catalog, "layout": scene_config.layout()}

    @app.get("/api/report")
    def report():
        result = bus.snapshot()["state"].get("report")
        if result is None:
            return JSONResponse({"error": "Todavía no hay un informe de sesión."}, status_code=404)
        return JSONResponse(result, headers={"Content-Disposition": 'attachment; filename="informe-sesion.json"'})

    @app.get("/api/runtime-log")
    def runtime_log():
        name = bus.snapshot()["state"].get("runtime_log")
        if not name or not Path(name).exists():
            return JSONResponse({"error": "Todavía no hay registro de ejecución."}, status_code=404)
        return FileResponse(name, media_type="text/plain", filename="registro-sesion.log")

    @app.post("/api/voice")
    async def voice(request: Request) -> dict:
        """The browser's microphone recording, transcribed.

        One WAV blob in, one sentence out - the text goes back, and the browser puts it in the
        command box so the person can correct it before sending. Nothing is auto-sent: hearing
        the model mis-hear the order go straight to a robot arm is worse than one extra click.
        """
        payload = await request.body()
        if not payload:
            return JSONResponse({"error": "audio vacío"}, status_code=400)
        if len(payload) > 4_000_000:
            return JSONResponse({"error": "La grabación es demasiado grande."}, status_code=413)
        model_name = request.query_params.get("model") or STT_MODEL
        language = request.query_params.get("language") or None
        # Transcription is seconds of CPU. It runs in a worker thread, so the event loop - and
        # with it the live picture and the controls - keeps answering while it works.
        if stt is not None:
            text = await asyncio.to_thread(stt, payload)
        else:
            try:
                text = await asyncio.to_thread(transcribe_wav, payload, model_name, language)
            except ImportError:
                return JSONResponse(
                    {"error": "faster-whisper no está instalado en este entorno"},
                    status_code=503)
            except Exception as exc:  # noqa: BLE001 - bad audio must answer, not 500
                return JSONResponse({"error": f"no se pudo transcribir: {exc}"},
                                    status_code=400)
        return {"ok": True, "text": text}

    @app.get("/")
    def index():
        if not INDEX.exists():
            return JSONResponse({"error": "No se encuentra el archivo del panel."}, status_code=500)
        # La página se edita en caliente; sin esto el navegador servía una copia guardada y el
        # panel no reflejaba los cambios hasta vaciar la caché a mano.
        return FileResponse(INDEX, media_type="text/html",
                            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                                     "Pragma": "no-cache", "Expires": "0"})

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
            return JSONResponse({"error": "Todavía no hay un fotograma."}, status_code=404)
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
        # Every request is answered at once with its id and where it stands. The robot moves it
        # on (RUNNING, then SUCCESS or FAILED) when it gets to it; the page follows the snapshot.
        if action == "start":
            cid = bus.send_control("start")
            bus.open_start()
            return _ack(bus, cid, action="start")
        if text:
            cid = bus.submit(text)
            return _ack(bus, cid, queued=text)
        if action:
            params = {k: v for k, v in payload.items() if k != "action"}
            cid = bus.send_control(action, **params)
            return _ack(bus, cid, queued=action)
        return JSONResponse({"error": "Envía una orden de texto o una acción."}, status_code=400)

    @app.post("/api/env")
    async def env(payload: dict, request: Request) -> dict:
        action = str(payload.get("action") or "").strip()
        if not action:
            return JSONResponse({"error": "Falta la acción solicitada."}, status_code=400)
        params = {k: v for k, v in payload.items() if k != "action"}
        if action == "start":
            # The setup the dashboard chose travels with the go-ahead: the demo is already
            # past its CLI by the time this button is pressed, so these override its defaults.
            cid = bus.send_control("start", **params)
            bus.open_start()
            return _ack(bus, cid, action="start", params=params)
        cid = bus.send_control(action, **params)
        return _ack(bus, cid, queued=action, params=params)

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        """Push snapshots and frames; read commands and controls.

        Two tasks, because this socket has to do both without polling: one waits for the bus
        to change and pushes, the other reads what the browser sent. Polling the bus on a timer
        would either be late or would flood the socket, and the bus knows exactly when
        something changed.

        Two things about the push side are load-bearing and both were bugs first:

        **A connection is answered immediately, without waiting for a change.** The dashboard
        is meant to be open *before* the simulator starts, and that is exactly when nothing is
        happening: ``main`` publishes the bench sheet, then waits up to two minutes for somebody
        to press start. A handler that only spoke when the bus changed left the page blank for
        all of it, so the one screen that exists to explain the wait showed nothing at all. It
        read as a server that did not come up. Measured: connecting and receiving nothing for
        five seconds, with the process healthy and ``/api/state`` answering the whole time.

        **Pushes are keyed on a version counter and rate-limited**, never on an event that can
        stay set. See ``PUSH_MIN_INTERVAL_S`` for the measurement - 2171 snapshots per second
        from a single bus change - and for why that made both the page and the simulator look
        broken. ``pending["v"]`` is bumped by the hook from whichever thread the robot is on,
        and a push happens only when it differs from what was last sent, so a burst of writes
        coalesces into one push and a quiet bus produces no pushes at all.
        """
        await websocket.accept()
        bus.add_to_metric("websocket_clients", 1)
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        pending = {"v": 0}

        def hook() -> None:
            # Called from whichever thread the robot is on - possibly the one inside
            # ``env.step`` - so the hand-off to the loop has to be explicit. Bumping a counter
            # is what decides whether a push is due: several changes between two pushes are
            # then one push rather than several. The event is only a "go and look" knock, so
            # waking on it when nothing changed is harmless - the loop re-reads the version and
            # goes back to sleep.
            loop.call_soon_threadsafe(_bump)

        def _bump() -> None:
            pending["v"] += 1
            wake.set()

        bus.add_hook(hook)
        reader = asyncio.create_task(_reader(websocket, bus))
        sent = -1          # the bus version already pushed; -1 means "nothing yet"
        sent_events = None  # the last log sequence number pushed; None sends the recent history
        last_push = 0.0
        last_frame = 0
        try:
            while True:
                version = pending["v"]
                if version == sent:
                    # Nothing new. Sleep until either the bus changes or the browser speaks.
                    # The event is cleared first, because it is only ever *set* and leaving it
                    # set would make the next waiter return at once - which is the spin this
                    # loop is written to not have. The version is re-read afterwards so a
                    # change landing in that window is not lost.
                    wake.clear()
                    if pending["v"] != version:
                        continue
                    waiter = asyncio.create_task(wake.wait())
                    done, _pending = await asyncio.wait(
                        {reader, waiter}, return_when=asyncio.FIRST_COMPLETED)
                    waiter.cancel()
                    if reader in done:
                        break  # the browser hung up, or sent something malformed
                    continue

                # Something changed. Push at most once per PUSH_MIN_INTERVAL_S: if a push is
                # not due yet, wait for it and re-read the version, so the writes that arrive
                # in the meantime cost nothing.
                due = last_push + PUSH_MIN_INTERVAL_S
                now = time.monotonic()
                if sent >= 0 and now < due:
                    await asyncio.sleep(due - now)
                    continue
                sent = pending["v"]
                last_push = time.monotonic()
                # Only the log lines this socket has not been sent yet: the page already holds the
                # rest, and it deduplicates by sequence number, so a delta is enough.
                snap = bus.snapshot(events_after=sent_events)
                await websocket.send_json({"type": "snapshot", **snap})
                sent_events = snap["seq"]
                _seq, jpeg, meta = bus.frame()
                if jpeg is not None and _seq > last_frame:
                    await websocket.send_json({"type": "frame", "seq": _seq, "meta": meta})
                    await websocket.send_bytes(jpeg)
                    last_frame = _seq
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            bus.drop_hook(hook)
            reader.cancel()
            bus.add_to_metric("websocket_clients", -1)

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

    def __init__(self, bus: telemetry.Bus, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 stt=None):
        import uvicorn

        self.bus = bus
        self.host = host
        self.port = port
        self.app = build_app(bus, stt)
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
          open_browser: bool = False, announce=lambda _line: None, stt=None) -> Monitor:
    """Start the dashboard and return it once it is listening."""
    monitor = Monitor(bus, host, port, stt=stt).start()
    if not monitor.wait_ready():
        announce(f"  the dashboard did not come up on {host}:{port}; is the port in use?")
        return monitor
    announce(f"  dashboard: {monitor.url}")
    if open_browser:
        monitor.open_browser()
    return monitor
