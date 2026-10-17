"""Does a WebSocket survive being served from a process that has booted Isaac Sim?

Measured after the demo's own WebSocket started answering ``403 Forbidden`` while every plain
HTTP endpoint worked. There were two candidate explanations: a mistake in ``server.py``, or
Isaac Kit breaking the WebSocket handshake for any server in the process. Only one of those is
fixable by editing ``server.py``, so this measures which it is.

The probe boots Kit -- and *only* Kit -- then serves the smallest FastAPI app that can possibly
expose a WebSocket, in a background thread, exactly as ``demo12.server`` does it. No cell, no
physics, no camera, so it runs in well under a minute.

Measured result, and the reason ``server.py`` does not use ``@app.websocket``::

    /raw       raw ASGI, no framework        -> connects, streams
    /starlette starlette.routing.WebSocketRoute -> connects, streams
    /ws        fastapi @app.websocket        -> 403 Forbidden, handler never entered

FastAPI's websocket route matches the request and then answers ``websocket.close`` before
accept, and uvicorn turns that into a bare 403 with an empty body. A plain HTTP ping is served
normally throughout, which is what made this look like a working server.

    python -u -m demo12.tools.probe_ws_isaac
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
# _HERE = .../07_multi_robot_orchestration/demo12/tools/probe_ws_isaac.py
# parents[1] = demo12, [2] = 07_multi_robot_orchestration, [3] = demos
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[3] / "01_simulation_basics"))

import env_loader  # noqa: E402

# Imported *before* Kit boots on purpose. Isaac appends its own bundled FastAPI/Starlette from
# ``omni.kit.pip_archive-*/pip_prebundle`` to ``sys.path``; whichever copy is imported first
# wins, and a pip FastAPI running against Isaac's Starlette is the mismatch measured below.
import fastapi  # noqa: E402
import starlette  # noqa: E402
import uvicorn  # noqa: E402
import websockets  # noqa: E402

print(f"[probe] web stack imported early: fastapi {fastapi.__version__} "
      f"from {fastapi.__file__}", flush=True)
print(f"[probe] starlette {starlette.__version__} from {starlette.__file__}", flush=True)


def describe_loop(label: str) -> None:
    """Report which event loop this thread would hand to uvicorn's websocket impl.

    ``uvicorn.protocols.websockets.websockets_impl`` does
    ``self.loop = _loop or asyncio.get_event_loop()`` -- with no ``_loop`` passed, it asks asyncio
    for the current loop rather than using the one uvicorn is running on. If Kit has replaced
    the loop policy, those are not the same object, and the ASGI task gets posted to a loop that
    is not the one servicing the socket.
    """
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except Exception as exc:                       # noqa: BLE001
        print(f"[probe] {label}: get_event_loop raised {type(exc).__name__}: {exc}", flush=True)
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    print(f"[probe] {label}: policy={type(asyncio.get_event_loop_policy()).__name__} "
          f"current={id(loop)} running={id(running) if running else None} "
          f"same={'yes' if running is loop else 'no'}", flush=True)


def report_modules(label: str) -> None:
    """Print which copy of the web stack is actually imported.

    Isaac ships its own FastAPI/Starlette inside ``omni.kit.pip_archive-*/pip_prebundle`` and
    puts that directory on ``sys.path``. Which copy wins is decided purely by import order, and
    the two copies are not interchangeable -- so this is worth printing rather than assuming.
    """
    for name in ("starlette", "fastapi", "uvicorn", "websockets"):
        module = sys.modules.get(name)
        if module is None:
            print(f"[probe] {label}: {name} not imported yet", flush=True)
            continue
        version = getattr(module, "__version__", "?")
        origin = getattr(module, "__file__", "?")
        bundled = "pip_prebundle" in str(origin)
        print(f"[probe] {label}: {name:<11} {version:<10} "
              f"{'BUNDLED BY ISAAC' if bundled else 'site-packages'}  {origin}", flush=True)


def main() -> int:
    report_modules("before Kit boot")
    print("[probe] booting Kit (no cell)", flush=True)
    env_loader.prepare_runtime(headless=True)
    print("[probe] Kit up", flush=True)
    report_modules("after Kit boot")

    describe_loop("main thread, Kit booted")

    import uvicorn
    import websockets
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    hit = threading.Event()

    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        hit.set()
        await websocket.accept()
        try:
            while True:
                await websocket.send_text(json.dumps({"hello": "from the Isaac process"}))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    # Starlette's own WebSocketRoute, on the same FastAPI app: same endpoint, same process,
    # but bypassing FastAPI's dependency solving. If this connects and ``/ws`` does not, the
    # fault is FastAPI's dispatch and the fix is to stop asking FastAPI to do it.
    from starlette.routing import WebSocketRoute

    async def starlette_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_text(json.dumps({"hello": "starlette route"}))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    app.router.routes.append(WebSocketRoute("/starlette", starlette_ws))

    config = uvicorn.Config(app, host="127.0.0.1", port=8123, log_level="info")
    server = uvicorn.Server(config)

    # Wrap the app so every ASGI scope is recorded. If the upgrade never produces a
    # ``websocket`` scope here, the rejection is happening in uvicorn's handshake before the
    # app is involved at all -- which is a different bug with a different fix.
    inner = config.app

    async def traced(scope, receive, send):
        print(f"[probe] scope type={scope['type']} path={scope.get('path')} "
              f"subprotocols={scope.get('subprotocols')}", flush=True)

        # Raw-ASGI control: do the handshake ourselves, with no FastAPI and no Starlette
        # routing in the path at all. If this connects while the FastAPI-routed ``/ws`` does
        # not, the defect is in the framework dispatch and can simply be avoided.
        if scope["type"] == "websocket" and scope.get("path") == "/raw":
            message = await receive()
            print(f"[probe] raw: first message {message['type']}", flush=True)
            await send({"type": "websocket.accept"})
            print("[probe] raw: accepted", flush=True)
            for index in range(2):
                await send({"type": "websocket.send",
                            "text": json.dumps({"raw": index})})
                await asyncio.sleep(0.2)
            print("[probe] raw: sent frames", flush=True)
            await send({"type": "websocket.close"})
            return
        if scope["type"] == "websocket":
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            print(f"[probe] websocket scope loop={id(asyncio.get_event_loop_policy().get_event_loop())} "
                  f"running={id(running) if running else None}", flush=True)

            # Ask the router itself, route by route, instead of inferring from the 403.
            # ``re`` is the thing I would suspect if a path regex stopped matching, and
            # nothing else in the process explains HTTP routes matching while websocket
            # routes do not.
            for route in getattr(inner, "routes", []):
                try:
                    matched, _ = route.matches(scope)
                except Exception as exc:           # noqa: BLE001
                    matched = f"{type(exc).__name__}: {exc}"
                print(f"[probe]   route {getattr(route, 'path', '?'):<24} "
                      f"{type(route).__name__:<18} matches={matched}", flush=True)

            async def watch():
                message = await receive()
                print(f"[probe]   <- {message['type']}", flush=True)
                return message

            async def traced_send(message):
                print(f"[probe]   -> {message['type']}", flush=True)
                await send(message)

            await inner(dict(scope), watch, traced_send)
            return
        await inner(scope, receive, send)

    async def guarded(scope, receive, send):
        import traceback
        try:
            await traced(scope, receive, send)
        except BaseException as exc:               # noqa: BLE001
            print(f"[probe] ASGI raised {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            raise

    traced_app = guarded

    traced_app = traced
    config.app = traced_app

    # Also record the loop uvicorn's websocket protocol binds itself to. It resolves this as
    # ``_loop or asyncio.get_event_loop()``; if Kit has left a different loop current for the
    # HTTP thread, the ASGI task lands on a loop that is not servicing the socket.
    from uvicorn.protocols.websockets import websockets_impl
    _original_init = websockets_impl.WebSocketProtocol.__init__

    def traced_init(protocol_self, *args, **kwargs):
        _original_init(protocol_self, *args, **kwargs)
        print(f"[probe] WebSocketProtocol bound to loop id={id(protocol_self.loop)}", flush=True)

    websockets_impl.WebSocketProtocol.__init__ = traced_init

    threading.Thread(target=server.run, name="probe-http", daemon=True).start()

    for _ in range(60):
        if server.started:
            break
        time.sleep(0.5)
    print(f"[probe] uvicorn started={server.started}", flush=True)

    thread_result: list[str] = []

    def in_thread() -> None:
        describe_loop("http thread")

    probe_thread = threading.Thread(target=in_thread, name="probe-describer", daemon=True)
    probe_thread.start()

    async def attempt() -> None:
        for path in ("/raw", "/starlette", "/ws"):
            try:
                async with websockets.connect(f"ws://127.0.0.1:8123{path}",
                                              open_timeout=20) as sock:
                    message = await asyncio.wait_for(sock.recv(), timeout=15)
                    thread_result.append(f"{path} OK -> {message}")
            except Exception as exc:               # noqa: BLE001
                thread_result.append(f"{path} FAIL {type(exc).__name__}: {str(exc)[:100]}")

    asyncio.run(attempt())
    probe_thread.join(timeout=5)

    print(f"[probe] FastAPI handler entered: {hit.is_set()}", flush=True)
    print("[probe] --- comparison, same process, same app, Kit booted ---", flush=True)
    for line in thread_result:
        print(f"[probe]   {line}", flush=True)
    print("[probe] -- a plain HTTP ping is served the whole time; only the upgrade fails",
          flush=True)
    print("[probe] -- demo12/server.py therefore registers /ws as a starlette WebSocketRoute",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
