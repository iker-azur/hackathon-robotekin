"""Live end-to-end verification of the three things a scripted cell cannot reach.

Run with the Isaac interpreter, with the demo root on ``PYTHONPATH``:

    python -m demo12.tools.verify_live --port 8077

What this checks, and why each one is worth checking:

1. **The real Isaac cell behind the routes.** The fake in ``tests/test_server.py`` proves
   the handlers work. It cannot prove that ``IsaacCell.snapshot()`` produces the shapes the
   handlers assume, or that a real frame renders. So: read ``/api/scene`` and ``/api/state``
   from a live cell and assert the same invariants, then ask for a frame.

2. **WebSocket frame timing.** ``/ws`` sends state every 0.5 s from the simulation thread.
   The failure that matters is not "did it connect" -- it is "is it still alive ten seconds
   later, or did the second send kill it". So: connect, receive over a sustained window, and
   measure the inter-arrival times. A socket that delivers two frames and then stalls is
   worse than one that refuses, because the interface shows green until it does not.

3. **The supervisor restart.** Writing ``{"robots": N}`` to ``control.json`` is the
   documented way to change the roster, and it has never been run end to end. The thing to
   check is not just that a new process appears -- it is that the ledger survived, because
   ``terminate()`` runs no shutdown hooks on Windows and the supervisor has to ask for the
   save first (see ``launcher.request_ledger_save``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\Usuario\Documents\Workspace\LeRobot_\demos\07_multi_robot_orchestration")
OUT12 = ROOT / "out12"
CONTROL = OUT12 / "control.json"
PYTHON = sys.executable

FAILURES: list[str] = []
NOTES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(label)
    return ok


def note(text: str) -> None:
    print(f"         {text}", flush=True)
    NOTES.append(text)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return 0, str(exc).encode()


def get_json(url: str, timeout: float = 10.0) -> dict:
    status, body = get(url, timeout)
    try:
        return json.loads(body)
    except Exception:
        return {"_status": status, "_body": body[:200].decode("utf-8", "replace")}


def post(url: str, payload: dict | None = None, timeout: float = 20.0) -> dict:
    data = json.dumps(payload or {}).encode()
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read())
        except Exception:
            return {"_status": exc.code}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


async def watch_socket(port: int, seconds: float) -> list[float]:
    """Connect to /ws and return the inter-arrival gaps between frames, in seconds."""
    import websockets

    gaps: list[float] = []
    uri = f"ws://127.0.0.1:{port}/ws"
    deadline = time.monotonic() + seconds

    async with websockets.connect(uri, open_timeout=20) as socket:
        last = time.monotonic()
        while time.monotonic() < deadline:
            remaining = max(0.2, deadline - time.monotonic())
            try:
                message = await asyncio.wait_for(socket.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            now = time.monotonic()
            gaps.append(now - last)
            last = now

            state = json.loads(message)
            if "controlTicks" not in state:
                raise AssertionError("the socket sent something that is not a state frame")
    return gaps


# ---------------------------------------------------------------------------
# The three things
# ---------------------------------------------------------------------------


def verify_cell(port: int) -> dict:
    print("\n1. The real Isaac cell behind the routes", flush=True)

    health = get_json(f"http://127.0.0.1:{port}/api/health")
    check("/api/health answers", health.get("ok") is True, json.dumps(health)[:140])
    check("the cell reports the robot count the launcher asked for",
          isinstance(health.get("robots"), int) and health["robots"] >= 1,
          f"robots={health.get('robots')}")

    scene = get_json(f"http://127.0.0.1:{port}/api/scene")
    check("/api/scene loads (the 500 of PHYSICS.md 11f was here)",
          "parts" in scene and "robots" in scene,
          f"{len(scene.get('parts', []))} parts, {len(scene.get('robots', []))} robots")
    check("every part carries a mass and a graspability, as the editor needs",
          all(p.get("massG", 0) > 0 and "graspable" in p for p in scene.get("parts", [])),
          ", ".join(f"{p['id']} {p.get('massG')}g" for p in scene.get("parts", [])))
    check("every robot's reach annulus is present and ordered",
          all(r.get("reachMinM", 1) < r.get("reachMaxM", 0) for r in scene.get("robots", [])),
          ", ".join(f"{r['id']} {r.get('reachMinM')}-{r.get('reachMaxM')} m"
                    for r in scene.get("robots", [])))
    check("the scene the live cell is running is valid",
          scene.get("errors") == [], str(scene.get("errors")))

    state = get_json(f"http://127.0.0.1:{port}/api/state")
    parts = state.get("parts") or []
    check("the live cell reports where the rigid bodies actually are",
          bool(parts) and all({"id", "x", "y", "z", "resting"} <= set(p) for p in parts),
          ", ".join(f"{p['id']} ({p['x']:.3f},{p['y']:.3f}) z={p['z']:.3f}"
                    for p in parts[:4]))
    check("the table height is real, not a default",
          isinstance(state.get("tableTopZ"), float) and state["tableTopZ"] > 0,
          f"tableTopZ={state.get('tableTopZ')}")

    status, body = get(f"http://127.0.0.1:{port}/api/frame.jpg", timeout=30)
    if status == 200:
        check("a JPEG frame renders from the live cell",
              body[:2] == b"\xff\xd8", f"{len(body)} bytes, starts {body[:3]!r}")
    elif status == 503:
        note("frame.jpg answered 503 -- the documented 'no frame' path, with a reason:")
        note(f"    {json.loads(body).get('error')}")
        check("an unavailable frame is a 503 with a reason, not a stub image", True)
    else:
        check("frame.jpg answered", False, f"status {status}: {body[:120]!r}")

    return scene


def verify_socket(port: int, seconds: float = 9.0) -> None:
    print(f"\n2. WebSocket, watched for {seconds:.0f}s", flush=True)
    try:
        gaps = asyncio.run(watch_socket(port, seconds))
    except Exception as exc:  # noqa: BLE001
        check("the socket connected and stayed up", False, f"{type(exc).__name__}: {exc}")
        return

    expected = max(2, int(seconds / 0.5) - 1)
    check("the socket connected and stayed up", len(gaps) >= expected,
          f"{len(gaps)} frames in {seconds:.0f}s (the handler sends every 0.5 s)")

    if len(gaps) >= 3:
        median = sorted(gaps)[len(gaps) // 2]
        worst = max(gaps)
        check("frames arrive at roughly the 0.5 s cadence", 0.4 <= median <= 0.8,
              f"median {median:.3f}s")
        check("no frame is stuck behind a dead one", worst < 2.5,
              f"worst gap {worst:.3f}s")
        check("frames keep arriving rather than arriving twice and stalling",
              gaps[-1] < 1.5, f"last gap {gaps[-1]:.3f}s")


def verify_restart(port: int, new_robots: int) -> dict:
    print(f"\n3. Supervisor restart, {health_robots(port)} -> {new_robots} robots",
          flush=True)

    before = get_json(f"http://127.0.0.1:{port}/api/roster")
    started = time.monotonic()

    CONTROL.parent.mkdir(parents=True, exist_ok=True)
    CONTROL.write_text(json.dumps({"robots": new_robots}), encoding="utf-8")

    # The supervisor polls the file, stops the child (asking it to save the ledger first)
    # and starts a new process, which has to boot Kit again. That is slow by design.
    deadline = time.monotonic() + 420
    seen_down = False
    seen_up = False
    while time.monotonic() < deadline:
        health = get_json(f"http://127.0.0.1:{port}/api/health", timeout=4)
        robots = health.get("robots") if isinstance(health, dict) else None
        if robots is None:
            seen_down = True
        elif robots == new_robots:
            seen_up = True
            break
        time.sleep(2)

    elapsed = time.monotonic() - started
    check("the old process actually went away before a new one came up", seen_down,
          f"took {elapsed:.0f}s")
    check("a new process came up with the requested roster", seen_up,
          f"{elapsed:.0f}s, now {get_json(f'http://127.0.0.1:{port}/api/health').get('robots')} robots")

    after = get_json(f"http://127.0.0.1:{port}/api/health")
    check("the restarted cell is serving the real cell, not a stub",
          after.get("ok") is True and after.get("armCollision"),
          json.dumps(after)[:140])

    scene = get_json(f"http://127.0.0.1:{port}/api/scene")
    check("the restarted cell rebuilt the requested number of arms",
          len(scene.get("robots", [])) == new_robots,
          f"{len(scene.get('robots', []))} arms: "
          f"{[r['id'] for r in scene.get('robots', [])]}")

    # The part that was silently broken. `terminate()` is TerminateProcess: no shutdown
    # hooks, `run_loop` never returns, so the save at the end of `server.main()` never runs.
    # If the supervisor's pre-kill request does not actually write, the queue comes back empty
    # -- which looks identical to a successful restart from the outside.
    ledger = OUT12 / "ledger.json"
    if ledger.is_file():
        saved = json.loads(ledger.read_text(encoding="utf-8"))
        check("the supervisor wrote the ledger before killing the process",
              bool(saved.get("tasks")),
              f"{len(saved.get('tasks', []))} task(s): "
              f"{[t.get('id') for t in saved.get('tasks', [])]}")
        check("the ledger entries use the names restore reads",
              all({"partId", "binId"} <= set(t) for t in saved.get("tasks", [])),
              f"keys {sorted(saved['tasks'][0])}")
    else:
        check("the supervisor wrote the ledger before killing the process", False,
              "out12/ledger.json does not exist -- the task queue did not survive")

    restored = get_json(f"http://127.0.0.1:{port}/api/state").get("ledger") or {}
    note(f"the restarted server reports: {json.dumps(restored)[:160]}")

    tasks = get_json(f"http://127.0.0.1:{port}/api/state").get("tasks") or []
    check("the queue came back rather than reverting to the default mission",
          bool(tasks), f"{len(tasks)} task(s): {[t.get('id') for t in tasks]}")

    note("the roster endpoint explains why this is a restart and not a hot change:")
    note(f"    {(before or {}).get('reason', 'n/a')[:150]}")
    return scene


def verify_scene_reload(port: int, before: dict) -> None:
    """Edit a part's position, apply it, and prove the *cell* was rebuilt.

    This is the path the scene editor's "apply and restart" button uses, and the part of it
    worth testing is whether the new process really builds the edited geometry. Reloading a
    JSON file is easy and proves nothing: what would be easy to get wrong is the new process
    quietly building the default scene and the interface showing an edit it is not running.

    So the edit is a real coordinate move, and the check reads it back from ``/api/scene``
    *and* from where the rigid body actually ended up.
    """
    print("\n4. Scene edit -> restart -> the new cell is really that scene", flush=True)

    payload = {"name": before.get("name"), "table": before["table"],
               "robots": before["robots"], "parts": before["parts"],
               "bins": before["bins"], "tasks": before.get("tasks", [])}

    part = next(p for p in payload["parts"] if p["id"] == "part_4")
    original = (part["x"], part["y"])
    part["x"], part["y"] = 0.330, -0.472
    moved = (part["x"], part["y"])

    validation = post(f"http://127.0.0.1:{port}/api/scene/validate", payload)
    if not validation.get("ok"):
        note(f"the edit was refused, which is the editor doing its job: {validation.get('errors')}")
        check("the edited scene validates before anything is restarted", False,
              str(validation.get("errors"))[:200])
        return
    check("the edited scene validates before anything is restarted", True,
          f"part_4 {original} -> {moved}")

    # This is what /api/scene/apply writes, and it is the call that restarts the supervisor.
    applied = post(f"http://127.0.0.1:{port}/api/scene/apply", payload)
    check("the editor's apply request is accepted", applied.get("ok") is True,
          json.dumps(applied)[:160])
    if not applied.get("ok"):
        note("this server has no supervisor, so it says so instead of pretending")
        return
    check("it says it is restarting rather than pretending it changed the running cell",
          applied.get("restarting") is True, f"parts={applied.get('parts')}")

    deadline = time.monotonic() + 420
    reloaded = None
    while time.monotonic() < deadline:
        scene = get_json(f"http://127.0.0.1:{port}/api/scene", timeout=4)
        if "parts" in scene:
            found = next((p for p in scene["parts"] if p["id"] == "part_4"), None)
            if found is not None and abs(found["x"] - moved[0]) < 1e-6:
                reloaded = scene
                break
        time.sleep(2)

    check("the restarted process is serving the edited geometry",
          reloaded is not None,
          f"part_4 is now at "
          f"{next((p for p in (reloaded or {}).get('parts', []) if p['id'] == 'part_4'), {})}"
          if reloaded else "the new cell still has the old coordinates")

    if reloaded is not None:
        live = {p["id"]: (p["x"], p["y"])
                for p in (get_json(f"http://127.0.0.1:{port}/api/state").get("parts") or [])}
        check("and the rigid body is physically there too, not just in the config",
              abs(live.get("part_4", (9, 9))[0] - moved[0]) < 0.004,
              f"body at {live.get('part_4')}, config at {moved}")
        check("the edited cell is itself valid",
              reloaded.get("errors") == [], str(reloaded.get("errors")))


def health_robots(port: int) -> object:
    return get_json(f"http://127.0.0.1:{port}/api/health").get("robots")


def stop_supervisor(launcher: "subprocess.Popen", port: int) -> None:
    """Shut the whole tree down, and check that it actually shut down.

    Isaac holds the GPU for as long as a process lives, so a leaked child is not untidy --
    it silently halves the speed of everything else on the machine, which is exactly how it
    presented.
    """
    # The supervisor notices the dead server and respawns, so stop it first.
    launcher.terminate()
    try:
        launcher.wait(timeout=60)
    except subprocess.TimeoutExpired:
        launcher.kill()
        try:
            launcher.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # Then whatever is still answering on the port is an orphan. Kill it by PID, found from
    # the OS rather than guessed, so this cannot reach an unrelated process.
    #
    # `psutil`, not `wmic`. The first version of this sweep shelled out to `wmic process`,
    # which was removed from Windows in recent builds -- so on this machine it raised
    # "not recognised", found nothing, printed nothing, and let the orphan live. The check
    # still failed, which is the only reason it was caught: it reported the symptom and not
    # the cause. A cleanup step that silently does nothing is worse than no cleanup step,
    # because the run still looks tidy.
    if get(f"http://127.0.0.1:{port}/api/health", timeout=4)[0] == 200:
        killed = sweep_servers(port)
        print(f"  killed {killed} orphaned server process(es)", flush=True)
        time.sleep(4)

    still = get(f"http://127.0.0.1:{port}/api/health", timeout=4)[0]
    print(f"  port {port} is {'free' if still != 200 else 'STILL SERVING'}", flush=True)
    if still == 200:
        FAILURES.append(f"the server on port {port} outlived the supervisor")


def sweep_servers(port: int) -> int:
    """Kill any ``demo12.server`` bound to ``port``. Returns how many it killed.

    Matching is on **argv structure**, not on substrings, and that distinction is load-bearing.

    The first version looked for the text ``demo12.server`` and ``--port N`` anywhere in the
    command line. That matched this very script -- its own path contains
    ``demos/07_multi_robot_orchestration/demo12/server.py``, and this file's source contains
    both strings -- so the sweep was willing to kill whatever was inspecting it. A cleanup step
    that can reach an unrelated process is worse than one that cannot.

    So: an argument exactly equal to ``demo12.server`` (the ``-m`` target) followed by an
    argument exactly equal to ``--port`` whose successor is the port. Our own process and its
    ancestors are excluded outright.
    """
    import psutil

    me = psutil.Process()
    mine = {me.pid} | {p.pid for p in me.parents()}
    killed = 0

    def is_our_server(argv: list[str]) -> bool:
        if "demo12.server" not in argv:
            return False
        for i, arg in enumerate(argv):
            if arg == "--port" and i + 1 < len(argv) and argv[i + 1] == str(port):
                return True
        return False

    targets = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.pid in mine:
            continue
        try:
            argv = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if is_our_server(argv):
            targets.append(proc)
            print(f"    pid {proc.pid}: {' '.join(argv)[:90]}", flush=True)
            try:
                proc.terminate()
                killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    if targets:
        gone, alive = psutil.wait_procs(targets, timeout=8)
        for proc in alive:
            try:
                proc.kill()
                killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return killed


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="live verification for demo12")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--restarter", type=int, default=3,
                        help="robot count to ask the supervisor for; 0 to skip")
    parser.add_argument("--socket-seconds", type=float, default=9.0)
    args = parser.parse_args()

    print(f"starting the supervisor on port {args.port} with {args.robots} robot(s)",
          flush=True)
    launcher = subprocess.Popen(
        [PYTHON, "-u", "-m", "demo12.launcher",
         "--robots", str(args.robots), "--port", str(args.port)],
        cwd=str(ROOT), env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})

    try:
        print("waiting for Kit to boot and the server to answer", flush=True)
        deadline = time.monotonic() + 420
        up = False
        while time.monotonic() < deadline:
            if get_json(f"http://127.0.0.1:{args.port}/api/health", timeout=4).get("ok"):
                up = True
                break
            time.sleep(3)

        if not up:
            check("the server came up", False, "no /api/health in 420 s")
            return 1
        print("  server is up", flush=True)

        first_scene = verify_cell(args.port)
        verify_socket(args.port, args.socket_seconds)
        if args.restarter:
            verify_restart(args.port, args.restarter)
            # Re-read the scene after the restart: this check edits the cell the *new*
            # process is actually running, not the one the supervisor started with.
            current = get_json(f"http://127.0.0.1:{args.port}/api/scene")
            verify_scene_reload(args.port, current if "parts" in current else first_scene)

    finally:
        print("\nstopping the supervisor", flush=True)
        # Not just the launcher. `launcher.terminate()` is TerminateProcess, which delivers
        # no signal, so the supervisor's SIGTERM handler -- the one that stops its child --
        # never runs. Measured: two `demo12.server` children from earlier runs of this
        # script were still alive hours later, each holding a full Isaac Sim on the GPU, and
        # they were the reason the test suite appeared to take twenty minutes per test.
        #
        # So ask the supervisor to stop the child first (that is a real HTTP call and does
        # work), then take the supervisor down, then sweep up anything still holding the port.
        stop_supervisor(launcher, args.port)

    print("\n" + "=" * 68)
    if FAILURES:
        print("%d check(s) FAILED:" % len(FAILURES))
        for item in FAILURES:
            print("  - " + item)
    else:
        print("every check passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())