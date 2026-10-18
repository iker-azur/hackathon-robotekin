"""Demo 12 -- process supervisor, the only way to change the robot count.

Isaac Lab cannot insert an articulation into a live scene, and the in-process rebuild hangs
(measured: 370 CPU-seconds, no return -- ``docs/PHYSICS.md`` section 9). So the roster is
fixed when the process starts, and changing it means starting a different process.

This is that process. It owns a child ``demo12.server`` and restarts it with a new
``--robots`` when asked. The parts do not survive that, so before each restart it saves the
task ledger (see :mod:`demo12.ledger`) and the server loads it back.

Control is a file, not an HTTP call, and deliberately so: the thing being asked to do the work
is the process that has to die. ``out12/control.json`` is polled by this supervisor; writing
it is something any process can do without holding a socket open to the one it wants to
replace.

    launcher.py --robots 2 --port 8000
    # from anywhere, including a UI button:
    echo '{"robots": 3}' > out12/control.json

The supervisor prints what it did, so a relaunch is legible in the terminal rather than being
something that happened silently.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_PARENT = _HERE.parents[1]
DEMOS_ROOT = _HERE.parents[3]

CONTROL_PATH = _PARENT / "out12" / "control.json"
LEDGER_PATH = _PARENT / "out12" / "ledger.json"
SCENE_PATH = _PARENT / "out12" / "scene.json"

# How often to look for a control request. 1 s is fast enough to feel immediate and slow
# enough not to matter.
POLL_SECONDS = 1.0

# Grace period for the child to shut down before it is killed. Kit's teardown can hang, and a
# supervisor that waits forever is a supervisor that never restarts anything.
SHUTDOWN_GRACE_S = 12.0

# Backoff between automatic restarts, and the point at which the supervisor gives up. A child
# that cannot start will never start, and respawning it in a tight loop turns one clear error
# into an unreadable wall of identical ones.
RESTART_BACKOFF_S = 3.0
RESTART_BACKOFF_MAX_S = 20.0
MAX_AUTO_RESTARTS = 4


def read_control(path: Path) -> dict | None:
    """Read a control request, tolerating a BOM and a half-written file.

    Both matter on Windows. PowerShell 5.1's ``Set-Content -Encoding UTF8`` writes a BOM,
    which ``json.loads`` rejects -- so a request written from the obvious PowerShell
    one-liner was silently ignored and the supervisor sat there looking healthy. It also
    polls a file another process is writing, so a partial read is normal rather than
    exceptional.
    """
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8-sig")
        return json.loads(text)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def clear_control(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def python_executable() -> str:
    """The Isaac interpreter, not whatever ``python`` happens to be on PATH.

    The PATH python on this machine is 3.12 and has no ``isaaclab`` at all, so starting the
    child with it would fail in a way that looks like a Demo 12 bug.
    """
    override = os.environ.get("DEMO12_PYTHON")
    if override:
        return override
    candidates = [
        Path.home() / "miniconda3/envs/hackathon-robotekin/python.exe",
        Path.home() / "miniconda3/envs/hackathon-robotekin/bin/python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def spawn(robot_count: int, port: int, gui: bool,
          scene: str | None = None) -> subprocess.Popen:
    command = [python_executable(), "-u", "-m", "demo12.server",
               "--robots", str(robot_count), "--port", str(port)]
    if scene:
        command += ["--scene", scene]
    if gui:
        command.append("--gui")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PARENT) + (os.pathsep + existing if existing else "")
    print(f"[launcher] starting: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=str(DEMOS_ROOT.parent), env=env)


def resolve_scene(value: object) -> str | None:
    """Turn a control request's ``scene`` field into a path the child can open.

    A bare filename resolves against ``out12`` next to the control file, which is where the
    server writes it. An absolute path is honoured as given.

    A missing file resolves to ``None`` rather than to the path anyway, and says so: passing
    a path that does not exist would start a child that silently falls back to the default
    cell, and the interface would show an edited scene it is not actually running.
    """
    if not value:
        return None
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = SCENE_PATH.parent / candidate
    if not candidate.is_file():
        print(f"[launcher] ignoring the scene request: {candidate} does not exist",
              flush=True)
        return None
    return str(candidate)


def request_ledger_save(port: int, timeout: float = 20.0) -> str:
    """Ask the running server to write the ledger to disk, *now*, before stopping it.

    This has to be an explicit HTTP call rather than relying on shutdown handlers, and the
    reason is Windows-specific and worth writing down: ``Popen.terminate()`` sends
    ``TerminateProcess``, which kills the process immediately and runs **no** shutdown hooks.
    Uvicorn's ``on_event("shutdown")`` never fires, and ``run_loop`` never returns either, so
    the save at the end of ``server.main()`` never runs. The state is only saved if somebody
    asks for it first.

    Measured, twice. The first roster change went 2 -> 3 robots perfectly and came back with
    ``{"restored": false}`` -- the task queue had silently reverted to the default scene. And
    when this function was first written, the endpoint it called *queued* the save instead of
    performing it, so the launcher got ``{"queued": "relaunch_plan"}``, called ``terminate()``,
    and the file was never written. Asking is not enough; the answer has to say it worked.

    So this checks ``saved`` in the reply and keeps asking for a few seconds if the server
    says it did not. A supervisor that kills without asking loses the state it is supposed to
    be preserving, and a supervisor that asks and does not check loses it just the same.
    """
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/relaunch-plan"
    deadline = time.monotonic() + timeout
    last = "no attempt made"

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="POST"),
                                        timeout=max(2.0, deadline - time.monotonic())) as r:
                body = json.loads(r.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            last = f"could not reach the server to save the ledger: {exc}"
            time.sleep(0.5)
            continue

        if body.get("saved") is True:
            return f"saved {body.get('tasks')} task(s) to {body.get('path')}"

        last = json.dumps(body)[:200]
        # The server answered but did not save. Give it a moment rather than killing it and
        # losing the queue, but do not retry forever.
        time.sleep(0.5)

    return f"NOT SAVED -- the task queue will not survive this restart ({last})"


def stop(child: subprocess.Popen | None, port: int | None = None,
         save_first: bool = False) -> None:
    if child is None or child.poll() is not None:
        return

    if save_first and port is not None:
        print("[launcher] asking the server to save its ledger first", flush=True)
        outcome = request_ledger_save(port)
        # Said loudly, and said as a warning rather than a log line. A restart that loses the
        # task queue looks exactly like a successful one from the outside -- same robots, same
        # scene, empty queue -- so the only place it can be caught is here.
        if outcome.startswith("NOT SAVED"):
            print(f"[launcher]   *** {outcome}", flush=True)
            print("[launcher]   *** the task queue will come back as the default mission",
                  flush=True)
        else:
            print(f"[launcher]   {outcome}", flush=True)

    print("[launcher] asking the server to stop", flush=True)
    try:
        child.terminate()
    except OSError:
        pass
    deadline = time.monotonic() + SHUTDOWN_GRACE_S
    while time.monotonic() < deadline and child.poll() is None:
        time.sleep(0.3)
    if child.poll() is None:
        # Kit's shutdown can wedge in a loader thread. A supervisor that waits forever never
        # restarts anything, so it is killed.
        print("[launcher] did not stop in time; killing", flush=True)
        try:
            child.kill()
        except OSError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo 12 supervisor")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--scene", default=None,
                        help="a scene JSON to build instead of the default cell")
    args = parser.parse_args()

    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clear_control(CONTROL_PATH)

    robot_count = args.robots
    scene_file = resolve_scene(args.scene) if args.scene else None
    print(f"[launcher] Demo 12 supervisor: {robot_count} robot(s) on port {args.port}",
          flush=True)
    if scene_file:
        print(f"[launcher] scene: {scene_file}", flush=True)
    print(f"[launcher] change the roster with "
          f"'{{\"robots\": N}}' in {CONTROL_PATH.name}", flush=True)
    print(f"[launcher] change the scene with "
          f"'{{\"scene\": \"scene.json\"}}' in {CONTROL_PATH.name}", flush=True)

    child = spawn(robot_count, args.port, args.gui, scene_file)
    restarts = 0
    crashes = 0

    def on_signal(signum, _frame):
        print(f"\n[launcher] signal {signum}; stopping the server", flush=True)
        stop(child, args.port, save_first=True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, on_signal)
    try:
        signal.signal(signal.SIGTERM, on_signal)
    except (AttributeError, ValueError):
        pass                                  # not available on every platform

    try:
        while True:
            request = read_control(CONTROL_PATH)
            if request is not None:
                clear_control(CONTROL_PATH)
                # A request may carry a roster, a scene, or both. Both are honoured in
                # one restart rather than racing two restarts, and the ledger is saved
                # before the child dies because terminate() runs no shutdown hooks here.
                raw_robots = request.get("robots")
                wanted_scene = resolve_scene(request.get("scene"))

                roster_change = None
                if isinstance(raw_robots, int) and 1 <= raw_robots <= 3:
                    if raw_robots != robot_count:
                        roster_change = raw_robots
                elif raw_robots is not None:
                    print(f"[launcher] ignoring a request that is not 1-3 robots: "
                          f"{raw_robots!r}", flush=True)

                scene_change = (wanted_scene
                                if wanted_scene and wanted_scene != scene_file else None)

                if roster_change is None and scene_change is None:
                    if isinstance(raw_robots, int):
                        print(f"[launcher] already running with {raw_robots} robot(s)",
                              flush=True)
                    elif request.get("scene"):
                        print("[launcher] already running with that scene", flush=True)
                else:
                    restarts += 1
                    if roster_change is not None:
                        print(f"\n[launcher] roster change: {robot_count} -> "
                              f"{roster_change}", flush=True)
                    if scene_change is not None:
                        print(f"\n[launcher] scene change: {scene_file or 'default'} -> "
                              f"{scene_change}", flush=True)
                    print("[launcher] stopping the server; the ledger is saved "
                          "beforehand, because terminate() runs no shutdown hooks on "
                          "Windows", flush=True)
                    stop(child, args.port, save_first=True)
                    if roster_change is not None:
                        robot_count = roster_change
                    if scene_change is not None:
                        scene_file = scene_change
                    print(f"[launcher] restarting with {robot_count} robot(s), "
                          f"scene {scene_file or 'default'}", flush=True)
                    child = spawn(robot_count, args.port, args.gui, scene_file)

            if child.poll() is not None:
                print(f"[launcher] the server exited with {child.returncode}", flush=True)
                # Back off before restarting. A child that dies immediately -- a bad flag, a
                # missing module -- would otherwise be respawned as fast as the poll loop can
                # turn, which buries the error under hundreds of identical stack traces and
                # looks like a hang rather than a failure.
                crashes += 1
                if crashes > MAX_AUTO_RESTARTS:
                    print(f"[launcher] {crashes} restarts in a row; giving up. The last "
                          f"exit code was {child.returncode}. Fix the cause before "
                          f"starting the supervisor again.", flush=True)
                    stop(None)
                    return 1
                delay = min(RESTART_BACKOFF_S * crashes, RESTART_BACKOFF_MAX_S)
                print(f"[launcher] restarting in {delay:.0f}s", flush=True)
                time.sleep(delay)
                child = spawn(robot_count, args.port, args.gui, scene_file)
            else:
                crashes = 0

            time.sleep(POLL_SECONDS)
    except SystemExit:
        stop(child, args.port, save_first=True)
        print(f"[launcher] stopped after {restarts} roster change(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())