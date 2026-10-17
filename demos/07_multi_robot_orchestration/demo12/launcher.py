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


def spawn(robot_count: int, port: int, gui: bool) -> subprocess.Popen:
    command = [python_executable(), "-u", "-m", "demo12.server",
               "--robots", str(robot_count), "--port", str(port)]
    if gui:
        command.append("--gui")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PARENT) + (os.pathsep + existing if existing else "")
    print(f"[launcher] starting: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=str(DEMOS_ROOT.parent), env=env)


def request_ledger_save(port: int, timeout: float = 20.0) -> str:
    """Ask the running server to requeue and save its ledger, before stopping it.

    This has to be an explicit HTTP call rather than relying on shutdown handlers, and the
    reason is Windows-specific and worth writing down: ``Popen.terminate()`` sends
    ``TerminateProcess``, which kills the process immediately and runs **no** shutdown hooks.
    Uvicorn's ``on_event("shutdown")`` never fires, so the ledger is simply never written.

    Measured: the first roster change went 2 -> 3 robots perfectly and came back with
    ``{"restored": false}`` -- the task queue had silently reverted to the default scene
    because nothing had saved it. A supervisor that kills without asking loses the state it
    is supposed to be preserving.
    """
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/relaunch-plan"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="POST"),
                                    timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")[:200]
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return f"could not reach the server to save the ledger: {exc}"


def stop(child: subprocess.Popen | None, port: int | None = None,
         save_first: bool = False) -> None:
    if child is None or child.poll() is not None:
        return

    if save_first and port is not None:
        print("[launcher] asking the server to save its ledger first", flush=True)
        print(f"[launcher]   {request_ledger_save(port)}", flush=True)

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
    args = parser.parse_args()

    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    clear_control(CONTROL_PATH)

    robot_count = args.robots
    print(f"[launcher] Demo 12 supervisor: {robot_count} robot(s) on port {args.port}",
          flush=True)
    print(f"[launcher] change the roster by writing "
          f"'{{\"robots\": N}}' to {CONTROL_PATH}", flush=True)

    child = spawn(robot_count, args.port, args.gui)
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
                wanted = request.get("robots")
                if not isinstance(wanted, int) or wanted < 1 or wanted > 3:
                    print(f"[launcher] ignoring a request that is not 1-3 robots: "
                          f"{request!r}", flush=True)
                elif wanted == robot_count:
                    print(f"[launcher] already running with {wanted} robot(s)",
                          flush=True)
                else:
                    restarts += 1
                    print(f"\n[launcher] roster change: {robot_count} -> {wanted}",
                          flush=True)
                    print("[launcher] stopping the server; the ledger is saved "
                          "beforehand, because terminate() runs no shutdown hooks on "
                          "Windows", flush=True)
                    stop(child, args.port, save_first=True)
                    robot_count = wanted
                    print(f"[launcher] restarting with {robot_count} robot(s)", flush=True)
                    child = spawn(robot_count, args.port, args.gui)

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
                child = spawn(robot_count, args.port, args.gui)
            else:
                crashes = 0

            time.sleep(POLL_SECONDS)
    except SystemExit:
        stop(child, args.port, save_first=True)
        print(f"[launcher] stopped after {restarts} roster change(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())