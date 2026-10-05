"""Run demo 11 until Isaac actually starts, because on this machine it often does not.

    python tools/run_demo11.py                                        # the demo, with retries
    python tools/run_demo11.py --text "separa las piezas por colores"
    python tools/run_demo11.py --attempts 8 --stall 120 --cap 900
    python tools/run_demo11.py --setup-only                          # nothing to retry, passes through

**Why this exists.** Isaac Sim 5.1 faults while starting up on this machine, and it is not the
demo's fault - ``10_command_to_motion.py`` does the same. Measured here over one session, a launch
either gets all the way into a round (and then runs) or dies in the first thirty seconds with:

    Windows fatal exception: code 0xc0000139      (STATUS_ENTRYPOINT_NOT_FOUND)
    Windows fatal exception: access violation

on a thread with **no Python frame**, i.e. inside Kit's native extension loader rather than in any
Python code, sometimes followed by a process that stops producing output entirely. Roughly two
launches in three were affected. What was tried, and what it proved:

* **Clearing Kit's caches** (``%LOCALAPPDATA%\\ov\\cache``: shaders, DerivedDataCache, texturecache,
  637 MB) - no change. They are rebuilt on the next launch.
* **Giving Isaac's own HDF5 the process.** Both Isaac and h5py ship a DLL called ``hdf5.dll``;
  h5py's is HDF5 2.0.0 and Isaac's is the 1.14-era library its ``hdf5_cpp.dll`` was linked against.
  ``env_loader.py`` imports h5py *before* Kit to stop Isaac's copy shadowing it, which is what
  produces this on every run:
  ``Could not load the dynamic library ... generic_mo_io.dll``. Preloading Isaac's four HDF5
  modules with ``RTLD_GLOBAL`` does make that error disappear - and then ``isaacsim.sensors.rtx``
  loads for the first time, brings torch in during Kit's start-up, and IsaacLab's later
  ``import torch._prims`` fails with *"Only a single TORCH_LIBRARY can be used to register the
  namespace prims"*. So the HDF5 clash is real and provable, and it is **not** the fatal exception:
  the working configuration is the one demo 10 already used, where that extension simply never
  loads.
* **Excluding the unused extensions** (``isaacsim.sensors.rtx``, ``omni.sensors.nv.lidar``) by
  carb setting - the setting is forwarded by ``SimulationApp`` but did not stop either extension
  from being attempted, so it is not a lever this launcher can pull.

The fault is therefore in the vendor's loader, not in the demo, and the useful thing to do about it
is not to wait: relaunch. Each attempt costs about a minute when it works and about forty seconds
when it does not, and the demo itself has nothing to do with the difference.

What this tool does
-------------------
Launches the demo, watches the log for progress, and decides per attempt:

* **completed** - the child printed its summary. Exit code 0 (everything verified) or 2 (something
  was missed) are both a successful launch and are reported as they are.
* **faulted** - the child exited early. The last signature of the fault is pulled out of its output
  so the log says *why* it is being retried rather than just doing it again.
* **stalled** - no output for ``--stall`` seconds. This is the case that would otherwise hang a
  session: it kills the child and starts again.

Relauchs kill the previous process tree, because the process that faulted can leave a Kit kernel
holding the GPU, and the next launch then stalls behind it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMO = ROOT / "11_real_sorting_with_monitor.py"

# The fault's signatures, in the order they appear, so a retry can say which one it saw.
SIGNATURES = (
    ("entry point", "STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139) in Kit's extension loader"),
    ("access violation", "access violation on a native thread"),
    ("single TORCH_LIBRARY", "torch namespace 'prims' registered twice"),
    ("Could not load the dynamic library", "an extension's DLL could not be resolved"),
    ("Recursive unloadAllPlugins", "Kit aborted while unloading"),
)

SUMMARY = re.compile(r"=== summary ===")


def fault_in(text: str) -> str:
    """The last fault signature in ``text``, or "" if there is none."""
    found = ""
    for needle, description in SIGNATURES:
        if needle in text:
            found = description
    return found


def child_processes(pid: int) -> list[int]:
    """``pid`` and any Kit kernel it started, read from the process table.

    ``taskkill`` is used rather than killing only the parent: a faulted Kit leaves its kernel
    running and holding the GPU, and a launch behind it stalls with no output - which is the
    failure this tool exists to avoid reproducing.
    """
    if os.name != "nt":
        return [pid]
    listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
    pids = [pid]
    for line in listing.splitlines():
        match = re.search(r'"[^"]+","(\d+)"', line)
        if match:
            pids.append(int(match.group(1)))
    return pids


def kill_tree(pid: int) -> None:
    for target in child_processes(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(target)],
                           capture_output=True)
        else:  # pragma: no cover - this repository's runs are on Windows
            try:
                os.kill(target, 9)
            except ProcessLookupError:
                pass


def attempt(index: int, argv: list[str], stall: float, cap: float, log_dir: Path):
    """One launch. Returns ``(verdict, log_path, detail)``."""
    log = log_dir / f"demo11_attempt{index}.log"
    started = time.monotonic()
    with log.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen([sys.executable, str(DEMO), *argv], cwd=str(ROOT),
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        last_size = -1
        last_change = started
        while True:
            time.sleep(5.0)
            if process.poll() is not None:
                text = log.read_text(encoding="utf-8", errors="replace")
                if SUMMARY.search(text):
                    code = process.returncode
                    verdict = "completed" if code == 0 else f"completed with exit code {code}"
                    # No fault to report: "Recursive unloadAllPlugins" is in every run's shutdown
                    # noise, including successful ones, so reporting it here would name a fault
                    # that did not happen.
                    detail = (f"verified everything" if code == 0
                              else f"ran, and reported {code} for an unverified placement")
                else:
                    verdict = "faulted"
                    detail = fault_in(text) or f"exited after {time.monotonic() - started:.0f}s"
                return verdict, log, detail
            size = log.stat().st_size
            if size != last_size:
                last_size = size
                last_change = time.monotonic()
            if time.monotonic() - last_change > stall:
                kill_tree(process.pid)
                return "stalled", log, f"no output for {stall:.0f}s"
            if time.monotonic() - started > cap:
                kill_tree(process.pid)
                return "capped", log, f"still running after {cap:.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempts", type=int, default=6,
                        help="launches to try before giving up")
    parser.add_argument("--stall", type=float, default=150.0, metavar="S",
                        help="seconds without output before a launch counts as stalled")
    parser.add_argument("--cap", type=float, default=900.0, metavar="S",
                        help="seconds to let one launch run before giving up on it")
    parser.add_argument("--log-dir", default=None,
                        help="where to keep each attempt's log (default: out11/runs)")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="everything after -- goes to the demo")
    options = parser.parse_args()

    passthrough = list(options.args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if not passthrough:
        passthrough = ["--no-wait-for-start"]

    log_dir = Path(options.log_dir) if options.log_dir else ROOT / "out11" / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"launching demo 11 with retries: up to {options.attempts} attempt(s), "
          f"{options.stall:.0f}s without output counts as stalled", flush=True)
    for index in range(1, options.attempts + 1):
        verdict, log, detail = attempt(index, passthrough, options.stall, options.cap, log_dir)
        print(f"attempt {index}: {verdict} - {detail}", flush=True)
        print(f"  log: {log}", flush=True)
        if verdict.startswith("completed"):
            return 0 if verdict == "completed" else 2
        if index < options.attempts:
            print("  relaunching", flush=True)
            time.sleep(5)

    print(f"\n{given_up(options.attempts)} launches, none of them got into a round. Isaac is "
          f"faulting during start-up on this machine - demo 10 does the same - so this is not "
          f"something the demo can fix. Try again in a minute, or run demo 10 to confirm it is "
          f"the machine and not the code.", flush=True)
    return 3


def given_up(attempts: int) -> str:
    return f"{attempts} of {attempts}"


if __name__ == "__main__":
    raise SystemExit(main())