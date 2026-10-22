"""Spawnable session doubles: exercise the real process transport without Isaac."""
import json
import queue
import sys
import time
from pathlib import Path


def run(arguments, inbox, updates, snapshots, stopping, report_path, log_path):
    bus_state = {"phase": "waiting", "detections": [{"kind": "ball", "color": "red"}]}
    _run(arguments, inbox, updates, snapshots, stopping, report_path, log_path, bus_state)


def _run(arguments, inbox, updates, snapshots, stopping, report_path, log_path, bus_state):
    import telemetry

    # Accept either a Namespace or a plain dict: the supervisor hands the worker a
    # Namespace built from the launch arguments, and a test that calls this directly
    # hands it a dict. Reading both the same way is the difference between a test
    # that exercises the retry and one that silently passes with nothing failing.
    def setting(name, default=0):
        if isinstance(arguments, dict):
            return arguments.get(name, default)
        return getattr(arguments, name, default)

    attempt = int(setting("session_attempt", 1))
    fail_times = int(setting("fail_times", 0))
    hang_until = float(setting("hang_until", 0.0) or 0.0)
    if attempt <= fail_times:
        _die_like_a_native_fault(report_path, log_path, setting("signature", ""))
    if hang_until:
        # The other half of the fault: a Kit kernel holding the GPU, so the launch produces
        # nothing at all until the supervisor's startup timeout kills it. The deadline is
        # an absolute monotonic stamp, so a re-launched child keeps hanging.
        while time.monotonic() < hang_until:
            time.sleep(.1)

    bus = telemetry.Bus()
    bus.set_state(**bus_state)
    snapshots.put({"snapshot": bus.snapshot(), "frame": bus.frame()})
    Path(log_path).write_text("fake session\n", encoding="utf-8")
    while not stopping.is_set():
        try:
            request = inbox.get(timeout=.05)
        except queue.Empty:
            continue
        updates.put({"type": "command", "id": request["id"], "state": "SUCCESS", "note": "simulado"})
    bus.set_state(phase="done", status="ABORTED")
    report = bus.snapshot()
    Path(report_path).write_text(json.dumps(report), encoding="utf-8")
    updates.put({"type": "result", "snapshot": report})


def _die_like_a_native_fault(report_path, log_path, signature):
    """Exit the way a failed Isaac launch does: no report, a signature in the log.

    ``0xc0000139`` is ``STATUS_ENTRYPOINT_NOT_FOUND`` and it happens inside Kit's extension
    loader, on a thread with no Python frame, so there is nothing to catch: the process
    simply goes. The log is the only evidence, and it is what the supervisor reads to
    decide why it is retrying.
    """
    default = "STATUS_ENTRYPOINT_NOT_FOUND in Kit's extension loader"
    text = f"Windows fatal exception: code 0xc0000139\n{signature or default}\n"
    Path(log_path).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    sys.stdout.flush()
    raise SystemExit(3)


