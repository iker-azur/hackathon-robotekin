"""Spawnable session double: exercise the real process transport without Isaac."""
import json
import queue
from pathlib import Path


def run(arguments, inbox, updates, snapshots, stopping, report_path, log_path):
    import telemetry
    bus = telemetry.Bus()
    bus.set_state(phase="waiting", detections=[{"kind": "ball", "color": "red"}])
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
