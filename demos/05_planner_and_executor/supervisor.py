"""Persistent dashboard; one isolated Isaac process per session (Windows spawn)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import telemetry

HERE = Path(__file__).resolve().parent


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def worker(arguments, inbox, updates, snapshots, stopping, report_path, log_path):
    """Own the simulator main thread. Never load Isaac in the supervisor."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        # Capture native Kit diagnostics as well as Python output.
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout = sys.stderr = log
        args = argparse.Namespace(**arguments)
        args.worker = True
        args.wait_for_start = False

        class WorkerBus(telemetry.Bus):
            def new_command(self, *a, **kw):
                cid = super().new_command(*a, **kw)
                updates.put({"type": "register", "request": self.command_state(cid)})
                return cid

            def set_command_state(self, cid, state, note=""):
                changed = super().set_command_state(cid, state, note)
                if changed:
                    updates.put({"type": "command", "id": cid, "state": state, "note": note})
                return changed

            def set_state(self, **state):
                super().set_state(**state)
                if state.get("phase") == "done":
                    atomic_json(Path(report_path), self.snapshot())
                    updates.put({"type": "result", "snapshot": self.snapshot()})

        bus = WorkerBus()
        bus._id_prefix = f"{args.session_id}_"
        done = threading.Event()

        def receive():
            while not done.is_set():
                if stopping.is_set():
                    bus._stop.set()
                try:
                    request = inbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                cid = request["id"]
                with bus._lock:
                    bus._cmd_log[cid] = dict(request)
                if request["type"] == "command":
                    bus._commands.put((request["text"], request["source"], cid))
                else:
                    bus._controls.put({"action": request["type"], "params": request["params"], "id": cid})

        def publish():
            revision = -1
            while not done.wait(0.05):
                snap = bus.snapshot()
                if snap["rev"] == revision:
                    continue
                revision = snap["rev"]
                packet = {"snapshot": snap, "frame": bus.frame()}
                try:
                    snapshots.put_nowait(packet)
                except queue.Full:
                    # Camera and state are latest-value channels, never an accumulating backlog.
                    try:
                        snapshots.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        snapshots.put_nowait(packet)
                    except queue.Full:
                        pass

        threading.Thread(target=receive, daemon=True).start()
        threading.Thread(target=publish, daemon=True).start()
        try:
            spec = importlib.util.spec_from_file_location("demo11_worker", HERE / "11_real_sorting_with_monitor.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            code = module.main(args=args, bus=bus)
        except BaseException as exc:
            reason = f"La sesión ha fallado: {type(exc).__name__}: {exc}"
            bus.set_state(phase="failed", error=reason, status="FAILED")
            bus.fail_active(reason)
            atomic_json(Path(report_path), bus.snapshot())
            updates.put({"type": "result", "snapshot": bus.snapshot()})
            import traceback
            traceback.print_exc()
            code = 2
        finally:
            done.set()
        raise SystemExit(code)


class Supervisor:
    def __init__(self, args, bus=None, target=worker):
        self.args = args
        self.bus = bus or telemetry.Bus()
        self.context = mp.get_context("spawn")
        self.target = target
        self.child = None
        self.start_id = ""
        self.last_event = 0
        self.last_frame = 0
        self.session_id = ""
        self.ready = False
        self.result = None
        self.stop_at = None
        self.bus.set_state(phase="waiting_start", sim="stopped", report=None)

    def start(self, control):
        cid = control["id"]
        if self.child is not None:
            self.bus.set_command_state(cid, telemetry.FAILED, "Ya hay una sesión activa.")
            return
        arguments = vars(self.args).copy()
        for key in ("variant", "camera", "rounds", "seed"):
            if key in control["params"]:
                arguments[key] = control["params"][key]
        if arguments["camera"] not in ("top", "side", "front") or arguments["variant"] not in (
                "contained", "open", "out_of_reach") or not isinstance(arguments["rounds"], int) or arguments["rounds"] < 1:
            self.bus.set_command_state(cid, telemetry.FAILED, "Configuración de entorno no válida.")
            return
        self.session_id = uuid.uuid4().hex
        arguments["session_id"] = self.session_id
        self.directory = Path(getattr(self.args, "session_dir", HERE / "out11" / "sessions")) / self.session_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.report_path = self.directory / "report.json"
        self.log_path = self.directory / "runtime.log"
        self.inbox = self.context.Queue(maxsize=telemetry.MAX_QUEUED_COMMANDS + telemetry.MAX_QUEUED_CONTROLS)
        self.updates = self.context.Queue()
        self.snapshots = self.context.Queue(maxsize=2)
        self.stopping = self.context.Event()
        self.last_event = self.last_frame = 0
        self.ready = False
        self.result = None
        self.stop_at = None
        self.start_id = cid
        self.started_at = time.monotonic()
        self.bus.prepare_session(self.session_id)
        self.bus.set_state(phase="loading", error="", status="", session_id=self.session_id,
                           runtime_log=str(self.log_path), **{k: arguments[k] for k in ("variant", "camera", "rounds", "seed")})
        self.child = self.context.Process(target=self.target, args=(arguments, self.inbox, self.updates,
                                         self.snapshots, self.stopping, str(self.report_path), str(self.log_path)))
        self.child.start()

    def merge(self, snapshot):
        state = dict(snapshot["state"])
        # The supervisor owns process lifecycle and preserves the last completed report.
        if state.get("phase") == "stopped":
            state.pop("phase")
        self.bus.set_state(**state)
        for node in snapshot["nodes"]:
            self.bus.node(node["id"], node["state"], node.get("note", ""), detail=node.get("detail", ""))
        for event in snapshot["events"]:
            if event["seq"] > self.last_event:
                self.bus.log(event["message"], level=event["level"], node=event.get("node", ""))
                self.last_event = event["seq"]
        for name, value in snapshot.get("metrics", {}).items():
            self.bus.set_metric(name, value)
        for request in snapshot.get("commands", []):
            current = self.bus.command_state(request["id"])
            if current:
                self.bus.set_command_state(request["id"], request["state"], request.get("note", ""))
        if not self.ready and state.get("detections") and state.get("phase") not in ("failed", "loading", "starting"):
            self.ready = True
            self.bus.set_command_state(self.start_id, telemetry.SUCCESS, "Escena y cámara listas.")

    def forward(self, cid):
        request = self.bus.command_state(cid)
        if not self.child or not self.ready or self.bus.stop_requested():
            self.bus.set_command_state(cid, telemetry.FAILED, "El entorno no está listo para aceptar órdenes.")
            return
        # take_command marks RUNNING in the supervisor. The worker starts from QUEUED.
        request["state"] = telemetry.QUEUED
        try:
            self.inbox.put_nowait(request)
        except queue.Full:
            self.bus.set_command_state(cid, telemetry.FAILED, "La sesión tiene demasiadas órdenes pendientes.")

    def poll(self):
        if self.child is None:
            return
        while True:
            try:
                message = self.updates.get_nowait()
            except queue.Empty:
                break
            if message["type"] == "register":
                request = message["request"]
                with self.bus._lock:
                    self.bus._cmd_log[request["id"]] = request
            elif message["type"] == "command":
                self.bus.set_command_state(message["id"], message["state"], message["note"])
            else:
                self.result = message["snapshot"]
                self.merge(self.result)
                self.bus.set_state(report=self.result)
        newest = None
        while True:
            try:
                newest = self.snapshots.get_nowait()
            except queue.Empty:
                break
        if newest:
            self.merge(newest["snapshot"])
            seq, jpeg, meta = newest["frame"]
            if jpeg and seq != self.last_frame:
                self.last_frame = seq
                self.bus.put_frame(jpeg, meta)
        if self.bus.stop_requested() and self.stop_at is None:
            self.stop_at = time.monotonic()
            self.stopping.set()
            self.bus.set_state(phase="stopping")
        if not self.ready and time.monotonic() - self.started_at > self.args.startup_timeout:
            self.bus.set_state(error="Se agotó el tiempo de arranque.")
            self.terminate()
        elif self.stop_at is not None and time.monotonic() - self.stop_at > self.args.stop_timeout:
            self.terminate()
        if not self.child.is_alive():
            self.child.join()
            if self.report_path.exists():
                self.result = json.loads(self.report_path.read_text(encoding="utf-8"))
                self.merge(self.result)
                self.bus.set_state(report=self.result)
            error = self.bus.snapshot()["state"].get("error", "")
            if not self.result and not self.bus.stop_requested():
                error = error or f"Isaac terminó sin informe (código {self.child.exitcode}). Consulte el registro de la sesión."
            self.bus.fail_active(error or "La sesión ha terminado antes de completar esta orden.")
            self.bus.complete_stop()
            self.bus.set_state(phase="failed" if error else "stopped", sim="stopped", error=error)
            for channel in (self.inbox, self.updates, self.snapshots):
                channel.close()
            self.child = None

    def terminate(self):
        if self.child is not None and self.child.is_alive():
            if os.name == "nt":
                import subprocess
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.child.pid)], capture_output=True)
            else:
                self.child.terminate()
            self.child.join(timeout=2)


def run(args):
    from monitor.server import serve
    supervisor = Supervisor(args)
    bus = supervisor.bus
    monitor = serve(bus, args.host, args.port, open_browser=args.open)
    if not monitor.server.started:
        raise RuntimeError(f"No se pudo abrir el panel en {args.host}:{args.port}.")
    print(f"Panel: {monitor.url}", flush=True)
    try:
        if not args.wait_for_start or args.text:
            bus.send_control("start")
        while True:
            supervisor.poll()
            control = bus.take_control()
            if control:
                if control["action"] == "start":
                    supervisor.start(control)
                else:
                    supervisor.forward(control["id"])
            if supervisor.ready:
                text = bus.take_command()
                if text:
                    supervisor.forward(bus.active_command_id())
            elif not supervisor.child:
                text = bus.take_command()
                if text:
                    bus.set_command_state(bus.active_command_id(), telemetry.FAILED, "Inicia el entorno antes de enviar una orden.")
            if not supervisor.child and bus.stop_requested():
                bus.complete_stop()
            time.sleep(0.02)
    except KeyboardInterrupt:
        supervisor.terminate()
    finally:
        monitor.stop()
    return 0
