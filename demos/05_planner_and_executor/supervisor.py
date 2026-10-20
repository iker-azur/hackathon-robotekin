"""Persistent dashboard; one isolated Isaac process per session (Windows spawn)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

import telemetry

HERE = Path(__file__).resolve().parent

# How many times a session is launched before giving up. Isaac Sim faults natively
# during start-up on this machine - a thread with no Python frame dies with
# STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139) inside Kit's extension loader - and there
# is nothing in this repository that can prevent it. So the supervisor retries, and
# only retries, because the alternative is a dashboard that says "no ha arrancado"
# and a person pressing the button again by hand.
MAX_START_ATTEMPTS = 3

# How long a child may take to become ready before it counts as a failed attempt.
# Measured warm: ~45 s from launch to "escena y cámara listas". The default leaves
# roughly twice that, because a cold asset cache is a slow asset cache.
DEFAULT_STARTUP_TIMEOUT_S = 300.0

# The signatures a failed Isaac start-up leaves in its log, matched so the
# supervisor can say WHY it is retrying instead of only that it is. Ordered as
# they appear in the log, so the last match is the most specific.
FAULT_SIGNATURES = (
    (re.compile(r"fatal exception: code 0xc0000139"), "STATUS_ENTRYPOINT_NOT_FOUND en el cargador de extensiones de Kit"),
    (re.compile(r"fatal exception: access violation"), "violación de acceso en un hilo nativo"),
    (re.compile(r"single TORCH_LIBRARY"), "el espacio de nombres 'prims' de torch se registró dos veces"),
    (re.compile(r"Could not load the dynamic library"), "una extensión no pudo resolver su DLL nativa"),
    (re.compile(r"Recursive unloadAllPlugins"), "Kit abortó al descargar sus plugins"),
)


def fault_in(text: str) -> str:
    """The most specific fault signature in ``text``, or "" if there is none.

    ''Recursive unloadAllPlugins'' is in every run's shutdown noise, including
    successful ones, so it is the last entry and is only reached when nothing else
    matched - which is exactly the case where it is worth reporting.
    """
    found = ""
    for pattern, description in FAULT_SIGNATURES:
        if pattern.search(text):
            found = description
    return found


def fault_in_log(path) -> str:
    """``fault_in`` over a session log, tolerant of one that was never written."""
    try:
        return fault_in(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""


def worker_error_before_ready(state: dict) -> str:
    """Find a Python startup error so it is not mistaken for an Isaac native crash."""
    if state.get("phase") == "failed":
        return str(state.get("error") or "")
    return ""


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)



class SessionLog:
    """The session log as a stream that Kit cannot take away from us.

    Pointing fd 1 and 2 at this file (so Kit's C++ diagnostics land next to Python's) and
    installing it as ``sys.stdout``/``sys.stderr`` is not enough, and the reason is not in this
    file. Isaac Lab's ``AppLauncher._create_app`` sets ``sys.stdout`` to devnull while Kit boots
    and then restores ``sys.__stdout__`` - its own comment calls it "purge the print statements
    from the simulation app" - and Kit's ``omni.kit.app`` installs a ``StreamInterceptor`` over
    both streams that forwards to ``sys.__stdout__`` and ``sys.__stderr__``. So after
    ``prepare_runtime`` returns, anything that prints prints through whichever of those won, and
    the last one is fd 1: a descriptor this process dup2'd onto the log and has no other claim
    on. On this machine that descriptor goes invalid part-way through Kit's start-up, and the
    next ``print`` into it raises ``OSError: [WinError 6] Controlador no válido`` - invalid
    handle - which is what killed a session at the first ``print`` in
    ``isaaclab_tasks.utils.parse_cfg`` (``load_cfg_from_registry``'s
    ``[INFO]: Parsing configuration from: ...``) about four seconds into a Kit start-up that had
    otherwise succeeded, and demo 11 reports that as "el entorno no ha arrancado" because it
    wraps the whole environment build in ``except BaseException``.

    Two things here make that class of failure impossible rather than unlikely:

    * the log is installed on all four names - ``sys.stdout``, ``sys.stderr`` and the
      ``sys.__stdout__``/``sys.__stderr__`` pair every replace-and-restore dance lands back on -
      so no matter who swaps the streams or when, the next write comes back here;
    * writes go through the file object this class owns, and if that write fails anyway the log
      is reopened in append mode and the write retried once. A log that cannot be written must
      not be able to take a session down with it, so ``write`` does not raise.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._stream = open(self.path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
        # Native Kit diagnostics as well as Python output; the pair ``sys.__std*__`` is set
        # below, and this is the part of it that makes C++ writes land in the same file.
        os.dup2(self._stream.fileno(), 1)
        os.dup2(self._stream.fileno(), 2)
        sys.stdout = sys.stderr = sys.__stdout__ = sys.__stderr__ = self

    def fileno(self) -> int:
        return self._stream.fileno()

    def write(self, text: str) -> int:
        try:
            return self._stream.write(text)
        except (OSError, ValueError):
            # The handle this process prints through went away. Reopen and retry once: either
            # the log is writable again, or there is nothing left to say about it.
            if self._reopen():
                try:
                    return self._stream.write(text)
                except (OSError, ValueError):
                    pass
            return 0

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        with suppress(OSError, ValueError):
            self._stream.flush()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def close(self) -> None:
        self.flush()
        with suppress(OSError, ValueError):
            self._stream.close()

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    def __getattr__(self, name):
        # Anything a library expects of a stream (encoding, errors, buffer, closed, __iter__)
        # is answered by the stream underneath, so this object is a drop-in for one. The lookup
        # goes through ``__dict__`` so a stream that was never built raises AttributeError
        # instead of recursing into today.
        try:
            return getattr(self.__dict__["_stream"], name)
        except KeyError:
            raise AttributeError(name) from None

    def _reopen(self) -> bool:
        with suppress(OSError, ValueError):
            self._stream.close()
        try:
            self._stream = open(self.path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        except OSError:
            return False
        with suppress(OSError, ValueError):
            os.dup2(self._stream.fileno(), 1)
            os.dup2(self._stream.fileno(), 2)
        return True


def worker(arguments, inbox, updates, snapshots, stopping, report_path, log_path):
    """Own the simulator main thread. Never load Isaac in the supervisor."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with SessionLog(log_path) as log:
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
        # How many launches a start button may get, and which one is running.
        self.max_attempts = max(1, int(getattr(args, "start_attempts", MAX_START_ATTEMPTS)))
        self.attempt = 0
        self.session_arguments = None
        self.session_config = None
        self.session_items = None
        self.session_catalog = []
        self.bus.set_state(phase="waiting_start", sim="stopped", report=None,
                           max_start_attempts=self.max_attempts)

        import scene_config
        saved = HERE / "out11" / "scene_config.json"
        try:
            config, items = scene_config.load(getattr(args, "scene_config", None) or (saved if saved.exists() else None))
            self.bus.set_state(scene_config=config, catalog=scene_config.catalog(config, items),
                               available_colors=list(scene_config.AVAILABLE_COLORS))
            if scene_config.LAST_LOAD_NOTICE:
                # The saved scene could not be run (for example a tray the arm cannot reach);
                # the person must see why the panel shows the demo scene instead of theirs.
                self.bus.log(scene_config.LAST_LOAD_NOTICE, level="warn")
        except (ValueError, OSError) as exc:
            self.bus.set_state(error=f"No se pudo leer la escena: {exc}", phase="failed")

    def start(self, control):
        cid = control["id"]
        self.bus.set_command_state(cid, telemetry.RUNNING)
        if self.child is not None:
            self.bus.set_command_state(cid, telemetry.FAILED, "Ya hay una sesión activa.")
            return
        arguments = vars(self.args).copy()
        import scene_config
        try:
            config = control["params"].get("scene_config") or self.bus.snapshot()["state"].get("scene_config")
            config, items = scene_config.build(config)
        except (ValueError, TypeError) as exc:
            self.bus.set_command_state(cid, telemetry.FAILED, f"Escena no válida: {exc}")
            return
        arguments["scene_data"] = config
        for key in ("variant", "camera", "rounds", "seed"):
            if key in control["params"]:
                arguments[key] = control["params"][key]
        if arguments["camera"] not in ("top", "side", "front") or arguments["variant"] not in (
                "contained", "open", "out_of_reach") or type(arguments["rounds"]) is not int or not 1 <= arguments["rounds"] <= 100 or type(arguments["seed"]) is not int:
            self.bus.set_command_state(cid, telemetry.FAILED, "Configuración de entorno no válida.")
            return
        # Everything the attempts below need, kept so a retry reuses the scene and
        # the session settings rather than rebuilding them from the start button.
        self.session_arguments = arguments
        self.session_config = config
        self.session_items = items
        self.session_catalog = scene_config.catalog(config, items)
        self.attempt = 0
        self.start_id = cid
        self.launch()

    def launch(self):
        """Spawn one Isaac worker. ``attempt`` is 1-based and owned by the caller."""
        import scene_config
        arguments = self.session_arguments
        self.attempt += 1
        self.session_id = uuid.uuid4().hex
        arguments["session_id"] = self.session_id
        # Which attempt this is, for the worker. A spawn target gets a fresh copy of its
        # own module on every launch, so nothing else carries it: a child told how many
        # launches to fail can then fail exactly the first two and work on the third,
        # which is what the start-up tests exercise.
        arguments["session_attempt"] = self.attempt
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
        self.started_at = time.monotonic()
        self.bus.prepare_session(self.session_id)
        self.bus.set_state(phase="loading", error="", status="", session_id=self.session_id,
                           scene_config=self.session_config, catalog=self.session_catalog,
                           available_colors=list(scene_config.AVAILABLE_COLORS),
                           runtime_log=str(self.log_path),
                           start_attempt=f"{self.attempt}/{self.max_attempts}",
                           **{k: arguments[k] for k in ("variant", "camera", "rounds", "seed")})
        note = (f"Arrancando Isaac Sim (intento {self.attempt} de {self.max_attempts})"
                if self.attempt > 1 else "Arrancando Isaac Sim")
        self.bus.log(note, level="info" if self.attempt == 1 else "warn")
        if self.start_id:
            self.bus.set_command_state(self.start_id, telemetry.RUNNING, note)
        self.child = self.context.Process(target=self.target, args=(arguments, self.inbox, self.updates,
                                          self.snapshots, self.stopping, str(self.report_path), str(self.log_path)))
        self.child.start()

    def retry_or_give_up(self, reason: str) -> None:
        """Retry a worker that died before it was ready, or say so and stop.

        Called only from ``poll`` when the child is gone and never became ready. A
        stop requested by a person, and a session that completed, are not retried:
        the first is somebody's decision and the second already did its job.
        """
        # Whichever way this ends, the child of THIS attempt is finished. Closing it
        # here is what stops ``poll`` from seeing a live process and walking the
        # whole timeout path again - which is the bug this replaced: an exhausted
        # session left the child object set and retried forever, one timeout per
        # poll, until somebody killed the process.
        for channel in (self.inbox, self.updates, self.snapshots):
            with suppress(Exception):
                channel.close()
        self.child = None
        if self.attempt >= self.max_attempts:
            note = (f"No se ha podido iniciar Isaac Sim después de {self.attempt} intentos. "
                    f"Último motivo: {reason}")
            self.bus.set_state(phase="failed", error=note, sim="stopped")
            self.bus.log(note, level="fail")
            if self.start_id:
                self.bus.set_command_state(self.start_id, telemetry.FAILED, note)
            self.bus.fail_active(note)
            return
        self.bus.log(f"Error de arranque ({reason}). Reintentando "
                     f"({self.attempt + 1}/{self.max_attempts}).", level="warn")
        self.bus.set_state(phase="loading", error="",
                           start_attempt=f"{self.attempt + 1}/{self.max_attempts}")
        # A faulted Kit can leave its kernel holding the GPU, and the next launch
        # then stalls behind it with no output at all - which is the failure this
        # whole retry exists to not reproduce. Kill the tree, not just the parent.
        self.terminate()
        # The log is on disk per attempt, so the next one starts clean and the old
        # one stays downloadable from its own session folder.
        self.launch()


    def merge(self, snapshot):
        state = dict(snapshot["state"])
        # The supervisor owns process lifecycle and preserves the last completed report.
        if state.get("phase") == "stopped":
            state.pop("phase")
        self.bus.set_state(**state)
        current_nodes = {node["id"]: node for node in self.bus.snapshot()["nodes"]}
        for node in snapshot["nodes"]:
            previous = current_nodes.get(node["id"], {})
            if any(node.get(key) != previous.get(key) for key in ("state", "note", "detail")):
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
            self.bus.set_state(start_attempt="", start_note="Simulación lista")
            self.bus.log(f"Isaac Sim listo en el intento {self.attempt} de {self.max_attempts}.",
                         level="info")
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
            # A launch that hangs is the other half of the native fault: the child
            # is alive and says nothing. Kill it and let the retry decide.
            fault = fault_in_log(self.log_path) or "el arranque se quedó sin responder"
            self.bus.log(f"Se agotó el tiempo de arranque: {fault}.", level="warn")
            self.terminate()
            for channel in (self.inbox, self.updates, self.snapshots):
                with suppress(Exception):
                    channel.close()
            if not self.bus.stop_requested():
                self.child = None
                self.retry_or_give_up(fault)
                return
            self.child = None
        elif self.stop_at is not None and time.monotonic() - self.stop_at > self.args.stop_timeout:
            self.terminate()
        if not self.child.is_alive():
            self.child.join()
            # The worker never got as far as a camera and a scene: this attempt
            # died during start-up. That is the case Isaac faults in, and the only
            # one where relaunching is the right answer - a session that ran and
            # then died has already produced what it was going to produce.
            if not self.ready and not self.bus.stop_requested():
                application_error = worker_error_before_ready(self.bus.snapshot()["state"])
                if application_error:
                    for channel in (self.inbox, self.updates, self.snapshots):
                        with suppress(Exception):
                            channel.close()
                    self.child = None
                    self.bus.fail_active(application_error)
                    self.bus.set_state(phase="failed", sim="stopped", error=application_error,
                                       start_attempt="")
                    return
                reason = fault_in_log(self.log_path) or (
                    f"el proceso terminó con el código {self.child.exitcode}")
                for channel in (self.inbox, self.updates, self.snapshots):
                    channel.close()
                self.child = None
                self.retry_or_give_up(reason)
                return
            if self.report_path.exists():
                self.result = json.loads(self.report_path.read_text(encoding="utf-8"))
                self.merge(self.result)
                self.bus.set_state(report=self.result)
            error = self.bus.snapshot()["state"].get("error", "")
            if not self.result and not self.bus.stop_requested():
                error = error or f"Isaac terminó sin informe (código {self.child.exitcode}). Consulte el registro de la sesión."
            if not self.result:
                status = "ABORTED" if self.bus.stop_requested() else "FAILED"
                reason = error or "El entorno se detuvo antes de generar su informe."
                self.bus.set_state(status=status, termination_reason=reason)
                self.result = self.bus.snapshot()
                atomic_json(self.report_path, self.result)
                self.bus.set_state(report=self.result)
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
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.child.pid)],
                                   capture_output=True, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if self.child.is_alive():
                self.child.terminate()
            self.child.join(timeout=2)
            # A Windows process tree may survive a failed/partial taskkill. Never leave the
            # launcher waiting for that orphan to finish a long startup timeout on its own.
            if self.child.is_alive():
                kill = getattr(self.child, "kill", None)
                if callable(kill):
                    kill()
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
            control = bus.take_control(start=False)
            if control:
                if control["action"] == "start":
                    supervisor.start(control)
                else:
                    supervisor.forward(control["id"])
            if supervisor.ready:
                text = bus.take_command(start=False)
                if text:
                    supervisor.forward(bus.active_command_id())
            elif not supervisor.child:
                text = bus.take_command(start=False)
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
