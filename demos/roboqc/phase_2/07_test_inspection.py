#!/usr/bin/env python3
"""Inspection workflow in the terminal, without any robot.

A background thread keeps detecting and tracking while you type commands:

    > Inspecciona las piezas rojas.
    > ¿Cuántas piezas azules hay?
    > Ahora las piezas azules son defectuosas.
    > :report              (inventory and summary)
    > :save informe.json   (structured report)
    > :quit

Headless check: pass --command repeatedly with --frames to run without typing.
"""

import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from roboqc.commands.pipeline import understand_text
from roboqc.vision.calibration import load_calibration
from roboqc.vision.camera import CameraError, FpsMeter, open_source
from roboqc.vision.config import DEFAULT_CONFIG_PATH, load_config
from roboqc.vision.inspection import build_inspection_report, format_report
from roboqc.vision.pipeline import VisionPipeline
from roboqc.vision.resolver import resolve_command


class CaptureThread(threading.Thread):
    """Reads and processes frames until stopped. The world state is locked internally."""

    def __init__(self, src, pipeline: VisionPipeline, max_frames: int = 0):
        super().__init__(daemon=True)
        self.src = src
        self.pipeline = pipeline
        self.max_frames = max_frames
        self.meter = FpsMeter()
        self._stop_event = threading.Event()  # not _stop: that name is threading.Thread's own method

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self.max_frames and self.pipeline.frame_index >= self.max_frames:
                break
            t0 = time.perf_counter()
            frame = self.src.read()
            read_seconds = time.perf_counter() - t0
            if frame is None:
                break
            self.meter.tick(read_seconds)
            self.pipeline.process(frame)

    def stop(self) -> None:
        self._stop_event.set()


def _run_command(text: str, pipeline: VisionPipeline) -> None:
    cmd = understand_text(text, source="text")  # rule-based: no speech model required
    result = resolve_command(cmd, pipeline.world)
    print(f"[{result.status}]")
    print(result.text)
    print()


def _run_meta(text: str, pipeline: VisionPipeline) -> bool:
    """Handle :commands. Returns False when the session should end."""
    parts = text.split(maxsplit=1)
    name = parts[0]
    if name == ":quit":
        return False
    if name == ":report":
        print(format_report(build_inspection_report(pipeline.world.snapshot())))
        print()
    elif name == ":save" and len(parts) == 2:
        with open(parts[1], "w", encoding="utf-8") as f:
            json.dump(build_inspection_report(pipeline.world.snapshot()), f, indent=2, ensure_ascii=False)
        print(f"Informe guardado: {parts[1]}")
    else:
        print("Comandos: :report, :save RUTA.json, :quit")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Flujo de inspeccion en terminal (RoboQC, sin robot).")
    ap.add_argument("--source", default=None, help="Indice de camara, video o imagen.")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Fichero de configuracion JSON.")
    ap.add_argument("--calibration", default=None, help="JSON de calibracion opcional.")
    ap.add_argument("--command", action="append", default=[],
                    help="Ejecuta esta instruccion tras procesar frames y termina. Repetible.")
    ap.add_argument("--frames", type=int, default=30, help="Frames a procesar antes de --command (modo no interactivo).")
    ap.add_argument("--report-json", default=None, help="Guarda el informe final en este JSON (modo no interactivo).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg["camera"]["index"]
    pipeline = VisionPipeline(cfg, calibration=load_calibration(args.calibration or cfg.get("calibration_path")))
    try:
        src = open_source(source, cfg["camera"]["width"], cfg["camera"]["height"])
    except CameraError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    non_interactive = bool(args.command)
    with src:
        print(f"Fuente: {src.description}")
        thread = CaptureThread(src, pipeline, max_frames=args.frames if non_interactive else 0)
        thread.start()
        try:
            if non_interactive:
                thread.join()
                for text in args.command:
                    print(f"> {text}")
                    _run_command(text, pipeline)
                print(format_report(build_inspection_report(pipeline.world.snapshot())))
                if args.report_json:
                    with open(args.report_json, "w", encoding="utf-8") as f:
                        json.dump(build_inspection_report(pipeline.world.snapshot()), f, indent=2, ensure_ascii=False)
                    print(f"Informe guardado: {args.report_json}")
            else:
                print("Escribe una instruccion (:quit para salir, :report para el inventario).")
                while True:
                    line = input("> ").strip()
                    if not line:
                        continue
                    if line.startswith(":"):
                        if not _run_meta(line, pipeline):
                            break
                    else:
                        _run_command(line, pipeline)
        except (KeyboardInterrupt, EOFError):
            print()
        finally:
            thread.stop()
            thread.join(timeout=2.0)
    print(f"Frames procesados: {pipeline.frame_index}  FPS medio: {thread.meter.fps:.1f}")


if __name__ == "__main__":
    main()
