#!/usr/bin/env python3
"""Real-time detection, tracking and quality classification on webcam, video or image.

Each object is printed as a structured record (object_id, type, color, position_px,
bounding_box, quality, quality_reasons, detection_status, ...). World coordinates
appear only when a calibration file is given.

Keys in the window: q or Esc quit.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from roboqc.commands.pipeline import understand_text
from roboqc.vision.calibration import load_calibration
from roboqc.vision.camera import CameraError, FpsMeter, open_source
from roboqc.vision.config import DEFAULT_CONFIG_PATH, load_config
from roboqc.vision.inspection import build_inspection_report
from roboqc.vision.overlay import draw_overlay, draw_text_lines
from roboqc.vision.pipeline import VisionPipeline
from roboqc.vision.resolver import resolve_command

try:
    import cv2
except Exception:
    print("ERROR: OpenCV no esta instalado. Ejecuta: pip install opencv-python")
    sys.exit(1)


def _print_frame(result):
    print(f"--- frame {result.frame_index}  {result.processing_ms:.1f} ms  "
          f"entran={result.events.entered} salen={result.events.disappeared}")
    for obj in result.snapshot.objects:
        if obj.in_view:
            print(json.dumps(obj.to_dict(), ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Deteccion y clasificacion en tiempo real (RoboQC).")
    ap.add_argument("--source", default=None, help="Indice de camara, video o imagen (por defecto, la camara de la configuracion).")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Fichero de configuracion JSON.")
    ap.add_argument("--calibration", default=None, help="JSON de calibracion (opcional) para coordenadas de mundo.")
    ap.add_argument("--command", default=None,
                    help="Instruccion aplicada antes de empezar, p. ej. 'Ahora las piezas azules son defectuosas'.")
    ap.add_argument("--frames", type=int, default=0, help="Procesa N frames y termina (0 = hasta pulsar q).")
    ap.add_argument("--no-window", action="store_true", help="Sin ventana: solo imprime los registros estructurados.")
    ap.add_argument("--json-out", default=None, help="Guarda el ultimo estado del mundo en este JSON.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = args.source if args.source is not None else cfg["camera"]["index"]
    calibration = load_calibration(args.calibration or cfg.get("calibration_path"))
    pipeline = VisionPipeline(cfg, calibration=calibration)

    if args.command:
        # Phase 1 rule-based understanding; no ML model is needed for this script.
        cmd = understand_text(args.command, source="text")
        result = resolve_command(cmd, pipeline.world)
        print(f"Comando: {args.command}\nEstado: {result.status}\n{result.text}\n")

    try:
        src = open_source(source, cfg["camera"]["width"], cfg["camera"]["height"])
    except CameraError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    meter = FpsMeter()
    window = "RoboQC deteccion (q salir)"
    last = None
    with src:
        print(f"Fuente: {src.description}")
        while args.frames == 0 or pipeline.frame_index < args.frames:
            t0 = time.perf_counter()
            frame = src.read()
            read_seconds = time.perf_counter() - t0
            if frame is None:
                print("No llegan mas frames.")
                break
            meter.tick(read_seconds)
            result = pipeline.process(frame)
            last = result
            if args.no_window:
                _print_frame(result)
                continue
            view = draw_overlay(frame, result.snapshot.objects, roi=cfg.get("roi"))
            counts = result.snapshot.counts()
            view = draw_text_lines(view, [
                f"FPS {meter.fps:.1f}  proceso {result.processing_ms:.1f} ms",
                f"visibles {counts['visible']}  buenas {counts['GOOD']}  "
                f"defectuosas {counts['DEFECTIVE']}  inciertas {counts['UNCERTAIN']}",
            ])
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    if not args.no_window:
        cv2.destroyAllWindows()

    if last is not None and args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(build_inspection_report(last.snapshot), f, indent=2, ensure_ascii=False)
        print(f"Estado guardado: {args.json_out}")
    print(f"Frames procesados: {pipeline.frame_index}  FPS medio: {meter.fps:.1f}")


if __name__ == "__main__":
    main()
