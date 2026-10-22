#!/usr/bin/env python3
"""Camera diagnostics for the RoboQC workspace: list, preview, measure, save, select ROI.

Keys in the preview window: q or Esc quit, s save a frame, r select the workspace ROI
(saved to the config file).
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from roboqc.vision.camera import CameraError, FpsMeter, list_cameras, open_source
from roboqc.vision.config import DEFAULT_CONFIG_PATH, load_config, save_roi

try:
    import cv2
except Exception:
    print("ERROR: OpenCV no esta instalado. Ejecuta: pip install opencv-python")
    sys.exit(1)


def _overlay_info(frame, meter, roi):
    if roi:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
    height, width = frame.shape[:2]
    text = f"{width}x{height}  FPS {meter.fps:5.1f}  latencia {meter.latency_ms:5.1f} ms"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run_preview(src, args, roi):
    meter = FpsMeter()
    window = "RoboQC camara (q salir, s guardar, r ROI)"
    while True:
        t0 = time.perf_counter()
        frame = src.read()
        read_seconds = time.perf_counter() - t0
        if frame is None:
            print("No llegan mas frames. Fin de la prueba.")
            break
        meter.tick(read_seconds)
        cv2.imshow(window, _overlay_info(frame.copy(), meter, roi))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = os.path.join(args.save_dir, f"frame_{int(time.time())}.png")
            os.makedirs(args.save_dir, exist_ok=True)
            cv2.imwrite(path, frame)
            print(f"Frame guardado: {path}")
        if key == ord("r"):
            x, y, w, h = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
            if w > 0 and h > 0:
                roi = [int(x), int(y), int(w), int(h)]
                save_roi(args.config, roi)
                print(f"ROI guardada en {args.config}: {roi}")
    cv2.destroyAllWindows()


def run_headless(src, args):
    meter = FpsMeter()
    frames = 0
    while frames < args.frames:
        t0 = time.perf_counter()
        frame = src.read()
        read_seconds = time.perf_counter() - t0
        if frame is None:
            break
        meter.tick(read_seconds)
        frames += 1
    print(f"Frames leidos: {frames}  FPS: {meter.fps:.1f}  latencia media: {meter.latency_ms:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostico de camara (RoboQC).")
    ap.add_argument("--list", action="store_true", help="Lista las camaras disponibles (indices 0-3).")
    ap.add_argument("--source", default=None,
                    help="Indice de camara, ruta de video o de imagen. Por defecto: camera.index de la configuracion.")
    ap.add_argument("--width", type=int, default=None, help="Ancho de captura (p. ej. 640, 1280).")
    ap.add_argument("--height", type=int, default=None, help="Alto de captura (p. ej. 480, 720).")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Fichero de configuracion JSON.")
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "samples"), help="Carpeta para las capturas.")
    ap.add_argument("--save-frame", default=None, metavar="RUTA.png", help="Guarda un frame y termina.")
    ap.add_argument("--frames", type=int, default=0,
                    help="Sin ventana: lee N frames y muestra FPS y latencia (para equipos sin pantalla).")
    args = ap.parse_args()

    if args.list:
        try:
            found = list_cameras()
        except CameraError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        print(f"Camaras disponibles: {found if found else 'ninguna'}")
        return

    cfg = load_config(args.config)
    cam = cfg["camera"]
    source = args.source if args.source is not None else cam["index"]
    width = args.width or cam["width"]
    height = args.height or cam["height"]
    try:
        src = open_source(source, width, height)
    except CameraError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    with src:
        print(f"Fuente: {src.description}")
        if args.save_frame:
            frame = src.read()
            if frame is None:
                print("ERROR: no se pudo leer ningun frame.")
                sys.exit(1)
            os.makedirs(os.path.dirname(os.path.abspath(args.save_frame)), exist_ok=True)
            cv2.imwrite(args.save_frame, frame)
            print(f"Frame guardado: {args.save_frame}")
        elif args.frames:
            run_headless(src, args)
        else:
            run_preview(src, args, cfg.get("roi"))


if __name__ == "__main__":
    main()
