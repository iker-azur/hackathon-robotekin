"""Ubuntu SO-101 teleoperation recorder. Run --help before connecting hardware."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import select
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from recorder import (
    IMAGE_KEYS,
    JOINTS,
    Config,
    Session,
    dataset_lock,
    positions,
    require_version,
    sample,
    validate,
)

LOG = logging.getLogger(__name__)


def device_checks(cfg):
    devices = (cfg.follower_port, cfg.leader_port, cfg.wrist_camera, cfg.overhead_camera)
    for device in devices:
        if "REEMPLAZAR" in str(device):
            raise ValueError("Sustituye todos los dispositivos de ejemplo en la configuración")
        if isinstance(device, str) and device.startswith("/dev/") and not Path(device).exists():
            raise FileNotFoundError(device)
    if Path(cfg.follower_port).resolve() == Path(cfg.leader_port).resolve():
        raise ValueError("Leader y follower deben usar puertos distintos")
    wrist, overhead = cfg.wrist_camera, cfg.overhead_camera
    if str(wrist) == str(overhead) or (
        isinstance(wrist, str) and isinstance(overhead, str)
        and Path(wrist).resolve() == Path(overhead).resolve()
    ):
        raise ValueError("Las dos vistas deben usar cámaras distintas")


@contextlib.contextmanager
def hardware(cfg):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    device_checks(cfg)
    cameras = {
        name: OpenCVCameraConfig(index_or_path=device, width=cfg.width, height=cfg.height,
                                 fps=cfg.fps, color_mode="rgb")
        for name, device in (("wrist", cfg.wrist_camera), ("overhead", cfg.overhead_camera))
    }
    follower = SO101Follower(SO101FollowerConfig(
        port=cfg.follower_port, id=cfg.follower_id, use_degrees=True, cameras=cameras,
        calibration_dir=Path(cfg.follower_calibration_dir) if cfg.follower_calibration_dir else None,
        max_relative_target=cfg.max_relative_target,
    ))
    leader = SO101Leader(SO101LeaderConfig(
        port=cfg.leader_port, id=cfg.leader_id, use_degrees=True,
        calibration_dir=Path(cfg.leader_calibration_dir) if cfg.leader_calibration_dir else None,
    ))
    provenance = {}
    for name, device in (("follower", follower), ("leader", leader)):
        if not device.calibration:
            raise RuntimeError(f"Falta calibración {name}: {device.calibration_fpath}. "
                               "Ejecuta primero lerobot-calibrate con este id")
        provenance[name] = {key: asdict(value) for key, value in device.calibration.items()}
    try:
        # Never trigger interactive calibration as a side effect of recording.
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        if not leader.is_calibrated or not follower.is_calibrated:
            raise RuntimeError("La calibración local no coincide con los motores; calibra antes de grabar")
        yield follower, leader, provenance
    finally:
        # is_connected can be false after a partial camera connection. Close the bus anyway.
        for device in (follower, leader):
            for camera in getattr(device, "cameras", {}).values():
                with contextlib.suppress(Exception):
                    camera.disconnect()
            if device.bus.is_connected:
                with contextlib.suppress(Exception):
                    device.bus.disconnect()


class Keys:
    """Terminal-local keys work on Wayland and SSH as well as X11."""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("record necesita una terminal interactiva (TTY)")
        if os.name != "nt":
            import termios
            import tty
            self.original = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def poll(self):
        if os.name == "nt":
            import msvcrt
            return msvcrt.getwch().lower() if msvcrt.kbhit() else ""
        return sys.stdin.read(1).lower() if select.select([sys.stdin], [], [], 0)[0] else ""

    def wait(self, choices):
        while True:
            key = self.poll()
            if key and key in choices:
                return key
            time.sleep(0.03)

    def __exit__(self, *_):
        if os.name != "nt":
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original)


class Preview:
    def __init__(self, enabled):
        self.rr = None
        if enabled:
            import rerun as rr
            rr.init("so101_recording", spawn=True)
            self.rr = rr

    def show(self, frame, episode):
        if self.rr is None:
            return
        rr = self.rr
        rr.set_time("episode_time", duration=np.asarray(frame["capture_time_s"]).item())
        prefix = f"episodes/{episode}"
        for name in IMAGE_KEYS:
            rr.log(f"{prefix}/{name}", rr.Image(frame[name]))
        for name in ("observation.state", "action", "leader_state"):
            for joint, value in zip(JOINTS, frame[name], strict=True):
                rr.log(f"{prefix}/{name}/{joint}", rr.Scalars(float(value)))


def capture(cfg, session, follower, leader, keys, preview):
    """Return (choice, report). A failed sensor read never becomes an accepted frame."""
    session.begin()
    start = time.perf_counter()
    ticks = []
    period = 1 / cfg.fps
    choice = "e"
    while time.perf_counter() - start < cfg.episode_time_s:
        key = keys.poll()
        if key in ("e", "d", "q"):
            choice = key
            break
        tick = time.perf_counter()
        frame = sample(cfg, follower, leader, tick - start)
        session.add(frame)
        preview.show(frame, session.count)
        ticks.append(tick - start)
        time.sleep(max(0, period - (time.perf_counter() - tick)))
    intervals = np.diff(ticks)
    report = {
        "frames": len(ticks), "target_fps": cfg.fps,
        "measured_fps": float(1 / intervals.mean()) if len(intervals) else 0,
        "max_interval_s": float(intervals.max()) if len(intervals) else 0,
        "slow_intervals": int((intervals > 1.5 * period).sum()),
    }
    return choice, report


def record(cfg, resume):
    with dataset_lock(cfg):
        session = Session(cfg, resume=resume)
        with hardware(cfg) as (follower, leader, provenance), Keys() as keys:
            session.provenance = provenance
            existing = Path(cfg.root) / "recording.json"
            if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["calibrations"] != provenance:
                raise ValueError("Las calibraciones cambiaron; usa otro dataset")
            preview = Preview(cfg.display_data)
            try:
                if session.count > session.state["uploaded_episodes"]:
                    print("Hay episodios pendientes; reintentando subida antes de continuar.", flush=True)
                    session.sync()
                while True:
                    print(f"\nLocales: {session.count}; última subida: {session.state['uploaded_episodes']}. "
                          "PAUSA: p = habilitar teleoperación, q = salir. "
                          "El follower limita el salto por articulación.", flush=True)
                    if keys.wait("pq") == "q":
                        break
                    print("PREPARACIÓN (sin grabar): s = grabar, q = salir.", flush=True)
                    while True:
                        key = keys.poll()
                        if key in ("s", "q"):
                            break
                        tick = time.perf_counter()
                        frame = sample(cfg, follower, leader, 0)
                        preview.show(frame, "preparation")
                        time.sleep(max(0, 1 / cfg.fps - (time.perf_counter() - tick)))
                    if key == "q":
                        break
                    print("GRABANDO: e = terminar/revisar, d = descartar, q = salir.", flush=True)
                    choice, report = capture(cfg, session, follower, leader, keys, preview)
                    print(json.dumps(report, indent=2), flush=True)
                    if choice == "q":
                        session.discard()
                        break
                    if choice == "d" or report["frames"] == 0:
                        session.discard()
                        continue
                    if report["measured_fps"] < 0.9 * cfg.fps:
                        print("Cadencia inferior al 90% del objetivo; revisa antes de aceptar.", flush=True)
                    print("REVISIÓN: inspecciona cámaras y articulaciones en Rerun. "
                          "a = aceptar y subir públicamente, d = descartar, q = salir.", flush=True)
                    choice = keys.wait("adq")
                    if choice != "a":
                        session.discard()
                        if choice == "q":
                            break
                        continue
                    session.accept(report)
                    print("Guardado y cerrado. Subiendo a Hugging Face...", flush=True)
                    if not session.sync():
                        print("Subida pendiente. Puedes continuar; upload permite reintentar.", flush=True)
            finally:
                if session.dataset is not None:
                    if session.state["phase"] == "recording":
                        session.discard()
                    else:
                        # A failed save must remain marked for inspection, never silently repaired.
                        with contextlib.suppress(Exception):
                            session.dataset.finalize()


def check(cfg):
    import av
    from huggingface_hub import HfApi
    from PIL import Image

    cfg.local.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=cfg.local) as handle:
        handle.write(b"write check")
    # Actually encode one frame, rather than only checking an encoder name.
    with av.open(io.BytesIO(), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=cfg.fps)
        stream.width, stream.height, stream.pix_fmt = cfg.width, cfg.height, "yuv420p"
        image = av.VideoFrame.from_ndarray(np.zeros((cfg.height, cfg.width, 3), np.uint8), format="rgb24")
        for packet in stream.encode(image):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    user = HfApi().whoami()
    print(f"Hub autenticado como {user['name']}; destino público: {cfg.repo_id}")
    print("check no crea repositorios ni envía consignas de movimiento.")
    with hardware(cfg) as (follower, leader, _):
        times = []
        for _ in range(cfg.fps * 2):
            start = time.perf_counter()
            obs = follower.get_observation()
            positions(obs)
            positions(leader.get_action())
            for name in ("wrist", "overhead"):
                frame = obs[name]
                if frame.shape != (cfg.height, cfg.width, 3) or frame.dtype != np.uint8:
                    raise ValueError(f"Formato de cámara incorrecto: {name}")
            times.append(time.perf_counter() - start)
            time.sleep(max(0, 1 / cfg.fps - times[-1]))
        for name in ("wrist", "overhead"):
            Image.fromarray(obs[name]).save(cfg.local / f"check-{name}.png")
        print(f"Lectura conjunta media: {1000 * np.mean(times):.1f} ms. "
              f"Revisa las imágenes en {cfg.local}")
    print("Diagnóstico completado. La cadencia con movimiento se mide durante record.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "record", "validate", "upload"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Continuar un dataset local de este grabador")
    parser.add_argument("--root", type=Path, help="Solo validate: validar una copia descargada del Hub")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.resume and args.command != "record":
        parser.error("--resume solo se usa con record")
    if args.root and args.command != "validate":
        parser.error("--root solo se usa con validate")
    try:
        cfg = Config.load(args.config)
        require_version()
        if args.command == "check":
            with dataset_lock(cfg):
                check(cfg)
        elif args.command == "record":
            record(cfg, args.resume)
        elif args.command == "validate":
            with dataset_lock(cfg):
                print(json.dumps(validate(cfg, args.root), indent=2))
        else:
            with dataset_lock(cfg):
                session = Session(cfg, resume=True)
                if not session.sync():
                    return 1
                print(f"Sincronizados {session.count} episodios: https://huggingface.co/datasets/{cfg.repo_id}")
    except KeyboardInterrupt:
        LOG.info("Interrumpido; el episodio sin aceptar se descarta si el cierre es normal")
        return 130
    except Exception:
        LOG.exception("Operación fallida")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
