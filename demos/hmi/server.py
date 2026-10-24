"""Read-only SO-101 HMI: local assets, pose telemetry and camera frames.

Run: python demos/hmi/server.py --open (animated demo, no hardware)
An existing RecoveryKit instance can supply Isaac Sim poses and JPEGs with
--source URL. A process that already reads a physical robot can POST measured
link poses to /api/telemetry and JPEG bytes to /api/frame.jpg. No endpoint
sends commands to the robot.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from commands import make_plan
from speech import SpeechService, SpeechUnavailable
from comparison import demo_joints, comparison_sample, tcp_position, PERIOD
RECOVERY = ROOT.parent / "08_recovery_kit"
VENDOR = RECOVERY / "server" / "static" / "vendor"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
INSPECTION_CYCLE = 20.0


def inspection_joints(elapsed, sample_index, with_pin):
    """Illustrative pick, lift-to-camera, inspect and sort cycle (radians)."""
    sources = ((-.0594, -.0863, 1.1981, -.4390, 0.0, .7),
               (-.2952, .5730, .3915, -.4691, 0.0, .7),
               (-.7046, .2037, .8679, -.4541, 0.0, .7))
    source = sources[sample_index]
    inspect = (-.8292, -1.7411, 1.5708, -.1742, 0.0, .7)
    with_pin_bin = (.89223, .37366, .96018, -.88222, 0.0, .8)
    without_pin_bin = (.33269, 1.44709, -.99183, -.22136, -.17332, .8)
    destination = with_pin_bin if with_pin else without_pin_bin
    phase = elapsed % INSPECTION_CYCLE
    keys = ((0, source), (2.2, source[:5] + (.15,)),
            (4.5, inspect), (9.5, inspect), (12, destination),
            (14, destination[:5] + (1.35,)), (17, source), (INSPECTION_CYCLE, source))
    for (t0, q0), (t1, q1) in zip(keys, keys[1:]):
        if phase <= t1:
            u = max(0.0, min(1.0, (phase - t0) / (t1 - t0)))
            u = u * u * (3 - 2 * u)
            return dict(zip(JOINTS, (a + (b - a) * u for a, b in zip(q0, q1))))
    return dict(zip(JOINTS, source))


def inspection_stage(elapsed):
    phase = elapsed % INSPECTION_CYCLE
    if phase < 2.2:
        return "recogiendo", "Cesta roja"
    if phase < 4.5:
        return "elevando", "Elevando"
    if phase < 9.5:
        return "inspeccionando", "Cara plana"
    if phase < 12:
        return "clasificando", "Perno · ejemplo"
    if phase < 14:
        return "depositando", "A la caja"
    if phase < 17:
        return "soltando", "Depositando"
    return "siguiente", "Siguiente pieza"


def joint_poses(joints):
    """Convert measured USD-convention radians using the existing SO-101 chain."""
    if set(joints) != set(JOINTS):
        raise ValueError("joints must contain all six SO-101 joints, in radians")
    values = {key: float(value) for key, value in joints.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("joint values must be finite")
    if str(RECOVERY) not in sys.path:
        sys.path.insert(0, str(RECOVERY))
    from recoverykit.kinematics import fk, mat_to_quat
    return {name: [matrix[:3, 3].tolist(), mat_to_quat(matrix[:3, :3]).tolist()]
            for name, matrix in fk(values).items()}


def validate_poses(data):
    if not isinstance(data, dict):
        raise ValueError("telemetry must be an object")
    data = dict(data)
    if "joints" in data:
        data["links"] = joint_poses(data["joints"])
    links = data.get("links")
    if not isinstance(links, dict) or not links or "base" not in links:
        raise ValueError("links must include base and measured link poses")
    for name, pose in links.items():
        if not isinstance(name, str) or not isinstance(pose, (tuple, list)) or len(pose) != 2:
            raise ValueError("each link requires [position_xyz, quaternion_wxyz]")
        pos, quat = pose
        if len(pos) != 3 or len(quat) != 4:
            raise ValueError("pose requires 3 position values and 4 quaternion values")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [*pos, *quat]):
            raise ValueError("pose values must be finite numbers")
        if sum(v * v for v in quat) < 1e-12:
            raise ValueError("quaternion must be nonzero")
    return data


class Telemetry:
    def __init__(self, source):
        self.lock = threading.Lock()
        self.source = source or "Publicador local"
        self.poses = None
        self.pose_time = None
        self.jpeg = None
        self.frame_time = None
        self.frame_id = 0
        self.error = "Esperando fuente de datos"
        self.mode = "live"
        self.paused = False

    def publish(self, poses):
        poses = validate_poses(poses)
        with self.lock:
            self.poses = poses
            self.pose_time = time.monotonic()
            self.error = ""

    def frame(self, jpeg):
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("expected JPEG bytes")
        with self.lock:
            self.jpeg = jpeg
            self.frame_time = time.monotonic()
            self.frame_id += 1

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            age = None if self.pose_time is None else now - self.pose_time
            frame_age = None if self.frame_time is None else now - self.frame_time
            return {"source": self.source, "mode": self.mode, "paused": self.paused,
                    "connected": age is not None and age < 2,
                    "pose_age": age, "frame_age": frame_age,
                    "frame_id": self.frame_id, "poses": self.poses, "error": self.error}


def demo_worker(telemetry, stop):
    """Timed inspection route with pre-labelled samples; no physical robot or classifier."""
    telemetry.mode = "demo"
    telemetry.source = "SO-101 · Demo animada"
    elapsed = 0.0
    previous = time.monotonic()
    while not stop.is_set():
        now = time.monotonic()
        with telemetry.lock:
            paused = telemetry.paused
        if not paused:
            elapsed += now - previous
        previous = now
        sample_index = int(elapsed // INSPECTION_CYCLE) % 3
        sample_cycle = int(elapsed // INSPECTION_CYCLE)
        samples = (
            ("pieza_01", True, [.035, -.265, .042], [-.20, -.22, .042]),
            ("pieza_02", False, [.12, -.35, .042], [-.12, -.43, .042]),
            ("pieza_03", True, [.205, -.24, .042], [-.20, -.22, .042]),
        )
        sample_id, has_pin, source_pos, destination_pos = samples[sample_index]
        joints = inspection_joints(elapsed, sample_index, has_pin)
        reference = joint_poses(demo_joints(elapsed, JOINTS, follower=False))
        robot = joint_poses(joints)
        comparison = comparison_sample(elapsed, reference, robot)
        phase = elapsed % INSPECTION_CYCLE
        held = 2.2 <= phase < 14
        classified = phase >= 9.5
        pieces = {}
        for index, (name, pin, start, end) in enumerate(samples):
            cycle_for_piece = sample_cycle - ((sample_cycle - index) % 3)
            if cycle_for_piece < 0:
                cycle_for_piece = index
            done = elapsed >= (cycle_for_piece * INSPECTION_CYCLE + 14)
            is_active = name == sample_id
            pose = (tcp_position(robot) if is_active and held else end if is_active and classified
                    else end if done else start)
            gripper_pose = robot["gripper"]
            if is_active and held:
                w, x, y, z = gripper_pose[1]
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                quaternion = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            else:
                quaternion = [1, 0, 0, 0]
            pieces[name] = {"pos": pose, "quat": quaternion, "color": "gray", "shape": "casting",
                            "size": .052, "height": .012, "with_pin": pin, "visible": True}
        telemetry.publish({"joints": joints, "t": elapsed, "comparison": comparison, "pieces": pieces,
                           "workflow": {"stage": inspection_stage(elapsed)[0],
                                        "detail": inspection_stage(elapsed)[1],
                                        "piece": sample_id, "with_pin": has_pin,
                                        "result_is_example": True}})
        stop.wait(1 / 30)


def mirror(telemetry, source, stop, local_camera=False):
    """Poll read-only endpoints; each successful read refreshes even a stationary pose."""
    previous_frame_id = None
    next_camera = 0
    while not stop.is_set():
        try:
            with urlopen(source + "/api/poses", timeout=1) as response:
                poses = json.load(response)
            if poses:
                telemetry.publish(poses)
            if not local_camera and time.monotonic() >= next_camera:
                next_camera = time.monotonic() + .1
                with urlopen(source + "/api/state", timeout=1) as response:
                    state = json.load(response)
                frame_id = state.get("frame_id")
                if frame_id is not None and frame_id != previous_frame_id:
                    try:
                        with urlopen(source + "/api/frame.jpg", timeout=1) as response:
                            telemetry.frame(response.read())
                        previous_frame_id = frame_id
                    except HTTPError as exc:
                        if exc.code != 404:
                            raise
        except (OSError, URLError, ValueError, TypeError, KeyError) as exc:
            with telemetry.lock:
                telemetry.error = f"Fuente no disponible: {exc}"
        stop.wait(.04)


def camera_worker(telemetry, device, stop):
    import cv2
    capture = cv2.VideoCapture(int(device) if device.isdecimal() else device)
    try:
        if not capture.isOpened():
            print(f"No se puede abrir la cámara {device}", file=sys.stderr, flush=True)
            return
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not stop.is_set():
            ok, image = capture.read()
            if not ok:
                print(f"Se ha perdido la imagen de {device}", file=sys.stderr, flush=True)
                return
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                telemetry.frame(encoded.tobytes())
            stop.wait(1 / 30)
    finally:
        capture.release()


def make_handler(telemetry, speech=None):
    speech = speech or SpeechService()
    reference_path = None
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            nonlocal reference_path
            path = urlsplit(self.path).path
            if path == "/api/demo-reference":
                if reference_path is None:
                    reference_path = [tcp_position(joint_poses(demo_joints(i * PERIOD / 180, JOINTS, follower=False))) for i in range(181)]
                self.send(200, json.dumps({"kind": "illustrative", "period": PERIOD, "points": reference_path}).encode())
                return
            if path == "/api/state":
                self.send(200, json.dumps(telemetry.snapshot(), allow_nan=False).encode())
                return
            if path == "/api/voice/status":
                self.send(200, json.dumps(speech.status(), ensure_ascii=False).encode())
                return
            if path == "/api/frame.jpg":
                with telemetry.lock:
                    jpeg = telemetry.jpeg
                self.send(200 if jpeg else 404, jpeg or b'{"detail":"Sin imagen"}',
                          "image/jpeg" if jpeg else "application/json")
                return
            if path == "/api/reference":
                poses = {"links": joint_poses(dict.fromkeys(JOINTS, 0.0))}
                self.send(200, json.dumps(poses).encode())
                return
            assets = {"/": ROOT / "index.html", "/index.html": ROOT / "index.html",
                      "/app.js": ROOT / "app.js", "/style.css": ROOT / "style.css",
                      "/voice.js": ROOT / "voice.js",
                      "/scene-plan.js": ROOT / "scene-plan.js",
                      "/robot-model.js": ROOT / "robot-model.js",
                      "/buddy.js": ROOT / "buddy.js",
                      "/robot_meshes.json": RECOVERY / "server/static/robot_meshes.json"}
            if path.startswith("/vendor/") and path.removeprefix("/vendor/") in {
                "three.module.min.js", "three.core.min.js", "OrbitControls.js", "THREE_LICENSE.txt"
            }:
                assets[path] = VENDOR / path.removeprefix("/vendor/")
            file = assets.get(path)
            if file and file.is_file():
                kind = "text/javascript" if file.suffix == ".js" else mimetypes.guess_type(file)[0]
                self.send(200, file.read_bytes(), kind or "application/octet-stream")
            else:
                self.send(404, b'{"detail":"No encontrado"}')

        def do_POST(self):
            path = urlsplit(self.path).path
            if path == "/api/demo/toggle" and telemetry.mode == "demo":
                with telemetry.lock:
                    telemetry.paused = not telemetry.paused
                self.send(200, b'{"ok":true}')
                return
            if path not in ("/api/telemetry", "/api/frame.jpg", "/api/command", "/api/voice/transcribe"):
                self.send(404, b'{"detail":"No encontrado"}')
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8_000_000:
                    raise ValueError("invalid payload size")
                body = self.rfile.read(size)
                if path == "/api/command":
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        raise ValueError("La orden debe ser un objeto con texto.")
                    plan = make_plan(payload.get("text"), payload.get("source", "text"))
                    self.send(200, json.dumps(plan, ensure_ascii=False).encode())
                    return
                elif path == "/api/voice/transcribe":
                    result = speech.transcribe(body)
                    self.send(200, json.dumps(result, ensure_ascii=False).encode())
                    return
                elif path == "/api/telemetry":
                    telemetry.publish(json.loads(body))
                else:
                    telemetry.frame(body)
            except SpeechUnavailable as exc:
                self.send(503, json.dumps({"detail": str(exc)}, ensure_ascii=False).encode())
                return
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                self.send(400, json.dumps({"detail": str(exc)}).encode())
                return
            self.send(200, b'{"ok":true}')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="interfaz de escucha (usa 0.0.0.0 para aceptar telemetría de otro PC)")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--source", default="demo",
                        help="'demo', RecoveryKit URL, or 'publisher' for measured telemetry POSTs")
    parser.add_argument("--camera", help="Optional local camera index or /dev/videoN")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--whisper-model", default="tiny", help="Local Whisper model size or path")
    parser.add_argument("--prepare-voice", action="store_true", help="Download/load the voice model, then exit")
    args = parser.parse_args()
    speech = SpeechService(args.whisper_model)
    if args.prepare_voice:
        try:
            speech.load()
        except SpeechUnavailable as exc:
            parser.exit(1, str(exc) + "\n")
        print("Whisper local preparado.", flush=True)
        return
    source = None if args.source in ("publisher", "demo") else args.source.rstrip("/")
    if source and (urlsplit(source).scheme not in ("http", "https") or not urlsplit(source).netloc):
        parser.error("--source must be an http(s) URL or 'publisher'")
    telemetry = Telemetry(source)
    stop = threading.Event()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(telemetry, speech))
    if args.source == "demo":
        threading.Thread(target=demo_worker, args=(telemetry, stop), daemon=True).start()
    elif args.source == "publisher":
        telemetry.source = "SO-101 físico · LeRobot"
    if source:
        threading.Thread(target=mirror, args=(telemetry, source, stop, bool(args.camera)), daemon=True).start()
    if args.camera:
        threading.Thread(target=camera_worker, args=(telemetry, args.camera, stop), daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    address = f"0.0.0.0:{args.port}" if args.host == "0.0.0.0" else f"{args.host}:{args.port}"
    print(f"HMI: {url} · escucha: {address} · fuente: {telemetry.source}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
