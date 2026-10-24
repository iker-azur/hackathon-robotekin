"""LeRobot teleoperation entry point with a read-only HMI pose mirror.

This wraps the robot instance created by ``lerobot-teleoperate`` and reuses its
existing observation read. It never opens a second motor bus and never changes
the actions sent to the robot.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import queue
import sys
import threading
import time
from urllib.request import Request, urlopen

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_LIMITS_RAD = (-0.1745329917784602, 1.745329917784602)


def observation_payload(observation, invert_gripper: bool = False) -> dict | None:
    """Convert LeRobot's measured degree/% positions to the HMI USD radians."""
    try:
        joints = {name: math.radians(float(observation[f"{name}.pos"])) for name in ARM_JOINTS}
        gripper_percent = float(observation["gripper.pos"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (*joints.values(), gripper_percent)):
        return None
    gripper_percent = max(0.0, min(100.0, gripper_percent))
    if invert_gripper:
        gripper_percent = 100.0 - gripper_percent
    low, high = GRIPPER_LIMITS_RAD
    joints["gripper"] = low + (high - low) * gripper_percent / 100.0
    return {"joints": joints}


class LatestPosePublisher:
    """Send the newest pose at 20 Hz without blocking LeRobot's control loop."""

    def __init__(self, endpoint: str, invert_gripper: bool = False):
        self.endpoint = endpoint
        self.invert_gripper = invert_gripper
        self.pending: queue.Queue[dict] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="hmi-pose-publisher", daemon=True)
        self.thread.start()
        self._reported_error = False

    def observe(self, observation) -> None:
        payload = observation_payload(observation, self.invert_gripper)
        if payload is None:
            return
        try:
            self.pending.put_nowait(payload)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self.pending.put_nowait(payload)
            except queue.Full:
                pass

    def _run(self) -> None:
        next_send = 0.0
        latest = None
        while not self.stop_event.is_set():
            timeout = max(0.0, next_send - time.monotonic()) if latest else 0.05
            try:
                latest = self.pending.get(timeout=timeout)
            except queue.Empty:
                pass
            if latest is None or time.monotonic() < next_send:
                continue
            request = Request(
                self.endpoint,
                data=json.dumps(latest, separators=(",", ":")).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=0.5) as response:
                    response.read(64)
                self._reported_error = False
            except Exception as exc:  # telemetry loss must never interrupt robot control
                if not self._reported_error:
                    print(f"[HMI] No se pudo publicar la pose (solo lectura): {exc}", file=sys.stderr, flush=True)
                    self._reported_error = True
            latest = None
            next_send = time.monotonic() + 0.05

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=0.7)


def main() -> None:
    cli = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    cli.add_argument("--hmi-url", default="http://127.0.0.1:8790/api/telemetry")
    cli.add_argument("--hmi-gripper-invert", action="store_true",
                     help="invertir el porcentaje de apertura de la pinza en la vista")
    bridge_args, lerobot_args = cli.parse_known_args()
    sys.argv = [sys.argv[0], *lerobot_args]

    teleoperate = importlib.import_module("lerobot.scripts.lerobot_teleoperate")
    original_factory = teleoperate.make_robot_from_config

    def instrumented_factory(config):
        robot = original_factory(config)
        publisher = LatestPosePublisher(bridge_args.hmi_url, bridge_args.hmi_gripper_invert)
        read_observation = robot.get_observation
        disconnect = robot.disconnect

        def get_observation(*args, **kwargs):
            observation = read_observation(*args, **kwargs)
            publisher.observe(observation)
            return observation

        def close_robot(*args, **kwargs):
            try:
                return disconnect(*args, **kwargs)
            finally:
                publisher.close()

        robot.get_observation = get_observation
        robot.disconnect = close_robot
        return robot

    teleoperate.make_robot_from_config = instrumented_factory
    teleoperate.main()


if __name__ == "__main__":
    main()
