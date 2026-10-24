"""Publish calibrated SO-101 follower positions to the local HMI."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen


HMI_TELEMETRY_URL = "http://127.0.0.1:8790/api/telemetry"
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
KINEMATICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "demos/08_recovery_kit/assets/so101/kinematics.json"
)
with KINEMATICS_PATH.open(encoding="utf-8") as kinematics_file:
    USD_LIMITS_RAD = {
        joint["name"]: (joint["lower"], joint["upper"])
        for joint in json.load(kinematics_file)["joints"]
    }
MOTOR_RESOLUTION = 4095
PUBLISH_PERIOD_S = 1 / 20
# Add 180° to the previous 90° correction; the equivalent wrist offset is -90°.
USD_ZERO_OFFSETS_RAD = {"wrist_roll": -1.5707963267948966}

logger = logging.getLogger("hmi_telemetry")


def observation_to_usd_radians(observation, calibration, use_degrees):
    """Convert calibrated LeRobot positions into the HMI's SO-101 USD radians."""
    joints = {}
    for name in JOINTS:
        value = float(observation[f"{name}.pos"])
        range_min = calibration[name].range_min
        range_max = calibration[name].range_max
        usd_min_rad, usd_max_rad = USD_LIMITS_RAD[name]

        if name == "gripper":
            # SOFollower normalizes the gripper to RANGE_0_100, independent of use_degrees.
            fraction = value / 100.0
            if calibration[name].drive_mode:
                fraction = 1.0 - fraction
        elif use_degrees:
            # DEGREES is (encoder - midpoint) * 360 / (resolution - 1).
            encoder_position = value * MOTOR_RESOLUTION / 360 + (range_min + range_max) / 2
            fraction = (encoder_position - range_min) / (range_max - range_min)
        else:
            # RANGE_M100_100 maps calibrated minimum/maximum to -100/+100.
            fraction = (value + 100.0) / 200.0
            if calibration[name].drive_mode:
                fraction = 1.0 - fraction

        joints[name] = (
            usd_min_rad + fraction * (usd_max_rad - usd_min_rad)
            + USD_ZERO_OFFSETS_RAD.get(name, 0.0)
        )

    return {"joints": joints}


class LatestTelemetryPublisher:
    """POST only the latest sample, at no more than 20 Hz, on a daemon thread."""

    def __init__(self):
        self._condition = threading.Condition()
        self._latest = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="hmi-telemetry", daemon=True)

    def start(self):
        self._thread.start()

    def offer(self, payload):
        with self._condition:
            self._latest = payload
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def _run(self):
        last_post = 0.0
        last_warning = 0.0
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._latest is not None or self._stopping)
                if self._stopping:
                    return
                delay = PUBLISH_PERIOD_S - (time.monotonic() - last_post)
                while delay > 0:
                    self._condition.wait(timeout=delay)
                    if self._stopping:
                        return
                    delay = PUBLISH_PERIOD_S - (time.monotonic() - last_post)
                payload = self._latest
                self._latest = None

            request = Request(
                HMI_TELEMETRY_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=1.0) as response:
                    response.read()
            except Exception as exc:
                now = time.monotonic()
                if now - last_warning >= 5.0:
                    logger.warning("No se pudo publicar telemetría en la HMI: %s", exc)
                    last_warning = now
            last_post = time.monotonic()
