"""Run LeRobot teleoperation and stream measured follower joints to the HMI."""

from __future__ import annotations

import json
import logging
import threading
import time
from math import pi
from urllib.request import Request, urlopen

from lerobot.scripts import lerobot_teleoperate


HMI_TELEMETRY_URL = "http://10.186.203.205:8790/api/telemetry"
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
USD_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}
MOTOR_RESOLUTION = 4095
PUBLISH_PERIOD_S = 1 / 20

logger = logging.getLogger("hmi_telemetry")


def observation_to_usd_radians(observation, calibration, use_degrees):
    """Convert calibrated LeRobot positions into the HMI's SO-101 USD radians."""
    joints = {}
    for name in JOINTS:
        value = float(observation[f"{name}.pos"])
        range_min = calibration[name].range_min
        range_max = calibration[name].range_max
        usd_min_deg, usd_max_deg = USD_LIMITS_DEG[name]

        if name == "gripper":
            # SOFollower always uses RANGE_0_100 for the gripper.
            fraction = value / 100.0
        elif use_degrees:
            # DEGREES is (encoder - midpoint) * 360 / (resolution - 1).
            encoder_position = value * MOTOR_RESOLUTION / 360 + (range_min + range_max) / 2
            fraction = (encoder_position - range_min) / (range_max - range_min)
        else:
            # RANGE_M100_100 maps calibrated minimum/maximum to -100/+100.
            fraction = (value + 100.0) / 200.0

        usd_degrees = usd_min_deg + fraction * (usd_max_deg - usd_min_deg)
        joints[name] = usd_degrees * pi / 180

    return {"joints": joints}


class LatestTelemetryPublisher:
    """POST the latest observation at no more than 20 Hz on a daemon thread."""

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


def _teleop_loop_with_hmi_telemetry(*args, **kwargs):
    robot = kwargs["robot"]
    publisher = LatestTelemetryPublisher()
    original_get_observation = robot.get_observation
    use_degrees = robot.config.use_degrees
    calibration = robot.calibration

    def get_observation_with_telemetry():
        observation = original_get_observation()
        try:
            publisher.offer(observation_to_usd_radians(observation, calibration, use_degrees))
        except Exception:
            logger.exception("No se pudo convertir la observación para la telemetría HMI")
        return observation

    robot.get_observation = get_observation_with_telemetry
    publisher.start()
    try:
        return lerobot_teleoperate.teleop_loop_original(*args, **kwargs)
    finally:
        robot.get_observation = original_get_observation
        publisher.stop()


def main():
    lerobot_teleoperate.teleop_loop_original = lerobot_teleoperate.teleop_loop
    lerobot_teleoperate.teleop_loop = _teleop_loop_with_hmi_telemetry
    return lerobot_teleoperate.main()


if __name__ == "__main__":
    main()
