"""Run LeRobot teleoperation with SO-101 follower telemetry to the local HMI."""

from __future__ import annotations

import logging

from comandos_so101.hmi_telemetry import (
    HMI_TELEMETRY_URL,
    JOINTS,
    KINEMATICS_PATH,
    LatestTelemetryPublisher,
    MOTOR_RESOLUTION,
    PUBLISH_PERIOD_S,
    USD_LIMITS_RAD,
    observation_to_usd_radians,
)
from lerobot.scripts import lerobot_teleoperate


logger = logging.getLogger("hmi_telemetry")


def _teleop_loop_with_hmi_telemetry(*args, **kwargs):
    """Fallback for interpreters where the project venv hook is not installed."""
    robot = kwargs["robot"]
    if getattr(type(robot), "_hmi_telemetry_installed", False):
        return lerobot_teleoperate.teleop_loop_original(*args, **kwargs)

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
