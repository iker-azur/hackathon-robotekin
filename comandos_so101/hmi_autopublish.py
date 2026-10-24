"""Attach HMI telemetry to every LeRobot SOFollower observation in this process."""

from __future__ import annotations

import logging

from comandos_so101.hmi_telemetry import LatestTelemetryPublisher, observation_to_usd_radians


logger = logging.getLogger("hmi_telemetry")


def install(follower_class=None):
    """Wrap SOFollower observation reads without adding reads or changing actions."""
    if follower_class is None:
        from lerobot.robots.so_follower import SOFollower

        follower_class = SOFollower

    if getattr(follower_class, "_hmi_telemetry_installed", False):
        return

    original_get_observation = follower_class.get_observation
    original_disconnect = follower_class.disconnect

    def get_observation_with_telemetry(robot, *args, **kwargs):
        observation = original_get_observation(robot, *args, **kwargs)
        try:
            publisher = getattr(robot, "_hmi_telemetry_publisher", None)
            if publisher is None:
                publisher = LatestTelemetryPublisher()
                publisher.start()
                robot._hmi_telemetry_publisher = publisher
            publisher.offer(observation_to_usd_radians(
                observation, robot.calibration, robot.config.use_degrees
            ))
        except Exception:
            if not getattr(robot, "_hmi_telemetry_error_reported", False):
                logger.exception("No se pudo publicar la observación del follower en la HMI")
                robot._hmi_telemetry_error_reported = True
        return observation

    def disconnect_with_telemetry(robot, *args, **kwargs):
        try:
            return original_disconnect(robot, *args, **kwargs)
        finally:
            publisher = getattr(robot, "_hmi_telemetry_publisher", None)
            if publisher is not None:
                publisher.stop()
                del robot._hmi_telemetry_publisher

    follower_class.get_observation = get_observation_with_telemetry
    follower_class.disconnect = disconnect_with_telemetry
    follower_class._hmi_telemetry_installed = True
