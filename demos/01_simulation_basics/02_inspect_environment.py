"""Create the official LeRobot/LeIsaac SO-101 task and inspect its Gym interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_loader import DEFAULT_ENV_SOURCE, load_environment  # noqa: E402


def describe(value: Any, indent: int = 0) -> None:
    """Print a compact, readable description of nested observations."""
    prefix = " " * indent
    if isinstance(value, dict):
        print(f"{prefix}dict with keys: {list(value)}")
        for key, item in value.items():
            print(f"{prefix}{key}:")
            describe(item, indent + 2)
    else:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"{prefix}{type(value).__name__}, shape={shape}, dtype={dtype}")


def main() -> int:
    """Reset the task and print observation and action information."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_ENV_SOURCE, help="LeRobot EnvHub source")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulator without a window (useful on remote desktops)",
    )
    args = parser.parse_args()

    try:
        env = load_environment(args.source, headless=args.headless)
    except Exception as error:
        print(f"ERROR: the SO-101 environment could not be created: {error}")
        return 1

    try:
        observation, info = env.reset()
        print(f"Environment: {args.source}")
        print("\nObservation structure:")
        describe(observation)
        print(f"\nReset info keys: {list(info) if isinstance(info, dict) else type(info).__name__}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        print(
            "\nThe action is the simulator control vector. Its exact meaning and bounds "
            "are intentionally read from action_space rather than assumed."
        )
        print(
            "The lift-cube task exposes robot/object observations; camera observations, "
            "if enabled by the task, appear as image-shaped arrays."
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
