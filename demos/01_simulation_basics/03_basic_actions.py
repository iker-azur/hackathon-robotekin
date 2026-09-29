"""Apply a few deterministic, small actions to the SO-101 simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_loader import DEFAULT_ENV_SOURCE, load_environment  # noqa: E402


def main() -> int:
    """Reset the task and apply zero and small bounded actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_ENV_SOURCE, help="LeRobot EnvHub source")
    parser.add_argument("--steps", type=int, default=5, help="Steps per action pattern")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulator without a window (useful on remote desktops)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as error:
        print(f"ERROR: PyTorch is required by the simulation: {error}")
        return 1

    try:
        env = load_environment(args.source, headless=args.headless)
    except Exception as error:
        print(f"ERROR: the SO-101 environment could not be created: {error}")
        return 1

    try:
        env.reset()
        action_space = env.action_space
        if not hasattr(action_space, "shape") or action_space.shape is None:
            raise RuntimeError(f"This example expects a Box-like action space, got {action_space}")

        # Isaac Lab is a Torch simulator: actions must be Torch tensors placed on the
        # simulation device, not NumPy arrays.
        device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cuda:0"
        low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)

        zero_action = torch.zeros(action_space.shape, dtype=torch.float32, device=device)
        small_action = torch.full(action_space.shape, 0.1, dtype=torch.float32, device=device)
        actions = (("zero / hold", zero_action), ("small positive command", small_action))

        for label, action in actions:
            action = torch.clamp(action, low, high)
            print(f"Applying {label}: {action.cpu().numpy()}")
            for step_number in range(args.steps):
                _, reward, terminated, truncated, info = env.step(action)
                print(
                    f"  step={step_number + 1} reward={reward} "
                    f"terminated={terminated} truncated={truncated} info={info}"
                )
                if bool(terminated.any()) or bool(truncated.any()):
                    print("Episode ended; resetting before the next action pattern.")
                    env.reset()
                    break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
