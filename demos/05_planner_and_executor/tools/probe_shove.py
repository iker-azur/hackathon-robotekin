"""Can a resting object be shoved? Needs Isaac.

A shove to a settled object reports a velocity of 11-24 cm/s that never decays over 45 steps,
while its position changes by 0.0-1.0 mm. A body cannot move at 24 cm/s for 0.75 s and go
nowhere, so the value being read back is the one that was written rather than the one the
solver has: PhysX has the objects asleep and ignoring the command.

This compares the ways of waking one, because the fix has to be measured, not guessed:

  write velocity         the current approach
  velocity after a nudge waking the body with a 1 mm pose write first
  pose teleport          writing the displaced pose, which place_items already does

    python tools/probe_shove.py --headless
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import dynamics  # noqa: E402
import executor as executor_module  # noqa: E402
import isaac_io  # noqa: E402
import scene_spec  # noqa: E402

SPEED = 0.20
SETTLE = 60


def position(env, prim):
    return env.scene[prim].data.root_pos_w[0].detach().cpu().numpy().copy()


def measure(env, prim, before):
    after = position(env, prim)
    return math.dist(before[:2], after[:2]) * 1000, after[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    items = [i for i in scene_spec.DEFAULT_SCENE if not i.container]
    isaac_io.prepare_runtime(headless=args.headless)
    env, sensor = isaac_io.build_env("top", list(scene_spec.DEFAULT_SCENE))
    try:
        env.reset()
        isaac_io.aim_camera(env, "top", sensor)
        bot = executor_module.Executor(env)
        bot.park()

        prims = [i.prim for i in items]
        print(f"shove {SPEED * 100:.0f} cm/s, settle {SETTLE} steps\n")
        print("| object | method | moved mm | speed read back cm/s |")
        print("|---|---|---|---|")

        heading = 0.0
        direction = (math.cos(heading), math.sin(heading), 0.0)

        for prim in prims:
            for method in ("write velocity", "velocity after nudge", "pose teleport"):
                isaac_io.place_items(env, items)
                obj = env.scene[prim]
                before = position(env, prim)
                velocity = torch.tensor([[direction[0] * SPEED, direction[1] * SPEED, 0.0]],
                                        device=env.device)
                zeros = torch.zeros_like(velocity)

                if method == "write velocity":
                    obj.write_root_velocity_to_sim(torch.cat([velocity, zeros], dim=-1))
                elif method == "velocity after nudge":
                    # 1 mm along the push, to wake the body before commanding it.
                    obj.write_root_pose_to_sim(
                        torch.cat([torch.tensor(before, device=env.device)[None, :]
                                    + torch.tensor(
                                        [[direction[0] * 0.001, direction[1] * 0.001, 0.0]],
                                        device=env.device),
                            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)], dim=-1))
                    env.sim.step(render=False)
                    obj.write_root_velocity_to_sim(torch.cat([velocity, zeros], dim=-1))
                else:
                    shifted = before.copy()
                    shifted[0] += direction[0] * 0.02
                    shifted[1] += direction[1] * 0.02
                    obj.write_root_pose_to_sim(
                        torch.cat([torch.tensor([shifted], device=env.device),
                                   torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)], dim=-1))
                    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

                for _ in range(SETTLE):
                    env.sim.step(render=False)
                env.scene.update(dt=env.physics_dt)

                moved, _z = measure(env, prim, before)
                speed = float(torch.linalg.norm(env.scene[prim].data.root_lin_vel_w[0])) * 100
                print(f"| {prim} | {method} | {moved:.1f} | {speed:.0f} |")
            print("")

        del dynamics
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())