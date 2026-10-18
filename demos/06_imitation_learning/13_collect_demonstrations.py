"""Collect imitation-learning demonstrations: the scripted expert drives the SO-101.

This is step 1 of the playground. It runs the waypoint expert in the simulation,
converts every step into the LeRobot dataset format, and writes a real dataset
directory that the training demos can load.

Run:

    python .\\demos\\06_imitation_learning\\13_collect_demonstrations.py --episodes 5

What you get:

    demos/06_imitation_learning/data/so101_lift_cube_demos/
        meta/info.json        frames, fps, features
        data/chunk-000/...    one row per step: state, action
        videos/...            the front camera, encoded
        13_episodes.jsonl     one line per episode: what moved, and if it worked

Each line in ``13_episodes.jsonl`` records what actually happened, so a dataset
can be judged before it is trained on: how far the cube moved, whether it was
lifted, and how many steps the teacher needed.
"""

from __future__ import annotations

import argparse
import sys
import time
from math import degrees as to_degrees
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_EPISODES = 5
DEFAULT_DATASET = "so101_lift_cube_demos"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="number of demonstrations to record")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"dataset directory name under data/ (default {DEFAULT_DATASET})")
    parser.add_argument("--keep-failed", action="store_true",
                        help="also store episodes the teacher failed on")
    parser.add_argument("--headless", action="store_true", help="run without a window")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete the existing dataset directory first")
    args = parser.parse_args()

    import numpy as np
    import shutil

    import expert as expert_module
    import il_dataset as data
    import il_env

    dataset_root = data.default_dataset_dir(args.dataset)
    if dataset_root.exists():
        if not args.overwrite:
            print(f"ERROR: {dataset_root} already exists.")
            print("Pass --overwrite to replace it, or --dataset <name> for a new one.")
            return 1
        shutil.rmtree(dataset_root)

    # The dataset is created before Kit starts: writing frames needs no simulator,
    # and starting it takes the better part of a minute.
    dataset = data.create_dataset(repo_id=args.dataset, root=dataset_root)
    print(f"dataset: {dataset_root}")

    from il_env import prepare_runtime

    prepare_runtime(headless=args.headless)

    # The teacher's control space: the 8-number inverse-kinematics action.
    env = il_env.build_env(kind="ik_teacher", record=True, device=args.device)
    il_env.assert_joint_order(env)

    log: list[dict] = []
    kept = failed = 0
    started = time.time()

    try:
        for episode in range(args.episodes):
            expert = expert_module.Expert(env)
            env.reset()
            # Back to the canonical rest pose. Without this the arm carries the
            # previous episode's end pose into the next one and the dataset's
            # starting states drift with the episode index.
            il_env.reset_arm(env)
            spot = il_env.sample_cube_spot(episode, args.episodes)
            yaw = il_env.sample_cube_yaw(episode, args.episodes)
            cube_start = expert.bind(spot, yaw)
            # The observation the plan starts from: the arm at rest, the cube on
            # its spot. It is read here, before any of the plan is applied, so the
            # first frame pairs it with the first action.
            observation = env.unwrapped.observation_manager.compute()

            step = 0
            episode_start = time.time()

            # Pairing: frame ``i`` holds observation ``i`` and the action chosen at
            # ``i``. The observation is read *before* the step, and the action
            # recorded is the one being applied - not the joint positions the IK
            # solver ended up commanding, which is a different space.
            #
            # Recording the *resulting* joints is the mistake worth documenting:
            # they lag the command by a couple of steps, so the recorded action is
            # nearly the same number as the recorded state, and a policy trained
            # on that learns the identity and stands still.
            while not expert.done:
                state, image = data.observation_from_env(observation)
                action = expert.action()
                dataset.add_frame(
                    {
                        data.STATE_KEY: state,
                        data.IMAGE_KEY: image,
                        data.ACTION_KEY: data.action_from_teacher(action),
                        "task": data.TASK_DESCRIPTION,
                    }
                )
                observation = il_env.step_obs(env, action)
                expert.after_step()
                step += 1

            cube_end = expert.cube_position()
            moved = float(np.linalg.norm(np.array(cube_end[:2]) - np.array(cube_start[:2])))
            error = expert.placement_error()
            above_base = expert.height_above_base()
            ok = expert.succeeded()
            if ok or args.keep_failed:
                dataset.save_episode()
                kept += 1
            else:
                dataset.clear_episode_buffer()
                failed += 1

            log.append(
                {
                    "episode": episode,
                    "success": bool(ok),
                    "steps": step,
                    "cube_spot": [round(v, 4) for v in spot],
                    "cube_yaw_deg": round(to_degrees(yaw), 1),
                    "cube_start": [round(v, 4) for v in cube_start],
                    "cube_end": [round(v, 4) for v in cube_end],
                    "moved_m": round(moved, 4),
                    "placement_error_m": round(error, 4),
                    "height_above_base_m": round(above_base, 4),
                    "wall_clock_s": round(time.time() - episode_start, 1),
                }
            )
            print(
                f"  episode {episode}: {'ok' if ok else 'FAILED'} steps={step} "
                f"moved={moved * 100:.1f}cm final_error={error * 100:.1f}cm "
                f"({time.time() - episode_start:.0f}s)"
            )

    finally:
        dataset.finalize()
        data.write_episode_log(dataset_root / f"{Path(__file__).stem}_episodes.jsonl", log)
        env.close()

    print("")
    print(f"kept {kept} episode(s), discarded {failed}")
    print(f"frames: {dataset.num_frames}, episodes: {dataset.num_episodes}")
    print(f"total time: {time.time() - started:.0f}s")
    print(f"log: {dataset_root / (Path(__file__).stem + '_episodes.jsonl')}")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
