"""Collect imitation-learning demonstrations: the scripted expert drives the SO-101.

This is step 1 of the playground. It runs the waypoint expert in the simulation,
converts every step into the LeRobot dataset format, and writes a real dataset
directory that the training demos can load.

Run:

    python .\\demos\\06_imitation_learning\\13_collect_demonstrations.py --episodes 40 --headless

What you get:

    demos/06_imitation_learning/data/so101_lift_cube_physical/
        meta/info.json        frames, fps, features
        data/chunk-000/...    one row per step: state, action, plan clock, cube position
        videos/...            the front camera, encoded
        13_collect_demonstrations_episodes.jsonl   one line per episode

What is different from the first recordings
-------------------------------------------

* **The cube is held by friction.** The expert used to pin the cube to the gripper
  while the jaw was closed, so a demonstration "succeeded" even where the fingers
  could not have held it. Here nothing is pinned (``--pin`` brings the old
  behaviour back), and an episode is kept only if the *physical* result passes
  ``il_metrics``: the cube was lifted, carried in the hand for a second, and put
  down on the drop point. A demonstration that did not really lift the cube is
  discarded instead of being taught.
* **The camera is fresh.** The first two frames of every earlier episode showed the
  previous episode's scene (``il_env.refresh_camera``).
* **The cube is located once.** ``il_env.perceive_cube`` finds the cube with the
  colour detector (averaged over a few renders) before the plan starts, and that
  pixel position is stored on every frame as ``observation.cube_uv``. The plan
  clock is stored as ``observation.plan_clock``.

``--overwrite`` never deletes anything: the existing dataset is renamed to
``<name>.replaced-<timestamp>`` next to it.
"""

from __future__ import annotations

import argparse
import sys
import time
from math import degrees as to_degrees
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_EPISODES = 5
DEFAULT_DATASET = "so101_lift_cube_physical"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="number of demonstrations to attempt")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"dataset directory name under data/ (default {DEFAULT_DATASET})")
    parser.add_argument("--first-spot", type=int, default=0,
                        help="index of the first cube position in the Halton sequence "
                             "(default 0; evaluation uses 1000 and up, so keep this below that)")
    parser.add_argument("--keep-failed", action="store_true",
                        help="also store episodes whose physical result failed")
    parser.add_argument("--pin", action="store_true",
                        help="hold the cube kinematically while the jaw is closed, as the first "
                             "recordings did (a simulator shortcut)")
    parser.add_argument("--headless", action="store_true", help="run without a window")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing dataset of this name. The old one is renamed "
                             "to <name>.replaced-<timestamp>, never deleted")
    args = parser.parse_args()

    import os

    import numpy as np

    import expert as expert_module
    import il_dataset as data
    import il_env
    import il_metrics

    dataset_root = data.default_dataset_dir(args.dataset)
    if dataset_root.exists():
        if not args.overwrite:
            print(f"ERROR: {dataset_root} already exists.")
            print("Pass --dataset <name> for a new one, or --overwrite to set it aside "
                  "(it is renamed, not deleted).")
            return 1
        parked = dataset_root.with_name(f"{dataset_root.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}")
        dataset_root.rename(parked)
        print(f"existing dataset kept as {parked}")

    if args.first_spot + args.episodes > il_env.EVAL_EPISODE_OFFSET:
        print(f"ERROR: spots {args.first_spot}..{args.first_spot + args.episodes - 1} would reach "
              f"the held-out evaluation spots, which start at {il_env.EVAL_EPISODE_OFFSET}.")
        return 1

    # The dataset is created before Kit starts: writing frames needs no simulator,
    # and starting it takes the better part of a minute.
    dataset = data.create_dataset(repo_id=args.dataset, root=dataset_root, with_perception=True)
    print(f"dataset: {dataset_root}")
    print(f"cube held by: {'a kinematic pin (shortcut)' if args.pin else 'friction only'}")

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
            spot_index = args.first_spot + episode
            expert = expert_module.Expert(env)
            env.reset()
            # Back to the canonical rest pose. Without this the arm carries the
            # previous episode's end pose into the next one and the dataset's
            # starting states drift with the episode index.
            il_env.reset_arm(env)
            spot = il_env.sample_cube_spot(spot_index, args.episodes)
            yaw = il_env.sample_cube_yaw(spot_index, args.episodes)
            cube_start = expert.bind(spot, yaw)

            # Where the detector sees the cube, before the plan moves anything.
            # This also leaves the camera showing the scene as it is now.
            perception = {}
            cube_uv = il_env.perceive_cube(env, info=perception)
            if cube_uv is None:
                print(f"  episode {episode}: the detector found no cube, skipped")
                failed += 1
                log.append({"episode": episode, "success": False, "skipped": "no_detection",
                            "cube_spot": [round(v, 4) for v in spot]})
                continue

            # The observation the plan starts from: the arm at rest, the cube on
            # its spot. It is read here, before any of the plan is applied, so the
            # first frame pairs it with the first action.
            observation = env.unwrapped.observation_manager.compute()
            judge = il_metrics.EpisodeJudge(target_xy=expert.target())

            step = 0
            episode_start = time.time()

            # Pairing: frame ``i`` holds observation ``i`` and the action chosen at
            # ``i``. The observation is read *before* the step, and the action
            # recorded is the one being applied - not the joint positions the IK
            # solver ended up commanding, which is a different space. Recording the
            # resulting joints is the mistake worth documenting: they lag the
            # command by a couple of steps, so the recorded action is nearly the
            # same number as the recorded state, and a policy trained on that
            # learns the identity and stands still.
            bounds = data.phase_bounds()
            while not expert.done:
                state, image = data.observation_from_env(observation)
                action = expert.action()
                dataset.add_frame(
                    {
                        data.STATE_KEY: state,
                        data.IMAGE_KEY: image,
                        data.PHASE_KEY: data.phase_vector(step, bounds),
                        data.PLAN_CLOCK_KEY: np.array([data.plan_clock(step)], dtype=np.float32),
                        data.CUBE_UV_KEY: np.array(cube_uv, dtype=np.float32),
                        data.ACTION_KEY: data.action_from_teacher(action),
                        "task": data.TASK_DESCRIPTION,
                    }
                )
                observation = il_env.step_obs(env, action)
                if args.pin:
                    expert.after_step()
                judge.update(expert.cube_position(), il_env.cube_hold_distance(env))
                step += 1

            verdict = judge.verdict()
            cube_end = expert.cube_position()
            moved = float(np.linalg.norm(np.array(cube_end[:2]) - np.array(cube_start[:2])))
            ok = verdict.success
            if ok or args.keep_failed:
                dataset.save_episode()
                kept += 1
            else:
                dataset.clear_episode_buffer()
                failed += 1

            log.append(
                {
                    "episode": episode,
                    "spot_index": spot_index,
                    "success": bool(ok),
                    "steps": step,
                    "cube_spot": [round(v, 4) for v in spot],
                    "cube_yaw_deg": round(to_degrees(yaw), 1),
                    "cube_start": [round(v, 4) for v in cube_start],
                    "cube_end": [round(v, 4) for v in cube_end],
                    "cube_uv": [round(v, 5) for v in cube_uv],
                    "cube_uv_spread_px": round(perception["spread_px"], 2),
                    "moved_m": round(moved, 4),
                    "placement_error_m": round(expert.placement_error(), 4),
                    "verdict": verdict.as_dict(),
                    "pinned": bool(args.pin),
                    "grasp_bias_m": list(expert_module.GRASP_BIAS_M),
                    "wall_clock_s": round(time.time() - episode_start, 1),
                }
            )
            print(
                f"  episode {episode}: {'ok' if ok else 'FAILED (' + verdict.failure + ')'} steps={step} "
                f"lift={verdict.max_lift_m * 100:.1f}cm carry={verdict.longest_carry_steps}steps "
                f"final_error={verdict.final_error_m * 100:.1f}cm "
                f"({time.time() - episode_start:.0f}s)",
                flush=True,
            )

    finally:
        dataset.finalize()
        data.write_episode_log(dataset_root / f"{Path(__file__).stem}_episodes.jsonl", log)

    print("")
    print(f"kept {kept} episode(s), discarded {failed}")
    print(f"frames: {dataset.num_frames}, episodes: {dataset.num_episodes}")
    print(f"total time: {time.time() - started:.0f}s")
    print(f"log: {dataset_root / (Path(__file__).stem + '_episodes.jsonl')}")
    sys.stdout.flush()
    # ``env.close()`` can hang for minutes with cameras (Kit shutdown); the dataset
    # is already finalised on disk, so leave without it.
    os._exit(0 if kept else 1)


if __name__ == "__main__":
    raise SystemExit(main())
