"""Check a clock + cube policy against recorded episodes, without the simulator.

A policy whose only inputs are the plan clock and the cube's pixel position (see
``tools/21_build_clock_cube_dataset.py``) never reads the arm, so what it commands
over an episode does not depend on how the arm responds. That makes its rollout
reproducible offline: feed it the clock and the cube position the dataset holds,
let it chunk its actions exactly as demo 16 does, and compare the command stream
with the expert's. The simulator then only has to answer one question - does the
cube stay in the gripper - and this answers the other: *is the command right?*

Meant to be run on episodes the policy was **not** trained on
(``15_train_act.py --episodes 0-31`` leaves 32-39 unseen).

The number to read is the position error while the gripper closes. The grasp
tolerates roughly +-10 mm around the expert's grasp point (measured by
offsetting the expert, see ``results/diagnosis.md``); a command more than a
centimetre off there will not pick the cube up, however good the rest looks.

Run from the repository root:

    python .\\demos\\06_imitation_learning\\tools\\22_validate_offline.py ^
        --policy .\\demos\\06_imitation_learning\\outputs\\15_act_clock\\checkpoints\\last\\pretrained_model ^
        --dataset so101_lift_cube_clock_cube --episodes 32-39
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR))

import il_dataset  # noqa: E402

OFFLINE_DIR = DEMO_DIR / "results" / "offline"


def parse_episodes(text: str) -> list[int]:
    episodes: list[int] = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            episodes.extend(range(int(first), int(last) + 1))
        else:
            episodes.append(int(part))
    return episodes


def load_policy(path: Path, action_steps: int | None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(str(path))
    config.device = "cpu"
    if action_steps is not None:
        config.n_action_steps = min(action_steps, config.chunk_size)
    policy = ACTPolicy.from_pretrained(str(path), config=config).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post


def read_dataset(name: str):
    import pandas as pd
    import pyarrow.parquet as pq

    root = il_dataset.default_dataset_dir(name)
    frames = [pq.read_table(shard).to_pandas() for shard in sorted((root / "data").glob("*/*.parquet"))]
    table = pd.concat(frames, ignore_index=True)
    action = np.asarray(table["action"].tolist(), dtype=np.float32)
    env_state = np.asarray(table[il_dataset.ACT_ENV_STATE_KEY].tolist(), dtype=np.float32)
    return table["episode_index"].to_numpy(), table["frame_index"].to_numpy(), action, env_state


def rollout_commands(policy, pre, post, env_state_rows):
    """The command stream demo 16 would produce: one ``select_action`` per step."""
    import torch

    policy.reset()
    out = []
    with torch.no_grad():
        for row in env_state_rows:
            batch = {
                il_dataset.ACT_ENV_STATE_KEY: torch.as_tensor(row, dtype=torch.float32).unsqueeze(0),
                # Read for its device only (LeRobot 0.4.4's ACT); not a network input.
                il_dataset.STATE_KEY: torch.zeros(1, 6),
            }
            out.append(post(policy.select_action(pre(batch))).squeeze(0).numpy())
    return np.asarray(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="checkpoint directory (.../pretrained_model)")
    parser.add_argument("--dataset", default="so101_lift_cube_clock_cube")
    parser.add_argument("--episodes", default="32-39", help="episodes to check, e.g. 32-39")
    parser.add_argument("--action-steps", type=int, default=None)
    parser.add_argument("--report-name", default=None,
                        help="write results/offline/<name>.json (never overwritten)")
    args = parser.parse_args()

    policy_path = Path(args.policy).resolve()
    policy, pre, post = load_policy(policy_path, args.action_steps)
    width = int(policy.config.input_features[il_dataset.ACT_ENV_STATE_KEY].shape[0])
    kind = il_dataset.resolve_env_state_kind(policy_path, width)
    if kind != "clock_cube":
        print(f"this tool only replays clock + cube policies; this checkpoint is {kind!r}")
        return 1

    episode_ids, frame_ids, action, env_state = read_dataset(args.dataset)
    bounds = il_dataset.phase_bounds()
    names = [name for _, _, name in bounds]

    # Per-episode error, in millimetres, of the commanded position against the expert's.
    per_phase: dict[str, list[np.ndarray]] = {name: [] for name in names}
    grasp_xy, grasp_z, episodes_report = [], [], []
    for episode in parse_episodes(args.episodes):
        rows = np.flatnonzero(episode_ids == episode)
        if len(rows) == 0:
            print(f"episode {episode} is not in {args.dataset}")
            continue
        rows = rows[np.argsort(frame_ids[rows])]
        commands = rollout_commands(policy, pre, post, env_state[rows])
        error = (commands[:, :3] - action[rows][:, :3]) * 1000.0
        for first, last, name in bounds:
            per_phase[name].append(error[first:last])
        close_first, close_last = next((f, l) for f, l, n in bounds if n == "close")
        # The grasp happens while the jaw closes; the pose then is the one that matters.
        xy = float(np.linalg.norm(error[close_first:close_last, :2], axis=1).max())
        z = float(np.abs(error[close_first:close_last, 2]).max())
        grasp_xy.append(xy)
        grasp_z.append(z)
        jaw_ok = float(np.mean(np.sign(commands[:, 7]) == np.sign(action[rows][:, 7])))
        rms = float(np.sqrt((error ** 2).mean()))
        episodes_report.append({"episode": episode, "grasp_xy_error_mm": round(xy, 2),
                                "grasp_z_error_mm": round(z, 2), "jaw_command_agreement": round(jaw_ok, 4),
                                "whole_plan_xyz_rms_mm": round(rms, 2)})
        print(f"  episode {episode}: error while closing xy {xy:5.1f} mm, z {z:5.1f} mm; "
              f"jaw command agrees on {100 * jaw_ok:.1f}% of steps; whole plan rms {rms:.1f} mm")

    if not grasp_xy:
        return 1

    print("")
    print("phase          mean |dx| |dy| |dz| (mm)        worst xyz (mm)")
    for name in names:
        stack = np.abs(np.concatenate(per_phase[name]))
        mean, worst = stack.mean(0), stack.max(0)
        print(f"  {name:13s} {mean[0]:6.1f} {mean[1]:6.1f} {mean[2]:6.1f}          "
              f"{worst[0]:6.1f} {worst[1]:6.1f} {worst[2]:6.1f}")
    print("")
    print(f"error while the jaw closes, over {len(grasp_xy)} unseen episode(s):")
    print(f"  xy: mean {np.mean(grasp_xy):.1f} mm, worst {np.max(grasp_xy):.1f} mm   "
          f"z: mean {np.mean(grasp_z):.1f} mm, worst {np.max(grasp_z):.1f} mm")
    print("  the grasp tolerates about +-10 mm around the expert's grasp point")

    if args.report_name:
        OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
        path = OFFLINE_DIR / f"{args.report_name}.json"
        if path.exists():
            print(f"refusing to overwrite {path}")
            return 1
        path.write_text(json.dumps({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "policy": str(policy_path), "dataset": args.dataset, "episodes": args.episodes,
            "grasp_xy_error_mm": {"mean": float(np.mean(grasp_xy)), "worst": float(np.max(grasp_xy))},
            "grasp_z_error_mm": {"mean": float(np.mean(grasp_z)), "worst": float(np.max(grasp_z))},
            "per_episode": episodes_report,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
