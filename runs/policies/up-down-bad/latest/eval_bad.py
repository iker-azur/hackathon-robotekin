"""Offline held-out evaluation of an ACT checkpoint: joint error vs baselines and gripper open/close accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

GRIPPER_OPEN = 30.0
HORIZONS = (0, 10, 25, 50, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.device = args.device
    policy = ACTPolicy.from_pretrained(args.policy, config=cfg).to(args.device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    chunk = policy.config.chunk_size
    root = Path(args.dataset_root).resolve()
    episodes = [int(x) for x in args.episodes.split(",")]
    ds = LeRobotDataset(root.name, root=str(root), episodes=episodes, video_backend="pyav",
                        delta_timestamps={"action": [i / 30 for i in range(chunk)]})

    err = np.zeros((6, chunk)); base = np.zeros((6, chunk)); count = np.zeros(chunk)
    grip_pred, grip_true = [], []
    for i in range(0, len(ds), args.stride):
        item = ds[i]
        valid = ~item["action_is_pad"].numpy()
        batch = {k: v.unsqueeze(0) for k, v in item.items() if k.startswith("observation.")}
        with torch.no_grad():
            pred = post(policy.predict_action_chunk(pre(batch))).squeeze(0).float().cpu().numpy()
        truth, state = item["action"].numpy(), item["observation.state"].numpy()
        err[:, valid] += np.abs(pred - truth)[valid].T
        base[:, valid] += np.abs(state[None] - truth)[valid].T
        count[valid] += 1
        # gripper decision over the first 25 steps (what the robot executes before re-planning)
        grip_pred.extend(pred[:25][valid[:25], 5] > GRIPPER_OPEN)
        grip_true.extend(truth[:25][valid[:25], 5] > GRIPPER_OPEN)

    gp, gt = np.array(grip_pred), np.array(grip_true)
    tp, fp, fn = (gp & gt).sum(), (gp & ~gt).sum(), (~gp & gt).sum()
    pol, cp = err.sum() / (6 * count.sum()), base.sum() / (6 * count.sum())
    names = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
    rep = {
        "policy": args.policy, "frames": int(count[0]),
        "mae_policy": round(float(pol), 3), "mae_copy_state": round(float(cp), 3),
        "improvement_vs_copy_state": round(float(1 - pol / cp), 3),
        "mae_first_step": round(float(err[:, 0].mean() / count[0]), 3),
        "per_joint_mae": dict(zip(names, (err.sum(1) / count.sum()).round(2).tolist())),
        "mae_by_horizon": {h: round(float(err[:, h].mean() / count[h]), 2) for h in HORIZONS if h < chunk and count[h]},
        "gripper_accuracy": round(float((gp == gt).mean()), 3),
        "gripper_open_f1": round(float(2 * tp / max(1, 2 * tp + fp + fn)), 3),
    }
    print(json.dumps(rep, indent=1))
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
