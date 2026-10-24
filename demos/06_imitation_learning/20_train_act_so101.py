"""Train ACT on the real SO-101 recordings, on the local GPU, and check it on a held-out session.

Input is the dataset ``19_build_so101_act_dataset.py`` writes: front camera + wrist
camera + the six joint positions in, a chunk of future joint targets out, all at 30 fps
and 640 x 480. Training is LeRobot's own ``train()`` (the function ``lerobot-train``
runs), so no loss or optimiser is reimplemented here.

Which recordings to train on is the point of ``--regimes``. The data comes from two
camera configurations that frame the tray differently (wide 16:9 recorded at 10 fps,
tight 4:3 recorded at 30 fps); the 10 fps group was up-sampled to 30 fps by the build
script, which adds no real information. Pooling gives more episodes, but a policy trained
on both may fit today's setup worse than one trained on the 30 fps footage alone. Settle
it by measurement, not by argument: train twice, validate both on the same held-out
session, and keep the better one:

    python .\\demos\\06_imitation_learning\\20_train_act_so101.py --output act_all
    python .\\demos\\06_imitation_learning\\20_train_act_so101.py --regimes 30fps-480p --output act_30fps

What the validation number means. For every held-out frame, the policy is asked for a
chunk of future actions through the same path it uses on the robot (no peeking at the
answer) and the error is compared with a **hold-still baseline**: "stay at the current
joint positions". An earlier camera + joint-state ACT in the simulator looked fine and
then failed because it copied the arm's own pose; a policy that only copies the state
scores about the same as hold-still, so the figure to read is how far *below* hold-still
the policy gets. Errors are in joint units (about -100..100, gripper 0..100), averaged
over the chunk. This is an offline check; it says nothing about whether the arm
grasps a part. Normalisation statistics come from the whole dataset, held-out session
included, which only affects the scaling and is the same in both runs.

Run from the repository root, in the ``hackathon-robotekin`` environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "data" / "real_so101"
DATASET_REPO_ID = "local/so101_act"  # must match 19_build_so101_act_dataset.py

DEFAULT_STEPS = 20000
DEFAULT_BATCH = 8
# At 30 fps a chunk of 90 is 3 s: long enough for a reach + grasp, shorter than the
# 4-5 s up-down episodes. (ACT's default of 100 is 3.3 s at this rate.)
DEFAULT_CHUNK = 90
# Replan every second instead of every 3 s; this is only an inference setting and can
# be changed on the saved policy without retraining.
DEFAULT_ACTION_STEPS = 30


def load_sim_trainer():
    """The sim trainer (15_train_act.py) holds the Windows fixes for LeRobot's trainer."""
    spec = importlib.util.spec_from_file_location("train_act_sim", DEMO_DIR / "15_train_act.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_sources(root: Path) -> list[dict]:
    return json.loads((root / "meta" / "sources.json").read_text(encoding="utf-8"))


def select_episodes(sources: list[dict], regimes: list[str] | None, groups: list[str]) -> list[int]:
    return [
        s["episode_index"] for s in sources
        if not s["holdout"] and s["group"] in groups and (regimes is None or s["regime"] in regimes)
    ]


def validate(policy_dir: Path, root: Path, device: str, stride: int, batch_size: int = 16) -> dict:
    """Policy vs hold-still error on the held-out episodes, per group and per joint."""
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    sources = read_sources(root)
    held = [s["episode_index"] for s in sources if s["holdout"]]
    task_group = {s["task"]: s["group"] for s in sources}
    config = PreTrainedConfig.from_pretrained(str(policy_dir))
    config.device = device
    policy = ACTPolicy.from_pretrained(str(policy_dir), config=config).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(policy_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    delta = resolve_delta_timestamps(policy.config, LeRobotDatasetMetadata(DATASET_REPO_ID, root=root))
    dataset = LeRobotDataset(DATASET_REPO_ID, root=root, episodes=held, delta_timestamps=delta)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(0, len(dataset), stride)),
        batch_size=batch_size, num_workers=0,
    )

    joints = dataset.meta.features["action"]["names"]
    total = {}
    for batch in loader:
        truth, pad = batch["action"].clone(), batch["action_is_pad"].clone()
        state, tasks = batch["observation.state"].clone(), list(batch["task"])
        with torch.no_grad():
            pred = post(policy.predict_action_chunk(pre(batch))).cpu()
        valid = (~pad).unsqueeze(-1).float()
        errors = {"policy": (pred - truth).abs() * valid,
                  "hold_still": (state.unsqueeze(1) - truth).abs() * valid}
        for i, task in enumerate(tasks):
            group = task_group.get(task, "unknown")
            acc = total.setdefault(group, {"policy": np.zeros(len(joints)), "hold_still": np.zeros(len(joints)),
                                           "steps": 0.0, "frames": 0})
            for kind, err in errors.items():
                acc[kind] += err[i].sum(0).numpy()
            acc["steps"] += float(valid[i, :, 0].sum())
            acc["frames"] += 1

    report = {"policy": str(policy_dir), "stride": stride, "joints": joints, "groups": {}}
    for group, acc in sorted(total.items()):
        per_joint = {k: (acc[k] / acc["steps"]).tolist() for k in ("policy", "hold_still")}
        policy_err, still_err = float(np.mean(per_joint["policy"])), float(np.mean(per_joint["hold_still"]))
        report["groups"][group] = {"frames": acc["frames"], "policy": policy_err, "hold_still": still_err,
                                   "below_hold_still": 1 - policy_err / still_err, "per_joint": per_joint}
    overall = {k: float(sum(np.sum(a[k]) for a in total.values()) / sum(a["steps"] * len(joints) for a in total.values()))
               for k in ("policy", "hold_still")}
    report["overall"] = {**overall, "below_hold_still": 1 - overall["policy"] / overall["hold_still"]}
    return report


def print_report(report: dict) -> None:
    print("")
    print(f"held-out validation ({report['policy']}):")
    print("  group       frames   policy  hold-still  below hold-still")
    for group, r in report["groups"].items():
        print(f"  {group:10s} {r['frames']:7d} {r['policy']:8.2f} {r['hold_still']:11.2f} {100 * r['below_hold_still']:14.0f} %")
    o = report["overall"]
    print(f"  {'overall':10s} {'':7s} {o['policy']:8.2f} {o['hold_still']:11.2f} {100 * o['below_hold_still']:14.0f} %")
    print("  per joint, policy / hold-still:")
    for i, joint in enumerate(report["joints"]):
        pairs = ", ".join(f"{g} {r['per_joint']['policy'][i]:.1f}/{r['per_joint']['hold_still'][i]:.1f}"
                          for g, r in report["groups"].items())
        print(f"    {joint:15s} {pairs}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="act_30fps", help="dataset folder under data/real_so101")
    parser.add_argument("--regimes", nargs="+", default=None,
                        help="train only on these, e.g. 30fps-480p (default: every regime in the dataset)")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="train only on these groups of the dataset (default: every group in it)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS)
    parser.add_argument("--lr", type=float, default=None, help="learning rate (default: ACT's own 1e-5)")
    parser.add_argument("--no-augment", action="store_true",
                        help="turn off LeRobot's random colour / sharpness / affine image augmentation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default="act_all",
                        help="checkpoint folder under outputs/ (an existing one is refused)")
    parser.add_argument("--val-stride", type=int, default=3, help="validate on every Nth held-out frame")
    parser.add_argument("--validate-only", metavar="PRETRAINED_MODEL_DIR", default=None,
                        help="skip training and only validate this checkpoint")
    args = parser.parse_args()

    root = DATA_DIR / args.dataset
    if not (root / "meta" / "sources.json").exists():
        print(f"ERROR: {root} is not a dataset built by 19_build_so101_act_dataset.py")
        return 1

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to the CPU. This will be very slow.")
        args.device = "cpu"

    if args.validate_only:
        if not any(s["holdout"] for s in read_sources(root)):
            print("ERROR: the dataset has no held-out episodes, so there is nothing to validate on")
            return 1
        print_report(validate(Path(args.validate_only).resolve(), root, args.device, args.val_stride))
        return 0

    sources = read_sources(root)
    args.groups = args.groups or sorted({s["group"] for s in sources})
    episodes = select_episodes(sources, args.regimes, args.groups)
    if not episodes:
        print(f"ERROR: no episodes match --regimes {args.regimes} --groups {args.groups}; "
              f"the dataset has {sorted({s['regime'] for s in sources})}")
        return 1
    chosen = [s for s in sources if s["episode_index"] in set(episodes)]
    print(f"training on {len(chosen)} episodes ({sum(s['frames'] for s in chosen)} frames):")
    for regime in sorted({s["regime"] for s in chosen}):
        for group in args.groups:
            part = [s for s in chosen if s["regime"] == regime and s["group"] == group]
            print(f"  {regime:12s} {group:10s} {len(part):4d} episodes, {sum(s['frames'] for s in part):6d} frames")
    held = [s for s in sources if s["holdout"]]
    print(f"held out for validation: {len(held)} episodes from {sorted({s['source_repo'] for s in held})}")

    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.transforms import ImageTransformsConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    output_dir = DEMO_DIR / "outputs" / args.output
    policy = ACTConfig(
        device=args.device, chunk_size=args.chunk, n_action_steps=args.action_steps, push_to_hub=False,
        **({"optimizer_lr": args.lr} if args.lr is not None else {}),
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_REPO_ID, root=str(root), episodes=episodes,
                              image_transforms=ImageTransformsConfig(enable=not args.no_augment)),
        policy=policy, output_dir=output_dir, job_name=f"act_so101_{args.output}", seed=args.seed,
        batch_size=args.batch, steps=args.steps, num_workers=args.num_workers,
        save_freq=max(1000, args.steps // 5), log_freq=100,
    )

    from lerobot.scripts.lerobot_train import train

    load_sim_trainer().patch_last_checkpoint()
    started = time.time()
    train(cfg)
    print(f"trained in {time.time() - started:.0f} s")

    last = output_dir / "checkpoints" / "last" / "pretrained_model"
    (output_dir / "run.json").write_text(json.dumps({
        "regimes": args.regimes, "groups": args.groups, "episodes": len(episodes), "steps": args.steps,
        "batch": args.batch, "chunk": args.chunk, "augment": not args.no_augment,
    }, indent=2) + "\n", encoding="utf-8")
    torch.cuda.empty_cache() if args.device == "cuda" else None
    if not held:
        print("no held-out episodes in this dataset, so no validation was run")
        print(f"checkpoint: {last}")
        return 0
    report = validate(last, root, args.device, args.val_stride)
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"\ncheckpoint: {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
