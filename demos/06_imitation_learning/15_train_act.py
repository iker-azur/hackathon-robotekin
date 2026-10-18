"""Train an ACT policy with LeRobot's own training pipeline.

This is step 3 of the playground. ACT (Action Chunking with Transformers) is
the imitation-learning policy LeRobot ships and recommends first. The demo trains
it on the dataset demo 13 produced, using ``lerobot.scripts.lerobot_train`` - the
same function the ``lerobot-train`` command runs - so nothing here reimplements a
loss or an optimiser.

What ACT adds over Behavior Cloning
----------------------------------

Instead of predicting the next action, ACT predicts a chunk of ``chunk_size``
actions at once and replays a few of them before predicting again
(``n_action_steps``). That repairs the classic failure of plain cloning: a policy
that is slightly wrong at step one drifts, sees a state it never saw in the
demonstrations, and is then wrong by more. With a chunk, a small mistake is
absorbed into a trajectory that was planned ahead.

The transformer also attends to the last few observations and the whole action
chunk, so the gripper's open/close decision is made in the context of the arm's
motion rather than by a single frame.

Run:

    python .\\demos\\06_imitation_learning\\15_train_act.py --steps 20000

Outputs a checkpoint under ``outputs/15_act/`` that demo 16 loads directly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_STEPS = 20000
DEFAULT_BATCH = 8

# ACT defaults from LeRobot 0.4.4, except chunk size. The shipped default of
# 100 actions per chunk is 1.7 seconds at 60 fps - longer than the whole grasp
# takes - so 20 gets a chunk that spans the descent and the close, without
# spending the budget predicting steps far past the episode.
DEFAULT_CHUNK = 20
DEFAULT_ACTION_STEPS = 20

# How many past observations ACT is shown.
#
# Kept at 1, and that is a limitation of LeRobot 0.4.4 rather than a choice:
# ``ACTConfig.__post_init__`` raises ``ValueError("Multiple observation steps not
# handled yet")`` for any ``n_obs_steps > 1`` (configuration_act.py:149).
#
# It matters, because the task as recorded is not a function of a single
# observation. The arm revisits the *same* configuration in different phases:
# frames 299 and 300 match to 8e-6 degrees while the plan switches from ``lift``
# to ``carry``, and frames 391 and 420 match to 0 degrees while it switches from
# ``carry`` to ``put down``. Counting every pair of frames in an episode sharing
# a pose within 1 degree but demanding targets up to 22 mm apart gives **449
# ambiguous pairs per episode**. Given six joint angles and asked for a target,
# the best any single-observation policy can do on those pairs is their average -
# a target that goes nowhere. That is the remaining known blocker, and the fix
# is either a stack that does accept a history, or removing the revisits from the
# expert's plan. See ``results/diagnosis.md``.
DEFAULT_OBS_STEPS = 1

# With a 480x640 camera and a 16 GB card shared with the simulator, batch 8 is
# the largest that holds comfortably. LeRobot's own SO-101 tutorial uses 8 too.
BATCH_WARNING_THRESHOLD = 8


def patch_last_checkpoint() -> None:
    """Make LeRobot's "last checkpoint" link work on Windows without admin rights.

    LeRobot 0.4.4 points ``checkpoints/last`` at the newest step with a symlink,
    which needs the ``SeCreateSymbolicLinkPrivilege`` privilege - an
    administrator/symlink permission this machine does not grant. The training
    itself is unaffected; only this bookkeeping fails.

    The fix replaces the function with one that copies the directory instead. A
    copy costs an extra 80 MB per save, which is why the demo's default
    ``save_freq`` is coarse, and it removes the need to raise the process's
    privileges or enable Developer Mode.
    """
    import shutil

    import lerobot.utils.train_utils as train_utils
    from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK

    def update_last_checkpoint(checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        link_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
        if link_dir.is_symlink() or link_dir.is_file():
            link_dir.unlink()
        elif link_dir.is_dir():
            shutil.rmtree(link_dir)
        shutil.copytree(checkpoint_dir, link_dir)
        return link_dir

    # ``lerobot_train`` imports the function by name, so patching the module
    # alone is not enough: the caller's own reference has to be replaced too.
    import lerobot.scripts.lerobot_train as lerobot_train

    train_utils.update_last_checkpoint = update_last_checkpoint
    lerobot_train.update_last_checkpoint = update_last_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="so101_lift_cube_demos")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS)
    parser.add_argument(
        "--obs-steps", type=int, default=DEFAULT_OBS_STEPS,
        help="how many past observations the policy is shown. Only 1 is "
             "supported: LeRobot 0.4.4's ACTConfig raises for any larger value. "
             "Left here because the recorded task needs a history to be "
             "well-posed - see the comment on DEFAULT_OBS_STEPS.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--job-name", default="act_so101_lift_cube")
    parser.add_argument("--output", default="15_act", help="checkpoint directory under outputs/")
    parser.add_argument(
        "--no-vae",
        action="store_true",
        help="train ACT without its variational bottleneck. On a dataset this "
             "small the VAE pulls the prediction towards the average "
             "demonstration, and the arm ends up parked in a valid pose that "
             "is nobody's trajectory.",
    )
    args = parser.parse_args()

    import il_dataset as data

    root = data.default_dataset_dir(args.dataset)
    if not (root / "meta" / "info.json").exists():
        print(f"ERROR: {root} does not hold a dataset. Collect one with demo 13 first.")
        return 1

    if args.batch > BATCH_WARNING_THRESHOLD:
        print(f"note: batch {args.batch} is larger than {BATCH_WARNING_THRESHOLD}, "
              "the size this 16 GB card was checked with. If it runs out of memory, "
              f"lower it to {BATCH_WARNING_THRESHOLD}.")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to the CPU. This will be very slow.")
        args.device = "cpu"

    # LeRobot's own configuration and training entry points. Importing them here
    # rather than at the top keeps ``--help`` fast and import-clean on a machine
    # where only the dataset tools are wanted.
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    # LeRobot creates this itself, and refuses to run into a directory that
    # already exists unless ``resume`` is set - which protects a checkpoint from
    # being written over by a started-by-accident second run. Creating it here
    # instead would trip that check on the first run.
    output_dir = Path(__file__).resolve().parent / "outputs" / args.output

    policy = ACTConfig(
        device=args.device,
        chunk_size=args.chunk,
        n_action_steps=args.action_steps,
        n_obs_steps=args.obs_steps,
        use_vae=not args.no_vae,
        # Dropout is what acts as regularisation on a dataset this small; 0.1 is
        # ACT's default and is left alone.
        push_to_hub=False,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.dataset, root=str(root)),
        policy=policy,
        output_dir=output_dir,
        job_name=args.job_name,
        batch_size=args.batch,
        steps=args.steps,
        # Checkpoints are LARGE: an ACT policy with a ResNet-18 backbone is about
        # 80 MB in safetensors. Saving every 2000 steps keeps the directory under
        # a few hundred megabytes instead of the default 20,000.
        save_freq=max(1000, args.steps // 10),
        log_freq=100,
    )

    print("dataset:", root)
    print(f"policy:  ACT, chunk {args.chunk}, {args.action_steps} actions per step, "
          f"{args.obs_steps} observation(s), batch {args.batch}")
    print(f"device:  {args.device}")
    print(f"steps:   {args.steps}")
    print("")

    started = time.time()
    from lerobot.scripts.lerobot_train import train

    patch_last_checkpoint()
    train(cfg)

    print("")
    print(f"trained in {time.time() - started:.0f}s")
    print(f"checkpoint: {output_dir / 'checkpoints' / 'last' / 'pretrained_model'}")
    print(f"evaluate it with demo 16: 16_evaluate_policy.py --policy {output_dir / 'checkpoints' / 'last' / 'pretrained_model'} --type act")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
