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

What ACT does *not* do, stated because it is easy to assume otherwise: it sees
**one** observation per prediction. ``n_obs_steps`` is 1 and LeRobot 0.4.4 refuses
any larger value (``configuration_act.py:149``); the transformer attends over the
current image's feature map, the current joint state, the optional environment
state and the action chunk it is producing - never over past frames. Anything that
depends on "where the arm has been" has to be inside that one observation.

Which inputs the policy gets is decided by the dataset it is trained on, not by a
flag: whatever features the dataset declares (camera, joint state, environment
state) become the network's inputs. ``tools/21_build_clock_cube_dataset.py`` builds
a dataset with only an environment state, which is what the demo's working policy
uses.

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
# What it costs: the arm revisits the same pose in different phases of the plan
# (frames 299 and 300 match to 8e-6 degrees while the plan switches from ``lift``
# to ``carry``), so from joint angles alone those frames cannot be told apart. The
# demo's working policy sidesteps that by being given the plan clock instead of a
# history. It was *not* the cause of the camera + joint-state policies' failure:
# with the phase as input the ambiguity is gone and they still drifted, because
# they copy the arm's own pose (``results/diagnosis.md``, section 4).
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


def patch_episode_metadata_loader() -> None:
    """Avoid a LeRobot/Datasets hang on the episode stats columns.

    LeRobot 0.4.4 loads the whole episode parquet through
    ``datasets.Dataset.from_parquet`` and only then removes every ``stats/*``
    column. On this dataset that conversion hangs in Datasets' schema inference.
    The stats are also available in ``meta/stats.json`` and the standard loader
    discards the per-episode copies, so read the small parquet shards with
    PyArrow and create the HF Dataset from only the columns LeRobot keeps.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    def load_episodes(local_dir):
        episodes_dir = Path(local_dir) / "meta" / "episodes"
        paths = sorted(episodes_dir.glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"Provided directory does not contain any episode parquet file: {episodes_dir}"
            )

        tables = []
        for path in paths:
            parquet = pq.ParquetFile(path)
            columns = [name for name in parquet.schema.names if not name.startswith("stats/")]
            tables.append(parquet.read(columns=columns))

        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        return lerobot_dataset.datasets.Dataset(table)

    lerobot_dataset.load_episodes = load_episodes


def patch_frame_data_loader() -> None:
    """Load the frame parquet through Arrow instead of Datasets' slow cast.

    Datasets.from_parquet also stalls on the dataset's 25,600 row frame shard
    when LeRobot supplies a Features schema. The Arrow table is small and
    already has the correct fixed-size vectors; scalar one-element fields
    (timestamp and indices) need to be wrapped as
    length-one vectors, while the reserved environment-state field is kept as a
    vector for ACT's encoder.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    import il_dataset as data

    env_state_key = data.ACT_ENV_STATE_KEY
    original_load_info = lerobot_dataset.load_info
    original_load_stats = lerobot_dataset.load_stats

    def load_info_with_act_environment_state(root):
        info = original_load_info(root)
        features = info.get("features", {})
        if data.PHASE_KEY in features:
            features[env_state_key] = features.pop(data.PHASE_KEY)
        return info

    def load_stats_with_act_environment_state(root):
        stats = original_load_stats(root)
        if stats and data.PHASE_KEY in stats:
            stats[env_state_key] = stats.pop(data.PHASE_KEY)
        return stats

    lerobot_dataset.load_info = load_info_with_act_environment_state
    lerobot_dataset.load_stats = load_stats_with_act_environment_state

    def load_nested_dataset(pq_dir, features=None, episodes=None):
        pq_dir = Path(pq_dir)
        paths = sorted(pq_dir.glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"Provided directory does not contain any parquet file: {pq_dir}"
            )

        tables = []
        for path in paths:
            parquet = pq.ParquetFile(path)
            table = parquet.read()
            if data.PHASE_KEY in table.column_names:
                table = table.rename_columns([
                    env_state_key if name == data.PHASE_KEY else name
                    for name in table.column_names
                ])
            if episodes is not None and "episode_index" in table.column_names:
                mask = pc.is_in(
                    table["episode_index"], value_set=pa.array(episodes)
                )
                table = table.filter(mask)
            if features is not None:
                for name, feature in features.items():
                    if name not in table.column_names:
                        continue
                    length = getattr(feature, "length", None)
                    if name == env_state_key and length is None:
                        # LeRobot may store a one-element environment state as
                        # an Arrow scalar, while ACT expects a batched vector.
                        length = 1
                    column = table[name].combine_chunks()
                    if (
                        length is not None
                        and not pa.types.is_list(column.type)
                        and not pa.types.is_large_list(column.type)
                        and not pa.types.is_fixed_size_list(column.type)
                    ):
                        array = pa.FixedSizeListArray.from_arrays(
                            column, list_size=length
                        )
                        table = table.set_column(
                            table.column_names.index(name), name, array
                        )
            tables.append(table)

        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        return lerobot_dataset.datasets.Dataset(table)

    lerobot_dataset.load_nested_dataset = load_nested_dataset


def parse_episodes(text: str | None) -> list[int] | None:
    """``"0-31"`` or ``"0,1,5"`` -> a list of episode indices; ``None`` means all."""
    if not text:
        return None
    episodes: list[int] = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            episodes.extend(range(int(first), int(last) + 1))
        else:
            episodes.append(int(part))
    return episodes


def write_inputs_file(dataset_root: Path, output_dir: Path, dataset_name: str) -> None:
    """Record what the policy's environment-state input holds, next to the checkpoints.

    ACT gives that input a width and no name, so demo 16 would otherwise have to
    guess. The dataset's own ``meta/il_inputs.json`` is the source when it has one
    (tools/21 writes it); older datasets fall back on the width, which demo 16
    already understands.
    """
    import json

    import il_dataset as data

    source = dataset_root / "meta" / data.INPUTS_FILE
    if not source.is_file() or not output_dir.is_dir():
        return
    record = json.loads(source.read_text(encoding="utf-8"))
    record["trained_on"] = dataset_name
    text = json.dumps(record, indent=2) + "\n"
    (output_dir / data.INPUTS_FILE).write_text(text, encoding="utf-8")
    # And inside every checkpoint, so a ``pretrained_model`` folder that is copied or
    # zipped for another machine still says what its input means.
    for checkpoint in (output_dir / "checkpoints").glob("*/pretrained_model"):
        (checkpoint / data.INPUTS_FILE).write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="so101_lift_cube_clock_cube",
                        help="dataset under data/ (default: the clock + cube set built by "
                             "tools/21_build_clock_cube_dataset.py)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--action-steps", type=int, default=DEFAULT_ACTION_STEPS)
    parser.add_argument(
        "--obs-steps", type=int, default=DEFAULT_OBS_STEPS,
        help="how many observations the policy is shown per prediction. Only 1 is "
             "supported: LeRobot 0.4.4's ACTConfig raises for any larger value. "
             "Left here so the limit is visible - see the comment on DEFAULT_OBS_STEPS.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--job-name", default="act_so101_lift_cube")
    parser.add_argument("--output", default="15_act_clock",
                        help="checkpoint directory under outputs/ (an existing one is refused, "
                             "so a finished run is never overwritten)")
    parser.add_argument(
        "--no-vae",
        action="store_true",
        help="train ACT without its variational bottleneck. On a dataset this "
             "small the VAE pulls the prediction towards the average "
             "demonstration, and the arm ends up parked in a valid pose that "
             "is nobody's trajectory.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="learning rate (default: ACT's own 1e-5, tuned for a pretrained ResNet; a policy "
             "with no camera has nothing pretrained to protect and learns far faster at 1e-4)",
    )
    parser.add_argument("--backbone", choices=("resnet18", "resnet34"), default="resnet18")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.937,
                        help="AdamW beta1 (not SGD momentum)")
    parser.add_argument("--lr-final-fraction", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--dataset-fraction", type=float, default=1.0)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--save-freq", type=int, default=None,
                        help="checkpoint interval (default: max(1000, steps/10))")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-split", type=float, default=0.0,
                        help="percentage of selected episodes held out for LeRobot validation")
    parser.add_argument("--eval-freq", type=int, default=500,
                        help="training steps between offline validation runs")
    parser.add_argument("--pretrained-policy", default=None,
                        help="local pretrained ACT policy folder used for fine-tuning")
    parser.add_argument("--dim-model", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--amp", action="store_true", help="enable mixed precision when supported")
    parser.add_argument("--image-transforms-json", default=None,
                        help="ImageTransformsConfig JSON for camera augmentation during training")
    parser.add_argument("--split-mode", choices=("automatic", "manual"), default="automatic")
    parser.add_argument("--train-percent", type=int, default=100)
    parser.add_argument("--validation-percent", type=int, default=0)
    parser.add_argument("--test-percent", type=int, default=0)
    parser.add_argument("--validation-episodes", default=None)
    parser.add_argument("--test-episodes", default=None)
    parser.add_argument("--wandb", action="store_true", help="log the run with Weights & Biases")
    parser.add_argument("--wandb-project", default="robotekin-so101")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--seed", type=int, default=1000, help="training seed (default 1000)")
    parser.add_argument(
        "--episodes", default=None,
        help="train on these episodes only, e.g. 0-31 or 0,1,5; the rest stay unseen for "
             "tools/22_validate_offline.py (default: all)",
    )
    args = parser.parse_args()
    if not 0 <= args.eval_split < 100:
        parser.error("--eval-split must be between 0 and 100")
    if args.eval_freq <= 0:
        parser.error("--eval-freq must be greater than zero")
    if args.dim_model % args.attention_heads:
        parser.error("--dim-model must be divisible by --attention-heads")
    if not 0 < args.dataset_fraction <= 1:
        parser.error("--dataset-fraction must be greater than 0 and at most 1")
    if not 0 < args.lr_final_fraction <= 1:
        parser.error("--lr-final-fraction must be greater than 0 and at most 1")

    import il_dataset as data

    root = data.default_dataset_dir(args.dataset)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: {root} does not hold a dataset. Collect one with demo 13 first.")
        return 1
    dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    available_episodes = int(dataset_info.get("total_episodes", 0))
    episode_ids = parse_episodes(args.episodes) or list(range(available_episodes))
    if args.dataset_fraction < 1.0 and episode_ids:
        import random

        random.Random(args.seed).shuffle(episode_ids)
        keep = max(1, int(len(episode_ids) * args.dataset_fraction))
        episode_ids = sorted(episode_ids[:keep])
        print(f"dataset fraction: {len(episode_ids)}/{available_episodes} episodes selected")
    validation_episode_ids = parse_episodes(args.validation_episodes)
    test_episode_ids = parse_episodes(args.test_episodes)
    if args.split_mode == "automatic":
        if args.train_percent + args.validation_percent + args.test_percent != 100 or args.train_percent <= 0:
            parser.error("automatic train/validation/test percentages must total 100 and train must be positive")
        import random

        random.Random(args.seed).shuffle(episode_ids)
        train_count = max(1, int(len(episode_ids) * args.train_percent / 100))
        validation_count = int(len(episode_ids) * args.validation_percent / 100)
        # Reserve the test tail and let LeRobot hold out the validation fraction
        # from the remaining selected episodes.
        test_count = max(0, len(episode_ids) - train_count - validation_count)
        held_test = episode_ids[-test_count:] if test_count else []
        train_validation = episode_ids[:-test_count] if test_count else episode_ids
        validation_fraction = validation_count / max(1, len(train_validation))
        episode_ids = sorted(train_validation)
        validation_episode_ids = []
        test_episode_ids = held_test
    else:
        if not episode_ids:
            parser.error("manual split requires --episodes with at least one training episode")
        if validation_episode_ids is None:
            validation_episode_ids = []
        if test_episode_ids is None:
            test_episode_ids = []
        training_set = set(episode_ids)
        validation_set = set(validation_episode_ids)
        test_set = set(test_episode_ids)
        if training_set & validation_set or training_set & test_set or validation_set & test_set:
            parser.error("manual train, validation and test episode lists must not overlap")
        validation_fraction = 0.0
        print(f"manual episode split: train={len(episode_ids)}, validation={len(validation_episode_ids)}, test={len(test_episode_ids)}")

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
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    # LeRobot creates this itself, and refuses to run into a directory that
    # already exists unless ``resume`` is set - which protects a checkpoint from
    # being written over by a started-by-accident second run. Creating it here
    # instead would trip that check on the first run.
    output_dir = Path(__file__).resolve().parent / "outputs" / args.output

    # A dataset that says it holds a clock + cube environment state (tools/21) trains
    # a policy whose *only* input is that state. The dataset still carries the joint
    # state - LeRobot's ACT reads ``batch["observation.state"].device`` and crashes
    # without the key - but declaring the input features explicitly keeps it out of
    # the network, which is the point: with it the policy copies the arm's own pose.
    input_features = {}
    inputs_file = root / "meta" / data.INPUTS_FILE
    if inputs_file.is_file():
        import json

        kind = json.loads(inputs_file.read_text(encoding="utf-8")).get("env_state")
        if kind == "clock_cube":
            input_features = {
                data.ACT_ENV_STATE_KEY: PolicyFeature(
                    type=FeatureType.ENV, shape=(len(data.ENV_KINDS[kind]),)
                )
            }
            print(f"inputs:  environment state only ({', '.join(data.ENV_KINDS[kind])})")

    import inspect

    policy_options = {
        "input_features": input_features,
        "device": args.device,
        "chunk_size": args.chunk,
        "n_action_steps": args.action_steps,
        "n_obs_steps": args.obs_steps,
        "use_vae": not args.no_vae,
        "normalization_mapping": {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        "push_to_hub": False,
        "vision_backbone": args.backbone,
        "pretrained_backbone_weights": (
            None if args.no_pretrained_backbone
            else {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
            }[args.backbone]
        ),
        "optimizer_lr": args.lr,
        "optimizer_weight_decay": args.weight_decay,
        "optimizer_lr_backbone": args.backbone_lr,
        "dropout": args.dropout,
        "pretrained_path": Path(args.pretrained_policy) if args.pretrained_policy else None,
        "dim_model": args.dim_model,
        "n_encoder_layers": args.encoder_layers,
        "n_decoder_layers": args.decoder_layers,
        "n_heads": args.attention_heads,
        "latent_dim": args.latent_dim,
        "kl_weight": args.kl_weight,
        "use_amp": args.amp,
    }
    if args.pretrained_policy and "pretrained_path" not in inspect.signature(ACTConfig).parameters:
        raise RuntimeError("This LeRobot version does not support fine-tuning from a pretrained policy.")
    # LeRobot has moved config fields between releases. Keep this demo usable
    # with its documented 0.4.4 environment and newer compatible installs.
    policy_options = {
        key: value for key, value in policy_options.items()
        if key in inspect.signature(ACTConfig).parameters
        and (value is not None or key == "pretrained_backbone_weights")
    }
    policy = ACTConfig(
        **policy_options,
    )

    dataset_options = {
        "repo_id": args.dataset,
        "root": str(root),
        "episodes": episode_ids,
        "eval_split": validation_fraction,
    }
    if args.image_transforms_json:
        try:
            from lerobot.transforms import ImageTransformConfig, ImageTransformsConfig
        except ImportError as error:
            raise RuntimeError(
                "This LeRobot installation does not provide image transforms. "
                "Update LeRobot or disable training-time augmentation."
            ) from error
        transform_data = json.loads(args.image_transforms_json)
        transform_signature = inspect.signature(ImageTransformConfig).parameters
        transform_configs = {
            name: ImageTransformConfig(**{
                key: value for key, value in values.items() if key in transform_signature
            })
            for name, values in transform_data.get("tfs", {}).items()
        }
        transforms_signature = inspect.signature(ImageTransformsConfig).parameters
        transforms_options = {
            key: value for key, value in transform_data.items()
            if key in transforms_signature and key != "tfs"
        }
        if "tfs" in transforms_signature:
            transforms_options["tfs"] = transform_configs
        dataset_options["image_transforms"] = ImageTransformsConfig(**transforms_options)
    dataset_signature = inspect.signature(DatasetConfig).parameters
    if validation_fraction and "eval_split" not in dataset_signature:
        raise RuntimeError("This LeRobot version does not support a dataset evaluation split.")
    if args.image_transforms_json and "image_transforms" not in dataset_signature:
        raise RuntimeError("This LeRobot version does not support training-time image transforms.")
    dataset_options = {key: value for key, value in dataset_options.items() if key in dataset_signature}

    pipeline_options = {
        "num_workers": args.num_workers,
        "cudnn_deterministic": args.deterministic,
        "eval_steps": args.eval_freq if validation_fraction else 0,
    }
    pipeline_options = {
        key: value for key, value in pipeline_options.items()
        if key in inspect.signature(TrainPipelineConfig).parameters
    }
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(**dataset_options),
        policy=policy,
        output_dir=output_dir,
        job_name=args.job_name,
        seed=args.seed,
        batch_size=args.batch,
        steps=args.steps,
        # Checkpoints are LARGE: an ACT policy with a ResNet-18 backbone is about
        # 80 MB in safetensors. Saving every 2000 steps keeps the directory under
        # a few hundred megabytes instead of the default 20,000.
        save_freq=args.save_freq or max(1000, args.steps // 10),
        log_freq=args.log_freq,
        **pipeline_options,
    )
    if args.wandb:
        wandb_config = getattr(cfg, "wandb", None)
        if wandb_config is None or not hasattr(wandb_config, "enable"):
            raise RuntimeError("This LeRobot version does not expose W&B configuration.")
        wandb_config.enable = True
        if hasattr(wandb_config, "project"):
            wandb_config.project = args.wandb_project
        if hasattr(wandb_config, "entity"):
            wandb_config.entity = args.wandb_entity

    if args.momentum != 0.9:
        try:
            from lerobot.optim.optimizers import AdamWConfig
        except ImportError as error:
            raise RuntimeError("This LeRobot version cannot configure AdamW beta1.") from error
        train_fields = inspect.signature(TrainPipelineConfig).parameters
        if "optimizer" not in train_fields:
            raise RuntimeError("This LeRobot version does not expose an optimizer configuration.")
        cfg.optimizer = AdamWConfig(
            lr=args.lr or policy.optimizer_lr,
            weight_decay=args.weight_decay,
            betas=(args.momentum, 0.999),
        )
        if hasattr(cfg, "use_policy_training_preset"):
            cfg.use_policy_training_preset = False

    if args.cos_lr:
        try:
            from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        except ImportError as error:
            raise RuntimeError("This LeRobot version does not expose the cosine decay scheduler.") from error
        if not hasattr(cfg, "scheduler"):
            raise RuntimeError("This LeRobot version does not expose scheduler configuration.")
        warmup_steps = int(args.warmup_epochs * max(1, len(episode_ids)) * int(dataset_info.get("total_frames", 1))
                           / max(1, available_episodes) / args.batch)
        cfg.scheduler = CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=warmup_steps,
            num_decay_steps=args.steps,
            peak_lr=args.lr or policy.optimizer_lr,
            decay_lr=(args.lr or policy.optimizer_lr) * args.lr_final_fraction,
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
    patch_episode_metadata_loader()
    patch_frame_data_loader()
    train(cfg)
    write_inputs_file(root, output_dir, args.dataset)

    print("")
    print(f"trained in {time.time() - started:.0f}s")
    print(f"checkpoint: {output_dir / 'checkpoints' / 'last' / 'pretrained_model'}")
    print(f"evaluate it with demo 16: 16_evaluate_policy.py --policy {output_dir / 'checkpoints' / 'last' / 'pretrained_model'} --type act")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
