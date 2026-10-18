"""Train a Behavior Cloning policy: a small network that copies the expert.

This is step 2 of the playground, and the simplest one. Behavior Cloning is
supervised regression on demonstrations:

    network(state, image) -> action

Nothing else. No rewards, no trying things out, no search. The policy is trained
to reproduce the expert's action given the expert's observation. Everything that
goes wrong later - compounding errors, poor generalisation - comes from that one
line, which is exactly what this demo is meant to make visible.

What the network is
-------------------

``BCPolicy`` is a two-branch MLP: a small convolutional trunk for the image, a
dense trunk for the joint state, and a merged head for the action. It is written
out here in full, on purpose. ACT and Diffusion Policy are far better policies,
but they hide the fact that imitation learning is just regression. Seeing the
whole thing in fifty lines is the point.

Actions are normalised with the dataset's own statistics, so the six joints have
comparable scale before the loss sees them.

Run:

    python .\\demos\\06_imitation_learning\\14_train_behavior_cloning.py --steps 4000

Outputs checkpoints under ``outputs/14_bc/`` and a ``results/14_bc.csv`` summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn

DEFAULT_STEPS = 4000
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
HIDDEN = 256


class BCPolicy(nn.Module):
    """The Behavior Cloning network, in full.

    Kept deliberately plain: two trunks into a merged head. The point of this
    class is that it is small enough to read, not that it is the best possible
    architecture.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN):
        super().__init__()
        # The image trunk: three convolutions down from 480x640 to a 60x80 feature
        # map, then pooled. The kernel sizes are chosen so each block halves the
        # spatial size, which is the usual habit rather than a tuned result.
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        # The state trunk: the strongest signal in this dataset, because the arm's
        # current pose already explains most of where it has to go next.
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, image, state):
        # The dataset hands frames back in one of two layouts depending on the
        # video backend installed: (B, C, H, W) from the decoder, or (B, 1, H, W, C)
        # when a frame has been decoded as an image array. Both are put into
        # (B, C, H, W) float in zero to one here, so the whole "what the network
        # sees" lives in one place.
        if image.dim() == 5:  # (B, 1, H, W, C)
            image = image[:, 0]
            image = image.permute(0, 3, 1, 2)
        elif image.shape[1] in (1, 3):
            pass  # already (B, C, H, W)
        else:
            image = image.permute(0, 3, 1, 2)
        image = image.float() / 255.0
        features = torch.cat([self.image(image), self.state(state)], dim=-1)
        return self.head(features)


def normalise(values, mean, scale):
    """Put actions on a common scale, as ``(value - mean) / std``.

    The shoulder travels over tens of degrees and the gripper over two; without
    this the shoulder would dominate the loss and the gripper would never be
    learned. Done in plain brackets rather than through LeRobot's processors so
    the arithmetic is visible.

    A dimension that never moves has a standard deviation of zero, and dividing
    by it turns every prediction for it into NaN - which then poisons the whole
    loss, not just that dimension. The teacher's action contains several fixed
    numbers (the gripper orientation, which never changes), so the guard is not
    defensive: it is needed on this dataset.
    """
    values = torch.as_tensor(values, dtype=torch.float32)
    mean = torch.as_tensor(mean, dtype=torch.float32)
    scale = torch.as_tensor(scale, dtype=torch.float32)
    safe = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    return (values - mean) / safe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="so101_lift_cube_demos",
                        help="dataset directory name under data/")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="14_bc", help="checkpoint directory under outputs/")
    args = parser.parse_args()

    import numpy as np
    import torch

    import il_dataset as data

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to the CPU. This will be slow.")
        args.device = "cpu"

    root = data.default_dataset_dir(args.dataset)
    dataset = data.load_dataset(args.dataset, root=root)
    if len(dataset) == 0:
        print(f"ERROR: {root} is empty. Collect demonstrations with demo 13 first.")
        return 1

    output_dir = Path(__file__).resolve().parent / "outputs" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = BCPolicy(state_dim=data.env_cfg.STATE_DIM, action_dim=data.env_cfg.ACTION_DIM).to(args.device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    # Normalisation comes from the dataset's own statistics. Without it the
    # shoulder, which moves over tens of degrees, would dominate the gripper,
    # which moves over two.
    stats = dataset.meta.stats[data.ACTION_KEY]
    action_mean = torch.tensor(stats["mean"], device=args.device)
    action_std = torch.tensor(np.sqrt(stats["std"]), device=args.device)
    torch.save(
        {"action_mean": action_mean.cpu(), "action_std": action_std.cpu(), "steps": args.steps},
        output_dir / "normalisation.pt",
    )

    generator = np.random.default_rng(0)
    rows = len(dataset)
    started = time.time()
    history: list[tuple[int, float]] = []

    for step in range(args.steps):
        index = generator.choice(rows, size=min(args.batch, rows), replace=False)
        actions = torch.stack([dataset[i]["action"] for i in index]).float().to(args.device)
        states = torch.stack([dataset[i]["observation.state"] for i in index]).float().to(args.device)
        images = torch.stack([dataset[i][data.IMAGE_KEY] for i in index]).to(args.device)
        if images.dim() == 5:  # decoded video frames arrive as (B, 1, H, W, C)
            images = images[:, 0]
        predicted = policy(images, states)
        target = normalise(actions, action_mean, action_std)
        loss = nn.functional.mse_loss(predicted, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        history.append((step, float(loss)))
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  step {step + 1}/{args.steps} loss {float(loss):.4f}")

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "state_dim": data.env_cfg.STATE_DIM,
            "action_dim": data.env_cfg.ACTION_DIM,
        },
        output_dir / "bc_policy.pt",
    )
    write_summary(output_dir, history, args)

    print("")
    print(f"trained on {dataset.num_episodes} episode(s), {rows} frames, "
          f"{args.steps} steps, {time.time() - started:.0f}s")
    print(f"checkpoint: {output_dir / 'bc_policy.pt'}")
    print(f"summary:    {output_dir / 'summary.md'}")
    return 0


def write_summary(output_dir: Path, history, args) -> None:
    lines = [
        "# Behavior Cloning on the SO-101 lift-cube demonstrations",
        "",
        f"- dataset: `{args.dataset}`",
        f"- steps: {args.steps}, batch: {args.batch}, learning rate: {args.lr}",
        f"- device: {args.device}",
        f"- loss: {history[0][1]:.4f} at step 1, {history[-1][1]:.4f} at step {args.steps}",
        "",
        "The mean squared error is on normalised actions, so it is comparable",
        "between joints even though they have different ranges.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
