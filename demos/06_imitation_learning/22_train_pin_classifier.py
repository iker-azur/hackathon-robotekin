"""Train a very small two-input CNN: piece or piece with pin (perno), from two cameras.

Input is the dataset ``21_build_pin_dataset.py`` writes: the last frames of the front and
the wrist camera of each labelled pick-cube episode. Each camera has its own tiny
convolutional stream (four conv + batch-norm + ReLU + max-pool blocks, about 80 k
parameters each); both are pooled (mean and max, so a small pin is not averaged away),
concatenated and classified by one linear layer.

``--backbone resnet18`` swaps those streams for an ImageNet-pretrained ResNet18 (torchvision
weights, downloaded once) shared by both cameras and fine-tuned (backbone at a tenth of the head
learning rate; ``--freeze`` trains only the head). ``--augment strong`` (default) adds random
rotation, zoom, shift, flip, saturation / white-balance changes, blur and noise on the GPU;
``--augment basic`` is the shift / brightness / flip of the first version.

How good it is on unseen data is measured by cross-validation over whole *sessions*: the
frames of one episode are near-duplicates and the pieces/lighting repeat inside a session,
so a random split would look much better than it is. ``--folds 0`` skips that and only
trains the final model, which is then trained on all the data and saved to
``outputs/<output>/model.pt`` with ``metrics.json``.

The labels are a visual judgement (see ``pin_labels.json``); ``--clear-only`` trains and
tests only on the ones marked ``clear``. The report also shows the accuracy on the clear
labels alone and the "always answer the commonest class" baseline: read the gap to that
baseline, not the raw accuracy.

Use a saved model from other code with ``load_model`` and ``predict`` below.

Run from the repository root, in the ``hackathon-robotekin`` environment:

    python .\\demos\\06_imitation_learning\\22_train_pin_classifier.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEMO_DIR = Path(__file__).resolve().parent
DATA_DIR = DEMO_DIR / "data" / "real_so101"
CLASS_NAMES = ("nopin", "pin")
CHANNELS = (16, 32, 64, 96)


def conv_stream(channels) -> nn.Sequential:
    layers, previous = [], 3
    for c in channels:
        layers += [nn.Conv2d(previous, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                   nn.MaxPool2d(2)]
        previous = c
    return nn.Sequential(*layers)


class PinNet(nn.Module):
    """front and wrist image (N, 3, H, W), already normalised -> 2 logits (nopin, pin)."""

    def __init__(self, channels=CHANNELS, dropout: float = 0.3) -> None:
        super().__init__()
        self.front = conv_stream(channels)
        self.wrist = conv_stream(channels)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * channels[-1], 2))

    @staticmethod
    def pool(x):
        return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)

    def forward(self, front, wrist):
        return self.head(torch.cat([self.pool(self.front(front)), self.pool(self.wrist(wrist))], 1))

    @staticmethod
    def prepare(x):
        return (x - 0.5) / 0.25


class ResNetPinNet(nn.Module):
    """Same interface as PinNet, but one ImageNet-pretrained ResNet18 (without its classifier) for both cameras."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * 512, 2))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def prepare(self, x):
        return (x - self.mean) / self.std

    def forward(self, front, wrist):
        n = len(front)
        pooled = PinNet.pool(self.backbone(torch.cat([front, wrist])))
        return self.head(torch.cat([pooled[:n], pooled[n:]], 1))


def build_model(backbone: str, pretrained: bool = True) -> nn.Module:
    return ResNetPinNet(pretrained) if backbone == "resnet18" else PinNet()


def to_tensor(images: np.ndarray, device) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB -> (N, 3, H, W) float in [0, 1]."""
    return torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random shift (up to 8 px), brightness / contrast and horizontal flip, per image."""
    n, _, h, w = x.shape
    padded = F.pad(x, (8, 8, 8, 8), mode="reflect")
    out = torch.empty_like(x)
    for i in range(n):
        dy, dx = (int(v) for v in torch.randint(0, 17, (2,), generator=generator))
        crop = padded[i, :, dy:dy + h, dx:dx + w]
        if torch.rand(1, generator=generator).item() < 0.5:
            crop = crop.flip(2)
        contrast = 0.75 + 0.5 * torch.rand(1, generator=generator).item()
        brightness = 0.2 * torch.rand(1, generator=generator).item() - 0.1
        out[i] = ((crop - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)
    return out


def augment_strong(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Per image: rotation (+-12 deg), zoom (+-15 %), shift (+-10 %), flip, brightness / contrast /
    saturation / per-channel gain, sometimes blur, light noise. Whole batch on the GPU."""
    n = x.shape[0]

    def rnd(*shape):
        return torch.rand(*shape, generator=generator).to(x.device)

    angle = (rnd(n) - 0.5) * 2 * math.radians(12)
    zoom = 1 / (0.85 + 0.3 * rnd(n))
    flip = torch.where(rnd(n) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) * zoom, torch.sin(angle) * zoom
    shift = (rnd(n, 2) - 0.5) * 0.2
    theta = torch.stack([torch.stack([cos * flip, -sin, shift[:, 0]], 1),
                         torch.stack([sin * flip, cos, shift[:, 1]], 1)], 1)
    out = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                        padding_mode="reflection", align_corners=False)
    gray = out.mean(1, keepdim=True)
    saturation = (0.6 + 0.8 * rnd(n)).view(n, 1, 1, 1)
    out = gray + (out - gray) * saturation
    out = out * (0.9 + 0.2 * rnd(n, 3)).view(n, 3, 1, 1)
    contrast = (0.7 + 0.6 * rnd(n)).view(n, 1, 1, 1)
    out = (out - 0.5) * contrast + 0.5 + ((rnd(n) - 0.5) * 0.3).view(n, 1, 1, 1)
    blur = (rnd(n) < 0.3).float().view(n, 1, 1, 1) * (0.4 + 0.5 * rnd(n)).view(n, 1, 1, 1)
    out = out + blur * (F.avg_pool2d(out, 5, 1, 2, count_include_pad=False) - out)
    out = out + (rnd(n) * 0.03).view(n, 1, 1, 1) * (rnd(*out.shape) - 0.5) * 2
    return out.clamp(0, 1)


def train_model(front, wrist, labels, epochs: int, batch: int, lr: float, device, seed: int,
                backbone: str = "small", augmentation: str = "basic", freeze: bool = False) -> nn.Module:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = build_model(backbone).to(device)
    counts = np.bincount(labels, minlength=2)
    weight = torch.tensor(len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    if backbone == "resnet18":
        groups = [{"params": model.head.parameters(), "lr": lr}]
        if freeze:
            model.backbone.requires_grad_(False)
        else:
            groups.append({"params": model.backbone.parameters(), "lr": lr * 0.1})
        optimiser = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-2)
        max_lr = [g["lr"] for g in groups]
    else:
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        max_lr = lr
    steps = epochs * int(np.ceil(len(labels) / batch))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, total_steps=steps)
    targets = torch.from_numpy(labels).to(device)
    rng = np.random.default_rng(seed)
    augment_fn = augment_strong if augmentation == "strong" else augment
    for _ in range(epochs):
        model.train()
        if freeze:
            model.backbone.eval()
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            if len(idx) < 2:  # batch norm needs more than one sample
                continue
            f = model.prepare(augment_fn(to_tensor(front[idx], device), generator))
            w = model.prepare(augment_fn(to_tensor(wrist[idx], device), generator))
            loss = F.cross_entropy(model(f, w), targets[idx], weight=weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
    return model


@torch.no_grad()
def predict_proba(model: PinNet, front, wrist, device) -> np.ndarray:
    """Probability of "pin" for (N, H, W, 3) uint8 front / wrist arrays."""
    model.eval()
    out = []
    for start in range(0, len(front), 64):
        f = model.prepare(to_tensor(front[start:start + 64], device))
        w = model.prepare(to_tensor(wrist[start:start + 64], device))
        out.append(torch.softmax(model(f, w), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def load_model(path: Path, device="cpu") -> tuple[PinNet, dict]:
    saved = torch.load(path, map_location=device, weights_only=True)
    backbone = saved.get("backbone", "small")
    model = (build_model(backbone, pretrained=False) if backbone == "resnet18"
             else PinNet(tuple(saved["channels"]))).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved


def predict(model: PinNet, saved: dict, front_rgb: np.ndarray, wrist_rgb: np.ndarray, device="cpu") -> tuple[str, float]:
    """One pair of RGB camera frames of any size -> (class name, probability of "pin")."""
    spec = importlib.util.spec_from_file_location("build_act", DEMO_DIR / "19_build_so101_act_dataset.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    size = (saved["height"], saved["width"])
    front = build.fit(front_rgb, *size)[None]
    wrist = build.fit(wrist_rgb, *size)[None]
    p = float(predict_proba(model, front, wrist, device)[0])
    return CLASS_NAMES[int(p >= 0.5)], p


def by_episode(samples, indices) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for i in indices:
        groups.setdefault((samples[i]["session"], samples[i]["episode"]), []).append(i)
    return groups


def summarise(truth: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    tp, tn = int(((pred == 1) & (truth == 1)).sum()), int(((pred == 0) & (truth == 0)).sum())
    fp, fn = int(((pred == 1) & (truth == 0)).sum()), int(((pred == 0) & (truth == 1)).sum())
    recalls = [tn / max(tn + fp, 1), tp / max(tp + fn, 1)]
    return {"n": len(truth), "accuracy": (tp + tn) / max(len(truth), 1), "balanced_accuracy": float(np.mean(recalls)),
            "baseline": float(max(np.mean(truth == 0), np.mean(truth == 1))),
            "confusion": {"nopin_ok": tn, "nopin_as_pin": fp, "pin_as_nopin": fn, "pin_ok": tp}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="pin_cls", help="dataset folder under data/real_so101")
    parser.add_argument("--clear-only", action="store_true", help="use only the labels marked clear")
    parser.add_argument("--folds", type=int, default=5, help="session-grouped cross-validation folds (0 = none)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None, help="head learning rate (default 3e-3 small, 1e-3 resnet18)")
    parser.add_argument("--backbone", choices=("small", "resnet18"), default="small")
    parser.add_argument("--augment", choices=("basic", "strong"), default="strong")
    parser.add_argument("--freeze", action="store_true", help="resnet18: train only the classifier head")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="pin_classifier",
                        help="folder under outputs/ (an existing one is refused)")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.backbone == "resnet18" else 3e-3

    root = DATA_DIR / args.dataset
    if not (root / "pin_cls.npz").exists():
        print(f"ERROR: {root} is not a dataset built by 21_build_pin_dataset.py")
        return 1
    output = DEMO_DIR / "outputs" / args.output
    if output.exists():
        print(f"ERROR: {output} already exists; pass another --output or delete it.")
        return 1
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using the CPU (this model is small enough).")
        args.device = "cpu"
    device = torch.device(args.device)

    data = np.load(root / "pin_cls.npz")
    samples = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    keep = np.array([not args.clear_only or s["confidence"] == "clear" for s in samples])
    front, wrist, labels = data["front"][keep], data["wrist"][keep], data["label"][keep]
    samples = [s for s, k in zip(samples, keep) if k]
    height, width = front.shape[1:3]
    episodes = {(s["session"], s["episode"]) for s in samples}
    print(f"{len(samples)} samples, {len(episodes)} episodes, {width}x{height}, "
          f"nopin {int((labels == 0).sum())} / pin {int((labels == 1).sum())} frames")
    print(f"parameters: {sum(p.numel() for p in build_model(args.backbone, pretrained=False).parameters()):,} "
          f"({args.backbone}, {args.augment} augmentation{', frozen backbone' if args.freeze else ''})")

    report: dict = {"clear_only": args.clear_only, "episodes": len(episodes), "samples": len(samples),
                    "backbone": args.backbone, "augment": args.augment, "freeze": args.freeze}
    fit_args = (args.epochs, args.batch, args.lr, device)
    train_kwargs = {"backbone": args.backbone, "augmentation": args.augment, "freeze": args.freeze}
    sessions = sorted({s["session"] for s in samples})
    if args.folds > 1:
        folds = min(args.folds, len(sessions))
        fold_of = {session: i % folds for i, session in enumerate(sessions)}
        prob = np.full(len(samples), np.nan)
        started = time.time()
        for fold in range(folds):
            test = np.array([fold_of[s["session"]] == fold for s in samples])
            if len(set(labels[~test])) < 2:
                print(f"  fold {fold}: the training part has one class only, skipped")
                continue
            model = train_model(front[~test], wrist[~test], labels[~test], *fit_args, args.seed + fold,
                                **train_kwargs)
            prob[test] = predict_proba(model, front[test], wrist[test], device)
            fold_acc = float(((prob[test] >= 0.5) == labels[test]).mean())
            print(f"  fold {fold}: {int(test.sum())} frames from {sum(1 for v in fold_of.values() if v == fold)} "
                  f"session(s), accuracy {fold_acc:.2f}", flush=True)
        done = ~np.isnan(prob)
        groups = by_episode(samples, np.flatnonzero(done))
        ep_truth = np.array([labels[idx[0]] for idx in groups.values()])
        ep_prob = np.array([prob[idx].mean() for idx in groups.values()])
        ep_clear = np.array([all(samples[i]["confidence"] == "clear" for i in idx) for idx in groups.values()])
        report["cross_validation"] = {
            "folds": folds, "per_frame": summarise(labels[done], prob[done]),
            "per_episode": summarise(ep_truth, ep_prob),
            "per_episode_clear_labels_only": summarise(ep_truth[ep_clear], ep_prob[ep_clear]) if ep_clear.any() else None,
        }
        sources = np.array([s.get("source", "episode") for s in samples])
        report["cross_validation"].update({
            f"per_frame_{name}_only": summarise(labels[done & (sources == name)], prob[done & (sources == name)])
            for name in sorted(set(sources[done]))})
        print(f"cross-validation over {folds} session groups ({time.time() - started:.0f} s):")
        for name in [k for k in report["cross_validation"] if k.startswith("per_")]:
            r = report["cross_validation"][name]
            if r:
                print(f"  {name:30s} n={r['n']:3d} accuracy {r['accuracy']:.2f} (baseline {r['baseline']:.2f}) "
                      f"balanced {r['balanced_accuracy']:.2f}  {r['confusion']}")

    model = train_model(front, wrist, labels, *fit_args, args.seed, **train_kwargs)
    fit_prob = predict_proba(model, front, wrist, device)
    report["training_set_accuracy"] = float(((fit_prob >= 0.5) == labels).mean())
    output.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "channels": list(CHANNELS), "height": int(height),
                "width": int(width), "classes": list(CLASS_NAMES), "backbone": args.backbone}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"final model (trained on everything, training-set accuracy {report['training_set_accuracy']:.2f}, "
          f"not a test result): {output / 'model.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
