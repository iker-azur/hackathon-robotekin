"""Supervised training of a two-view pin classifier, independent of ACT."""

import math
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch
from model import ARCHITECTURE, CAMERA_SETS, PREPROCESS, PairedClassifier, load_image, normalize
from paired_data import CAMERAS, CLASSES, load_pairs, split_pairs, write_json
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# Probabilities are 0..1 and 0 disables a transform; the other values set its maximum strength.
AUGMENTATIONS = ("horizontal_flip", "vertical_flip", "rotation", "translation", "perspective")
DEFAULT_AUGMENTATION = {
    "horizontal_flip": 0.0, "vertical_flip": 0.0, "rotation": 0.0, "translation": 0.0, "perspective": 0.0,
    "rotation_degrees": 15.0, "translation_fraction": 0.1, "perspective_scale": 0.3,
}
AUGMENTATION_LIMITS = {"rotation_degrees": 180.0, "translation_fraction": 0.5, "perspective_scale": 1.0}


@dataclass
class TrainConfig:
    data_root: str
    output: str
    selected_repos: list[str] | None = None
    cameras: list[str] = field(default_factory=lambda: list(CAMERAS))
    epochs: int = 30
    batch_size: int = 16
    backbone_lr: float = 1e-5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    pretrained: bool = True
    augmentation: dict = field(default_factory=lambda: dict(DEFAULT_AUGMENTATION))
    stop_file: str | None = None

    def validate(self):
        for name in ("epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser un entero positivo")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed debe estar entre 0 y 2**32-1")
        for name in ("backbone_lr", "learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"{name} debe ser finito y positivo (weight_decay permite cero)")
        if type(self.pretrained) is not bool:
            raise ValueError("pretrained debe ser booleano")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device debe ser auto, cpu o cuda")
        if not isinstance(self.cameras, list) or tuple(self.cameras) not in CAMERA_SETS:
            raise ValueError("cameras debe ser [front, wrist], [front] o [wrist]")
        if not isinstance(self.augmentation, dict) or not set(self.augmentation) <= set(DEFAULT_AUGMENTATION):
            raise ValueError("augmentation solo admite: " + ", ".join(DEFAULT_AUGMENTATION))
        self.augmentation = {**DEFAULT_AUGMENTATION, **self.augmentation}
        for name, value in self.augmentation.items():
            limit = AUGMENTATION_LIMITS.get(name, 1.0)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError(f"augmentation.{name} debe estar entre 0 y {limit:g}")


def build_augmentation(settings):
    """Random transforms drawn again every time a sample is loaded, hence every epoch."""
    translate = (settings["translation_fraction"],) * 2
    return v2.Compose([
        v2.RandomHorizontalFlip(settings["horizontal_flip"]),
        v2.RandomVerticalFlip(settings["vertical_flip"]),
        v2.RandomApply([v2.RandomRotation(settings["rotation_degrees"])], p=settings["rotation"]),
        v2.RandomApply([v2.RandomAffine(0, translate=translate)], p=settings["translation"]),
        v2.RandomPerspective(settings["perspective_scale"], p=settings["perspective"]),
    ])


class Cancelled(Exception):
    pass


class PairedImages(Dataset):
    def __init__(self, pairs, cameras, augment=None):
        self.pairs, self.cameras, self.augment = pairs, cameras, augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        # Each view gets its own draw: front and wrist are separate cameras.
        views = (load_image(pair[camera]) for camera in self.cameras)
        if self.augment is not None:
            views = (self.augment(view) for view in views)
        return (*(normalize(view) for view in views), CLASSES.index(pair["label"]))


def classification_metrics(matrix):
    """Rows are actual classes, columns are predictions, in CLASSES order."""
    total = sum(sum(row) for row in matrix)
    per_class = {}
    for index, label in enumerate(CLASSES):
        true_positive = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        actual = sum(matrix[index])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    return {"accuracy": sum(matrix[i][i] for i in range(2)) / total if total else 0.0,
            "macro_f1": sum(row["f1"] for row in per_class.values()) / 2,
            "per_class": per_class, "confusion_matrix": matrix, "class_order": list(CLASSES)}


def _epoch(model, loader, device, optimizer, stop_file):
    model.train(optimizer is not None)
    total_loss, examples = 0.0, 0
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    for *views, labels in loader:
        if stop_file and Path(stop_file).exists():
            raise Cancelled()
        views, labels = [view.to(device) for view in views], labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(*views)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita; entrenamiento detenido")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predictions = logits.detach().argmax(-1)
        matrix += torch.bincount((labels * 2 + predictions).cpu(), minlength=4).reshape(2, 2)
        total_loss += loss.item() * len(labels)
        examples += len(labels)
    return {"loss": total_loss / examples, **classification_metrics(matrix.tolist())}


def selection_key(validation):
    """Lower validation loss wins; among equal losses, higher accuracy wins."""
    return (validation["loss"], -validation["accuracy"])


def _save_checkpoint(path, model, cameras, epoch, metrics):
    content = {"model": model.state_dict(), "architecture": ARCHITECTURE, "preprocess": PREPROCESS,
               "classes": list(CLASSES), "cameras": list(cameras), "epoch": epoch, "metrics": metrics}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def train(settings, emit=lambda event: None):
    config = TrainConfig(**settings)
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible en el intérprete seleccionado")
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    emit({"event": "status", "text": "Verificando imágenes y separación por sesiones…"})
    report = load_pairs(config.data_root, config.selected_repos)
    split = split_pairs(report["pairs"], config.seed)
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "started_at": datetime.now(UTC).isoformat(), "device": device}
    write_json(output / "run_config.json", {**asdict(config), "architecture": ARCHITECTURE,
                                           "preprocess": PREPROCESS, "classes": list(CLASSES)})
    write_json(output / "split.json", split)
    write_json(output / "data_report.json", report)
    write_json(output / "status.json", state)
    history, best_key, best_metrics, best_epoch = [], None, None, None
    try:
        if config.stop_file and Path(config.stop_file).exists():
            raise Cancelled()
        emit({"event": "status", "text": f"Inicializando ResNet18 en {device} con {' + '.join(config.cameras)}; "
              f"{len(split['train'])} parejas train y {len(split['validation'])} validación"})
        # Download failures propagate. Do not silently replace pretrained weights with random weights.
        model = PairedClassifier(pretrained=config.pretrained, views=len(config.cameras)).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": config.backbone_lr},
            {"params": model.fusion.parameters(), "lr": config.learning_rate},
        ], weight_decay=config.weight_decay)
        generator = torch.Generator().manual_seed(config.seed)
        enabled = [name for name in AUGMENTATIONS if config.augmentation[name] > 0]
        if enabled:
            emit({"event": "status", "text": "Aumentación por época: " + ", ".join(
                f"{name} {config.augmentation[name]:.0%}" for name in enabled)})
        augment = build_augmentation(config.augmentation) if enabled else None
        train_loader = DataLoader(PairedImages(split["train"], config.cameras, augment), batch_size=config.batch_size,
                                  shuffle=True, generator=generator, num_workers=0)
        val_loader = DataLoader(PairedImages(split["validation"], config.cameras), batch_size=config.batch_size, num_workers=0)
        for epoch in range(1, config.epochs + 1):
            train_metrics = _epoch(model, train_loader, device, optimizer, config.stop_file)
            validation = _epoch(model, val_loader, device, None, config.stop_file)
            row = {"epoch": epoch, "train": train_metrics, "validation": validation}
            _save_checkpoint(output / "last.pt", model, config.cameras, epoch, row)
            if best_key is None or selection_key(validation) < best_key:
                _save_checkpoint(output / "best.pt", model, config.cameras, epoch, row)
                best_key, best_metrics, best_epoch = selection_key(validation), validation, epoch
            history.append(row)
            write_json(output / "history.json", history)
            write_json(output / "metrics.json", {"best_epoch": best_epoch, "validation": best_metrics,
                                                 "selection": "minimum_validation_loss_then_maximum_accuracy"})
            emit({"event": "epoch", "epochs": config.epochs, **row})
        state["status"] = "completed"
    except (Cancelled, KeyboardInterrupt):
        state["status"] = "interrupted"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state.update(finished_at=datetime.now(UTC).isoformat(), completed_epochs=len(history))
        write_json(output / "status.json", state)
    return {"event": "result", "status": state["status"], "output": str(output),
            "best_epoch": best_epoch, "validation": best_metrics}
