"""One or two ordered RGB views, one binary decision. No robot or LeRobot dependency."""

from pathlib import Path

import torch
from paired_data import CAMERAS, CLASSES
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

IMAGE_SIZE = 224
ARCHITECTURE = {"backbone": "resnet18", "shared_backbone": True, "fusion_hidden": 128, "dropout": 0.2}
CAMERA_SETS = (CAMERAS, ("front",), ("wrist",))
PREPROCESS = {"size": IMAGE_SIZE, "resize": "contain_pad", "color": "RGB",
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def load_image(path):
    """Padded 0..1 tensor before normalization, so augmentation fills with the same black as padding."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image = ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    return TF.to_tensor(image)


def normalize(image):
    return TF.normalize(image, PREPROCESS["mean"], PREPROCESS["std"])


def preprocess(path):
    return normalize(load_image(path))


class PairedClassifier(nn.Module):
    def __init__(self, pretrained=True, views=len(CAMERAS)):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.fusion = nn.Sequential(nn.Linear(512 * views, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, *views):
        if not views or any(view.shape != views[0].shape for view in views) or views[0].ndim != 4 \
                or views[0].shape[1] != 3:
            raise ValueError("las vistas deben tener la misma forma B×3×H×W")
        features = self.backbone(torch.cat(views, dim=0))
        return self.fusion(torch.cat(features.chunk(len(views), dim=0), dim=1))


def load_classifier(checkpoint, device="cpu"):
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    cameras = saved.get("cameras")
    if saved.get("classes") != list(CLASSES) or cameras not in [list(option) for option in CAMERA_SETS]:
        raise ValueError("Checkpoint incompatible: clases u orden de cámaras")
    if saved.get("architecture") != ARCHITECTURE or saved.get("preprocess") != PREPROCESS:
        raise ValueError("Checkpoint incompatible: arquitectura o preprocesado")
    model = PairedClassifier(pretrained=False, views=len(cameras))
    model.load_state_dict(saved["model"])
    model.cameras = tuple(cameras)
    return model.to(device).eval()


@torch.inference_mode()
def predict(checkpoint, images, device="cpu"):
    """images maps camera name to path; it must contain every camera the checkpoint uses."""
    model = load_classifier(checkpoint, device)
    missing = [camera for camera in model.cameras if not images.get(camera)]
    if missing:
        raise ValueError("Falta la imagen de la cámara: " + ", ".join(missing))
    probabilities = model(*(preprocess(images[camera]).unsqueeze(0).to(device)
                            for camera in model.cameras)).softmax(-1)[0].cpu().tolist()
    return {"label": CLASSES[max(range(2), key=lambda i: probabilities[i])],
            "probabilities": dict(zip(CLASSES, probabilities)), "cameras": list(model.cameras)}
