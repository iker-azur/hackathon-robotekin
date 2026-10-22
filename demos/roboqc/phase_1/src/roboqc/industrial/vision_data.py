"""Real industrial annotation import and grouped splits, called only by Colab."""

import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from .vision_inference import CLASSES, QUALITY


def _safe(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Image path outside dataset: {relative}")
    return path


def import_coco(root, annotation_file):
    from pycocotools import mask as mask_utils
    root = Path(root)
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in data["categories"]}
    if set(categories.values()) - set(CLASSES):
        raise ValueError(f"Use these category names: {CLASSES}")
    grouped = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            raise ValueError("Crowd regions cannot be used as individual graspable instances")
        grouped.setdefault(ann["image_id"], []).append(ann)
    records = []
    for image in data["images"]:
        path = _safe(root, image["file_name"])
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable image: {path}")
        h, w = frame.shape[:2]
        if [h, w] != [image["height"], image["width"]]:
            raise ValueError(f"Annotation dimensions differ: {path}")
        instances = []
        for ann in grouped.get(image["id"], []):
            segmentation = ann["segmentation"]
            if isinstance(segmentation, list):
                rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
            elif isinstance(segmentation.get("counts"), list):
                rle = mask_utils.frPyObjects(segmentation, h, w)
            else:
                rle = segmentation
            mask = mask_utils.decode(rle).astype(np.uint8)
            if mask.shape != (h,w) or not mask.any():
                raise ValueError(f"Invalid mask in annotation {ann['id']}")
            attrs = ann.get("attributes", {})
            quality = attrs.get("quality", "unknown")
            color = attrs.get("color")
            if quality not in QUALITY + ["unknown"] or color not in [None,"red","green","blue","unknown"]:
                raise ValueError("Unsupported instance attributes")
            instances.append({"class": categories[ann["category_id"]], "mask": mask,
                              "quality": quality, "color": color})
        session = image.get("session_id")
        if session is None:
            raise ValueError("Each COCO image needs session_id to avoid capture-session leakage")
        records.append({"path": str(path), "session": str(session), "instances": instances})
    return records


def import_yolo(root, yaml_file, sessions_file, attributes_file=None):
    import yaml
    root = Path(root)
    data = yaml.safe_load(Path(yaml_file).read_text(encoding="utf-8"))
    names = data["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    if set(names) - set(CLASSES):
        raise ValueError(f"Use these categories: {CLASSES}")
    sessions = json.loads(Path(sessions_file).read_text(encoding="utf-8"))
    attributes = json.loads(Path(attributes_file).read_text(encoding="utf-8")) if attributes_file else {}
    records = []
    # Imported train/val/test folders are regrouped by session; no augmentations yet.
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"] or "images" not in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in sessions:
            raise ValueError(f"Missing capture session for {relative}")
        parts = list(path.parts)
        parts[max(i for i,p in enumerate(parts) if p == "images")] = "labels"
        label = Path(*parts).with_suffix(".txt")
        if not label.exists():
            raise ValueError(f"Missing label file, including for empty images: {label}")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable image: {path}")
        h, w = frame.shape[:2]
        instances = []
        for i, line in enumerate(label.read_text().splitlines()):
            values = np.array([float(v) for v in line.split()])
            if len(values) < 7 or (len(values)-1) % 2 or not np.isfinite(values).all() or np.any(values[1:] < 0) or np.any(values[1:] > 1):
                raise ValueError(f"Expected normalized segmentation polygons: {label}:{i+1}")
            class_id = int(values[0])
            if values[0] != class_id or not 0 <= class_id < len(names):
                raise ValueError(f"Invalid class ID: {label}:{i+1}")
            polygon = np.rint(values[1:].reshape(-1,2) * [w,h]).astype(np.int32)
            mask = np.zeros((h,w), np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            attrs = attributes.get(relative, [{}] * (i+1))
            attrs = attrs[i] if i < len(attrs) else {}
            instances.append({"class": names[class_id], "mask": mask, "quality": attrs.get("quality", "unknown"), "color": attrs.get("color")})
        records.append({"path": str(path), "session": str(sessions[relative]), "instances": instances})
    return records


def prepare(records, output, definitions=None, seed=42):
    import yaml
    definitions = definitions or {}
    if not records:
        raise ValueError("No real annotated images found. Import a COCO or YOLO dataset first.")
    groups = sorted({r["session"] for r in records})
    if len(groups) < 3:
        raise ValueError("Capture at least three independent sessions for train/validation/test")
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups)*.15)); n_test = max(1, round(len(groups)*.15))
    split_of = {g: ("val" if i < n_val else "test" if i < n_val+n_test else "train") for i,g in enumerate(groups)}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    hashes, manifest, losses = {}, [], []
    for record in records:
        path = Path(record["path"])
        split = split_of[record["session"]]
        frame = cv2.imread(str(path))
        digest = hashlib.sha256(frame.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate image {path} (already in {hashes[digest]}); remove before splitting")
        hashes[digest] = str(path)
        h,w = frame.shape[:2]
        stem = digest[:20]
        for folder in ["images", "labels", "masks"]:
            (out / folder / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"{stem}.png"), frame)
        polygons, instances = [], []
        for i, instance in enumerate(record["instances"]):
            mask = instance["mask"]
            if not mask.any():
                raise ValueError("Empty mask")
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            external = [c for j,c in enumerate(contours) if hierarchy[0,j,3] < 0]
            contour = max(external, key=cv2.contourArea).reshape(-1,2)
            if len(contour) < 3:
                raise ValueError("Degenerate contour")
            if len(external) > 1 or np.any(hierarchy[0,:,3] >= 0):
                losses.append({"image": stem, "instance": i, "reason": "YOLO polygon drops holes/disconnected regions; original PNG retained"})
            coordinates = (contour / [w,h]).clip(0,1).reshape(-1)
            polygons.append(str(CLASSES.index(instance["class"])) + " " + " ".join(f"{v:.6f}" for v in coordinates))
            mask_relative = f"masks/{split}/{stem}_{i}.png"
            cv2.imwrite(str(out / mask_relative), mask*255)
            instances.append({k:v for k,v in instance.items() if k != "mask"} | {"mask_path": mask_relative})
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(polygons), encoding="utf-8")
        manifest.append({"image": f"images/{split}/{stem}.png", "split": split, "session": record["session"], "instances": instances})
    for split in ["train","val","test"]:
        present = {i["class"] for r in manifest if r["split"] == split for i in r["instances"]}
        if present != set(CLASSES):
            raise ValueError(f"{split} needs real examples of all classes; missing {set(CLASSES)-present}. Add sessions or change seed.")
    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASSES}), encoding="utf-8")
    (out / "instances.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "quality_definitions.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    (out / "conversion_warnings.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")
    return manifest


def augment_training(output, copies=1, seed=42):
    """Matched image/mask transforms, train only; never alter hue labels."""
    out = Path(output)
    records = json.loads((out / "instances.json").read_text())
    rng = np.random.default_rng(seed)
    added = []
    for record in records:
        if record["split"] != "train" or record.get("augmentation_parent"):
            continue
        original = cv2.imread(str(out / record["image"]))
        masks = [cv2.imread(str(out / i["mask_path"]),0) for i in record["instances"]]
        for index in range(copies):
            turns = int(rng.integers(1,4))
            frame = np.rot90(original,turns).copy()
            transformed = [np.rot90(m,turns).copy() for m in masks]
            if rng.random() < .5:
                frame = cv2.flip(frame,1)
                transformed = [cv2.flip(m,1) for m in transformed]
            frame = np.clip(frame.astype(np.float32)*rng.uniform(.85,1.15),0,255).astype(np.uint8)
            stem = Path(record["image"]).stem + f"_aug{index}"
            image_path = f"images/train/{stem}.png"
            if (out / image_path).exists():
                raise FileExistsError("Augmentations already exist; use a new run")
            cv2.imwrite(str(out / image_path),frame)
            h,w = frame.shape[:2]
            annotations, lines = [], []
            for j,(instance,mask) in enumerate(zip(record["instances"],transformed)):
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                polygon = max(contours,key=cv2.contourArea).reshape(-1,2) / [w,h]
                lines.append(str(CLASSES.index(instance["class"])) + " " + " ".join(f"{v:.6f}" for v in polygon.reshape(-1)))
                mask_path = f"masks/train/{stem}_{j}.png"
                cv2.imwrite(str(out / mask_path),mask)
                annotations.append({**instance,"mask_path":mask_path})
            (out / f"labels/train/{stem}.txt").write_text("\n".join(lines))
            added.append({**record,"image":image_path,"instances":annotations,"augmentation_parent":record["image"]})
    (out / "instances.json").write_text(json.dumps(records+added,indent=2))
    return records+added
