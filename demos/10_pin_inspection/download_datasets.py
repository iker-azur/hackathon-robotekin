"""Discover and download explicitly labelled pin/no-pin image datasets from the Hub."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import unicodedata

from PIL import Image
import yaml


IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_LABELS = {
    "con_perno": ["Pieza_con_perno", "con_perno", "ConPerno", "with_pin", "piece_with_pin", "pin"],
    "sin_perno": ["Pieza_sin_perno", "sin_perno", "SinPerno", "without_pin", "piece_without_pin", "nopin", "no_pin"],
}
METADATA_PATTERNS = ["*.json", "*.yaml", "*.yml", "*classes.txt", "*.names"]


def normalize(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def label_map(with_pin=(), without_pin=()):
    result = {}
    for canonical, extra in (("con_perno", with_pin), ("sin_perno", without_pin)):
        for label in DEFAULT_LABELS[canonical] + list(extra):
            key = normalize(label)
            if not key or (key in result and result[key] != canonical):
                raise ValueError(f"Conflicting or empty class name: {label!r}")
            result[key] = canonical
    return result


def definitions(root):
    """Read original class IDs; never assume that 0/1 have a particular meaning."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            content = json.loads(path.read_text(encoding="utf-8-sig"))
            if path.name == "index.json" and path.parent.name == "pin_snapshots":
                if not isinstance(content, dict) or not isinstance(content.get("images"), list):
                    raise ValueError(f"Invalid snapshot index: {path}")
                labels = {item["label"]: item["label"] for item in content["images"]}
                found.append((path, "pin_snapshots", labels, content))
            elif isinstance(content, dict) and isinstance(content.get("categories"), list):
                names = {int(c["id"]): c["name"] for c in content["categories"]}
                found.append((path, "coco", names, content))
        elif suffix in {".yaml", ".yml"}:
            content = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            if isinstance(content, dict) and isinstance(content.get("names"), (dict, list)):
                names = content["names"]
                names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
                found.append((path, "yolo", names, content))
        elif path.name.endswith("classes.txt") or suffix == ".names":
            names = dict(enumerate(path.read_text(encoding="utf-8-sig").splitlines()))
            found.append((path, "yolo", names, None))
    return found


def matching_classes(root, files, aliases):
    evidence = []
    for path, kind, names, _ in definitions(root):
        for class_id, name in names.items():
            if normalize(name) in aliases:
                evidence.append({"source": path.relative_to(root).as_posix(), "format": kind,
                                 "class_id": class_id, "original_label": name,
                                 "label": aliases[normalize(name)]})
    for filename in files:
        path = PurePosixPath(filename)
        if path.suffix.lower() in IMAGES:
            for part in path.parts[:-1]:
                if normalize(part) in aliases:
                    evidence.append({"source": part, "format": "imagefolder",
                                     "original_label": part, "label": aliases[normalize(part)]})
    return list({json.dumps(e, sort_keys=True): e for e in evidence}.values())


def inventory(root, aliases):
    """One row per labelled instance (or folder-labelled image), preserving mixed scenes."""
    root = root.resolve()
    images = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGES]
    by_name = {}
    for image in images:
        by_name.setdefault(image.name, []).append(image)
    rows, annotated = [], set()

    def add(image, original, source, **extra):
        canonical = aliases.get(normalize(original))
        if canonical is not None:
            rows.append({"image": image.relative_to(root).as_posix(), "label": canonical,
                         "original_label": original, "label_source": source.relative_to(root).as_posix(), **extra})
            annotated.add(image)

    def resolve(filename, annotation):
        # Exports may store file_name relative to the annotation or images directory.
        candidates = {p.resolve() for p in (annotation.parent / filename, annotation.parent / "images" / filename,
                                           root / filename) if p.is_file()}
        candidates = {p for p in candidates if p.is_relative_to(root.resolve()) and p.suffix.lower() in IMAGES}
        if len(candidates) == 1:
            return next(iter(candidates))
        matches = by_name.get(Path(filename).name, [])
        if not candidates and len(matches) == 1:
            return matches[0]
        raise ValueError(f"Missing or ambiguous image {filename!r} in {annotation}")

    defs = definitions(root)
    for annotation, kind, names, content in defs:
        if kind == "pin_snapshots":
            for number, item in enumerate(content["images"]):
                if normalize(item["label"]) not in aliases:
                    continue
                for camera in ("front", "wrist"):
                    image = (annotation.parent / item[camera]).resolve()
                    if not image.is_relative_to(annotation.parent.resolve()) or not image.is_file():
                        raise ValueError(f"Missing or unsafe snapshot image: {image}")
                    add(image, item["label"], annotation, camera=camera, location=item.get("location"),
                        pair_id=f"{annotation.relative_to(root).as_posix()}:{number}")
        if kind != "coco" or not any(normalize(n) in aliases for n in names.values()):
            continue
        image_names = {item["id"]: item["file_name"] for item in content.get("images", [])}
        for obj in content.get("annotations", []):
            original = names[obj["category_id"]]
            if normalize(original) in aliases:
                image = resolve(image_names[obj["image_id"]], annotation)
                add(image, original, annotation, annotation_id=obj.get("id"), class_id=obj["category_id"])

    for image in images:
        relative = image.relative_to(root)
        parts = list(relative.parts)
        if "images" in parts[:-1]:
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            annotation = root.joinpath(*parts).with_suffix(".txt")
            if annotation.is_file():
                configs = [d for d in defs if d[1] == "yolo" and image.is_relative_to(d[0].parent)]
                if not configs:
                    raise ValueError(f"No YOLO class names for {annotation}")
                # The nearest definition applies to this subtree, never a different split's YAML.
                depth = max(len(d[0].parent.parts) for d in configs)
                maps = [d[2] for d in configs if len(d[0].parent.parts) == depth]
                if any(m != maps[0] for m in maps):
                    raise ValueError(f"Conflicting YOLO class maps for {annotation}")
                names = maps[0]
                for number, line in enumerate(annotation.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if line.strip():
                        class_id = int(line.split()[0])
                        add(image, names[class_id], annotation, line=number, class_id=class_id)
        if image not in annotated:
            labels = {aliases[normalize(p)] for p in relative.parts[:-1] if normalize(p) in aliases}
            if len(labels) > 1:
                raise ValueError(f"Conflicting folder labels for {image}")
            for original in relative.parts[:-1]:
                if normalize(original) in aliases:
                    add(image, original, root / relative.parent)
                    break
    return rows


def classify_images(samples, output):
    """Copy decodable, unambiguously labelled photos; this is not a visual review."""
    folders = {"con_perno": "ConPerno", "sin_perno": "SinPerno"}
    for folder in folders.values():
        (output / folder).mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in samples:
        grouped.setdefault(row["local_image"], []).append(row)
    report = {"basis": "original_labels", "visual_review": "not_performed",
              "counts": {folder: 0 for folder in folders.values()}, "files": [], "pending": []}
    by_hash = {}
    for filename, rows in grouped.items():
        try:
            source = Path(filename)
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                image.load()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            item = by_hash.setdefault(digest, {"sources": [], "labels": set(), "source": source})
            item["sources"].append(filename)
            item["labels"].update(row["label"] for row in rows)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            report["pending"].append({"sources": [filename], "reason": f"invalid_image: {exc}"})
    for digest, item in by_hash.items():
        if len(item["labels"]) != 1:
            report["pending"].append({"sources": item["sources"], "reason": "mixed_or_conflicting_labels"})
            continue
        label = next(iter(item["labels"]))
        source = item["source"]
        destination = output / folders[label] / (digest + source.suffix.lower())
        opposite = output / folders["sin_perno" if label == "con_perno" else "con_perno"]
        if any(opposite.glob(digest + ".*")):
            report["pending"].append({"sources": item["sources"], "reason": "previous_copy_in_opposite_class"})
            continue
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Destination was modified: {destination}")
        else:
            shutil.copy2(source, destination)
        report["files"].append({"sources": item["sources"], "label": label,
                                "destination": str(destination), "sha256": digest})
        report["counts"][folders[label]] += 1
    (output / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def download(owner, output, aliases, api, fetch):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"owner": owner, "status": "incomplete", "datasets": [], "errors": []}
    samples = []
    try:
        for dataset in api.list_datasets(author=owner):
            entry = {"repo_id": dataset.id, "status": "inspecting"}
            report["datasets"].append(entry)
            print(f"Inspecting {dataset.id}", flush=True)
            try:
                info = api.dataset_info(dataset.id)
                revision = info.sha
                files = api.list_repo_files(dataset.id, repo_type="dataset", revision=revision)
                root = output / "raw" / dataset.id / revision
                fetch(repo_id=dataset.id, repo_type="dataset", revision=revision,
                      local_dir=str(root), allow_patterns=METADATA_PATTERNS)
                entry["revision"] = revision
                entry["classes"] = matching_classes(root, files, aliases)
                entry["containers_not_inspected"] = [f for f in files if Path(f).suffix.lower() in
                                                     {".zip", ".tar", ".gz", ".parquet", ".arrow"}]
                if not entry["classes"]:
                    entry["status"] = "no_explicit_target_labels"
                    info_path = root / "meta" / "info.json"
                    if info_path.is_file():
                        metadata = json.loads(info_path.read_text(encoding="utf-8"))
                        if metadata.get("robot_type") and metadata.get("features"):
                            entry["status"] = "lerobot_without_pin_snapshots"
                    continue
                print(f"Downloading {dataset.id}: {entry['classes']}", flush=True)
                options = {}
                if all(c["format"] == "pin_snapshots" for c in entry["classes"]):
                    options["allow_patterns"] = [c["source"].rsplit("/", 1)[0] + "/*" for c in entry["classes"]]
                    options["allow_patterns"] = sorted(set(options["allow_patterns"]))
                fetch(repo_id=dataset.id, repo_type="dataset", revision=revision, local_dir=str(root), **options)
                rows = inventory(root, aliases)
                for row in rows:
                    row.update(repo_id=dataset.id, revision=revision,
                               local_image=str(root / row["image"]))
                samples.extend(rows)
                entry.update(status="downloaded", instances=len(rows),
                             images=len({r["image"] for r in rows}), counts=dict(Counter(r["label"] for r in rows)))
            except Exception as exc:
                entry["status"] = "error"
                report["errors"].append({"repo_id": dataset.id, "error": str(exc)})
    except Exception as exc:
        report["errors"].append({"error": str(exc)})
    report["counts"] = dict(Counter(r["label"] for r in samples))
    try:
        report["classification"] = classify_images(samples, output)
    except (OSError, ValueError) as exc:
        report["errors"].append({"stage": "classification", "error": str(exc)})
    uninspected = any(d.get("containers_not_inspected") and not d.get("classes") and
                     d["status"] != "lerobot_without_pin_snapshots" for d in report["datasets"])
    if set(report["counts"]) == {"con_perno", "sin_perno"} and not report["errors"] and not uninspected:
        classification = report["classification"]
        if all(classification["counts"].values()) and not classification["pending"]:
            report["status"] = "complete"
    (output / "inventory.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples), encoding="utf-8")
    (output / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['counts']}. Report: {output / 'download_report.json'}", flush=True)
    for error in report["errors"]:
        print(f"ERROR: {error}", flush=True)
    return 0 if report["status"] == "complete" else 1


def main():
    from download_gui import main as open_window

    return open_window()


if __name__ == "__main__":
    raise SystemExit(main())
