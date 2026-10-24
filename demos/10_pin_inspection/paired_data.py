"""Build paired front/wrist examples without losing session or duplicate provenance."""

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CLASSES = ("sin_perno", "con_perno")
DISPLAY_CLASSES = ("Pieza sin perno", "Pieza con perno")
CAMERAS = ("front", "wrist")


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_pairs(data_root, selected_repos=None):
    root = Path(data_root).resolve()
    selected = set(selected_repos) if selected_repos is not None else None
    groups = defaultdict(list)
    rejected = []
    with (root / "inventory.jsonl").open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if selected is not None and row.get("repo_id") not in selected:
                continue
            if not row.get("pair_id"):
                rejected.append({"image": row.get("image"), "reason": "unpaired_image"})
                continue
            groups[(row.get("repo_id"), row.get("revision"), row["pair_id"])].append(row)
    verified = {}
    candidates = defaultdict(list)
    for (repo, revision, pair_id), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        try:
            if not repo or not revision or len(rows) != 2 or {r.get("camera") for r in rows} != set(CAMERAS):
                raise ValueError("incomplete_or_duplicate_camera")
            labels = {row.get("label") for row in rows}
            if len(labels) != 1 or not labels.issubset(CLASSES):
                raise ValueError("conflicting_or_unknown_labels")
            pair = {"repo_id": repo, "revision": revision, "pair_id": pair_id,
                    "label": next(iter(labels)), "location": rows[0].get("location")}
            for row in rows:
                # Prefer relocatable raw paths; old inventories also contain absolute paths.
                image = (root / "raw" / repo / revision / row["image"]).resolve()
                if not image.is_file():
                    image = Path(row["local_image"]).resolve()
                if not image.is_relative_to(root):
                    raise ValueError("image_outside_dataset_directory")
                if image not in verified:
                    with Image.open(image) as photo:
                        photo.verify()
                    with Image.open(image) as photo:
                        photo.load()
                    verified[image] = hashlib.sha256(image.read_bytes()).hexdigest()
                camera = row["camera"]
                pair[camera] = str(image)
                pair[camera + "_sha256"] = verified[image]
            pair["id"] = hashlib.sha256(
                (pair["front_sha256"] + pair["wrist_sha256"]).encode()
            ).hexdigest()
            pair["sources"] = [{"repo_id": repo, "revision": revision, "pair_id": pair_id}]
            candidates[pair["id"]].append(pair)
        except (OSError, ValueError, KeyError, Image.DecompressionBombError) as error:
            rejected.append({"repo_id": repo, "revision": revision, "pair_id": pair_id, "reason": str(error)})
    pairs = []
    duplicates = 0
    for digest, versions in sorted(candidates.items()):
        if len({pair["label"] for pair in versions}) != 1:
            rejected.append({"id": digest, "reason": "identical_pair_with_conflicting_labels",
                             "sources": [source for pair in versions for source in pair["sources"]]})
            continue
        pair = versions[0]
        pair["sources"] = [source for version in versions for source in version["sources"]]
        duplicates += len(versions) - 1
        pairs.append(pair)
    # Sessions sharing even one image must stay on the same side of the split.
    parents = {source["repo_id"]: source["repo_id"] for pair in pairs for source in pair["sources"]}

    def find(session):
        while parents[session] != session:
            parents[session] = parents[parents[session]]
            session = parents[session]
        return session

    def join(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    seen_images = {}
    for pair in pairs:
        sessions = [source["repo_id"] for source in pair["sources"]]
        for session in sessions:
            join(sessions[0], session)
        for camera in CAMERAS:
            digest = pair[camera + "_sha256"]
            if digest in seen_images:
                join(sessions[0], seen_images[digest])
            seen_images[digest] = sessions[0]
    for pair in pairs:
        pair["group"] = find(pair["repo_id"])
    return {"pairs": pairs, "rejected": rejected, "duplicates_removed": duplicates,
            "counts": dict(Counter(pair["label"] for pair in pairs))}


def split_pairs(pairs, seed=42):
    """Approximately 80/20, with both classes on each side and no shared sessions/images."""
    if {pair["label"] for pair in pairs} != set(CLASSES):
        raise ValueError("Se necesitan parejas válidas de ambas clases")
    groups = sorted({pair["group"] for pair in pairs})
    if len(groups) < 2:
        raise ValueError("No hay suficientes sesiones independientes para separar entrenamiento y validación")
    counts = {group: Counter(pair["label"] for pair in pairs if pair["group"] == group) for group in groups}
    total = Counter(pair["label"] for pair in pairs)
    rng = random.Random(seed)
    order = groups.copy()
    rng.shuffle(order)
    masks = itertools.product((False, True), repeat=len(order)) if len(order) <= 12 else (
        [rng.random() < 0.2 for _ in order] for _ in range(4096)
    )
    best = None
    for mask in masks:
        validation = {group for group, chosen in zip(order, mask) if chosen}
        val_counts = Counter()
        for group in validation:
            val_counts.update(counts[group])
        if any(not 0 < val_counts[label] < total[label] for label in CLASSES):
            continue
        score = sum(abs(val_counts[label] / total[label] - 0.2) for label in CLASSES)
        candidate = (score, tuple(group for group in order if group in validation))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("No se pueden reservar sesiones con ambas clases en train y validación. "
                         "Selecciona o recoge más sesiones; las imágenes compartidas no se separan")
    validation = set(best[1])
    return {"seed": seed, "classes": list(CLASSES), "cameras": list(CAMERAS),
            "train": [pair for pair in pairs if pair["group"] not in validation],
            "validation": [pair for pair in pairs if pair["group"] in validation]}
