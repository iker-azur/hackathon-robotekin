"""Image/archive I/O for human-reviewed segmentation proposals. No model execution."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import uuid
import zipfile

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_files(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if any(not p.resolve().is_relative_to(root) for p in paths):
        raise ValueError("Image symlink escapes input folder")
    if not paths:
        raise ValueError(f"No supported images found recursively in {root}")
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def pack_images(root, destination):
    """Package existing photos only, preserving relative directories and original bytes."""
    root, destination = Path(root).resolve(), Path(destination)
    paths = image_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
    return {"images": len(paths), "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def extract_images(archive_path, destination, maximum_bytes=10*1024**3, maximum_files=50000):
    """Extract only images into a fresh directory; preflight ZIP traversal and size limits."""
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        members = []
        names = set()
        for member in archive.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("ZIP symlinks are not supported")
            if member.is_dir() or name.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = name.as_posix().casefold()
            if key in names:
                raise ValueError("Duplicate ZIP image path")
            names.add(key)
            members.append((member, name))
        if not members or len(members) > maximum_files or sum(m.file_size for m, _ in members) > maximum_bytes:
            raise ValueError("Empty image ZIP or configured extraction limits exceeded")
        destination.mkdir(parents=True, exist_ok=False)
        for member, name in members:
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return destination


def register_images(source, bundle):
    """Save originals and EXIF-normalized RGB PNGs once; proposals reference these IDs."""
    from PIL import Image, ImageOps
    source, bundle = Path(source).resolve(), Path(bundle)
    rows = []
    for path in image_files(source):
        relative = path.relative_to(source).as_posix()
        digest = sha256(path)
        image_id = hashlib.sha256((relative + "\0" + digest).encode()).hexdigest()[:24]
        folder = bundle / "images" / image_id
        folder.mkdir(parents=True, exist_ok=False)
        original = folder / ("original" + path.suffix.lower())
        shutil.copy2(path, original)
        row = {"image_id": image_id, "source_relative_path": relative, "sha256": digest,
               "original": original.relative_to(bundle).as_posix(), "status": "ok"}
        try:
            with Image.open(path) as opened:
                rgb = ImageOps.exif_transpose(opened).convert("RGB")
                rgb.save(folder / "rgb.png")
                row.update(width=rgb.width, height=rgb.height, rgb=(folder / "rgb.png").relative_to(bundle).as_posix())
        except (OSError, ValueError) as exc:
            row.update(status="invalid_image", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        write_json(bundle / "images.json", rows)
    reviews = {row["image_id"]: {"status": "unreviewed", "selected_proposal_id": None,
               "instances": [], "split_group": None} for row in rows}
    write_json(bundle / "reviews.json", reviews)
    write_json(bundle / "registration_complete.json", {"images": len(rows)})
    return rows


def settings_compatible(saved, current):
    """Allow export/discovery-only changes, never changed inference source or settings."""
    def comparable(value):
        value = dict(value)
        hashes = dict(value.get("runtime_sha256", {}))
        hashes.pop("roboqc/industrial/review_bundle.py", None)
        value["runtime_sha256"] = hashes
        return value
    return comparable(saved) == comparable(current)


def expected_proposals(settings, samples):
    active = [s for s in samples if s["status"] == "ok"]
    if settings["max_images"] is not None:
        active = active[:settings["max_images"]]
    return {f"{sample['image_id']}:{method}:{pi}"
            for sample in active for method in settings["methods"]
            for pi in (range(len(settings["prompts"])) if method == "sam3_text" else [0])}


def discover_bundles(root, settings, progress=None):
    """Rank manifests, not thousands of masks; prune bundle/export/input subtrees."""
    root = Path(root).resolve()
    rows = []
    for directory, children, _ in os.walk(root):
        path = Path(directory)
        children[:] = sorted(name for name in children if name not in {"inputs", "exports"})
        if not path.name.startswith("review_bundle_"):
            continue
        children[:] = []
        row = {"path": str(path), "successful": 0, "remaining": None, "compatible": False}
        try:
            saved = json.loads((path / "settings.json").read_text())
            samples = json.loads((path / "images.json").read_text())
            if not (path / "registration_complete.json").is_file():
                raise ValueError("Input registration incomplete")
            index = path / "proposals.json"
            records = json.loads(index.read_text()) if index.exists() else []
            expected = expected_proposals(saved, samples)
            completed = {r["proposal_id"] for r in records if r["status"] == "ok"} & expected
            row.update(successful=len(completed), remaining=len(expected)-len(completed),
                       images=len(samples), compatible=settings_compatible(saved, settings))
            if not row["compatible"]:
                row["error"] = "Inference settings/source differ"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress:
            progress("Find saved bundles", len(rows), None)
    return sorted(rows, key=lambda row: (row["compatible"], row["successful"], row["path"]), reverse=True)


def archive_bundle(bundle, destination, *, staging_directory=None, progress=None):
    """Read Drive sources once; build/verify locally, then copy a new ZIP to Drive."""
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    if destination.is_relative_to(bundle):
        raise ValueError("Archive must be outside the bundle")
    if destination.exists():
        raise FileExistsError(destination)
    def report(phase, done, total):
        if progress:
            progress(phase, done, total)
    files = []
    for path in bundle.rglob("*"):
        if path.is_file() and path != bundle / "checksums.json":
            if not path.resolve().is_relative_to(bundle):
                raise ValueError("Bundle symlink escapes its folder")
            files.append((path, path.stat().st_size))
            report("Scan files", len(files), None)
    files.sort()
    report("Scan files", len(files), len(files))
    total = sum(size for _, size in files)
    if staging_directory is not None:
        staging_directory = Path(staging_directory).resolve()
        if staging_directory.is_relative_to(bundle):
            raise ValueError("Local staging must be outside the bundle")
        staging_directory.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="review_export_", dir=staging_directory) as temporary:
        local = Path(temporary) / "results.zip"
        if shutil.disk_usage(temporary).free < total + len(files)*1024 + 1024**2:
            raise OSError("Insufficient local disk space for ZIP staging. Free local space, not saved Drive outputs.")
        inventory = {}
        done = 0
        report("Archive + hash", done, total)
        # PNG/JPEG/NPZ are already compressed. ZIP_STORED avoids expensive recompression.
        with zipfile.ZipFile(local, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path, size in files:
                digest = hashlib.sha256()
                with path.open("rb") as source, archive.open(path.relative_to(bundle).as_posix(), "w", force_zip64=True) as output:
                    read = 0
                    for chunk in iter(lambda: source.read(1024*1024), b""):
                        digest.update(chunk); output.write(chunk)
                        read += len(chunk); done += len(chunk)
                        report("Archive + hash", done, total)
                if read != size:
                    raise OSError(f"Bundle changed during export: {path}. Stop inference/review writes before exporting.")
                inventory[path.relative_to(bundle).as_posix()] = digest.hexdigest()
            archive.writestr("checksums.json", json.dumps(inventory, ensure_ascii=False, indent=2))
        write_json(bundle / "checksums.json", inventory)
        with zipfile.ZipFile(local) as archive:
            total = sum(member.file_size for member in archive.infolist())
            done = 0
            report("Verify local ZIP", done, total)
            for member in archive.infolist():
                digest = hashlib.sha256()
                with archive.open(member) as source:
                    for chunk in iter(lambda: source.read(1024*1024), b""):
                        digest.update(chunk); done += len(chunk)
                        report("Verify local ZIP", done, total)
                # Reading each member also validates its ZIP CRC.
                if member.filename in inventory and digest.hexdigest() != inventory[member.filename]:
                    raise OSError(f"ZIP checksum mismatch: {member.filename}")
        # Do not publish a partial archive under a completed .zip filename.
        partial = destination.with_name(destination.name + ".partial_" + uuid.uuid4().hex)
        digest = hashlib.sha256(); done = 0; total = local.stat().st_size
        report("Copy ZIP to Drive", done, total)
        with local.open("rb") as source, partial.open("xb") as output:
            for chunk in iter(lambda: source.read(4*1024*1024), b""):
                output.write(chunk); digest.update(chunk); done += len(chunk)
                report("Copy ZIP to Drive", done, total)
        if partial.stat().st_size != total:
            raise OSError(f"Incomplete destination ZIP: {partial}")
        if destination.exists():
            raise FileExistsError(destination)
        partial.rename(destination)
    return {"path": str(destination), "sha256": digest.hexdigest(), "files": len(inventory)+1,
            "bytes": total, "verified": "local_crc_and_sha256; destination_size_checked"}
