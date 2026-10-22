"""Photo input helper embedded as source in public notebooks; never embeds photos."""

from pathlib import Path
import uuid


def select_image_paths(image_path, run, uploader=None):
    suffixes = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    if image_path:
        source = Path(image_path).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
    else:
        if uploader is None:
            from google.colab import files
            uploader = files.upload
        uploaded = uploader()
        if not uploaded:
            raise ValueError('Upload one or more photos, or set IMAGE_PATH to a Drive image/folder')
        if any(Path(name).name != name or '\\' in name or ':' in name or
               Path(name).suffix.lower() not in suffixes for name in uploaded):
            raise ValueError('Upload image files only (not ZIPs); use IMAGE_PATH for nested folders')
        source = Path(run) / ('uploaded_photos_' + uuid.uuid4().hex)
        source.mkdir(parents=True, exist_ok=False)
        for name, blob in uploaded.items():
            with (source / name).open('xb') as handle:
                handle.write(blob)
    paths = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
    paths = [p for p in paths if p.suffix.lower() in suffixes]
    if not paths:
        raise ValueError('No supported photos found; set IMAGE_PATH or upload photos')
    return paths
