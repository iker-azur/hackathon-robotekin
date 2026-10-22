"""Suggestions derived from source paths. Nothing here is stored until a human accepts.

* ``suggest_class`` reads the class from the folder (``Pieza_con_perno`` -> that class).
* ``auto_groups`` splits a folder of near-duplicate burst frames into time windows.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import PurePosixPath
import re

_STAMP = re.compile(r'(\d{8})_(\d{6})')
_CLASS_FOLDER = re.compile(r'^pieza[\s_-]', re.IGNORECASE)


def parse_timestamp(name):
    """Capture time from ``img_YYYYMMDD_HHMMSS_...`` file names, or None."""
    match = _STAMP.search(PurePosixPath(name).name)
    if not match:
        return None
    try:
        return datetime.strptime(match[1] + match[2], '%Y%m%d%H%M%S')
    except ValueError:
        return None


def folder_of(path):
    parent = str(PurePosixPath(path).parent)
    return '' if parent == '.' else parent


def suggest_class(path, known=()):
    """Class name implied by a folder of ``path``, or None (e.g. the ``tests/`` folder)."""
    folders = PurePosixPath(path).parts[:-1]
    registered = {name.casefold(): name for name in known}
    for part in folders:
        if part.casefold() in registered:
            return registered[part.casefold()]
    for part in folders:
        if _CLASS_FOLDER.match(part):
            return part
    return None


def seed_classes(images):
    """Distinct folder-implied class names over a bundle, in a stable order."""
    return sorted({name for row in images if (name := suggest_class(row['source_relative_path']))})


def auto_groups(images, window_seconds):
    """Map image_id -> capture group.

    With ``window_seconds`` <= 0 the group is the folder (safe: whole bursts stay together).
    Otherwise each folder is cut into consecutive blocks no longer than ``window_seconds``;
    frames without a timestamp keep the folder group. Neighbouring blocks of one burst are
    still near-duplicates, so a short window can leak between train and validation.
    """
    result = {row['image_id']: folder_of(row['source_relative_path']) for row in images}
    if window_seconds <= 0:
        return result
    folders = defaultdict(list)
    for row in images:
        stamp = parse_timestamp(row['source_relative_path'])
        if stamp is not None:
            folders[folder_of(row['source_relative_path'])].append((stamp, row['image_id']))
    for folder, frames in folders.items():
        frames.sort()
        block, start = 0, frames[0][0]
        for stamp, image_id in frames:
            if (stamp - start).total_seconds() > window_seconds:
                block, start = block + 1, stamp
            result[image_id] = f'{folder}#{block + 1:02d}'
    return result
