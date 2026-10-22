"""Augmentation engine shared by the live preview and the offline dataset export (no Qt).

Parameters that exist in Ultralytics (flips, rotation, scale, translate, shear, perspective,
HSV, mosaic, mixup, cutmix, copy-paste) are passed to training unchanged (``train_kwargs``).
The geometric part mirrors Ultralytics' ``random_perspective`` so previews are faithful.
``contrast``, ``blur``, ``noise`` and ``erasing`` exist only here (preview and offline export);
Ultralytics ignores ``erasing`` for segmentation, so it is not forwarded.
Mosaic, mixup and cutmix happen only inside Ultralytics' training loop.
"""

from pathlib import Path
import json
import shutil

import cv2
import numpy as np

from .core import polygon, write_json

MIN_AREA = 64

# group, key, label, kind, min, max, step, default, scope, tooltip
_FIELDS = [
    ('Geometric', 'fliplr', 'Flip left-right (probability)', 'float', 0, 1, .05, .5, 'train', 'Mirrors the image horizontally.'),
    ('Geometric', 'flipud', 'Flip up-down (probability)', 'float', 0, 1, .05, .5, 'train', 'Mirrors the image vertically; parts lie flat on a table so this is realistic.'),
    ('Geometric', 'degrees', 'Rotation ± degrees', 'float', 0, 180, 5, 180, 'train', 'Random rotation. 180 = any orientation, right for parts seen from above.'),
    ('Geometric', 'translate', 'Translation (fraction of size)', 'float', 0, .9, .05, .1, 'train', 'Random shift.'),
    ('Geometric', 'scale', 'Scale ± gain', 'float', 0, .95, .05, .5, 'train', 'Random zoom in / out.'),
    ('Geometric', 'shear', 'Shear ± degrees', 'float', 0, 45, 1, 0, 'train', 'Skews the image.'),
    ('Geometric', 'perspective', 'Perspective', 'float', 0, .001, .0001, 0, 'train', 'Camera tilt simulation (Ultralytics range 0-0.001).'),
    ('Photometric', 'hsv_h', 'Hue shift', 'float', 0, 1, .01, 0, 'train', 'Keep 0 to preserve the colour of the parts.'),
    ('Photometric', 'hsv_s', 'Saturation shift', 'float', 0, 1, .05, 0, 'train', 'Keep 0 to preserve the colour of the parts.'),
    ('Photometric', 'hsv_v', 'Brightness shift', 'float', 0, 1, .05, .2, 'train', 'Lighting changes.'),
    ('Photometric', 'contrast', 'Contrast ±', 'float', 0, .6, .05, 0, 'offline', 'Preview / offline export only.'),
    ('Photometric', 'blur', 'Blur (probability)', 'float', 0, 1, .05, 0, 'offline', 'Preview / offline export only.'),
    ('Photometric', 'noise', 'Sensor noise (probability)', 'float', 0, 1, .05, 0, 'offline', 'Preview / offline export only.'),
    ('Occlusion', 'erasing', 'Random occluder (probability)', 'float', 0, 1, .05, 0, 'offline',
     'Covers part of the image and removes the covered pixels from the masks. Preview / offline only: Ultralytics ignores erasing for segmentation.'),
    ('Composition', 'mosaic', 'Mosaic (probability)', 'float', 0, 1, .05, .2, 'train', 'Four images in one. Training only.'),
    ('Composition', 'mixup', 'Mixup (probability)', 'float', 0, 1, .05, 0, 'train', 'Blends two images. Training only.'),
    ('Composition', 'cutmix', 'CutMix (probability)', 'float', 0, 1, .05, 0, 'train', 'Pastes a crop of another image. Training only.'),
    ('Composition', 'copy_paste', 'Copy-paste (probability)', 'float', 0, 1, .05, 0, 'train', 'Pastes mirrored copies of your masks into the image (uses the reviewed masks).'),
    ('Composition', 'copy_paste_mode', 'Copy-paste mode', 'choice', ('flip', 'mixup'), None, None, 'flip', 'train', 'Ultralytics copy-paste source.'),
    ('Composition', 'close_mosaic', 'Switch mosaic off for last epochs', 'int', 0, 100, 1, 10, 'train', 'Last N epochs train without mosaic.'),
]
FIELDS = [dict(zip(('group', 'key', 'label', 'kind', 'min', 'max', 'step', 'default', 'scope', 'tip'), row))
          for row in _FIELDS]
GROUPS = ['Geometric', 'Photometric', 'Occlusion', 'Composition']
TRAIN_KEYS = [f['key'] for f in FIELDS if f['scope'] == 'train']

PRESETS = {
    'Metal parts (default)': {},
    'Conservative': dict(fliplr=.5, flipud=0, degrees=15, translate=.05, scale=.2, hsv_v=.1, mosaic=0),
    'Heavy': dict(degrees=180, translate=.2, scale=.7, shear=5, perspective=.0005, hsv_v=.3, contrast=.2,
                  blur=.2, noise=.2, erasing=.3, mosaic=.8, mixup=.1, copy_paste=.3),
    'Off': dict(fliplr=0, flipud=0, degrees=0, translate=0, scale=0, hsv_v=0, mosaic=0, close_mosaic=0),
}


def defaults():
    return {f['key']: f['default'] for f in FIELDS}


def clean(config=None):
    """Validated copy: unknown keys dropped, numbers clamped to their range."""
    result = defaults()
    for field in FIELDS:
        if not config or field['key'] not in config:
            continue
        value = config[field['key']]
        if field['kind'] == 'choice':
            result[field['key']] = value if value in field['min'] else field['default']
        else:
            number = float(value)
            number = min(float(field['max']), max(float(field['min']), number))
            result[field['key']] = int(round(number)) if field['kind'] == 'int' else number
    return result


def preset(name):
    return clean(PRESETS[name])


def train_kwargs(config):
    """The subset of the configuration Ultralytics understands (written into job.json)."""
    config = clean(config)
    return {key: config[key] for key in TRAIN_KEYS}


# -- geometry -------------------------------------------------------------------------

def random_matrix(config, size, rng):
    """3x3 matrix = translate @ shear @ rotate/scale @ perspective @ centre (as Ultralytics)."""
    w, h = size
    centre = np.eye(3)
    centre[0, 2], centre[1, 2] = -w / 2, -h / 2
    persp = np.eye(3)
    persp[2, 0] = rng.uniform(-config['perspective'], config['perspective'])
    persp[2, 1] = rng.uniform(-config['perspective'], config['perspective'])
    rotate = np.eye(3)
    rotate[:2] = cv2.getRotationMatrix2D((0, 0), rng.uniform(-config['degrees'], config['degrees']),
                                         rng.uniform(1 - config['scale'], 1 + config['scale']))
    shear = np.eye(3)
    shear[0, 1] = np.tan(rng.uniform(-config['shear'], config['shear']) * np.pi / 180)
    shear[1, 0] = np.tan(rng.uniform(-config['shear'], config['shear']) * np.pi / 180)
    move = np.eye(3)
    move[0, 2] = rng.uniform(.5 - config['translate'], .5 + config['translate']) * w
    move[1, 2] = rng.uniform(.5 - config['translate'], .5 + config['translate']) * h
    return move @ shear @ rotate @ persp @ centre


def warp(image, masks, matrix, perspective):
    h, w = image.shape[:2]
    if perspective:
        out = cv2.warpPerspective(image, matrix, (w, h), borderValue=(114, 114, 114))
        moved = [cv2.warpPerspective(m.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST) > 0 for m in masks]
    else:
        out = cv2.warpAffine(image, matrix[:2], (w, h), borderValue=(114, 114, 114))
        moved = [cv2.warpAffine(m.astype(np.uint8), matrix[:2], (w, h), flags=cv2.INTER_NEAREST) > 0 for m in masks]
    return out, moved


def adjust_hsv(image, config, rng):
    gains = rng.uniform(-1, 1, 3) * np.array([config['hsv_h'], config['hsv_s'], config['hsv_v']]) + 1
    if np.allclose(gains, 1):
        return image
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=gains.dtype)
    hsv = cv2.merge((cv2.LUT(hue, ((x * gains[0]) % 180).astype(np.uint8)),
                     cv2.LUT(sat, np.clip(x * gains[1], 0, 255).astype(np.uint8)),
                     cv2.LUT(val, np.clip(x * gains[2], 0, 255).astype(np.uint8))))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def copy_paste(image, masks, labels, rng):
    """Paste mirrored copies of some instances into the image (Ultralytics 'flip' mode)."""
    if not masks:
        return image, masks, labels
    image, masks, labels = image.copy(), list(masks), list(labels)
    mirrored = image[:, ::-1]
    count = int(rng.integers(1, len(masks) + 1))
    for index in rng.choice(len(masks), count, replace=False):
        flipped = masks[index][:, ::-1]
        area = int(flipped.sum())
        if not area:
            continue
        existing = np.zeros(flipped.shape, bool)
        for mask in masks:
            existing |= mask
        if (flipped & existing).sum() / area >= .3:
            continue
        image[flipped] = mirrored[flipped]
        masks = [mask & ~flipped for mask in masks] + [flipped]
        labels.append(labels[index])
    return image, masks, labels


def random_erase(image, masks, rng):
    """1-3 rectangular occluders; their pixels leave every mask."""
    h, w = image.shape[:2]
    image = image.copy()
    masks = list(masks)
    for _ in range(int(rng.integers(1, 4))):
        area = rng.uniform(.02, .15) * w * h
        aspect = rng.uniform(.4, 2.5)
        rw, rh = int(min(w, np.sqrt(area * aspect))), int(min(h, np.sqrt(area / aspect)))
        x, y = int(rng.integers(0, w - rw + 1)), int(rng.integers(0, h - rh + 1))
        image[y:y + rh, x:x + rw] = rng.integers(0, 256, 3)
        masks = [m & ~_box(m.shape, x, y, rw, rh) for m in masks]
    return image, masks


def _box(shape, x, y, w, h):
    box = np.zeros(shape, bool)
    box[y:y + h, x:x + w] = True
    return box


def tidy(masks, labels, min_area=MIN_AREA):
    """Keep each instance's largest connected piece; drop ones that became tiny. Returns (masks, labels, dropped)."""
    kept_masks, kept_labels = [], []
    for mask, label in zip(masks, labels):
        count, parts, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[biggest, cv2.CC_STAT_AREA] < min_area:
            continue
        kept_masks.append(parts == biggest)
        kept_labels.append(label)
    return kept_masks, kept_labels, len(masks) - len(kept_masks)


def augment_sample(image, masks, labels, config, rng):
    """One random augmented sample. Returns dict(image, masks, labels, dropped)."""
    config = clean(config)
    image, masks, labels = image, [np.asarray(m, bool) for m in masks], list(labels)
    if rng.random() < config['copy_paste']:
        image, masks, labels = copy_paste(image, masks, labels, rng)
    if rng.random() < config['fliplr']:
        image, masks = image[:, ::-1], [m[:, ::-1] for m in masks]
    if rng.random() < config['flipud']:
        image, masks = image[::-1], [m[::-1] for m in masks]
    image = np.ascontiguousarray(image)
    masks = [np.ascontiguousarray(m) for m in masks]
    if any(config[k] for k in ('degrees', 'translate', 'scale', 'shear', 'perspective')):
        matrix = random_matrix(config, (image.shape[1], image.shape[0]), rng)
        image, masks = warp(image, masks, matrix, bool(config['perspective']))
    image = adjust_hsv(image, config, rng)
    if config['contrast'] and rng.random() < .5 + config['contrast']:
        alpha = 1 + rng.uniform(-config['contrast'], config['contrast'])
        image = np.clip((image.astype(np.float32) - 128) * alpha + 128, 0, 255).astype(np.uint8)
    if rng.random() < config['blur']:
        size = int(rng.choice([3, 5, 7]))
        image = cv2.GaussianBlur(image, (size, size), 0)
    if rng.random() < config['noise']:
        noise = rng.normal(0, rng.uniform(2, 10), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < config['erasing']:
        image, masks = random_erase(image, masks, rng)
    masks, labels, dropped = tidy(masks, labels)
    return {'image': image, 'masks': masks, 'labels': labels, 'dropped': dropped}


def fit_max_side(image, masks, max_side):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image, masks
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return (cv2.resize(image, size, interpolation=cv2.INTER_AREA),
            [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0 for m in masks])


def draw_preview(image, masks, colors, labels=None):
    """Translucent fill + outline per mask, for the preview grid."""
    out = image.copy()
    for index, mask in enumerate(masks):
        color = np.array(colors[index % len(colors)], np.float32)
        out[mask] = (.62 * out[mask] + .38 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), max(1, image.shape[1] // 400))
    return out


# -- offline export -------------------------------------------------------------------

def yolo_line(class_id, mask):
    points, holes = polygon(mask, allow_holes=True)
    return f'{class_id} ' + ' '.join(f'{v:.8f}' for v in points.ravel()), holes


def export_augmented(dataset, destination, copies, config, seed=42, max_side=1280, progress=None):
    """Copy a reviewed dataset and add `copies` augmented samples per **train** image.

    val/test files are copied untouched, so augmentation can never leak across splits.
    """
    dataset, destination = Path(dataset).resolve(), Path(destination).resolve()
    manifest_path = dataset / 'manifest.json'
    if not manifest_path.is_file() or not (dataset / 'data.yaml').is_file():
        raise ValueError('Select a reviewed dataset exported by this app')
    if destination.exists():
        raise FileExistsError('Choose a fresh output folder')
    if copies < 1:
        raise ValueError('Need at least one copy per image')
    config = clean(config)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    shutil.copytree(dataset, destination)
    (destination / 'manifest.json').unlink()
    train = [row for row in manifest['images'] if row['split'] == 'train']
    added, skipped = [], 0
    for number, row in enumerate(train, 1):
        image = cv2.cvtColor(cv2.imread(str(dataset / 'images' / 'train' / (row['image_id'] + '.png'))), cv2.COLOR_BGR2RGB)
        masks = [cv2.imread(str(dataset / inst['mask']), cv2.IMREAD_GRAYSCALE) > 0 for inst in row['instances']]
        classes = [inst['class_id'] for inst in row['instances']]
        image, masks = fit_max_side(image, masks, max_side)
        for k in range(copies):
            rng = np.random.default_rng((seed, number, k))
            sample = augment_sample(image, masks, classes, config, rng)
            if classes and not sample['masks']:
                skipped += 1
                continue
            lines = []
            for mask, class_id in zip(sample['masks'], sample['labels']):
                try:
                    lines.append(yolo_line(class_id, mask)[0])
                except ValueError:
                    continue
            if classes and not lines:
                skipped += 1
                continue
            name = f"{row['image_id']}__aug{k:02d}"
            cv2.imwrite(str(destination / 'images' / 'train' / (name + '.png')), cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            (destination / 'labels' / 'train' / (name + '.txt')).write_text('\n'.join(lines), encoding='utf-8')
            added.append({'image_id': name, 'augmented_from': row['image_id'], 'split': 'train',
                          'group': row['group'], 'instances': len(lines)})
        if progress:
            progress(number, len(train))
    result = {**manifest, 'schema': 'roboqc-augmented-dataset-1', 'source_dataset': str(dataset),
              'augmentation': {'config': config, 'seed': seed, 'copies': copies, 'max_side': max_side,
                               'applied_to': 'train', 'skipped_empty': skipped},
              'images': manifest['images'] + added}
    write_json(destination / 'manifest.json', result)
    return {'train_images': len(train), 'added': len(added), 'skipped': skipped, 'destination': destination}
