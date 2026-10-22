"""Lazy read-only bundle access, immutable review revisions and dataset export."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
import uuid
import zipfile

import cv2
import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def safe_name(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or '..' in path.parts or any(':' in p for p in path.parts):
        raise ValueError(f'Unsafe bundle path: {name}')
    return path


def decode_image(data, gray=False):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Unreadable image')
    return image if gray else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_mask(path, mask):
    ok, data = cv2.imencode('.png', np.asarray(mask, np.uint8) * 255)
    if not ok:
        raise OSError('Mask PNG encoding failed')
    Path(path).write_bytes(data.tobytes())


class Bundle:
    """Reads only manifests and requested RGB/masks; never extracts or runs archive code."""

    def __init__(self, source):
        self.source = Path(source).resolve()
        self.archive = zipfile.ZipFile(self.source) if self.source.is_file() else None
        self.prefix = ''
        if self.archive:
            names = set()
            for member in self.archive.infolist():
                safe_name(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('Archive symlinks are not supported')
                key = member.filename.casefold()
                if key in names:
                    raise ValueError('Duplicate archive path')
                names.add(key)
            roots = [n.filename[:-len('images.json')] for n in self.archive.infolist()
                     if PurePosixPath(n.filename).name == 'images.json']
            if len(roots) != 1:
                raise ValueError('Expected one 04e images.json (not the input-only image ZIP)')
            self.prefix = roots[0]
        self.images = self.json('images.json')
        self.proposals = self.json('proposals.json')
        self.by_image = {row['image_id']: row for row in self.images}
        if len(self.by_image) != len(self.images):
            raise ValueError('Duplicate image IDs')
        self.by_proposal = {row['proposal_id']: row for row in self.proposals}
        self.alternatives = defaultdict(list)
        for row in self.proposals:
            self.alternatives[row['image_id']].append(row)
        self.identity = hashlib.sha256(json.dumps(self.images, sort_keys=True).encode()).hexdigest()

    def close(self):
        if self.archive:
            self.archive.close()

    def read(self, relative):
        path = safe_name(relative)
        if self.archive:
            member = self.archive.getinfo(self.prefix + path.as_posix())
            if member.file_size > 512 * 1024**2:
                raise ValueError('Single member exceeds 512 MiB safety limit')
            return self.archive.read(member)
        target = self.source.joinpath(*path.parts).resolve()
        if not target.is_relative_to(self.source):
            raise ValueError('Bundle symlink escapes source')
        return target.read_bytes()

    def json(self, relative):
        return json.loads(self.read(relative))

    def rgb(self, image_id):
        row = self.by_image[image_id]
        image = decode_image(self.read(row['rgb']))
        if image.shape[:2] != (row['height'], row['width']):
            raise ValueError('RGB size differs from manifest')
        return image

    def candidates(self, proposal_id):
        row = self.by_proposal[proposal_id]
        if row['status'] != 'ok':
            raise ValueError(row.get('error', 'Proposal failed'))
        manifest = self.json(row['candidates'])
        parent = PurePosixPath(row['candidates']).parent
        image = self.by_image[row['image_id']]
        result = []
        for candidate in manifest['candidates']:
            if not candidate.get('kept') or not candidate.get('mask_file'):
                continue
            mask = decode_image(self.read((parent / safe_name(candidate['mask_file'])).as_posix()), True)
            if mask.shape != (image['height'], image['width']) or not np.isin(mask, [0, 1, 255]).all():
                raise ValueError('Expected original-size binary candidate mask')
            result.append({'id': candidate['id'], 'label': 'part', 'mask': mask > 0,
                           'keep': True, 'source': candidate})
        return result


class Project:
    def __init__(self, folder, bundle):
        self.folder = Path(folder).resolve()
        if self.folder == bundle.source or (bundle.source.is_dir() and self.folder.is_relative_to(bundle.source)):
            raise ValueError('Review project must be outside the source bundle')
        self.folder.mkdir(parents=True, exist_ok=True)
        self.bundle = bundle
        self.index_path = self.folder / 'project.json'
        if self.index_path.exists():
            self.state = json.loads(self.index_path.read_text(encoding='utf-8'))
            if self.state['bundle_identity'] != bundle.identity:
                raise ValueError('Project belongs to different images')
        else:
            self.state = {'schema': 'roboqc-review-project-1', 'bundle_identity': bundle.identity,
                          'source': str(bundle.source), 'reviews': {}}
            write_json(self.index_path, self.state)

    def decision(self, image_id):
        return self.state['reviews'].get(image_id)

    def load_masks(self, image_id):
        decision = self.decision(image_id)
        if not decision or decision['status'] != 'accepted':
            return []
        result = []
        for row in decision['instances']:
            mask = decode_image((self.folder / row['mask_file']).read_bytes(), True) > 0
            result.append({**row, 'mask': mask, 'keep': True})
        return result

    def save(self, image_id, proposal_id, instances, status='accepted', group='', negative=False):
        if status not in ('accepted', 'excluded'):
            raise ValueError('Review must be accepted or excluded')
        image = self.bundle.by_image[image_id]
        selected = [row for row in instances if row.get('keep', True)] if status == 'accepted' else []
        if status == 'accepted' and not selected and not negative:
            raise ValueError('Empty labels require explicit confirmed negative image')
        if proposal_id is not None and self.bundle.by_proposal[proposal_id]['image_id'] != image_id:
            raise ValueError('Proposal belongs to another image')
        for row in selected:
            mask = np.asarray(row['mask'])
            if mask.dtype != bool or mask.shape != (image['height'], image['width']) or not mask.any():
                raise ValueError('Every selected instance needs a nonempty original-size bool mask')
            if not row['label'].strip():
                raise ValueError('Every selected instance needs a class name')
        revision = self.folder / 'revisions' / uuid.uuid4().hex
        revision.mkdir(parents=True, exist_ok=False)
        records = []
        for number, row in enumerate(selected):
            target = revision / f'instance_{number:04d}.png'
            save_mask(target, row['mask'])
            records.append({'id': number, 'label': row['label'].strip(),
                            'mask_file': target.relative_to(self.folder).as_posix(),
                            'source_id': row.get('id')})
        review = {'status': status, 'proposal_id': proposal_id, 'instances': records,
                  'negative': bool(negative and not records), 'group': group.strip() or image_id,
                  'image_id': image_id, 'revision': revision.name}
        write_json(revision / 'review.json', review)
        self.state['reviews'][image_id] = review
        write_json(self.index_path, self.state)
        return review


def split_reviews(project, seed=42):
    """Keep declared capture groups AND exact original hashes together across splits."""
    reviews = [r for r in project.state['reviews'].values() if r['status'] == 'accepted']
    parent = {r['image_id']: r['image_id'] for r in reviews}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    owners = {}
    for review in reviews:
        image_id = review['image_id']
        image = project.bundle.by_image[image_id]
        for key in [('group', review['group']), ('sha', image['sha256'])]:
            if key in owners:
                parent[find(image_id)] = find(owners[key])
            owners[key] = image_id
    groups = defaultdict(list)
    for review in reviews:
        groups[find(review['image_id'])].append(review['image_id'])
    groups = sorted(groups.values(), key=lambda g: sorted(g))
    if len(groups) < 3:
        raise ValueError('Need at least 3 independent accepted capture groups for train/val/test')
    random.Random(seed).shuffle(groups)
    ntest = max(1, round(len(groups) * .15))
    nval = max(1, round(len(groups) * .15))
    result = {}
    for index, group in enumerate(groups):
        split = 'test' if index < ntest else 'val' if index < ntest + nval else 'train'
        result.update({image_id: split for image_id in group})
    return result


def polygon(mask, allow_holes=False):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError('Empty mask')
    external = [c for c, h in zip(contours, hierarchy[0]) if h[3] == -1]
    holes = any(h[3] != -1 for h in hierarchy[0])
    if len(external) != 1:
        raise ValueError('Disconnected instance: split or repair mask before YOLO export')
    if holes and not allow_holes:
        raise ValueError('Mask has holes. Explicitly allow hole filling for lossy YOLO polygon export')
    points = external[0].reshape(-1, 2)
    if len(points) < 3 or cv2.contourArea(external[0]) <= 0:
        raise ValueError('Mask too small/thin for a YOLO polygon')
    return points / np.array([mask.shape[1], mask.shape[0]]), holes


def export_dataset(project, destination, seed=42, allow_holes=False, progress=None):
    """Copy accepted normalized images, exact masks and YOLO polygons; never mutate inputs."""
    import yaml
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a fresh export folder')
    splits = split_reviews(project, seed)
    reviews = project.state['reviews']
    names = sorted({i['label'] for key in splits for i in reviews[key]['instances']})
    if not names:
        raise ValueError('Need at least one labeled foreground instance')
    classes = {name: index for index, name in enumerate(names)}
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'roboqc-reviewed-dataset-1', 'seed': seed, 'classes': names,
                'split_policy': '70/15/15 capture groups; exact duplicates joined', 'images': [],
                'warning': 'YOLO polygons fill approved holes; exact reviewed PNG masks retained'}
    for split in ('train', 'val', 'test'):
        (destination / 'images' / split).mkdir(parents=True)
        (destination / 'labels' / split).mkdir(parents=True)
    for number, (image_id, split) in enumerate(sorted(splits.items()), 1):
        image, review = project.bundle.by_image[image_id], reviews[image_id]
        lines, instance_rows = [], []
        mask_folder = destination / 'reviewed_masks' / image_id
        mask_folder.mkdir(parents=True)
        for index, row in enumerate(project.load_masks(image_id)):
            try:
                points, holes = polygon(row['mask'], allow_holes)
            except ValueError as exc:
                raise ValueError(f"{image['source_relative_path']}, instance {index}: {exc}") from exc
            lines.append(str(classes[row['label']]) + ' ' + ' '.join(f'{v:.8f}' for v in points.ravel()))
            target = mask_folder / f'{index:04d}.png'
            shutil.copyfile(project.folder / row['mask_file'], target)
            instance_rows.append({'class_id': classes[row['label']], 'mask': target.relative_to(destination).as_posix(),
                                  'holes_filled_in_yolo': holes})
        (destination / 'images' / split / (image_id + '.png')).write_bytes(project.bundle.read(image['rgb']))
        (destination / 'labels' / split / (image_id + '.txt')).write_text('\n'.join(lines), encoding='utf-8')
        manifest['images'].append({'image_id': image_id, 'source': image['source_relative_path'],
                                  'sha256': image['sha256'], 'split': split, 'group': review['group'],
                                  'proposal_id': review['proposal_id'], 'revision': review['revision'],
                                  'instances': instance_rows})
        if progress:
            progress(number, len(splits))
    if not any(r['instances'] for r in manifest['images'] if r['split'] == 'train'):
        raise ValueError('Training split has no foreground. Review groups/seeds before training')
    config = {'path': '.', 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': names}
    (destination / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    write_json(destination / 'manifest.json', manifest)
    return destination


def zip_dataset(folder, target, progress=None):
    folder, target = Path(folder).resolve(), Path(target).resolve()
    if target.is_relative_to(folder):
        raise ValueError('ZIP must be outside dataset folder')
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_STORED) as archive:
        for number, path in enumerate(files, 1):
            if not path.resolve().is_relative_to(folder):
                raise ValueError('Dataset symlink escapes folder')
            archive.write(path, path.relative_to(folder).as_posix())
            if progress:
                progress(number, len(files))
    return target
