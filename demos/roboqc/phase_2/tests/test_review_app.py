"""Handmade masks only: no model import, inference, download or training."""

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from review_app.core import Bundle, Project, export_dataset, polygon, safe_name, save_mask, split_reviews, write_json, zip_dataset


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / 'bundle'
    folder.mkdir()
    images, proposals = [], []
    for index in range(8):
        image_id = f'image_{index}'
        rgb = np.full((64, 96, 3), 100 + index, np.uint8)
        image_dir = folder / 'images' / image_id
        image_dir.mkdir(parents=True)
        ok, encoded = cv2.imencode('.png', rgb)
        assert ok
        (image_dir / 'rgb.png').write_bytes(encoded.tobytes())
        images.append({'image_id': image_id, 'source_relative_path': f'session_{index}/nested/photo.png',
                       'rgb': f'images/{image_id}/rgb.png', 'original': f'images/{image_id}/rgb.png',
                       'width': 96, 'height': 64, 'status': 'ok',
                       'sha256': hashlib.sha256(encoded.tobytes()).hexdigest()})
        proposal_id = image_id + ':sam3_text:0'
        mask_dir = folder / 'proposals' / image_id / 'masks'
        mask_dir.mkdir(parents=True)
        mask = np.zeros((64, 96), bool)
        mask[10:40, 20:60] = True
        save_mask(mask_dir / 'mask_0000.png', mask)
        write_json(mask_dir / 'candidates.json', {'candidates': [
            {'id': 0, 'kept': True, 'mask_file': 'mask_0000.png'},
            {'id': 1, 'kept': False, 'mask_file': None}]})
        proposals.append({'proposal_id': proposal_id, 'image_id': image_id, 'method': 'sam3_text',
                          'status': 'ok', 'kept_masks': 1, 'candidates': f'proposals/{image_id}/masks/candidates.json'})
    write_json(folder / 'images.json', images)
    write_json(folder / 'proposals.json', proposals)
    source = Bundle(folder)
    yield source
    source.close()


def digest_tree(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def approve(project, count=6):
    for image in project.bundle.images[:count]:
        proposal = image['image_id'] + ':sam3_text:0'
        project.save(image['image_id'], proposal, project.bundle.candidates(proposal), group=image['image_id'])


def test_lazy_zip_and_nested_root(bundle, tmp_path):
    archive = tmp_path / 'results.zip'
    with zipfile.ZipFile(archive, 'x') as handle:
        for path in bundle.source.rglob('*'):
            if path.is_file():
                handle.write(path, 'review_bundle/' + path.relative_to(bundle.source).as_posix())
    loaded = Bundle(archive)
    try:
        assert loaded.identity == bundle.identity
        assert loaded.rgb('image_0').shape == (64, 96, 3)
        assert len(loaded.candidates('image_0:sam3_text:0')) == 1
        assert not (tmp_path / 'review_bundle').exists()
    finally:
        loaded.close()


@pytest.mark.parametrize('name', ['../x', '/x', 'C:/x', 'a\\b', 'a/../../b', ''])
def test_unsafe_paths(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_zip_traversal_rejected(tmp_path):
    path = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(path, 'x') as handle:
        handle.writestr('../images.json', '[]')
    with pytest.raises(ValueError, match='Unsafe'):
        Bundle(path)


def test_review_versions_never_modify_source(bundle, tmp_path):
    before = digest_tree(bundle.source)
    project = Project(tmp_path / 'project', bundle)
    proposal = 'image_0:sam3_text:0'
    masks = bundle.candidates(proposal)
    first = project.save('image_0', proposal, masks)
    masks[0]['mask'][0, 0] = True
    second = project.save('image_0', proposal, masks)
    assert first['revision'] != second['revision']
    assert (project.folder / first['instances'][0]['mask_file']).exists()
    assert project.load_masks('image_0')[0]['mask'][0, 0]
    project.save('image_0', proposal, [], 'excluded')
    assert project.decision('image_0')['status'] == 'excluded'
    assert digest_tree(bundle.source) == before
    reopened = Project(project.folder, bundle)
    assert reopened.decision('image_0')['status'] == 'excluded'


def test_explicit_negative_and_nonempty_mask_validation(bundle, tmp_path):
    project = Project(tmp_path / 'project', bundle)
    with pytest.raises(ValueError, match='confirmed negative'):
        project.save('image_0', None, [])
    project.save('image_0', None, [], negative=True)
    assert project.decision('image_0')['negative']
    with pytest.raises(ValueError, match='nonempty'):
        project.save('image_1', None, [{'mask': np.zeros((64, 96), bool), 'label': 'part'}])
    with pytest.raises(ValueError, match='another image'):
        project.save('image_0', 'image_1:sam3_text:0', [], negative=True)


def test_project_must_be_separate(bundle):
    with pytest.raises(ValueError, match='outside'):
        Project(bundle.source / 'project', bundle)


def test_deterministic_group_and_duplicate_safe_splits(bundle, tmp_path):
    project = Project(tmp_path / 'project', bundle)
    approve(project)
    project.state['reviews']['image_0']['group'] = 'same_capture'
    project.state['reviews']['image_1']['group'] = 'same_capture'
    bundle.by_image['image_2']['sha256'] = bundle.by_image['image_1']['sha256']
    result = split_reviews(project)
    assert result == split_reviews(project)
    assert result['image_0'] == result['image_1'] == result['image_2']
    assert set(result.values()) == {'train', 'val', 'test'}


def test_insufficient_groups_rejected(bundle, tmp_path):
    project = Project(tmp_path / 'project', bundle)
    approve(project, 2)
    with pytest.raises(ValueError, match='3 independent'):
        split_reviews(project)


def test_export_only_accepted_exact_masks_and_normalized_polygons(bundle, tmp_path):
    project = Project(tmp_path / 'project', bundle)
    approve(project)
    project.save('image_6', None, [], 'excluded')
    before = digest_tree(bundle.source)
    updates = []
    target = export_dataset(project, tmp_path / 'dataset', progress=lambda *args: updates.append(args))
    manifest = json.loads((target / 'manifest.json').read_text())
    assert len(manifest['images']) == 6
    assert updates[-1] == (6, 6)
    assert len(list((target / 'images').rglob('*.png'))) == 6
    for path in (target / 'labels').rglob('*.txt'):
        values = list(map(float, path.read_text().split()))
        assert values[0] == 0 and all(0 <= x <= 1 for x in values[1:])
    mask_path = target / manifest['images'][0]['instances'][0]['mask']
    assert (cv2.imdecode(np.frombuffer(mask_path.read_bytes(), np.uint8), 0) > 0).sum() == 1200
    assert digest_tree(bundle.source) == before
    archive = zip_dataset(target, tmp_path / 'dataset.zip')
    with zipfile.ZipFile(archive) as opened:
        assert opened.testzip() is None
        assert 'data.yaml' in opened.namelist()
    with pytest.raises(FileExistsError):
        export_dataset(project, target)
    with pytest.raises(ValueError, match='outside'):
        zip_dataset(target, target / 'inside.zip')


def test_polygon_loss_is_explicit():
    mask = np.zeros((30, 30), bool)
    mask[3:20, 3:20] = True
    mask[8:12, 8:12] = False
    with pytest.raises(ValueError, match='holes'):
        polygon(mask)
    points, holes = polygon(mask, True)
    assert holes and len(points) >= 3
    mask[25:28, 25:28] = True
    with pytest.raises(ValueError, match='Disconnected'):
        polygon(mask, True)


def test_worker_static_only():
    import ast
    source = (Path(__file__).resolve().parents[2] / 'review_app/worker.py').read_text()
    ast.parse(source)
    assert 'YOLOEPESegTrainer' in source
    assert 'workers=0' in source
    assert "if __name__ == '__main__'" in source


# The Qt tests (window structure, shortcuts, review flow, canvas, augmentation page) live in
# test_review_app_ui.py; the augmentation engine is covered by test_review_app_augment.py.
