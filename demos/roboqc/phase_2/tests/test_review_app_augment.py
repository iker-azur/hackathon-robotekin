"""Augmentation engine, suggestions, class registry and worker mapping (no Qt, no model run)."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from review_app import augment
from review_app.core import Project, export_dataset, polygon
from review_app.suggest import auto_groups, parse_timestamp, seed_classes, suggest_class
from review_app.theme import PALETTE, contrast
from test_review_app import approve, bundle  # noqa: F401  (bundle is a fixture)

ONLY_GEOMETRY = dict(fliplr=0, flipud=0, degrees=0, translate=0, scale=0, shear=0, perspective=0,
                     hsv_h=0, hsv_s=0, hsv_v=0, mosaic=0, copy_paste=0)


def scene(color=(0, 0, 255)):
    image = np.full((120, 160, 3), 90, np.uint8)
    first, second = np.zeros((120, 160), bool), np.zeros((120, 160), bool)
    first[30:80, 20:70] = True
    second[60:100, 100:150] = True
    image[first] = color
    image[second] = color
    return image, [first, second], [0, 1]


def config(**values):
    return augment.clean({**ONLY_GEOMETRY, **values})


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def test_clean_clamps_unknown_and_presets():
    cleaned = augment.clean({'degrees': 999, 'fliplr': -3, 'copy_paste_mode': 'bad', 'zzz': 1, 'close_mosaic': 7.4})
    assert cleaned['degrees'] == 180 and cleaned['fliplr'] == 0 and cleaned['copy_paste_mode'] == 'flip'
    assert 'zzz' not in cleaned and cleaned['close_mosaic'] == 7
    assert augment.preset('Metal parts (default)') == augment.defaults()
    for name in augment.PRESETS:
        assert augment.clean(augment.preset(name)) == augment.preset(name)


def test_train_kwargs_are_exactly_the_ultralytics_subset_and_match_worker_defaults():
    spec = importlib.util.spec_from_file_location('roboqc_worker', Path(__file__).resolve().parents[2] / 'review_app/worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)                   # imports no model library at module level
    kwargs = augment.train_kwargs(None)
    assert kwargs == worker.AUGMENTATION_DEFAULTS
    assert not {'erasing', 'contrast', 'blur', 'noise'} & set(kwargs)
    # a job file written by the previous version still trains the same way
    old = worker.augmentation_kwargs({'epochs': 5, 'degrees': 90, 'brightness': .3})
    assert old['hsv_v'] == .3 and old['degrees'] == 90 and old['close_mosaic'] == 5 and old['hsv_h'] == 0
    source = (Path(__file__).resolve().parents[2] / 'review_app/worker.py').read_text()
    assert '**augmentation_kwargs(config)' in source and 'erasing' not in source


def test_warped_masks_agree_with_their_polygons():
    image, masks, labels = scene()
    for seed in range(8):
        sample = augment.augment_sample(image, masks, labels, config(degrees=40, scale=.3, translate=.1, shear=8),
                                        np.random.default_rng(seed))
        for mask in sample['masks']:
            points, _ = polygon(mask, allow_holes=True)
            raster = np.zeros(mask.shape, np.uint8)
            h, w = mask.shape
            cv2.fillPoly(raster, [np.round(points * [w, h]).astype(np.int32)], 1)
            assert iou(mask, raster > 0) > .95


def test_same_seed_same_sample_different_seed_differs():
    image, masks, labels = scene()
    settings = config(degrees=45, hsv_v=.4, translate=.2)
    a = augment.augment_sample(image, masks, labels, settings, np.random.default_rng((7, 1)))
    b = augment.augment_sample(image, masks, labels, settings, np.random.default_rng((7, 1)))
    c = augment.augment_sample(image, masks, labels, settings, np.random.default_rng((7, 2)))
    assert np.array_equal(a['image'], b['image']) and all(np.array_equal(x, y) for x, y in zip(a['masks'], b['masks']))
    assert not np.array_equal(a['image'], c['image'])


def test_default_keeps_part_colour_hue():
    image, masks, labels = scene(color=(0, 0, 255))
    hue_before = np.median(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 0][masks[0]])
    for seed in range(6):
        sample = augment.augment_sample(image, masks, labels, augment.defaults(), np.random.default_rng(seed))
        hsv = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2HSV)[..., 0]
        # interpolation at the edges may blend with the grey background, so look at the mask core
        core = cv2.erode(sample['masks'][0].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if core.any():
            assert abs(np.median(hsv[core]) - hue_before) <= 2


def test_erasing_only_removes_mask_pixels():
    image, masks, labels = scene()
    before = sum(int(m.sum()) for m in masks)
    areas = []
    for seed in range(20):
        sample = augment.augment_sample(image, masks, labels, config(erasing=1.0), np.random.default_rng(seed))
        areas.append(sum(int(m.sum()) for m in sample['masks']))
    assert max(areas) <= before and min(areas) < before


def test_copy_paste_adds_labelled_instances_without_overlap():
    image = np.full((120, 160, 3), 90, np.uint8)
    left, lower = np.zeros((120, 160), bool), np.zeros((120, 160), bool)
    left[20:50, 10:40] = True                     # mirrored copies land in empty space
    lower[80:110, 20:50] = True
    image[left | lower] = (0, 0, 255)
    masks, labels = [left, lower], [0, 1]
    added = 0
    for seed in range(20):
        sample = augment.augment_sample(image, masks, labels, config(copy_paste=1.0), np.random.default_rng(seed))
        assert len(sample['masks']) == len(sample['labels'])
        added = max(added, len(sample['masks']) - len(masks))
        stacked = np.sum([m.astype(int) for m in sample['masks']], axis=0)
        assert stacked.max() <= 1                      # pasted pixels leave the masks underneath
    assert added >= 1


def test_tidy_keeps_largest_piece_and_drops_tiny():
    mask = np.zeros((60, 60), bool)
    mask[5:30, 5:30] = True
    mask[50:52, 50:52] = True
    tiny = np.zeros((60, 60), bool)
    tiny[0:3, 0:3] = True
    masks, labels, dropped = augment.tidy([mask, tiny], [0, 1])
    assert len(masks) == 1 and labels == [0] and dropped == 1
    assert masks[0].sum() == 625


def test_fit_max_side_shrinks_image_and_masks_together():
    image, masks, _ = scene()
    small, small_masks = augment.fit_max_side(image, masks, 80)
    assert max(small.shape[:2]) == 80 and all(m.shape == small.shape[:2] for m in small_masks)
    same, same_masks = augment.fit_max_side(image, masks, 4000)
    assert same is image and same_masks is masks


def digest(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def test_offline_export_never_touches_val_or_test(bundle, tmp_path):
    project = Project(tmp_path / 'project', bundle)
    approve(project)
    source = export_dataset(project, tmp_path / 'reviewed')
    manifest = json.loads((source / 'manifest.json').read_text())
    result = augment.export_augmented(source, tmp_path / 'augmented', 2, config(degrees=20, translate=.05), seed=3)
    target = tmp_path / 'augmented'
    train = [r for r in manifest['images'] if r['split'] == 'train']
    assert result['train_images'] == len(train)
    assert result['added'] + result['skipped'] == 2 * len(train)
    before, after = digest(source), digest(target)
    for name, value in before.items():
        if name == 'manifest.json':
            continue
        assert after[name] == value                    # every original file, val/test included, is byte-identical
    extra = set(after) - set(before)
    assert extra and all('/train/' in name and '__aug' in name for name in extra)
    out = json.loads((target / 'manifest.json').read_text())
    assert out['schema'] == 'roboqc-augmented-dataset-1' and out['augmentation']['applied_to'] == 'train'
    augmented = [r for r in out['images'] if 'augmented_from' in r]
    assert len(augmented) == result['added'] and all(r['split'] == 'train' for r in augmented)
    assert {r['group'] for r in augmented} <= {r['group'] for r in train}
    for label in (target / 'labels' / 'train').glob('*__aug*.txt'):
        rows = [line.split() for line in label.read_text().splitlines() if line]
        assert rows and all(0 <= float(v) <= 1 for row in rows for v in row[1:])
    with pytest.raises(FileExistsError):
        augment.export_augmented(source, target, 1, augment.defaults())
    with pytest.raises(ValueError, match='reviewed dataset'):
        augment.export_augmented(tmp_path, tmp_path / 'other', 1, augment.defaults())


def test_suggest_class_from_folder_names():
    assert suggest_class('Robotekin/Pieza_con_perno/individual/img_a.png') == 'Pieza_con_perno'
    assert suggest_class('Robotekin/Pieza_sin_perno/grupo/ordenado/b.png') == 'Pieza_sin_perno'
    assert suggest_class('tests/image_test_1.jpg') is None
    assert suggest_class('Robotekin/bolts/x.png', known=['Bolts']) == 'Bolts'
    rows = [{'source_relative_path': p} for p in ('A/Pieza_x/1.png', 'A/Pieza_y/2.png', 'tests/3.png')]
    assert seed_classes(rows) == ['Pieza_x', 'Pieza_y']


def test_timestamps_and_time_window_groups():
    assert parse_timestamp('a/img_20261021_174255_994365_acae.png').second == 55
    assert parse_timestamp('a/photo.png') is None
    rows = [{'image_id': f'i{n}', 'source_relative_path': f'Robotekin/A/img_20261021_1742{n * 2:02d}_000001_x.png'}
            for n in range(25)] + [{'image_id': 'plain', 'source_relative_path': 'Robotekin/A/photo.png'}]
    folder = auto_groups(rows, 0)
    assert set(folder.values()) == {'Robotekin/A'}
    windowed = auto_groups(rows, 20)
    assert windowed['i0'] == 'Robotekin/A#01' and windowed['i24'] != windowed['i0']
    assert len({v for k, v in windowed.items() if k != 'plain'}) == 3      # 48 s span cut into blocks of <= 20 s
    assert windowed['plain'] == 'Robotekin/A'                              # no timestamp keeps the folder group


def test_class_registry_rename_merge_delete_and_old_projects(bundle, tmp_path):
    project = Project(tmp_path / 'project', bundle)
    approve(project)
    assert project.class_names() == ['part']
    assert project.color_of('part').startswith('#')
    revisions = {k: v['revision'] for k, v in project.state['reviews'].items()}
    project.ensure_class('after')
    colour = project.color_of('part')
    assert project.rename_class('part', 'bracket') == 6
    assert project.class_names() == ['bracket', 'after'] and project.color_of('bracket') == colour   # keeps its slot and colour
    project.delete_class('after')
    assert project.class_usage()['bracket'] == 6
    assert all(project.decision(k)['revision'] != revisions[k] for k in revisions)       # new immutable revisions
    assert all(row['label'] == 'bracket' for k in revisions for row in project.load_masks(k))
    with pytest.raises(ValueError, match='used by'):
        project.delete_class('bracket')
    project.ensure_class('clip')
    project.set_class_color('clip', '#123456')
    project.move_class('clip', -1)
    assert project.class_names() == ['clip', 'bracket']
    assert project.rename_class('bracket', 'clip') == 6                                    # merge into an existing class
    assert project.class_names() == ['clip'] and project.color_of('clip') == '#123456'
    project.ensure_class('unused')
    project.delete_class('unused')
    with pytest.raises(ValueError):
        project.rename_class('clip', 'clip')
    # a project written before classes/prefs existed still opens
    state = json.loads((project.folder / 'project.json').read_text())
    state.pop('classes'), state.pop('prefs', None)
    (project.folder / 'project.json').write_text(json.dumps(state))
    reopened = Project(project.folder, bundle)
    assert reopened.class_names() == [] and reopened.pref('method') is None
    reopened.set_pref('method', ['sam3_text', 'metal bracket'])
    assert Project(project.folder, bundle).pref('method') == ['sam3_text', 'metal bracket']
    assert reopened.stats() == {'total': 8, 'accepted': 6, 'excluded': 0, 'unreviewed': 2}


def test_theme_text_contrast_meets_wcag_aa():
    pairs = [('text', 'bg'), ('text', 'surface'), ('muted', 'bg'), ('muted', 'surface'), ('on_primary', 'primary'),
             ('on_primary', 'primary_hover'), ('primary', 'surface'), ('primary', 'primary_soft'),
             ('success', 'success_soft'), ('warn', 'warn_soft'), ('danger', 'danger_soft'), ('danger', 'surface'),
             ('text', 'primary_soft'), ('on_primary', 'tooltip')]
    for foreground, background in pairs:
        assert contrast(PALETTE[foreground], PALETTE[background]) >= 4.5, (foreground, background)
