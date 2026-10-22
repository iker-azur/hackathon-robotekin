"""Fixtures and document construction only; no notebook or model execution."""

import ast
import json
from pathlib import Path
import re
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2] / 'colab'


def selector():
    return runpy.run_path(str(ROOT / 'image_inputs.py'))['select_image_paths']


def test_recursive_drive_folder_and_single_image(tmp_path):
    folder = tmp_path / 'photos'
    (folder / 'nested').mkdir(parents=True)
    (folder / 'nested/part.PNG').write_bytes(b'fixture')
    (folder / 'notes.txt').write_text('not a photo')
    function = selector()
    assert function(folder, tmp_path) == [folder / 'nested/part.PNG']
    assert function(folder / 'nested/part.PNG', tmp_path) == [folder / 'nested/part.PNG']


def test_uploads_create_fresh_folders_without_replacing_previous_files(tmp_path):
    function = selector()
    first = function('', tmp_path, lambda: {'part.png': b'first fixture'})
    second = function('', tmp_path, lambda: {'part.png': b'second fixture'})
    assert first[0] != second[0]
    assert first[0].read_bytes() == b'first fixture'
    assert second[0].read_bytes() == b'second fixture'


@pytest.mark.parametrize('name', ['../part.png', 'nested/part.png', 'C:\\part.png', 'images.zip'])
def test_upload_rejects_unsafe_paths_and_non_images(tmp_path, name):
    with pytest.raises(ValueError):
        selector()('', tmp_path, lambda: {name: b'fixture'})
    assert list(tmp_path.iterdir()) == []


def test_cancel_missing_and_empty_inputs(tmp_path):
    with pytest.raises(ValueError, match='Upload'):
        selector()('', tmp_path, lambda: {})
    with pytest.raises(FileNotFoundError):
        selector()(tmp_path / 'missing', tmp_path)
    with pytest.raises(ValueError, match='No supported'):
        selector()(tmp_path, tmp_path)


def assert_public_notebook(notebook):
    assert not notebook['metadata'].get('widgets')
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        assert not re.search(r'[A-Za-z0-9+/]{2000,}={0,2}', source), 'Embedded binary payload'
        assert not re.search(r'Robotekin2026/(?:inputs/[a-z0-9]+_data_|\d{8}_)', source), 'Personal run path'
        if cell['cell_type'] == 'code':
            ast.parse(source)
            assert cell['execution_count'] is None and not cell['outputs']


def test_public_notebooks_do_not_publish_photos_outputs_or_personal_paths():
    for name in ['04b_try_sam2_stamped_parts.ipynb', '04c_compare_pretrained_segmentation.ipynb',
                 '04d_try_sam3_pretrained.ipynb', '04e_export_segmentation_review.ipynb']:
        assert_public_notebook(json.loads((ROOT / name).read_text(encoding='utf-8')))


def test_generators_cannot_reintroduce_local_photo_payloads(monkeypatch):
    module = runpy.run_path(str(ROOT / 'build_notebooks.py'))
    generated = {}
    def capture(name, cells):
        generated[name] = cells
    def forbidden(*args, **kwargs):
        raise AssertionError('Public notebook generators must not read photo bytes')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    for name in ['sam_notebook', 'comparison_notebook', 'sam3_notebook']:
        monkeypatch.setitem(module[name].__globals__, 'write', capture)
        module[name]()  # Constructs documents in memory, never executes notebook cells.
    assert len(generated) == 3
    for name, cells in generated.items():
        assert_public_notebook({'metadata': {}, 'cells': cells})
        saved = json.loads((ROOT / name).read_text(encoding='utf-8'))
        assert cells == saved['cells']
