"""Fixture-only tests of the pasteable exporter, never notebook/model execution."""

import ast
import base64
import json
from pathlib import Path
import runpy
import zipfile

import pytest


def exporter():
    script = Path(__file__).resolve().parents[2] / 'colab/export_last_cell.py'
    return runpy.run_path(str(script))['export_saved_bundle']


def make_bundle(tmp_path):
    bundle = tmp_path / 'review_bundle_fixture'; bundle.mkdir()
    (bundle / 'images.json').write_text('[]')
    (bundle / 'proposals.json').write_text('[{"proposal_id":"one:sam3_text:0","status":"ok"}]')
    nested = bundle / 'images/one'; nested.mkdir(parents=True)
    (nested / 'original.jpg').write_bytes(b'fixture-original' * 50)
    masks = bundle / 'proposals/one/masks'; masks.mkdir(parents=True)
    (masks / 'mask.png').write_bytes(b'fixture-mask' * 100)
    return bundle


def test_standalone_export_exact_bytes_no_duplicates_and_completed_zip_reuse(tmp_path):
    bundle = make_bundle(tmp_path)
    export = exporter()
    final = export(bundle, tmp_path / 'exports', batch_size=1)
    with zipfile.ZipFile(final) as archive:
        expected = {p.relative_to(bundle).as_posix(): p.read_bytes() for p in bundle.rglob('*') if p.is_file()}
        assert len(archive.infolist()) == len(expected)
        assert {name: archive.read(name) for name in archive.namelist()} == expected
        assert archive.testzip() is None
    assert export(bundle, tmp_path / 'exports') == final
    assert len(list((tmp_path / 'exports').glob('*.zip'))) == 1


def test_resume_recovers_committed_zip_after_abrupt_garbage_append(tmp_path,monkeypatch):
    bundle = make_bundle(tmp_path)
    export = exporter()
    original_open = Path.open
    def interrupted_open(path,*args,**kwargs):
        if path.name == 'proposals.json' and path.parent == bundle and args and args[0] == 'rb':
            raise KeyboardInterrupt('simulate disconnect during second batch')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',interrupted_open)
    with pytest.raises(KeyboardInterrupt): export(bundle,tmp_path / 'exports',batch_size=2)
    state_path = next((tmp_path / 'exports').glob('*.export.json'))
    state = json.loads(state_path.read_text())
    work = Path(state['archive'])
    # Simulate an unclosed append after a runtime/power interruption: no usable EOCD.
    with original_open(work,'r+b') as handle:
        handle.seek(state['directory_offset']); handle.write(b'interrupted-new-file' * 10000); handle.truncate()
    with pytest.raises(zipfile.BadZipFile): zipfile.ZipFile(work)
    read_sources = []
    def record_sources(path,*args,**kwargs):
        if args and args[0] == 'rb' and path.is_relative_to(bundle): read_sources.append(path.relative_to(bundle).as_posix())
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',record_sources)
    final = export(bundle,tmp_path / 'exports',batch_size=2)
    # The first committed batch (images.json and original.jpg) is not reread.
    assert 'images.json' not in read_sources
    assert 'images/one/original.jpg' not in read_sources
    assert set(read_sources) == {'proposals.json','proposals/one/masks/mask.png'}
    with zipfile.ZipFile(final) as archive:
        assert archive.testzip() is None and len(archive.infolist()) == 4


def test_old_invalid_archive_is_preserved_and_not_mistaken_for_finished_export(tmp_path):
    bundle = make_bundle(tmp_path)
    exports = tmp_path / 'exports'; exports.mkdir()
    damaged = exports / (bundle.name + '_old.zip'); damaged.write_bytes(b'partial-old-archive')
    final = exporter()(bundle,exports)
    assert final != damaged and damaged.read_bytes() == b'partial-old-archive'
    with zipfile.ZipFile(final) as archive: assert archive.testzip() is None


def test_export_reports_missing_predictions_without_running_inference(tmp_path,capsys):
    bundle = make_bundle(tmp_path)
    (bundle / 'images.json').write_text('[{"image_id":"one","status":"ok"}]')
    (bundle / 'settings.json').write_text(json.dumps({'methods':['sam2_auto','sam3_text'],
        'prompts':['bracket','clip'],'max_images':None}))
    exporter()(bundle,tmp_path / 'exports')
    assert 'Expected: 3 | missing/failed: 2' in capsys.readouterr().out


def test_reviews_all_legacy_attempts_and_resumes_most_payload_bytes(tmp_path,monkeypatch,capsys):
    bundle = make_bundle(tmp_path)
    exports = tmp_path / 'exports'; exports.mkdir()
    small = exports / (bundle.name + '_small.zip')
    advanced = exports / (bundle.name + '_advanced.zip')
    with zipfile.ZipFile(small,'w') as archive:
        for name in ['images.json','proposals.json']: archive.write(bundle / name,name)
    with zipfile.ZipFile(advanced,'w') as archive:
        archive.write(bundle / 'proposals/one/masks/mask.png','proposals/one/masks/mask.png')
    originals = {p:p.read_bytes() for p in [small,advanced]}
    original_open = Path.open; reads = []
    def recorded(path,*args,**kwargs):
        if args and args[0]=='rb' and path.is_relative_to(bundle): reads.append(path.relative_to(bundle).as_posix())
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',recorded)
    final = exporter()(bundle,exports)
    output = capsys.readouterr().out
    assert f'Most export progress: {advanced}' in output
    assert str(small) in output and str(advanced) in output
    assert 'proposals/one/masks/mask.png' not in reads
    assert 'images/one/original.jpg' in reads
    assert all(p.read_bytes()==blob for p,blob in originals.items())
    with zipfile.ZipFile(final) as archive: assert archive.testzip() is None and len(archive.infolist())==4


def test_finds_old_local_staging_and_drive_partial_copy(tmp_path,capsys):
    bundle = make_bundle(tmp_path)
    local = tmp_path / 'local/review_export_old'; local.mkdir(parents=True)
    with zipfile.ZipFile(local / 'results.zip','w') as archive:
        for p in bundle.rglob('*'):
            if p.is_file(): archive.write(p,p.relative_to(bundle).as_posix())
    exports = tmp_path / 'exports'; exports.mkdir()
    partial = exports / (bundle.name + '_previous.zip.partial_123')
    partial.write_bytes(b'bad incomplete bytes')
    final = exporter()(bundle,exports,search_dirs=[local.parent])
    output = capsys.readouterr().out
    assert f'Most export progress: {local / "results.zip"}' in output
    assert str(partial) in output and 'no usable checkpoint' in output
    assert partial.read_bytes()==b'bad incomplete bytes'
    with zipfile.ZipFile(final) as archive: assert archive.testzip() is None


def test_selects_most_advanced_checkpoint_without_modifying_other_attempt(tmp_path,monkeypatch,capsys):
    bundle = make_bundle(tmp_path)
    export = exporter(); original_open = Path.open
    def interrupt(path,*args,**kwargs):
        if path==bundle / 'proposals.json' and args and args[0]=='rb': raise KeyboardInterrupt()
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',interrupt)
    with pytest.raises(KeyboardInterrupt): export(bundle,tmp_path / 'exports',batch_size=2)
    monkeypatch.setattr(Path,'open',original_open)
    state_path = next((tmp_path / 'exports').glob('*.export.json'))
    advanced = json.loads(state_path.read_text())
    empty = tmp_path / 'exports' / (bundle.name + '_resume_empty.work.zip')
    with zipfile.ZipFile(empty,'x') as archive: pass
    less = {**advanced,'archive':str(empty),'directory_offset':0,'tail':base64.b64encode(empty.read_bytes()).decode()}
    less_path = tmp_path / 'exports' / (bundle.name + '_less.export.json')
    less_path.write_text(json.dumps(less)); unchanged = empty.read_bytes()
    final = export(bundle,tmp_path / 'exports')
    output = capsys.readouterr().out
    assert f'Most export progress: {state_path}' in output and str(less_path) in output
    assert empty.read_bytes()==unchanged and json.loads(less_path.read_text())==less
    with zipfile.ZipFile(final) as archive: assert archive.testzip() is None


def test_replacement_cell_does_not_require_export_local_dir_variable():
    script = Path(__file__).resolve().parents[2] / 'colab/export_last_cell.py'
    tree = ast.parse(script.read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Name) and n.id=='EXPORT_LOCAL_DIR' for n in ast.walk(tree))
    assert "globals().get('EXPORT_LOCAL_DIR','/content/roboqc_review_exports')" in script.read_text(encoding='utf-8')


def test_inventory_uses_cached_directory_entries_and_resumes_listing(tmp_path,monkeypatch):
    bundle = make_bundle(tmp_path)
    collect = exporter().__globals__['list_export_files']
    import os
    original = os.scandir; scanned = []
    def interrupted(path):
        scanned.append(Path(path))
        if Path(path)==bundle / 'proposals/one/masks': raise KeyboardInterrupt('fixture listing interruption')
        return original(path)
    monkeypatch.setattr(os,'scandir',interrupted)
    with pytest.raises(KeyboardInterrupt): collect(bundle,tmp_path / 'cache',workers=1)
    assert list((tmp_path / 'cache').glob('*.inventory.jsonl'))
    scanned.clear()
    def tracked(path): scanned.append(Path(path)); return original(path)
    monkeypatch.setattr(os,'scandir',tracked)
    files = collect(bundle,tmp_path / 'cache',workers=1)
    assert len(files)==4
    assert bundle / 'images/one' not in scanned
    assert bundle / 'proposals/one/masks' in scanned
    scanned.clear()
    assert collect(bundle,tmp_path / 'cache',workers=4)==files
    assert scanned==[bundle]  # Root manifest guard only; no nested re-enumeration.


def test_inventory_invalidates_changed_directories_and_root_manifests(tmp_path):
    import os
    bundle = make_bundle(tmp_path)
    collect = exporter().__globals__['list_export_files']
    first = collect(bundle,tmp_path / 'cache')
    added = bundle / 'proposals/one/masks/another.png'; added.write_bytes(b'new fixture')
    parent = added.parent.stat()
    os.utime(added.parent,ns=(parent.st_atime_ns,parent.st_mtime_ns+1000000000))
    second = collect(bundle,tmp_path / 'cache')
    assert len(second)==len(first)+1 and 'proposals/one/masks/another.png' in second
    (bundle / 'reviews.json').write_text('{"review":"updated"}')
    third = collect(bundle,tmp_path / 'cache')
    assert 'reviews.json' in third and len(list((tmp_path / 'cache').glob('*.inventory.jsonl')))==2


def test_inventory_explicit_refresh_detects_in_place_mask_edits(tmp_path):
    bundle = make_bundle(tmp_path)
    collect = exporter().__globals__['list_export_files']
    first = collect(bundle,tmp_path / 'cache')
    mask = bundle / 'proposals/one/masks/mask.png'; mask.write_bytes(b'changed mask fixture')
    refreshed = collect(bundle,tmp_path / 'cache',refresh=True)
    assert refreshed['proposals/one/masks/mask.png']!=first['proposals/one/masks/mask.png']
    assert len(list((tmp_path / 'cache').glob('*.inventory.jsonl')))==1
    assert collect(bundle,tmp_path / 'cache')==refreshed


def test_interrupted_forced_refresh_cannot_reuse_old_mask_metadata(tmp_path,monkeypatch):
    import os
    bundle=make_bundle(tmp_path); collect=exporter().__globals__['list_export_files']
    collect(bundle,tmp_path / 'cache')
    mask=bundle / 'proposals/one/masks/mask.png'; mask.write_bytes(b'edited fixture')
    original=os.scandir
    def interrupt(path):
        if Path(path)==mask.parent: raise KeyboardInterrupt('refresh interrupted')
        return original(path)
    monkeypatch.setattr(os,'scandir',interrupt)
    with pytest.raises(KeyboardInterrupt): collect(bundle,tmp_path / 'cache',workers=1,refresh=True)
    monkeypatch.setattr(os,'scandir',original)
    resumed=collect(bundle,tmp_path / 'cache',workers=1)
    assert resumed['proposals/one/masks/mask.png'][0]==len(b'edited fixture')


def test_inventory_ignores_torn_journal_line_and_rejects_symlinks(tmp_path):
    bundle = make_bundle(tmp_path)
    collect = exporter().__globals__['list_export_files']
    files = collect(bundle,tmp_path / 'cache')
    journal = next((tmp_path / 'cache').glob('*.inventory.jsonl'))
    with journal.open('a') as handle: handle.write('\n{"unfinished":')
    assert collect(bundle,tmp_path / 'cache')==files
    # No platform symlink privilege needed: a fake DirEntry stat marks a link.
    import os,stat,types
    original = os.scandir
    class Entries:
        def __enter__(self): return iter([types.SimpleNamespace(name='outside',stat=lambda **kw:types.SimpleNamespace(st_mode=stat.S_IFLNK))])
        def __exit__(self,*args): pass
    from unittest.mock import patch
    with patch.object(os,'scandir',lambda path:Entries() if Path(path)==bundle else original(path)):
        with pytest.raises(ValueError,match='symlink'): collect(bundle,tmp_path / 'cache')


def test_inventory_metadata_latency_benchmark(tmp_path,monkeypatch,capsys):
    """Handmade files and simulated metadata latency; NOT a Google Drive benchmark."""
    import os,time
    bundle = make_bundle(tmp_path)
    for i in range(8):
        folder = bundle / f'fixture_{i}'; folder.mkdir()
        for j in range(25): (folder / f'file_{j}.bin').write_bytes(b'fixture')
    collect = exporter().__globals__['list_export_files']
    original_stat,original_scandir = os.stat,os.scandir
    calls = {'path':0,'entry':0}
    def slow_stat(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).is_relative_to(bundle):
            calls['path']+=1; time.sleep(.001)
        return original_stat(path,*args,**kwargs)
    class Entry:
        def __init__(self,entry): self.entry=entry; self.name=entry.name
        def stat(self,**kwargs): calls['entry']+=1; time.sleep(.001); return self.entry.stat(**kwargs)
    class Entries:
        def __init__(self,path): self.handle=original_scandir(path)
        def __enter__(self): return (Entry(entry) for entry in self.handle)
        def __exit__(self,*args): self.handle.close()
    monkeypatch.setattr(os,'stat',slow_stat)
    # Baseline is the original listing loop, unchanged.
    start=time.perf_counter(); baseline={}
    for path in bundle.rglob('*'):
        if path.is_file():
            assert path.resolve().is_relative_to(bundle)
            info=path.stat(); baseline[path.relative_to(bundle).as_posix()]=[info.st_size,info.st_mtime_ns]
    old_time=time.perf_counter()-start; old_calls=calls['path']
    monkeypatch.setattr(os,'scandir',lambda path:Entries(path))
    calls.update(path=0,entry=0)
    start=time.perf_counter(); result=collect(bundle,tmp_path / 'cache',workers=4)
    cold=time.perf_counter()-start; cold_calls=sum(calls.values())
    calls.update(path=0,entry=0)
    start=time.perf_counter(); cached=collect(bundle,tmp_path / 'cache',workers=4)
    warm=time.perf_counter()-start; warm_calls=sum(calls.values())
    assert result==baseline==cached and cold_calls<old_calls and warm_calls<cold_calls
    with capsys.disabled():
        print(f'Fixture metadata benchmark (1ms simulated latency): old={old_time:.3f}s/{old_calls} calls; '
              f'new={cold:.3f}s/{cold_calls} calls; cached={warm:.3f}s/{warm_calls} calls. NOT Drive throughput.')


def test_standalone_exporter_has_no_model_imports_or_inference_calls():
    script = Path(__file__).resolve().parents[2] / 'colab/export_last_cell.py'
    tree = ast.parse(script.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert not node.module.startswith(('roboqc','torch','transformers','ultralytics'))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'predict','generate','fit','train','from_pretrained'}
