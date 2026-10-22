"""Only handmade file/image fixtures; never load models or execute notebook cells."""

import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

from roboqc.industrial.review_bundle import image_files, pack_images, extract_images, register_images, archive_bundle, sha256
from roboqc.industrial import review_bundle


def test_recursive_zip_roundtrip_and_duplicate_basenames(tmp_path):
    source = tmp_path / 'data'
    for name,blob in [('a/part.JPG',b'first'),('b/deep/part.png',b'second')]:
        path = source / name; path.parent.mkdir(parents=True); path.write_bytes(blob)
    (source / 'ignored.txt').write_text('not an image')
    archive = tmp_path / 'inputs.zip'
    report = pack_images(source,archive)
    assert report['images'] == 2 and report['sha256'] == sha256(archive)
    restored = extract_images(archive,tmp_path / 'restore')
    assert [p.relative_to(restored).as_posix() for p in image_files(restored)] == ['a/part.JPG','b/deep/part.png']
    assert (restored / 'a/part.JPG').read_bytes() == b'first'
    assert (restored / 'b/deep/part.png').read_bytes() == b'second'
    with pytest.raises(FileExistsError): pack_images(source,archive)
    with pytest.raises(FileExistsError): extract_images(archive,restored)


@pytest.mark.parametrize('name',['../escape.png','/absolute.png','C:/escape.png','nested/../../escape.png','..\\escape.png'])
def test_zip_rejects_unsafe_paths_before_extracting(tmp_path,name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive,'w') as handle: handle.writestr(name,b'bad')
    with pytest.raises(ValueError,match='Unsafe'): extract_images(archive,tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('case',['bytes','count','duplicate','symlink','empty'])
def test_zip_preflight_limits_duplicates_and_symlinks(tmp_path,case):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive,'w') as handle:
        if case == 'empty': handle.writestr('readme.txt','none')
        elif case == 'symlink':
            info = zipfile.ZipInfo('link.png'); info.external_attr = 0o120777 << 16
            handle.writestr(info,'elsewhere')
        else:
            handle.writestr('a.png','abc')
            handle.writestr('A.png' if case == 'duplicate' else 'b.png','abc')
    with pytest.raises(ValueError):
        extract_images(archive,tmp_path / 'out',maximum_bytes=1 if case == 'bytes' else 100,
                       maximum_files=1 if case == 'count' else 10)
    assert not (tmp_path / 'out').exists()


def test_review_registry_originals_ids_invalid_images_and_snapshot(tmp_path,monkeypatch):
    # Local verification has no Pillow; fake only normalization and file saving.
    class FakeImage:
        width,height = 8,6
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def convert(self,mode): assert mode == 'RGB'; return self
        def save(self,path): Path(path).write_bytes(b'normalized-rgb-fixture')
    def open_image(path):
        if Path(path).name == 'broken.png': raise OSError('bad image')
        return FakeImage()
    monkeypatch.setitem(sys.modules,'PIL',types.SimpleNamespace(Image=types.SimpleNamespace(open=open_image),
        ImageOps=types.SimpleNamespace(exif_transpose=lambda image:image)))
    source = tmp_path / 'data'
    for name in ['one/part.JPG','two/part.JPG','broken.png']:
        path = source / name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'original-fixture')
    out = tmp_path / 'bundle'
    rows = register_images(source,out)
    assert len({r['image_id'] for r in rows}) == 3
    assert sum(r['status'] == 'ok' for r in rows) == 2
    assert all((out / r['original']).read_bytes() == b'original-fixture' for r in rows)
    assert all(r['width'] == 8 and r['height'] == 6 for r in rows if r['status'] == 'ok')
    reviews = json.loads((out / 'reviews.json').read_text())
    assert len(reviews) == 3
    assert all(r['status'] == 'unreviewed' and r['instances'] == [] and r['selected_proposal_id'] is None for r in reviews.values())
    report = archive_bundle(out,tmp_path / 'snapshot.zip')
    with zipfile.ZipFile(report['path']) as archive:
        assert archive.testzip() is None
        inventory = json.loads(archive.read('checksums.json'))
        assert inventory['reviews.json'] == sha256(out / 'reviews.json')
    with pytest.raises(FileExistsError): archive_bundle(out,tmp_path / 'snapshot.zip')
    with pytest.raises(ValueError): archive_bundle(out,out / 'recursive.zip')


def test_empty_image_folder(tmp_path):
    with pytest.raises(ValueError,match='No supported images'): image_files(tmp_path)


def saved_settings():
    return {'methods':['sam2_auto','sam3_text'],'prompts':['bracket','clip'],'max_images':None,
            'minimum_area':100,'config':{'cache':'/content/cache'},
            'runtime_sha256':{'roboqc/industrial/sam3.py':'inference-v1',
                              'roboqc/industrial/review_bundle.py':'export-v1'}}


def make_saved_bundle(root,name,successes=0,errors=0,settings=None,complete=True):
    folder = root / name; folder.mkdir(parents=True)
    settings = settings or saved_settings()
    (folder / 'settings.json').write_text(json.dumps(settings))
    samples = [{'image_id':'one','status':'ok'},{'image_id':'two','status':'ok'}]
    (folder / 'images.json').write_text(json.dumps(samples))
    if complete: (folder / 'registration_complete.json').write_text('{"images":2}')
    ids = ['one:sam2_auto:0','two:sam2_auto:0','one:sam3_text:0','one:sam3_text:1','two:sam3_text:0','two:sam3_text:1']
    records = [{'proposal_id':p,'status':'ok'} for p in ids[:successes]]
    records.extend({'proposal_id':p,'status':'error'} for p in ids[successes:successes+errors])
    (folder / 'proposals.json').write_text(json.dumps(records))
    return folder


def test_auto_discovery_ranks_unique_successes_not_failures_or_file_count(tmp_path):
    first = make_saved_bundle(tmp_path,'run1/review_bundle_first',successes=2,errors=4)
    best = make_saved_bundle(tmp_path,'run2/nested/review_bundle_best',successes=4)
    # Stray outputs and duplicate index rows must not increase completed count.
    rows = json.loads((first / 'proposals.json').read_text())
    rows += [rows[0]]*10 + [{'proposal_id':'unknown:sam2_auto:0','status':'ok'}]
    (first / 'proposals.json').write_text(json.dumps(rows))
    for i in range(12): (first / f'error_{i}.txt').write_text('not successful')
    ignored = saved_settings(); ignored['minimum_area'] = 999
    make_saved_bundle(tmp_path,'run3/review_bundle_incompatible',successes=6,settings=ignored)
    make_saved_bundle(tmp_path,'run4/review_bundle_incomplete',successes=6,complete=False)
    make_saved_bundle(tmp_path,'exports/review_bundle_export_copy',successes=6)
    # Never traverse proposal trees while looking for other bundles.
    make_saved_bundle(first,'proposals/review_bundle_not_a_run',successes=6)
    events = []
    rows = review_bundle.discover_bundles(tmp_path,saved_settings(),progress=lambda *args:events.append(args))
    compatible = [r for r in rows if r['compatible']]
    assert compatible[0]['path'] == str(best)
    assert compatible[0]['successful'] == 4 and compatible[0]['remaining'] == 2
    assert next(r for r in rows if r['path'] == str(first))['successful'] == 2
    assert len(rows) == 4 and events


def test_old_run_allows_export_helper_change_but_not_inference_or_config_change():
    old,new = saved_settings(),saved_settings()
    new['runtime_sha256']['roboqc/industrial/review_bundle.py'] = 'export-v2'
    assert review_bundle.settings_compatible(old,new)
    new['runtime_sha256']['roboqc/industrial/sam3.py'] = 'inference-v2'
    assert not review_bundle.settings_compatible(old,new)
    new = saved_settings(); new['minimum_area'] = 99
    assert not review_bundle.settings_compatible(old,new)


def test_discovery_reports_corrupt_index_without_overwriting_it(tmp_path):
    path = make_saved_bundle(tmp_path,'review_bundle_bad')
    (path / 'proposals.json').write_text('broken-json')
    rows = review_bundle.discover_bundles(tmp_path,saved_settings())
    assert len(rows) == 1 and not rows[0]['compatible'] and rows[0]['error']
    assert (path / 'proposals.json').read_text() == 'broken-json'


def test_streaming_archive_reads_sources_once_and_reports_each_stage(tmp_path,monkeypatch):
    folder = tmp_path / 'bundle'; folder.mkdir()
    (folder / 'mask.png').write_bytes(b'png-fixture' * 4000)
    (folder / 'reviews.json').write_text('{"one":{"status":"accepted"}}')
    (folder / 'checksums.json').write_text('stale-checksum-inventory')
    read_paths = []
    original_open = Path.open
    def tracked_open(path,*args,**kwargs):
        if args and args[0] == 'rb' and path.parent == folder: read_paths.append(path.name)
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',tracked_open)
    events = []
    report = archive_bundle(folder,tmp_path / 'drive/results.zip',
                            staging_directory=tmp_path / 'local',progress=lambda *args:events.append(args))
    assert sorted(read_paths) == ['mask.png','reviews.json']
    assert report['sha256'] == sha256(report['path'])
    assert report['verified'] == 'local_crc_and_sha256; destination_size_checked'
    phases = {phase for phase,_,_ in events}
    assert phases == {'Scan files','Archive + hash','Verify local ZIP','Copy ZIP to Drive'}
    for phase in phases:
        updates = [(done,total) for name,done,total in events if name == phase]
        assert updates[-1][0] == updates[-1][1]
        assert all(a[0] <= b[0] for a,b in zip(updates,updates[1:]))
    with zipfile.ZipFile(report['path']) as archive:
        assert archive.read('mask.png') == b'png-fixture' * 4000
        assert json.loads(archive.read('checksums.json')) == json.loads((folder / 'checksums.json').read_text())
    assert not list((tmp_path / 'local').iterdir())


def test_streaming_archive_refuses_insufficient_local_space(tmp_path,monkeypatch):
    folder = tmp_path / 'bundle'; folder.mkdir(); (folder / 'mask.png').write_bytes(b'fixture')
    monkeypatch.setattr(review_bundle.shutil,'disk_usage',lambda path:types.SimpleNamespace(free=0))
    with pytest.raises(OSError,match='local disk space'):
        archive_bundle(folder,tmp_path / 'out.zip',staging_directory=tmp_path / 'local')
    assert not (tmp_path / 'out.zip').exists()
