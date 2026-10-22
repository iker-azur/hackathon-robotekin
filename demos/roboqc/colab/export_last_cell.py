"""Paste this whole file into 04e's LAST cell. No model imports or inference."""

import base64
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile
from tqdm.auto import tqdm


def list_export_files(bundle, cache_dir, workers=4, refresh=False):
    """One stat per entry, parallel folders, resumable inventory of FROZEN outputs.

    Directory mtimes catch additions/removals; root manifests guard normal edits.
    Use refresh=True after manually editing files in-place, without updating manifests.
    """
    bundle,cache_dir = Path(bundle).resolve(),Path(cache_dir).resolve()
    if not isinstance(workers,int) or not 1 <= workers <= 16:
        raise ValueError('Listing workers must be an integer from 1 to 16.')
    if cache_dir.is_relative_to(bundle): raise ValueError('Inventory cache must be outside the bundle.')
    cache_dir.mkdir(parents=True,exist_ok=True)

    def scan(relative,cached=None):
        folder = bundle / relative
        before = os.stat(folder,follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode): raise ValueError('Directory symlink or unsupported source folder.')
        if cached and cached['mtime_ns']==before.st_mtime_ns: return cached,True
        files,children = {},[]
        with os.scandir(folder) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)  # DirEntry: one metadata lookup, no resolve per file.
                if stat.S_ISLNK(info.st_mode): raise ValueError('Source symlinks are not supported in the export inventory.')
                if stat.S_ISDIR(info.st_mode): children.append(entry.name)
                elif stat.S_ISREG(info.st_mode): files[entry.name]=[info.st_size,info.st_mtime_ns]
        after = os.stat(folder,follow_symlinks=False)
        if before.st_mtime_ns!=after.st_mtime_ns:
            raise RuntimeError('Source folder changed while listing. Stop concurrent inference/review writes.')
        return {'directory':relative,'mtime_ns':after.st_mtime_ns,'files':files,'directories':sorted(children)},False

    root,_ = scan('')  # Cheap root-manifest guard, even on cached runs.
    signature = hashlib.sha256(json.dumps([str(bundle),root['files']],sort_keys=True).encode()).hexdigest()[:16]
    journal = cache_dir / (bundle.name + '_' + signature + '.inventory.jsonl')
    cache = {}; ignored=0
    if journal.exists() and not refresh:
        with journal.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    if not isinstance(row,dict): raise ValueError('Invalid inventory record')
                    if row.get('reset_inventory') is True:
                        cache.clear()
                        continue
                    if not isinstance(row.get('files'),dict) or not isinstance(row.get('directories'),list):
                        raise ValueError('Invalid inventory record')
                    relative = PurePosixPath(row['directory'])
                    names = [*row['files'],*row['directories']]
                    if (relative.is_absolute() or '..' in relative.parts or '\\' in row['directory']
                            or any(':' in part for part in relative.parts)
                            or any(not n or n in {'.','..'} or '/' in n or '\\' in n for n in names)):
                        raise ValueError('Unsafe inventory path')
                    if not isinstance(row['mtime_ns'],int): raise ValueError('Invalid directory timestamp')
                    cache[row['directory']]=row
                except (ValueError,KeyError,TypeError): ignored+=1
    if ignored: print('Ignored incomplete inventory records:',ignored,'; affected folders will be rescanned.')
    print('Inventory cache:',journal)
    print('Outputs must stay unchanged; set REFRESH_EXPORT_INVENTORY=True after manual mask edits.')
    result={}; reused=0; fresh=0; queue=deque(); running={}
    # Leading newlines separate new records from a torn last write after a disconnect.
    with journal.open('a',encoding='utf-8') as handle, tqdm(desc=f'List saved files ({workers} workers)',unit='file') as bar:
        if refresh:
            # A durable reset lets the next run resume even an interrupted forced refresh.
            handle.write('\n' + json.dumps({'reset_inventory':True}))
            handle.flush()
        def accept(row,from_cache):
            nonlocal reused,fresh
            relative=row['directory']
            for name,metadata in row['files'].items():
                result[(PurePosixPath(relative) / name).as_posix()]=metadata
            queue.extend((PurePosixPath(relative) / name).as_posix() for name in row['directories'])
            if from_cache: reused+=1
            else:
                fresh+=1
                handle.write('\n' + json.dumps(row,ensure_ascii=False))
                if fresh % 16==0: handle.flush()
            bar.update(len(row['files']))
            bar.set_postfix(cached_dirs=reused,scanned_dirs=fresh,queued_dirs=len(queue)+len(running))
        accept(root,cache.get('')==root)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while queue or running:
                while queue and len(running)<workers:
                    relative=queue.popleft()
                    running[pool.submit(scan,relative,cache.get(relative))]=relative
                finished,_ = wait(running,return_when=FIRST_COMPLETED)
                for future in finished:
                    running.pop(future)
                    row,from_cache=future.result()
                    accept(row,from_cache)
    print('Inventory:',len(result),'files | reused directories:',reused,'| scanned directories:',fresh)
    return result


def export_saved_bundle(bundle, export_dir, batch_size=50, search_dirs=(), listing_workers=4, refresh_inventory=False):
    bundle, export_dir = Path(bundle).resolve(), Path(export_dir).resolve()
    if not (bundle / 'images.json').is_file():
        raise ValueError('Set BUNDLE_TO_EXPORT to your saved review_bundle folder.')
    if export_dir.is_relative_to(bundle):
        raise ValueError('Export folder must be outside the input bundle.')
    export_dir.mkdir(parents=True, exist_ok=True)
    records_path = bundle / 'proposals.json'
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    completed = {r['proposal_id'] for r in records if r['status'] == 'ok'}
    print('Saved successful proposals:', len(completed))
    if (bundle / 'settings.json').exists():
        settings = json.loads((bundle / 'settings.json').read_text())
        images = [s for s in json.loads((bundle / 'images.json').read_text()) if s['status'] == 'ok']
        if settings['max_images'] is not None: images = images[:settings['max_images']]
        expected = {f"{s['image_id']}:{m}:{i}" for s in images for m in settings['methods']
                    for i in (range(len(settings['prompts'])) if m == 'sam3_text' else [0])}
        print('Expected:', len(expected), '| missing/failed:', len(expected-completed))
    print('Export only: no inference or processing of images will run.')

    files = list_export_files(bundle,export_dir,workers=listing_workers,refresh=refresh_inventory)
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    def checked_members(target):
        # Read only the ZIP directory. Payload CRC/SHA is NOT re-verified here.
        with zipfile.ZipFile(target) as archive:
            members = archive.infolist()
        result = {m.filename: m.file_size for m in members}
        if len(result) != len(members) or any(n not in files or size != files[n][0] for n, size in result.items()):
            raise ValueError('Archive entries differ from this snapshot (names/sizes/duplicates).')
        return result

    attempts = set()
    extra_roots = {Path(p).resolve() for p in search_dirs}
    roots = {export_dir, bundle.parent, *extra_roots}
    with tqdm(desc='Find export attempts', unit='attempt') as bar:
        for root in sorted(roots):
            for directory, children, names in os.walk(root):
                children[:] = [n for n in children if not n.startswith('review_bundle_')
                               and n not in {'images', 'proposals', 'runtime', 'inputs'}]
                for name in names:
                    if ((name.startswith(bundle.name) and ('.zip' in name or name.endswith('.export.json')))
                            or (root in extra_roots and name == 'results.zip')):
                        path = Path(directory) / name
                        if path not in attempts: attempts.add(path); bar.update(1)
    rows = []
    covered = set()
    # Checkpoint tails can be inspected without touching an interrupted work ZIP.
    for path in tqdm(sorted(p for p in attempts if p.name.endswith('.export.json')), desc='Review checkpoints', unit='attempt'):
        row = {'path':str(path), 'kind':'checkpoint', 'bytes':0, 'files':0, 'usable':False}
        try:
            state = json.loads(path.read_text())
            work = Path(state['archive']).resolve()
            covered.add(work)
            if (state['bundle'] != str(bundle) or state['fingerprint'] != fingerprint
                    or work.parent != export_dir or not work.name.startswith(bundle.name + '_resume_')
                    or not work.name.endswith('.work.zip')):
                raise ValueError('Checkpoint is for a different snapshot or unsupported work path.')
            offset = state['directory_offset']
            if not isinstance(offset,int) or offset < 0 or work.stat().st_size < offset:
                raise ValueError('Committed ZIP prefix is missing.')
            members = checked_members(io.BytesIO(base64.b64decode(state['tail'],validate=True)))
            row.update(usable=True, bytes=sum(members.values()), files=len(members), state=state)
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            row['reason'] = str(exc)
        rows.append(row)
    for path in tqdm(sorted(p for p in attempts if not p.name.endswith('.export.json') and p.resolve() not in covered), desc='Review ZIP attempts', unit='attempt'):
        row = {'path':str(path), 'kind':'archive', 'bytes':0, 'files':0, 'usable':False}
        try:
            members = checked_members(path)
            row.update(usable=True, bytes=sum(members.values()), files=len(members))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            row['reason'] = str(exc) + ' (preserved; no usable checkpoint)'
        rows.append(row)
    total = sum(size for size, _ in files.values())
    rows.sort(key=lambda r:(r['usable'],r['bytes'],r['files'],
                           r['kind']=='archive' and r['path'].endswith('.zip') and not r['path'].endswith('.work.zip')), reverse=True)
    print('Export attempts (ZIP directory checks only; not full payload verification):')
    for row in rows:
        pct = 100*row['bytes']/total if total else 0
        print(f"{pct:6.1f}% | {row['files']}/{len(files)} files | {row['kind']} | "
              f"{'recoverable' if row['usable'] else row.get('reason','unusable')} | {row['path']}")
    best = next((row for row in rows if row['usable']),None)
    if best:
        print('Most export progress:',best['path'])
        candidate = Path(best['path'])
        if best['kind']=='archive' and best['files']==len(files) and candidate.suffix=='.zip' and not candidate.name.endswith('.work.zip'):
            print('Existing complete ZIP (file list/sizes checked):',candidate)
            return candidate

    state_path = (Path(best['path']) if best and best['kind']=='checkpoint' else
                  export_dir / (bundle.name + '_' + fingerprint[:16] + '_' + uuid.uuid4().hex[:8] + '.export.json'))

    def save_checkpoint(work, directory_offset):
        # Back up the committed central directory BEFORE the next append overwrites it.
        # Restoring this tail discards only uncommitted bytes in this owned work ZIP.
        with work.open('rb') as handle:
            handle.seek(directory_offset)
            tail = base64.b64encode(handle.read()).decode('ascii')
        state = {'bundle': str(bundle), 'fingerprint': fingerprint, 'archive': str(work),
                 'directory_offset': directory_offset, 'tail': tail}
        temporary = state_path.with_name(state_path.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(state), encoding='utf-8')
        temporary.replace(state_path)

    if best and best['kind']=='checkpoint':
        state = best['state']
        work = Path(state['archive']).resolve()
        if (state['bundle'] != str(bundle) or state['fingerprint'] != fingerprint
                or work.parent != export_dir or not work.name.startswith(bundle.name + '_resume_')
                or not work.name.endswith('.work.zip')):
            raise ValueError('Export checkpoint does not match this bundle; existing files were preserved.')
        offset = state['directory_offset']
        tail = base64.b64decode(state['tail'], validate=True)
        if not isinstance(offset, int) or offset < 0 or work.stat().st_size < offset:
            raise ValueError('Committed ZIP bytes are missing. Existing files were preserved.')
        with work.open('r+b') as handle:
            handle.seek(offset)
            handle.write(tail)
            handle.truncate(offset + len(tail))
        print('Resuming saved export:', work)
    else:
        work = export_dir / (bundle.name + '_resume_' + uuid.uuid4().hex + '.work.zip')
        if best:
            # Clone a readable legacy partial ZIP. Never append to the original attempt.
            source_path = Path(best['path'])
            with source_path.open('rb') as source, work.open('xb') as destination, tqdm(
                    total=source_path.stat().st_size,desc='Preserve previous ZIP progress',unit='B',unit_scale=True,unit_divisor=1024) as bar:
                for chunk in iter(lambda:source.read(4*1024*1024),b''):
                    destination.write(chunk); bar.update(len(chunk))
            with zipfile.ZipFile(work) as archive: pass
        else:
            with zipfile.ZipFile(work, 'x', compression=zipfile.ZIP_STORED) as archive: pass
        save_checkpoint(work, archive.start_dir)
        print('New resumable export:', work)

    with zipfile.ZipFile(work) as archive:
        committed = {m.filename: m.file_size for m in archive.infolist()}
    if any(name not in files or size != files[name][0] for name, size in committed.items()):
        raise ValueError('Checkpoint contains unexpected files. Existing source data were preserved.')
    pending = sorted(name for name in files if name not in committed)
    print('Files already exported:', len(committed), '| files remaining:', len(pending))

    # Store already-compressed PNG/JPEG/NPZ directly. No extra SHA pass or full ZIP reread.
    with tqdm(total=total, initial=sum(committed.values()), desc='Export saved files',
              unit='B', unit_scale=True, unit_divisor=1024) as bar:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            with zipfile.ZipFile(work, 'a', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name in batch:
                    path = bundle / name
                    before = path.stat()
                    if [before.st_size, before.st_mtime_ns] != files[name]:
                        raise RuntimeError('Source changed. Stop inference/review writes before exporting.')
                    written = 0
                    with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as destination:
                        for chunk in iter(lambda: source.read(4*1024*1024), b''):
                            destination.write(chunk)
                            written += len(chunk)
                            bar.update(len(chunk))
                    after = path.stat()
                    if written != files[name][0] or [after.st_size, after.st_mtime_ns] != files[name]:
                        raise RuntimeError('Source changed during export. Stop concurrent writes and rerun.')
            save_checkpoint(work, archive.start_dir)
            bar.set_postfix(files_committed=len(committed) + min(start + batch_size, len(pending)))

    if len(checked_members(work)) != len(files):
        raise RuntimeError('Final ZIP file list/size check failed; work ZIP and checkpoints preserved.')
    final = work.with_name(work.name.removesuffix('.work.zip') + '.zip')
    if final.exists():
        raise FileExistsError(final)
    work.rename(final)
    print('Drive ZIP ready:', final)
    print('File list/sizes checked; full payload CRC/SHA re-verification skipped for speed.')
    return final


if __name__ == '__main__':
    # After a runtime reset, paste the EXACT saved bundle path here; no earlier cells needed.
    BUNDLE_TO_EXPORT = globals().get('OUT') or globals().get('RESUME_BUNDLE', '')
    EXPORT_FOLDER = Path(globals().get('ROOT', '/content/drive/MyDrive/Robotekin2026')) / 'exports'
    if not BUNDLE_TO_EXPORT or str(BUNDLE_TO_EXPORT) == 'auto':
        raise ValueError('Set BUNDLE_TO_EXPORT above to your exact saved review_bundle folder.')
    if not Path('/content/drive/MyDrive').exists():
        from google.colab import drive
        drive.mount('/content/drive')
    export_saved_bundle(BUNDLE_TO_EXPORT, EXPORT_FOLDER,
        search_dirs=[globals().get('EXPORT_LOCAL_DIR','/content/roboqc_review_exports')],
        listing_workers=globals().get('EXPORT_LISTING_WORKERS',4),
        refresh_inventory=globals().get('REFRESH_EXPORT_INVENTORY',False))
