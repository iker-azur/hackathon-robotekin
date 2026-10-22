"""Build self-contained notebook documents; does NOT execute their cells.

Run after editing shared runtime code. Embedded source avoids a private Git URL
or a second code upload in Colab. Only datasets/weights need uploading.
"""

import ast
import json
from pathlib import Path
import runpy
import textwrap

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True), "id": __import__("hashlib").sha256(source.encode()).hexdigest()[:12]}
    if kind == "code":
        ast.parse(source)
        value.update(execution_count=None, outputs=[])
    return value


def md(source):
    return cell("markdown", source)


def code(source):
    return cell("code", source)


def bootstrap(vision=False):
    paths = ["industrial/__init__.py", "industrial/voice.py", "industrial/dataset.py", "industrial/models.py", "industrial/training.py", "intent/jepa_model.py"]
    payload = {"roboqc/__init__.py": "", "roboqc/intent/__init__.py": ""}
    for path in paths:
        payload["roboqc/" + path] = (DEMO / "phase_1/src/roboqc" / path).read_text(encoding="utf-8")
    if vision:
        for name in ("vision_data.py", "vision_training.py", "vision_inference.py", "yoloe.py"):
            payload["roboqc/industrial/" + name] = (DEMO / "phase_1/src/roboqc/industrial" / name).read_text(encoding="utf-8")
    return code("""
        import sys, json, zipfile, shutil, uuid
        from pathlib import Path
        RUNTIME = Path('/content/roboqc_runtime')
        SOURCES = PAYLOAD
        for relative, source in SOURCES.items():
            target = RUNTIME / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding='utf-8')
        sys.path.insert(0, str(RUNTIME))
        def package_runtime(folder):
            destination = Path(folder) / 'runtime'
            shutil.copytree(RUNTIME / 'roboqc', destination / 'roboqc', dirs_exist_ok=True)
        def local_requirements(folder, packages):
            from importlib.metadata import version, PackageNotFoundError
            rows = []
            for package in packages:
                try:
                    # Colab CUDA build suffixes are not portable to Windows CPU wheels.
                    rows.append(package + '==' + version(package).split('+')[0])
                except PackageNotFoundError:
                    pass
            (Path(folder) / 'requirements-local.txt').write_text('\\n'.join(rows) + '\\n')
        def upload_folder():
            from google.colab import files
            uploaded = files.upload()
            target = RUN / ('uploaded_' + uuid.uuid4().hex[:8])
            target.mkdir()
            for name, blob in uploaded.items():
                path = target / Path(name).name
                path.write_bytes(blob)
                if path.suffix == '.zip':
                    with zipfile.ZipFile(path) as archive:
                        root = target.resolve()
                        for member in archive.infolist():
                            if not (root / member.filename).resolve().is_relative_to(root):
                                raise ValueError('Unsafe ZIP member')
                        archive.extractall(target)
            return target
        def archive_export(folder):
            package_runtime(folder)
            archive = shutil.make_archive(str(folder), 'zip', folder)
            print('Export:', archive)
            return archive
    """.replace("PAYLOAD", repr(payload)))


def config(extra=""):
    return code(textwrap.dedent("""
        # Change these values BEFORE running subsequent cells.
        USE_DRIVE = True
        ROOT = '/content/drive/MyDrive/Robotekin2026'
        SEED = 42
        DATASET_PATH = ''  # Existing dataset folder on Drive; empty opens upload.
        MODEL_PATH = ''    # Existing exported model folder; empty opens upload.
        RESUME_RUN = ''    # Existing run folder for checkpoint continuation.
        EXTRA
    """).replace("EXTRA", extra))


def setup(deps):
    return code("""
        import os, random, subprocess, sys, uuid
        from datetime import datetime, timezone
        from pathlib import Path
        print('Python:', sys.version, '| executable:', sys.executable, flush=True)
        install_log = Path('/content/roboqc_pip_install.log')
        install_log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', DEPS]
        with install_log.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            exit_code = process.wait()
        if exit_code:
            raise RuntimeError(f'pip failed (exit {exit_code}). Read the ERROR lines above. Full log: {install_log}')
        if USE_DRIVE:
            from google.colab import drive
            drive.mount('/content/drive')
        else:
            ROOT = '/content/Robotekin2026'
        RUN = Path(RESUME_RUN) if RESUME_RUN else Path(ROOT) / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        RUN.mkdir(parents=True, exist_ok=bool(RESUME_RUN))
        import numpy as np, torch
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        GPU = torch.cuda.is_available()
        DEVICE = 'cuda' if GPU else 'cpu'
        print('Device:', DEVICE, '| Run:', RUN)
        (RUN / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    """.replace("DEPS", ", ".join(repr(d) for d in deps)))


COMMON = ["numpy>=1.26,<3", "scikit-learn>=1.4,<2", "joblib>=1.3,<2", "matplotlib>=3.8,<4", "pandas>=2.2,<4", "psutil>=5.9,<8"]
TRAIN = COMMON + ["torch>=2.5,<3", "sentence-transformers>=3.4,<7", "transformers>=5.2,<6", "accelerate>=1.2,<2", "peft>=0.18,<1", "datasets>=3,<6", "onnx>=1.17,<2", "onnxruntime>=1.20,<2"]


def write(name, cells):
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}, "colab": {"provenance": []}}, "cells": cells}
    (HERE / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def voice_notebooks():
    write("01_generate_voice_dataset.ipynb", [
        md("""
        # 01 · Spanish industrial voice dataset
        Run manually in Google Colab. No API keys or external text-generation service.
        Labels originate from semantic records, not predictions of the parser being tested.
        Synthetic text is not evidence of noisy-microphone performance. Collect real recordings later.
        Families are reserved by split; canonical text is globally deduplicated.
        """), config("N_EXAMPLES = 25000"), setup(COMMON + ["torch>=2.5,<3"]), bootstrap(),
        code("""
        from roboqc.industrial.dataset import generate, export
        splits = generate(N_EXAMPLES, SEED)
        DATA = RUN / 'voice_dataset'
        report = export(splits, DATA)
        (DATA / 'config.json').write_text(json.dumps({'seed': SEED, 'n': N_EXAMPLES, 'split': [0.70, 0.15, 0.15]}, indent=2))
        print(json.dumps(report, indent=2))
        """), md("Inspect class coverage and cross-split near duplicates. Prefix and polite-word variants are stripped for this diagnostic; investigate hits before training."),
        code("""
        import pandas as pd, matplotlib.pyplot as plt
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        counts = pd.DataFrame({s: r['status'] for s, r in report.items()}).fillna(0)
        display(counts)
        counts.plot.bar(title='Status counts by split'); plt.show()
        for s, rows in splits.items():
            display(pd.DataFrame(rows).sample(min(5, len(rows)), random_state=SEED)[['text', 'expected']])
            assert {'sort', 'stop', 'resume', 'cancel', 'unsupported'} <= {r['intent'] for r in rows}, s
        # Bounded diagnostic, not a claim that synthetic data has no semantic overlap.
        texts = {s: [r['text'] for r in rows[:2000]] for s, rows in splits.items()}
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
        x = vectorizer.fit_transform(texts['train'])
        nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(x)
        near = {}
        for split in ['validation', 'test']:
            distances, indices = nn.kneighbors(vectorizer.transform(texts[split]))
            near[split] = [{'text': texts[split][i], 'train_match': texts['train'][int(indices[i, 0])], 'similarity': float(1-d[0])}
                           for i, d in enumerate(distances) if d[0] < .02]
        (DATA / 'near_duplicate_audit.json').write_text(json.dumps(near, ensure_ascii=False, indent=2))
        print('Near-duplicate audit counts:', {k: len(v) for k, v in near.items()})
        """), code("archive = archive_export(DATA)\nprint('Use this folder/ZIP in notebook 02. CSV and JSONL are both included.')")])

    write("02_train_intent_model.ipynb", [
        md("""
        # 02 · Intent training and artifact export
        Choose a GPU runtime for JEPA/OpenJev; TF-IDF and the MiniLM head also work on CPU.
        MiniLM uses a multilingual pretrained encoder and trains a linear head. OpenJev is
        experimental: Spanish generalization and CPU latency must be measured, not assumed.
        JEPA is the existing compact character encoder, not the proprietary Jev model.
        Only validation selects thresholds/models. Test remains untouched until notebook 03.
        Artifact bundles use trusted Python/joblib files: load only your own exports.
        """), config("MODELS = ['tfidf', 'minilm', 'jepa', 'openjev']\nEPOCHS = 30\nPATIENCE = 5\nCALIBRATION_LIMIT = 300  # Same held-out validation subset for all candidates\nMINILM = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'"),
        setup(TRAIN), bootstrap(),
        code("""
        from roboqc.industrial.training import load_records, seed_all
        seed_all(SEED)
        source = Path(DATASET_PATH) if DATASET_PATH else upload_folder()
        matches = list(source.rglob('train.jsonl'))
        if (source / 'train.jsonl').exists():
            DATA = source
        elif len(matches) == 1:
            DATA = matches[0].parent
        else:
            raise ValueError('Choose exactly one generated dataset folder')
        train = load_records(DATA / 'train.jsonl')
        validation = load_records(DATA / 'validation.jsonl')
        assert not {r['family'] for r in train} & {r['family'] for r in validation}
        OUT = RUN / 'intent_bundle'
        OUT.mkdir(exist_ok=bool(RESUME_RUN))
        manifest_path = OUT / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema_version': 'industrial-1.0', 'selected': 'tfidf', 'models': {}, 'seed': SEED}
        """),
        code("""
        from roboqc.industrial.training import train_linear, train_jepa, train_openjev, select_and_calibrate
        for kind in MODELS:
            if kind in manifest['models']:
                print('Already completed:', kind); continue
            if kind == 'openjev' and not GPU:
                print('PENDING: OpenJev requires a GPU. Resume this run with a GPU.'); continue
            if kind == 'tfidf':
                metrics = train_linear(train, validation, OUT, seed=SEED, epochs=EPOCHS, patience=PATIENCE)
            elif kind == 'minilm':
                metrics = train_linear(train, validation, OUT, encoder_name=MINILM, seed=SEED, epochs=EPOCHS, patience=PATIENCE)
            elif kind == 'jepa':
                metrics = train_jepa(train, validation, OUT, seed=SEED, epochs=EPOCHS, patience=PATIENCE)
            elif kind == 'openjev':
                metrics = train_openjev(train, validation, OUT, seed=SEED)
            else:
                raise ValueError(kind)
            manifest['models'][kind] = metrics
            manifest['selected'] = next(iter(manifest['models']))
            manifest_path.write_text(json.dumps(manifest, indent=2))
            package_runtime(OUT)
        calibration = random.Random(SEED).sample(validation,min(CALIBRATION_LIMIT,len(validation)))
        manifest = select_and_calibrate(OUT, calibration)
        print(json.dumps(manifest, indent=2))
        """), md("Optional optimized MiniLM encoder export. Verify embedding equivalence and downstream decisions before choosing ONNX locally. The default local loader uses the original CPU encoder; OpenJev exports merged Transformers weights."),
        code("""
        if 'minilm' in manifest['models']:
            from roboqc.industrial.training import export_minilm_onnx
            check = export_minilm_onnx(OUT, [r['text'] for r in validation[:100]])
            print(check)
            (OUT / 'quantization_equivalence.json').write_text(json.dumps(check,indent=2))
            if check['passed']:
                manifest['models']['minilm'].update(encoder_backend='onnx_native',encoder_path='encoder_int8')
                manifest_path.write_text(json.dumps(manifest,indent=2))
            else:
                print('Keep original encoder: quantized behavior did not meet equivalence criteria.')
        """), code("""
        import shutil
        shutil.copy2(RUN / 'environment.txt', OUT / 'requirements-colab.txt')
        local_requirements(OUT,['numpy','scikit-learn','joblib','torch','sentence-transformers','transformers','onnxruntime','tokenizers','safetensors'])
        archive = archive_export(OUT)
        print('Download ZIP from the Colab Files panel or Google Drive. Do not replace existing models.')
        """)])

    write("03_evaluate_intent_model.ipynb", [
        md("""
        # 03 · Held-out evaluation, CPU latency and dialogue
        Load notebook 01's dataset ZIP and notebook 02's model ZIP (or Drive folders).
        CPU timing in Colab is not the timing of your hackathon laptop. No retraining occurs here.
        The adversarial examples below were written independently of the synthetic generator.
        """), config("CPU_THREADS = 2\nOPENJEV_EVAL_LIMIT = 200  # Set None for the full CPU test; can be slow"), setup(TRAIN + ["jiwer>=3,<5", "faster-whisper>=1.1,<2", "soundfile>=0.12,<1"]), bootstrap(),
        code("""
        from roboqc.industrial.training import load_records, evaluate
        from roboqc.industrial.models import IntentBundle
        torch.set_num_threads(CPU_THREADS)
        dataset_root = Path(DATASET_PATH) if DATASET_PATH else upload_folder()
        data_files = list(dataset_root.rglob('test.jsonl'))
        if not data_files: raise FileNotFoundError('Upload the generated dataset ZIP')
        records = load_records(data_files[0])
        model_root = Path(MODEL_PATH) if MODEL_PATH else upload_folder()
        manifests = [p for p in model_root.rglob('manifest.json') if json.loads(p.read_text()).get('schema_version') == 'industrial-1.0']
        if len(manifests) != 1: raise ValueError('Choose one exported intent bundle')
        BUNDLE = manifests[0].parent
        manifest = json.loads(manifests[0].read_text())
        reports = {}
        import time, psutil, pandas as pd, matplotlib.pyplot as plt
        for kind in manifest['models']:
            start = time.perf_counter()
            model = IntentBundle(BUNDLE, kind)
            load_ms = (time.perf_counter() - start) * 1000
            evaluation_records = records if kind != 'openjev' or OPENJEV_EVAL_LIMIT is None else random.Random(SEED).sample(records,min(OPENJEV_EVAL_LIMIT,len(records)))
            for r in evaluation_records[:10]: model.predict(r['text'])
            report, predictions = evaluate(evaluation_records, model)
            report['evaluated_subset'] = len(evaluation_records) != len(records)
            report.update(load_ms=load_ms, process_rss_mb=psutil.Process().memory_info().rss / 2**20)
            reports[kind] = report
            (RUN / f'{kind}_predictions.jsonl').write_text('\\n'.join(json.dumps({'text': r['text'], 'expected': r['expected'], 'prediction': p}, ensure_ascii=False) for r,p in zip(evaluation_records,predictions)))
            plt.imshow(report['confusion_matrix']); plt.xticks(range(len(report['confusion_labels'])), report['confusion_labels'], rotation=45, ha='right')
            plt.yticks(range(len(report['confusion_labels'])), report['confusion_labels']); plt.title(kind + ' status confusion'); plt.tight_layout(); plt.savefig(RUN / f'{kind}_confusion.png'); plt.show()
            del model
        display(pd.DataFrame(reports).T[['complete_command_accuracy','false_acceptance_rate','relation_f1','load_ms']])
        (RUN / 'intent_report.json').write_text(json.dumps(reports, indent=2))
        pd.json_normalize([{'model': k, **v} for k,v in reports.items()]).to_csv(RUN / 'intent_report.csv', index=False)
        """),
        code("""
        from roboqc.industrial.voice import Dialogue, understand, canonical
        model = IntentBundle(BUNDLE)
        difficult = [
            ('Pon las piezas buenas en la caja verde, las defectuosas tipo uno en la roja y las tipo dos en la azul.', 'valid'),
            ('Clasifica las piezas rojas en la caja azul, las verdes en la roja y las azules en la verde.', 'valid'),
            ('Recoge solamente las piezas defectuosas.', 'needs_clarification'),
            ('Pon las rojas en la azul y las rojas en la verde.', 'contradictory'),
            ('Pon las buenas en la amarilla.', 'unsupported'),
            ('No pares la clasificacion.', 'unsupported'),
            ('Clasifica las rojas excepto las buenas.', 'needs_clarification'),
            ('Deten la clasificacion.', 'valid'), ('Continua clasificando.', 'valid'),
            ('', 'invalid')]
        errors = []
        for text, status in difficult:
            prediction = understand(text, model)
            if prediction['status'] != status: errors.append({'text': text, 'expected': status, 'prediction': prediction})
        print('Adversarial failures:', json.dumps(errors, ensure_ascii=False, indent=2))
        dialogue = Dialogue()
        turns = [dialogue.submit(t, model) for t in ['Recoge solamente las defectuosas', 'la azul', 'Deten la clasificacion', 'Continua clasificando']]
        assert [r['status'] for r in turns] == ['needs_clarification','valid','valid','valid']
        (RUN / 'challenge_report.json').write_text(json.dumps({'errors': errors, 'turns': turns}, ensure_ascii=False, indent=2))
        """), md("Interactive command cell: edit CUSTOM_COMMAND and rerun. The dialogue persists between calls so you can answer a clarification."),
        code("CUSTOM_COMMAND = 'Pon las piezas buenas en la caja verde'\nprint(json.dumps(dialogue.submit(CUSTOM_COMMAND, model), ensure_ascii=False, indent=2))"),
        md("Optional real-audio evaluation: provide a JSONL manifest with `audio_path`, `reference` and `expected` (industrial envelope), plus WAV files. Leave AUDIO_MANIFEST empty to mark ASR evaluation pending. Relative WAV paths resolve against the manifest folder."),
        code("""
        AUDIO_MANIFEST = ''
        if AUDIO_MANIFEST:
            import soundfile as sf
            from faster_whisper import WhisperModel
            from jiwer import wer
            asr = WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=CPU_THREADS)
            rows = load_records(AUDIO_MANIFEST); audio_results = []
            for row in rows:
                wav = Path(row['audio_path'])
                if not wav.is_absolute(): wav = Path(AUDIO_MANIFEST).parent / wav
                start = time.perf_counter()
                segments, _ = asr.transcribe(str(wav), language='es', vad_filter=True, beam_size=1)
                transcription = ' '.join(s.text.strip() for s in segments)
                prediction = understand(transcription, model)
                audio_results.append({'reference': row['reference'], 'transcription': transcription,
                    'wer': wer(row['reference'], transcription), 'latency_ms': (time.perf_counter()-start)*1000,
                    'command_exact': canonical(prediction['command']) == canonical(row['expected']['command']) and prediction['status'] == row['expected']['status']})
            (RUN / 'audio_report.json').write_text(json.dumps(audio_results, ensure_ascii=False, indent=2))
            display(pd.DataFrame(audio_results))
        else:
            (RUN / 'audio_report.json').write_text(json.dumps({'status': 'pending', 'reason': 'No real recordings supplied'}))
        archive = shutil.make_archive(str(RUN.parent / (RUN.name + '_intent_evaluation')), 'zip', RUN)
        print('Report:', RUN)
        """)])


VISION = COMMON + ["torch>=2.5,<3", "torchvision>=0.20,<1", "ultralytics>=8.4.0,<9", "opencv-python-headless>=4.10,<6", "pycocotools>=2.0.8,<3", "PyYAML>=6,<7", "onnx>=1.17,<2", "onnxruntime>=1.20,<2"]


def labelled(source):
    return code("if EVALUATION_MODE == 'labelled':\n" + textwrap.indent(textwrap.dedent(source).strip(), "    "))


def yoloe_notebook():
    write("04a_try_yoloe_pretrained.ipynb", [
        md("""
        # 04a · Start here: pretrained YOLOE, no dataset or training
        Ready-to-run photos: `USE_DEMO_IMAGES=True` downloads five real Wikimedia Commons
        photos: fasteners, a single angle bracket, four gray brackets, a formed amplifier
        bracket (stereo pair), and an assembled channel connector. No photo upload or labels are needed.
        Set it False to use your own Drive folder or upload images/a ZIP.
        Demo photos are not a labelled dataset and do not establish performance on your parts.
        Credits and license links are saved beside the downloaded originals in ATTRIBUTION.txt.
        Downloads reuse ROOT/demo_images across runs. Transient HTTP 429/5xx responses
        use bounded retries and respect Retry-After. If some photos remain unavailable,
        the notebook warns and uses the verified subset; see download_report.json.
        No available photos is an error. You may always supply your own IMAGE_PATH.
        Sources: [Sbrennan83, CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Bolts,Nuts_%26_Washers.jpg),
        [Fructibus, CC0](https://commons.wikimedia.org/wiki/File:Bronze_angle_bracket_-_4_x_4_cm_-_A.jpg).
        Additional photos: [four brackets, Fructibus, CC0](https://commons.wikimedia.org/wiki/File:4_x_Gray_angle_bracket_-_12.5_cm_-_A.jpg),
        [amplifier bracket, Robbie Sproule, CC BY 2.0](https://commons.wikimedia.org/wiki/File:Bracket_for_amp._(25960984807).jpg),
        [channel connector, Best Tech Nick 25, CC0](https://commons.wikimedia.org/wiki/File:Hilti_Schienen_Eckverbinder_90%C2%B0.jpg).
        These are related bracket examples, not exact substitutes for loose stamped clips.
        The stereo image repeats one object; the assembled connector is a clutter test.
        For your own stamped parts, set IMAGE_PATH to a Drive folder of photos. You can
        try PROMPTS['bracket'] = 'sheet metal bracket' or 'stamped metal part' in separate
        runs; prompt wording is an experiment, not evidence of improved accuracy.

        This notebook downloads pretrained YOLOE and its text encoder, configures four prompts,
        exports a local bundle and displays CPU predictions. It never calls model.train().
        Defaults: YOLOE-26 nano; prompts bolt / metal nut / metal washer / metal bracket.
        Adjust wording after inspecting results. New prompts require re-exporting the model.

        Predictions are not ground truth. Quality stays unknown, grasp points are candidates,
        and single RGB images provide no validated 3D pose. Review the Ultralytics license.
        Source/API: https://docs.ultralytics.com/models/yoloe/
        """), config("USE_DEMO_IMAGES = True  # Download credited real sample photos; no upload needed\nIMAGE_PATH = ''  # Nonempty path overrides demo photos; otherwise False above opens upload\nCHECKPOINT = 'yoloe-26n-seg.pt'\nPROMPTS = {'bolt':'bolt','nut':'metal nut','washer':'metal washer','bracket':'metal bracket'}\nIMGSZ = 640\nCONFIDENCE = 0.25\nMAX_IMAGES = 50\nCPU_THREADS = 2\nEXPORT_ONNX = True"),
        setup(VISION), bootstrap(True),
        code("""
        from roboqc.industrial.yoloe import find_images, prepare_bundle, inspect_images, download_demo_images
        torch.set_num_threads(CPU_THREADS)
        source = Path(IMAGE_PATH).resolve() if IMAGE_PATH else download_demo_images(Path(ROOT) / 'demo_images') if USE_DEMO_IMAGES else upload_folder()
        images = find_images(source,MAX_IMAGES)
        import cv2, matplotlib.pyplot as plt
        for path in images[:10]:
            frame = cv2.imread(str(path))
            if frame is None: raise ValueError(f'Unreadable photo: {path}')
            plt.figure(figsize=(8,6)); plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); plt.title(path.name); plt.axis('off'); plt.show()
        print('Input photos:',len(images),'| next cell prepares the pretrained model, without training.')
        """),
        code("""
        OUT = (RUN / ('yoloe_bundle_' + uuid.uuid4().hex)).resolve()
        print('New export attempt (previous folders preserved):',OUT,flush=True)
        CACHE = (Path(ROOT) / 'pretrained_cache/yoloe').resolve()
        CACHE.mkdir(parents=True,exist_ok=True)
        original_directory = Path.cwd()
        try:
            os.chdir(CACHE)  # Reuse downloaded weights/text encoder between Colab sessions.
            manifest = prepare_bundle(OUT,images,CHECKPOINT,PROMPTS,IMGSZ,CONFIDENCE,EXPORT_ONNX,SEED)
        finally:
            os.chdir(original_directory)
        print(json.dumps(manifest,indent=2))
        print('No training performed. Bundle:',OUT)
        """),
        code("""
        PREVIEW = (RUN / ('yoloe_preview_' + uuid.uuid4().hex)).resolve()
        report,rows = inspect_images(OUT,images,PREVIEW,CONFIDENCE,CPU_THREADS)
        import cv2, matplotlib.pyplot as plt, pandas as pd
        for row in rows:
            if row['overlay']:
                view = cv2.imread(str(PREVIEW / row['overlay']))
                plt.figure(figsize=(10,6)); plt.imshow(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
            print(Path(row['image']).name,[(obj['type'],round(obj['confidence'],3),obj['color']) for obj in row['objects']])
        display(pd.json_normalize(report))
        print('No labelled data: mAP, recall and classification accuracy are NOT measured.')
        """),
        code("""
        shutil.copy2(RUN / 'environment.txt',OUT / 'requirements-colab.txt')
        local_requirements(OUT,['ultralytics','onnxruntime','opencv-python','numpy','torch','torchvision'])
        (OUT / 'local_config.example.json').write_text(json.dumps({
            'industrial':{'enabled':True,'model_dir':'REPLACE_WITH_ABSOLUTE_EXTRACTED_BUNDLE_PATH',
                          'prefer_onnx':EXPORT_ONNX,'confidence':CONFIDENCE,'source_region':None,
                          'destinations':{'red':None,'green':None,'blue':None}}},indent=2))
        archive = archive_export(OUT)
        print('Use this ZIP locally, or set MODEL_PATH to this folder in notebook 06:',OUT)
        print('Notebook 05 is optional fine-tuning; skip it for this pretrained route.')
        """)])


def sam_notebook():
    payload = {'roboqc/__init__.py':'','roboqc/industrial/__init__.py':''}
    for name in ['vision_inference.py','sam_masks.py']:
        payload['roboqc/industrial/' + name] = (DEMO / 'phase_1/src/roboqc/industrial' / name).read_text(encoding='utf-8')
    deps = ['numpy>=1.26,<3','torch>=2.5.1,<3','torchvision>=0.20.1,<1',
            'matplotlib>=3.9,<4','Pillow>=10,<13','opencv-python-headless>=4.10,<6',
            'huggingface-hub>=0.26,<2','hydra-core>=1.3.2,<2','iopath>=0.1.10,<1','tqdm>=4.66,<5']
    write('04b_try_sam2_stamped_parts.ipynb',[
        md('''
        # 04b · Segment Anything 2.1 — your four stamped-part photos
        Standalone alternative to YOLOE: no training, no class vocabulary or prior model bundle.
        No photos are bundled. Set IMAGE_PATH to a Drive image/folder or leave it blank
        to upload your own photos. Dependencies and weights need internet on first use.
        Choose **Runtime → Change runtime type → T4 GPU** before running. CPU works but is slow.

        Default: SAM 2.1 Large, automatic grid-of-points mask generation. This proposes regions,
        not guaranteed one-mask-per-part instances. It may split a part, include shadows/holes,
        return overlapping/nested masks or miss parts. It does NOT identify types or defects.
        An optional box-prompt cell lets you request a particular part without training.
        No ground-truth masks exist here: predicted IoU is a model score, NOT measured accuracy.

        User-supplied images: no open redistribution license established; verify rights before
        publishing this notebook. Model/code: https://github.com/facebookresearch/sam2
        Apache 2.0; review upstream licenses. No notebook/model execution by the coding agent.
        '''),
        config("MODEL_ID = 'facebook/sam2.1-hiera-large'  # Also: facebook/sam2.1-hiera-small\nMODEL_REVISION = 'main'  # Resolved checkpoint revision/hash is recorded; pin it for repeat runs\nSAM2_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'\nIMAGE_PATH = ''  # Drive image/folder; blank uploads one or more photos\nPOINTS_PER_SIDE = 32\nPOINTS_PER_BATCH = 16  # Lower to 8 or 4 if GPU memory is insufficient\nCROP_N_LAYERS = 0  # Try 1 for small/missed parts; more compute\nPRED_IOU_THRESH = 0.8\nSTABILITY_SCORE_THRESH = 0.95\nUSE_M2M = True\nMIN_MASK_AREA = 100\nMAX_MASK_FRACTION = 0.85  # Optional large-background filter, not ground truth\nMAX_MASK_PREVIEWS = 16\nBOX_IMAGE_INDEX = 0\nBOX_XYXY = []  # Optional [x1,y1,x2,y2] in original image pixels\nCPU_THREADS = 2"),
        setup(deps),
        code('''
        # Disable optional CUDA extension compilation; preserve mask holes (no fill-hole postprocess).
        os.environ['SAM2_BUILD_CUDA'] = '0'
        command = [sys.executable,'-m','pip','install','--no-build-isolation',
                   'git+https://github.com/facebookresearch/sam2.git@' + SAM2_COMMIT]
        with install_log.open('a',encoding='utf-8') as log:
            process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line)
            if process.wait(): raise RuntimeError(f'SAM 2 install failed. Full log: {install_log}')
        (RUN / 'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        torch.set_num_threads(CPU_THREADS)
        if not GPU:
            print('WARNING: CPU runtime. Large automatic segmentation can be very slow. GPU recommended.')
        '''),
        code((HERE / 'image_inputs.py').read_text(encoding='utf-8')),
        code('''
        import json, shutil, time
        from PIL import Image, ImageOps
        import matplotlib.pyplot as plt
        SOURCES = SAM_SOURCES
        RUNTIME = Path('/content/roboqc_sam_runtime')
        for relative,source in SOURCES.items():
            destination = RUNTIME / relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(source,encoding='utf-8')
        sys.path.insert(0,str(RUNTIME))
        from roboqc.industrial.sam_masks import save_candidates
        paths = select_image_paths(IMAGE_PATH,RUN)
        for index,path in enumerate(paths):
            with Image.open(path) as opened: rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
            plt.figure(figsize=(9,6)); plt.imshow(rgb); plt.title(f'{index}: {path.name} — {rgb.shape[1]} x {rgb.shape[0]}'); plt.axis('on'); plt.show()
        '''.replace('SAM_SOURCES',repr(payload))),
        code('''
        # Public checkpoints, no Hugging Face access approval/token needed for these SAM 2.1 weights.
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2, HF_MODEL_ID_TO_FILENAMES
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        if MODEL_ID not in HF_MODEL_ID_TO_FILENAMES: raise ValueError('Select an official SAM 2/2.1 model ID')
        CACHE = Path(ROOT) / 'pretrained_cache/sam2'
        config_name,checkpoint_name = HF_MODEL_ID_TO_FILENAMES[MODEL_ID]
        checkpoint = Path(hf_hub_download(MODEL_ID,checkpoint_name,revision=MODEL_REVISION,cache_dir=str(CACHE)))
        digest = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
        model = build_sam2(config_name,str(checkpoint),device=DEVICE,apply_postprocessing=False)
        model.eval()
        generator = SAM2AutomaticMaskGenerator(model,points_per_side=POINTS_PER_SIDE,
            points_per_batch=POINTS_PER_BATCH,pred_iou_thresh=PRED_IOU_THRESH,
            stability_score_thresh=STABILITY_SCORE_THRESH,crop_n_layers=CROP_N_LAYERS,
            crop_n_points_downscale_factor=2,min_mask_region_area=0,use_m2m=USE_M2M,output_mode='binary_mask')
        settings = {'model_id':MODEL_ID,'requested_revision':MODEL_REVISION,'resolved_revision':checkpoint.parent.name,
            'checkpoint':str(checkpoint),'checkpoint_sha256':digest.hexdigest(),'sam2_commit':SAM2_COMMIT,
            'device':DEVICE,'seed':SEED,'points_per_side':POINTS_PER_SIDE,'points_per_batch':POINTS_PER_BATCH,
            'crop_n_layers':CROP_N_LAYERS,'crop_n_points_downscale_factor':2,'pred_iou_thresh':PRED_IOU_THRESH,
            'stability_score_thresh':STABILITY_SCORE_THRESH,'use_m2m':USE_M2M,
            'minimum_area':MIN_MASK_AREA,'maximum_fraction':MAX_MASK_FRACTION,'apply_postprocessing':False}
        print(json.dumps(settings,indent=2))
        '''),
        code('''
        # Each rerun is a new attempt. Save each completed image immediately.
        OUT = RUN / ('sam2_results_' + uuid.uuid4().hex)
        OUT.mkdir()
        (OUT / 'settings.json').write_text(json.dumps(settings,indent=2))
        summaries = []
        for index,path in enumerate(paths):
            with Image.open(path) as opened: rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
            if GPU: torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(): annotations = generator.generate(rgb)
            if GPU: torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            folder = OUT / f'image_{index:02d}'
            result,overlay = save_candidates(rgb,annotations,folder,MIN_MASK_AREA,MAX_MASK_FRACTION,SEED)
            Image.fromarray(rgb).save(folder / 'input.png')
            summaries.append({'image':str(path),'output':folder.name,'raw_masks':result['raw_masks'],
                              'kept_candidates':result['kept_candidates'],'inference_seconds':elapsed})
            report = {'model':settings,'images':summaries,'accuracy':'not_measured',
                      'latency_note':'One automatic grid/crop pass per image; includes cold-start effects; not webcam FPS',
                      'classes':'unknown','quality':'unknown','status':'in_progress'}
            (OUT / 'report.json').write_text(json.dumps(report,indent=2))
            fig,axes = plt.subplots(1,2,figsize=(16,6))
            axes[0].imshow(rgb); axes[0].set_title(path.name)
            axes[1].imshow(overlay); axes[1].set_title(f"{result['kept_candidates']} candidate regions — {elapsed:.1f}s")
            for axis in axes: axis.axis('off')
            plt.show()
            kept = [r for r in result['candidates'] if r['kept']][:MAX_MASK_PREVIEWS]
            if kept:
                fig,axes = plt.subplots((len(kept)+3)//4,4,figsize=(16,4*((len(kept)+3)//4)),squeeze=False)
                for axis in axes.flat: axis.axis('off')
                for axis,row in zip(axes.flat,kept):
                    mask = np.asarray(Image.open(folder / row['mask_file'])) > 0
                    axis.imshow(np.where(mask[...,None],rgb,255)); axis.set_title(f"ID {row['id']} | predicted IoU {row['model_predicted_iou']:.2f}")
                plt.show()
            del annotations
        report['status'] = 'complete'
        (OUT / 'report.json').write_text(json.dumps(report,indent=2))
        print('Results:',OUT,'| No classification, training, or ground-truth accuracy measurement.')
        '''),
        md('''
        ## Optional: segment a specific part using a box
        If automatic masks are poor, set BOX_IMAGE_INDEX and BOX_XYXY in the initial
        configuration cell (or directly in a new cell), then run the next cell.
        Coordinates are [left, top, right, bottom] in the ORIGINAL photo pixels.
        Enclose one visible part, not its whole shadow. Empty BOX_XYXY skips this step.
        This is prompted segmentation, not autonomous object detection.
        '''),
        code('''
        if BOX_XYXY:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            if not 0 <= BOX_IMAGE_INDEX < len(paths): raise ValueError('BOX_IMAGE_INDEX is outside the image list')
            with Image.open(paths[BOX_IMAGE_INDEX]) as opened: rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
            box = np.asarray(BOX_XYXY,dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= rgb.shape[1] and 0 <= box[1] < box[3] <= rgb.shape[0]):
                raise ValueError('Use a valid finite [x1,y1,x2,y2] box inside the original image')
            predictor = SAM2ImagePredictor(model)
            with torch.inference_mode():
                predictor.set_image(rgb)
                masks,scores,_ = predictor.predict(box=box,multimask_output=False)
            annotations = [{'segmentation':m,'predicted_iou':float(s)} for m,s in zip(masks,scores)]
            folder = OUT / ('box_prompt_' + uuid.uuid4().hex)
            result,overlay = save_candidates(rgb,annotations,folder,1,1.0,SEED)
            (folder / 'prompt.json').write_text(json.dumps({'image':str(paths[BOX_IMAGE_INDEX]),'box_xyxy':box.tolist()}))
            plt.figure(figsize=(10,7)); plt.imshow(overlay); plt.axis('off'); plt.show()
            print('Prompted masks saved:',folder)
        else:
            print('Skipped: set BOX_XYXY to segment a specific part.')
        '''),
        code('''
        # Exports preserve binary masks (including holes and overlaps), not lossy polygons.
        shutil.copy2(RUN / 'environment.txt',OUT / 'requirements-colab.txt')
        shutil.copytree(RUNTIME / 'roboqc',OUT / 'runtime/roboqc')
        (OUT / 'README.txt').write_text('SAM 2.1 mask candidates only, not a RoboQC industrial vision bundle. No class/defect labels. Raw masks: raw_masks.npz; filtered masks: PNG; scores/geometry: candidates.json. Checkpoint stays in the Drive cache recorded in settings.json. Images are user-provided; verify redistribution rights.')
        archive = shutil.make_archive(str(OUT),'zip',OUT)
        print('Download results ZIP from the Files panel or Drive:',archive)
        print('This export does not include the large pretrained weights; their cached path/hash are in settings.json.')
        ''')])


def comparison_notebook():
    payload = {'roboqc/__init__.py':'','roboqc/industrial/__init__.py':''}
    for name in ['vision_inference.py','sam_masks.py','pretrained_compare.py']:
        payload['roboqc/industrial/' + name] = (DEMO / 'phase_1/src/roboqc/industrial' / name).read_text(encoding='utf-8')
    deps = ['numpy>=1.26,<3','torch>=2.5.1,<3','torchvision>=0.20.1,<1',
        'transformers==5.2.0','huggingface-hub>=1.3,<2','ultralytics>=8.4.0,<9',
        'matplotlib>=3.9,<4','Pillow>=10,<13','opencv-python-headless>=4.10,<6',
        'pandas>=2.2,<4','scipy>=1.13,<2','hydra-core>=1.3.2,<2','iopath>=0.1.10,<1','tqdm>=4.66,<5']
    write('04c_compare_pretrained_segmentation.ipynb',[
        md('''
        # 04c · Compare pretrained methods on your actual industrial parts
        **Use a fresh Colab T4 GPU session.** No training and no API-paid inference.
        No dataset/reference photos are bundled. Set IMAGE_PATH to a Drive image/folder
        or leave it blank to upload photos. First-use dependencies
        and weights still require internet and several GB of local runtime disk space.

        Five default methods, not five sizes of one model:
        1. Larger YOLOE text-prompted segmentation (direct masks).
        2. SAM 2.1 automatic grid masks (class-agnostic baseline).
        3. Grounding DINO + SAM 2.1 (text detection, box-prompted masks).
        4. OWLv2 + SAM 2.1 (different open-vocabulary detector, same mask decoder).
        5. Florence-2 + SAM 2.1 (generative grounding, same mask decoder).
        Optional sixth: SAM 3 text segmentation, requiring approved checkpoint access.

        All methods see the same images and optional workspace crop. No forced six-mask
        selection. No expected counts are assumed for uploaded photos.
        Count agreement does NOT prove correct segmentation. Mask scores are NOT comparable between
        architectures, and no accuracy winner is selected without human ground truth.
        Detector-plus-SAM methods can only segment objects whose boxes were proposed.

        This is a development comparison, not an independent held-out benchmark.
        No method is guaranteed to segment these pieces correctly. Masks are not safe
        robot grasps, and no hidden surfaces, height, defect labels or robot control are inferred.
        User-supplied image rights: verify before redistribution.

        Sources: [YOLOE](https://docs.ultralytics.com/models/yoloe/),
        [SAM 2](https://github.com/facebookresearch/sam2),
        [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base),
        [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble),
        [Florence-2](https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/florence2),
        [SAM 3](https://huggingface.co/facebook/sam3).
        Review each model's license/access terms. Coding agent has not run inference.
        '''),
        config("IMAGE_PATH = ''  # Drive image/folder; blank uploads one or more photos\nMETHODS = ['yoloe_text','sam2_auto','grounding_dino_sam2','owlv2_sam2','florence2_sam2']\nENABLE_SAM3 = False  # Optional sixth; requires approved facebook/sam3 access and HF_TOKEN\nPROMPTS = ['metal bracket','sheet metal part','metal mounting clip']\nROI_NORM = None  # Optional shared [left,top,right,bottom] fractions; None = whole photo\nGT_MANIFEST_PATH = ''  # Optional JSON: filename -> list of full-frame binary instance-mask PNG paths\nMODEL_REVISIONS = {}  # HF model ID -> pinned revision; resolved revisions are recorded\nSAM2_ID = 'facebook/sam2.1-hiera-large'  # Set small for less memory, consistently across methods\nSAM2_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'\nYOLOE_CHECKPOINT = 'yoloe-26l-seg.pt'\nIMGSZ = 960\nYOLOE_THRESHOLD = 0.25\nDINO_THRESHOLD = 0.25\nDINO_TEXT_THRESHOLD = 0.20\nOWL_THRESHOLD = 0.10\nSAM3_THRESHOLD = 0.50\nBOX_NMS_IOU = 0.70\nMAX_DETECTIONS = 50  # Resource limit, not expected part count\nPOINTS_PER_SIDE = 32\nPOINTS_PER_BATCH = 8\nCROP_N_LAYERS = 0\nSAM_PRED_IOU = 0.80\nSAM_STABILITY = 0.95\nFLORENCE_MAX_NEW_TOKENS = 1024\nMIN_MASK_AREA = 100\nMAX_MASK_FRACTION = 0.85\nWARMUP = True  # Extra untimed prediction on first image, per method\nCPU_THREADS = 2"),
        code("MODEL_CACHE = '/content/roboqc_pretrained_cache'  # All model weights on local disk, NOT Drive\nSAM2_CACHE = '/content/roboqc_sam2_cache'  # Local SAM cache, reused within this Colab session"),
        md('''
        SAM checkpoint recovery: SAM 2.1 weights are cached on local Colab disk,
        validated with `weights_only=True`, then loaded strictly. If deserialization
        fails, one new download is attempted in a fresh subfolder; previous files
        are preserved. Later methods reuse the recovered SAM cache. Detector weights
        also use local disk (`MODEL_CACHE`), with one fresh-cache retry on a
        SafetensorError. No unsafe pickle fallback is used. Results remain on Drive;
        all local model caches disappear on runtime reset and must download again.
        Failed methods print their full traceback and also save it in the results.
        '''),
        setup(deps),
        code('''
        # Same pinned official SAM 2 install as 04b; skip optional CUDA extension builds.
        os.environ['SAM2_BUILD_CUDA'] = '0'
        command = [sys.executable,'-m','pip','install','--no-build-isolation',
                   'git+https://github.com/facebookresearch/sam2.git@' + SAM2_COMMIT]
        with install_log.open('a',encoding='utf-8') as log:
            process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout: print(line,end='',flush=True); log.write(line)
            if process.wait(): raise RuntimeError(f'SAM install failed. See {install_log}')
        (RUN / 'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        torch.set_num_threads(CPU_THREADS)
        if not GPU: print('WARNING: CPU detected. Five-method comparison can be very slow; choose a GPU runtime.')
        CACHE = Path(MODEL_CACHE).resolve()
        CACHE.mkdir(parents=True,exist_ok=True)
        os.environ['HF_HOME'] = str(CACHE / 'huggingface')
        if ENABLE_SAM3:
            # Store an approved token in Colab Secrets as HF_TOKEN; never print it.
            try:
                from google.colab import userdata
                token = userdata.get('HF_TOKEN')
            except Exception:
                token = os.environ.get('HF_TOKEN')
            if token:
                from huggingface_hub import login
                login(token=token,add_to_git_credential=False)
                del token
            print('SAM 3 enabled: missing access approval will be reported as a method error, not substituted.')
        '''),
        code((HERE / 'image_inputs.py').read_text(encoding='utf-8')),
        code('''
        import hashlib, json, shutil, time, gc, traceback
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image, ImageOps
        SOURCES = SOURCE_PAYLOAD
        RUNTIME = Path('/content/roboqc_compare_runtime')
        for relative,source in SOURCES.items():
            target = RUNTIME / relative; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source,encoding='utf-8')
        sys.path.insert(0,str(RUNTIME))
        from roboqc.industrial.pretrained_compare import DEFAULT_METHODS, PretrainedMethod, crop_image, restore_masks, mask_metrics
        from roboqc.industrial.sam_masks import save_candidates
        if not PROMPTS or any(not isinstance(p,str) or not p.strip() for p in PROMPTS): raise ValueError('Use nonempty text prompts')
        if MAX_DETECTIONS < 1 or POINTS_PER_SIDE < 1 or POINTS_PER_BATCH < 1 or MAX_MASK_FRACTION <= 0 or MAX_MASK_FRACTION > 1:
            raise ValueError('Use positive limits and a mask fraction in (0,1]')
        selected = list(dict.fromkeys(METHODS + (['sam3_text'] if ENABLE_SAM3 else [])))
        if not selected or any(m not in DEFAULT_METHODS+['sam3_text'] for m in selected): raise ValueError('Unknown/empty METHODS')
        if len(selected) < 5: print('WARNING: fewer than five methods selected; default METHODS compares five.')
        paths = select_image_paths(IMAGE_PATH,RUN)
        samples = []
        for index,path in enumerate(paths):
            with Image.open(path) as opened: rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
            crop,roi = crop_image(rgb,ROI_NORM)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = None  # No count or mask ground truth is inferred from filenames
            samples.append({'index':index,'path':str(path),'sha256':digest,'expected_count':expected,'roi':list(roi)})
            plt.figure(figsize=(8,6)); plt.imshow(rgb); plt.title(f'{index}: {path.name} | processed ROI shown in green')
            from matplotlib.patches import Rectangle
            plt.gca().add_patch(Rectangle((roi[0],roi[1]),roi[2]-roi[0],roi[3]-roi[1],fill=False,edgecolor='green')); plt.axis('off'); plt.show()
        references = {}
        if GT_MANIFEST_PATH:
            gt_path = Path(GT_MANIFEST_PATH)
            manifest = json.loads(gt_path.read_text())
            for sample in samples:
                name = Path(sample['path']).name
                if name not in manifest: continue  # Missing annotation is NOT an empty ground-truth scene.
                with Image.open(sample['path']) as opened: shape = np.asarray(ImageOps.exif_transpose(opened)).shape[:2]
                masks = []
                for filename in manifest[name]:
                    with Image.open(gt_path.parent / filename) as opened: mask = np.asarray(opened.convert('L')) > 0
                    if mask.shape != shape or not mask.any(): raise ValueError(f'Invalid ground-truth mask: {filename}')
                    masks.append(mask)
                references[sample['index']] = masks
        print('Selected:',selected,'| images:',len(samples),'| ground-truth scenes:',len(references))
        '''.replace('SOURCE_PAYLOAD',repr(payload))),
        code('''
        OUT = RUN / ('pretrained_comparison_' + uuid.uuid4().hex); OUT.mkdir()
        cfg = {'cache':str(CACHE),'sam2_cache':SAM2_CACHE,'revisions':MODEL_REVISIONS,'sam2_id':SAM2_ID,'yoloe_checkpoint':YOLOE_CHECKPOINT,
            'prompts':PROMPTS,'imgsz':IMGSZ,'yoloe_threshold':YOLOE_THRESHOLD,'dino_threshold':DINO_THRESHOLD,
            'dino_text_threshold':DINO_TEXT_THRESHOLD,'owl_threshold':OWL_THRESHOLD,'sam3_threshold':SAM3_THRESHOLD,
            'box_nms':BOX_NMS_IOU,'max_detections':MAX_DETECTIONS,'points_per_side':POINTS_PER_SIDE,
            'points_per_batch':POINTS_PER_BATCH,'crop_layers':CROP_N_LAYERS,'sam_iou':SAM_PRED_IOU,
            'sam_stability':SAM_STABILITY,'florence_max_tokens':FLORENCE_MAX_NEW_TOKENS}
        settings = {'methods':selected,'config':cfg,'sam2_commit':SAM2_COMMIT,'seed':SEED,'device':DEVICE,
            'hardware':{'python':sys.version,'torch':torch.__version__,'cpu_threads':CPU_THREADS,
                        'gpu':torch.cuda.get_device_name(0) if GPU else None},
            'roi_norm':ROI_NORM,'minimum_area':MIN_MASK_AREA,'maximum_fraction':MAX_MASK_FRACTION,'warmup':WARMUP,
            'images':samples,'ground_truth_manifest':GT_MANIFEST_PATH,'weights_in_export':False}
        (OUT / 'settings.json').write_text(json.dumps(settings,indent=2))
        records = []
        for method in selected:
            folder = OUT / method; folder.mkdir()
            runner = None
            print('Loading',method,flush=True)
            try:
                start = time.perf_counter(); runner = PretrainedMethod(method,cfg,DEVICE)
                load_seconds = time.perf_counter()-start
                (folder / 'provenance.json').write_text(json.dumps(runner.provenance,indent=2))
                if WARMUP:
                    with Image.open(samples[0]['path']) as opened: rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
                    crop,_ = crop_image(rgb,ROI_NORM)
                    runner.predict(crop)  # Untimed, separate from reported per-image latency.
                    if GPU: torch.cuda.synchronize()
                for sample in samples:
                    image_out = folder / f"image_{sample['index']:02d}"
                    row = {'method':method,'image_index':sample['index'],'image':Path(sample['path']).name,
                           'load_seconds_including_downloads':load_seconds,'expected_count':sample['expected_count']}
                    try:
                        with Image.open(sample['path']) as opened: rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
                        crop,roi = crop_image(rgb,ROI_NORM)
                        if GPU: torch.cuda.synchronize()
                        start = time.perf_counter(); annotations,details = runner.predict(crop)
                        if GPU: torch.cuda.synchronize()
                        seconds = time.perf_counter()-start
                        annotations = restore_masks(annotations,rgb.shape,roi)
                        fraction = MAX_MASK_FRACTION * crop.shape[0]*crop.shape[1]/(rgb.shape[0]*rgb.shape[1])
                        result,_ = save_candidates(rgb,annotations,image_out,MIN_MASK_AREA,fraction,SEED)
                        Image.fromarray(rgb).save(image_out / 'input.png')
                        kept = [a['segmentation'] for a,r in zip(annotations,result['candidates']) if r['kept']]
                        row.update(status='ok',elapsed_s=seconds,raw_masks=result['raw_masks'],kept_masks=len(kept),
                            overlay=str((image_out / 'overlay.png').relative_to(OUT)),details=details,
                            count_abs_error=abs(len(kept)-sample['expected_count']) if sample['expected_count'] is not None else None)
                        row['accuracy_status'] = 'measured_on_supplied_masks' if sample['index'] in references else 'not_measured'
                        if sample['index'] in references: row['metrics'] = mask_metrics(kept,references[sample['index']])
                    except Exception as exc:
                        row.update(status='error',error=f'{type(exc).__name__}: {exc}')
                        (folder / f"error_image_{sample['index']:02d}.txt").write_text(traceback.format_exc())
                        print('ERROR:',method,row['image'],row['error'],flush=True)
                        traceback.print_exc(file=sys.stdout)
                    records.append(row)
                    (OUT / 'comparison.json').write_text(json.dumps(records,indent=2))
                    pd.json_normalize(records).to_csv(OUT / 'comparison.csv',index=False)
                    print(method,row['image'],row['status'],'| masks:',row.get('kept_masks'),'| seconds:',row.get('elapsed_s'),flush=True)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                (folder / 'load_or_warmup_error.txt').write_text(traceback.format_exc())
                records.append({'method':method,'status':'load_or_warmup_error','error':error})
                print('ERROR:',method,error,flush=True)
                traceback.print_exc(file=sys.stdout)
                (OUT / 'comparison.json').write_text(json.dumps(records,indent=2))
                pd.json_normalize(records).to_csv(OUT / 'comparison.csv',index=False)
            finally:
                del runner
                gc.collect()
                if GPU: torch.cuda.empty_cache()
        completed = [m for m in selected if sum(r['status'] == 'ok' and r.get('method') == m for r in records) == len(samples)]
        (OUT / 'run_status.json').write_text(json.dumps({'configured':selected,'completed_all_images':completed,
            'five_completed':len(completed)>=5,'accuracy_winner':None,'note':'No automatic winner; count agreement alone is not correct segmentation'},indent=2))
        print('Completed all images:',completed)
        if len(completed) < 5: print('WARNING: fewer than five methods completed. Inspect error files; failures were NOT replaced by zero predictions.')
        '''),
        code('''
        display(pd.json_normalize(records))
        for sample in samples:
            columns = 3; count = len(selected)+1
            fig,axes = plt.subplots((count+columns-1)//columns,columns,figsize=(17,6*((count+columns-1)//columns)),squeeze=False)
            for axis in axes.flat: axis.axis('off')
            with Image.open(sample['path']) as original: axes.flat[0].imshow(ImageOps.exif_transpose(original).convert('RGB'))
            axes.flat[0].set_title(Path(sample['path']).name + ' | original')
            for axis,method in zip(list(axes.flat)[1:],selected):
                row = next((r for r in records if r['method'] == method and r.get('image_index') == sample['index']),None)
                if row and row['status'] == 'ok':
                    with Image.open(OUT / row['overlay']) as view: axis.imshow(view)
                    axis.set_title(f"{method}: {row['kept_masks']} masks | {row['elapsed_s']:.2f}s")
                else:
                    axis.text(.05,.5,'Failed/unavailable: inspect error report',wrap=True); axis.set_title(method)
            plt.tight_layout(); plt.savefig(OUT / f"comparison_image_{sample['index']:02d}.png",dpi=120); plt.show()
        print('Predicted IoU / detector confidence are not shared accuracy scales.')
        print('Count error is only a diagnostic. Mask P/R/F1@IoU0.5 requires reviewed per-instance ground truth; no mAP claimed.')
        '''),
        md('''
        ## Optional ground-truth evaluation and interpreting results
        To measure mask precision/recall/F1 at IoU 0.5, provide a JSON file mapping
        image filenames to lists of binary PNG mask paths relative to that JSON.
        Example: `{"image_test_1.jpg":["masks/part_1.png","masks/part_2.png"]}`.
        Label EVERY visible target instance, not just two. Each mask must match the
        full original image dimensions; nonzero pixels mean visible target surface.
        Missing filename = unlabelled scene; explicit [] = a verified empty scene.
        Holes stay background. Occluded surfaces are not invented. Supply masks before
        running the comparison; the notebook never creates ground truth from predictions.

        Inspect whole-part coverage, merged/fragmented instances and background masks.
        Lowering thresholds adds candidates, not necessarily correct pieces. If using
        a workspace crop, ensure it includes all annotated target pieces for fair recall.
        Latency excludes loading/downloads, warmup and export, but includes both detector
        and SAM for two-stage methods. Three images do NOT establish production performance.
        If GPU memory is insufficient: try SAM2_ID='facebook/sam2.1-hiera-small' for ALL
        methods using SAM; lower POINTS_PER_BATCH; rerun as a new comparison.
        SAM 3 requires approval at https://huggingface.co/facebook/sam3 and a token in
        Colab Secrets named HF_TOKEN. No secrets are saved to exported reports.
        '''),
        code('''
        shutil.copy2(RUN / 'environment.txt',OUT / 'requirements-colab.txt')
        shutil.copytree(RUNTIME / 'roboqc',OUT / 'runtime/roboqc')
        (OUT / 'README.txt').write_text('Pretrained development comparison, not an industrial detector bundle. No training. All weights use local MODEL_CACHE/SAM2_CACHE and disappear on runtime reset; results stay in the configured run folder. Cache recovery preserves previous files. Model revisions/checksums/paths recorded where available. No accuracy claims without supplied masks. Verify user image and model redistribution rights.')
        archive = shutil.make_archive(str(OUT),'zip',OUT)
        print('Comparison ZIP (reports, images, masks and runtime; no weights):',archive)
        ''' )])


def vision_notebooks():
    write("04_prepare_vision_dataset.ipynb", [
        md("""
        # 04 · Real industrial instance dataset preparation
        No dataset yet? Start with 04a_try_yoloe_pretrained.ipynb instead.
        It includes automatically downloaded demo photos; 06 also supports those photos.
        This notebook is the optional annotated-data route for later fine-tuning.
        Classes: bolt, nut, washer, bracket. Annotate individual visible instance masks,
        including occluded objects where boundaries are supported by evidence.
        No synthetic shape demo is represented as an industrial training dataset.
        Capture at least three sessions with different layouts/lighting; include every class
        in each split. Add more sessions if grouped splitting cannot achieve coverage.

        COCO: images need `session_id`; annotation `attributes` may contain `color`
        (`red/green/blue/unknown`) and `quality` (`good/defect_1/defect_2/unknown`).
        YOLO: supply sessions.json mapping relative image paths to session IDs, and optional
        attributes.json mapping image paths to lists of attributes in label-line order.
        Define defects only after choosing visible, repeatable criteria for actual parts.
        """), config("FORMAT = 'coco'\nCOCO_JSON = 'annotations.json'\nIMAGE_ROOT = ''  # Optional absolute folder against which COCO file_name resolves\nYOLO_YAML = 'data.yaml'\nSESSIONS_JSON = 'sessions.json'\nATTRIBUTES_JSON = ''\nQUALITY_DEFINITIONS = {}  # e.g. {'good': '...', 'defect_1': '...', 'defect_2': '...'}\nAUGMENT_COPIES = 1"),
        setup(VISION), bootstrap(True),
        code("""
        from roboqc.industrial.vision_data import import_coco, import_yolo, prepare, augment_training
        source = Path(DATASET_PATH) if DATASET_PATH else upload_folder()
        if FORMAT == 'coco':
            matches = list(source.rglob(COCO_JSON))
            if len(matches) != 1: raise ValueError('Set COCO_JSON to one real annotation filename')
            annotation = matches[0]
            records = import_coco(Path(IMAGE_ROOT) if IMAGE_ROOT else annotation.parent, annotation)
        elif FORMAT == 'yolo':
            matches = list(source.rglob(YOLO_YAML))
            if len(matches) != 1: raise ValueError('Select one YOLO dataset')
            root = matches[0].parent
            records = import_yolo(root, matches[0], root / SESSIONS_JSON, root / ATTRIBUTES_JSON if ATTRIBUTES_JSON else None)
        else:
            raise ValueError('FORMAT must be coco or yolo')
        DATA = RUN / 'vision_dataset'
        manifest = prepare(records, DATA, QUALITY_DEFINITIONS, SEED)
        manifest = augment_training(DATA, AUGMENT_COPIES, SEED)
        print('Dataset:', DATA, '| images:', len(manifest))
        """),
        code("""
        import cv2, pandas as pd, matplotlib.pyplot as plt
        counts = pd.DataFrame([{'split': r['split'], 'class': i['class'], 'quality': i['quality']} for r in manifest for i in r['instances']])
        display(pd.crosstab(counts['class'],counts['split']))
        display(pd.crosstab(counts['quality'],counts['split']))
        for record in random.sample(manifest,min(6,len(manifest))):
            frame = cv2.imread(str(DATA / record['image']))
            view = frame.copy()
            for instance in record['instances']:
                mask = cv2.imread(str(DATA / instance['mask_path']),0) > 0
                view[mask] = (.5*view[mask] + .5*np.array([40,200,60])).astype(np.uint8)
            plt.figure(figsize=(8,5)); plt.imshow(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)); plt.title(record['split'] + ': ' + record['image']); plt.axis('off'); plt.show()
        warnings = json.loads((DATA / 'conversion_warnings.json').read_text())
        print('Polygon conversion warnings:', len(warnings), '(see conversion_warnings.json)')
        (DATA / 'preparation_report.json').write_text(json.dumps({'seed':SEED,'images':len(manifest),'conversion_warnings':len(warnings),'augmentation_copies':AUGMENT_COPIES},indent=2))
        archive = archive_export(DATA)
        """)])

    write("05_train_segmentation.ipynb", [
        md("""
        # 05 · Lightweight industrial instance segmentation
        Optional: skip this notebook when using the pretrained bundle from notebook 04a.
        Train YOLO11n-seg on real masks. GPU recommended; CPU is supported but slow.
        This notebook stops if notebook 04's validated dataset is missing.
        Hue-changing augmentation is disabled to preserve red/green/blue semantics.
        MobileNetV3 visible-quality training runs only with criteria and all labels in all splits.
        Original geometry heuristics are not defect labels for nuts/washers/brackets.
        Export includes CPU ONNX, original weights, classes, thresholds and runtime source.
        Ultralytics licensing applies; review it before distributing a closed product.
        """), config("EPOCHS = 80\nIMGSZ = 640\nTRAIN_QUALITY = True\nQUALITY_EPOCHS = 20"), setup(VISION), bootstrap(True),
        code("""
        source = Path(DATASET_PATH) if DATASET_PATH else upload_folder()
        manifests = list(source.rglob('instances.json'))
        if len(manifests) != 1: raise ValueError('Choose notebook 04 dataset ZIP/folder')
        DATA = manifests[0].parent
        assert (DATA / 'data.yaml').exists() and (DATA / 'quality_definitions.json').exists()
        OUT = RUN / 'vision_bundle'
        OUT.mkdir(exist_ok=bool(RESUME_RUN))
        from roboqc.industrial.vision_training import train_segmentation, train_quality
        if not GPU: print('CPU detected. Consider EPOCHS=3 for a workflow check; this is not a useful trained model.')
        if not (OUT / 'vision_manifest.json').exists() or not json.loads((OUT / 'vision_manifest.json').read_text()).get('export_verified'):
            summary = train_segmentation(DATA,OUT,EPOCHS,SEED,IMGSZ,bool(RESUME_RUN))
            print(summary)
        if TRAIN_QUALITY:
            print(train_quality(DATA,OUT,epochs=QUALITY_EPOCHS,seed=SEED))
        """),
        code("""
        import shutil
        assert json.loads((OUT / 'vision_manifest.json').read_text()).get('export_verified'), 'Complete export verification before packaging'
        shutil.copy2(RUN / 'environment.txt',OUT / 'requirements-colab.txt')
        shutil.copy2(DATA / 'quality_definitions.json',OUT / 'quality_definitions.json')
        local_requirements(OUT,['ultralytics','onnxruntime','opencv-python','numpy','torch','torchvision'])
        print((OUT / 'vision_manifest.json').read_text())
        archive = archive_export(OUT)
        print('Keep this export alongside, not over, existing models.')
        """)])

    write("06_evaluate_vision.ipynb", [
        md("""
        # 06 · Held-out industrial vision evaluation
        Default unlabelled mode: load a pretrained bundle from 04a and automatically
        download the same five credited demo photos. Set USE_DEMO_IMAGES=False to upload
        your own photos, or set IMAGE_PATH to a Drive image/folder (overrides demos).
        Photos: Sbrennan83, CC BY-SA 4.0 (fasteners); Fructibus, CC0 (angle brackets);
        Robbie Sproule, CC BY 2.0 (amplifier bracket); Best Tech Nick 25, CC0 (channel connector).
        The amplifier photo is a stereo pair; the connector is assembled, not loose.
        Source and license links are saved in ROOT/demo_images/ATTRIBUTION.txt.
        Shared cached photos are reused; downloads retry transient errors. A warning
        and download_report.json identify any missing photos; a nonempty verified
        subset can still be inspected. No photos available is an error.
        It reports overlays and CPU latency; accuracy/mAP remain unmeasured without labels.
        Labelled mode: load notebook 04 dataset and notebook 05 (or 04a) export. Test is never used for training,
        early stopping, calibration or threshold selection. No training occurs here.
        Masks, centers, planar principal axes and grasp candidates are image estimates.
        Symmetric objects have no unique planar orientation. Holes lost in YOLO polygons
        and occlusion can invalidate grasp candidates; no physical grasp is certified.
        """), config("EVALUATION_MODE = 'unlabelled'  # 'labelled' uses the prepared dataset\nUSE_DEMO_IMAGES = True  # Download credited real sample photos in unlabelled mode\nIMAGE_PATH = ''  # Nonempty path overrides demo photos; otherwise False above opens upload\nCONFIDENCE = 0.25\nMAX_IMAGES = 50\nCPU_THREADS = 2\nCUSTOM_IMAGE = ''  # Optional unseen image path on Drive\nMAX_VISUALIZATIONS = 10"), setup(VISION), bootstrap(True),
        code("""
        import yaml, time, cv2, pandas as pd, matplotlib.pyplot as plt
        from roboqc.industrial.vision_inference import Segmenter, draw_instances
        from roboqc.industrial.yoloe import find_images, inspect_images, download_demo_images
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix, classification_report
        if EVALUATION_MODE not in ['unlabelled','labelled']:
            raise ValueError('EVALUATION_MODE must be unlabelled or labelled')
        torch.set_num_threads(CPU_THREADS)
        source = Path(MODEL_PATH) if MODEL_PATH else upload_folder()
        manifests = [source / 'vision_manifest.json'] if (source / 'vision_manifest.json').is_file() else list(source.rglob('vision_manifest.json'))
        if len(manifests) != 1: raise ValueError('Select one exported vision bundle')
        BUNDLE = manifests[0].parent
        segmenter = Segmenter(BUNDLE,confidence=CONFIDENCE)
        if EVALUATION_MODE == 'unlabelled':
            source = Path(IMAGE_PATH).resolve() if IMAGE_PATH else download_demo_images(Path(ROOT) / 'demo_images') if USE_DEMO_IMAGES else upload_folder()
            images = find_images(source,MAX_IMAGES)
            report,rows = inspect_images(BUNDLE,images,RUN / 'unlabelled_results',CONFIDENCE,CPU_THREADS,MAX_VISUALIZATIONS)
            for row in rows:
                if row['overlay']:
                    view = cv2.imread(str(RUN / 'unlabelled_results' / row['overlay']))
                    plt.figure(figsize=(10,6)); plt.imshow(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
            display(pd.json_normalize(report))
            print('Accuracy/mAP not measured: no ground-truth annotations supplied.')
        """),
        labelled("""
        source = Path(DATASET_PATH) if DATASET_PATH else upload_folder()
        datasets = [source / 'instances.json'] if (source / 'instances.json').is_file() else list(source.rglob('instances.json'))
        if len(datasets) != 1: raise ValueError('Select one prepared vision dataset')
        DATA = datasets[0].parent
        dataset_config = yaml.safe_load((DATA / 'data.yaml').read_text())
        dataset_config['path'] = str(DATA.resolve())
        evaluation_yaml = RUN / 'evaluation_data.yaml'
        evaluation_yaml.write_text(yaml.safe_dump(dataset_config))
        val = segmenter.model.val(data=str(evaluation_yaml),split='test',device='cpu',imgsz=segmenter.metadata.get('imgsz',640),project=str(RUN),name='test_metrics')
        report = {'segmentation': {k:float(v) for k,v in val.results_dict.items()},
                  'per_class': {'names':val.names,'box_maps':val.box.maps.tolist(),'mask_maps':val.seg.maps.tolist()}}
        records = [r for r in json.loads((DATA / 'instances.json').read_text()) if r['split'] == 'test']
        assert records
        segmenter.predict(cv2.imread(str(DATA / records[0]['image'])))  # warmup
        """),
        labelled("""
        rows, timings, truth_types, pred_types, qualities, colors = [], [], [], [], [], []
        for number, record in enumerate(records):
            frame = cv2.imread(str(DATA / record['image']))
            start = time.perf_counter(); objects = segmenter.predict(frame); timings.append((time.perf_counter()-start)*1000)
            gold = record['instances']
            masks = [cv2.imread(str(DATA / a['mask_path']),0) > 0 for a in gold]
            overlaps = np.array([[np.logical_and(m,o['mask']).sum()/max(1,np.logical_or(m,o['mask']).sum()) for o in objects] for m in masks])
            matches = []
            if gold and objects:
                a,b = linear_sum_assignment(-overlaps)
                matches = [(int(i),int(j),float(overlaps[i,j])) for i,j in zip(a,b) if overlaps[i,j] >= .5]
            for i,j,iou in matches:
                truth_types.append(gold[i]['class']); pred_types.append(objects[j]['type'])
                if gold[i]['quality'] != 'unknown': qualities.append((gold[i]['quality'],objects[j]['quality']))
                if gold[i].get('color') in ['red','green','blue']: colors.append((gold[i]['color'],objects[j]['color']))
            serializable = [{k:v for k,v in obj.items() if k != 'mask'} for obj in objects]
            rows.append({'image':record['image'],'instances':serializable,'mask_matches':matches,'latency_ms':timings[-1]})
            if number < MAX_VISUALIZATIONS:
                view = draw_instances(frame,objects)
                cv2.imwrite(str(RUN / f'prediction_{number:03d}.png'),view)
                plt.figure(figsize=(10,6)); plt.imshow(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
        report['latency_ms'] = {'p50':float(np.percentile(timings,50)),'p95':float(np.percentile(timings,95)),'mean':float(np.mean(timings))}
        report['matched_type_classification'] = classification_report(truth_types,pred_types,output_dict=True,zero_division=0) if truth_types else {'status':'no_matches'}
        for key,values in [('quality',qualities),('color',colors)]:
            report[key] = {'matched_annotated_instances':len(values),'accuracy':sum(a==b for a,b in values)/len(values)} if values else {'status':'pending','reason':'No matched instances with reference labels'}
        if truth_types:
            labels = sorted(set(truth_types+pred_types))
            matrix = confusion_matrix(truth_types,pred_types,labels=labels)
            plt.imshow(matrix); plt.xticks(range(len(labels)),labels); plt.yticks(range(len(labels)),labels); plt.title('Classes at mask IoU >= 0.5'); plt.savefig(RUN / 'class_confusion.png'); plt.show()
        (RUN / 'vision_report.json').write_text(json.dumps(report,indent=2))
        (RUN / 'predictions.json').write_text(json.dumps(rows,indent=2))
        pd.json_normalize(rows).to_csv(RUN / 'vision_predictions.csv',index=False)
        display(pd.json_normalize(report))
        """),
        code("""
        if CUSTOM_IMAGE:
            frame = cv2.imread(CUSTOM_IMAGE)
            if frame is None: raise FileNotFoundError(CUSTOM_IMAGE)
            objects = segmenter.predict(frame)
            plt.figure(figsize=(12,7)); plt.imshow(cv2.cvtColor(draw_instances(frame,objects),cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
            print(json.dumps([{k:v for k,v in o.items() if k != 'mask'} for o in objects],indent=2))
        archive = shutil.make_archive(str(RUN.parent / (RUN.name + '_vision_evaluation')),'zip',RUN)
        print('Evaluation report:',archive)
        """)])


def sam3_notebook():
    payload = {'roboqc/__init__.py':'','roboqc/industrial/__init__.py':''}
    for name in ['vision_inference.py','sam_masks.py','pretrained_compare.py','sam3.py']:
        payload['roboqc/industrial/' + name] = (DEMO / 'phase_1/src/roboqc/industrial' / name).read_text(encoding='utf-8')
    write('04d_try_sam3_pretrained.ipynb',[
        md('''
        # 04d · SAM 3 only — CPU or CUDA industrial part segmentation

        No training. Uses **facebook/sam3** through Transformers, not SAM 3.1 or SAM 2.1.
        Each image and concept prompt is independent; prompt outputs are not merged.

        Use a **fresh Colab CPU or GPU runtime**. No A100 requirement: T4 and other
        CUDA GPUs use float16; CPU uses float32. Native PyTorch SDPA is used,
        with no external FlashAttention/CUDA extensions or source compilation.
        Allow several GB of local disk and sufficient host/GPU memory. CPU can be
        very slow; start with one image and one prompt. Hardware support is not
        a promise of acceptable speed or memory use on every machine.
        No dataset/reference photos are bundled. Set IMAGE_PATH to a Drive image/folder
        or leave it blank to upload your own photos.

        Add a Hugging Face read token to **Colab Secrets → HF_TOKEN**, enable notebook
        access, and ensure that same account is approved for **facebook/sam3**.
        Approval for facebook/sam3.1 alone may not grant access to this repository.
        Tokens are never printed or included in exports. No approval bypass or model fallback.

        Sources: [checkpoint/access](https://huggingface.co/facebook/sam3),
        [Transformers SAM 3](https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/sam3).
        Model and supplied image redistribution rights must be checked separately.
        '''),
        config("IMAGE_PATH = ''  # Drive image/folder; blank uploads one or more photos\nPROMPTS = ['metal bracket','sheet metal part','metal mounting clip']\nROI_NORM = None  # Optional [left,top,right,bottom] fractions\nMODEL_CACHE = '/content/roboqc_sam3_cache'  # Local disk, never Drive\nMODEL_REVISION = 'main'  # Resolved checkpoint revision recorded\nDEVICE_CHOICE = 'auto'  # 'auto', 'cpu', 'cuda'; CPU=float32, CUDA=float16\nCONFIDENCE = 0.40\nMIN_MASK_AREA = 100\nMAX_MASK_FRACTION = 0.85\nCUSTOM_IMAGE = ''  # Optional additional image for the final interactive cell\nCUSTOM_PROMPT = 'metal bracket'"),
        md('''
        Installation uses a pinned Transformers release and the existing compatible
        PyTorch when available. No Meta source package or NumPy metadata workaround
        is required. Do not mix with an earlier 04b/04c/SAM 3.1 session.
        Installation errors retain full pip output and a saved log.
        '''),
        setup(['numpy>=1.26,<3','matplotlib>=3.9,<4','Pillow>=10,<13',
            'opencv-python-headless>=4.10,<6','pandas>=2.2,<4','scipy>=1.13,<2',
            'torch>=2.5.1,<3','torchvision>=0.20.1,<1','transformers==5.2.0',
            'huggingface-hub>=1.3,<2','safetensors>=0.5,<1']),
        code('''
        if DEVICE_CHOICE not in ['auto','cpu','cuda']: raise ValueError('Invalid DEVICE_CHOICE')
        if DEVICE_CHOICE == 'cuda' and not GPU: raise ValueError('CUDA unavailable; use auto or cpu')
        if DEVICE_CHOICE != 'auto': DEVICE = DEVICE_CHOICE
        GPU = DEVICE == 'cuda'
        print('Selected device:',DEVICE,'| dtype:', 'float16' if GPU else 'float32')
        if not GPU: print('CPU inference can be slow and needs several GB of RAM; start with one image/prompt.')
        CACHE = Path(MODEL_CACHE).resolve()
        if CACHE.is_relative_to(Path('/content/drive').resolve()):
            raise ValueError('Keep MODEL_CACHE on local Colab disk, not the Drive mount.')
        CACHE.mkdir(parents=True,exist_ok=True)
        os.environ['HF_HOME'] = str(CACHE / 'huggingface')
        from huggingface_hub import login
        from getpass import getpass
        token = os.environ.get('HF_TOKEN')
        if not token:
            try:
                from google.colab import userdata
                token = userdata.get('HF_TOKEN')
            except Exception:
                pass
        if not token: token = getpass('Hugging Face read token (hidden): ')
        if not token: raise ValueError('HF_TOKEN is required for the approved checkpoint.')
        login(token=token,add_to_git_credential=False)
        del token
        (RUN / 'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        '''),
        code((HERE / 'image_inputs.py').read_text(encoding='utf-8')),
        code('''
        import json, shutil, time, traceback
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image, ImageOps
        SOURCES = SOURCE_PAYLOAD
        RUNTIME = Path('/content/roboqc_sam3_runtime')
        for relative,source in SOURCES.items():
            target = RUNTIME / relative; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source,encoding='utf-8')
        sys.path.insert(0,str(RUNTIME))
        from roboqc.industrial.sam3 import load_predictor, predict_image
        from roboqc.industrial.pretrained_compare import crop_image, restore_masks
        from roboqc.industrial.sam_masks import save_candidates
        if not PROMPTS or any(not isinstance(p,str) or not p.strip() for p in PROMPTS):
            raise ValueError('Use nonempty text prompts')
        if not 0 <= CONFIDENCE <= 1: raise ValueError('Invalid confidence threshold')
        if MIN_MASK_AREA < 1 or not 0 < MAX_MASK_FRACTION <= 1: raise ValueError('Invalid mask area filters')
        paths = select_image_paths(IMAGE_PATH,RUN)
        print('Images:',[p.name for p in paths],'| independent concept prompts:',PROMPTS)
        '''.replace('SOURCE_PAYLOAD',repr(payload))),
        code('''
        # Rerunning creates a NEW output folder; previous attempts remain intact.
        OUT = RUN / ('sam3_' + uuid.uuid4().hex); OUT.mkdir()
        settings = {'model_id':'facebook/sam3','model_revision':MODEL_REVISION,
            'seed':SEED,'prompts':PROMPTS,'roi_norm':ROI_NORM,'confidence':CONFIDENCE,
            'minimum_area':MIN_MASK_AREA,'maximum_fraction':MAX_MASK_FRACTION,
            'device':DEVICE,'dtype':'float16' if GPU else 'float32',
            'gpu':torch.cuda.get_device_name(0) if GPU else None,'torch':torch.__version__,
            'attention':'pytorch_sdpa','compile':False,'mode':'independent_image_prompts',
            'accuracy':'not_measured','weights_in_export':False}
        (OUT / 'settings.json').write_text(json.dumps(settings,indent=2))
        if 'predictor' not in globals() or predictor is None:
            start = time.perf_counter()
            try:
                predictor,provenance = load_predictor(CACHE,MODEL_REVISION,DEVICE)
            except Exception:
                (OUT / 'load_error.txt').write_text(traceback.format_exc())
                raise
            print('Model load including download (seconds):',round(time.perf_counter()-start,2))
            LOADED_CONFIGURATION = (MODEL_REVISION,DEVICE)
        elif LOADED_CONFIGURATION != (MODEL_REVISION,DEVICE):
            raise RuntimeError('Model configuration changed: restart runtime before reloading.')
        (OUT / 'provenance.json').write_text(json.dumps(provenance,indent=2))
        '''),
        code('''
        def segment_and_save(path,prompt,parent):
            attempt = parent / ('preview_' + uuid.uuid4().hex); attempt.mkdir(parents=True)
            with Image.open(path) as opened: rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB'))
            crop,roi = crop_image(rgb,ROI_NORM)
            try:
                if GPU: torch.cuda.synchronize()
                start = time.perf_counter()
                annotations = predict_image(predictor,crop,prompt,CONFIDENCE)
                if GPU: torch.cuda.synchronize()
                seconds = time.perf_counter()-start
            except Exception:
                (attempt / 'error.txt').write_text(traceback.format_exc())
                raise
            metadata = [{k:v for k,v in a.items() if k != 'segmentation'} for a in annotations]
            annotations = restore_masks(annotations,rgb.shape,roi)
            fraction = MAX_MASK_FRACTION * crop.shape[0]*crop.shape[1]/(rgb.shape[0]*rgb.shape[1])
            result,_ = save_candidates(rgb,annotations,attempt / 'masks',MIN_MASK_AREA,fraction,SEED)
            Image.fromarray(rgb).save(attempt / 'input.png')
            (attempt / 'object_metadata.json').write_text(json.dumps(metadata,indent=2))
            record = {'image':Path(path).name,'input_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'prompt':prompt,'status':'ok','seconds_including_pre_and_postprocessing':seconds,
                'raw_masks':result['raw_masks'],'kept_masks':result['kept_candidates'],
                'roi':list(roi),'accuracy':'not_measured','overlay':str((attempt / 'masks/overlay.png').relative_to(OUT))}
            (attempt / 'result.json').write_text(json.dumps(record,indent=2))
            return record
        records = []
        for index,path in enumerate(paths):
            for prompt_index,prompt in enumerate(PROMPTS):
                parent = OUT / f'image_{index:02d}_prompt_{prompt_index:02d}'
                try:
                    row = segment_and_save(path,prompt,parent)
                except Exception as exc:
                    parent.mkdir(parents=True,exist_ok=True)
                    (parent / 'error.txt').write_text(traceback.format_exc())
                    traceback.print_exc(file=sys.stdout)
                    row = {'image':path.name,'prompt':prompt,'status':'error','error':f'{type(exc).__name__}: {exc}'}
                records.append(row)
                (OUT / 'report.json').write_text(json.dumps(records,indent=2))
                pd.DataFrame(records).to_csv(OUT / 'report.csv',index=False)
                print(row,flush=True)
        '''),
        code('''
        display(pd.DataFrame(records))
        for path in paths:
            matches = [row for row in records if row['image'] == path.name]
            fig,axes = plt.subplots(1,len(matches)+1,figsize=(6*(len(matches)+1),7),squeeze=False)
            with Image.open(path) as opened: axes[0,0].imshow(ImageOps.exif_transpose(opened).convert('RGB'))
            axes[0,0].set_title(path.name + ' | original'); axes[0,0].axis('off')
            for axis,row in zip(axes[0,1:],matches):
                axis.axis('off'); axis.set_title(row['prompt'])
                if row['status'] == 'ok':
                    with Image.open(OUT / row['overlay']) as view: axis.imshow(view)
                    axis.set_title(f"{row['prompt']} | {row['kept_masks']} masks")
                else: axis.text(.05,.5,row['error'],wrap=True)
            plt.tight_layout(); plt.show()
        print('Scores are model confidence, not observed mask accuracy. No ground truth or mAP is claimed.')
        print('Inspect each physical part: holes, attached pins, touching pieces and occlusion remain difficult.')
        '''),
        md('''
        ## Optional custom image/prompt

        Set `CUSTOM_IMAGE` to a Drive photo and `CUSTOM_PROMPT` to a short English
        concept, then run the next cell. Each attempt creates a new folder. No model
        retraining or weight download is needed while the loaded predictor stays alive.
        For another uploaded/Drive image, use a path printed in the image-selection cell.
        '''),
        code('''
        if CUSTOM_IMAGE:
            row = segment_and_save(Path(CUSTOM_IMAGE),CUSTOM_PROMPT,OUT / 'custom')
            print(row)
            with Image.open(OUT / row['overlay']) as view:
                plt.figure(figsize=(10,8)); plt.imshow(view); plt.axis('off'); plt.show()
        '''),
        code('''
        # Bundle results and inference helper source, not large gated model weights or tokens.
        shutil.copy2(RUN / 'environment.txt',OUT / 'requirements-colab.txt')
        if not (OUT / 'runtime').exists(): shutil.copytree(RUNTIME / 'roboqc',OUT / 'runtime/roboqc')
        (OUT / 'README.txt').write_text('SAM 3 only via Transformers; CPU=float32, CUDA=float16, actual device recorded. Independent image and prompt experiments. No training, no accuracy claims without reviewed masks, no robot execution. Raw masks and filtered PNGs preserve visible holes. Geometry is planar and grasp candidates are unvalidated. Gated weights stay local, are excluded from this archive and disappear on runtime reset. Check image/model redistribution rights.')
        archive = shutil.make_archive(str(OUT.parent / (OUT.name + '_export_' + uuid.uuid4().hex)),'zip',OUT)
        print('Results ZIP:',archive)
        print('No masks are guaranteed safe for robot execution. No hidden geometry or quality defects are inferred.')
        ''')])


if __name__ == "__main__":
    voice_notebooks()
    if (DEMO / 'phase_1/src/roboqc/industrial/vision_training.py').exists():
        yoloe_notebook()
        sam_notebook()
        comparison_notebook()
        sam3_notebook()
        runpy.run_path(str(HERE / 'build_review_notebook.py'))['build']()
        vision_notebooks()
