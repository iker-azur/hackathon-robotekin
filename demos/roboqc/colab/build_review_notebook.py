"""Build 04e JSON only. Never execute notebook cells or model inference."""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent


def build():
    helpers = runpy.run_path(str(HERE / "build_notebooks.py"))
    md, code, config, setup, write = [helpers[name] for name in ["md", "code", "config", "setup", "write"]]
    runtime = HERE.parent / "phase_1/src/roboqc/industrial"
    payload = {"roboqc/__init__.py": "", "roboqc/industrial/__init__.py": ""}
    for name in ["vision_inference.py", "sam_masks.py", "pretrained_compare.py", "sam3.py", "review_bundle.py"]:
        payload["roboqc/industrial/" + name] = (runtime / name).read_text(encoding="utf-8")
    write("04e_export_segmentation_review.ipynb", [
        md('''
        # 04e · Six-method proposals → human-review ZIP

        Combines the five methods in 04c with the portable **facebook/sam3** adapter
        from 04d. No training, automatic winner selection, forced object count or
        robot execution. The output is a **proposal bank, not ground truth**.

        Start a **fresh Colab runtime**, preferably GPU. CPU is supported but slow.
        One model pipeline is loaded at a time; RAM/VRAM needs still apply. SAM 3
        uses CPU float32 / CUDA float16, not SAM 3.1 or an A100-only implementation.
        Add your approved Hugging Face read token as the Colab secret **HF_TOKEN**.

        ## Input: your complete `/data` tree, including new photos and subfolders

        On your PC run `python demos/roboqc/colab/prepare_04e_inputs.py` from the
        project root. This packages existing images only (no model execution) as
        `demos/roboqc/colab/assets/data_04e_inputs.zip`. Upload that ZIP in the input
        cell, or put it in Drive and set `INPUT_PATH` to its full Drive path.
        An existing Drive image folder also works. Recursive discovery includes
        uppercase extensions. Relative paths and originals are preserved; identical
        filenames in different folders cannot overwrite one another.

        Empty `INPUT_PATH` opens an upload, saves a new copy of the input ZIP in
        Drive, then reads that Drive copy. On later sessions use the printed path.
        Input ZIPs are copied/extracted locally for speed; weights stay local too.
        Outputs/checkpoints and the final ZIP are stored in Drive.

        All images run by default. With hundreds of images and multiple prompts,
        this can take hours and produce a large archive. `MAX_IMAGES=3` is a useful
        smoke test; set it back to `None` for the full batch. No image bytes are
        embedded in this notebook. Check data/model redistribution rights.
        '''),
        config("INPUT_PATH = ''  # Drive ZIP or image folder; ignored when resuming\nMETHODS = ['yoloe_text','sam2_auto','grounding_dino_sam2','owlv2_sam2','florence2_sam2','sam3_text']\nPROMPTS = ['metal bracket','sheet metal part','metal mounting clip']\nROI_NORM = None  # Optional common [left,top,right,bottom] fractions\nMAX_IMAGES = None  # None = ALL recursively discovered images\nDEVICE_CHOICE = 'auto'  # auto, cpu, cuda\nMODEL_CACHE = '/content/roboqc_review_cache'  # Never Drive\nSAM2_ID = 'facebook/sam2.1-hiera-large'  # small variant can reduce memory\nSAM2_COMMIT = '2b90b9f5ceec907a1c18123530e92e794ad901a4'\nMODEL_REVISIONS = {}  # HF model ID -> pinned revision; resolved revisions exported\nYOLOE_CHECKPOINT = 'yoloe-26l-seg.pt'\nIMGSZ = 960\nSAM3_THRESHOLD = 0.40\nMIN_MASK_AREA = 100\nMAX_MASK_FRACTION = 0.85\nRESUME_BUNDLE = 'auto'  # 'auto': most successful compatible bundle under ROOT; '' = NEW; or exact folder\nEXPORT_ONLY = False  # True = export saved results without model loading/inference\nEXPORT_LOCAL_DIR = '/content/roboqc_review_exports'  # ZIP staging on local disk, never Drive\nPREVIEW_IMAGES = 3\nMAX_EXTRACT_GB = 10\nMAX_EXTRACT_FILES = 50000"),
        setup(['numpy>=1.26,<3','torch>=2.5.1,<3','torchvision>=0.20.1,<1',
               'transformers==5.2.0','huggingface-hub>=1.3,<2','safetensors>=0.5,<1',
               'ultralytics>=8.4.0,<9','matplotlib>=3.9,<4','Pillow>=10,<13',
               'opencv-python-headless>=4.10,<6','pandas>=2.2,<4','scipy>=1.13,<2',
               'hydra-core>=1.3.2,<2','iopath>=0.1.10,<1','tqdm>=4.66,<5']),
        code('''
        import gc, hashlib, json, shutil, time, traceback
        import pandas as pd
        from PIL import Image, ImageOps
        from tqdm.auto import tqdm
        if not USE_DRIVE: raise ValueError('04e requires Drive for persistent inputs and exports.')
        if DEVICE_CHOICE not in ['auto','cpu','cuda']: raise ValueError('Invalid DEVICE_CHOICE')
        if DEVICE_CHOICE == 'cuda' and not GPU: raise ValueError('CUDA is unavailable')
        if DEVICE_CHOICE != 'auto': DEVICE = DEVICE_CHOICE
        GPU = DEVICE == 'cuda'
        CACHE = Path(MODEL_CACHE).resolve()
        if CACHE.is_relative_to(Path('/content/drive').resolve()): raise ValueError('Weights must stay on local disk')
        CACHE.mkdir(parents=True,exist_ok=True)
        os.environ['HF_HOME'] = str(CACHE / 'huggingface')
        os.environ['SAM2_BUILD_CUDA'] = '0'
        if not EXPORT_ONLY and any(m == 'sam2_auto' or m.endswith('_sam2') for m in METHODS):
            command = [sys.executable,'-m','pip','install','--no-build-isolation',
                       'git+https://github.com/facebookresearch/sam2.git@' + SAM2_COMMIT]
            with install_log.open('a',encoding='utf-8') as log:
                process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in process.stdout: print(line,end='',flush=True); log.write(line)
                if process.wait(): raise RuntimeError(f'SAM 2 installation failed; see {install_log}')
        if not EXPORT_ONLY and 'sam3_text' in METHODS:
            from huggingface_hub import login
            from getpass import getpass
            token = os.environ.get('HF_TOKEN')
            if not token:
                try:
                    from google.colab import userdata
                    token = userdata.get('HF_TOKEN')
                except Exception: pass
            if not token: token = getpass('Approved facebook/sam3 read token (hidden): ')
            if not token: raise ValueError('SAM 3 requires an approved account/token')
            login(token=token,add_to_git_credential=False)
            del token
        print('Device:',DEVICE,'| SAM 3 dtype:', 'float16' if GPU else 'float32')
        if not GPU: print('WARNING: CPU batch inference may be very slow.')
        '''),
        code('''
        SOURCES = SOURCE_PAYLOAD
        RUNTIME = Path('/content/roboqc_review_runtime')
        for relative,source in SOURCES.items():
            target = RUNTIME / relative; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source,encoding='utf-8')
        sys.path.insert(0,str(RUNTIME))
        from roboqc.industrial.review_bundle import image_files, extract_images, register_images, write_json, archive_bundle
        from roboqc.industrial.review_bundle import discover_bundles, settings_compatible, expected_proposals
        from roboqc.industrial.pretrained_compare import DEFAULT_METHODS, PretrainedMethod, crop_image, restore_masks
        from roboqc.industrial.sam3 import load_predictor, predict_image
        from roboqc.industrial.sam_masks import save_candidates
        selected = list(dict.fromkeys(METHODS))
        if not selected or any(m not in DEFAULT_METHODS + ['sam3_text'] for m in selected): raise ValueError('Unknown/empty METHODS')
        if not PROMPTS or any(not isinstance(p,str) or not p.strip() for p in PROMPTS): raise ValueError('Empty prompt')
        if MAX_IMAGES is not None and (not isinstance(MAX_IMAGES,int) or MAX_IMAGES < 1): raise ValueError('MAX_IMAGES must be positive or None')
        if MIN_MASK_AREA < 1 or not 0 < MAX_MASK_FRACTION <= 1 or not 0 <= SAM3_THRESHOLD <= 1:
            raise ValueError('Invalid mask filters/confidence')
        # Other thresholds match 04c. Edit here before starting a new bundle.
        cfg = {'cache':str(CACHE),'sam2_cache':str(CACHE / 'sam2'),'revisions':MODEL_REVISIONS,
            'sam2_id':SAM2_ID,'yoloe_checkpoint':YOLOE_CHECKPOINT,'prompts':PROMPTS,'imgsz':IMGSZ,
            'yoloe_threshold':.25,'dino_threshold':.25,'dino_text_threshold':.20,'owl_threshold':.10,
            'box_nms':.70,'max_detections':50,'points_per_side':32,'points_per_batch':8,
            'crop_layers':0,'sam_iou':.80,'sam_stability':.95,'florence_max_tokens':1024}
        settings = {'schema_version':'roboqc-review-1.0','methods':selected,'prompts':PROMPTS,
            'roi_norm':ROI_NORM,'sam3_threshold':SAM3_THRESHOLD,'minimum_area':MIN_MASK_AREA,
            'maximum_fraction':MAX_MASK_FRACTION,'sam2_commit':SAM2_COMMIT,'seed':SEED,
            'max_images':MAX_IMAGES,'config':cfg,'weights_in_export':False,'accuracy':'not_measured',
            'runtime_sha256':{name:hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCES.items()}}
        chosen_bundle = RESUME_BUNDLE
        if chosen_bundle == 'auto':
            with tqdm(desc='Find saved bundles',unit='bundle') as search_progress:
                candidates = discover_bundles(ROOT,settings,
                    progress=lambda phase,done,total:search_progress.update(done-search_progress.n))
            if candidates: display(pd.DataFrame(candidates))
            compatible = [row for row in candidates if row['compatible']]
            chosen_bundle = compatible[0]['path'] if compatible else ''
            if candidates and not compatible:
                raise ValueError('Saved bundles found but none are compatible. Inspect table; do not remove the settings safeguard. Use an exact bundle with matching settings or explicitly set RESUME_BUNDLE="" for a new run.')
            if chosen_bundle: print('Auto-selected most completed compatible bundle:',chosen_bundle)
        if EXPORT_ONLY and not chosen_bundle:
            raise ValueError('EXPORT_ONLY needs an existing compatible bundle. Set RESUME_BUNDLE to auto or an exact folder.')
        if chosen_bundle:
            OUT = Path(chosen_bundle)
            if not (OUT / 'registration_complete.json').exists():
                raise ValueError('Input registration was interrupted. Start a new bundle from the original input ZIP.')
            previous = json.loads((OUT / 'settings.json').read_text())
            if not settings_compatible(previous,settings):
                raise ValueError('Configuration differs from saved bundle. Restore it or start a new bundle.')
            samples = json.loads((OUT / 'images.json').read_text())
            print('Resuming from saved originals; INPUT_PATH is ignored.')
        else:
            if INPUT_PATH:
                source = Path(INPUT_PATH)
            else:
                from google.colab import files
                uploaded = files.upload()
                if len(uploaded) != 1: raise ValueError('Upload exactly one input ZIP')
                name,blob = next(iter(uploaded.items()))
                if Path(name).suffix.lower() != '.zip': raise ValueError('Upload a ZIP of the /data folder')
                source = Path(ROOT) / 'inputs' / (uuid.uuid4().hex + '_' + Path(name).name)
                source.parent.mkdir(parents=True,exist_ok=True)
                with source.open('xb') as handle: handle.write(blob)
                del uploaded, blob
                print('Input saved in Drive; reuse INPUT_PATH =',repr(str(source)))
            if not source.exists(): raise FileNotFoundError(source)
            if source.is_file():
                local_zip = Path('/content') / ('review_input_' + uuid.uuid4().hex + '.zip')
                shutil.copy2(source,local_zip)
                source = extract_images(local_zip,Path('/content') / ('review_images_' + uuid.uuid4().hex),
                    maximum_bytes=int(MAX_EXTRACT_GB*1024**3),maximum_files=MAX_EXTRACT_FILES)
            discovered = image_files(source)
            OUT = RUN / ('review_bundle_' + uuid.uuid4().hex); OUT.mkdir()
            write_json(OUT / 'settings.json',settings)
            # Register ALL input photos, even when MAX_IMAGES limits inference.
            samples = register_images(source,OUT)
            print('Recursively discovered:',len(discovered),'| invalid:',sum(s['status'] != 'ok' for s in samples))
        active = [s for s in samples if s['status'] == 'ok']
        if MAX_IMAGES is not None: active = active[:MAX_IMAGES]
        if not active: raise ValueError('No decodable images')
        print('Bundle:',OUT,'| images to process:',len(active),'| models:',selected)
        print('For recovery, set RESUME_BUNDLE =',repr(str(OUT)))
        '''.replace('SOURCE_PAYLOAD',repr(payload))),
        md('''
        ## Generate proposals (long-running; execute manually)

        Each SAM 3 prompt is a separate alternative. The other methods use the
        configured prompt list together (SAM automatic masks ignore text). Never
        union alternatives blindly: they may duplicate objects or segment subparts.
        No confidence-based cross-model ranking is used. Every completed prediction
        is saved immediately, including zero detections. Errors remain explicit.
        Rerun with `RESUME_BUNDLE` and identical settings to skip successful proposals
        and retry failures into fresh folders. Existing masks/review edits survive.
        `RESUME_BUNDLE='auto'` searches run folders under ROOT, prunes result trees,
        displays compatible/incompatible candidates and chooses the most unique
        successful indexed predictions. Bundles are NOT merged. Verify the selected
        dataset/path; a different image collection can share inference settings.
        Exact paths override auto; `''` explicitly starts a new bundle.
        Export-helper updates are compatible with older 04e runs; changed inference
        source/parameters remain incompatible. Set `EXPORT_ONLY=True` if inference
        is already done or you only want a partial-results ZIP: no model download,
        HF login or inference, and a CPU runtime is sufficient.
        Inference progress includes saved successes and new attempts. Failed attempts
        advance the attempt bar but remain errors, not successful results. ETA is
        approximate across architectures; model download/load can pause the bar.
        '''),
        code('''
        index_path = OUT / 'proposals.json'
        records = json.loads(index_path.read_text()) if index_path.exists() else []
        expected = expected_proposals(settings,samples)
        completed = {r['proposal_id'] for r in records if r['status'] == 'ok'} & expected
        print('Successful:',len(completed),'| remaining:',len(expected)-len(completed),'| total:',len(expected))
        inference_progress = tqdm(total=len(expected),initial=len(completed),desc='Inference attempts',unit='proposal',disable=EXPORT_ONLY)
        for method in ([] if EXPORT_ONLY else selected):
            variants = list(enumerate(PROMPTS)) if method == 'sam3_text' else [(0,None)]
            pending = [(sample,pi,prompt) for sample in active for pi,prompt in variants
                       if f"{sample['image_id']}:{method}:{pi}" not in completed]
            if not pending: continue
            runner = None
            inference_progress.set_postfix_str('Loading ' + method + ' (downloads may take time)')
            try:
                started = time.perf_counter()
                if method == 'sam3_text':
                    runner,provenance = load_predictor(CACHE / 'sam3',MODEL_REVISIONS.get('facebook/sam3','main'),DEVICE)
                else:
                    runner = PretrainedMethod(method,cfg,DEVICE)
                    provenance = runner.provenance
                load_seconds = time.perf_counter()-started
                load_error = None
            except Exception:
                load_error = traceback.format_exc(); print(load_error)
            try:
                for sample,pi,prompt in pending:
                    proposal_id = f"{sample['image_id']}:{method}:{pi}"
                    attempt = OUT / 'proposals' / sample['image_id'] / method / (f'prompt_{pi:02d}_' + uuid.uuid4().hex)
                    attempt.mkdir(parents=True)
                    row = {'proposal_id':proposal_id,'image_id':sample['image_id'],'method':method,
                           'prompt':prompt,'prompts':PROMPTS,'path':attempt.relative_to(OUT).as_posix(),
                           'status':'error','device':DEVICE,'accuracy':'not_measured'}
                    try:
                        if load_error: raise RuntimeError('Model load failed; see traceback below.\\n' + load_error)
                        write_json(attempt / 'provenance.json',provenance)
                        with Image.open(OUT / sample['rgb']) as opened: rgb = np.asarray(opened.convert('RGB'))
                        crop,roi = crop_image(rgb,ROI_NORM)
                        if GPU: torch.cuda.synchronize()
                        started = time.perf_counter()
                        if method == 'sam3_text':
                            annotations = predict_image(runner,crop,prompt,SAM3_THRESHOLD)
                            details = {'score_type':'SAM 3 confidence; not observed accuracy'}
                        else:
                            annotations,details = runner.predict(crop)
                        if GPU: torch.cuda.synchronize()
                        seconds = time.perf_counter()-started
                        annotations = restore_masks(annotations,rgb.shape,roi)
                        fraction = MAX_MASK_FRACTION*crop.shape[0]*crop.shape[1]/(rgb.shape[0]*rgb.shape[1])
                        result,_ = save_candidates(rgb,annotations,attempt / 'masks',MIN_MASK_AREA,fraction,SEED)
                        row.update(status='ok',elapsed_s=seconds,load_seconds_including_downloads=load_seconds,
                            raw_masks=result['raw_masks'],kept_masks=result['kept_candidates'],roi=list(roi),details=details,
                            candidates=(attempt / 'masks/candidates.json').relative_to(OUT).as_posix(),
                            overlay=(attempt / 'masks/overlay.png').relative_to(OUT).as_posix())
                        completed.add(proposal_id)
                    except Exception as exc:
                        row['error'] = f'{type(exc).__name__}: {exc}'
                        (attempt / 'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                        traceback.print_exc(file=sys.stdout)
                    write_json(attempt / 'result.json',row)
                    # Index the latest attempt; old attempt directories are deliberately preserved.
                    records = [r for r in records if r['proposal_id'] != proposal_id] + [row]
                    write_json(index_path,records)
                    pd.json_normalize(records).to_csv(OUT / 'summary.csv',index=False)
                    inference_progress.update(1)
                    inference_progress.set_postfix(method=method,successful=len(completed),errors=sum(r['status'] != 'ok' for r in records))
            finally:
                runner = None
                gc.collect()
                if GPU: torch.cuda.empty_cache()
        inference_progress.close()
        if records: display(pd.DataFrame(records).groupby(['method','status']).size().rename('proposals').reset_index())
        print('Still missing/failed:',len(expected)-len(completed),'| successful:',len(completed))
        '''),
        code('''
        import matplotlib.pyplot as plt
        for sample in active[:PREVIEW_IMAGES]:
            choices = [r for r in records if r['image_id'] == sample['image_id']]
            fig,axes = plt.subplots((len(choices)+3)//3,3,figsize=(18,6*((len(choices)+3)//3)),squeeze=False)
            for axis in axes.flat: axis.axis('off')
            with Image.open(OUT / sample['rgb']) as view: axes.flat[0].imshow(view)
            axes.flat[0].set_title(sample['source_relative_path'])
            for axis,row in zip(list(axes.flat)[1:],choices):
                axis.set_title(row['method'] + (' / ' + row['prompt'] if row['prompt'] else ''))
                if row['status'] == 'ok':
                    with Image.open(OUT / row['overlay']) as view: axis.imshow(view)
                else: axis.text(.02,.5,'FAILED — see error.txt',wrap=True)
            plt.tight_layout(); plt.show()
        print('Confidence scales differ. No reviewed truth: no accuracy, mAP or automatic winner.')
        '''),
        md('''
        ## Export to Drive / contract for the later review app

        `images.json`: stable image IDs, nested source paths, dimensions, original
        hashes, original bytes and EXIF-normalized RGB paths. All proposal masks
        use the normalized RGB coordinates, never the unnormalized original.

        `proposals.json`: latest outcome per image/model/prompt, provenance, timings,
        errors and relative paths. Per-proposal `raw_masks.npz` contains ALL boolean
        instances, including filtered ones; `candidates.json` maps raw indices to
        PNGs, bounding boxes and scores. Separate masks preserve holes and overlaps.
        Empty successful predictions differ from failed methods and unprocessed images.

        `reviews.json`: ONE future annotation record per image, initially unreviewed.
        The app should choose a proposal, allow adding/deleting/splitting/merging and
        painting masks, assign class IDs, then save each accepted instance as a
        separate binary PNG in `reviewed/<image_id>/`. Set `status='accepted'`,
        `selected_proposal_id`, `split_group`, and `instances=[{'instance_id':1,
        'class_id':0,'mask_file':'reviewed/.../instance_1.png'}]`.
        One annotation set per image does NOT mean one merged mask for all pieces.
        An accepted empty scene must be explicit; unreviewed empty lists are NOT negatives.

        Later: export reviewed labels to COCO masks or YOLO segmentation, train a
        compact segmentation model and evaluate on untouched reviewed holdouts.
        YOLO polygon export needs an explicit policy for holes/disconnected masks;
        retain these lossless PNG/NPZ masters. Split by capture session/scene BEFORE
        augmentation; near-identical frames and their derivatives stay in one split.
        Use joint image/mask geometry transforms, image-only lighting/color changes,
        and reviewed-mask cut/paste onto new backgrounds. Do not change defect/color
        labels through augmentation, invent hidden surfaces or treat synthetic
        shadows as real defect evidence. Validate on real, unaugmented test images.

        This notebook prepares the archive contract, NOT the annotation app or
        training/augmentation implementation. Human review remains required before training.
        Check model licenses, especially before redistribution or deployment.

        Export progress has four stages: discover files, read/hash/archive sources,
        verify the local ZIP, copy it to Drive. Each stage shows its own rate/ETA
        once a total is known. The local ZIP uses ZIP_STORED: images/NPZ are already
        compressed, so recompression is avoided (the final ZIP may be larger).
        Source bytes are read from Drive only once for the archive. Checksums and
        ZIP CRC are verified locally; Drive copy is checked for size, not reread.
        Allow local disk space approximately equal to the complete bundle size.
        Stop inference/annotation writes before snapshotting. Interrupted copies
        remain `.partial_*`, never a completed ZIP. Retry export to create a fresh
        ZIP without rerunning models; byte-level ZIP construction is not resumed.
        '''),
        code('''
        # Export works after resume/discovery even if the inference cell was skipped.
        records = json.loads((OUT / 'proposals.json').read_text()) if (OUT / 'proposals.json').exists() else []
        export_local = Path(EXPORT_LOCAL_DIR).resolve()
        if export_local.is_relative_to(Path('/content/drive').resolve()):
            raise ValueError('EXPORT_LOCAL_DIR must be on Colab local disk, not Drive.')
        # New snapshot on every execution; no model weights, tokens or input ZIP duplication.
        (OUT / 'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        (OUT / 'README.txt').write_text('roboqc-review-1.0: images.json + proposals.json + reviews.json. All mask coordinates refer to images/<image_id>/rgb.png (EXIF-normalized). Originals preserved. Review records are NOT ground truth until explicitly accepted. Multiple instance masks per image, not one merged mask. Raw NPZ includes filtered proposals. Scores are not comparable accuracy. No robot execution. Use matching runtime source and settings for provenance. Model weights and credentials excluded.',encoding='utf-8')
        runtime_out = OUT / 'runtime'
        runtime_out.mkdir(exist_ok=True)
        for relative,source in SOURCES.items():
            target = runtime_out / relative; target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): target.write_text(source,encoding='utf-8')
        snapshot = Path(ROOT) / 'exports' / (OUT.name + '_' + uuid.uuid4().hex + '.zip')
        export_bar = None
        def export_progress(phase,done,total):
            global export_bar
            if export_bar is None or export_bar.desc != phase:
                if export_bar is not None: export_bar.close()
                export_bar = tqdm(total=total,desc=phase,unit='file' if phase == 'Scan files' else 'B',
                                  unit_scale=phase != 'Scan files',unit_divisor=1024)
            if total is not None: export_bar.total = total
            export_bar.update(done-export_bar.n)
        try:
            exported = archive_bundle(OUT,snapshot,staging_directory=export_local,progress=export_progress)
        finally:
            if export_bar is not None: export_bar.close()
        print('Drive results ZIP:',exported)
        print('Input images:',len(samples),'| inference-selected images:',len(active))
        print('Errors:',sum(r['status'] != 'ok' for r in records),'| unreviewed: all until app approval')
        expected = expected_proposals(settings,samples)
        successful = {r['proposal_id'] for r in records if r['status'] == 'ok'} & expected
        print('Successful proposals:',len(successful),'| still missing/failed:',len(expected)-len(successful))
        '''),
    ])


if __name__ == "__main__":
    build()
