# Robotekin 2026 · manual Colab workflow

The six stage notebooks plus the YOLOE, SAM 2, pretrained-comparison and SAM 3 notebooks are standalone Google Colab uploads. Their
code cells have **not been executed by the coding agent**. Static validation and
local unit tests do not establish training accuracy, GPU compatibility or camera
performance. Run the notebooks yourself using the routes below. Phase 1 can be used
without any vision dataset or hardware.

## Run order

For the new complete-image-tree proposal export, see [04e review workflow](04E_REVIEW_WORKFLOW.md).
Generate `assets/data_04e_inputs.zip` from your local images using
`prepare_04e_inputs.py`, then upload it once in 04e to save/reuse the input in Drive.
Private image archives and supplied reference photos are not included in Git.

| Notebook | Input | Output |
|---|---|---|
| 01_generate_voice_dataset | Configuration, no external data | ≥20,000 commands, CSV/JSONL splits, statistics, leakage audit |
| 02_train_intent_model | 01 dataset ZIP or Drive folder | TF-IDF, multilingual MiniLM, compact JEPA, experimental OpenJev; selected model bundle |
| 03_evaluate_intent_model | 01 dataset + 02 bundle | Held-out metrics, JSON predictions, confusions, CPU latency, interactive dialogue, optional ASR evaluation |
| 04_prepare_vision_dataset | Real COCO/YOLO images and instance masks | Grouped splits, original masks, YOLO labels, train-only augmentations, metadata |
| 04a_try_yoloe_pretrained | Ordinary part photos, no labels | Pretrained YOLOE nano PT/ONNX bundle, masks, overlays, CPU latency |
| 04b_try_sam2_stamped_parts | Your photos: upload or Drive file/folder | SAM 2.1 automatic/prompted mask candidates, PNG/NPZ/JSON exports |
| 04c_compare_pretrained_segmentation | Your photos: upload or recursive Drive folder | Five methods, optional sixth SAM 3, side-by-side overlays, mask/latency reports |
| 04d_try_sam3_pretrained | Your photos plus approved facebook/sam3 account + read token | SAM 3 only, independent concept-prompt masks and reports; CPU or CUDA including T4 |
| 04e_export_segmentation_review | Complete recursive `/data` ZIP or Drive image folder | All five 04c methods + corrected 04d SAM 3, original images and lossless proposals in a review ZIP on Drive |
| 05_train_segmentation | 04 prepared dataset | YOLO11n-seg PT/ONNX; optional visible-quality classifier |
| 06_evaluate_vision | 04a bundle + photos, or 04 dataset + 05 bundle | Unlabelled inspection, or held-out metrics when annotations exist |

1. Upload a notebook at <https://colab.research.google.com/>. Edit its first Python
   configuration cell **before** running the setup. For 02/05 choose a GPU runtime.
2. Keep `USE_DRIVE=True` to mount Drive. Change `ROOT` if desired. Each notebook
   creates a new timestamped run folder; existing models/datasets are not replaced.
3. Set `DATASET_PATH` and `MODEL_PATH` to uncompressed Drive folders. Leave them
   empty to open Colab's upload widget; ZIP uploads are extracted automatically.
   For evaluation, upload the dataset and model ZIP at their separate prompts.
4. Read the statistics and error tables. A synthetic text score does not establish
   speech recognition quality; a schematic image is not real industrial evidence.
5. Download the export ZIP through the Files panel or Drive. Data preparation and
   training packages include embedded runtime source. Evaluation archives contain
   reports and visualizations, not newly trained weights.

Set `RESUME_RUN` to the **original run directory**, not its `intent_bundle` or
`vision_bundle` subfolder, to resume training. Completed intent candidates are
skipped. JEPA/OpenJev and vision training save intermediate checkpoints. The linear
heads save the best artifact each epoch; interrupted linear fits restart their
short fit. Do not change a resumed run's dataset, seed or architecture. Re-run
notebooks 01/04 in a new run instead of setting `RESUME_RUN`.

OpenJev training needs a GPU; without one notebook 02 marks it pending and still
exports completed candidates. Its upstream repository/subfolder and NLI label
mapping are checked explicitly. Upstream incompatibilities should surface as
errors, not silently substitute a different model. The MiniLM ONNX optimization
cell is optional: only validated exports are selected for local loading.
Voice setup uses Transformers 5 for OpenJev and exports MiniLM directly through
PyTorch/ONNX Runtime. Do not add `optimum[onnxruntime]` to this environment: its
ONNX dependency currently requires Transformers below 4.58. If installation fails,
read the pip `ERROR` lines printed in the cell; the complete output is also saved
to `/content/roboqc_pip_install.log`. Re-upload updated notebooks into a fresh
Colab session after changing dependency setup.
Threshold selection uses a reproducibly sampled validation subset (default 300).
OpenJev evaluation defaults to 200 test examples to bound CPU time; the report
marks this subset explicitly. Set `OPENJEV_EVAL_LIMIT=None` for a full test run.

## Start vision without an annotated dataset

Public copies of 04b–04d contain no embedded photos. Set `IMAGE_PATH` to a Drive
image or folder (subfolders are searched), or leave it blank and upload image
files when prompted. Upload ZIPs through 04e instead. Supply only photos you
have permission to use. 04e's published configuration starts a new run; set
`RESUME_BUNDLE` explicitly to recover an existing run. Saved notebook outputs
and personal run paths were removed; local pre-publication originals are backed
up under the ignored repository-root `local/` directory.

Run **04a_try_yoloe_pretrained**, not 04 or 05. It uses pretrained YOLOE26 nano
with text prompts for bolts, nuts, washers and brackets. No fitting or training is
performed. You still need ordinary photos of your parts to inspect predictions;
instance labels are not required. By default, `USE_DEMO_IMAGES=True` automatically
downloads five real sample photos (fasteners and several formed brackets), so no image upload is
needed. Input photos are displayed before model preparation, followed by predicted
masks and labels. The first run downloads the checkpoint and text
encoder into a reusable Drive cache. Internet access is needed for those downloads.

Set the initial configuration cell, for example:

```python
USE_DRIVE = True
USE_DEMO_IMAGES = False  # Keep True with IMAGE_PATH='' for the included sample photos.
IMAGE_PATH = '/content/drive/MyDrive/Robotekin2026/part_photos'
CHECKPOINT = 'yoloe-26n-seg.pt'
EXPORT_ONNX = True
```

Alternatively set `USE_DEMO_IMAGES=False`, leave `IMAGE_PATH=''` and upload photos
or a ZIP when prompted. A nonempty `IMAGE_PATH` always overrides demo photos.
The original fastener/bracket photos are by [Sbrennan83 (CC BY-SA 4.0)](https://commons.wikimedia.org/wiki/File:Bolts,Nuts_%26_Washers.jpg)
and [Fructibus (CC0)](https://commons.wikimedia.org/wiki/File:Bronze_angle_bracket_-_4_x_4_cm_-_A.jpg).
Three additional photos cover multiple brackets, a formed amplifier bracket and
an assembled channel connector. See [demo image sources and stamped-part references](DEMO_IMAGES.md)
for all credits, scene limitations and instructions to test your own reference image.
Their source/license credits are saved in `demo_images/ATTRIBUTION.txt`; downloads
are checksum-checked and existing mismatched files are never replaced. These
public photos have no instance annotations and are not an industrial benchmark.
Verified downloads are cached at `ROOT/demo_images` across runs. Requests are
paced; HTTP 429 and transient server errors get at most two retries with backoff
and numeric/date `Retry-After` handling. Long server delays are deferred, not
bypassed. Persistent rate limiting stops further network requests in that call.
The notebook continues with a clearly reported nonempty verified subset and saves
missing filenames/reasons to `download_report.json`; no available images is an
error. You can select an older run's `demo_images` folder via IMAGE_PATH to use
already downloaded images immediately, without making new image requests.
Review the overlays before trusting detections. The notebook saves a new
`yoloe_bundle_<unique-id>` folder and ZIP in its timestamped run directory, including the
prompt-conditioned PT checkpoint, optional ONNX, manifest and local requirements.
An export mismatch raises an error rather than declaring the ONNX bundle ready.
Both backends explicitly use square padding (`rect=False`), the same IoU/detection
limits and class-agnostic NMS for YOLOE. This avoids comparing different padding
or YOLOE-vs-YOLO suppression defaults. Verification still requires equal instance
counts, matching classes and mask IoU at least 0.95. A mismatch saves per-image
counts, boxes, confidence scores and settings in `export_equivalence.json`, and
leaves `export_verified=False`. Inspect that report if a mismatch persists.
With `EXPORT_ONNX=False`, only the saved PyTorch checkpoint is used. Prompts are
embedded in the saved model; change them in 04a and export a new bundle when needed.
Rerunning the export or preview cell creates a new uniquely named output folder,
so interrupted attempts and previous exports are preserved. Use the successful
bundle path printed by the export cell, not an older failed attempt.

Skip notebook 05. For additional images, open notebook 06 with:

```python
EVALUATION_MODE = 'unlabelled'
USE_DEMO_IMAGES = False  # Keep True and leave IMAGE_PATH empty to reuse demo subjects.
MODEL_PATH = '/content/drive/MyDrive/Robotekin2026/<04a-run>/yoloe_bundle_<unique-id>'
IMAGE_PATH = '/content/drive/MyDrive/Robotekin2026/new_photos'
```

Use the actual 04a run path shown by the notebook. An empty MODEL_PATH prompts for
the model ZIP; photos download automatically unless demos are disabled. The same
photos in 04a and 06 are a smoke test, not an independent held-out test.
Unlabelled evaluation reports masks,
classes, positions, planar orientations and CPU latency, **not accuracy or mAP**.
Quality stays `unknown`: these object prompts do not identify defects. Bare gray
metal also has no red/green/blue color category. Once annotated capture sessions
exist, 04 then 05 is the optional custom-training route, and 06's `labelled` mode
can measure held-out performance.

YOLOE may miss small reflective parts, merge touching instances or produce poor
washer-hole masks. Test real overhead views before planning picks. See the
[official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/) and
[Ultralytics licensing](https://www.ultralytics.com/license) before distributing
the application; the upstream AGPL/Enterprise terms need review.

## 04b: Segment Anything on the four supplied stamped-part photos

Upload `04b_try_sam2_stamped_parts.ipynb` to a **fresh Colab GPU runtime**, then run
top to bottom. It is standalone: no notebooks 04/04a/05 or existing model bundle
are needed. Leave `IMAGE_PATH=''` to upload your own photos, or set it to a Drive
file/folder. No supplied photo bytes are bundled. Internet is still required to
install dependencies and download the
pretrained SAM checkpoint on first use. Drive caches weights between sessions.

The default is `facebook/sam2.1-hiera-large`; `facebook/sam2.1-hiera-small` is a
lighter comparison. The official SAM source is commit-pinned; the checkpoint's
resolved Hugging Face revision and SHA-256 are recorded in `settings.json`.
The optional CUDA extension is not compiled, and mask hole-filling postprocessing
is disabled. GPU is auto-detected; CPU is supported but automatic Large inference
can be very slow. If GPU memory runs out, lower `POINTS_PER_BATCH` to 8 or 4.

Each image displays input/overlay and an individual-mask gallery. A new result
folder is created per inference attempt. Each completed image is saved immediately:

- `raw_masks.npz`: all generated boolean masks, retaining holes and overlaps.
- `mask_*.png`: candidates retained by configurable area filters.
- `candidates.json`: boxes, centers, planar geometry, scores and filter reasons.
- `overlay.png`, `input.png`, plus run-level settings, report and export ZIP.

The area filter removes very small/large regions, not semantically verified
background. Set `MAX_MASK_FRACTION=1.0` to retain large masks. Nested masks may
describe subparts; candidate count is **not** physical part count. Predicted IoU
is a model score, not measured mask accuracy. Categories and quality stay unknown.

For a missed part, set `BOX_IMAGE_INDEX` and `BOX_XYXY=[x1,y1,x2,y2]` in original
pixel coordinates, then run the optional box-prompt cell. You can change these
variables in a new cell without rerunning setup or automatic generation. A new
prompt-result folder preserves each attempt. Use one tight box per visible part.

04b performs no training and is not a guaranteed improvement over YOLOE on these
photos. It exports **SAM candidates, not an industrial vision bundle**: notebook
06 and the robot planner cannot consume it as a classified detector. A later
classifier/validated instance-selection stage would be required. Weights stay in
the Drive cache, whose path/hash is recorded; the results ZIP excludes the large
checkpoint. No local CPU real-time claim or defect recognition is made.

The four user-provided photos have no established open redistribution license;
verify rights before publishing the notebook/results. See the
[official SAM 2 repository](https://github.com/facebookresearch/sam2) for model/API
documentation and licenses. The coding agent did not execute inference or notebooks.

## 04c: compare five pretrained methods on the actual parts

Use a fresh Colab T4 GPU session. The default comparison is YOLOE-26 Large text
segmentation, SAM 2.1 automatic masks, Grounding DINO + SAM, OWLv2 + SAM, and
Florence-2 + SAM. These are five pipelines, not five sizes of one model. SAM 3 is
an optional sixth requiring approved Hugging Face access. See the
[primary-source model review and comparison guide](PRETRAINED_REVIEW.md).

Set `IMAGE_PATH` to your Drive image/folder, or leave it blank to upload photos.
Nested folders and uppercase image extensions are supported. No private dataset
or reference photos are bundled. Weights/dependencies
still download initially. All weights are cached on local Colab disk using
`MODEL_CACHE` and `SAM2_CACHE` (cleared on runtime reset), not the Drive mount.
Unreadable SAM checkpoints or detector `SafetensorError` failures trigger one new
download in a fresh local subfolder, without replacing previous files. Detector
loading explicitly uses safetensors; there is no unsafe pickle fallback.
Results still save under `ROOT` on Drive. Full tracebacks are printed and saved.
All methods process the same
optional `ROI_NORM`; full-frame masks/positions are retained in exports.

Each method saves per-image masks, overlays and timing; the notebook displays a
comparison grid and writes CSV/JSON plus a ZIP. Downloads/loading are timed
separately. Error statuses/tracebacks remain visible; the report explicitly checks
whether five methods actually completed all inputs. No unverified "best" method
or measured accuracy is invented. If labelled masks are supplied through
`GT_MANIFEST_PATH`, one-to-one mask precision/recall/F1 at IoU 0.5 can be measured.
No expected object counts are assumed for uploaded images. The notebook never
forces six predictions. It does not train or command a robot.

## 04d: SAM 3 only, CPU or CUDA

1. Upload `04d_try_sam3_pretrained.ipynb` to a **fresh Colab CPU or GPU runtime**.
   Leave `DEVICE_CHOICE='auto'`: CPU uses float32 and CUDA uses float16, including
   T4. No A100/BF16 requirement remains. You can explicitly set `cpu` or `cuda`.
   CPU may be very slow; sufficient RAM/VRAM and several GB of local disk are needed.
2. In Colab Secrets add `HF_TOKEN`, a read token belonging to the account approved
   at [facebook/sam3](https://huggingface.co/facebook/sam3), and enable notebook
   access. Approval for facebook/sam3.1 is not assumed to cover facebook/sam3.
   If unavailable, the notebook asks for a token using a hidden input. Never paste
   tokens into configuration cells or include them with exported reports.
3. Set `IMAGE_PATH` to your Drive image/folder, or leave it blank to upload photos.
   No dataset/reference photos are embedded in the public notebook. Adjust short
   English `PROMPTS` or the shared `ROI_NORM`, then run in order. Each concept is
   a separate experiment: masks from synonyms are not merged or forced to six.
4. Inspect overlays, per-instance PNGs, `raw_masks.npz`, `object_metadata.json`,
   candidate geometry, CSV/JSON reports and the final ZIP. The optional custom cell
   reuses the loaded predictor. Reruns use fresh folders and preserve previous results.

This replaces the previous 04d SAM 3.1-only notebook. It uses `Sam3Model` and
`Sam3Processor` from pinned Transformers, not Meta's CUDA-only multiplex code.
Native PyTorch SDPA avoids external FlashAttention dependencies. Floating image
tensors match model dtype; token IDs remain integer. All CUDA timing calls are
guarded so CPU runs do not attempt GPU access. Images and prompts are independent;
no temporal tracking or model substitution is performed.

All weights stay on local `MODEL_CACHE`, with explicit safetensors loading and one
fresh-cache recovery attempt. Results save to Drive under `ROOT`. Local weights
vanish on runtime reset. No source install or NumPy pin workaround is needed.
Actual Colab execution, CPU latency and GPU memory use remain manual checks.
There is no fallback to another model, compilation, training, or robot execution. Masks and
planar grasp candidates require review; hidden surfaces and quality are unknown.

## Voice data and behavior

Each JSONL record has `id`, `text`, `family`, `group`, `split`, `intent`, `mode` and
`expected`. The expected envelope is:

```json
{
  "schema_version": "industrial-1.0",
  "status": "valid",
  "command": {
    "intent": "sort",
    "mode": "quality",
    "rules": [{"category": "good", "destination": "green"}]
  },
  "issues": [],
  "clarification": null
}
```

Categories: quality `good/defect_1/defect_2`; color `red/green/blue`. Destinations:
`red/green/blue`. Other statuses: `needs_clarification`, `contradictory`,
`unsupported`, `invalid`. A `filter` lists explicitly restricted categories.
Repeated identical rules are deduplicated; conflicting destinations are rejected.
Multiple categories may share a destination. Mixed color/quality requests require
clarification. Negations outside the supported grammar require reformulation.

Every missing destination prompts a question, including when rules were previously
active. A bare answer such as “la azul” supplies that destination to all currently
missing categories; category-specific answers can complete the request in several
turns. No new plan starts while clarification is pending. `stop` preserves the
plan, `resume` resumes an existing plan, and `cancel` discards it.

Text examples are generated from semantic templates rather than an LLM/API. Split
grammars are reserved before rendering, normalized texts are globally unique, and
derived examples stay within the reserved split. Review the bounded near-duplicate
audit: it diagnoses high lexical overlap rather than proving natural-language
independence. Add a manually collected, independently written Spanish test set
before making generalization claims.

Notebook 03 accepts an optional audio manifest:

```json
{"audio_path":"recordings/operator01.wav","reference":"pon las buenas en la verde","expected":{"status":"valid","command":{"intent":"sort","mode":"quality","rules":[{"category":"good","destination":"green"}]}}}
```

## Import local voice models

Extract 02's ZIP into a **new** folder, for example `demos/roboqc/artifacts/intent_bundle`.
Do not extract over `phase_1/models`. Load only trusted joblib/PT files.

From the demo directory in PowerShell:

```powershell
pip install -r phase_1/requirements.txt
pip install -r artifacts/intent_bundle/requirements-local.txt
$env:ROBOQC_MODEL_DIR = (Resolve-Path artifacts/intent_bundle).Path
$env:ROBOQC_MODEL = 'industrial'
python phase_1/04_app.py
```

The model dropdown includes `Industrial (Colab)`. Without a bundle, the new sorting
parser still works with deterministic rules. Legacy model options and inspection
commands remain available. The command summary shows the complete industrial JSON
and clarification message. Speech defaults to Whisper base INT8; Vosk remains
selectable. `ROBOQC_CPU_THREADS=2` controls Whisper CPU threads. No ASR weights are
retrained; Spanish ASR must be downloaded once before an offline demonstration.

For a headless consumer, add `phase_1/src` to `PYTHONPATH` and use:

```python
from roboqc.industrial.models import IntentBundle
from roboqc.industrial.voice import Dialogue

model = IntentBundle('artifacts/intent_bundle')
dialogue = Dialogue()
response = dialogue.submit('Recoge solamente las piezas defectuosas', model)
print(response)  # needs_clarification
print(dialogue.submit('la azul', model))
```

The runtime shipped inside each bundle is under `runtime/roboqc`. That folder can
be added to `PYTHONPATH` outside this repository for inference. The original demo
uses its own matching source modules.

## Real vision annotation format

Use classes `bolt`, `nut`, `washer`, `bracket`, in that canonical output order.
In COCO, each image includes a `session_id`. Its `file_name` resolves against the
annotation directory unless `IMAGE_ROOT` is configured. For existing exports with
an `annotations/` subdirectory, set `IMAGE_ROOT` to the actual image root.

Example metadata additions to ordinary COCO annotations:

```json
{"images":[{"id":1,"file_name":"images/001.png","width":640,"height":480,"session_id":"capture_A"}],"annotations":[{"id":1,"image_id":1,"category_id":1,"segmentation":[[10,10,50,10,50,50,10,50]],"attributes":{"color":"red","quality":"unknown"}}],"categories":[{"id":1,"name":"bolt"}]}
```

This illustrates metadata only; capture all classes in independent sessions before
training. COCO RLE retains holes and disconnected visible regions. The prepared
dataset preserves original masks as PNG; YOLO polygon conversion uses the largest
outer contour and records its losses in `conversion_warnings.json`. Such masks can
fill washer/nut holes in predictions, so candidate grasp points remain unvalidated.

For YOLO input, put `sessions.json` beside `data.yaml`:

```json
{"images/train/001.png":"capture_A","images/val/002.png":"capture_B"}
```

Optional `attributes.json` has lists matching each image's label-line order:

```json
{"images/train/001.png":[{"quality":"unknown","color":"red"}]}
```

Notebook 04 regroups imported images by session, checks exact decoded-image
duplicates and class coverage, then augments **training only**. Keep adjacent video
frames, repeat views and near-duplicate captures in the same session. It does not
automatically discover cross-session near duplicates of a physical scene.

Leave `QUALITY_DEFINITIONS={}` until you have explicit visible acceptance criteria.
To enable quality training, define all `good/defect_1/defect_2` criteria and label
real instances in every split. Missing or low-confidence evidence stays `unknown`.
Color rules route colored/coated parts; unpainted gray metal stays color-unknown.

## Local vision and simulation

Extract 04a's or 05's model ZIP into a new folder. For example, name the extracted
bundle `artifacts/vision_bundle` (the folder containing `vision_manifest.json`). Install:

```powershell
pip install -r phase_2/requirements-industrial.txt
pip install -r artifacts/vision_bundle/requirements-local.txt
```

Copy `phase_2/config/industrial.example.json` to a new local configuration and set
an **absolute** `industrial.model_dir`. Configure the overhead webcam index,
`roi` (source-box crop), `source_region` and destination rectangles `[x,y,w,h]` in
full-frame pixels. Destination colors are explicit box identities: the system does
not assume it can find colored boxes automatically.

The calibration JSON extends the existing format:

```json
{
  "frame":"table","units":"mm",
  "pixel_points":[[80,60],[560,60],[560,420],[80,420]],
  "world_points":[[0,0],[300,0],[300,200],[0,200]],
  "max_error_px":2.0,
  "robot_transform":[[1,0,0],[0,1,0],[0,0,1]]
}
```

**These numbers illustrate the file format; they are not a hardware calibration.**
Measure correspondences and the robot-frame transform. Optional `camera_matrix`
and `distortion` enable distortion correction. Four points exactly fit a homography;
use additional independent points and physically check projection error. World
coordinates outside the calibrated hull are unavailable. The rigid 2D robot
transform requires millimetres. No Z coordinate is inferred.

Launch the integrated tabs with the same voice environment variables:

```powershell
python phase_2/08_app.py --config phase_2/config/industrial.local.json --calibration phase_2/config/calibration.local.json
```

The inspection tab displays masks, candidates, industrial quality, rules and plan.
`Simular un paso` processes one ready item and emits `action_requested`,
`simulated_result`, `simulated_verification`. No LeRobot connection is attempted.
The planner blocks incomplete regions, unknown categories, partial objects,
missing candidates and absent robot-frame calibration. Pixels/masks use full-frame
coordinates. Serialized masks use custom row-major binary RLE (`size/order/counts`),
not COCO's column-major compressed representation.

## Technical review, priorities and remaining risks

Priority 1 replaces brittle per-conjunction interpretation with relational sorting
rules and explicit dialogue. It adds whole-command evaluation and an artifact
contract. The classifier assists action recognition; deterministic validation
controls accepted output. Both raw-model metrics and composed-pipeline metrics
are reported, so parser success is not mislabeled as classifier accuracy.

Priority 2 replaces saturated-color contour detection with learned instances in
industrial mode. Legacy geometric demos remain intact. Pretrained YOLOE26 nano
provides the no-annotation starting point; YOLO11n-seg remains the optional custom
training path. Mask R-CNN is a heavier baseline and SAM can
assist annotation but does not by itself supply the required industrial labels.

Priority 3 connects rules and world snapshots to a small simulated planner.
`SortingCoordinator` is the execution boundary for a future LeRobot adapter;
physical transport, inverse kinematics, gripper calibration, collision checking and
robot verification are not implemented in this delivery.

Known limits: one RGB view cannot establish reliable full 3D pose, stack height,
hidden defects, graspability or physical success. Overlap may merge/miss instances;
distance-based tracking can swap IDs. Symmetric parts have no unique orientation.
Generic metallic backgrounds, reflections and unseen defects require real data.
Whisper/Vosk and text rejection are not an emergency-stop safety system. Low
latency and industrial accuracy must be measured on the actual laptop/camera.

Sources: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
[OpenJev](https://huggingface.co/AlexWortega/openjev),
[TypeSafe Jev](https://docs.typesafe.ai/introduction),
[YOLO11](https://docs.ultralytics.com/models/yolo11/).
OpenJev is a community NLI model, distinct from proprietary Jev and from JEPA.
Its benchmark claims are not evidence for Spanish commands or this robot.

## Development checks

`build_notebooks.py` rebuilds JSON notebook documents from shared runtime source;
it does not execute cells. After changing shared code, run it and the local tests:

```powershell
python demos/roboqc/colab/build_notebooks.py
python -m pytest demos/roboqc/phase_1/tests -q -p no:cacheprovider
python -m pytest demos/roboqc/phase_2/tests -q -p no:cacheprovider
```

In the managed Windows sandbox, Python's user-only temporary-directory ACL can
exclude the restricted process token. `python demos/roboqc/colab/verify_local.py
phase_1` (or `phase_2`) runs the same tests using new workspace temporary folders
that inherit workspace permissions. `verify_local.py screenshot` captures the UI
with fixture data and an explicitly loaded available font; it opens no camera and loads no models.

No training/dataset-generation script is part of those checks. Notebook validity,
parser behavior and mock vision integration can be checked locally; Colab training,
real model exports and hardware behavior remain manual acceptance steps.
