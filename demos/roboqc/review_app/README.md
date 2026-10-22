# Robotekin segmentation review workstation

Four-page **PySide6 desktop application**, independent of the robot/microphone demo.
Original images, proposal files and previous reviews are never deleted or overwritten.
Training and validation only start after explicit confirmation in the application.

## 1. Download the right 04e output

The notebook is **04e_export_segmentation_review.ipynb** (not `043`). Its last cell
prints a completed ZIP path in `MyDrive/Robotekin2026/exports`. In Google Drive,
right-click that completed `.zip` → **Download**. Download the file itself, not the
whole exports directory. Do not choose `.work.zip`, `.partial_*`, `.export.json`,
or the input-only `data_04e_inputs.zip`.

The result must contain `images.json`, `proposals.json`, `images/.../rgb.png` and
`proposals/.../masks/candidates.json` with individual binary PNG masks. An enclosing
bundle directory is supported. The app reads this ZIP directly without extracting
tens of thousands of files. You can also open an already extracted bundle folder.
No archive Python code is imported or executed. Optional checksums in the export
are not fully rehashed on opening; ZIP CRC is checked for each member read.

## 2. Install and launch

Use Python **3.11 or 3.12**, preferably a separate virtual environment to avoid
changing the demo's ML dependencies. From the repository root on Windows:

```powershell
py -3.11 -m venv .venv-review
.\.venv-review\Scripts\python.exe -m pip install --upgrade pip
# First install the CPU or CUDA PyTorch build appropriate for your machine:
# https://pytorch.org/get-started/locally/
.\.venv-review\Scripts\python.exe -m pip install -r demos/roboqc/review_app/requirements.txt
Set-Location demos/roboqc
..\..\.venv-review\Scripts\python.exe -m review_app
```

On Linux/macOS, activate an equivalent environment, install the requirements,
`cd demos/roboqc`, then `python -m review_app`. Reviewing/exporting needs no GPU
and no Hugging Face token. Training YOLOE may download a text encoder and weights;
use trusted checkpoints only. Check Ultralytics licensing before redistribution.

## 3. Pages and workflow

1. **Explorer**: open the results ZIP/folder, then choose a separate empty project
   folder. Reopen the same project later to restore decisions. Search nested paths
   or review status. Counts come from manifests, not an expensive mask scan.
   Double-click an image. Opening images/masks and exporting run in a bounded
   background thread; only the displayed proposal's masks are loaded, with a
   128 MiB normalized-image cache (one oversized current image can exceed it).
2. **Review**: switch model/prompt alternatives, inspect individual numbered masks,
   uncheck background/subparts, assign a class name to each selected instance.
   Wheel zooms; Pan drags. Paint/Erase edits the active instance. Add an empty
   instance to paint a missed part. Erase/split manually by adding separate masks.
   Assign a capture/session group: near-identical frames of the same scene MUST
   share a group. The default parent folder is only a starting point, not an
   automatic similarity detector. Accept saves an immutable revision and updates
   the current decision. Exclude affects training eligibility, not source files.
   An empty scene requires the explicit negative checkbox. Unsaved edits trigger
   a discard prompt on navigation/proposal changes. Previous/Next respects the
   Explorer filter. Multiple instances remain separate; one chosen annotation set
   per image does not mean one merged object mask.
3. **Training**: first export approved data in Explorer. Choose YOLOE with
   `yoloe-26s-seg.pt` (or another official YOLOE segmentation checkpoint), or YOLO
   with e.g. `yolo11n-seg.pt`. Set epochs, batch, resolution, CPU/CUDA, patience,
   rotation/flip/brightness/mosaic. Start creates a unique output run and separate
   Python process. See streamed logs, epoch progress, live box/mask mAP curves and
   the run folder. Early stopping can finish before the configured epoch count.
   Saved artifacts include `train/results.csv`, plots, `train/weights/best.pt`,
   `last.pt`, `job.json`, and `console.log`. Stop terminates the worker; the last
   fully saved epoch survives, not necessarily the current epoch. No training
   starts on startup/import and the UI does not automatically install packages.
4. **Validation**: choose exported dataset and trusted trained `best.pt`, with
   matching model family/device on Training. Run `val` during model development;
   reserve `test` for final evaluation after model selection. Shows metrics JSON,
   box/mask mAP, per-image stage timings and generated plots/confusion matrices.
   The best training checkpoint automatically populates the weight field.

## Export contract

Only accepted images and checked instances are copied into a fresh dataset:

```
data.yaml
manifest.json                  # provenance, selected proposal/revision, groups/splits
images/{train,val,test}/<image_id>.png
labels/{train,val,test}/<image_id>.txt
reviewed_masks/<image_id>/0000.png   # exact reviewed binary masks
```

Optional sibling ZIP contains the same tree, ready to copy/upload elsewhere.
RGB images are EXIF-normalized, matching 04e mask coordinates; originals remain
in the source ZIP. Class names are explicit human assignments, never inferred
from folder names, prompt labels or incomparable model confidence scores.
YOLO rows are `class_id x1 y1 x2 y2 ...`, normalized to image dimensions.
Export does **not** copy unreviewed alternatives. The original 04e ZIP remains
the complete proposal archive; keep it together with the review project.

The reproducible split is approximately 70/15/15 by independent capture group,
not image count. Exact original SHA256 duplicates join groups transitively.
At least three independent accepted groups and training foreground are required.
This cannot detect near-duplicates or guarantee balanced classes: inspect
`manifest.json` and group data appropriately. Augmentation occurs only while
training, after splits. Photometric augmentation preserves hue/saturation.
Background replacement is not implemented; inaccurate masks or implausible
lighting can make it harmful. No synthetic data leaks into validation/test.

YOLO polygon labels cannot represent holes. By default export rejects holes;
explicitly enable **Allow hole filling** to approve that lossy conversion while
retaining exact PNG masks. Disconnected masks are rejected rather than silently
dropping components: repair/split them in Review first. Overlapping instances
remain separate. A failed export leaves its incomplete fresh directory for
inspection and never publishes `manifest.json`; select a new output attempt.

## Training elsewhere / Colab

Download/copy the reviewed dataset ZIP, extract it in Colab, install a compatible
PyTorch/Ultralytics environment, and resolve the `path` field in `data.yaml` to
the extracted folder. `path: .` is portable storage metadata; do not assume an
external trainer resolves it relative to this YAML. The app's worker writes a
separate absolute-path YAML and never changes the exported one. For the same
fine-tuning recipe see `worker.py`: YOLOE requires `YOLOEPESegTrainer`; ordinary
YOLO uses `YOLO.train`. Fine-tuned YOLOE is closed-set for the reviewed class
names, not a newly trained general open-vocabulary segmenter.

## Verification and limitations

Fixture tests are in `phase_2/tests/test_review_app.py`. They cover lazy ZIP input,
unsafe paths, source preservation, immutable revisions, explicit negatives,
duplicate/group-safe splits, approved-only export, polygon limitations and an
offscreen four-page UI screenshot. They do not execute models, training or Colab.
The actual CUDA/CPU training stack must be tested on the user's chosen machine.
The worker currently starts fresh fine-tuning runs; restoring optimizer state
from an interrupted run is not exposed in the UI. Full masks may still be
expensive on very large images. Quality labels are not automatically inferred.
No robot execution or calibration is added by this application.

References: [YOLOE fine-tuning](https://docs.ultralytics.com/models/yoloe/#train-usage),
[Qt asynchronous processes](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html).
