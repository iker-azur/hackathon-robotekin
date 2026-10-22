# Robotekin segmentation review workstation

Five-page **PySide6 desktop application**, independent of the robot/microphone demo:
**Explorer → Review → Augmentation → Training → Validation**.
Original images, proposal files and previous reviews are never deleted or overwritten.
Training and validation only start after explicit confirmation in the application.
Light theme, one set of sizes on every page, and every action has a key (shown on its button,
listed under **F1**).

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
Minimum window 1280×720.

## 3. Pages and workflow

1. **Explorer** (`Ctrl+1`): open the results ZIP (`Ctrl+O`) or folder, then choose a separate
   project folder; reopen it later to restore decisions. A progress bar and chips show
   accepted / excluded / to review. Filter by status or by text (path, class, group), sort any
   column, and see the suggested or saved class and group per image. **Review next
   unreviewed** jumps in. Double-click an image to review it. **Groups…** sets how
   unreviewed images get their default capture group (see below). Opening images/masks and
   exporting run in a background thread; only the displayed proposal's masks are loaded, with
   a 128 MiB normalized-image cache.
2. **Review** (`Ctrl+2`): the main workspace.
   * *Method* combo (`↑`/`↓`) switches between the 8 model/prompt alternatives; the last method
     you used opens first on the next image. Tick **add masks from other methods** (`Shift+M`)
     to append another method's masks to your working set instead of replacing it, so you can
     combine the best of several models.
   * *Masks* list: the tick means "kept in the dataset". Click a mask on the image to select it
     (`Ctrl`-click adds); double-click toggles keep; dropped masks stay faintly outlined so you
     can bring them back. **All / None / Invert** (`A`, `Shift+A`, `I`) triage quickly; `K`
     toggles the selected mask, `Shift+K` keeps only it.
   * *Edit tools*: **Select** `V`, **Pan** `P` (or hold `Space`, or middle-drag), **Brush**
     `B`, **Erase** `E`, **Lasso** `L` (click points, `Enter` adds, `Shift+Enter` subtracts,
     `Esc` cancels). `[` / `]` change the brush size. Brushing with nothing selected starts a
     new mask of the current class. **New** `Ctrl+N`, **Delete** `Del`, **Merge** `M`, **Split
     into pieces** `S`, duplicate `Ctrl+D`, fill holes and remove specks (under **More**) cover
     the cases that otherwise block export (holes, disconnected masks).
   * *Undo / redo* `Ctrl+Z` / `Ctrl+Y` for every edit (strokes, classes, deletes, merges…); the
     window title shows `*` only while the image differs from its saved state.
   * *Classes*: one button per class (keys `1`-`9`) labels the selected masks, or all kept masks
     when none is selected. Each class keeps its colour. **Manage…** adds, renames, recolours,
     reorders, merges and deletes classes (a rename re-saves affected images as new revisions).
   * *Scene*: capture group and the explicit "confirmed empty scene" box for negative images.
   * *Decision*: **Accept & next** `Ctrl+Enter`, **Accept** `Ctrl+S`, **Exclude** `Ctrl+E`.
     Accept saves an immutable revision; Exclude only affects training eligibility. Leaving an
     image with unsaved edits asks: *Accept & continue / Discard / Cancel*. `←`/`→` follow the
     Explorer's current filter and sort; `N` jumps to the next unreviewed image.
3. **Augmentation** (`Ctrl+3`): parameters for geometric, photometric, occlusion and composition
   augmentation, presets (*Metal parts (default)*, *Conservative*, *Heavy*, *Off*), and a live
   3×3 preview on your own masks (`R` re-rolls). The configuration is stored in the review
   project and passed to Training. **Offline export** writes a copy of a reviewed dataset with
   extra augmented training images (`val` and `test` are copied byte-for-byte), for Colab or
   other trainers.
4. **Training** (`Ctrl+4`): first export approved data in Explorer. Choose YOLOE with
   `yoloe-26s-seg.pt` (or another official YOLOE segmentation checkpoint), or YOLO with e.g.
   `yolo11n-seg.pt`. Set epochs, batch, resolution, CPU/CUDA and patience; augmentation comes
   from the Augmentation page. Start creates a unique output run and a separate Python
   process, with streamed logs, epoch progress and live box/mask mAP curves. Early stopping can
   end before the configured epoch count. Artifacts include `train/results.csv`, plots,
   `train/weights/best.pt`, `last.pt`, `job.json` and `console.log`. Stop terminates the worker;
   the last fully saved epoch survives. No training starts on startup/import and the UI never
   installs packages.
5. **Validation** (`Ctrl+5`): choose the exported dataset and a trusted `best.pt`. Run `val`
   during development; reserve `test` for the final evaluation after model selection. Shows
   metrics JSON, box/mask mAP, stage timings and generated plots/confusion matrices. The best
   training checkpoint populates the weight field automatically.

## 4. Keyboard shortcuts

Press **F1** (or the `?` button on any page) for the full searchable list. Hover any button to
see its key. Single keys pause while you type in a text field.

| Area | Keys |
|---|---|
| Images | `←` `→` previous / next · `N` next unreviewed · `↑` `↓` method · `Ctrl+1…5` pages |
| Decision | `Ctrl+Enter` accept & next · `Ctrl+S` accept · `Ctrl+E` exclude · `Ctrl+Z` / `Ctrl+Y` undo / redo |
| Tools | `V` select · `P` pan · `B` brush · `E` erase · `L` lasso · `Space` hold = pan · `[` `]` brush size |
| View | `F` fit · `0` 100 % · `+` `-` zoom · `H` hide masks · `O` opacity |
| Masks | `Tab` / `Shift+Tab` next / previous · `K` keep · `Del` delete · `M` merge · `S` split · `Ctrl+N` new · `Ctrl+D` duplicate · `A` `Shift+A` `I` keep all / none / invert |
| Classes | `1`…`9` assign class · `Ctrl+Shift+C` manage classes |
| Files | `Ctrl+O` open ZIP · `Ctrl+Shift+O` open folder · `Ctrl+Shift+E` export dataset |

## 5. Classes and groups

Class names are human decisions. When a folder name implies the class (`Pieza_con_perno`,
`Pieza_sin_perno`), the app **suggests** it: new masks are pre-labelled and a badge says so, but
nothing is stored until you **Accept**. Folders without a hint (for example `tests/`) get no
suggestion. Class names are never taken from prompt labels or model confidence scores.

Near-identical frames of one scene **must share a capture group**, or the same scene lands in both
training and validation. The default group is the image's folder, which is safe when a folder is
one continuous recording. **Groups…** can instead cut each folder into time blocks (using the
`img_YYYYMMDD_HHMMSS` file names) for a finer split; neighbouring blocks of one burst still look
alike, so shorter windows trade some leakage risk for more groups. Already accepted images keep
the group they were saved with.

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
in the source ZIP. YOLO rows are `class_id x1 y1 x2 y2 ...`, normalized to image dimensions.
Export does **not** copy unreviewed alternatives. The original 04e ZIP remains
the complete proposal archive; keep it together with the review project.

The reproducible split is approximately 70/15/15 by independent capture group,
not image count. Exact original SHA256 duplicates join groups transitively.
At least three independent accepted groups and training foreground are required.
This cannot detect near-duplicates or guarantee balanced classes: inspect
`manifest.json` and group data appropriately.

YOLO polygon labels cannot represent holes. By default export rejects holes;
explicitly enable **Allow hole filling** to approve that lossy conversion while
retaining exact PNG masks (or use *Fill holes* in Review). Disconnected masks are rejected rather
than silently dropping components: *Split* or repair them in Review first. Overlapping instances
remain separate. A failed export leaves its incomplete fresh directory for inspection and never
publishes `manifest.json`; select a new output attempt.

## Augmentation

Training-time augmentation is performed by Ultralytics after the split, so nothing leaks into
`val`/`test`. The page exposes the Ultralytics parameters (flips, rotation, scale, translation,
shear, perspective, HSV, mosaic, mixup, CutMix, copy-paste) and forwards them unchanged. Defaults
keep hue and saturation fixed, because the colour of a metal part carries information, and allow
any rotation, because parts are photographed from above. The preview mirrors Ultralytics'
geometry (`random_perspective`) on your reviewed masks; mosaic, mixup and CutMix happen only
inside training and are not previewed.

Contrast, blur, sensor noise and random occluders exist only in the preview and the offline
export (Ultralytics ignores `erasing` for segmentation, so it is not forwarded). A random
occluder also removes the covered pixels from the masks. The offline export adds augmented copies
of **train** images only, with masks and polygons recomputed (the largest connected piece of each
mask is kept; tiny or vanished masks are dropped) and downscaled to the chosen longest side.
Background replacement is not implemented; inaccurate masks or implausible lighting can make
augmentation harmful.

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

Tests are in `phase_2/tests/test_review_app.py` (data layer, export, worker),
`test_review_app_augment.py` (augmentation engine, class registry, suggestions, theme contrast)
and `test_review_app_ui.py` (offscreen Qt: key events, undo/redo, canvas tools, dialogs,
augmentation page, layout standards). They use handmade masks and do not execute models,
training or Colab. The actual CUDA/CPU training stack must be tested on your chosen machine.

Known limits: the worker starts fresh fine-tuning runs (restoring optimizer state from an
interrupted run is not exposed). Brush strokes and selection stay instant on 12 MP photos, but a
full recompose of 50 large masks on one takes a few seconds (`All` / `None`, zoom steps that change
the outline width); the dataset's 640×480 images are not affected. The augmentation preview
approximates, not reproduces, Ultralytics' pipeline. Quality labels are not inferred. No robot
execution or calibration is added by this application.

References: [YOLOE fine-tuning](https://docs.ultralytics.com/models/yoloe/#train-usage),
[Qt asynchronous processes](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html).
