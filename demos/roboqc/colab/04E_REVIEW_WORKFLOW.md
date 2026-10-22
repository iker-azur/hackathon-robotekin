# 04e: all-model proposals for human review

The local [PySide6 review workstation](../review_app/README.md) now reads the
completed results ZIP directly. Download the completed `.zip` printed by the
last cell from Drive (not a `.work.zip` or `.partial_*` attempt). The application
keeps review choices and mask edits in a separate project, exports approved
YOLO datasets, and offers explicit training/validation without changing this notebook.

Upload [04e_export_segmentation_review.ipynb](04e_export_segmentation_review.ipynb)
to a fresh Colab session. It runs the five 04c methods and the corrected SAM 3
implementation from 04d. It does not train models or select a winner automatically.

For an existing Colab run needing only export, copy the entire contents of
[export_last_cell.py](export_last_cell.py) into the active export cell (above any
commented original). It is standalone and does not require `EXPORT_LOCAL_DIR` or
new configuration fields. Leave inference and earlier cells unchanged. It reviews
all discoverable attempts for the selected bundle and resumes the one with the most
recoverable payload bytes. It preserves original legacy attempts, reports unreadable
uncheckpointed ZIPs, and checkpoints new exports on Drive. Do not rebuild a manually
edited notebook merely to update its last cell.

The standalone cell's listing uses four folder workers, one DirEntry metadata
lookup per entry, and a resumable `.inventory.jsonl` outside the source bundle.
Unchanged directories reuse cached file metadata; folder timestamps detect additions/
removals, and root-manifest metadata changes invalidate the relevant inventory.
The cache assumes a **frozen, finished bundle**: do not edit masks or run inference
concurrently. After manually editing existing files in place, add
`REFRESH_EXPORT_INVENTORY = True` before the final call to force a new listing;
set it back to `False` afterward. `EXPORT_LISTING_WORKERS` optionally changes the
default four workers (1–16). The first scan is still necessary; an old pre-cache
listing cannot be resumed retroactively. If that old listing already finished,
let the current export continue instead of restarting merely for this optimization.

## Run

1. From the project root, package the current photos with:

   ```powershell
   python demos/roboqc/colab/prepare_04e_inputs.py
   ```

   The prepared `assets/data_04e_inputs.zip` includes images recursively from
   `/data`, with their subfolder paths. The current snapshot contains 382 photos.
   The script refuses to overwrite an existing ZIP. For new photos, choose a new
   destination: `--output demos/roboqc/colab/assets/data_04e_inputs_v2.zip`.
   Packaging only archives existing bytes; it does not generate datasets or labels.

2. Enable Colab Secrets access to `HF_TOKEN` for an account approved for
   `facebook/sam3`. A read token is sufficient. GPU is recommended; CPU works but
   may be too slow for the full collection. No A100-only requirement.
3. For a NEW dataset/run, set `RESUME_BUNDLE=''`. Leave `INPUT_PATH=''` and upload the ZIP when asked. The notebook saves it in
   Drive under `ROOT/inputs/`, prints the path, then reads that copy. Alternatively
   upload it yourself and set, for example:

   ```python
   INPUT_PATH = '/content/drive/MyDrive/Robotekin2026/inputs/data_04e_inputs.zip'
   ```

   A Drive folder is also supported, recursively. Uploading only the notebook does
   NOT make your PC's `/data` folder accessible to Colab.
4. Optionally start with `MAX_IMAGES=3`; the default `None` processes all images.
   All input images are included in the archive even during a limited smoke test;
   omitted inference results remain absent, not empty predictions. With three SAM 3
   prompts, each processed image has eight alternatives across six model methods.
5. Run cells manually in order. Weights download to local Colab disk, never Drive.
   Each image/model/prompt saves an outcome immediately to the Drive bundle.
   Failure reports include full tracebacks; failed methods are not silently replaced.
6. The final cell builds and verifies a new ZIP locally, then copies it to `ROOT/exports/`, with a printed
   SHA-256. Download from Drive. It includes all originals, normalized RGB images,
   per-model masks/overlays, errors, settings, provenance, runtime source and review
   placeholders. No gated weights or tokens are included.

## Resume automatically or export without inference

For interrupted runs, use:

```python
RESUME_RUN = ''
RESUME_BUNDLE = 'auto'
EXPORT_ONLY = False
```

Discovery searches run folders below `ROOT`, reads small manifests rather than
every mask, and skips the contents of already identified bundle folders and
`inputs/exports`. It displays a table and selects the compatible bundle with the
most unique successful indexed proposals. Failed attempts, duplicate records and
stray files do not increase that count. Different bundles are NOT merged. Check
the selected path/image count: different datasets can share inference settings.
Set an exact bundle path to override auto-discovery; set `''` to explicitly start
a NEW bundle. If bundles exist but none are compatible, discovery stops with an
explanation rather than silently redoing the collection.

Keep the original inference parameters, image limit and prompts. Updated export/
discovery helper code is compatible with earlier 04e settings; inference-source or
parameter changes remain blocked. Successful proposals are skipped, failed/missing
ones retry into fresh attempt folders, and existing masks/reviews are preserved.
An interrupted input-registration step requires a new bundle from the input ZIP.

The inference progress bar starts at saved successful work and advances per new
attempt. Failures are counted explicitly, not reported as successful masks; the
final missing/failed count can be nonzero even at 100% of this session's attempts.
ETA is approximate across different models, and downloads/loading can pause the bar.

If only the last cell was interrupted, use:

```python
RESUME_BUNDLE = 'auto'  # or the exact saved folder
EXPORT_ONLY = True
```

Run setup and discovery, then the final export cell. You can skip the inference
and preview cells. Export-only does not log into Hugging Face, install SAM 2's source
package, download model weights or load/run a model; common dependencies still install.
A CPU runtime is sufficient. It can also export incomplete results deliberately.
The final cell rereads the saved proposal index, so it does not depend on a prior
inference cell's in-memory `records` variable.

Export progress shows **Scan files → Archive + hash → Verify local ZIP → Copy ZIP
to Drive**, with byte totals/rates/ETA after the inventory is known. Source bytes
are read once from Drive while hashing and writing a local ZIP. Verification is
local (member CRC and SHA-256); the Drive copy gets a size check, not another full
remote read. `EXPORT_LOCAL_DIR` must be local disk with space approximately equal
to the entire bundle. Existing compressed PNG/JPEG/NPZ are stored without ZIP
recompression to reduce CPU work; the ZIP may be larger than before. Actual Drive
speed still depends on the mount and number of files; it was not benchmarked locally.

Do not run inference or annotation writes concurrently with export. Every export
uses a fresh archive name. Interrupted Drive copies remain `.partial_*`, not a
completed `.zip`; old complete ZIPs and source masks are untouched. Restarting
export rebuilds the ZIP—it does not resume midway through ZIP bytes—but never
requires redoing model inference. Allow time for Drive to finish syncing files.

## Archive contract

```text
images.json                         # IDs, relative source paths, hashes, dimensions
images/<id>/original.<ext>           # exact original bytes
images/<id>/rgb.png                  # EXIF-normalized RGB; ALL mask coordinates
proposals.json                      # latest outcome per image/method/prompt
proposals/<id>/<method>/<attempt>/
  provenance.json
  result.json                       # status and coordinates; error.txt on failure
  masks/candidates.json             # mask indices, filters, scores, geometry
  masks/raw_masks.npz               # ALL binary masks, including filtered masks
  masks/mask_XXXX.png               # separate retained instance masks, 0/255
  masks/overlay.png
reviews.json                        # one unreviewed record per input image
registration_complete.json
settings.json
summary.csv
checksums.json                      # SHA-256 of every other file
environment.txt
runtime/
```

Image IDs hash the nested relative path and original content, so duplicate filenames
in different directories cannot collide. A PNG per instance preserves holes and
overlaps; a merged foreground mask would lose the boundaries needed for picking.
Prompt labels are proposals, not confirmed part classes. Geometric grasp candidates
are unvalidated. Neither confidence nor a correct object count proves accuracy.

## Next implementation stages

1. **Review app:** load the ZIP safely, show original and alternative overlays,
   select a starting proposal, then add/delete/split/merge/paint instance masks and
   assign class IDs. Save ONE final annotation set per image, containing MULTIPLE
   instance masks. `reviews.json` records acceptance, source proposal and session
   grouping. Preserve the original proposal bank and save edited masks separately
   under `reviewed/<image_id>/`. Explicitly accepted empty scenes differ from
   unreviewed images. The reviewer defines the class taxonomy; folder names are not
   automatically trusted as instance labels.
2. **Dataset export:** use accepted annotations only; freeze class IDs and grouped
   train/validation/test splits by capture session/scene before augmentation.
   Consecutive frames, duplicates and augmented descendants belong to one split.
   Keep PNG masks as masters. COCO masks preserve holes; YOLO polygon conversion
   needs an explicit policy for holes and disconnected components, with visual QA.
3. **Training:** expose a compact YOLO segmentation baseline first, with an optional
   YOLOE fine-tuning path subject to its training API and licensing. Reuse the
   existing 05/06 supervised training/evaluation stages after validated export;
   do not treat the unreviewed 04e archive as training-ready ground truth.
4. **Augmentation:** joint geometry transforms for image and masks; image-only
   illumination/noise changes; reviewed-mask cut/paste/background replacement and
   controlled overlap with updated visible masks. Preserve semantic color/defect
   labels, holes, instance identity and plausible scale. Avoid synthetic defects
   without reliable labels. Apply augmentation to training only and evaluate on
   untouched real images, including clutter and reflections.

The notebook is implemented now; the app, reviewed-label exporter and new training
controls are separate next steps. Model quality, runtime memory and GPU compatibility
still require your manual Colab run; no model inference was executed locally.
