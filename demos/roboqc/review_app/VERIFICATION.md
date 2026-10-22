# Verification scope

The implementation was verified with handmade 64×96 RGB images and binary masks, and,
for interaction and performance, by driving the offscreen window against the real 04e
export (382 images, 3056 proposals). Never with model inference, pretrained downloads,
training or notebook execution.

Regression tests cover, in `phase_2/tests/`:

* `test_review_app.py`: read-only lazy ZIP and nested-bundle access, traversal rejection,
  immutable review revisions, explicit negative images, original-byte preservation,
  independent-group/duplicate-safe splits, selected-only YOLO export and ZIP CRC, explicit
  hole/disconnected-mask handling, static checks of the training worker.
* `test_review_app_augment.py`: augmentation geometry (warped mask vs recomputed polygon),
  determinism by seed, hue preserved by default, occluders and copy-paste, offline export
  leaving `val`/`test` byte-identical, folder class suggestion, time-window groups, the class
  registry (rename keeps slot and colour, merge, protected delete, projects written before it
  existed), worker defaults equal to the previous hard-coded values, and WCAG AA contrast of
  every text/background pair in the light theme.
* `test_review_app_ui.py`: shortcut table without conflicts, plain keys paused while typing,
  the key on every button matching the table, same dimensions on every page at 1280×720 and
  1440×900 with no sideways clipping, real key events through the review flow (navigation,
  method change, tools, classes, undo to the saved state, accept/exclude, accept-and-next,
  the three-way unsaved prompt), brush/eraser/lasso strokes and undo, local overlay
  recomposition, mask operations with undo/redo identity, merge-from-method, explorer
  filter/sort order, the augmentation page (presets, preview, per-project storage, offline
  export), training configuration and metric parsing from saved output, class and group dialogs,
  and background callbacks running on the GUI thread.

Run all Phase 2 tests from the repository root:

```powershell
.\.venv\Scripts\python.exe demos/roboqc/colab/verify_local.py phase_2
```

Screenshots of every page are written to the git-ignored `verification/` folder by the UI
tests (`*_fixture.png`) and were inspected at 1280×720, 1440×900 and 1920×1080: no clipped or
overlapping controls, key caps visible on buttons, light theme throughout.

The actual Ultralytics training/validation worker was only checked statically and through its
configuration mapping. It remains an explicit user-started operation. CPU/CUDA performance,
convergence, downloaded-model compatibility and real segmentation accuracy are not established
by fixture tests.

Results: **242 Phase 2 tests passed, 1 skipped** (59 of them for this application). On a
3024×4032 photo with 50 masks: open 0.4 s, load the 50-mask method 1.3 s, select a mask 0.05 s,
brush stroke 0.12 s, undo 0.07 s, accept 1.6 s; a full recompose of all 50 large masks (`All` /
`None`, zoom steps that change the outline width) takes about 2.5 s. On 640×480 images every
action is instant.
