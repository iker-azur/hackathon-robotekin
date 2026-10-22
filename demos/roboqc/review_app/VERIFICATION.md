# Verification scope

The implementation was verified with handmade 64×96 RGB images and binary masks,
never with model inference, pretrained downloads, training or notebook execution.

New regression tests (`phase_2/tests/test_review_app.py`) cover read-only lazy ZIP
and nested-bundle access, traversal rejection, immutable review revisions, explicit
negative images, original-byte preservation, independent-group/duplicate-safe
splits, selected-only YOLO export and ZIP CRC, explicit hole/disconnected-mask
handling, four-page Qt rendering, background proposal loading/review saving and
live metrics/log parsing from saved fixture output.

Run all Phase 2 tests from the repository root:

```powershell
.\.venv\Scripts\python.exe demos/roboqc/colab/verify_local.py phase_2
```

The review screenshot at `verification/review_fixture.png` is a new application
view, not a before/after comparison with an existing GUI. It was visually inspected:
four tabs, numbered active mask, instance checklist, class assignment, group input,
paint/erase controls and accept/exclude actions render without overlapping controls.

The actual Ultralytics training/validation worker was only statically checked.
It remains an explicit user-started operation. CPU/CUDA hardware performance,
convergence, downloaded-model compatibility and real segmentation accuracy are
not established by fixture tests.

Results: all **209 Phase 2 tests passed**; all **18 review-app tests passed** in
a separate focused rerun. Phase 1 had **74 passed, 1 failed**: the already edited
04e notebook's active export cell differs from `export_last_cell.py`, which an
existing exact-source-match assertion rejects. Neither file was changed by this
application task; no notebook cells were rebuilt to conceal that mismatch.
