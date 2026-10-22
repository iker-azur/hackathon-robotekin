# Verification performed

Publication cleanup supersedes the historical notebook-preservation notes below:
04b–04d now contain upload/Drive photo inputs instead of private image payloads.
Their generators no longer read local photos. 04e outputs/widget metadata and
run-specific Drive paths were cleared while preserving the customized export
implementation. Exact pre-cleanup copies were checksum-verified under ignored
`/local/notebook_backup_3fd399272d864024b980d3f8cf072baa/`. New fixture/static tests
cover recursive and single-file input, safe fresh uploads, cancellation, invalid
paths, no published payloads/outputs/private paths, and generators that cannot
read photo bytes. No notebook cells, inference or training were executed.

The 04e export contract is checked structurally, not by equating the user's
customized implementation with a different standalone helper revision.
After cleanup: **84 Phase 1 + 209 Phase 2 tests passed**. The candidate-file scan
found no checked credential patterns, embedded notebook photos, saved outputs or
run-specific Drive paths. Git staging was blocked by `.git/index.lock` permissions;
no commit or push was performed by the agent.

Export listing optimization: only the active 04e export cell was synchronized
with the standalone helper; the non-export cell/output/metadata hash below remains
unchanged. Six new fixture tests cover interrupted directory inventory recovery,
cache reuse without nested enumeration, changed-folder and root-manifest
invalidation, explicit refresh after in-place edits, interrupted refresh safety,
torn journal lines, symlink rejection and a metadata-latency benchmark.
In the 204-file fixture with 1 ms simulated metadata latency, one measured run
was baseline **1.088 s / 626 calls**, new parallel inventory **0.141 s / 246 calls**,
cached inventory **0.052 s / 28 calls**. This is NOT a Google Drive throughput
benchmark or guaranteed speedup. Initial discovery still reads every directory;
reruns assume frozen outputs and recheck directory timestamps/root-manifest
metadata. In-place mask edits without manifest changes require explicit refresh.
Current results: **75 Phase 1 + 191 Phase 2 tests passed**. No notebook cell,
learned model or industrial-data export was executed.

User-edited Colab 04e correction: replaced ONLY active export cell 9 (above the
commented original), clearing that failing cell's old output. All other cells,
saved Colab execution outputs and notebook metadata retain the exact same
JSON-object SHA-256:
`42652066dfe451f2f2adcadcb15b6914df9955d1b9b3fc666eafca2c251c40e6`.
The standalone cell no longer loads a bare EXPORT_LOCAL_DIR variable or calls
the user's older embedded export helper. Four additional fixture tests cover
ranking legacy attempts by reusable payload bytes rather than file count,
checkpoint selection without modifying losing attempts, discovery of local
staging/Drive partial copies, and the missing-variable regression. All older
checkpoint/interruption tests remain green. Current checks: **75 Phase 1 + 185
Phase 2 tests passed**. Notebook static checks intentionally preserve this user's
executed cells and older embedded export helper instead of rebuilding them.
Only metadata is checked when discovering old archives; unreadable old ZIPs with
no checkpoint are reported/preserved, not claimed recoverable. No notebook,
model, training script or real industrial dataset export was executed locally.

Standalone last-cell replacement (`export_last_cell.py`): no notebook, inference
cell, builder or initial-configuration changes. Five new fixture-only tests cover
exact exported bytes/CRC, reuse of a completed ZIP, recovery from an unreadable
interrupted append using a saved central-directory checkpoint, skipping committed
source reads, preservation of damaged old ZIPs, missing-inference reporting, and
absence of model imports/calls. Phase 2 now passes **181 tests**. The replacement
intentionally checks final member names/sizes without a full payload reread;
fixture tests perform full CRC checks separately. Old uncheckpointed incomplete
archives are preserved, not claimed as recoverable. New batch checkpoints are
stored beside the work ZIP on Drive and can resume without the original kernel.
Only handmade fixture files were archived in this verification; no industrial
dataset export, notebook cell or learned model was executed.

2026-10-22 04e resume/export update: five new fixture regressions failed against
the previous implementation and pass after adding ranked manifest discovery,
backward-compatible export-helper settings, source-read-once local ZIP staging,
per-stage progress callbacks and local disk-space checks. Tests verify selection
by unique successful proposals (not errors/files), pruning nested outputs,
incompatible/incomplete/corrupt manifests, matching bytes/hashes/CRC, monotonic
progress reaching each stage total, and old-output overwrite protection.
Notebook static checks cover auto selection, export-only model/auth guards,
inference progress initialization, and export independent of inference-cell state.
The archive fixture records exactly one read per original source file; no Drive
mount/network throughput benchmark or real-data export was performed.

04e addition: eleven notebook documents pass AST, unexecuted-cell and embedded-source
consistency checks. New static checks cover six-method selection, corrected SAM 3
adapter reuse, per-prompt alternatives, incremental Drive indexes, resume skipping,
one-at-a-time model cleanup, guarded CUDA synchronization and no training calls.
Thirteen new fixture-only archive tests cover recursive discovery, duplicate
basenames, ZIP traversal/absolute/Windows paths, symlinks, extraction limits,
case-insensitive duplicate entries, invalid images, lossless original copies,
review placeholders, snapshot hashes/CRC and overwrite protection.
The current existing photos were ARCHIVED, not generated: the input ZIP contains
382 images, 162,971,782 bytes, with exact recursive member-path equality to `/data`
and a successful CRC check. SHA-256:
`27ecc5e1b5867dc1403393104dd146c0720b389da30c6ea194574a5e993f04b7`.
No notebook cells, models, training scripts or learned-model inference were run.
Actual six-model Colab execution, memory use and segmentation quality remain manual
checks for the user. The review app and new training/augmentation controls are not
implemented by this notebook-only stage.

Latest local checks on 2026-10-22, using `.venv/Scripts/python.exe` in the existing
Windows virtual environment (the default system Python does not contain pytest):

- Phase 1: **75 tests passed** through `verify_local.py phase_1`, including static
  checks that the pretrained route never trains, labelled evaluation stays guarded,
  and 04a export/preview retries select fresh output directories. The retry regression
  failed against the original fixed paths and passed after rebuilding the notebook.
- Phase 2: **191 tests passed** through `verify_local.py phase_2`, including mocked
  YOLOE prompt ordering, saved-checkpoint loading, export comparison and unlabelled reporting.
- Backend parity regressions failed on the original wrapper and pass with explicit
  square padding and matching class-agnostic NMS for PT/ONNX. A mocked count mismatch
  saves diagnostics and leaves the manifest unverified. Real-image equivalence must
  still be checked by the user's Colab run; no tolerance was relaxed.
- Demo-photo downloads are tested with in-memory JPEG fixtures: attribution,
  checksum checking, cache reuse, refusal to replace modified files and offline errors.
  Rate-limit regressions failed on the original downloader and pass with bounded
  retries, numeric/date Retry-After handling for 429/503, long-delay deferral,
  stopped network requests after persistent 429, and partial-cache reporting.
  The five-photo catalog includes three additional formed-bracket/connector photos;
  tests check their credits, checksums and stereo/assembled-scene caveats. Source
  pages and original-file links were checked, but new-photo downloads and model
  predictions were not executed locally.
  Wikimedia source pages and licenses were checked through browsing; direct image
  retrieval from the local Windows process was blocked by sandbox networking.
  Live downloads and inference on those photos remain a manual Colab check.
- 04b embeds the exact bytes of all four supplied PNGs, with matching SHA-256,
  decodable images and no image download calls. Tests verify no training calls.
  SAM export utilities preserve holes/overlapping masks, report area filters and
  unknown classes/quality, handle zero predictions and refuse output overwrites.
  These tests use handmade masks, not learned SAM predictions. SAM package install,
  checkpoint retrieval, GPU memory/latency and actual segmentation remain untested.
- 04c embeds all three exact `/data` photos plus the four reference photos and
  configures five default pipelines plus optional SAM 3. Fixture-only tests cover
  shared crops/full-frame mask restoration, detector-box NMS, distinct detector
  confidence vs mask-IoU fields, and one-to-one ground-truth matching that does not
  mistake count agreement for correct masks. Fake external APIs exercise all
  comparison adapter branches; no real forward pass or model installation occurred.
  Four mocked checkpoint cases verify safe initial loading, recovery from the
  reported IndexError, rejection of an invalid state dictionary, and bounded
  failure when both reads fail. Checks preserve the original cache, reuse the
  recovery directory, keep weights_only=True and load model state strictly.
  Florence adapter fixtures check that input conversion receives the model's
  float16 or float32 dtype before generation while token IDs remain intact.
  Twelve detector-loading cases reproduce safetensors header failures using fake
  external APIs. They check successful loading, a fresh-cache recovery, a bounded
  second failure, and no retries for unrelated errors across all three detectors.
  The recovery regressions failed before the patch and now pass. Original files,
  requested revisions and explicit safetensors loading are preserved. Static
  checks confirm all 04c weight caches default to local Colab disk, not Drive.
- 04d now embeds the seven exact supplied photos and uses only facebook/sam3
  through pinned Transformers, replacing the SAM 3.1/A100-only notebook.
  Static checks cover local cache, token handling, unique attempts, guarded CUDA
  timing and no training. Thirty-five mocked SAM 3 cases cover CPU/auto/CUDA
  selection, CPU float32 vs CUDA float16, safe loading/recovery, normal/empty masks,
  holes and shape/score validation. Binary int64 masks match the pinned upstream
  postprocessor's torch.long output. Six binary-numeric regressions failed with
  the original bool-only check and pass after lossless conversion; non-binary
  probabilities, invalid integers, NaN masks and invalid scores remain rejected.
  No BF16 guard, Meta source install or NumPy
  metadata workaround remains. No real model or notebook cell was run.
  Dependency installation, approved access, real CPU/T4/GPU latency and memory use,
  and actual segmentation remain manual Colab checks, not verified compatibility claims.
- All ten notebook documents parse as notebook v4 JSON. Python cells and embedded
  runtime modules pass AST parsing; cells have no execution counts or outputs.
- Embedded runtime text is checked against local source to detect stale notebooks.
- Voice notebooks no longer combine Transformers 5 with Optimum ONNX's
  incompatible Transformers constraint. Native ONNX token batching is checked
  with a mocked runtime; actual encoder export remains a manual Colab check.
- Fixture-only UI checks pass: industrial JSON display, world-state update, masks,
  sorting-plan readiness and simulated verification. Screenshots are under
  `verification/voice_after.png` and `verification/vision_after.png`.
- Before/after layout was inspected. Offscreen Qt initially rendered missing
  glyphs; the fixture runner now explicitly loads the available Arial font.

The Windows test wrapper preserves the test assertions and changes only temporary
directory creation inside its own workspace scratch directory to inherit the
workspace ACL. Direct pytest invocation in this sandbox encountered restricted
temporary-directory permissions.

Compatibility issues discovered and fixed during the full vision suite: blur grew
an ROI silhouette by one pixel; a parked MuJoCo object remained visible below the
table before its scheduled entry; equivalent quality-grouping actions produced
duplicate output. Their original tests now pass.

**Not performed:** notebook execution, dataset generation, model training,
pretrained-weight downloads, real exported-model inference, GPU/Colab execution,
microphone/camera acquisition or physical robot motion. Colab dependency/API
compatibility, learned accuracy, optimized-model equivalence and performance must
be established by the notebook runs. The notebooks contain those checks; this
report does not claim their results in advance.

Existing trained weights and datasets were not replaced. Real quality definitions,
annotated industrial images, recorded Spanish audio, measured calibration and
eventual SO-101/LeRobot validation remain necessary for a hardware demonstration.
