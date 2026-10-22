# RoboQC — Phase 1 and Phase 2

> Phase 2 code lives in this folder. It uses the Phase 1 package at `../phase_1/src/roboqc` and the models at `../phase_1/models`. See the Phase 1 README for the parser and speech setup.

Offline Spanish voice understanding (Phase 1) and lightweight, deterministic
computer vision for industrial inspection (Phase 2). CPU only, no GPU, no
robot control yet.

> **Status:** Phase 2 code has been written and reviewed by reading only. It has
> not been executed. The first run is the validation step. Expect to adjust HSV
> ranges and area thresholds for your camera and lighting.

---

## 1. Phase 1 review

### What was checked
Speech engines, intent classifier and its metrics, entity extraction, the rule-based
parser, the PySide6 window, workers, configuration and tests. Not reviewed: the dataset
generator and templates (`dataset/generator.py`, `dataset/templates.py`), and the
JEPA and MiniLM training code. Their metrics files were read, not their implementation.

### Findings and changes

| # | Finding | Status |
|---|---------|--------|
| 1 | `sí`/`si` matched as a substring: "visible" or "posible" triggered `CONFIRM_ACTION`. | **Fixed**: word-boundary match (`parser.py`). |
| 2 | Plural `malas`/`malos` were not quality words, so "las piezas malas" lost its quality. | **Fixed** (`entities.py`). |
| 3 | Verbless clauses inherited the previous intent even when they were visual queries. | **Fixed for visual queries** only (`parser.py`). |
| 4 | Worker signals named `finished` shadowed `QThread.finished`. | **Fixed**: renamed to `recorded`, `transcribed`, `classified`. |
| 5 | Model path defaulted to `models` relative to the working directory, so the app failed when launched from elsewhere. | **Fixed** in `phase_1/04_app.py`. |
| 6 | Two tests asserted nothing (`or True`, and an `or` that is nearly always true). | **Fixed** (`test_parser.py`). Two other weak tests remain: `test_negation_scope_variants` has no assertion and `test_two_colors_roles` accepts either color. |
| 7 | `config/default.json` is never read. | **Not changed.** Wire it in or delete it. |
| 8 | Dead code in `parser.py`: a no-op safety branch in `classify_intent_rule`, unused `_inherit_intent` and `_DEST_PREPS`. | **Not changed** (out of scope). |
| 9 | Trained-model quality on quality-rule commands: MiniLM (the selected model) reaches recall 0.54 on `CHANGE_QUALITY_RULE`, JEPA 0.48, TF-IDF 0.92. `CHANGE_DESTINATION` has F1 0 in TF-IDF. | **Mitigated**, not retrained. Visual queries are resolved by deterministic rules before the classifier (see §3.4). |
| 10 | `ClassificationWorker` runs the classifier per segment and again for top-k, so each command is inferred twice. | **Not changed.** Efficiency note. |
| 11 | The Phase 1 window loads the model synchronously on startup and blocks the UI. | **Not changed** in `--classic`. The Phase 2 workstation loads it in a worker. |

### Why the classifier was not retrained
The new visual intents live outside `Intent` and `ALL_INTENTS` on purpose.
`test_every_intent_is_validated_and_tested` requires dataset coverage for every
value in `ALL_INTENTS`, so adding them there would force a dataset regeneration and
retraining. The deterministic layer answers those queries reliably, so retraining is
not needed for Phase 2.

---

## 2. Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Phase 2 adds `opencv-python`. Phase 1 needs the Vosk model and optionally
faster-whisper (see Phase 1 notes below). `scipy` is used by the tracker.

Install PyTorch CPU only if you use the JEPA or MiniLM models:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Phase 2 architecture

```
camera ─► detector ─► tracker ─► world_state ◄─ quality rules
(OpenCV)  (HSV, contours)  (Hungarian IDs)  (thread-safe)   ▲
                                 │                          │
                    ┌────────────┴──────────┐     resolver (commands)
                    ▼                       ▼               ▲
            inspection report         PySide6 UI    Phase 1 pipeline
                                                    (voice / text)
```

### 3.1 Modules (`src/roboqc/vision/`)

| Module | Responsibility |
|--------|----------------|
| `camera.py` | Webcam, video file or still image sources. Explicit `CameraError` messages. FPS and latency meter. |
| `color.py` | HSV color classification by pixel vote (dominant color with minimum fraction). |
| `shape.py` | Contour features: vertices, circularity, solidity, minimum-area rectangle, orientation. |
| `detector.py` | Foreground mask, morphology, contours, ROI handling. Returns single-frame `Detection`s. |
| `tracker.py` | Stable IDs (`OBJ_001`, …) via centroid distance and Hungarian assignment. Min-hits confirmation, occlusion memory, enter/move/disappear events. |
| `quality.py` | `GOOD` / `DEFECTIVE` / `UNCERTAIN` from observable criteria and runtime rules. |
| `world_state.py` | Thread-safe `WorldState` and immutable `WorldSnapshot`. No Qt, no OpenCV. |
| `calibration.py` | Optional planar homography pixel → table coordinates. Only from user-supplied points. |
| `pipeline.py` | Detector + tracker + world state for one frame. |
| `resolver.py` | Answers structured commands from the snapshot. Rule-based. Never claims a physical action. |
| `inspection.py` | Structured JSON report and terminal table. |
| `overlay.py` | Drawing for the UI and the diagnostic scripts. |
| `config.py` | Defaults merged with `config/vision.json`. |

Layers are separate so the robot planner can depend only on `world_state`.

### 3.2 Pipeline per frame

1. Crop to ROI (if set), Gaussian blur, convert to HSV.
2. Foreground = saturated pixels (`min_saturation`, `min_value`). Dark or bright achromatic objects are opt-in.
3. Morphological open then close removes noise and fills holes.
4. External contours, area filter (`min_area_px`–`max_area_px`).
5. Per object: color vote over its own pixels; shape from contour features.
6. Tracker associates detections with tracks (Hungarian, `max_distance_px`).
   A track is confirmed after `min_hits`. Unmatched tracks are kept as `OCCLUDED`
   for `max_missed_frames`.
7. Quality is evaluated when a snapshot is taken, so rule changes apply to the current frame.

### 3.3 Structured detection output

```json
{
  "object_id": "OBJ_001",
  "type": "cubo",
  "color": "rojo",
  "shape": "cuadrado",
  "position_px": [320.0, 180.0],
  "bounding_box": [290, 150, 60, 60],
  "area_px": 3580.0,
  "orientation_deg": 12.4,
  "quality": "GOOD",
  "quality_reasons": [],
  "detection_status": "VISIBLE",
  "motion": "STATIONARY",
  "confidence": 0.9,
  "world_position": null,
  "selected": false
}
```

`world_position` is non-null only when a calibration file is configured. It contains
`frame`, `units` and `xy`. Pixel coordinates are never converted into millimetres
without calibration.

### 3.4 Command interpretation for visual queries

`commands/vision_intents.py` detects visual queries before the trained classifier runs:

| Intent | Example | Behaviour |
|--------|---------|-----------|
| `COUNT_OBJECTS` | "¿Cuántas piezas rojas hay?", "¿Hay alguna pieza defectuosa?" | Counts visible matches. |
| `LOCATE_OBJECT` | "¿Dónde está la pieza roja?" | Position in px. Asks for clarification when the singular reference matches several objects. |
| `SHOW_OBJECTS` | "Muéstrame las piezas verdes" | Highlights matches on the camera view. |
| `SELECT_OBJECTS` | "Selecciona las tres piezas azules" | Selects up to N matches and says so if fewer exist. |
| `INSPECT_*`, `CLASSIFY_OBJECT`, `SORT_OBJECTS` | "Inspecciona todos los cubos azules", "Clasifica las piezas buenas y malas" | Inspection detail, or groups by quality. |
| `CHANGE_QUALITY_RULE` | "Ahora las piezas azules son defectuosas" | Adds a runtime rule. The classification changes at once. No retraining. |
| `REQUEST_STATUS` | "¿Cuántas piezas quedan?" | Phase 1 label kept. Answers with the inspection summary. |

Robot commands (pick, place, move, …) are recognized but answered with
"no controla el robot. No se ha ejecutado ninguna acción física."

### 3.5 Quality classification

Verdicts are checked in this order:

1. **Active runtime rule** (newest first). A rule for the same color/type replaces the older one.
   `DEFECTIVE` rules reject the object. `GOOD` rules accept the color.
2. **Geometric and size checks**, when the object is fully visible:
   - Solidity below `quality.min_solidity` (0.88): missing corners or chipped outline.
   - Area outside `expected_area_px` ± tolerance (only when configured).
   - Color not in `expected_colors`, or type not in `expected_types` (only when configured).
3. **Uncertainty** is returned when evidence is insufficient: partially visible object,
   inconclusive color (below `min_color_confidence`), unrecognized outline. Uncertainty
   never becomes `GOOD`.

Every verdict carries `quality_reasons`, so the UI can show why a piece was rejected.

---

## 4. Supported objects and colors

| Object (`type`) | Shape from above | Notes |
|-----------------|------------------|-------|
| `cubo` | square outline | Tilted cubes with visible side faces may become `desconocido` → `UNCERTAIN`. |
| `prisma` | rectangle or triangle | Triangle prisms seen as triangular sections. |
| `circular` | circle | **Cylinder and sphere are indistinguishable from above.** Queries for "cilindros" or "esferas" match `circular` objects and the answer says so. |
| `desconocido` | other outlines | Always `UNCERTAIN`. |

Colors (defaults in `config/vision.json`, OpenCV HSV scale): `rojo`, `naranja`, `amarillo`,
`verde`, `azul`, `morado`, `blanco`, `gris`, `negro`.
Pink (`rosa`) and brown (`marrón`) are recognized in voice commands but have no default HSV range.
`white`, `gray` and `black` need the opposite-tone background. See §6.

---

## 5. Camera setup

- Fixed overhead webcam, lens perpendicular to the table, the whole work area in view.
- Use a matte, uniformly colored background that contrasts with the objects (e.g. dark
  gray mat for colored parts). Avoid patterns, reflective or colored surfaces that match
  the objects.
- Lighting: diffuse, constant, from above. Avoid glare on glossy parts and strong shadows.
  Changing light changes HSV values, so re-check the color ranges after moving lamps.
- Keep the camera still. Re-select the ROI after moving it (`05_test_camera.py`, key `r`).
- Keep the camera-to-table distance constant. Pixel areas depend on distance; set
  `expected_area_px` only for a fixed setup.

---

## 6. Recommended lighting and background

- Colored objects on a neutral mat: default settings work.
- Black objects: set `foreground.detect_dark` to `true` on a light background.
- White or gray objects: set `foreground.detect_bright_achromatic` to `true` on a
  dark background. Expect more false positives from reflections.
- Mixed white and black on one table is not supported by the default settings.

---

## 7. Quality rules and configuration

Settings live in `config/vision.json`. Missing keys fall back to built-in defaults.
Key sections:

- `roi`: `[x, y, w, h]` in camera pixels, or `null`. Save it with `05_test_camera.py` (key `r`).
- `foreground`, `morphology`, `shape`, `color` (per-color HSV ranges): segmentation.
- `tracking`: `max_distance_px`, `max_missed_frames`, `min_hits`, `motion_threshold_px`.
- `quality`: `min_solidity`, `expected_colors`, `expected_types`, `expected_area_px`.
- `calibration_path`: optional JSON with `pixel_points` and `world_points` (≥ 4 pairs).

Runtime rules from voice or text are kept in memory only. Restarting the application
restores the configuration. Persisting rules is not implemented.

---

## 7.5 Simulated workspace (no camera needed)

Every Phase 2 script and the workstation accept a simulated source. Scenes are rendered
with OpenCV, with optional noise, moving objects, and objects that enter at a given frame.
The default source is `sim:demo` (`camera.index` in `config/vision.json`), so the
commands in §8 run without hardware. To use a real webcam, pass `--source 0` or set
`camera.index` to `0`.

Built-in scenes:

| Name | Content |
|------|---------|
| `sim:demo` | Two red cubes, a blue cylinder-like disc, a green prism moving slowly to the right, and a yellow cube entering at frame 45. Loops every 300 frames. |
| `sim:static` | Three objects without motion and without noise. Useful for exact checks. |

Your own scene: write a JSON file and pass its path as `sim:path/to/scene.json`.

Your own scene: write a JSON file and pass its path as `sim:path/to/scene.json`.

```json
{
  "width": 640, "height": 480, "background": [40, 40, 40],
  "noise_sigma": 3.0, "seed": 42, "loop_frames": 300,
  "objects": [
    {"id": "cubo_rojo", "shape": "square", "color": "rojo", "center": [150, 140], "size": [60, 60]},
    {"id": "cilindro_azul", "shape": "circle", "color": "azul", "center": [320, 240], "radius": 40, "velocity": [0.5, 0.0]},
    {"id": "prisma_verde", "shape": "rect", "color": "verde", "center": [480, 340], "size": [110, 55], "enter_frame": 30}
  ]
}
```

- `shape`: `square`, `rect`, `circle`, `triangle` (triangle uses `radius` as half-size).
- `color`: `rojo`, `naranja`, `amarillo`, `verde`, `azul`, `morado`, `blanco`, `gris`, `negro`.
- `velocity` is in px per frame. `enter_frame` and `exit_frame` set visibility.
- The `id` field is only for your reference. The pipeline assigns its own `OBJ_###` IDs.

Known positions are available in code with `simulation.ground_truth(scene, frame)`.
Tests use it to compare detections with the truth.

Simulated frames are generated as fast as the CPU allows, not at camera frame rate.
Use `display_fps` to limit the preview rate.

### 7.6 Physics workspace (`sim3d:`)

`sim3d:` runs a MuJoCo physics scene: a table, a camera with shadows, and rigid objects.
Objects fall under gravity, rest on the table and can be knocked over. Objects with
`velocity` are kinematic: they follow a fixed path, which stands in for a conveyor.
Ground truth is read from the physics state, so the reported positions match the render.

```
python 08_app.py --source sim3d:demo3d
python 06_test_detection.py --source sim3d:demo3d
```

It needs `pip install mujoco` (listed in `requirements.txt`) and an OpenGL-capable GPU or
driver. Shading is rasterised with shadows, not ray traced: realistic in lighting and physics,
not photoreal. Use Isaac Sim for photoreal rendering, which needs an NVIDIA RTX GPU.

The table is neutral grey on purpose. The detector treats saturated pixels as objects, so a
warm wooden table would be segmented as one large object and hide the pieces on it. The real
workspace needs a neutral work surface for the same reason.

| Name | Content |
|------|---------|
| `sim3d:demo3d` | Two red cubes, a blue cylinder, a green box moving along the table, and a yellow cube dropped in at frame 45. Loops every 300 frames. |

Your own 3D scene: a JSON file with `camera` (`eye`, `target` in mm, `fovy` in degrees),
`table` (`size` = [width, depth] in mm), and `objects`:

- `kind`: `box` (`size` = [width, depth, height] in mm) or `cylinder` (`radius`, `height` in mm).
- `center` is the position on the table in mm. `velocity` is mm per frame; an object with
  `velocity` must start at frame 0.
- `enter_frame` drops the object onto the table from 0.15 m above it.

Ground truth gives each object's pixel centre (`center`) and table position (`center_mm`).
World positions in the workstation still need a calibration file (`calibration_path`, §7).
The simulated table corners are not written to one.

Not run yet: the physics and rendering have not been executed on this machine, so the first run
may need adjustments to the scene XML or the test thresholds.

## 8. Phase 2 scripts

All three scripts are independent and can be run on their own. They do not control a robot.

### 8.1 `05_test_camera.py` — camera diagnostics

```bash
python 05_test_camera.py --list                              # list webcams 0-3
python 05_test_camera.py --source 0                          # live preview (q quits)
python 05_test_camera.py --source 1 --width 1280 --height 720
python 05_test_camera.py --source samples/setup.mp4          # video file
python 05_test_camera.py --save-frame samples/setup.png      # save one frame and exit
python 05_test_camera.py --frames 200                        # headless: FPS and latency
```

In the preview: `s` saves a frame into `samples/`, `r` selects the workspace ROI and
writes it to `config/vision.json`.

### 8.2 `06_test_detection.py` — detection, tracking, quality

```bash
python 06_test_detection.py                                  # webcam from config
python 06_test_detection.py --source samples/setup.png --frames 1 --no-window
python 06_test_detection.py --source samples/setup.mp4 --frames 300 --json-out samples/state.json
python 06_test_detection.py --command "Ahora las piezas azules son defectuosas"
python 06_test_detection.py --calibration config/calibration.json
```

`--no-window` prints one JSON record per object and frame. `--command` applies a
visual rule before processing.

### 8.3 `07_test_inspection.py` — inspection workflow

```bash
python 07_test_inspection.py                                 # interactive terminal
python 07_test_inspection.py --command "Inspecciona las piezas rojas." --frames 60
python 07_test_inspection.py --command "¿Cuántas piezas azules hay?" --command "Ahora las piezas azules son defectuosas." --frames 60
python 07_test_inspection.py --source samples/setup.mp4 --command "¿Hay alguna pieza defectuosa?" --report-json samples/report.json
```

Interactive commands: type an instruction; `:report` prints the inventory; `:save RUTA.json` writes the report; `:quit` exits.
In `--command` mode the `:` commands are not processed. Use `--report-json` to write the report to a file.

---

## 9. PySide6 workstation

```bash
python 08_app.py                                 # two tabs: Comandos (type and speak here) and Inspeccion (camera, inventory, results)
python 08_app.py --source sim:demo                 # simulated scene instead of webcam
```

Layout:

- **Camera:** live feed with bounding boxes, IDs, type, color and quality. Selected objects are highlighted in white.
- **Inventory:** one row per visible or occluded object (ID, type, color, shape, quality, view status, motion, confidence). Click rows to highlight them on the camera view.
- **Instruction:** voice (record, transcribe, edit) or text. Ctrl+Enter runs the command.
- **Summary:** visible, good, defective, uncertain.
- **Result:** status, interpreted intents and parameters, and the answer with reasons.

Threads: camera capture and vision run in `CameraWorker`. Speech and classification run in
the Phase 1 workers. The UI thread only renders state.

---

## 10. CPU performance recommendations

- Use 640×480 at 15 fps for the camera. Higher resolution multiplies contour and HSV work.
- Set a ROI so only the work area is processed.
- Keep `display_fps` at 10–15. The processing loop still runs every frame for tracking; only the UI emission is throttled.
- `morphology.kernel_size` 5 is a good default. Larger kernels are slower.
- Use Whisper `tiny` or `base` on low-end CPUs (Vosk small is the lightest option).
- Run the JEPA and MiniLM models only if needed. Visual queries do not use them.
- Close other camera applications. A webcam usually opens in one process at a time.

Measure on your hardware with `05_test_camera.py --frames 200` (FPS and latency) and
the `processing_ms` value printed by `06_test_detection.py`.

---

## 11. Known limitations

- **Not executed yet.** No test, script or application has been run. Expect small
  issues in HSV ranges, thresholds and UI layout.
- Overhead view: cylinders and spheres cannot be told apart; cubes at an angle may be
  `UNCERTAIN`.
- Pixel areas and positions depend on distance and lighting. No calibration is
  performed automatically.
- White and black objects need a background of the opposite tone.
- Tracking is centroid-based. Two objects that cross paths may swap IDs when their
  colors and sizes match.
- Quality rules target color or object type. Marking one specific piece by hand is not
  supported.
- Runtime quality rules are not saved between runs.
- The trained intent models do not know the visual intents. Those commands depend on the rule layer, and
  paraphrases outside its patterns fall back to the classifier.
- `CHANGE_DESTINATION` and other robot commands are recognized but never executed.
- Selection with a quantity below the number of matches picks the first N by ID. It does
  not choose by distance.

---

## 12. Preparation for SO-101 integration

- `WorldState.snapshot()` is the interface for the planner: IDs, types, colors, pixel
  positions, bounding boxes, quality, visibility, motion, timestamps and optional world coordinates.
- `calibration.py` already provides the homography interface. Add correspondences by
  measuring points on the table with the robot.
- Motion is the `motion` field plus `first_seen`/`last_seen`. Grasp decisions can use the
  quality verdict and reasons.
- No robot code exists in this phase. The resolver's refusal messages mark where robot
  actions will plug in.

---

## 13. Tests (manual execution only)

Install `pytest`, then from this folder:

```bash
pytest -q                                              # all tests
pytest -q tests/test_vision_intents.py                 # Phase 1 fixes and visual intents
pytest -q tests/test_vision_color_shape.py             # color and shape
pytest -q tests/test_vision_pipeline_units.py          # detection on synthetic frames, tracking, quality
pytest -q tests/test_vision_world_resolver.py          # world state and resolver scenarios
pytest -q tests/test_vision_io.py                      # camera errors, calibration, config
```

Vision tests skip automatically when OpenCV is not installed. They draw synthetic
frames, so they do not need a camera. For real-camera validation use the scripts in §8.

---

## 14. Phase 1 usage (unchanged)

```bash
python 01_test_voice.py --list-devices
python 01_test_voice.py --mic
python 01_test_voice.py --file samples/test.wav
python 01_test_voice.py --compare --file samples/test.wav
python 02_generate_dataset.py --samples 25000 --seed 42
python 03_train_model.py --model baseline
python 03_train_model.py --model auto
```

Environment variables: `ROBOQC_MODEL_DIR` (default: `models` next to this file), `ROBOQC_MODEL`
(`baseline`, `minilm` or `jepa`), `VOSK_MODEL_PATH`.

Vosk Spanish model: extract `vosk-model-small-es` to `models/vosk-model-small-es/`, or run
`python install_vosk_model.py --yes`.

---

## 15. Architecture summary (Phase 1)

```
Voice or Text -> STT -> Normalization -> Segmentation -> Intent
-> Entity extraction -> Validation -> Structured Command -> UI
```

Phase 2 adds a deterministic visual-query layer before the intent classifier and a
resolver after it, which answers from the world state.
