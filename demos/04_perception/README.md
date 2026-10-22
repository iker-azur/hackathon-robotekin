# 04 — Perception and grounding

A camera on the table, and a list of what is on it in a file that the robot can act on.

This is the demo that demo 08 promises. `08_voice_hybrid.py` answers a question about
the table with *"Todavía no puedo ver la mesa: la cámara llegará en la siguiente demo"*,
and its README names the camera demo as the thing that replaces that answer. This is
that demo:

```text
"coge la bola roja de la izquierda"
  -> command  {"kind": "ball", "color": "red", "position": "left"}      (demo 03)
  -> scene    {"objects": [...], "containers": [...]}                    (this demo)
  -> target   object #3 at (0.42, -0.36, 0.056)                          (this demo)
```

Three things are produced, and they can be run separately because they fail separately:

| | Script | What it does |
| --- | --- | --- |
| the camera | `09_camera_to_scene.py` | adds a camera to the simulation task, fills the table, writes frames |
| the scene | `perceive.py` | image to a scene JSON. No Isaac, no torch, no models |
| the answer | `09_ground_command.py` | command plus scene, or a question back |

## What is measured and what is not

**Measured, off rendered frames of the top camera** (`--calibrate` reads them back):

- The HSV bands in `colors.py`. Two findings changed the file. The rendered colours are
  far less saturated than their albedos suggest — the blue cube's centre measures S = 75
  where the albedo predicts 235, because a 1000-intensity dome light washes the chroma
  out, so `sat_min` is 60 and not 110. And the wood table measures H = 18, S = 82, which
  is inside the yellow hue band; yellow needs `sat_min = 130` against the wood's 95 peak,
  and it is off by default.
- `colors.CONTAINER_MIN_AREA_PX = 1200`, between the largest small object measured
  (702 px) and the smallest zone (about 1900 px).
- `perceive.BALL_FILL = 0.985`. Over 20 trials, 3 cm cubes fill 0.9562–0.9780 and 4 cm
  balls fill 0.9985–1.0001, so the threshold sits in a gap the data agrees on.
- The camera convention. Both lateral signs are negative (`u = cx - fx·y/x`,
  `v = cy - fy·z/x`); see `scene.CameraModel.to_camera` for why checking only that objects
  land *in frame* is not enough to settle it.

**Still not measured:** anything about a scene other than the one in `scene_spec.py`.
Change the objects, the lighting or the camera and every number above has to be
re-measured; `--calibrate` is the tool for that.

**Measured, from the shipped USD rather than assumed:** the table's top surface is at
z = 0.0409 and spans x 0.0032..0.7032 and y -0.6768..-0.0268; the shipped cube is a
3 cm textured glossy **red** cube at (0.3461, -0.3616, 0.0615); the robot base sits at
(0.35, -0.64) facing +y, so the robot's left is -x. These come from walking
`assets/scenes/table_with_cube/scene.usd` with `pxr`.

**Derived, then re-measured:** the top camera's ground resolution. It comes out at
**746 px/m** on the table, not the ~1200 px/m the nominal focal length and height suggest,
because the camera is tilted 67 degrees and foreshortens the view direction.
`CameraModel.pixels_per_m` measures it by projecting two points 10 cm apart rather than
computing it, because the number is not the same in both axes.

**Checked by `scene_spec.clearance()`, reported by `--calibrate`:** the default layout and
its rearrangements keep every same-colour pair at least 66 mm apart (38 px at 746 px/m,
against a 3 px morphological opening) and nothing sits further than 0.33 m from the robot
base, inside the SO-101's advertised 350 mm reach. The layout was fitted to those four
constraints rather than chosen by eye; the search and what it rejected are written up at
`DEFAULT_SCENE` in `scene_spec.py`.

## Install

Nothing new. `opencv-python-headless`, `numpy` and `transformers` are already in the
`hackathon-robotekin` environment, and `transformers` is what OWLv2 needs, so the
optional detector adds no dependency at all. The version matrix is the one in
[`../01_simulation_basics/README.md`](../01_simulation_basics/README.md).

## Running it

```powershell
# 0. the bands and the layout, in half a second, without starting Isaac Sim
python .\demos\04_perception\09_camera_to_scene.py --source spec

# 1. build the scene, look at it, score every detection against the simulator
python .\demos\04_perception\09_camera_to_scene.py

# 2. read the real HSV values off a rendered frame and paste them into colors.py
python .\demos\04_perception\09_camera_to_scene.py --calibrate

# 3. the same with 20 rearrangements of the layout, which is the number for the board
python .\demos\04_perception\09_camera_to_scene.py --trials 20 --headless

# 4. the same pipeline on the table at home
python .\demos\04_perception\09_camera_to_scene.py --source webcam
python .\demos\04_perception\09_camera_to_scene.py --source folder out\frames

# 5. a command against a scene, without a microphone
python .\demos\04_perception\09_ground_command.py --scene out\scenes\trial000.json
python .\demos\04_perception\09_ground_command.py --scene out\scenes\trial000.json --text "coge la bola roja" --lang es
python .\demos\04_perception\09_ground_command.py --scene out\scenes\trial000.json --text "coge la bola roja" --answer "la de la izquierda" --lang es
```

`--source spec` is the only route that never starts Isaac Sim, so it is the one to use
when you only want to check the bands and the layout. Note that `--calibrate` **does**
start the simulator, because the whole point is to read the bands off a real frame.

`--camera top` (the default) looks from the **far** side of the table, back towards the
robot, tilted 67° over the working area. That is the whole trick: the arm occupies
y −0.68..−0.50, so any camera behind the robot sees the arm first and loses the
far-side objects behind the gripper. From y = −0.22 the objects are nearest the camera
and the arm is only ever background. `--camera side` is a lower, more human-facing view
that also stays clear of the arm.

`--camera front` sits behind the robot at y = −0.86 and is kept as the deliberate worst
case: it looks straight through the arm. Running it beside `--camera top` is how the
claim above gets checked rather than asserted.

`--detect` chooses which colours to segment. The default is `red,green,blue`, which is
`colors.DEFAULT_DETECT`; `white` and `black` are in `COLORS` because `vocabulary.yaml`
needs them to name a detected object, but both match the background and the robot's own
links here, so including them turns the whole frame into detections.

`09_ground_command.py --command` takes a command JSON inline, but a JSON blob has to
survive PowerShell's quoting, so `--command-file path.json` is usually the one to reach
for on Windows.

## The files

| File | Needs Isaac | Needs torch | Role |
| --- | --- | --- | --- |
| `colors.py` | no | no | the named HSV bands. **The file to edit at the hackathon**, in the same way `vocabulary.yaml` is for the command model |
| `scene.py` | no | no | `Object`, `Container`, `Scene`, the JSON schema, and the pinhole maths |
| `perceive.py` | no | no | image to scene, and the annotated overlay |
| `ground.py` | no | no | a command item to object ids, or a question back |
| `describe.py` | no | no | a scene to a sentence in Spanish, English or Basque |
| `detector.py` | no | yes | optional open-vocabulary detector, **off by default** |
| `scene_spec.py` | yes | yes | what is on the table, in one editable list |
| `09_camera_to_scene.py` | yes | yes | the simulator half, and the scoring |
| `09_ground_command.py` | no | yes | demo 03's command model plus a scene |

The split is deliberate. Everything that decides *what is on the table* imports neither
Isaac Sim nor torch, so it runs on the hackathon laptop even when the simulator is
broken, and so the same code path is exercised by the simulator, by a webcam and by a
folder of saved frames. That is what lets you tune on saved frames and be sure the
thing you tuned is the thing that runs.

## How it works

**Colour, not RGB.** The task's light is `DomeLightCfg(intensity=1000.0)`, so brightness
changes with facing and distance. Hue survives that; saturation is what separates a
painted object from the light wood of the table. Each colour is one HSV band, with the
saturation floor doing most of the work.

**Shape from the second moments, not from the bounding box.** A square rotated inside its
own bounding box has an extent anywhere between 0.5 and 1.0, so the extent test cannot
tell a cube from a ball. The second moments of a disc are isotropic and equal `r²/4` in
any orientation, and the fill ratio (area over the area of the ellipse with the same
moments) is 1.000 for a disc and 0.954 for a square. The threshold sits at **0.985**,
which is where 160 measured blobs put a gap: cubes 0.9562–0.9780, balls 0.9985–1.0001.
It is not derived from the ideal shapes — a real thresholded mask is not an ideal square,
and that is why 0.975 (inside the cube range) mislabelled the largest blue cube.

**Position from one ray and the table plane.** Every object rests on the table, so a
single pixel ray intersected with a known plane height is enough — and it is the method
that also works on a webcam, which has no depth sensor. Where the simulator does give a
depth image, the same point is computed a second way and stored beside the first in
`pos_m_depth`, because the gap between the two is a measurement worth having.

Camera geometry: a pixel plus an orthogonal depth `d` gives the camera-frame point
`K⁻¹[u,v,1]·d`, and the world point is `R(quat)·p_cam + pos`. The axis signs are **not**
Isaac Lab's documented ones — the camera looks along its own `+x`, and **both** lateral
axes are negative (`u = cx - fx·y/x`, `v = cy - fy·z/x`). Those signs were measured, and
they are worth reading about in `scene.CameraModel.to_camera`, because the failure mode is
silent: with the opposite signs every object still landed comfortably inside the frame,
because the layout is compact and centred. Only comparing the projection against the
*measured blob centroids* of the same frame showed the 130 px average error, with
`640 - u, 480 - v` landing exactly on each blob (1.9 px median over six objects). The
same bug was in the webcam path, where it put objects metres off the table.

`--check-convention` draws the truth positions on the overlay so this can be checked by
eye. Looking at whether they land in the frame is not enough.

**Left is the robot's left.** Read off `robot.data.root_quat_w` at run time and written
into every scene JSON as `frame.left_axis`, so demos 10 and 12 never re-derive it and it
does not flip when the camera changes. It evaluates to -x with the shipped task.

## The scene JSON

The contract between this demo and the ones after it.

```json
{"schema": "leisaac-demo/perception/scene/1",
 "source": "isaac:top:trial0",
"frame": {"space": "world", "left_axis": [1, 0], "front_axis": [0, -1],
            "robot_xy": [0.35, -0.64]},
  "width": 640, "height": 480,
  "camera": {"pos_w": [0.375, -0.22, 0.55], "quat_wxyz": [-0.567, -0.406, -0.378, 0.609],
             "matrix": [[...]], "width": 640, "height": 480},
  "plane_z": 0.0409, "ground_px_per_m": 746.5,
  "objects": [{"id": 2, "kind": "cube", "color": "blue", "confidence": 0.398,
               "bbox_px": [366, 166, 27, 29], "centroid_px": [379.1, 180.1],
               "pos_m": [0.267, -0.520, 0.0409], "pos_m_depth": [0.273, -0.505, 0.067],
               "area_px": 684, "mean_hsv": [111, 179, 197], "source": "segmentation"}],
  "containers": [{"id": 0, "kind": "box", "color": "blue", "pos_m": [0.269, -0.372, 0.0409],
                  "inner_rect_m": [0.233, -0.410, 0.072, 0.076],
                  "area_px": 2577, "...": "..."}],
  "relations": [{"subject": 3, "predicate": "left_of", "object": 1}],
  "notes": ["looking for: red, green, blue. ..."]}
```

`id` is the currency demo 10 spends: it is assigned biggest-first then left-to-right, so
it is stable while nothing moves. `space` is `world` (metres) when a camera pose and a
table height are known and `image` (normalised 0..1) when only a picture is; left and
right work in both, and `pos_m` is simply absent in the second case rather than faked.

## Grounding

| status | meaning | what the robot does |
| --- | --- | --- |
| `ok` | exactly one object answers the description | acts |
| `ok_filtered` | several matched and a position word picked one | acts |
| `ok_count` | several matched and a number picked how many | acts |
| `ambiguous` | several matched and nothing narrowed them | asks |
| `not_found` | nothing matched | says what it *can* see, so the person can correct it |

A position word never removes a candidate, it only orders them, so the question is not
"did the word leave one candidate" but "does the word actually separate them". That is
the difference between *"the red ball on the left"* picking one of two reds side by
side, and the same words meaning nothing when the two reds are one behind the other. In
that case it asks, and the question names the two ends of whichever axis does separate
them:

```text
coge la bola roja                   -> Hay dos bolas rojas. ¿Cuál, la de la izquierda o la de la derecha?
coge la bola roja de la izquierda   -> Vale, cojo #3.
```

The loop closes with no new training, because demo 03 already has `answer_value` trained
on *"La de la izquierda"*. Two turns of a model that exists.

A position word that cannot separate the candidates is not treated as one that did:
`ground()` asks whether the word *separates* them, not whether it left one survivor. So
if two reds sit one behind the other and the person says "on the left", the robot says
so rather than picking arbitrarily.

`describe.py` is what replaces the fixed reply for a `question` intent. It does real
agreement per language rather than translating one list: Spanish agrees in gender and
number (*dos cubos azules*, not *dos cubos azul*), English puts the adjective first (*two
blue cubes*), and Basque puts the numeral first and then uses the bare form of everything
after it, because a numeral already means plural (*bi kubo urdin*, never *bi kubo
urdinak*; singular is *kubo urdin bat*).

## Known gaps

- **Kind is the weakest step, and it is measured rather than hidden.** Over 20 trials it
  scores 117/120, the three misses all being `cube_blue` read as a ball. A disc and a
  square differ by only 4.5% in fill ratio and a 3 cm cube is about 30 px across here, so
  this is noisy by construction; `BALL_FILL` in `perceive.py` is the constant to move, and
  the measured ranges above it say how much room there is. When a shape is neither, the
  kind is reported as `"any"`, a legal answer that lets grounding fall back on colour and
  position.
- **Yellow against the wood table.** The table is `light_wood_planks_0.png` and measures
  H 18, S 82 — inside the yellow hue band. Separation therefore rests entirely on
  saturation (yellow needs 130, the wood peaks at 95), which is thin, and nothing in
  `scene_spec.py` is yellow. It is off by default; `--detect yellow` works but should be
  measured before it is believed.
- **The two zones are no longer told apart by shape.** They were 9 x 7 cm and
  8.5 x 8.5 cm, distinguished by aspect; they are now both 6 cm, because a 9 cm tray
  cannot fit in a third row inside a 33 m arm. `perceive.BOX_ASPECT` is no longer
  load-bearing for this scene. If demo 10 needs to tell them apart, colour and height
  (1.2 cm against 3 cm) are what it has.
- **`scene.relations` is empty for this scene.** It only reports pairs sharing both kind
  and colour, which is what a person needs to disambiguate, and every object in
  `DEFAULT_SCENE` is unique. Ambiguity is exercised by the synthetic scene in the smoke
  tests and by `--source` layouts that duplicate an object, not by the default one.
- **Two same-coloured objects touching merge into one blob.** A 3 px opening is applied,
  and `clearance()` proves the layout never brings a same-colour pair within 10 px of
  each other, so the opening cannot break a real pair apart. Whether it is enough when
  something does touch is something `--trials` should be asked directly.
- **Containers are coloured zones, not hollow boxes.** Demo 09 only needs to see where one
  is, and demo 10's placement is scripted the way demo 05 writes the cube's pose. Making
  one hollow is a demo 10 change. Their footprints are now both 6 cm — see the note above
  on why the aspect-ratio distinction had to go — so perception calls both of them `box`
  and `scene_spec` declares both as `box` to match.
- **The open-vocabulary detector is off by default and would not help in the simulation.**
  The extra cubes and balls are flat-shaded primitives, and a detector asked for a banana
  should not be expected to find one. Score it on a webcam pointed at real objects, or on
  real USD assets. YOLO-World would mean a new `ultralytics` install into an environment
  whose versions are pinned deliberately, so OWLv2 was chosen because `transformers`
  already has it.
- **A webcam's positions are an approximation.** The pose assumes the camera is directly
  above the table looking straight down, and the focal length comes from `--fov-h`.
  `--no-intrinsics` turns it off and leaves the scene in the image frame, where left and
  right still work and positions are honestly unknown.
- **There are no committed tests.** Correctness here is measured by
  `results/segmentation_report.md`, the same way demos 01–03 measure it. That is a
  deliberate choice, not an oversight. The synthetic checks that do exist
  (a painted scene through the real `perceive`, a static import/definition audit) live
  outside the repo in a temp directory, which is a real weakness: they are the only
  thing that catches a fixture disagreeing with a constant, and they are not versioned.
  Promoting them into `demos/04_perception/tests/` is the obvious next step, and it
  would have caught the two wrong-camera bugs on its own.
- **`--source image` and the webcam path were never checked against the simulator.** They
  were only exercised on painted fixtures. Running one rendered frame through both paths
  and comparing the positions is what exposed the transposed rotation, and it is cheap.

## What the next demos get

- **Demo 10 (planner and executor)** consumes `objects[].pos_m` and `containers[].inner_rect_m`
  and reuses demo 05's waypoints, driven by camera positions instead of hardcoded ones.
  `ground_command()` returns `executable` per item, which is exactly the "one of the two
  objects is missing" case the executor has to survive.
- **Demo 11 (imitation learning)** records from the same scene, so the dataset and the
  perception agree about what object #3 is.
- **Demo 12 (show demo and dashboard)** draws the annotated overlay and prints the
  grounding transcript, and puts the `--trials` totals on the board.

## References

- [Level 2 vision: un detector identifica objetos](../docs/guia-so-101-autonomous-agent.md)
- [`isaaclab.utils.math.unproject_depth`](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/isaaclab/utils/math.html)
- [LeIsaac tasks](https://lightwheelai.github.io/leisaac/docs/)
- [OWLv2, open-vocabulary detection](https://huggingface.co/docs/transformers/model_doc/owlv2)
