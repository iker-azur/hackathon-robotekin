# Perception results

`python .\demos\04_perception\09_camera_to_scene.py --trials 1` on the top camera of `LeIsaac-SO101-LiftCube-v0`.

- ground resolution on the table: **747 px/m**
- table plane used: z = 0.0409 m
- looking for: red, green, blue. not by default: yellow, white, black (white and black match the background and the robot's own links; yellow shares its hue band with the wood table). Add one with --detect.
- frames written to `out/frames/`, scenes to `out/scenes/`, overlays to `out/annotated/`

| trial | objects | found | right colour | right kind | plane err (mm) | depth err (mm) | missed | wrong colour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6 | 6 | 6 | 5 | 10 | 6 | - | - |

## Totals

- objects in the scene: **6**
- objects detected: **6/6** (100%)
- correct colour: **6/6** (100%)
- correct kind: **5/6** (83%)
- plane position error: median **10 mm**, p95 **12 mm**, worst **12 mm**
- depth position error: median **6 mm**, p95 **7 mm**, worst **7 mm**
- **the difference between those two columns is the bias of the plane method**: 5 mm
- blob centroid against the projected object centre: median **1 px**
- wrong kind: **1** (cube_blue read as ball x1). Shape is the one noisy column: see `perceive.BALL_FILL` for the measured ranges.

## How to read these numbers

- *Found* means a detection's position on the table is within 6 cm of where the object really is.
- **The plane column is biased on purpose and the depth column is not.** The
  plane method starts from a pixel on the object's *top face* and walks that ray
  down to the table, so it reports where the object appears to touch the table:
  an error of height/tan(elevation), about 47 mm for a 3 cm cube at 67 degrees.
  The depth method reads the object's actual surface, so it is unbiased. Neither
  is wrong; they answer different questions, and only the simulator has depth.
- *Right kind* is the honest one to watch. It is the fill ratio of the mask, and
  a 3 cm cube is only about 30 px across, so it is the noisiest column here.
  Measured over 6 blobs: cubes fill 0.9562-0.9780, balls 0.9985-1.0001,
  so the threshold in `perceive.BALL_FILL` (0.985) sits in a gap
  the data agrees on rather than near a boundary.
- The bands in `colors.py` are measured, not derived from the albedos. If the
  scene, the lighting or the camera changes, re-run `--calibrate` before
  trusting the colour column.
