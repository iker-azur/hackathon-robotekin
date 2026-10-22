# The working area, and why it did not get bigger

Asked for: *"i want the working area to be bigger"*.

Answer: it cannot be, and the reason is worth having in writing, because the area that shipped
was already larger than the arm can cover.

## The area before and after

| | x | y | area |
|---|---|---|---|
| before | 16 cm | 21 cm | 336 cm² |
| after | **25 cm** | 12 cm | 300 cm² |

It is 56% wider and 43% shallower. What changed is not the size of the number — it is that every
point inside the new area was measured to work.

## Why not bigger

The arm's reach is `MAX_REACH_M = 0.33 m` from a base at (0.35, -0.64). The largest rectangle
that fits inside a disc of radius *r* has area 2*r*², which for *r* = 0.33 is **218 cm²**.

The old working area was **336 cm²**. It was larger than the largest rectangle inscribable in the
region the arm can reach, so parts of it could never have been worked in.

That was not theoretical. `tools/probe_grasp.py`, which runs the real pick cycle at each point of
a grid, found the jaws closing on nothing at x = 0.56 and along the y = -0.34 row — both comfortably
inside the declared area:

```
        -0.62 -0.58 -0.54 -0.50 -0.46 -0.42 -0.38 -0.34
  x=+0.10 .  .  .  .  .  .  .  .  .  .  .
  x=+0.15 .  .  .  .  .  .  .  .  .  .  .
  x=+0.20 .  .  .  .  .  .  .  .  .  .  .
  x=+0.25 .  .  .  .  .  .  .  .  .  .  .
  x=+0.30 .  .  .  o  o  o  o  .  .  .  .
  x=+0.35 .  .  o  o  o  .  .  .  .  .  .
  x=+0.40 .  .  o  o  o  o  o  .  .  .  .
  x=+0.45 o  o  o  o  o  o  o  o  o  .  .
  x=+0.50 o  o  o  o  o  o  o  o  .  .  .
  x=+0.55 o  o  o  o  o  .  .  .  .  .  .
  x=+0.60 o  o  o  o  .  .  .  .  .  .  .
```

`MAX_REACH_M` arithmetic predicts a 66 cm span at y = -0.62 and 22 cm at y = -0.34. The arm
manages 15 cm and 6 cm. The limit is the arm's joint travel and its self-collision, and it has to
be measured; it cannot be computed from a radius.

## What the new area is fitted to

`tools/probe_grasp.py`, real pick cycle, both a cube and a ball, table cleared:

```
        +0.10 +0.20 +0.29 +0.39 +0.48 +0.58
y=-0.54    K     K     x     K     K     x
y=-0.47    K     x     K     K     K     K
y=-0.40    K     x     K     K     x     K
```

Every point lifts in x 0.29–0.48, y −0.47…−0.40. The two neighbouring corners do not — (0.29, −0.54)
and (0.48, −0.40) fail for both kinds — so they are left outside the area.

`WORK_X = (0.27, 0.52)`, `WORK_Y = (-0.54, -0.42)`. The bounds are a few millimetres wider than the
layout needs, which is not tidiness: an item placed exactly on the boundary fails `clearance` on a
floating-point comparison and is reported as outside its own area.

### The far row is the camera's limit, not the arm's

Perception localises to 10.3–12.5 mm inside the area and to **53 mm** along y = −0.34. At 67°
elevation and that distance the objects approach the frame edge, where confidence falls and the
plane-ray estimate degrades. Enlarging the area towards the camera would enlarge the region the
robot can see badly, which is the opposite of what was asked for.

## The layout inside it

Four loose objects in the near row, two receptacles in the far row. Three constraints decide it,
and the second costs the most:

1. **Receptacles far, objects near.** The camera is at y = −0.22, so the receptacles must not stand
   between the camera and the objects. A first attempt put the trays in the near row and recorded
   2 of 4 objects per frame.
2. **Same-coloured pairs at least 70 mm apart in x.** Two blobs of one colour that touch come back
   from segmentation as one. Measured blob half-widths put the threshold at about 63 mm — a 24 px
   tray plus a 12 px cube. An earlier version of this same list put the red ball 10 mm from the red
   box and recorded a 3890 px red blob where a box alone measures about 2600 px, with the ball
   simply gone.
3. **Loose objects 70 mm apart**, a 40 mm gap between two 3 cm objects: 50 px of daylight at
   747 px/m against a 3 px morphological opening.

```
near row   cube(red) 0.290   ball_red 0.360   cube_blue 0.430   ball_green 0.500
far row    tray_blue 0.320                            box_red  0.440
```

Read the x column and the same-colour gaps are 150 mm, 110 mm and 80 mm.

## Result

All 20 recorded frames now see every object and every container, which the previous layout did not:

```
20/20 trials: 4 objects, 2 containers, position error 10.3-12.5 mm
```

`tools/measure_kinds.py` labels every blob against the simulator's own ground truth: **80/80
correct**.

## What would actually make it bigger

Not a constant in this file. The robot would need to be further back from the table, or a longer
arm. The working area is currently the largest rectangle that fits inside what the arm and the
camera can do, which is a different thing from the largest number that looks impressive.