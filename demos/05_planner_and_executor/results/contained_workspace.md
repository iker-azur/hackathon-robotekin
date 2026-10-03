# Variant 1: a walled working area

`--variant contained` walls the working area in so a disturbed object cannot leave it.
`--variant open` is the original pen-free demo, kept deliberately as the control.

## Why it exists

The arm knocks whatever it reaches for. A ball ends up 57–62 mm away on the way in and then
rolls 100–370 mm more — measured leaving the table entirely at x = −0.22 against a table edge
at x = 0.003, and once at x = −2.1. Once outside the working area two things go wrong at once:
the object cannot be reached, and perception stops calling a ball a ball (the shape thresholds
in `colors.py` were fitted near the centre of the frame).

## How it is built

`pen.py`. One kinematic rigid body whose four collision shapes are walls and which has no floor
— the table is the floor. That is `hollow.py`'s tray mechanism with the floor part left off, so
`tray_parts` now emits four parts instead of five when `floor` is 0. `check_trays.py` still
asserts five parts for the trays, which are unchanged.

Kinematic for the same reason the trays are: a wall the arm can knock over is not a wall.

The inner face sits exactly on the working-area boundary, so containment is precise — an object
is either inside the area where the robot can reach it, or stopped at the edge.

## Measured, with and without

`tools/probe_pen.py`, 3 rounds × 4 shoves per round, shoves at 0.45 m/s — about 60% faster
than the demo's own, to test the kerb rather than the demo.

| | escapes | worst overhang | verdict |
|---|---|---|---|
| `--no-pen`, 0.45 m/s | **5 of 12** | **447 mm** | objects leave the area |
| `--variant contained`, 0.45 m/s | **0 of 12** | **0.0 mm** | nothing leaves |
| `--variant contained`, 0.20 m/s | 0 of 12 | 0.0 mm | nothing leaves |

The kerb contains completely, and the result is not marginal: nothing escaped even against
shoves well faster than the demo produces, and the worst overhang is 0.0 mm rather than
"within tolerance".

### Kerb height does not matter for containment

Swept, and all three hold everything in:

| kerb | escapes | worst overhang |
|---|---|---|
| 1.5 cm | 0 of 12 | 0.0 mm |
| 2.0 cm | 0 of 12 | 0.0 mm |
| 3.0 cm | 0 of 12 | 0.0 mm |

2.5 cm is kept, on the reasoning that a 4 cm ball's centre is 2 cm above the table so a kerb
reaching past the centre catches it rather than being climbed — but the measurement says even
1.5 cm does, and a lower kerb occludes less.

## What the pen does not fix

**Balls are still misread, and it is not the pen's doing.** Between 3 and 5 of 6 balls per run
stopped being called balls, at *every* kerb height and *with no kerb at all*. Identical counts
with and without the wall is the measurement that says so. This is the shape classifier
failing away from the centre of the frame, and it was already known and documented before the
pen existed.

So the pen fixes containment, which is what it was built for, and leaves the classifier alone.
The two are separable, and separating them is what makes the pen worth having: a contained
object is a re-plannable object even when it has rolled somewhere awkward, because there is
always somewhere inside the area to aim from.

## The cost, and it is not yet paid

The arm's base is at y = −0.64, beyond the working area's near edge at y = −0.53, so **every
move crosses the near kerb**. A 2.5 cm wall between the arm and everything it reaches is a real
risk of the forearm clipping it on the way in.

`probe_pen.py` shoves objects but does not drive the arm, so it does not measure this. What it
has shown is that a cube pick still completes with the pen present — 1/1 verified, 0 mm
tracking — but that is one pick, not a sweep, and it is not evidence the forearm never clips.
`--variant contained` should be run against `tools/check_executor.py --regrasp 0` before being
trusted, and that is the obvious next measurement.