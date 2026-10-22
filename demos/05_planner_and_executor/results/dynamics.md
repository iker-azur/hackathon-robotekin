# Dynamics: what moves, and what it costs

Measured with `tools/check_dynamics.py`, 16 rounds at a 30 mm layout spread, plus
`tools/probe_shove.py` for the disturbance physics.

## Randomised layouts

`dynamics.randomize` rejection-samples a layout from `scene_spec.DEFAULT_SCENE` and rejects
any that overlaps, puts two same-coloured things within 8 mm, leaves the reach envelope, or
leaves the working area.

| | |
|---|---|
| rounds | 16, seeds 0..15, spread 30 mm |
| largest single relocation | 38.6 mm |
| worst same-colour gap seen | 11.3 mm |
| objects found by perception | 4 of 4 in every round |
| containers found | 2 of 2 in every round |

Over 200 seeds checked without the simulator, the mean largest relocation is 18.0 mm and the
worst same-colour gap across *all* of them is 13.7 mm, so acceptance is not marginal. The
randomiser is deterministic in its seed: seed 7 is the same layout on every machine.

Detection never dropped an object. That is the number that matters most here - a layout the
camera cannot read is not a randomised layout, it is a broken round - and it held across
every seed at both 18 mm and 30 mm spread.

## Mid-task disturbance

`dynamics.disturb` picks one object that the arm is not holding and moves it, so that a plan
already computed from the previous camera frame is stale before it is executed. That is the
case the recovery loop exists for.

A shove is only a disturbance for a sphere. From `tools/probe_shove.py`, a 20 cm/s shove:

| object | velocity write | after a 1 mm nudge | with a tip | pose move of 20 mm |
|---|---|---|---|---|
| `ball_red` | 69.4 mm | 70.2 mm | - | 2.5 mm |
| `ball_green` | 141.8 mm | 142.5 mm | - | 1.8 mm |
| `cube_blue` | 2.7 mm | 2.6 mm | 1.0 mm | 18.5 mm |
| `cube` (shipped) | 1.4 mm | 1.5 mm | 0.3 mm | 17.9 mm |

A sphere rolls and goes somewhere nobody predicted, which is the interesting case. A cube on a
flat face slides two millimetres and stops, and cannot be tipped by pushing at its base - the
"with a tip" column is the same failure with a spin added, so the spin was removed rather
than kept as a decoration.

So `disturb` shoves spheres and displaces everything else directly, and says which it did in
`Disturbance.method`. The demo still gets a mid-task disturbance for every object; it just
does not claim the two are the same physics.

Disturbance size over the 16 rounds: 15.3 to 133.9 mm, against a required minimum of 5 mm.
20 mm was chosen because it invalidates a planned pose without approaching the edge of the
table, and because it is larger than the worst same-colour gap so the moved object cannot be
confused for a neighbour of its own colour.

## What is not disturbed

The hollow trays. They are kinematic rigid bodies - `hollow.py` builds them that way so that
an object dropped inside stays inside - and a kinematic body cannot be moved by a velocity or
a shove; writing one moves it 0.0 mm. An early version of `disturb` took a list of prim names
and cheerfully shoved trays, which is how it first looked broken on 5 rounds out of 6. It now
takes the item list and skips containers itself.