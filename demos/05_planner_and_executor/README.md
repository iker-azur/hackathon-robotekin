# Demo 10 — from a camera frame to a moving arm

Type a sentence. The robot looks at the table, works out what you meant, moves, and then
checks whether it actually did what you asked.

```
python 10_command_to_motion.py                      # window, live camera, type at the prompt
python 10_command_to_motion.py --headless           # no window
python 10_command_to_motion.py --rounds 5 --disturb 0.5
python 10_command_to_motion.py --text "pon el cubo rojo en la caja roja"
```

Everything here is measured. Where a number is quoted below, a tool in `tools/` reproduces
it, and where something does not work, the README says so and says why.

## What it does

| stage | module | what it is responsible for |
|---|---|---|
| see | `perceive.py`, `colors.py`, `detector.py` | one camera frame → objects, kinds, colours, positions |
| understand | `command_model_v2/` | a sentence in Spanish, English or Basque → a command |
| resolve | `ground.py` | that command → *this* table's objects, asking when it cannot tell |
| plan | `planner.py` | those objects → arm poses |
| move | `executor.py` | poses → motor commands, with grasp fallback |
| check | `10_command_to_motion.py` | did it land where it was asked to? if not, try again |
| disturb | `dynamics.py` | move the layout, shove something mid-task |

## Running it

| flag | default | |
|---|---|---|
| `--headless` | off | no window; needed for any scripted run |
| `--rounds N` | 1 | re-randomise the layout between rounds |
| `--seed N` | 0 | the layout is a pure function of the seed |
| `--spread M` | 0.018 | how far a randomised layout may move things, in metres |
| `--disturb P` | 0 | probability of shoving the scene between planning and executing |
| `--text "..."` | — | run this instead of waiting; repeatable |
| `--detect red green blue` | red green blue | colours to look for |
| `--lang es` | es | language for replies |
| `--offset DX DY` | 0 0 | systematic grasp correction, in metres |

Keys, in the window: `q`/`escape` quit, `a` masks the arm out of perception, `s` saves a
frame, `r` re-looks, `n` next round.

Typed words that the loop answers itself: `q`/`salir`, `n`/`siguiente`, `r`/`mirar`,
`s`/`guardar`.

## Measured

**The arm moves as well as demo 05.** `tools/check_executor.py --headless` reproduces
demo 05's reference move — the shipped cube 10 cm in x — to **0 mm** worst-case gripper
tracking error, landing 10.2 cm from a 10 cm target, 0.2 cm error.

**Perception is good enough to aim at.** Localises a tray to **11 mm** and an object to
**9–11 mm**; 80 of 80 cubes and balls classified correctly against the hollow trays.

**Randomised layouts do not hide anything.** `tools/check_dynamics.py --rounds 16 --spread
0.03`: 4 of 4 objects and 2 of 2 containers found in **every** round, worst same-colour gap
11.3 mm. Over 200 seeds with no simulator, the mean largest relocation is 18.0 mm and the
worst same-colour gap across all of them is 13.7 mm.

**Mid-task disturbances really move things.** 15.3 to 133.9 mm over those 16 rounds, against
a 5 mm minimum. The plan is recomputed afterwards and the demo reports when it replanned.

**The command model.** 136 hand-written sentences in three languages: **84.6%** exact whole
command, 90.4% on items, 95.6% on the destination, 100% on intent. On demo 03's own 98 legacy
sentences this scores **85.7%**, up from 84.7% before the destination-colour fix below, and
level with the 85.7% demo 03's own committed report records for its model. Stop is caught
100%, and **0%** of the 54 sentences that must not move the robot cause any motion at all.

**Grounding and planning**: 294 checks over three recorded scenes, `tests/test_planner.py`.

## How it is put together, and why

**The layout moves every round.** If the same command works across twenty randomised
layouts, nothing downstream can be reading a hardcoded coordinate. `dynamics.randomize`
rejection-samples a layout and rejects anything that overlaps, brings two same-coloured
things within 8 mm, or leaves the reach envelope.

**Placements are verified, not assumed.** Perception's localisation error is larger than the
margin a 4 cm ball has inside a 5.6 cm cavity, so a release can miss. After the arm finishes,
the demo checks the simulator's own positions and, if an object is not in the tray it was
sent to, replans that one object and tries once more. The summary counts what was verified,
what needed a retry and what never verified.

**Grasp honesty.** A grasp is judged by how much of the gripper's own travel the object
followed, measured in 3-D at the end of the lift. A 4 cm ball closed on by jaws that shut to
0.15 rad follows 46% of the lift and is reported as a kinematic carry rather than a grasp.
An object that moved *further* than the gripper was knocked out of reach, and is not carried
at all — carrying it would teleport a ghost back onto the table.

**Parking comes before populating, and before every look.** A plan ends with the arm over the
table, and a camera looking down from the far side sees the arm first; looking from there
found 0 of 4 objects. Parking sweeps the forearm across the table and moves the balls, so the
demo looks *after* parking, plans for where things actually came to rest, and puts back
anything the sweep moved more than 25 mm — saying so out loud.

## Known limitations

These are real, reproduced by the tools, and not worked around silently.

**The arm's parking sweep disturbs the table.** The forearm sits in the object row at the
retreat pose, so driving back to park shoves the balls: 16.6 mm in the gentlest measured
case, and up to 168 mm coming back from a release, which is off the table entirely. It is not
a gripper-clearance problem — drift was identical at four different hover heights — and a
*slower* park is much worse (1200 steps drifted 1893 mm) because the forearm then leans on
the balls the whole way. `tools/probe_park_speed.py` has the table. The structural fix is to
put the trays between the robot and the loose objects so the return path never crosses them;
that is a scene-layout change and would need every calibration re-measured.

**A tray holds one object.** The cavity is 5.6 cm square, so a 3 cm cube or a 4 cm ball fits
but not both. A command that asks for a second object in the same tray is refused by the
verification step, but reaching into a tray that already holds something can squeeze the
object hard enough for the solver to eject it metres away. This shows up as an object
reported as "knocked out of reach".

**A 3 cm cube into the 5.6 cm cavity has about 1 mm of margin.** That is arithmetic, not
pessimism: the cavity half-width less the cube half-width less the 12 mm localisation error.
It lands in some runs and misses in others. This is the case the verify-and-retry loop exists
for, and it will fire often enough to be worth revisiting.

**Orange and pink are not detectable.** Measured on rendered frames: orange sits 5 hue units
from yellow and pink 11 from red with overlapping saturation. `--detect yellow purple` widens
the palette to five colours.

**Every grasp in this build is kinematic.** Friction grasping a 3 cm cube is unreliable in
this simulation, which demo 05 already documented. The fallback is detected and reported per
grasp rather than assumed.

## Layout of the code

```
10_command_to_motion.py   the demo loop: rounds, commands, verification, recovery
dynamics.py               randomised layouts and mid-task shoves
executor.py               the only module that needs Isaac to move anything
planner.py                command + scene -> arm poses, no Isaac
ground.py                 command -> this scene's objects, and the questions to ask
perceive.py               frame -> Scene
scene_spec.py             what is on the table, and the geometry it was fitted to
hollow.py                 the trays: one kinematic body, five collision shapes
isaac_io.py               simulator setup, camera capture, placing objects
viewer.py                 window, off-thread perception, frame writer
colors.py                 the measured colour bands
tools/                    every measurement quoted above
results/                  the reports those tools write
```

## Reproducing the numbers

```
python tools/check_executor.py --headless          # motion parity, then a camera-derived pick
python tools/check_dynamics.py --headless --rounds 16 --spread 0.03
python tools/probe_park_speed.py --headless        # the parking sweep table
python tools/probe_shove.py --headless             # how a shove affects each kind
python tests/test_planner.py                       # 294 grounding and planning checks
python command_model_v2/evaluate.py                # the 136-sentence evaluation
python tools/check_trays.py --headless             # hollow tray containment
python tools/calibrate_colours.py --headless       # the nine colour bands
python tools/measure_kinds.py --headless           # cube/ball thresholds
```

Isaac Sim prints a `_generic_model_output` DLL warning and a few `omni.fabric.plugin` messages
on every run, including successful ones. They are not the run failing.