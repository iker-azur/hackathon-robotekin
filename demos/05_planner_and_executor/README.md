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
| move | `executor.py` | poses → motor commands, re-aiming from the camera as it goes |
| check | `10_command_to_motion.py` | did it land where it was asked to? if not, try again |
| disturb | `dynamics.py` | move the layout, shove something mid-task |

## The two variants

One script, `--variant` picks the scene.

| variant | what it is |
|---|---|
| `contained` | the working area walled in by a 2.5 cm kerb, so a knocked object stays reachable |
| `out_of_reach` | two objects placed beyond the arm's reach; the robot refuses them and says why |
| `open` | the original, no kerb — the control that shows what the wall is for |

```
python 10_command_to_motion.py --variant contained
python 10_command_to_motion.py --variant out_of_reach
python 10_command_to_motion.py --variant open
```

`out_of_reach` refuses in your language, and the reason is a number rather than an excuse:

```
> pon el cubo azul en la caja roja
  Veo un cubo azul, pero no puedo llegar: esta a 35 cm de la base y el brazo llega a 33. Muévelo más cerca.
```

## Running it

| flag | default | |
|---|---|---|
| `--variant` | `contained` | `contained`, `open` or `out_of_reach` |
| `--headless` | off | no window; needed for any scripted run |
| `--voice` | off | listen to each command instead of typing it |
| `--voice-model` | small | `tiny`/`base`/`small`/`medium`; all already cached locally |
| `--voice-language` | detect | force `es`, `en` or `eu` instead of detecting |
| `--raw` | off | show the unannotated camera feed, not the drawn detections |
| `--rounds N` | 1 | re-randomise the layout between rounds |
| `--seed N` | 0 | the layout is a pure function of the seed |
| `--spread M` | 0.018 | how far a randomised layout may move things, in metres |
| `--disturb P` | 0 | probability of shoving the scene between planning and executing |
| `--regrasp N` | 2 | re-perceive and replan when the arm knocks the target out of reach |
| `--text "..."` | — | run this instead of waiting; repeatable |
| `--detect red green blue` | red green blue | colours to look for |
| `--lang es` | es | language for replies |
| `--offset DX DY` | 0 0 | systematic grasp correction, in metres |

Keys, in the window: `q`/`escape` quit, `a` masks the arm out of perception, `s` saves a
frame, `r` re-looks, `n` next round. Masking is switched on by itself while the arm moves —
those are the frames that need it — and `a` is there to turn it off and compare.

Typed words that the loop answers itself: `q`/`salir`, `n`/`siguiente`, `r`/`mirar`,
`s`/`guardar`.

## Seeing what the camera sees

The window shows the **annotated** frame: every blob outlined in its colour, labelled with
what was decided about it, with a crosshair at the position it was measured at, plus the
scale (`746 px/m on the table`). That is deliberately not the raw feed — the raw feed looks
fine either way, which is the problem. Looking at the annotated frame is how you tell whether
the robot saw the green ball as a *ball* at a given position or as an unclassified blob, and
that is the question worth answering by eye.

```
python 10_command_to_motion.py                          # annotated, with a window
python 10_command_to_motion.py --raw                    # the plain camera feed
python 10_command_to_motion.py --save-every 1           # write every frame to out/*.png
```

`--save-every 1` writes the same annotated frames to `out/` and works with `--headless`, so a
run can be inspected afterwards. In the window, `s` saves the current frame.

To check which camera is being used, `--camera side` or `--camera front` are also wired up;
`front` is the deliberate worst case, looking through the arm.

## Saying it instead of typing it

`--voice` listens for one utterance, transcribes it, and runs it exactly as if it had been
typed. Typing still works as a fallback, because a microphone is not always the right input.

```
python 10_command_to_motion.py --voice
python 10_command_to_motion.py --voice --voice-language es
python 10_command_to_motion.py --voice --voice-model base
```

Both packages it needs are already installed (`faster-whisper` 1.2.1, `sounddevice` 0.5.6) and
`Systran/faster-whisper-small` is already in the Hugging Face cache, so **nothing is
downloaded on the first run** and it works offline.

Recording starts when you speak and stops 1.2 s after you stop, or after 10 s, whichever comes
first; nothing heard means it falls through to the keyboard. What it heard is printed before
it is run, so a mis-transcription is visible rather than mysterious — Whisper returns
lower-case text with no punctuation, and demo 03's command model was trained with a
`train_asr.jsonl` set of the same sentences exactly as speech recognition returns them, so
that shape of input is one it has already seen. All three languages are in Whisper's language
list, Basque included; `--voice-language` forces one if auto-detection picks wrong.

The model is loaded before the simulator, for the same reason the command model is: two heavy
native libraries loading after Isaac has booted crashes the process.

## Measured

**The arm moves as well as demo 05.** `tools/check_executor.py --headless` reproduces
demo 05's reference move — the shipped cube 10 cm in x — to **0 mm** worst-case gripper
tracking error, landing 10.1 cm from a 10 cm target.

**And it now picks a ball from the camera and puts it in a tray.** The same tool, second test:
the red ball is found to within **7.0 mm** of ground truth, carried, and lands **5.7 mm** from the
tray centre — inside the 48 mm cavity. Before the work in `results/ball_handling.md` the same check
reported it landing **118 mm** from the centre, outside the cavity, and failed.

**Perception is good enough to aim at.** 20 of 20 recorded frames see **4 of 4 objects and 2 of 2
containers**, position error **10.3–12.5 mm**; **80 of 80** cubes and balls classified correctly
against the hollow trays (`tools/measure_kinds.py`, `tools/record_fixtures.py`).

**And the aiming error is smaller than that, because the bias is subtracted.** Casting a ray from a
blob centroid to the table plane overshoots by a fixed amount — the centroid is on the object's top
face, 3 cm above the table. Measured over 120 blobs (`tools/measure_bias.py`, no simulator needed):

| blobs | dx | dy | before | after |
|---|---|---|---|---|
| balls | +2.1 mm | **−7.3 mm** | 10.3–12.5 mm | **4.8 mm** |
| cubes | +1.0 mm | **−6.5 mm** | 10.3–12.5 mm | **4.2 mm** |
| trays | +0.3 mm | **−10.3 mm** | — | 8.0 mm |

That matters because `GRASP_REACH_M` is 20 mm: a third of the budget was being spent on a one-signed
error before the arm had moved at all. Applied where the arm *aims* — `grasp_pose`, `place_pose`,
`locate` — and not inside `perceive`, so `pos_m` stays the raw measurement.

**A sort works.** `separa las piezas por colores` sorts **3 of 3**, one pick per pass.

**A ball only moves when something touches it.** `tools/probe_ball_physics.py --section roll`:
a shove of 10–45 cm/s moves it **33–45 mm** and it stops there. Before, the same shove sent it
**388–1898 mm** and it was still moving after 11.7 s — the balls were spawned with zero linear and
angular damping, so a sphere had no reason to stop rolling. Full sweep in
`results/ball_handling.md`.

**Nothing goes back to where it started.** Objects are placed once per round and left where they
come to rest. The park sweep is measured and reported, not undone: with the arm's closest link
**166 mm** from the balls, they move **1.7–3.6 mm**, which is resting-contact jitter.

**The gripper is 36.2 mm wide and cannot be opened further.** `tools/probe_gripper.py` reads
36.2 mm at every close command from 0.0 to 1.0 rad and every open command from 1.0 to the 1.745 rad
joint limit. That is why the balls are **3 cm** and not 4: a 4 cm ball does not fit, so the jaws
close on it and push it off the table. No amount of re-aiming fixes a geometric impossibility,
which is why three earlier attempts all failed.

**The working area is 25 × 12 cm, and it could not be larger.** The old 16 × 21 cm area was
336 cm², and the largest rectangle that fits inside a 0.33 m reach disc is 218 cm² — the old area
was already bigger than the arm could cover, and `tools/probe_grasp.py` found the jaws closing on
nothing inside it. The new one is 56% wider and every point in it was measured to lift a cube and
a ball. See `results/working_area.md`.

**The arm follows what it is reaching for.** `Executor.track` re-reads the camera every 20 steps
through the approach and the descent, and says how far it had to correct:

```
above  green: above #5  [aimed at (+0.509, -0.444), re-aimed 3x (largest 4 mm)]
down   green: down onto #5  [aimed at (+0.507, -0.445), re-aimed 2x (largest 5 mm)]
```

**Every grasp is kinematic, and says so.** Nothing is held by friction in this simulation — demo
05 says so of the cube it inherited — so the fallback carries the object and the report prints
`kinematic` rather than claiming a grip.

**The command model.** 136 hand-written sentences in three languages: **84.6%** exact whole
command, 90.4% on items, 95.6% on the destination, 100% on intent. On demo 03's own 98 legacy
sentences this scores **85.7%**, up from 84.7% before the destination-colour fix below, and
level with the 85.7% demo 03's own committed report records for its model. Stop is caught
100%, and **0%** of the 54 sentences that must not move the robot cause any motion at all.

**Grounding and planning**: **1468 checks over 20 recorded scenes**, `tests/test_planner.py`.

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
followed, measured in 3-D at the end of the lift. A ball nudged by the closing jaws follows only
46% of the lift — then drops out — and is reported as a kinematic carry rather than a grasp. An
object that moved *further* than the gripper was knocked out of reach, and is not carried at all:
carrying it would teleport a ghost back onto the table. And an object that did not move at all
*and* is no longer where the jaws were aimed is reported as **stale** rather than carried, because
that combination means the jaws closed beside it.

**The approach is closed-loop.** `above` and `down` do not drive once at a pose computed from a
frame that is already stale. `Executor.track` re-reads the camera every 20 steps and corrects the
aim, and reports how far it had to correct. The arm is masked out of its own view while it moves
— painted black, because the old mask halved brightness and every colour band is keyed on hue and
saturation, so it removed nothing — and the camera stops being consulted once the gripper is
within 10 cm of the table, where it would be standing on top of its own target.

**Objects stay where they stop.** An object is placed once per round and from then on it is
wherever it came to rest. `Robot.park` reports what the sweep moved and leaves it; `ensure_on_table`
repairs only an object that has left the table altogether, which is a broken state rather than a
movement.

## Known limitations

These are real, reproduced by the tools, and not worked around silently.

**Multi-object sorts run one object at a time, and a failure costs one object rather than all of
them.** `separa las piezas por colores` goes 3 of 3. A command naming several objects is cut into
one pick per pass — `handle` routes any multi-pick plan straight to that path — and each pass
re-looks, re-plans and re-aims, because a 32-step plan built from one camera frame is wrong by the
time it reaches its last step. A pick that fails is retried twice and then named in the summary
rather than retried forever. Getting there meant finding a 92 mm disagreement between two functions
that both move the arm; the full account is in `results/ball_handling.md`.

**A release can still leave an object balanced on the tray rim.** Verification reports it as a miss
and the retry may or may not fix it. The cavity is 56 mm for a 30 mm object, so the margin is real;
the loss is in the last millimetres of the descent.

**A tracked ball is often read as a `cube` while the gripper is near it.** The mask removes the
arm's own pixels, but the ball's remaining blob is no longer a clean disc, and cube-versus-ball is
the least reliable step in perception by its own documentation. Tracking falls back to matching on
colour alone, allowed only for a colour that is unique on the table — green has one object, red has
three, so red keeps asking.

**No grasp in this demo is held by friction.** Friction grasping is unreliable in this simulation
for the cube demo 05 already carried, so the kinematic fallback is the mechanism that completes a
pick, and the report prints `kinematic` rather than claiming a grip. Every number above that says
"picked" means the jaws arrived at the object and it was carried.

**The arm's parking sweep still brushes the table.** The forearm sits in the object row at the
retreat pose, so driving back to park passes over it. With the balls damped this is now **1.7–3.6
mm** — solver jitter, not a shove — but it is not zero, and it is reported above that threshold
rather than undone. A *slower* park is much worse (1200 steps drifted 1893 mm) because the forearm
leans on the balls the whole way, so the demo parks briskly.

**A ball knocked out of the working area cannot be re-planned for.** Outside it perception stops
calling a ball a ball, so there is nothing to aim at. The kerb is what prevents this and
`tools/probe_pen.py` measures it (0 of 12 shoves escaped at 0.45 m/s).

**A 3 cm cube into the 5.6 cm cavity has about 1 mm of margin.** That is arithmetic, not
pessimism: the cavity half-width less the cube half-width less the 12 mm localisation error.
It lands in some runs and misses in others. This is the case the verify-and-retry loop exists
for, and it will fire often enough to be worth revisiting.

**Orange and pink are not detectable.** Measured on rendered frames: orange sits 5 hue units
from yellow and pink 11 from red with overlapping saturation. `--detect yellow purple` widens
the palette to five colours.

## Demo 11 — the same demo, on physics you could build, watched from a browser

```
python 11_real_sorting_with_monitor.py --open                # dashboard first, then Isaac
python 11_real_sorting_with_monitor.py --setup-only          # the bench sheet and dashboard, no Isaac
python 11_real_sorting_with_monitor.py --no-wait-for-start --text "separa las piezas por colores"
python 11_real_sorting_with_monitor.py --voice --web
```

Demo 11 changes two things and inherits everything else, including demo 10's loop: it loads
`10_command_to_motion.py` by path and subclasses its `Robot`, so demo 10's behaviour cannot
regress.

**The physics are ones a bench could have.** Parts spawn at the mass their material gives them —
28 g for a 30 mm ABS cube rather than the 50 g demo 10 used for everything — with measured
friction and restitution instead of PhysX's defaults, and rolling resistance declared rather
than faked with damping on everything. The grip is no longer faked: demo 10's `Executor.grab`
writes a part's pose from the gripper frame whenever friction fails, which is how it picked up a
3 cm ball in a world where nothing is ever held by friction, and which no SO-101 can do. Demo 11
refuses to carry a part the jaws did not hold, and re-plans instead.

Two findings came out of that refusal, both measured by `tools/probe_grip_hold.py` and written up
in [`results/grip_and_release.md`](results/grip_and_release.md):

| | demo 10 | demo 11 | why |
|---|---|---|---|
| jaw close command | LeIsaac's 0.4 rad | 0.20 rad (`--close-command`) | at 0.4 rad the jaws hold a 30 mm cube **0%** of the time; 0.4 is tuned for upstream's 8 cm orange |
| gripper stiffness | 17.8 | 250 (`--grip-stiffness`) | a sphere meets the pads at a point, so the 10 g ball needs the stiffer servo: 79% of the lift at 17.8, 94% at 250 |
| release height | rim + 12 mm | rim + 32 mm of finger + 6 mm clearance (`--release rim` for demo 10's) | at demo 10's height the fingertips are 20 mm *inside* the wall, so the part is dragged along the cavity instead of dropped in |
| placement margin | 0.5 mm, absorbed by the retry loop | 0.5 mm, reported as a **failure** | cavity 56 mm − part 30 mm − camera error 12.5 mm; `real_world.recommended_tray()` is the 70 mm tray that fixes it |

`real_world.py` holds all of it and imports nothing from Isaac: materials, masses, the aperture,
the friction-cone grasp test, the release height, the placement margins, `preflight()` findings, and
`setup_sheet()` — a bill of materials plus the five measurements someone has to take on a real
bench, with the ones this repository has *not* measured labelled as assumptions.

### The dashboard

`--web` (on by default) serves `http://127.0.0.1:8765/` on a thread while Isaac keeps the main
thread. It is up **before** the simulator starts, which is where the environment gets set up and
the run gets started. Six panels: the setup form and its preflight findings, the decision-path
diagram, the camera frame the robot actually planned from, what perception found in it, the voice
state, and the log.

The diagram is built from `telemetry.FLOW`, the same list the robot publishes against, so it
cannot drift from the pipeline:

```
setup -> input -> understand -> capture -> perceive -> link -> ground -> plan -> execute -> verify
                                     ^                                          |           |
                                     +--------------- recover -----------------+-----------+
verify -> report
```

Inputs are three ways at once — the browser's box, the microphone, the terminal — and `q` in the
browser aborts a run mid-move, the same guarantee the viewer keys give.

| flag | default | |
|---|---|---|
| `--web` / `--no-web` | on | serve the dashboard on loopback |
| `--port` | 8765 | `0` asks the OS for a free port |
| `--open` | off | open a browser at it |
| `--no-wait-for-start` | waits | start Isaac immediately; implied by any `--text` |
| `--setup-only` | off | print the bench sheet and serve, without starting Isaac |
| `--close-command` | 0.20 rad | how far the jaws close |
| `--grip-stiffness` | 250 | gripper actuator stiffness |
| `--release` | `clear` | `rim` reproduces demo 10's release height |
| `--allow-kinematic` | off | permit demo 10's carry-the-part-by-hand fallback; results so obtained are **not** reproducible on hardware |

## Layout of the code

```
10_command_to_motion.py   the demo loop: rounds, commands, tracking, verification, recovery
11_real_sorting_with_monitor.py  demo 10's loop on real physics, instrumented for the dashboard
real_world.py             materials, masses, the friction cone, release height, the bench sheet
telemetry.py              the bus between the robot and the dashboard: nodes, events, frames
monitor/server.py         FastAPI + uvicorn on a thread; websocket, REST, the start gate
monitor/static/index.html the whole front end: no build step, no network
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
python tools/check_executor.py --headless          # motion parity, then a camera-derived ball pick
python tools/probe_ball_physics.py --headless      # ball damping sweep, contact, free-space reach
python tools/probe_gripper.py --headless           # the 36.2 mm aperture, open and closed
python tools/probe_grip_hold.py --headless         # demo 11: does friction hold, and at what close command
python tools/run_demo11.py                         # demo 11 with relaunch-on-fault, for a flaky Isaac
python tools/probe_grasp.py --headless             # does a pick work, and where
python tools/probe_pen.py --headless               # containment with and without the kerb
python tools/record_fixtures.py --headless         # re-record tests/fixtures from this layout
python tests/test_planner.py                       # 1468 grounding and planning checks
python tests/test_demo11.py                        # real-world model, bus and dashboard; no Isaac
python tools/measure_kinds.py --headless           # cube/ball thresholds
python command_model_v2/evaluate.py                # the 136-sentence evaluation
```

The first six need the simulator and take a few minutes each; the last four do not.

Isaac Sim prints a `_generic_model_output` DLL warning and a few `omni.fabric.plugin` messages
on every run, including successful ones. They are not the run failing.

Isaac also faults intermittently during start-up on this machine — `Windows fatal exception: code
0xc0000139`, then an access violation, on a thread with no Python frame — for demo 10 and demo 11
alike, about one launch in four. It is in Kit's native extension loader, not in any Python code:
clearing Kit's caches and excluding the two unused extensions both changed nothing. Run through the
retry launcher rather than waiting for it:

```
python tools/run_demo11.py                                   # demo 11, relaunching on a fault or stall
python tools/run_demo11.py -- --text "separa las piezas por colores"
```

`results/grip_and_release.md` has the full account, including a failure that looked exactly like a
simulator fault and was a `TypeError`.