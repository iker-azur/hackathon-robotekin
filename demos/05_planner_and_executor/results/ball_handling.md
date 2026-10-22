# Ball handling: what was actually wrong, and what fixed it

Three complaints, one cause for two of them, and one hard geometric limit that made the third
impossible rather than merely difficult. Everything below is measured, and the tool that
measures it is named next to the number.

Reproduce with:

```
python tools/probe_ball_physics.py --headless      # roll decay, reach, contact
python tools/probe_gripper.py --headless           # the aperture
python tools/probe_grasp.py --headless             # does a pick actually work, and where
```

---

## 1. "The balls move with anything touching them"

**Cause: the balls were spawned with no damping at all.**

`isaac_io.rigid_cfg` built them with a bare `RigidBodyPropertiesCfg()`, whose defaults are zero
linear and zero angular damping. A sphere with neither has no reason to stop rolling.

Measured on a clear 58 cm strip of table (`tools/probe_ball_physics.py --section roll`), a 4 cm
ball shoved along +x:

| shove | rolled | stopped? |
|---|---|---|
| 10 cm/s | 388 mm | never — still moving after 11.7 s |
| 25 cm/s | 818 mm | never |
| 45 cm/s | 1898 mm | never |

The demo's own disturbance range is 0.08–0.28 m/s and the arm's transit across the table moves
a ball at up to 0.45 m/s, so every interaction this demo could produce sent the ball off the
table. Nothing in the planner can answer that; it is a property of the spawn.

### The fix, and the sweep that chose it

Damping is a choice, so it was measured rather than guessed. Same ball, same shove, same strip,
sweeping damping, friction and PhysX's own rolling-resistance term:

| linear | angular | friction | 10 cm/s | 25 cm/s | 45 cm/s | residual creep |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.5 | 353 mm | 488 mm | 104 mm | 49 mm/s |
| 0 | 2 | 0.5 | 270 mm | 368 mm | 408 mm | 24 mm/s |
| 2 | 8 | 0.5 | 149 mm | 163 mm | 181 mm | 9 mm/s |
| **10** | **40** | **0.5** | **33 mm** | **37 mm** | **45 mm** | **2.6 mm/s** |
| 25 | 100 | 0.5 | 9 mm | 11 mm | 14 mm | 0.7 mm/s |

`linear_damping = 10`, `angular_damping = 40` is the pick. A shove of any speed the demo can
produce now moves the ball 33–45 mm: enough that being touched is visibly something, small
enough that it stops where it was knocked and stays there. The next setting up, 25 and 100,
only travels 9–14 mm, which is closer to a ball glued to the table than to one an arm can move.

Two knobs that sound like they should help were measured and do not:

- **Friction.** 0.5 → 1.0 changed the roll to the millimetre at every damping setting — 33/37/45
  mm either way. A rolling ball is not held back by sliding friction; that is the reason it
  rolls.
- **PhysX's `torsional_patch_radius`.** Set to 0.02 it returns numbers identical to no patch at
  all, on every run. It does not act on a sphere resting on a plane in this build.

Only balls get damping. A cube rests on a flat face and does not roll, and the cubes measured
0.0 mm of drift under a park sweep while the balls moved.

---

## 2. "…and then they return to their previous place"

**Cause: the demo did it on purpose, in three places.**

| what | where |
|---|---|
| every object written back to the declared layout at the start of a round | `Robot.populate` |
| anything the park sweep moved more than 25 mm put back | `Robot.park` |
| anything outside the working area put back on its square | `Robot.ensure_on_table` |

So a ball the robot had just placed somewhere was teleported back to where the layout said it
started, and the table never reflected what had actually happened.

**The fix.** An object is now placed once per round and from then on it is wherever it came to
rest. `park` *reports* what the sweep moved and leaves it; `ensure_on_table` now repairs only an
object that has left the table altogether, which is a broken state rather than a movement.

The park sweep is still measured rather than ignored — `PARK_REPORT_M` decides what is worth a
line of output — because the arm's forearm really does pass over the near half of the table.
With the balls damped it moves them a millimetre or two, and 5 mm is above the solver jitter a
settled ball shows with the arm nowhere near it.

### "The arm isn't close to them and they move"

This was measured rather than argued, by stepping a park and recording the closest robot link to
each ball on every step:

```
ball_red     closest link 176.4 mm, moved  1.7 mm
ball_green   closest link 166.2 mm, moved  3.6 mm
```

So the arm genuinely was not touching them, and they still moved a couple of millimetres. That
residual is resting-contact jitter, not a push: it is below the 5 mm reporting threshold and
below anything a person would call the ball having moved. Before the damping fix the same
measurement showed the balls creeping 49 mm/s.

---

## 3. "The ball getting must be improved so much"

### The limit: this gripper cannot hold a 4 cm ball

`tools/probe_gripper.py` measures the distance between the two finger bodies while the gripper is
commanded to every position the joints allow:

| close command | 0.00 → 1.00 rad | open command | 1.00 → 1.75 rad |
|---|---|---|---|
| gap | **36.2 mm at every value** | gap | **36.2 mm at every value** |

The fingers are fixed to their mounts; the joint drives them along their own axes and never
widens the span between the two bodies. 40 mm is therefore not "hard to grip", it does not fit.
The jaws close onto a ball wider than the distance between them and push it off the table.

This is why the three fixes recorded in the previous version of this file all failed — slower
descent, `--regrasp`, and re-aiming. None of them could have worked, because no aiming puts 40 mm
through a 36 mm gap.

**The fix: the balls are 3 cm across.** That leaves 6.2 mm of clearance in a 36.2 mm aperture, and
the balls are now the same size as the cubes, so the two are still told apart by shape — which is
what perception uses anyway, since it separates a disc from a square by their second moments and
not by how many pixels across they are.

Re-measured against ground truth over 20 randomised layouts (`tools/measure_kinds.py`):

```
ball  40 blobs  elongation 0.922..0.998  fill 0.9982..1.0000
cube  40 blobs  elongation 0.833..0.893  fill 0.9570..0.9840
kind accuracy: 80/80 (100.0%)
position error: 10.3..12.5 mm
```

### The other half: stop aiming at a stale position

Even a ball that fits gets knocked on the way in, because a plan is computed from one camera
frame and executed over several hundred steps. The old code re-aimed **once**, after the hover.

`Executor.track` now re-aims **throughout** the approach and the descent, re-reading the camera
every 20 steps and correcting the aim:

```
above    green: above #5  [aimed at (+0.509, -0.444), re-aimed 3x (largest 4 mm)]
down     green: down onto #5  [aimed at (+0.507, -0.445), re-aimed 2x (largest 5 mm)]
close    green: close on #5
lift     green: lift #5  [followed 21% of the lift (moved 15.7 of 75.0 mm), carrying kinematically]
camera sees 4 object(s) and 2 container(s); 1/1 placement(s) in a tray
```

Two things this needed, both found by the failure they fix:

- **The arm has to be masked out of its own camera.** With the gripper hovering, a tracked green
  ball read as a `cube` and a blue cube left the frame altogether — the log read *"the camera sees
  no blue at all (it sees cube red, ball red, cube green)"*. The old mask halved the brightness of
  a 46 px disc, which changes V and nothing else, and every colour band in `colors.py` is keyed on
  hue and saturation — so it removed nothing. The mask is now an 18 px disc painted **black**, and
  it is switched on automatically while the arm moves rather than left to a keypress.
- **The camera stops being consulted once the gripper is on top of its target.** Below 10 cm above
  the table the object is behind the gripper, so the aim is locked in from the hover and held. The
  stale-grasp check after the jaws close verifies it rather than assuming it.

### What grasping actually is in this simulation

Nothing here is held by friction. Demo 05 says so of the cube it inherited — *"Real friction
grasping of the 3 cm cube was unreliable in this sim"* — and `GRASP_FOLLOW_MIN` exists to catch a
partial hold that slides out. So the grasp that completes is the **kinematic** fallback, and the
demo says so in its own output:

```
1 real grasp(s), 2 kinematic
```

That is reported, never hidden. What the work above changed is that the jaws now arrive where the
object is, so the fallback is applied to an object that is genuinely between them.

### The result

`tools/check_executor.py` is the end-to-end check and it now passes both of its tests:

```
== parity: hardcoded pick, move 10 cm in x ==
  worst gripper tracking error: 0 mm
  cube moved by (+10.1, +0.0) cm, wanted (+10.0, +0.0) cm
== camera-derived: pick the red ball, drop it in the red box ==
  red ball at (+0.3601, -0.4420), ground truth (+0.3600, -0.4350), 7.0 mm apart
  ball landed 5.7 mm from the tray centre, at z 0.0619
RESULT: SUCCESS
```

Before this work the same check reported *"the ball landed 118 mm from the tray centre, outside the
48 mm cavity"* and failed.

`tools/probe_grasp.py`, which runs the whole pick at each point of a grid, went from 8 of 9 ball
picks lifting with the target moving 57–371 mm, to 13 of 20 lifting with the target moving about
16 mm.

---

## 4. Sorts, and a frame error worth 92 mm

### A multi-object sort went 0 of 3 → 3 of 3

Two bugs, both found by looking at what the code did rather than what it said.

**The loop did the thing its own comment said it must not.** `_one_at_a_time` re-planned from a
fresh frame between objects, but the combined plan was run *first*, and each pass then ran whole
batches again. A plan for four objects is 32 steps built from a single camera frame; by its later
steps two objects are inside trays and the arm is standing over one. Now `handle` routes any
multi-pick command straight to the one-at-a-time path, and each pass runs **one** pick, cut at its
`lift`.

**A pick that "succeeded" but missed the tray was never counted as a failure.** The grasp closed on
the object, the fallback carried it, and the release landed beside the tray — which looks like
success to every counter in the file. The sort looped on it. All three of `stale`, `aborted` and
`not verified` now count, with `MAX_PICK_ATTEMPTS = 2` before it moves on and names the reason.

### The real cause: `track` and `do` disagreed about which frame they were in

`do` receives a **hold point** — where the object should be — and converts it to a gripper *body*
pose with `goal = pose - HOLD`. `track` was handed the same hold point and interpolated `z`
verbatim, so it drove the body **92 mm too low** on every descent: the fingers went through the
table.

The symptom was geometric and unmistakable once measured:

```
not verified: cube_blue came to rest at (+0.432, -0.442), 85 mm outside the cavity in x
retry lift: followed 29% of the lift (moved 18.5 of 152.2 mm)      <- should be ~60 mm
```

A lift that travels 152 mm for a 60 mm raise is the 92 mm appearing twice. `track` now converts
`target_z` into body coordinates the same way `do` does, and the measured lift dropped to
**63.6 mm** — and the sort went to **3 of 3**.

`TRACK_OBSERVE_ABOVE_Z` moved with it, from a hold-point 0.1409 to a body height of 0.20. The hover
sits at a body height of 0.223 and the bottom of a descent at 0.148, so the camera is consulted
through the approach and locked in below that.

### A systematic 7 mm aiming error, now subtracted

`perceive` casts one ray from a blob centroid to the **table plane**, which assumes the centroid is
on the table. It is on the object's top face, 3 cm higher, so the ray overshoots by a fixed amount.
Measured over 120 blobs in the 20 recorded frames (`tools/measure_bias.py`, no simulator needed —
the layout each frame came from is a pure function of the trial number):

| blobs | n | dx | dy | mean abs err |
|---|---|---|---|---|
| balls | 40 | +2.1 | **−7.3** | 4.8 mm |
| cubes | 40 | +1.0 | **−6.5** | 4.2 mm |
| trays | 40 | +0.3 | **−10.3** | 8.0 mm |

Subtracting it takes the residual from 10.3–12.5 mm to **4.2–4.8 mm**. That matters because
`GRASP_REACH_M` is 20 mm: 7 mm of one-signed error is a third of the whole budget, spent before the
arm has moved. Trays get their own figure because a tray blob's centroid is over the rim-and-interior
shape, not the footprint. Applied in `scene_spec.aim_point`, used by `grasp_pose`, `place_pose` and
`locate` — deliberately not inside `perceive`, so `pos_m` stays the raw measurement with
`pos_m_depth` beside it, and grounding is left alone because 7 mm does not change whether something
is inside the working area.

### And the retry path now closes the loop too

`_retry_one` used to drive `above` and `down` open-loop, which is the one place in the file allowed
to know the answer and the one that most needed the camera: the object it is retrying is one that
has just been dropped somewhere wrong. It now tracks, and **skips the retry rather than aiming at a
guess** when the camera loses the object.

### Misplacements now say which axis

`not verified: cube_blue is not in the tray it was sent to` became:

```
not verified: cube_blue came to rest at (+0.432, -0.442), 85 mm outside the cavity in x and 27 mm in y
not verified: cube_blue is at (+0.327, -0.484) and just inside the cavity, but settled 0.4 mm from
              the resting height - it was balanced on the rim
```

Two different faults with nothing in common, and the old message sent the investigation to the wrong
axis both times.

## 5. Still not solved

- **A tracked ball is often read as a `cube`** while the gripper is near it. The mask removes the
  arm's own pixels, but the ball's remaining blob is no longer a clean disc. Tracking falls back to
  matching on colour alone, which is allowed only for a colour unique on the table.
- **No grasp in this demo is held by friction.** That is a property of the simulator and the gripper
  asset, not something the code can change.
- **A release can still leave an object balanced on the tray rim**, which verification correctly
  reports as a miss and the retry may or may not fix. The tray cavity is 56 mm for a 30 mm object, so
  the margin is real; the loss is in the last millimetres of the descent.