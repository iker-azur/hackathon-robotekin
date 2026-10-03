# Why the robot aimed at the wrong place, and why it looked like it worked

Measured with `tools/probe_approach_drift.py`.

## The symptom

The robot reached for a ball that was no longer where it had been seen, closed its jaws on
empty air, and the demo reported a successful placement.

## Why it looked successful

`Executor.grab` writes the held object's pose from the gripper frame every step. That is demo
05's deliberate kinematic-carry mechanism, and it is what makes the shipped 3 cm cube
reproducible. But nothing checked whether the object was actually there first, so the same
mechanism silently picked an object out of thin air: the jaws closed 371 mm from the ball,
`grab` teleported the ball into the gripper, and the placement verified as correct.

A check that cannot fail is not a check. This is the same failure as the earlier
"verification passed vacuously" bug, one layer deeper: that one could not detect a *missing*
placement, this one could not detect a *fake* one.

## How far the target actually moves

Planned from a real camera frame, then driven `above` and `down`, comparing where the jaws
aimed with where the object was:

| target | moved during `above` | distance at `down` | jaws can reach |
|---|---|---|---|
| `cube_blue` | 0.0 mm | 9.6–11.7 mm | 15 mm |
| `cube` (shipped) | 0.0 mm | 10.1–11.3 mm | 15 mm |
| `ball_red` | 57.6–60.4 mm | **192–371 mm** | 20 mm |

The jaws closed more than half an object away on **3 of 9** approaches. Every one was a ball.
A cube moves 10–12 mm over the same approach, which is perception error and well inside reach.

## The descent is not the cause; the ball keeps rolling

Sweeping the descent duration:

| `down` steps | seconds | ball drift at `down` |
|---|---|---|
| 15 | 0.25 | 145 mm |
| 25 | 0.42 | 119 mm |
| 40 | 0.67 | 206 mm |
| 60 | 1.00 | 146 mm |
| 100 (default) | 1.67 | 373 mm |
| 300 | 5.00 | 1860 mm |
| 800 | 13.4 | 6791 mm |

Two things follow. The ball is not squeezed out by the descending fingers — at 15 steps,
where it has almost no time to react, it is still 145 mm away. And the relationship is not
monotonic at the fast end but is brutal at the slow end, which is the signature of **something
rolling freely for the duration of the motion**. The arm knocks the ball during its transit
across the table, and the ball then rolls away on its own while the gripper descends.

The same signature appears in the parking sweep, where duration is likewise how long the
forearm leans on the balls: 1200 park steps drifted 1893 mm against 16.6 mm for 40.

## What was tried, and what the measurements ruled out

Four candidate fixes were tested. Three are disproven by measurement, which is worth recording
as carefully as the thing that worked, because they are the obvious next ideas.

**1. A gentler descent — disproven, and backwards.** Sweeping the step count:

| `down` steps | 15 | 25 | 40 | 60 | 100 | 300 | 800 |
|---|---|---|---|---|---|---|---|
| ball drift | 145 mm | 119 mm | 206 mm | 146 mm | 373 mm | 1860 mm | 6791 mm |

At 15 steps, where the ball has almost no time to react, it is still 145 mm away — so the
descending fingers are not squeezing it out. And the slow end is catastrophic, which is the
signature of something **rolling freely for the duration**. Same signature as the parking
sweep, where 1200 steps drifted 1893 mm against 16.6 mm for 40.

**2. Reversing the layout — implemented, measured, reverted.** Put the receptacles between the
robot and the loose objects, with all four loose objects in one row beyond them, so the arm's
transit crosses only kinematic trays. Validated offline first (reaches 214–284 mm against a
330 mm envelope, minimum same-colour gap 35 mm, and a 40 mm row gap against a 13 mm occlusion
shadow, so nothing hides anything), and `conflicts` was clean over 400 random seeds.

It changed nothing. Balls still moved 57–62 mm during `above` and ended 106–245 mm from the
aim — the same as before. The parking sweep was also unchanged, 16.3 mm against 16.6 mm.

The reason is that the arm **descends through the height of the object row on the way to the
target**, so it collides with the target itself no matter which side of it the other objects
are on. Since the layout flip would have invalidated the colour, shape and tray calibrations
for no measured gain, it was reverted.

**3. Approaching in two legs — disproven, badly.** Travel to a waypoint high above the target,
then descend vertically, so the diagonal descent through the row never happens:

| waypoint z | 0.20 | 0.30 | 0.40 |
|---|---|---|---|
| ball drift | 1772 mm | 1793 mm | 1073 mm |

Worse by an order of magnitude. The extra motion at table height does more damage than the
diagonal it was meant to avoid.

**4. Refusing the fake grasp — this one worked.** See above.

## Option 2, built and measured: re-aiming after the arm knocks its own target

Rather than stop the arm knocking things, join the roll: after the hover, wait for the target
to stop moving, look again, and aim at where it actually ended up. Implemented as
`Executor.run(retarget=...)` plus `make_retarget` in the demo.

**The two preconditions both hold**, which was not guaranteed:

- the ball does come to rest — displaced 157 mm, taking 6.7 s (400 steps);
- the camera still sees it from the hover pose, to within **14 mm**. The arm hovering above
  where the ball *was* does not hide it, which was the thing that could have made this
  impossible.

So the mechanism works: it recovers a ball that rolled out of the working area (measured at
x = −0.22 on a table whose near edge is x = 0.003), puts it back, re-perceives and replans.

**And it still does not rescue balls.** Re-aimed at a freshly measured position, the descent
displaces the ball **77 mm** — past the 20 mm the jaws can reach, so the grasp is correctly
refused as stale.

That is the finding that matters, and it revises the diagnosis. The ball is not only knocked
on the way in; **the descent ejects it even when the aim is correct.** Re-aiming fixes the
aiming error and leaves the real cause untouched.

## What that points at instead

The jaws cannot close on a 4 cm ball. `OPEN, CLOSE = 1.0, -1.0` drives the gripper action to
`JAW_CLOSED_RAD = 0.15`, which is essentially shut, and a 4 cm sphere does not fit between
fingers that close to nothing — it is squeezed out, which is the 77 mm. A 3 cm cube survives
the same approach at 10 mm, which is why cubes work.

The fix is therefore not perception and not planning. It is to stop treating "close" as binary
and to drive the gripper joint to an opening sized to the object — about 2 cm for a 4 cm ball,
closed for a cube. That is a change to how the executor commands the gripper rather than a
change to the scene, and it is the one thing left that addresses the cause.

It is not implemented here: the gripper action term is configured in the LeIsaac Hub module
that `env_loader` downloads, so `closed_command` is not a parameter this demo owns, and
overriding the action term was judged a larger change than it should be made untested.

Every motion of this arm near a 4 cm ball ejects it, and the ejection scales with how long the
ball is free to roll. The objects are 50 g spheres on a low-friction table, so a knock sends
them metres. The one lever that follows directly from that is **damping the rolling** — the
drift is a function of free-roll time, so it is the free-roll time that has to change. That is
a physics change to the scene rather than a code fix, and it has not been tried.

The alternative, which needs no physics change, is to **not require the ball to stay still**:
accept that the arm will always disturb its target, and grasp it anyway by reaching for where
it has rolled to. That is a prediction problem rather than a rejection problem — it needs a
second perception pass timed to the end of the approach, when the ball is stationary — and it
is what "the camera should work more" would have to mean in practice.

## What is fixed

- `GRASP_REACH_M` (20 mm, a ball's own radius): at the end of `close`, if the object is not
  within reach of where the jaws aimed, the grasp is marked `stale`, the run **aborts** rather
  than carrying on, and no kinematic carry is started. The demo says *"nothing to close on:
  the object is 113 mm from where the jaws aimed"*.
- `--regrasp N` (default 2) re-perceives and replans on a stale grasp, which is the right
  response to a target that moved.
- A stale or lost grasp is no longer counted as a placement.

## What is not fixed

**Replanning does not converge for balls**, measured: 4 attempts, 4 stale grasps. The reason
is that the robot cannot observe a ball without disturbing it — parking the arm to see the
table sweeps the same row, and the ball rolls further each time. Perception and action are
fighting over the same object.

Cubes are unaffected and work: 10–12 mm of drift against 15 mm of reach.

The two ways out, neither of which is a code tweak:

1. **Stop the ball rolling.** The drift scales with how long the ball is free, so damping it
   is the direct fix. Current rolling behaviour is not what a rubber ball on a table does —
   a real one stops within a few tens of centimetres.
2. **Move the trays between the robot and the loose objects.** The arm's transit crosses the
   object row only because the receptacles are on the far side of it. Putting them on the near
   side means only the target is disturbed. This is the layout change already identified as
   the fix for the parking sweep, and it would fix this too — at the cost of re-measuring the
   colour, shape and tray calibrations.