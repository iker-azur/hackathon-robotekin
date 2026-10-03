# Variant 2: objects the robot can see but cannot reach

`--variant out_of_reach`. A **static** table with two objects placed beyond the arm, and the
robot has to say so rather than try.

```
python 10_command_to_motion.py --variant out_of_reach
python 10_command_to_motion.py --variant out_of_reach --text "pon el cubo azul en la caja roja"
```

## What it shows

The robot can see more than it can touch. Asked for something out of reach it names the
object, gives the reason with numbers, and says what would make it possible — in the language
the command came in:

```
ask (es): Veo un cubo azul, pero no puedo llegar: esta a 35 cm de la base y el brazo
          llega a 33. Muévelo más cerca.
ask (en): I can see one blue cube, but I cannot reach it: it is 35 cm from the base and the
          arm reaches 33. Move it closer.
ask (eu): Ikusten dut kubo urdina bat, baina iristu ezin dut: 35 cm dago ohatik dago eta
          burua 33 cm-rera iristen. Hurbildu gertuago.
```

Two reasons are distinguished, because they need different responses:

* **beyond the arm's length** — the person's problem; move it closer.
* **outside the working area** — a decision this demo made; move it into the area.

If one candidate is reachable and another is not, the reachable one wins silently. Asking a
question there would be pedantry: if there are two red cubes and only one is in reach, the one
they can have is the obvious answer.

## The layout, and why these positions

Both out-of-reach objects sit at y = −0.325, just past the working area's far edge, at x = 0.190
and x = 0.510 — symmetric, so the demo is not accidentally kinder on one side.

**Chosen so they are genuinely unreachable, not merely outside the declared area.** At that
depth the base is 0.315 m away in y, so the reach envelope leaves |dx| ≤ 0.10 m: anything past
x = 0.45 or before x = 0.25 is physically out of the arm's length. Both are **35 cm from the
base against a 33 cm arm**, a 2 cm margin against a localisation error of about 11 mm for a
tray and up to 28 mm for a ball.

Placing them merely outside `WORK_X` — at x = 0.205, say — would have been 23.5 cm from the base
and comfortably inside the arm's length. The robot could have reached it, and the refusal would
have been policy dressed up as physics.

**The first attempt put the blue cube at x = 0.245, where it did not exist.** It overlapped the
blue tray's footprint, and the rendered frame showed three objects and no blue cube anywhere.
`clearance` did not catch it either — it compares centre distances minus half-extents, and an
L-shaped overlap can come out positive. x = 0.190 is 85 mm from the tray in x, well past the
50 mm their combined half-widths need. This is the same shape of mistake as the one that hid
balls behind the arm earlier: a layout that looks fine in the numbers and is wrong in the
render.

## How it is built

* `scene_spec.why_unreachable(x, y)` — one definition of "the robot cannot work here", used by
  both planning and grounding. It used to be two copies of the same arithmetic, once in
  `planner` and once implicitly, which is how a "no" from one and a plan from the other
  happens. `planner.in_workspace` now delegates to it.
* `scene_spec.reach_facts(x, y)` — the numbers behind that sentence, so `ground` can build the
  reply in the user's language. Rendering the reason in English and splicing it in produced
  "Veo un cubo azul, pero no puedo llegar: esta 35 cm from the base, past the 33 cm the arm
  can reach".
* `ground` filters candidates to the reachable ones and returns `not_reachable` with a
  three-language question when none are. `planner` surfaces that status; it used to surface
  only `ambiguous`, so the robot answered "nothing to pick up" about an object it could plainly
  see.

The scene is **static** and the round loop does not re-randomise it. The point is that "that
cube is out there and cannot be reached" is a stable fact about the table rather than a coin
flip each round.

## What it does not fix

The reachable ball in this scene still cannot be grasped — that is the descent-ejects-balls
problem in `approach_drift.md`, unchanged by anything here. What this variant adds is that the
robot no longer *pretends* otherwise: it says it cannot reach the two far objects, and when it
fails on the near one it says that too.

Widening `LINK_TOLERANCE_M` from 25 to 35 mm was needed along the way. A ball is seen up to
28 mm from where it is, and at 25 mm the link silently failed, the grasp was skipped with no
note, and the placement was reported unverified for an object the robot was looking at.