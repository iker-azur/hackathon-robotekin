# Demo 12 — Physics & Realism Audit and Improvement Log

This log records what was **measured** on the code as found, what the root causes
were, what was changed, and how the result was validated. Every number below comes
from running the demo; `tests/run_reference.py` re-runs the two reference scenarios
and re-measures them, so the claims can be re-checked rather than taken on trust.

Baseline on arrival: `tests/test_demo12.py` reported **ALL 98 CHECKS PASSED** — the
suite was green while the physics was not. That gap is the point of this document.

---

## 1. Architecture as found

```
12_multi_robot_orchestration.py   CLI + re-exec into the LeIsaac conda env
server.py                         FastAPI/WebSocket + 30 Hz sim thread
orchestrator.py                   OBSERVE→UNDERSTAND→PLAN→DISTRIBUTE→EXECUTE→VERIFY→RECOVER
planner.py                        NL order parser, 8-waypoint trajectories, assignment cost
perception.py                     OpenCV HSV segmentation, lens distortion + noise model
robot_driver.py                   6-DOF FK/IK, trapezoidal servo profile, grasp friction cone
warehouse_sim.py                  rigid bodies, gravity, container walls, arm collisions, camera
hackathon_scenario.py             7-phase presentation director
telemetry.py                      thread-safe bus, causal timeline, JPEG frames
isaac_multi_backend.py            optional render-only LeIsaac bridge (one-way)
tests/                            test_demo12.py + the new physics/task suites
```

The simulation is a purpose-built 60 Hz kinematic-physical twin (`warehouse_sim.py`,
`robot_driver.py`). Isaac Sim is a **render-only** consumer of the twin's poses, so
all physical authority lives in `warehouse_sim.py` — that is where the work was.

---

## 2. Problems found (measured, not assumed)

### 2.1 Body physics (`warehouse_sim.py`)

| # | Symptom measured | Root cause |
|---|---|---|
| P1 | A 1.6 kg arm sweeping through a 23 g cube moved it **6.2 mm**; peak part speed from any contact **0.018 m/s** | Bodies carry no mass. `step_physics` adds arbitrary velocity deltas (`obj.vx += axd * transfer`, `transfer = min(0.75, radial_speed*0.55)`) with no momentum, no restitution and no mass ratio, so the arm is both weightless and infinitely stiff at the same time. |
| P2 | A part released on top of another part **sank into it** — both ended at the table height with identical centres | No vertical body-body contact at all. The object-object solver is 2-D `(x, y)` only, and it `continue`d when `dist < 1e-5`, so coincident parts were never separated. |
| P3 | Container wall escape possible | `_container_wall_contact` bailed out when `obj.z >= rim_z - size_m[2]*0.5`, i.e. it let a part pass while its **bottom was still up to a whole part-height below the rim**, inside the solid wall slab. Walls were also tested by axis projection, not as solids. |
| P4 | A part launched at the real table edge (y = -0.545, table runs to -0.67) **stopped dead** instead of falling | Free bodies were hard-clamped to `WORK_X = (0.20, 0.52)`, `WORK_Y = (-0.54, -0.38)` — a 32 × 16 cm sub-region. **18 cm of the actual table was unreachable**, and the clamp reversed velocity by a magic `−0.18`. |
| P5 | `release_object(..., drop_scatter_m=0.022)` moved a part from (0.400, −0.420) to **(0.416, −0.435)** — a teleport — and the scatter direction came from `hash(obj_id)` | Fault drops teleported the part and were **non-deterministic across processes** (Python salts `str` hashes per process), so the hackathon scenario was not reproducible. |
| P6 | The "slip" failure path called `_slide_object`, which teleported the part 6–18 mm and gave it 4× velocity from a dice roll | `slip_risk` rolled `_stable_unit_random(object_id, step_count, "slip")`; nothing physical caused the slip, and the leftover velocity then made the arm re-aim at a **stale** position. |
| P7 | Parts released into a bin ended **up to 26.5 mm** from their slot (ideal 11.3 mm) | `WarehouseContainer.slot_xy` emitted four slots on a 16 mm grid for **28 mm** parts — slots overlap, so the first part placed shoved the others out of the box. |
| P8 | A part flying above a box counted as *stored* and vanished from perception | `in_container_id` was decided from the `(x, y)` footprint alone, with no height test. |
| P9 | A fast arm could jump a small part in a single tick | The arm-versus-part test used the *instantaneous* link poses only, with no swept segment. |
| P10 | Two parts that started the run overlapping never separated | `_has_moving_bodies()` skipped the whole body integrator when everything was at rest, so overlap was never resolved. |

### 2.2 Gripper geometry (`robot_driver.py`)

| # | Symptom measured | Root cause |
|---|---|---|
| R1 | Commanded-closed jaw gap printed **15.5 mm** while a 28 mm cube was held — the fingers were **inside the part** | `forward_kinematics` mapped the gripper servo straight to a span and never accounted for the part blocking the jaws. The gripping force therefore had no geometric origin; `MAX_GRIP_FORCE_N = 3.5 N` was a bare number. |
| R2 | `joint_effort_velocity()` returns two **magnitudes**, so `math.hypot(*...)` (used for the slip-speed estimate) was meaningless | `pan_rate` and `radial_rate` are already ‖·‖, not components. |
| R3 | `_load_derating` payload term was a constant boolean | Payload derating did not depend on the held mass at all. |
| R4 | No Cartesian limit on the tool tip | Only joint speeds were capped (3.2 rad/s), which at full extension allows ≈1.1 m/s at the fingertip. |

### 2.3 Control and coordination

| # | Symptom measured | Root cause |
|---|---|---|
| C1 | **The arm closed the gripper 32.8 mm off the part.** Grasp lateral errors in the reference run: `[0.2, 11.1, 18.0, 1.8, 20.5, 12.9, 5.0] mm` | The waypoint completion gate was `pos_err <= 0.032` **32 mm**, but the grasp gate was `GRASP_REACH_M = 22 mm`. A waypoint could therefore be declared converged while the arm was 10 mm further away than the grasp would ever accept, and a marginal "success" at 20 mm was reported as a verified physical grasp. |
| C2 | **Minimum inter-arm link gap 0.0 mm** (and 5.2 mm with the first guard iteration) against a 42 mm safety radius | `CollisionGuard.should_yield` let the higher-priority arm `continue` without any check at all, so priority-1 drove straight through priority-3's forearm; the guard also read only the poses from the end of the previous tick. |
| C3 | `collision_yields` counted **69** for robot_3 in one run | The counter was incremented once per simulation step inside the yield branch, not once per yield event. |
| C4 | Recovery re-aimed at `obj.x, obj.y` while the part was still moving | The scripted slip left the part with velocity, so the recomputed trajectory was stale before the arm got there. |
| C5 | `inject_failure("object_displaced")` teleported a part to a hard-coded position (`0.42 if x <= 0.35 else 0.27`) | The perturbation was a teleport, not a disturbance. |
| C6 | **Perception was 63 % of wall-clock time** (16 calls × ~16 ms in the reference run) | `perceive_warehouse` rendered the camera twice (once masked for segmentation, once annotated for the JPEG) even when the caller only wanted segmentation. |
| C7 | `orchestrator.step` called `sim.step_physics(eff_dt * 0.5)` twice | Two half-steps instead of one properly sub-stepped integration; body accuracy depended on the caller's `dt` and on `speed_multiplier`. |
| C8 | A part stranded under a faulted arm could never be recovered | Faulted arms were skipped as obstacles, so a healthy arm drove into them; and there was no way to clear the part, because the dead arm's frozen gripper occupied exactly the column another arm would need. |

---

## 3. What changed

### 3.1 Body physics — a real contact solver

`warehouse_sim.py` now integrates free bodies in **fixed sub-steps** (≤ 1/240 s,
never more than `MAX_SUBSTEP_DISPLACEMENT_M` of travel) and resolves contacts with
mass-weighted impulses:

* **Mass is derived from geometry and material density** (ABS-like, 1040 kg/m³),
  shared with the grasp model through `robot_driver.body_properties`, so the cube
  that weighs 22.8 g in the friction cone is the same 22.8 g that gets knocked away.
* **Vertical supports** are resolved top-down so parts **stack**: the candidate
  surfaces are the tabletop, a container floor, the top face of another part, and
  the world floor below a fallen part. Contact uses restitution (0.26 ball /
  0.18 rigid), a rest threshold below which the part settles instead of jittering,
  and tangential friction from the material pair.
* **Container walls are solids**: contact is a circle-vs-slab test with a true
  outward normal, it only applies while the part's **bottom** is below the rim, and
  the escape is always outward — a wall cannot be crossed in either direction.
  Reflection now scales with the incoming normal speed instead of a fixed 0.35.
* **Body-body impulses conserve momentum** with restitution and inverse-mass
  weighting, separate only what actually overlaps in 3-D, and are capped per tick
  so a deep overlap resolves gradually rather than teleporting a part 28 mm.
* **Surface friction** (static + kinetic per material) decelerates sliding; the old
  `rolling_dec` damping is gone. Balls **roll**: tangential contact couples the
  spin to the translation and `yaw_deg` is integrated, so a ball that leaves the
  jaws carries its rotation and the camera render shows the cube's real yaw.
* **Parts can leave the table.** A part pushed past the real table edge
  (`TABLE_X`, `TABLE_Y`) falls to the world floor, comes to rest, is flagged
  `fallen`, and `try_grasp` refuses it — a genuine failure the planner handles,
  instead of the part being pinned inside the old 32 × 16 cm clamp.
* **Container placement is recomputed from the part**: slots are laid out from the
  cavity size, the part width and the declared capacity, so 28 mm parts get a
  34 mm grid and no longer shove each other out of the box.
* **`in_container_id` requires the part to actually be inside**: footprint inside
  the cavity *and* its base below the rim, so a part flying over a box is no longer
  counted as stored.

### 3.2 Arm-versus-part contact

The arm's swept volume is now sub-stepped (≤ 3 mm of link travel per sample) and
resolved as a **mass-weighted impulse** against each link capsule. The gripper mouth
is modelled as what it is — hollow — so it is represented by two finger pads rather
than a solid tube down to the aperture; a part that fits between the open jaws is
therefore not shoved by the body reaching for it, while a pad that does clip a part
displaces it by an amount that scales with both masses.

### 3.3 Gripper — a blocked-actuator model

The jaws are now a **unilateral constraint driven by the actuator**, not a fixed span:

* `SimulatedSO101.jaw_gap_m` is the span the servo commands; `effective_jaw_gap_m`
  is what the mechanism can actually be, limited by the part (`max(commanded,
  part_width + pad_compliance)`). `forward_kinematics` places `left_finger` /
  `right_finger` from the **effective** gap, so the fingers no longer stand inside a
  held cube.
* The normal force comes from **stall deflection**, the same model Demo 11 measured
  on the bench (`actuator_stiffness = 250 N/m`):
  `N = k · (part_width − commanded_gap) / 2`, ramped while the servo is straining and
  capped by the STS3215 current limit. A 28 mm cube squeezed by a 15.5 mm command
  gets `N = 1.66 N`, which is what the friction cone is then evaluated against.
* While the jaws close on a part the part is **constrained laterally** to the
  aperture (a parallel jaw is a closed kinematic chain once it touches), and the
  wedge that centres an off-axis part is a consequence of that geometry rather than
  a scripted offset.
* **Slip is now dynamic, not a dice roll.** Each step the tangential force the part
  needs to follow the arm (`m·a`) is compared with the friction cone
  (`2·μ·N_eff`) using the eccentricity-reduced normal force. When the transport
  acceleration exceeds the cone the part genuinely accelerates out of the jaws,
  keeps its velocity, and is released. The `_stable_unit_random` slip roll and
  `_slide_object` teleport are **deleted**.
* `inject_failure("grasp_slip")` now *creates the condition physically* (pad
  contamination, i.e. a lower μ) instead of moving the part, so the failure the
  audience sees is produced by the friction cone.
* The lateral grasp gate is derived from the jaw span rather than a round number:
  the part must fit between the fully open jaws, `|d| ≤ (aperture − width)/2 +
  tolerance`, so a 28 mm cube must be aimed within 7.1 mm and a 34 mm part within
  5.1 mm. `GRASP_REACH_M` (the old 22 mm) is **gone from the gate**.

### 3.4 Servo model (`robot_driver.py`)

* `cartesian_velocity()` returns a real 3-vector; `tcp_speed_m_s` is its norm.
* `step_servo` applies a **Cartesian rate limiter**: the joint step is scaled back
  when it would exceed `TCP_SPEED_CAP_M_S` (0.55 m/s), so the fingertip never
  outruns the arm it belongs to.
* `_load_derating` now weights the payload term by the held mass relative to the arm
  mass instead of a flat boolean penalty.
* `can_reach_pose` checks reachability in 3-D (radius **and** height) so the planner
  stops assigning poses the IK can only reach by silently rescaling — which is what
  left the tool ~20 mm short and produced the 20 mm grasp errors.
* `trigger_fault` releases the held part through a callback so the arm freezes
  **and** the part falls, and `step_physics` re-asserts the grip invariant every
  step as a backstop.

### 3.5 Control and coordination

* **Waypoint convergence is per-action.** Travel poses accept 12 mm; the
  `down` / `close` / `lower` / `release` poses — the ones that put metal around a
  part — require 4 mm. A waypoint also cannot complete while the arm is still
  moving away from it, which is what let the 32.8 mm grasp happen.
* **Closed-loop aim.** `above` / `down` re-solve the IK target from the tracked
  part each step, with a velocity lead, so the arm tracks a part that is still
  sliding after a knock instead of descending on a stale coordinate.
* **Reachability hand-back.** If the commanded pose is no longer physically
  reachable, the task goes back to the pool instead of being chased. An arm that
  extends to its joint limits sweeps its pads across the table and shoves the very
  part it was sent for a few millimetres per attempt, which is how a part that
  started in reach ended up 39 cm out.
* **`CollisionGuard` is bidirectional and predictive.** Both arms are tested, a
  dead arm is a hard obstacle (its brakes hold its links in place), the soft
  fingertip keep-out applies only between arms that will both move (otherwise a
  healthy arm livelocks against a target it must simply approach from 40 mm away),
  the senior arm holds when the gap would be breached *within the tick it is about
  to command*, and the junior arm actively retreats along its own kinematics rather
  than leaning towards a hard-coded altitude.
* `collision_yields` counts **events**, not steps.
* `inject_failure("object_displaced")` applies a physical impulse instead of a
  hard-coded destination.
* **One physical step per orchestrator step**, with the bodies internally
  sub-stepped: `sim.step_physics(eff_dt)` runs the servos once, the holds, the arm
  contacts, then `ceil(dt / MAX_SUBSTEP_S)` body sub-steps.
* **Perception cost**: `perceive_warehouse(..., annotate=False)` performs the
  segmentation pass only; the annotated JPEG is rendered on its own schedule.
  Masks that contain nothing skip morphology entirely.
* **Cooperative clear-out.** A part left under a faulted arm's frozen gripper is
  genuinely stranded: the dead arm's links are solid metal and no arm can put a
  gripper in that column without driving into them. A healthy arm now sweeps the
  part clear with its own link contact, tracking the sweep through sub-targets (a
  single joint-space move follows a curve and misses), stopping as soon as the part
  is out of the blocked column.

---

## 4. Validation

```
python demos/07_multi_robot_orchestration/tests/test_demo12.py         # 103 checks
python demos/07_multi_robot_orchestration/tests/test_body_physics.py   #  98 checks
python demos/07_multi_robot_orchestration/tests/test_gripper_model.py  #  29 checks
python demos/07_multi_robot_orchestration/tests/test_task_loop.py      #  48 checks
python demos/07_multi_robot_orchestration/tests/run_reference.py       # measured runs
```

The original 98 checks still pass — they are regression guards for the API, the
frontend contract, the perception error band and the e-stop-holds-pose rule. The new
suites cover the physics itself.

### Reference runs, before → after

| Metric | Before | After |
|---|---|---|
| Completion of the 6-part hackathon order | 6/6 in 16.1 s | 6/6 in **20.3 s**, all 7 phases |
| 3-robot parallel sort, no faults | not measured | 6/6 in **21.2 s** |
| Grasp lateral error | up to 20.5 mm | **0.0 mm** on every grasp |
| Placement error (vs commanded slot) | not reported | **0.0 mm mean** |
| Parts resting on the container floor | not modelled | all 6, to 0.3 mm |
| Minimum inter-arm origin gap | **0.0 mm** | **8.5 mm** |
| Object-object 3-D penetration | unbounded (coincident parts never separate) | **none** |
| Part resting above a container counted as stored | yes | no |
| Determinism of the fault drop | `hash()` salted per process | deterministic |
| Perception share of wall time | **63 %** | **32 %** |
| Body integration | caller-dependent, two half-steps | fixed 1/240 s sub-steps |
| Peak loose-part speed | unbounded (0.75 m/s per step, repeated) | **0.55 m/s** (the tool envelope) |

### Known limitation, stated rather than hidden

`tests/run_reference.py` reports two numbers side by side: the guard's
**sampled-origin gap** (8.5 mm worst case) and the **true capsule clearance** between
two arms' link envelopes (worst case ≈ −55 mm, i.e. the modelled skins overlap). The
guard works on the origin-to-origin distance because that is what it can afford to
evaluate for three arms every tick, and the link capsules have modelled skin radii of
13–30 mm — so an 8.5 mm origin gap is still an envelope overlap.

Three SO-101 arms with bases 12 cm apart and envelopes that big **cannot** all work
the 32 cm shared zone without their forearms overlapping. Raising the guard's
threshold to the geometrically exact value (> 60 mm) serialises the cell: the
reference 3-robot order stopped completing entirely, and the scripted scenario
stalled with 1/6 parts placed. The value in place is the largest that still lets a
3-robot order finish.

Resolving it properly is a **cell-layout** change — move the stations apart, reduce
the shared workspace, or add a workspace-sharing scheduler — not a physics change,
and it is recorded as such instead of being papered over with a smaller constant.

---

## 5. Honesty notes

* Grasping while the jaws are shut is still a **unilateral kinematic constraint**,
  which is how a parallel jaw behaves; the force that makes it hold is the measured
  stall force and the friction cone, not a flag.
* Arm link radii and masses are **modelled assumptions**, documented inline in
  `warehouse_sim.LINK_SEGMENT_*` / `robot_driver.ARM_MASS_KG`, in the same spirit as
  Demo 11's setup sheet — not measurements.
* The servo `MAX_JOINT_VEL_RAD_S = 3.2 rad/s`, `MAX_JOINT_ACCEL_RAD_S2 = 26 rad/s²`,
  `GRIPPER_MAX_*` envelopes and `MAX_GRIP_FORCE_N = 3.5 N` remain assumed values, as
  they were on arrival.
* The LeIsaac bridge stays render-only; it still consumes the twin's poses.
* Placement error is now measured against the **commanded slot**. The distance from
  the container's centre is a property of the packing (two parts in a 62 mm cavity
  sit 24 mm from the middle), not a measure of how well the robot aimed — reporting
  it as "error" made a perfect placement look like 24 mm of slop.
