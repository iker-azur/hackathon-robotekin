# Demo 12 — handoff brief for a coding agent

## What Demo 12 is

Multi-robot orchestration on **real Isaac Lab / Isaac Sim 5.1**. Several SO-101 arms share
a bench, pick parts and sort them into trays. The user-facing workflow is:

**CONFIGURE ENVIRONMENT → LAUNCH ISAAC LAB → EXECUTE MULTI-ROBOT TASK → OBSERVE RESULTS**

Isaac Sim is the primary observation surface. The web UI is deliberately minimal.

### Non-negotiable requirements

1. **Real Isaac Lab only.** No mocks, no simulated test modes, no artificial execution
   results, no hardcoded success states, no debug panels.
2. **Physically correct.** Real masses, dimensions, inertia, friction. Natural resting.
   Reliable physical grasping. No teleporting a part into a grip, no scripted
   object-follow constraint. Robot-robot collision avoidance. Document every approximation
   and every measured constant.
3. **Completion is verified from physical state** — the part must actually be in the tray.
4. **Simple over complex; correct physics over visual effects; stability over features;
   maintainability over patches.** Clear separation: frontend / orchestration / simulation.
5. **Orchestration reads real sim events.** Independent and collaborative tasks, correct
   shared-workspace handling, no robot interference, clear errors, safe recovery.

### The old implementation must be deleted

`warehouse_sim.py` (1,840), `robot_driver.py` (891), `perception.py` (355),
`planner.py` (541), `hackathon_scenario.py` (249), `telemetry.py` (255),
`isaac_multi_backend.py` (445), the old `orchestrator.py` (1,482), `server.py`, the old
`12_multi_robot_orchestration.py` entrypoint, `static/index.html` and the old `tests/` are
**fake physics** — they exist to be replaced, not extended. The old demo still runs; do not
assume it validates anything.

---

## Environment (this machine)

| | |
|---|---|
| Isaac Lab | `C:\Users\Usuario\Documents\Workspace\IsaacLab` |
| Isaac Sim | 5.1, LeIsaac 0.4.0 |
| Python | `C:\Users\Usuario\miniconda3\envs\hackathon-robotekin\python.exe` (3.11) |
| `conda` | **not on PATH** — call the python by absolute path |

**Booting headless (the only working way on this machine):**

```python
import sys
sys.path.insert(0, r"...\LeRobot_\demos\01_simulation_basics")
import env_loader; env_loader.prepare_runtime(headless=True)   # Kit bootstrap
```

`prepare_runtime` already handles the DLL exclusions, EULA and derived experience. **Do not
duplicate it.**

**Exiting:** call `os._exit(rc)` when done. Kit hangs on normal return and on exceptions.

**Cost:** each Isaac start is ~10–40 s. Batch your measurements into one run — a probe that
prints a whole table beats a probe that prints one number per launch.

---

## Current state

`demos/07_multi_robot_orchestration/demo12/` — library code only. **No runner, no
orchestration, no server, no frontend, no docs, no working grasp.**

| File | Lines | Role |
|---|---|---|
| `materials.py` | 543 | Physical constants, each labelled S1/S2/S3 by how it was obtained |
| `config.py` | 693 | Scene model + validation, runs without a GPU |
| `isaac_env.py` | 1,705 | Isaac Lab scene — the sole physics authority |
| `robot.py` | 347 | `ArmController`: DLS IK, joint commands, state reads |

### Verified working in real Isaac Sim

- Exact asset masses. SO-101 follower: 0.632 kg total (base 0.147, shoulder 0.100,
  upper_arm 0.103, lower_arm 0.104, wrist 0.079, gripper 0.087, jaw 0.012).
- **0.00 mm** resting error; **0.00 mm** drift over 5 s.
- IK converges to **0.49 mm**; pad tracking to **0.5 mm**.
- Parts rest undisturbed on approach (`linSpeed = 0.0000`).
- **Finger linkage acts and sweeps** — see below.
- `default_scene(1|2|3)` validates with no errors.

### The finger mechanism (new, working)

The gripper is a **two-prismatic-joint linkage**: two pad rigid bodies, each joined to the
jaw by a prismatic joint along the jaw's local Y.

```
joints: 8  [... 'gripper', 'pad_L_slide', 'pad_R_slide']
bodies: 9  [... 'jaw', 'pad_L', 'pad_R']
open    joint = [9.99, 9.97] mm   span 22.0 mm
closed  joint = [0.0, 0.0] mm     span 12.0 mm
```

`IsaacCell.set_fingers(robot_id, open_)` is the gripper interface;
`ArmController.set_jaws(open_)` delegates to it, so callers are unchanged.

**Two things that took real work to find:**

- Isaac Lab *does* pick up an authored prismatic joint as a first-class articulation joint —
  but `set_joint_position_target` on it did nothing (stayed at exactly `-0.00000` for 180
  ticks). Isaac Lab processes `ArticulationCfg.actuators` when the `Articulation` is
  *constructed*, and the joint didn't exist yet. Writing gains straight to the PhysX view
  afterwards changes nothing either, because `apply_actuator_model` only writes forces for
  joints with a registered actuator. **Fix in place:** `_rebuild_with_linkage_actuators()`
  constructs a second `Articulation` with an `ImplicitActuatorCfg` covering
  `pad_L_slide`/`pad_R_slide`. Joint and body ids are resolved in
  `_capture_robot_frames`, *not* in that rebuild — a fresh `Articulation` has no PhysX
  views until `sim.reset()`.
- A USD `DriveAPI` needs `maxForce`. It defaults to 0, i.e. a drive that applies no force.

---

## The immediate blocker

**Adding two joints broke the controller's joint indexing, so the arm no longer tracks.**
The tool lands ~130 mm away in x/y. `robot.py` has 6-wide assumptions where the
articulation is now 8-wide — notably `ArmController.joints` returns the full
`joint_pos[0]` row, and the home pose / `GRIPPER_*_RAD` constants assume 6 joints.

Fix this first, then re-run a grasp-and-lift probe and confirm:

1. arm tracks to < 1 mm at the grasp pose
2. `pad_span_m` open ≥ 36 mm, shut < 18 mm
3. `grip_force_n` > 0 on a part
4. the part rises with the jaw and stays in the tray

---

## Four gripper designs already tried and why they failed

This is the expensive part. Do not repeat any of it.

**Root cause of the original impossibility.** `so101_follower.usd` models the gripper's
collision as **one solid 22.3 × 48 mm slab**: `/colliders/jaw/moving_jaw_so101_v1/mesh` has
84,810 points, local x `-12.30..+10.00 mm`, and 14,886 of them within 4 mm of x=0 spanning
the full z range. A slab cannot straddle anything — it presses down on top. The 14
`so101_follower_demo11_pads_*` variants change only *visual* meshes; all fourteen behave
identically.

| Design | Result |
|---|---|
| Pads **rigidly parented to the jaw** | **Impossible, by proof.** A revolute preserves every point's distance from its axis and turns every point by the same angle, so the distance between any two points on a rigid body is invariant. Measured 42.74 mm at every angle 0.2→1.4 rad, constant to the millimetre. (An earlier derivation claiming `2R·cos θ` was wrong — that describes the *direction* the pads point, not the distance between them.) |
| Pads as **standalone kinematic bodies** | Infinite mass, so no normal force: nothing to squeeze with. Closing across a part means 6 mm of interference — the part is fired out of the grip at **0.62 m/s**. |
| Pads as **standalone dynamic bodies, velocity servo** | Track to 0.5 mm against a 10 mm target, but a velocity command every tick drives them *through* the contact they are gripping. Contact never develops; sensor reads **0.00 N**. |
| Pads as **standalone dynamic bodies, force servo** | Track to 1.70 mm, then **stop being simulated**. Not the force term, not a pose write, and not `sleepThreshold` (swept −1 → 1000, changed the frozen state by exactly zero). Only a velocity write revives a pad — and that is the thing that breaks contact. |

Other measured findings worth keeping:

- Kinematic pads spawned at the world origin (inside the bench) are flung by the solver:
  descents landed 0.09–0.46 m for commands spanning 32 mm.
- Pads centred 10.4 mm below the jaw pivot reach *backwards* into the wrist: **84,975 N** on
  the upper arm.
- Pad tops landing on a part's top corners shove it 100 mm and spin it 51°.
- Pad yaw following `shoulder_pan` launches the part at 1.5–1.7 m/s — a 30 mm cube has a
  21.2 mm half-diagonal against an 18.1 mm pad face.
- A resting pad is **held down by PhysX** and a servo force cannot lift it: pads pinned 7 mm
  below target for 90 ticks while the servo applied +0.3475 N (3× the pad's weight).
- The `gripper` joint rotates about the jaw's **local Z** — measured, by differencing the
  jaw quaternion across a joint sweep: `(-0.0000, -0.0000, +1.0000)` at every sample.
- At the grasp pose the jaw's local axes are X → world +Z (up), Y → world −Y (horizontal),
  Z → world +X. So the pads close sideways while staying level.
- Isaac Lab has **no `disable_sleep`** on `RigidBodyPropertiesCfg`; `sleep_threshold=0.0` is
  the PhysX way to say never sleep (and it did not help).
- `ContactSensor` must point at a **rigid body**. A pad that is a collision child of the jaw
  fails with "did not match any rigid bodies".
- Isaac Lab disables contact processing during `SimulationContext` setup; without
  `sim.set_setting("/physics/disableContactProcessing", False)` the contacts are real and
  unreported.

---

## Scene / build-order rules already established

- Tune `/physicsScene` → spawn assets → bind materials and contact reporters → build contact
  sensors → `sim.reset()` → `sim.forward()` → re-apply actuator gains → capture frames →
  build arms. Binding colliders after reset, or tuning after spawning, invalidates the
  PhysX tensor views ("Failed to get root link transforms from backend").
- The SO-101 USD ships **zero** `PhysicsDrive` prims; drives must come from
  `UsdFileCfg.joint_drive_props` / actuator configs.
- With a bare `SimulationContext` there is no env loop, so each control tick must call
  `art.write_data_to_sim()` then step, then `sim.forward()` and `art.update(dt)` — otherwise
  every cached read is stale and the orchestration reads a world that has not moved.
- Joint drive gains are written by `_process_actuators_cfg` before `sim.reset()` and are
  **discarded** by it. Re-apply afterwards.
- At zero joint angles the arm points −Y, so robots need `base_yaw_deg = 180`.
- `demos/01_simulation_basics/env_loader.py` is the only working Kit bootstrap on this
  machine.
- `demos/05_planner_and_executor/real_world.py` is where the measured material constants
  came from.

---

## Known costs to disclose, not hide

- `ARM_BODIES_WITH_COLLISION = ()` — the arm's own links have collision **off**. Keeping them
  collidable while pads are kinematic and teleported into the wrist is unstable, and dropping
  them kept descent at 0.3 mm. The arm therefore cannot be stopped by the bench. This needs
  a PhysX collision-group filter to restore, and `physxScene:collisionGroups` is a
  token-array *relationship* that `pxr.Usd.Relationship` cannot write (only path targets).
  If the new linkage is stable, **try re-enabling arm collision first** — it may no longer
  be needed.
- The pads are axis-aligned with the world rather than following wrist roll.
- `wrist_roll` is held at zero.
- The linkage's actuator gains (kp 4000, kd 80) are **tuning, stated as tuning** — the asset
  models no finger actuator.

---

## What to build, in order

1. **Fix the 8-joint indexing** and confirm the grasp-and-lift probe passes end to end.
2. **Try re-enabling arm collision** (`ARM_BODIES_WITH_COLLISION`). If stable, that removes
   the largest disclosed approximation.
3. **`orchestrator.py`** — task state machine that reads real sim state only. Never write a
   pose into the sim; only joint targets, as a real controller would.
4. **`worker.py`** — Isaac Sim process owning Kit.
5. **`server.py`** — FastAPI, REST + WebSocket.
6. **`web/`** — Vite + React + TypeScript. The user chose this over zero-build ES modules and
   FastAPI+HTMX. Deliberately minimal; structurally unlike the old UI.
7. **Tests**, then **`docs/`** including `docs/PHYSICS.md` recording every constant's
   provenance and every approximation.
8. **Delete the superseded engine** listed above.

## Definition of done

A user can launch the demo, configure a 1–3 robot scene, run a multi-robot task, and watch
Isaac Sim; every task completion is verified from the part's actual physical state; and any
constant that is not a measurement is labelled as an approximation in both code and docs.