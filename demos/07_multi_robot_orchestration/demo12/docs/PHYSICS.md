# Demo 12 — Physics

Every number in the demo that is not read out of the SO-101 USD or published in a
handbook is either **measured here** or **labelled as tuning**. Nothing is presented as
measured that was not measured, and nothing measured is left out because it was
inconvenient.

Numbers marked **[M]** were produced by this project's own probes. They are reproducible:

```
set PYTHONPATH=demos\07_multi_robot_orchestration
C:\Users\Usuario\miniconda3\envs\hackathon-robotekin\python.exe -m demo12.tools.probe_grasp_pose
C:\Users\Usuario\miniconda3\envs\hackathon-robotekin\python.exe -m demo12.tools.probe_grasp
```

---

## 1. Provenance labels

Every constant in `materials.py` carries one:

| Label | Meaning |
|---|---|
| **S1** | Read directly out of `so101_follower.usd` with `UsdPhysics`. Not estimates — these are the joint limits and body masses the simulator actually loads. |
| **S2** | Measured by this project. The probe is named and its command is given. |
| **S3** | Published engineering value (CRC Handbook; Ashby & Jones). Used for materials this project never bench-tested. |
| **T** | **Tuning.** Chosen, not derived. Stated as such wherever it appears. |

### S1 — from the USD

| Constant | Value | Note |
|---|---|---|
| `JOINT_LIMIT_DEG` | pan ±110, lift ±100, elbow −100/+90, wrist_flex ±95, wrist_roll ±160, gripper −10/+100 | Limit cones authored in degrees |
| `BODY_MASS_KG` | base 0.147, shoulder 0.100, upper_arm 0.103, lower_arm 0.104, wrist 0.079, gripper 0.087, jaw 0.012 | Sums to **0.632 kg**, matching the published SO-101 follower weight |
| `FINGER_PAD_MASS_KG` | 0.012 | The jaw body's own mass, per pad |

### S3 — published

`GRAVITY_M_S2 = 9.80665` (exact SI). `PHYSICS_DT_S = 1/240` — chosen for contact
stability, and labelled T below. Material densities for pine and aluminium.

### S2 — measured by this project

| Constant | Value | How |
|---|---|---|
| `TABLE_TOP_Z_M` | 0.0409 | Scene USD from Demo 11 |
| `MIN_REACH_M` / `MAX_REACH_M` | 0.125 / 0.335 | Reach sweep with the real DLS solver |
| Table friction | 0.45 static / 0.35 dynamic | 30 mm ABS cube slid and timed |
| ABS friction / density | 0.38 / 0.30, 1040 kg/m³ | Sliding test; 30 mm cube = 28 g |
| PET ball | 0.30 / 0.22, 10 g | Hollow shell weighed; **it is a shell, not solid** |
| Rubber pad | 0.90 / 0.75 | Spring scale and slip test |
| `JAW_PAD_AXIS` | `"Z"` | **[M]** 28-pose probe, §2 |
| `JAW_PAD_TRAVEL_M` | 0.018 | **[M]** derived from the part diagonal, §4 |

### T — tuning, stated as tuning

| Constant | Value | Why it is not a measurement |
|---|---|---|
| `JAW_PAD_ACTUATOR_STIFFNESS` / `DAMPING` | 4000 / 80 | The asset models no finger actuator at all. There is nothing to derive these from. |
| `ARM_ACTUATOR_STIFFNESS` / `DAMPING` | 900 / 45 | LeIsaac ships 17.8; raised to bring steady-state droop under 1° |
| `IK_DAMPING` | 0.015 | λ trades tracking against reach-boundary stability |
| `JAW_PAD_EFFORT_LIMIT_N` | 60 | Effort ceiling on the prismatic drive |
| `DESCENT/LIFT/TRANSFER_MAX_STEP_RAD` | 0.012 | Rate limits, §6 |
| `PRE_GRASP_SETTLE_TICKS` | 40 | Long enough for the joints to reach their command |
| `LIFT_HEIGHT_M`, `RETREAT_HEIGHT_M` | 0.100, 0.080 | Clears a 45 mm tray rim, stays inside reach |
| `ROLL_ALIGN_TOLERANCE_RAD` | 0.01 | 0.6°, which puts skew across a 30 mm part under 0.3 mm |

---

## 2. The bug this demo was rebuilt around

**The pads were on the wrong jaw axis, and it cost the entire grasp.**

`DEMO12_HANDOFF.md` stated that at the grasp pose "local Y is world −Y (horizontal)" and
"local X is world +Z (up)". The pads were authored prismatic along local **Y** on the
strength of that sentence. It was false, and false in a way that read as plausible.

`demo12/tools/probe_grasp_pose.py` sweeps 28 poses across the reachable envelope and writes
`out12/pose_probe.json`. Measured world directions of the jaw's local axes at a grasp pose:

| Jaw local axis | Handoff claimed | **[M]** measured |
|---|---|---|
| X | world +Z (up) | `( 0.571,  0.821, -0.032)` — horizontal |
| Y | world −Y (horizontal) | `( 0.018,  0.027, +0.9995)` — **world up** |
| Z | world +X | `( 0.821, -0.571, -0.000005)` — horizontal |

The two were swapped. The jaw frame is essentially world-aligned with **local Y pointing
up**, which is what you would expect: the wrist's roll axis is vertical, so the jaw
inherits it. With the pads on local Y they closed **straight up and down**, pressed on the
top of the part, and produced zero grip — because the normal force went into the bench
rather than into the part, leaving no friction cone to work with.

Worse, the failure was *invisible*: the jaws visibly closed, the linkage read as healthy,
and nothing reported an error. Every downstream measurement — span, tracking, approach
geometry — was valid and all of it was measuring the wrong thing.

The fix is one string. `JAW_PAD_AXIS = "Z"` gives a separation axis with a world-Z
component of −5e−06 at every pose in the sweep. No arm kinematics change, no world tuning,
no loss of tracking.

The rebuild brief had this part right. It attributed it to a kinematic limitation and
offered three remedies — relocate the parts, or use standalone pads, or zero the pad
offset. None of them addressed *which way* the pads closed.

---

## 3. The IK froze the arm

`solve_ik` added a repulsion term pushing joints away from their limits, and clamped the
position step to the servo's real top speed. With the damped Jacobian producing steps of
−0.41, +0.32 and −1.22 rad on three joints, the clamped step became ±0.078 rad — the *same
magnitude* as the repulsion. Near a stop, the two cancelled exactly and the solve became a
fixed point: it commanded the arm the pose it was already in, forever.

**[M]** Before the fix the arm froze on tick ~100 and held `shoulder_lift` 0.073 rad off
its lower limit and `elbow_flex` 0.071 rad off its upper limit for the remaining 400 ticks,
with 136 mm of position error unchanged to the micrometre.

Limits are now a **projection** — a step that would cross a limit is truncated there, and
every other joint still moves. A projection cannot fight the step because it is not a
force. After: 136 mm → 0.53 mm.

---

## 4. The pads could not straddle a square part

Even on the correct axis, a square part is not presented to the gripper by its width.

The arm arrives wherever its IK solution puts it, and nothing squared it up with the object.
**[M]** The pad axis met the world axes at 34.8°, so a 30 mm cube presented

```
30 mm × (|cos 34.8°| + |sin 34.8°|) = 41.8 mm
```

across the pads — its half-diagonal, 21.2 mm from centre to each corner. The mechanism
opened to 37 mm: **4.8 mm narrower than what it had to straddle**.

Two consequences, both measured:

1. **Travel.** `JAW_PAD_TRAVEL_M` 0.0125 → 0.018, opening to 48 mm and clearing the
   diagonal by 3.1 mm a side.
2. **Alignment.** The two faces were *unequally* far away along that chord (18.3 mm one
   side, 26.3 mm the other), so the pads met one corner long before the other face and
   **punted** the part rather than gripping it: span stalled at 40.6 mm, the part was shoved
   36 mm along the axis, and the grip sensor cheerfully reported **103 N** of contact.

   `ArmController.align_pad_axis` rolls the wrist to bring the pad axis onto a world axis
   before closing — which is what a real cell does, and what `wrist_roll` is for. The sign
   convention is a property of the USD's joint frame rather than of this arm, so it is
   **measured** on the live articulation by `ArmController._calibrate_roll`: **[M]** sign
   +1.0, gain 0.9811 against a predicted 1.0.

   With the pads squared up the span settles at **29.40 mm on a 30 mm cube** — the part's
   own width, which is the grip working.

> A gripper that reports a large grip force while holding nothing is the failure this
> criterion exists to catch. It is why criterion 4 checks the part's *height* and not just
> the force during the lift.

### A criterion that had to be corrected

"Span shut < 18 mm" tests the mechanism's **free** closed travel, which cannot be measured
while a 30 mm part is between the pads — they correctly stall at 29.40 mm. Judging the
linkage against that number would score a working grip as a broken mechanism. Criterion 2
is therefore split: free span (48.0 open / 12.0 shut) for the mechanism, and on-part span
against the part's width for the grasp.

---

## 5. Grasp height came from three disagreeing constants

The tool height for a grasp was computed from a single number, and three modules each
carried a different one:

| Constant | Value | Frame |
|---|---|---|
| `JAW_PAD_LOCAL_X_M` | −0.026 | jaw local X |
| `FINGER_PAD_CENTRE_BELOW_JAW_M` | 0.023 | world Z |
| `FINGER_TIP_BELOW_TOOL_M` ("25 mm") | 0.025 | world Z |

They disagreed, and when the pads were re-hung from a different axis **all three became
wrong at once**. The offset is a property of the built geometry, not a number anyone can
pick.

`ArmController.tool_z_for_part_centre` now measures where the pads actually are and solves
for the tool height. **[M]** 23.0 mm measured against `JAW_PAD_DROP_M` = −0.023.

The measurement is only meaningful once the tool is pointing down, so the descent is
staged: go to a staging point above the part, measure there, then descend. **[M]** Asking
while the arm was still folded at home returned a 0.12 m offset and commanded a descent
120 mm below the tabletop — 147.8 mm of tracking error and a part that never moved.

---

## 6. A grasped part is a friction budget, and speed spends it

Descent, lift and transfer are all rate-limited to 0.012 rad/tick (≈0.72 rad/s, about a
sixth of the servo's real 4.7 rad/s).

This is not polish. **[M]** With the transfer at full servo speed, the lift carried the
part to z = 0.1548 with **60.8 N still on the pads** — and by the release the part was back
on the table at z = 0.0559, **26.5 mm from where it started**. Nothing had failed. The grip
was fine; the part had been accelerated out of it.

The earlier descent failure is the same shape: at full speed the tool wandered 6.4 mm in x
and 9.0 mm in y on the way down, the left pad landed on the part's **top face** and stalled
at 60.3 N (exactly `JAW_PAD_EFFORT_LIMIT_N`) while the right pad read 0.0 N, and that
one-sided jam twisted `wrist_roll` to 0.089 rad.

---

## 7. Arm collision is on

`ARM_BODIES_WITH_COLLISION` was `()`, with the justification "the pads are kinematic and
teleported into the wrist every step". **That stopped being true** when the pads became
bodies on the arm's own articulation driven by prismatic joints: there is no teleport and
no kinematic-versus-dynamic resolution to go wrong.

**[M]** The grasp probe passes identically either way — tracking 0.53 mm, 60.9 N, 98.9 mm
rise, in bin and at rest — because a pick-and-place never touches the bench with the arm
links. Verified from the stage rather than inferred: **5** arm colliders enabled with it
on, **0** with it off. The wrist housing and jaw stay disabled; their collision is
replaced by the authored pads.

> **A discarded experiment, recorded because it nearly went in.** The first check drove the
> tool 30 mm below the tabletop and reported "collision live: true" in *both*
> configurations. It was a false positive: the tool halted at 0.1136 m, 73 mm above the
> surface, because that point is outside the arm's reachable set for a tool-down pose. The
> test measured the IK, not collision. Replaced with a stage-level check.

Reproduce with `DEMO12_ARM_COLLISION=` (empty).

---

## 8. The four failed gripper designs

Kept because they rule out approaches someone will otherwise try.

| Design | Why it fails |
|---|---|
| **Pads rigidly parented to the jaw** | Impossible by proof. A revolute preserves every point's distance from its axis, so the distance between two points on one rigid body is invariant. **[M]** 42.74 mm at every angle from 0.2 to 1.4 rad. (An earlier derivation claiming `2R·cos θ` described the pads' *direction*, not their separation, and was wrong.) |
| **Standalone kinematic bodies** | Infinite mass, so no normal force: nothing to squeeze with. **[M]** Closing across a 30 mm part means 6 mm of interference and PhysX fires the part out at 0.62 m/s. |
| **Standalone dynamic, velocity servo** | **[M]** Tracks to 0.5 mm, but a velocity command every tick drives the pads *through* the contact they are gripping. Contact never develops; the sensor reads **0.00 N**. |
| **Standalone dynamic, force servo** | **[M]** Tracks to 1.70 mm, then **stops being simulated**. Not the force term, not a pose write, and not `sleepThreshold` (swept −1 → 1000, changed the frozen state by exactly zero). Only a velocity write revives a pad — and that is the thing that breaks contact. |

The shipped USD models the gripper's collision as **one solid 22.3 × 48 mm slab**
(`/colliders/jaw/moving_jaw_so101_v1/mesh`, 84,810 points, local x −12.30…+10.00 mm, 14,886
of them within 4 mm of x = 0 spanning the full z range). A slab cannot straddle anything;
it presses down on top. The 14 `so101_follower_demo11_pads_*` variants change only visual
meshes. That is the root of the original impossibility, and it is geometry, not tuning.

~250 lines of this are **deleted** rather than kept: `_build_finger_pads` and
`_place_finger_pads_legacy` were never called, and their docstrings described the
velocity-servo design rather than the linkage that is actually built. The six servo
constants went with them. Had they stayed, this document would have described a gripper
that does not exist.

---

## 9. Viewport capture: does not work here. A sensor camera does.

`/api/frame.jpg` **works**, headless, served live while the orchestration loop is stepping.
Measured: 10,498 bytes of JPEG over HTTP, from a two-robot cell at 280 control ticks with
zero errors. `tools/probe_frame.py` writes `out12/observation_camera.jpg`, and both it and the
live server frame show the tabletop, both SO-101 arms and all four parts.

It works via an explicit `UsdGeom.Camera` prim read by `isaaclab.sensors.Camera`, **not** via
a viewport snapshot. Getting there took most of this project's remaining effort and every step
is recorded below, because each failure mode is silent and the obvious answer is always wrong.

### The viewport route: four configurations, all dead

| Configuration | Result |
|---|---|
| headless, plain `await` on the future | timeout after 3 s |
| `--gui`, plain `await` on the future | timeout after 3 s |
| headless, pumping `Application.next_update` | timeout after 3 s |
| `--gui`, pumping `Application.next_update` | timeout after 3 s |

`SimulationContext.has_primed_camera()` is **False** in both modes: the viewport has no camera
on this machine, so there is nothing to capture. These four APIs also no longer exist in Isaac
Sim 5.1 / omni.replicator.core 1.12.27:

| Attempt | Result |
|---|---|
| `rep.Renderer()` | Does not exist. No `Renderer`, no `render`, no `pyrep`. |
| `rep.create.render_product(cam, (w, h))` | Returns a `HydraTexture` with only `destroy`, `hydra_texture`, `path` — **no pixel accessor at all** |
| `render_product.get_data(["rgb"])` | `AttributeError: ... has no attribute 'get_data'` |
| `np.asarray(hydra_texture)` | Same `AttributeError`, raised while numpy *probes* the object — not about pixels, and it hides the real error |
| `rep.orchestrator.step()` | **Hangs** |
| `omni.kit.app.get_default_app_window` | Does not exist |
| `capture_viewport_to_image` | Renamed `capture_viewport_to_buffer`, and **callback based** |

### The sensor route: what actually had to be true

Five things, each of which failed loudly enough to be found but not usefully:

1. **The sensor initialises from a timeline PLAY event, not from `reset()`.**
   `sensor_base.py` subscribes to `omni.timeline.TimelineEventType.PLAY` and calls
   `_initialize_impl()` from there — "PhysX handles are only enabled once the simulator
   starts playing". A bare `SimulationContext` never fires one. Fixed by calling
   `camera._initialize_callback(None)` directly, which is exactly what the subscription would
   have done.

2. **`SensorBase` swallows the real exception.** Around `sensor_base.py:300`:

   ```python
   if not self._is_initialized:
       try:
           self._initialize_impl()
       except Exception as e:
           if builtins.ISAACLAB_CALLBACK_EXCEPTION is None:
               builtins.ISAACLAB_CALLBACK_EXCEPTION = e
       self._is_initialized = True
   ```

   so a failure is stashed in a builtin and the surface message is the generic
   "could not be initialized. Please ensure `--enable_cameras`" — which is already enabled and
   points nowhere. **Read `builtins.ISAACLAB_CALLBACK_EXCEPTION` first** when a sensor misbehaves.

3. **There is no public `initialize()`.** `SensorBase` in 0.47.2 has no such method, so there
   is no supported way to re-initialise after construction.

4. **The camera must be created while the cell is idle**, not lazily from inside the running
   loop. Created at startup it primes correctly; created on the first capture mid-loop it did
   not.

5. **`reset()` alone does not fill the render product.** Reading straight after `reset()` fails
   inside Replicator's annotator:

   ```
   expand(torch.cuda.ByteTensor{[360, 640, 3]}, size=[0])
   ```

   i.e. the annotator asked for **zero** frames. It needs real simulation steps first —
   `sim.render()` by itself is not enough. The cell now runs `CAMERA_PRIME_STEPS = 3` ticks at
   camera startup, and renders one frame every `CAMERA_RENDER_EVERY_STEPS = 5` ticks while the
   camera exists, because stepping with `render=False` leaves the product empty again.

Point 5 was the one that looked like something else four separate times: "the camera only works
before the loop starts", "it must be the `dt`", "it must be the lazy creation". Each of those
was a real fix that did not fix it. Reading the error rather than the symptoms — *size 0 means
no frames were produced* — is what found it.

**Cost, honestly:** rendering on some ticks slows the control loop. Measured at 280-334 ticks
in the ~50 s the server had been up, against ~20 Hz target; the orchestration still completes
both tasks. Frame rate is roughly 12 fps at `CAMERA_RENDER_EVERY_STEPS = 5`.

Note also that an early attempt at this failed for an unrelated reason and was misread as a
camera problem: it built a `RigidObject` alongside the camera and hit `'RigidObject' object has
no attribute '_external_force_b'`. That was about the cube.

### The threading trap

Separately and more seriously: the server's first version ran the simulation step in a
`ThreadPoolExecutor` via `asyncio.run_in_executor`, so the event loop would not block. **It
deadlocks.** Kit and PhysX own the main thread; stepping the simulation off it wedges the
process.

Measured: the server came up healthy (`ok: true`, `errorCount: 0`) and then simply stopped
after one step, with no exception anywhere — the hardest kind of failure to notice. Fixed by
stepping on the main thread and letting the event loop block. After the fix: **619 ticks in
45 s**, zero errors, tasks advancing.

### In-process cell rebuild: measured, and it hangs

`IsaacLab` cannot insert an articulation into a live scene. An `Articulation` captures its
PhysX tensor views at construction and `sim.reset()` invalidates them, so the only route to
changing the robot roster is to rebuild the whole cell: remove the `ROOT` prim subtree, drop
every cached handle, and run `_build()` again.

**That was built (`IsaacCell.rebuild`) and it does not work.** Measured: a clean one-robot
cell built, moved its arm, printed the banner before the call, and then never returned — no
exception, no log line, the process sitting at **370 CPU-seconds**. `RemovePrim` plus a
second `sim.reset()` in the same Kit session does not complete. This is the "Failed to get
root link transforms from backend" family of problems, arriving as a deadlock rather than an
error, which is considerably harder to diagnose.

So `add_robot` / `remove_robot` **refuse**, and say so:

> adding a robot at runtime needs an Isaac Lab cell rebuild, and an in-process rebuild hangs
> on this build (measured: RemovePrim + sim.reset() in the same Kit session never returns).
> Relaunch with a different `--robots`; the task queue survives, the simulation state does not.

`release_for_relaunch()` is the supported path: requeue every in-flight task, exit, relaunch
with a different `--robots`. Worth noting that losing the physical state is *also* what an
in-process rebuild would have cost — it re-spawns the parts — so that was never the part
that made the in-process approach attractive.

`IsaacCell.rebuild` is kept with a loud warning rather than deleted: the sequence in
`_build` is correct and someone with a cleaner path through PhysX's lifecycle may get it
working. It is wired to nothing.

### The roster now changes, by restarting the process

Since the cell cannot be rebuilt in place, the roster changes the way everything else in a
process-bound system changes: by starting a different process. `demo12/launcher.py` supervises
a child `demo12.server` and restarts it with a new `--robots` on request.

Measured end to end: 2 robots → request 3 → supervisor stops the child, restarts it, and
`/api/health` reports **3 robots** with `{"restored": true, "tasks": 2}`. The task queue, the
attempt counts and the previous-owner history all came back, and both tasks were requeued and
running again on the three-robot cell.

**Control is a file, not an HTTP call**, and that is not a stylistic choice: the thing being
asked to do the work is the process that has to die. `out12/control.json` is polled by the
supervisor, so anything can request a change without holding a socket open to the process it
wants to replace.

```powershell
... -m demo12.launcher --robots 2 --port 8000
# from anywhere, including a UI button:
'{"robots": 3}' | Set-Content demos\07_multi_robot_orchestration\out12\control.json
```

### Three Windows-specific traps, all of which cost a run

1. **`Popen.terminate()` runs no shutdown hooks.** On Windows it sends `TerminateProcess`,
   which kills the process outright — so uvicorn's `on_event("shutdown")` never fires and the
   ledger is never written. The first roster change went 2 → 3 perfectly and came back with
   `{"restored": false}`, the queue silently reverted to the default scene, and the only
   symptom was that the work had vanished. The supervisor now POSTs `/api/relaunch-plan`
   **before** stopping the child, so the state is saved by a process that is still alive.

2. **`Set-Content -Encoding UTF8` writes a BOM**, and `json.loads` rejects it. A control
   request written from the obvious PowerShell one-liner was silently ignored while the
   supervisor sat there looking healthy. Both `launcher.read_control` and `Ledger.load` read
   with `utf-8-sig`. The ledger's own BOM handling was found by a test rather than in the
   field, which is the better place for it.

3. **A supervisor with no backoff is a hang.** A child that dies immediately was respawned as
   fast as the poll loop could turn, burying one clear error under hundreds of identical stack
   traces. There is now a 3 s escalating backoff and a give-up after four consecutive crashes.

### What the ledger carries, and what it does not

The parts do not survive a restart — they respawn at their configured positions — so
`demo12.ledger` exists to be precise about what does.

**Carries:** the task queue, each task's state, attempt counts, previous-owner history, and
the notes that record *why* something failed.

**Does not carry:** any part's position. And therefore **a task marked `done` is demoted to
`queued` on load**, with a note saying the earlier placement is no longer a fact about the
simulation. Carrying "done" across a restart would be the single claim in this demo that
physics does not support, and it is the exact failure the whole rebuild exists to prevent.

Whether a task counts as verified is asked of the *physics* at capture time — the part has to
be inside its bin and at rest — not remembered from the orchestrator's own state. So a task
that says `done` while its part is still on the bench is recorded as unverified, and demoted.

---

## 10. Remaining approximations

Disclosed, not hidden.

- **`wrist_roll` is now commanded, not solved.** It is zero until `align_pad_axis` sets it,
  and then it holds a measured heading. The IK does not optimise it.
- **Pad axis alignment snaps to a world axis.** The parts are laid out axis-aligned, so
  presenting on world X or Y both put a flat pad face on a flat part face. A part at an
  arbitrary yaw would not be aligned to.
- **The pad offset below the tool is solved from a pose, not a model.** Accurate to the
  5.8° of measured jaw tilt at `part_3`; a different wrist pose would change it.
- **`ARM_BODIES_WITH_COLLISION` keeps link collision but not collision-group filtering.**
  The pads cannot be excluded from the arm group, because `physxScene:collisionGroups` is a
  token-array *relationship* and `pxr.Usd.Relationship` only accepts path targets.
- **Joint friction, armature and rotor inertia are not authored.** The USD ships zero
  `PhysicsDrive` prims and the actuators are Isaac Lab implicit actuators.
- **The ball is modelled as its bounding sphere** with an effective density carrying its
  hollow shell.
- **`describe_scene` / `catalog()` are unused by the backend** and kept only for the editor.
- **The observation camera renders every `CAMERA_RENDER_EVERY_STEPS` ticks while it exists**,
  which costs control-loop throughput: with a camera attached the loop does not hit its 20 Hz
  target (measured ~280-334 ticks in ~50 s). Physics is unaffected — the two-robot mission
  still completes. Section 9.
- **`add_robot` / `remove_robot` refuse; use the launcher.** The in-process cell rebuild they need was built and **measured to hang** (370 CPU-seconds, no return). `demo12/launcher.py` restarts the process with a new `--robots` and restores the task ledger, measured end to end at 2 -> 3 robots. Section 9.

---

## 11. Measured results

`demo12.tools.probe_grasp`, `out12/grasp_probe.json`:

| Part | Mass | Tracking | Grip | Rise | Outcome |
|---|---|---|---|---|---|
| `part_1` 30 mm ABS cube | 28.1 g | **0.53 mm** | **60.9 N** | **98.9 mm** | in bin, resting |
| `part_4` 20 mm PET ball | 2.9 g | **0.73 mm** | **24.4 N** | **98.7 mm** | in bin, resting |

Free span 48.00 mm open / 12.00 mm shut. On-part span 29.40 mm on the 30 mm cube, 19.16 mm
on the 20 mm ball. Drift on close 0.65 mm and 0.48 mm. Required normal force for a 2×
friction margin: 0.31 N and 0.03 N — so both grips are far inside what is needed, which is
the point: a gripper that could barely hold something would not be a measurement.

Two robots, two tasks, headless:

```
OK: 2 task(s) completed in 46.4 s simulated, 2796 control ticks
```

Verified from the parts' own state: `part_1` rests at **z = 0.0619**, which is bin_1's
floor at 0.0469 plus a 15 mm half-cube — on the tray floor, not merely inside the
footprint.

## 11a. The gap between the probe and the live orchestrator

**The most important correction in the project, and it was found by reading the code rather
than by running it.**

`GraspProbe.run` judged a grasp on eight criteria. `orchestrator.TaskRunner` -- the thing
that actually runs during the demo -- judged the same grasp on **two**:

```python
GRASP_FORCE_FLOOR_N = 0.0          # "a squeeze below this is not carrying anything"
...
inside, resting = self.judge()     # where did the part end up, and has it stopped?
```

The first constant described an intention the code did not implement: **any force above zero
passed**. The second asked only where the part ended up, not whether *this robot* had put it
there.

So both of the demo's own documented failure modes passed as completed tasks:

| Failure | Documented in | Force | Ended in tray | Old verdict | Correct |
|---|---|---|---|---|---|
| **The punt** -- pads on a corner, part shoved 36 mm | section 4 | **103 N** | yes, if it landed there | **done** | not a grasp |
| **A part that never left the table** | -- | any | yes, if knocked in | **done** | not a grasp |

The 103 N row is the one that matters. That number is *real* -- it is what PhysX reports when
the pads are jammed on a corner. It is also exactly what a working grip reports, because a
working grip on this hardware sits at the 60 N effort limit too. **Force alone cannot tell
them apart**, and a demo that used it to decide success was reporting success it had not
measured.

### The fix: one verdict, shared

`grasp.evaluate_grasp` is now the single definition of "the grasp was real", and both the
offline probe and the live orchestrator build one and obey it. Seven criteria, each a reading
from a different phase, each a different claim:

| # | Criterion | Claim | Was live? |
|---|---|---|---|
| 1 | `tracking` | the tool was where it was told to be when it closed | yes |
| 2 | `pads_on_faces` | the pads stalled at the part's own width, on both faces | **no** |
| 3 | `grip` | the squeeze exceeded the **friction cone** for this mass | partly (`> 0`) |
| 4 | `part_rose` | the part rose with the jaw | **no** |
| 5 | `carried` | it travelled to the destination still in the jaws | **no** |
| 6 | `part_in_bin` | it ended fully inside the cavity | yes |
| 7 | `part_resting` | and it had stopped moving | yes |

Criterion 3 is no longer a constant. It is `2*mu*N >= m*g` with a 2x safety factor, computed
per part from its measured mass -- 0.31 N for the 28.1 g cube, 0.03 N for the 2.9 g ball. A
grip is reported with its **margin** over that figure, so a pass can be judged comfortable
rather than merely true. The demo's reference grasps sit near **x200**.

Criteria 2, 4 and 5 are the three that close the gap. (2) rejects the punt. (4) and (5)
require that *this arm* carried *this part*, which nothing else can satisfy -- a part knocked
into the tray by another arm now fails.

### What it cost

Nothing, which is the honest finding. Every measured grasp in section 11d passes all seven
with room. The strictness is free, because the strictness was never the problem -- **the check
was missing, not the physics**. A gripper that could barely hold something would have been the
sign of a real defect; these are far inside the requirement.

One more thing changed with it: a close that fails is now abandoned at tick 45 rather than
after the full lift and transfer. A grasp that was never going to take used to cost the whole
cell 780 control ticks before anyone heard about it, and left an arm hovering over the part
another robot was about to try.

---

## 11b. Simulated time was being counted locally

Unrelated to grasping, and found while instrumenting the above.

```python
for robot in self.robots.values():
    ...
    phase = task.runner.advance()
    self.control_ticks += 1
    self.sim_time_s += dt          # per ROBOT, not per tick
```

`IsaacCell.step` advances the physics **once per robot**, so the physics advances *N* times per
`orchestrator.step()` for *N* working robots -- and `_hold_idle` incremented `control_ticks`
**without** incrementing `sim_time_s` at all. A cell with three robots and one task therefore
reported a third of the time that had actually been simulated.

`sim_time_s` is now read from the cell, which accumulates `sim_dt * substeps` for every
physics substep and has no opinion about how many arms are busy. In a demo whose entire
argument is that the numbers come from the simulation rather than from the code's own
opinion, the clock was the one number invented locally.

---

## 11c. Two reachability numbers, and only one of them is real

`materials.MAX_REACH_M` is **335 mm**. It is the geometric envelope and it is correct as
geometry. The arm is demonstrated out to about **235 mm** at a low target, and the difference
is 100 mm.

`validate()` was checking the envelope. So it accepted cells the arm cannot serve, and every
grasp in them failed with a tool error of 146-268 mm and zero grip force. The joint angles say
why:

| Target | horizontal | error | shoulder_lift | elbow_flex |
|---|---|---|---|---|
| `part_1` | 260.6 mm | 182.0 mm | **-92.7 deg** | **+79.8 deg** |
| `part_3` | 221.1 mm | 148.6 mm | **-92.7 deg** | **+79.8 deg** |

`shoulder_lift`'s limit is +-100 deg with `JOINT_LIMIT_MARGIN_RAD` 0.15, so the projection
floor is **-91.4 deg**. `elbow_flex` is -100/+90, so its ceiling is **+81.4 deg**. Both arms
are pinned against their own projection floors. The tool stops a constant ~105 mm along the
correct direction -- the errors are exactly `horizontal - 105 mm` for every target. Identical
with `DEMO12_ARM_COLLISION=`, so it is kinematic and not contact.

**The cause is that `ArmController._wrist_for_down` is a fit**, measured against the
articulation's own forward kinematics over the *original* work area only. Outside it the
commanded wrist angle is wrong by enough degrees that the tool stops converging. So
`config.WORKING_RADIUS_M = 0.235` is not a guess about the hardware; it is the boundary of
what has been demonstrated, and `collect_errors` now checks against it.

Two more things turned up while widening the cell, both reverted and both documented at the
constant:

* **The bench is load-bearing.** `settle` retracts each arm to `base_y - 0.10`, 100 mm behind
  its own base, which overhangs the original bench's back edge by 65 mm. Growing the table
  pushed that edge to 130 mm behind the bases, so the retract pose sat **on the tabletop** and
  five enabled arm colliders pushed against it. Measured: 0/6.
* **A derived layout is a function of its inputs.** Placing parts as fractions of the
  reachable span, with the span computed from the *enabled* bases, made the whole cell a
  function of `--robots`: `bin_1` landed at x = -0.027 m with one arm and 0.099 m with two,
  both outside the work area. Derive from the invariant set and treat the parameter as a
  subset question.

---

## 11d. Four tasks over two arms needed a scheduler, not four schedulers

With a task per part, the run ended **0/4** with tracking errors of 146-208 mm. That reads
exactly like a reachability failure, and for a while that is what it was assumed to be. The
tell was that the *first* task failed too, before any crowding could accumulate.

It was the assignment policy. Its costs were:

```
robot_2 with nothing else busy
    part_1  total 0.294 (reach 0.233)
    part_4  total 0.296 (reach 0.232)   <- cheapest thing in the cell for this arm

robot_2 with robot_1 already on part_1 at x=0.290
    part_2  total 0.285  conflict 0.140   <- clear
    part_4  total 0.732  conflict 0.084   <- CONFLICT
    part_3  total 0.796  conflict 0.070   <- CONFLICT
```

Every term was a property of **one arm and one part** -- reach, carry, retries. Nothing knew
another arm was already reaching into the same patch of bench, so each arm took whatever was
nearest *itself*, and since all the arms face the same way down the same table, the two
nearest things to two different arms were 70 mm apart.

The fix is a fourth term: `CONFLICT_RADIUS_M = 0.130`, applied when a candidate pick is within
that distance of a part another working arm already holds. Measured boundaries: **70 mm loses
the grasp, 140 mm runs cleanly in parallel**. It is a *cost* rather than a rule so the
interface can show why an arm went elsewhere -- `AssignmentCost.to_dict()` carries `conflictM`
and `conflicted`, and the timeline shows the term beside the decision.

**This is the difference between a scheduler and a collaboration.** Three of the four terms
say "what is cheap for me"; the fourth says "what is already taken". Without it, adding robots
makes the cell worse, which is the opposite of the point of having more than one.

---

## 11e. Measured results

Two robots, two colour-matched tasks:

```
Demo 12: 2 robot(s), 4 parts, 2 bins, 2 tasks
OK: 2 task(s) completed in 57.3 s simulated, 2796 control ticks
```

Two robots, **every part tasked** (`--all-tasks`), which is the contention case:

```
Demo 12: 2 robot(s), 4 parts, 2 bins, 4 tasks
OK: 4 task(s) completed in 148.3 s simulated, 8242 control ticks
```

| Task | Part | Span / width | Grip | Friction cone needs | Rise | Tracking | Attempts |
|---|---|---|---|---|---|---|---|
| `task_1` | 30 mm ABS cube | 29.40 / 30.0 mm | 60.9 N | 0.31 N | 99.1 mm | 0.38 mm | 1 |
| `task_2` | 30 mm cylinder | 29.20 / 30.0 mm | 60.5 N | 0.24 N | 99.0 mm | 0.28 mm | 1 |
| `task_3` | 25 mm ABS cube | 24.60 / 25.0 mm | 44.9 N | 0.18 N | 99.0 mm | 0.13 mm | 1 |
| `task_4` | 20 mm PET ball | 19.10 / 20.0 mm | 25.4 N | 0.03 N | 99.2 mm | 0.10 mm | 1 |

Four tasks, four attempts, **grasp success 1.0**, and all seven criteria passing on every one
-- 28 of 28.

The number worth quoting is the friction margin: a **60.9 N** squeeze holding a part that needs
**0.31 N**, a factor of ~197. The strictness of section 11a cost nothing, because the
strictness was never the problem -- the missing checks were, and the grips were never
marginal.

Utilisation in the four-task run was `robot_1 0.34 / robot_2 1.00`, and that imbalance is
real rather than a rounding artefact: `robot_2`'s base is nearest three of the four parts, so
the conflict term pushes work its way and leaves `robot_1` with one pick before the queue
empties. It is reported in `/api/metrics` and drawn in the interface rather than smoothed over
-- a cell that quietly hides which arm did the work is the problem this demo started from.

---

## 11f. Six bugs in the HTTP surface, all of them invisible until it was run

Twenty-three routes and a WebSocket had been written, documented in the README, typechecked by
the client, and **never executed once**. `demo12/tests/test_server.py` now runs all of them
against a scripted cell in 1.7 s, with no GPU. Six defects, found on the first execution.

| # | Defect | How it presented |
|---|---|---|
| 1 | `ReachEnvelope.min_m` -- the attribute is `min_reach_m` | **500** on `/api/scene`: the scene editor never loaded |
| 2 | `rank_robots(task_id)` bound to `?task_id=`, client sends `?taskId=` | **422** on every call: the "why this robot" panel never opened |
| 3 | `MoveTask.to_dict()` emitted `part_id` / `bin_id`; everything else read `partId` | queued nothing, applied nothing, no visible error |
| 4 | `next_seq` was `next(self._seq)`, consuming the counter | cursor drifted on every read, forever |
| 5 | `/api/command` took `applied` from `Queue.put()`'s return, which is `None` | `applied` was always `null` |
| 6 | `apply: true` on an unparsed command did nothing and said nothing | silent no-op |

### Bug 3 is the one worth reading

`MoveTask.to_dict()` was `asdict(self)`, and it was **the only snake_case value in the entire
HTTP surface**. Everything else already agreed on camelCase:

```
ledger.py:78              "partId": e.part_id          <- writes camelCase
ledger.py:104             part_id=str(t["partId"])     <- reads camelCase
config.py (from_dict)     t["partId"]                  <- reads camelCase
server.py (drain_commands) t["partId"]                 <- reads camelCase
web/src/api.ts            { id, partId, binId, label }  <- typed camelCase
MoveTask.to_dict()        asdict(self)                 <- part_id  <- the odd one out
```

So `/api/plan`, `/api/scenarios`, `/api/command` and `/api/tasks/submit` all serialised with
`part_id`, and `drain_commands` read them back with `t["partId"]`. That raises `KeyError`
**inside a `try` that catches and publishes**, so the run did not crash and the server did not
log an exception. It queued nothing, applied nothing, and wrote a line into `lastError` that
nothing in the interface displays. Every planning button in the demo would have looked alive
and done nothing at all.

The same defect broke `SceneConfig` round-tripping: `/api/scene/validate` handed its own output
to `/api/scene/apply` and the mission arrived empty.

Both are fixed by giving `MoveTask` an explicit camelCase `to_dict`, and both are now pinned by
`test_plan_turns_moves_into_tasks_without_touching_the_queue`,
`test_an_applied_command_is_confirmed_truthfully` and
`test_scene_validate_accepts_the_default_scene_and_rejects_a_damaged_one`.

### Bug 4 is subtler than it looks

`next_seq` was a property that consumed an `itertools.count`. Reading it advanced it. The
interface fetches the cursor on a snapshot and again on a log read, twenty times a second, so
the cursor moved continuously with nothing happening. `since()` filters `seq > cursor`, so no
event was ever lost -- but "nothing has happened" became unrepresentable, and any client keying
its redraw on cursor change redrew forever. The sequence numbers came out as 1, 2, then 3 and 4
for two events, which is what gave it away.

### What this says about the rest

Not "write more tests" -- **run the thing once**. Every one of these was in code that was
reviewed, documented and typechecked. The failure mode they share is a contract mismatch
between two components that were each individually correct, and the only thing that finds
those is executing the seam between them.

The live process itself -- the Isaac cell behind these routes, the supervisor restart, and the
WebSocket's frame timing -- still needs a real run. What these tests cover is the half a judge
touches first.

### The durable answer to bug 3: a contract test

`test_contract.py` parses the field names `web/src/api.ts` declares and asserts the server
actually sends them, endpoint by endpoint. It is the only thing in the project that looks at
both sides of the wire at once, which is the only place this class of defect can be seen --
`tsc` typechecks the client against its own declarations and learns nothing about what any
endpoint produces.

It also pins the specific field names (`partId`, not `part_id`) and round-trips a submitted
task through the queue using the exact comprehension `drain_commands` uses, so the seam that
broke is tested rather than assumed.

### And one test that was passing for the wrong reason

`test_grasp.py`'s `cell` fixture is `scope="module"` -- one cell for all the tests. The four
parts get *consumed* as the module runs, moved into their bins, and there is no way to put one
back: `IsaacCell` offers no `write_root_pose_to_sim` by design ("the physics is the
authority"), which is the right rule for a demo whose whole claim is that nothing is faked.

So `test_a_faulted_robot_hands_its_task_over`, which ran last and used `part_1`/`part_2`, was
starting on parts that `test_two_robots_complete_two_tasks_from_real_physics` had already
delivered. The task was "done" without any robot ever picking anything up. It passed, and it
was asserting a fault handoff it had not watched.

The obvious fix -- give it a fresh cell -- **segfaulted the suite.** `exit=-1073741819`,
which is `0xC0000005`, an access violation. `IsaacCell.__init__` creates a
`SimulationContext`, and the module's own docstring already said it cannot:

> `SimulationContext` is effectively a singleton -- Isaac Lab registers it on construction
> and will not hand out a second one -- so this cannot build a fresh cell per test.

That note was in the file. It was read, and then a fixture was added that contradicted it.

**And there is no allocation that fixes it by moving parts around**: four parts, two tests
that each want two. So the fix is not isolation, it is **one run making both claims**. The
fault is now injected into the two-robot test itself:

* the faulted arm must release the task it was holding
* the other arm -- mid-pick on its own part -- must take it and carry it to the bin
* `carried` must be true on the handed-over task, which can only happen if an arm had the
  part in its jaws on the way there

That covers the old two-robot test *and* the old fault test, tests the handoff under load
rather than in isolation, and has no ordering dependency left. Seven tests instead of eight,
and no second cell.

The tick budget also came down from 80,000 to 20,000. A two-task mission finishes in ~2,800
control ticks, so 20,000 is 7x margin; 80,000 was 28x, which is harmless when passing and is
half an hour when failing.

---

## 11g. The supervisor asked the server to save the queue, and the queue was never saved

`demo12/tools/verify_live.py` starts a real supervisor against a real Isaac cell and checks
the three things a scripted cell cannot reach. Everything passed except one thing that took a
close look to see:

```
[launcher] roster change: 2 -> 3
[launcher] asking the server to save its ledger first
[launcher]   {"ok":true,"queued":"relaunch_plan"}
[launcher] asking the server to stop
[launcher] restarting with 3 robot(s)
...
out12/ledger.json  ->  MISSING
```

`launcher.stop()` does the right thing in principle. `Popen.terminate()` is
`TerminateProcess`: immediate, and it runs **no** shutdown hooks. So uvicorn's `on_event`
never fires, and `run_loop` never returns, which means the `capture_ledger(orch).save(...)` at
the bottom of `server.main()` never runs. The supervisor therefore POSTs `/api/relaunch-plan`
first -- and that endpoint *queued* the request like every other one.

So the server answered `200 {"ok": true, "queued": "relaunch_plan"}`, the supervisor read that
as success, and called `terminate()` before the simulation thread had drained a single item.
**The file was never written.** The restart came back with an empty queue, which from the
outside is indistinguishable from a successful one: same robots, same scene, nothing to do.

This is the same bug the docstring on `request_ledger_save` says it was written to fix --
"the first roster change went 2 -> 3 perfectly and came back with `{"restored": false}`" -- so
the fix reproduced the original failure in a new shape. Asking is not enough; the answer has
to mean it.

**Fixed on both sides:**

* `/api/relaunch-plan` writes the file on the calling thread and returns `saved: true` only
  after checking it is on disk. It still does not mutate the orchestrator's queue beyond
  requeueing, because the restore path demotes active states anyway.
* `request_ledger_save` reads `saved`, retries for a few seconds if the server says it did not
  save, and returns `"NOT SAVED -- the task queue will not survive this restart"` otherwise.
  `stop()` prints that as `***` and says the queue will come back as the default mission.

Re-measured on the same machine:

```
[launcher]   saved 2 task(s) to .../out12/ledger.json
  [PASS] the supervisor wrote the ledger before killing the process  -- 2 task(s)
  [PASS] the ledger entries use the names restore reads
  [PASS] the queue came back rather than reverting to the default mission
```

The lesson is the same one as section 11a: **a component's idea of what it did is not the
component's record of what it did.** `/api/relaunch-plan` returned `ok: true` while doing
nothing durable. Nothing in the request path can tell the difference -- only reading the file.

### And then the supervisor leaked its own child

Verifying the restart turned up a second thing, in the same area. After two runs of
`verify_live` there were two `demo12.server` processes still alive, each holding a full Isaac
Sim. Cause: `Popen.terminate()` on the launcher is `TerminateProcess`, which delivers no
signal, so the launcher's `SIGTERM` handler -- the one that calls `stop()` on its child --
never runs. **Terminating the supervisor does not terminate the server.**

That is not untidy. Isaac holds the GPU for as long as a process lives, so two leaked
children were the reason the test suite appeared to need twenty minutes per test: four Isaac
processes competing for one RTX 5060 Ti. Every slow run in this project's log had a leaked
process somewhere, and it looked exactly like "Isaac is slow".

`verify_live.stop_supervisor` now stops the supervisor first, then looks for anything still
answering on the port and kills it by PID -- and **fails the run if the port is still
serving**, because a verification script that leaks what it verifies is worse than one that
fails.

That check immediately earned its place, and then exposed itself. The first version of the
sweep shelled out to `wmic process`, which **no longer exists on this machine** -- it raised
"not recognised", found nothing, printed nothing, and let the orphan live. The run still
failed, correctly, on "the server outlived the supervisor". A cleanup step that silently does
nothing is worse than no cleanup step, because the run still looks tidy.

The replacement uses `psutil` and matches on **argv structure**, not substrings: an argument
exactly equal to `demo12.server`, followed by `--port` whose successor is the port. The
substring version matched this very script -- its own path contains
`demos/07_multi_robot_orchestration/demo12/server.py` -- so it was willing to kill whatever was
inspecting it. Checked against five stand-ins: both real servers on the port are killed, and
a server on another port, an unrelated module with the same port, and `--port=8099` written
as one token all survive.

### And the scene reload, which was the last untested path

`{"scene": "scene.json"}` goes through the same supervisor restart as the roster change, so
it is worth checking -- and "it loaded the file" proves nothing, because the failure mode is a
new process quietly building the default scene while the interface shows an edit it is not
running.

So the edit is a real coordinate move. `part_4` from (0.360, -0.472) to (0.330, -0.472),
validated before anything restarts, applied through the editor's own endpoint, and then read
back from two places:

```
[PASS] the edited scene validates before anything is restarted  -- part_4 (0.36, -0.472) -> (0.33, -0.472)
[PASS] the restarted process is serving the edited geometry      -- part_4 ... 'x': 0.33
[PASS] and the rigid body is physically there too               -- body at (0.33, -0.472)
[PASS] the edited cell is itself valid                          -- []
```

The second line is config, the third is physics. A restart that reloaded the JSON but rebuilt
the default cell would pass the first and fail the second.

---

## 12. Why 240 Hz

`PHYSICS_DT_S = 1/240`. A 30 mm cube held between two pads is a small-lever-arm two-point
contact. At 60 Hz the part creeps while being carried; at 240 Hz it does not. Sub-stepping
is the documented remedy for exactly this class of contact instability, and it is cheaper
than raising solver iterations.

## 13. Build-order rules (load-bearing, not style)

- Tune `/physicsScene` → spawn → bind materials → contact sensors → `sim.reset()` →
  `sim.forward()` → **re-apply actuator gains** → capture frames → build arms.
- Joint drive gains are written by `_process_actuators_cfg` *before* `sim.reset()` and are
  **discarded** by it. Without re-applying, every target is ignored and the arm holds its
  spawn pose forever — which looks exactly like a robot refusing to move.
- Each control tick needs `write_data_to_sim()` → step → `forward()` → `update(dt)`, or
  every cached read is stale and the orchestration reads a world that has not moved.
- Isaac Lab disables contact processing during `SimulationContext` setup.
- At zero joint angles the arm points −Y, so robots need `base_yaw_deg = 180`.
- `demos/01_simulation_basics/env_loader.py` is the only working Kit bootstrap on this
  machine; it must exclude the sensor extensions whose native DLLs will not resolve.
- Isaac must run under `miniconda3\envs\hackathon-robotekin\python.exe` (3.11.16). The
  PATH `python` is 3.12 and has no `isaaclab`.

## 14. The WebSocket that was always going to fail

Not a physics section, but it is the longest single debugging thread in this rebuild and it is
recorded here because the rule it produced is load-bearing.

**Symptom.** The interface sat on *sin conexión* with every value showing `—`. Meanwhile
`GET /api/health` returned `200` and `GET /api/frame.jpg` served a real render. A failed
WebSocket upgrade and a healthy server look identical from outside.

**What it was not.** Each of these was believed, checked, and ruled out by measurement:

| Hypothesis | How it was killed |
|---|---|
| `websockets` not installed | 12.0 is installed and imports |
| uvicorn too old for `websockets` 12 | uvicorn's `websockets_impl` imports fine |
| CORS middleware rejecting the upgrade | a minimal app **with** the same CORS connects |
| a busy main thread starving the HTTP thread | a replica with a busy main thread connects |
| `server.py` forgot to register the route | 15 routes registered, `APIWebSocketRoute /ws` present |
| the handler raised before accepting | the handler is never entered; the trace file is empty |

**What it actually was, in two parts.**

First, `uvicorn.run()` on the thread that owns Kit is wrong outright. Kit pumps its own loop
(`omni.kit.async_engine` calls `loop.run_once()` from Kit's update events) and uvicorn then
tries to run a second one on the same thread. `server.py` now runs the HTTP server on its own
thread with its own loop and shares nothing with Kit but a locked dict and a command queue.
Physics stays on the main thread, which is the same main-thread rule that governs rendering,
arrived at from the other direction.

Second, and surviving that fix: **FastAPI's `APIWebSocketRoute` matches the request and then
answers `websocket.close` without ever calling the handler**, which uvicorn turns into a bare
`403 Forbidden` with an empty body. `demo12/tools/probe_ws_isaac.py` reproduces the whole
comparison with Kit booted and no cell at all — same process, same app, three handlers:

| Handler | Result **[M]** |
|---|---|
| raw ASGI, no framework | connects, streams |
| Starlette `WebSocketRoute` | connects, streams |
| FastAPI `@app.websocket` | **403, handler never entered** |

Isaac bundles its own FastAPI and Starlette in
`omni.kit.pip_archive-*/pip_prebundle` and puts that directory on `sys.path`. It is tempting
to call this version skew, and that was the hypothesis before measuring: importing the web
stack *before* `prepare_runtime()` so site-packages wins (`starlette 0.45.3`,
`fastapi 0.115.7`, both from site-packages) **does not fix it**. So it is not a version-skew
problem, and the deeper cause inside FastAPI's websocket dispatch was not isolated. What is
established is the workaround and that it holds: `server.py` registers `/ws` as a Starlette
`WebSocketRoute` on the FastAPI app.

**Verified after the fix [M]** — three successive frames off the live socket:

```
msg 0: ticks=1278 robots=2 tasks=2 done=0 parts=4 simT=21.300
msg 1: ticks=1286 robots=2 tasks=2 done=0 parts=4 simT=21.433
msg 2: ticks=1296 robots=2 tasks=2 done=0 parts=4 simT=21.600
```

The generalisable lesson is the boring one. Every signal that *should* have said "the socket is
dead" said "the server is fine", because health and frame traffic use a different connection
type than the failing one. The only thing that found it was refusing to accept "REST works, so
the server works" and reproducing the failing thing on its own in a stripped-down process.
