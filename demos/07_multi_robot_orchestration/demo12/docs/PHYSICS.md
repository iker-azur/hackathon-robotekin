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