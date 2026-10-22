# The grip, and where the part is released

Two questions demo 10 answered by never asking them, because its executor picked parts up by
writing their pose from the gripper frame whenever friction failed. Both are answerable, and the
answers are what demo 11 is built on.

Every number below was measured on this machine, in this scene, with demo 11's own physics -
`real_world.mass_kg` for the masses and the materials' own friction and restitution - by
`tools/probe_grip_hold.py`, which drives the real pick cycle (above, down, close, lift) onto a
part at a declared position with no camera and no perception involved.

    python tools/probe_grip_hold.py --headless --part cube
    python tools/probe_grip_hold.py --headless --part ball

## 1. The jaws close on nothing, and that is why nothing was ever held

`probe_grip_hold.py`, 30 mm cube (28.1 g, matte ABS), following = how much of the 100 mm lift
the part made. `executor.GRASP_FOLLOW_MIN` is 0.8.

| close command | gripper stiffness | follow | verdict |
|---|---|---|---|
| 0.20 rad | 17.8 | 100% | held |
| 0.20 rad | 80 | 100% | held |
| 0.20 rad | 300 | 100% | held |
| 0.30 rad | 17.8 | 0% | not held |
| 0.30 rad | 80 | 0% | not held |
| 0.30 rad | 300 | 0% | not held |
| 0.40 rad (LeIsaac's default) | 17.8 | 0% | not held |
| 0.40 rad | 80 | 0% | not held |
| 0.40 rad | 300 | 0% | not held |

30 mm ball (9.9 g, hollow PET), same sweep:

| close command | stiffness | follow | verdict |
|---|---|---|---|
| 0.20 rad | 17.8 | 79% | under the 80% threshold |
| 0.20 rad | 80 | 72% | under the threshold |
| 0.20 rad | 300 | 94% | held |
| 0.30 rad | any | - | knocked out of the jaws ("lost") |
| 0.40 rad | 17.8 | 57% | not held |
| 0.40 rad | 80 | 83% | held |
| 0.40 rad | 300 | 83% | held |

**LeIsaac's shipped close target is 0.4 rad**, and it exists because upstream drops an 8 cm
orange. On this scene it holds a 30 mm cube **0%** of the time. At 0.20 rad the same cube is held
outright at every stiffness tried.

Two consequences, and they are the reason demo 11 exists:

* **`real_world.GRIPPER.close_command_rad = 0.20`.** Applied in `11_real_sorting_with_monitor.set_gripper`
  after the environment is built, because the action term owns the number and re-reads it every
  step. Override with `--close-command`.
* **`real_world.GRIPPER.actuator_stiffness = 250`.** A sphere meets the pads at a point and a cube
  at a face, so the lighter part needs the stiffer servo to be carried at all: at the shipped 17.8
  the ball follows 79% of the lift, at 250 it follows 94%. Set in the task config by
  `set_gripper_actuator`, **before** `gym.make`, and overridable with `--grip-stiffness`.

The measured finger-body separation is 36.2 mm at every command, which is `probe_gripper.py`'s
finding rather than this one's: the finger bodies rotate about their mounts and their origins do
not move, so the close command sets how far the pads swing in, not how wide they end up.

**Nothing in demo 10 could have found this.** Its `Executor.grab` writes the part's pose from the
gripper frame whenever the lift test says the jaws did not hold it, so every one of the rows above
that says "not held" would have been reported as a successful pick. Demo 11 refuses instead
(`StrictExecutor.grab`), which is why it found the problem on its first run.

### The jaws also need longer than the executor gives them

`executor.STEP_COUNTS` spends **50 steps (0.83 s)** on `close`; this probe used 90. That
difference decides the grasp on its own: with 50 the lift is judged while the pads are still
closing, the part has not been squeezed, and demo 11 refuses the grasp it would have got. Demo 11
therefore sets `CLOSE_SETTLE_STEPS = 90` on its own `close` step and leaves demo 10's table alone.

With 90 steps and the 0.20 rad command, `pon el cubo rojo en la caja roja` completes and verifies:

```
above    red: above #3  [aimed at (+0.289, -0.441), re-aimed 2x (largest 11 mm)]
down     red: down onto #3  [aimed at (+0.289, -0.441)]
close    red: close on #3
lift     red: lift #3  [followed 99% of the lift (74.4 of 75.0 mm)]
carry    red: carry to #0
lower    red: lower over #0
release  red: release into #0
retreat  red: retreat from #0
camera sees 2 object(s) and 2 container(s); 1/1 placement(s) in a tray
placements verified : 1      grasps carried by hand : 0
```

**99% of the lift, carried by friction, placed and verified - with nothing written to the part's
pose.** That is the first placement in this repository that a real SO-101 could reproduce.

### The binding constraint is centring, not force

A 30 mm part between pads 36.2 mm apart has **3.1 mm of lateral slack in each direction**. The
aim error is larger than that:

| part | where it sits | aim | outcome |
|---|---|---|---|
| `cube` (red) | x 0.281 | (+0.289, -0.441) | held, 99% of the lift |
| `cube_blue` | x 0.445 | (+0.445, -0.441) | refused, three attempts, never held |

Same size, same material, same jaws, same grasp pose height - opposite outcomes, and the only
thing that differs is where on the table the part is. The pads are rubber and can wrap a corner,
which is why an 8 mm error sometimes still holds, but it is luck rather than a budget: at this
camera the localisation residual is 4-6 mm per part and the systematic y offset is ~7-10 mm near
the far row, both of which are outside the 3.1 mm the geometry allows.

This is the number a bench needs, and it is not about the simulator:

* **re-measure the aiming correction for your own camera** (`scene_spec.AIM_CORRECTION_*` are fitted
  to this render) - see item 4 on the setup sheet;
* **or use a wider aperture**. 36.2 mm is what these jaws give; a 40 mm aperture would put 5 mm of
  slack either side of a 30 mm part, which is the same order as the localisation error;
* **or grip the part's faces deliberately** - a parallel jaw that closes on a 30 mm cube from a
  known orientation removes the lateral error from the budget entirely.

What demo 11 will not do is close the jaws, find the part did not follow, and write its pose from
the gripper frame anyway - which is what made this invisible for as long as it was.

## 5. The grasp envelope: found, then fixed by servoing the tool centre

### What was wrong with the first two explanations

**The measurement was an assumption.** The jaws-vs-command error was computed from
`executor.HOLD` - 7 mm forward and 92 mm below the gripper body - rotated into the world by the
wrist's orientation. That is a *belief* about where the fingers are, and it reported 13-18 mm at the
far end of the table. The articulation has one body named `gripper` and one named `jaw`, and the
midpoint of their origins is the aperture centre by definition: measured, no assumption.

**The correction was applied to the wrong quantity.** `executor.track` re-reads the camera every 20
steps and points the commanded pose at the part. Correcting the *aim* was therefore undone on the
next tick - which is exactly what the log showed, and why the first attempt changed nothing.

### What works: correct the command, inside the loop

`StrictExecutor.track` replaces the aim-on-the-part rule with a servo on the tool:

    command += (where the camera says the part is) - (where the aperture actually is)

every tick the camera is consulted, capped at `2 x LATERAL_SLACK_M` per correction so a
full-gain swing cannot throw the arm. This is a tool-centre-point correction using the arm's own
feedback, and a fixed offset between command and reality cancels on the next tick instead of being
fought. Two guards keep it honest: a shift past `TRACK_MAX_SHIFT_M` is still counted as lost (that
is the part being knocked away), and if the residual stops shrinking over `STALL_WINDOW`
corrections the servo stops and says the pose is unreachable - because an arm at a joint limit
obeys every command and goes nowhere, which is a different fact with a different fix.

Measured, before and after:

| | before | after |
|---|---|---|
| `cube_blue` at x 0.44, single command | refused 3/3 | **held, 99% of the lift, placed, verified** |
| `separa las piezas por colores`, seed 0 | **0 of 3** sorted, 5 refusals | **2 of 3** sorted, 2 verified, 3 refusals |
| same, seed 3 | - | **2 of 3** sorted, 2 verified, 1 refusal |
| parts carried by hand | 0 | **0** |

The hover still reports the honest thing: *"the aperture is still 17.1 mm from the part after 6
correction(s), so this pose is outside what the arm can hold it at"*. The arm cannot reach the
commanded pose from park - and the servo still gets the descent close enough to grip, because it is
correcting continuously rather than once.

### What is still missing

One object in three still needs a retry. The residue is not centring - it is that a marginal grasp
sometimes does not take on the first approach at a pose the arm reaches awkwardly. The remaining
levers, in the order worth trying on a bench:

1. **Proper TCP calibration**, applied to the commanded pose inside the loop, from residuals
   measured at several poses rather than closed on the fly. The servo above is a poor man's version
   and it works; a fitted table is the real thing.
2. **A wider-aperture gripper.** 36.2 mm gives 3.1 mm of slack on a 30 mm part; 40 mm would give
   5 mm, which is the same order as the residual near the middle of the table.
3. **Centre the work on the robot.** The SO-101's base is at x 0.35 while the declared working area
   runs 0.27..0.52, so the far end of the table is 170 mm to one side of the arm. A workspace
   symmetric about the base would put every part inside the envelope the arm actually has, and would
   also remove the need for a 190 mm-wide layout in a 90 mm-wide envelope.

### One caveat, stated because it matters for a bench

The force the pads actually apply cannot be read out of this simulation. `force = stiffness x
deflection` needs the deflection, and the part - not the servo - is what stops the jaws: at a
0.20 rad command the gripper joint rests at 0.201-0.207 rad, i.e. at or past the command. So
`real_world.GRIPPER.max_grip_force_n = 20` is a **declared** figure and the friction-cone margins in
`real_world.setup_sheet` are indicative. Measuring the real number is item 2 on the setup sheet.

## 2. Releasing at rim + 12 mm drives the jaws into the tray

`planner.place_pose` releases with the part's centre 12 mm above the tray rim. Where the fingers
are at that moment:

    HOLD.z                       -0.092 m   a held part's centre sits 92 mm below the gripper body
    finger tip below the part    -0.032 m   assumed (setup sheet item 3)
    ------------------------------------------------------------
    fingertip z at release       = rim + 12 - 32 mm = rim - 20 mm

The fingertips are **20 mm below the rim** while the jaws are still closed around the part, on a
tray whose wall is 50 mm tall and whose floor slab is 6 mm. The arm is dragging the part along
the inside of the cavity on the way down, which is exactly demo 10's observed failure: a ball
reported as "30.3 mm from the resting height" and sitting near the cavity edge, i.e. perched
against the inner wall rather than resting on the floor.

`real_world.safe_release_z(tray)` is the height a real arm can use:

    release z = rim + finger tips + 6 mm clearance = 0.0909 + 0.032 + 0.006 = 0.129 m
    fall to the cavity floor                                       = 67 mm

`--release clear` (the default) raises every `lower` and `release` step to that height. It is
129 mm, which is 2 mm below demo 10's own travel height of 130.9 mm - so on this tray the part is
effectively carried to the destination and dropped, rather than lowered in. That is the honest
consequence of a 50 mm rim and fingers that hang 32 mm below the part: **you cannot lower a part
into this tray.** The tray is the wrong shape for this arm, which is why
`real_world.recommended_tray()` exists - a 70 mm cavity and a 30 mm rim, which leaves 7.5 mm of
placement margin instead of 0.5 mm and shortens the fall to about 47 mm.

## 3. The placement margin on the simulated tray is 0.5 mm

    cavity half-width        28.0 mm
    part half-width          15.0 mm
    camera localisation error 12.5 mm   (worst of 120 blobs over the 20 recorded frames)
    ------------------------------------------------------------------
    margin                     0.5 mm

`real_world.MIN_PLACEMENT_MARGIN_M` is 5 mm, so `preflight` reports this as a **failure** for every
part-tray pair, and names the tray that would fix it. This is not a complaint about the simulator;
it is the number a person building the bench needs, and it is the same number demo 10's recovery
loop is quietly working around by re-dropping misses.

## 4. Isaac faulting at start-up on this machine, and what it was

A run that dies silently right after Kit comes up looks exactly like a simulator fault, and for a
while that is what this was taken to be. It was not, in the end. Three separate things, in the
order they turned up.

### 4a. A configclass is not subscriptable - this was the real blocker

`set_gripper_actuator` did `cfg.scene["robot"]`. IsaacLab's scene config is a `configclass`, which
takes `setattr` - the same reason `isaac_io.build_env` uses `setattr(cfg.scene, term, None)`. So
every launch raised

```
TypeError: 'LiftCubeSceneCfg' object is not subscriptable
```

a few seconds after "starting Isaac Sim", and the process exited. From the outside that is
indistinguishable from a fault: no round, no error message the reader is looking for, and a log
that ends at the same place every time. Eleven launches were spent on this before the traceback was
read end to end.

The lesson worth keeping: **read the whole log, including the traceback frames.** The Kit banner
above the crash is identical in a healthy run and a broken one.

### 4b. h5py's HDF5 and Isaac's HDF5 are the same filename

`demos/01_simulation_basics/env_loader.py` imports h5py *before* Kit so Isaac's copy does not
shadow it - which is correct, and it is why this appears in every run's log as

```
Could not load the dynamic library ... isaacsim.sensors.rtx .../generic_mo_io.dll
```

Both ship a DLL called `hdf5.dll`: h5py's is HDF5 2.0.0 (h5py 3.16), Isaac's is the 1.14-era
library its own `hdf5_cpp.dll` links against. Windows resolves an already-loaded module by base
name, so whoever arrives first wins for the whole process and the other can never load. Preloading
Isaac's four HDF5 modules with `RTLD_GLOBAL` does clear that error - and then
`isaacsim.sensors.rtx` loads for the first time, imports torch during Kit's start-up, and
IsaacLab's later `import torch._prims` fails with *"Only a single TORCH_LIBRARY can be used to
register the namespace prims"*.

So the clash is real and provable, and it is **not** what was killing the runs: the working
configuration is the one demo 10 already used, where that extension simply never loads. Nothing in
this demo's path needs h5py - all three references in IsaacLab and the LeIsaac task are in dataset
*writing*, and every run here sets `cfg.recorders = None` - but the working order was left alone
rather than "fixed" into a worse one.

### 4c. A real, intermittent fault, handled by relaunching

With 4a fixed, launches get into a round most of the time. When they do not, the signature is:

```
Windows fatal exception: code 0xc0000139      (STATUS_ENTRYPOINT_NOT_FOUND)
Windows fatal exception: access violation
Thread 0x... (most recent call first):
  <no Python frame>
```

i.e. inside Kit's native extension loader, on a thread with no Python frame - not in any Python
code - sometimes leaving a process that produces no further output at all. Roughly one launch in
four was affected across this session, for demo 10 and demo 11 alike. Clearing Kit's caches
(`%LOCALAPPDATA%\ov\cache`, 637 MB of shaders, DerivedDataCache and texturecache) changed nothing,
and neither did excluding `isaacsim.sensors.rtx` and `omni.sensors.nv.lidar`: `SimulationApp`
forwards `--/app/extensions/disabled=` to carb, and both extensions still started.

Since it is in the vendor's loader, the useful response is not to wait for it:

```
python tools/run_demo11.py -- --headless --text "separa las piezas por colores"
```

`tools/run_demo11.py` launches the demo, watches for progress, and relaunches on a fault or a
stall, printing which signature it saw. Each attempt costs ~40 s when it fails and about four
minutes when it works, and the demo itself has nothing to do with the difference.