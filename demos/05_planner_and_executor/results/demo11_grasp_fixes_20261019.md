# Demo 11 grasping: root causes, fixes and strict-physical results (2026-10-19)

Everything below was measured with `tools/benchmark_scene_matrix.py --strict-physical` (top
camera, open variant, 30 mm parts, one target and a red tray; "cluster" adds two same-colour
balls 80 mm to the side). A scenario passes only when the task is `SUCCESS`, at least one
placement is verified, **at least one physical grasp was counted and zero simulator-assisted
grasps** (`capture_passed` enforces the last two; before today it did not).

## Result

Two regions (same target position, per-repetition poses), 10 scenarios each, 2 runs per scenario
on the finished code (seeds 17000 and 17200); "before" is one run of the code at the start of the day:

| shape / layout | before (central, 1 run) | after, central | after, forward_pick |
|---|---|---|---|
| cube single | pass | **2/2** | **2/2** |
| cube cluster | pass | **2/2** | **2/2** |
| ball single | **fail** | **2/2** | **2/2** |
| ball cluster | **fail** | **2/2** | **2/2** |
| cylinder single | **fail** | **2/2** | **2/2** |
| cylinder cluster | pass | **2/2** | **2/2** |
| triangular prism single | fail | **0/2** | **0/2** |
| triangular prism cluster | fail | **0/2** | **0/2** |
| hexagonal prism single | pass | **1/2** | **2/2** |
| hexagonal prism cluster | pass | **2/2** | **2/2** |
| total | 5/10 | **15/20** | **16/20** |

Reports: `baseline_strict_central_20261019.json`, `final2_strict_central_20261019.json`,
`final2_strict_forward_pick_20261019.json` (plus the 90 mm hover A/B,
`hover_ab_0.09_20261019.json`: 10/20). All 31 passes have physical grasps >= 1, simulator-assisted
grasps 0 and a verified placement, and no run ended `SUCCESS` yet was rejected.

Not yet passing, and why (from the logs, not guessed):

* **Triangular prism, 0/4.** The close and the first lift are fine (hold frame +8 mm,
  followed 88-94 % of the lift); the part is then lost: either "friction did not carry it"
  (it sits 4.8-5.2 mm off the pad midpoint, which is a triangle's centroid offset between a flat
  pad and a vertex) or it slips during the 400-step carry. A harder squeeze does not help
  (close 0.10 rad: 0/3; 0.05 rad flicks the part: 0/3) and neither does the close ramp (0, 20, 45
  steps: 0/9). When the wrist is rotated -60 degrees the part is hidden for the whole hover and
  the approach is abandoned (`TRACK_MAX_SCENE_LINKED_MISSES = 6`).
* **Hexagon single, repetition 2 (scene yaw 30 degrees), central region: 1 failure in the final
  run, and 3 of 4 trials failed across the hover variants.** The aperture settles 4.3-4.7 mm from
  the part against a ~3.1 mm gate, twice, and the descent is cancelled before contact. It passed
  in the final forward_pick run, so it is marginal rather than broken.

Not covered: front and side cameras, the contained-tray variant and 20 mm parts were **not run** in
this session. The two regions share the same target position and per-repetition poses, so they are
not independent samples, and 2 runs per cell is a small sample.

## Root causes found

1. **Unreachable ball hover.** The 30 mm ball hovered 150 mm above its centre (90 mm for other
   shapes). `tools/probe_hover_reach.py` shows the pads drift off the grip point as the hover
   rises (aperture error beyond the open jaws' constant 32.7 mm: <= 8 mm at 60 mm, up to 28 mm at
   90 mm, 25-70 mm at 120-150 mm over 0.20-0.30 m from the base). Three independent runs logged
   "pre-grasp hover did not converge: aperture remains ~150 mm". Fix: `planner.hover_reach_limit_m`
   (90 mm within 0.215 m of the base, else 75 mm, the travel plane); every shape hovers alike. A/B
   over the whole matrix: 75 mm 15/20, old 90 mm 10/20 (ball 0/4).
2. **Ball pre-contact pause lifted the arm back up.** It paused "at clearance" at a fixed 0.285 m,
   a height from the old hover. Now capped by where the descent began (`ceiling_z`).
3. **Ball close flicked the ball.** The stock fast close moved it 22 mm in three ticks and the 8 mm
   drift guard aborted the close. The 45-step ramp (previously opt-in) is now the default for the
   30 mm ball.
4. **The open jaw clipped the tray.** The stock opening (pads 70 mm apart) put the moving jaw ~10 mm
   from a tray wall 90 mm away. The jaw was pushed open to 0.9-1.0 rad and stayed wedged while its
   target ramped to 0.2 rad, so nothing was gripped. Signature in the log:
   `closed-jaw hold frame` x about +31..+36 mm instead of +8..+10. Fix:
   `gripper_contact_pads.open_command_for` opens only as wide as the part needs (pad separation
   about 14 + 68 x angle mm), applied on each pick approach. Ball cluster went 0/5 to 4/4.
5. **Detection lost when the gripper hides the part.** The late-descent coast budget was 2 looks;
   a rotated wrist hides the part for the whole descent, so a perfectly centred approach (aperture
   0.2-2.4 mm) was cancelled. The budget is now 20 looks, bounded by the descent. The aperture
   keeps servoing to the scene-linked anchor from the arm's own feedback meanwhile, and a part
   knocked in the dark still fails the close-time reach and drift guards; a longer blackout still
   cancels (tested).
6. **A lone rotated cube was refused as an "unresolved group".** At yaw 30 degrees its oblique
   silhouette is 65 x 74 px (49.6 mm), over the single-part extent limit (1.5 x 30 mm = 67 px).
   It was flagged `ambiguous`, dropped by the catalog linker, and the order was refused with
   "no se identifican todas las piezas" before any motion. `SINGLE_OBJECT_EXTENT_FACTOR` is 1.8 and
   `SINGLE_OBJECT_AREA_ALLOWANCE` 1.25. Replayed over all 109 saved first frames (308 objects):
   308/308 matched, 0 false detections before and after; ambiguous lone parts 2 -> 0.
7. **The harness could not prove a grasp was physical.** `capture_passed` ignored the grasp
   counters, and the unit tests' `check()` only appended to a list that pytest never read.
8. **A physical regrasp was invisible to the report.** Only `Executor.run` counted friction grasps,
   so a retry that gripped (99 % lift follow) and placed the part still reported 0 physical
   grasps and was rejected by the stricter criterion. The command now keeps a cumulative total
   that includes the recovery lift. The same scenario went from `FAIL (SUCCESS)`, 0 physical, to a
   pass with 1.

## Tests added or changed

* `tests/conftest.py`: pytest now fails a test whose `check()` recorded a failure.
* Hover limit across radii and shapes; probe table kept consistent with the planner limit.
* Ball pause ceiling; open-width helper and its apply/restore hook; occlusion budget (coasts a long run,
  still stops a longer one); lone rotated cube fixture (`tests/fixtures/lone_cube_yaw30_top`,
  fails on the old limits); `capture_passed` rejects assisted, non-physical and unproven reports.
* The cumulative physical-grasp counter.
* Results: `pytest tests` 216 passed; `python tests/test_demo11.py` 1572 checks and
  `python tests/test_planner.py` 828 checks, both SUCCESS.

## Part 2 (2026-10-20): why `python 11_real_sorting_with_monitor.py --open --variant contained ...` grabbed nothing

The interactive default is **not** what Part 1 measured: it loads the scene the dashboard saved
(`out11/scene_config.json`), uses the `contained` variant (a 25 mm kerb around the working area)
and a full multi-part scene. Reproduced with the exact flags (seed 56000, spread 0.004), it
refused or failed every order for six separate reasons, each isolated with an A/B (same scene and
seed, one thing changed):

| # | Cause | Evidence | Fix |
|---|---|---|---|
| 1 | **Phantom objects.** The depth-layer recovery for a part pressed against a same-colour tray took each tray's cavity floor (a 4,300 px region) for a "blue ball" / "red cube". They competed with the real trays for their catalog identity, so every order was refused before any motion. | Replay of the saved frame: phantoms with depth, none without; present with the old and the new size limits. | `split_depth_layered_container` now rejects children larger than a declared part (`perceive.py`). Fixture `tests/fixtures/tray_floor_top`. |
| 2 | **Kerb too close to the robot.** The kerb's near wall sits on the working-area edge (saved scene: y = -0.589, 51 mm from the base). Same cube, same seed: near wall at -0.555 passed, at -0.565, -0.575, -0.585, -0.595 all failed; open variant passed. | `pre-grasp hover did not converge ... la pose no es alcanzable`. | `pen.PEN_NEAR_LIMIT_FROM_BASE_M` = 90 mm: the near wall is never closer. |
| 3 | **Kerb too close to the parts.** With the quarter-turn wrist the jaws span y and reach ~33 mm from the part. Cube 22 mm from the far wall: contained fails, open passes (same scene and seed); 45 mm worked. | Contained 0/1 vs open 1/1. | `pen.PEN_JAW_CLEARANCE_M` = 30 mm: far and side walls stand out from the working area. |
| 4 | **Trays inside the arm's placement limit.** The saved trays were 0.12 m from the base, and all three shipped presets had them at 0.14-0.17 m. Strict trials: tray at 0.21, 0.18, 0.175, 0.165 m placed; 0.15 m gripped the cube then stopped the carry 20.3 mm short of a 20 mm tolerance (also with my narrower jaws disabled); 0.12 m never gripped. | `transporte detenido: error final de pinza 91.4 mm`. Nothing checked a tray against the arm's near limit (only picks had one). | `scene_spec.MIN_PLACE_REACH_M` = 0.17, enforced in scene validation, command grounding (a sentence: "...está a 14 cm de la base... Aléjalo un poco de la base") and the setup sheet. Presets moved: trays y -0.46 (0.18/0.20 m), parts row y -0.38, working area y -0.555..-0.355. A saved scene that breaks the rule prints why and falls back to `demo` (also in the dashboard log). Fixture `tests/fixtures/scenes/saved_scene_trays_near_base.json` is the scene that failed. |
| 5 | **The "distance error" you saw.** Your red cube measured 0.2063 m from the base, 1.3 mm beyond the 0.205 m pick limit, so a few mm of camera error flipped it between "no puedo llegar: está a 20 cm de la base" and an attempted pick. The setup sheet also failed the 56 mm tray cavity (0.5 mm of placement margin after the assumed 12.5 mm camera error; it recommends 70 mm). | Variants A-D all refused with that sentence. | The scene is now moved clear of the limit; the cavity warning is real and remains (the shipped 70 mm trays are the minimum that leaves 0.5 mm; use 84 mm trays for margin). |
| 6 | **A wrist-shifted camera frame vetoed centred grasps.** As the wrist covers the part its mask shrinks and the centroid walks: in the trace `seen` drifted 0.427 -> 0.422 m while the aperture was 1 mm from the anchor. The final contact check trusted any frame within 5 mm of the anchor, so it measured "5.0 mm from the part" and cancelled the close, three attempts running (the same "4.3-5.3 mm outside the slack" seen on cubes and hexagons). | Same eight seeds: 4/8 before, 8/8 after, nothing else changed. | `FINAL_CONTACT_OBSERVATION_TOLERANCE_M` 5 mm -> 2 mm: a frame can only refine the validated anchor within the alignment gate. |

I also tried an integral "plateau" step in the servo; it never fired in any run, so it was removed.
An experiment with `DEMO11_EXPERIMENTAL_APERTURE_SLACK_MM` looked like a fix (5/6 -> 6/6) but that
override only applies to a 30 mm ball with V-pads, so it did nothing for cubes: the change was
run-to-run variation, and I did not keep anything based on it.

### Results with the finished code (strict-physical, top camera, 30 mm, central region, 2 runs per cell)

| shape / layout | `contained` (your default) | `open` |
|---|---|---|
| cube single / crowded | 2/2, 2/2 | 2/2, 2/2 |
| ball single / crowded | 2/2, 2/2 | 2/2, 2/2 |
| cylinder single / crowded | 2/2, 2/2 | 2/2, 2/2 |
| hexagonal prism single / crowded | 2/2, 2/2 | 2/2, 2/2 |
| **triangular prism single / crowded** | **0/2, 0/2** | **0/2, 0/2** |
| total | **16/20** | **16/20** |

Reports: `final3_strict_contained_20261020.json`, `final3_strict_open_20261020.json`; all 32 passes
have physical grasps >= 1, zero assisted, a verified placement, and no `SUCCESS` run was rejected.

Your own flow (default `demo` scene, `contained`, `--strict-physical`, "separa las piezas por
colores"): **8/8** on seeds 21100-21107 after fix 6 (it was 4/8 on the same seeds before), 8/8 on
21200-21207 with the finished code, and 5/6 then 6/6 earlier on 21000-21005 before fix 6. In every
run, both cubes were picked physically and verified in their matching tray.

Still not working: the **triangular prism** (0/8 across both variants; see Part 1). Not run:
front and side cameras, 20 mm parts, and `forward_pick` after these last changes.

Tests: `pytest tests` 219 passed; `python tests/test_demo11.py` 1603 checks and
`python tests/test_planner.py` 828 checks, both SUCCESS. New: tray-floor phantom fixture, kerb
near-wall and jaw-clearance rules (including every preset), placement-reach rule and its
fallback, contact-anchor drift.

## Part 3 - default (non-strict) run: "task done but reported FAILED" and "separar por colores"

User report: «pon el cubo rojo en la caja roja» ended FALLIDA with the cube physically in the red box,
and «separa las piezas por colores» did not work. The default run is non-strict (kinematic grasp
assist), which Parts 1-2 had not exercised. Causes and fixes:

1. **False FALLIDA** - `_inside` demanded the cube centre above the tray rim height; a cube resting on
   the cavity floor was judged outside. Now `real_world.cavity_verdict` (x,y inside the cavity, z from
   rest_z - 6 mm to rest_z + 0.4 * part height); the failure text is logged (`not verified: ...`).
2. **Assisted close dragged the part** - the post-close drift check was applied to assisted grasps; it is
   skipped when the grasp is `_pregrasp_kinematic`.
3. **"Separar" refused/re-sorted pieces already in trays** - `_ignore_parts_already_in_trays` removes
   them from sort/clear/sequence plans (logged).
4. **Second command started from a different posture** - `Robot.park` now restores the first canonical
   park joints (`isaac_io.hold_arm_pose(positions=...)`).
5. **Premature stall at start of `down`** - with the wrist near its limit the aperture lags for a few
   corrections and `STALL_WINDOW = 6` declared "centring stalled" although the error then shrank to
   1-3 mm (seeds 24204/24205). `STALL_WINDOW` is now 8 (`DEMO11_STALL_WINDOW` overrides); the stall
   unit test runs 400 steps so the window can fill. 12 was tried and is too long: strict colour sort
   fell to 6/8 on seeds 25100-25107 (6 and 8 both give 8/8).

Non-strict results (contained, demo preset, spread 0.004): order alone 4/4 (24400-24403); separa alone
4/4 (24300-24303); order then separa in one session 6/6 (24200-24205; it was 0/6 before fix 4, 4/6
before fix 5). Strict colour sort after these changes (window 8): 8/8 on 25100-25107, all physical grasps.

## Part 4 (2026-10-21): the triangular prism now grips (0/12 -> 12/12)

Cause, measured with the new `DEMO11_TRACE_PART=1` trace (`[part]` lines: clearance to each pad, offset
along/across the jaw axis, tilt, jaw angle): the **V45 insert tips the 24 mm-tall triangle while the jaw
closes** (tilt 5-11 degrees, part lifted off the table) and the vertex never seats against the fixed pad;
with the 0.15 rad close the pads end ~29.8 mm apart at the surfaces against a ~24 mm part, so the jaw
reaches its target without squeezing. Tighter closes (0.0 / 0.04 / 0.08 rad) only made the jaw drag the
part 12-15 mm (drift guard abort): 0/3 each. Stock wide opening: 1/3. **Flat pads: 3/3.**

Fix: `gripper_contact_pads.flat_profile_for` / `automatic_scene_pad_shape` select **flat pads** for the 30 mm
triangular prism (stiffness 700, 45-step ramp, 400-step carry unchanged). The 20 mm triangle keeps V45.

| strict, top camera, central | before | after |
|---|---|---|
| triangular prism, open+contained x single+crowded, 3 reps (seed 21500) | 0/12 | **12/12** |
| all five shapes, open, single+crowded, 2 reps (seed 24000, default path, no env overrides) | triangle 0 | **20/20** |
| triangle, front camera, open, 2 reps (seed 24500) | - | 2/4 |
| triangle, default (non-strict) contained, 2 reps (seed 24700) | - | 4/4 tasks SUCCESS (simulator assist) |

Hexagon 12/12 strict in the same baseline run. **Still open:** front-camera triangle 2/4 - both misses are
reach failures before any grasp ("pre-grasp hover did not converge", wrist -30 degrees at scene yaw 30 and one
crowded layout), not pad/grip. Side camera, 20 mm parts and the contained variant on other cameras not run.

## Part 5 (2026-10-21): coverage of the flat-pad triangle, front camera and 20 mm parts

Measured after Part 4 (strict, no env overrides unless noted):

| scenario | result |
|---|---|
| 30 mm triangle, side camera, open+contained x single+crowded, 3 reps | **12/12** |
| 30 mm triangle, front camera, same matrix (V-groove wrist yaw) | 9/12 |
| all five shapes, 20 mm, top, open, single+crowded, 2 reps | 18/20 (cube, ball, cylinder, hexagon 2/2 each layout; 20 mm triangle 2/4) |

The three front-camera misses all chose a **-30 / -60 degree wrist** ("pre-grasp hover did not converge",
"la pose no es alcanzable"). That yaw only existed to seat the vertex in the V-groove. Flat pads close an
edge and the opposite vertex at any yaw, so `tool_yaw_for` now returns a **neutral wrist (0)** for the
30 mm triangle (`DEMO11_TRIANGLE30_YAW_MODE=legacy` restores the old angles). A/B on the three failing
scenes: 2/3; then the whole 30 mm triangle matrix on top, front and side (seed 28000): **35/36**
(top 12/12, side 12/12, front 11/12). `forward_pick` region, top+front (seed 29000): 14/16.

Still failing, with causes from the logs:
* **front camera, ~1 in 8:** the part is only partly seen (occlusion) and the arm cannot reach the
  pre-grasp pose (seeds 26002, 28013, 29008). Flaky: the same pose passes on other seeds.
* **top, forward_pick, contained crowded, 1 of 2:** closed 3.9 mm off-centre (outside the 3.1 mm slack),
  the flat pads did not hold it (seed 29007).
* **20 mm triangle on V45 pads, yaw 0 (2 of 3 yaw-0 scenes):** `executor` aborts the close with "cierre
  desplazo la pieza 13-15 mm" although the trace shows the pads centred on the part (0.7 mm, no tilt).
  The reference is the camera's pre-close target, which is ~13 mm off for this part. Not fixed
  (relaxing the other drift guard in `11_real_sorting_with_monitor.py` had no effect, so it was reverted).
  Flat pads cannot hold a 20 mm triangle (about 15 mm across, narrower than the closed pads).
* Not run: 20 mm parts on front/side cameras, the contained variant on 20 mm parts.

## Part 6 (2026-10-22): 20 mm triangle close, front-camera reach

**Correction to Part 5.** The 20 mm triangle's "cierre desplazo la pieza 13-15 mm" is *not* a camera
offset: `executor` compares the part's true position at the end of the descent with its position during
the close, so the jaws really do move it. A per-step trace (`DEMO11_TRACE_PART_EVERY=1`) shows a
vertex catching the V45 pad within a few ticks of the close.

Baseline on 12 fresh scenes (top, open, single+crowded, seed 31000): **7/12** - three closes shoved the
part 9-14 mm, two had the part centred and untilted (16 mm across the jaws) but not held. Same scenes:

| variant | result |
|---|---|
| ramp 20 steps, close 0.10 rad (old default) | 7/12 |
| ramp 45 steps | 10/12 |
| close 0.06 rad | 10/12 |
| **ramp 45 steps + close 0.07 rad** | **12/12** |

Both are now the default for the 20 mm triangle on V pads (`TRIANGLE_CLOSE_RAMP_STEPS = 45`,
`close_command_for` -> 0.07). Independent check on 24 new scenes (seeds 32000/33000, top, front, side x
open, contained x single, crowded): **18/24** - side 8/8, top 7/8, front 3/8.

**Front camera, 20 mm triangle (5 of 8 fail), cause established, not fixed.** All five end before any
close with "pre-grasp hover did not converge". The part is hidden from the front camera by the arm, but
that is not the cause: raising the occlusion budgets to 1000 (`TRACK_MAX_OCCLUDED`,
`TRACK_MAX_SCENE_LINKED_MISSES`) and letting the hover correct without a sighting changed nothing. The
hover trace shows the jaws start 31 mm from the target in a different arm configuration (shoulder_lift
0.42 / elbow -0.63 against 0.26 / -0.39 in a passing run of the same scene), `wrist_flex` pinned at its
1.658 rad limit, and the corrections move the body the wrong way. It is a path-dependent IK branch, so the
same scene passes or fails from run to run. Those experiments were reverted. The 30 mm triangle shows the
same failure at about 1 in 8.

Still open: the front-camera reach (a different approach path or hover height is the lead), seed 29007
(first attempt 3.9 mm off-centre after a wrist-clipped mask, retry slipped 81 mm during carry), top 33000
(close shove on a retry), and 20 mm cube/ball/cylinder/hexagon on the front and side cameras.

## Part 7 (2026-10-22): front-camera reach fixed - the front park pose

**Supersedes the "front camera reach" item in Part 6.** The path-dependent failure was real but its
cause was the *starting pose*, not occlusion, hover height or the IK. `main` parks the arm for the front
camera at hold point `(0.20, -0.58, 0.39)` (a pose out of the camera's sight line). From there the
first vertical move to the hover height stops at body z 0.26 m instead of 0.223 m, so the whole lateral
slide to the part runs 37 mm too high. At that height the arm can only reach x >= 0.44, takes an
extended configuration (shoulder_lift 0.42, elbow -0.63, `wrist_flex` pinned at 1.658 rad) and the
pre-grasp hover then never converges. The top camera starts from the standard park pose and reaches
0.224 m directly (shoulder_lift 0.05, elbow -0.12). Found by diffing the `DEMO11_TRACE_TRACK=1` /
`DEMO11_TRACE_MOTION=1` traces of a passing and a failing front run of the same scene.

Fix: front park pose `(0.28, -0.58, 0.39)`. Same 8 scenes (20 mm triangle, front, seeds 32004-7 and
33004-7): 3/8 at x = 0.20, **8/8** at x = 0.28, 8/8 at the standard `(0.35, -0.58, 0.46)`.
Regression check, front camera, every shape x 20/30 mm x open/contained x single/crowded (40 scenes,
seed 34000, strict, 0 simulator-assisted grasps):

| park pose | pass | per shape (20 mm / 30 mm) |
|---|---|---|
| **new, x = 0.28** | **40/40** | ball, cube, cylinder, hexagon, triangle all 4/4 at both sizes |
| old, x = 0.20 | 25/40 | ball 2/4 and 1/4, cube 3/4 and 4/4, cylinder 4/4 and 1/4, hexagon 4/4 and 3/4, triangle 0/4 and 3/4 |

`DEMO11_FRONT_PARK_POSE=x,y,z` overrides the pose for A/B runs. Ruled out on the way, all reverted:
bigger occlusion budgets (`TRACK_MAX_OCCLUDED`, `TRACK_MAX_SCENE_LINKED_MISSES`) and a blind pre-grasp
hover (no change), a lower hover via a `DEMO11_HOVER_FAR_M` sweep (no effect, the plan is floored at
`TRAVEL_Z`) and a lowered pick approach (2/8 and 1/8, worse).

Still open: seed 29007 (30 mm triangle, forward_pick, crowded tray: first attempt 3.9 mm off-centre
after a wrist-clipped mask, retry slipped 81 mm in the carry); top 33000 (20 mm triangle, close shove
on a retry); front camera with `forward_pick`; side camera for the other shapes at 20 mm.

## Reproduce

```
export HF_HUB_OFFLINE=1
python tools/probe_hover_reach.py --headless
python tools/benchmark_scene_matrix.py --headless --strict-physical --cameras top --variants open \
    --layouts single cluster --repetitions 2 --region central --seed 16000
```

Use a fresh `--seed` per run: log names are `<seed>_<camera>_...` and older campaigns reuse them.
`DEMO11_HOVER_FAR_M` and `DEMO11_DISABLE_NARROW_OPEN=1` exist for A/B runs; `DEMO11_TRACE_PART=1` logs the
prism's position between the pads during close/lift/carry (`DEMO11_TRACE_PART_EVERY=1` for every tick).
