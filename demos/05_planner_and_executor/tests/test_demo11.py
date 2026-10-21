"""Tests for demo 11's parts that need no simulator. No GPU, no Isaac.

    python tests/test_demo11.py

Four groups, in the order they would stop a run:

* **The real-world numbers.** Masses come from materials, the jaws have an aperture, the finger
  tips clear the rim, and a placement has to have room left after the camera's own error. These
  are the claims that make the demo a replica rather than a simulation, so they are checked as
  arithmetic rather than trusted.
* **The bus and the dashboard.** The two threads meet only through ``telemetry.Bus``, and the
  browser only ever queues requests, so a snapshot taken while a robot thread is publishing has
  to be self-consistent and the HTTP surface has to answer.
* **The flow diagram's integrity.** The picture is built from ``telemetry.FLOW``; if an edge names
  a node that does not exist the dashboard draws a line to nowhere and nobody notices for a week.
* **The dashboard's own script.** ``tests/dashboard_client.js`` loads the real ``index.html`` into
  a small DOM and replays a run's snapshots through it, reporting what it painted. This is here
  because two of the worst bugs in this demo were in the browser's code and no Python test could
  see them: a write into a null element that froze ten of the twelve boxes of the decision route
  on every single frame, and a write to an ``id`` the page never declared that silently killed
  the whole plan card. It is skipped when node is not installed.
"""

from __future__ import annotations

import json
import importlib.util
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import executor as executor_module  # noqa: E402
import gripper_contact_pads  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402
import telemetry  # noqa: E402

failures: list[str] = []
checks = 0


def test_pre_grasp_hover_never_exceeds_the_height_the_arm_can_hold():
    """Strict logs: a 150 mm ball hover left the aperture ~150 mm from the part at 0.24 m.

    ``tools/probe_hover_reach.py`` measured that the pads drift off the part as the hover rises,
    so the planner may not ask for a taller hover than ``hover_reach_limit_m`` at that radius,
    for any geometry. The ball is no longer special-cased to a taller one.
    """
    import planner
    from planner import Plan, _grasp_and_hold, _pick_and_place
    from scene import Thing

    base_x, base_y = scene_spec.ROBOT_XY
    for radius in (.20, .215, .23, .25, .27, .30):
        for lateral in (-.06, 0.0, .06):
            x, y = base_x + lateral, base_y + math.sqrt(radius ** 2 - lateral ** 2)
            limit = planner.hover_reach_limit_m(x, y)
            for kind in ("cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"):
                size = [0.03, 0.03, 0.03]
                thing = Thing(id=7, kind=kind, color="blue", centroid_px=[0, 0],
                              bbox_px=[0, 0, 1, 1], area_px=1, mean_hsv=[0, 0, 0],
                              confidence=1.0, source="test", pos_m=[x, y, 0.056],
                              size_m=size, aim_xy=[x, y])
                plan = Plan(intent="pick")
                _grasp_and_hold(plan, thing, (0.0, 0.0), kind)
                above = next(step.pose for step in plan.steps if step.action == "above")
                grasp = planner.grasp_pose(thing)
                clearance = above[2] - grasp[2]
                allowed = max(limit, planner.TRAVEL_Z - grasp[2])
                check(clearance <= allowed + 1e-9,
                      f"{kind} at {radius:.3f} m hovers {clearance * 1000:.0f} mm, "
                      f"arm holds {allowed * 1000:.0f} mm")
    check(planner.hover_reach_limit_m(base_x, base_y + .20) == planner.GRASP_APPROACH_CLEARANCE_M,
          "near the base the stock 90 mm hover is still reachable")
    check(planner.hover_reach_limit_m(base_x, base_y + .25) < planner.GRASP_APPROACH_CLEARANCE_M,
          "from 0.23 m outwards the hover is limited")
    far_ball = Thing(id=7, kind="ball", color="blue", centroid_px=[0, 0], bbox_px=[0, 0, 1, 1],
                     area_px=1, mean_hsv=[0, 0, 0], confidence=1.0, source="test",
                     pos_m=[0.377, -0.399, 0.056], size_m=[0.03, 0.03, 0.03],
                     aim_xy=[0.377, -0.399])
    far_plan = Plan(intent="pick")
    _grasp_and_hold(far_plan, far_ball, (0.0, 0.0), "ball")
    far_above = next(step.pose for step in far_plan.steps if step.action == "above")
    check(far_above[2] - planner.grasp_pose(far_ball)[2] < 0.10,
          "the central 30 mm ball (0.243 m from the base) no longer hovers 150 mm up")
    del _pick_and_place


def test_hover_reach_probe_table_matches_the_planner_limit():
    """The limit is calibrated from a measured table; keep the two from drifting apart."""
    import json
    import planner

    path = ROOT / "results" / "hover_reach_20261019.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    baseline = min(row["aperture_xy_error_mm"] for row in rows
                   if row["aperture_xy_error_mm"] is not None)
    for row in rows:
        if row["aperture_xy_error_mm"] is None:
            continue
        limit = planner.hover_reach_limit_m(row["x"], row["y"])
        if row["clearance_m"] <= limit + 1e-9:
            excess = row["aperture_xy_error_mm"] - baseline
            check(excess <= 25.0,
                  f"allowed hover {row['clearance_m'] * 1000:.0f} mm at r="
                  f"{row['radius_m']:.3f} leaves {excess:.1f} mm of aperture error")


def test_large_ball_keeps_validated_lift_height():
    """The ball's hover must not become a taller post-grasp lift."""
    import planner
    from planner import Plan, _grasp_and_hold, _pick_and_place
    from scene import Thing

    ball = Thing(id=7, kind="ball", color="blue", centroid_px=[0, 0], bbox_px=[0, 0, 1, 1],
                 area_px=1, mean_hsv=[0, 0, 0], confidence=1.0, source="test",
                 pos_m=[0.38, -0.40, 0.056], size_m=[0.03, 0.03, 0.03],
                 aim_xy=[0.38, -0.40])
    plan = Plan(intent="pick")
    _grasp_and_hold(plan, ball, (0.0, 0.0), "ball")
    above = next(step.pose for step in plan.steps if step.action == "above")
    lift = next(step.pose for step in plan.steps if step.action == "lift")
    check(above[2] - ball.pos_m[2] <= 0.09 + 1e-9,
          f"30 mm ball hovers within the reachable 90 mm, got {above[2] - ball.pos_m[2]:.3f} m")
    check(abs((lift[2] - planner.grasp_pose(ball)[2]) - 0.09) < 1e-9,
          "the ball's post-grasp lift remains at the validated 90 mm height")
    tray = Thing(id=9, kind="box", color="red", centroid_px=[0, 0], bbox_px=[0, 0, 1, 1],
                 area_px=1, mean_hsv=[0, 0, 0], confidence=1.0, source="test",
                 pos_m=[0.473, -0.4095, 0.056], size_m=[0.084, 0.084, 0.03],
                 cavity_m=[0.07, 0.07], aim_xy=[0.473, -0.4095])
    placed_plan = Plan(intent="sort")
    _pick_and_place(placed_plan, ball, tray, (0.0, 0.0), "ball")
    transit = next(step.pose for step in placed_plan.steps if step.action == "carry")
    placed_lift = next(step.pose for step in placed_plan.steps if step.action == "lift")
    check(transit[2] >= placed_lift[2],
          "horizontal transit stays at or above the physical lift height through clustered objects")
    cube = Thing(id=8, kind="cube", color="red", centroid_px=[0, 0], bbox_px=[0, 0, 1, 1],
                 area_px=1, mean_hsv=[0, 0, 0], confidence=1.0, source="test",
                 pos_m=[0.38, -0.40, 0.056], size_m=[0.03, 0.03, 0.03],
                 aim_xy=[0.38, -0.40])
    check(planner.grasp_approach_clearance(cube.kind, cube.size_m) ==
          planner.GRASP_APPROACH_CLEARANCE_M,
          "other object shapes keep the established approach clearance")


def test_above_tracking_finishes_xy_alignment_before_descending():
    """The pre-grasp XY sweep stays high; only its final phase descends vertically."""
    start_z, target_z = .388, .298
    check(executor_module.tracked_body_z("above", start_z, target_z, .50) == start_z
          and executor_module.tracked_body_z("above", start_z, target_z, .65) == start_z,
          "the first 65 percent of the hover approach keeps its starting height")
    check(abs(executor_module.tracked_body_z("above", start_z, target_z, .825) - .343) < 1e-9
          and executor_module.tracked_body_z("above", start_z, target_z, 1.0) == target_z,
          "the final hover phase descends vertically to its planned clearance")
    check(abs(executor_module.tracked_body_z("down", start_z, target_z, .50) - .343) < 1e-9,
          "the contact descent keeps its ordinary full linear Z interpolation")


def test_scene_approach_clearance_covers_the_tallest_configured_obstacle():
    """Custom tall props raise the horizontal pre-grasp travel plane automatically."""
    from types import SimpleNamespace

    demo = load_demo11()
    ordinary = demo.safe_approach_clearance_z([SimpleNamespace(height=.03)])
    tall_scene = demo.safe_approach_clearance_z([
        SimpleNamespace(height=.03), SimpleNamespace(height=.10)])
    check(ordinary >= demo.planner.TRAVEL_Z,
          "the default approach stage is at least the calibrated travel plane")
    check(tall_scene >= scene_spec.TABLE_TOP_Z + .14,
          "a 100 mm object raises the approach plane above its top and safety margin")


def test_pregrasp_hover_correction_is_damped_and_bounded():
    """Extra hover alignment moves toward the measured aperture error without a large jump."""
    demo = load_demo11()
    dx, dy = demo.bounded_servo_delta((.040, -.030))
    check(dx > 0 and dy < 0 and abs(math.hypot(dx, dy) - .008) < 1e-9,
          "the hover servo corrects in the error direction with an 8 mm step cap")
    check(demo.bounded_servo_delta((0.0, 0.0)) == (0.0, 0.0)
          and demo.bounded_servo_delta((float('nan'), 1.0)) == (0.0, 0.0),
          "zero and non-finite servo errors cannot move the tool")
    check(demo.supports_large_ball_hover("ball", (.030, .030, .030))
          and not demo.supports_large_ball_hover("ball", (.020, .020, .020))
          and not demo.supports_large_ball_hover("cube", (.030, .030, .030)),
          "the extra hover controller is limited to 30 mm balls")
    from types import SimpleNamespace
    check(all(demo.requires_pregrasp_alignment(SimpleNamespace(kind=kind, container=False))
              for kind in ("cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"))
          and not demo.requires_pregrasp_alignment(SimpleNamespace(kind="cube", container=True))
          and not demo.requires_pregrasp_alignment(None),
          "all supported movable shapes require a verified stationary hover before contact")


def test_feedback_servo_rebases_each_correction_on_the_live_body_pose():
    """A lagging arm must not accumulate aperture error into a runaway command."""
    demo = load_demo11()
    body = (.330, -.450)
    aperture = (.290, -.450)
    target = (.300, -.450)
    first = demo.feedback_body_command(
        body, aperture, target, max_step_m=.008)
    repeated = demo.feedback_body_command(
        body, aperture, target, max_step_m=.008)
    check(first == repeated and abs(first[0] - .338) < 1e-9,
          f"repeated stale feedback must not wind up the command: {first}, {repeated}")

    moved_body = (.336, -.450)
    updated_aperture = (.296, -.450)
    updated = demo.feedback_body_command(
        moved_body, updated_aperture, target, max_step_m=.008)
    check(abs(updated[0] - .340) < 1e-9,
          f"the next command is based on live body feedback, got {updated}")
    invalid = demo.feedback_body_command(
        body, (float('nan'), 0.0), target, max_step_m=.008)
    check(invalid == body,
          "invalid camera/aperture feedback holds the measured pose without drifting")

    measured_x = .330
    target_x = .300
    residuals = []
    for _ in range(20):
        aperture_x = measured_x - .040
        residuals.append(abs(target_x - aperture_x))
        command_x, _ = demo.feedback_body_command(
            (measured_x, -.45), (aperture_x, -.45), (target_x, -.45),
            max_step_m=.008)
        check(abs(command_x - measured_x) <= .008 + 1e-9,
              "each measured-pose correction stays within its motion cap")
        measured_x += .5 * (command_x - measured_x)  # half-step actuator lag
    check(all(later <= earlier for earlier, later in zip(residuals, residuals[1:]))
          and residuals[-1] < .001,
          f"a lagging first-order arm converges without command wind-up: {residuals[-1]:.4f} m")


def test_final_contact_uses_only_a_trusted_scene_anchor_and_measured_recenter_pose():
    demo = load_demo11()
    anchor = (.30, -.45)
    check(demo.trusted_contact_target((.3015, -.45), anchor) == (.3015, -.45)
          and demo.trusted_contact_target((.303, -.45), anchor) == anchor
          and demo.trusted_contact_target(None, anchor) == anchor,
          "a wrist-clipped final mask beyond 2 mm cannot redirect contact from its scene anchor")
    # Strict trace, seed 21102: the aperture was 1 mm from the anchor at the bottom of the
    # descent, but the camera had drifted from x 0.427 to 0.422 as the wrist covered the cube.
    # Trusting that frame measured "5.0 mm from the part" and cancelled a centred grasp.
    anchor = (.427, -.382)
    drifted = (.422, -.381)
    aperture = (.427, -.383)
    gap_to_frame = math.dist(aperture, drifted)
    target = demo.trusted_contact_target(drifted, anchor)
    gap_to_target = math.dist(aperture, target)
    check(gap_to_frame > .0031 and gap_to_target < .0031,
          f"the drifted frame would reject the grasp ({gap_to_frame * 1000:.1f} mm); the anchor "
          f"accepts it ({gap_to_target * 1000:.1f} mm)")
    corrected = demo.feedback_body_command(
        (.34, -.49), (.30, -.45), (.31, -.44), max_step_m=.025)
    check(all(abs(value - expected) < 1e-9
              for value, expected in zip(corrected, (.35, -.48))),
          f"safe recenter is based on measured body and aperture poses: {corrected}")



def test_ball_precontact_hold_waits_for_lateral_alignment():
    """A large sphere's descent pauses above contact until the aperture is centred."""
    demo = load_demo11()
    held_z, holding = demo.ball_precontact_hold_z(
        action="down", large_ball=True, requested_z=.232,
        lateral_error_m=.019, holding=False)
    check(holding and held_z == demo.BALL_PRECONTACT_HOLD_Z_M,
          "a 30 mm ball pauses descent at the pre-contact height while error exceeds 8 mm")
    resumed_z, holding = demo.ball_precontact_hold_z(
        action="down", large_ball=True, requested_z=.222,
        lateral_error_m=.007, holding=True)
    check(not holding and resumed_z == .222,
          "the vertical approach resumes once the aperture error is within 8 mm")
    other_z, holding = demo.ball_precontact_hold_z(
        action="down", large_ball=False, requested_z=.222,
        lateral_error_m=.040, holding=False)
    check(not holding and other_z == .222,
          "other geometries retain their existing descent path")


def test_jaws_open_only_as_wide_as_the_part_needs():
    """Strict traces: the stock 70 mm opening clips a tray rim 90 mm away and wedges the jaw."""
    stock, close = 0.822, 0.20
    ball = gripper_contact_pads.open_command_for("ball", (.03, .03, .03), stock, close)
    small = gripper_contact_pads.open_command_for("cube", (.02, .02, .02), stock, 0.20)
    wide = gripper_contact_pads.open_command_for("cube", (.08, .08, .08), stock, 0.20)
    check(0.40 < ball < 0.46, f"a 30 mm part opens to about 0.43 rad, got {ball:.3f}")
    separation = (gripper_contact_pads.JAW_SEPARATION_INTERCEPT_MM
                  + gripper_contact_pads.JAW_SEPARATION_MM_PER_RAD * ball)
    check(separation < 50.0, f"pads are {separation:.0f} mm apart, not the stock 70 mm")
    check(small < ball, "a smaller part needs a narrower opening")
    check(wide == stock, "a part wider than the stock opening never exceeds it")
    check(gripper_contact_pads.open_command_for("ball", (), stock, close) == stock,
          "without a size the stock opening is kept")
    check(gripper_contact_pads.open_command_for("ball", (.03, .03, .03), stock, 0.50) >= 0.60,
          "the opening always stays above the close target")
    # The moving jaw's reach beyond the part centre must clear a tray wall 48 mm away by more
    # than the stock opening did (the stock pad was ~39 mm out, the wall's outer face 58 mm).
    stock_reach = (gripper_contact_pads.JAW_SEPARATION_INTERCEPT_MM
                   + gripper_contact_pads.JAW_SEPARATION_MM_PER_RAD * stock) / 2
    new_reach = separation / 2
    check(stock_reach - new_reach > 12.0,
          f"narrowing frees {stock_reach - new_reach:.0f} mm beside the part")


def test_pick_open_width_is_set_on_approach_and_restored_on_retreat():
    """The narrow opening is per pick; the stock one comes back for everything else."""
    import types

    import torch

    demo = load_demo11()
    term = types.SimpleNamespace(_open_command=torch.tensor([0.822]))
    manager = types.SimpleNamespace(get_term=lambda name: term)
    robot = types.SimpleNamespace(
        env=types.SimpleNamespace(action_manager=manager),
        item_kinds={"ball_a": "ball", "box": "box"},
        item_sizes={"ball_a": (.03, .03, .03), "box": (.084, .084, .05)},
        close_command_override=None, log=lambda *args, **kwargs: None)
    set_width = demo.StrictExecutor._set_pick_open_width
    set_width(robot, "down", "ball_a")
    check(abs(float(term._open_command[0]) - 0.822) < 1e-6,
          "only the approach (and its retreat) changes the opening")
    set_width(robot, "above", "ball_a")
    check(0.40 < float(term._open_command[0]) < 0.46,
          f"approach opens to the part width, got {float(term._open_command[0]):.3f}")
    set_width(robot, "retreat", "box")
    check(abs(float(term._open_command[0]) - 0.822) < 1e-6,
          "the retreat restores the stock opening")
    set_width(robot, "above", "ball_a")
    set_width(robot, "above", "unknown_prim")
    check(abs(float(term._open_command[0]) - 0.822) < 1e-6,
          "an unknown target falls back to the stock opening")


def test_ball_precontact_pause_never_lifts_the_arm_above_where_the_descent_started():
    """Strict logs: pausing at the fixed 0.285 m sent a 0.223 m hover back up to 0.285 m.

    That height was the old 150 mm hover; the arm cannot hold it at 0.25 m radius, so each pause
    left the aperture 60-70 mm off the ball and centring stalled before any contact.
    """
    demo = load_demo11()
    held_z, holding = demo.ball_precontact_hold_z(
        action="down", large_ball=True, requested_z=.222,
        lateral_error_m=.061, holding=False, ceiling_z=.223)
    check(holding and held_z == .223,
          f"the pause holds at the hover the descent began from, not above it: {held_z}")
    still_z, holding = demo.ball_precontact_hold_z(
        action="down", large_ball=True, requested_z=.215,
        lateral_error_m=.061, holding=True, ceiling_z=.223)
    check(holding and still_z == .223 and still_z <= .223,
          "an ongoing pause stays at the start height")
    high_z, holding = demo.ball_precontact_hold_z(
        action="down", large_ball=True, requested_z=.280,
        lateral_error_m=.020, holding=False, ceiling_z=.300)
    check(holding and high_z == demo.BALL_PRECONTACT_HOLD_Z_M,
          "a descent that starts above the pause height still pauses at the pause height")
    resumed_z, holding = demo.ball_precontact_hold_z(
        action="down", large_ball=True, requested_z=.215,
        lateral_error_m=.005, holding=True, ceiling_z=.223)
    check(not holding and resumed_z == .215, "the descent resumes once aligned")


def test_ball_descent_does_not_jump_after_precontact_pause():
    """The held trajectory may advance internally, but its released Z command stays smooth."""
    demo = load_demo11()
    released = demo.bounded_ball_descent_z(
        action="down", large_ball=True, requested_z=.223, previous_z=.285)
    check(abs(released - .284) < 1e-9,
          "a 62 mm accumulated descent is released as at most a 1 mm command step")
    regular = demo.bounded_ball_descent_z(
        action="down", large_ball=True, requested_z=.280, previous_z=.281)
    check(abs(regular - .280) < 1e-9,
          "normal small descent steps are not delayed")
    other = demo.bounded_ball_descent_z(
        action="down", large_ball=False, requested_z=.223, previous_z=.285)
    check(other == .223,
          "the ball-specific descent cap does not alter other geometry profiles")


def test_experimental_ball_vee_slack_is_narrowly_scoped():
    """Experimental contact slack cannot leak into stock fingers or other geometries."""
    demo = load_demo11()
    ball = (.030, .030, .030)
    cube = (.030, .030, .030)
    measured = demo.aperture_slack_for_size(ball)
    check(abs(demo.aperture_slack_for_profile(ball, kind="ball", pad_shape="stock",
                                               experimental_slack_mm=4.5) - measured) < 1e-9,
          "stock fingers retain measured ball slack")
    check(abs(demo.aperture_slack_for_profile(ball, kind="ball", pad_shape="vee",
                                               experimental_slack_mm=4.5) - .0045) < 1e-9,
          "the bounded experimental override is available for a 30 mm ball with V pads")
    check(abs(demo.aperture_slack_for_profile(cube, kind="cube", pad_shape="vee",
                                               experimental_slack_mm=4.5) - measured) < 1e-9,
          "other shapes keep their measured aperture slack")

# How long a websocket session waits before concluding that nothing more is coming, and how long
# one read waits for the next message.
#
# The read timeout is the one that decides the wall clock: a session ends as soon as a read comes
# back empty, so this is what each of the three sessions costs. It only has to tell "the frame is
# on its way" from "there is no frame coming", and the handler writes the snapshot, the frame's
# header and the JPEG back to back with nothing but the socket write in between - microseconds.
# 0.25 s is a very large margin for that, and three sessions of it is most of a second saved.
QUIET_WINDOW_S = 0.5
QUIET_READ_S = 0.25

# Whether node is known to be missing, so it is looked for once rather than per test.
#
# The report itself is NOT cached across tests: the two tests that need the page want different
# batches replayed into it, so one node process each. What is shared is the loader.
_CLIENT_UNAVAILABLE = False


def dashboard_client_report(page_batches):
    """The report for these batches, loading the page under node once.

    Returns None when node is not installed, which is the only case in which there is nothing to
    report - and the caller says so rather than skipping quietly.
    """
    global _CLIENT_UNAVAILABLE
    if _CLIENT_UNAVAILABLE:
        return None
    report = run_dashboard_client(page_batches)
    if report is None:
        _CLIENT_UNAVAILABLE = True
    return report


def check(condition, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def load_recorded_scene(path):
    """Load a scene recorded by an earlier campaign, without today's tray-reach rule.

    Those files put the trays at y -0.50, 0.14-0.17 m from the base, nearer than
    ``scene_spec.MIN_PLACE_REACH_M``. The tests that use them exercise the randomizer and the
    perception-clearance logic over a real layout, not reach, so they must still load; the
    rule has its own tests.
    """
    import scene_config

    with patch.object(scene_spec, "MIN_PLACE_REACH_M", 0.0):
        return scene_config.load(path)


# -- the real-world numbers -----------------------------------------------------------


def test_masses_come_from_materials():
    """A 30 mm cube is not 50 g, and a hollow ball is not a solid one."""
    cube = next(item for item in scene_spec.DEFAULT_SCENE if item.kind == "cube")
    ball = next(item for item in scene_spec.DEFAULT_SCENE if item.kind == "ball")
    check(20.0 < real_world.mass_kg(cube) * 1000 < 35.0,
          f"a 30 mm ABS cube should be around 28 g, got {real_world.mass_kg(cube) * 1000:.1f}")
    check(real_world.mass_kg(ball) * 1000 < real_world.mass_kg(cube) * 1000,
          "a hollow ball must be lighter than a solid cube of the same size")
    # Demo 10 spawns 0.05 kg for everything; this is the number demo 11 exists to replace.
    check(all(real_world.mass_kg(item) < 0.05 for item in scene_spec.DEFAULT_SCENE
              if not item.container),
          "every loose part should now be lighter than the 50 g demo 10 spawned")
    check(real_world.volume_m3(ball) < real_world.volume_m3(cube),
          "a 30 mm sphere has less volume than a 30 mm cube")


def test_the_jaws_have_an_aperture_and_a_width_must_fit_it():
    """A part wider than the aperture cannot be held at any force. Demo 10 proved this."""
    too_wide = scene_spec.Item(prim="ball_40", kind="ball", color="red", x=0.4, y=-0.44,
                               shape="sphere", size=(0.04, 0.04, 0.04))
    too_narrow = scene_spec.Item(prim="ball_6", kind="ball", color="red", x=0.4, y=-0.44,
                                 shape="sphere", size=(0.006, 0.006, 0.006))
    check(not real_world.grasp(too_wide).feasible, "a 40 mm ball must not be graspable")
    check("wider than" in real_world.grasp(too_wide).reason,
          f"the reason should name the aperture: {real_world.grasp(too_wide).reason}")
    check(not real_world.grasp(too_narrow).feasible, "a 6 mm ball has no contact patch")
    for item in scene_spec.DEFAULT_SCENE:
        if not item.container:
            check(real_world.grasp(item).feasible, f"{item.prim} must be graspable: "
                                                    f"{real_world.grasp(item).report()}")


def test_the_friction_cone_is_what_carries_the_part():
    """2 * mu * N against m * g: doubling the mass doubles the force needed."""
    cube = next(item for item in scene_spec.DEFAULT_SCENE if item.kind == "cube")
    heavy = scene_spec.Item(**{**cube.__dict__, "size": (0.03, 0.03, 0.06),
                               "prim": "cube_heavy"})
    light_grip = real_world.Gripper(max_grip_force_n=0.05)
    check(real_world.grasp(heavy, light_grip).required_force_n >
          real_world.grasp(cube, light_grip).required_force_n,
          "a heavier part must need more force")
    check(not real_world.grasp(cube, light_grip).feasible,
          "a gripper too weak for the part must be reported as such")
    check("more than" in real_world.grasp(cube, light_grip).reason,
          "the reason should name the available force")
    check(real_world.grasp(cube).margin > 1.0,
          f"the declared gripper holds a cube with margin {real_world.grasp(cube).margin:.1f}x")


def test_grasp_benchmark_covers_every_supported_loose_geometry():
    import importlib.util

    path = ROOT / "tools" / "benchmark_grasp_objects.py"
    spec = importlib.util.spec_from_file_location("demo11_grasp_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module.cases(["cube", "ball", "cylinder"])
    kinds = {item.kind for item in selected if not item.container}
    check(kinds == {"cube", "ball", "cylinder"},
          f"the physical benchmark covers all supported loose geometries: {kinds}")
    check(all(real_world.grasp(item).feasible for item in selected if not item.container),
          "every trial size passes the declared aperture and force preflight")
    check(not any(item.size[0] > real_world.GRIPPER.aperture_m - real_world.GRIP_WIDTH_MARGIN_M
                  for item in selected if not item.container),
          "the physical trial set excludes geometries already known not to fit")


def test_contact_benchmark_uses_a_spawned_rigid_body_target():
    import importlib.util

    path = ROOT / "tools" / "benchmark_grasp_objects.py"
    spec = importlib.util.spec_from_file_location("demo11_contact_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module.cases(["cube"])
    target = module.contact_diagnostic_target(selected, ["cube"])
    check(target != scene_spec.SHIPPED_CUBE,
          "contact diagnostics avoid the stock cube's custom scene-USD spawn adapter")
    check(any(item.prim == target for item in selected),
          "contact diagnostics select an object present in the scene")


def test_unconfigured_task_cube_is_not_a_hidden_scene_object():
    import importlib.util

    path = ROOT / "11_real_sorting_with_monitor.py"
    spec = importlib.util.spec_from_file_location("demo11_scene_asset_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ball = scene_spec._ball("ball_red", "red", .38, -.435, .02)
    cube = scene_spec._cube(scene_spec.SHIPPED_CUBE, "red", .38, -.435, .03)
    check(not module.shipped_cube_is_configured([ball]),
          "a ball-only editor scene must not count the task USD cube as configured")
    check(module.shipped_cube_is_configured([cube]),
          "the task USD cube stays active when explicitly listed in the scene")
    source = path.read_text(encoding="utf-8")
    check("imageable.MakeInvisible()" in source
          and "CreateCollisionEnabledAttr().Set(False)" in source,
          "an unconfigured task cube must be removed from both vision and PhysX")


def test_experimental_gripper_pads_are_opt_in_and_radius_bounded():
    """The experimental USD must not alter the normal robot or accept unsafe pad sizes."""
    source = (ROOT / "11_real_sorting_with_monitor.py").read_text(encoding="utf-8")
    check("experimental_contact_pads: bool = False" in source,
          "the stock SO-101 USD remains the production default")
    check(gripper_contact_pads.PAD_RADIUS_M == 0.005
          and set(gripper_contact_pads.PAD_CENTRES_LOCAL_M) == {"gripper", "jaw"},
          "experimental pad geometry is explicit and attached to both calibrated finger links")
    with tempfile.TemporaryDirectory() as directory:
        source_usd = Path(directory) / "source.usd"
        source_usd.write_bytes(b"placeholder")
        rejected = 0
        for kwargs in ({"radius_m": 0.0005}, {"radius_m": 0.011},
                       {"shape": "unknown"}, {"shape": "vee", "vee_angle_deg": 50}):
            try:
                gripper_contact_pads.make_padded_robot_asset(source_usd, **kwargs)
            except ValueError:
                rejected += 1
        check(rejected == 4,
              "experimental pad radii outside the tested 1–10 mm bounds are rejected before USD import")
    check('choices=("sphere", "flat", "vee")' in
          (ROOT / "tools" / "benchmark_grasp_objects.py").read_text(encoding="utf-8"),
          "the V-groove remains available for isolated physical A/B benchmarks")
    check('"--experimental-contact-pad-angle-deg"' in
          (ROOT / "tools" / "benchmark_grasp_objects.py").read_text(encoding="utf-8"),
          "V-pad angle is controllable for repeatable A/B simulation")
    check('"--auto-validated-contact-pads"' in
          (ROOT / "tools" / "benchmark_grasp_objects.py").read_text(encoding="utf-8")
          and '"--grasp-profile-kind"' in
          (ROOT / "tools" / "benchmark_grasp_objects.py").read_text(encoding="utf-8"),
          "the grasp benchmark can exercise automatic and target-scoped validated profiles")
    check('"--contact-diagnostics-all-robot-links"' in
          (ROOT / "tools" / "benchmark_grasp_objects.py").read_text(encoding="utf-8"),
          "the grasp benchmark can isolate unintended contact with every arm link")
    check(gripper_contact_pads.vee_profile_for("cube", (.020, .020, .020), 0)
          and gripper_contact_pads.vee_profile_for("cube", (.020, .020, .020), 90)
          and gripper_contact_pads.vee_profile_for("ball", (.020, .020, .020), 217)
          and not gripper_contact_pads.vee_profile_for("cube", (.030, .030, .030), 0)
          and gripper_contact_pads.vee_profile_for("hexagonal_prism", (.020, .020, .018), 30)
          and gripper_contact_pads.vee_profile_for("hexagonal_prism", (.020, .020, .018), 0)
          and gripper_contact_pads.vee_profile_for("cylinder", (.020, .020, .015), 231)
          and gripper_contact_pads.vee_profile_for("triangular_prism", (.020, .020, .018), 30)
          and gripper_contact_pads.vee_profile_for("triangular_prism", (.020, .020, .018), 60)
          and gripper_contact_pads.vee_profile_for("triangular_prism", (.030, .030, .024), 0)
          and gripper_contact_pads.vee_profile_for("triangular_prism", (.030, .030, .024), 30)
          and gripper_contact_pads.vee_profile_for("triangular_prism", (.030, .030, .024), 60)
          and not gripper_contact_pads.vee_profile_for("hexagonal_prism", (.020, .020, .018), 15)
          , "V contact profiles include tested triangle sizes across yaw using wrist alignment")
    check(not gripper_contact_pads.scene_uses_only_vee_profiles([
              ("hexagonal_prism", (.020, .020, .018), 30),
              ("ball", (.030, .030, .030))])
          and gripper_contact_pads.scene_uses_only_vee_profiles([
              ("hexagonal_prism", (.020, .020, .018), 30),
              ("hexagonal_prism", (.020, .020, .018), 0)])
          and not gripper_contact_pads.scene_uses_only_vee_profiles([
              ("hexagonal_prism", (.020, .020, .018), 30),
              ("triangular_prism", (.030, .030, .024), 0)])
          and not gripper_contact_pads.scene_uses_only_vee_profiles([
              ("cube", (.020, .020, .020), 0),
              ("hexagonal_prism", (.020, .020, .018), 30)])
          and gripper_contact_pads.scene_uses_only_vee_profiles([
              ("hexagonal_prism", (.020, .020, .018), 30),
              ("hexagonal_prism", (.020, .020, .018), 90)]),
          "V pads require a validated compatible profile across all movable scene items")
    from types import SimpleNamespace
    check(gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="cube20", kind="cube", size=(.02, .02, .02),
                              yaw_deg=0.0, container=False)]) == "vee"
          and gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="ball20", kind="ball", size=(.02, .02, .02),
                              yaw_deg=0.0, container=False)]) == "vee"
          and gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="cylinder20", kind="cylinder", size=(.02, .02, .015),
                              yaw_deg=90.0, container=False)]) == "vee"
          and gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="triangle20", kind="triangular_prism", size=(.02, .02, .018),
                              yaw_deg=60.0, container=False)]) == "vee"
          and gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="cube30", kind="cube", size=(.03, .03, .03),
                              yaw_deg=0.0, container=False)]) is None
          and gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="tri", kind="triangular_prism", size=(.03, .03, .024),
                              yaw_deg=0.0, container=False)]) == "flat"
          and gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="tri", kind="triangular_prism", size=(.03, .03, .024),
                              yaw_deg=30.0, container=False),
              SimpleNamespace(prim="ball", kind="ball", size=(.03, .03, .03),
                              yaw_deg=0.0, container=False)]) is None
          and gripper_contact_pads.automatic_scene_pad_shape([
              SimpleNamespace(prim="hex1", kind="hexagonal_prism", size=(.02, .02, .018),
                              yaw_deg=30.0, container=False),
              SimpleNamespace(prim="tri", kind="triangular_prism", size=(.03, .03, .024),
              yaw_deg=0.0, container=False)]) is None,
          "the live scene rejects one shared V profile when validated geometries need different angles")
    mixed_cluster = [
        SimpleNamespace(prim="tri", kind="triangular_prism", size=(.03, .03, .024),
                        yaw_deg=0.0, container=False),
        SimpleNamespace(prim="distractor", kind="ball", size=(.03, .03, .03),
                        yaw_deg=0.0, container=False),
    ]
    check(gripper_contact_pads.automatic_scene_pad_shape(mixed_cluster) is None
          and gripper_contact_pads.automatic_scene_pad_shape(
              mixed_cluster, target_kind="triangular_prism") == "flat",
          "a validated target profile may ignore distant distractors only when target geometry is explicit")
    check(gripper_contact_pads.flat_profile_for("triangular_prism", (.030, .030, .024))
          and not gripper_contact_pads.flat_profile_for("triangular_prism", (.020, .020, .018))
          and not gripper_contact_pads.flat_profile_for("hexagonal_prism", (.030, .030, .024))
          and not gripper_contact_pads.flat_profile_for("cube", (.030, .030, .030)),
          "only the 30 mm triangular prism uses flat pads; the 20 mm triangle keeps its V45 profile")
    mixed_ball_sizes = [
        SimpleNamespace(prim="target_ball20", kind="ball", size=(.02, .02, .02),
                        yaw_deg=0.0, container=False),
        SimpleNamespace(prim="distractor_ball30", kind="ball", size=(.03, .03, .03),
                        yaw_deg=0.0, container=False),
    ]
    check(gripper_contact_pads.automatic_scene_pad_shape(
              mixed_ball_sizes, target_kind="ball") == "vee"
          and gripper_contact_pads.automatic_scene_pad_shape(mixed_ball_sizes) is None,
          "a validated 20 mm ball target can select its pads without inheriting a 30 mm distractor profile")
    check([item.prim for item in gripper_contact_pads.validated_profile_items(
              mixed_ball_sizes, target_kind="ball")] == ["target_ball20"],
          "the builder's final pad validation ignores only unvalidated target-size distractors")
    check(gripper_contact_pads.automatic_scene_grip_stiffness([
              SimpleNamespace(kind="hexagonal_prism", size=(.03, .03, .024),
                              container=False)], 250) == 500
          and gripper_contact_pads.automatic_scene_grip_stiffness([
              SimpleNamespace(kind="cylinder", size=(.03, .03, .024),
                              container=False)], 250) == 1000
          and gripper_contact_pads.automatic_scene_grip_stiffness([
              SimpleNamespace(kind="cylinder", size=(.03, .03, .024),
                              container=False),
              SimpleNamespace(kind="ball", size=(.03, .03, .03), container=False)],
              250, target_kind="cylinder") == 1000
          and gripper_contact_pads.automatic_scene_grip_stiffness([
              SimpleNamespace(kind="cylinder", size=(.03, .03, .024),
                              container=False),
              SimpleNamespace(kind="ball", size=(.03, .03, .03), container=False)], 250) == 250
          and gripper_contact_pads.automatic_scene_grip_stiffness([
              SimpleNamespace(kind="hexagonal_prism", size=(.03, .03, .024),
                              container=False),
              SimpleNamespace(kind="ball", size=(.03, .03, .03), container=False)], 250) == 250,
          "the 30 mm cylinder stiffness profile is target-scoped and mixed scenes stay unchanged")
    check(not gripper_contact_pads.vee_profile_for("hexagonal_prism", (.030, .030, .024))
          and not gripper_contact_pads.vee_profile_for("ball", (.030, .030, .030)),
          "the V profile stays disabled for the unvalidated 30 mm hexagon and ball controls")
    check(real_world.jaw_sweep_approach_offset(
              "hexagonal_prism", (.030, .030, .024)) == (0.0, 0.0)
          and real_world.jaw_sweep_approach_offset(
              "hexagonal_prism", (.020, .020, .018)) == (0.0, 0.0),
          "hexagons stay centred until the isolated offset validates in clustered scenes")
    demo = load_demo11()
    check(demo.aperture_gap_is_graspable(0.003)
          and demo.aperture_gap_is_graspable(.00310001, .0031)
          and not demo.aperture_gap_is_graspable(.00316, .0031)
          and not demo.aperture_gap_is_graspable(0.0032)
          and not demo.aperture_gap_is_graspable(None)
          and not demo.aperture_gap_is_graspable(float("nan"))
          and abs(demo.aperture_slack_for_size((.03, .03, .024)) - .0031) < 1e-9
          and abs(demo.aperture_slack_for_size((.02, .02, .018)) - .0081) < 1e-9,
          "the executor refuses contact outside the measured aperture slack")
    check(gripper_contact_pads.close_command_for("ball", (.030, .030, .030), .10) == .20
          and gripper_contact_pads.close_command_for(
              "ball", (.020, .020, .020), .20, pad_shape="vee") == .20
          and gripper_contact_pads.close_command_for(
              "ball", (.020, .020, .020), .20) == .20
          and gripper_contact_pads.close_command_for(
              "triangular_prism", (.030, .030, .024), .10) == .15
          and gripper_contact_pads.close_command_for(
              "hexagonal_prism", (.020, .020, .018), .20) == .20
          and gripper_contact_pads.close_command_for(
              "hexagonal_prism", (.030, .030, .024), .20) == .15,
          "object close targets match the measured profile for each size and geometry")
    check(gripper_contact_pads.close_command_for(
              "cylinder", (.020, .020, .015), .20, pad_shape="vee") == .20
          and gripper_contact_pads.close_command_for(
              "triangular_prism", (.020, .020, .018), .20, pad_shape="vee") == .10,
          "validated 20 mm V profiles use the measured cylinder and triangle close targets")
    check(gripper_contact_pads.close_command_for(
              "ball", (.020, .020, .020), .20, override=.10) == .10
          and gripper_contact_pads.close_command_for(
              "triangular_prism", (.030, .030, .024), .15, override=.12) == .12,
          "explicit close-angle experiments override the automatic per-object profile")
    check(gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.020, .020, .020), 90) == 45
          and gripper_contact_pads.staged_close_ramp_steps(
              "cylinder", (.020, .020, .020), 90) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "cylinder", (.030, .030, .030), 90) == 20
          and gripper_contact_pads.staged_close_ramp_steps(
              "hexagonal_prism", (.030, .030, .024), 90) == 20
          and gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.030, .030, .030), 90) == 45
          and gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.030, .030, .030), 90, ramp_steps=0) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.040, .040, .040), 90) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "cube", (.020, .020, .020), 90) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "cube", (.030, .030, .030), 90) == 45
          and gripper_contact_pads.staged_close_ramp_steps(
              "cube", (.040, .040, .040), 90) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.020, .020, .020), 1) == 0,
          "ball20, cylinder30, hexagon30 and cube30 use bounded staged closure with time left to settle")
    check(gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.020, .020, .020), 90, ramp_steps=45) == 45
          and gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.020, .020, .020), 90, ramp_steps=0) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "ball", (.020, .020, .020), 20, ramp_steps=45) == 19,
          "ball close timing can be varied safely for physical A/B trials")
    check(gripper_contact_pads.staged_close_ramp_steps(
              "triangular_prism", (.020, .020, .016), 90) == 20
          and gripper_contact_pads.staged_close_ramp_steps(
              "triangular_prism", (.030, .030, .024), 90) == 45
        and gripper_contact_pads.staged_close_ramp_steps(
            "hexagonal_prism", (.020, .020, .018), 90) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "triangular_prism", (.020, .020, .016), 90, ramp_steps=0) == 0
          and gripper_contact_pads.staged_close_ramp_steps(
              "triangular_prism", (.020, .020, .016), 90, ramp_steps=30) == 30
          and gripper_contact_pads.staged_close_ramp_steps(
              "triangular_prism", (.020, .020, .016), 1, ramp_steps=20) == 0,
          "small prism staged closure is bounded and configurable for physical A/B trials")
    check(gripper_contact_pads.tool_yaw_for("hexagonal_prism", (.030, .030, .024), 0) == 30
          and gripper_contact_pads.tool_yaw_for(
              "hexagonal_prism", (.030, .030, .024), 30) == 0
          and gripper_contact_pads.tool_yaw_for(
              "hexagonal_prism", (.030, .030, .024), 15) == 15
          and gripper_contact_pads.tool_yaw_for(
              "hexagonal_prism", (.030, .030, .024), 60) == 30
          and gripper_contact_pads.tool_yaw_for(
              "hexagonal_prism", (.020, .020, .018), 0) == 30
          and gripper_contact_pads.tool_yaw_for(
              "hexagonal_prism", (.020, .020, .018), 30) == 0
          and gripper_contact_pads.tool_yaw_for(
              "triangular_prism", (.030, .030, .024), 0) == 0
          and gripper_contact_pads.tool_yaw_for(
              "triangular_prism", (.030, .030, .024), 30) == 0
          and gripper_contact_pads.tool_yaw_for(
              "triangular_prism", (.030, .030, .024), 60) == 0
          and gripper_contact_pads.tool_yaw_for(
              "triangular_prism", (.020, .020, .018), 0) == 0
          and gripper_contact_pads.tool_yaw_for(
              "triangular_prism", (.020, .020, .018), 30) == 0
          and gripper_contact_pads.tool_yaw_for(
              "triangular_prism", (.020, .020, .018), 60) == 60
          and gripper_contact_pads.tool_yaw_for(
              "triangular_prism", (.020, .020, .018), 120) == 0
          and gripper_contact_pads.tool_yaw_for("cube", (.030, .030, .024), 30) == 30,
          "wrist alignment accounts for hexagon, triangle and cube symmetry before grasping")
    check(gripper_contact_pads.alternate_tool_yaw_for(
              "triangular_prism", (.030, .030, .024), 0) == 60
          and gripper_contact_pads.alternate_tool_yaw_for(
              "triangular_prism", (.030, .030, .024), 30) == 60
          and gripper_contact_pads.alternate_tool_yaw_for(
              "hexagonal_prism", (.030, .030, .024), 0) == 0
          and gripper_contact_pads.alternate_tool_yaw_for(
              "hexagonal_prism", (.030, .030, .024), 30) == -30
          and gripper_contact_pads.alternate_tool_yaw_for(
              "cube", (.030, .030, .024), 0) == 90
          and gripper_contact_pads.alternate_tool_yaw_for(
              "ball", (.030, .030, .030), 0) == 30
          and gripper_contact_pads.alternate_tool_yaw_for(
              "cylinder", (.030, .030, .030), 0) == 30
          and gripper_contact_pads.alternate_tool_yaw_for(
              "triangular_prism", (.020, .020, .018), 0) is None,
          "symmetric geometry gets a safe alternate wrist posture on a failed physical approach")
    check(gripper_contact_pads.tool_yaw_for_workspace(
              "triangular_prism", (.030, .030, .024), 0, .241) == 0
          and gripper_contact_pads.tool_yaw_for_workspace(
              "triangular_prism", (.030, .030, .024), 0, .253) == 60
          and gripper_contact_pads.tool_yaw_for_workspace(
              "hexagonal_prism", (.030, .030, .024), 0, .247) == 30
          and gripper_contact_pads.tool_yaw_for_workspace(
              "hexagonal_prism", (.030, .030, .024), 30, .260) == 0,
          "30 mm triangle contact orientation adapts to inner versus outer reach")
    check(gripper_contact_pads.tool_yaw_for_workspace(
              "cube", (.030, .030, .030), 0, .241) == 90
          and gripper_contact_pads.tool_yaw_for_workspace(
              "cube", (.030, .030, .030), 0, .220) == 0,
          "30 mm cubes use the physically observed reachable wrist pose at forward-pick reach")
    check(gripper_contact_pads.tool_yaw_for_view(
              "triangular_prism", (.030, .030, .024), 0, .254, "side") == 0
          and gripper_contact_pads.tool_yaw_for_view(
              "triangular_prism", (.030, .030, .024), 0, .254, "front") == 60,
          "side-camera 30 mm triangle keeps the neutral contact axis")
    check(gripper_contact_pads.retry_tool_yaw_for(60, 0, 60) == 0
          and gripper_contact_pads.retry_tool_yaw_for(60, 0, 0) == 60
          and gripper_contact_pads.retry_tool_yaw_for(30, 0, 30) == 0
          and gripper_contact_pads.retry_tool_yaw_for(30, 0, 0) == 30
          and gripper_contact_pads.retry_tool_yaw_for(90, 0, 90) == 0
          and gripper_contact_pads.retry_tool_yaw_for(90, 0, 0) == 90
          and gripper_contact_pads.retry_tool_yaw_for(0, None, 0) == 0,
          "successive retries alternate between two geometry-equivalent wrist postures")
    demo = load_demo11()
    from unittest import mock
    from types import SimpleNamespace
    executor = demo.StrictExecutor.__new__(demo.StrictExecutor)
    executor.layout = [SimpleNamespace(prim="cube", kind="cube", size=(.030, .030, .030),
                                       yaw_deg=0.0, container=False)]
    executor.camera_name = "top"
    executor.failures = {"cube": 1}
    executor._initial_object_grasp_yaw = {"cube": 90.0}
    executor._last_object_grasp_yaw = {"cube": 90.0}
    selected = []
    executor.set_tool_yaw = selected.append
    executor.log = lambda *_args: None
    with mock.patch.object(gripper_contact_pads, "tool_yaw_for_view", return_value=90.0):
        executor.set_object_grasp_yaw("cube")
    check(selected[-1] == 0.0,
          "a failed workspace-selected cube yaw retries the true alternate, not itself")
    executor._last_object_grasp_yaw = {"cube": 0.0}
    with mock.patch.object(gripper_contact_pads, "tool_yaw_for_view", return_value=90.0):
        executor.set_object_grasp_yaw("cube")
    check(selected[-1] == 90.0,
          "a subsequent cube retry returns to the workspace-reachable preferred yaw")
    check(gripper_contact_pads.carry_steps_for(
              "triangular_prism", (.030, .030, .024), 120) == 400
          and gripper_contact_pads.carry_steps_for(
              "hexagonal_prism", (.030, .030, .024), 120) == 120,
          "only the narrow-margin 30 mm triangular prism gets a 400-step carry")
    check(gripper_contact_pads.vee_angle_for("ball", (.020, .020, .020), 40) == 35
          and gripper_contact_pads.vee_angle_for("hexagonal_prism", (.020, .020, .018), 35, 30) == 40
          and gripper_contact_pads.vee_angle_for(
              "cube", (.020, .020, .020), 35, 0) == 30
          and gripper_contact_pads.vee_angle_for(
              "hexagonal_prism", (.020, .020, .018), 35, 0) == 40
          and gripper_contact_pads.vee_angle_for(
              "triangular_prism", (.020, .020, .016), 35, 0) == 45
          and gripper_contact_pads.vee_angle_for(
              "triangular_prism", (.030, .030, .024), 35, 0) == 45,
          "profiles select V35 for ball20, V30 for cube20, V45 for triangles, and V40 for validated prisms")
    check(gripper_contact_pads.pad_inset_for(
              "triangular_prism", (.030, .030, .024)) == .001
          and gripper_contact_pads.pad_span_for(
              "triangular_prism", (.030, .030, .024)) == .030
          and gripper_contact_pads.pad_inset_for(
              "cube", (.030, .030, .030)) == 0.0
          and gripper_contact_pads.pad_span_for(
              "cube", (.030, .030, .030)) == .018
          and gripper_contact_pads.pad_inset_for(
              "triangular_prism", (.030, .030, .024), "0") == 0.0,
          "the repeated 30 mm triangle contact profile uses a 1 mm inset and 30 mm span")
    check(gripper_contact_pads.vee_angle_for(
              "ball", (.020, .020, .020), 35, override_deg=40) == 40,
          "the ball V angle can be overridden for a controlled physical A/B experiment")
    try:
        gripper_contact_pads.vee_angle_for(
            "ball", (.020, .020, .020), 35, override_deg=80)
        valid_angle_override = False
    except ValueError:
        valid_angle_override = True
    check(valid_angle_override, "experimental V-angle overrides reject unsafe ranges")
    check(gripper_contact_pads.vee_stiffness_for(
              "cube", (.020, .020, .020), 250) == 500
          and gripper_contact_pads.vee_stiffness_for(
              "ball", (.020, .020, .020), 250) == 500
          and gripper_contact_pads.vee_stiffness_for(
              "hexagonal_prism", (.020, .020, .018), 250) == 500
          and gripper_contact_pads.vee_stiffness_for(
              "triangular_prism", (.030, .030, .024), 250) == 700
          and gripper_contact_pads.vee_stiffness_for(
              "cylinder", (.020, .020, .015), 250) == 500
          and gripper_contact_pads.vee_stiffness_for(
              "triangular_prism", (.020, .020, .018), 250) == 500
          and gripper_contact_pads.vee_stiffness_for(
              "ball", (.030, .030, .030), 250) == 250,
          "only validated V profiles raise the actuator stiffness")
    check(gripper_contact_pads.last_visual_correction_height(
              "cube", (.020, .020, .020), "vee", .20, 0) == .20
          and gripper_contact_pads.last_visual_correction_height(
              "hexagonal_prism", (.020, .020, .018), "vee", .20, 30) == .13
          and gripper_contact_pads.last_visual_correction_height(
              "hexagonal_prism", (.020, .020, .018), "vee", .20, 0) == .13
          and gripper_contact_pads.last_visual_correction_height(
              "hexagonal_prism", (.020, .020, .018), "vee", .19, 0) == .13
          and gripper_contact_pads.last_visual_correction_height(
              "triangular_prism", (.020, .020, .018), "vee", .19, 30) == .19
          and gripper_contact_pads.last_visual_correction_height(
              "triangular_prism", (.030, .030, .024), "vee", .19, 30) == .13
          and gripper_contact_pads.last_visual_correction_height(
              "hexagonal_prism", (.030, .030, .024), None, .20, 0) == .20
          and gripper_contact_pads.last_visual_correction_height(
              "ball", (.030, .030, .030), None, .20) == .20,
          "low visual correction is limited to validated geometry, size, yaw, and pad profile")


def test_simulator_assistance_is_the_reliable_demo_default():
    """The demo defaults to a labelled simulation grasp; strict mode remains available."""
    module = load_demo11()
    defaults = module.build_parser().parse_args(["--setup-only"])
    check(defaults.sim_assist_policy == "preemptive",
          "Demo 11 defaults to a contact-gated, labelled simulator grasp")
    strict = module.StrictExecutor.__new__(module.StrictExecutor)
    strict.allow_kinematic = False
    strict.auto_sim_assist = True
    strict.item_kinds = {"ball_red": "ball"}
    strict.auto_sim_assist_policy = "preemptive"
    check(strict._uses_pregrasp_assist("ball_red"),
          "the legacy policy still permits pre-contact simulation assistance")
    strict.auto_sim_assist_policy = "after_physical_miss"
    check(strict._uses_simulator_assist("ball_red")
          and not strict._uses_pregrasp_assist("ball_red"),
          "the explicit friction-first policy remains available for physical-grasp experiments")
    strict.auto_sim_assist = False
    check(not strict._uses_simulator_assist("ball_red"),
          "strict-physical mode continues to disable both assist timings")


def test_simulated_tracking_uses_scene_pose_after_camera_identity_is_resolved():
    import numpy as np
    import torch
    from types import SimpleNamespace

    module = load_demo11()
    strict = module.StrictExecutor.__new__(module.StrictExecutor)
    strict.allow_kinematic = False
    strict.auto_sim_assist = True
    strict.item_kinds = {"target_ball": "ball"}
    strict.item_sizes = {"target_ball": (.030, .030, .030)}
    strict.auto_sim_assist_policy = "preemptive"
    strict._sim_tracking_logged = set()
    strict.log = lambda _message: None
    strict.env = SimpleNamespace(scene={"target_ball": SimpleNamespace(
        data=SimpleNamespace(root_pos_w=torch.tensor([[.381, -.402, .0559]])))})
    vision = lambda _prim: (.379, -.400)
    check(np.allclose(strict.tracking_observation("target_ball", vision), (.381, -.402),
                      atol=1e-7),
          "the final simulation servo uses the selected prim pose through camera occlusion")
    strict.auto_sim_assist = False
    check(strict.tracking_observation("target_ball", vision) == (.379, -.400),
          "strict physical mode remains camera-localized")


def test_simulated_grasp_transform_preserves_the_object_pose_at_attachment():
    import math
    import numpy as np

    q90 = (math.sqrt(.5), 0.0, 0.0, math.sqrt(.5))
    q180 = (0.0, 0.0, 0.0, 1.0)
    local_pos, local_quat = executor_module.relative_pose_wxyz(
        (1.0, 0.0, 0.0), q90, (1.0, 2.0, 0.0), q180)
    check(np.allclose(local_pos, (2.0, 0.0, 0.0), atol=1e-8)
          and np.allclose(local_quat, (math.sqrt(.5), 0.0, 0.0, math.sqrt(.5)), atol=1e-8),
          "the grasp stores the object's exact relative position and orientation")


def test_the_fingertips_clear_the_rim():
    """The release height is a function of the fingers, not a constant someone liked."""
    tray = next(item for item in scene_spec.DEFAULT_SCENE if item.container)
    release = real_world.safe_release_z(tray)
    tips = release - real_world.GRIPPER.finger_tip_below_hold_m
    check(tips >= real_world.rim_z(tray) + real_world.GRIPPER.rim_clearance_m - 1e-9,
          f"the fingertips would be {tips * 1000:.0f} mm, inside a rim at "
          f"{real_world.rim_z(tray) * 1000:.0f} mm")
    check(release > planner_release_height(),
          "demo 11's release must be higher than demo 10's, which drives the jaws into the wall")
    # And the two heights differ by exactly "the finger depth, less the clearance we leave",
    # which is the whole argument in one subtraction.
    lift = (release - planner_release_height()
            - (real_world.GRIPPER.finger_tip_below_hold_m - real_world.GRIPPER.rim_clearance_m))
    check(abs(lift) < 1e-9,
          f"demo 11's release should be demo 10's plus the finger depth less the clearance, "
          f"off by {lift * 1000:.1f} mm")
    drop = real_world.drop_height_m(tray)
    check(0.0 < drop <= real_world.MAX_SAFE_DROP_M,
          f"the part falls {drop * 1000:.0f} mm, outside the {real_world.MAX_SAFE_DROP_M:.2f} m "
          f"a bouncing part can be trusted inside")
    shorter = real_world.recommended_tray()
    check(shorter.height < tray.height and shorter.inner[0] > tray.inner[0],
          "the recommended tray must be both wider and lower, and say why in its docstring")


def planner_release_height() -> float:
    """``planner.place_pose``'s z, computed from the scene spec rather than importing Isaac."""
    tray = next(item for item in scene_spec.DEFAULT_SCENE if item.container)
    return scene_spec.TABLE_TOP_Z + tray.height + 0.012


def test_a_placement_has_to_have_room_for_the_camera():
    """Cavity minus part minus the camera's error is the whole margin."""
    cube = next(item for item in scene_spec.DEFAULT_SCENE if item.kind == "cube")
    tray = next(item for item in scene_spec.DEFAULT_SCENE
                if item.container and item.color == cube.color)
    margin = real_world.placement_margin(cube, tray)
    check(not margin["ok"],
          f"the sim's own tray leaves only {margin['margin_m'] * 1000:.1f} mm; that is a finding, "
          f"not a pass")
    check(margin["margin_m"] < real_world.MIN_PLACEMENT_MARGIN_M,
          "0.5 mm of margin cannot be called enough")
    roomy = real_world.recommended_tray()
    better = real_world.placement_margin(cube, roomy)
    check(better["ok"] and better["margin_m"] > margin["margin_m"],
          f"the recommended tray should help: {better['margin_m'] * 1000:.1f} mm")
    # Half the error, twice the margin: the error is the only uncertain term in it.
    half = real_world.placement_margin(cube, tray, real_world.PERCEPTION_ERROR_M / 2.0)
    check(abs((half["margin_m"] - margin["margin_m"])
              - real_world.PERCEPTION_ERROR_M / 2.0) < 1e-9,
          "the margin must move with the camera's error")


def test_preflight_reports_the_findings_and_does_not_invent_them():
    findings = real_world.preflight(scene_spec.DEFAULT_SCENE)
    check(findings, "there are findings and they must be reported")
    check(all(isinstance(level, str) and isinstance(text, str) for level, text in findings),
          "findings are (level, text) pairs")
    levels = {level for level, _ in findings}
    check("fail" in levels, f"the 56 mm cavity is a failure at 12.5 mm of camera error: {levels}")
    check("note" in levels, "the assumptions must be reported as assumptions")
    check(any("green" in text for _level, text in findings),
          "the green part has no receptacle and must be said so")
    # A layout that is entirely out of reach must produce reach failures.
    far = [scene_spec.Item(**{**item.__dict__, "x": 0.15}) for item in scene_spec.DEFAULT_SCENE]
    check(any(level == "fail" and "reach" in text
              for level, text in real_world.preflight(far)), "out-of-reach parts must be caught")
    clean = real_world.preflight(scene_spec.DEFAULT_SCENE, camera="top")
    check(len(clean) == len(findings), "preflight must be deterministic")


def test_configured_scene_rejects_the_soa101_near_base_dead_zone():
    """A point inside max reach can still be impossible at the arm's folded limit."""
    import scene_config
    near_base = scene_spec.why_unreachable(.3723801, -.45121697,
                                           require_pick_envelope=True)
    check(near_base is not None and "minimum reach" in near_base,
          "a pose that left 36 mm physical tracking residual is outside the tested pick envelope")
    config = scene_config.default_config()
    target = next(row for row in config["objects"] if row["id"] == "cube")
    config["objects"] = [target]
    container = next(row for row in config["containers"] if row["id"] == "box_red")
    container["x"], container["y"] = .48, -.50
    config["containers"] = [container]
    target["x"], target["y"] = .35, -.48  # 160 mm from the base; inside the measured dead zone.
    try:
        scene_config.build(config)
    except ValueError as exc:
        check("mínimo" in str(exc).lower() or "minimum" in str(exc).lower()
              or "dead zone" in str(exc).lower(),
              f"the rejection should explain near-base reach, got: {exc}")
    else:
        check(False, "environment editor must reject object centres inside the near-base dead zone")
    target["x"], target["y"] = .419, -.454
    container["x"], container["y"] = .48, -.50
    try:
        scene_config.build(config)
    except ValueError as exc:
        check("pick envelope" in str(exc).lower() or "205 mm" in str(exc).lower(),
              f"scene validation should reject the saved 198 mm pick pose: {exc}")
    else:
        check(False, "scene validation must reject the saved cube pose inside the tested pick margin")
    facts = scene_spec.reach_facts(.35, -.48)
    check(facts["inside_dead_zone"] and not facts["beyond_arm"],
          "reach facts distinguish the conservative near-base pick zone from max reach")
    edge_facts = scene_spec.reach_facts(.419, -.454)
    edge_reason = scene_spec.why_unreachable(.419, -.454, require_pick_envelope=True)
    check(edge_facts["distance_m"] > scene_spec.MIN_REACH_M
          and edge_facts["outside_pick_envelope"] and edge_reason is not None,
          "a 198 mm pick pose is rejected by the tested 205 mm near-base envelope")
    import ground
    message = ground.unreachable_question([(.35, -.48)], {"kind": "ball", "color": "red"})["es"]
    check("zona cercana" in message and "Aléjalo" in message,
          f"the user should be told to move an object away from the base: {message}")


def test_the_jaws_close_far_enough_to_grip_anything():
    """The measured finding: LeIsaac's shipped close target grips nothing on this scene.

    ``tools/probe_grip_hold.py`` drove the real pick cycle at 0.20/0.30/0.40 rad and at three
    actuator stiffnesses: the cube followed the lift 100% at 0.20 rad and 0% at 0.30 and 0.40,
    and the ball needed the stiffer servo. The shipped target is 0.4 rad because upstream drops
    an 8 cm orange. These two checks keep that from quietly reverting.
    """
    check(real_world.GRIPPER.close_command_rad < 0.4,
          f"the close target must stay below the shipped 0.4 rad, which grips nothing: "
          f"{real_world.GRIPPER.close_command_rad}")
    check(real_world.GRIPPER.actuator_stiffness >= 250.0,
          f"a 10 g ball needs the stiffer servo: {real_world.GRIPPER.actuator_stiffness}")
    for item in scene_spec.DEFAULT_SCENE:
        if item.container:
            continue
        held = real_world.grasp(item)
        check(held.feasible,
              f"{item.prim} must survive the friction cone at the shipped friction: {held.reason}")


def test_the_setup_sheet_carries_the_measurements():
    sheet = "\n".join(real_world.setup_sheet(scene_spec.DEFAULT_SCENE))
    for needed in ("jaw aperture", "grip force", "finger tip", "camera offset", "RECOMMENDED",
                   "working area"):
        check(needed in sheet, f"the setup sheet must tell someone to measure {needed!r}")
    check("friction cone" in sheet, "the sheet must state the grip criterion")
    small_item = scene_spec.replace(scene_spec.DEFAULT_SCENE[0], size=(.020, .020, .020))
    small_sheet = "\n".join(real_world.setup_sheet([small_item]))
    check("20 x 20 x 20 mm" in small_sheet and "all 30 mm" not in small_sheet,
          "the setup sheet reports each configured size instead of claiming every object is 30 mm")


# -- the bus ---------------------------------------------------------------------------


def test_bus_publishes_a_consistent_snapshot():
    bus = telemetry.Bus()
    bus.node("plan", telemetry.ACTIVE, "planning")
    bus.log("a line", level="warn")
    bus.set_state(phase="command", verified=3)
    bus.put_frame(b"\xff\xd8jpeg", {"note": "annotated"})

    snap = bus.snapshot()
    check(len(snap["nodes"]) == len(telemetry.FLOW), "every node is in the snapshot")
    plan = next(node for node in snap["nodes"] if node["id"] == "plan")
    check(plan["state"] == telemetry.ACTIVE and plan["note"] == "planning",
          f"the node state should be the last one set: {plan}")
    check(snap["state"]["verified"] == 3, "state is merged, not replaced")
    check(any(event["level"] == "warn" for event in snap["events"]),
          "the log level must survive to the dashboard")
    json.dumps(snap)  # the websocket sends this verbatim, so it must be JSON already
    seq, jpeg, meta = bus.frame()
    check(jpeg == b"\xff\xd8jpeg" and meta["note"] == "annotated" and seq > 0,
          "the frame and its metadata must come back together")

    telemetry.reset_nodes(bus)
    check(all(node["state"] == telemetry.IDLE for node in bus.snapshot()["nodes"]),
          "reset_nodes must clear the diagram between commands")
    bus.node("verify", telemetry.OK)
    telemetry.reset_nodes(bus, {"verify": telemetry.OK})
    states = {node["id"]: node["state"] for node in bus.snapshot()["nodes"]}
    check(states["verify"] == telemetry.OK and states["plan"] == telemetry.IDLE,
          "reset_nodes must keep the states it is asked to keep")


def test_bus_wakes_a_listener_and_queues_commands():
    bus = telemetry.Bus()
    fired = threading.Event()

    def hook():
        fired.set()

    bus.add_hook(hook)
    bus.log("something happened")
    check(fired.wait(1.0), "a change must wake whoever is watching")

    bus.submit("  separa las piezas  ")
    check(bus.take_command(timeout=0.1) == "separa las piezas",
          "commands are stripped and handed over in order")
    check(bus.take_command(timeout=0.05) is None, "an empty queue returns None, not an error")

    bus.send_control("relook", why="testing")
    control = bus.take_control(timeout=0.1)
    check(control and control["action"] == "relook" and control["params"] == {"why": "testing"},
          f"a control action carries its parameters: {control}")

    check(not bus.wait_start(timeout=0.01), "the gate is shut until the browser opens it")
    bus.open_start()
    check(bus.wait_start(timeout=0.01), "and open afterwards")

    def hammer() -> None:
        for index in range(200):
            bus.log(f"line {index}", node="execute")

    thread = threading.Thread(target=hammer)
    thread.start()
    thread.join()
    check(len(bus.snapshot()["events"]) <= 400, "the event history is bounded")
    check(bus.snapshot()["nodes"], "and a concurrent hammer does not lose the diagram")


def test_flow_graph_is_consistent():
    ids = [node.id for node in telemetry.FLOW]
    check(len(ids) == len(set(ids)), "node ids are unique")
    check([node.row for node in telemetry.FLOW] == sorted(node.row for node in telemetry.FLOW),
          "the flow is ordered by row, so the diagram reads top to bottom")
    for source, target, _label in telemetry.EDGES:
        check(source in ids, f"edge names an unknown node: {source}")
        check(target in ids, f"edge names an unknown node: {target}")
    pairs = {(source, target) for source, target, _label in telemetry.EDGES}
    check(("recover", "capture") in pairs and ("recover", "plan") in pairs,
          "the recovery loop must point back at the camera and the planner, or the diagram "
          "cannot show that this system re-looks")
    check(("verify", "recover") in pairs and ("verify", "report") in pairs,
          "verification has two exits: everything landed, or something to recover")


# -- the dashboard's HTTP surface ------------------------------------------------------


def test_the_dashboard_answers_over_http():
    """A loopback socket, a snapshot and a queued command: the whole browser contract."""
    from monitor import server as monitor_server

    bus = telemetry.Bus()
    bus.node("verify", telemetry.ACTIVE, "checking")
    bus.put_frame(b"\xff\xd8\xff\xe0fake", {"note": "annotated"})

    monitor = monitor_server.serve(bus, port=0)
    try:
        check(monitor.wait_ready(READY := 15.0), "the dashboard must come up on port 0")
        check(monitor.port > 0, f"port 0 must resolve to the real port, got {monitor.port}")
        base = monitor.url

        with urllib.request.urlopen(base + "api/state", timeout=5) as response:
            snap = json.loads(response.read())
        check(len(snap["nodes"]) == len(telemetry.FLOW), "the browser gets the whole diagram")
        check(any(node["state"] == telemetry.ACTIVE for node in snap["nodes"]),
              "and the current stage in it")

        with urllib.request.urlopen(base + "api/frame.jpg", timeout=5) as response:
            check(response.headers.get_content_type() == "image/jpeg",
                  "the frame endpoint serves a JPEG")
            check(len(response.read()) > 0, "and it is not empty")

        request = urllib.request.Request(
            base + "api/command", data=json.dumps({"text": "pon el cubo rojo en la caja roja"})
            .encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            check(json.loads(response.read())["ok"], "a command must be accepted")
        check(bus.take_command(timeout=0.5) == "pon el cubo rojo en la caja roja",
              "and reach the robot's own queue, un-run")

        request = urllib.request.Request(
            base + "api/env", data=json.dumps({"action": "start"}).encode(),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            json.loads(response.read())
        check(bus.started(), "the start action must open the gate")

        request = urllib.request.Request(
            base + "api/env", data=json.dumps({"action": "relook"}).encode(),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            json.loads(response.read())
        actions = []
        while True:
            control = bus.take_control(timeout=0.5)
            if control is None:
                break
            actions.append(control.get("action"))
        # The start request itself occupies one slot (its params travel with it); the demo's
        # start-up drains it, and the robot's loop must still receive the relook.
        check(actions == ["start", "relook"], f"a live control must be queued, got {actions}")

        with urllib.request.urlopen(base, timeout=5) as response:
            page = response.read().decode("utf-8")
        check("text/html" in response.headers.get_content_type(), "the root serves HTML")
        for needed in ("ruta de decisi\u00f3n", "entorno", "c\u00e1mara", "registro",
                       "clasificaci"):
            check(needed in page, f"the dashboard is missing its {needed} panel")
    finally:
        monitor.stop(timeout=5.0)


def test_the_dashboard_transcribes_what_the_microphone_records():
    """The browser sends one WAV blob; a stand-in STT answers with the sentence.

    The WAV is real, at 16 kHz mono like the browser produces, so the path that has to survive
    contact with faster-whisper is also the path the test exercises.
    """
    import io as io_module
    import wave as wave_module

    from monitor import server as monitor_server

    bus = telemetry.Bus()
    seen = []

    def stand_in(payload: bytes) -> str:
        seen.append(payload)
        return "pon el cubo rojo en la caja roja"

    buf = io_module.BytesIO()
    with wave_module.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x10\x20" * 3200)  # 0.2 s of something
    blob = buf.getvalue()

    samples, rate = monitor_server.decode_wav(blob)
    check(rate == 16000, f"the blob is a 16 kHz wav, got {rate}")
    check(len(samples) == 3200 and abs(float(samples[0]) - 0x2010 / 32767.0) < 1e-4,
          "the decoder reads int16 pcm as float in [-1, 1]")

    monitor = monitor_server.serve(bus, port=0, stt=stand_in)
    try:
        check(monitor.wait_ready(READY := 15.0), "the dashboard with an STT seam must come up")
        base = monitor.url
        request = urllib.request.Request(
            base + "api/voice", data=blob, method="POST",
            headers={"content-type": "audio/wav"})
        with urllib.request.urlopen(request, timeout=10) as response:
            answer = json.loads(response.read())
        check(answer["ok"] and answer["text"] == "pon el cubo rojo en la caja roja",
              f"the dashboard must answer with the transcribed text, got {answer}")
        check(seen and seen[0] == blob, "and pass the same bytes to the transcriber")
    finally:
        monitor.stop(timeout=5.0)


def test_the_dashboard_reports_what_the_model_and_the_plan_said():
    """classification and plan travel as data, not only as log lines.

    Driven through ModelSpy and MonitoredRobot._publish_plan with stand-ins - no simulator,
    no real model - because the contract being pinned is what the bus carries.
    """
    import importlib.util as util

    spec = util.spec_from_file_location("demo11_report", ROOT /
                                        "11_real_sorting_with_monitor.py")
    module = util.module_from_spec(spec)
    sys.modules["demo11_report"] = module
    spec.loader.exec_module(module)

    bus = telemetry.Bus()

    class FakeModel:
        def parse(self, text, *args, **kwargs):
            return {"intent": "pick_and_place",
                    "items": [{"kind": "cube", "color": "blue", "count": 1,
                               "place": {"kind": "box", "color": "blue"}}],
                    "place": None, "sort_by": "none"}

    spy = module.ModelSpy(FakeModel(), bus)
    command = spy.parse("pon el cubo azul en la caja azul")
    snap = bus.snapshot()
    cls = snap["state"]["classification"]
    check(command["intent"] == "pick_and_place", "the parse still returns the command")
    check(cls["intent"] == "pick_and_place" and cls["items"][0]["kind"] == "cube",
          "the classification carries intent and items")
    check(cls["items"][0]["place"] == {"kind": "box", "color": "blue"},
          "and each item's own destination")
    check(any("cyan" not in str(node) for node in snap["nodes"]), "nothing invented")

    class FakeStep:
        def __init__(self, action, note, pose):
            self.action, self.note, self.pose = action, note, pose

    class FakePlan:
        intent = "pick_and_place"
        picks = [3]
        steps = [FakeStep("above", "pass to hover", (0.29, 0.06, 0.22))]
        moves = True
        # what grounding could not decide, as a real plan carries it
        question = {}
        unresolved = []

    spy_robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    spy_robot.bus = bus
    spy_robot._plan_n = 0
    spy_robot._publish_plan(FakePlan(), note="")
    plan = bus.snapshot()["state"]["plan"]
    check(plan["n"] == 1 and plan["steps"][0]["action"] == "above", "the plan is published")
    check(abs(plan["steps"][0]["pose"][2] - 0.22) < 1e-9, "the pose survives as floats")


def test_start_settings_travel_with_the_go_ahead():
    """The dashboard's variant/camera/rounds/seed override the CLI defaults at start-up."""
    import importlib.util as util

    spec = util.spec_from_file_location("demo11_start", ROOT /
                                        "11_real_sorting_with_monitor.py")
    module = util.module_from_spec(spec)
    sys.modules["demo11_start"] = module
    spec.loader.exec_module(module)

    bus = telemetry.Bus()
    bus.send_control("start", variant="open", camera="side", rounds=3, seed=7,
                     command="lleva el cilindro rojo al recipiente azul")
    bus.open_start()

    class NS:
        def __init__(self):
            self.variant = "contained"
            self.camera = "top"
            self.rounds = 1
            self.seed = 0

    args = NS()
    check(bus.started(), "the gate opens")
    module.apply_start_params(args, bus)
    check(args.variant == "open" and args.camera == "side", "variant and camera applied")
    check(args.rounds == 3 and args.seed == 7, "rounds and seed applied")
    check(args.text == ["lleva el cilindro rojo al recipiente azul"],
          "the startup command is queued to run after Isaac is ready")
    control = bus.take_control()
    check(control is None, "the start control was consumed, so keys() never sees it")


def test_startup_grasp_profile_requires_one_resolved_geometry():
    import importlib.util as util

    spec = util.spec_from_file_location("demo11_profile_inference", ROOT /
                                        "11_real_sorting_with_monitor.py")
    module = util.module_from_spec(spec)
    sys.modules["demo11_profile_inference"] = module
    spec.loader.exec_module(module)

    class Model:
        parsed = {}
        def parse(self, text):
            return self.parsed[text]

    model = Model()
    model.parsed = {
        "cilindro": {"intent": "pick", "items": [{"kind": "cylinder"}]},
        "otro cilindro": {"intent": "pick", "items": [{"kind": "cylinder"}]},
        "mezcla": {"intent": "sort", "items": [{"kind": "ball"}, {"kind": "cube"}]},
        "sin pieza": {"intent": "question", "items": []},
        "no resuelta": {"intent": "pick", "items": [{"kind": "any"}]},
    }
    check(module.infer_startup_grasp_profile(model, ["cilindro"]) == "cylinder",
          "a single resolved target geometry activates its validated profile")
    check(module.infer_startup_grasp_profile(model, ["cilindro", "otro cilindro"]) == "cylinder",
          "multiple startup commands keep the profile when they share one geometry")
    for commands in (["mezcla"], ["cilindro", "mezcla"], ["sin pieza"], ["no resuelta"]):
        check(module.infer_startup_grasp_profile(model, commands) is None,
              "mixed, empty, or unresolved commands stay on the general profile")

    from intent_parser import HybridParser
    spanish = HybridParser()
    check(module.infer_startup_grasp_profile(
        spanish, ["mueve el cilindro verde a la caja roja"]) == "cylinder",
        "the real Spanish rules parser selects the cylinder profile from a valid task")
    check(module.infer_startup_grasp_profile(
        spanish, ["mueve la bola roja a la caja azul"]) == "ball",
        "the real Spanish rules parser resolves a ball task")
    check(module.infer_startup_grasp_profile(
        spanish, ["mueve el prisma triangular azul a la caja roja"]) == "triangular_prism",
        "the real Spanish rules parser resolves a triangular-prism task")


def test_hybrid_intent_parser_falls_back_to_rules_when_torch_is_missing():
    """A missing optional model dependency must not prevent Isaac from starting in hybrid mode."""
    module = load_demo11()

    def missing_torch():
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    parser, degraded = module.initialize_intent_parser("hybrid", [], missing_torch)
    parsed = parser.parse("mueve el cilindro verde en la caja roja")
    check(degraded and parsed["parser"] == "rules"
          and parsed["items"][0]["kind"] == "cylinder"
          and parsed["items"][0]["color"] == "green"
          and parsed["place"]["color"] == "red",
          "hybrid mode starts with explicit color-and-shape commands using the rule parser")
    try:
        module.initialize_intent_parser("model", [], missing_torch)
    except ModuleNotFoundError as exc:
        check(exc.name == "torch", "an explicitly requested model backend reports missing torch")
    else:
        check(False, "an explicitly requested model backend must not silently degrade")


def test_executor_assists_all_supported_simulated_shapes_and_labels_them():
    """Simulation assist covers every demo shape; friction trials can still disable it."""
    check(hasattr(executor_module.Executor, "grab"),
          "demo 10's executor must still have the hook this demo overrides")
    import importlib.util as util

    spec = util.spec_from_file_location("demo11_under_test", ROOT /
                                        "11_real_sorting_with_monitor.py")
    # Imported for the class definition only; the module body is guarded by __main__.
    module = util.module_from_spec(spec)
    sys.modules["demo11_under_test"] = module
    spec.loader.exec_module(module)

    strict = module.StrictExecutor.__new__(module.StrictExecutor)
    strict.allow_kinematic = False
    strict.refused = 0
    strict.kinematic = 0
    strict.item_kinds = {"cube": "cube", "ball": "ball", "cyl": "cylinder", "tri": "triangular_prism",
                         "hex": "hexagonal_prism"}

    from unittest import mock

    held = []
    with mock.patch.object(module.executor_module.Executor, "grab",
                           lambda self, prim: held.append(prim)):
        strict.grab("ball")
        strict.grab("cyl")
        strict.grab("tri")
        strict.grab("hex")
        strict.grab("cube")
    check(held == ["ball", "cyl", "tri", "hex", "cube"] and strict.kinematic == 5,
          "all supported shapes use the labeled simulator assist by default")
    check(module.LATERAL_SLACK_M < module.SIM_ASSIST_MAX_LATERAL_M <= 0.030,
          "sim assist can absorb bounded residuals without changing physical-grasp slack")
    check(module.grasp_height_error(.101, .193, -.092) < 0.001,
          "sim assist compares object height with the calibrated hold point, not body origin")
    check(abs(module.grasp_height_error(.101, .193, 0.0) - .092) < 1e-9,
          "the regression fixture reproduces the false 92 mm body-frame error")
    calibrated = module.calibrated_grasp_offset((.007, 0.0, -.092), (-.009, .012, .010))
    check(calibrated == (-.009, .012, -.092),
          "the grasp offset uses measured jaw-centre X/Y but keeps its fitted contact height")
    strict.auto_sim_assist = False
    try:
        strict.grab("ball")
    except module.GraspRefused:
        pass
    check(held == ["ball", "cyl", "tri", "hex", "cube"],
          "the strict benchmark can disable default simulator assistance")
    strict.allow_kinematic = True
    with mock.patch.object(module.executor_module.Executor, "grab",
                           lambda self, prim: held.append(prim)):
        strict.grab("cube")
    check(held[-1] == "cube" and strict.kinematic == 6,
          "--allow-kinematic enables the simulator assist for cubes too")


def test_grasp_width_uses_projected_shape_and_calibrated_pad_axis():
    """Prism feasibility follows the actual jaw axis, not the diameter field alone."""
    import math
    import numpy as np
    import torch
    from types import SimpleNamespace

    module = load_demo11()
    q0 = (1.0, 0.0, 0.0, 0.0)
    q30 = (math.cos(math.radians(15)), 0.0, 0.0, math.sin(math.radians(15)))
    width_flat = module.projected_grip_width(
        "triangular_prism", (.040, .040, .024), q0, (1.0, 0.0))
    width_rotated = module.projected_grip_width(
        "triangular_prism", (.040, .040, .024), q30, (1.0, 0.0))
    check(abs(width_flat - .030) < 1e-6,
          "a 40 mm triangular prism aligned face-to-vertex presents 30 mm across X")
    check(abs(width_rotated - .034641016) < 1e-6,
          "the same prism presents its wider projected footprint after 30 degree rotation")
    check(abs(module.projected_grip_width(
        "cube", (.030, .020, .010), q30, (1.0, 0.0)) - .035980762) < 1e-6,
          "a rectangular cube uses both dimensions and object yaw for aperture checks")

    strict = module.StrictExecutor.__new__(module.StrictExecutor)
    strict.items_by_prim = {"tri40": .040}
    strict.item_sizes = {"tri40": (.040, .040, .024)}
    strict.item_kinds = {"tri40": "triangular_prism"}
    strict.robot = SimpleNamespace(body_names=["gripper", "jaw"])
    strict.finger_contact_points = lambda: {
        "gripper": np.array((.368, -.435, .05)),
        "jaw": np.array((.391, -.435, .05)),
    }
    strict.aperture_centre = lambda: (.3795, -.435, .05)
    strict.env = SimpleNamespace(scene={
        "tri40": SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[.38, -.435, .0529]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])))})
    geometry = strict.part_geometry("tri40")
    check(geometry["fits"] and abs(geometry["grip_width_m"] - .030) < 1e-6,
          "a 40 mm prism is feasible when its actual yaw presents a 30 mm jaw width")


def test_simulator_assisted_lift_does_not_use_the_physical_friction_threshold():
    from unittest import mock
    import numpy as np

    module = load_demo11()
    strict = module.StrictExecutor.__new__(module.StrictExecutor)
    strict._pregrasp_kinematic = "cube_red"
    strict._held = "cube_red"
    strict.frame_hook = None
    strict.log = lambda _message: None
    recorded = {}

    def base_do(_self, *args, **kwargs):
        recorded.update(kwargs)
        return executor_module.StepResult(action=args[0], note="", ok=True)

    with mock.patch.object(executor_module.Executor, "do", base_do):
        result = strict.do("lift", np.array([.38, -.45, .17]), .2,
                           prim="cube_red", on_close=np.array([.38, -.45, .057]),
                           check="follow")
    check(recorded.get("on_close") is None and recorded.get("check") is None,
          "a simulated constraint is not judged by the physical vertical-only displacement test")
    check(result.ok and result.grasp == "kinematic" and "contact constraint" in result.note,
          "the held part is reported as simulator-assisted through lift")


def test_simulator_assisted_close_skips_only_the_transient_physical_reach_check():
    from unittest import mock
    import numpy as np

    module = load_demo11()
    strict = module.StrictExecutor.__new__(module.StrictExecutor)
    strict.allow_kinematic = False
    strict.auto_sim_assist = True
    strict.item_kinds = {"cube_red": "cube"}
    strict.auto_sim_assist_policy = "preemptive"
    strict.kinematic = 0
    strict._held = None
    strict._pregrasp_kinematic = None
    strict.frame_hook = None
    strict.log = lambda _message: None
    geometry = {"fits": True, "lateral_m": .005, "z": .0559}
    strict.part_geometry = lambda _prim: geometry
    captured = {}

    def base_do(_self, *args, **kwargs):
        captured.update(kwargs)
        return executor_module.StepResult(action=args[0], note="", ok=True)

    with mock.patch.object(executor_module.Executor, "do", base_do), \
            mock.patch.object(executor_module.Executor, "grab",
                              lambda self, prim: setattr(self, "_held", prim)):
        result = strict.do("close", np.array([.38, -.45, .0559]), .2,
                           prim="cube_red", check="reach")
    check(captured.get("check") is None and strict.holding == "cube_red",
          "the bounded assisted close does not fail on the pre-carry contact transient")
    check(result.ok, "and it leaves the ordinary physical close path untouched")


def test_simulator_assist_lateral_gate_covers_measured_camera_residuals_only():
    from unittest import mock
    import numpy as np

    module = load_demo11()
    strict = module.StrictExecutor.__new__(module.StrictExecutor)
    strict.allow_kinematic = False
    strict.auto_sim_assist = True
    strict.item_kinds = {"cube_red": "cube"}
    strict.auto_sim_assist_policy = "preemptive"
    strict.kinematic = 0
    strict._held = None
    strict._pregrasp_kinematic = None
    strict.frame_hook = None
    strict.log = lambda _message: None
    lateral = {"value": .029}
    strict.part_geometry = lambda _prim: {
        "fits": True, "lateral_m": lateral["value"], "z": .0559}
    with mock.patch.object(executor_module.Executor, "do",
                           lambda self, *args, **kwargs:
                           executor_module.StepResult(action=args[0], note="", ok=True)), \
            mock.patch.object(executor_module.Executor, "grab",
                              lambda self, prim: setattr(self, "_held", prim)):
        strict.do("close", np.array([.38, -.45, .0559]), .2, prim="cube_red")
        check(strict.holding == "cube_red",
              "the simulation assist accepts the empirically observed 29 mm residual")
        strict._held = None
        strict._pregrasp_kinematic = None
        lateral["value"] = .031
        strict.do("close", np.array([.38, -.45, .0559]), .2, prim="cube_red")
        check(strict.holding is None,
              "the simulation assist remains capped and refuses residuals above 30 mm")
        strict.auto_sim_assist = False
        lateral["value"] = .029
        strict.do("close", np.array([.38, -.45, .0559]), .2, prim="cube_red")
        check(strict.holding is None,
              "strict physical mode still refuses simulator assistance below the sim-only ceiling")


def test_a_part_measured_ungraspable_stops_being_planned():
    """Once the close-time geometry says the grip is impossible, the re-plans stop including it.

    This is the measured attempt-storm from out11's logs: five descents, five refusals, a part
    tipping further onto its edge each time, and a sort that ended 0 of 3. The strip is
    demonstrated here against a stand-in scene - no simulator needed.
    """
    import types

    import copy as copy_module
    from types import SimpleNamespace

    import importlib.util as util

    spec = util.spec_from_file_location("demo11_impossible", ROOT /
                                        "11_real_sorting_with_monitor.py")
    module = util.module_from_spec(spec)
    sys.modules["demo11_impossible"] = module
    spec.loader.exec_module(module)

    layout = [SimpleNamespace(container=True, prim="tray_blue", x=0.30, y=-0.51),
              SimpleNamespace(container=False, prim="cube_blue", x=0.449, y=-0.448),
              SimpleNamespace(container=False, prim="cube_red", x=0.281, y=-0.448),
              SimpleNamespace(container=False, prim="ball_red", x=0.364, y=-0.451)]
    scene = SimpleNamespace(objects=[
        SimpleNamespace(color="blue", kind="cube", pos_m=(0.448, -0.448, 0.015), id=2),
        SimpleNamespace(color="red", kind="ball", pos_m=(0.364, -0.451, 0.015), id=4),
        SimpleNamespace(color="red", kind="cube", pos_m=(0.281, -0.448, 0.015), id=3)],
        containers=[])
    before = copy_module.deepcopy(scene)

    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = layout
    robot.bot = SimpleNamespace(impossible=set())
    robot.say = lambda text: None
    check(robot.before_plan(object(), scene, (0.0, 0.0)) is scene,
          "with nothing impossible, the scene passes through untouched")

    robot.bot.impossible.add("cube_blue")
    shaped = robot.before_plan(object(), scene, (0.0, 0.0))
    kept_kinds = sorted(str(getattr(thing, "kind", "?")) for thing in shaped.objects if
                        hasattr(thing, "kind"))
    check(kept_kinds == ["ball", "cube"] or len(shaped.objects) == 2,
          "the impossible part's blob was taken out")
    check(copy_module.deepcopy(scene.containers) == before.containers,
          "containers are never touched by the strip")
    check(len(scene.objects) == 3 and scene is not before,
          "the original scene object is left alone for the robot's own link()")

    # The marginal case too: two refusals without the geometry saying impossible was the
    # measured pattern that kept the sort looping on one cube.
    fresh = copy_module.deepcopy(before)
    robot.bot.impossible.clear()
    robot.bot.failures = {"cube_red": module.MAX_PICK_FAILURES}
    fresher = robot.before_plan(object(), fresh, (0.0, 0.0))
    reds_left = [thing for thing in fresher.objects
                 if getattr(thing, "color", "") == "red" and getattr(thing, "kind", "") == "cube"]
    check(not reds_left, f"a part refused twice stops being planned, margin or not "
                         f"(left: {[(getattr(t, 'kind', None), getattr(t, 'color', None)) for t in reds_left]}")
    check(any(getattr(thing, "color", "") == "red" for thing in fresher.objects),
          "the other red is still planned")


def test_the_retry_masks_the_arm():
    """The bug that kept a retry blind: the unmasked gripper clipped the blob from above."""
    import importlib.util as util

    spec = util.spec_from_file_location("demo11_retry", ROOT /
                                        "11_real_sorting_with_monitor.py")
    module = util.module_from_spec(spec)
    sys.modules["demo11_retry"] = module
    spec.loader.exec_module(module)

    from unittest import mock

    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.bus = telemetry.Bus()
    robot.mask_arm = False
    seen = []

    def fake_retry(self, prim):
        seen.append(self.mask_arm)
        return True

    with mock.patch.object(module.MonitoredRobot, "_retry_placed", fake_retry):
        result = robot._retry_one("cube_blue")
    check(seen == [True], "the retry ran with the arm masked")
    check(robot.mask_arm is False, "and restored the setting afterwards")
    check(result is True, "and reports what the retry did")


def test_the_websocket_answers_immediately_and_does_not_run_away():
    """The two ways the live socket used to be useless, both measured.

    **A connection that is never spoken to looks exactly like a dead server.** The dashboard is
    meant to be open before the simulator, and that is when nothing is happening - ``main``
    publishes the bench sheet and then waits for somebody to press start. The handler used to
    speak only when the bus changed, so for that whole wait the page was blank: it read as
    localhost not working, while ``/api/state`` was answering the entire time.

    **One bus change used to become thousands of snapshots per second.** The handler waited on
    an ``asyncio.Event`` nothing ever cleared, so once set it stayed set and each new waiter
    returned at once. Measured with a client connected and nothing else happening: 2171 pushes
    per second, forever, from one ``bus.node`` call. That froze the tab and burned a whole core
    of the process that is also running PhysX - 5.97 CPU-seconds in 6.02 seconds of wall clock,
    and a stand-in physics step from 40822 to 17259 per second on the GIL.

    So: connect and be answered, change something and be answered once, and then go quiet. The
    last assertion is the one that would catch the spin coming back - a correct implementation
    sends nothing at all once the bus stops moving.
    """
    import asyncio

    from monitor import server as monitor_server

    bus = telemetry.Bus()
    for index in range(140):  # a full log, so the snapshots are the size real ones are
        bus.log(f"  a line of the kind the robot prints while it works, number {index}")
    bus.put_frame(b"\xff\xd8\xff\xe0" + b"x" * 2000, {"note": "en vivo", "annotated": False})

    monitor = monitor_server.serve(bus, port=0)
    try:
        check(monitor.wait_ready(15.0), "the dashboard must come up on port 0")

        async def session(trigger, want=None):
            """What one browser sees: what arrives, and when.

            ``want`` stops as soon as that many SNAPSHOTS have arrived, so a session that expects
            one reply does not sit out the whole window waiting to be sure no second one comes.
            It counts snapshots rather than messages because one push is three messages - the
            snapshot, the frame's own header, and the JPEG - so counting messages would stop on the
            frame and never see the snapshot the test is actually asking about.

            The window itself is the guard that matters: it is how "and then it went quiet" becomes
            a measurement rather than an absence.
            """
            import websockets

            url = f"ws://{monitor.host}:{monitor.port}/ws"
            arrived = []
            snapshots = 0
            async with websockets.connect(url, open_timeout=10) as socket:
                if trigger is not None:
                    trigger()
                deadline = time.monotonic() + QUIET_WINDOW_S
                while time.monotonic() < deadline:
                    try:
                        message = await asyncio.wait_for(socket.recv(), timeout=QUIET_READ_S)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(message, bytes):
                        arrived.append(("bytes", len(message)))
                    else:
                        payload = json.loads(message)
                        arrived.append((payload.get("type"), payload.get("seq")))
                        if payload.get("type") == "snapshot":
                            snapshots += 1
                    if want is not None and snapshots >= want:
                        break
            return arrived

        quiet = asyncio.run(session(None, want=3))
        check(quiet and quiet[0][0] == "snapshot",
              f"a connection must be answered without waiting for a change, got {quiet}")
        # the JPEG follows the snapshot that announced it, so three messages is the whole exchange
        check(("bytes", 2004) in quiet,
              f"and the picture already published must come with it, got {quiet}")

        # The connect snapshot plus the one the change causes.
        after_one = asyncio.run(
            session(lambda: bus.node("setup", telemetry.OK, "started"), want=2))
        snapshots = [row for row in after_one if row[0] == "snapshot"]
        check(len(snapshots) <= 2,
              f"one change must not turn into a flood; got {len(snapshots)} snapshots")
        check(len(snapshots) == 2,
              f"the change itself must still be pushed on top of the connect one, got "
              f"{len(snapshots)}")

        # The spin would show up here as hundreds of pushes inside one floor interval.
        burst = asyncio.run(
            session(lambda: [bus.node("execute", telemetry.ACTIVE, str(i))
                             for i in range(100)], want=3))
        burst_snapshots = [row for row in burst if row[0] == "snapshot"]
        check(len(burst_snapshots) <= 3,
              f"a hundred writes must coalesce into one push, got {len(burst_snapshots)}")
    finally:
        monitor.stop(timeout=5.0)


def test_a_snapshot_says_whether_it_is_different_from_the_last_one():
    """``rev`` must move on every change, not only when a line is logged.

    The dashboard skips a repaint when the ``rev`` it was last given is the one it has now, so a
    ``rev`` that misses a change leaves a stale number on screen. ``set_state`` and ``put_frame``
    are the two that add no log line and used to be invisible to such a signature: a frame
    arriving and the phase moving from *ready* to *scene* are both real changes to the picture.
    """
    bus = telemetry.Bus()

    before = bus.snapshot()["rev"]
    bus.set_state(phase="ready", sim="running")
    after_state = bus.snapshot()["rev"]
    check(after_state > before, "a state merge must count as a change")

    bus.put_frame(b"\xff\xd8\xff\xe0fake", {"note": "en vivo"})
    after_frame = bus.snapshot()["rev"]
    check(after_frame > after_state, "a camera frame must count as a change too")

    quiet = bus.snapshot()["rev"]
    check(quiet == after_frame, "and a snapshot that changes nothing must not move it")


def load_demo11():
    """Import demo 11 by path, once, for the tests that drive its classes directly.

    The module body is guarded by ``__main__`` so importing it starts nothing; it registers
    itself in ``sys.modules`` because it defines dataclasses that resolve their own module at
    class-creation time.
    """
    import importlib.util as util

    if "demo11_under_test" in sys.modules:
        return sys.modules["demo11_under_test"]
    spec = util.spec_from_file_location("demo11_under_test", ROOT /
                                        "11_real_sorting_with_monitor.py")
    module = util.module_from_spec(spec)
    sys.modules["demo11_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_contact_diagnostics_enable_reporters_on_robot_and_target():
    from types import SimpleNamespace

    module = load_demo11()
    robot_spawn = SimpleNamespace(activate_contact_sensors=False)
    target_spawn = SimpleNamespace(activate_contact_sensors=False)
    module.enable_contact_reporting(robot_spawn, target_spawn)
    check(0.0 < robot_spawn.activate_contact_sensors < 1e-3,
          "robot contact reports use a sub-millinewton threshold, not bool True's 1 N")
    check(0.0 < target_spawn.activate_contact_sensors < 1e-3,
          "target contact reports use the same low force threshold")

    fallback_robot = SimpleNamespace(activate_contact_sensors=False)
    module.enable_contact_reporting(fallback_robot)
    check(0.0 < fallback_robot.activate_contact_sensors < 1e-3,
          "scene-USD targets still enable contact reporters on the robot")


def test_a_command_typed_in_the_browser_reaches_the_robot():
    """The three ways a sentence gets in, and the two that used to be dropped on the floor.

    Nothing was ever lost - the HTTP endpoint put the sentence on the bus every time, verified
    here from the browser's side - but the robot's own wait was where it died:

    * **A dashboard-only run quit instead of waiting.** ``take_command(timeout=None)`` means
      "do not block" in ``Bus``, so an empty poll returned the literal string ``"quit"``, which
      ``handle`` reads as the quit word and stops on. Measured on the real method: ``"quit"`` in
      0.2 s, every single time, for a person who had simply not typed yet.
    * **In a terminal run the browser waited behind the keyboard.** The 0.2 s poll came first
      and then the robot fell into ``input()``, so the browser's sentence sat there until
      somebody also pressed Enter. Measured: the terminal's answer came back while the
      browser's own command was still queued.

    So ``ask`` now waits for all three inputs through one queue, and answers the panel's
    ``stop``/``next`` while it is waiting - ``keys()`` cannot run there, because there are no
    executor steps to run between them.
    """
    import importlib.util
    import threading

    module = load_demo11()
    demo10 = module.load_demo10()
    demo10.Robot.ask = lambda self: "FROM_TERMINAL"   # do not block on input() in a test

    class Probe(module.MonitoredRobot):
        def __init__(self, web_input, voice=None, keyboard=None):
            self.bus = telemetry.Bus()
            self.voice = voice
            self.web_input = web_input
            self.keyboard = keyboard
            self.advance = False
            self.say = lambda *a, **k: None

    class Voice:
        def listen(self):
            # A real listen() records for seconds. This only has to be long enough that the
            # browser command below arrives while the microphone has the floor, and `ask` polls
            # every WEB_COMMAND_POLL_S, so 0.05 s is inside that window.
            time.sleep(0.05)
            return ""

    def submit_later(bus, text, delay=0.05):
        """Queue a command while ``ask`` is already waiting for one.

        The delay is short because ``Queue.get(timeout=...)`` returns the moment something is put,
        not when the timeout expires. So a submit at 0.05 s is answered at 0.05 s even though the
        wait asked for 0.2 s - the sleep only has to be long enough to be *during* the wait, not to
        outlast it. That is the whole difference between this test taking 1.8 s and 0.3 s.
        """
        def go():
            time.sleep(delay)
            bus.submit(text)
        threading.Thread(target=go, daemon=True).start()

    def ask_now(probe, timeout=6.0):
        """ask() in a thread, so a test can fail on a wait instead of hanging on one."""
        box = {}

        def go():
            box["text"] = module.MonitoredRobot.ask(probe)
        thread = threading.Thread(target=go, daemon=True)
        thread.start()
        thread.join(timeout)
        return box.get("text", "<STILL WAITING>")

    # A browser command that is already there comes straight back.
    probe = Probe(web_input=False, keyboard=True)
    probe.bus.submit("separa las piezas por colores")
    check(ask_now(probe) == "separa las piezas por colores",
          "a queued browser command must be what ask() returns")

    # Dashboard-only, nothing queued yet: it must WAIT, not quit.
    probe = Probe(web_input=True)
    submit_later(probe.bus, "coge la bola verde")
    check(ask_now(probe) == "coge la bola verde",
          "with no keyboard the dashboard must be waited on, not given up on")

    # Dashboard-only, and the person presses stop while it waits.
    probe = Probe(web_input=True)
    threading.Timer(0.05, lambda: probe.bus.send_control("stop")).start()
    check(ask_now(probe) == "quit",
          "the panel's stop must reach ask() even with no executor steps running")

    # Dashboard-only, and the person asks for the next round while it waits.
    probe = Probe(web_input=True)
    threading.Timer(0.05, lambda: probe.bus.send_control("next")).start()
    check(ask_now(probe) == "" and probe.advance,
          "the panel's next round must break the round, not run a command nobody sent")

    # With --voice the microphone has the floor for a moment; the browser is checked after it.
    probe = Probe(web_input=False, voice=Voice(), keyboard=True)
    submit_later(probe.bus, "pon el cubo rojo en la caja roja", delay=0.02)
    check(ask_now(probe) == "pon el cubo rojo en la caja roja",
          "a browser command typed during listen() must not be dropped")


def test_the_panel_can_see_what_is_queued():
    """The queue has to be visible, or a queued command is indistinguishable from a lost one.

    The page sends the text, the HTTP call returns, and from that moment nothing on screen says
    anything until the robot's loop reaches the queue - and the robot is usually mid-pick. That
    silence is the whole of "I pressed send and nothing happened", so ``submit`` publishes the
    sentence, and where it came from.
    """
    bus = telemetry.Bus()
    bus.submit("separa las piezas por colores")
    snap = bus.snapshot()["state"]
    check(snap.get("queued") == "separa las piezas por colores",
          f"a submitted command must be visible in the state, got {snap.get('queued')!r}")
    check(snap.get("queued_from") == "dashboard", "and it must say it came from the dashboard")

    bus.submit("coge la bola verde", source="terminal")
    check(bus.snapshot()["state"].get("queued_from") == "terminal",
          "a command from the keyboard reader must be distinguishable from the panel's")
    check(bus.command_source() == "", "reading nothing must not report a source")

    check(bus.take_command(timeout=0.5) == "separa las piezas por colores", "first out first")
    check(bus.command_source() == "dashboard", "and its source comes back with it")
    check(bus.take_command(timeout=0.5) == "coge la bola verde", "then the second")
    check(bus.command_source() == "terminal", "with its own source")

    check(not bus.control_pending(), "an empty control queue must not look pending")
    bus.send_control("relook")
    check(bus.control_pending(), "a queued control must be visible without being consumed")
    check(bus.take_control().get("action") == "relook", "and still be there to be taken")


def test_the_dashboard_has_no_id_it_writes_to_and_does_not_declare():
    """Every id the page's script touches has to exist, and be declared only once.

    Two real panels died on exactly this and nothing said so, because a ``catch`` around the
    painter turned the failure into silence:

    * ``paintPlan`` wrote to ``#plan-n``, which the page never declared. The throw came before
      the steps table, the note and the list of parts were filled, so the plan card showed its
      placeholder for ever.
    * ``paintNodes`` wrote the live note of all twelve boxes into ``painted.live``, but
      ``buildFlow`` only makes that element for the boxes with no content of their own -
      ``input``, ``understand`` and ``plan`` - so it was null from the second box onwards.

    Checked here rather than in the browser because the ids are all in the file, and the first
    version of this check only looked at ``el("x")``, which is how ``#plan-n`` slipped through:
    ``text("plan-n", ...)`` reaches the same place one call later.
    """
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    script = page[page.index("<script>"):]

    declared = re.findall(r'id="([^"]+)"', page)
    duplicated = sorted({i for i in declared if declared.count(i) > 1})

    touched: set[str] = set()
    for pattern in (r'\bel\("([^"]+)"\)', r'\btext\("([^"]+)"', r'getElementById\("([^"]+)"\)'):
        touched |= set(re.findall(pattern, script))
    check(not sorted(touched - set(declared)),
          f"the script writes to ids the page does not declare: "
          f"{sorted(touched - set(declared))}")
    check(not duplicated, f"an id declared twice paints whichever comes first: {duplicated}")

    # Every box that carries its own content needs somewhere to put its live note, because that
    # is what paintNodes writes into for it.
    for node_id in re.findall(r'nodeId === "([a-z]+)"', script):
        block = script[script.index(f'nodeId === "{node_id}"'):]
        check(f'id="note-{node_id}"' in block[:block.index("return null")],
              f"the {node_id} box has content of its own but no #note-{node_id} to write its "
              f"live note into, which is what made paintNodes throw")

    check("if (painted.live)" in script,
          "paintNodes must not assume every box has somewhere to write its note")
    check("id=\"paint-error\"" in page,
          "a panel that fails to paint has to say so instead of going quiet")


LONG_NOTE = ("judging 2 placements against the tray geometry and the declared cavity, with a "
            "very long tail that has to be clipped rather than drawn over the box next to it, "
            "because that is exactly what used to happen on this page")


def _stress_snapshots():
    """One snapshot carrying LONG_NOTE in the narrowest boxes, to prove it cannot escape them."""
    bus = telemetry.Bus()
    for stage in ("capture", "execute", "verify", "ground"):
        bus.node(stage, telemetry.ACTIVE, LONG_NOTE)
    bus.set_state(phase="command", command="separa las piezas por colores")
    return [bus.snapshot()]


def test_nothing_can_paint_outside_its_own_box():
    """No text may escape its panel - the one thing that makes this diagram look broken.

    It was happening three ways, and all three are fixed by putting the text in a div that clips
    rather than in an SVG ``<text>``, which neither wraps nor clips:

    * a stage's detail was a ``<text>`` at a fixed x, so a long one ran off to the right and was
      read over the box next to it - which is what the four perception boxes were doing;
    * a stage's live note was appended to a box with no room for it, so it fell out of the bottom;
    * an edge label was a ``<text>`` too, and hung out of its gap between boxes.

    What is checked is that the clipping is *declared*: the class is on the element and the
    containers are ``overflow: hidden``. Whether a browser honours ``-webkit-line-clamp`` is not
    something this DOM shim can answer, and the declaration is what makes it true in a real one.
    """
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")

    check("foreignObject { overflow: hidden; }" in page,
          "a foreignObject that can show more than its box is how text escapes")
    for level in ("clip-1", "clip-2", "clip-3"):
        check(f".{level}" in page, f"the {level} clipping rule must exist")
    check("overflow: hidden" in re.search(r"\.fo \{.*?\}", page, re.S).group(0),
          "the box's own content must clip")
    # the live note of a plain box, and the note of a box with content, both carry a clip
    check(re.search(r'<div class="live clip-2" id="live-\$\{node.id\}">', page),
          "the live note must carry a clipping class")
    block = page[page.index('nodeId === "input"'):]
    check("clip-1" in block[:block.index("return null")], "the input box's note must be clipped")
    # understand and plan are no longer route boxes: they live on the main screen, where the
    # sentence and the option rows clip on their own elements
    check('id="note-understand"' not in page,
          "the understanding box has no sentence line: it shows the classification only")
    check('class="sure clip-1" id="cls-sure"' in page, "the confidence line must clip to one line")
    check("text-overflow: ellipsis" in page,
          "an edge label that cannot ellipsis will run over the box beside it")
    # the checklist's rows, one per line
    tasks = re.search(r"#hero-tasks-list \.what \{(.*?)\}", page, re.S)
    check(tasks and "text-overflow: ellipsis" in tasks.group(1),
          "a task label longer than the plan box must ellipsis, not overflow")

    # And the long notes, from a page loaded once with the stress batch as its only content: the run
    # itself is irrelevant here, so it is not sent, and a second load of the same page costs
    # 200 ms to learn the same thing again.
    report = dashboard_client_report([_stress_snapshots()])
    if report is None:
        # Only when node is missing. Not a silent pass: it is a real gap in what was checked.
        print("  (SKIPPED the long-note half: node is not installed, so the page was never "
              "loaded and there is nothing to read.)")
        return
    check(not report["throws"], f"a long note must not throw: {report['throws']}")
    check(not [w for w in report["warnings"] if w.startswith("ERROR")],
          f"a long note must not break a panel: "
          f"{[w for w in report['warnings'] if w.startswith('ERROR')]}")
    # The whole note reaches its box; clipping it is the box's business, done in CSS. A note
    # truncated here instead would mean the data was lost, which is a different failure.
    for stage in ("capture", "execute", "verify"):
        check(len(report["notes"][stage]) > 150,
              f"the long note must reach the {stage} box whole, got "
              f"{len(report['notes'][stage])} characters")


def run_dashboard_client(page_batches):
    """Load the shipped page once, replay these batches into it, and report what it painted.

    ``page_batches`` is a list of snapshot lists. The first is the run; any others replay into the
    page as it already stands, without a reload, which is the more realistic case - a page that
    has already painted has to keep working when more state arrives.

    One node process per call. Spawning node costs ~200 ms and it is the single biggest thing in
    the suite, so this is only split where a test genuinely needs a different set of batches; the
    two tests that share the run's own content go through ``dashboard_client_report`` instead, so
    they still only spawn node between them.
    """
    node = shutil.which("node")
    if node is None:
        return None
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    # COLORS.idle is the colour buildFlow gives every box, so it is the only value "this box has
    # not been visited yet" can be compared against. Read from the page so it cannot drift.
    idle = re.search(r'idle:\s*"(#[0-9a-fA-F]{6})"', page)
    payload = [list(batch) for batch in page_batches]
    with tempfile.TemporaryDirectory() as folder:
        dump = Path(folder) / "snapshots.json"
        # ensure_ascii off, so the file holds the accented Spanish the browser will render
        dump.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        done = subprocess.run([node, str(ROOT / "tests" / "dashboard_client.js"),
                               str(ROOT / "monitor" / "static" / "index.html"), str(dump),
                               idle.group(1) if idle else "#4a5164"],
                              capture_output=True, text=True, encoding="utf-8", timeout=120)
    if done.returncode != 0:
        return {"loadError": done.stderr[-400:], "throws": [], "warnings": [], "nodes": 0,
                "boxes": 0, "dark": [], "noNote": [], "paintError": "", "layout": {},
                "gutter": None, "run": {}, "selected": {}, "intentRows": 0, "taskRows": 0}
    return json.loads(done.stdout)


def test_the_dashboard_repaints_the_whole_route_during_a_run():
    """The shipped page's own script, fed a whole run's snapshots, must paint every box.

    This is the test for "the decision route stops updating", and it runs the actual
    ``index.html`` script rather than a description of it: the bug was in the browser's code and
    nothing in Python could see it. ``tests/dashboard_client.js`` loads the page into a DOM shim,
    replays the snapshots below, and reports what it painted.

    The snapshots are built from ``telemetry.Bus`` here, so the data is the data the robot
    actually publishes - every stage of the decision path, in order, with the state that goes
    with it.
    """
    node = shutil.which("node")
    if node is None:
        print("  (node not found: the dashboard's own script was not exercised)")
        return

    bus = telemetry.Bus()
    stages = [("setup", telemetry.ACTIVE, "waiting for the dashboard to start the environment"),
              ("setup", telemetry.OK, "Isaac running; variant contained, camera top"),
              ("input", telemetry.ACTIVE, "separa las piezas por colores"),
              ("understand", telemetry.ACTIVE, "separa las piezas por colores"),
              ("understand", telemetry.OK, "separar las piezas por color"),
              ("capture", telemetry.ACTIVE, "frame"),
              ("capture", telemetry.OK, "rendered one frame"),
              ("perceive", telemetry.OK, "4 object(s), 2 container(s)"),
              ("link", telemetry.OK, "4/4 blob(s) matched a part"),
              ("ground", telemetry.OK, "2 pick(s) resolved to 2 destination(s)"),
              ("plan", telemetry.ACTIVE, "2 step(s)"),
              ("execute", telemetry.ACTIVE, "above"),
              ("execute", telemetry.OK, "above 1 red: above #3"),
              ("verify", telemetry.WARN, "1/2 inside; missed cube"),
              ("recover", telemetry.OK, "cube re-placed"),
              ("report", telemetry.OK, "2 verified, 1 missed, 0 grasp(es) refused")]
    # The task list as the robot publishes it: three rows in the three states the panel shows -
    # done, measured and missed, and in progress. All three have to be distinguishable.
    TASKS = [
        {"key": "ball_red", "part": "bola roja", "destination": "cesta roja",
         "label": "bola roja \u2192 cesta roja", "done": True, "state": "done"},
        {"key": "cube", "part": "cubo rojo", "destination": "cesta roja",
         "label": "cubo rojo \u2192 cesta roja", "done": False, "state": "fail"},
        {"key": "cube_blue", "part": "cubo azul", "destination": "cesta azul",
         "label": "cubo azul \u2192 cesta azul", "done": False, "state": "doing"},
    ]

    states = [{"phase": "setup"}, {"phase": "starting"}, {"phase": "ready", "sim": "running"},
              {"phase": "scene", "round": 0},
              {"phase": "command", "command": "separa las piezas por colores",
               "command_src": "dashboard"},
              {"phase": "command", "command": "separa las piezas por colores",
               "classification": {
                   "intent": "sort", "sort_by": "color",
                   "items": [{"kind": "cube", "color": "red", "count": 1,
                              "place": {"kind": "box", "color": "red"}}],
                   "place": {"kind": "box", "color": "red"},
                   # the sentence the robot publishes, so the panel never has to reassemble the
                   # decision out of a grid of options
                   "summary": "separar las piezas por color",
                   # the model's own per-field scores, as `parse` now publishes them
                   "confidence": 0.742,
                   "confidences": {"intent": 0.98, "place_kind": 0.91,
                                   "place_color": 0.742, "sort_by": 0.88}}},
              {"phase": "command", "command": "separa las piezas por colores",
               "tasks": TASKS},
              {"phase": "command", "command": "separa las piezas por colores",
               # where in the plan the arm is: which step it is running, of how many
               "progress": "above", "step_index": 3, "step_total": 8,
               "plan": {"n": 1, "intent": "sort", "picks": ["3"],
                        "question": "", "unresolved": [],
                        "steps": [{"action": "above", "note": "above #3",
                                   "pose": [0.3, -0.4, 0.12]}]}},
              {"phase": "done", "verified": 2, "missed": 1}]

    snapshots = []
    for index, (node_id, state, note) in enumerate(stages):
        bus.node(node_id, state, note)
        # The log line is the stage's own note, not a wrapper around it, because the page reads
        # the first word of an execute line as the step in progress: a wrapper there made every
        # plan/execute line read as the word "etapa".
        bus.log(note, node=node_id)
        if index < len(states):
            bus.set_state(**states[index])
        snapshots.append(bus.snapshot())

    node = shutil.which("node")
    if node is None:
        print("  (node not found: the dashboard's own script was not exercised)")
        return
    report = dashboard_client_report([snapshots, _stress_snapshots()])

    check(not report.get("loadError"), f"the dashboard's script must load: "
                                       f"{report.get('loadError')}")
    check(report["loadError"] is None, f"the page's script threw while loading: "
                                       f"{report['loadError']}")
    check(not report["throws"],
          f"a snapshot must never throw on the way to the screen: {report['throws']}")
    check(not [w for w in report["warnings"] if w.startswith("ERROR")],
          f"no panel may fail while painting: "
          f"{[w for w in report['warnings'] if w.startswith('ERROR')]}")
    check(not report["paintError"], f"the failure banner should be empty, got "
                                    f"{report['paintError']!r}")
    check(report["boxes"] == report["nodes"],
          f"the diagram must draw all {report['nodes']} boxes, drew {report['boxes']}")
    check(not report["dark"],
          f"every box must take part in the route during a run; these never lit up: "
          f"{report['dark']}")
    check(not report["noNote"],
          f"every box must show what the robot said at it; these showed nothing: "
          f"{report['noNote']}")
    run = report["run"]
    check(run["command"] == "separa las piezas por colores",
          f"the command box must show the sentence, got {run['command']!r}")
    check(report["run"]["planRows"] >= 1,
          f"the plan card must fill its steps table, got {run['planRows']} rows")
    check(report["run"]["logLines"] > 0, "the log must grow as the run publishes")
    check(report["run"]["phase"] == "terminado", f"the phase pill must follow the run, got "
                                     f"{run['phase']!r}")

    # ---- the layout itself -------------------------------------------------
    layout = report["layout"]
    check(all(layout.values()), f"every box must publish its geometry: "
                                f"{[k for k, v in layout.items() if not v]}")

    # Row A: entorno and command, same size. Row C: the four of the perception chain, same size.
    # These were the explicit complaint - boxes of different widths on the same row.
    for row, names in (("A", ["setup", "input"]), ("C", ["capture", "perceive", "link", "ground"])):
        widths = {round(layout[n]["w"], 1) for n in names}
        heights = {round(layout[n]["h"], 1) for n in names}
        check(len(widths) == 1, f"row {row} must have equal widths, got {sorted(widths)}")
        check(len(heights) == 1, f"row {row} must have equal heights, got {sorted(heights)}")
        tops = {round(layout[n]["y"], 1) for n in names}
        check(len(tops) == 1, f"row {row} must be one row, got tops {sorted(tops)}")

    # Nothing may overlap, and the rows must be in the order they were asked for.
    names = [n for n, v in layout.items() if v]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = layout[a], layout[b]
            overlaps = (A["x"] < B["x"] + B["w"] and B["x"] < A["x"] + A["w"]
                        and A["y"] < B["y"] + B["h"] and B["y"] < A["y"] + A["h"])
            check(not overlaps, f"{a} and {b} overlap")

    order = [layout[n]["y"] for n in ("setup", "understand", "capture", "plan", "report")]
    check(order == sorted(order), f"the five rows must come in order, got {order}")

    # The intention row spans the screen, and so does the report.
    for name in ("understand", "report"):
        # the full width is what row A adds up to, measured from the drawn boxes
        full = layout["setup"]["w"] * 2 + report["gutter"]
        check(layout[name]["w"] == full,
              f"{name} must span the full width: {layout[name]['w']} against {full}")
        check(layout[name]["x"] == layout["setup"]["x"],
              f"{name} must start at the left margin, got {layout[name]['x']}")

    # The plan is on the left and the loop is to its right.
    check(layout["plan"]["x"] == layout["execute"]["x"] or
          layout["plan"]["x"] + layout["plan"]["w"] <= layout["execute"]["x"],
          "the plan must sit to the left of the loop")
    for name in ("execute", "verify", "recover"):
        check(layout[name]["y"] == layout["plan"]["y"],
              f"{name} must share the plan's row")
        check(layout[name]["w"] > 100, f"{name} is too narrow to hold a note: "
                                       f"{layout[name]['w']}")

    # The checklist: one row per task, the finished ones marked, and a count of how far along.
    run = report["run"]
    tasks_html = run["tasks"]
    check(run["taskRows"] == len(TASKS),
          f"the objective's checklist must list every task, got {run['taskRows']} rows")
    check(tasks_html.count('class="done"') == 1,
          f"the one finished task must be marked done: {tasks_html[:200]}")
    check("1 de 3" in run["tally"],
          f"the tally under the checklist must say how far along it is: {run['tally']!r}")
    check('class="doing"' in tasks_html,
          f"and the task being moved must be marked in the checklist, not just counted")
    check("1 fallida" in run["tally"],
          f"and how many failed: {run['tally']!r}")
    for wanted in ("bola roja", "cesta roja", "cubo azul", "cesta azul"):
        check(wanted in tasks_html, f"the checklist must show {wanted!r}: {tasks_html[:240]}")
    # The three states that occurred in this run are three different icons. "Pendiente" has no row
    # here - nothing is left unstarted once the arm is working on the list - so its empty circle is
    # checked against the panel's own mark table below rather than against the rendered rows.
    check("&#10003;" in tasks_html, "a finished task must show a tick")
    check("&#10007;" in tasks_html, "a task that was measured and missed must show a cross")
    check("&#9679;" in tasks_html, "the task in progress must show its own icon")
    check('class="doing"' in tasks_html, "and the row must say it is the one being done")
    # A pending task is not in this run's data, so its icon is asserted from the marks table
    # rather than from the rendered rows: nothing here should be *pending* once the arm is working.
    check("&#9675;" not in tasks_html or 'class="todo"' in tasks_html,
          "an unstarted task must show the empty circle, not a mark for work in progress")
    check('class="doing"' in tasks_html and "&#9679;" in tasks_html,
          "the task in progress must be distinguishable from one that is merely pending")
    # The four marks exist, and they are four different glyphs: a task the arm is doing, one that
    # was measured and finished, one that was measured and failed, and one not yet started.
    panel = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    marks = dict(re.findall(r'(\w+):\s*\{ icon: "&#(\d+);", word:', panel))
    check(len(marks) == 4, f"the panel must define four task marks, got {marks}")
    check(len(set(marks.values())) == 4,
          f"and the four marks must be four different glyphs, got {marks}")

    # ---- la clasificación: la frase, la fiabilidad, y el abanico completo --------------------
    #
    # La caja responde a dos preguntas, no a una. «¿Qué ha dicho el robot?» la responde la frase.
    # «¿Qué podría haber dicho?» la responde el abanico: todas las opciones de cada campo, porque
    # ver «separar» al lado de «recoger» y de «parar» es lo que hace que la elegida signifique algo.
    # Sin el abanico, la frase se lee como un resultado y no como una decisión.
    # La fiabilidad sólo aparece cuando es baja, y nombra el campo flojo: un número sin dueño
    # ("74%") no permite arreglar nada, "el color del destino (74%)" sí.
    check(run["clsSure"] == "lo menos claro: el color del destino (74%)",
          f"the weakest field must be named, not a bare number, got {run['clsSure']!r}")
    check(run["clsWarn"] == "", f"this run doubted nothing, got {run['clsWarn']!r}")

    # El abanico: seis campos, cada uno con TODAS sus opciones, y exactamente una elegida.
    # Los recuentos son los del vocabulario del modelo, no de esta ronda.
    for field, expected in (("opt-intent", 16), ("opt-kind", 7), ("opt-color", 9),
                            ("opt-count", 6), ("opt-place", 8), ("opt-sort", 3)):
        row = run["options"][field]
        check(row["total"] == expected,
              f"{field} must offer all {expected} of its options, got {row['total']}")
        check(row["chosen"] <= 1,
              f"at most one option per field may be the chosen one, got {row}")
        check(row["divs"] == row["total"],
              f"every option in {field} must be a pill, not stray markup: {row}")
    # This run: a sort, by colour, over every part, so the chosen ones are those three.
    check(run["options"]["opt-intent"]["chosen"] == 1,
          f"the run decided an intent, so one must be lit: {run['options']['opt-intent']}")
    check(run["options"]["opt-sort"]["chosen"] == 1,
          f"the run ordered by something: {run['options']['opt-sort']}")
    # The probability belongs to the chosen option only, and the run's intent was trusted at 98%:
    # that number has to be on the intent pill, and nowhere else in the row.
    check("separar" in report["optionHtml"]["opt-intent"],
          "the chosen intent must be among the ones offered")
    check("98%" in report["optionHtml"]["opt-intent"],
          f"the chosen intent must carry the model's own score: "
          f"{report['optionHtml']['opt-intent'][:200]}")
    for field in ("opt-intent", "opt-kind", "opt-color", "opt-count", "opt-place", "opt-sort"):
        scores = [mark for mark in report["optionScores"][field]]
        check(len(scores) <= 1,
              f"only the chosen option in {field} may carry a score, got {scores}")
    # Nothing is on the table in this run's data, so nothing may be marked as existing there.
    check(sum(row["here"] for row in run["options"].values()) == 0,
          f"no detection means no option may claim to be on the table: {run['options']}")

    # ---- la línea "ahora", sobre el mapa -------------------------------------------------
    # La pregunta que un mapa no responde de un vistazo: qué hace el brazo EN ESTE INSTANTE. Se
    # monta de lo que el robot publica -la palabra de acción, la tarea marcada como en curso, y el
    # contador de pasos-, no de lo que la página deduzca. Lo que se ve aquí es el final de la
    # ronda: la orden ya se ejecutó, así que la línea NO sigue hablando en presente.
    check(run["routeNow"] == "terminado",
          f"a finished run must leave the line at its phase, not claim motion that ended, "
          f"got {run['routeNow']!r}")
    # Y una orden recién dada, sin ningún paso todavía, dice sólo que el brazo no se ha movido:
    # la orden ya está en el objetivo.
    check(report["routeNow"] == "el brazo aún no se ha movido",
          "a command with no step yet says the arm has not moved; the order itself is in the "
          f"objective, not repeated here, got {report['routeNow']!r}")

    # ---- los pasos: sólo el verbo ---------------------------------------------------------
    # La matriz tiene una columna por tarea y una fila por movimiento. La pose en milímetros y el
    # código "#2" se han ido a propósito: nadie mueve un brazo con esos números, y su sitio es el
    # registro.
    check(run["stepColumns"] >= 1,
          f"the movement matrix must have a column per task, got {run['stepColumns']}")
    check("pose" not in run["tasks"] and "mm" not in run["tasks"],
          "no millimetres in the panel")


def test_the_dout_is_the_plan_s_question_and_the_panel_sees_it():
    """La duda vive en el plan, y la caja la lee de ahí.

    Era una línea muerta: ``paintClassification`` leía ``unresolved`` y ``question`` de
    ``classification``, y ``parse`` nunca los publicó allí -viven en el plan, que es quien para a
    preguntar «¿cuál de las dos rojas?»-. La línea existía en la página y no se veía nunca. Ahora
    ``_publish_plan`` los publica y la caja los lee de ``window.state.plan``.

    Un solo snapshot basta: la pregunta se pinta en cuanto el plan llega, que es lo que se comprueba.
    """
    if shutil.which("node") is None:
        print("  (node not found: the page's own script was not exercised)")
        return
    bus = telemetry.Bus()
    bus.node("ground", telemetry.WARN, "Aclaración: hay dos bolas rojas")
    bus.set_state(phase="command", command="coge la bola roja",
                  classification={"intent": "pick", "items": [{"kind": "ball", "color": "red"}],
                                  "summary": "coger una bola roja"},
                  plan={"question": "hay dos bolas rojas, ¿cuál?",
                        "unresolved": ["la bola roja: dos candidatos"]})
    report = dashboard_client_report([[bus.snapshot()]])
    if report is None:
        print("  (SKIPPED the doubt line: node is not installed, so the page was never loaded "
              "and there is nothing to read.)")
        return
    check(not report.get("loadError"), f"the page must load: {report.get('loadError')}")
    check(not report["throws"], f"a snapshot must never throw: {report['throws']}")
    run = report["run"]
    check("hay dos bolas rojas" in run["clsWarn"],
          f"the robot's question must reach the classification box, got {run['clsWarn']!r}")
    check("dos candidatos" in run["clsWarn"],
          f"and so must what it could not resolve, got {run['clsWarn']!r}")
    check(run["said"] == "", f"the box does not repeat the command, got {run['said']!r}")


def test_the_line_above_the_map_says_what_the_arm_is_doing():
    """«ahora: trasladar · cubo azul → cesta azul · paso 3 de 8», while it is happening.

    The map answers "where is this command going", which is a question about the process. The one
    asked while watching is "what is the robot doing RIGHT NOW", and it used to be assembled by eye
    out of the execution box's note, the lit row of the plan and the action word of the step. It is
    one line now, built from what the robot publishes: the step it is on, of how many, and the task
    it marked as being executed.
    """
    if shutil.which("node") is None:
        print("  (node not found: the page's own script was not exercised)")
        return
    bus = telemetry.Bus()
    bus.node("execute", telemetry.ACTIVE, "trasladar")
    bus.set_state(phase="sorting", command="separa las piezas por colores",
                  moving=True,
                  progress="carry", step_index=3, step_total=8,
                  tasks=[{"key": "c:bola roja:cesta roja", "label": "bola roja \u2192 cesta roja",
                          "done": True, "state": "done"},
                         {"key": "c:cubo azul:cesta azul", "label": "cubo azul \u2192 cesta azul",
                          "done": False, "state": "doing"}],
                  plan={"steps": [{"action": "carry", "note": "carry #7", "pose": None}]})
    report = dashboard_client_report([[bus.snapshot()]])
    if report is None:
        print("  (SKIPPED the now line: node is not installed, so the page was never loaded "
              "and there is nothing to read.)")
        return
    check(not report.get("loadError"), f"the page must load: {report.get('loadError')}")
    check(not report["throws"], f"a snapshot must never throw: {report['throws']}")
    check(report["routeNow"] == "ahora: trasladar \u00b7 cubo azul \u2192 cesta azul \u00b7 paso 3 de 8",
          f"the line must say what is happening now, with the part and the step, "
          f"got {report['routeNow']!r}")


def test_the_now_line_stops_saying_the_arm_is_moving_when_it_is_not():
    """The verb in the present tense only while the robot says the arm is moving.

    ``progress``, ``step_index`` and ``step_total`` are written per step and were never
    cleared, and the phase stays ``command``/``sorting`` throughout verification and
    recovery. So after the last step of a plan the panel kept asserting the last verb -
    «ahora: retirar · paso 8 de 8» - while the robot was measuring where the part came
    to rest. The robot now publishes ``moving``, and the page keys its present tense on
    that flag instead of on the phase.

    This is the regression test for a dashboard that lies about what the arm is doing,
    which is the one failure a live demo cannot survive.
    """
    if shutil.which("node") is None:
        print("  (node not found: the page's own script was not exercised)")
        return
    # A finished plan: the step counter and the action word are still there, and the
    # phase is still the command's, and the task is still the one being placed.
    bus = telemetry.Bus()
    bus.node("execute", telemetry.OK, "8 pasos, 1 agarre(s)")
    bus.set_state(phase="sorting", command="separa las piezas por colores",
                  moving=False, progress="retreat", step_index=0, step_total=0,
                  tasks=[{"key": "c:cubo azul:cesta azul", "label": "cubo azul \u2192 cesta azul",
                          "done": True, "state": "done"}])
    report = dashboard_client_report([[bus.snapshot()]])
    if report is None:
        print("  (SKIPPED the now line: node is not installed, so the page was never loaded "
              "and there is nothing to read.)")
        return
    said = report["routeNow"]
    check("retreat" not in said and "paso" not in said,
          f"a finished plan must not still name its last step, got {said!r}")
    check(not said.startswith("ahora:"),
          f"a finished plan must not claim the arm is moving, got {said!r}")

    # And the two phases that are not motion at all get their own sentence.
    for phase, expected in (("verifying", "midiendo"), ("recovering", "recuperando")):
        bus = telemetry.Bus()
        bus.set_state(phase=phase, command="separa las piezas por colores",
                      moving=False, progress="", step_index=0, step_total=0, tasks=[])
        report = dashboard_client_report([[bus.snapshot()]])
        if report is None:
            print("  (SKIPPED the now line: node is not installed)")
            return
        check(expected in report["routeNow"],
              f"{phase} has to say what it is doing, got {report['routeNow']!r}")


def test_the_option_space_is_there_before_any_command():
    """El abanico completo está desde el primer fotograma, y los puntos siguen a la percepción.

    Dos cosas, y las dos son de la misma idea: la caja de clasificación es ESTÁTICA.

    1. El espacio de opciones no depende de la orden -¿qué podría haber dicho el modelo?- así que
       aparece con la página y lo único que cambia con el tiempo es qué opción va resaltada.
       Pintarlo al llegar la primera clasificación dejaba la caja vacía justo el rato en el que
       alguien se está preguntando para qué sirve.
    2. Los puntos verdes -"esta opción existe de verdad en la mesa"- dependen de la percepción, no
       de la orden, así que se repintan cuando llegan detecciones nuevas. Sin eso se quedaban con
       las de la primera mirada para siempre, porque la clasificación no había cambiado y el
       repintado estaba colgado de ella.
    """
    if shutil.which("node") is None:
        print("  (node not found: the page's own script was not exercised)")
        return
    SPACE = (("opt-intent", 16), ("opt-kind", 7), ("opt-color", 9), ("opt-count", 6),
             ("opt-place", 8), ("opt-sort", 3))

    # 1. Nothing has been understood yet: the whole space is on screen, nothing lit.
    bus = telemetry.Bus()
    bus.node("setup", telemetry.OK, "Isaac running; variant contained, camera top")
    bus.set_state(phase="ready", sim="running")
    first = dashboard_client_report([[bus.snapshot()]])
    if first is None:
        print("  (SKIPPED the option space: node is not installed, so the page was never "
              "loaded and there is nothing to read.)")
        return
    check(not first.get("loadError"), f"the page must load: {first.get('loadError')}")
    for field, expected in SPACE:
        row = first["options"][field]
        check(row["total"] == expected,
              f"{field} must offer all {expected} of its options from the start, got {row['total']}")
        check(row["chosen"] == 0, f"nothing is chosen before a command: {row}")
        check(row["here"] == 0, f"nothing is on the table before the first look: {row}")
    # Before any command the sentence only says that it is waiting; it never states a decision.
    check(first["said"] == "", f"the box has no sentence to state a decision in, got {first['said']!r}")

    # 2. A command is understood with nothing on the table yet, and then the robot looks.
    bus = telemetry.Bus()
    bus.node("understand", telemetry.OK, "coger y colocar un cubo rojo en una cesta roja")
    bus.set_state(phase="command", command="pon el cubo rojo en la caja roja",
                  classification={
                      "intent": "pick_and_place", "sort_by": "none",
                      "items": [{"kind": "cube", "color": "red",
                                 "place": {"kind": "box", "color": "red"}}],
                      "place": {"kind": "box", "color": "red"},
                      "summary": "coger y colocar un cubo rojo en una cesta roja",
                      "confidence": 0.74, "confidences": {"intent": 0.98}})
    before_looking = bus.snapshot()
    bus.set_state(detections=[{"id": 1, "kind": "cube", "color": "red", "prim": "cube_red",
                               "container": False},
                              {"id": 2, "kind": "box", "color": "red", "prim": "box_red",
                               "container": True},
                              {"id": 3, "kind": "tray", "color": "green", "prim": "tray_green",
                               "container": True},
                              {"id": 4, "kind": "ball", "color": "blue", "prim": "ball_blue",
                               "container": False}])
    after_looking = bus.snapshot()
    report = dashboard_client_report([[before_looking], [after_looking]])
    check(report is not None and not report.get("loadError"),
          f"the page must load: {report and report.get('loadError')}")
    check(not report["throws"], f"a snapshot must never throw: {report['throws']}")
    options = report["options"]
    # The command's own choices are lit, and only those. A chosen pill is not counted as "exists
    # on the table" - it is the decision, and the dot says the other thing - so the two are added
    # to get "everything that is real for this field".
    check(options["opt-intent"]["chosen"] == 1, f"one intent is chosen: {options['opt-intent']}")
    check(options["opt-kind"]["chosen"] == 1, f"one kind is chosen: {options['opt-kind']}")
    check(options["opt-color"]["chosen"] == 1, f"one colour is chosen: {options['opt-color']}")
    check(options["opt-place"]["chosen"] == 1, f"one destination is chosen: {options['opt-place']}")
    # And the dots follow the look, which arrived with the classification unchanged: the balls and
    # the trays are on the table even though nothing was said about them.
    marked_kinds = options["opt-kind"]["chosen"] + options["opt-kind"]["here"]
    check(marked_kinds == 2,
          f"cube and ball are both on the table: {options['opt-kind']}")
    marked_colours = options["opt-color"]["chosen"] + options["opt-color"]["here"]
    check(marked_colours == 3,
          f"red, green and blue are all on the table: {options['opt-color']}")
    check(marked_colours < 9,
          f"but not the six the model knows and this table does not have: {options['opt-color']}")
    marked_places = options["opt-place"]["chosen"] + options["opt-place"]["here"]
    check(marked_places == 2,
          f"box and tray are both destinations on the table: {options['opt-place']}")
    # The chosen one keeps its score through all of it.
    check("98%" in report["optionHtml"]["opt-intent"],
          "the chosen intent still carries its own score")
    # And "sin ordenar" is lit, because this command is not a sort: the model decided there is no
    # order. The difference against the case above - nothing decided yet, nothing lit - is the
    # whole point of the pill.
    check(options["opt-sort"]["chosen"] == 1,
          f"this command orders nothing, and that is a decision: {options['opt-sort']}")


def test_the_robot_says_what_it_understood_in_one_sentence():
    """``separar las piezas por color``, not a highlighted cell in a grid of options.

    The panel's classification box reads one sentence, and the sentence is built by
    ``describe.understood`` from the decision itself - the same ``phrase`` the log notes and the
    grounding questions use, so there is no third Spanish in the system. What is checked here is
    that each intent reads as something a person would say, and that a destination is included
    only when the sentence named one.
    """
    module = load_demo11()
    module.load_demo10()
    import describe

    cases = [
        # (command, expected)
        ({"intent": "sort", "items": [], "sort_by": "color", "place": None},
         "separar las piezas por color"),
        ({"intent": "sort", "items": [{"kind": "cube", "color": "red", "quantity": 1}],
          "sort_by": "kind", "place": None},
         "separar un cubo rojo por tipo"),
        ({"intent": "pick_and_place",
          "items": [{"kind": "cube", "color": "red", "quantity": 1}],
          "place": {"kind": "box", "color": "red"}, "sort_by": "none"},
         "coger y colocar un cubo rojo en una cesta roja"),
        # A pick with nowhere to put it says so by not saying a destination.
        ({"intent": "pick", "items": [{"kind": "ball", "color": "green", "quantity": 1}],
          "place": None, "sort_by": "none"},
         "coger una bola verde"),
        ({"intent": "pick", "items": [{"kind": "ball", "color": "blue", "quantity": 3}],
          "place": None, "sort_by": "none"},
         "coger tres bolas azules"),
        ({"intent": "clear", "items": [], "place": None, "sort_by": "none"},
         "recoger todo lo que hay en la mesa"),
        # An intent the model does not use is not translated into a broken sentence.
        ({"intent": "not_an_intent", "items": [], "place": None, "sort_by": "none"},
         "no he entendido la orden"),
    ]
    for command, expected in cases:
        got = describe.understood(command)
        check(got == expected, f"{command['intent']}: expected {expected!r}, got {got!r}")
    # And the demo publishes it, so the page never builds a sentence of its own.
    bus = telemetry.Bus()

    class Model:
        def parse(self, text):
            return {"intent": "sort", "items": [], "sort_by": "color", "place": None,
                    "confidences": {"intent": 0.9}, "confidence": 0.9, "parser": "rules"}

    module.ModelSpy(Model(), bus).parse("separa las piezas por colores")
    classification = bus.snapshot()["state"]["classification"]
    check(classification["summary"] == "separar las piezas por color",
          f"parse must publish the sentence, got {classification.get('summary')!r}")


def test_the_plan_marks_the_task_the_arm_is_on():
    """``doing`` lo pone el robot, no la página: es el único que sabe qué pieza está moviendo.

    The panel used to infer "in progress" by comparing a row's key (a part's name) with the action
    word the executor published ("lift"), and the two never match, so the row never lit up in a
    real run. ``run_plan`` marks the tasks of the plan it starts, and leaves alone the ones
    ``check`` has already measured.
    """
    module = load_demo11()
    module.load_demo10()
    import planner

    class NS:
        lang = "es"

    class Stub(module.MonitoredRobot):
        def __init__(self):
            self.bus = telemetry.Bus()
            self.args = NS()
            self._task_rows = {}
            self._plan_n = 0
            self._step_i = self._step_n = 0
            self.ledger = module.TaskLedger()

    from scene import Scene

    recorded = next((Scene.read(path) for path in sorted((ROOT / "tests" / "fixtures").glob("*.json"))
                     if len(Scene.read(path).containers) >= 2
                     and len(Scene.read(path).objects) >= 2), None)
    check(recorded is not None, "a recorded scene with two baskets is needed for this")
    if recorded is None:
        return
    plan = planner.plan_command({"intent": "sort", "sort_by": "color", "items": [],
                                 "sort_filter": []}, recorded)
    premise = {pick: f"thing#{pick}" for pick in plan.picks}
    for place in plan.places:
        premise.setdefault(place, f"tray#{place}")

    robot = Stub()
    robot._note_tasks(plan, premise)
    states = [row["state"] for row in robot.bus.snapshot()["state"]["tasks"]]
    check(all(state == "todo" for state in states),
          f"nothing is in progress until a plan runs, got {states}")

    def one_of(task):
        sub = planner.Plan(intent=plan.intent)
        sub.picks = [task.pick]
        sub.places = [task.place]
        sub.tasks = [task]
        return sub

    robot._mark_doing(one_of(plan.tasks[0]), premise)
    rows = robot.bus.snapshot()["state"]["tasks"]
    check(rows[0]["state"] == "doing",
          f"the running plan's task must say it is being done: {rows[0]}")
    check(all(row["state"] == "todo" for row in rows[1:]),
          "and the rest must be back to pending, not left as if they were running")
    # A finished one keeps its verdict: re-planning it is not a new attempt.
    robot.ledger.record(robot._task_key(plan.tasks[-1], premise), True)
    robot._settle_rows()
    robot._mark_doing(one_of(plan.tasks[-1]), premise)
    done = [row for row in robot.bus.snapshot()["state"]["tasks"] if row["done"]]
    check(len(done) == 1 and done[0]["state"] == "done",
          f"a verified task must not go back to in progress: {done}")


def test_the_plan_lists_its_tasks_and_ticks_off_the_finished_ones():
    """One row per thing the plan intends to move, marked with the verdict ``check`` gave.

    A multi-object command is not executed as one plan: ``_one_at_a_time`` re-plans around each
    object in turn and calls ``check`` after every one, so the plan being run holds a single task
    while the plan the person asked for held all of them. The list therefore has to be
    accumulated across the command and keyed by part - a re-plan of a finished task must not add a
    second row for it, nor un-tick the first one.

    Driven through the real ``MonitoredRobot`` helpers on a stub, with the ``misses`` list
    ``check`` returns standing in for the simulator's own verdict.
    """
    module = load_demo11()
    module.load_demo10()

    class NS:
        lang = "es"

    class Stub(module.MonitoredRobot):
        def __init__(self):                      # the real one needs an Isaac env
            self.bus = telemetry.Bus()
            self.args = NS()
            self._task_rows = {}
            self._plan_n = 0
            self.ledger = module.TaskLedger()

    import planner
    from scene import Scene

    recorded = next((Scene.read(path) for path in sorted((ROOT / "tests" / "fixtures").glob("*.json"))
                     if len(Scene.read(path).containers) >= 2 and len(Scene.read(path).objects) >= 2),
                    None)
    check(recorded is not None, "a recorded scene with two baskets is needed for this")
    if recorded is None:
        return

    plan = planner.plan_command({"intent": "sort", "sort_by": "color", "items": [],
                                 "sort_filter": []}, recorded)
    check(len(plan.tasks) == len(plan.picks) == len(plan.places),
          f"a plan must record one task per pick: {len(plan.tasks)} tasks, "
          f"{len(plan.picks)} picks, {len(plan.places)} places")
    check(all(task.place is not None for task in plan.tasks),
          "a sort task has a destination, so none of them may be left without one")
    check([t.pick for t in plan.tasks] == plan.picks
          and [t.place for t in plan.tasks] == plan.places,
          "the tasks must line up with the picks and places they came from")

    names = {0: "box_red", 1: "tray_blue", 2: "cube_blue", 3: "cube", 4: "ball_red",
             5: "ball_green"}
    premise = {task.pick: names.get(task.pick, f"thing{task.pick}") for task in plan.tasks}
    for place in plan.places:
        premise.setdefault(place, names.get(place, f"tray{place}"))

    robot = Stub()

    def rows():
        return robot.bus.snapshot()["state"].get("tasks", [])

    def one_of(plan_tasks):
        """The single-task plan `_one_at_a_time` would run for one object."""
        sub = planner.Plan(intent="sort")
        sub.picks = [plan_tasks.pick]
        sub.places = [plan_tasks.place]
        sub.tasks = [plan_tasks]
        return sub

    robot._note_tasks(plan, premise)
    labels = [row["label"] for row in rows()]
    check(len(rows()) == len(plan.tasks), f"every task must be listed, got {labels}")
    check(not any(row["done"] for row in rows()), "nothing is done before the arm has moved")
    check(all(row["state"] == "todo" for row in rows()),
          f"nothing is in progress before the arm moves, got {[r['state'] for r in rows()]}")
    check(all("→" in label for label in labels),
          f"a task must read as part-then-destination, got {labels}")
    check(any("bola roja" in label and "cesta roja" in label for label in labels),
          f"the red ball's task must name both ends, got {labels}")

    # the per-object re-plan must not shrink the list to its own single task
    sub = one_of(plan.tasks[0])
    robot._note_tasks(sub, premise)
    check(len(rows()) == len(plan.tasks),
          f"a one-object re-plan must not replace the whole job, got {len(rows())} rows")

    robot.ledger.record(robot._task_key(plan.tasks[0], premise), True)
    robot._settle_rows()
    check(rows()[0]["done"], "the verified task must be marked done")
    check(rows()[0]["state"] == "done", "and its state must say it was done")
    check(sum(1 for row in rows() if row["done"]) == 1,
          "and only that one")

    # A miss is not a pending task: `check` measured it and found it outside the tray, which is a
    # different fact from never having tried.
    last = one_of(plan.tasks[-1])
    robot._note_tasks(last, premise)
    robot.ledger.record(robot._task_key(plan.tasks[-1], premise), False)
    robot._settle_rows()
    check(not rows()[-1]["done"],
          f"a part left outside the tray must not be marked done: {rows()[-1]}")
    check(rows()[-1]["state"] == "fail",
          f"and it must be marked failed rather than merely pending: {rows()[-1]}")
    check(sum(1 for row in rows() if row["done"]) == 1, "the earlier task keeps its mark")

    # re-planning a finished task must not add a row or clear the mark
    robot._note_tasks(sub, premise)
    check(len(rows()) == len(plan.tasks), f"a re-plan must not duplicate rows, got {len(rows())}")
    check(rows()[0]["done"], "a re-plan must not un-tick what was already verified")
    check(rows()[0]["state"] == "done", "and must not drop it back to pending")
    check(rows()[-1]["state"] == "fail",
          "nor un-fail one that was measured and missed")

    # a new command starts an empty list
    robot._task_rows.clear()
    robot._publish_tasks()
    check(rows() == [], f"a new command starts with no tasks, got {rows()}")


def test_a_receptacle_is_named_not_called_an_object():
    """``cesta roja``, not ``objeto rojo``.

    Perception calls a container ``box``, and ``describe`` had no noun for it, so every
    destination came out as "a red object" - vague, and wrong about gender, since a tray is what
    things go *into* and "objeto" is masculine where the Spanish adjective has to agree.
    """
    import describe

    check(describe.phrase("box", "red", 1, "es", bare=True) == "cesta roja",
          "the receptacle needs its own Spanish noun")
    check(describe.phrase("box", "red", 1, "es").startswith("una cesta"),
          "and it is feminine, so it takes 'una'")
    check(describe.phrase("ball", "red", 1, "es", bare=True) == "bola roja",
          "the parts keep the nouns they had")
    check(describe.phrase("cube", "blue", 1, "es", bare=True) == "cubo azul",
          "including the masculine ones")


def test_raising_a_release_height_keeps_the_tasks():
    """``_release_adjusted`` rebuilds the plan by hand, and must carry the tasks over.

    It copies fields one at a time because it is making a new ``Plan`` with different poses. A
    field it forgets is a field the plan silently loses, and for the task list that would mean
    the checklist emptying itself the moment a release height had to be lifted - which is the
    common case on this bench, since ``clear`` raises most of them.
    """
    module = load_demo11()
    module.load_demo10()
    import planner

    # real Steps, because `_release_adjusted` rebuilds them with `dataclasses.replace`
    plan = planner.Plan(intent="sort", picks=[3, 4], places=[0, 1],
                        tasks=[planner.Task("2 red", 3, "cube", "red", 0, "box", "red"),
                               planner.Task("1 blue", 4, "ball", "blue", 1, "box", "blue")])
    plan.steps = [planner.Step("lower", (0.30, -0.43, 0.06)),
                  planner.Step("release", (0.30, -0.43, 0.06))]

    class Robot(module.MonitoredRobot):
        def __init__(self):
            self.layout = []
            self.release_mode = "clear"
            self.bus = telemetry.Bus()

    robot = Robot()
    import scene_spec
    robot.layout = [item for item in scene_spec.DEFAULT_SCENE if item.container]
    adjusted = robot._release_adjusted(plan)
    check(adjusted is not plan, "this plan's releases must actually be raised, or nothing is tested")
    check(all(step.pose[2] > 0.06 for step in adjusted.steps if step.pose is not None),
          "the poses must have been lifted")
    check(len(adjusted.tasks) == len(plan.tasks),
          f"raising a release height must not drop the tasks: {len(adjusted.tasks)} of "
          f"{len(plan.tasks)}")
    check([(t.kind, t.color, t.place) for t in adjusted.tasks]
          == [(t.kind, t.color, t.place) for t in plan.tasks],
          "and they must be the same tasks, not new ones")


# -- correctness: no hidden failure may read as success -------------------------------------
#
# The checks below patch only what they need, with ``unittest.mock.patch`` so the shared classes
# are restored afterwards. The older tests in this file that assign to a class and ``del`` it
# are the ones this suite should not copy.


def _stub_robot(module, layout, placed):
    """A MonitoredRobot without an Isaac env: the attributes the verification path reads.

    ``placed(prim)`` says whether the simulator would measure that part inside its tray.
    """
    from types import SimpleNamespace

    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.bus = telemetry.Bus()
    robot.layout = layout
    robot.args = SimpleNamespace(lang="es", offset=(0.0, 0.0))
    robot._task_rows = {}
    robot._plan_n = 0
    robot.ledger = module.TaskLedger()
    robot.verified = robot.missed = robot.recovered = 0
    robot.replanned = robot.regrasped = robot.tracked = robot.reaimed = 0
    robot.mask_arm = False
    robot.tracked_xy = {}
    robot.bot = SimpleNamespace(settle=lambda: None, refused=0, corrected=0, kinematic=0,
                                lost=0)
    robot.release_mode = "clear"
    robot.say = lambda text="": None
    robot.park = lambda: None
    robot.look = lambda caption="": SimpleNamespace(objects=[], containers=[], everything=[])
    robot._inside = lambda prim, tray: placed(prim)
    return robot


def test_a_check_counts_each_task_once_and_a_failed_retry_is_not_a_recovery():
    import planner
    from types import SimpleNamespace

    module = load_demo11()
    layout = [SimpleNamespace(prim="tray", container=True)]
    robot = _stub_robot(module, layout, lambda prim: prim == "a")
    robot._retry_one = lambda prim: False          # the retry does not put the part back
    plan = planner.Plan(intent="sort")
    plan.picks, plan.places = [1, 2, 3], [10, 10, 10]
    misses = robot.check(plan, {1: "a", 2: "b", 10: "tray"}, announce=False)

    check(robot.ledger.counts() == (1, 2),
          f"three tasks, one placed: the ledger must say (1, 2), got {robot.ledger.counts()}")
    check(robot.verified == 1 and robot.missed == 2,
          f"the counters are read from the ledger: verified={robot.verified} missed={robot.missed}")
    check(robot.recovered == 0, "a retry that did not place the part is not a recovery")
    check(sorted(misses) == ["b", "thing #3"], f"misses name the tasks: {misses}")


def test_a_successful_retry_is_a_recovery_counted_once():
    import planner
    from types import SimpleNamespace

    module = load_demo11()
    layout = [SimpleNamespace(prim="tray", container=True)]
    state = {"b": False}
    robot = _stub_robot(module, layout, lambda prim: prim == "a" or state.get(prim, False))

    def retry(prim):
        state[prim] = True                          # the retry places it, and it is measured
        return True

    robot._retry_one = retry
    plan = planner.Plan(intent="sort")
    plan.picks, plan.places = [1, 2], [10, 10]
    robot.check(plan, {1: "a", 2: "b", 10: "tray"}, announce=False)

    check(robot.ledger.counts() == (2, 0), f"both placed: {robot.ledger.counts()}")
    check(robot.recovered == 1, f"one retry placed a part: recovered={robot.recovered}")


def test_ambiguous_language_replan_retries_original_identity_and_verifies_it():
    from types import SimpleNamespace

    module = load_demo11()
    robot = _stub_robot(module, [], lambda _prim: False)
    attempts = []
    robot._retry_one = lambda prim: attempts.append(prim) or True
    robot._placement_verdicts = lambda plan, premise: [("ball→tray", True, "ball_red")]
    robot._settle_rows = lambda: None
    task = SimpleNamespace(pick=1, place=2)
    plan = SimpleNamespace(tasks=[task])
    verified = robot._retry_original_targets(plan, {1: "ball_red", 2: "tray_red"})
    check(verified and attempts == ["ball_red"],
          "ambiguous reparsing retries the original catalog object only")
    check(robot.ledger.verdict("ball→tray") == module.TaskLedger.OK
          and robot.recovered == 1 and robot.verified == 1 and robot.missed == 0,
          "direct recovery syncs counters only for measured placements inside the original tray")


def test_failed_recovery_carry_stops_before_lower_release_or_drop():
    from types import SimpleNamespace
    import numpy as np

    module = load_demo11()
    item = SimpleNamespace(prim="ball_red", kind="ball", color="red",
                           size=(.02, .02, .02), x=.38, y=-.435, container=False)
    tray = SimpleNamespace(prim="tray_red", kind="box", color="red",
                           size=(.07, .07, .05), x=.44, y=-.50,
                           height=.05, floor=.006, container=True, inner=(.056, .056))
    ball = SimpleNamespace(id=1, entity_id=None, kind="ball", color="red",
                           size_m=list(item.size), pos_m=[.38, -.435, .0509],
                           aim_xy=[.38, -.435])
    seen = SimpleNamespace(objects=[ball], containers=[SimpleNamespace(
        entity_id="tray_red", pos_m=[.44, -.50, .066], aim_xy=[.44, -.50],
        size_m=list(tray.size))])
    calls, speech = [], []

    class TensorLike:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    def do(action, *_args, **_kwargs):
        calls.append(action)
        return SimpleNamespace(ok=action != "carry", note="carry stalled")

    bot = SimpleNamespace(track=lambda **_kwargs: SimpleNamespace(ok=True, note=""),
                          do=do, log=lambda _message: None, settle=lambda: None,
                          drop=lambda: calls.append("drop"))
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = [item, tray]
    robot._tray_of = lambda _prim: tray
    robot.look = lambda _caption="", **_kwargs: seen
    robot.link = lambda _seen: {}
    robot.tracked_xy = {item.prim: (item.x, item.y)}
    robot.args = SimpleNamespace(offset=(0.0, 0.0))
    robot.env = SimpleNamespace(scene={item.prim: SimpleNamespace(data=SimpleNamespace(
        root_pos_w=[TensorLike([.38, -.435, .0509])]))})
    robot.bot = bot
    robot.say = speech.append
    robot._task_slots = {}
    robot._placed_slots = {}

    success = robot._retry_placed(item.prim)
    check(not success and "carry" in calls,
          "a failed recovery transport is reported as a failed retry")
    check("lower" not in calls and "release" not in calls and "drop" not in calls,
          "the recovery never lowers or opens after carry fails")


def test_a_failed_grasp_is_a_failed_task_and_a_later_placement_replaces_the_verdict():
    import planner
    from types import SimpleNamespace
    from unittest import mock

    module = load_demo11()
    layout = [SimpleNamespace(prim="tray", container=True)]
    placed = {"a": False}
    robot = _stub_robot(module, layout, lambda prim: placed.get(prim, False))
    plan = planner.Plan(intent="pick")
    plan.picks, plan.places, plan.steps = [1], [10], []
    premise = {1: "a", 10: "tray"}
    refused_run = SimpleNamespace(stale_grasps=1, aborted=False, aborted_at="close", reaimed=0,
                                  tracked=0, real_grasps=0, kinematic_grasps=0, steps=[],
                                  lost_grasps=0,
                                  summary=lambda: "1 stale")
    with mock.patch.object(robot.bot, "run", create=True, return_value=refused_run) as motion:
        robot.run_plan(plan, premise, "coge la pieza")
        check(motion.call_args.kwargs["retarget"] is None,
              "motion cannot replace the requested identities and destinations through demo 10's generic replan")

    check(robot.ledger.status() == "FAILED",
          f"one grasp failed and nothing placed must read FAILED, got {robot.ledger.status()}")
    check(robot.ledger.failed_labels() == ["a"], "the failed task is named")

    placed["a"] = True
    robot.check(plan, premise, announce=False)
    check(robot.ledger.status() == "SUCCESS",
          "a later measured placement replaces the failure, and the result follows it")


def test_a_lost_grasp_is_a_failed_grasp_and_is_not_carried():
    import numpy as np
    from unittest import mock

    module = load_demo11()
    sx = module.StrictExecutor.__new__(module.StrictExecutor)
    sx.allow_kinematic, sx.refused, sx.kinematic, sx.lost = False, 0, 0, 0
    sx.impossible, sx.failures, sx.at_close, sx.items_by_prim = set(), {}, None, {}
    sx.frame_hook, sx.log = None, (lambda text: None)
    sx.failure_hook = None

    def parent_do(self, action, pose=None, grip=None, prim=None, **_):
        return executor_module.StepResult(action=action, note="knocked out of reach", ok=False,
                                          grasp="lost")

    with mock.patch.object(executor_module.Executor, "do", parent_do):
        result = sx.do("lift", np.zeros(3), -1.0, prim="cube_a")
    check(result.ok is False and result.grasp == "stale",
          f"a lost grasp must stop the plan like a stale one: {result.grasp!r} ok={result.ok}")
    check(sx.lost == 1 and sx.failures.get("cube_a") == 1,
          "a lost grasp is counted, and against the part")


def test_a_close_time_miss_enables_bounded_target_reacquisition():
    import numpy as np
    from unittest import mock

    module = load_demo11()
    sx = module.StrictExecutor.__new__(module.StrictExecutor)
    sx.allow_kinematic = False
    sx.auto_sim_assist = False
    sx.auto_sim_assist_policy = "after_physical_miss"
    sx.close_command_override = None
    sx.failures = {}
    sx.refused = sx.kinematic = sx.lost = 0
    sx.item_kinds, sx.item_sizes, sx.items_by_prim = {}, {}, {}
    sx.frame_hook = None
    sx.log = lambda _message: None
    sx.part_geometry = lambda _prim: {"lateral_m": 0.033, "fits": True}
    failures = []
    sx.failure_hook = lambda *values: failures.append(values)

    def parent_do(self, action, *_args, **_kwargs):
        return executor_module.StepResult(action=action, note="object moved on close",
                                          ok=False, grasp="stale")

    with mock.patch.object(executor_module.Executor, "do", parent_do):
        result = sx.do("close", np.zeros(3), -1.0, prim="blue_ball")
    check(not result.ok and result.grasp == "stale",
          "a pushed-away target remains a failed physical close")
    check(sx.failures.get("blue_ball") == 1 and failures
          and failures[0][0:2] == ("blue_ball", "missed"),
          "the failure is recorded before the recovery scan can re-anchor the moved target")


def test_a_failed_approach_stops_before_the_gripper_closes():
    """An unreachable approach must not continue into a blind descent and close."""
    import planner

    robot = executor_module.Executor.__new__(executor_module.Executor)
    robot._held = None
    robot.log = lambda _message: None
    calls = []

    def do(action, *_args, **_kwargs):
        calls.append(action)
        return executor_module.StepResult(action=action, ok=False,
                                          note="pose no alcanzable")

    robot.do = do
    plan = planner.Plan(intent="pick")
    plan.steps = [planner.Step(action, (0.35, -0.45, 0.14), executor_module.OPEN, 0, action)
                  for action in ("above", "down", "close", "lift")]
    result = robot.run(plan, resolve=lambda _target: "cube")
    check(calls == ["above"], f"a failed pre-grasp must stop later actions, got {calls}")
    check(result.aborted and result.aborted_at.startswith("above:"),
          f"the run reports where it stopped, got {result.aborted_at!r}")
    check(result.stale_grasps == 1 and result.stale_prims == ["cube"],
          "a failed tracked approach requests a fresh observation and identity-preserving retry")


def test_contact_gated_pregrasp_survives_open_descent_waypoint():
    """The down waypoint is open, but a contact-gated sim grasp must reach close intact."""
    import numpy as np
    from types import SimpleNamespace
    from planner import Plan, Step

    robot = executor_module.Executor.__new__(executor_module.Executor)
    robot._held = None
    robot._pregrasp_kinematic = None
    robot.track_every_steps = executor_module.TRACK_EVERY
    robot.approach_clearance_z = None
    robot.log = lambda _message: None
    robot.set_tool_yaw = lambda _yaw: None
    robot.drop_calls = []

    def drop():
        robot.drop_calls.append(robot._held)
        robot._held = None

    robot.drop = drop
    robot.env = SimpleNamespace(scene={
        "ball": SimpleNamespace(data=SimpleNamespace(root_pos_w=[
            SimpleNamespace(detach=lambda: SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: np.array([.40, -.45, .056]))))]))})
    robot.body_pose_from_current = lambda: np.array([.407, -.45, -.036])

    def track(**_kwargs):
        robot._held = "ball"
        robot._pregrasp_kinematic = "ball"
        return executor_module.StepResult(action="down", note="contact", ok=True)

    robot.track = track
    plan = Plan(intent="pick")
    plan.steps = [Step("down", (.40, -.45, .056), executor_module.OPEN, target="ball")]
    robot.run(plan, resolve=lambda _target: "ball", observe=lambda _prim: (.40, -.45))
    check(robot.drop_calls == [] and robot._held == "ball",
          "the open-gripper descent keeps its explicitly marked contact pregrasp attached")

    robot._pregrasp_kinematic = None
    robot.do = lambda action, *_args, **_kwargs: executor_module.StepResult(
        action=action, note="release", ok=True)
    release = Plan(intent="release")
    release.steps = [Step("release", (.40, -.45, .10), executor_module.OPEN)]
    robot.run(release)
    check(robot.drop_calls == ["ball"] and robot._held is None,
          "an ordinary open-gripper release still drops a grasp without the pregrasp marker")


def test_closed_loop_grasp_pose_is_preserved_for_close_and_vertical_lift():
    """Tracked XY corrections must not be undone by the planned close/lift steps."""
    import numpy as np
    from types import SimpleNamespace
    from planner import Plan, Step

    robot = executor_module.Executor.__new__(executor_module.Executor)
    robot._held = None
    robot.approach_clearance_z = None
    robot.track_every_steps = executor_module.TRACK_EVERY
    robot.log = lambda _message: None
    robot.env = SimpleNamespace(scene={
        "ball": SimpleNamespace(data=SimpleNamespace(root_pos_w=[
            SimpleNamespace(detach=lambda: SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: np.array([.40, -.45, .056]))))]))})
    robot.body_pose_from_current = lambda: np.array([.417, -.431, -.036])
    robot.track = lambda **kwargs: executor_module.StepResult(
        action=kwargs["action"], note="tracked", ok=True)
    calls = []

    def do(action, pose, grip, **kwargs):
        calls.append((action, np.asarray(pose, dtype=float).copy(), kwargs))
        return executor_module.StepResult(action=action, note="", ok=True,
                                          grasp="real" if action in {"close", "lift"} else "")

    robot.do = do
    plan = Plan(intent="pick_and_place")
    plan.steps = [
        Step("above", (.40, -.45, .15), executor_module.OPEN, 0),
        Step("down", (.40, -.45, .056), executor_module.OPEN, 0),
        Step("close", (.40, -.45, .056), executor_module.CLOSE, 0),
        Step("lift", (.40, -.45, .15), executor_module.CLOSE, 0),
        Step("carry", (.47, -.50, .20), executor_module.CLOSE, None),
    ]
    robot.run(plan, resolve=lambda _target: "ball", observe=lambda _prim: (.40, -.45))
    poses = {action: (pose, kwargs) for action, pose, kwargs in calls}
    check(np.allclose(poses["close"][0], (.417, -.431, -.036))
          and poses["close"][1].get("body_target") is True
          and np.allclose(poses["lift"][0][:2], (.417, -.431))
          and poses["lift"][1].get("body_target") is True,
          "close and lift preserve the corrected world-space gripper body pose")
    check(np.allclose(poses["lift"][0][2], .058),
          "vertical lift uses the requested 94 mm tool motion")
    check(np.allclose(poses["carry"][0], (.487, -.481, .108))
          and poses["carry"][1].get("body_target") is True
          and np.allclose(poses["carry"][1].get("feedback_target"), (.47, -.50)),
          "carry translates the measured body pose by the requested object-center displacement")


def test_physical_object_must_follow_carry_before_lower_or_release():
    """A gripper reaching the tray is not a successful carry if the part stayed on the table."""
    import numpy as np
    from types import SimpleNamespace
    from planner import Plan, Step

    class Position:
        def __init__(self, value): self.value = np.asarray(value, dtype=float)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value

    position = Position((.40, -.45, .056))
    robot = executor_module.Executor.__new__(executor_module.Executor)
    robot._held = None
    robot._pregrasp_kinematic = None
    robot.approach_clearance_z = None
    robot.track_every_steps = executor_module.TRACK_EVERY
    robot.log = lambda _message: None
    robot.set_tool_yaw = lambda _yaw: None
    robot.body_pose_from_current = lambda: np.array([.407, -.45, -.036])
    robot.env = SimpleNamespace(scene={"ball": SimpleNamespace(
        data=SimpleNamespace(root_pos_w=[position]))})
    actions = []
    simulate_slip = {"enabled": True}

    def do(action, *_args, **_kwargs):
        actions.append(action)
        if action == "carry" and not simulate_slip["enabled"]:
            position.value[:] = (.47, -.50, .15)
        elif action == "lower" and not simulate_slip["enabled"]:
            position.value[:] = (.47, -.50, .10)
        return executor_module.StepResult(
            action=action, note="", ok=True,
            grasp="real" if action == "lift" else "")

    robot.do = do
    plan = Plan(intent="pick_and_place")
    plan.steps = [
        Step("close", (.40, -.45, .056), executor_module.CLOSE, target=1),
        Step("lift", (.40, -.45, .15), executor_module.CLOSE, target=1),
        Step("carry", (.47, -.50, .15), executor_module.CLOSE),
        Step("lower", (.47, -.50, .10), executor_module.CLOSE),
        Step("release", (.47, -.50, .10), executor_module.OPEN),
    ]
    result = robot.run(plan, resolve=lambda _target: "ball")
    check(actions == ["close", "lift", "carry"]
          and result.aborted and result.aborted_at.startswith("carry:")
          and result.lost_grasps == 1
          and "deslizamiento durante el transporte" in result.steps[-1].note,
          "a slipping actor aborts before lower/release and is counted as a lost physical grasp")

    # A genuine physical object at the requested carry waypoint remains admissible.
    simulate_slip["enabled"] = False
    position.value[:] = (.47, -.50, .15)
    actions.clear()
    result = robot.run(plan, resolve=lambda _target: "ball")
    check(not result.aborted and actions[-2:] == ["lower", "release"]
          and "seguimiento físico del objeto: error 0.0 mm" in result.steps[2].note,
          "the carry check permits lower/release when the simulator actor reaches the waypoint")


def test_motion_feedback_offset_is_bounded_when_a_pose_is_unreachable():
    import numpy as np
    demo = load_demo11()

    local = demo.world_position_to_local_offset((1, 2, 3),
                                                (.70710678, 0, 0, .70710678),
                                                (1, 3, 3))
    check(np.allclose(local, (1, 0, 0), atol=1e-6),
          "simulator recovery recalibrates the carried object from its measured world pose")
    check(demo.simulator_recovery_hold_is_plausible((.007, 0, -.092))
          and not demo.simulator_recovery_hold_is_plausible((-.095, .012, -.183)),
          "simulator recovery refuses an implausible gripper-to-object height jump")
    check(executor_module.carry_follow_tolerance(simulator_assisted=False) == .025
          and executor_module.carry_follow_tolerance(simulator_assisted=True) == .035,
          "simulator assistance gets a bounded compliance allowance without loosening physical-slip checks")
    offset = executor_module.bounded_feedback_offset((.025, .0), (.025, .025))
    assert np.linalg.norm(offset) <= executor_module.MOTION_FEEDBACK_TOTAL_CORRECTION_M + 1e-9
    assert np.all(offset > 0.0)


def test_follow_ratio_tolerates_small_simulation_overshoot_but_rejects_a_launch():
    import contextlib
    import types
    import numpy as np

    class Tensor(np.ndarray):
        def clone(self): return self.copy()
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self)

    fake_torch = types.SimpleNamespace(
        no_grad=contextlib.nullcontext,
        as_tensor=lambda value, device=None: np.asarray(value, dtype=float).view(Tensor),
        linalg=types.SimpleNamespace(norm=np.linalg.norm),
    )

    def lift(movement):
        hook_events = []
        part = types.SimpleNamespace(data=types.SimpleNamespace(
            root_pos_w=np.zeros((1, 3), dtype=float).view(Tensor)))
        env = types.SimpleNamespace(scene={"part": part})
        env.step = lambda _action: setattr(
            part.data, "root_pos_w", np.array([[0.0, 0.0, movement]]).view(Tensor))
        sx = executor_module.Executor.__new__(executor_module.Executor)
        sx.torch, sx.device, sx.env = fake_torch, "cpu", env
        sx.current = np.zeros((1, 3), dtype=float).view(Tensor)
        sx._hold = np.zeros((1, 3), dtype=float).view(Tensor)
        sx.hold_offset_world = lambda: sx._hold
        sx.pause_hook = None
        sx._action = lambda point, _grip: point
        sx._carry_if_held = lambda: None
        sx.physics_step_hook = lambda action, prim: hook_events.append((action, prim))
        result = sx.do("lift", (0.0, 0.0, 1.0), executor_module.CLOSE,
                       prim="part", on_close=np.zeros(3), steps=1, check="follow")
        return result, hook_events

    small, hook_events = lift(1.003)
    launched, _ = lift(1.20)
    check(small.grasp == "real",
          f"a 0.3% state-buffer overshoot is still a held object: {small.grasp}")
    check(launched.grasp == "lost",
          f"a launch 20% beyond the gripper path remains a lost grasp: {launched.grasp}")
    check(hook_events == [("lift", "part")],
          f"simulator assist hooks run after each physics tick with its target, got {hook_events}")


def test_motion_interpolation_starts_at_live_gripper_pose_not_last_command():
    import contextlib
    import types
    import numpy as np

    class Tensor(np.ndarray):
        def clone(self): return self.copy()

    fake_torch = types.SimpleNamespace(
        no_grad=contextlib.nullcontext,
        as_tensor=lambda value, device=None: np.asarray(value, dtype=float).view(Tensor),
    )
    robot = executor_module.Executor.__new__(executor_module.Executor)
    robot.torch, robot.device = fake_torch, "cpu"
    robot.current = np.zeros((1, 3), dtype=float).view(Tensor)
    robot._held = None
    robot.measured = np.array([[.2, .3, .4]], dtype=float).view(Tensor)
    robot._gripper_pos = lambda: robot.measured
    robot.hold_offset_world = lambda: np.zeros((1, 3), dtype=float).view(Tensor)
    robot.position_feedback = None
    robot.pause_hook = None
    robot.env = types.SimpleNamespace(step=lambda _action: None)
    issued = []
    robot._action = lambda pose, _grip: issued.append(np.asarray(pose).copy())
    robot.do("above", (1., 1., 1.), executor_module.OPEN,
             body_target=True, steps=2)
    expected_first = np.array([.2, .3, .4]) + .625 * (np.ones(3) - np.array([.2, .3, .4]))
    check(np.allclose(issued[0][0], expected_first),
          f"first IK target interpolates from live pose, got {issued[0][0]}")


def test_physical_close_stops_immediately_when_contact_pushes_the_object():
    import contextlib
    import types
    import numpy as np

    class Tensor(np.ndarray):
        def clone(self): return self.copy()
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self)

    fake_torch = types.SimpleNamespace(
        no_grad=contextlib.nullcontext,
        as_tensor=lambda value, device=None, dtype=None: np.asarray(value, dtype=float).view(Tensor),
    )
    ball = types.SimpleNamespace(data=types.SimpleNamespace(
        root_pos_w=np.zeros((1, 3), dtype=float).view(Tensor)))
    env = types.SimpleNamespace(scene={"ball": ball})
    def physics_step(_action):
        ball.data.root_pos_w[0, 0] += .005
    env.step = physics_step
    robot = executor_module.Executor.__new__(executor_module.Executor)
    robot.torch, robot.device = fake_torch, "cpu"
    robot.current = np.zeros((1, 3), dtype=float).view(Tensor)
    robot._held = None
    robot._gripper_pos = lambda: robot.current
    robot.hold_offset_world = lambda: np.zeros((1, 3), dtype=float).view(Tensor)
    robot.position_feedback = None
    robot.pause_hook = None
    robot.gripper_step_hook = None
    robot.env = env
    robot._action = lambda _pose, _grip: None
    robot._carry_if_held = lambda: None
    result = robot.do("close", (0., 0., 0.), executor_module.CLOSE,
                      prim="ball", on_close=np.zeros(3), steps=20,
                      check="reach", body_target=True, feedback_target=(0., 0.))
    check(not result.ok and result.grasp == "stale" and "10.0 mm" in result.note,
          f"close aborts at the measured 10 mm push instead of continuing: {result.note}")
    check(abs(float(ball.data.root_pos_w[0, 0]) - .010) < 1e-9,
          "the jaw stops applying contact after the first drift-bound violation")


def test_a_failed_retarget_stops_the_plan_instead_of_running_stale_steps():
    import planner
    import numpy as np  # noqa: F401  (kept for the shape the executor expects)
    from unittest import mock

    module = load_demo11()
    sx = module.StrictExecutor.__new__(module.StrictExecutor)
    sx.log = lambda text: None
    plan = planner.Plan(intent="pick")
    plan.steps = [planner.Step("above", (0.3, -0.45, 0.25), grip=1.0, target=1)]
    continued = []

    def parent_run(self, plan, resolve=None, keys=lambda: set(), fps=1, retarget=None,
                   observe=None):
        retarget(plan.steps[0], "cube_a")           # the re-plan finds nothing ...
        continued.append("kept going")              # ... and must not be followed by this
        return executor_module.RunResult()

    with mock.patch.object(executor_module.Executor, "run", parent_run):
        out = sx.run(plan, resolve=lambda target: "cube_a", retarget=lambda step, prim: None)
    check(continued == [], "an empty re-target must end the plan, not carry on with old steps")
    check(out.stale_grasps == 1 and out.stale_prims == ["cube_a"],
          f"the plan reports the target as lost, against the part: {out.stale_prims}")


def test_failed_unheld_lift_can_retarget_but_held_motion_cannot():
    check(executor_module.failed_step_can_retarget("lift", False)
          and not executor_module.failed_step_can_retarget("lift", True)
          and executor_module.failed_step_can_retarget("close", False)
          and not executor_module.failed_step_can_retarget("close", True)
          and executor_module.failed_step_can_retarget("down", True),
          "missed contact may re-localize only while the gripper is empty")


def _track_rig(module, aperture_at, observe_at):
    """A StrictExecutor with only what ``track`` reads, and a stand-in torch for its tensors."""
    import contextlib
    import types
    from types import SimpleNamespace

    import numpy as np

    class FT(np.ndarray):
        def clone(self):
            return self.copy()
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.asarray(self)

    fake = types.ModuleType("torch")
    fake.no_grad = contextlib.nullcontext
    fake.tensor = lambda data, device=None: np.array(data, dtype=float).view(FT)
    s = module.StrictExecutor.__new__(module.StrictExecutor)
    s.current = np.array([[0.30, -0.45, 0.25]], dtype=float).view(FT)
    s._hold = np.array([[0.007, 0.0, -0.092]], dtype=float)
    s.device = "cpu"
    s.env = SimpleNamespace(step=lambda action: None)
    commands = []
    s._action = lambda pose, grip: commands.append(float(pose[0][0]))
    s.frame_hook = None
    s.pause_hook = None
    s.track_log = []
    s.log = s.track_log.append
    s.correcting, s.corrected = True, 0
    s.aperture_centre = aperture_at
    s._gripper_pos = lambda: s.current.copy().view(FT)
    return s, commands, fake, observe_at


def _run_track(module, aperture_at, observe_at, steps=200):
    from unittest import mock

    s, commands, fake, observe = _track_rig(module, aperture_at, observe_at)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        res = s.track(observe=observe, target_z=0.20, steps=steps, aim=(0.30, -0.45))
    return res, s, commands


def test_delayed_ik_hover_servo_converges_for_every_supported_geometry():
    """A coupled, lagged tool model must settle every shape at hover before contact."""
    from types import SimpleNamespace
    from unittest import mock
    import numpy as np

    module = load_demo11()
    target = (.30, -.45)
    cases = (("cube", (.03, .03, .03)), ("ball", (.03, .03, .03)),
             ("cylinder", (.03, .03, .04)),
             ("triangular_prism", (.03, .03, .03)),
             ("hexagonal_prism", (.03, .03, .03)))
    for kind, size in cases:
        s, _commands, fake, _observe = _track_rig(
            module, lambda: target, lambda: target)
        s._tracking_prim = "target"
        s.layout = [SimpleNamespace(prim="target", kind=kind, size=size,
                                    container=False)]
        s._assist_ball_states = {}
        s._held = None
        body = np.array([.34, -.49, .25], dtype=float)
        tensor_type = type(s.current)

        def aperture():
            z_delta = body[2] - .25
            return (float(body[0] + .075 - .30 * z_delta),
                    float(body[1] + .035 + .45 * z_delta),
                    float(body[2] + .10))

        def step(_action):
            desired = np.asarray(s.current[0], dtype=float)
            body[:] += .28 * (desired - body)

        s.env.step = step
        s._gripper_pos = lambda: np.asarray(body.reshape(1, 3)).view(tensor_type)
        s.aperture_centre = aperture
        with mock.patch.dict(sys.modules, {"torch": fake}):
            result = s.track(observe=lambda: target, target_z=.20, steps=200,
                             action="above", every=10, aim=target)
        residual = math.dist(aperture()[:2], target)
        check(result.ok and residual <= module.PREGRASP_ALIGNMENT_TOLERANCE_M,
              f"{kind} reaches measured hover tolerance under coupled IK lag: "
              f"residual={residual * 1000:.1f} mm, {result.note}")


def test_a_constant_residual_stalls_and_the_servo_stops_correcting():
    module = load_demo11()
    part = (0.30, -0.45)
    res, s, commands = _run_track(module, lambda: (part[0] - 0.020, part[1]),
                                  lambda: part, steps=400)
    check("centrado detenido" in res.note, f"the stall is reported: {res.note}")
    check(res.ok is False, "a stalled descent is not a clean one")
    check(s.correcting is False and s.corrected <= module.STALL_WINDOW,
          f"no corrections after the stall: corrected={s.corrected}")
    check(commands[-1] == commands[-20],
          "the commanded pose stops moving once the stall is declared")


def test_tracking_phase_seed_preserves_the_measured_body_to_aperture_offset():
    import numpy as np
    module = load_demo11()
    body = (.30, -.45)
    aperture = (.28, -.46)
    target = (.30, -.45)
    command = module.body_command_for_aperture(body, aperture, target)
    check(np.allclose(command, (.32, -.44), atol=1e-9),
          f"phase handoff compensates measured aperture offset: {command}")


def test_tracking_starts_from_live_aperture_alignment_instead_of_raw_object_xy():
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)
    s, commands, fake, _observe = _track_rig(
        module, lambda: (part[0] - .020, part[1] + .010), lambda: part)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        s.track(observe=lambda: (part[0] + .080, part[1]), target_z=.20,
                steps=1, every=1,
                aim=part)
    expected = module.feedback_body_command(
        part, (part[0] - .020, part[1] + .010), part,
        max_step_m=module.LATERAL_SLACK_M * 4.0)
    check(abs(commands[0] - expected[0]) < 1e-9,
          f"first move uses a bounded live-aperture correction, got {commands[0]:.4f}")


def test_tracking_does_not_wind_up_when_the_arm_has_not_followed_its_command():
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)
    s, commands, fake, observe = _track_rig(
        module, lambda: (part[0] - .020, part[1]), lambda: part)
    measured_body = s.current.copy()
    s._gripper_pos = lambda: measured_body.copy()
    with mock.patch.dict(sys.modules, {"torch": fake}):
        s.track(observe=observe, target_z=.20, steps=4, every=1,
                action="above", aim=part)
    expected = part[0] + module.LATERAL_SLACK_M * 4.0
    check(len(commands) == 4 and all(abs(value - expected) < 1e-9 for value in commands),
          f"stale measured pose must hold one bounded target, not ratchet it: {commands}")


def test_tracking_converges_with_a_lagging_articulation_feedback_plant():
    from unittest import mock
    import numpy as np
    module = load_demo11()
    part = (.30, -.45)
    s, commands, fake, _observe = _track_rig(
        module, lambda: None, lambda: part)
    actual_body = np.array([.330, -.450, .250], dtype=float)
    aperture_offset = np.array([-.040, 0.0], dtype=float)

    def measured_body():
        return fake.tensor([actual_body], device="cpu")

    def delayed_step(pose):
        actual_body[:2] += .5 * (np.asarray(pose[0, :2], dtype=float) - actual_body[:2])

    s._gripper_pos = measured_body
    s.aperture_centre = lambda: tuple(actual_body[:2] + aperture_offset)
    s._action = lambda pose, grip: (commands.append(float(pose[0][0])) or pose)
    s.env.step = delayed_step
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: part, target_z=.20, steps=8, every=1,
                         action="above", aim=part)
    residual = np.linalg.norm(actual_body[:2] + aperture_offset - np.asarray(part))
    check(result.ok and residual < module.LATERAL_SLACK_M,
          f"the measured aperture converges through the integrated track loop: {residual:.4f} m")
    check(max(commands) - min(commands) < .001,
          f"delayed articulation feedback must not ratchet the body command: {commands}")


def test_contact_descent_aborts_before_motion_without_aperture_feedback():
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)
    s, commands, fake, observe = _track_rig(
        module, lambda: None, lambda: part)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=observe, target_z=.055, steps=20, every=1,
                         action="down", aim=part)
    check(not result.ok and result.grasp == "stale",
          "a contact descent reports a stale grasp if jaw-centre feedback is absent")
    check(not commands,
          f"the arm must not move toward contact without aperture measurement: {commands}")


def test_contact_descent_stops_on_the_first_lost_aperture_feedback_sample():
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)
    s, commands, fake, _observe = _track_rig(
        module, lambda: None, lambda: part)
    state = {"samples": 0, "steps": 0}

    def aperture_until_lost():
        state["samples"] += 1
        if state["samples"] >= 5:
            return None
        return (part[0] - .010, part[1])

    s.aperture_centre = aperture_until_lost
    s.env.step = lambda _action: state.__setitem__("steps", state["steps"] + 1)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: part, target_z=.055, steps=20,
                         every=1, action="down", aim=part)
    check(not result.ok and result.grasp == "stale",
          "mid-descent aperture feedback loss invalidates the contact approach")
    check(state["steps"] == 1 and len(commands) == 1,
          f"the down motion halts on the first missing sample: {state}")


def test_a_steadily_converging_residual_is_not_a_stall():
    module = load_demo11()
    part = (0.30, -0.45)
    looks = {"k": 0}

    def centre():
        looks["k"] += 1
        return (part[0] - max(0.020 - 0.001 * looks["k"], 0.0), part[1])

    res, s, _commands = _run_track(module, centre, lambda: part)
    check("centrado detenido" not in res.note,
          f"a residual shrinking by 1 mm per correction is converging, not stuck: {res.note}")
    check(s.correcting is True, "the servo is still correcting")


def test_a_blob_further_than_a_same_object_move_is_not_followed():
    module = load_demo11()
    part = (0.30, -0.45)
    res, s, commands = _run_track(module, lambda: (part[0] - 0.020, part[1]),
                                  lambda: (part[0] + 0.080, part[1]))
    check(res.ok is False, "a sighting 80 mm from the command is a different object, so lost")
    check(s.corrected == 0, "and no correction is made towards it")
    check(any("two consecutive target jumps" in line for line in s.track_log),
          "an isolated bad frame is rejected before the target is declared lost")
    check(module.SAME_OBJECT_MAX_SHIFT_M < 0.064,
          "the same-object gate must stay under the closest same-colour pair (64 mm)")


def test_one_isolated_tracking_jump_is_ignored_and_the_next_valid_frame_recovers():
    from unittest import mock
    module = load_demo11()
    part = (0.30, -0.45)
    frames = iter([(part[0] + 0.080, part[1]), part])
    s, _commands, fake, _ = _track_rig(
        module, lambda: (part[0] - 0.010, part[1]), lambda: next(frames, part))
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: next(frames, part), target_z=.20,
                         steps=120, every=20, aim=part)
    check(result.ok and "ignored 1 isolated target-jump frame" in result.note,
          f"one false blob does not abort a grasp and is reported: {result.note}")
    check(s.corrected > 0, "normal centring resumes after the next valid frame")


def test_one_anchor_violation_inside_the_step_gate_is_ignored_then_recovers():
    from unittest import mock
    module = load_demo11()
    part = (0.30, -0.45)
    frames = iter([(part[0] + .012, part[1]), part])
    s, _commands, fake, _ = _track_rig(
        module, lambda: part, lambda: next(frames, part))
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: next(frames, part), target_z=.20,
                         steps=120, every=20, action="down", aim=part)
    check(result.ok and "ignored 1 isolated target-jump frame" in result.note,
          f"one 12 mm anchor shift under the 25 mm step gate is treated as an outlier: {result.note}")


def test_repeated_anchor_violation_inside_the_step_gate_stops_descent():
    from unittest import mock
    module = load_demo11()
    part = (0.30, -0.45)
    shifted = (part[0] + .012, part[1])
    s, commands, fake, _ = _track_rig(module, lambda: part, lambda: shifted)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: shifted, target_z=.20,
                         steps=120, every=20, action="down", aim=part)
    check(not result.ok and len(commands) < 120
          and "two consecutive anchor violations" in " ".join(s.track_log),
          "persistent 12 mm motion is rejected before the gripper follows it")


def test_front_camera_can_coast_through_bounded_arm_occlusion():
    from unittest import mock
    module = load_demo11()
    s, commands, fake, _observe = _track_rig(
        module, lambda: (0.30, -0.45), lambda: None)
    s.allow_occluded_tracking = True
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: None, target_z=.20, steps=120,
                         every=20, aim=(.30, -.45))
    check(result.ok and "occluded look(s)" in result.note,
          f"front view retains a scene-linked aim during temporary arm occlusion: {result.note}")
    check(len(commands) == 120,
          "the bounded front-view fallback finishes this approach without changing its target")


def test_small_object_pregrasp_cancels_persistent_outlier_frames():
    from types import SimpleNamespace
    from unittest import mock
    module = load_demo11()
    part = (0.30, -0.45)
    s, commands, fake, _observe = _track_rig(
        module, lambda: part, lambda: (part[0] + 0.080, part[1]))
    s._tracking_prim = "small_ball"
    s.layout = [SimpleNamespace(prim="small_ball", kind="ball", size=(.02, .02, .02))]
    s._assist_ball_states = {}
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: (part[0] + 0.080, part[1]),
                         target_z=.20, steps=120, every=20,
                         action="above", aim=part)
    check(not result.ok and len(commands) > 120 and "could not find it" in result.note,
          f"20 mm targets coast above, then cancel if the final hover keeps seeing a shifted blob: {result.note}")


def test_small_object_downward_tracking_keeps_the_repeated_jump_stop():
    from types import SimpleNamespace
    from unittest import mock
    module = load_demo11()
    part = (0.30, -0.45)
    s, commands, fake, _observe = _track_rig(
        module, lambda: part, lambda: (part[0] + 0.080, part[1]))
    s._tracking_prim = "small_ball"
    s.layout = [SimpleNamespace(prim="small_ball", size=(.02, .02, .02))]
    s._assist_ball_states = {}
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: (part[0] + 0.080, part[1]),
                         target_z=.20, steps=120, every=20,
                         action="down", aim=part)
    check(not result.ok and len(commands) < 120,
          "small-object coasting ends before descent when the target ID still jumps")


def test_top_view_stops_after_the_bounded_scene_linked_coast():
    from unittest import mock
    module = load_demo11()
    s, commands, fake, _observe = _track_rig(
        module, lambda: (0.30, -0.45), lambda: None)
    # Even when the target is never seen again, the pre-approach scene link only authorizes a
    # short coast; it cannot become an open-loop motion lasting the full descent.
    s.current[0, 2] = .40
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=lambda: None, target_z=.20, steps=160,
                         every=20, aim=(.30, -.45))
    check(not result.ok and len(commands) == 139 and "6 occluded look(s)" in result.note,
          "top/side views stop after six scene-linked misses rather than continuing open-loop")


def test_a_late_top_view_occlusion_uses_bounded_last_seen_servo_updates():
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)
    looks = {"count": 0}

    def observe():
        looks["count"] += 1
        return part if looks["count"] == 1 else None

    s, commands, fake, _ = _track_rig(
        module, lambda: (part[0] - .010, part[1]), lambda: None)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=observe, target_z=.20, steps=120, every=20, aim=part)
    check(result.ok and "occluded look(s)" in result.note and len(commands) == 120,
          "a late miss can use the last scene-linked target after an earlier valid sighting")


def test_target_occlusion_below_visibility_gate_keeps_last_aim_for_two_looks():
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)
    s, commands, fake, _ = _track_rig(module, lambda: part, lambda: part)
    s.current[0, 2] = .30
    looks = {"count": 0}

    def observe():
        looks["count"] += 1
        return part if looks["count"] <= 4 else None

    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=observe, target_z=.163, steps=120,
                         every=20, action="down", aim=part)
    check(result.ok and len(commands) == 120 and "2 occluded look(s)" in result.note,
          "two expected late wrist occlusions retain the last aim through contact height")


def test_a_wrist_that_hides_the_part_for_several_looks_does_not_cancel_a_centred_descent():
    """Strict logs: "8 occluded look(s) ... 1 look(s) could not find it" at 0.3-2.4 mm aperture.

    A prism's rotated wrist hides the part from the top camera for the whole descent. The
    aperture is servoed to the scene-linked anchor from the arm's own feedback meanwhile, and a
    part knocked in the dark fails the close-time reach and drift guards, so the misses are
    coasted through up to ``TRACK_MAX_LATE_OCCLUSION``; an unbounded blackout still stops the
    descent before the jaws close.
    """
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)

    def descend(seen_first, steps, every):
        s, commands, fake, _ = _track_rig(module, lambda: part, lambda: part)
        s.current[0, 2] = .25
        looks = {"count": 0}

        def observe():
            looks["count"] += 1
            return part if looks["count"] <= seen_first else None

        with mock.patch.dict(sys.modules, {"torch": fake}):
            return s.track(observe=observe, target_z=.163, steps=steps, every=every,
                           action="down", aim=part), looks["count"] - seen_first

    coasted, misses = descend(seen_first=2, steps=240, every=20)
    check(8 < misses <= module.TRACK_MAX_LATE_OCCLUSION and coasted.ok
          and "occluded look(s)" in coasted.note and "could not find it" not in coasted.note,
          f"{misses} consecutive misses are coasted: ok={coasted.ok}, {coasted.note[-90:]}")
    blind, misses = descend(seen_first=2, steps=400, every=10)
    check(misses > module.TRACK_MAX_LATE_OCCLUSION and not blind.ok
          and "could not find it" in blind.note,
          "a blackout longer than the budget still cancels the descent before contact")


def test_late_occlusion_coast_applies_to_every_grasped_geometry():
    from unittest import mock
    module = load_demo11()
    part = (.30, -.45)
    looks = {"count": 0}

    def observe():
        looks["count"] += 1
        return part if looks["count"] == 1 or looks["count"] >= 4 else None

    s, commands, fake, _ = _track_rig(module, lambda: part, lambda: part)
    s.current[0, 2] = .30
    s._tracking_prim = "cube"
    s.item_kinds = {"cube": "cube"}
    s.item_sizes = {"cube": (.030, .030, .030)}
    s._assist_ball_states = {}
    s._held = None
    with mock.patch.dict(sys.modules, {"torch": fake}):
        result = s.track(observe=observe, target_z=.163, steps=120,
                         every=20, action="down", aim=part)
    check(result.ok and len(commands) == 120 and looks["count"] >= 4,
          "a cube keeps the validated anchor through two expected wrist occlusions: "
          f"ok={result.ok}, commands={len(commands)}, note={result.note}")


def test_clearance_descent_starts_from_measured_aperture_not_body_offset():
    import numpy as np
    from types import SimpleNamespace

    module = load_demo11()
    body = np.array([0.40, -0.50, 0.30])
    hold = np.array([-0.01, 0.02, -0.09])
    measured = SimpleNamespace(aperture_centre=lambda: (0.37, -0.44, 0.21))
    aim = module.executor_module.lowering_start_aim(measured, body, hold)
    check(aim == (0.37, -0.44),
          "the clearance descent continues from measured jaw-centre XY")
    fallback = SimpleNamespace()
    aim = module.executor_module.lowering_start_aim(fallback, body, hold)
    check(aim == (0.39, -0.48),
          "robots without aperture feedback retain the calibrated body-offset fallback")


def test_the_exit_code_is_the_status_of_the_ledger():
    module = load_demo11()
    ledger = module.TaskLedger
    cases = [
        ([("a", True), ("b", True)], False, "SUCCESS", 0),
        ([("a", False)], False, "FAILED", 2),
        ([("a", True), ("b", False)], False, "PARTIAL_SUCCESS", 2),
        ([], False, "NO_TASKS", 0),
        ([("a", True)], True, "ABORTED", 3),
    ]
    for records, aborted, status, code in cases:
        led = ledger()
        for key, ok in records:
            led.record(key, ok)
        check(led.status(aborted=aborted) == status,
              f"{records} aborted={aborted} must be {status}, got {led.status(aborted=aborted)}")
        check(module.EXIT_CODE[status] == code, f"{status} must exit {code}")


# -- phase 2: one command pipeline, with ids, states and a stop that cannot be lost ---------


def test_a_stop_is_never_queued_behind_anything_and_cancels_what_waits():
    bus = telemetry.Bus()
    first = bus.submit("separa las piezas por colores")
    second = bus.submit("coge la bola verde")
    relook = bus.send_control("relook")
    stop = bus.send_control("stop")

    check(bus.stop_requested(), "STOP sets its flag at once, whatever is queued")
    check(bus.take_command(timeout=0.05) is None,
          "the queued sentences are cancelled, not run after the stop")
    check(bus.take_control() is None, "and the queued controls are cancelled too")
    for cid in (first, second, relook):
        state = bus.command_state(cid)
        check(state["state"] == telemetry.CANCELLED and state["note"],
              f"{cid} must end CANCELLED with its reason, got {state}")
    check(bus.command_state(stop)["state"] == telemetry.QUEUED,
          "the stop waits to be acknowledged by the robot, not assumed done")
    bus.complete_stop()
    check(bus.command_state(stop)["state"] == telemetry.SUCCESS,
          "the robot's acknowledgement completes the stop")


def test_repeated_relook_is_coalesced_and_the_next_one_is_queued_again():
    bus = telemetry.Bus()
    first = bus.send_control("relook")
    second = bus.send_control("relook")
    third = bus.send_control("relook")
    check(bus.command_state(first)["state"] == telemetry.QUEUED,
          "the first relook is queued")
    check(bus.command_state(second)["state"] == telemetry.CANCELLED
          and "esperando" in bus.command_state(second)["note"],
          "a repeat while one waits is cancelled with the reason, not run again")
    check(bus.command_state(third)["state"] == telemetry.CANCELLED,
          "every repeat is answered, not just the first")
    taken = bus.take_control()
    check(taken and taken["id"] == first and bus.take_control() is None,
          "one relook is carried out, not three")
    again = bus.send_control("relook")
    check(bus.command_state(again)["state"] == telemetry.QUEUED,
          "once the waiting one is taken, a new relook is accepted again")


def test_a_command_moves_only_along_the_legal_lifecycle():
    bus = telemetry.Bus()
    cid = bus.submit("pon el cubo rojo en la caja roja")
    check(bus.command_state(cid)["state"] == telemetry.QUEUED, "accepted: QUEUED")
    check(bus.take_command(timeout=0.1) == "pon el cubo rojo en la caja roja",
          "the robot takes it")
    check(bus.command_state(cid)["state"] == telemetry.RUNNING, "taken: RUNNING")
    check(bus.set_command_state(cid, telemetry.SUCCESS), "RUNNING may end SUCCESS")
    check(not bus.set_command_state(cid, telemetry.RUNNING),
          "a finished command cannot go back to RUNNING")
    check(bus.command_state(cid)["state"] == telemetry.SUCCESS, "and it stays SUCCESS")


def test_a_full_queue_refuses_with_a_reason_and_never_drops_silently():
    bus = telemetry.Bus()
    ids = [bus.submit(f"orden {index}") for index in range(telemetry.MAX_QUEUED_COMMANDS)]
    overflow = bus.submit("una más")
    check(bus.command_state(overflow)["state"] == telemetry.FAILED
          and "demasiadas" in bus.command_state(overflow)["note"],
          "the sentence that does not fit is FAILED with the reason")
    check(len(set(ids + [overflow])) == len(ids) + 1, "every request has its own id")
    check(all(bus.command_state(cid)["state"] == telemetry.QUEUED for cid in ids),
          "the sentences that did fit are still waiting, untouched")
    snap_ids = [cmd["id"] for cmd in bus.snapshot()["commands"]]
    check(overflow in snap_ids, "the page can see the refused request in the snapshot")


def test_a_control_waiting_in_the_wait_loop_is_carried_out_not_dropped():
    from unittest import mock

    module = load_demo11()
    calls = []

    class Probe(module.MonitoredRobot):
        def __init__(self):
            self.bus = telemetry.Bus()
            self.voice = None
            self.web_input = True
            self.keyboard = None
            self.advance = False
            self.say = lambda *a, **k: None
            self.look = lambda caption="": calls.append(caption)

    probe = Probe()
    relook = probe.bus.send_control("relook")
    threading.Timer(0.3, lambda: probe.bus.submit("coge la bola verde")).start()
    got = module.MonitoredRobot.ask(probe)
    check(got == "coge la bola verde", f"the sentence still arrives: {got!r}")
    check(calls == ["re-observación del panel"],
          f"the relook was carried out while waiting, not logged as 'later': {calls}")
    check(probe.bus.command_state(relook)["state"] == telemetry.SUCCESS,
          "and its command says so")


def test_a_failed_relook_is_a_failed_command_with_the_reason():
    module = load_demo11()

    class Probe(module.MonitoredRobot):
        def __init__(self):
            self.bus = telemetry.Bus()
            self.say = lambda *a, **k: None
            self.shown = None

        def look(self, caption=""):
            raise RuntimeError("la cámara no responde")

    probe = Probe()
    cid = probe.bus.send_control("relook")
    probe.bus.take_control()
    module.MonitoredRobot._apply_control(probe, {"action": "relook", "id": cid, "params": {}})
    state = probe.bus.command_state(cid)
    check(state["state"] == telemetry.FAILED and "la cámara no responde" in state["note"],
          f"a failed look is FAILED, with the reason, not swallowed: {state}")


def test_keys_carries_out_a_bounded_number_of_controls_per_step():
    from unittest import mock

    module = load_demo11()
    counts = {"look": 0, "next": 0}

    class Probe(module.MonitoredRobot):
        def __init__(self):
            self.bus = telemetry.Bus()
            self.say = lambda *a, **k: None
            self.cam_fps = 0.0
            self.advance = False
            self.shown = None

        def look(self, caption=""):
            counts["look"] += 1

    probe = Probe()
    for action in ("relook", "next", "next", "next", "next"):
        probe.bus.send_control(action)
    with mock.patch.object(module.MonitoredRobot.__bases__[0], "keys", lambda self: set()):
        probe.keys()
    waiting = probe.bus.snapshot()["commands"]
    still_queued = [c for c in waiting if c["state"] == telemetry.QUEUED]
    check(len(still_queued) == 5 - module.MAX_CONTROLS_PER_STEP,
          f"only {module.MAX_CONTROLS_PER_STEP} controls run per step; the rest wait in order: "
          f"{len(still_queued)} still queued")
    check(counts["look"] == 1, "a burst of relooks costs at most the perception pass it asks for")


def test_stop_in_a_step_quits_the_run_even_with_controls_waiting():
    from unittest import mock

    module = load_demo11()

    class Probe(module.MonitoredRobot):
        def __init__(self):
            self.bus = telemetry.Bus()
            self.say = lambda *a, **k: None
            self.cam_fps = 0.0
            self.advance = False
            self.shown = None

    probe = Probe()
    probe.bus.send_control("next")
    probe.bus.send_control("stop")
    with mock.patch.object(module.MonitoredRobot.__bases__[0], "keys", lambda self: set()):
        pressed = probe.keys()
    check("q" in pressed and probe.quit, "STOP quits the run at this step")


def test_the_dashboard_acknowledges_with_the_command_id_and_its_state():
    from monitor import server as monitor_server

    bus = telemetry.Bus()
    cid = bus.submit("separa las piezas")
    answer = monitor_server._ack(bus, cid, queued="separa las piezas")
    check(answer["id"] == cid and answer["state"] == telemetry.QUEUED and answer["ok"],
          f"the page gets the id and the state it is in: {answer}")


def test_a_snapshot_after_a_sequence_number_carries_only_the_new_log_lines():
    bus = telemetry.Bus()
    for index in range(5):
        bus.log(f"linea {index}")
    first = bus.snapshot()
    check(len(first["events"]) == 5, "a fresh page gets the recent history")
    seen = first["seq"]
    bus.log("linea 5")
    bus.log("linea 6")
    delta = bus.snapshot(events_after=seen)
    check([e["message"] for e in delta["events"]] == ["linea 5", "linea 6"],
          f"the delta is exactly what came after the last line sent: {delta['events']}")
    check(bus.snapshot(events_after=delta["seq"])["events"] == [],
          "and an up-to-date socket gets no log lines at all")
    check(len(bus.snapshot()["events"]) == 7, "a full snapshot still has the whole recent log")


def test_the_bus_measures_rates_depths_and_command_latency():
    bus = telemetry.Bus()
    for _ in range(10):
        bus.tick("physics")
    check(bus.rate("physics") > 0, "ticks become a rate")
    check(bus.rate("camera") == 0.0, "a stream that never ticked reads zero, not an error")
    cid = bus.submit("coge la bola verde")
    bus.take_command(timeout=0.1)
    metrics = bus.snapshot()["metrics"]
    check("physics_per_s" in metrics and metrics["physics_per_s"] > 0,
          f"the snapshot carries the rate: {metrics}")
    check("command_latency_ms" in metrics and metrics["command_latency_ms"] >= 0,
          "the latency from request to pick-up is measured")
    check(metrics["command_queue_depth"] == 0 and metrics["control_queue_depth"] == 0,
          "queue depths are reported, and empty here")
    bus.add_to_metric("websocket_clients", 1)
    bus.add_to_metric("websocket_clients", 1)
    bus.add_to_metric("websocket_clients", -1)
    check(bus.snapshot()["metrics"]["websocket_clients"] == 1, "client count goes up and down")
    check(bus.command_state(cid)["state"] == telemetry.RUNNING, "the command is RUNNING")


def test_end_to_end_over_http_a_command_is_acknowledged_taken_finished_and_stopped():
    """Phase 5's first steps, without Isaac: the real server, the real HTTP calls, a robot stand-in."""
    from monitor import server as monitor_server

    bus = telemetry.Bus()
    monitor = monitor_server.serve(bus, port=0)
    try:
        check(monitor.wait_ready(15.0), "the dashboard comes up")

        def call(path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            request = urllib.request.Request(monitor.url + path.lstrip("/"), data=data,
                                             method="POST" if body is not None else "GET",
                                             headers={"content-type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.loads(response.read())

        def command_in_page(cid):
            return next((c for c in call("api/state")["commands"] if c["id"] == cid), None)

        ack = call("api/command", {"text": "separa las piezas por colores"})
        check(ack["ok"] and ack["id"].startswith("cmd_") and ack["state"] == telemetry.QUEUED,
              f"the browser is acknowledged at once, with an id and QUEUED: {ack}")
        check(command_in_page(ack["id"])["state"] == telemetry.QUEUED,
              "and the page sees the same state")

        check(bus.take_command(timeout=0.5) == "separa las piezas por colores",
              "the robot takes it")
        check(command_in_page(ack["id"])["state"] == telemetry.RUNNING,
              "the page sees RUNNING once the robot has it")
        bus.set_command_state(ack["id"], telemetry.SUCCESS, "pieza colocada")
        check(command_in_page(ack["id"])["state"] == telemetry.SUCCESS,
              "and SUCCESS, with the robot's own note")

        waiting = call("api/command", {"text": "coge la bola verde"})
        stop = call("api/env", {"action": "stop"})
        check(stop["ok"] and stop["state"] == telemetry.QUEUED,
              "STOP is acknowledged at once, and waits for the robot to confirm it")
        check(command_in_page(waiting["id"])["state"] == telemetry.CANCELLED,
              "a sentence still waiting when STOP arrives is cancelled, and says so")
        check(bus.stop_requested(), "the robot sees the stop flag")
        bus.complete_stop()
        check(command_in_page(stop["id"])["state"] == telemetry.SUCCESS,
              "the stop is SUCCESS once the robot has halted")
        check("command_latency_ms" in call("api/state")["metrics"],
              "the page's metrics include the measured latency")
    finally:
        monitor.stop(timeout=5.0)


def test_the_start_up_phases_follow_the_real_stages_and_a_failure_is_recorded():
    """"Isaac is starting" may only be published once Isaac is starting, and a failure must say so."""
    source = (ROOT / "11_real_sorting_with_monitor.py").read_text(encoding="utf-8")
    wait = source.index('phase="waiting_start"')
    loading = source.index('phase="loading"')
    starting = source.index('phase="starting", voice=')
    check(wait < loading < starting,
          "the order is waiting for the press, then loading the models, then Isaac starting")
    startup = source[starting:source.index("    viewer = None", starting)]
    check('phase="failed"' in startup and "except BaseException as exc" in startup,
          "a failure while Isaac starts is recorded as failed, with its reason")
    check("tiempo de espera agotado" in source, "a start that times out is logged, not silent")
    check("loading:" in (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8"),
          "the page names the loading stage")


def test_the_pre_flight_findings_live_in_the_info_popup_behind_an_icon_button_in_the_header():
    """The findings are not on the main screen; an icon button in the top bar opens them."""
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    header = page[page.index("<header>"):page.index("</header>")]
    button = header[header.index('id="info-btn"'):]
    check("<svg" in button and "</svg>" in button, "the info button carries an icon")
    import re
    inner = button[button.index(">") + 1:button.index("</button>")]
    check(re.sub(r"<svg.*?</svg>", "", inner, flags=re.S).strip() == "",
          "and no text beside the icon")
    popup = page[page.index('id="info-modal"'):page.index("</main>")]
    check('id="findings"' in popup, "the findings are in the info popup")
    main_part = page[page.index("<main>"):page.index("</main>")]
    check('id="findings"' not in main_part, "and not on the main screen any more")


def test_the_log_is_a_popup_behind_an_icon_button_in_the_header_and_not_on_the_main_screen():
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    header = page[page.index("<header>"):page.index("</header>")]
    check('id="log-btn"' in header, "the registro button is in the top bar")
    main_part = page[page.index("<main>"):page.index("</main>")]
    check('id="log"' not in main_part and 'id="diag"' not in main_part,
          "the log and its diagnostics are not on the main screen")
    popup = page[page.index('id="log-modal"'):page.index("<main>")]
    check('id="log"' in popup and 'id="diag"' in popup, "the log and diagnostics are in its popup")


def test_the_camera_leads_the_screen_and_control_sits_with_the_environment():
    """From the top: the stage rail, then entorno and comando side by side, then what the robot
    understood (70% of the width) beside the objective (30%), then the camera, the perception and the
    plan in one row, the camera widest. Start, stop and the moving bar share the entorno card, and
    without a frame the camera says it is not connected.

    The first version of this page had the camera nested in a control column that was never closed:
    it was 2 px tall at 1366x768 and nothing caught it, because the old check only read strings.
    The layout itself is measured in a browser by the screenshot run; this test pins the order of
    the blocks so a column cannot silently move to the wrong place.
    """
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    check(page.index('id="stage-rail"') < page.index('class="ctrl-row"')
          < page.index('class="top-row"') < page.index('class="row4"'),
          "the order is: stage rail, entorno and comando, the top row, then camera, perception and plan")
    top = page[page.index('class="top-row"'):page.index('class="row4"')]
    understand = top[top.index('class="card understand-card"'):top.index('id="objetivo-card"')]
    objetivo = top[top.index('id="objetivo-card"'):]
    row4 = page[page.index('class="row4"'):page.index('id="paint-error"')]
    camera = row4[row4.index('class="card cam-card"'):row4.index('class="card perc-card"')]
    perc = row4[row4.index('class="card perc-card"'):row4.index('class="card plan-card"')]
    plan = row4[row4.index('class="card plan-card"'):]
    check('id="opt-intent"' in understand and 'id="note-understand"' not in understand,
          "what the robot understood sits in the wide card of the top row, as its classification")
    check('id="hero-tasks-list"' in objetivo and 'id="result-command"' in objetivo,
          "the objective and its tasks sit beside it, in the narrow card")
    check('id="frame"' in camera and 'id="cam-empty"' in camera and 'id="route-now"' in camera,
          "the camera, with the line that says what the arm is doing, is in the row")
    check('id="detections"' in perc, "the perception table is the middle card of the row")
    check('id="exec-steps"' in plan and 'id="hero-tasks-list"' not in plan,
          "the plan is the last card, and it is the movement matrix: the checklist lives in the objective")
    check(row4.index('class="card cam-card"') < row4.index('class="card perc-card"') < row4.index('class="card plan-card"'),
          "the cards of the row read camera, perception, plan, left to right")
    check("cámara aún no conectada" in page, "without a frame the camera says it is not connected")
    bar = page[page.index('class="ctrl-row"'):page.index('class="top-row"')]
    check('id="start"' in bar and 'id="stop"' in bar and 'id="start-bar"' in bar,
          "start, stop and the moving bar sit in the entorno card")
    check('id="env-summary"' in bar and 'id="send"' in bar,
          "the environment summary and the command box share the row")
    check('class="help"' in understand and 'class="help"' in objetivo,
          "the understanding and the objective each carry an info icon in their title bar")
    check("startPending" in page and "RUNNING_PHASES" in page,
          "a press blocks the start button until the environment is running")


def test_the_environment_configuration_is_one_popup_opened_left_of_the_start_button():
    """The configuration used to be an expandable panel inside the environment card, which grew the
    right column until the camera was below the fold: what a watched run needs is the state and the
    camera, and what a one-off start needs is a form. Now every setting that decides how the
    environment is built lives in a single popup, opened by a button to the left of "iniciar
    entorno", and the environment group is left saying in one line what will be used."""
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    main_part = page[page.index("<main>"):page.index("</main>")]
    entorno = main_part[main_part.index('id="env-group"'):main_part.index('id="cmd-group"')]
    popup = page[page.index('id="env-modal"'):page.index("<main>")]
    button = entorno[entorno.index('id="env-config"'):]
    check(entorno.index('id="env-config"') < entorno.index('id="start"'),
          "the configuration button sits to the left of iniciar entorno")
    check("<svg" in button[:button.index("</button>")], "and it carries an icon, not a word on its own")
    for name in ("scene-objects", "scene-containers", "scene-add-object", "scene-add-container",
                 "scene-save", "scene-simple", "scene-start", "scene-message",
                 "variant-sel", "camera-sel", "rounds", "seed-in", "startup-command"):
        check(f'id="{name}"' in popup, f"{name} lives in the configuration popup")
        check(f'id="{name}"' not in entorno, f"and no longer inside the environment panel")
    check('id="scene-editor"' not in page and '<textarea id="scene-json"' not in page
          and '<textarea id="scene-import"' not in page,
          "the old expandable panel and JSON text boxes are gone")
    check('id="scene-map"' in popup and 'id="scene-summary"' in popup,
          "the popup draws the table and states in one line what is on it")
    for name in ("work-area-x", "work-area-y", "work-area-width", "work-area-depth"):
        check(f'id="{name}"' in popup, f"{name} configures the working area")
    check('data-work-area="true"' in page and 'closest("[data-id], [data-work-area]")' in page,
          "the working area can be moved directly on the scene map")
    check("width: min(1440px, calc(100vw - 32px))" in page
          and "height: min(96vh, 1080px)" in page,
          "the environment editor uses the expanded desktop dimensions")
    check(popup.index('id="variant-sel"') < popup.index('id="scene-map"'),
          "the session comes first, and the table under it")
    check(popup.index('id="scene-map"') < popup.index('id="scene-objects"'),
          "the table is read before the lists it comes from")
    check('class="split"' in popup and popup.index('id="scene-objects"') < popup.index('id="scene-containers"'),
          "the pieces and the receptacles sit side by side, each with its own heading")
    check('{ modal: "env-modal", open: "env-config", close: "env-close" }' in page,
          "the popup opens and closes with that button, and with Escape and a click outside")
    check('setPopupOpen(POPUPS.find(p => p.modal === "env-modal"), false)' in page,
          "starting the environment closes it, so the start bar is visible while Isaac boots")
    check('command: el("startup-command").value.trim()' in page,
          "the initial task is sent with the environment settings")
    check('id="env-summary"' in entorno and "paintEnvSummary()" in page,
          "the environment group still states, in one line, what the next start will use")
    foot = popup[popup.index('id="scene-message"'):]
    for name in ("scene-simple", "scene-save", "scene-start"):
        check(f'id="{name}"' in foot, f"{name} is an action of the footer, not a stray in the form")
    check(".modal-foot .actions button { flex: 1 1 0;" in page,
          "the three actions share the width of the footer, a third each")


def run_dashboard_controls():
    """Load the shipped page, press its configuration button, and report what each press did.

    ``tests/dashboard_controls.js`` runs the page's own script against a DOM that keeps the
    handlers it was given, so a click is a click and a change is a change - the part of "is this
    button wired to that window" that no amount of markup reading can answer. Returns None when
    node is not installed, and a report with a ``loadError`` when the page's script threw.
    """
    node = shutil.which("node")
    if node is None:
        return None
    done = subprocess.run([node, str(ROOT / "tests" / "dashboard_controls.js"),
                           str(ROOT / "monitor" / "static" / "index.html")],
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    if done.returncode != 0:
        return {"loadError": done.stderr[-400:]}
    return json.loads(done.stdout)


def test_the_environment_button_opens_the_popup_and_start_reads_what_it_holds():
    """Markup can be perfect and the button dead; this presses it, for real, in the page.

    The configuration moved out of the panel and into a window, and the risk of a move like that
    is silent: the ids are in the page either way, so a markup test passes on a button wired to
    nothing. So the page's own script runs here, the button is clicked, the form is changed, and
    the start press is watched to see what it would have sent.
    """
    report = run_dashboard_controls()
    if report is None:
        print("  (node not found: the dashboard's own controls were not exercised)")
        return
    check(not report.get("loadError"), f"the page's script loads ({report.get('loadError')})")
    for entry in report.get("steps", []):
        check(entry["ok"], f"{entry['name']}" + (f": {entry['detail']}" if not entry["ok"] else ""))
    check(not report.get("warnings"), f"nothing on the page warns: {report.get('warnings')}")


def _presentation_snapshot(phase, node_states, tasks=None, detections=None,
                           command="", classification=None, status=""):
    """A real bus snapshot shaped for the presentation layer test."""
    bus = telemetry.Bus()
    for node_id, (state, note) in node_states.items():
        bus.node(node_id, state, note)
    fields = {"phase": phase, "moving": phase in ("sorting", "command")}
    if tasks is not None:
        fields["tasks"] = tasks
    if detections is not None:
        fields["detections"] = detections
    if command:
        fields["command"] = command
    if classification is not None:
        fields["classification"] = classification
    if status:
        fields["status"] = status
    bus.set_state(**fields)
    return [bus.snapshot()]


def test_the_presentation_rail_and_tasks_mirror_real_telemetry():
    """The judge-facing six-stage rail, the objective and the task list are read from the robot's
    own state, never invented. The rail groups the twelve map stages into VER, ENTENDER,
    PLANIFICAR, ACTUAR, VERIFICAR, RECUPERARSE; the task list never marks a task completed before
    the robot says so.
    """
    report = dashboard_client_report([
        _presentation_snapshot(
            "sorting",
            {"capture": (telemetry.OK, "2 cubos"), "perceive": (telemetry.OK, ""),
             "link": (telemetry.OK, ""), "ground": (telemetry.OK, ""),
             "understand": (telemetry.OK, "ordenar por color"), "plan": (telemetry.OK, "2 pasos"),
             "execute": (telemetry.ACTIVE, "trasladar")},
            tasks=[{"key": "a", "part": "cubo rojo", "destination": "roja",
                    "label": "cubo rojo \u2192 roja", "done": True, "state": "done"},
                   {"key": "b", "part": "cubo azul", "destination": "azul",
                    "label": "cubo azul \u2192 azul", "done": False, "state": "doing"}],
            detections=[{"id": 0, "kind": "cube", "color": "red", "prim": "red", "container": False},
                        {"id": 1, "kind": "cube", "color": "blue", "prim": "blue", "container": False}],
            command="separa las piezas por colores",
            classification={"summary": "ordenar las piezas por color"}),
    ])
    if report is None:
        print("  (SKIPPED: node is not installed, so the page was never loaded)")
        return
    check(not report.get("loadError"), f"the page loads: {report.get('loadError')}")
    check(not report["throws"], f"painting the presentation must not throw: {report['throws']}")
    p = report.get("present") or {}
    check(p.get("stageCount") == 6, f"six stages, got {p.get('stageCount')}")
    check(p.get("stageNames") == ["VER", "ENTENDER", "PLANIFICAR", "ACTUAR", "VERIFICAR", "RECUPERARSE"],
          f"the stages read as the story: {p.get('stageNames')}")
    check(p.get("stageClasses", [])[:3] == ["stage ok", "stage ok", "stage ok"],
          f"ver, entender and plan have finished: {p.get('stageClasses')}")
    check(p.get("stageClasses", [None] * 4)[3] == "stage active", "the acting stage is highlighted")
    check(p.get("stageClasses", [None] * 6)[4] == "stage pending"
          and p.get("stageClasses", [None] * 6)[5] == "stage pending",
          "verify and recover stay pending until they happen")
    check(p.get("heroObjective", "").startswith("Orden: «")
          and "separa las piezas por colores" in p.get("heroObjective", ""),
          f"the objective is the real command: {p.get('heroObjective')}")
    check(p.get("heroTaskRows") == 2, f"both tasks are listed: {p.get('heroTaskRows')}")
    check(p.get("heroStates") == {"doing": 1, "done": 1, "fail": 0, "todo": 0},
          f"the task states mirror the robot, done only after verification: {p.get('heroStates')}")
    check("1 de 2 completadas" in p.get("heroTally", ""), f"the tally counts verified tasks: {p.get('heroTally')}")
    check(p.get("demoMessage", "").startswith("Ejecutando la orden")
          and "cubo" not in p.get("demoMessage", ""),
          f"the demo line names the phase, and the camera line says what is moving: {p.get('demoMessage')}")


def test_the_presentation_reports_success_and_recovery_honestly():
    """A finished run reads as completed, a recovery reads as recovering - not as either success or
    a hard failure - and nothing is completed that was not physically verified.
    """
    done = dashboard_client_report([
        _presentation_snapshot(
            "done",
            {"capture": (telemetry.OK, ""), "perceive": (telemetry.OK, ""), "link": (telemetry.OK, ""),
             "ground": (telemetry.OK, ""), "understand": (telemetry.OK, ""), "plan": (telemetry.OK, ""),
             "execute": (telemetry.OK, ""), "verify": (telemetry.OK, "")},
            tasks=[{"key": "a", "label": "a", "done": True, "state": "done"},
                   {"key": "b", "label": "b", "done": True, "state": "done"}],
            command="separa las piezas por colores", status="SUCCESS"),
    ])
    rec = dashboard_client_report([
        _presentation_snapshot(
            "recovering",
            {"capture": (telemetry.OK, ""), "execute": (telemetry.OK, ""), "plan": (telemetry.OK, ""),
             "verify": (telemetry.WARN, "no dentro"), "recover": (telemetry.ACTIVE, "reintentando")},
            tasks=[{"key": "a", "label": "a", "done": True, "state": "done"},
                   {"key": "b", "label": "b", "done": False, "state": "doing"}],
            command="separa las piezas por colores"),
    ])
    if done is None or rec is None:
        print("  (SKIPPED: node is not installed, so the page was never loaded)")
        return
    dp = done.get("present") or {}
    check(dp.get("demoStatus", "").startswith("ok"), f"a full success reads ok: {dp.get('demoStatus')}")
    check("completado" in dp.get("demoMessage", ""), f"and says completed: {dp.get('demoMessage')}")
    check(dp.get("heroStates", {}).get("done") == 2 and dp.get("heroStates", {}).get("fail") == 0,
          f"both tasks completed after physical verification: {dp.get('heroStates')}")
    rp = rec.get("present") or {}
    check(rp.get("demoStatus", "").startswith("warn"), f"recovery reads warn, not a failure: {rp.get('demoStatus')}")
    check(rp.get("heroRecovering") >= 1, f"the piece being retried says RECUPERANDO: {rp.get('heroRecovering')}")
    check(rp.get("heroStates", {}).get("done") == 1 and rp.get("heroStates", {}).get("doing") == 1,
          f"one verified, one still in recovery: {rp.get('heroStates')}")


def test_the_presentation_does_not_invent_a_scene_before_it_exists():
    """An idle dashboard shows every stage pending and says so, rather than inventing progress."""
    report = dashboard_client_report([_presentation_snapshot("waiting_start", {})])
    if report is None:
        print("  (SKIPPED: node is not installed, so the page was never loaded)")
        return
    p = report.get("present") or {}
    check(p.get("stageCount") == 6 and all(c == "stage pending" for c in p.get("stageClasses", [])),
          f"every stage is pending before anything happens: {p.get('stageClasses')}")
    check("Sin orden" in p.get("heroObjective", ""), f"no objective is invented: {p.get('heroObjective')}")
    check(p.get("demoStatus", "").startswith("idle"), f"an idle dashboard says so: {p.get('demoStatus')}")


def test_the_one_click_demo_is_in_the_header_and_the_advanced_controls_stay_reachable():
    """The primary action is one button in the header; the environment, the full route and the log
    stay reachable behind their buttons, and the camera keeps its place as the biggest element.
    """
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    header = page[page.index("<header>"):page.index("</header>")]
    check('id="demo-run"' in header and "Iniciar demo" in header,
          "the one-click demo is the header's primary action")
    check('id="route-btn"' in header and 'id="route-modal"' in page
          and 'id="flow-svg"' in page and 'id="route-now"' in page,
          "the full twelve-stage route stays reachable in its own window")
    check('id="stage-rail"' in page, "the six-stage rail is on the main screen")
    for name in ("hero-objective", "hero-tasks-list", "hero-tally", "demo-status"):
        check(f'id="{name}"' in page, f"{name} is declared for the presentation")
    main_part = page[page.index("<main>"):page.index("</main>")]
    check('id="stage-rail"' in main_part and 'id="demo-status"' in main_part,
          "the rail and the demo status are in the main screen, not hidden in a window")
    check('id="findings"' not in main_part and 'id="log"' not in main_part,
          "findings and the log stay off the main screen")


def test_the_motion_loop_does_not_render_every_physics_step():
    """The per-step render is what ran the simulation at half real time; it is turned off while an
    Executor drives, and the camera still gets its frames from explicit renders.

    Measured with tools/probe_step_cost.py: a motion step is ~32 ms with a render and ~4 ms of
    physics without, so the render was ~88% of the step - and it is redundant, because the live
    picture and every tracked re-aim read the camera through ``isaac_io.capture``/``capture_rgb``,
    which render themselves. This guards the two halves: the Executor raises the render interval,
    and the capture path still renders.
    """
    source = (ROOT / "executor.py").read_text(encoding="utf-8")
    check(re.search(r"MOTION_RENDER_INTERVAL\s*=\s*(1000000|1_000_000)", source) is not None,
          "the render interval the motion loop uses is far beyond any single command's step count")
    check("self.env.cfg.sim.render_interval = MOTION_RENDER_INTERVAL" in source,
          "the Executor turns off the per-step render while it drives")
    io_source = (ROOT / "isaac_io.py").read_text(encoding="utf-8")
    for name in ("capture_rgb", "capture"):
        idx = io_source.index(f"def {name}(")
        nxt = io_source.find("\ndef ", idx + 1)
        block = io_source[idx: nxt if nxt > 0 else len(io_source)]
        check("_render_camera_rgb(env, sensor" in block,
              f"{name} uses the bounded camera warm-up capture")
    helper = io_source[io_source.index("def _render_camera_rgb("):
                       io_source.index("def capture_rgb(")]
    check("env.sim.render()" in helper and "force_recompute=True" in helper,
          "camera warm-up renders and refreshes each attempted buffer")
    check("CAMERA_CAPTURE_MAX_ATTEMPTS" in helper
          and "no se interpreta como mesa vacía" in helper,
          "camera warm-up is bounded and never converts a failed sensor into an empty scene")
    demo_source = (ROOT / "11_real_sorting_with_monitor.py").read_text(encoding="utf-8")
    look = demo_source[demo_source.index("    def look("):
                       demo_source.index("    def _grasp_failure(")]
    check("telemetry.FAIL" in look and "self._camera_error(reason)" in look,
          "an exhausted camera retry appears as a camera failure in the dashboard")
    check("isaac_io.capture(self.env, self._active_sensor)" in look,
          "the configured camera is the primary capture; overhead is fallback only")
    benchmark_source = (ROOT / "tools" / "benchmark_demo11.py").read_text(encoding="utf-8")
    capture_wrapper = benchmark_source[benchmark_source.index("    def look(self, caption=", benchmark_source.index("def capture(arguments):")):
                                       benchmark_source.index("    demo.MonitoredRobot.look = look")]
    check("def look(self, caption=\"\", *look_args, **kwargs):" in capture_wrapper
          and "original(self, caption, *look_args, **kwargs)" in capture_wrapper,
          "the benchmark camera wrapper forwards keyword-only recovery controls")


def test_oblique_camera_falls_back_only_for_invalid_or_occluded_scene():
    demo = load_demo11()
    fallback = demo.should_use_backup_camera
    primary, backup = object(), object()
    check(demo.allow_occluded_tracking_for_camera("front", primary, primary)
          and not demo.allow_occluded_tracking_for_camera("front", backup, primary)
          and not demo.allow_occluded_tracking_for_camera("top", primary, primary),
          "front-view occlusion tolerance turns off after switching to the overhead camera")
    check(demo.experimental_approach_clearance_z(None) is None
          and demo.experimental_approach_clearance_z("0.25") == 0.25,
          "the optional safe-height approach accepts a valid world-frame clearance")
    try:
        demo.experimental_approach_clearance_z("0.05")
        check(False, "the optional safe-height approach rejects an unsafe clearance")
    except ValueError:
        check(True, "the optional safe-height approach rejects an unsafe clearance")
    check(fallback(has_backup=True, primary_capture_failed=True, objects_seen=2,
                   containers_expected=True, containers_seen=1, tasks_pending=False,
                   robot_holding=False),
          "an invalid primary frame switches to the backup camera")
    check(fallback(has_backup=True, primary_capture_failed=False, objects_seen=2,
                   containers_expected=True, containers_seen=0, tasks_pending=True,
                   robot_holding=False),
          "a missing destination triggers an overhead recheck")
    check(fallback(has_backup=True, primary_capture_failed=False, objects_seen=0,
                   containers_expected=True, containers_seen=1, tasks_pending=True,
                   robot_holding=False),
          "an empty oblique view during unfinished work triggers an overhead recheck")
    check(fallback(has_backup=True, primary_capture_failed=False, objects_seen=2,
                   containers_expected=True, containers_seen=1, tasks_pending=True,
                   robot_holding=False, grasp_failed=True),
          "a failed physical grasp forces overhead localization before the retry")
    check(not fallback(has_backup=True, primary_capture_failed=False, objects_seen=2,
                       containers_expected=True, containers_seen=1, tasks_pending=True,
                       robot_holding=True, grasp_failed=True),
          "an overhead recovery view is not requested while the robot is holding an object")
    check(not fallback(has_backup=True, primary_capture_failed=False, objects_seen=0,
                       containers_expected=True, containers_seen=1, tasks_pending=False,
                       robot_holding=False),
          "an empty scene after all tasks finish does not switch views")
    check(not fallback(has_backup=True, primary_capture_failed=False, objects_seen=0,
                       containers_expected=True, containers_seen=1, tasks_pending=True,
                       robot_holding=True),
          "an empty view while carrying is not misread as a missing part")
    check(not fallback(has_backup=False, primary_capture_failed=True, objects_seen=0,
                       containers_expected=True, containers_seen=0, tasks_pending=True,
                       robot_holding=False),
          "top-only configurations do not request a nonexistent backup")

    assist = demo.off_center_assist_candidate
    pregrasp = demo.should_pregrasp_assist
    check(pregrasp("after_physical_miss", assist_enabled=True,
                   kind="cube", size_mm=20.0)
          and pregrasp("after_physical_miss", assist_enabled=True,
                       kind="hexagonal_prism", size_mm=30.0)
          and pregrasp("after_physical_miss", assist_enabled=True,
                       kind="triangular_prism", size_mm=30.0)
          and pregrasp("after_physical_miss", assist_enabled=True,
                       kind="cylinder", size_mm=30.0)
          and not pregrasp("after_physical_miss", assist_enabled=True,
                           kind="cube", size_mm=24.0)
          and not pregrasp("after_physical_miss", assist_enabled=True,
                           kind="hexagonal_prism", size_mm=32.0)
          and not pregrasp("after_physical_miss", assist_enabled=True,
                           kind="triangular_prism", size_mm=32.0)
          and not pregrasp("after_physical_miss", assist_enabled=True,
                           kind="cylinder", size_mm=32.0)
          and not pregrasp("after_physical_miss", assist_enabled=True,
                           kind="ball", size_mm=20.0)
          and not pregrasp("after_physical_miss", assist_enabled=False,
                           kind="cube", size_mm=20.0),
          "only empirically fragile cube, prism and cylinder sizes use pre-contact assist under the physical-first policy")
    physical_miss_assist = demo.physical_miss_assist_candidate
    check(physical_miss_assist({"fits": True, "lateral_m": 0.004},
                               after_physical_miss=True, assist_enabled=True),
          "a centred lost lift can be recovered by explicitly labelled simulator assist")
    check(not physical_miss_assist({"fits": True, "lateral_m": 0.031},
                                   after_physical_miss=True, assist_enabled=True)
          and not physical_miss_assist({"fits": True, "lateral_m": 0.004},
                                       after_physical_miss=False, assist_enabled=True)
          and not physical_miss_assist({"fits": True, "lateral_m": 0.004},
                                       after_physical_miss=True, assist_enabled=False),
          "physical-miss assist rejects off-centre, preemptive and strict-physical cases")
    geometry = {"fits": True, "lateral_m": 0.023}
    check(assist(geometry, after_physical_miss=True, assist_enabled=True),
          "bounded off-centre follow is secured as simulator assist")
    check(not assist({"fits": True, "lateral_m": 0.031},
                     after_physical_miss=True, assist_enabled=True),
          "off-centre assist never attaches outside the validated reach bound")
    check(not assist(geometry, after_physical_miss=False, assist_enabled=True),
          "preemptive and strict policies do not relabel physical grasps")
    check(not assist(geometry, after_physical_miss=True, assist_enabled=False),
          "strict physical mode never attaches a simulator constraint")

    stabilize = demo.stabilize_fallback_position
    step_limit = demo.fallback_tracking_step_limit
    reliable = demo.tracking_measurement_is_reliable
    check(reliable(420, .8, None)
          and reliable(210, .8, 400)
          and not reliable(150, .8, 400)
          and not reliable(300, .19, 400)
          and not reliable(0, .9, None),
          "clipped or weak masks cannot replace the last trusted object pose")
    kept, rejected = stabilize((0.374, -0.477), (0.410, -0.458), 0.025)
    check(rejected and kept == (0.410, -0.458),
          "a backup-view localization jump preserves the latest scene-linked target")
    accepted, rejected = stabilize((0.412, -0.460), (0.410, -0.458), 0.025)
    check(not rejected and accepted == (0.412, -0.460),
          "small backup-view refinements remain available to the grasp servo")
    check(step_limit(after_grasp_failure=False) == 0.025
          and step_limit(after_grasp_failure=True) == 0.060
          and step_limit(after_grasp_failure=True,
                         ambiguous_same_color_peer=True) == 0.025
          and stabilize((0.428, -0.450), (0.400, -0.450),
                        step_limit(after_grasp_failure=True)) == ((0.428, -0.450), False)
          and stabilize((0.461, -0.450), (0.400, -0.450),
                        step_limit(after_grasp_failure=True)) == ((0.400, -0.450), True),
          "a failed grasp may reacquire 60 mm only when color uniquely identifies the target")
    pad_friction = demo.experimental_pad_friction
    check(pad_friction(None) == (0.90, 0.75)
          and pad_friction("1.20,1.00") == (1.20, 1.00),
          "an experimental pad-friction override is opt-in and leaves measured defaults intact")
    check(demo.experimental_carry_steps(None) == 120
          and demo.experimental_carry_steps("200") == 200,
          "slower carry timing is opt-in and preserves the measured default")
    check(gripper_contact_pads.experimental_pad_inset_m(None) == 0.0
          and gripper_contact_pads.experimental_pad_inset_m("9") == 0.009,
          "the experimental pad inset is opt-in and converted from millimetres")
    check(gripper_contact_pads.experimental_pad_radius_m(None) == 0.005
          and gripper_contact_pads.experimental_pad_radius_m("8.5") == 0.0085,
          "spherical pad radius overrides are opt-in and converted from millimetres")
    check(gripper_contact_pads.experimental_pad_span_m(None) == 0.018
          and gripper_contact_pads.experimental_pad_span_m("24") == 0.024,
          "experimental V-pad face spans are opt-in and converted from millimetres")
    check(gripper_contact_pads.experimental_vee_angle_deg(None) == 35.0
          and gripper_contact_pads.experimental_vee_angle_deg("40") == 40.0,
          "the V-pad angle override applies to explicitly forced experimental profiles")
    check(not gripper_contact_pads.force_unvalidated_experimental_pads(None)
          and not gripper_contact_pads.force_unvalidated_experimental_pads("0")
          and gripper_contact_pads.force_unvalidated_experimental_pads("1"),
          "unvalidated V-pad geometry can only be force-enabled by an explicit experiment flag")
    for invalid in ("1.0", "-0.1,0.2", "0.5,0.8", "nan,0.2", "2.1,1.0"):
        try:
            pad_friction(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid experimental pad friction was accepted: {invalid}")
    try:
        gripper_contact_pads.experimental_pad_inset_m("12.1")
    except ValueError:
        pass
    else:
        raise AssertionError("an experimental pad inset above the validated range was accepted")
    try:
        demo.experimental_carry_steps("401")
    except ValueError:
        pass
    else:
        raise AssertionError("an experimental carry duration above the safe range was accepted")
    try:
        gripper_contact_pads.experimental_pad_radius_m("10.1")
    except ValueError:
        pass
    else:
        raise AssertionError("a spherical pad radius above the supported range was accepted")
    try:
        gripper_contact_pads.experimental_pad_span_m("40.1")
    except ValueError:
        pass
    else:
        raise AssertionError("a V-pad face span above the supported range was accepted")
    tolerance = demo.fallback_link_tolerance
    check(abs(tolerance(0.035, backup_view=True, is_container=False,
                        max_tracking_shift_m=0.025) - 0.060) < 1e-9,
          "backup objects can associate within the coordinate stabilization radius")
    check(tolerance(0.035, backup_view=True, is_container=True,
                    max_tracking_shift_m=0.025) == 0.035,
          "container association keeps its stricter, fixed-pose threshold")
    check(tolerance(0.035, backup_view=False, is_container=False,
                    max_tracking_shift_m=0.025) == 0.035,
          "the selected camera retains the original association threshold")


def test_camera_readiness_rejects_uninitialized_buffers_and_accepts_a_rendered_scene():
    """Black/flat-gray startup buffers must be retried; a real tabletop frame must pass."""
    import numpy as np
    import isaac_io

    black = np.zeros((24, 32, 3), dtype=np.uint8)
    gray = np.full((24, 32, 3), 220, dtype=np.uint8)
    softly_shaded = np.full((24, 32, 3), 218, dtype=np.uint8)
    softly_shaded[:, :8] = 210  # RTX's near-blank startup buffer has only a 8-level range.
    textured = np.full((24, 32, 3), 180, dtype=np.uint8)
    textured[::2, ::2] = (110, 70, 35)
    check(not isaac_io._camera_frame_is_ready(black),
          "an uninitialized black render target is not an empty table")
    check(not isaac_io._camera_frame_is_ready(gray),
          "an uninitialized flat-gray render target is retried")
    check(not isaac_io._camera_frame_is_ready(softly_shaded),
          "a softly shaded RTX startup buffer cannot masquerade as a rendered scene")
    check(isaac_io._camera_frame_is_ready(textured),
          "a rendered tabletop with texture is accepted for perception")
    check(not isaac_io._camera_frame_is_ready(np.full((24, 32), 128, dtype=np.uint8)),
          "malformed camera output cannot pass readiness checks")

    class Buffer:
        def __init__(self, value): self.value = value
        def __getitem__(self, index): return Buffer(self.value[index])
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self.value)

    class FakeEnv:
        physics_dt = 1 / 60
        class Sim:
            def __init__(self): self.renders = 0; self.steps = 0
            def render(self): self.renders += 1
            def step(self, render=True): self.steps += 1
        def __init__(self): self.sim = self.Sim()

    class FakeSensor:
        def __init__(self, frames):
            self.frames = frames
            self.index = 0
            self.resets = 0
            self.data = type("Data", (), {"output": {"rgb": Buffer(frames[0][None])}})()
        def reset(self): self.resets += 1
        def update(self, _dt, force_recompute=False):
            check(force_recompute, "every warm-up render forces the camera buffer to refresh")
            frame = self.frames[min(self.index, len(self.frames) - 1)]
            self.index += 1
            self.data.output["rgb"] = Buffer(frame[None])

    env = FakeEnv()
    sensor = FakeSensor([black, gray, black, gray, textured])
    frame, attempts = isaac_io._render_camera_rgb(env, sensor)
    check(attempts == 5 and env.sim.renders == 5
          and isaac_io._camera_frame_is_ready(frame),
          "camera warm-up retries empty buffers and returns the first valid rendered frame")

    env = FakeEnv()
    sensor = FakeSensor([black] * isaac_io.CAMERA_CAPTURE_MAX_ATTEMPTS + [textured])
    frame, attempts = isaac_io._render_camera_rgb(env, sensor,
                                                  advance_physics_on_recovery=True)
    check(attempts == isaac_io.CAMERA_CAPTURE_MAX_ATTEMPTS + 1
          and env.sim.steps == 1 and isaac_io._camera_frame_is_ready(frame),
          "camera recovery advances a physics frame when RTX stays blank after scene reset")

    env = FakeEnv()
    sensor = FakeSensor([black])
    try:
        isaac_io._render_camera_rgb(env, sensor)
        failed_explicitly = False
    except RuntimeError as exc:
        failed_explicitly = "no se interpreta como mesa vacía" in str(exc)
    check(failed_explicitly
          and env.sim.renders == (isaac_io.CAMERA_CAPTURE_MAX_ATTEMPTS
                                 + isaac_io.CAMERA_CAPTURE_RECOVERY_STEPS
                                 + isaac_io.CAMERA_CAPTURE_REINITIALIZE_STEPS)
          and env.sim.steps == 0 and sensor.resets == 1,
          "a persistently empty sensor fails explicitly at the configured retry limit")


def test_each_fact_is_painted_once_the_checklist_in_the_objective_and_the_matrix_in_the_plan():
    """Each fact has one place on the screen. The objective owns the checklist, the count of finished
    tasks and the last result; the plan card owns the movements, one row per move and one column per
    task. The checklist and its count used to be painted in both cards, so the reader had to compare
    two lists to find what was new.

    The poses in millimetres and the blob code are gone on purpose: nobody moves an arm with those
    numbers. The detail stays in the session log.
    """
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    plan_card = page[page.index('class="card plan-card"'):page.index("</main>")]
    check('id="exec-steps"' in plan_card, "the plan card holds the movement matrix")
    check("hero-tasks-list" not in plan_card and "hero-tally" not in plan_card,
          "the plan card does not repeat the checklist or its count")
    check(page.count('id="hero-tasks-list"') == 1, "the checklist is declared once, in the objective")
    check('id="plan-tasks"' not in page and 'id="tally"' not in page,
          "the old list and its count are gone from the page")
    check(page.count("completadas") == 1, "the count of finished tasks is written in one place only")
    check(page.count('id="result-session"') == 1 and 'id="result-session"' in plan_card,
          "the session line sits at the bottom of the plan card, once")
    check('id="route-now"' in page, "what is in progress is said in one line, next to the camera")
    check("paintSteps(plan)" in page and 'el("exec-steps")' in page,
          "the matrix is painted from the plan")
    steps = page[page.index("function paintSteps("):]
    steps = steps[:steps.index("\n}\n")]
    code = "\n".join(line for line in steps.splitlines() if not line.strip().startswith("//"))
    check("pose" not in code, "no millimetres in the steps table: their place is the session log")
    check("step.pose" not in steps, "the pose is not even read into the panel")


def test_the_page_names_every_status_command_state_and_phase_the_robot_can_publish():
    """The page's Spanish maps must cover what the robot really publishes, or a state shows raw.

    The phases are read from the demo's own ``phase=`` calls, and the statuses from the ledger,
    so a new state added on the Python side fails here until the page can say it in Spanish.
    """
    import re

    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")

    def keys_of(name):
        block = re.search(rf"const {name} = \{{(.*?)\}};", page, re.S)
        return set(re.findall(r"(\w+):", block.group(1))) if block else set()

    source = (ROOT / "11_real_sorting_with_monitor.py").read_text(encoding="utf-8")
    phases = set(re.findall(r'phase="(\w+)"', source))
    check(phases and phases <= keys_of("PHASE_ES"),
          f"the page has no Spanish for these phases: {sorted(phases - keys_of('PHASE_ES'))}")
    statuses = {module_status for module_status in
                ("SUCCESS", "PARTIAL_SUCCESS", "FAILED", "ABORTED", "NO_TASKS")}
    check(statuses <= keys_of("STATUS_ES"),
          f"missing final statuses: {sorted(statuses - keys_of('STATUS_ES'))}")
    states = {"RECEIVED", "QUEUED", "RUNNING", "SUCCESS", "FAILED", "CANCELLED"}
    check(states <= keys_of("COMMAND_ES"),
          f"missing command states: {sorted(states - keys_of('COMMAND_ES'))}")
    check("EN CURSO" in page and "COMPLETADA" in page and "FALLIDA" in page and "PENDIENTE" in page,
          "the checklist speaks the four task states in Spanish")



def test_start_is_running_until_the_environment_is_ready():
    from types import SimpleNamespace
    module = load_demo11()
    bus = telemetry.Bus()
    cid = bus.send_control("start", camera="top")
    args = SimpleNamespace(variant="contained", camera="top", rounds=1, seed=0)
    module.apply_start_params(args, bus)
    check(bus.command_state(cid)["state"] == telemetry.RUNNING,
          "start is not successful before Isaac is ready")


def test_pending_tasks_cannot_produce_success():
    ledger = load_demo11().TaskLedger()
    ledger.expect("command-one:ball:red-box", "bola roja")
    check(ledger.status() == "FAILED", "an expected but unverified task prevents success")
    ledger.record("command-one:ball:red-box", True)
    ledger.expect("command-two:ball:blue-box")
    check(ledger.counts() == (1, 1), "a second order on the same ball is a separate task")


def test_controls_do_not_replace_the_running_text_identity():
    bus = telemetry.Bus()
    cid = bus.submit("coge la bola roja")
    bus.take_command()
    bus.send_control("relook")
    bus.take_control()
    check(bus.active_command_id() == cid, "controls cannot steal the active text id")
    bus.set_command_state(cid, telemetry.NEEDS_CLARIFICATION, "¿Qué caja?")
    check(bus.command_state(cid)["state"] == telemetry.NEEDS_CLARIFICATION,
          "a clarification is an explicit terminal outcome")


def test_supervisor_distinguishes_python_setup_errors_from_isaac_crashes():
    from supervisor import worker_error_before_ready

    error = "No se pudo cargar el modelo: No module named 'torch'"
    check(worker_error_before_ready({"phase": "failed", "error": error}) == error,
          "a reported Python dependency error is surfaced directly without Isaac retry wording")
    check(worker_error_before_ready({"phase": "starting", "error": ""}) == "",
          "a worker that has not reported a Python error remains eligible for native startup retries")



def test_supervisor_can_stop_keep_the_report_and_start_a_new_process():
    from types import SimpleNamespace
    from supervisor import Supervisor
    from fake_session import run
    with tempfile.TemporaryDirectory() as directory:
        args = SimpleNamespace(camera="top", variant="contained", rounds=1, seed=0,
                               startup_timeout=5, stop_timeout=1, session_dir=directory)
        supervisor = Supervisor(args, target=run)
        bus = supervisor.bus
        def until(predicate):
            deadline = time.monotonic() + 5
            while not predicate() and time.monotonic() < deadline:
                supervisor.poll()
                time.sleep(.02)
            check(predicate(), "session process reached the expected state")
        try:
            for _ in range(2):
                cid = bus.send_control("start")
                supervisor.start(bus.take_control())
                until(lambda: supervisor.ready)
                check(bus.command_state(cid)["state"] == telemetry.SUCCESS, "start follows camera readiness")
                command = bus.submit("coge la bola roja")
                bus.take_command()
                supervisor.forward(command)
                until(lambda: bus.command_state(command)["state"] == telemetry.SUCCESS)
                bus.request_stop()
                until(lambda: supervisor.child is None)
                check(bus.snapshot()["state"]["report"]["state"]["status"] == "ABORTED",
                      "the report survives process shutdown")
        finally:
            supervisor.terminate()


# --- arranque: reintentos automáticos cuando Isaac muere antes de estar listo ----------
#
# Isaac Sim faults natively during start-up on this machine - a thread with no Python frame
# dies with STATUS_ENTRYPOINT_NOT_FOUND inside Kit's extension loader - and the supervisor
# used to answer that by terminating the session and leaving the dashboard saying "no ha
# arrancado". These tests exercise the retry with a child that dies on demand instead of a
# real Isaac, so they run in seconds and can be run by anyone.


def _fake_supervisor(directory, fail_times=1, start_attempts=3, signature="",
                     startup_timeout=30, hang=False):
    """A supervisor whose child fails ``fail_times`` launches, then behaves.

    ``fail_times`` and the fault signature travel with the session arguments, because the
    child is re-spawned per attempt and gets a fresh copy of its own module each time -
    only the arguments survive. What the test asserts on is what the **supervisor**
    publishes, which is the right thing anyway: ``start_attempt`` is the field the
    dashboard renders, so asserting on it tests what the person actually sees.
    """
    from types import SimpleNamespace
    from supervisor import Supervisor
    import fake_session
    args = SimpleNamespace(camera="top", variant="contained", rounds=1, seed=0,
                           startup_timeout=startup_timeout, stop_timeout=1,
                           session_dir=directory, start_attempts=start_attempts,
                           fail_times=fail_times, signature=signature,
                           hang_until=time.monotonic() + 120 if hang else 0.0)
    return Supervisor(args, target=fake_session.run)


def _start_and_wait(supervisor, timeout=60):
    """Press start and poll until the session is ready or the child gives up."""
    bus = supervisor.bus
    cid = bus.send_control("start")
    supervisor.start(bus.take_control())
    deadline = time.monotonic() + timeout
    while supervisor.child is not None and not supervisor.ready and time.monotonic() < deadline:
        supervisor.poll()
        time.sleep(.02)
    return cid


def test_a_failed_launch_is_retried_and_the_second_attempt_succeeds():
    """One dead worker, then a good one: the session has to reach READY by itself.

    This is the whole point of the retry. The properties that matter are that the attempt
    number is published (so the dashboard can say «intento 2/3»), that the scene survives
    the relaunch, and that nobody had to press start twice.
    """
    with tempfile.TemporaryDirectory() as directory:
        supervisor = _fake_supervisor(directory, fail_times=1)
        bus = supervisor.bus
        cid = _start_and_wait(supervisor, timeout=40)
        try:
            check(supervisor.ready, "the session recovered on its own")
            check(bus.snapshot()["state"].get("start_attempt") == "",
                  "the attempt counter is cleared once ready")
            check(bus.command_state(cid)["state"] == telemetry.SUCCESS,
                  "start ends SUCCESS once the camera is up, even after a retry")
            snapshot_state = bus.snapshot()["state"]
            check(len(snapshot_state.get("catalog") or []) > 0,
                  "the scene the person chose survives a relaunch")
            check(any("intento 2 de 3" in event["message"]
                      for event in bus.snapshot()["events"]),
                  "the retry is visible in the log, not silent")
            check(any("Reintentando" in event["message"] or "Error de arranque" in event["message"]
                      for event in bus.snapshot()["events"]),
                  "the reason for the retry is in the log")
        finally:
            bus.request_stop()
            supervisor.terminate()


def test_exhausted_attempts_report_a_reason_and_stop():
    """Three dead workers: the dashboard must say why, and must not loop for ever."""
    with tempfile.TemporaryDirectory() as directory:
        supervisor = _fake_supervisor(directory, fail_times=9, start_attempts=3,
                                      signature="entry point")
        bus = supervisor.bus
        cid = _start_and_wait(supervisor, timeout=60)
        try:
            check(bus.snapshot()["state"].get("phase") == "failed",
                  "the session gives up with a failed phase")
            check(bus.command_state(cid)["state"] == telemetry.FAILED,
                  "the start command ends FAILED with a reason, not RUNNING")
            reason = bus.snapshot()["state"].get("error", "")
            check("3 intentos" in reason and "ENTRYPOINT" in reason.upper(),
                  f"the reason names both the count and the fault, got {reason!r}")
        finally:
            supervisor.terminate()


def test_an_intentional_stop_is_never_retried():
    """A stop pressed by a person is a decision, not a failure: it must not relaunch."""
    with tempfile.TemporaryDirectory() as directory:
        supervisor = _fake_supervisor(directory, fail_times=0)
        bus = supervisor.bus
        _start_and_wait(supervisor, timeout=20)
        bus.request_stop()
        deadline = time.monotonic() + 20
        while supervisor.child is not None and time.monotonic() < deadline:
            supervisor.poll()
            time.sleep(.02)
        time.sleep(.4)
        supervisor.poll()
        try:
            check(bus.snapshot()["state"].get("phase") in (None, "stopped", "failed"),
                  f"the phase after a stop is terminal, got {bus.snapshot()['state'].get('phase')!r}")
            check(not any("Reintentando" in event["message"]
                          for event in bus.snapshot()["events"]),
                  "an intentional stop relaunches nothing")
        finally:
            supervisor.terminate()


def test_a_hung_startup_is_retried_too():
    """A launch that says nothing is the other half of the native fault.

    A faulted Kit kernel can hold the GPU, and the next launch then stalls behind it with no
    output - so the timeout path has to retry just like the crash path does. A child that
    produces nothing until the timeout stands in for it.
    """
    with tempfile.TemporaryDirectory() as directory:
        supervisor = _fake_supervisor(directory, fail_times=0, start_attempts=2,
                                      startup_timeout=2, hang=True)
        bus = supervisor.bus
        _start_and_wait(supervisor, timeout=60)
        try:
            check(any("intento 2 de 2" in event["message"]
                      for event in bus.snapshot()["events"]),
                  "a hung launch was relaunched")
            check(bus.snapshot()["state"].get("phase") == "failed",
                  "and then gave up with a failed phase")
            reason = bus.snapshot()["state"].get("error", "")
            check("2 intentos" in reason, f"the give-up reason names the count, got {reason!r}")
        finally:
            supervisor.terminate()


def test_a_hung_startup_is_retried_too():
    """A launch that says nothing is the other half of the native fault.

    A faulted Kit kernel can hold the GPU, and the next launch then stalls behind it with no
    output - so the timeout path has to retry just like the crash path does. A child that
    produces nothing until the timeout stands in for it.
    """
    with tempfile.TemporaryDirectory() as directory:
        supervisor = _fake_supervisor(directory, fail_times=0, start_attempts=2,
                                      startup_timeout=2, hang=True)
        bus = supervisor.bus
        _start_and_wait(supervisor, timeout=90)
        try:
            check(any("intento 2 de 2" in event["message"]
                      for event in bus.snapshot()["events"]),
                  "a hung launch was relaunched")
            check(bus.snapshot()["state"].get("phase") == "failed",
                  "and then gave up with a failed phase")
            reason = bus.snapshot()["state"].get("error", "")
            check("2 intentos" in reason, f"the give-up reason names the count, got {reason!r}")
        finally:
            supervisor.terminate()



def test_session_log_survives_kit_taking_the_standard_streams_over():
    """The 19:26 session died on a print, not on physics: see ``supervisor.SessionLog``.

    Isaac Lab's ``AppLauncher._create_app`` replaces ``sys.stdout`` with devnull while Kit boots
    and then restores ``sys.__stdout__``; Kit's ``omni.kit.app`` installs a ``StreamInterceptor``
    that forwards to the ``sys.__std*__`` pair it captured. The last thing in that chain is fd 1,
    a descriptor the worker dup2'd onto the log and has no other claim on, and a write into it
    raised ``OSError: [WinError 6] Controlador no válido`` at the first print of
    ``isaaclab_tasks.utils.parse_cfg``. This replays that sequence in a child process - the same
    arrangement the worker runs in - and asks whether the session log still receives it, which is
    the property that keeps a logging failure from being reported as "el entorno no ha arrancado".

    The child says it finished by writing a sentinel, not by exiting 0: a print that raises takes
    the session down, and it is that survival this test is about.
    """
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "runtime.log"
        done_path = Path(directory) / "done"
        child = (
            "import os, sys\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from supervisor import SessionLog\n"
            "log = SessionLog(Path(%r))\n"
            "captured_out, captured_err = sys.__stdout__, sys.__stderr__\n"
            "sys.stdout = open(os.devnull, 'w')  # noqa: SIM115 - AppLauncher's devnull\n"
            "os.close(1)  # the handle the restore dance lands on, gone: WinError 6 from inside\n"
            "sys.stdout = sys.__stdout__  # AppLauncher's restore after Kit booted\n"
            "sys.stderr = captured_err  # what Kit's StreamInterceptor leaves behind\n"
            "print('[INFO]: Parsing configuration from: leisaac.tasks.lift_cube', flush=True)\n"
            "captured_out.write('[py stdout]: forwarded\\n')\n"
            "captured_err.write('[py stderr]: forwarded\\n')\n"
            "log.flush()\n"
            "Path(%r).write_text('done', encoding='utf-8')\n"
        ) % (str(ROOT), str(log_path), str(done_path))
        finished = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                                  encoding="utf-8", timeout=60)
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        check(finished.returncode == 0 and done_path.exists(),
              f"the worker survives AppLauncher and Kit swapping the streams over "
              f"(rc={finished.returncode}, stderr={finished.stderr[-200:]!r})")
        check("Parsing configuration from: leisaac" in text,
              "a print after the restore still reaches the session log")
        check("[py stdout]: forwarded" in text and "[py stderr]: forwarded" in text,
              "a write on the streams Kit captured still reaches the session log")


def test_session_log_retries_a_write_whose_handle_died():
    """A log that cannot be written must not be able to take the run down with it."""
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "runtime.log"
        done_path = Path(directory) / "done"
        child = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from supervisor import SessionLog\n"
            "log = SessionLog(Path(%r))\n"
            "log.write('before the handle died\\n')\n"
            "log._stream.close()  # what OSError WinError 6 feels like from the inside\n"
            "log.write('after the handle died\\n')\n"
            "log.flush()\n"
            "Path(%r).write_text('done', encoding='utf-8')\n"
        ) % (str(ROOT), str(log_path), str(done_path))
        finished = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                                  encoding="utf-8", timeout=60)
        check(finished.returncode == 0 and done_path.exists(),
              f"a write through a dead handle does not raise (rc={finished.returncode}, "
              f"stderr={finished.stderr[-200:]!r})")
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        check("before the handle died" in text, "what was written before the handle died is kept")
        check("after the handle died" in text, "a retried write lands in the reopened log")


def test_scene_configuration_is_validated_and_reproducible():
    import copy
    import scene_config
    from unittest import mock
    # Isolate from the saved scene: this test is about the DEFAULT catalog, and a file
    # left on disk by a dashboard session must not decide what "default" means here.
    with mock.patch.object(scene_config, "SAVED_SCENE",
                           Path("no-existe-scene-config.json")):
        config, items = scene_config.load()
        check(len(items) == len(scene_spec.DEFAULT_SCENE), "default catalog covers the scene")
        check(scene_config.build(config)[1] == items, "scene generation is deterministic")
        original_work = (scene_spec.WORK_X, scene_spec.WORK_Y)
        moved = copy.deepcopy(config)
        moved["working_area"] = {"x_m": .405, "y_m": -.46, "width_mm": 270, "depth_mm": 200}
        canonical_moved, moved_items = scene_config.build(moved)
        moved_layout = scene_config.layout(canonical_moved)
        check(all(abs(actual - expected) < 1e-9
                  for actual, expected in zip(moved_layout["work_x"] + moved_layout["work_y"],
                                             [.27, .54, -.56, -.36])),
              f"a moved and resized work area reaches the map, got {moved_layout}")
        check(all(moved_layout["work_x"][0] <= item.x <= moved_layout["work_x"][1]
                  and moved_layout["work_y"][0] <= item.y <= moved_layout["work_y"][1]
                  for item in moved_items), "generated objects follow the configured work area")
        check((scene_spec.WORK_X, scene_spec.WORK_Y) == original_work,
              "validating a custom area does not mutate process-wide runtime bounds")
        # Configurations written before working_area existed remain loadable and are upgraded.
        legacy = copy.deepcopy(config); legacy.pop("working_area", None)
        legacy_canonical, _ = scene_config.build(legacy)
        check(legacy_canonical["working_area"] == scene_config.DEFAULT_WORKING_AREA,
              "legacy scenes receive the default work area")
        for invalid_area in (
            {"x_m": .5, "y_m": -.48, "width_mm": 500, "depth_mm": 120},
            {"x_m": .5, "y_m": -.3, "width_mm": 300, "depth_mm": 200},
        ):
            bad_area = copy.deepcopy(config); bad_area["working_area"] = invalid_area
            try: scene_config.build(bad_area)
            except ValueError: check(True, "working areas outside the tabletop or safe reach are refused")
            else: check(False, "invalid working area was accepted")
        custom = {"version":1, "objects":[{"id":"balls","kind":"ball","color":"red","count":2}],
                  "containers":[{"id":"left_box","name":"Izquierda","color":"red"},
                                {"id":"right_box","name":"Derecha","color":"blue"}]}
        _, custom_items = scene_config.build(custom)
        check(len(custom_items) == 4, "counts expand into stable physical identities")
        for bad in [dict(custom, version=2), dict(custom, surprise=True)]:
            try: scene_config.build(bad)
            except ValueError: check(True, "invalid scene is refused")
            else: check(False, "invalid scene was accepted")
        for field, value in [("color","pink"),("count",0),("count",65),("size_mm",40),("x",float("nan"))]:
            bad=copy.deepcopy(custom); bad["objects"][0][field]=value
            try: scene_config.build(bad)
            except ValueError: check(True, "invalid object is refused")
            else: check(False, "invalid object was accepted")


def test_catalog_linking_suppresses_nearby_same_entity_fragments_only():
    from types import SimpleNamespace
    import scene_config

    item = SimpleNamespace(prim="red_cube", color="red", container=False,
                           size=(.03, .03, .03))
    linked = SimpleNamespace(id=1, color="red", aim_xy=[.356, -.442], pos_m=None)
    duplicate = SimpleNamespace(id=2, color="red", aim_xy=[.379, -.453], pos_m=None)
    nearby_real = SimpleNamespace(id=3, color="red", aim_xy=[.401, -.442], pos_m=None)
    other_color = SimpleNamespace(id=4, color="green", aim_xy=[.357, -.443], pos_m=None)
    kept, suppressed = scene_config.suppress_unlinked_fragments(
        [linked, duplicate, nearby_real, other_color], set(), {1: "red_cube"}, [item])
    check(suppressed == [2] and kept == [linked, nearby_real, other_color],
          "catalog linking removes a close duplicate blob but preserves separated objects and other colours")

    tray = SimpleNamespace(prim="red_tray", color="red", container=True,
                           size=(.07, .07, .05))
    red_container = SimpleNamespace(id=5, color="red", aim_xy=[.45, -.45], pos_m=None)
    red_object = SimpleNamespace(id=6, color="red", aim_xy=[.45, -.45], pos_m=None)
    kept, suppressed = scene_config.suppress_unlinked_fragments(
        [red_container, red_object], {5}, {5: "red_tray"}, [tray])
    check(not suppressed and kept == [red_container, red_object],
          "a nearby same-colour object is not discarded as a container fragment")


def test_out_of_reach_variant_handles_custom_geometries_and_keeps_clearance():
    import scene_config

    config = scene_config.default_config()
    config["objects"] = [
            {"id": f"custom_{kind}", "kind": kind, "color": color, "count": 1,
             "size_mm": 24, "height_mm": 24, "x": x,
             "y": -.380,
         "yaw_deg": 20 if kind.endswith("prism") else 0}
        for kind, color, x in (
                ("cube", "red", .29), ("ball", "blue", .355),
            ("cylinder", "green", .39), ("triangular_prism", "yellow", .44),
            ("hexagonal_prism", "purple", .49),
        )
        ]
    _canonical, configured = scene_config.build(config)
    moved = scene_spec.out_of_reach_layout(configured)
    loose = [item for item in moved if not item.container]
    check({item.kind for item in loose} == {
        "cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"},
        "the out-of-reach layout retains every custom geometry")
    check(all(scene_spec.reach_facts(item.x, item.y, scene_spec.TABLE_X,
                                     scene_spec.TABLE_Y)["beyond_arm"] for item in loose),
          "every custom object is physically outside the arm reach, independent of its ID")
    original_containers = {item.prim: (item.x, item.y) for item in configured if item.container}
    moved_containers = {item.prim: (item.x, item.y) for item in moved if item.container}
    check(moved_containers == original_containers,
          "custom out-of-reach scenes keep configured receptacles in place")

    for index, item in enumerate(moved):
        hx, hy = scene_config._aabb_half_extents(item)
        for other in moved[index + 1:]:
            ox, oy = scene_config._aabb_half_extents(other)
            overlaps = (abs(item.x - other.x) < hx + ox + .003
                        and abs(item.y - other.y) < hy + oy + .003)
            check(not overlaps,
                  f"custom out-of-reach placement keeps {item.prim} clear of {other.prim}")

    defaults = scene_spec.out_of_reach_scene()
    by_prim = {item.prim: item for item in defaults}
    check((by_prim["cube_blue"].x, by_prim["cube_blue"].y)
          == scene_spec.OUT_OF_REACH_ITEMS["cube_blue"],
          "the existing blue-cube out-of-reach demo pose remains stable")
    check((by_prim["ball_green"].x, by_prim["ball_green"].y)
          == scene_spec.OUT_OF_REACH_ITEMS["ball_green"],
          "the existing green-ball out-of-reach demo pose remains stable")
    single_default_part = [item for item in scene_spec.DEFAULT_SCENE
                           if item.prim in {"cube", "box_red"}]
    partial = {item.prim: item for item in
               scene_spec.out_of_reach_layout(single_default_part)}
    check(scene_spec.reach_facts(partial["cube"].x, partial["cube"].y,
                                 scene_spec.TABLE_X, scene_spec.TABLE_Y)["beyond_arm"],
          "a partial scene using a legacy ID still applies the out-of-reach variant")
    check((partial["box_red"].x, partial["box_red"].y)
          == next((item.x, item.y) for item in single_default_part if item.prim == "box_red"),
          "a partial out-of-reach scene leaves its destination where configured")


def test_prisms_are_spawnable_graspable_and_recognized_by_top_contour():
    import cv2
    import numpy as np
    import polygon_prism
    import perceive
    import scene_config

    cfg = scene_config.default_config()
    cfg["objects"] = [
        {"id": "tri", "kind": "triangular_prism", "color": "red", "size_mm": 30,
         "height_mm": 24, "x": .36, "y": -.400},
        {"id": "hex", "kind": "hexagonal_prism", "color": "blue", "size_mm": 30,
         "height_mm": 24, "x": .43, "y": -.400},
    ]
    _, items = scene_config.build(cfg)
    prisms = [item for item in items if item.kind.endswith("prism")]
    check([item.kind for item in prisms] == ["triangular_prism", "hexagonal_prism"],
          "both prism geometries survive scene validation")
    check(all(real_world.grasp(item).feasible for item in prisms),
          "30 mm prisms fit the gripper's measured width envelope")
    tri_volume = real_world.volume_m3(prisms[0])
    hex_volume = real_world.volume_m3(prisms[1])
    check(0 < tri_volume < hex_volume, "polygon side count contributes to physical mass")
    for sides, kind in ((3, "triangular_prism"), (6, "hexagonal_prism")):
        vertices, faces = polygon_prism.mesh_arrays(.03, .024, sides)
        check(vertices.shape == (2 * sides, 3) and faces.shape == (4 * sides - 4, 3),
              f"{kind} mesh has capped, triangulated sides")
        image = np.zeros((100, 100), dtype=np.uint8)
        points = np.asarray([[[int(50 + 30 * np.cos(2 * np.pi * i / sides)),
                               int(50 + 30 * np.sin(2 * np.pi * i / sides))]]
                             for i in range(sides)], dtype=np.int32)
        cv2.fillPoly(image, [points], 1)
        measured = perceive.polygon_sides(image)
        label = perceive.kind_of(.8, .96, measured)
        check(measured == sides and label == kind,
              f"top-view contour distinguishes the {kind}, got {measured}/{label}")

def test_the_saved_scene_is_what_a_plain_run_loads():
    """A scene saved in the dashboard is the scene a bare ``python 11_…`` runs.

    ``scene_config.load(None)`` used to mean ``default_config()``, so a run with no
    ``--scene-config`` silently ignored the file the panel had written and put four
    parts - two of them balls - on the table. For a live demo that is the difference
    between the scene that was measured and one that was not, and the fix is that
    ``load()`` falls back to the saved file before it falls back to the default.
    """
    import scene_config
    from unittest import mock
    with tempfile.TemporaryDirectory() as directory:
        saved = Path(directory) / "scene_config.json"
        saved.write_text(json.dumps(scene_config.demo_config()), encoding="utf-8")
        with mock.patch.object(scene_config, "SAVED_SCENE", saved):
            _, items = scene_config.load(None)
            check([i.prim for i in items if not i.container] == ["cube_red", "cube_blue"],
                  f"the saved scene is loaded, got {[i.prim for i in items]}")
            check(not any(i.kind == "ball" for i in items),
                  "the saved two-cube preset has no balls")
            # And with no file there, the default scene is still the default.
            saved.unlink()
            _, items = scene_config.load(None)
            check(len(items) == len(scene_spec.DEFAULT_SCENE),
                  "with no saved file the default scene is used")


def test_the_demo_preset_is_the_measured_two_cube_scene():
    """The preset for a live demo: two cubes, two trays, positions that were run.

    Balls are the failure mode this avoids. Measured over the recorded sessions, a ball
    command ended FAILED with 0/1 verified and the seeds that finished needed refused
    grasps; the seed that finished 2/2 with zero refusals was the two-cube one. The
    preset is that scene written down, so a preset and a measurement cannot drift.
    """
    import scene_config
    config, items = scene_config.build(scene_config.demo_config())
    loose = [i for i in items if not i.container]
    check(len(loose) == 2 and all(i.kind == "cube" for i in loose),
          "the demo preset is two cubes, not balls")
    check(len([i for i in items if i.container]) == 2,
          "and two trays, one of each colour")
    colours = {i.color for i in loose}
    check(colours == {t.color for t in items if t.container},
          f"each part has a tray of its own colour, got {sorted(colours)}")
    for item in items:
        check(scene_spec.WORK_X[0] <= item.x <= scene_spec.WORK_X[1]
              and scene_spec.WORK_Y[0] <= item.y <= scene_spec.WORK_Y[1],
              f"{item.prim} is inside the working area")
    check(abs(loose[0].x - loose[1].x) >= 0.045,
          "the two cubes are far enough apart for the segmentation to separate them")



def test_hybrid_parser_keeps_counts_destinations_and_unknown_references():
    from intent_parser import HybridParser
    import scene_config
    cfg,items=scene_config.load(); parser=HybridParser(catalog=scene_config.catalog(cfg,items))
    for text in ["pon dos bolas rojas en la caja azul", "put two red balls in the blue box", "hartu bi bola gorri eta sartu kutxa urdinean"]:
        result=parser.parse(text)
        check(result["intent"]=="pick_and_place" and result["items"][0]["quantity"]==2 and result["place"]["color"]=="blue", "three languages preserve count and destination")
        check(not result.get("errors"), "complete supported sentence is accepted")
    colour_roles = parser.parse("pon el cilindro verde en la caja roja")
    check(colour_roles["items"][0]["kind"] == "cylinder"
          and colour_roles["items"][0]["color"] == "green"
          and colour_roles["place"] == {"kind": "box", "color": "red"},
          "the spoken green object and red destination keep separate colour roles")
    name=cfg["containers"][0]["name"]
    check(parser.parse("pon la bola roja en "+name)["place"].get("entity_id")==cfg["containers"][0]["id"], "catalog names resolve to stable identities")
    for text in ["pon la bola magenta en la caja roja", "pon la bola roja en la caja inventada", "no cojas la bola roja", "coge la bola redwood"]:
        check(bool(parser.parse(text).get("errors")), "unknown references or negation require clarification")
    check(parser.parse("para")["intent"]=="stop", "stop uses the immediate rule path")


def test_move_verbs_and_prism_nouns_parse_for_supported_shapes():
    from intent_parser import HybridParser

    parser = HybridParser()
    commands = {
        "mueve la bola roja a la caja roja": "ball",
        "mueve el cilindro rojo a la caja roja": "cylinder",
        "mueve el prisma triangular rojo a la caja roja": "triangular_prism",
        "mueve el prisma hexagonal rojo a la caja roja": "hexagonal_prism",
    }
    for sentence, kind in commands.items():
        parsed = parser.parse(sentence)
        check(parsed["intent"] == "pick_and_place" and parsed["items"][0]["kind"] == kind
              and parsed["place"]["kind"] == "box",
              f"'{sentence}' names the supported {kind} and its destination, got {parsed}")
        check(not parsed.get("errors"), f"'{sentence}' does not fall back to an unknown object")


def test_explicit_geometry_and_destination_words_override_language_model_classifications():
    from command_model_v2.command_model import ITEM_KINDS, _destination_in, kind_in_phrase
    supported = {"ball", "cube", "cylinder", "triangular_prism", "hexagonal_prism"}
    check(supported <= set(ITEM_KINDS),
          "the trained item-kind schema includes every geometry the robot can manipulate")
    for phrase, expected in (("el cilindro verde", "cylinder"),
                             ("triangular prism", "triangular_prism"),
                             ("prisma hexagonal azul", "hexagonal_prism"),
                             ("zilindro berdea", "cylinder")):
        check(kind_in_phrase(phrase) == expected,
              f"an explicitly named {expected} is recovered from the object's own phrase")
    red = _destination_in("pon el cilindro verde en la caja roja", " en la caja roja")
    blue = _destination_in("pon la bola roja en la caja azul", " en la caja azul")
    check(red == {"kind": "box", "color": "red"}
          and blue == {"kind": "box", "color": "blue"},
          "destination colour is read from the words beside each named receptacle")
    multi_red = _destination_in("pon la bola verde en la caja roja y el cubo azul en la caja azul",
                                " en la caja roja y el ")
    multi_blue = _destination_in("pon la bola verde en la caja roja y el cubo azul en la caja azul",
                                 " en la caja azul")
    check(multi_red == {"kind": "box", "color": "red"}
          and multi_blue == {"kind": "box", "color": "blue"},
          "separate destinations retain their own colours instead of sharing one classifier label")


def test_front_depth_layers_split_same_colour_cube_from_its_container():
    import numpy as np
    import perceive

    fixture = ROOT / "tests" / "fixtures" / "front_red_cube_tray_depth.npz"
    with np.load(fixture) as sample:
        scene = perceive.perceive(sample["image_bgr"], depth=sample["depth"],
                                  looking_for=["red"], robust=True)
    red_parts = [thing for thing in scene.objects if thing.color == "red"]
    red_boxes = [thing for thing in scene.containers if thing.color == "red"]
    check(len(red_parts) == 1 and red_parts[0].kind == "cube",
          "depth layers recover the cube when its red mask touches the red tray")
    check(len(red_boxes) == 1,
          "splitting the part keeps the original connected component as one tray")


def test_container_detection_returns_large_blobs_and_excludes_balls():
    import numpy as np
    import perceive

    def blob(area, fill):
        return perceive.Blob(color="red", area_px=area, bbox_px=[0, 0, 50, 50],
                             centroid_px=[25.0, 25.0], mean_hsv=[0, 0, 0],
                             elongation=1.0, fill=fill, mask=np.ones((50, 50), np.uint8))

    check(not perceive.is_container(blob(1100, .96), 1200),
          "small object blobs stay outside the container class")
    check(perceive.is_container(blob(1200, .96), 1200),
          "large box-like blobs enter the container class at the configured floor")
    check(not perceive.is_container(blob(2400, .999), 1200),
          "a large round ball is never classified as a container")


def test_scene_catalog_corrects_camera_dependent_container_classification():
    from types import SimpleNamespace
    import scene as scene_module
    import scene_config

    module = load_demo11()
    config = scene_config.simple_config()
    _, items = scene_config.build(config)
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = items
    robot.demo10 = SimpleNamespace(LINK_TOLERANCE_M=.08)
    robot.tracked_xy = {item.prim: (item.x, item.y) for item in items}
    robot.catalog = scene_config.catalog(config, items)
    robot.args = SimpleNamespace(legacy_perception=False)
    robot.bus = SimpleNamespace(node=lambda *_args, **_kwargs: None)

    # Side-camera perspective made all four blobs look like objects in the captured frame.
    rows = [
        ("red", "cube", 8469, [.42784, -.50149, .0409]),
        ("blue", "cube", 3922, [.30526, -.50136, .0409]),
        ("blue", "cube", 2030, [.41274, -.43026, .0409]),
        ("red", "any", 1225, [.35104, -.43022, .0409]),
    ]
    objects = [scene_module.Thing(
        id=index, kind=kind, color=color, centroid_px=[0, 0], bbox_px=[0, 0, 1, 1],
        area_px=area, mean_hsv=[], confidence=.8, source="segmentation", pos_m=position)
        for index, (color, kind, area, position) in enumerate(rows)]
    seen = scene_module.Scene(frame=scene_module.Frame.from_image(640, 480),
                              source="side-camera", objects=objects)
    robot.scene = seen
    mapping = robot.link(seen)

    mapped = {thing.entity_id: thing for thing in seen.everything}
    expected_containers = {item.prim for item in items if item.container}
    expected_objects = {item.prim for item in items if not item.container}
    check(set(mapping.values()) == expected_containers | expected_objects,
          f"scene positions associate all side-view blobs, got {mapping}")
    check({thing.entity_id for thing in seen.containers} == expected_containers,
          "catalog association restores both containers despite perspective size confusion")
    check({thing.entity_id for thing in seen.objects} == expected_objects,
          "catalog association keeps the two loose cubes pickable")
    check(all(mapped[prim].kind == "box" for prim in expected_containers),
          "container geometry is restored from the scene catalog")


def test_grasp_tracking_does_not_replace_small_object_anchor_with_a_far_blob():
    from types import SimpleNamespace
    module = load_demo11()
    red_tray_fragment = SimpleNamespace(entity_id=None, color="red", kind="ball",
                                        aim_xy=(.434, -.522),
                                        pos_m=(.434, -.522, .05))
    near_ball = SimpleNamespace(entity_id=None, aim_xy=(.401, -.438),
                                pos_m=(.401, -.438, .05))
    linked_ball = SimpleNamespace(entity_id="small_ball", aim_xy=(.401, -.438),
                                  pos_m=(.401, -.438, .05))
    far_linked_ball = SimpleNamespace(entity_id="small_ball", color="red", kind="ball",
                                      aim_xy=(.50, -.52),
                                       pos_m=(.50, -.52, .05))
    candidates = [red_tray_fragment, near_ball, linked_ball, far_linked_ball]
    kept = module.target_tracking_candidates(candidates, "small_ball", (.38, -.435))
    check(kept == [linked_ball],
          "an exact catalog association wins over farther same-colour fragments")
    kept = module.target_tracking_candidates(candidates[:2], "small_ball", (.38, -.435))
    check(kept == [near_ball],
          "unlinked detections can refine an anchor only inside the 25 mm ordinary tracking gate")
    check(module.target_tracking_candidates([far_linked_ball], "small_ball", (.38, -.435)) == [],
          "even a catalog-linked detection cannot jump past the maximum object displacement")
    recoverable_ball = SimpleNamespace(entity_id="small_ball", aim_xy=(.428, -.435),
                                        pos_m=(.428, -.435, .05))
    check(module.target_tracking_candidates([recoverable_ball], "small_ball", (.38, -.435),
                                             max_shift_m=.060) == [recoverable_ball],
          "a 60 mm reacquisition remains available when a recovery path explicitly permits it")
    item = SimpleNamespace(color="red", kind="ball", size=(.02, .02, .02))
    raw_ball = SimpleNamespace(entity_id=None, color="red", kind="ball",
                               aim_xy=(.405, -.438), pos_m=(.405, -.438, .05))
    linked_as_tray = SimpleNamespace(entity_id="box_red", color="red", kind="box",
                                     aim_xy=(.405, -.438), pos_m=(.405, -.438, .05))
    recovered = module.reacquire_target_from_raw_proposals(
        [[raw_ball, linked_as_tray]], item, "small_ball", (.38, -.435),
        module.MonitoredRobot._nearest)
    check(recovered is raw_ball and recovered.entity_id == "small_ball",
          "recovery can use a raw ball proposal before catalog linking reclassifies it as a tray")
    check(module.reacquire_target_from_raw_proposals(
              [[red_tray_fragment, far_linked_ball]], item, "small_ball", (.38, -.435),
              module.MonitoredRobot._nearest) is None,
          "raw recovery rejects same-colour tray fragments outside the target motion gate")
    check(linked_as_tray.kind == "box",
          "the regression fixture models catalog reclassification without mutating raw proposals")
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = [SimpleNamespace(prim="small_ball", kind="ball", size=(.02, .02, .02))]
    trusted = (.38, -.435)
    robot.tracked_xy = {"small_ball": trusted}

    def locate(_prim, _last_xy, kind_hint=None):
        robot.tracked_xy["small_ball"] = (.44, -.50)  # same-colour tray picked by the detector
        return (.44, -.50)

    robot.locate = locate
    point = robot._observe_grasp_target("small_ball")
    check(point is None,
          "a grasp observation 60 mm away is treated as an occluded/outlier target")
    check(robot.tracked_xy["small_ball"] == trusted,
          "the last scene-linked small-object anchor survives the bad association")


def test_ambiguous_perception_to_catalog_assignment_abstains_instead_of_swapping_objects():
    import numpy as np

    module = load_demo11()
    ambiguous = np.asarray([[.010, .011], [.011, .010]])
    clear = np.asarray([[.010, .040], [.050, .010]])
    check(module.unambiguous_catalog_assignment(ambiguous, max_cost=.060) == [],
          "nearly tied same-colour detections are left unlinked instead of assigned arbitrarily")
    check(module.unambiguous_catalog_assignment(clear, max_cost=.060) == [(0, 0), (1, 1)],
          "well-separated detector-to-object matches preserve both identities")
    outside_gate = np.asarray([[.061]])
    check(module.unambiguous_catalog_assignment(outside_gate, max_cost=.060) == [],
          "detections outside the coordinate gate cannot acquire a catalog identity")


def test_ambiguous_catalog_link_is_marked_unusable_for_grasp_planning():
    from types import SimpleNamespace
    import scene as scene_module
    import scene_config
    import scene_spec

    module = load_demo11()
    items = [scene_spec._cube("red_a", "red", .400, -.450),
             scene_spec._cube("red_b", "red", .420, -.450)]
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = items
    robot.tracked_xy = {item.prim: (item.x, item.y) for item in items}
    robot.demo10 = SimpleNamespace(LINK_TOLERANCE_M=.035)
    robot.catalog = scene_config.catalog(scene_config.default_config(), items)
    robot.args = SimpleNamespace(legacy_perception=False)
    robot.bus = SimpleNamespace(node=lambda *_args, **_kwargs: None)
    detection = scene_module.Thing(
        id=1, kind="cube", color="red", centroid_px=None, bbox_px=[0, 0, 10, 10],
        area_px=100, mean_hsv=[], confidence=.9, source="segmentation",
        pos_m=[.410, -.450, scene_spec.TABLE_TOP_Z + .015], aim_xy=[.410, -.450])
    seen = scene_module.Scene(frame=scene_module.Frame.from_image(100, 100),
                              source="ambiguous-cluster", objects=[detection])
    robot.scene = seen
    mapping = robot.link(seen)
    check(not mapping and detection.ambiguous
          and detection.ambiguity_reason == "identidad de escena ambigua",
          "a blob equidistant from two same-colour parts stays visible as ungraspable ambiguity")


def test_populating_a_new_scene_refreshes_identity_links_before_perception():
    from types import SimpleNamespace
    from unittest import mock
    import scene as scene_module
    import scene_config

    module = load_demo11()
    config = scene_config.default_config()
    config["objects"] = [{"id": "custom_ball", "kind": "ball", "color": "red",
                          "count": 1, "size_mm": 24, "x": .38, "y": -.400}]
    config["containers"] = [next(item for item in config["containers"]
                                 if item["color"] == "red")]
    _canonical, initial = scene_config.build(config)
    changed = scene_spec.out_of_reach_layout(initial)
    target = next(item for item in changed if not item.container)

    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = list(initial)
    robot.tracked_xy = {item.prim: (item.x, item.y) for item in initial}
    robot.demo10 = SimpleNamespace(LINK_TOLERANCE_M=.08)
    robot.catalog = scene_config.catalog(config, initial)
    robot.args = SimpleNamespace(legacy_perception=False)
    robot.bus = SimpleNamespace(node=lambda *_args, **_kwargs: None)
    rendered_steps = []
    robot.env = SimpleNamespace(sim=SimpleNamespace(
        step=lambda *, render: rendered_steps.append(render)))

    parent = module.MonitoredRobot.__mro__[1]
    with mock.patch.object(parent, "populate",
                           lambda self, layout: setattr(self, "layout", list(layout))):
        robot.populate(changed)

    check(rendered_steps == [True] * module.SCENE_SETTLE_RENDER_STEPS,
          "a populated scene advances synchronized render steps before its first camera look")

    thing = scene_module.Thing(
        id=0, kind="ball", color=target.color, centroid_px=None, bbox_px=[0, 0, 10, 10],
        area_px=100, mean_hsv=[], confidence=.9, source="segmentation",
        pos_m=[target.x, target.y, scene_spec.TABLE_TOP_Z + target.height / 2],
        aim_xy=[target.x, target.y])
    seen = scene_module.Scene(frame=scene_module.Frame.from_image(100, 100),
                              source="new-scene", objects=[thing])
    robot.scene = seen
    mapping = robot.link(seen)
    check(mapping.get(thing.id) == target.prim,
          "a moved custom part links to its catalog identity after scene population")
    check(robot.tracked_xy[target.prim] == (target.x, target.y),
          "the identity anchor is refreshed to the current scene position")


def test_nearby_same_colour_instances_keep_distinct_identities_after_randomization():
    from types import SimpleNamespace
    from unittest import mock
    import scene as scene_module
    import scene_config
    import dynamics
    import importlib.util

    matrix_spec = importlib.util.spec_from_file_location(
        "demo11_cluster_matrix", ROOT / "tools" / "benchmark_scene_matrix.py")
    matrix = importlib.util.module_from_spec(matrix_spec)
    matrix_spec.loader.exec_module(matrix)
    config = matrix.scene_config("cube", 0, "cluster")
    _area, work_x, work_y = scene_config.working_area(config)
    previous_work_area = scene_spec.WORK_X, scene_spec.WORK_Y
    scene_spec.set_work_area(work_x, work_y)
    _canonical, initial = scene_config.build(config)
    randomized = dynamics.randomize(initial, seed=11500, spread=.008,
                                    allow_same_colour_clusters=True)
    module = load_demo11()
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = list(initial)
    robot.tracked_xy = {item.prim: (item.x, item.y) for item in initial}
    robot.demo10 = SimpleNamespace(LINK_TOLERANCE_M=.035)
    robot.catalog = scene_config.catalog(config, initial)
    robot.args = SimpleNamespace(legacy_perception=False)
    robot.bus = SimpleNamespace(node=lambda *_args, **_kwargs: None)
    parent = module.MonitoredRobot.__mro__[1]
    with mock.patch.object(parent, "populate",
                           lambda self, layout: setattr(self, "layout", list(layout))):
        robot.populate(randomized)

    detections = []
    for identifier, item in enumerate(item for item in randomized if not item.container):
        detections.append(scene_module.Thing(
            id=identifier, kind=item.kind, color=item.color, centroid_px=None,
            bbox_px=[0, 0, 12, 12], area_px=120, mean_hsv=[], confidence=.9,
            source="segmentation", pos_m=[item.x, item.y,
                                           scene_spec.TABLE_TOP_Z + item.height / 2],
            aim_xy=[item.x, item.y]))
    seen = scene_module.Scene(frame=scene_module.Frame.from_image(100, 100),
                              source="cluster-camera", objects=detections)
    robot.scene = seen
    mapping = robot.link(seen)
    green = [item for item in randomized if item.color == "green"]
    check(len(green) == 2 and green[0].prim != green[1].prim,
          "the close-cluster scene contains two separately catalogued green instances")
    check(all(mapping.get(thing.id) == next(item.prim for item in green
                                            if item.x == thing.pos_m[0] and item.y == thing.pos_m[1])
               for thing in detections if thing.color == "green"),
          "one-to-one linking preserves both nearby same-colour object identities")
    scene_spec.set_work_area(*previous_work_area)


def test_randomizer_respects_rotated_polygon_footprints_and_really_varies_yaw30_hexagons():
    import math
    import dynamics
    import scene_config
    import scene_spec

    config_path = ROOT / "results" / "demo11-hex20-vee-e2e-scene.json"
    data, items = load_recorded_scene(config_path)
    _, work_x, work_y = scene_config.working_area(data)
    previous = scene_spec.WORK_X, scene_spec.WORK_Y
    scene_spec.set_work_area(work_x, work_y)
    try:
        randomized = dynamics.randomize(items, seed=17180, spread=.018)
        original = next(item for item in items if not item.container)
        moved = next(item for item in randomized if not item.container)
        check((moved.x, moved.y) != (original.x, original.y),
              "a rotated hexagonal layout is randomized instead of silently falling back")
        check(not dynamics.conflicts(randomized),
              "rotated footprints stay inside the work area and keep physical clearance")
        target = next(item for item in randomized if not item.container)
        target_reach = math.hypot(target.x - scene_spec.ROBOT_XY[0],
                                  target.y - scene_spec.ROBOT_XY[1])
        check(target_reach >= scene_spec.MIN_REACH_M + dynamics.MIN_PICK_REACH_MARGIN_M,
              "randomized pick poses retain localization margin at the near-base reach limit")
        # The recorded trays are nearer the base than today's placement limit; this test is about
        # footprints and the pick margin, so check them without that rule.
        with patch.object(scene_spec, "MIN_PLACE_REACH_M", 0.0):
            for item in randomized:
                scene_config._check_position(
                    item, [other for other in randomized if other is not item], work_x, work_y)
    finally:
        scene_spec.set_work_area(previous[0], previous[1])


def test_randomizer_rejects_near_base_poses_with_less_than_the_pick_margin():
    from dataclasses import replace
    import dynamics
    import scene_config
    import scene_spec

    _, items = load_recorded_scene(ROOT / "results" / "demo11-hex20-vee-e2e-scene.json")
    target = next(item for item in items if not item.container)
    too_close = replace(target, x=scene_spec.ROBOT_XY[0] + scene_spec.MIN_REACH_M + .005,
                        y=scene_spec.ROBOT_XY[1])
    problems = dynamics.conflicts([too_close])
    check(any("clearance" in problem for problem in problems),
          "the layout validator rejects a pick only 5 mm outside the tested near-base limit")


def test_randomizer_keeps_matching_object_and_container_visually_separated():
    from dataclasses import replace
    import math
    import dynamics
    import scene_config

    _config, items = load_recorded_scene(
        ROOT / "results" / "demo11-mixed-v40-e2e-scene.json")
    observed_failure_layout = [
        replace(item, x=.318, y=-.490) if item.prim == "box_red" else item
        for item in items]
    problems = dynamics.conflicts(observed_failure_layout)
    check(any("tri_red and box_red" in problem or "box_red and tri_red" in problem
              for problem in problems),
          "same-colour source and tray at the observed 63.6 mm gap are rejected for perception")
    triangle = next(item for item in observed_failure_layout if item.prim == "tri_red")
    tray = next(item for item in observed_failure_layout if item.prim == "box_red")
    try:
        scene_config._check_position(triangle, [tray])
        config_rejected = False
    except ValueError as error:
        config_rejected = "75 mm" in str(error)
    check(config_rejected,
          "saved scene validation rejects the same ambiguous object-to-tray placement")

    randomized = dynamics.randomize(items, seed=17303, spread=.012)
    red = {item.prim: item for item in randomized if item.color == "red"}
    check(math.dist((red["tri_red"].x, red["tri_red"].y),
                    (red["box_red"].x, red["box_red"].y))
          >= dynamics.MIN_OBJECT_CONTAINER_GAP_M,
          "randomization preserves the object-to-matching-tray visual clearance")


def test_catalog_link_projects_side_camera_detections_at_candidate_height():
    from types import SimpleNamespace
    from unittest import mock
    import scene as scene_module
    import scene_config

    module = load_demo11()
    config = scene_config.simple_config()
    _, items = scene_config.build(config)
    target = next(item for item in items if item.color == "red" and not item.container)
    catalog = next(row for row in scene_config.catalog(config, items) if row["id"] == target.prim)
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = [target]
    robot.demo10 = SimpleNamespace(LINK_TOLERANCE_M=.035)
    robot.tracked_xy = {target.prim: (target.x, target.y)}
    robot.catalog = [catalog]
    robot.args = SimpleNamespace(legacy_perception=False)
    robot.bus = SimpleNamespace(node=lambda *_args, **_kwargs: None)

    # A low side view's generic table-plane intersection is 60 mm off. The known
    # object-height projection lands on the configured item and should win identity.
    detected = scene_module.Thing(
        id=0, kind="any", color=target.color, centroid_px=[40, 50], bbox_px=[30, 40, 20, 20],
        area_px=400, mean_hsv=[], confidence=.8, source="segmentation",
        pos_m=[target.x + .060, target.y, scene_spec.TABLE_TOP_Z],
    )
    seen = scene_module.Scene(frame=scene_module.Frame.from_image(100, 100),
                              source="side-camera", objects=[detected], camera={"side": True})

    class ProjectedCamera:
        @classmethod
        def from_dict(cls, _camera):
            return cls()

        def on_plane(self, _u, _v, _z):
            return [target.x, target.y, scene_spec.TABLE_TOP_Z + target.height / 2]

    robot.scene = seen
    with mock.patch.object(scene_module.CameraModel, "from_dict", ProjectedCamera.from_dict):
        mapping = robot.link(seen)
    check(mapping.get(detected.id) == target.prim,
          "catalog matching uses raised-plane projection before enforcing side-view link tolerance")


def test_catalog_link_keeps_trusted_pose_when_gripper_clips_object_mask():
    from types import SimpleNamespace
    import scene as scene_module
    import scene_config

    module = load_demo11()
    config = scene_config.simple_config()
    _canonical, items = scene_config.build(config)
    target = next(item for item in items if not item.container)
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = [target]
    robot.demo10 = SimpleNamespace(LINK_TOLERANCE_M=.035)
    robot.tracked_xy = {target.prim: (target.x, target.y)}
    robot.tracked_area_px = {target.prim: 400}
    robot.catalog = scene_config.catalog(config, items)
    robot.args = SimpleNamespace(legacy_perception=False)
    robot.bus = SimpleNamespace(node=lambda *_args, **_kwargs: None)
    detected = scene_module.Thing(
        id=0, kind="any", color=target.color, centroid_px=[40, 50], bbox_px=[35, 45, 10, 10],
        area_px=120, mean_hsv=[], confidence=.8, source="segmentation",
        pos_m=[target.x + .014, target.y, scene_spec.TABLE_TOP_Z + target.height / 2],
    )
    seen = scene_module.Scene(frame=scene_module.Frame.from_image(100, 100),
                              source="overhead", objects=[detected])
    robot.scene = seen
    mapping = robot.link(seen)
    check(mapping.get(detected.id) == target.prim,
          "a clipped mask can retain the catalog identity for display")
    check(robot.tracked_xy[target.prim] == (target.x, target.y)
          and detected.aim_xy == [target.x, target.y],
          "a clipped centroid cannot overwrite the pose used by the grasp controller")
    check(robot.tracked_area_px[target.prim] == 400,
          "occlusion cannot ratchet down the full-view mask reference")


def test_catalog_link_uses_configured_pose_for_static_simulator_receptacles():
    from types import SimpleNamespace
    import scene as scene_module
    import scene_config

    module = load_demo11()
    config = scene_config.simple_config()
    _canonical, items = scene_config.build(config)
    tray = next(item for item in items if item.container)
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.layout = [tray]
    robot.demo10 = SimpleNamespace(LINK_TOLERANCE_M=.035)
    robot.tracked_xy = {tray.prim: (tray.x, tray.y)}
    robot.catalog = scene_config.catalog(config, items)
    robot.args = SimpleNamespace(legacy_perception=False)
    robot.bus = SimpleNamespace(node=lambda *_args, **_kwargs: None)
    detected = scene_module.Thing(
        id=0, kind="any", color=tray.color, centroid_px=[40, 50], bbox_px=[30, 40, 20, 20],
        area_px=500, mean_hsv=[], confidence=.45, source="segmentation",
        pos_m=[tray.x + .020, tray.y + .010, scene_spec.TABLE_TOP_Z], hollow=True,
        inner_rect_m=[tray.x, tray.y, .05, .05])
    seen = scene_module.Scene(frame=scene_module.Frame.from_image(100, 100),
                              source="front-camera", containers=[detected])
    robot.scene = seen
    mapping = robot.link(seen)
    check(mapping.get(detected.id) == tray.prim,
          "a partially occluded tray still associates to its configured simulator identity")
    check(detected.aim_xy == [tray.x, tray.y],
          "the static receptacle uses its configured centre instead of the biased rim centroid")
    check(abs(detected.pos_m[0] - tray.x) < 1e-9 and abs(detected.pos_m[1] - tray.y) < 1e-9,
          "container planning and placement verification share the configured world pose")


def test_robust_perception_handles_brightness_and_touching_balls():
    import cv2
    import numpy as np
    import perceive
    image=np.zeros((150,200,3),dtype=np.uint8)
    cv2.circle(image,(65,75),19,(0,0,180),-1)
    cv2.circle(image,(99,75),19,(0,0,180),-1)
    result=perceive.perceive(image,looking_for=["red"],robust=True)
    check(len(result.objects)==2, "watershed separates a pair of touching balls")
    check(all(o.kind=="ball" for o in result.objects), "separated objects retain their kind")
    image=np.zeros((150,220,3),dtype=np.uint8)
    for x in (49, 82, 115):
        cv2.circle(image,(x,75),19,(0,0,180),-1)
    result=perceive.perceive(image,looking_for=["red"],robust=True)
    check(len(result.objects)==3, "watershed separates a compact three-ball cluster")
    check(all(o.area_px > 150 for o in result.objects), "cluster split rejects tiny watershed fragments")
    image=np.zeros((150,180,3),dtype=np.uint8)
    cv2.circle(image,(53,75),22,(0,0,180),-1)
    cv2.circle(image,(84,75),13,(0,0,180),-1)
    result=perceive.perceive(image,looking_for=["red"],robust=True)
    check(len(result.objects)==2, "local maxima preserve a smaller object touching a larger one")
    image=np.zeros((150,200,3),dtype=np.uint8);cv2.circle(image,(80,75),19,(0,0,115),-1)
    result=perceive.perceive(image,looking_for=["red"],robust=True)
    check(len(result.objects)==1 and result.objects[0].kind=="ball", "a shaded red ball stays detectable")


def test_unresolved_oversized_blob_is_ambiguous_and_cannot_match_a_grasp_target():
    import cv2
    import numpy as np
    import perceive
    import scene as scene_module

    mask = np.zeros((100, 120), dtype=np.uint8)
    cv2.ellipse(mask, (60, 50), (34, 18), 0, 0, 360, 1, -1)
    area = int(mask.sum())
    blob = perceive.Blob("red", area, [26, 32, 69, 37], [60, 50],
                         [0, 255, 180], .8, .9, mask.astype(bool))
    kept = perceive.split_touching(blob, max_extent_px=150, min_area_px=80,
                                   single_object_extent_px=100,
                                   single_object_area_px=500)
    check(len(kept) == 1 and kept[0].ambiguous_group,
          "a region larger than any configured part is not silently accepted as one object")

    thing = scene_module.Thing(
        id=0, kind="cube", color="red", centroid_px=[60, 50], bbox_px=[26, 32, 69, 37],
        area_px=area, mean_hsv=[0, 255, 180], confidence=.8, source="segmentation",
        ambiguous=True, ambiguity_reason="no hay semillas de separación fiables")
    check(not thing.matches("any", "red"),
          "ambiguous regions cannot satisfy an object description")
    restored = scene_module.Thing.from_dict(thing.to_dict())
    check(restored.ambiguous and restored.ambiguity_reason == thing.ambiguity_reason,
          "ambiguity survives the scene JSON round trip")


def test_oblique_camera_does_not_mark_a_single_measured_cube_as_an_ambiguous_cluster():
    import json
    import numpy as np
    import perceive
    import scene_spec
    from scene import CameraModel

    metadata = json.loads((ROOT / "tests" / "fixtures" / "small_triangle_top" / "scene.json")
                          .read_text(encoding="utf-8"))
    camera_data = metadata["camera"]
    camera_data["matrix"] = [[value * 2 for value in row] for row in camera_data["matrix"]]
    camera_data["width"] *= 2
    camera_data["height"] *= 2
    camera = CameraModel.from_dict(camera_data)
    ppm = camera.pixels_per_m(scene_spec.TABLE_TOP_Z)
    extent_px = max(8, int(round(ppm * .030 * 1.5)))
    area_limit_px = perceive.projected_single_object_area_px(.030 * .030, ppm)
    measured_pixels = 2762  # blue 30 mm cube from the 1280x960 top-camera trial
    region = np.zeros((61, 50), dtype=bool)
    region.flat[:measured_pixels] = True
    blob = perceive.Blob(
        "blue", measured_pixels, [591, 483, 50, 61], [616, 513],
        [110, 220, 130], .80, .96, region)
    kept = perceive.split_touching(
        blob, max_extent_px=256, min_area_px=80,
        single_object_extent_px=extent_px,
        single_object_area_px=area_limit_px)
    check(len(kept) == 1 and not kept[0].ambiguous_group,
          "one obliquely projected cube within its size envelope remains a confirmed object")


def test_configured_color_prior_skips_depth_clustering_for_unrelated_tray_colors():
    import cv2
    import numpy as np
    import perceive

    image = np.zeros((180, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (15, 35), (105, 145), (0, 0, 180), -1)
    cv2.rectangle(image, (155, 75), (184, 104), (180, 0, 0), -1)
    depth = np.full(image.shape[:2], .50, dtype=np.float32)
    attempted_colors = []
    original = perceive.split_depth_layered_container

    def track_split(blob, *args, **kwargs):
        attempted_colors.append(blob.color)
        return original(blob, *args, **kwargs)

    perceive.split_depth_layered_container = track_split
    try:
        seen = perceive.perceive(
            image, depth=depth, looking_for=["red", "blue"], robust=True,
            object_bounds_by_color={"blue": (.030, .0009)})
    finally:
        perceive.split_depth_layered_container = original
    check("red" not in attempted_colors and any(item.color == "red" for item in seen.containers),
          "a red tray remains detected without running depth clustering when only blue parts exist")


def test_scene_reset_never_writes_velocity_to_kinematic_or_unknown_bodies():
    import isaac_io
    from types import SimpleNamespace
    import torch

    writes = []
    actor = SimpleNamespace(write_root_velocity_to_sim=lambda value: writes.append(value))
    with patch.object(isaac_io, "root_is_kinematic", return_value=True):
        check(not isaac_io.zero_dynamic_root_velocity(actor, "zero"),
              "kinematic bodies are excluded from velocity resets")
    with patch.object(isaac_io, "root_is_kinematic", return_value=None):
        check(not isaac_io.zero_dynamic_root_velocity(actor, "zero"),
              "an unverified body is not sent to PhysX's dynamic-only velocity API")
    with patch.object(isaac_io, "root_is_kinematic", return_value=False):
        check(isaac_io.zero_dynamic_root_velocity(actor, "zero"),
              "dynamic bodies still have their residual velocity cleared")
    check(writes == ["zero"], "only the confirmed dynamic body receives a velocity write")

    def fake_rigid(name):
        calls = []
        actor = SimpleNamespace(
            name=name,
            data=SimpleNamespace(default_root_state=torch.zeros((1, 13))),
            write_root_pose_to_sim=lambda pose, **kwargs: calls.append("pose"),
            write_root_velocity_to_sim=lambda vel, **kwargs: calls.append("velocity"),
        )
        return actor, calls

    dynamic, dynamic_calls = fake_rigid("dynamic")
    fixed, fixed_calls = fake_rigid("fixed")
    unknown, unknown_calls = fake_rigid("unknown")
    env = SimpleNamespace(scene=SimpleNamespace(
        rigid_objects={"dynamic": dynamic, "fixed": fixed, "unknown": unknown},
        articulations={}, deformable_objects={}, env_origins=torch.zeros((1, 3))))
    states = {id(dynamic): False, id(fixed): True, id(unknown): None}
    with patch.object(isaac_io, "root_is_kinematic", side_effect=lambda actor: states[id(actor)]):
        isaac_io.reset_scene_to_default_without_kinematic_velocity_writes(env, torch.tensor([0]))
    check(dynamic_calls == ["pose", "velocity"],
          "confirmed dynamic scene objects reset their pose and velocity")
    check(fixed_calls == ["pose"] and unknown_calls == ["pose"],
          "kinematic or unverified scene objects reset pose without invalid velocity writes")


def test_large_touching_clusters_split_without_reclassifying_the_children_as_containers():
    import cv2
    import numpy as np
    import perceive

    for centers, radii, expected in (
        ([(65, 75), (99, 75)], [19, 19], 2),
        ([(49, 75), (82, 75), (115, 75)], [19, 19, 19], 3),
        ([(53, 75), (84, 75)], [22, 13], 2),
    ):
        image = np.zeros((150, 220, 3), dtype=np.uint8)
        for (x, y), radius in zip(centers, radii):
            cv2.circle(image, (x, y), radius, (0, 0, 180), -1)
        result = perceive.perceive(image, looking_for=["red"], robust=True)
        assert len(result.objects) == expected
        assert not result.containers
        assert all(item.kind == "ball" for item in result.objects)


def test_top_camera_tray_walls_are_not_split_into_fake_objects():
    """A real top view splits one red tray into large lobes unless the parent is kept."""
    import cv2
    import json
    import perceive
    import scene_spec
    from scene import CameraModel

    fixture = ROOT / "tests" / "fixtures" / "container_fragment_top"
    image = cv2.imread(str(fixture / "frame.png"))
    metadata = json.loads((fixture / "scene.json").read_text(encoding="utf-8"))
    scene = perceive.perceive(
        image, camera=CameraModel.from_dict(metadata["camera"]),
        plane_z=scene_spec.TABLE_TOP_Z, looking_for=["red", "blue"], robust=True,
        min_area_px=80)
    red_objects = [item for item in scene.objects if item.color == "red"]
    red_containers = [item for item in scene.containers if item.color == "red"]
    blue_objects = [item for item in scene.objects if item.color == "blue"]
    check(not red_objects, "large red tray wall fragments are not emitted as pickable objects")
    check(len(red_containers) == 1 and red_containers[0].kind == "box",
          "top-camera tray remains one correctly typed receptacle")
    check(len(blue_objects) == 1 and blue_objects[0].kind == "cube",
          "shading peaks on one blue cube do not create a duplicate object")



def _lone_cube_yaw30_scene(area_allowance=None, extent_factor=None):
    """Perceive the saved top frame of one 30 mm cube at yaw 30 degrees."""
    import cv2
    import perceive
    import scene_spec
    from scene import CameraModel

    fixture = ROOT / "tests" / "fixtures" / "lone_cube_yaw30_top"
    image = cv2.imread(str(fixture / "frame.png"))
    metadata = json.loads((fixture / "scene.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(
        "matrix_priors", ROOT / "tools" / "benchmark_scene_matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    priors = matrix.perception_geometry_priors(metadata["truth"])
    bounds = matrix.perception_work_bounds(metadata["truth"], metadata)
    saved = (perceive.SINGLE_OBJECT_AREA_ALLOWANCE, perceive.SINGLE_OBJECT_EXTENT_FACTOR,
             scene_spec.WORK_X, scene_spec.WORK_Y)
    try:
        if area_allowance is not None:
            perceive.SINGLE_OBJECT_AREA_ALLOWANCE = area_allowance
        if extent_factor is not None:
            perceive.SINGLE_OBJECT_EXTENT_FACTOR = extent_factor
        scene_spec.WORK_X, scene_spec.WORK_Y = bounds["work_x"], bounds["work_y"]
        return perceive.perceive(
            image, camera=CameraModel.from_dict(metadata["camera"]),
            plane_z=scene_spec.TABLE_TOP_Z, looking_for=["blue", "red"], robust=True,
            min_area_px=perceive.colors.min_area_for_object_size(priors["smallest_part_m"]),
            max_object_extent_m=priors["largest_part_m"],
            max_object_footprint_m2=priors["largest_part_footprint_m2"],
            object_bounds_by_color=priors["object_bounds_by_color"]), metadata
    finally:
        (perceive.SINGLE_OBJECT_AREA_ALLOWANCE, perceive.SINGLE_OBJECT_EXTENT_FACTOR,
         scene_spec.WORK_X, scene_spec.WORK_Y) = saved


def test_a_physical_regrasp_is_counted_in_the_final_report():
    """Strict matrix: first close failed, the retry gripped (99 % lift) and the task succeeded,
    yet the report said 0 physical grasps because only ``Executor.run`` counted them."""
    import types

    demo = load_demo11()
    robot = types.SimpleNamespace()
    count = demo.MonitoredRobot._count_real_grasps
    check(count(robot, 0) == 0, "a run with no physical grasp adds nothing")
    check(count(robot, 1) == 1 and count(robot, 1) == 2,
          "later runs and a recovery grasp add to the total instead of replacing it")
    source = (ROOT / "11_real_sorting_with_monitor.py").read_text(encoding="utf-8")
    check('real_grasps=getattr(robot, "real_grasps_total", 0)' in source,
          "the final state publishes the command's cumulative physical-grasp total")
    check('lifted.ok and getattr(lifted, "grasp", None) == "real"' in source,
          "the recovery lift counts a physical grasp")


def test_a_sort_leaves_out_parts_that_are_already_in_a_tray():
    """After one order placed the red cube, "separa las piezas por colores" was refused 6/6.

    The cube in the red tray came back from perception as an unlinked red object at the tray, the
    sort planned a pick for it, could not name it, and answered "No puedo identificar todas las
    piezas y sus destinos".
    """
    from types import SimpleNamespace

    demo = load_demo11()
    tray = SimpleNamespace(pos_m=[.44, -.46, .056], size_m=[.07, .07, .03])
    placed = SimpleNamespace(aim_xy=[.434, -.474], pos_m=[.434, -.474, .056])
    loose = SimpleNamespace(aim_xy=[.43, -.38], pos_m=[.43, -.38, .056])
    beside = SimpleNamespace(aim_xy=[.44 + .04, -.46], pos_m=[.48, -.46, .056])
    seen = SimpleNamespace(containers=[tray], objects=[placed, loose, beside])
    ignored = demo.MonitoredRobot._ignore_parts_already_in_trays(seen)
    check(ignored == 1 and seen.objects == [loose, beside],
          f"only the part inside the tray outline is left out, kept {len(seen.objects)}")
    bare = SimpleNamespace(containers=[], objects=[loose])
    check(demo.MonitoredRobot._ignore_parts_already_in_trays(bare) == 0 and bare.objects == [loose],
          "with no tray nothing is left out")


def test_a_part_resting_in_the_cavity_is_verified_even_if_it_leans_and_one_on_the_wall_is_not():
    """The dashboard said FALLIDA for an order the arm had carried out correctly.

    Seed 24002: the red cube came to rest at (0.432, -0.462) in a tray centred (0.44, -0.46),
    inside the cavity in x and y, with its centre 6.2 mm above the flat-rest height (it leaned on
    the cavity wall). The check allowed 6 mm, so the order was judged "balanced on the rim" and
    failed, and the retry then looked for the cube at its old position.
    """
    import scene_config

    _, parts = scene_config.build(scene_config.demo_config())
    tray = next(item for item in parts if item.prim == "box_red")
    flat = scene_spec.TABLE_TOP_Z + tray.floor + 0.015

    def verdict(dx, dy, z):
        return real_world.cavity_verdict((tray.x + dx, tray.y + dy, z), 0.03, tray)

    check(verdict(0.0, 0.0, flat)[0], "a cube lying flat in the middle is inside")
    check(verdict(-0.008, -0.002, flat + 0.0062)[0],
          "seed 24002's cube (centre 6.2 mm above flat rest, inside the cavity) is inside")
    check(verdict(0.012, 0.012, flat + 0.011)[0],
          "a cube standing on a corner in the cavity (the highest a part on the floor can be) is inside")
    ok, reason = verdict(0.02, 0.0, real_world.rim_z(tray) + 0.015)
    check(not ok and "balanced on the wall" in reason,
          f"a cube whose centre is a half-part above the rim is on the wall, got {reason!r}")
    ok, reason = verdict(0.045, 0.0, flat)
    check(not ok and "outside the cavity" in reason,
          f"a cube beside the tray is not inside, got {reason!r}")
    ok, reason = verdict(0.0, 0.0, flat - 0.012)
    check(not ok and "below the floor" in reason,
          f"a part under the floor-rest height is not on the floor, got {reason!r}")


def test_a_tray_the_arm_cannot_lower_a_part_into_is_refused_with_a_reason():
    """The saved scene that broke the demo had trays 0.12 m from the base.

    Strict trials: a tray at 0.165 m and beyond took the part (verified); at 0.15 m the carry
    stopped 20.3 mm short of its 20 mm tolerance; at 0.12 m the part was never gripped. Nothing
    checked a tray against the arm's near limit, so every order gripped the cube and then stopped.
    """
    import copy
    import scene_config
    import scene_spec
    from ground import ground_place
    from scene import Scene

    facts_near = scene_spec.reach_facts(0.299, -0.53)
    facts_far = scene_spec.reach_facts(0.32, -0.46)
    check(facts_near["outside_place_envelope"] and not facts_far["outside_place_envelope"],
          "reach facts say a tray at 0.121 m is too near and one at 0.18 m is fine")
    check("nearer than" in (scene_spec.why_unreachable(
              0.299, -0.53, require_place_envelope=True) or ""),
          "the reason names the placement limit")
    check(scene_spec.why_unreachable(0.299, -0.53) is None,
          "callers that do not ask for the placement rule are unchanged")

    saved = json.loads((ROOT / "tests" / "fixtures" / "scenes" / "saved_scene_trays_near_base.json")
                       .read_text(encoding="utf-8"))
    try:
        scene_config.build(saved)
        rejected = ""
    except ValueError as error:
        rejected = str(error)
    check("tray_blue" in rejected and "170 mm" in rejected,
          f"the saved scene is rejected naming the tray and the limit, got {rejected!r}")

    # Every shipped preset must be a scene the arm can actually carry out.
    for name, build_preset in scene_config.PRESETS.items():
        try:
            _, parts = scene_config.build(build_preset())
            ok = all(math.hypot(p.x - scene_spec.ROBOT_XY[0], p.y - scene_spec.ROBOT_XY[1])
                     >= scene_spec.MIN_PLACE_REACH_M for p in parts if p.container)
        except ValueError as error:
            ok = False
            print(f"  preset {name}: {error}")
        check(ok, f"preset '{name}' places its trays within the arm's placement reach")
        # And every loose part stands at least 45 mm from the kerb walls the contained variant
        # builds, so the open jaws (33 mm from the part's centre with the quarter-turn wrist)
        # never touch them.
        import pen
        canonical, built = scene_config.build(build_preset())
        _, work_x, work_y = scene_config.working_area(canonical)
        previous = scene_spec.WORK_X, scene_spec.WORK_Y
        scene_spec.set_work_area(work_x, work_y)
        try:
            (wall_x0, wall_x1), (_near, wall_far) = pen.contained_region()
        finally:
            scene_spec.set_work_area(*previous)
        for part in (p for p in built if not p.container):
            gap = min(part.x - wall_x0, wall_x1 - part.x, wall_far - part.y)
            check(gap >= 0.045 - 1e-9,
                  f"preset '{name}': {part.prim} is {gap * 1000:.0f} mm from the kerb, jaws need 45")

    # A command whose only destination is too near is refused in words, not attempted.
    scene = Scene.read(ROOT / "tests" / "fixtures" / "trial001.json")
    result = ground_place(scene, {"kind": "box", "color": "red"})
    check(result.status == "not_reachable" and "cm de la base" in result.question["es"],
          f"an unreachable recorded tray is refused: {result.status} {result.question}")

    # And a saved scene that breaks the rule falls back to the demo scene with a notice.
    with tempfile.TemporaryDirectory() as directory:
        bad = Path(directory) / "scene_config.json"
        bad.write_text(json.dumps(copy.deepcopy(saved)), encoding="utf-8")
        with patch.object(scene_config, "SAVED_SCENE", bad):
            _, items = scene_config.load()
            notice = scene_config.LAST_LOAD_NOTICE
            _, same = scene_config.load(bad)
        check(notice and "tray_blue" in notice and "demo" in notice,
              f"the fallback says why, got {notice!r}")
        check({i.prim for i in items} == {"cube_red", "cube_blue", "tray_blue", "box_red"}
              and [i.prim for i in same] == [i.prim for i in items],
              "the demo scene runs instead, whether the saved path is implicit or explicit")


def test_the_containment_kerb_never_puts_its_near_wall_where_the_arm_cannot_work():
    """The default `contained` variant refused to pick anything in a saved scene.

    The kerb's inner face sits on the working-area edge. The scene the dashboard saved reached
    y = -0.589, 51 mm from the base axis; strict trials of one cube with the near face at -0.555
    passed and at -0.565, -0.575, -0.585 and -0.595 every one failed ("pre-grasp hover did not
    converge"), while the same cube in the open variant passed. The kerb now stays at least
    ``PEN_NEAR_LIMIT_FROM_BASE_M`` from the base axis, whatever area was saved.
    """
    import pen
    import scene_spec

    saved = scene_spec.WORK_X, scene_spec.WORK_Y
    limit_y = scene_spec.ROBOT_XY[1] + pen.PEN_NEAR_LIMIT_FROM_BASE_M
    try:
        scene_spec.set_work_area((.242, .492), (-.589, -.429))   # the saved scene that failed
        x_range, y_range = pen.contained_region()
        clearance = pen.PEN_JAW_CLEARANCE_M
        check(abs(y_range[0] - limit_y) < 1e-9 and abs(y_range[1] - (-.429 + clearance)) < 1e-9
              and all(abs(a - b) < 1e-9 for a, b in zip(x_range, (.242 - clearance, .492 + clearance))),
              f"the near wall is held at {limit_y:.3f} and the others stand {clearance * 1000:.0f} "
              f"mm out for the jaws, got {x_range} {y_range}")
        scene_spec.set_work_area((.285, .525), (-.555, -.355))   # the validated benchmark area
        _, y_range = pen.contained_region()
        check(y_range[0] >= -.555 and y_range[0] <= limit_y + 1e-9,
              "the validated area is at most nudged away from the base, never towards it")
        scene_spec.set_work_area((.285, .525), (-.45, -.30))     # farther out: untouched
        near_edge, far_edge = pen.contained_region()[1]
        check(near_edge == -.45 and abs(far_edge - (-.30 + clearance)) < 1e-9,
              "an area already beyond the near limit keeps its own near edge")
        scene_spec.set_work_area((.285, .525), (-.62, -.60))     # nonsense strip near the base
        near, far = pen.contained_region()[1]
        check(far - near >= .05 - 1e-9, "a degenerate area still gets a kerb of positive size")
    finally:
        scene_spec.set_work_area(*saved)
    check(.080 <= pen.PEN_NEAR_LIMIT_FROM_BASE_M <= .100,
          "the limit stays tied to the measured 85 mm pass and 75 mm failure")


def test_a_tray_cavity_floor_is_not_recovered_as_a_pickable_part():
    """The default scene (seed 56000, top, contained) refused every order.

    The depth-layer recovery meant for a part pressed against a same-colour tray wall took each
    tray's own cavity floor (4,300 px on a 70 mm tray, bigger than any 30 mm part) for a "blue
    ball" and a "red cube". Those phantoms competed with the real trays for their catalog
    identity, so nothing could be planned and the arm never moved.
    """
    import cv2
    import numpy as np
    import perceive
    import scene_spec
    from scene import CameraModel

    fixture = ROOT / "tests" / "fixtures" / "tray_floor_top"
    image = cv2.imread(str(fixture / "frame.png"))
    depth = np.load(fixture / "depth.npz")["depth"]
    metadata = json.loads((fixture / "scene.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(
        "matrix_priors_tray", ROOT / "tools" / "benchmark_scene_matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    priors = matrix.perception_geometry_priors(metadata["truth"])
    bounds = matrix.perception_work_bounds(metadata["truth"], metadata)
    saved = scene_spec.WORK_X, scene_spec.WORK_Y
    try:
        scene_spec.WORK_X, scene_spec.WORK_Y = bounds["work_x"], bounds["work_y"]
        scene = perceive.perceive(
            image, camera=CameraModel.from_dict(metadata["camera"]),
            plane_z=scene_spec.TABLE_TOP_Z, depth=depth, robust=True,
            looking_for=sorted({row["color"] for row in metadata["truth"]}),
            min_area_px=perceive.colors.min_area_for_object_size(priors["smallest_part_m"]),
            max_object_extent_m=priors["largest_part_m"],
            max_object_footprint_m2=priors["largest_part_footprint_m2"],
            object_bounds_by_color=priors["object_bounds_by_color"])
    finally:
        scene_spec.WORK_X, scene_spec.WORK_Y = saved
    parts = [row for row in metadata["truth"] if not row["container"]]
    check(len(scene.objects) == len(parts) == 3,
          f"only the 3 real parts are objects, got {[(o.kind, o.color, o.area_px) for o in scene.objects]}")
    check(len(scene.containers) == 2, "and the two trays stay receptacles")
    check(not any(o.area_px > 3500 for o in scene.objects),
          "no 'object' is larger than a 30 mm part could be")
    for row in parts:
        nearest = min((np.hypot((o.aim_xy or o.pos_m)[0] - row["position"][0],
                                (o.aim_xy or o.pos_m)[1] - row["position"][1])
                       for o in scene.objects if o.color == row["color"]), default=9.0)
        check(nearest < .012, f"{row['id']} is localised within 12 mm (got {nearest * 1000:.1f} mm)")
    check(not any(o.ambiguous for o in scene.objects), "none of the real parts is flagged ambiguous")


def test_a_lone_rotated_cube_is_not_flagged_as_an_unresolved_group():
    """Strict matrix: a lone 30 mm cube at yaw 30 deg was 'ambiguous' and the order rejected.

    From the oblique top camera it spans 65 x 74 px (49.6 mm), over the old 1.5x single-part
    limit of 67 px, and covers 2,985 px against a 2,869 px trigger. With no separation seeds
    the detector kept it as a flagged group, the catalog linker dropped flagged detections, and
    the command was refused with "no se identifican todas las piezas" before any motion.
    """
    import numpy as np

    scene, metadata = _lone_cube_yaw30_scene()
    cubes = [item for item in scene.objects if item.color == "blue"]
    truth = next(row for row in metadata["truth"] if row["id"] == "matrix_piece")
    check(len(cubes) == 1 and cubes[0].kind == "cube",
          f"one blue cube is detected, got {[(c.kind, c.area_px) for c in cubes]}")
    check(cubes and not cubes[0].ambiguous,
          f"the lone cube is a resolved part, not '{cubes[0].ambiguity_reason if cubes else ''}'")
    check(cubes and np.linalg.norm(
        np.asarray(cubes[0].aim_xy or cubes[0].pos_m[:2]) - np.asarray(truth["position"][:2]))
          < .010, "and it is localised within 10 mm of the simulator pose")
    old, _ = _lone_cube_yaw30_scene(area_allowance=1.10, extent_factor=1.5)
    old_cubes = [item for item in old.objects if item.color == "blue"]
    check(old_cubes and old_cubes[0].ambiguous,
          "with the former limits this frame is flagged, so the fixture exercises the fix")
    import perceive
    check(2.0 > perceive.SINGLE_OBJECT_EXTENT_FACTOR,
          "two parts side by side (2x a part) still exceed the single-part extent")


def test_small_triangle_from_real_top_camera_capture_survives_size_aware_threshold():
    """A 20 mm triangle rendered at only 133 pixels was dropped by the old 150 px floor."""
    import cv2
    import numpy as np
    import colors
    import perceive
    import scene_spec
    from scene import CameraModel

    fixture = ROOT / "tests" / "fixtures" / "small_triangle_top"
    image = cv2.imread(str(fixture / "frame.png"))
    metadata = json.loads((fixture / "scene.json").read_text(encoding="utf-8"))
    camera = CameraModel.from_dict(metadata["camera"])
    target = metadata["target_xy"]
    ordinary = perceive.perceive(
        image, camera=camera, plane_z=scene_spec.TABLE_TOP_Z,
        looking_for=["red", "green"], robust=True,
        min_area_px=colors.MIN_AREA_PX)
    adaptive = perceive.perceive(
        image, camera=camera, plane_z=scene_spec.TABLE_TOP_Z,
        looking_for=["red", "green"], robust=True,
        min_area_px=colors.min_area_for_object_size(.020))
    red_objects = [thing for thing in adaptive.objects if thing.color == "red"]
    check(not any(thing.area_px < 150 for thing in ordinary.objects),
          "the conservative 150 px threshold continues to reject small components")
    check(len(red_objects) == 1 and red_objects[0].area_px == 133,
          "the adaptive small-object threshold recovers the 133 px triangle without duplicates")
    check(red_objects and red_objects[0].kind == "triangular_prism",
          "RGB contour evidence identifies the 133 px triangle without simulator depth")
    check(red_objects and np.linalg.norm(np.asarray(red_objects[0].pos_m[:2]) - target) < .015,
          "the recovered triangle projects within 15 mm of its simulator pose")
    check(colors.min_area_for_object_size(.030) == colors.MIN_AREA_PX,
          "30 mm scenes retain the original noise-rejection floor")
    check(colors.min_area_for_object_size(.020) == 80,
          "the 80 px small-object floor admits the 110 px triangle measured in the 20 mm top-camera capture")


def test_perception_pixel_thresholds_scale_with_camera_resolution():
    import perceive

    check(perceive.image_pixel_scale(640, 480) == 1.0,
          "the calibrated 640x480 camera keeps its original pixel thresholds")
    check(perceive.image_pixel_scale(1280, 960) == 2.0,
          "doubling both image dimensions yields a 2x linear morphology scale")


def test_depth_corrects_oblique_object_centroid_to_half_object_height():
    import json
    import numpy as np
    import perceive
    import scene_spec
    from scene import CameraModel

    metadata = json.loads((ROOT / "tests" / "fixtures" / "small_triangle_top" / "scene.json")
                          .read_text(encoding="utf-8"))
    camera = CameraModel.from_dict(metadata["camera"])
    plane_z = scene_spec.TABLE_TOP_Z
    centre = np.array([.36, -.43, plane_z + .010])
    top_z = plane_z + .020
    pixel = camera.project(centre)
    u, v = (int(round(value)) for value in pixel)
    mask = np.zeros((camera.height, camera.width), dtype=bool)
    mask[v-5:v+6, u-5:u+6] = True
    depth = np.zeros(mask.shape, dtype=np.float32)
    for py, px in zip(*np.nonzero(mask)):
        direction = camera.rotation @ camera.ray(px, py)
        depth[py, px] = (top_z - camera.origin[2]) / direction[2]
    blob = perceive.Blob("blue", int(mask.sum()), [u-5, v-5, 11, 11],
                         [float(u), float(v)], [110, 180, 180], .9, .99, mask)
    estimated_z = perceive.object_center_z(blob, camera, depth, plane_z)
    aimed = np.asarray(camera.on_plane(u, v, estimated_z))[:2]
    check(abs(estimated_z - centre[2]) < .001,
          "depth estimates the 20 mm object's centre height from its flat top")
    check(np.linalg.norm(aimed - centre[:2]) < .003,
          "height-corrected centroid ray returns within 3 mm in the oblique camera")


def test_depth_and_contour_features_separate_supported_shapes():
    import cv2
    import numpy as np
    from unittest import mock
    import perceive

    size = 161
    centre = size // 2
    masks = {}
    circle = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(circle, (centre, centre), 48, 1, -1)
    masks["circle"] = circle
    for kind, sides in (("triangle", 3), ("square", 4), ("hexagon", 6)):
        angles = np.linspace(0.0, 2 * np.pi, sides, endpoint=False) - np.pi / 2
        points = np.rint(np.stack((centre + 48 * np.cos(angles),
                                   centre + 48 * np.sin(angles)), axis=1)).astype(np.int32)
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(mask, points, 1)
        masks[kind] = mask

    circle_harmonics = perceive.contour_harmonics(masks["circle"])
    triangle_harmonics = perceive.contour_harmonics(masks["triangle"])
    square_harmonics = perceive.contour_harmonics(masks["square"])
    hexagon_harmonics = perceive.contour_harmonics(masks["hexagon"])
    check(max(circle_harmonics) < .01,
          "an ellipse-normalized circular contour has little polygon harmonic energy")
    check(triangle_harmonics[0] > triangle_harmonics[1] * 2,
          "triangle contour evidence is strongest at the third harmonic")
    check(square_harmonics[1] > square_harmonics[0] * 2,
          "square contour evidence is strongest at the fourth harmonic")
    check(hexagon_harmonics[2] > hexagon_harmonics[0] * 2,
          "hexagon contour evidence is strongest at the sixth harmonic")

    spherical_surface = np.linspace(0.0, .030, 400)
    flat_cap_and_sides = np.concatenate((np.full(280, .030), np.linspace(0.0, .030, 120)))
    sphere_top = perceive.surface_top_fraction(spherical_surface)
    cylinder_top = perceive.surface_top_fraction(flat_cap_and_sides)
    check(sphere_top is not None and sphere_top < perceive.BALL_DEPTH_TOP_FRACTION,
          "curved depth surfaces have a small uppermost height band")
    check(cylinder_top is not None and cylinder_top > perceive.BALL_DEPTH_TOP_FRACTION,
          "flat cylinder caps have a larger uppermost height band than spheres")

    def blob(kind, fill, elongation):
        mask = masks[kind]
        return perceive.Blob("red", int(mask.sum()), [0, 0, size, size],
                             [centre, centre], [0, 255, 200], elongation, fill,
                             mask.astype(bool), 0)

    with mock.patch.object(perceive, "depth_top_fraction", return_value=.20):
        ball_blob = blob("circle", .999, .98)
        check(perceive.robust_kind(ball_blob, camera=object(), depth=np.ones((1, 1))) == "ball",
              "a curved depth profile classifies the circular silhouette as a ball")
        ball_confidence = perceive.confidence_of(ball_blob, .3, "ball", object(), np.ones((1, 1)))
        check(ball_confidence >= .75,
              f"round-object confidence uses ball evidence, not a cube-only fill score ({ball_confidence})")
        edge_confidence = perceive.confidence_of(
            ball_blob, .05, "ball", object(), np.ones((1, 1)))
        centre_confidence = perceive.confidence_of(
            ball_blob, .60, "ball", object(), np.ones((1, 1)))
        check(centre_confidence > edge_confidence,
              "unclipped central detections receive more confidence than edge detections")
    with mock.patch.object(perceive, "depth_top_fraction", return_value=.70):
        triangle_blob = blob("triangle", .92, .87)
        hexagon_blob = blob("hexagon", .996, .84)
        check(perceive.robust_kind(triangle_blob, camera=object(), depth=np.ones((1, 1))) == "triangular_prism",
              "third-harmonic contour evidence classifies a triangular prism")
        triangle_confidence = perceive.confidence_of(triangle_blob, .3, "triangular_prism")
        hexagon_confidence = perceive.confidence_of(hexagon_blob, .3, "hexagonal_prism")
        check(triangle_confidence >= .75 and hexagon_confidence >= .75,
              "triangular and hexagonal confidence uses their matching contour harmonics "
              f"({triangle_confidence}, {hexagon_confidence})")
        check(perceive.robust_kind(blob("square", .97, .87), camera=object(), depth=np.ones((1, 1))) == "cube",
              "fourth-harmonic contour evidence classifies a cube")
        check(perceive.robust_kind(blob("hexagon", .996, .84), camera=object(), depth=np.ones((1, 1))) == "hexagonal_prism",
              "sixth-harmonic contour evidence classifies a hexagonal prism")
        cylinder_blob = blob("circle", .999, .98)
        cylinder_confidence = perceive.confidence_of(
            cylinder_blob, .3, "cylinder", object(), np.ones((1, 1)))
        check(cylinder_confidence >= .75,
              "cylinder confidence uses the flat-cap depth evidence")
        with mock.patch.object(perceive, "contour_harmonics",
                               return_value=(.0060, .0077, .0088)):
            check(perceive.robust_kind(blob("hexagon", .9979, .87), camera=object(),
                                       depth=np.ones((1, 1))) == "hexagonal_prism",
                  "sixth-harmonic dominance recovers the measured low-resolution top-camera hexagon")
        with mock.patch.object(perceive, "contour_harmonics",
                               return_value=(.0070, .0161, .0086)):
            check(perceive.robust_kind(blob("circle", .9981, .87), camera=object(),
                                       depth=np.ones((1, 1))) == "cylinder",
                  "a measured cylinder with stronger fourth than sixth harmonic stays cylindrical")
    with mock.patch.object(perceive, "depth_top_fraction", return_value=None):
        check(perceive.robust_kind(blob("triangle", .92, .87), camera=object()) == "triangular_prism",
              "RGB-only third-harmonic evidence identifies a triangular prism")
        check(perceive.robust_kind(blob("square", .97, .87), camera=object()) == "cube",
              "RGB-only fourth-harmonic evidence identifies a cube")
        check(perceive.robust_kind(blob("hexagon", .996, .84), camera=object()) == "hexagonal_prism",
              "RGB-only sixth-harmonic evidence identifies a hexagonal prism")
        check(perceive.robust_kind(blob("circle", .999, .98)) == "ball",
              "a round RGB silhouette in the calibrated spherical range remains a ball")
        check(perceive.robust_kind(blob("circle", .999, .84), camera=object()) == "any",
              "a depthless round cylinder-like silhouette is left ambiguous instead of called a ball")


def test_depth_top_fraction_matches_full_camera_projection():
    """The scalar world-Z projection must preserve the previous 3-D ray result."""
    from types import SimpleNamespace
    import numpy as np
    import perceive

    size = 12
    mask = np.ones((size, size), dtype=bool)
    blob = perceive.Blob("red", size * size, [0, 0, size, size],
                         [size / 2, size / 2], [0, 255, 200], 1.0, .99, mask, 0)
    ax, ay = .31, -.22
    rx = np.array([[1., 0., 0.], [0., np.cos(ax), -np.sin(ax)],
                   [0., np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0., np.sin(ay)], [0., 1., 0.],
                   [-np.sin(ay), 0., np.cos(ay)]])
    rotation = rx @ ry
    camera = SimpleNamespace(
        k=np.array([[110., 0., 5.5], [0., 105., 5.5], [0., 0., 1.]]),
        rotation=rotation,
        origin=np.array([.12, -.31, .72]),
    )
    depth = .45 + np.arange(size * size, dtype=np.float64).reshape(size, size) * .0007
    ys, xs = np.indices(depth.shape)
    rays = np.stack((np.ones_like(xs), -(xs - 5.5) / 110.,
                     -(ys - 5.5) / 105.), axis=-1)
    expected_world = (rotation @ (rays.reshape(-1, 3) * depth.reshape(-1, 1)).T).T
    expected_world += camera.origin
    expected = perceive.surface_top_fraction(expected_world[:, 2])
    actual = perceive.depth_top_fraction(blob, camera, depth)
    check(np.isclose(actual, expected, atol=1e-12),
          "optimized depth top-profile projection matches full world-coordinate projection")


def test_capacity_reserves_slots_and_refuses_overfill():
    import capacity
    import scene_config
    from types import SimpleNamespace
    from unittest import mock
    # The default scene, not whatever the dashboard last saved: this test names the
    # prims of that scene.
    with mock.patch.object(scene_config, "SAVED_SCENE",
                           Path("no-existe-scene-config.json")):
        cfg, items = scene_config.load()
    tasks = [SimpleNamespace(pick=1, place=3), SimpleNamespace(pick=2, place=3)]
    premise = {1: "cube", 2: "ball_red", 3: "box_red"}
    check(len(capacity.allocate(tasks[:1], premise, items)) == 1, "one part fits the default cavity")
    try:
        capacity.allocate(tasks, premise, items)
    except ValueError:
        check(True, "two parts are refused before motion when the cavity is full")
    else:
        check(False, "overfilled cavity was accepted")
    tray = next(i for i in items if i.prim == "box_red")
    tray.inner = (.12, .08)
    slots = capacity.allocate(tasks, premise, items)
    check(slots["cube"][1] != slots["ball_red"][1], "larger cavities reserve distinct placements")
    with mock.patch.object(scene_config, "SAVED_SCENE",
                           Path("no-existe-scene-config.json")):
        fresh = scene_config.load()[1]
    try:
        capacity.allocate(tasks[1:], premise, fresh, {"cube": ("box_red", (0, 0))})
    except ValueError:
        check(True, "previous verified placements reserve space across commands")
    else:
        check(False, "occupied cavity was reused")


def test_frame_encoder_publishes_and_closes():
    import numpy as np
    from frame_encoder import FrameEncoder
    bus = telemetry.Bus(); errors = []
    encoder = FrameEncoder(bus, errors.append)
    encoder.offer(np.zeros((40, 60, 3), dtype=np.uint8), {"captured_at": 1}, 75)
    encoder.close()
    check(bus.frame()[1] is not None and bus.frame()[2]["captured_at"] == 1, "encoder drains the latest frame at shutdown")
    check(not errors and not encoder.thread.is_alive(), "encoder worker closes cleanly")


def test_forwarding_does_not_mark_queued_commands_as_running():
    bus = telemetry.Bus(); cid = bus.submit("coge el cubo azul")
    bus.take_command(start=False)
    check(bus.command_state(cid)["state"] == telemetry.QUEUED, "IPC forwarding keeps the text queued until robot consumption")
    cid = bus.send_control("relook"); bus.take_control(start=False)
    check(bus.command_state(cid)["state"] == telemetry.QUEUED, "IPC forwarding keeps controls queued too")
    stop = bus.request_stop(); bus.fail_active("La sesión terminó."); bus.complete_stop()
    check(bus.command_state(stop)["state"] == telemetry.SUCCESS, "actual process termination acknowledges stop")
    for text in ("para", "stop", "gelditu"):
        bus = telemetry.Bus(); pending = bus.submit("coge el cubo azul"); stop = bus.submit(text)
        check(bus.stop_requested() and bus.command_state(stop)["type"] == "stop", "typed stop bypasses the motion queue")
        check(bus.command_state(pending)["state"] == telemetry.CANCELLED, "stop cancels queued text")


def test_catalog_prevents_occluded_parts_and_hidden_box_ambiguity():
    module = load_demo11()
    import scene_config
    from unittest import mock
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    # The four-part scene, which is the one this test is about: it has a red ball AND a red
    # cube, so hiding one of them is a real ambiguity. (A dashboard session may have left a
    # two-cube file on disk, and that scene has no red ball to occlude.)
    with mock.patch.object(scene_config, "SAVED_SCENE",
                           Path("no-existe-scene-config.json")):
        _, robot.layout = scene_config.load()
    robot._placed_slots = {}
    premise = {index:item.prim for index,item in enumerate(robot.layout) if item.prim != "ball_red"}
    check(bool(robot._catalog_gaps({"intent":"sort","items":[]},premise)), "an occluded piece cannot disappear from all")
    check(not robot._catalog_gaps({"intent":"pick","items":[{"kind":"cube","color":"blue"}]},premise), "unrelated occlusions do not block a fully resolved reference")
    from dataclasses import replace
    tray = next(item for item in robot.layout if item.prim == "box_red")
    robot.layout.append(replace(tray,prim="other_red_box"))
    command={"intent":"pick_and_place","items":[{"kind":"cube","color":"blue"}],"place":{"color":"red","kind":"box"}}
    check(bool(robot._catalog_gaps(command,premise)), "hidden same-colour boxes remain ambiguous")
    command["place"]["entity_id"]="box_red"
    check(not robot._catalog_gaps(command,premise), "named destinations disambiguate the catalog")


def test_dashboard_shows_clarification_and_preserved_session_report():
    import scene_config
    cfg, parts = scene_config.load(); bus = telemetry.Bus()
    bus.set_state(phase="stopped",catalog=scene_config.catalog(cfg,parts),
                  command_result={"status":"NEEDS_CLARIFICATION","reason":"Indica el recipiente."},
                  report={"state":{"status":"PARTIAL_SUCCESS","verified":1}})
    report=run_dashboard_client([[bus.snapshot()]])
    check(report and not report["loadError"] and not report["warnings"], "new result panels render without script errors")
    if report and not report["loadError"]:
        check("aclaración" in report["resultCommand"] and "Indica" in report["resultCommand"], "clarification is visible in the result card")
        check("parcialmente" in report["resultSession"], "a previous session result remains visible after stop")
        check(bool(report["examples"]), "examples derive from the live catalog")



def test_simple_preset_uses_uniform_custom_cubes():
    import scene_config
    cfg, parts = scene_config.build(scene_config.simple_config())
    check({o["id"] for o in cfg["objects"]} == {"cube_red", "cube_blue"}, "simple cubes both use calibrated spawned material")
    check(len(parts) == 4 and all(o["kind"] == "cube" for o in cfg["objects"]), "simple preset has two cubes and two containers")
    cfg["containers"][0]["name"] = "Destino Ágil"
    cfg["containers"][1]["name"] = "Destino Agil"
    try:
        scene_config.build(cfg)
    except ValueError:
        check(True, "accent-equivalent names are ambiguous and rejected")
    else:
        check(False, "accent-equivalent names were accepted")


def test_release_height_is_resolved_per_container():
    import planner, scene_config, real_world
    from dataclasses import replace
    module = load_demo11()
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    _, robot.layout = scene_config.load()
    low = next(t for t in robot.layout if t.prim == "box_red")
    low = replace(low, size=(low.size[0], low.size[1], .03))
    robot.layout = [low if t.prim == low.prim else t for t in robot.layout]
    high = next(t for t in robot.layout if t.prim == "tray_blue")
    robot.release_mode = "clear"; robot.bus = telemetry.Bus()
    robot.say = lambda *args: None
    plan = planner.Plan(intent="sort", steps=[planner.Step("release", (.3, -.45, .01), target=1),
                                             planner.Step("release", (.4, -.45, .01), target=2)])
    adjusted = robot._release_adjusted(plan, {1:low.prim, 2:high.prim})
    check(adjusted.steps[0].pose[2] == real_world.safe_release_z(low), "low container gets its own release height")
    check(adjusted.steps[1].pose[2] == real_world.safe_release_z(high), "high container gets its own release height")


def test_verification_measures_after_parking():
    from types import SimpleNamespace
    module = load_demo11()
    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    order = []
    robot.bot = SimpleNamespace(settle=lambda:order.append("settle"))
    robot._placement_verdicts = lambda *args: order.append("verify") or [("task", True, "cube")]
    robot.park = lambda:order.append("park")
    robot.ledger = module.TaskLedger()
    robot.recovered = 0
    robot._task_label = lambda *args:"cubo rojo"
    robot._settle_rows = lambda:None
    robot.look = lambda *args:SimpleNamespace(objects=[], containers=[])
    plan = SimpleNamespace(picks=[1])
    robot._verify_and_retry(plan, {}, announce=False)
    check(order == ["settle", "verify", "park", "settle", "verify"], "final placement verdict follows the final arm motion")


def test_handle_finishes_failure_and_diagram_without_motion():
    from types import SimpleNamespace
    module = load_demo11()
    for parsing, expected in (({"errors":["Indica el destino."],"intent":"pick_and_place"}, telemetry.NEEDS_CLARIFICATION),
                              ({"intent":"greeting"}, telemetry.SUCCESS),
                              (RuntimeError("modelo no disponible"), telemetry.FAILED)):
        robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
        robot.bus = telemetry.Bus(); robot.bot = SimpleNamespace(impossible=set(),failures=set())
        robot._task_rows = {}; robot.demo10 = SimpleNamespace(QUIT_WORDS=set())
        def parse(text):
            if isinstance(parsing, Exception): raise parsing
            return parsing
        robot.model = SimpleNamespace(parse=parse)
        robot.handle("comando de prueba")
        snap = robot.bus.snapshot()
        check(snap["commands"][-1]["state"] == expected, "parser errors and nonmoving intents have honest terminal states")
        check(not any(n["state"] == telemetry.ACTIVE for n in snap["nodes"]), "completed commands leave no stale active stage")

def test_every_example_button_is_a_command_the_scene_can_carry_out():
    """The example buttons are built from the scene being run, so each one names pieces and boxes
    that exist in it. Every command a button sends must read without errors into an action the robot
    knows, and for the demo scene each one must plan motion without asking anything.

    What is covered: the rule parser, which is what reads these sentences without the GPU model, and
    the planner, against a recorded scene that has the demo's two cubes and two boxes (its balls are
    removed so the demo's colour sort is the same problem). What is not covered here: the model
    itself, which takes the sentences the rules do not read.
    """
    import scene_config
    import intent_parser
    import planner
    from scene import Scene
    demo_commands, demo_catalog = [], None
    for preset in ("demo", "default"):
        canonical, parts = scene_config.build(scene_config.PRESETS[preset]())
        catalog = scene_config.catalog(canonical, parts)
        bus = telemetry.Bus()
        bus.set_state(catalog=catalog)
        report = dashboard_client_report([[bus.snapshot()]])
        if report is None:
            print("  (SKIPPED: node is not installed, so the example buttons were never built)")
            return
        commands = [text for text in report.get("exampleCommands", []) if text]
        check(0 < len(commands) <= 12, f"{preset}: one to twelve examples, got {len(commands)}")
        parser = intent_parser.HybridParser(fallback=None, catalog=catalog)
        for text in commands:
            command = parser.parse(text)
            check(command["intent"] in ("pick", "pick_and_place", "sort") and not command.get("errors"),
                  f"{preset}: '{text}' must read as an action without errors, got "
                  f"{command['intent']} {command.get('errors')}")
        if preset == "demo":
            demo_commands, demo_catalog = commands, catalog
    recorded = Scene.read(ROOT / "tests" / "fixtures" / "trial001.json")
    recorded.objects = [thing for thing in recorded.objects if thing.kind == "cube"]
    # The recording has its boxes at y -0.51..-0.52, nearer the base than the arm can lower a part
    # from (``scene_spec.MIN_PLACE_REACH_M``). Bring them out to 0.19 m keeping their x, as the
    # shipped demo scene now does; this test is about what the commands plan, not about reach.
    for tray in recorded.containers:
        lateral = min(abs(tray.pos_m[0] - scene_spec.ROBOT_XY[0]), 0.18)
        y = scene_spec.ROBOT_XY[1] + (0.19 ** 2 - lateral ** 2) ** 0.5
        if y > tray.pos_m[1]:
            tray.pos_m = [tray.pos_m[0], y, tray.pos_m[2]]
            tray.aim_xy = [tray.pos_m[0], y]
    for text in demo_commands:
        command = intent_parser.HybridParser(fallback=None, catalog=demo_catalog).parse(text)
        plan = planner.plan_command(command, recorded)
        check(plan.moves and not plan.question and not plan.unresolved,
              f"the demo's '{text}' must plan motion without asking: {plan.unresolved} {plan.question}")


def main() -> int:
    # Demo 11 is imported once, here, before any test runs. Three tests need its classes, and each
    # one that went through `load_demo11` first would otherwise import it again - ~500 ms, almost
    # all of it importing torch and gliner2 through the command model at the top of the file.
    # `load_demo11` caches, so this is the same work done once and in the right order.
    started = time.perf_counter()
    load_demo11()
    warmup = time.perf_counter() - started

    slowest = []
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            began = time.perf_counter()
            function()
            slowest.append((time.perf_counter() - began, name))
    slowest.sort(reverse=True)

    if "-v" in sys.argv or "--timing" in sys.argv:
        print(f"import of demo 11: {warmup * 1000:.0f} ms")
        for seconds, name in slowest[:8]:
            print(f"  {seconds * 1000:7.0f} ms  {name}")
    print(f"{checks} checks over the demo 11 real-world model, the bus and the dashboard")
    if failures:
        print(f"{len(failures)} FAILED:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("RESULT: SUCCESS")
    return 0
def test_the_current_action_is_said_once_and_the_objective_is_the_typed_order():
    """What the arm is doing now is said by the camera line, and nowhere else: the status bar says
    the phase, and the rail's acting stage says the task's position. The objective is the order as
    it was typed; what the robot made of it is in "lo que entendió el robot", not repeated there.
    """
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")

    def body(name):
        start = page.index(f"function {name}(")
        return page[start:page.index("\n}\n", start)]

    demo = body("paintDemo")
    check("ACTION_ES" not in demo and "Ejecutando la orden" in demo,
          "the status bar says the phase, not the action the arm is taking")
    stage = body("stageNote")
    check("ACTION_ES" not in stage and "tarea ${at + 1} de ${rows.length}" in stage,
          "the rail's acting stage says which task is in progress, not the action")
    hero = body("paintHero")
    check("classification" not in hero, "the objective does not take the robot's sentence")
    check("Orden: " in hero, "the objective shows the order as it was typed")


def test_an_example_fills_the_command_box_and_waits_for_enviar():
    """Choosing an example puts its command in the box and leaves it there: the robot acts only when
    the operator presses enviar, so a stray click on an example cannot start a motion."""
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    start = page.index("button.onclick = () => { el(\"cmd\").value = command;")
    line = page[start:page.index("\n", start)]
    check('el("cmd").value = command' in line, "an example writes its command into the box")
    check('el("send")' not in line and ".click()" not in line,
          "and does not send it: enviar is the only thing that sends")


def test_scene_matrix_does_not_report_an_interrupted_campaign_as_successful():
    import importlib.util
    path = ROOT / "tools" / "benchmark_scene_matrix.py"
    spec = importlib.util.spec_from_file_location("demo11_scene_matrix_completion", path)
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    partial = {"total": 4, "expected_total": 18, "failed": 0}
    complete = {"total": 18, "expected_total": 18, "failed": 0}
    failed = {"total": 18, "expected_total": 18, "failed": 1}
    check(matrix.report_exit_code(partial) != 0,
          "a stopped campaign with pending seeds cannot be reported as passing")
    check(matrix.report_exit_code(complete) == 0
          and matrix.report_exit_code(failed) != 0,
          "only a complete campaign with every trial passing has a zero exit code")


def test_scene_matrix_replays_live_size_priors_for_cluster_perception():
    import importlib.util
    path = ROOT / "tools" / "benchmark_scene_matrix.py"
    spec = importlib.util.spec_from_file_location("demo11_scene_matrix_priors", path)
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    truth = [
        {"id": "cube_red", "color": "red", "container": False,
         "size": [.03, .03, .03], "position": [.380, -.400, .0559]},
        {"id": "ball_red", "color": "red", "container": False,
         "size": [.02, .02, .02], "position": [.300, -.320, .0509]},
        {"id": "tray_red", "color": "red", "container": True,
         "size": [.084, .084, .03], "position": [.465, -.4095, .0559]},
    ]
    priors = matrix.perception_geometry_priors(truth)
    check(priors["smallest_part_m"] == .02
          and priors["largest_part_m"] == .03
          and priors["largest_part_footprint_m2"] == .0009,
          "cluster scoring derives the live small and largest object bounds while excluding containers")
    check(priors["object_bounds_by_color"]["red"] == (.03, .0009),
          "same-color clusters receive the largest configured piece footprint used by live segmentation")
    exact = matrix.perception_work_bounds(truth, {
        "work_area": {"work_x": [.285, .507], "work_y": [-.555, -.305]}})
    legacy = matrix.perception_work_bounds(truth, {})
    check(exact == {"work_x": (.285, .507), "work_y": (-.555, -.305)},
          "new captures replay the exact configured workspace ROI")
    check(legacy["work_y"][1] > -.32 and legacy["work_x"][0] < .30,
          "legacy cluster captures infer an ROI that does not clip valid edge objects")


def test_scene_matrix_builds_each_shape_and_spanish_task_without_isaac():
    """The repeatable simulator matrix must cover every supported shape and parse its task."""
    import scene_config
    import intent_parser
    import scene_spec
    import importlib.util
    import dynamics
    import math
    previous_work_area = scene_spec.WORK_X, scene_spec.WORK_Y

    path = ROOT / "tools" / "benchmark_scene_matrix.py"
    spec = importlib.util.spec_from_file_location("demo11_scene_matrix", path)
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    check(all(matrix.grasp_profile_kind_for(kind, 20) == kind for kind in
              ("cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"))
          and matrix.grasp_profile_kind_for("triangular_prism", 30) == "triangular_prism"
          and matrix.grasp_profile_kind_for("ball", 30) is None
          and matrix.grasp_profile_kind_for("triangular_prism", 25) is None,
          "the physical matrix selects validated target pads amid mixed-scene distractors")
    central_cfg = matrix.scene_config("ball", 0, "cluster", region="central")
    _central_area, central_x, central_y = scene_config.working_area(central_cfg)
    scene_spec.set_work_area(central_x, central_y)
    _central_canonical, central_items = scene_config.build(central_cfg)
    central_target = next(item for item in central_items if not item.container)
    central_tray = next(item for item in central_items if item.container)
    central_reach = scene_spec.reach_facts(central_target.x, central_target.y)
    check(abs(central_target.x - .380) < .001 and abs(central_target.y + .400) < .001
          and central_reach["distance_m"] > scene_spec.MIN_REACH_M + .010
          and math.dist((central_target.x, central_target.y),
                        (central_tray.x, central_tray.y)) > .075,
          "central matrix fixture keeps its target outside the measured dead zone and clear of the tray")
    central_distractors = [item for item in central_items
                           if not item.container and item.prim != central_target.prim]
    dx, dy = central_tray.x - central_target.x, central_tray.y - central_target.y
    carry_length_sq = dx * dx + dy * dy
    carry_clearances = []
    for item in central_distractors:
        t = max(0.0, min(1.0, ((item.x - central_target.x) * dx
                               + (item.y - central_target.y) * dy) / carry_length_sq))
        closest = (central_target.x + t * dx, central_target.y + t * dy)
        carry_clearances.append(math.dist((item.x, item.y), closest))
    central_area = scene_config.working_area(central_cfg)[0]
    check(len(central_distractors) == 2 and min(carry_clearances) > .070
          and central_area["width_mm"] == 222
          and central_area["depth_mm"] == 250,
          "cluster distractors stay outside the source-to-tray corridor and inside the configured work area")
    for kind in matrix.KINDS:
        for layout, expected_loose in (("single", 1), ("cluster", 3)):
            scenario = matrix.scene_config(kind, 1, layout)
            _area, work_x, work_y = scene_config.working_area(scenario)
            scene_spec.set_work_area(work_x, work_y)
            canonical, items = scene_config.build(scenario)
            check(len([item for item in items if not item.container]) == expected_loose,
                  f"{kind}/{layout}: matrix creates the expected loose object set")
            check(len([item for item in items if item.container]) == 1,
                  f"{kind}/{layout}: matrix creates exactly one target container")
            target = next(item for item in items if not item.container)
            tray = next(item for item in items if item.container)
            check(scene_spec.why_unreachable(tray.x, tray.y) is None
                  and (target.color != tray.color
                       or math.dist((target.x, target.y), (tray.x, tray.y)) >= .075),
                  f"{kind}/{layout}: the matrix tray is reachable and visually separated")
            check(real_world.placement_margin(target, tray)["margin_m"] >= .005,
                  f"{kind}/{layout}: the tray leaves the required localization margin")
            catalog = scene_config.catalog(canonical, items)
            command = intent_parser.HybridParser(catalog=catalog).parse(matrix.command_for(kind))
            check(command["intent"] == "pick_and_place" and not command.get("errors"),
                  f"{kind}/{layout}: Spanish matrix task parses as pick and place")
            check(command["items"][0]["kind"] == kind and command["items"][0]["color"] == "blue",
                  f"{kind}/{layout}: parsed task selects the intended object")
            check(command["place"] == {"kind": "box", "color": "red"},
                  f"{kind}/{layout}: parsed task selects the matching red box")
        for size_mm in (20, 30):
            sized, sized_items = scene_config.build(
                matrix.scene_config(kind, 1, "single", size_mm=size_mm))
            target = next(item for item in sized_items if not item.container)
            check(round(target.size[0] * 1000) == size_mm,
                  f"{kind}/{size_mm} mm: matrix preserves the requested object size")
            if kind in ("cylinder", "triangular_prism", "hexagonal_prism"):
                check(round(target.size[2] * 1000) == round(size_mm * .8),
                      f"{kind}/{size_mm} mm: matrix scales prism/cylinder height")
    try:
        matrix.scene_config("ball", 0, size_mm=40)
    except ValueError:
        size_bounds_checked = True
    else:
        size_bounds_checked = False
    check(size_bounds_checked, "matrix refuses sizes outside Demo 11's configured scene limits")
    front_cfg = matrix.scene_config("cylinder", 1, "single", camera="front")
    _area, work_x, work_y = scene_config.working_area(front_cfg)
    scene_spec.set_work_area(work_x, work_y)
    _front_canonical, front_items = scene_config.build(front_cfg)
    front_target = next(item for item in front_items if not item.container)
    front_box = next(item for item in front_items if item.container)
    check(abs(front_target.x - .384) <= .001 and abs(front_target.y + .400) <= .001
          and math.hypot(front_target.x - .35, front_target.y + .64)
          > scene_spec.MIN_REACH_M + .035,
          "camera fixtures keep a larger margin from the inner reach singularity")
    front_cluster = matrix.scene_config("triangular_prism", 1, "cluster", camera="front")
    _area, work_x, work_y = scene_config.working_area(front_cluster)
    scene_spec.set_work_area(work_x, work_y)
    _front_cluster, front_cluster_items = scene_config.build(front_cluster)
    check(len([item for item in front_cluster_items if not item.container]) == 3,
          "the front camera keeps the close same-colour cluster fixture valid")
    for kind in matrix.KINDS:
        for layout in matrix.LAYOUTS:
            forward_cfg = matrix.scene_config(kind, 0, layout, region="forward_pick")
            _area, work_x, work_y = scene_config.working_area(forward_cfg)
            scene_spec.set_work_area(work_x, work_y)
            _forward, forward_items = scene_config.build(forward_cfg)
            forward_target = next(item for item in forward_items if not item.container)
            forward_box = next(item for item in forward_items if item.container)
            expected_forward_x = .370 if layout == "cluster" else .380
            check(abs(forward_target.x - expected_forward_x) <= .001
                  and abs(forward_target.y + .400) <= .001
                  and abs(forward_box.x - .480) <= .001
                  and abs(forward_box.y + .4095) <= .001
                  and scene_spec.why_unreachable(forward_target.x, forward_target.y) is None
                  and scene_spec.why_unreachable(forward_box.x, forward_box.y) is None,
                  f"{kind}/{layout}: forward-pick fixture is reachable and scene-valid")
            check(real_world.placement_margin(forward_target, forward_box)["margin_m"] >= .005,
                  f"{kind}/{layout}: forward-pick receptacle keeps localization margin")
    success_report = {"snapshot": {"state": {
        "phase": "done", "status": "SUCCESS", "completed_tasks": 1,
        "failed_tasks": 0, "verified": 1,
        "real_grasps": 1, "kinematic_grasps": 0}}, "perception": {"passed": True}}
    check(matrix.capture_passed(0, success_report),
          "matrix requires a completed task and verified placement")

    def with_counters(**counters):
        state = {**success_report["snapshot"]["state"]}
        state.pop("real_grasps"), state.pop("kinematic_grasps")
        state.update(counters)
        return {"snapshot": {"state": state}}

    check(not matrix.capture_passed(0, with_counters(real_grasps=1, kinematic_grasps=1)),
          "a verified placement that used a simulator-assisted grasp is not a physical pass")
    check(not matrix.capture_passed(0, with_counters(real_grasps=0, kinematic_grasps=0)),
          "a verified placement with no physical grasp is not a physical pass")
    check(not matrix.capture_passed(0, with_counters()),
          "a report that cannot prove its grasps were physical does not pass")
    check(matrix.capture_passed(0, {**success_report, "perception": {"passed": False}}),
          "a verified task may recover from a missed initial frame through bounded re-observation")
    cluster_scenario = matrix.scene_config("cube", 0, "cluster")
    _area, work_x, work_y = scene_config.working_area(cluster_scenario)
    scene_spec.set_work_area(work_x, work_y)
    cluster_config, cluster_items = scene_config.build(cluster_scenario)
    del cluster_config
    truth = [{"id": item.prim, "kind": item.kind, "color": item.color,
              "container": item.container, "position": [item.x, item.y, 0.0]}
             for item in cluster_items]
    linked = [{"entity_id": item.prim, "kind": item.kind, "color": item.color,
               "container": item.container, "position": [item.x, item.y, 0.0],
               "aim_xy": [item.x, item.y]} for item in cluster_items]
    exact = matrix.score_linked_detections(truth, linked)
    check(exact["passed"] and exact["system"]["matched_ids"] == len(truth),
          "catalog-linked detections pass with exact IDs and sub-25-mm positions")
    green_ids = [item.prim for item in cluster_items if item.color == "green"]
    swapped = [dict(row) for row in linked]
    green_rows = [row for row in swapped if row["entity_id"] in green_ids]
    green_rows[0]["entity_id"], green_rows[1]["entity_id"] = (
        green_rows[1]["entity_id"], green_rows[0]["entity_id"])
    check(not matrix.score_linked_detections(truth, swapped)["passed"],
          "linked scoring rejects swapped identities in a close same-colour pair")
    duplicate = linked + [{**linked[0], "id": 100, "entity_id": "untracked"}]
    check(matrix.score_linked_detections(truth, duplicate)["system"]["false_positives"] == 1,
          "linked scoring counts unmatched detections as false positives")
    check(not matrix.capture_passed(1, success_report),
          "matrix rejects a nonzero simulator process exit")
    check(not matrix.capture_passed(0, {"snapshot": {"state": {
        "phase": "done", "status": "FAILED", "completed_tasks": 0,
        "failed_tasks": 1, "verified": 0}}, "perception": {"passed": True}}),
        "matrix rejects an unverified or failed task")
    ordinary_conflicts = dynamics.conflicts(cluster_items)
    check(any("same colour" in item for item in ordinary_conflicts),
          "cluster fixture actually exercises the normally rejected same-colour proximity")
    randomized = dynamics.randomize(cluster_items, seed=11500, spread=.008,
                                    allow_same_colour_clusters=True)
    check(not dynamics.conflicts(randomized, allow_same_colour_clusters=True),
          "cluster-aware randomization retains physical clearance and reach checks")
    check(any(math.hypot(moved.x - original.x, moved.y - original.y) > .001
              for original, moved in zip(cluster_items, randomized)),
          "seeded cluster layout is actually randomized instead of silently falling back")
    balls = [item for item in randomized if item.kind == "ball"]
    separation = ((balls[0].x - balls[1].x) ** 2 + (balls[0].y - balls[1].y) ** 2) ** .5
    check(separation < dynamics.MIN_SAME_COLOUR_GAP_M,
          "seeded cluster remains inside the color-blob merging range")
    args = load_demo11().build_parser().parse_args(["--allow-same-colour-clusters"])
    check(args.allow_same_colour_clusters,
          "the simulation CLI exposes the explicit close-cluster test mode")
    profile_args = load_demo11().build_parser().parse_args([
        "--grasp-profile-kind", "triangular_prism"])
    check(profile_args.grasp_profile_kind == "triangular_prism",
          "the CLI can select a validated contact profile for one explicitly resolved target geometry")
    close_args = load_demo11().build_parser().parse_args(["--close-command", "0.10"])
    check(close_args.close_command == .10,
          "the CLI accepts a close-angle override for physical-grasp experiments")

    loader_path = ROOT.parent / "01_simulation_basics" / "env_loader.py"
    loader_spec = importlib.util.spec_from_file_location("demo11_env_loader_test", loader_path)
    loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader)
    with tempfile.TemporaryDirectory() as temporary:
        source_app = Path(temporary) / "isaaclab.python.kit"
        source_app.write_text(
            '[dependencies]\n"isaacsim.sensors.camera" = {}\n'
            '"isaacsim.sensors.rtx" = {}\n"omni.physx.tensors" = {}\n',
            encoding="utf-8")
        derived = loader.make_demo11_kit_experience(source_app, Path(temporary) / "cache")
        derived_text = derived.read_text(encoding="utf-8")
        check(derived.name.startswith("isaaclab.python.demo11-")
              and derived.suffix == ".kit" and derived.exists(),
              "Demo 11 derives a loadable IsaacLab app experience")
        check('"isaacsim.sensors.rtx"' not in derived_text,
              "derived app removes the unavailable RTX sensor dependency")
        check('"isaacsim.sensors.camera"' in derived_text
              and '"omni.physx.tensors"' in derived_text,
              "derived app preserves camera and physics dependencies")
        check('"omni.replicator.core" = {}' in derived_text
              and '"omni.syntheticdata" = {}' in derived_text
              and "cameras_enabled = true" in derived_text,
              "derived app retains the headless rendering prerequisites for Demo 11's camera")
        with patch.object(loader.sys, "platform", "win32"):
            windows_app = Path(temporary) / "windows" / "isaaclab.python.kit"
            windows_app.parent.mkdir()
            windows_app.write_text(
                '[settings.app]\nvulkan = true\n[dependencies]\n'
                '"isaacsim.sensors.rtx" = {}\n', encoding="utf-8")
            windows_derived = loader.make_demo11_kit_experience(
                windows_app, Path(temporary) / "windows-cache")
            windows_text = windows_derived.read_text(encoding="utf-8")
            check("vulkan = false" in windows_text and "vulkan = true" not in windows_text,
                  "derived Windows app selects Direct3D instead of Vulkan")
        flags = loader.disable_unused_sensor_extensions().split()
        check(flags == [
            "--/app/extensions/excluded/0=isaacsim.sensors.rtx",
            "--/app/extensions/excluded/1=omni.sensors.nv.lidar",
            "--/app/extensions/excluded/2=omni.sensors.nv.radar",
            "--/app/extensions/excluded/3=omni.sensors.nv.ultrasonic",
        ], "extension exclusions use Kit's indexed array syntax")

        lab_root = Path(temporary) / "local_lab"
        app_dir = lab_root / "apps"
        source_dir = lab_root / "source"
        app_dir.mkdir(parents=True)
        local_app = app_dir / "isaaclab.python.kit"
        local_app.write_text(
            '[dependencies]\n"isaaclab" = {order = 1000}\n'
            '"isaaclab_assets" = {order = 1000}\n'
            '"isaaclab_tasks" = {order = 1000}\n'
            '"isaaclab_mimic" = {order = 1000}\n'
            '"isaaclab_rl" = {order = 1000}\n'
            '"isaacsim.sensors.rtx" = {}\n'
            '[settings.app.exts]\nfolders = ["${app}/../source"]\n', encoding="utf-8")
        local_versions = {
            "isaaclab": "0.47.2", "isaaclab_assets": "0.2.3",
            "isaaclab_tasks": "0.11.6", "isaaclab_mimic": "1.0.15",
            "isaaclab_rl": "0.4.4",
        }
        for extension, version in local_versions.items():
            manifest = source_dir / extension / "config" / "extension.toml"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(f'[package]\nversion = "{version}"\n', encoding="utf-8")
        pinned = loader.make_demo11_kit_experience(local_app).read_text(encoding="utf-8")
        check(f'"{source_dir.as_posix()}"' in pinned
              and '"${app}/../source"' not in pinned,
              "cached Kit experience retains the real IsaacLab source extension path")
        check(all(f'"{extension}" = {{ version = "{version}", order = 1000}}' in pinned
                  for extension, version in local_versions.items()),
              "derived app pins all IsaacLab dependencies to the adjacent source checkout")
    scene_spec.set_work_area(*previous_work_area)


def test_tracking_rejects_per_frame_and_cumulative_camera_drift():
    spec = importlib.util.spec_from_file_location("demo11_executor_tracking", ROOT / "executor.py")
    executor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = executor
    spec.loader.exec_module(executor)
    origin = (0.4, -0.45)
    assert executor.tracking_correction_is_safe((0.41, -0.45), origin, origin)
    # Each update is under 25 mm, but the accumulated camera walk is not.
    assert not executor.tracking_correction_is_safe((0.43, -0.45), (0.42, -0.45), origin)
    assert not executor.tracking_correction_is_safe((0.44, -0.45), origin, origin)
    assert executor.tracked_target_motion_is_safe((0.409, -0.45), origin,
                                                  during_descent=True)
    assert not executor.tracked_target_motion_is_safe((0.411, -0.45), origin,
                                                       during_descent=True)
    assert executor.tracked_target_motion_is_safe((0.419, -0.45), origin,
                                                  during_descent=False)
    assert executor.close_center_drift_is_safe(origin, (.407, -.45))
    assert not executor.close_center_drift_is_safe(origin, (.409, -.45))
    assert not executor.tool_alignment_needs_correction(.0016, .0031)
    assert executor.tool_alignment_needs_correction(.0032, .0031)
    assert executor.pregrasp_coast_supported("ball", (.030, .030, .030))
    assert executor.pregrasp_coast_supported("cube", (.020, .020, .020))
    assert not executor.pregrasp_coast_supported("cube", (.030, .030, .030))
    assert executor.tracking_cadence_for("ball", 20) == 10
    assert executor.tracking_cadence_for("hexagonal_prism", 20) == 10
    assert executor.tracking_cadence_for("cube", 20) == 10
    assert executor.tracking_steps_for("ball", (.030, .030, .030), "down", 200) == 400
    assert executor.tracking_steps_for("cube", (.030, .030, .030), "down", 200) == 400
    assert executor.tracking_steps_for("ball", (.020, .020, .020), "down", 200) == 200
    assert executor.tracking_steps_for("cube", (.020, .020, .020), "down", 200) == 200
    assert executor.tracking_steps_for("ball", (.030, .030, .030), "above", 120) == 120


if __name__ == "__main__":
    raise SystemExit(main())
