"""Demo 12 -- the physics tests. These run inside real Isaac Sim.

Everything here reads PhysX. There is no mock path, no simulated mode and no way to get a
green run without the gripper actually gripping, because that is the whole point: the
failure this demo was rebuilt to eliminate was a grasp that reported nothing, and only a
simulation can tell you whether it now works.

Run with the Isaac interpreter -- the PATH python has no ``isaaclab``:

    set PYTHONPATH=demos\\07_multi_robot_orchestration
    C:\\Users\\Usuario\\miniconda3\\envs\\hackathon-robotekin\\python.exe -m pytest \\
        demos\\07_multi_robot_orchestration\\demo12\\tests\\test_grasp.py -v

Takes several minutes: Kit boots once and every grasp is a real motion.

**One cell for the whole module.** ``SimulationContext`` is effectively a singleton --
Isaac Lab registers it on construction and will not hand out a second one -- so this cannot
build a fresh cell per test the way an ordinary fixture would. Instead the module builds
one two-robot cell and the tests take disjoint parts from it: the single-robot tests work
on ``part_3`` and ``part_4``, the multi-robot tests on ``part_1`` and ``part_2``. pytest
runs tests in definition order, so those sets do not collide.

Skipped rather than failed when there is no GPU, so the fast tests in
``test_orchestrator.py`` stay usable on a machine without one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_HERE = Path(__file__).resolve()
_DEMOS_ROOT = _HERE.parents[3]


def _kit_available() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device: Isaac Lab needs a GPU"
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        return False, "isaaclab is not importable in this interpreter"
    return True, ""


AVAILABLE, SKIP_REASON = _kit_available()

pytestmark = pytest.mark.skipif(not AVAILABLE, reason=SKIP_REASON)

# Parts reserved for the single-robot tests, and for the multi-robot ones. Disjoint, so a
# test that moves its part cannot invalidate another test's premise.
#
# The single-robot tests use robot_2, not robot_1, and that is not arbitrary. part_3 and
# part_4 sit at x = 0.36, directly in front of robot_2's base at x = 0.36. Robot_1's base
# is at x = 0.20, so reaching across to them puts robot_1's arm through robot_2's -- and
# since arm collision is enabled (Step 2) that is a real collision, not a near miss. With
# robot_1 the single-robot tests failed on tracking; with robot_2 they pass.
SINGLE_ROBOT_ID = "robot_2"
SINGLE_ROBOT_PARTS = ("part_3", "part_4")
MULTI_ROBOT_PARTS = ("part_1", "part_2")


@pytest.fixture(scope="module")
def cell():
    """Boot Kit once and build the one cell the whole module shares."""
    if not AVAILABLE:
        pytest.skip(SKIP_REASON)

    import os

    for path in (str(_DEMOS_ROOT / "07_multi_robot_orchestration"),
                 str(_DEMOS_ROOT / "01_simulation_basics")):
        if path not in sys.path:
            sys.path.insert(0, path)

    import env_loader

    env_loader.prepare_runtime(headless=True)

    from demo12.config import default_scene
    from demo12.isaac_env import IsaacCell

    scene = default_scene(2)
    built = IsaacCell(scene, device="cuda:0", headless=True)
    built.settle(90)
    yield built

    # Deliberately NOT os._exit here, unlike the probe scripts. Forcing an exit from
    # fixture teardown discards pytest's own report *and* forces the process exit code to
    # 0 -- so a run with four failing tests would be reported as a pass. Kit's shutdown is
    # left to do what it does; if it ever becomes the problem, the fix is to fix it rather
    # than to hide the exit code.


@pytest.fixture
def arm1(cell):
    return cell.arm(SINGLE_ROBOT_ID)


def _grasp(cell, part_id: str, bin_id: str):
    from demo12.grasp import GraspProbe

    return GraspProbe(cell, SINGLE_ROBOT_ID).run(part_id, bin_id)


# ---------------------------------------------------------------------------
# The four acceptance criteria
# ---------------------------------------------------------------------------


def test_grasp_and_place_all_four_criteria_pass(cell):
    """The gate: tracking, span, grip, and lift-and-release, all measured.

    A part that is lifted and then lands back on the table passes three of these and is not
    a grasp, which is why criterion four reads the part's final physical state rather than
    the force during the lift.
    """
    result = _grasp(cell, SINGLE_ROBOT_PARTS[0], "bin_1")
    m = result.measurements

    assert m["tool_error_m"] < 0.001, f"tracking {m['tool_error_m'] * 1000:.2f} mm"
    assert m["span_free_open_m"] >= 0.036, f"open {m['span_free_open_m'] * 1000:.1f} mm"
    assert m["span_free_shut_m"] < 0.018, f"shut {m['span_free_shut_m'] * 1000:.1f} mm"
    assert m["grip_force_n"] > 0.0, "no force between the pads"

    # The pads must have stalled on the part's faces, not on a corner.
    assert abs(m["span_on_part_m"] - m["part_grasp_width_m"]) <= 0.002, (
        f"span on part {m['span_on_part_m'] * 1000:.1f} mm is not the part's "
        f"{m['part_grasp_width_m'] * 1000:.1f} mm width")

    assert m["part_rise_m"] >= 0.050, f"part rose only {m['part_rise_m'] * 1000:.1f} mm"
    assert result.criteria["part_in_bin"], f"part ended at {m['part_final']}"
    assert result.criteria["part_resting"]
    assert result.ok, result.failures


def test_a_ball_is_grasped_and_placed_too(cell):
    """A different shape and mass, so the result is not a single-part fluke.

    The 20 mm PET ball is the easiest thing a friction gripper can hold -- a sphere meets
    the pads at a point rather than a face -- so it should be at least as reliable as the
    cube and it exercises a different contact geometry.
    """
    result = _grasp(cell, SINGLE_ROBOT_PARTS[1], "bin_1")

    assert result.measurements["grip_force_n"] > 0.0
    assert result.measurements["part_rise_m"] >= 0.050
    assert result.ok, result.failures


def test_closing_does_not_punt_the_part(cell):
    """The corner-catch regression.

    A square part presented on its diagonal to a narrower gripper gets closed on at a
    corner and punted. The grip sensor still reports a large force, so only the part's
    displacement catches it -- which is why this is a separate test rather than an
    assertion buried in the one above.
    """
    result = _grasp(cell, SINGLE_ROBOT_PARTS[1], "bin_1")

    drift = result.measurements["part_drift_on_close_m"]
    assert drift < 0.010, f"closing moved the part {drift * 1000:.1f} mm"


def test_pad_separation_axis_is_horizontal_at_the_grasp_pose(cell):
    """The regression test for the bug this demo was rebuilt around.

    The pads slide along the jaw's local Z. At a grasp pose that axis is close to
    horizontal, so the pads close sideways and can straddle a part. When they were on local
    Y they closed up and down, pressed on top of the part, and produced no grip -- and it
    read as "the jaws closed", not as a fault.

    The tolerance is derived from the pad geometry rather than picked. If the axis is
    skewed by ``t`` from horizontal, then across a part of width ``w`` the two pads meet its
    faces at heights differing by ``w * tan(t)``. That skew has to stay inside the pad's own
    contact height, or one pad rides over the face and the other does the gripping:

        w * tan(t) < pad_height   ->   t < atan(pad_height / w)

    For a 30 mm part and a 10 mm pad that is 18.4 deg, i.e. a world-Z component below 0.32.

    Measured: 0.9995 at part_1 (axis exactly vertical -- that is the *broken* configuration,
    and this test would catch it by a factor of three), and 0.1017 at part_3, where the jaw
    happens to sit 5.8 deg off. Both are inside the bound; only the first is a fault.
    """
    import math

    import torch
    from isaaclab.utils.math import quat_apply

    from demo12 import materials as M

    part_width = 0.030
    skew_limit = math.atan(M.FINGER_PAD_HEIGHT_M / part_width)
    z_limit = math.sin(skew_limit)

    arm = cell.arm(SINGLE_ROBOT_ID)
    part = cell.part_state(SINGLE_ROBOT_PARTS[0])
    staging = (part.x, part.y, part.z + 0.120)
    for _ in range(500):
        arm.command_joints(arm.solve_ik(staging, 0.078))
        cell.step()
        if arm.tool_error_m(staging) <= 0.003:
            break

    axes = quat_apply(arm.tool_quat[0],
                      torch.tensor([[0.0, 0.0, 1.0]],
                                   dtype=torch.float32, device=cell.device))[0]
    vertical = abs(float(axes[2].item()))

    assert vertical < z_limit, (
        f"pad separation axis has world-Z component {vertical:.4f} "
        f"({math.degrees(math.asin(min(1.0, vertical))):.1f} deg of skew), beyond the "
        f"{math.degrees(skew_limit):.1f} deg the pads can straddle a {part_width * 1000:.0f} "
        f"mm part; the pads are closing on top of it instead of around it")


def test_nothing_between_the_pads_reads_zero_force(cell):
    """The converse of the failure this demo had for a long time.

    A nonzero grip reading that does not correspond to holding anything. Parked well clear
    of every part, the pads must report nothing.
    """
    arm = cell.arm(SINGLE_ROBOT_ID)
    park = (arm.tool_pos[0].item(), arm.tool_pos[1].item(), 0.30)
    for _ in range(300):
        arm.command_joints(arm.solve_ik(park, 0.078))
        cell.step()
        if arm.tool_error_m(park) <= 0.005:
            break

    arm.set_jaws(False)
    for _ in range(80):
        arm.command_joints(arm.solve_ik(park, 0.02))
        cell.step()

    reported = cell.grip_force_n(SINGLE_ROBOT_ID)
    assert reported < 0.05, f"{reported} N reported with nothing between the pads"


# ---------------------------------------------------------------------------
# Multi-robot
# ---------------------------------------------------------------------------


def test_two_robots_complete_two_tasks_from_real_physics(cell):
    """End to end, with two arms working at once.

    Both tasks are verified from the parts' own final state: inside the target bin's
    cavity, at rest, and resting on the tray floor rather than hovering over it. The
    orchestration logic is covered far more cheaply in ``test_orchestrator.py``; what this
    adds is that the parts really moved, which a scripted fake cannot tell you.
    """
    from demo12.config import MoveTask
    from demo12.orchestrator import MultiRobotOrchestrator

    supplied = [
        MoveTask(id="task_1", part_id=MULTI_ROBOT_PARTS[0], bin_id="bin_1",
                 label="cube 1"),
        MoveTask(id="task_2", part_id=MULTI_ROBOT_PARTS[1], bin_id="bin_2",
                 label="cube 2"),
    ]
    orch = MultiRobotOrchestrator(cell, cell.scene, supplied)

    for _ in range(80_000):
        if not orch.work_outstanding():
            break
        orch.step()

    assert orch.all_tasks_completed(), [t.to_dict() for t in orch.tasks.values()]

    for task_id, bin_id, part_id in (("task_1", "bin_1", MULTI_ROBOT_PARTS[0]),
                                     ("task_2", "bin_2", MULTI_ROBOT_PARTS[1])):
        bin_cfg = cell.bins[bin_id]
        part_cfg = cell.part_cfgs[part_id]
        state = cell.part_state(part_id)
        assert bin_cfg.contains_footprint(state.x, state.y,
                                          part_cfg.footprint_radius_m), (
            f"{part_id} at ({state.x:.4f}, {state.y:.4f}) is not inside {bin_id}")
        assert state.resting, f"{part_id} is still moving at {state.lin_speed} m/s"
        expected = bin_cfg.cavity_floor_z_m + part_cfg.size_m[2] / 2.0
        assert abs(state.z - expected) < 0.004, (
            f"{part_id} rests at z={state.z:.4f}, expected {expected:.4f}")

    owners = {t.assigned_robot for t in orch.tasks.values()}
    assert owners == {"robot_1", "robot_2"}, f"one robot did everything: {owners}"


def test_a_faulted_robot_hands_its_task_over(cell):
    """Fault one arm mid-task and the task has to finish on another."""
    from demo12.config import MoveTask
    from demo12.orchestrator import MultiRobotOrchestrator

    supplied = [
        MoveTask(id="task_f1", part_id="part_1", bin_id="bin_1", label="cube 1"),
        MoveTask(id="task_f2", part_id="part_2", bin_id="bin_2", label="cube 2"),
    ]
    orch = MultiRobotOrchestrator(cell, cell.scene, supplied)

    for _ in range(200):
        orch.step()

    busy = [r for r in orch.robots.values() if r.task_id is not None]
    assert busy, "expected a robot to have picked something up by now"

    victim, held = busy[0].id, busy[0].task_id
    result = orch.inject_failure(victim)
    assert result["ok"] and held in result["released"]

    orch.repair_robot(victim)
    for _ in range(80_000):
        if not orch.work_outstanding():
            break
        orch.step()

    task = orch.tasks[held]
    assert task.state == "done", f"{held} ended {task.state}: {task.note}"
    assert victim in task.previous_robots