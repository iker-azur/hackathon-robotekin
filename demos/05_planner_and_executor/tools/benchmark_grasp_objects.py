"""Repeatable physical grasp benchmark for Demo 11's supported loose-object geometries.

Runs the real Demo 11 material configuration and ``StrictExecutor``. It aims from simulator
truth on purpose: this isolates contact, IK, and grip performance from camera-localisation
error. Run the perception-to-grasp benchmark separately with ``probe_grasp.py`` after this
passes.

    python tools/benchmark_grasp_objects.py --headless --repetitions 5
    python tools/benchmark_grasp_objects.py --headless --kinds cube ball --repetitions 10

The output JSON is suitable for comparing changes across commits. A "success" requires a real
friction grasp (never the kinematic fallback), a measured lift, and retention during a short
lateral carry.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep the largest supported size sweep (11 objects) separated along the table's back edge.
PARKING = ((.05, -.62), (.12, -.62), (.19, -.62), (.55, -.62),
          (.62, -.62), (.68, -.62), (.27, -.62), (.46, -.62),
          (.12, -.68), (.26, -.68), (.40, -.68), (.54, -.68), (.68, -.68))

import executor  # noqa: E402
import gripper_contact_pads  # noqa: E402
import isaac_io  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402


def demo11_module():
    spec = importlib.util.spec_from_file_location("demo11_grasp_benchmark", ROOT / "11_real_sorting_with_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cases(kinds: list[str], size_sweep: bool = False):
    """Use the shipped 30 mm baseline per kind; optionally add feasible 20 mm variants."""
    out = [next(i for i in scene_spec.DEFAULT_SCENE if i.prim == scene_spec.SHIPPED_CUBE)]
    # The size sweep creates up to eleven objects (five geometries at two sizes plus
    # the shipped cube). Keep their initial scene poses distinct even though trials later
    # place one target at a time in the measured reachable region.
    for kind in kinds:
        if kind == "cube" and not size_sweep:
            continue  # the shipped USD cube is the 30 mm baseline
        sizes = (20, 30) if size_sweep else (30,)
        for mm in sizes:
            size = mm / 1000
            if kind == "cube":
                item = scene_spec._cube(f"bench_cube_{mm}", "blue", *PARKING[len(out) - 1], size)
            elif kind == "ball":
                item = scene_spec._ball(f"bench_ball_{mm}", "green", *PARKING[len(out) - 1], size)
            elif kind == "cylinder":
                height = {20: .015, 30: .030, 34: .045}[mm]
                item = scene_spec.Item(prim=f"bench_cylinder_{mm}", kind="cylinder",
                                      color="red", x=PARKING[len(out) - 1][0],
                                      y=PARKING[len(out) - 1][1], shape="cylinder",
                                      size=(size, size, height))
            else:
                kind_name = "triangular_prism" if kind == "triangular_prism" else "hexagonal_prism"
                sides = 3 if kind_name == "triangular_prism" else 6
                item = scene_spec.Item(prim=f"bench_{kind_name}_{mm}", kind=kind_name,
                                      color="purple", x=PARKING[len(out) - 1][0],
                                      y=PARKING[len(out) - 1][1], shape=f"prism_{sides}",
                                      size=(size, size, .024 if mm == 30 else .018))
            if real_world.grasp(item).feasible:
                out.append(item)
            else:
                print(f"SKIP {item.prim}: {real_world.grasp(item).reason}", flush=True)
    return out


def contact_diagnostic_target(items: list[scene_spec.Item], kinds: list[str]) -> str:
    """Choose a separately spawned body so filtered contact sensors can inspect it.

    The stock cube is a child of LeIsaac's scene USD, then exposed through a custom
    ``spawn_from_prim_path`` adapter. That adapter does not apply PhysX contact reporters. Prefer
    one of Demo 11's separately spawned rigid objects; if the requested benchmark contains only
    the stock cube, append a known 30 mm probe cube rather than silently configuring a sensor
    that cannot initialize.
    """
    target = next((item for item in items
                   if not item.container and item.kind in kinds
                   and item.prim != scene_spec.SHIPPED_CUBE), None)
    if target is None:
        target = scene_spec._cube("demo11_contact_probe", "blue", .12, -.62, .03)
        items.append(target)
    return target.prim


def set_pose(env, item, x, y, yaw_deg, preserve_settled_pose=False):
    import torch

    half = math.radians(yaw_deg) / 2
    pose = torch.tensor([[x, y, item.z, math.cos(half), 0.0, 0.0, math.sin(half)]],
                        device=env.device)
    zeros = torch.zeros(1, 6, device=env.device)
    robot = env.scene["robot"]
    arm_hold = robot.data.joint_pos.clone()
    actor = env.scene[item.prim]
    actor.write_root_pose_to_sim(pose)
    actor.write_root_velocity_to_sim(zeros)
    # Keep the articulation controller target aligned with the state writes below. Without
    # this, raw sim steps can revive the preceding park/IK target after object placement.
    with torch.inference_mode():
        robot.set_joint_position_target(arm_hold)
    # A short fixed settle mirrors the existing probes and avoids waiting on rolling-damping
    # thresholds that can take several hundred steps for a ball.
    for _ in range(45):
        with torch.inference_mode():
            robot.set_joint_position_target(arm_hold)
            robot.write_joint_position_to_sim(arm_hold)
            robot.write_joint_velocity_to_sim(torch.zeros_like(arm_hold))
        env.sim.step(render=False)
        with torch.inference_mode():
            robot.set_joint_position_target(arm_hold)
            robot.write_joint_position_to_sim(arm_hold)
            robot.write_joint_velocity_to_sim(torch.zeros_like(arm_hold))
    actor.write_root_velocity_to_sim(zeros)
    env.scene.update(dt=env.physics_dt)
    # Reasserting an exactly tangent sphere pose after settling can inject a fresh contact
    # impulse on the next tick. Keep this diagnostic option to distinguish that impulse from
    # persistent table slope / contact drift. Production trials retain the requested pose.
    if not preserve_settled_pose:
        actor.write_root_pose_to_sim(pose)
    actor.write_root_velocity_to_sim(zeros)
    env.scene.update(dt=env.physics_dt)


def park_all(env, items):
    import torch

    # Keep non-targets on the back edge of the table, spaced apart and outside the measured pick
    # area. Set all poses before advancing physics, so starting overlaps cannot impart impulses.
    zeros = torch.zeros(1, 6, device=env.device)
    for i, item in enumerate(items):
        x, y = PARKING[i]
        half = math.radians(item.yaw_deg) / 2
        pose = torch.tensor([[x, y, item.z, math.cos(half), 0.0, 0.0, math.sin(half)]],
                            device=env.device)
        actor = env.scene[item.prim]
        actor.write_root_pose_to_sim(pose)
        actor.write_root_velocity_to_sim(zeros)
    for _ in range(45):
        env.sim.step(render=False)
    for item in items:
        env.scene[item.prim].write_root_velocity_to_sim(zeros)
    env.scene.update(dt=env.physics_dt)


def truth(env, prim):
    p = env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
    return [float(p[0]), float(p[1]), float(p[2])]


def set_gripper_yaw(bot, yaw_deg: float) -> None:
    """Rotate the tool about world Z while retaining the base-relative IK convention."""
    import torch
    from isaaclab.utils.math import quat_mul

    half = math.radians(yaw_deg) / 2.0
    world_yaw = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]],
                             dtype=bot.grip_quat.dtype, device=bot.device)
    bot.grip_quat = quat_mul(bot._inverse, world_yaw)
    hold_x, hold_y = float(bot._hold[0, 0]), float(bot._hold[0, 1])
    bot._hold[0, 0] = hold_x * math.cos(math.radians(yaw_deg)) - hold_y * math.sin(math.radians(yaw_deg))
    bot._hold[0, 1] = hold_x * math.sin(math.radians(yaw_deg)) + hold_y * math.cos(math.radians(yaw_deg))


def run_trial(env, bot, item, rng, index, carry_steps=None, fixed_position=None,
              position_bounds=None,
              approach_clearance_z=None, track_every=None, sensor=None, frames_dir=None,
              hover_clearance=0.09, yaw_deg=None, grasp_z_offset=0.0,
              approach_offset=(0.0, 0.0), pre_approach_dwell_steps=0,
              preserve_settled_pose=False):
    # Keep trials inside the measured stable grasp envelope. Cubes are only tested at
    # axis-aligned yaw: a 30 mm cube at 45 degrees presents a 42.4 mm diagonal to a
    # 36.2 mm gripper and is physically ungraspable without orienting the wrist.
    if fixed_position is not None:
        x, y = fixed_position
    else:
        x_bounds, y_bounds = position_bounds or ((.32, .42), (-.46, -.43))
        x, y = rng.uniform(*x_bounds), rng.uniform(*y_bounds)
    yaw = rng.choice((0.0, 90.0)) if item.kind == "cube" else 0.0
    if yaw_deg is not None:
        yaw = float(yaw_deg)
    bot.drop()
    # Reach the camera-safe park pose, then freeze the measured arm state while the target is
    # placed. This prevents residual IK motion from sweeping the newly placed object.
    park_motion = bot.park(steps=executor.PARK_STEPS)
    isaac_io.hold_arm_pose(env)
    bot.current = bot._gripper_pos().clone()
    park_height = float(bot.current[0, 2])
    park_settle = executor.StepResult(
        action="park", ok=park_motion.ok and park_height >= 0.12,
        note=f"arm held at measured park pose ({park_height:.3f} m)")
    if not park_settle.ok:
        raise RuntimeError(f"Arm not clear before trial {index}: {park_settle.note}")
    robot = env.scene["robot"]
    joint_positions = robot.data.joint_pos[0].detach().cpu().numpy()
    joint_velocities = robot.data.joint_vel[0].detach().cpu().numpy()
    gripper_body = robot.data.body_pos_w[0, bot.gripper_id].detach().cpu().numpy()
    aperture = bot.aperture_centre()
    robot_start_state = {
        "joint_pos_rad": {str(name): round(float(joint_positions[i]), 7)
                           for i, name in enumerate(robot.joint_names)},
        "joint_vel_rad_s": {str(name): round(float(joint_velocities[i]), 7)
                             for i, name in enumerate(robot.joint_names)},
        "gripper_body_pos_w_m": [round(float(value), 7) for value in gripper_body],
        "aperture_centre_w_m": ([round(float(value), 7) for value in aperture]
                                if aperture is not None else None),
    }
    set_pose(env, item, x, y, yaw, preserve_settled_pose=preserve_settled_pose)
    start = truth(env, item.prim)
    actor = env.scene[item.prim]
    start_quat = actor.data.root_quat_w[0].detach().cpu().numpy()
    start_linear_velocity = actor.data.root_lin_vel_w[0].detach().cpu().numpy()
    start_angular_velocity = actor.data.root_ang_vel_w[0].detach().cpu().numpy()
    object_start_state = {
        "position_m": [round(float(value), 7) for value in start],
        "quat_wxyz": [round(float(value), 7) for value in start_quat],
        "linear_velocity_m_s": [round(float(value), 7) for value in start_linear_velocity],
        "angular_velocity_rad_s": [round(float(value), 7) for value in start_angular_velocity],
    }
    phase_trace = []
    active_phase = {"name": None, "samples": []}
    step_count = {"value": 0}

    def sample_phase(force=False):
        if active_phase["name"] is None:
            return
        step_count["value"] += 1
        if not force and step_count["value"] % 5:
            return
        position = truth(env, item.prim)
        sample = {
            "step": step_count["value"],
            "position_m": [round(value, 5) for value in position],
            "displacement_from_start_mm": round(math.dist(position, start) * 1000, 2),
        }
        try:
            actor = env.scene[item.prim]
            linear = actor.data.root_lin_vel_w[0].detach().cpu().numpy()
            angular = actor.data.root_ang_vel_w[0].detach().cpu().numpy()
            orientation = actor.data.root_quat_w[0].detach().cpu().numpy()
            sample["linear_velocity_m_s"] = [round(float(value), 5) for value in linear]
            sample["angular_velocity_rad_s"] = [round(float(value), 5) for value in angular]
            sample["orientation_wxyz"] = [round(float(value), 7) for value in orientation]
        except Exception:  # noqa: BLE001 - diagnostic fields must not break a trial
            pass
        try:
            robot = env.scene["robot"]
            body_positions = robot.data.body_pos_w[0].detach().cpu().numpy()
            body_quaternions = robot.data.body_quat_w[0].detach().cpu().numpy()
            body_linear_velocities = robot.data.body_lin_vel_w[0].detach().cpu().numpy()
            body_angular_velocities = robot.data.body_ang_vel_w[0].detach().cpu().numpy()
            distances = np.linalg.norm(body_positions - np.asarray(position), axis=1)
            nearest = int(np.argmin(distances))
            sample["nearest_link"] = str(robot.body_names[nearest])
            sample["nearest_link_origin_gap_mm"] = round(float(distances[nearest]) * 1000, 2)
            # Record the measured gripper frames during contact experiments. Their local
            # transforms are needed to place optional experimental pad colliders correctly;
            # world-space body origins alone are not enough to infer a jaw's contact face.
            link_poses = {}
            for name in ("gripper", "jaw"):
                matches = [index for index, body_name in enumerate(robot.body_names)
                           if str(body_name).lower() == name]
                if matches:
                    body_id = matches[0]
                    link_poses[name] = {
                        "position_m": [round(float(value), 6)
                                       for value in body_positions[body_id]],
                        "quat_wxyz": [round(float(value), 7)
                                      for value in body_quaternions[body_id]],
                        "linear_velocity_m_s": [round(float(value), 6)
                                                for value in body_linear_velocities[body_id]],
                        "angular_velocity_rad_s": [round(float(value), 6)
                                                   for value in body_angular_velocities[body_id]],
                    }
            if link_poses:
                sample["gripper_link_poses_w"] = link_poses
            joint_id = int(robot.find_joints("gripper")[0][0])
            joint_state = {}
            for key, output_key in (("joint_pos", "position_rad"),
                                    ("joint_vel", "velocity_rad_s"),
                                    ("joint_pos_target", "target_rad"),
                                    ("applied_torque", "applied_torque_nm")):
                value = getattr(robot.data, key, None)
                if value is not None:
                    joint_state[output_key] = round(
                        float(value[0, joint_id].detach().cpu()), 7)
            sample["gripper_joint_state"] = joint_state
        except Exception:  # noqa: BLE001 - diagnostic fields must not break a trial
            pass
        try:
            aperture = bot.aperture_centre()
            if aperture is not None:
                sample["aperture_m"] = [round(float(value), 5) for value in aperture]
        except Exception:  # noqa: BLE001 - diagnostic fields must not break a trial
            pass
        for link in ("gripper", "jaw", "target", "shoulder", "upper_arm",
                     "lower_arm", "wrist"):
            try:
                force_matrix = env.scene[f"demo11_contact_{link}"].data.force_matrix_w
                if force_matrix is not None:
                    forces = force_matrix[0].reshape(-1, 3).detach().cpu().numpy()
                    sample[f"contact_{link}_force_n"] = round(
                        float(np.linalg.norm(forces, axis=1).sum()), 5)
                net_forces = env.scene[f"demo11_contact_{link}"].data.net_forces_w
                if net_forces is not None:
                    values = net_forces[0].reshape(-1, 3).detach().cpu().numpy()
                    sample[f"contact_{link}_net_force_n"] = round(
                        float(np.linalg.norm(values, axis=1).sum()), 5)
            except Exception:  # noqa: BLE001 - optional contact diagnostics must never break a trial
                pass
        active_phase["samples"].append(sample)

    original_env_step = env.step

    def capture_env_step(*args, **kwargs):
        result = original_env_step(*args, **kwargs)
        sample_phase()
        return result

    env.step = capture_env_step
    stationary_dwell = max(0, int(pre_approach_dwell_steps))
    if stationary_dwell:
        active_phase["name"] = "stationary_hold"
        step_count["value"] = 0
        sample_phase(force=True)
        for _ in range(stationary_dwell):
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            sample_phase()
        sample_phase(force=True)
        phase_trace.append({"phase": "stationary_hold", "samples": active_phase["samples"]})
        active_phase["name"] = None
    # A small, explicit aim offset is useful for testing asymmetric jaw closure paths:
    # the object may be intentionally contacted off-centre so the jaw sweep settles it
    # between the fingers. The benchmark still records the true object pose separately.
    grasp_x, grasp_y = x + float(approach_offset[0]), y + float(approach_offset[1])
    grasp_z = item.z + float(grasp_z_offset)
    above = (grasp_x, grasp_y, max(.131, grasp_z + hover_clearance))
    grasp = (grasp_x, grasp_y, grasp_z)
    carry = (grasp_x + .025, grasp_y, above[2])

    plan = __import__("planner").Plan(intent="pick")
    for action, pose, grip in (
        ("above", above, executor.OPEN),
        ("down", grasp, executor.OPEN),
        ("close", grasp, executor.CLOSE),
        ("lift", above, executor.CLOSE),
        ("carry", carry, executor.CLOSE),
    ):
        plan.steps.append(__import__("planner").Step(action, pose, grip, 0, f"benchmark {action}"))
    plan.picks.append(0)
    # Read live simulator truth on every tracking callback, not the initial pose: a rolling ball
    # must be followed just as a camera-driven demo would follow a fresh observation.
    after_lift = {}
    after_close = {}
    close_joint_state = {}
    observe_trace = []
    original_do = bot.do
    original_track = bot.track

    def begin_phase(action):
        active_phase["name"] = action
        active_phase["samples"] = []
        step_count["value"] = 0
        sample_phase(force=True)

    def finish_phase(action, before):
        sample_phase(force=True)
        positions = [before] + [row["position_m"] for row in active_phase["samples"]]
        max_shift = max((math.dist(position, start) for position in positions), default=0.0)
        phase_trace.append({
            "phase": action,
            "start_pose_m": [round(value, 5) for value in before],
            "end_pose_m": [round(value, 5) for value in truth(env, item.prim)],
            "max_displacement_from_start_mm": round(max_shift * 1000, 2),
            "samples": active_phase["samples"],
        })
        active_phase["name"] = None

    def capture_lift(action, *args, **kwargs):
        if action == "carry" and carry_steps is not None:
            kwargs["steps"] = carry_steps
        before = truth(env, item.prim)
        begin_phase(action)
        result = original_do(action, *args, **kwargs)
        if action == "close":
            # Preserve the actuator state at the instant the grip is judged. This separates
            # insufficient commanded effort from missing/poor contact geometry in strict runs.
            try:
                robot = env.scene["robot"]
                joint_id = int(robot.find_joints("gripper")[0][0])
                for key in ("joint_pos", "joint_vel", "joint_pos_target", "applied_torque",
                            "joint_effort_limits"):
                    value = getattr(robot.data, key, None)
                    if value is not None:
                        close_joint_state[key] = float(value[0, joint_id].detach().cpu())
                close_joint_state["joint_id"] = joint_id
                close_joint_state["command_rad"] = float(
                    env.action_manager.get_term("gripper_action")._close_command[0])
            except Exception:  # diagnostics must not alter the grasp outcome
                pass
        finish_phase(action, before)
        if action == "close":
            after_close["position"] = truth(env, item.prim)
        if action == "lift":
            after_lift["position"] = truth(env, item.prim)
        return result

    def capture_track(observe, *args, **kwargs):
        action = kwargs.get("action", "track")
        before = truth(env, item.prim)
        begin_phase(action)
        try:
            result = original_track(observe, *args, **kwargs)
            if frames_dir is not None and sensor is not None:
                image, _camera = isaac_io.capture_rgb(env, sensor)
                import cv2
                Path(frames_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(frames_dir) / f"{item.prim}_{index}_{action}.png"), image)
            return result
        finally:
            finish_phase(action, before)

    def observe_target(prim):
        position = truth(env, prim)
        aperture = bot.aperture_centre()
        observe_trace.append({
            "phase": active_phase["name"],
            "step": step_count["value"],
            "object_xy_m": [round(position[0], 6), round(position[1], 6)],
            "aperture_xy_m": ([round(float(aperture[0]), 6), round(float(aperture[1]), 6)]
                              if aperture is not None else None),
        })
        return (position[0] + float(approach_offset[0]),
                position[1] + float(approach_offset[1]))

    bot.do = capture_lift
    bot.track = capture_track
    bot.approach_clearance_z = approach_clearance_z
    if track_every is not None:
        bot.track_every_steps = track_every
    try:
        result = bot.run(plan, resolve=lambda _id: item.prim,
                         observe=observe_target)
    finally:
        bot.do = original_do
        bot.track = original_track
        bot.approach_clearance_z = None
        bot.track_every_steps = executor.TRACK_EVERY
        env.step = original_env_step
    end = truth(env, item.prim)
    notes = [step.note for step in result.steps if step.note]
    lifted_mm = (after_lift.get("position", end)[2] - start[2]) * 1000
    # A real grasp must follow the lift, then stay attached as the arm carries 25 mm sideways.
    carried_from = after_close.get("position", start)
    carry_error_mm = math.dist((end[0], end[1]),
                               (carried_from[0] + .025, carried_from[1])) * 1000
    close_trace = next((phase for phase in phase_trace if phase["phase"] == "close"), None)
    contact_summary = None
    diagnostic_links = ("shoulder", "upper_arm", "lower_arm", "wrist",
                        "gripper", "jaw", "target")
    if close_trace is not None and any(
            f"contact_{link}_force_n" in row
            for row in close_trace["samples"] for link in diagnostic_links):
        contact_summary = {}
        for link in diagnostic_links:
            values = [row[f"contact_{link}_force_n"] for row in close_trace["samples"]
                      if f"contact_{link}_force_n" in row]
            net_values = [row[f"contact_{link}_net_force_n"] for row in close_trace["samples"]
                          if f"contact_{link}_net_force_n" in row]
            contact_summary[link] = {
                "samples": len(values),
                "peak_force_n": max(values) if values else None,
                "mean_force_n": round(sum(values) / len(values), 5) if values else None,
                "samples_over_0_05n": sum(value > 0.05 for value in values),
                "peak_net_force_n": max(net_values) if net_values else None,
                "net_samples_over_0_05n": sum(value > 0.05 for value in net_values),
            }
    lift_success = (result.real_grasps == 1 and result.kinematic_grasps == 0
                    and result.lost_grasps == 0 and result.stale_grasps == 0
                    and lifted_mm >= 50)
    simulator_lift_success = ((result.real_grasps + result.kinematic_grasps) == 1
                              and result.lost_grasps == 0 and result.stale_grasps == 0
                              and lifted_mm >= 50)
    retained_height = end[2] >= start[2] + .05
    carry_success = lift_success and retained_height and carry_error_mm <= 15
    simulator_carry_success = (simulator_lift_success and retained_height
                                and carry_error_mm <= 15)
    row = {
        "prim": item.prim, "kind": item.kind, "size_mm": [round(v * 1000, 2) for v in item.size],
        "x_m": round(x, 5), "y_m": round(y, 5), "yaw_deg": yaw, "trial": index,
        "grasp_z_offset_m": float(grasp_z_offset),
        "approach_offset_m": [float(approach_offset[0]), float(approach_offset[1])],
        "approach_clearance_z_m": approach_clearance_z,
        "pre_approach_dwell_steps": stationary_dwell,
        "robot_start_state": robot_start_state,
        "object_start_state": object_start_state,
        "close_joint_state": close_joint_state,
        "close_contact": contact_summary,
        "park_settle": park_settle.note,
        "grasp": "real" if result.real_grasps else ("kinematic" if result.kinematic_grasps else
                  "lost" if result.lost_grasps else
                  "stale" if result.stale_grasps else "none"),
        "lift_success": lift_success, "carry_success": carry_success,
        "simulator_lift_success": simulator_lift_success,
        "simulator_carry_success": simulator_carry_success,
        "success": carry_success, "lifted_mm": round(lifted_mm, 2),
        "carry_displacement_mm": round(math.dist(start[:2], end[:2]) * 1000, 2),
        "carry_target_error_mm": round(carry_error_mm, 2),
        "end_pose_m": [round(value, 5) for value in end],
        "phase_trace": phase_trace,
        "observe_trace": observe_trace,
        "notes": notes,
    }
    # Return to open gripper and let any failed part settle before the next randomized trial.
    bot.do("release", carry, executor.OPEN, steps=30)
    bot.settle(steps=30)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0",
                        help="physics/tensor device; use cpu to avoid sharing PhysX CUDA with "
                             "another Isaac session")
    parser.add_argument("--kinds", nargs="+", choices=("cube", "ball", "cylinder", "triangular_prism", "hexagonal_prism"),
                        default=["cube", "ball", "cylinder"])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--grip-stiffness", type=float, default=real_world.GRIPPER.actuator_stiffness,
                        help="gripper actuator stiffness; experimental sweep, default is production")
    parser.add_argument("--contact-diagnostics", action="store_true",
                        help="record filtered PhysX contact forces from both fingers against the target")
    parser.add_argument("--contact-diagnostics-all-robot-links", action="store_true",
                        help="also filter target contact forces from the arm links during approach")
    parser.add_argument("--experimental-contact-pads", action="store_true",
                        help="A/B-test the local SO-101 contact-pad USD; never enables grasp assistance")
    parser.add_argument("--auto-validated-contact-pads", action="store_true",
                        help="select only a bench-validated profile for the complete compatible scene")
    parser.add_argument("--grasp-profile-kind",
                        help="scope automatic pad selection to this resolved target geometry")
    parser.add_argument("--experimental-contact-pad-radius-mm", type=float, default=5.0,
                        help="radius for the experimental contact-pad USD (1–10 mm)")
    parser.add_argument("--experimental-contact-pad-shape", choices=("sphere", "flat", "vee"),
                        default="sphere", help="experimental pad collider shape")
    parser.add_argument("--experimental-contact-pad-thickness-mm", type=float, default=2.0,
                        help="thickness for flat pads (0.5–5 mm)")
    parser.add_argument("--experimental-contact-pad-angle-deg", type=float,
                        default=gripper_contact_pads.PAD_VEE_ANGLE_DEG,
                        help="angle for the experimental V profile (5–45 degrees)")
    parser.add_argument("--experimental-contact-pad-force-unvalidated", action="store_true",
                        help="force experimental pads for object profiles not validated for that shape")
    parser.add_argument("--keep-open", action="store_true",
                        help="leave the Isaac Sim window and environment open after the benchmark")
    parser.add_argument("--close-command", type=float, default=None,
                        help="explicit gripper close target in radians; overrides object profiles")
    parser.add_argument("--pad-static-friction", type=float,
                        default=real_world.RUBBER_PAD.static_friction,
                        help="experimental pad static friction override")
    parser.add_argument("--pad-dynamic-friction", type=float,
                        default=real_world.RUBBER_PAD.dynamic_friction,
                        help="experimental pad dynamic friction override")
    parser.add_argument("--carry-steps", type=int, default=executor.STEP_COUNTS["carry"],
                        help="carry interpolation duration; default is the production value")
    parser.add_argument("--fixed-position", nargs=2, type=float, metavar=("X", "Y"),
                        help="repeat every trial at the same world position (metres)")
    parser.add_argument("--position-bounds", nargs=4, type=float,
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                        help="sample randomized object poses within this measured workspace box")
    parser.add_argument("--yaw-deg", type=float,
                        help="fix object yaw for a geometry-orientation grasp experiment")
    parser.add_argument("--gripper-yaw-deg", type=float,
                        help="rotate the gripper about world Z for an object-aware grasp experiment")
    parser.add_argument("--grasp-z-offset", type=float, default=0.0,
                        help="shift the grasp hold point vertically from the part centre (m)")
    parser.add_argument("--approach-offset", nargs=2, type=float, default=(0.0, 0.0),
                        metavar=("DX", "DY"),
                        help="experimental XY aim offset from object centre (m); use to measure jaw-sweep compensation")
    parser.add_argument("--approach-clearance-z", type=float,
                        help="test an elevated hold-point approach; disabled by default")
    parser.add_argument("--compare-clearance-z", type=float,
                        help="run paired baseline and elevated-approach trials at identical poses")
    parser.add_argument("--ball-rolling-damping", type=float,
                        help="override ball angular damping for a physics-only experiment")
    parser.add_argument("--ball-static-friction", type=float,
                        help="experimental sphere static-friction override")
    parser.add_argument("--ball-dynamic-friction", type=float,
                        help="experimental sphere dynamic-friction override")
    parser.add_argument("--track-every", type=int,
                        help="experimental target-observation interval in physics steps")
    parser.add_argument("--pre-approach-dwell-steps", type=int, default=0,
                        help="diagnostic: hold the arm fixed after placing the target and record "
                             "object drift before approach")
    parser.add_argument("--preserve-settled-pose", action="store_true",
                        help="diagnostic: keep the object's settled pose instead of teleporting "
                             "it back to an exactly tangent table pose")
    parser.add_argument("--park-pose", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="experimental robot start pose before a pick, in metres")
    parser.add_argument("--size-sweep", action="store_true",
                        help="add 20 mm variants (more simulated rigid bodies and VRAM)")
    parser.add_argument("--only-size-mm", type=int, choices=(20, 30),
                        help="after --size-sweep, keep only this object size for isolated profile tests")
    parser.add_argument("--allow-kinematic", action="store_true",
                        help="allow the simulator-only, contact-gated carry fallback; report it separately from friction grasps")
    parser.add_argument("--auto-sim-assist", action="store_true",
                        help="exercise Demo 11's automatic assist policy for spheres, cylinders, and prisms")
    parser.add_argument("--frames-dir", type=Path,
                        help="save top-camera frames after approach phases for contact diagnosis")
    parser.add_argument("--hover-clearance", type=float, default=0.09,
                        help="height above the part centre for its hover pose, in metres")
    parser.add_argument("--seed", type=int, default=11011)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "grasp_benchmark.json")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least one")
    if args.pre_approach_dwell_steps < 0:
        parser.error("--pre-approach-dwell-steps must be non-negative")
    if args.grip_stiffness <= 0:
        parser.error("--grip-stiffness must be positive")
    if not 1.0 <= args.experimental_contact_pad_radius_mm <= 10.0:
        parser.error("--experimental-contact-pad-radius-mm must be between 1 and 10")
    if not 0.5 <= args.experimental_contact_pad_thickness_mm <= 5.0:
        parser.error("--experimental-contact-pad-thickness-mm must be between 0.5 and 5")
    if not 5.0 <= args.experimental_contact_pad_angle_deg <= 45.0:
        parser.error("--experimental-contact-pad-angle-deg must be between 5 and 45")
    if args.close_command is not None and not 0.0 <= args.close_command <= 1.745:
        parser.error("--close-command must be between 0 and the 1.745 rad joint limit")
    if (args.pad_static_friction <= 0 or args.pad_dynamic_friction <= 0
            or args.pad_dynamic_friction > args.pad_static_friction):
        parser.error("pad friction values must be positive and dynamic must not exceed static")
    if args.approach_clearance_z is not None and args.compare_clearance_z is not None:
        parser.error("choose --approach-clearance-z or --compare-clearance-z, not both")
    if args.ball_rolling_damping is not None and args.ball_rolling_damping < 0:
        parser.error("--ball-rolling-damping must be non-negative")
    ball_static = (real_world.PET_BALL.static_friction if args.ball_static_friction is None
                   else args.ball_static_friction)
    ball_dynamic = (real_world.PET_BALL.dynamic_friction if args.ball_dynamic_friction is None
                    else args.ball_dynamic_friction)
    if (ball_static <= 0 or ball_dynamic <= 0 or ball_dynamic > ball_static):
        parser.error("ball friction values must be positive and dynamic must not exceed static")
    if args.track_every is not None and args.track_every < 1:
        parser.error("--track-every must be at least one")
    if args.fixed_position is not None and args.position_bounds is not None:
        parser.error("choose --fixed-position or --position-bounds, not both")
    if args.position_bounds is not None:
        x_min, x_max, y_min, y_max = args.position_bounds
        if x_min >= x_max or y_min >= y_max:
            parser.error("position bounds must have increasing min/max values")

    import planner

    selected = cases(args.kinds, size_sweep=args.size_sweep)
    if args.only_size_mm is not None:
        if not args.size_sweep:
            parser.error("--only-size-mm requires --size-sweep")
        selected = [item for item in selected
                    if item.prim == scene_spec.SHIPPED_CUBE
                    or round(float(item.size[0]) * 1000) == args.only_size_mm]
    if args.park_pose is not None:
        executor.PARK_POSE = tuple(args.park_pose)
    if args.ball_rolling_damping is not None:
        real_world.PET_BALL = dataclasses.replace(
            real_world.PET_BALL, rolling_damping=args.ball_rolling_damping)
    if (args.ball_static_friction is not None or args.ball_dynamic_friction is not None):
        real_world.PET_BALL = dataclasses.replace(
            real_world.PET_BALL, static_friction=ball_static, dynamic_friction=ball_dynamic)
    if (args.pad_static_friction != real_world.RUBBER_PAD.static_friction
            or args.pad_dynamic_friction != real_world.RUBBER_PAD.dynamic_friction):
        real_world.RUBBER_PAD = dataclasses.replace(
            real_world.RUBBER_PAD,
            static_friction=args.pad_static_friction,
            dynamic_friction=args.pad_dynamic_friction)
    module = demo11_module()
    isaac_io.prepare_runtime(headless=args.headless)
    diagnostics_enabled = (args.contact_diagnostics
                           or args.contact_diagnostics_all_robot_links)
    diagnostic_target = (contact_diagnostic_target(selected, args.kinds)
                         if diagnostics_enabled else None)
    env, _sensor = module.build_real_env(
        "top", selected, grip_stiffness=args.grip_stiffness, device=args.device,
        auto_validated_contact_pads=args.auto_validated_contact_pads,
        contact_diagnostics_for=diagnostic_target if diagnostics_enabled else None,
        contact_diagnostics_all_robot_links=args.contact_diagnostics_all_robot_links,
        experimental_contact_pads=args.experimental_contact_pads,
        experimental_contact_pad_radius_m=args.experimental_contact_pad_radius_mm / 1000.0,
        experimental_contact_pad_shape=args.experimental_contact_pad_shape,
        experimental_contact_pad_thickness_m=args.experimental_contact_pad_thickness_mm / 1000.0,
        experimental_contact_pad_angle_deg=args.experimental_contact_pad_angle_deg,
        experimental_contact_pad_force_unvalidated=args.experimental_contact_pad_force_unvalidated,
        grasp_profile_kind=args.grasp_profile_kind)
    contact_setup = {}
    if diagnostics_enabled:
        sensor_links = (("shoulder", "upper_arm", "lower_arm", "wrist",
                         "gripper", "jaw", "target")
                        if args.contact_diagnostics_all_robot_links
                        else ("gripper", "jaw", "target"))
        for link in sensor_links:
            sensor_name = f"demo11_contact_{link}"
            try:
                contact_sensor = env.scene[sensor_name]
                data = contact_sensor.data
                matrix = data.force_matrix_w
                contact_setup[link] = {
                    "sensor_bodies": list(contact_sensor.body_names),
                    "num_bodies": int(contact_sensor.num_bodies),
                    "filter_count": int(contact_sensor.contact_physx_view.filter_count),
                    "filter_paths": list(contact_sensor.cfg.filter_prim_paths_expr),
                    "force_matrix_shape": list(matrix.shape) if matrix is not None else None,
                    "force_matrix_initial_abs_sum_n": (
                        float(matrix.abs().sum().item()) if matrix is not None else None),
                }
            except Exception as exc:  # optional introspection must not abort a benchmark
                contact_setup[link] = {"setup_error": f"{type(exc).__name__}: {exc}"}
        print(f"Contact sensor setup: {json.dumps(contact_setup, ensure_ascii=False)}", flush=True)
    rows = []
    started = time.perf_counter()
    try:
        env.reset()
        initial_close_command = (real_world.GRIPPER.close_command_rad
                                 if args.close_command is None else args.close_command)
        module.set_gripper(env, angle_rad=initial_close_command)
        # The benchmark's default is a pure friction-grasp measurement. Only the explicit CLI
        # switch enables the simulator object-follow fallback for every geometry; the optional
        # auto policy mirrors the presentation demo and only assists its unproven geometries.
        bot = module.StrictExecutor(
            env, allow_kinematic=args.allow_kinematic,
            auto_sim_assist=args.auto_sim_assist,
            close_command_override=args.close_command).with_layout(selected)
        if args.gripper_yaw_deg is not None:
            set_gripper_yaw(bot, args.gripper_yaw_deg)
        rng = random.Random(args.seed)
        park_all(env, selected)
        for item in selected:
            if item.kind not in args.kinds:
                continue
            if args.contact_diagnostics and item.prim != diagnostic_target:
                continue
            for index in range(1, args.repetitions + 1):
                variants = ([None, args.compare_clearance_z]
                            if args.compare_clearance_z is not None
                            else [args.approach_clearance_z])
                state = rng.getstate()
                for variant_index, approach_z in enumerate(variants):
                    if variant_index:
                        rng.setstate(state)
                    trial_yaw = args.yaw_deg
                    if (trial_yaw is None and args.experimental_contact_pads
                            and args.experimental_contact_pad_shape == "vee"
                            and item.kind == "hexagonal_prism"):
                        # The validated hexagonal contacts need a flat-on jaw presentation.
                        trial_yaw = 30.0
                    row = run_trial(env, bot, item, rng, index, args.carry_steps,
                                    tuple(args.fixed_position) if args.fixed_position else None,
                                    ((tuple(args.position_bounds[:2]),
                                      tuple(args.position_bounds[2:]))
                                     if args.position_bounds else None),
                                    approach_z, args.track_every, _sensor, args.frames_dir,
                                    args.hover_clearance, trial_yaw, args.grasp_z_offset,
                                    tuple(args.approach_offset),
                                    args.pre_approach_dwell_steps,
                                    args.preserve_settled_pose)
                    labels = []
                    if approach_z is not None:
                        labels.append(f"clearance_z_{approach_z:.3f}m")
                    if args.track_every is not None:
                        labels.append(f"track_every_{args.track_every}")
                    if args.ball_rolling_damping is not None:
                        labels.append(f"ball_damping_{args.ball_rolling_damping:g}")
                    if args.ball_static_friction is not None:
                        labels.append(f"ball_mu_{ball_static:g}_{ball_dynamic:g}")
                    if args.park_pose is not None:
                        labels.append("park_pose_override")
                    if args.hover_clearance != 0.09:
                        labels.append(f"hover_{args.hover_clearance:.3f}m")
                    if trial_yaw is not None:
                        labels.append(f"yaw_{trial_yaw:g}deg")
                    if args.grasp_z_offset:
                        labels.append(f"grasp_z_{args.grasp_z_offset * 1000:+g}mm")
                    row["variant"] = "+".join(labels) if labels else "baseline"
                    uses_assist_metric = (args.allow_kinematic or
                                          (args.auto_sim_assist and
                                           item.kind in module.SIM_ASSISTED_SHAPES))
                    row["policy_lift_success"] = (row["simulator_lift_success"]
                                                  if uses_assist_metric else row["lift_success"])
                    row["policy_carry_success"] = (row["simulator_carry_success"]
                                                   if uses_assist_metric else row["carry_success"])
                    rows.append(row)
                    print(f"{item.prim} {index}/{args.repetitions} {row['variant']}: "
                          f"lift {'PASS' if row['policy_lift_success'] else 'FAIL'}, "
                          f"carry {'PASS' if row['policy_carry_success'] else 'FAIL'} ({row['grasp']}, "
                          f"lift {row['lifted_mm']:.1f} mm, "
                          f"carry error {row['carry_target_error_mm']:.1f} mm)", flush=True)
    finally:
        if args.keep_open:
            print("Benchmark complete; Isaac Sim remains open. Press Enter here to close it.", flush=True)
            try:
                input()
            except EOFError:
                pass
        else:
            env.close()
    groups = {}
    for row in rows:
        group = groups.setdefault(row["prim"], {"kind": row["kind"], "trials": 0,
                                                  "lift_successes": 0, "successes": 0,
                                                  "policy_lift_successes": 0,
                                                  "policy_successes": 0})
        group["trials"] += 1
        group["lift_successes"] += int(row["lift_success"])
        group["successes"] += int(row["success"])
        group["policy_lift_successes"] += int(row["policy_lift_success"])
        group["policy_successes"] += int(row["policy_carry_success"])
    report = {"seed": args.seed, "repetitions": args.repetitions,
              "strict_executor": not (args.allow_kinematic or args.auto_sim_assist),
              "allow_kinematic": args.allow_kinematic,
              "auto_sim_assist": args.auto_sim_assist,
              "grip_stiffness": args.grip_stiffness,
              "experimental_contact_pads": args.experimental_contact_pads,
              "auto_validated_contact_pads": args.auto_validated_contact_pads,
              "grasp_profile_kind": args.grasp_profile_kind,
              "experimental_contact_pad_radius_mm": args.experimental_contact_pad_radius_mm,
              "experimental_contact_pad_shape": args.experimental_contact_pad_shape,
              "experimental_contact_pad_thickness_mm": args.experimental_contact_pad_thickness_mm,
              "experimental_contact_pad_angle_deg": args.experimental_contact_pad_angle_deg,
              "experimental_contact_pad_force_unvalidated": args.experimental_contact_pad_force_unvalidated,
              "contact_diagnostics": args.contact_diagnostics,
              "contact_diagnostics_all_robot_links": args.contact_diagnostics_all_robot_links,
              "contact_sensor_setup": contact_setup,
              "keep_open": args.keep_open,
              "close_command_rad": (real_world.GRIPPER.close_command_rad
                                    if args.close_command is None else args.close_command),
              "close_command_override_rad": args.close_command,
              "gripper_yaw_deg": args.gripper_yaw_deg,
              "pad_static_friction": args.pad_static_friction,
              "pad_dynamic_friction": args.pad_dynamic_friction,
              "ball_static_friction": ball_static,
              "ball_dynamic_friction": ball_dynamic,
              "device": args.device,
              "ball_rolling_damping_override": args.ball_rolling_damping,
              "track_every_steps_override": args.track_every,
              "preserve_settled_pose": args.preserve_settled_pose,
              "park_pose_override_m": args.park_pose,
              "aim_source": "simulator_truth (manipulation-isolated)", "elapsed_s": round(time.perf_counter() - started, 1),
              "groups": groups, "trials": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    total = len(rows)
    passes = sum(row["policy_carry_success"] for row in rows)
    print(f"\nTOTAL {passes}/{total} ({passes / total:.1%}); report: {args.output}")
    return 0 if passes == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
