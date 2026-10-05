"""Tests for demo 11's parts that need no simulator. No GPU, no Isaac.

    python tests/test_demo11.py

Three groups, in the order they would stop a run:

* **The real-world numbers.** Masses come from materials, the jaws have an aperture, the finger
  tips clear the rim, and a placement has to have room left after the camera's own error. These
  are the claims that make the demo a replica rather than a simulation, so they are checked as
  arithmetic rather than trusted.
* **The bus and the dashboard.** The two threads meet only through ``telemetry.Bus``, and the
  browser only ever queues requests, so a snapshot taken while a robot thread is publishing has
  to be self-consistent and the HTTP surface has to answer.
* **The flow diagram's integrity.** The picture is built from ``telemetry.FLOW``; if an edge names
  a node that does not exist the dashboard draws a line to nowhere and nobody notices for a week.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import executor as executor_module  # noqa: E402
import real_world  # noqa: E402
import scene_spec  # noqa: E402
import telemetry  # noqa: E402

failures: list[str] = []
checks = 0


def check(condition, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


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
        check((bus.take_control(timeout=0.5) or {}).get("action") == "relook",
              "a live control must be queued for the robot")

        with urllib.request.urlopen(base, timeout=5) as response:
            page = response.read().decode("utf-8")
        check("text/html" in response.headers.get_content_type(), "the root serves HTML")
        for needed in ("decision path", "environment setup", "camera", "voice", "log"):
            check(needed in page.lower(), f"the dashboard is missing its {needed} panel")
    finally:
        monitor.stop(timeout=5.0)


def test_the_executor_refuses_to_fake_a_grasp():
    """The strict executor's contract, without a simulator: ``grab`` refuses, ``do`` survives."""
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
    try:
        strict.grab("cube")
    except module.GraspRefused as refusal:
        check(str(refusal) == "cube", "the refusal names the part")
    else:
        check(False, "StrictExecutor.grab must refuse rather than carry the part by hand")

    strict.allow_kinematic = True
    held = []
    module.executor_module.Executor.grab = lambda self, prim: held.append(prim)
    try:
        strict.grab("cube")
        check(held == ["cube"] and strict.kinematic == 1,
              "--allow-kinematic must fall back to demo 10's mechanism and count it")
    finally:
        del module.executor_module.Executor.grab


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print(f"{checks} checks over the demo 11 real-world model, the bus and the dashboard")
    if failures:
        print(f"{len(failures)} FAILED:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("RESULT: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())