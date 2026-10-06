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
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
    bus.send_control("start", variant="open", camera="side", rounds=3, seed=7)
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
    control = bus.take_control()
    check(control is None, "the start control was consumed, so keys() never sees it")


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

    robot = module.MonitoredRobot.__new__(module.MonitoredRobot)
    robot.bus = telemetry.Bus()
    robot.mask_arm = False
    seen = []

    class Base:
        def _retry_one(self, prim):
            seen.append(robot.mask_arm)
            return prim == "cube_blue"

    module.MonitoredRobot.__bases__[0]._retry_one = lambda self, prim: seen.append(
        robot.mask_arm) or True
    try:
        robot._retry_one("cube_blue")
        check(seen == [True], "the retry ran with the arm masked")
        check(robot.mask_arm is False, "and restored the setting afterwards")
    finally:
        del module.MonitoredRobot.__bases__[0]._retry_one


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

        async def session(trigger):
            """What one browser sees: what arrives, and when."""
            import websockets

            url = f"ws://{monitor.host}:{monitor.port}/ws"
            arrived = []
            async with websockets.connect(url, open_timeout=10) as socket:
                if trigger is not None:
                    trigger()
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    try:
                        message = await asyncio.wait_for(socket.recv(), timeout=0.6)
                    except asyncio.TimeoutError:
                        break
                    if isinstance(message, bytes):
                        arrived.append(("bytes", len(message)))
                    else:
                        payload = json.loads(message)
                        arrived.append((payload.get("type"), payload.get("seq")))
            return arrived

        quiet = asyncio.run(session(None))
        check(quiet and quiet[0][0] == "snapshot",
              f"a connection must be answered without waiting for a change, got {quiet}")
        check(("bytes", 2004) in quiet,
              f"and the picture already published must come with it, got {quiet}")

        after_one = asyncio.run(session(lambda: bus.node("setup", telemetry.OK, "started")))
        snapshots = [row for row in after_one if row[0] == "snapshot"]
        check(len(snapshots) <= 2,
              f"one change must not turn into a flood; got {len(snapshots)} snapshots")
        check(len(snapshots) >= 2, "the change itself must still be pushed")

        # The spin would show up here as thousands of pushes in the two-second window.
        burst = asyncio.run(session(lambda: [bus.node("execute", telemetry.ACTIVE, str(i))
                                             for i in range(100)]))
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
            time.sleep(0.3)      # a real listen() records for seconds, not milliseconds
            return ""

    def submit_later(bus, text, delay=0.3):
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
    threading.Timer(0.3, lambda: probe.bus.send_control("stop")).start()
    check(ask_now(probe) == "quit",
          "the panel's stop must reach ask() even with no executor steps running")

    # Dashboard-only, and the person asks for the next round while it waits.
    probe = Probe(web_input=True)
    threading.Timer(0.3, lambda: probe.bus.send_control("next")).start()
    check(ask_now(probe) == "" and probe.advance,
          "the panel's next round must break the round, not run a command nobody sent")

    # With --voice the microphone has the floor for a moment; the browser is checked after it.
    probe = Probe(web_input=False, voice=Voice(), keyboard=True)
    submit_later(probe.bus, "pon el cubo rojo en la caja roja", delay=0.1)
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
              ("understand", telemetry.OK, "sort [cube ball]"),
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
    states = [{"phase": "setup"}, {"phase": "starting"}, {"phase": "ready", "sim": "running"},
              {"phase": "scene", "round": 0},
              {"phase": "command", "command": "separa las piezas por colores",
               "command_src": "dashboard"},
              {"phase": "command", "command": "separa las piezas por colores",
               "classification": {"intent": "sort",
                                  "items": [{"kind": "cube", "color": "red", "count": 1,
                                             "place": "box_red"}]}},
              {"phase": "command", "command": "separa las piezas por colores",
               "plan": {"n": 1, "intent": "sort", "picks": ["cube"],
                        "steps": [{"action": "above", "note": "above #3",
                                   "pose": [0.3, -0.4, 0.12]}]}},
              {"phase": "done", "verified": 2, "missed": 1}]

    snapshots = []
    for index, (node_id, state, note) in enumerate(stages):
        bus.node(node_id, state, note)
        bus.log(f"  etapa {node_id} -> {state}", node=node_id)
        if index < len(states):
            bus.set_state(**states[index])
        snapshots.append(bus.snapshot())

    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    # COLORS.idle is the colour buildFlow gives every box, so it is the only thing "this box has
    # not been visited yet" can be compared against. Read from the page so it cannot drift.
    idle = re.search(r'idle:\s*"(#[0-9a-fA-F]{6})"', page)
    with tempfile.TemporaryDirectory() as folder:
        dump = Path(folder) / "snapshots.json"
        dump.write_text(json.dumps(snapshots), encoding="utf-8")
        done = subprocess.run([node, str(ROOT / "tests" / "dashboard_client.js"),
                               str(ROOT / "monitor" / "static" / "index.html"), str(dump),
                               idle.group(1) if idle else "#4a5164"],
                              capture_output=True, text=True, timeout=120)
    check(done.returncode == 0, f"the dashboard's script must load: {done.stderr[:300]}")
    if done.returncode != 0:
        return
    report = json.loads(done.stdout)

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
    check(report["command"] == "separa las piezas por colores",
          f"the command box must show the sentence, got {report['command']!r}")
    check(report["planRows"] >= 1,
          f"the plan card must fill its steps table, got {report['planRows']} rows")
    check(report["planParts"] and "cube" in report["planParts"],
          f"the plan card must list the parts, got {report['planParts']!r}")
    check(report["intentChips"] > 0, "the classification card must offer its options")
    check(report["logLines"] > 0, "the log must grow as the run publishes")
    check(report["phase"] == "done", f"the phase pill must follow the run, got "
                                     f"{report['phase']!r}")


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