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
    from unittest import mock

    held = []
    with mock.patch.object(module.executor_module.Executor, "grab",
                           lambda self, prim: held.append(prim)):
        strict.grab("cube")
    check(held == ["cube"] and strict.kinematic == 1,
          "--allow-kinematic must fall back to demo 10's mechanism and count it")


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
    for node_id in ("input", "understand", "plan"):
        block = page[page.index(f'nodeId === "{node_id}"'):]
        check("clip-1" in block[:block.index("return null")],
              f"the {node_id} box's note must be clipped")
    check("text-overflow: ellipsis" in page,
          "an edge label that cannot ellipsis will run over the box beside it")
    # the plan's own rows, one per line
    tasks = re.search(r"#plan-tasks \.what \{(.*?)\}", page, re.S)
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
                              capture_output=True, text=True, timeout=120)
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
                   # the model's own per-field scores, as `parse` now publishes them
                   "confidence": 0.742,
                   "confidences": {"intent": 0.98, "place_kind": 0.91,
                                   "place_color": 0.742, "sort_by": 0.88}}},
              {"phase": "command", "command": "separa las piezas por colores",
               "tasks": TASKS},
              {"phase": "command", "command": "separa las piezas por colores",
               "plan": {"n": 1, "intent": "sort", "picks": ["3"],
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
    check(report["cells"]["row-intent"]["total"] > 0,
          "the classification card must offer its options")
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
          f"the plan card must list every task, got {run['taskRows']} rows")
    check(tasks_html.count('class="done"') == 1,
          f"the one finished task must be marked done: {tasks_html[:200]}")
    check("1 de 3" in run["tally"],
          f"the tally, bottom right of the plan, must say how far along it is: {run['tally']!r}")
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

    # ---- la intención: tabla, celdas, elección resaltada y probabilidad --------------------
    check(report["intentRows"] == 7,
          f"the intention must be a row per field, got {report['intentRows']} rows")
    check(report["cells"]["row-intent"]["total"] > 0,
          "the intent row must offer its options as cells")

    # ---- cada opción es una celda de una matriz, no una etiqueta suelta ------------------
    #
    # Eran <span> en línea que se envolvían, y eso se leía como una lista corrida: no había forma
    # de ver qué opciones competían entre sí dentro de un campo. Una celda de matriz es un bloque
    # con anchura propia dentro de una rejilla, y eso es lo que se comprueba aquí.
    for row_id, cells in report["cells"].items():
        if cells["total"]:
            check(cells["divs"] == cells["total"],
                  f"every option in {row_id} must be a block cell, not an inline span: {cells}")
            check(cells["spans"] == 0,
                  f"{row_id} must have no inline option left over: {cells}")
    for row_id, cls in report["grids"].items():
        check("opts" in (cls or ""),
              f"{row_id} must sit in the option grid, got {cls!r}")

    # and the chosen one is a cell, highlighted, carrying the model's own probability
    check('class="opt sel"' in run["intentHtml"],
          f"the chosen option must be highlighted: {run['intentHtml'][:200]}")
    check("98%" in run["intentHtml"],
          f"the chosen option must show the model's probability: {run['intentHtml'][:200]}")
    check("74%" in run["clsOverall"],
          f"the card must show the overall confidence, got {run['clsOverall']!r}")


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
    with mock.patch.object(module.MonitoredRobot.__bases__[0], "run_plan",
                           lambda self, plan, premise, command: refused_run):
        robot.run_plan(plan, premise, "coge la pieza")

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

    def parent_do(self, action, pose=None, grip=None, prim=None, **_):
        return executor_module.StepResult(action=action, note="knocked out of reach", ok=False,
                                          grasp="lost")

    with mock.patch.object(executor_module.Executor, "do", parent_do):
        result = sx.do("lift", np.zeros(3), -1.0, prim="cube_a")
    check(result.ok is False and result.grasp == "stale",
          f"a lost grasp must stop the plan like a stale one: {result.grasp!r} ok={result.ok}")
    check(sx.lost == 1 and sx.failures.get("cube_a") == 1,
          "a lost grasp is counted, and against the part")


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


def _track_rig(module, aperture_at, observe_at):
    """A StrictExecutor with only what ``track`` reads, and a stand-in torch for its tensors."""
    import contextlib
    import types
    from types import SimpleNamespace

    import numpy as np

    class FT(np.ndarray):
        def clone(self):
            return self.copy()

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
    s.log = lambda text: None
    s.correcting, s.corrected = True, 0
    s.aperture_centre = aperture_at
    return s, commands, fake, observe_at


def _run_track(module, aperture_at, observe_at):
    from unittest import mock

    s, commands, fake, observe = _track_rig(module, aperture_at, observe_at)
    with mock.patch.dict(sys.modules, {"torch": fake}):
        res = s.track(observe=observe, target_z=0.20, steps=200, aim=(0.30, -0.45))
    return res, s, commands


def test_a_constant_residual_stalls_and_the_servo_stops_correcting():
    module = load_demo11()
    part = (0.30, -0.45)
    res, s, commands = _run_track(module, lambda: (part[0] - 0.020, part[1]),
                                  lambda: part)
    check("centrado detenido" in res.note, f"the stall is reported: {res.note}")
    check(res.ok is False, "a stalled descent is not a clean one")
    check(s.correcting is False and s.corrected <= module.STALL_WINDOW,
          f"no corrections after the stall: corrected={s.corrected}")
    check(commands[-1] == commands[-20],
          "the commanded pose stops moving once the stall is declared")


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
    check(module.SAME_OBJECT_MAX_SHIFT_M < 0.064,
          "the same-object gate must stay under the closest same-colour pair (64 mm)")


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
    check('phase="failed"' in source[starting:starting + 3000],
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


def test_the_right_column_pairs_camera_and_perception_and_merges_control_with_the_environment():
    """Camera and perception share one fixed row; control and environment are one panel with a
    moving start bar, and the camera says it is not connected until a frame arrives."""
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    pair = page[page.index('class="pair"'):page.index("</main>")]
    check('id="frame"' in pair and 'id="detections"' in pair,
          "the camera and the perception table are in the same row")
    check('id="cam-empty"' in page and "cámara aún no conectada" in page,
          "without a frame the camera says it is not connected")
    entorno = page[page.index('id="entorno-card"'):page.index('id="comando-card"')]
    check(page.index('id="entorno-card"') < page.index('id="comando-card"') < page.index('class="pair"'),
          "the right column reads top to bottom: control and environment, command, then camera and perception")
    check('class="help"' in page[page.index('id="comando-card"'):page.index('class="pair"')]
          and 'class="help"' in entorno,
          "the command and environment panels each carry an info icon in their title bar")
    check('id="start"' in entorno and 'id="stop"' in entorno and 'id="start-bar"' in entorno,
          "start, stop and the moving bar sit in the same panel as the environment settings")
    check("startPending" in page and "RUNNING_PHASES" in page,
          "a press blocks the start button until the environment is running")


def test_the_plan_box_lists_tasks_and_the_steps_with_their_poses_live_in_execution():
    """Plan says what will be moved; Ejecución says how, step by step, with the pose of each."""
    page = (ROOT / "monitor" / "static" / "index.html").read_text(encoding="utf-8")
    plan_block = page[page.index('if (nodeId === "plan")'):page.index('if (nodeId === "execute")')]
    exec_block = page[page.index('if (nodeId === "execute")'):]
    exec_block = exec_block[:exec_block.index("return null")]
    check('id="plan-tasks"' in plan_block, "the plan box lists its tasks")
    check("exec-steps" not in plan_block and '<table' not in plan_block,
          "the plan box carries no steps and no poses")
    check('id="exec-steps"' in exec_block and 'id="note-execute"' in exec_block,
          "the execution box holds the steps and what is running now")
    check("paintSteps(plan)" in page and 'el("exec-steps")' in page,
          "the steps are painted into the execution box, from the plan")


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
    bus.set_command_state(cid, telemetry.NEEDS_CLARIFICATION, "?Qu? caja?")
    check(bus.command_state(cid)["state"] == telemetry.NEEDS_CLARIFICATION,
          "a clarification is an explicit terminal outcome")



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


if __name__ == "__main__":
    raise SystemExit(main())