"""Talk to the robot, and watch it do it. Demo 10.

    python 10_command_to_motion.py                 # window, live camera, type commands
    python 10_command_to_motion.py --headless      # no window
    python 10_command_to_motion.py --text "pon el cubo rojo en la caja roja"
    python 10_command_to_motion.py --rounds 3 --disturb 1.0

This is where demo 09's four pieces stop being four programs. The camera finds the objects
(``perceive``), the model turns a sentence into a command (``command_model``), grounding turns
that into positions on this table (``ground``), the planner turns positions into arm moves
(``planner``), and ``executor`` drives them. Anything you can say, it tries; anything it cannot
do, it says so in the language you used.

What is new here, and what each piece is for
---------------------------------------------
**The layout moves.** Every round re-randomises where everything is (``dynamics.randomize``),
by enough that nothing downstream can be reading a hardcoded coordinate. Perception still has
to find all four objects and both trays in all of them, which is checked rather than assumed.

**Things move mid-task.** With ``--disturb`` the scene is shoved between planning and
executing, which is the case that matters: the plan was computed from a camera frame that is
now wrong. The answer is to look again and plan again, not to finish a plan already known to
be stale. Whether the robot noticed is reported, because "it still worked" and "it replanned"
are different claims.

**Every placement is checked afterwards.** Perception localises a tray to about 11 mm and the
cavity leaves less margin than that for a 3 cm ball, so a release can miss. After the arm
finishes, the demo looks again and asks whether the object is actually inside. If it is not,
it replans that one object and tries once more. A demo that only reports what it attempted is
not reporting whether it worked.

**Parking comes first, populating second.** Driving the arm out of the camera's view sweeps
its own forearm across the table, and anything already down gets shoved; ``place_items`` is
what puts it back. See ``executor.PARK_POSE``.

**The two reds are deliberate.** ``ball_red``, the red cube and the red box are all red, so
"the red ball" is genuinely ambiguous and the robot has to ask which one - with the question
in your language. That moment is worth more than another green cube.

Keys, in the viewer window: ``q`` or ``escape`` quits, ``a`` masks the arm out of perception,
``s`` saves a frame, ``r`` re-looks, ``n`` starts the next round.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import colors  # noqa: E402
import describe  # noqa: E402
import dynamics  # noqa: E402
import executor as executor_module  # noqa: E402
import ground  # noqa: E402
import isaac_io  # noqa: E402
import perceive  # noqa: E402
import planner  # noqa: E402
import scene as scene_module  # noqa: E402
import scene_spec  # noqa: E402
import viewer as viewer_module  # noqa: E402
from command_model_v2.command_model import CommandModel  # noqa: E402

# How close a perceived thing has to be to a known object before it is accepted as that
# object.
#
# 35 mm, raised from 25 after measuring what it costs. Perception localises a tray to 11 mm and
# a cube to about 10, but a BALL is seen up to 28 mm from where it is - the same ball-
# classification weakness that leaves it rolling - and at 25 mm the link silently failed, the
# grasp was skipped with no note, and the placement was reported as unverified for an object
# the robot was looking straight at.
#
# 35 mm is still safe against confusing two objects: the closest same-coloured pair in the
# scene is 64 mm apart centre to centre, and the link takes the nearest unclaimed prim, so a
# ball cannot be handed a cube's name.
LINK_TOLERANCE_M = 0.035

# How far the arm's own parking sweep may move an object before the demo mentions it.
#
# Not a restore threshold any more: nothing on the table is put back (see ``Robot.park``).
# This only decides what is worth a line of output. 5 mm is above the 1.7-3.6 mm of solver
# jitter a settled ball shows with the arm nowhere near it - measured, closest link 166 mm -
# and below anything a person would call the ball having moved.
PARK_REPORT_M = 0.005

# How many times to re-look and retry a placement that did not verify.
RETRIES = 1

# How many times one object may be attempted inside a multi-object command before the sort
# moves on without it. Two, not one: the first attempt can fail because the object was knocked
# into a neighbour's blob, and a second look usually finds it clean. The old behaviour gave up
# on the whole command after the first stale grasp, so one awkward object cost the other three.
MAX_PICK_ATTEMPTS = 2

# How far from where it was last seen an object may be and still count as the same object
# during a tracked approach. A damped ball rolls 33-45 mm when the arm touches it, and the
# two same-coloured objects in the scene are 90 mm apart, so this sits between the two: wide
# enough to follow a knock, tight enough not to hand the arm a different object to grab.
TRACK_LAST_SEEN_M = 0.075

# How much of the frame around the gripper is painted out while the arm is moving. The gripper
# is about 5 cm across and the top camera measures 747 px/m at the table, so its body is about
# 37 px wide and this covers it. A larger disc starts eating the object being reached for - a
# 3 cm ball is only 22 px across - so this is deliberately tight, and the low-altitude rule in
# ``executor.TRACK_OBSERVE_ABOVE_Z`` is what stops the arm being tracked once it is on top of
# its own target.
MASK_RADIUS_PX = 18

# How low the gripper may come before the camera stops being consulted.
#
# Below this the gripper is directly over the target and occludes it completely: measured, a
# blue cube vanished from the frame entirely while the arm was on its way down to it. Rather
# than fight that, the aim is locked in from the hover and held. The object is 3 cm across and
# the last 10 cm of a descent lasts a third of a second, so it cannot get far in that time -
# and the jaws still find it, which the stale-grasp check verifies rather than assumes.
TRACK_ABOVE_Z = scene_spec.TABLE_TOP_Z + 0.10

# Keys the demo answers itself, so a scripted run and a typed one behave the same.
QUIT_WORDS = {"q", "quit", "exit", "salir", "salidas"}
NEXT_WORDS = {"n", "next", "siguiente"}
LOOK_WORDS = {"r", "relook", "mirar", "ver"}
SAVE_WORDS = {"s", "save", "guardar"}


class Robot:
    """The demo's whole loop, in one place.

    Owns the simulator, the arm and the last thing it saw. Keeping the order of operations in
    a single class is not tidiness: the order is the part that is easy to get wrong and
    impossible to read from a ``main``.
    """

    def __init__(self, env, sensor, args, viewer=None, model=None):
        self.env = env
        self.sensor = sensor
        self.args = args
        self.viewer = viewer
        self.model = model
        # Set to a voice.Listener by main() when --voice is passed; None means every command
        # arrives at the keyboard. Declared here rather than only in main() so ask() cannot
        # reach an attribute that does not exist if voice is off.
        self.voice = None
        self.base = list(scene_spec.DEFAULT_SCENE)
        self.layout = list(self.base)
        self.bot = executor_module.Executor(env, log=self.say)
        self.scene: scene_module.Scene | None = None
        self.last_image = None
        self.shown = None
        self.frames = 0
        self.label = "start"
        self.round_index = 0
        self.mask_arm = False
        self.verified = 0
        self.recovered = 0
        self.missed = 0
        self.replanned = 0
        # Commands that had to be replanned because the arm knocked its own target away.
        self.regrasped = 0
        # Commands re-aimed after the arm knocked the target and it finished rolling.
        self.reaimed = 0
        # Approach and descent moves that re-read the camera while the arm was moving.
        self.tracked = 0
        # Placements judged, including the ones inside a multi-object command.
        self.checked = 0
        self.premise: dict[int, str] = {}
        # How far the last park sweep moved each loose object, kept so the round's report can
        # say what the arm did to the table rather than leaving it to be guessed at.
        self.park_swept: dict[str, float] = {}
        # Last camera position of each object, used to tell a knocked object from a different
        # one while tracking. Seeded from the layout and replaced as the camera reports.
        self.tracked_xy: dict[str, tuple[float, float]] = {
            i.prim: (i.x, i.y) for i in self.base}
        # Why the last ``locate`` failed, empty when it succeeded. Reported rather than
        # swallowed, because "lost the target" has half a dozen causes and only one of them is
        # "the ball rolled".
        self.locate_note = ""
        # Always a writer, so the "save this frame" key works whatever --save-every says. With
        # --save-every 0 the counter is set out of reach, so nothing is written until asked.
        self.writer = viewer_module.FrameWriter(
            HERE / "out", every=args.save_every if args.save_every > 0 else 10 ** 9)

    # -- output ----------------------------------------------------------------

    def say(self, text: str = "") -> None:
        print(text, flush=True)

    # -- the round -------------------------------------------------------------

    def park(self) -> None:
        """Get the arm out of the camera's view, and let go of anything it is holding.

        **Nothing on the table is put back.** An earlier version measured what the sweep
        moved and restored anything past ``PARK_RESTORE_M``, and that is precisely the
        behaviour this demo was asked to lose: a ball the arm had just placed somewhere was
        teleported back to the layout, so the table never reflected what had actually
        happened. An object is now placed once per round and from then on it is wherever it
        came to rest.

        The sweep is still watched, because the arm's forearm genuinely does pass over the
        near half of the table. It is reported rather than undone: measured with the balls
        now damped (``isaac_io.BALL_LINEAR_DAMPING``), a park moves them by a millimetre or
        two, and anything larger than ``PARK_REPORT_M`` is worth saying out loud.
        """
        loose = {i.prim for i in self.layout if not i.container}

        # Put down anything still held. A "coge la bola verde" ends with the arm holding it,
        # and parking then drives to y = -0.58 with the ball still in the gripper: measured, it
        # ended up at y = -0.888, behind the robot and off the table. Releasing first is also
        # what a person does.
        if self.bot.holding:
            self.say(f"  putting down the {self.bot.holding} it was holding")
            self.bot.drop()
            self.bot.settle(20)

        before = {prim: self._xy(prim) for prim in loose}
        self.bot.park()

        moved = []
        for prim in sorted(loose):
            drift = math.dist(before[prim], self._xy(prim))
            if drift > PARK_REPORT_M:
                moved.append(f"{prim} {drift * 1000:.0f} mm")
        if moved:
            self.say(f"  the park sweep moved {', '.join(moved)}; leaving them where they are")
        self.park_swept = {prim: math.dist(before[prim], self._xy(prim)) for prim in loose}

    def _xy(self, prim: str) -> tuple[float, float]:
        where = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
        return float(where[0]), float(where[1])

    def ensure_on_table(self, prim: str) -> bool:
        """Rescue an object that has left the table altogether. True if it had to.

        Deliberately narrow, because the previous version of this was the other half of the
        "everything goes back where it started" behaviour: it put any object outside the
        working area back on its square, which undid the round's work for anything that had
        legitimately been moved.

        Now only a genuinely lost object is repaired - one that has left the table's own
        footprint, where there is no "where it came to rest" left to preserve and nothing
        for the camera to find. An object that is merely outside the working area is left
        where it is and said out loud, because the kerb is what is supposed to stop that
        happening and it is worth knowing when it did not.
        """
        item = next((i for i in self.layout if i.prim == prim), None)
        if item is None:
            return False
        x, y = self._xy(prim)
        off_the_table = (x < scene_spec.TABLE_X[0] - 0.02 or x > scene_spec.TABLE_X[1] + 0.02
                         or y < scene_spec.TABLE_Y[0] - 0.02 or y > scene_spec.TABLE_Y[1] + 0.02)
        if not off_the_table:
            return False
        isaac_io.place_items(self.env, [item])
        self.say(f"  {prim} had left the table at ({x:+.4f}, {y:+.4f}); it cannot be seen or "
                 f"reached from there, so it is back on the table")
        return True

    def outside_the_area(self) -> list[str]:
        """Loose objects that are out of the working area, reported rather than corrected."""
        out = []
        for item in self.layout:
            if item.container or not self._inside_area(item.prim):
                out.append(f"{item.prim} ({self._xy(item.prim)[0]:+.3f}, "
                           f"{self._xy(item.prim)[1]:+.3f})")
        return out

    def _inside_area(self, prim: str, slack: float = 0.02) -> bool:
        """Is this object still somewhere the arm can reach and the camera can see?"""
        x, y = self._xy(prim)
        item = next((i for i in self.layout if i.prim == prim), None)
        if item is None:
            return True
        return (scene_spec.WORK_X[0] - slack <= x <= scene_spec.WORK_X[1] + slack
                and scene_spec.WORK_Y[0] - slack <= y <= scene_spec.WORK_Y[1] + slack)

    def populate(self, layout) -> None:
        isaac_io.place_items(self.env, layout)
        self.layout = list(layout)

    def look(self, caption: str = "") -> scene_module.Scene:
        """One frame in, one Scene out, drawn in the viewer.

        What the viewer shows is the ANNOTATED frame - the picture with every blob outlined,
        labelled with what was decided about it - not the raw camera image. Those are
        different pictures, and showing the raw one hides the entire point: you cannot tell
        whether the robot saw the green ball as a ball at (0.366, -0.472) or as an unclassified
        blob, which is exactly the question you want answered by looking at the screen.
        ``--raw`` shows the unannotated feed instead, for checking the render itself.
        """
        image, depth, camera = isaac_io.capture(self.env, self.sensor)
        if self.mask_arm:
            image = self._mask(image, camera)
        self.scene = perceive.perceive(
            image, source=self.label, camera=camera,
            frame=isaac_io.robot_frame(self.env), plane_z=scene_spec.TABLE_TOP_Z,
            depth=depth, looking_for=colors.colors(self.args.detect))
        self.last_image = image

        caption = caption or self.scene.summary()
        shown = image if self.args.raw else perceive.annotate(image, self.scene, caption)
        self.shown = shown
        self.writer.offer(shown)
        if self.viewer is not None:
            self.viewer.show(shown, caption)
        self.frames += 1
        return self.scene

    def _mask(self, image, camera):
        """Paint the gripper out of the frame, so an arm in shot cannot be an object.

        The pixels are set to BLACK, not dimmed. The original version halved the brightness,
        which changes V and nothing else - and every colour band in ``colors.py`` is keyed on
        hue and saturation, so a half-bright gripper still matched whatever colour it reflected
        and the mask removed nothing. Black has zero saturation, so it fails the saturation
        floor every band applies and cannot be picked up as anything.

        This matters more than it used to. With the arm hovering over the table, a tracked
        green ball reads as a "cube" and a blue cube disappears from the frame entirely -
        measured, the log said *"the camera sees no blue at all (it sees cube red, ball red,
        cube green)"*. The arm was not being mistaken for an object; it was standing in front
        of them.
        """
        try:
            pixel = self.bot.gripper_pixel(camera)
        except Exception:  # the arm may not be where we think; masking is best-effort
            return image
        if pixel is None:
            return image
        u, v = pixel
        h, w = image.shape[:2]
        u, v = int(round(u)), int(round(v))
        radius = MASK_RADIUS_PX
        out = image.copy()
        y0, y1 = max(0, v - radius), min(h, v + radius)
        x0, x1 = max(0, u - radius), min(w, u + radius)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = 0
        return out

    # -- watching one object while the arm moves ---------------------------------

    def locate(self, prim: str, last_xy: tuple[float, float] | None = None,
               kind_hint: str | None = None):
        """Where the camera says ``prim`` is right now, as (x, y), or None.

        ``kind_hint`` overrides the object's declared kind when re-identifying it. It is not
        used while tracking from a plan - there the declared kind is right - but the retry path
        needs it: a ball that has been dropped somewhere wrong is often read as a "cube" because
        the gripper is clipping its blob, and a retry that insisted on the kind would refuse to
        look at the very object it was sent to fix.

        This is the closed loop. The executor calls it every few steps while the arm is moving,
        and corrects its aim to match, so the jaws arrive where the object is rather than where
        it was when the plan was made - which on a ball is worth 20-40 cm.

        How the right blob is picked matters as much as that it re-reads the camera. It is
        matched on the object's own colour and kind, then on closeness to where it was last
        seen. Matching on the simulator's position would be cheating - that is the answer, not
        an observation - and matching on colour alone would hand the arm the red ball when it
        was sent for the red cube, since this scene has two reds of each kind on purpose.

        Each call takes a fresh frame, so the viewer keeps updating during the approach. That is
        the point of the whole change: on screen you can see the robot following the ball.

        ``self.locate_note`` says why it failed, because "could not find it" with no reason is
        exactly the kind of message that costs an afternoon: the object may be gone, out of
        frame, misclassified, or simply further from where it was last seen than anything the
        arm could have knocked it, and those need four different fixes.
        """
        item = next((i for i in self.layout if i.prim == prim), None)
        if item is None:
            self.locate_note = f"{prim} is not in this layout"
            return None
        seen = self.look("tracking")
        self.scene = seen

        of_colour = [t for t in seen.objects if not item.color or t.color == item.color]
        want_kind = kind_hint or item.kind
        of_kind = [t for t in of_colour if t.kind in ("any", want_kind)]
        candidates = self._nearest(of_kind, last_xy)

        # The kind is the part of perception that is least reliable, by its own docstring: it
        # is a second-moment test on a blob about 30 px across, and the arm's own gripper
        # hovering over the target clips the disc and turns a ball into a "cube". Measured, a
        # tracked green ball reads as a cube in every frame taken with the arm above it.
        #
        # So when nothing matches the kind, colour alone is allowed to carry the match - but
        # only for a colour that is unique on this table. Green has exactly one object, so
        # "the green blob" is unambiguous however it is shaped. Red has three, and there a
        # colour-only match could hand the arm the red cube, so it is refused and the reason is
        # reported instead.
        if not candidates:
            unique = sum(1 for i in self.layout if i.color == item.color) == 1
            if unique and of_colour:
                candidates = self._nearest(of_colour, last_xy)
                if candidates:
                    self.locate_note = (f"the {item.color} blob read as "
                                        f"{sorted({t.kind for t in of_colour})}, matched on "
                                        f"colour alone (it is the only {item.color} thing here)")

        if not candidates:
            if not self.locate_note:
                self.locate_note = self._why_lost(prim, item, seen, of_colour, of_kind, last_xy)
            return None
        best = min(candidates, key=lambda pair: pair[0])[1]
        # Corrected for the plane-ray bias, exactly as planner.grasp_pose is: tracking that
        # aimed at the raw estimate would follow the object to a systematically wrong place,
        # and the correction is worth 7 mm of a 20 mm grasp budget.
        self.tracked_xy[prim] = scene_spec.aim_point(best.pos_m)
        return self.tracked_xy[prim]

    @staticmethod
    def _nearest(things, last_xy) -> list[tuple[float, object]]:
        """Things with a measured position, paired with how far they are from the last sighting."""
        out = []
        for thing in things:
            if thing.pos_m is None:
                continue
            gap = math.dist(thing.pos_m[:2], last_xy) if last_xy is not None else 0.0
            if last_xy is not None and gap > TRACK_LAST_SEEN_M:
                continue
            out.append((gap, thing))
        return out

    def _why_lost(self, prim, item, seen, of_colour, of_kind, last_xy) -> str:
        """The reason a tracked object could not be found, in one line, for the log."""
        if not of_colour:
            names = ", ".join(f"{t.kind} {t.color}" for t in seen.objects) or "nothing"
            return (f"the camera sees no {item.color} at all (it sees {names})")
        if not of_kind:
            kinds = ", ".join(sorted({t.kind for t in of_colour}))
            return (f"the {item.color} blob was classified as {kinds}, not {item.kind}")
        if last_xy is not None:
            far = ", ".join(f"({t.pos_m[0]:+.3f}, {t.pos_m[1]:+.3f})" for t in of_kind
                            if t.pos_m)
            return (f"the {item.color} {item.kind} is at {far}, more than "
                    f"{TRACK_LAST_SEEN_M * 1000:.0f} mm from where it was last seen at "
                    f"({last_xy[0]:+.3f}, {last_xy[1]:+.3f})")
        return f"no {item.color} {item.kind} blob carried a position"

    def run_plan(self, plan, premise, command):
        """Run one plan with the arm masked out of perception.

        Masking is switched on automatically rather than left to the ``a`` key, because the
        frames that most need it are the ones taken while the arm is moving - the whole point of
        tracking - and nobody watching a demo presses a key to fix a frame they did not know was
        wrong. With the arm masked, a tracked ball stays a ball for the whole approach; with it
        unmasked, the hover alone was enough to turn the green ball into a "cube" and delete the
        blue cube.
        """
        was = self.mask_arm
        self.mask_arm = True
        try:
            return self.bot.run(plan, resolve=premise.get, keys=lambda: self.keys(),
                                retarget=make_retarget(self, command, self.args),
                                observe=lambda p: self.locate(p, self.tracked_xy.get(p)))
        finally:
            self.mask_arm = was

    def observe_for(self, prim: str):
        """``locate`` as a one-argument callback, remembering where it last saw the object."""
        return lambda: self.locate(prim, self.tracked_xy.get(prim))

    # -- linking perception to the simulator -----------------------------------

    def link(self, seen: scene_module.Scene) -> dict[int, str]:
        """Map each perceived thing's id to the prim it is looking at.

        Perception does not know prim names - it knows pixels - so the arm cannot be told
        what to grab until the two are matched up. Nearest known object within
        ``LINK_TOLERANCE_M`` wins, and each prim is claimed once, so two things of the same
        colour cannot both resolve to the same object.
        """
        by_id: dict[int, str] = {}
        claimed: set[str] = set()
        for thing in sorted(seen.everything, key=lambda t: t.pos_m is None):
            if thing.pos_m is None:
                continue
            best, best_gap = None, LINK_TOLERANCE_M
            for item in self.layout:
                if item.prim in claimed:
                    continue
                gap = math.dist(thing.pos_m[:2], (item.x, item.y))
                if gap < best_gap:
                    best, best_gap = item, gap
            if best is not None:
                claimed.add(best.prim)
                by_id[thing.id] = best.prim
        return by_id

    # -- one command -----------------------------------------------------------

    def handle(self, text: str) -> bool:
        """Plan and run one command. Returns False to end the demo."""
        text = text.strip()
        if not text:
            return True
        lowered = text.lower()
        if lowered in QUIT_WORDS:
            return False
        if lowered in NEXT_WORDS:
            self.say("  (next round)")
            return True
        if lowered in LOOK_WORDS:
            seen = self.look("re-look")
            self.say(f"  {seen.summary()}")
            return True
        if lowered in SAVE_WORDS:
            if self.shown is None:
                self.say("  no frame yet")
            else:
                self.writer.force(self.shown)
                self.say(f"  frame saved ({self.writer.written} so far) to {self.writer.folder}")
            return True

        command = self.model.parse(text)
        self.say(f"  parsed: {command.get('intent')} "
                 f"[{' '.join(str(i.get('kind')) for i in command.get('items', []))}]")

        # Park, then look. Both halves are required and the order matters.
        #
        # A plan ends with the arm at the retreat pose, directly over the working area, and a
        # camera looking down from the far side sees the arm before it sees the table. Looking
        # from there found 0 of 4 objects, so every command after the first in a round was
        # planned against an empty table and reported "nothing to pick up" - while the round's
        # own opening frame, taken with the arm parked, had found all four.
        #
        # Parking sweeps the forearm across the table and shoves what is on it, which is why
        # this looks *after* parking rather than reusing the previous frame: whatever moved is
        # now seen where it actually came to rest, and the plan is built on that.
        self.park()
        seen = self.look()
        self.say(f"  sees: {seen.summary()}")
        premise = self.link(seen)
        plan = planner.plan_command(command, seen, offset=self.args.offset)

        if plan.question:
            for lang, line in plan.question.items():
                self.say(f"  ask ({lang}): {line}")
        if plan.reply and not plan.moves:
            self.say(f"  {plan.reply}")
            return True
        if plan.unresolved:
            self.say(f"  could not do: {', '.join(plan.unresolved)}")

        if not plan.moves:
            self.say(f"  {describe.answer(seen, command, lang=self.args.lang)}")
            return True

        # Mid-task disturbance: the plan is already computed, so a shove now makes it stale
        # before a single step runs. Look again and plan again rather than trusting it.
        if self.args.disturb > 0.0 and random.random() < self.args.disturb:
            held = {premise.get(pid) for pid in plan.picks}
            self.say(dynamics.disturb(self.env, self.layout, random.Random(),
                                      exclude={h for h in held if h}).summary())
            self.park()
            seen = self.look("after a shove")
            premise = self.link(seen)
            again = planner.plan_command(command, seen, offset=self.args.offset)
            if not again.moves:
                self.say(f"  nothing left to do after the shove: {again.reply}")
                return True
            self.replanned += 1
            self.say(f"  replanned: {plan.summary} -> {again.summary}")
            plan = again

        if len(plan.picks) >= 2:
            # A multi-object command goes straight to the one-at-a-time path, WITHOUT
            # running the combined plan first. That ordering was the bug: this used to run
            # all 8N steps and only then start re-planning between objects, so the stale
            # plan - built from one frame and executed after the table has changed under it
            # - was always the first thing that happened.
            return self._one_at_a_time(command, plan)

        # Run it, and if the arm knocked the target away on the way in, look again and
        # plan again. This is the case that used to be reported as a success: the jaws shut
        # on empty air, the kinematic fallback picked the object up out of thin air, and the
        # placement was verified as correct. See GRASP_REACH_M in executor.py.
        for attempt in range(1 + self.args.regrasp):
            self.say(f"  {plan.summary}")
            run = self.run_plan(plan, premise, command)
            self.say(f"  {run.summary()}")
            self.tracked += run.tracked
            if self.locate_note and run.tracked:
                self.say(f"  tracking lost the target: {self.locate_note}")
            if run.reaimed:
                self.reaimed += 1
                premise = self.premise

            if run.stale_grasps:
                self.regrasped += 1
                self.say(f"  the arm knocked the target away before closing "
                         f"({run.aborted_at})")
                if attempt < self.args.regrasp:
                    self.say(f"  looking again and replanning (attempt {attempt + 2} of "
                             f"{self.args.regrasp + 1})")
                    self.park()
                    seen = self.look("after a stale grasp")
                    premise = self.link(seen)
                    again = planner.plan_command(command, seen, offset=self.args.offset)
                    if not again.moves:
                        self.say(f"  nothing left to plan: {again.reply}")
                        return True
                    plan = again
                    continue
                self.say("  out of attempts; the placement did not happen")
                return True

            if run.aborted:
                self.say("  stopped early at your request")
                return True

            self.check(plan, premise)
            return True
        return True

    @staticmethod
    def _first_pick_only(plan) -> planner.Plan | None:
        """A plan cut down to its FIRST pick, whole: pickup AND delivery.

        The boundary is the SECOND ``above`` step, because every pick in every cycle begins with
        one. That is deliberately not spelled as "cut at ``lift``", which is what an earlier
        version did and which was wrong for half the plans in this file:

            _grasp_and_hold   above down close lift                        <- lift IS the end
            _pick_and_place   above down close lift carry lower release retreat

        Cutting at ``lift`` left a pick-and-place with only its first four steps, so the arm
        picked the object up and then stopped. It was never carried to the tray and never
        released, which is why every sort reported its objects landing back where they started -
        85 mm and 128 mm outside the cavity - and only the retry, which builds its own motion,
        actually placed them.

        Cutting at the second ``above`` is right for both cycles without knowing which one this is,
        and it leaves the arm at the retreat pose over the tray rather than halfway across the
        table.
        """
        cut = len(plan.steps)
        seen_above = 0
        for index, step in enumerate(plan.steps):
            if step.action == "above":
                seen_above += 1
                if seen_above == 2:
                    cut = index
                    break
        steps = list(plan.steps[:cut])
        if not steps:
            return None
        one = planner.Plan(intent=plan.intent)
        one.steps = steps
        one.picks = plan.picks[:1]
        one.places = plan.places[:1]
        one.unresolved = list(plan.unresolved)
        one.question = dict(plan.question)
        one.reply = plan.reply
        return one

    def _one_at_a_time(self, command, plan) -> bool:
        """Carry out a multi-object command one object per pass, re-looking in between.

        "separa las piezas por colores" is the case that matters: the plan names four objects
        and 32 steps of motion, all computed from one frame. A person's answer is not a better
        plan, it is to sort one thing, look again, and sort the next - and that is what this
        does, because the world genuinely is different after each placement.

        Re-planning terminates on its own: once an object is inside a tray it is part of that
        tray's blob and is no longer a loose object, so each pass has one fewer thing to do.
        The pass count is bounded anyway, in case that assumption is ever wrong.

        A pick that fails does not end the command. It gets ``MAX_PICK_ATTEMPTS`` goes, and then
        the sort moves on to the next object and says at the end which ones it could not manage.
        The previous behaviour aborted the whole sort on the first stale grasp, which meant one
        awkward object cost the other three.
        """
        total = len(plan.picks)
        self.say(f"  {total} objects to move; doing them one at a time and re-looking between")
        moved: list[str] = []
        gave_up: list[str] = []
        attempts: dict[str, int] = {}

        for _ in range(total + 2):
            self.park()
            seen = self.look("between objects")
            premise = self.link(seen)
            fresh = planner.plan_command(command, seen, offset=self.args.offset)
            if not fresh.moves:
                break
            one = self._first_pick_only(fresh)
            if one is None:
                break

            prim = premise.get(one.picks[0]) if one.picks else None
            self.say(f"  {one.summary}")
            run = self.run_plan(one, premise, command)
            self.say(f"  {run.summary()}")
            self.tracked += run.tracked
            if self.locate_note and run.tracked:
                self.say(f"  tracking lost the target: {self.locate_note}")

            # A quit is not a failure to retry: stop the whole command.
            if run.aborted and not run.stale_grasps:
                self.say("  stopped early at your request")
                return True

            missed = self.check(one, premise, announce=False)

            # Three ways to fail, and all three have to count. A stale grasp is the executor
            # saying the jaws closed on nothing. An abort is the user pressing q. And a miss is
            # the quiet one: the grasp worked, the object was carried, and the release landed
            # beside the tray - which looks like success to every other counter in this file.
            # Counting only the first two is what let a sort loop on one unplaceable object.
            key = prim or f"thing #{one.picks[0]}"
            failed = bool(run.stale_grasps or run.aborted or key in missed)
            if failed:
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] >= MAX_PICK_ATTEMPTS:
                    gave_up.append(key)
                    reason = ("nothing to close on" if run.stale_grasps
                              else "stopped" if run.aborted else "never landed in a tray")
                    self.say(f"  giving up on {key} after {attempts[key]} attempt(s): {reason}")
            else:
                moved.append(key)

        self.say(f"  sorted {len(moved)} of {total}"
                 + (f"; could not pick {', '.join(gave_up)}" if gave_up else ""))
        return True

    def check(self, plan, premise: dict[int, str], announce: bool = True) -> list[str]:
        """Did each object actually end up inside? Re-perceive, and retry once if not.

        Returns the prims still not in a tray after the retry, so a caller that is looping over a
        multi-object command can count a repeated miss and move on. A grasp that "succeeded" but
        whose release missed is not a failure the executor can see - the jaws did close on the
        object - so without this the loop would retry the same placement forever.

        Judged from the simulator's own positions, not from a camera frame. An earlier version
        re-looked after the plan and re-linked the scene to decide, and that passed vacuously:
        the arm was still over the table, so perception saw nothing, the link map came back
        empty, every object "matched" no tray and therefore nothing was reported missing. A
        check that cannot fail is not a check. Perception's job was to aim the arm; whether
        the release landed is a fact about the simulation.

        The camera is still looked at afterwards, with the arm parked, and what it sees is
        reported next to the verdict, because a disagreement between the two is worth knowing
        about.
        """
        self.bot.settle()
        misses = self._missed(plan, premise)
        self.verified += len(plan.picks) - len(misses)

        for attempt in range(RETRIES):
            if not misses:
                break
            self.missed += len(misses)
            for prim in misses:
                # Say which of the three tests failed and by how much. "not in the tray" on its
                # own is the kind of message that sends you looking in the wrong place: a
                # release that lands beside the tray and one that lands on the rim look
                # identical in a log, and they have nothing in common.
                self.say(f"    not verified: {prim} {self._why_not_inside(prim)}")
            self.say("  re-planning the ones that missed")
            for prim in misses:
                self._retry_one(prim)
                self.recovered += 1
            self.bot.settle()
            misses = self._missed(plan, premise)
            self.verified += len(plan.picks) - len(misses)

        if misses:
            self.missed += len(misses)

        self.park()
        seen = self.look("after the plan")
        if announce:
            self.say(f"  camera sees {len(seen.objects)} object(s) and "
                     f"{len(seen.containers)} container(s); "
                     f"{len(plan.picks) - len(misses)}/{len(plan.picks)} placement(s) in a tray")
        return misses

    def _missed(self, plan, premise: dict[int, str]) -> list[str]:
        """Which picked objects are not inside the container they were sent to."""
        out: list[str] = []
        for pick_id, place_id in zip(plan.picks, plan.places):
            prim = premise.get(pick_id)
            tray_prim = premise.get(place_id)
            if prim is None or tray_prim is None:
                out.append(f"thing #{pick_id}")
                continue
            tray = next((i for i in self.layout if i.prim == tray_prim), None)
            if tray is None or not self._inside(prim, tray):
                out.append(prim)
        return out

    def _settled_height(self, item, tray) -> float:
        """The z an object rests at once it is DOWN in this tray, not perched on the rim.

        ``_inside`` compared against the cavity floor plus half the object's height, which is
        right for a cube and wrong for a ball in a way that only shows up in the last
        millimetres. A sphere dropped into a 56 mm cavity settles against the floor at
        floor + radius, and one that clips the rim perches at rim height - which for this tray is
        50 mm up. Measured: a ball reported as "30.3 mm from the resting height" was not in the
        tray at all, it was sitting on the edge, and the old test could only say "not verified".

        So this returns the floor-level rest height, and the z test becomes a comparison against
        that plus a margin wide enough to accept a ball that has settled slightly off-centre but
        still well inside.
        """
        return scene_spec.TABLE_TOP_Z + tray.floor + item.height / 2.0

    def _inside(self, prim: str, tray) -> bool:
        """Is this prim inside this tray, judged by where it actually came to rest?

        Both the tray and the object are known ``scene_spec`` items, so this needs nothing from
        the camera: the position is the simulator's, which is the honest test of whether the
        release worked, and the cavity is the declared geometry rather than perception's
        estimate of it - ``tools/check_trays.py`` measures that estimate wrong by more than a
        wall thickness on one of the two trays.

        The z test is what catches the interesting failure. An object that misses the tray
        entirely is still on the table, 6 mm lower than the cavity floor, and an x-y-only test
        would call that a miss anyway; an object that clips the rim and stays half outside is
        caught by the x-y half, not by z.
        """
        item = next((i for i in self.layout if i.prim == prim), None)
        if item is None or tray is None:
            return False
        where = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
        half_w, half_d = tray.inner[0] / 2.0, tray.inner[1] / 2.0
        rest_z = scene_spec.TABLE_TOP_Z + tray.floor + item.height / 2.0
        return (abs(float(where[0]) - tray.x) <= half_w
                and abs(float(where[1]) - tray.y) <= half_d
                and abs(float(where[2]) - rest_z) < 0.006)

    def _why_not_inside(self, prim: str) -> str:
        """Which of the containment tests failed, and by how much, in one clause.

        A miss has three quite different causes and the log used to call all of them "not in the
        tray": the object is beside the tray in x, it is beside it in y, or it is on the table
        rather than down on the cavity floor. The last is the common one and the least obvious -
        an x-y-only test passes for anything that lands near the rim, which is why ``_inside``
        checks z at all, and why a log saying only "not verified" sent this demo looking at the
        wrong axis.
        """
        item = next((i for i in self.layout if i.prim == prim), None)
        tray = self._tray_of(prim)
        if item is None or tray is None:
            return "is not in the scene any more"
        where = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
        half_w, half_d = tray.inner[0] / 2.0, tray.inner[1] / 2.0
        rest_z = scene_spec.TABLE_TOP_Z + tray.floor + item.height / 2.0
        off_x = abs(float(where[0]) - tray.x) - half_w
        off_y = abs(float(where[1]) - tray.y) - half_d
        off_z = abs(float(where[2]) - rest_z)
        if off_x > 0.0 or off_y > 0.0:
            return (f"came to rest at ({where[0]:+.3f}, {where[1]:+.3f}), "
                    f"{off_x * 1000:.0f} mm outside the cavity in x and "
                    f"{off_y * 1000:.0f} mm in y")
        # off_z rounds to 0 mm here more often than it should not: the miss is computed before
        # the last settle and re-read after it, so an object balanced on the rim can have
        # dropped the last millimetre into place in between. Saying "0 mm off" would be
        # nonsense, so it says which way it went.
        return (f"is at ({where[0]:+.3f}, {where[1]:+.3f}) and just inside the cavity, but "
                f"settled {off_z * 1000:.1f} mm from the resting height - it was balanced on "
                f"the rim")

    def _retry_one(self, prim: str) -> bool:
        """Pick one object up again and put it back where it was sent.

        Aimed from where the object ACTUALLY is, not from where the scene declares it. The
        first version used the declared position and so reached past the tray the object was
        already sitting in, closed on empty table, and reported the placement as still missed
        without ever having touched it.
        """
        item = next((i for i in self.layout if i.prim == prim), None)
        tray = self._tray_of(prim)
        if item is None or tray is None:
            self.say(f"    {prim}: nowhere to put it back")
            return False

        here = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
        thing = self._as_thing(item, here)
        lift = np.asarray(planner.grasp_pose(thing)) + np.array([0, 0, executor_module.ABOVE])
        # Release over the TRAY's centre, not the object's: the object's own position is where
        # it failed to land.
        drop = np.asarray(planner.place_pose(self._as_thing(tray, (tray.x, tray.y, tray.z))))

        self.bot.log(f"    retry {prim} from ({here[0]:+.4f}, {here[1]:+.4f}) to "
                     f"({tray.x:+.4f}, {tray.y:+.4f})")
        # The approach is closed-loop here too, not only in the main path. This retry used to
        # drive ``above`` and ``down`` once at a pose computed from the simulator's own
        # position, which is the one place in this file that is allowed to know the answer and
        # the one place that most needs the closed loop: the object it is retrying is one that
        # has just been dropped somewhere wrong, and a fixed descent is what put it there. If
        # the camera cannot find it, the retry is skipped rather than aimed at a guess.
        grasp = np.asarray(planner.grasp_pose(thing))
        self.tracked_xy[prim] = (float(here[0]), float(here[1]))
        observe = lambda: self.locate(prim, self.tracked_xy.get(prim), kind_hint="any")
        approach = self.bot.track(observe=observe, target_z=float(lift[2]),
                                  steps=executor_module.TRACK_DESCEND_STEPS,
                                  action="above", grip=executor_module.OPEN,
                                  aim=(float(lift[0]), float(lift[1])))
        descent = self.bot.track(observe=observe, target_z=float(grasp[2]),
                                 steps=executor_module.TRACK_DESCEND_STEPS,
                                 action="down", grip=executor_module.OPEN,
                                 aim=(float(grasp[0]), float(grasp[1])))
        if not (approach.ok and descent.ok):
            self.bot.log(f"      skipped: lost sight of {prim} on the way in")
            return False
        before = here.copy()
        result = self.bot.do("lift", lift, executor_module.CLOSE, prim=prim, on_close=before,
                             check="follow")
        self.bot.log(f"      lift: {result.note}")
        if not result.ok:
            return False
        self.bot.do("carry", drop + np.array([0, 0, executor_module.ABOVE]),
                    executor_module.CLOSE)
        self.bot.do("lower", drop, executor_module.CLOSE)
        self.bot.drop()
        self.bot.do("release", drop, executor_module.OPEN)
        self.bot.do("retreat", drop + np.array([0, 0, executor_module.ABOVE]),
                    executor_module.OPEN)
        self.bot.settle()
        landed = self._xy(prim)
        self.bot.log(f"    retry left {prim} at ({landed[0]:+.4f}, {landed[1]:+.4f})")
        return True

    def _tray_of(self, prim: str):
        """The container this prim was sent to, matched by colour.

        By colour and not by the plan, because the plan's destination id is a perception id
        from a frame that is now stale. Two containers of different colours is the whole
        vocabulary here, so colour identifies the destination exactly.
        """
        item = next((i for i in self.layout if i.prim == prim), None)
        if item is None:
            return None
        return next((i for i in self.layout if i.container and i.color == item.color), None)

    @staticmethod
    def _as_thing(item, pos_m):
        """A minimal Thing so ``planner``'s pose helpers can be reused on a simulator position.

        Perception cannot tell where a released object came to rest inside a tray - it is
        looking down into a cavity - so the retry works from the simulation's own numbers.
        Building a Thing here keeps one definition of "where do I hold this" instead of two
        that can drift apart.
        """
        return scene_module.Thing(id=-1, kind=item.kind, color=item.color,
                                  centroid_px=[], bbox_px=[], area_px=0, mean_hsv=[],
                                  confidence=1.0, source="sim", pos_m=list(pos_m))

    # -- input -----------------------------------------------------------------

    def keys(self) -> set[str]:
        return self.viewer.drain() if self.viewer is not None else set()

    def ask(self) -> str:
        """One command, typed or spoken.

        With ``--voice`` the typed prompt is still there, because a microphone is not always
        the right input: a noisy room, a quiet voice, or a scripted run all argue for typing.
        So voice *adds* a way in rather than replacing one, and an empty line or nothing at
        all falls through to the keyboard.
        """
        if self.voice is not None:
            print("  listening...", end="", flush=True)
            print("\r", end="", flush=True)
            heard = self.voice.listen()
            if heard:
                print(f"  heard: {heard}")
                return heard
            print("  heard nothing; type instead")

        try:
            return input("> ")
        except EOFError:
            return "quit"


    # -- re-aiming after the arm knocks its own target --------------------------

def make_retarget(robot, command, args):
    """Build the ``retarget`` callback the executor calls when tracking loses the target.

    The closed loop in ``Executor.track`` handles the ordinary case: the arm knocks its target
    a few centimetres and follows it. This handles the one it cannot - the object has left the
    camera's view, so there is nothing to follow and the plan has to be built again from
    scratch against a fresh frame.

    It returns ``(steps, resolve)`` and both matter. A re-planned step's ``target`` is a
    perception id from the frame taken *here*, not from the one the plan started with, so
    handing back steps without their resolver makes the next steps aim at whatever object
    happens to carry that id in the old map.

    No parking here, deliberately. Parking is what sweeps the objects in the first place, so
    looking from the hover pose is not a compromise - it is the only place the robot can look
    from without disturbing what it is looking at.
    """
    state = {"done": False}

    def retarget(step, prim):
        if state["done"] or prim is None:
            return None
        state["done"] = True

        moved = robot.bot.settle_object(prim)
        robot.say(f"  the arm knocked {prim} moving; waited {moved} step(s) for it to stop")
        if robot.ensure_on_table(prim):
            moved = robot.bot.settle_object(prim)

        seen = robot.look("re-aiming")
        fresh = planner.plan_command(command, seen, offset=args.offset)
        if not fresh.moves:
            robot.say(f"  nothing left to aim at: {fresh.reply}")
            return None

        robot.say(f"  re-aimed: {fresh.summary}")
        fresh_premise = robot.link(seen)
        robot.premise = fresh_premise
        # Drop the fresh plan's leading hover: the arm is already there.
        tail = fresh.steps[1:] if fresh.steps and fresh.steps[0].action == "above" else fresh.steps
        return tail, fresh_premise.get

    return retarget

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--voice", action="store_true",
                        help="listen to each command instead of typing it; still accepts typing")
    parser.add_argument("--voice-model", default="small",
                        help="faster-whisper model: tiny, base, small or medium (all cached)")
    parser.add_argument("--voice-language", default=None,
                        help="force one language (es, en, eu) instead of detecting it")
    parser.add_argument("--voice-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--raw", action="store_true",
                        help="show the unannotated camera feed instead of the drawn detections")
    parser.add_argument("--camera", default="top", choices=sorted(scene_spec.CAMERA_CHOICES))
    parser.add_argument("--variant", default="contained",
                        choices=["contained", "open", "out_of_reach"],
                        help="'contained' walls the working area in so a disturbed object "
                             "cannot leave it; 'open' is the original pen-free demo, kept as "
                             "the control that shows what the wall is for; 'out_of_reach' puts "
                             "two objects beyond the arm's reach and expects the robot to say "
                             "so instead of trying")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spread", type=float, default=dynamics.DEFAULT_SPREAD_M,
                        help="how far a randomised layout may move things, in metres")
    parser.add_argument("--disturb", type=float, default=0.0, metavar="P",
                        help="probability of shoving the scene between planning and executing")
    parser.add_argument("--regrasp", type=int, default=2, metavar="N",
                        help="times to re-perceive and replan when the arm knocks the target "
                             "out of the jaws' reach before closing (default 2)")
    parser.add_argument("--text", action="append", default=[], metavar="COMMAND",
                        help="run this command instead of waiting; repeatable")
    parser.add_argument("--detect", nargs="+", default=list(colors.DEFAULT_DETECT),
                        help="colours to look for")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("DX", "DY"), help="systematic grasp correction, in metres")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--save-every", type=int, default=0,
                        help="write a frame every N frames to out/")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each --text command this many times")
    args = parser.parse_args()

    # Load the model BEFORE the simulator. Loading it afterwards crashes the process outright
    # (0xC0000005, inside gliner2's from_pretrained): Isaac's runtime and the model's own
    # imports do not coexist in that order. Demo 03 loads its model first for the same reason.
    model = CommandModel()

    listener = None
    if args.voice:
        import voice

        complaint = voice.check_microphone()
        if complaint:
            parser.error(f"--voice cannot work: {complaint}")
        # Also before the simulator, for the same reason the command model is: two heavy
        # native libraries loading at once is what crashed the process before.
        listener = voice.Listener(model=args.voice_model, language=args.voice_language,
                                  device=args.voice_device)

    isaac_io.prepare_runtime(headless=args.headless)
    if args.variant == "out_of_reach":
        # The static scene with two objects beyond the arm. No pen: this variant is about
        # refusing what cannot be reached, not about containing what can.
        layout = scene_spec.out_of_reach_scene()
        env, sensor = isaac_io.build_env(args.camera, layout, pen=False)
        print("  two objects placed beyond the arm's reach, on purpose:", flush=True)
        for item in scene_spec.OUT_OF_REACH_ITEMS:
            print(f"    {item}: {scene_spec.why_unreachable(*scene_spec.OUT_OF_REACH_ITEMS[item])}",
                  flush=True)
    else:
        env, sensor = isaac_io.build_env(args.camera, list(scene_spec.DEFAULT_SCENE),
                                         pen=args.variant == "contained")
        if args.variant == "contained":
            import pen as pen_module
            print(f"  working area walled in: {pen_module.PEN_HEIGHT * 100:.1f} cm kerb "
                  f"around x {scene_spec.WORK_X} y {scene_spec.WORK_Y}", flush=True)
    # Reset, then aim. The camera's pose is applied at spawn and only becomes readable after a
    # reset, and aim_camera only checks where it ended up - so without the reset it reports
    # the sensor at the origin, 701 mm from where scene_spec put it, and stops. Every other
    # tool in this demo resets first for the same reason.
    env.reset()
    isaac_io.aim_camera(env, args.camera, sensor)
    viewer = None
    robot = None
    try:
        use_viewer = not args.no_viewer and not args.headless
        if use_viewer:
            viewer = viewer_module.Viewer()
        robot = Robot(env, sensor, args, viewer=viewer, model=model)
        if args.variant == "out_of_reach":
            # The Robot starts from DEFAULT_SCENE; this variant's layout is its own.
            robot.base = layout
        robot.voice = listener

        scripted = list(args.text) * max(1, args.repeat) if args.text else []
        failures = 0

        for round_index in range(args.rounds):
            seed = args.seed + round_index
            if args.variant == "out_of_reach":
                # STATIC, deliberately. This variant is about a stable fact - that cube is out
                # there and cannot be reached - so re-randomising would make the robot's answer
                # change every round and mean nothing. Every round is the same table.
                layout = robot.base
                robot.say(f"\n=== round {round_index} (static layout) ===")
            else:
                layout = dynamics.randomize(robot.base, seed, spread=args.spread)
                robot.say(f"\n=== round {round_index} (seed {seed}) ===")

            robot.park()  # before populate: the park sweep moves whatever is down
            robot.populate(layout)
            robot.round_index = round_index
            robot.label = f"round{round_index}"
            seen = robot.look("the table")
            robot.say(f"  {seen.summary()}")
            robot.say(f"  {describe.what_i_see(seen, lang=args.lang)}")

            queue = list(scripted) if scripted else None
            while True:
                if queue is not None:
                    if not queue:
                        break
                    text = queue.pop(0)
                    robot.say(f"\n> {text}")
                else:
                    text = robot.ask()
                if not robot.handle(text):
                    break
                if robot.keys() & {"q", "escape"}:
                    break

            if viewer is not None and viewer.wants_quit(robot.keys()):
                break

        robot.say("\n=== summary ===")
        robot.say(f"  placements verified : {robot.verified}")
        robot.say(f"  needed a retry      : {robot.recovered}")
        robot.say(f"  never verified      : {robot.missed}")
        robot.say(f"  replanned mid-task  : {robot.replanned}")
        robot.say(f"  regrasp after a stale target : {robot.regrasped}")
        robot.say(f"  re-aimed at a knocked target : {robot.reaimed}")
        robot.say(f"  moves that re-tracked the target mid-approach : {robot.tracked}")
        failures += robot.missed
        if robot.writer:
            robot.writer.close()
        return 2 if failures else 0
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
