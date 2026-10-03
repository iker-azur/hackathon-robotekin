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
cavity leaves less margin than that for a 4 cm ball, so a release can miss. After the arm
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
# object. Perception is good to about 11 mm on this camera, so 25 mm is generous enough to
# survive a randomised layout and far tighter than the 51 mm smallest gap between two things
# of the same colour - which is what stops the red ball being linked to the red cube.
LINK_TOLERANCE_M = 0.025

# How far the arm's own parking sweep may move an object before the demo puts it back.
# See Robot.park: the sweep is unavoidable and launches a ball 168 mm when the arm comes back
# from a release over a tray, which is off the table entirely. 25 mm is above the 16.6 mm the
# sweep costs in the gentlest measured case and well under the 51 mm same-colour gap, so a
# restore never changes what anything looks like in the image.
PARK_RESTORE_M = 0.025

# How many times to re-look and retry a placement that did not verify.
RETRIES = 1

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
        # Always a writer, so the "save this frame" key works whatever --save-every says. With
        # --save-every 0 the counter is set out of reach, so nothing is written until asked.
        self.writer = viewer_module.FrameWriter(
            HERE / "out", every=args.save_every if args.save_every > 0 else 10 ** 9)

    # -- output ----------------------------------------------------------------

    def say(self, text: str = "") -> None:
        print(text, flush=True)

    # -- the round -------------------------------------------------------------

    def park(self) -> None:
        """Get the arm out of the camera's view, with nothing on the table to knock over.

        Parking sweeps the arm's forearm across the table and pushes the balls along it.
        Measured by tools/probe_park_speed.py: 16.6 mm per park from the retreat pose, always
        in the same direction, and unaffected by how high the arm travels - so it is the
        elbow's path, not the gripper clearing the objects by more. Four parks in a row walked
        a ball to x = -3.09 m, off a table spanning x 0.0032 to 0.7032, and an object that has
        left the table cannot be re-planned for.

        The sweep cannot be designed away, so it is watched instead: anything it moves by more
        than ``PARK_RESTORE_M`` is put back where the scene says it belongs, and the demo says
        so. Anything already resting inside a tray is left alone - putting that back would
        undo the round's work.
        """
        loose = {i.prim for i in self.layout if not i.container}

        # Put down anything still held. A "coge la bola verde" ends with the arm holding it,
        # and parking then drives to y = -0.58 with the ball still in the gripper: measured, it
        # ended up at y = -0.888, behind the robot and off the table, where perception then
        # found it as an unclassified green blob. Releasing first is also what a person does.
        if self.bot.holding:
            self.say(f"  putting down the {self.bot.holding} it was holding")
            self.bot.drop()
            self.bot.settle(20)

        before = {prim: self._xy(prim) for prim in loose}
        settled = {prim: self._xy(prim) for prim in loose}

        self.bot.park()

        trays = [i for i in self.layout if i.container]
        for prim in sorted(loose):
            x, y = self._xy(prim)
            moved = math.dist(settled[prim], (x, y)) * 1000
            if moved < PARK_RESTORE_M * 1000:  # PARK_RESTORE_M is metres, moved is mm
                continue
            if any(self._inside(prim, tray) for tray in trays):
                continue  # already placed: putting it back would undo the round's work
            item = next(i for i in self.layout if i.prim == prim)
            if not self._spot_is_free(item):
                # Restoring into an occupied spot is worse than not restoring. place_items
                # writes the pose outright, so an object written inside another is deeply
                # interpenetrating and PhysX resolves it violently: measured, a ball ejected
                # to x = -6.9 m, tens of times the length of the table. Leave it where it is
                # and let perception find it there.
                self.say(f"  the park sweep moved {prim} {moved:.0f} mm, but its spot is "
                         f"occupied; leaving it where it is")
                continue
            isaac_io.place_items(self.env, [item])
            self.say(f"  the park sweep moved {prim} {moved:.0f} mm to "
                     f"({x:+.4f}, {y:+.4f}); put it back")

    def _spot_is_free(self, item, slack: float = 0.005) -> bool:
        """Is this item's declared spot clear of everything else?"""
        for other in self.layout:
            if other.prim == item.prim:
                continue
            gap_x = abs(self._xy(other.prim)[0] - item.x) - scene_spec.half_width(item) \
                - scene_spec.half_width(other)
            gap_y = abs(self._xy(other.prim)[1] - item.y) - scene_spec.half_depth(item) \
                - scene_spec.half_depth(other)
            if max(gap_x, gap_y) < slack:
                return False
        return True

    def _xy(self, prim: str) -> tuple[float, float]:
        where = self.env.scene[prim].data.root_pos_w[0].detach().cpu().numpy()
        return float(where[0]), float(where[1])

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
        """Blank the gripper out, so an arm in frame cannot be mistaken for an object."""
        try:
            pixel = self.bot.gripper_pixel(camera)
        except Exception:  # the arm may not be where we think; masking is best-effort
            return image
        if pixel is None:
            return image
        u, v = pixel
        h, w = image.shape[:2]
        u, v = int(round(u)), int(round(v))
        radius = 46
        out = image.copy()
        y0, y1 = max(0, v - radius), min(h, v + radius)
        x0, x1 = max(0, u - radius), min(w, u + radius)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1].max(axis=1, keepdims=True) // 2
        return out

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

        self.say(f"  {plan.summary}")
        run = self.bot.run(plan, resolve=premise.get, keys=lambda: self.keys())
        self.say(f"  {run.summary()}")
        if run.aborted:
            self.say("  stopped early at your request")
            return True

        self.check(plan, premise)
        return True

    # -- verification and recovery ---------------------------------------------

    def check(self, plan, premise: dict[int, str]) -> None:
        """Did each object actually end up inside? Re-perceive, and retry once if not.

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
                self.say(f"    not verified: {prim} is not in the tray it was sent to")
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
        self.say(f"  camera sees {len(seen.objects)} object(s) and "
                 f"{len(seen.containers)} container(s); "
                 f"{len(plan.picks) - len(misses)}/{len(plan.picks)} placement(s) in a tray")

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
        self.bot.do("above", lift, executor_module.OPEN)
        self.bot.do("down", planner.grasp_pose(thing), executor_module.OPEN)
        before = here.copy()
        result = self.bot.do("lift", lift, executor_module.CLOSE, prim=prim, on_close=before)
        self.bot.log(f"      lift: {result.note}")
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
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spread", type=float, default=dynamics.DEFAULT_SPREAD_M,
                        help="how far a randomised layout may move things, in metres")
    parser.add_argument("--disturb", type=float, default=0.0, metavar="P",
                        help="probability of shoving the scene between planning and executing")
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
    env, sensor = isaac_io.build_env(args.camera, list(scene_spec.DEFAULT_SCENE))
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
        robot.voice = listener

        scripted = list(args.text) * max(1, args.repeat) if args.text else []
        failures = 0

        for round_index in range(args.rounds):
            seed = args.seed + round_index
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