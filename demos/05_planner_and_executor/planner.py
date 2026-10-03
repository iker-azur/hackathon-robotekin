"""Turn a spoken command and a scene into a list of robot steps. No Isaac, no torch.

This is the whole of demo 10's decision-making, and it deliberately depends on nothing but
numpy and the scene JSON, so that it can be tested against the recorded frames in
``tests/fixtures`` without a GPU. Everything that needs the simulator lives in
``executor.py``.

Where the numbers come from
---------------------------
Three sources, and the split is the point:

* **perception localises** - ``Thing.pos_m``, measured on this camera at 9-11 mm for both
  loose objects and receptacles.
* **declared geometry supplies sizes** - the arm can only hold a 3 cm cube and a 4 cm ball
  because that is what was put on the table, and the cavity is 5.6 cm because
  ``scene_spec._tray`` says so. Perception's own cavity estimate is kept and compared
  against this by ``tools/check_trays.py``, and it is not reliable enough to aim at: two
  identical trays measured 4.8 cm and 6.0 x 6.9 cm against a true 5.6 cm.
* **verification settles it** - after every placement the object is re-detected and checked
  against the declared cavity. That is what makes the 9-11 mm tolerable, and it is why the
  plan carries the ids it expects to have moved rather than assuming success.

Declared sizes are read out of ``scene_spec.DEFAULT_SCENE`` rather than repeated here, so a
change to the tray geometry cannot leave this file disagreeing with the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import ground
import scene_spec
from scene import Scene, Thing

# The one loose object in the default scene, used for its declared size. Reading the real
# objects out of the spec is why OBJECT_HEIGHT cannot drift from what is spawned.
_LOOSE = [i for i in scene_spec.DEFAULT_SCENE if not i.container]
_TRAY = next(i for i in scene_spec.DEFAULT_SCENE if i.container)

# Declared edge of each kind of loose object, in metres, by kind. Built by scanning the
# scene rather than indexing into it: DEFAULT_SCENE's first two loose objects are both
# cubes, so _LOOSE[1] is not a ball and an index would have silently given every ball the
# cube's size.
OBJECT_HEIGHT: dict[str, float] = {}
for _item in scene_spec.DEFAULT_SCENE:
    if not _item.container:
        OBJECT_HEIGHT[_item.kind] = _item.size[0]
DEFAULT_HEIGHT = OBJECT_HEIGHT.get("cube", 0.03)

# The receptacle: how thick its floor is, how wide its cavity is, and how deep.
TRAY_FLOOR = _TRAY.floor
TRAY_CAVITY = _TRAY.inner
TRAY_DEPTH = _TRAY.cavity_depth

# How high above the cavity floor an object is released, so it drops rather than being
# pushed in. Releasing at the floor itself means the jaws have to enter the cavity.
RELEASE_CLEARANCE = 0.012

# Intents that never move the arm. Spelled out rather than aliased to ``ground.NO_ACTION``:
# they are no longer the same set. ``ground`` decides what to look up in the scene, this
# decides whether the arm moves, and ``pick`` is the case that shows they differ - it has an
# item and moves, but no destination.
NO_MOTION = {"greeting", "off_topic", "not_doable", "question",
             "stop", "cancel", "repeat", "answer_yes", "answer_no", "answer_value"}

OPEN, CLOSE = 1.0, -1.0

# Travel height above the tallest thing on the table, which is a tray rim.
TRAVEL_Z = scene_spec.TABLE_TOP_Z + _TRAY.height + 0.04


@dataclass(frozen=True)
class Step:
    """One thing the arm should do. ``pose`` is where, in world metres."""

    action: str
    pose: tuple[float, float, float] | None = None
    grip: float = OPEN
    target: int | None = None
    note: str = ""


@dataclass
class Plan:
    """What to do about one command, and what could not be done."""

    intent: str
    steps: list[Step] = field(default_factory=list)
    picks: list[int] = field(default_factory=list)
    places: list[int] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    question: dict = field(default_factory=dict)
    reply: str = ""

    @property
    def moves(self) -> bool:
        return bool(self.steps)

    @property
    def summary(self) -> str:
        if not self.moves:
            return self.reply or f"{self.intent}: nothing to do"
        return (f"{self.intent}: {len(self.picks)} pick(s) -> {len(self.places)} destination(s), "
                f"{len(self.steps)} step(s)")


def grasp_pose(thing: Thing, offset: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, float]:
    """Where the gripper body should be to hold ``thing``, in world metres.

    x and y are perception's, plus an optional systematic offset. z is declared: the object's
    centre sits half its own height above the table.

    A measured correction is applied because the plane-ray estimate has a documented bias -
    it assumes the blob centroid is on the table, and a blob centroid is on the object's top
    face. ``calibration.py`` measures the mean residual; pass it as ``offset``.
    """
    x, y = thing.pos_m[0] + offset[0], thing.pos_m[1] + offset[1]
    height = OBJECT_HEIGHT.get(thing.kind, DEFAULT_HEIGHT)
    return (x, y, scene_spec.TABLE_TOP_Z + height / 2.0)


def place_pose(thing: Thing) -> tuple[float, float, float]:
    """Where to release an object over a receptacle, in world metres.

    The receptacle's own ``pos_m``, at a height that clears the rim. NOT its
    ``inner_rect_m``: that is an estimate, and ``tools/check_trays.py`` measures it wrong by
    more than the wall thickness on one of the two trays. The cavity size comes from
    ``scene_spec`` instead.
    """
    return (thing.pos_m[0], thing.pos_m[1], scene_spec.TABLE_TOP_Z + TRAY_FLOOR + RELEASE_CLEARANCE)


def in_workspace(point: tuple[float, float, float]) -> str | None:
    """Why this point is unreachable, or None if it is fine."""
    x, y, _z = point
    reach = math.dist((scene_spec.ROBOT_XY[0], scene_spec.ROBOT_XY[1]), (x, y))
    if reach > scene_spec.MAX_REACH_M:
        return (f"{reach * 100:.1f} cm from the base, over the {scene_spec.MAX_REACH_M * 100:.0f} cm reach")
    if not (scene_spec.WORK_X[0] <= x <= scene_spec.WORK_X[1]
            and scene_spec.WORK_Y[0] <= y <= scene_spec.WORK_Y[1]):
        return f"({x:+.3f}, {y:+.3f}) is outside the working area"
    return None


def _carry(plan: Plan, action: str, pose, grip, target, note="") -> None:
    """Add one step. The action name is what the executor times, so it has to be a real one."""
    plan.steps.append(Step(action, pose, grip, target, note))


def _pick_and_place(plan: Plan, thing: Thing, tray: Thing, offset, tag: str) -> None:
    """The eight-stage cycle demo 05 drives, pointed at measured poses instead of constants."""
    grasp = grasp_pose(thing, offset)
    drop = place_pose(tray)
    above_grasp = (grasp[0], grasp[1], TRAVEL_Z)
    above_drop = (drop[0], drop[1], TRAVEL_Z)

    _carry(plan, "above", above_grasp, OPEN, thing.id, f"{tag}: above #{thing.id}")
    _carry(plan, "down", grasp, OPEN, thing.id, f"{tag}: down onto #{thing.id}")
    _carry(plan, "close", grasp, CLOSE, thing.id, f"{tag}: close on #{thing.id}")
    _carry(plan, "lift", above_grasp, CLOSE, thing.id, f"{tag}: lift #{thing.id}")
    plan.picks.append(thing.id)
    _carry(plan, "carry", above_drop, CLOSE, thing.id, f"{tag}: carry to #{tray.id}")
    _carry(plan, "lower", drop, CLOSE, tray.id, f"{tag}: lower over #{tray.id}")
    _carry(plan, "release", drop, OPEN, tray.id, f"{tag}: release into #{tray.id}")
    plan.places.append(tray.id)
    _carry(plan, "retreat", above_drop, OPEN, tray.id, f"{tag}: retreat from #{tray.id}")


def _grasp_and_hold(plan: Plan, thing: Thing, offset, tag: str) -> None:
    """Pick an object up and keep it. `pick` has no destination, so there is no carry."""
    grasp = grasp_pose(thing, offset)
    above = (grasp[0], grasp[1], TRAVEL_Z)
    _carry(plan, "above", above, OPEN, thing.id, f"{tag}: above #{thing.id}")
    _carry(plan, "down", grasp, OPEN, thing.id, f"{tag}: down onto #{thing.id}")
    _carry(plan, "close", grasp, CLOSE, thing.id, f"{tag}: close on #{thing.id}")
    _carry(plan, "lift", above, CLOSE, thing.id, f"{tag}: lift #{thing.id} and hold")
    plan.picks.append(thing.id)


def _destinations_for_sort(scene: Scene, sort_by: str):
    """Group the scene by colour or kind, and pair each group with a receptacle.

    A ``sort`` that named no destinations - "separa las piezas por colores" - has to work out
    for itself which things go where, and the only rule a person would use is that each
    colour goes in the receptacle of that colour.

    Yields ``(key, label, members, receptacle)`` per group, the receptacle being None when
    nothing on the table matches, so the caller can report the groups that have nowhere to go
    instead of quietly doing half the sort.
    """
    groups: dict[str, list[Thing]] = {}
    for thing in scene.objects:
        key = str(thing.color if sort_by == "color" else thing.kind)
        groups.setdefault(key, []).append(thing)

    trays = list(scene.containers)
    for key in sorted(groups):
        members = groups[key]
        want_colour = key if sort_by == "color" else None
        want_kind = key if sort_by == "kind" else None
        match = next((t for t in trays
                      if (want_colour is None or t.color == want_colour)
                      and (want_kind is None or t.kind == want_kind)), None)
        yield key, f"{len(members)} {key}", members, match


def plan_command(command: dict, scene: Scene,
                 offset: tuple[float, float] = (0.0, 0.0)) -> Plan:
    """A command and a scene: what the arm should do about it.

    Never raises for a command the robot cannot carry out. An impossible command comes back
    as a Plan with no steps and a reason in ``reply``, because the demo's whole point is to
    answer a person rather than crash on them.
    """
    intent = command.get("intent")
    plan = Plan(intent=intent or "unknown")

    if intent in NO_MOTION:
        plan.reply = f"{intent}: nothing to move"
        return plan
    if intent in ("stop", "cancel", "repeat"):
        # Control intents are the caller's business - it owns the pause flag and the last
        # plan - not a motion to synthesise here.
        plan.reply = f"{intent}: handled by the loop, not by the planner"
        return plan

    resolved = ground.ground_command(scene, command)

    if intent == "sort" and not resolved["executable"]:
        return _plan_sort_from_scene(plan, scene, command, resolved, offset)

    if intent == "clear":
        return _plan_clear(plan, scene, resolved, offset)

    if not resolved["executable"]:
        plan.reply = f"nothing to pick up: {resolved['summary']}"
        for entry in resolved["items"]:
            if entry.status == "ambiguous" and entry.question:
                plan.question = dict(entry.question)
                plan.reply = entry.question.get("es") or plan.reply
            elif entry.status == "not_found":
                plan.unresolved.append(entry.say("es"))
        return plan

    place = resolved["place"]
    # `pick` is the one moving intent with nowhere to put anything: "coge la bola roja" means
    # hold it. Everything else needs a receptacle - but it may be the item's own rather than
    # the command's, which is how "pon el cubo rojo en la caja azul y la bola verde en la
    # bandeja roja" works with no command-level place at all.
    own_places = resolved.get("item_places") or []
    needs_place = intent != "pick"
    every_item_has_own = (needs_place and own_places
                          and all(p.ok and p.place is not None for p in own_places))
    if needs_place and not every_item_has_own and (not place.ok or place.place is None):
        plan.reply = "nowhere to put it: " + place.status
        if place.question:
            plan.question = dict(place.question)
        return plan
    default_tray = place.place if place.ok else None

    for index, entry in enumerate(resolved["items"]):
        if not entry.ok:
            plan.unresolved.append(entry.status)
            continue
        tray = default_tray
        if index < len(own_places) and own_places[index].ok and own_places[index].place is not None:
            tray = own_places[index].place
        for thing in entry.targets:
            pose = grasp_pose(thing, offset)
            problem = in_workspace(pose)
            if problem:
                plan.unresolved.append(f"#{thing.id} {thing.color} {thing.kind}: {problem}")
                continue
            tag = thing.color or thing.kind or "item"
            if tray is None:
                _grasp_and_hold(plan, thing, offset, tag)
            else:
                _pick_and_place(plan, thing, tray, offset, tag)

    if not plan.moves and not plan.unresolved:
        plan.reply = "nothing matched"
    return plan


def _plan_sort_from_scene(plan: Plan, scene: Scene, command: dict, resolved: dict, offset) -> Plan:
    """A bare sort: work the groups out from the scene and pair each with a receptacle.

    "separa las piezas por colores" names no colours and no receptacles. The rule a person
    would use is the only one available: each colour goes in the receptacle of that colour.
    Groups with no matching receptacle are reported rather than dropped, so a half-finished
    sort says so.
    """
    sort_by = command.get("sort_by") or "color"
    pairs = list(_destinations_for_sort(scene, sort_by))
    if not pairs:
        plan.reply = f"nothing on the table to sort by {sort_by}"
        return plan

    placed_any = False
    for _key, label, members, tray in pairs:
        if tray is None:
            plan.unresolved.append(f"{label} {sort_by}: no {sort_by} receptacle for it")
            continue
        for thing in members:
            pose = grasp_pose(thing, offset)
            problem = in_workspace(pose)
            if problem:
                plan.unresolved.append(f"#{thing.id} ({label}): {problem}")
                continue
            _pick_and_place(plan, thing, tray, offset, label)
            placed_any = True
    if not placed_any:
        plan.reply = f"sort by {sort_by}: nothing could be placed"
    return plan


def _plan_clear(plan: Plan, scene: Scene, resolved: dict, offset) -> Plan:
    """Clear: every loose object, or only the ones a filter named."""
    place = resolved["place"]
    if not place.ok or place.place is None:
        plan.reply = "nowhere to put them: " + place.status
        return plan
    tray = place.place

    named = [entry for entry in resolved["items"] if entry.ok]
    if named:
        things = [t for entry in named for t in entry.targets]
    else:
        things = list(scene.objects)

    for thing in things:
        pose = grasp_pose(thing, offset)
        problem = in_workspace(pose)
        if problem:
            plan.unresolved.append(f"#{thing.id}: {problem}")
            continue
        _pick_and_place(plan, thing, tray, offset, thing.color or thing.kind or "item")
    return plan
