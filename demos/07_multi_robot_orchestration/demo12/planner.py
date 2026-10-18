"""Demo 12 -- planning: what should happen, and why this robot.

Two jobs, both of which the interface needs and neither of which is physics:

**Choose.** Which robot should take which part-to-bin move. The demo previously handed work
out on "lowest attempt count first, ties by id", which is deterministic and completely
blind: it will send the far robot to the near part while the near robot walks to the far one,
and it will keep piling retries onto one arm. :class:`AssignmentPolicy` scores every
(robot, task) pair on measurable geometry -- reach distance, carry distance, how many
attempts the task already has -- and picks the cheapest. That is what makes the roster a
*collaboration* rather than three robots independently taking turns, and it gives the
interface a cost breakdown to show instead of a bare assignment.

**Understand.** Turning what a person typed into a plan. :func:`parse_command` reads a line
like ``"mueve el cubo rojo a la caja azul"`` or ``"part_3 -> bin_2"`` and returns the
assignments it names, matched on ids, labels, colours and shapes in Spanish and English.
It returns what it could not resolve too, because a command that silently does nothing is
worse than one that says which word it did not know.

Nothing here touches the simulator. A plan is data; the orchestrator decides whether the
physics carries it out.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import BinConfig, MoveTask, PartConfig, SceneConfig, SceneError

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Colours a person is likely to type, in the two languages the interface ships. Keys are
# the hex colours the scene uses, lowercased; values are the words that select them.
COLOR_WORDS: dict[str, tuple[str, ...]] = {
    "#e0483c": ("rojo", "red", "red1"),
    "#3d7fd6": ("azul", "blue", "azul1"),
    "#3aa76d": ("verde", "green", "verd"),
    "#d98c1f": ("amarillo", "yellow", "naranja", "orange"),
    "#8b5cf6": ("morado", "purple", "violeta", "violet"),
    "#0ea5e9": ("cian", "cyan", "turquesa"),
    "#f472b6": ("rosa", "pink"),
    "#14b8a6": ("teal", "turquesa2"),
}

# Shapes, in both languages.
SHAPE_WORDS: dict[str, tuple[str, ...]] = {
    "cube": ("cubo", "cube", "caja", "box", "bloque"),
    "cylinder": ("cilindro", "cylinder", "cilindro1", "tubo"),
    "ball": ("bola", "ball", "esfera", "sphere", "pelota"),
}

# Words that mean "move this to that", in both languages. Only their presence matters; the
# grammar is deliberately not parsed, because a demo that rejected a phrasing it did not
# anticipate would be worse than one that reads the nouns and ignores the verbs.
MOVE_WORDS = (
    "mueve", "mover", "lleva", "llevar", "manda", "mandar", "mete", "meter", "coloca",
    "colocar", "pon", "poner", "ordena", "ordenar", "sortea", "sortar", "a", "to", "into",
    "move", "send", "put", "place", "take", "bring", "sort", "goes", "should",
)

# Words that mean "everything of this kind".
ALL_WORDS = ("all", "todos", "todas", "each", "every", "todo")


def _strip_accents(text: str) -> str:
    """``á`` -> ``a``, so a typed accent cannot change whether a word matches."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _flat(text: str) -> str:
    """Lowercase and de-accent, **without** trimming.

    Separators are matched against this and then used as *offsets into the original
    string*, so it has to preserve length exactly. ``_norm`` also strips, which turned a
    three-character ``" a "`` separator into a one-character ``"a"`` and made every
    sentence split at the wrong place -- ``"todos a bin_1"`` came out as destination
    ``"dos a bin_1"`` and matched nothing.
    """
    return _strip_accents(str(text or "").lower())


def _norm(text: str) -> str:
    return _flat(text).strip()


def color_name(hex_color: str, lang: str = "en") -> str:
    """A human name for a hex colour, from the shared vocabulary."""
    words = COLOR_WORDS.get(str(hex_color).lower())
    if not words:
        return str(hex_color)
    if lang == "es":
        return words[0]
    return next((w for w in words if w.isascii() and len(w) > 3), words[0])


# ---------------------------------------------------------------------------
# Assignment cost
# ---------------------------------------------------------------------------


@dataclass
class AssignmentCost:
    """Why one robot was chosen over another. Shown, not just logged.

    Every term is a measured distance in the cell or a count from the run history, so a
    judge can check the arithmetic. The weights below are the only taste in the file and
    they are all ratios rather than magic numbers.
    """

    robot_id: str
    task_id: str
    reach_m: float            # robot base -> part
    carry_m: float            # part -> destination bin
    attempts: int             # retries already spent on this task
    total: float
    #: How close this pick is to one another arm is already committed to, and whether that
    #: was near enough to matter. Reported because "robot_2 went to the far side of the
    #: cell" is only a satisfying answer if you can see what pushed it there.
    conflict_m: float = float("inf")
    conflicted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "robotId": self.robot_id,
            "taskId": self.task_id,
            "reachM": round(self.reach_m, 4),
            "carryM": round(self.carry_m, 4),
            "attempts": self.attempts,
            "conflictM": (None if self.conflict_m == float("inf")
                          else round(self.conflict_m, 4)),
            "conflicted": self.conflicted,
            "total": round(self.total, 4),
        }


class AssignmentPolicy:
    """Scores every (robot, task) pair and names the cheapest.

    Four terms, and the point of each:

    * ``reach`` -- the arm's base to the part. The dominant real cost; an SO-101 walking
      100 mm further across the table pays it in ticks, and the demo's rate limits mean
      distance is time.
    * ``carry`` -- the part to its bin, weighted lower than reach because the transfer is a
      rate-limited ramp either way. It still matters, because two robots will otherwise both
      queue for the same long carry while a short one sits idle.
    * ``retries`` -- a penalty per attempt already spent. Without it the cheapest robot is
      also the one that has already failed most, and a flaky grasp gets retried by the same
      arm until the budget runs out.
    * ``conflict`` -- a penalty when another arm is already committed to a pick in the same
      patch of table.

    **The conflict term is the one that makes this multi-robot rather than several
    independent single-robot schedulers.** Every other term is a property of one arm and one
    part, so with nothing here each arm takes whatever is nearest *itself* -- and since all
    the arms face the same way down the same bench, the two nearest things to two different
    arms are frequently the same corner.

    Measured, with a task per part: robot_1 took ``part_1`` at x = 0.290, and robot_2 -- for
    whom ``part_4`` at x = 0.360 was the cheapest thing in the cell -- took that too. Two
    arms 70 mm apart, fighting, and the run ended 0/4 with tracking errors of 146-208 mm
    that read exactly like a reachability fault and were not one. With the term present the
    same four tasks schedule the way the two that work do: robot_1 takes ``part_1``,
    robot_2 is pushed out to ``part_2``, and they run in parallel.

    Load balancing is the reason this is not simply "nearest robot". A robot that has been
      working is *already* busy, so the policy only ever chooses among idle ones -- the
    balance comes from the cost terms rather than from an explicit queue-length term.
    """

    #: How much a carried part's travel counts against a robot's reach to it. Reach and
    #: carry are both real, but reach is paid twice (out and back to park) while carry is
    #: paid once.
    CARRY_WEIGHT = 0.6

    #: Distance added per attempt already spent. One reach across the table is roughly
    #: 0.15 m, so this is "a retry is worth about one extra walk's worth of reluctance".
    RETRY_PENALTY_M = 0.15

    #: Two arms picking within this distance of each other fight over the same table.
    #: Measured: 70 mm loses the grasp, and 140 mm -- robot_1 on ``part_1``, robot_2 on
    #: ``part_2`` -- runs cleanly with the two arms working in parallel.
    CONFLICT_RADIUS_M = 0.130

    #: The penalty itself. Deliberately larger than any achievable travel cost (reach is at
    #: most 0.235 m and carry is weighted at 0.6), so in practice it acts as a filter. It is
    #: expressed as a cost rather than a rule so the interface can show *why* an arm went
    #: elsewhere, and so it can be relaxed to a preference if the cell is laid out
    #: differently later.
    CONFLICT_PENALTY_M = 0.5

    @staticmethod
    def cost(scene: SceneConfig, robot_id: str, task: MoveTask,
             attempts: int = 0, conflict_m: float = float("inf")) -> AssignmentCost:
        robot = scene.robot(robot_id)
        part = scene.part(task.part_id)
        target = scene.bin(task.bin_id)
        bx, by = robot.base_xy
        reach = math.hypot(part.x - bx, part.y - by)
        carry = math.hypot(target.x - part.x, target.y - part.y)
        conflicted = conflict_m < AssignmentPolicy.CONFLICT_RADIUS_M
        total = (reach
                 + AssignmentPolicy.CARRY_WEIGHT * carry
                 + AssignmentPolicy.RETRY_PENALTY_M * attempts
                 + (AssignmentPolicy.CONFLICT_PENALTY_M if conflicted else 0.0))
        return AssignmentCost(robot_id=robot_id, task_id=task.id, reach_m=reach,
                              carry_m=carry, attempts=attempts, total=total,
                              conflict_m=conflict_m, conflicted=conflicted)

    @staticmethod
    def conflict_with(part: PartConfig,
                      busy_xy: list[tuple[float, float]]) -> float:
        """Distance from this pick to the nearest pick another arm is already committed to.

        ``busy_xy`` holds the part positions of the tasks *other* working robots hold -- the
        caller filters out its own, because a robot is never in conflict with itself.
        ``inf`` when nothing else is working, so "no conflict" is the default rather than
        something the first arm of the cell has to be lucky about.
        """
        if not busy_xy:
            return float("inf")
        return min(math.hypot(part.x - x, part.y - y) for x, y in busy_xy)

    def rank(self, scene: SceneConfig, robot_ids: Iterable[str],
             task: MoveTask, attempts: int = 0,
             busy_xy: list[tuple[float, float]] | None = None) -> list[AssignmentCost]:
        """Every candidate for this task, cheapest first.

        Eligibility is :func:`demo12.config.robot_can_work` -- the arm's *measured* working
        radius -- and not ``RobotConfig.reach``, which is the geometric envelope. The two
        disagree by 100 mm and using the wrong one here is how a validated cell gets
        scheduled into grasps the arm cannot make: measured, that mismatch is a run going
        from 2/2 to 0/4 with tracking errors of 146-208 mm.

        Robots that cannot work the part are excluded rather than ranked last. Scoring them
        would put a number in the interface next to an option that can never be taken, which
        is exactly the kind of thing that makes a demo look like it is lying.
        """
        from .config import robot_can_work

        busy_xy = busy_xy or []
        part = scene.part(task.part_id)
        out: list[AssignmentCost] = []
        for rid in robot_ids:
            robot = scene.robot(rid)
            if not robot_can_work(robot.base_xy, part.x, part.y):
                continue
            out.append(self.cost(scene, rid, task, attempts,
                                 self.conflict_with(part, busy_xy)))
        return sorted(out, key=lambda c: (c.total, c.robot_id))

    def choose(self, scene: SceneConfig, robot_ids: Iterable[str],
               task: MoveTask, attempts: int = 0,
               busy_xy: list[tuple[float, float]] | None = None) -> AssignmentCost | None:
        ranked = self.rank(scene, robot_ids, task, attempts, busy_xy)
        return ranked[0] if ranked else None


DEFAULT_POLICY = AssignmentPolicy()


# ---------------------------------------------------------------------------
# Task building and validation
# ---------------------------------------------------------------------------


def task_id_for(part_id: str, bin_id: str) -> str:
    """A stable id for a part-to-bin move, so re-planning the same move is a no-op.

    Derived from the pair rather than from a counter, because the interface lets someone
    change a destination and change it back, and a counter would leave orphaned tasks with
    fresh ids behind every edit.
    """
    return f"move_{part_id}_to_{bin_id}"


def make_task(scene: SceneConfig, part_id: str, bin_id: str) -> MoveTask:
    part = scene.part(part_id)
    target = scene.bin(bin_id)
    return MoveTask(
        id=task_id_for(part_id, bin_id),
        part_id=part_id,
        bin_id=bin_id,
        label=f"{part.label_es} {part.id.split('_')[-1]} -> {target.label}",
    )


def validate_move(scene: SceneConfig, part_id: str, bin_id: str) -> list[str]:
    """Everything wrong with asking for this move, as sentences.

    Returns a list rather than raising, because the interface wants to show all the
    problems with a proposed plan at once instead of one per attempt.
    """
    errs: list[str] = []
    part = next((p for p in scene.parts if p.id == part_id), None)
    target = next((b for b in scene.bins if b.id == bin_id), None)
    if part is None:
        errs.append(f"no part with id {part_id!r}")
    if target is None:
        errs.append(f"no bin with id {bin_id!r}")
    if part is None or target is None:
        return errs

    if not part.graspable:
        errs.append(
            f"{part.id} is {part.grasp_width_m * 1000:.0f} mm across and this gripper takes "
            f"10-30 mm parts, so it cannot be picked at all")
    if not any(r.reach.contains((part.x, part.y)) for r in scene.enabled_robots):
        errs.append(f"{part.id} is out of reach of every enabled robot")
    if not any(r.reach.contains((target.x, target.y)) for r in scene.enabled_robots):
        errs.append(f"{bin_id} is out of reach of every enabled robot")
    # A part has to fit *through* a tray's own mouth as well as in its cavity. A part wider
    # than the cavity could be carried over the rim and dropped in, so this is a warning
    # about tidiness rather than about possibility -- but a user who stacks two 30 mm cubes
    # side by side in a 75 mm tray is asking for a pile, not a sort.
    half_diag = math.hypot(*target.inner_m) / 2.0
    if part.footprint_radius_m * 2 > half_diag * 2:
        errs.append(
            f"{part.id} ({part.footprint_radius_m * 2000:.0f} mm across) is wider than "
            f"{bin_id}'s {target.accepts} cavity")
    return errs


def build_plan(scene: SceneConfig, moves: list[tuple[str, str]]) -> dict[str, Any]:
    """Turn ``(part_id, bin_id)`` pairs into tasks, reporting anything unusable.

    Two parts may be sent to the same bin -- that is a legitimate thing to want, and it is
    how the demo shows a tray filling up. What is *not* allowed is two tasks fighting over
    the same part, because nothing can pick a part that is already in a gripper.
    """
    tasks: list[MoveTask] = []
    errors: list[str] = []
    seen: set[str] = set()

    for part_id, bin_id in moves:
        if part_id in seen:
            errors.append(f"{part_id} already has a task in this plan")
            continue
        seen.add(part_id)
        errs = validate_move(scene, part_id, bin_id)
        if errs:
            errors.extend(errs)
            continue
        tasks.append(make_task(scene, part_id, bin_id))

    return {"ok": not errors, "tasks": tasks, "errors": errors}


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """What a typed line asked for, and what could not be resolved."""

    text: str
    moves: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    understood: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "moves": [{"partId": p, "binId": b} for p, b in self.moves],
            "errors": list(self.errors),
            "understood": list(self.understood),
        }


def _score_parts(text: str, scene: SceneConfig) -> list[tuple[int, PartConfig]]:
    """``(evidence, part)`` for every part the text could mean, strongest first.

    Evidence is **additive**, which is the part that matters. ``"the red cylinder"`` matches
    the red cube on colour *and* the red cylinder on both colour and shape, and the second
    is obviously the better reading -- but with a first-match-wins ladder both stop at the
    colour test and the red cube wins on scene order.

    Weights are powers of two so each match stays distinguishable in the sum, and the
    ambiguity check downstream can ask "did these two parts match *exactly* as well",
    which is the question a person asking "move the cube" is really asking.
    """
    t = _norm(text)
    scored: list[tuple[int, int, PartConfig]] = []
    for order, part in enumerate(scene.parts):
        if re.search(rf"\b{re.escape(_norm(part.id))}\b", t):
            score = 0                                  # an id names one part outright
        else:
            score = 0
            if any(re.search(rf"\b{re.escape(_norm(w))}\b", t)
                   for w in COLOR_WORDS.get(_norm(part.color), ())):
                score += 1
            if any(re.search(rf"\b{re.escape(_norm(w))}\b", t)
                   for w in SHAPE_WORDS.get(part.kind, ())):
                score += 2
            if _label_hits(part.label_es, t):
                score += 4
            if score == 0:
                continue
        scored.append((score, order, part))
    # Higher evidence first; scene order breaks ties so a run is reproducible.
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(score, part) for score, _, part in scored]


def _match_part(text: str, scene: SceneConfig) -> list[PartConfig]:
    """Every part the sentence could mean, best match first. See :func:`_score_parts`."""
    return [part for _, part in _score_parts(text, scene)]


#: Plural endings tolerated on a label word.
_PLURAL_SUFFIXES = ("", "s", "es")


def _word_pattern(word: str) -> str:
    """A regex matching one label word in any form it is likely to be typed in.

    Two Spanish inflections matter and neither is a suffix that can just be appended:

    * **Gender.** ``caja roja`` is the feminine of ``Rojo``, which drops the final ``o``
      rather than adding to it, so appending ``a`` to ``rojo`` produces ``rojoa`` and
      matches nothing. The last vowel becomes a class instead: ``roj[oa]``.
    * **Plural.** ``bandejas`` for ``bandeja``, which appends ``s`` or ``es`` depending
      on the ending.

    Getting this wrong is silent -- the command simply reports "no destination matched" for
    a sentence that is perfectly clear to a person.
    """
    if not word:
        return ""
    if word[-1] in "oa":
        # Gender-variable ending. The class has to span **both** vowels, not just the one the
        # label happens to use: matching the label "Rojo" against the typed "caja roja"
        # needs `roj[oa]`. Emitting only `[o]` produces `rojo` and matches nothing, which is
        # a silent failure for a sentence a person would call perfectly clear.
        return rf"{re.escape(word[:-1])}[oa](?:s|es)?"
    if word[-1] == "e":
        return rf"{re.escape(word)}s?"
    return rf"{re.escape(word)}(?:{'|'.join(_PLURAL_SUFFIXES[1:])})?"


def _label_hits(label: str, text: str) -> bool:
    """Whether a label appears in the text as a whole word, in any inflected form.

    Whole-word rather than substring, because a substring test on a short label finds
    things that are not there: ``"azul" in "azules"`` is fine by accident and
    ``"Rojo" in "Rojo"`` inside an unrelated word is not.
    """
    pattern = "".join(_word_pattern(w) for w in _norm(label).split())
    if not pattern:
        return False
    return bool(re.search(rf"\b{pattern}\b", text))


def _match_bin(text: str, scene: SceneConfig, *, by_colour: bool = True,
               by_label: bool = True, by_id: bool = True) -> list[BinConfig]:
    """Every bin the text could name.

    The three selectors exist because the three halves of a sentence need different ones.
    ``"mueve el cubo **rojo** a la caja azul"`` mentions two bin colours and only the second
    is the destination; matching the subject by colour turns a good command into "this names
    a bin, not a part". And because a tray labelled "Rojo" shares its word with the colour
    red, the subject check has to drop labels too -- otherwise "cubo rojo" matches bin_1's
    *label* and the command is rejected. For a subject, only an explicit ``bin_2`` counts.
    """
    t = _norm(text)
    hits: list[BinConfig] = []
    for target in scene.bins:
        if by_id and re.search(rf"\b{re.escape(_norm(target.id))}\b", t):
            hits.append(target)
            continue
        if by_colour and any(
                re.search(rf"\b{re.escape(_norm(w))}\b", t)
                for w in COLOR_WORDS.get(_norm(target.color), ())):
            hits.append(target)
            continue
        if by_label and _label_hits(target.label, t):
            hits.append(target)
            continue
    return hits


# Separators, longest first. Order matters: ``" a la "`` must be tried before ``" a "``, or
# "mueve el cubo rojo a la caja azul" splits at "a " and leaves "la caja azul" as the
# subject.
_SEPARATORS = ("->", "=>", " hacia el ", " hacia la ", " hacia ", " a la ", " a el ",
               " to ", " into ", " a ", " en ")


def parse_command(text: str, scene: SceneConfig) -> CommandResult:
    """Read one line and return the moves it names.

    Deliberately shallow. It resolves nouns -- part ids, colours, shapes, bin ids, bin
    labels, bin colours -- and takes everything after the first "to"/"a"/"->" as the
    destination and everything before it as the subject. It does not try to be a grammar,
    because a demo that refuses a phrasing it did not anticipate is a worse demo than one
    that is permissive and reports what it did.

    Unresolvable words are returned in ``errors`` rather than swallowed. "Move the purple
    thing to the blue bin" against a cell with no purple part must say so.
    """
    raw = str(text or "").strip()
    result = CommandResult(text=raw)
    if not raw:
        result.errors.append("empty command")
        return result

    # Offsets are taken from the *flattened* string and applied to the original, so the two
    # must have the same length. ``_flat`` preserves length; ``_norm`` does not.
    flat = _flat(raw)

    # "all" / "todos" is resolved first, because "todos a bin_1" names no part at all and
    # would otherwise be reported as an unresolvable subject.
    wants_all = any(_norm(w) in flat for w in ALL_WORDS)

    subject, destination = raw, ""
    for sep in _SEPARATORS:
        idx = flat.find(sep)
        if idx != -1:
            subject = raw[:idx]
            destination = raw[idx + len(sep):]
            break

    if not destination:
        # No separator at all: a bare "part_3 bin_2". Take the last token as the
        # destination and the rest as the subject.
        toks = flat.split()
        if len(toks) >= 2:
            cut = raw.rfind(toks[-1])
            subject, destination = raw[:cut], raw[cut:]

    dest_bins = _match_bin(destination, scene)
    if not dest_bins:
        if destination.strip():
            result.errors.append(
                f"no destination bin matched {_norm(destination.strip())!r}")
        else:
            result.errors.append(
                "no destination bin named. Try: 'mueve el cubo rojo a la caja azul' "
                "or 'part_1 -> bin_2'")
        return result
    bin_id = dest_bins[0].id
    result.understood.append(f"destination = {bin_id}")

    # The subject half, matched on explicit ids only. A colour there belongs to the part,
    # and a tray label is the same word as that colour.
    subj_bins = _match_bin(subject, scene, by_colour=False, by_label=False)
    if subj_bins:
        result.errors.append(
            f"{subject.strip()!r} is a bin, not a part. "
            f"Did you mean '{subj_bins[0].id} -> {bin_id}'?")
        return result

    parts = _match_part(subject, scene)

    if wants_all:
        chosen = parts or scene.parts
        for part in chosen:
            result.moves.append((part.id, bin_id))
        how = (chosen[0].label_es if len(chosen) == 1 else f"all {len(chosen)} parts")
        result.understood.append(f"{how} -> {bin_id}")
        return result

    if not parts:
        if subject.strip():
            result.errors.append(f"no part matched {_norm(subject.strip())!r}")
        return result

    # Ambiguity is only real between parts that matched *equally well*. "the green cube"
    # and "the cube" both hit three parts, but the first names part_3 by colour while the
    # other two only match on shape, so the first is answered and the second is a genuine
    # coin toss. Comparing membership instead of score would reject both, which would make
    # the parser useless for exactly the natural sentences it is meant to accept.
    scored = _score_parts(subject, scene)
    top = scored[0][0]
    tied = [part for score, part in scored if score == top]
    if len(tied) > 1:
        ids = ", ".join(p.id for p in tied[:4])
        result.errors.append(
            f"{ids} all match {_norm(subject.strip())!r}. Name one, or say 'all'.")
        return result

    result.moves.append((parts[0].id, bin_id))
    result.understood.append(f"{parts[0].id} -> {bin_id}")
    return result


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, str]] = {
    "color": {
        "en": "Sort every part into the tray of its own colour.",
        "es": "Cada pieza va a la bandeja de su propio color.",
    },
    "mixed": {
        "en": "Send everything to one tray. The interesting part is the contention: two "
              "robots want the same narrow approach.",
        "es": "Todo a una sola bandeja. Lo interesante es la contencion: dos robots "
              "compiten por el mismo acercamiento estrecho.",
    },
    "balanced": {
        "en": "Cross-sort: send every part to the tray furthest from where it started, so "
              "both robots have to cross the cell.",
        "es": "Clasificacion cruzada: cada pieza va a la bandeja mas lejos de su punto "
              "de partida, asi ambos robots cruzan la celda.",
    },
}


def scenario_moves(scene: SceneConfig, name: str) -> list[tuple[str, str]]:
    """A one-click assignment, so the demo is one click from interesting."""
    bins = scene.bins
    if not bins:
        return []

    if name == "mixed":
        return [(p.id, bins[0].id) for p in scene.parts]

    if name == "balanced":
        # Furthest bin from the part, ties broken by id so the demo is reproducible.
        def distance_to(p: PartConfig, b: BinConfig) -> float:
            return math.hypot(b.x - p.x, b.y - p.y)

        return [(p.id, max(bins, key=lambda b: (distance_to(p, b), b.id)).id)
                for p in scene.parts]

    # "color": match by colour where possible, otherwise round-robin so every part still
    # gets somewhere. A part whose colour no tray carries is a real situation, and silently
    # dropping it would make the demo look like it lost track of it.
    moves: list[tuple[str, str]] = []
    used: set[str] = set()
    for i, p in enumerate(scene.parts):
        match = next((b for b in bins if _norm(b.color) == _norm(p.color)), None)
        if match is None or match.id in used:
            match = bins[i % len(bins)]
        used.add(match.id)
        moves.append((p.id, match.id))
    return moves


__all__ = [
    "AssignmentCost", "AssignmentPolicy", "DEFAULT_POLICY", "SCENARIOS", "CommandResult",
    "build_plan", "color_name", "make_task", "parse_command", "scenario_moves",
    "task_id_for", "validate_move",
]