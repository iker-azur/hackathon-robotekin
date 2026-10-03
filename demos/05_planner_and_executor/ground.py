"""Match what someone said to what the camera actually sees, and ask when it cannot.

The command model in demo 03 turns "coge la bola roja de la izquierda" into an item
with ``kind: ball``, ``color: red``, ``position: left``. This file turns that item
into scene object ids, using the six position words demo 03 already produces and the
kinds and colours it already uses, so grounding is a lookup rather than a translation.

Four outcomes, and the fourth is the point of the demo:

``ok``
    exactly one object answers the description.
``ok_filtered``
    more than one matched but a position word picked one.
``ambiguous``
    more than one matched and nothing narrowed them. The question names the options
    rather than just asking, because "¿cuál?" on its own makes the person do the
    robot's work, while "¿la de la izquierda o la de la derecha?" is a yes/no.
``not_found``
    nothing matched. The question says what is on the table, so the person can
    correct the robot instead of repeating themselves.

The reply builders speak the three languages of the project and reuse the wording
style of ``08_voice_hybrid.py``. They are duplicated here rather than imported
because that file imports the two 0.6-1.1 GB models at module level, and grounding
must stay importable with nothing but a scene JSON and numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import describe
import scene_spec
from scene import POSITIONS, Scene, Thing

# Below this many matches the robot would rather ask than guess, even with a position
# word, because a wrong pick is the one failure a person notices.
ASK_BELOW = 2

COLOR_WORDS = {
    "red": {"es": "rojo", "en": "red", "eu": "gorri"},
    "green": {"es": "verde", "en": "green", "eu": "berde"},
    "blue": {"es": "azul", "en": "blue", "eu": "urdin"},
    "yellow": {"es": "amarillo", "en": "yellow", "eu": "hori"},
    "white": {"es": "blanco", "en": "white", "eu": "zuri"},
    "black": {"es": "negro", "en": "black", "eu": "beltz"},
}

# Spanish carries its article inside the value, because the question builds "de {}"
# and "de izquierda" is not Spanish. English and Basque add theirs in the template:
# "the left one" needs a bare "left", and "ezkerrekoa" is the bare form plus "-a".
POSITION_WORDS = {
    "left": {"es": "la izquierda", "en": "left", "eu": "ezkerreko"},
    "right": {"es": "la derecha", "en": "right", "eu": "eskuineko"},
    "front": {"es": "delante", "en": "front", "eu": "aurreko"},
    "back": {"es": "detrás", "en": "back", "eu": "atzeko"},
    "nearest": {"es": "la más cercana", "en": "nearest", "eu": "hurbilen"},
    "farthest": {"es": "la más lejana", "en": "farthest", "eu": "urrunen"},
}

# The word that names the other candidate, so a question can offer both ends of one
# axis instead of the same word twice.
OPPOSITE = {
    "left": ("left", "right"),
    "right": ("right", "left"),
    "front": ("front", "back"),
    "back": ("back", "front"),
    "nearest": ("nearest", "farthest"),
    "farthest": ("farthest", "nearest"),
}


@dataclass
class Grounding:
    """What grounding decided about one item, and what to say when it could not decide."""

    status: str
    targets: list[Thing] = field(default_factory=list)
    candidates: list[Thing] = field(default_factory=list)
    question: dict[str, str] = field(default_factory=dict)
    place: Thing | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "ok_filtered")

    @property
    def target_ids(self) -> list[int]:
        return [t.id for t in self.targets]

    def say(self, lang: str = "es") -> str:
        """The reply in one language, empty when there was nothing to ask."""
        return self.question.get(lang, "")

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "target_ids": self.target_ids,
            "candidate_ids": [t.id for t in self.candidates],
            "place_id": self.place.id if self.place else None,
            "question": self.question,
            "note": self.note,
        }


def describe_item(item: dict, lang: str = "es") -> str:
    """Name one item of a command, singular, the way a person said it.

    Deliberately short. The person who hears this already knows what they asked for; the
    sentence exists to say which of two things the robot cannot tell apart. Uses
    ``describe.phrase`` so the gender and the number agree in all three languages.
    """
    return describe.phrase(item.get("kind") or "any", item.get("color"), 1, lang)


def describe_thing(thing: Thing, lang: str = "es") -> str:
    """Name one detected object, singular, the way a person would say it once."""
    return describe.phrase(thing.kind, thing.color, 1, lang)


def _separates(scene: Scene, candidates: list[Thing], word: str) -> bool:
    """Whether a position word actually tells these candidates apart.

    Delegates to the scene, so that "does this word separate them" and "in what order
    does this word rank them" can never use different axes. Two red balls side by side
    differ on the left/right axis but not on near/far, so asking "the near one or the
    far one?" would be nonsense.
    """
    return scene.separates(candidates, word)


# Each pair is one axis; only the first word of a pair that separates the candidates
# is offered, because "left" and "right" are the same distinction read backwards and
# asking both would give the person four choices where there are two.
AXIS_PAIRS = (("left", "right"), ("front", "back"), ("nearest", "farthest"))


def separating_word(scene: Scene, candidates: list[Thing]) -> str | None:
    """The position word that best tells a set of candidates apart."""
    for pair in AXIS_PAIRS:
        for word in pair:
            if _separates(scene, candidates, word):
                return word
    return None


def options_of(scene: Scene, candidates: list[Thing]) -> dict[str, object]:
    """The two candidate ids a position word tells apart, or nothing.

    Returns ``{"position": word, "first": id, "second": id}``, where ``first`` is the one
    the word does *not* name, so that the question can offer the pair rather than a
    word the person has to interpret.
    """
    if len(candidates) < 2:
        return {}
    word = separating_word(scene, candidates)
    if word is None:
        return {}
    ranked = scene.rank(candidates, word)
    if ranked[0].id == ranked[-1].id:
        return {}
    return {"position": word, "first": ranked[-1].id, "second": ranked[0].id}


def ambiguous_question(scene: Scene, candidates: list[Thing],
                       color: str | None = None) -> dict[str, str]:
    """Ask which one, naming the options when a position word can tell them apart.

    The noun phrases come from ``describe.phrase`` rather than being built here, so that
    "there are two red balls" agrees in Spanish, English and Basque instead of being a
    translated singular with a plural bolted on.

    ``color`` is the colour that was ASKED for, and it is not the same as the candidates'
    colour. An earlier version took the colour from ``candidates[0]``, which quietly
    mislabelled every other candidate: "mete las bolas" - no colour asked for - matched the
    red ball and the green ball, and the question came out "there are two BLUE balls",
    naming a colour the person never said and that only one of them had. So the colour is
    used only when the person gave one, or when every candidate really does share it;
    otherwise the phrase carries no colour and the question stays true.
    """
    count = len(candidates)
    kind = candidates[0].kind
    if color is None and len({c.color for c in candidates}) == 1:
        color = candidates[0].color
    split = options_of(scene, candidates)
    if split:
        # The count is already inside describe.phrase for plurals, so it is not repeated
        # here: "hay dos bolas rojas", not "hay 2 dos bolas rojas".
        named = {lang: describe.phrase(kind, color, count, lang) for lang in ("es", "en", "eu")}
        # The two options are named with opposite words, which is what makes the question
        # answerable with a yes or a no: "the one on the left, or the one on the right".
        # Basque needs the article on an adjectival phrase, so -a is added to the bare
        # form: ezkerrekoa, eskuinekoa.
        here, opposite = OPPOSITE[split["position"]]
        offered = {
            "es": f"la de {POSITION_WORDS[here]['es']} o la de {POSITION_WORDS[opposite]['es']}",
            "en": f"the {POSITION_WORDS[here]['en']} one or the {POSITION_WORDS[opposite]['en']} one",
            "eu": f"{POSITION_WORDS[here]['eu']}a edo {POSITION_WORDS[opposite]['eu']}a",
        }
        return {
            "es": f"Hay {named['es']}. ¿Cuál, {offered['es']}?",
            "en": f"There are {named['en']}. Which one, {offered['en']}?",
            "eu": f"{named['eu']} daude. Zein, {offered['eu']}?",
        }
    ids = ", ".join(f"#{thing.id}" for thing in candidates)
    named = {lang: describe.phrase(kind, color, count, lang) for lang in ("es", "en", "eu")}
    return {
        "es": f"Hay {named['es']} y no puedo distinguirlos ({ids}). ¿Cuál cojo?",
        "en": f"There are {named['en']} and I cannot tell them apart ({ids}). Which one?",
        "eu": f"{named['eu']} daude eta ezin ditut bereizt ({ids}). Zein hartu?",
    }


def not_found_question(scene: Scene, item: dict) -> dict[str, str]:
    """Say what is on the table instead, so the person can correct the robot."""
    present = sorted({thing.color for thing in scene.objects if thing.color})
    if not present:
        return {
            "es": "No veo ningún objeto en la mesa.",
            "en": "I cannot see any object on the table.",
            "eu": "Ez ikusten dut objekturik mahaian.",
        }
    kind, color = item.get("kind") or "any", item.get("color")
    colours = {lang: ", ".join(COLOR_WORDS.get(c, {}).get(lang, c) for c in present)
               for lang in ("es", "en", "eu")}
    return {
        "es": f"No veo ningún {describe.phrase(kind, color, 1, 'es', bare=True)} en la mesa. "
              f"Veo: {colours['es']}.",
        "en": f"I cannot see any {describe.phrase(kind, color, 1, 'en', bare=True)} on the table. "
              f"I see: {colours['en']}.",
        "eu": f"Ez ikusten dut {describe.phrase(kind, color, 1, 'eu', bare=True)}rik mahaian. "
              f"Ikusten dut: {colours['eu']}.",
    }


def unreachable_question(points: list[tuple[float, float]], item: dict) -> dict[str, str]:
    """Say that it is there and cannot be reached, and what would make it reachable.

    The advice is the useful part. "I cannot reach it" leaves the person with nothing to do;
    "move it into the working area" tells them what the robot wants. The two reasons are named
    differently because they need different responses: the arm being too short is the person's
    problem to solve by moving the object closer, and being outside the working area is a
    decision this demo made.

    The numbers go in as numbers and the sentence around them is built per language. Rendering
    the reason in English first and splicing it in produced "Veo un cubo azul, pero no puedo
    llegar: esta 35 cm from the base, past the 33 cm the arm can reach".
    """
    kind = item.get("kind") or "any"
    color = item.get("color")
    what = {lang: describe.phrase(kind, color, 1, lang) for lang in ("es", "en", "eu")}

    x, y = points[0] if points else (0.0, 0.0)
    facts = scene_spec.reach_facts(x, y)
    distance = facts["distance_m"] * 100.0
    limit = facts["limit_m"] * 100.0
    beyond_arm = facts["beyond_arm"]

    where = {
        "es": (f"esta a {distance:.0f} cm de la base y el brazo llega a {limit:.0f}"
               if beyond_arm
               else f"está en ({x:+.3f}, {y:+.3f}), fuera del área de trabajo"),
        "en": (f"it is {distance:.0f} cm from the base and the arm reaches {limit:.0f}"
               if beyond_arm
               else f"it is at ({x:+.3f}, {y:+.3f}), outside the working area"),
        "eu": (f"{distance:.0f} cm dago ohatik dago eta burua {limit:.0f} cm-rera iristen"
               if beyond_arm
               else f"({x:+.3f}, {y:+.3f}) dago, lanetik kanpoan"),
    }
    advice = {
        "es": "Muévelo más cerca." if beyond_arm else "Muévelo dentro del área de trabajo.",
        "en": "Move it closer." if beyond_arm else "Move it into the working area.",
        "eu": "Hurbildu gertuago." if beyond_arm else "Eman utzi zonalera eramanu.",
    }
    lead = {
        "es": f"Veo {what['es']}, pero no puedo llegar: {where['es']}.",
        "en": f"I can see {what['en']}, but I cannot reach it: {where['en']}.",
        "eu": f"Ikusten dut {what['eu']}, baina iristu ezin dut: {where['eu']}.",
    }
    return {lang: f"{lead[lang]} {advice[lang]}" for lang in ("es", "en", "eu")}


def ground(scene: Scene, item: dict) -> Grounding:
    """One item of a command, resolved against a scene.

    A position word never removes anything: it only puts the matches in an order. So the
    question is not "did the word leave one candidate" but "does the word actually
    separate them", which is what ``scene.separates`` answers. That distinction is the
    whole point: "the red ball on the left" has to pick one of two reds, while "the red
    ball on the left" when the two reds are side by side in depth has to ask, because
    the word the person used cannot mean anything here.
    """
    kind, color, position = item.get("kind"), item.get("color"), item.get("position")
    quantity = item.get("quantity")
    if position and position not in POSITIONS:
        return Grounding(status="unsupported_position", note=f"unknown position word {position!r}")

    matches = scene.candidates(kind, color)
    if not matches:
        return Grounding(status="not_found", question=not_found_question(scene, item))

    # ADDED FOR THE out_of_reach VARIANT: drop what the arm cannot get to.
    #
    # The robot can see more than it can touch, and when the only thing matching is out of
    # reach the honest answer is to say so rather than to plan a move that cannot happen. Two
    # different reasons come back from `why_unreachable` - beyond the arm's length, which is
    # physics, and outside the declared working area, which is policy - and the person is told
    # which, because "move it closer" is useful advice and "that is where I am allowed to work"
    # is not the same request.
    #
    # Where one candidate is reachable and another is not, the reachable one wins silently. A
    # question here would be pedantry: if the person said "the red cube" and there are two and
    # only one is in reach, the one they can have is the obvious answer.
    reachable, unreachable = [], []
    for candidate in matches:
        why = scene_spec.why_unreachable(candidate.pos_m[0], candidate.pos_m[1]) \
            if candidate.pos_m else "its position was not measured"
        if candidate.pos_m and not why:
            reachable.append(candidate)
        else:
            unreachable.append(candidate)

    if not reachable:
        points = [(c.pos_m[0], c.pos_m[1]) for c in unreachable if c.pos_m]
        return Grounding(
            status="not_reachable",
            candidates=unreachable,
            question=unreachable_question(points, item),
            note=scene_spec.why_unreachable(*points[0]) if points
            else "its position was not measured")

    matches = reachable

    ranked = scene.rank(matches, position)

    if position:
        if not scene.separates(matches, position):
            return Grounding(status="ambiguous", candidates=matches,
                             question=ambiguous_question(scene, matches, color),
                             note=f"{position!r} does not tell these {len(matches)} apart")
        return Grounding(status="ok_filtered", targets=ranked[:1], candidates=matches)

    if isinstance(quantity, int) and 1 < quantity <= len(matches):
        # "coge dos bolas rojas" with three on the table: the word named a count, so take
        # that many in rank order rather than asking. With exactly two on the table this
        # is the whole set and there is nothing to disambiguate.
        if quantity == len(matches):
            return Grounding(status="ok", targets=ranked, candidates=matches)
        return Grounding(status="ok_count", targets=ranked[:quantity], candidates=matches)

    if len(matches) == 1:
        return Grounding(status="ok", targets=matches, candidates=matches)

    return Grounding(status="ambiguous", candidates=matches,
                     question=ambiguous_question(scene, matches, color))


def ground_place(scene: Scene, place: dict | None) -> Grounding:
    """Resolve a destination, by kind and colour, and say which one if it cannot.

    Kind is tried first because "the box" is what people say, then colour, because
    three containers of one kind are told apart by colour and not by shape. Containers
    come out of perception with the kind ``box`` only when their footprint is square, so
    the colour fallback is not a nicety: it is the usual path.
    """
    if not place:
        return Grounding(status="no_place")
    kind, color = place.get("kind"), place.get("color")
    by_kind = scene.container(kind, color) if color else []
    if not by_kind:
        by_kind = scene.container(kind, None)
    if not by_kind and kind in ("any", "other", None, ""):
        by_kind = sorted(scene.containers, key=lambda c: -c.area_px)

    # ADDED FOR THE out_of_reach VARIANT: a destination the arm cannot get to is refused for
    # the same reason an object is. A tray outside the working area is as unreachable as a
    # ball outside it, and saying "nowhere to put it" when the tray is plainly visible would
    # be a worse answer than naming the tray and the reason.
    usable, unusable = [], []
    for tray in by_kind:
        why = scene_spec.why_unreachable(tray.pos_m[0], tray.pos_m[1]) \
            if tray.pos_m else "its position was not measured"
        if tray.pos_m and not why:
            usable.append(tray)
        else:
            unusable.append(tray)
    if by_kind and not usable:
        points = [(t.pos_m[0], t.pos_m[1]) for t in unusable if t.pos_m]
        return Grounding(
            status="not_reachable",
            candidates=unusable,
            question=unreachable_question(points, {"kind": kind or "any", "color": color}),
            note=scene_spec.why_unreachable(*points[0]) if points
            else "its position was not measured")
    by_kind = usable if usable else by_kind
    if len(by_kind) == 1:
        return Grounding(status="ok", place=by_kind[0])
    if not by_kind:
        names = ", ".join(sorted({f"{c.kind} {c.color}" for c in scene.containers})) or "none"
        return Grounding(
            status="not_found",
            question={
                "es": f"No encuentro ese sitio. Tengo: {names}.",
                "en": f"I cannot find that place. I have: {names}.",
                "eu": f"Ez dut aurkiten. Hau ditut: {names}.",
            },
        )
    return Grounding(
        status="ambiguous",
        candidates=by_kind,
        question={
            "es": f"Hay {len(by_kind)} sitios así. ¿Cuál?",
            "en": f"There are {len(by_kind)} places like that. Which one?",
            "eu": f"{len(by_kind)} toki hori daude. Zein?",
        },
    )


# The intents that must not move the robot, so their items are never looked up. The same
# list as command_model_v2.ITEMLESS, repeated rather than imported because importing that
# module loads torch and gliner2 for what is a list of strings.
#
# CHANGED IN THIS COPY: "sort" and "clear" REMOVED. Demo 03's command model could not
# represent a sort into matching receptacles or a clear with a filter, so both intents were
# itemless and grounding them against the scene was pointless. The v2 model does both, and
# planner.py needs their items resolved. They are still in command_model_v2.ITEMLESS? No -
# they are deliberately not; the two lists must agree.
NO_ACTION = {
    "stop", "repeat", "cancel", "answer_yes", "answer_no",
    "greeting", "off_topic", "not_doable", "question",
}


def ground_command(scene: Scene, command: dict) -> dict:
    """A whole command, item by item, plus its destination.

    Returns a dict rather than a single Grounding because a command can be partly
    resolved: "coge las dos bolas rojas y ponlas en la caja azul" is executable even
    if only one of the two balls is on the table, and demo 10 needs to be told which
    half failed.

    Intents in ``NO_ACTION`` are answered, not executed, so their items are not looked
    up at all. "¿Qué ves?" carries an item for every object it names; grounding it
    against the scene would ask "¿cuál?" about a question that was never ambiguous.
    """
    intent = command.get("intent")
    if intent in NO_ACTION:
        return {
            "intent": intent,
            "items": [],
            "place": Grounding(status="no_place"),
            "item_places": [],
            "executable": [],
            "complete": False,
            "action": False,
            "summary": f"{intent}: nothing to execute",
        }

    items = [ground(scene, item) for item in command.get("items", [])]
    place = ground_place(scene, command.get("place"))

    # CHANGED IN THIS COPY: resolve each item's OWN destination.
    #
    # "Pon el cubo rojo en la caja azul y la bola verde en la bandeja roja" has no
    # command-level place, because there are two. Demo 03's schema had exactly one place per
    # command, so there was nothing to resolve here; the v2 command model gives each item a
    # destination and this is where it becomes a Thing with a position the arm can aim at.
    #
    # An item with no destination of its own inherits the command's, which is every item of
    # every single-destination command. The list is parallel to "items" so a caller can index
    # both with the same number.
    raw_items = command.get("items", [])
    item_places: list[Grounding] = []
    for raw in raw_items:
        own = raw.get("place") if isinstance(raw, dict) else None
        item_places.append(ground_place(scene, own) if own else place)

    executable = [g for g in items if g.ok]
    return {
        "intent": intent,
        "items": items,
        "place": place,
        "item_places": item_places,
        "executable": [g.target_ids for g in executable],
        "complete": bool(items) and len(executable) == len(items),
        "action": True,
        "summary": " ".join(
            filter(None, [f"{len(executable)}/{len(items)} objects" if items else "", place.status])
        ),
    }
