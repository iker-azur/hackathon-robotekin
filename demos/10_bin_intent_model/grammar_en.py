"""English sentences for the bin-sorting intent dataset (see ``grammar_common.py`` for the labels)."""

from __future__ import annotations

from grammar_common import Ctx

VERBS = {
    "move": {
        "train": ["move", "put", "place", "carry", "take", "bring", "transfer", "shift", "deposit", "send", "store",
                  "transport", "drop", "set", "load"],
        "calib": ["relocate", "convey"],
        "test": ["reposition", "ferry", "deliver"],
    },
    "sort": {
        "train": ["sort", "classify", "separate", "organize", "arrange", "group", "distribute", "set apart", "split",
                  "put away", "file"],
        "calib": ["segregate", "divide"],
        "test": ["categorize", "triage", "sift"],
    },
    "pick": {
        "train": ["pick up", "grab", "take", "lift", "collect", "get", "fetch", "hold"],
        "calib": ["seize"],
        "test": ["snatch", "retrieve"],
    },
}
MODALS = ["can you", "could you", "would you", "will you", "can you please", "I want you to", "I need you to",
          "I'd like you to", "you should", "you have to", "you need to", "let's", "go ahead and", "please",
          "it's time to", "I want to", "we need to", "you must"]
MARKERS = ["no, rather", "sorry,", "I mean", "no wait,", "actually,", "no,", "scratch that,", "correction,"]

NOUNS = {  # singular, plural
    "train": [("piece", "pieces")] * 4 + [("part", "parts"), ("component", "components"), ("item", "items"),
                                          ("unit", "units"), ("object", "objects")],
    "calib": [("product", "products"), ("element", "elements")],
    "test": [("bit", "bits"), ("article", "articles"), ("workpiece", "workpieces")],
}
BAD_ADJ = {"train": ["bad"] * 3 + ["defective", "faulty", "damaged", "broken", "rejected", "wrong", "incorrect", "no-good"],
           "calib": ["flawed", "spoiled"], "test": ["botched", "substandard"]}
BAD_PHRASE = {"train": ["in bad condition", "with defects", "with a defect", "that are bad", "that are defective"],
              "calib": ["with flaws"], "test": ["in poor shape"]}
GOOD_ADJ = {"train": ["good"] * 3 + ["OK", "fine", "correct", "valid", "acceptable", "perfect", "approved"],
            "calib": ["flawless", "sound"], "test": ["pristine", "satisfactory"]}
GOOD_PHRASE = {"train": ["in good condition", "without defects", "with no defects", "that are good", "that are OK"],
               "calib": ["in perfect shape"], "test": ["in great condition"]}
BOLT_PRE = {"train": ["bolted"], "calib": ["bolt-equipped"], "test": ["bolt-fitted"]}
BOLT_POST = {  # (singular, plural)
    "train": [("with a bolt", "with bolts"), ("with a bolt", "with a bolt"), ("that has a bolt", "that have a bolt"),
              ("that has a bolt", "that have bolts"), ("fitted with a bolt", "fitted with bolts"),
              ("that carries a bolt", "that carry a bolt"), ("with a bolt on it", "with a bolt on them")],
    "calib": [("equipped with a bolt", "equipped with bolts")],
    "test": [("featuring a bolt", "featuring bolts"), ("that includes a bolt", "that include a bolt")],
}
NO_BOLT_PRE = {"train": ["unbolted"], "calib": ["bolt-less"], "test": ["bolt-free"]}
NO_BOLT_POST = {
    "train": [("without a bolt", "without bolts"), ("without a bolt", "without a bolt"), ("with no bolt", "with no bolt"),
              ("that has no bolt", "that have no bolt"), ("missing the bolt", "missing the bolt"),
              ("missing a bolt", "missing their bolt"), ("lacking a bolt", "lacking a bolt"),
              ("that does not have a bolt", "that do not have a bolt")],
    "calib": [("that lacks a bolt", "that lack a bolt")],
    "test": [("devoid of a bolt", "devoid of bolts"), ("that has no bolt fitted", "that have no bolt fitted")],
}
DETS = {
    "all": ["the"] * 5 + ["all the", "all the", "all of the", "those", "these", "all those", ""],
    "one": ["a", "the", "the", "that", "this", "any", "one", "a single"],
    "two": ["two", "two", "the two", "those two", "these two"],
    "three": ["three", "three", "the three", "those three"],
}

BOX_NOUNS = {"train": ["box"] * 6 + ["bin", "container", "crate"], "calib": ["bucket", "tub"], "test": ["receptacle", "carton"]}
BIG = {"train": ["big"] * 3 + ["large", "bigger", "larger", "biggest", "largest", "big-sized"],
       "calib": ["huge", "oversized"], "test": ["jumbo", "giant"]}
WHITE = {"train": ["white"] * 5 + ["white-colored"], "calib": ["snow-white"], "test": ["whitish"]}
OTHER_PRE = {"train": ["small", "little", "smaller", "black", "red", "green", "blue", "yellow", "left", "right",
                       "left-hand", "right-hand", "second", "first", "other", "third", "smallest"],
             "calib": ["medium", "gray"], "test": ["brown", "orange", "purple", "middle"]}
OTHER_POST = ["on the left", "on the right", "at the back", "at the front", "on top", "next to the robot"]
PREPS = {"move": ["to"] * 3 + ["into"] * 3 + ["in"] * 3 + ["inside"], "sort": ["into"] * 4 + ["in"] * 3 + ["to"] * 2,
         "pick": ["into"]}

PREFIXES = ["please, ", "hey, ", "hey robot, ", "robot, ", "okay, ", "alright, ", "now ", "right now ", "well, ",
            "let's see, ", "perfect, ", "so ", "hi, ", "ok, ", "first ", "and now ", "um, "]
SUFFIXES = [", please", " please", " thanks", ", thank you", " now", " right away", " quickly", " when you can",
            " if you can", " at once", " as soon as possible", " carefully"]

CONTROL = {
    "stop": {"train": ["stop", "stop now", "stop it", "stop everything", "halt", "freeze", "hold it", "wait stop",
                       "stop the robot", "that's enough", "enough", "enough already", "don't move", "stop moving",
                       "stop right now", "emergency stop", "emergency", "stop stop", "stop, stop", "abort the motion"],
             "calib": ["cease", "interrupt", "interrupt the movement", "pause everything"],
             "test": ["halt immediately", "suspend the motion", "stand still", "cut it out", "knock it off"]},
    "cancel": {"train": ["cancel", "cancel that", "cancel the order", "cancel it", "forget it", "forget what I said",
                         "forget that", "never mind", "ignore that", "ignore the order", "scratch that", "drop it",
                         "better not", "it doesn't matter", "delete the order", "let's leave it"],
               "calib": ["discard the order", "call it off", "disregard that"],
               "test": ["withdraw the order", "annul that", "belay that", "scrap it"]},
    "repeat": {"train": ["repeat", "repeat that", "repeat the order", "again", "do it again", "once more", "one more time",
                         "say it again", "repeat the last order", "same again", "do that again", "run it again",
                         "repeat what you just did", "again please"],
               "calib": ["do the same again", "try again"],
               "test": ["reiterate the order", "execute it once more", "redo it", "go again"]},
    "yes": {"train": ["yes", "yes please", "yeah", "yep", "okay", "ok", "sure", "of course", "correct", "that's right",
                      "exactly", "go ahead", "affirmative", "do it", "right", "that one", "yes, that one", "confirmed",
                      "yes, correct", "absolutely"],
            "calib": ["indeed", "I confirm", "yes, sure"],
            "test": ["precisely", "definitely", "uh-huh", "by all means"]},
    "no": {"train": ["no", "no thanks", "nope", "negative", "incorrect", "not that one", "that's not it", "not that",
                     "no, the other one", "not at all", "no no", "wrong", "that's wrong", "no, not that", "nah"],
           "calib": ["no way", "not really"],
           "test": ["certainly not", "absolutely not", "never", "no chance"]},
    "hello": {"train": ["hello", "hi", "hi robot", "hello robot", "good morning", "good afternoon", "good evening", "hey",
                        "hey robot", "how are you?", "what's up?", "how's it going?", "greetings", "goodbye", "bye",
                        "see you later", "see you", "thanks", "thank you", "thank you robot", "good job", "well done",
                        "nice work", "cheers"],
              "calib": ["howdy", "hi there", "thanks a lot"],
              "test": ["pleased to meet you", "so long", "have a nice day", "much obliged", "good night"]},
    "chat": {"train": ["what time is it?", "what's the weather like?", "tell me a joke", "what's your name?",
                       "who made you?", "play some music", "put on the radio", "play a song", "what day is it today?",
                       "what's the capital of France?", "what is two plus two?", "do you like football?",
                       "what are we having for dinner?", "turn off the lights", "turn on the light", "turn up the volume",
                       "call my mother", "sing a song", "it's really hot in here", "the exam was hard",
                       "who won the match?", "open the window", "set an alarm for seven", "it's going to rain tomorrow",
                       "I'm hungry", "can you dance?", "how old are you?", "translate this into Spanish",
                       "where is the bathroom?", "I need a coffee", "remind me to call the doctor", "close the door",
                       "what a nice day", "this is so boring", "can you help me with my homework?",
                       "move the volume up", "put the table for dinner", "put on the TV", "take this home by car",
                       "what do you think about traffic?", "lower the blinds", "how is the cat?"],
             "calib": ["which movie do you recommend?", "put on the news", "can I borrow a pen?"],
             "test": ["do you think it will snow?", "play a podcast", "what time do the shops close?",
                      "recommend me a book", "pizza tonight", "when are the holidays?"]},
}
STATIC_ASK = {
    "train": ["what do you see?", "what are you doing?", "what can you do?", "are you done?", "is that it?",
              "how much is left?", "what boxes are there?", "how many boxes are there?", "how many pieces are left?",
              "are you ready?", "can you hear me?", "what pieces are there?", "tell me what you see",
              "tell me how many pieces are left", "tell me what you are doing", "what kinds of pieces are there?",
              "what are the boxes called?", "are there any bad pieces?", "are there any pieces with a bolt?",
              "what have you done so far?", "which is the best box for the bad ones?"],
    "calib": ["how many pieces do you have in front of you?", "can you tell me what you see?"],
    "test": ["how is the sorting going?", "how much time is left?", "what is the status of everything?", "report what you see"],
}
ASK_TEMPLATES = {
    "train": ["how many {npb} are there?", "how many {npb} are left?", "how many {npb} have you moved?",
              "are there any {npb}?", "where is {box}?", "which one is {box}?", "which box do {npd} go in?",
              "what is in {box}?", "how many pieces are in {box}?", "tell me how many {npb} are left",
              "tell me where {box} is", "do you see {npd}?", "is there a {boxadj} box left?"],
    "calib": ["how many {npb} do you have in front of you?", "can you tell me how many {npb} there are?"],
    "test": ["can you tell me where {box} is?", "which of the boxes is {box}?", "I need to know how many {npb} are left",
             "do you know how many {npb} there are?"],
}
STATEMENTS = {
    "train": ["{npd} are already in {box}", "I saw {npb} in {box} yesterday", "there were {npb} before", "{box} is full",
              "{box} is empty", "I already moved {npd}", "there are a lot of {npb} today", "I like {npd}",
              "{npd} are so pretty", "we sorted {npd} yesterday", "we will sort {npd} tomorrow",
              "I moved {npd} to {box} earlier"],
    "calib": ["{box} fell over", "the other day there were {npb}"],
    "test": ["{npd} were in {box} last night", "{box} looks nice to me"],
}
DESTROY = {"train": ["throw", "toss", "smash", "break", "crush", "burn", "kick", "destroy", "hit", "trash"],
           "calib": ["wreck"], "test": ["shatter", "demolish"]}
DESTROY_TAIL = ["on the floor", "in the trash", "out the window", "against the wall", "away", "into the fire"]
NEG_INTROS = ["don't", "never", "do not", "please don't", "under no circumstances", "do not ever"]
IMPOSSIBLE = ["the car", "the chair", "the table", "the computer", "the door", "my phone", "the cat", "my brother",
              "the building", "the monitor", "the big box", "the white box", "the arm", "the camera", "the lamp",
              "the whole box", "both boxes"]


class English:
    code = "en"
    markers = MARKERS

    def np(self, ctx: Ctx, f: str, q: str, count: str, det: str | None = None, pron: bool = True):
        """The pieces: ``(text, None)``. ``det=""`` forces a bare plural ("bad pieces with a bolt")."""
        noun_sg, noun_pl = ctx.pick(NOUNS)
        plural = count != "one"
        if det is None:
            det = ctx.rng.choice(DETS["all" if count == "none" else count])
        if count == "one" and det == "":
            det = "the"
        noun = noun_pl if plural else noun_sg
        adj = ctx.pick(BAD_ADJ if q == "bad" else GOOD_ADJ) if q != "any" else ""
        phrase = ctx.pick(BAD_PHRASE if q == "bad" else GOOD_PHRASE) if q != "any" else ""
        rel_sg, rel_pl = ctx.pick(BOLT_POST if f == "bolt" else NO_BOLT_POST) if f != "any" else ("", "")
        rel = rel_pl if plural else rel_sg
        fpre = ctx.pick(BOLT_PRE if f == "bolt" else NO_BOLT_PRE) if f != "any" else ""
        qpost = ""
        if q != "any" and ctx.chance(0.3):  # "pieces in bad condition" instead of "bad pieces"
            adj, qpost = "", phrase
        if pron and (f != "any" or q != "any") and det and ctx.chance(0.12):
            head = det if det in ("those", "these", "all those") else "the"
            return " ".join(p for p in (head, adj, "ones" if plural else "one", rel or qpost) if p), None
        if f != "any" and q != "any":
            pattern = ctx.weighted({"std": 0.5, "pre2": 0.1, "that": 0.2, "and": 0.2})
            if pattern == "pre2":
                return " ".join(p for p in (det, fpre, adj, noun, qpost) if p), None
            quality = adj or qpost
            if pattern == "that":
                joiner = "" if quality.startswith(("in ", "with", "that")) else "that are"
                return " ".join(p for p in (det, noun, rel, joiner, quality) if p), None
            if pattern == "and":
                feature = "a bolt" if f == "bolt" else "no bolt"
                have = "have" if plural else "has"
                if quality.startswith(("in ", "with")):
                    return " ".join((det, noun, quality, "and", have, feature)), None
                quality = quality.replace("that are ", "")
                return " ".join((det, noun, "that are", quality, "and", have, feature)), None
        parts = [det, adj, noun, rel, qpost]
        if f != "any" and q == "any" and ctx.chance(0.15):
            parts = [det, fpre, noun]
        return " ".join(p for p in parts if p), None

    def box_np(self, ctx: Ctx, label: str) -> str:
        noun = ctx.pick(BOX_NOUNS)
        art = ctx.rng.choice(["the"] * 4 + ["that", "this"])
        if label == "other":
            if ctx.chance(0.25):
                return f"{art} {noun} {ctx.rng.choice(OTHER_POST)}"
            adj = ctx.pick(OTHER_PRE)
        else:
            adj = ctx.pick(BIG if label == "big" else WHITE)
        if ctx.chance(0.12):
            return f"{art} {adj} one"
        return f"{art} {adj} {noun}"

    def box_ph(self, ctx: Ctx, label: str, intent: str = "move") -> str:
        return f"{ctx.rng.choice(PREPS[intent])} {self.box_np(ctx, label)}"

    def action(self, ctx: Ctx, intent: str, f: str, q: str, box: str, count: str) -> str:
        verb = ctx.pick(VERBS[intent])
        bx = self.box_ph(ctx, box, intent) if box != "none" else ""
        np_text, _ = self.np(ctx, f, q, count)
        weights = {"imp": 0.44, "boxfirst": 0.06, "pronoun": 0.10, "modal": 0.30, "inf": 0.04}
        if not bx:
            weights.pop("boxfirst")
        if intent == "move" and bx:
            weights["fragment"] = 0.06
        frame = ctx.weighted(weights)
        join = lambda *p: " ".join(x for x in p if x)
        if frame == "imp":
            return join(verb, np_text, bx)
        if frame == "boxfirst":
            return join(bx + ",", verb, np_text)
        if frame == "pronoun":
            text, _ = self.np(ctx, f, q, count, pron=False)
            return join(text + ",", verb, "them" if count != "one" else "it", bx)
        if frame == "modal":
            modal = ctx.rng.choice(MODALS)
            text = join(modal, verb, np_text, bx)
            return text + "?" if modal.startswith(("can you", "could you", "would you", "will you")) else text
        if frame == "inf":
            return join("to", verb, np_text, bx)
        return join(np_text, bx)  # fragment: "the bad pieces to the white box"

    def fragment(self, ctx: Ctx, f: str, q: str, box: str, count: str) -> str:
        return " ".join((self.np(ctx, f, q, count)[0], self.box_ph(ctx, box, "move")))

    def reply(self, ctx: Ctx, f: str, q: str, box: str, count: str) -> str:
        if box != "none" and f == "any" and q == "any" and count == "none":
            lead = ctx.rng.choice(["", "", "", "yes, ", "well ", "rather ", "no, ", "the "])
            if lead == "the ":
                adj = ctx.pick(BIG if box == "big" else WHITE if box == "white" else OTHER_PRE)
                return f"the {adj} one"
            return lead + self.box_ph(ctx, box, "move")
        if box == "none" and f == "any" and q == "any":
            return ctx.rng.choice({"all": ["all of them", "all", "everything", "all of it"],
                                   "one": ["one", "just one", "a single one", "that one"],
                                   "two": ["two", "the two", "just two", "both"],
                                   "three": ["three", "the three", "just three"]}[count])
        return self.np(ctx, f, q, count)[0]

    def both(self, ctx: Ctx) -> str:
        (f1, q1), (f2, q2) = self._two_groups(ctx)
        b1, b2 = ("big", "white") if ctx.chance(0.5) else ("white", "big")
        verb = ctx.pick(VERBS[ctx.rng.choice(["move", "sort"])])
        a, _ = self.np(ctx, f1, q1, "all")
        if ctx.chance(0.2):
            second = ctx.rng.choice(["the rest", "the others", "the remaining ones", "everything else"])
        else:
            second, _ = self.np(ctx, f2, q2, "all", det=ctx.rng.choice(["the", "all the"]))
        glue = ctx.rng.choice(["and", "and", "and", ", and", ";", "and then", "and after that"])
        first = " ".join((verb, a, self.box_ph(ctx, b1, "move")))
        return f"{first}{glue if glue in (';', ', and') else ' ' + glue} {second} {self.box_ph(ctx, b2, 'move')}"

    @staticmethod
    def _two_groups(ctx: Ctx):
        pairs = [(("bolt", "any"), ("no bolt", "any")), (("any", "bad"), ("any", "good")), (("bolt", "any"), ("any", "bad")),
                 (("any", "bad"), ("bolt", "any")), (("bolt", "good"), ("any", "bad")), (("no bolt", "any"), ("any", "good"))]
        return ctx.rng.choice(pairs)

    def control(self, ctx: Ctx, intent: str) -> str:
        slot = CONTROL[intent]
        if ctx.tier != "train":
            ctx.novel = True
            return ctx.rng.choice(slot[ctx.tier])
        return ctx.rng.choice(slot["train"])

    def _bare(self, ctx: Ctx):
        f, q = ctx.rng.choice([("bolt", "any"), ("no bolt", "any"), ("any", "bad"), ("any", "good"), ("bolt", "bad"),
                               ("no bolt", "good"), ("any", "any")])
        return self.np(ctx, f, q, "all", det="", pron=False)

    def ask(self, ctx: Ctx) -> str:
        if ctx.chance(0.4):
            return ctx.pick(STATIC_ASK)
        template = ctx.pick(ASK_TEMPLATES)
        npb, _ = self._bare(ctx)
        npd, _ = self.np(ctx, *ctx.rng.choice([("bolt", "any"), ("any", "bad"), ("any", "good"), ("no bolt", "any")]),
                         "all", det="the", pron=False)
        box = self.box_np(ctx, ctx.rng.choice(["big", "white", "white", "big", "other"]))
        return template.format(npb=npb, npd=npd, box=box, boxadj=ctx.pick(ctx.rng.choice([BIG, WHITE])))

    def refuse(self, ctx: Ctx) -> str:
        kind = ctx.weighted({"destroy": 0.22, "negate": 0.30, "thing": 0.24, "statement": 0.24})
        f, q = ctx.rng.choice([("any", "any"), ("bolt", "any"), ("any", "bad"), ("no bolt", "any"), ("any", "good")])
        npt, _ = self.np(ctx, f, q, "all", pron=False)
        box = ctx.rng.choice(["big", "white"])
        if kind == "destroy":
            return " ".join((ctx.pick(DESTROY), npt, ctx.rng.choice(DESTROY_TAIL)))
        if kind == "negate":
            verb = ctx.pick(VERBS[ctx.rng.choice(["move", "sort", "pick"])])
            bx = self.box_ph(ctx, box, "move") if ctx.chance(0.5) else ""
            style = ctx.weighted({"imp": 0.5, "nothing": 0.2, "want": 0.15, "need": 0.15})
            if style == "imp":
                return " ".join(p for p in (ctx.rng.choice(NEG_INTROS), verb, npt, bx) if p)
            if style == "nothing":
                return ctx.rng.choice(["don't touch anything", "don't move anything", "do nothing", "don't pick anything",
                                       "don't put anything in " + self.box_np(ctx, box)])
            if style == "want":
                return " ".join(p for p in ("I don't want you to", verb, npt, bx) if p)
            return " ".join(p for p in (ctx.rng.choice(["no need to", "you don't have to", "there's no need to"]),
                                        verb, npt, bx) if p)
        if kind == "thing":
            verb = ctx.pick(VERBS[ctx.rng.choice(["move", "pick"])])
            bx = self.box_ph(ctx, box, "move") if ctx.chance(0.5) else ""
            return " ".join(p for p in (verb, ctx.rng.choice(IMPOSSIBLE), bx) if p)
        template = ctx.pick(STATEMENTS)
        npd, _ = self.np(ctx, f, q, "all", det="the", pron=False)
        npb, _ = self.np(ctx, f, q, "all", det="", pron=False)
        return template.format(npd=npd, npb=npb, box=self.box_np(ctx, box))

    def wrap(self, ctx: Ctx, text: str, intent: str) -> str:
        busy = intent in ("move", "sort", "pick", "both", "ask", "refuse")
        if ctx.chance(0.30 if busy else 0.10):
            text = ctx.rng.choice(PREFIXES) + text
        if ctx.chance(0.22 if busy else 0.08) and not text.endswith("?"):
            text = text + ctx.rng.choice(SUFFIXES)
        return text
