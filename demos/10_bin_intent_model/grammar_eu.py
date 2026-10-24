"""Basque sentences for the bin-sorting intent dataset (see ``grammar_common.py`` for the labels).

Written by hand for this project and not checked by a native speaker: the word forms are explicit
(every adjective and box noun lists its absolutive, allative -ra and inessive -n forms) so the
morphology cannot drift, but the phrasing is a synthetic approximation of spoken Basque.
"""

from __future__ import annotations

from grammar_common import Ctx


def _verbs(rows):
    out = []
    for row in rows:
        part, nom, ine = row.split("|")
        out.append({"part": part, "nom": nom, "ine": ine == "1"})
    return out


VERBS = {
    "move": {"train": _verbs(["mugitu|mugitzea|0", "eraman|eramatea|0", "sartu|sartzea|1", "jarri|jartzea|1", "utzi|uztea|1",
                              "pasatu|pasatzea|0", "kokatu|kokatzea|1", "gorde|gordetzea|1", "mugitu|mugitzea|0",
                              "eraman|eramatea|0"]),
             "calib": _verbs(["garraiatu|garraiatzea|0", "bidali|bidaltzea|0"]),
             "test": _verbs(["lekualdatu|lekualdatzea|0", "aldatu|aldatzea|0"])},
    "sort": {"train": _verbs(["sailkatu|sailkatzea|1", "banandu|banantzea|1", "bereizi|bereiztea|1", "antolatu|antolatzea|1",
                              "ordenatu|ordenatzea|1", "taldekatu|taldekatzea|1", "banatu|banatzea|1",
                              "sailkatu|sailkatzea|1"]),
             "calib": _verbs(["iragazi|iragaztea|1", "zatitu|zatitzea|1"]),
             "test": _verbs(["multzokatu|multzokatzea|1", "kategorizatu|kategorizatzea|1"])},
    "pick": {"train": _verbs(["hartu|hartzea|0", "jaso|jasotzea|0", "harrapatu|harrapatzea|0", "hartu|hartzea|0"]),
             "calib": _verbs(["atzeman|atzematea|0"]), "test": _verbs(["eskuratu|eskuratzea|0", "bildu|biltzea|0"])},
}
AUX_PL = ["itzazu", "itzazu", "", "", "mesedez"]
AUX_SG = ["ezazu", "ezazu", "", "", "mesedez"]
MODAL_PL = ["nahi ditut", "behar dituzu", "ahal dituzu?", "ditzakezu?", "nahiko nituzke", "nahi ditugu"]
MODAL_SG = ["nahi dut", "behar duzu", "ahal duzu?", "dezakezu?", "nahiko nuke", "nahi dugu"]
MARKERS = ["ez, hobe", "barkatu", "hobe", "esan nahi dut", "ez, itxaron", "zuzenketa", "ez, ez", "alegia"]

NOUNS = {  # stem, singular absolutive, plural absolutive
    "train": [("pieza", "pieza", "piezak")] * 4 + [("osagai", "osagaia", "osagaiak"), ("elementu", "elementua", "elementuak"),
                                                    ("unitate", "unitatea", "unitateak"), ("objektu", "objektua", "objektuak")],
    "calib": [("produktu", "produktua", "produktuak")],
    "test": [("zati", "zatia", "zatiak"), ("txapa", "txapa", "txapak")],
}
BAD = {  # (stem, singular, plural)
    "train": [("txar", "txarra", "txarrak")] * 3 + [("akastun", "akastuna", "akastunak"), ("hondatu", "hondatua", "hondatuak"),
                                                     ("matxuratu", "matxuratua", "matxuratuak"), ("oker", "okerra", "okerrak"),
                                                     ("baliogabe", "baliogabea", "baliogabeak")],
    "calib": [("apurtu", "apurtua", "apurtuak")],
    "test": [("baztertu", "baztertua", "baztertuak")],
}
BAD_REL = {"train": [("egoera txarrean dagoen", "egoera txarrean dauden"), ("akatsak dituen", "akatsak dituzten"),
                     ("gaizki dagoen", "gaizki dauden")],
           "calib": [("hutsegiteak dituen", "hutsegiteak dituzten")], "test": [("matxurak dituen", "matxurak dituzten")]}
GOOD = {
    "train": [("on", "ona", "onak")] * 3 + [("zuzen", "zuzena", "zuzenak"), ("egoki", "egokia", "egokiak"),
                                             ("perfektu", "perfektua", "perfektuak"), ("onargarri", "onargarria", "onargarriak")],
    "calib": [("bikain", "bikaina", "bikainak")], "test": [("oso", "osoa", "osoak")],
}
GOOD_REL = {"train": [("egoera onean dagoen", "egoera onean dauden"), ("akatsik gabeko", "akatsik gabeko"),
                      ("ondo dagoen", "ondo dauden")],
            "calib": [("guztiz zuzen dagoen", "guztiz zuzen dauden")],
            "test": [("inolako akatsik ez duen", "inolako akatsik ez duten")]}
BOLT_REL = {  # (singular, plural)
    "train": [("perno duen", "perno duten"), ("pernoa duen", "pernoa duten"), ("pernodun", "pernodun"),
              ("perno bat duen", "perno bat duten"), ("pernoz hornitutako", "pernoz hornitutako"),
              ("pernoarekin dagoen", "pernoarekin dauden"), ("perno duen", "perno duten")],
    "calib": [("pernoa daraman", "pernoa daramaten")],
    "test": [("pernoa txertatuta duen", "pernoa txertatuta duten")],
}
NO_BOLT_REL = {
    "train": [("perno gabeko", "perno gabeko"), ("pernorik ez duen", "pernorik ez duten"),
              ("perno gabe dagoen", "perno gabe dauden"), ("pernoa falta zaion", "pernoa falta zaien"),
              ("pernorik gabeko", "pernorik gabeko"), ("pernorik ez daukan", "pernorik ez daukaten")],
    "calib": [("pernoa ez daraman", "pernoa ez daramaten")],
    "test": [("pernorik jarrita ez duen", "pernorik jarrita ez duten")],
}
NUMERALS = {"two": ["bi", "bi"], "three": ["hiru", "hiru"]}
SUFFIX_WORD = {"guztiak": "guztiak", "horiek": "horiek", "bat": "bat", "hori": "hori", "num": ""}

BOX_NOUNS = {  # stem, absolutive, allative, inessive
    "train": [("kutxa", "kutxa", "kutxara", "kutxan")] * 6 + [("edukiontzi", "edukiontzia", "edukiontzira", "edukiontzian"),
                                                               ("ontzi", "ontzia", "ontzira", "ontzian"),
                                                               ("kaxa", "kaxa", "kaxara", "kaxan")],
    "calib": [("gordailu", "gordailua", "gordailura", "gordailuan"), ("kutxatila", "kutxatila", "kutxatilara", "kutxatilan")],
    "test": [("kutxatxo", "kutxatxoa", "kutxatxora", "kutxatxoan"), ("kontenedore", "kontenedorea", "kontenedorera", "kontenedorean")],
}
BIG = {"train": [("handi", "handia", "handira", "handian")] * 5 + [("handiago", "handiagoa", "handiagora", "handiagoan"),
                                                                    ("handien", "handiena", "handienera", "handienean")],
       "calib": [("erraldoi", "erraldoia", "erraldoira", "erraldoian"), ("zabal", "zabala", "zabalera", "zabalean")],
       "test": [("tamaina handiko", "tamaina handikoa", "tamaina handikora", "tamaina handikoan")]}
WHITE = {"train": [("zuri", "zuria", "zurira", "zurian")] * 6, "calib": [("zuri-zuri", "zuri-zuria", "zuri-zurira", "zuri-zurian")],
         "test": [("kolore zuriko", "kolore zuriko", "kolore zurikora", "kolore zurikoan")]}
OTHER = {
    "train": [("txiki", "txikia", "txikira", "txikian"), ("beltz", "beltza", "beltzera", "beltzean"),
              ("gorri", "gorria", "gorrira", "gorrian"), ("berde", "berdea", "berdera", "berdean"),
              ("urdin", "urdina", "urdinera", "urdinean"), ("hori", "horia", "horira", "horian"),
              ("ezkerreko", "ezkerrekoa", "ezkerrekora", "ezkerrekoan"), ("eskuineko", "eskuinekoa", "eskuinekora", "eskuinekoan"),
              ("txikiago", "txikiagoa", "txikiagora", "txikiagoan"), ("atzeko", "atzekoa", "atzekora", "atzekoan")],
    "calib": [("ertain", "ertaina", "ertainera", "ertainean"), ("gris", "grisa", "grisera", "grisean")],
    "test": [("marroi", "marroia", "marroira", "marroian"), ("laranja", "laranja", "laranjara", "laranjan"),
             ("more", "morea", "morera", "morean"), ("erdiko", "erdikoa", "erdikora", "erdikoan")],
}
ORDINALS = ["bigarren", "lehen", "hirugarren", "beste"]

PREFIXES = ["mesedez, ", "kaixo, ", "robot, ", "ondo, ", "orain ", "orain bertan ", "tira, ", "ea, ", "bale, ", "beno, ",
            "lehenik ", "eta orain ", "e, "]
SUFFIXES = [" mesedez", " orain", " azkar", " ahal baduzu", " arretaz", " eskerrik asko", " lehenbailehen", " orain bertan"]

CONTROL = {
    "stop": {"train": ["gelditu", "gelditu orain", "gelditu ezazu", "stop", "geldi", "nahikoa", "nahikoa da", "utzi mugitzen",
                       "ez mugitu", "ez jarraitu", "larrialdia", "gelditu dena", "gelditu robota", "geldi geldi",
                       "gelditu berehala", "gelditu orain bertan", "stop stop"],
             "calib": ["eten", "eten mugimendua", "gelditu mugimendua"],
             "test": ["geratu geldirik", "gelditzeko", "ez segi", "gelditu behingoz"]},
    "cancel": {"train": ["ezeztatu", "ezeztatu agindua", "ahaztu", "ahaztu hori", "ahaztu esandakoa", "utzi hori", "ez axola",
                         "hobe ez", "berdin da", "bertan behera utzi", "bertan behera", "baliogabetu agindua",
                         "ezabatu agindua", "utzi alde batera"],
               "calib": ["baztertu agindua", "kendu agindua"],
               "test": ["ahaztu agindu hori", "ez egin kasurik", "deuseztatu agindua"]},
    "repeat": {"train": ["errepikatu", "errepikatu hori", "errepikatu agindua", "berriro", "berriz", "beste behin",
                         "egin berriro", "egin berriz", "berriro egin", "behin gehiago", "errepikatu azken agindua",
                         "berdin berriro", "esan berriro"],
               "calib": ["egin berdin berriro", "saiatu berriro"],
               "test": ["errepika ezazu", "exekutatu berriro", "ekin berriro"]},
    "yes": {"train": ["bai", "bai mesedez", "ados", "zuzen", "hori da", "hori bera", "aurrera", "noski", "egin ezazu", "ondo",
                      "oso ondo", "ok", "bai hori da", "hala da", "bai, zuzen"],
            "calib": ["horixe", "baiezten dut"],
            "test": ["jakina", "zalantzarik gabe", "tira ba", "bai noski"]},
    "no": {"train": ["ez", "ez hori", "ezetz", "ez da hori", "hori ez", "bestea", "ez inola ere", "ez eskerrik asko", "ezta ere",
                     "ez horrela", "ez, ez", "ez, bestea"],
           "calib": ["ez horixe", "ezta pentsatu ere"],
           "test": ["inola ere ez", "sekula ez", "ez ba"]},
    "hello": {"train": ["kaixo", "kaixo robot", "egun on", "arratsalde on", "gabon", "agur", "zer moduz?", "zer moduz zaude?",
                        "eskerrik asko", "mila esker", "bihar arte", "gero arte", "ondo egina", "lan ona", "hurrengora arte",
                        "eskerrik asko robot", "kaixo, zer moduz?"],
              "calib": ["ongi etorri", "eskerrak"],
              "test": ["pozten naiz zu ezagutzeaz", "egun ona izan", "eskerrik asko guztiagatik"]},
    "chat": {"train": ["zer ordu da?", "zer eguraldi dago?", "kontatu txiste bat", "nola duzu izena?", "nork egin zintuen?",
                       "jarri musika", "jarri irratia", "jarri abesti bat", "zer egun da gaur?", "zein da Frantziako hiriburua?",
                       "zenbat da bi gehi bi?", "gustatzen zaizu futbola?", "zer afalduko dugu?", "itzali argiak", "piztu argia",
                       "igo bolumena", "deitu nire amari", "abestu abesti bat", "bero handia dago hemen",
                       "azterketa zaila izan da", "nork irabazi du partida?", "ireki leihoa", "jarri esnagailua zazpietan",
                       "bihar euria egingo du", "goseak nago", "dantzatzen dakizu?", "zenbat urte dituzu?",
                       "itzuli hau ingelesera", "non dago komuna?", "kafe bat behar dut", "gogorarazi medikuari deitzea",
                       "itxi atea", "egun polita da", "hau aspergarria da", "lagunduko didazu etxerako lanekin?",
                       "jarri telebista", "eraman hau etxera autoz", "zer iruditzen zaizu trafikoa?", "jaitsi pertsiana",
                       "zer moduz dago katua?", "mugitu bolumena", "jarri mahaia afaltzeko"],
             "calib": ["zein filma gomendatzen duzu?", "jarri albisteak", "bolaluma bat uzten didazu?"],
             "test": ["elurra egingo duela uste duzu?", "jarri podcast bat", "noiz ixten dituzte dendak?",
                      "gomendatu liburu bat", "gaur pizza", "noiz dira oporrak?"]},
}
STATIC_ASK = {
    "train": ["zer ikusten duzu?", "zer egiten ari zara?", "zer egin dezakezu?", "amaitu duzu?", "prest zaude?",
              "entzuten nauzu?", "zenbat falta da?", "zer kutxa daude?", "zenbat kutxa daude?", "zenbat pieza geratzen dira?",
              "zer pieza daude?", "esan zer ikusten duzun", "esan zenbat pieza geratzen diren", "kontatu zer egiten ari zaren",
              "zer pieza mota daude?", "nola deitzen dira kutxak?", "pieza txarrik badago?", "pernodun piezarik badago?",
              "zer egin duzu orain arte?", "zein da kutxarik onena pieza txarrentzat?"],
    "calib": ["zenbat pieza dituzu aurrean?", "esango didazu zer ikusten duzun?"],
    "test": ["nola doa sailkapena?", "zenbat denbora geratzen da?", "zer egoeratan dago dena?", "jakinarazi zer ikusten duzun"],
}
ASK_TEMPLATES = {
    "train": ["zenbat {npn} daude?", "zenbat {npn} geratzen dira?", "zenbat {npn} mugitu dituzu?", "non dago {box}?",
              "zein da {box}?", "zein kutxatara doaz {npd}?", "zer dago {boxin}?", "zenbat pieza daude {boxin}?",
              "esan zenbat {npn} geratzen diren", "esan non dagoen {box}", "ikusten dituzu {npd}?"],
    "calib": ["zenbat {npn} dituzu aurrean?", "esango didazu zenbat {npn} dauden?"],
    "test": ["esango didazu non dagoen {box}?", "jakin nahi dut zenbat {npn} geratzen diren", "badakizu zenbat {npn} dauden?"],
}
STATEMENTS = {
    "train": ["{npd} dagoeneko {boxin} daude", "atzo {npd} ikusi nituen {boxin}", "lehen {npn} batzuk zeuden",
              "{box} beteta dago", "{box} hutsik dago", "dagoeneko mugitu ditut {npd}", "gustatzen zaizkit {npd}",
              "{npd} oso politak dira", "atzo {npd} sailkatu genituen", "bihar {npd} sailkatuko ditugu",
              "lehenago {npd} {boxto} eraman nituen"],
    "calib": ["{box} erori da", "tarteko egunean {npn} batzuk zeuden"],
    "test": ["{npd} {boxin} zeuden bart", "{box} polita iruditzen zait"],
}
DESTROY = {"train": ["bota", "apurtu", "txikitu", "hautsi", "erre", "zapaldu", "suntsitu", "jaurti"],
           "calib": ["birrindu"], "test": ["xehatu", "kolpatu"]}
DESTROY_TAIL = ["lurrera", "zaborrontzira", "leihotik kanpora", "paretaren kontra", "urrutira", "sutara"]
IMPOSSIBLE = ["autoa", "aulkia", "mahaia", "ordenagailua", "atea", "nire telefonoa", "katua", "nire anaia", "eraikina",
              "monitorea", "kutxa handia", "kutxa zuria", "besoa", "kamera", "lanpara", "kutxa osoa", "bi kutxak"]


class Basque:
    code = "eu"
    markers = MARKERS

    # -- noun phrases -------------------------------------------------------------------------------------

    def np(self, ctx: Ctx, f: str, q: str, count: str, style: str | None = None, pron: bool = True):
        """The pieces: ``(text, None)``. ``style="num"`` gives the bare form used after "zenbat" ("pieza txar")."""
        stem, sg, pl = ctx.pick(NOUNS)
        adj = ctx.pick(BAD if q == "bad" else GOOD) if q != "any" else None
        q_rel = ctx.pick(BAD_REL if q == "bad" else GOOD_REL) if q != "any" else None
        rel = ctx.pick(BOLT_REL if f == "bolt" else NO_BOLT_REL) if f != "any" else None
        plural = count != "one"
        if style is None:
            if count == "all":
                style = ctx.weighted({"def": 0.55, "guztiak": 0.33, "horiek": 0.12})
            elif count == "one":
                style = ctx.weighted({"bat": 0.45, "hori": 0.30, "def": 0.25})
            else:
                style = "numeral"
        rel_text = (rel[1] if plural else rel[0]) if rel else ""

        if pron and style == "def" and (f != "any" or q != "any") and ctx.chance(0.12):
            # no noun at all: "txarrak", "perno dutenak", "perno duten txarrak"
            if adj and rel:
                return f"{rel_text} {adj[2 if plural else 1]}", None
            if adj:
                return adj[2 if plural else 1], None
            return rel_text + ("ak" if plural else "a"), None

        def noun_with(entry):
            """The noun with an optional adjective; the article sits on the last word of the phrase."""
            if style == "def":
                return f"{stem} {entry[2 if plural else 1]}" if entry else (pl if plural else sg)
            stem_adj = " ".join(p for p in (stem, entry[0] if entry else "") if p)
            if style == "numeral":
                return f"{ctx.rng.choice(NUMERALS[count])} {stem_adj}"
            return " ".join(p for p in (stem_adj, SUFFIX_WORD[style]) if p)

        if f != "any" and q != "any" and ctx.chance(0.4):
            q_text = q_rel[1] if plural else q_rel[0]
            if ctx.chance(0.5):
                return f"{q_text} eta {rel_text} {noun_with(None)}", None
            return f"{rel_text} eta {q_text} {noun_with(None)}", None
        return " ".join(p for p in (rel_text, noun_with(adj)) if p), None

    def _box_forms(self, ctx: Ctx, label: str):
        """The box named three ways: absolutive, allative (-ra) and inessive (-n)."""
        stem, ab, to, ine = ctx.pick(BOX_NOUNS)
        if label == "other" and ctx.chance(0.25):
            ordinal = ctx.rng.choice(ORDINALS)
            return f"{ordinal} {ab}", f"{ordinal} {to}", f"{ordinal} {ine}"
        slot = {"big": BIG, "white": WHITE, "other": OTHER}[label]
        _, aab, ato, aine = ctx.pick(slot)
        if ctx.chance(0.12):  # "handira": the noun is understood
            return aab, ato, aine
        return f"{stem} {aab}", f"{stem} {ato}", f"{stem} {aine}"

    def box_np(self, ctx: Ctx, label: str) -> str:
        return self._box_forms(ctx, label)[0]

    def box_ph(self, ctx: Ctx, label: str, intent: str = "move", ine: bool = True) -> str:
        _, to, inessive = self._box_forms(ctx, label)
        return inessive if ine and ctx.chance(0.4) else to

    # -- action sentences -----------------------------------------------------------------------------------

    def action(self, ctx: Ctx, intent: str, f: str, q: str, box: str, count: str) -> str:
        verb = ctx.pick(VERBS[intent])
        bx = self.box_ph(ctx, box, intent, ine=verb["ine"]) if box != "none" else ""
        np_text, _ = self.np(ctx, f, q, count)
        plural = count != "one"
        aux = ctx.rng.choice(AUX_PL if plural else AUX_SG)
        modal = ctx.rng.choice(MODAL_PL if plural else MODAL_SG)
        weights = {"init": 0.28, "sov": 0.30, "modal": 0.18, "init_aux": 0.06, "topic": 0.06}
        if bx:
            weights["boxfirst"] = 0.08
            if intent == "move":
                weights["fragment"] = 0.06
        frame = ctx.weighted(weights)
        join = lambda *p: " ".join(x for x in p if x)
        if frame == "init":
            return join(verb["part"], np_text, bx)
        if frame == "sov":
            return join(np_text, bx, verb["part"], aux)
        if frame == "modal":
            return join(np_text, bx, verb["part"], modal)
        if frame == "init_aux":
            return join(verb["part"], aux if aux != "mesedez" else "", np_text, bx)
        if frame == "topic":
            return join(np_text + ":", verb["part"], aux, bx)
        if frame == "boxfirst":
            return join(bx, np_text, verb["part"], aux)
        return join(np_text, bx)  # fragment: "pieza txarrak kutxa zurira"

    def fragment(self, ctx: Ctx, f: str, q: str, box: str, count: str) -> str:
        return " ".join((self.np(ctx, f, q, count)[0], self.box_ph(ctx, box, "move", ine=False)))

    def reply(self, ctx: Ctx, f: str, q: str, box: str, count: str) -> str:
        if box != "none" and f == "any" and q == "any" and count == "none":
            absolutive, to, inessive = self._box_forms(ctx, box)
            lead = ctx.rng.choice(["", "", "", "bai, ", "hobe ", "ez, ", "beno "])
            return lead + ctx.rng.choice([absolutive, to, inessive, to])
        if box == "none" and f == "any" and q == "any":
            return ctx.rng.choice({"all": ["denak", "guztiak", "guztiak batera", "dena"],
                                   "one": ["bat", "bakarra", "bat bakarrik", "hori"],
                                   "two": ["bi", "biak", "bi bakarrik"],
                                   "three": ["hiru", "hiruak", "hiru bakarrik"]}[count])
        return self.np(ctx, f, q, count)[0]

    def both(self, ctx: Ctx) -> str:
        (f1, q1), (f2, q2) = self._two_groups(ctx)
        b1, b2 = ("big", "white") if ctx.chance(0.5) else ("white", "big")
        verb = ctx.pick(VERBS[ctx.rng.choice(["move", "sort"])])
        a, _ = self.np(ctx, f1, q1, "all", style="def")
        if ctx.chance(0.2):
            second = ctx.rng.choice(["gainerakoak", "besteak", "geratzen direnak", "gainerako guztiak"])
        else:
            second, _ = self.np(ctx, f2, q2, "all", style="def")
        first = f"{a} {self.box_ph(ctx, b1, 'move', ine=False)}"
        rest = f"{second} {self.box_ph(ctx, b2, 'move', ine=False)}"
        if ctx.chance(0.5):
            return f"{verb['part']} {first} eta {rest}"
        return f"{first} eta {rest} {verb['part']} {ctx.rng.choice(['itzazu', 'itzazu', ''])}".strip()

    @staticmethod
    def _two_groups(ctx: Ctx):
        pairs = [(("bolt", "any"), ("no bolt", "any")), (("any", "bad"), ("any", "good")), (("bolt", "any"), ("any", "bad")),
                 (("any", "bad"), ("bolt", "any")), (("bolt", "good"), ("any", "bad")), (("no bolt", "any"), ("any", "good"))]
        return ctx.rng.choice(pairs)

    # -- everything that moves nothing -------------------------------------------------------------------

    def control(self, ctx: Ctx, intent: str) -> str:
        slot = CONTROL[intent]
        if ctx.tier != "train":
            ctx.novel = True
            return ctx.rng.choice(slot[ctx.tier])
        return ctx.rng.choice(slot["train"])

    @staticmethod
    def _descriptor(ctx: Ctx):
        return ctx.rng.choice([("bolt", "any"), ("no bolt", "any"), ("any", "bad"), ("any", "good"), ("bolt", "bad"),
                               ("no bolt", "good"), ("any", "any")])

    def ask(self, ctx: Ctx) -> str:
        if ctx.chance(0.4):
            return ctx.pick(STATIC_ASK)
        template = ctx.pick(ASK_TEMPLATES)
        npn, _ = self.np(ctx, *self._descriptor(ctx), "all", style="num", pron=False)
        npd, _ = self.np(ctx, *self._descriptor(ctx), "all", style="def", pron=False)
        absolutive, _, inessive = self._box_forms(ctx, ctx.rng.choice(["big", "white", "white", "big", "other"]))
        return template.format(npn=npn, npd=npd, box=absolutive, boxin=inessive)

    def refuse(self, ctx: Ctx) -> str:
        kind = ctx.weighted({"destroy": 0.22, "negate": 0.30, "thing": 0.24, "statement": 0.24})
        f, q = ctx.rng.choice([("any", "any"), ("bolt", "any"), ("any", "bad"), ("no bolt", "any"), ("any", "good")])
        npd, _ = self.np(ctx, f, q, "all", style="def", pron=False)
        absolutive, to, inessive = self._box_forms(ctx, ctx.rng.choice(["big", "white"]))
        if kind == "destroy":
            part, tail = ctx.pick(DESTROY), ctx.rng.choice(DESTROY_TAIL)
            return f"{part} {npd} {tail}" if ctx.chance(0.5) else f"{npd} {tail} {part} {ctx.rng.choice(['itzazu', ''])}".strip()
        if kind == "negate":
            verb = ctx.pick(VERBS[ctx.rng.choice(["move", "sort", "pick"])])
            bx = to if ctx.chance(0.5) else ""
            style = ctx.weighted({"imp": 0.45, "nothing": 0.2, "want": 0.2, "need": 0.15})
            if style == "imp":
                if ctx.chance(0.5):
                    return " ".join(p for p in ("ez", verb["part"], npd, bx) if p)
                return " ".join(p for p in ("ez itzazu", npd, bx, verb["part"]) if p)
            if style == "nothing":
                return ctx.rng.choice(["ez ukitu ezer", "ez mugitu ezer", "ez egin ezer", "ez hartu ezer", f"ez jarri ezer {inessive}"])
            if style == "want":
                return " ".join(p for p in ("ez dut nahi", npd, bx, verb["nom"]) if p)
            return " ".join(p for p in ("ez da beharrezkoa", npd, bx, verb["nom"]) if p)
        if kind == "thing":
            verb = ctx.pick(VERBS[ctx.rng.choice(["move", "pick"])])
            bx = to if ctx.chance(0.5) else ""
            return " ".join(p for p in (verb["part"], ctx.rng.choice(IMPOSSIBLE), bx) if p)
        template = ctx.pick(STATEMENTS)
        npn, _ = self.np(ctx, f, q, "all", style="num", pron=False)
        return template.format(npd=npd, npn=npn, box=absolutive, boxin=inessive, boxto=to)

    def wrap(self, ctx: Ctx, text: str, intent: str) -> str:
        busy = intent in ("move", "sort", "pick", "both", "ask", "refuse")
        if ctx.chance(0.30 if busy else 0.10) and not text.endswith("?"):
            text = ctx.rng.choice(PREFIXES) + text
        if ctx.chance(0.22 if busy else 0.08) and not text.endswith("?"):
            text = text + ctx.rng.choice(SUFFIXES)
        return text
