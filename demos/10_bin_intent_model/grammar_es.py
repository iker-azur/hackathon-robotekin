"""Spanish sentences for the bin-sorting intent dataset (see ``grammar_common.py`` for the labels)."""

from __future__ import annotations

from grammar_common import Ctx

# --- verbs: infinitive, tu imperative, subjunctive, and the stem a clitic attaches to ----------------

def _verbs(rows):
    return [dict(zip(("inf", "imp", "subj", "acc"), row.split("|"))) for row in rows]


VERBS = {
    "move": {
        "train": _verbs(["mover|mueve|muevas|muéve", "poner|pon|pongas|pon", "colocar|coloca|coloques|colóca",
                         "llevar|lleva|lleves|lléva", "meter|mete|metas|méte", "pasar|pasa|pases|pása",
                         "guardar|guarda|guardes|guárda", "dejar|deja|dejes|déja",
                         "trasladar|traslada|traslades|trasláda", "transportar|transporta|transportes|transpórta",
                         "depositar|deposita|deposites|depósita", "mandar|manda|mandes|mánda"]),
        "calib": _verbs(["enviar|envía|envíes|envía", "desplazar|desplaza|desplaces|desplaza"]),
        "test": _verbs(["reubicar|reubica|reubiques|reubíca", "transferir|transfiere|transfieras|transfiére"]),
    },
    "sort": {
        "train": _verbs(["clasificar|clasifica|clasifiques|clasifíca", "separar|separa|separes|sepára",
                         "ordenar|ordena|ordenes|ordéna", "organizar|organiza|organices|organíza",
                         "repartir|reparte|repartas|repárte", "agrupar|agrupa|agrupes|agrúpa",
                         "distribuir|distribuye|distribuyas|distribúye", "apartar|aparta|apartes|apárta",
                         "filtrar|filtra|filtres|fíltra"]),
        "calib": _verbs(["dividir|divide|dividas|divíde", "reagrupar|reagrupa|reagrupes|reagrúpa"]),
        "test": _verbs(["segregar|segrega|segregues|ségrega", "cribar|criba|cribes|críba",
                        "catalogar|cataloga|cataloges|catalóga"]),
    },
    "pick": {
        "train": _verbs(["coger|coge|cojas|cóge", "agarrar|agarra|agarres|agárra", "tomar|toma|tomes|tóma",
                         "recoger|recoge|recojas|recóge", "levantar|levanta|levantes|levánta",
                         "sujetar|sujeta|sujetes|sújeta"]),
        "calib": _verbs(["atrapar|atrapa|atrapes|atrápa"]),
        "test": _verbs(["alzar|alza|alces|álza", "capturar|captura|captures|captúra"]),
    },
}

MODALS = ["puedes", "podrías", "puedes por favor", "quiero", "necesito", "quisiera", "tienes que", "tendrías que",
          "debes", "hay que", "vas a", "me gustaría", "toca", "es hora de", "te pido"]
SUBJ_INTROS = ["quiero que", "necesito que", "me gustaría que", "te pido que", "quisiera que", "es necesario que",
               "espero que", "quiero que ahora", "necesito que ahora"]
MARKERS = ["no, mejor", "perdón", "mejor", "quiero decir", "bueno, no", "digo", "no, espera", "corrijo"]

# --- pieces ---------------------------------------------------------------------------------------------

NOUNS = {  # singular, plural, gender
    "train": [("pieza", "piezas", "f")] * 4 + [("componente", "componentes", "m"), ("elemento", "elementos", "m"),
                                               ("unidad", "unidades", "f"), ("objeto", "objetos", "m")],
    "calib": [("producto", "productos", "m"), ("artículo", "artículos", "m")],
    "test": [("ítem", "ítems", "m"), ("trozo", "trozos", "m"), ("chapa", "chapas", "f")],
}
# adjectives: (masc sg, fem sg, masc pl, fem pl) or one invariant phrase
BAD = {
    "train": [("malo", "mala", "malos", "malas")] * 3 + [
        ("defectuoso", "defectuosa", "defectuosos", "defectuosas"), ("dañado", "dañada", "dañados", "dañadas"),
        ("estropeado", "estropeada", "estropeados", "estropeadas"), ("incorrecto", "incorrecta", "incorrectos", "incorrectas"),
        ("deficiente", "deficiente", "deficientes", "deficientes"), ("rechazado", "rechazada", "rechazados", "rechazadas"),
        ("erróneo", "errónea", "erróneos", "erróneas"), ("no válido", "no válida", "no válidos", "no válidas"),
        "en mal estado", "con defectos", "con algún defecto"],
    "calib": [("averiado", "averiada", "averiados", "averiadas"), ("inservible", "inservible", "inservibles", "inservibles"),
              "con fallos"],
    "test": [("descartado", "descartada", "descartados", "descartadas"), ("roto", "rota", "rotos", "rotas"),
             ("no conforme", "no conforme", "no conformes", "no conformes"), "con desperfectos"],
}
GOOD = {
    "train": [("bueno", "buena", "buenos", "buenas")] * 3 + [
        ("correcto", "correcta", "correctos", "correctas"), ("válido", "válida", "válidos", "válidas"),
        ("perfecto", "perfecta", "perfectos", "perfectas"), ("apto", "apta", "aptos", "aptas"),
        ("aceptable", "aceptable", "aceptables", "aceptables"), ("conforme", "conforme", "conformes", "conformes"),
        "en buen estado", "sin defectos", "en orden"],
    "calib": [("excelente", "excelente", "excelentes", "excelentes"), "en perfecto estado"],
    "test": [("impecable", "impecable", "impecables", "impecables"), ("aprobado", "aprobada", "aprobados", "aprobadas"),
             ("útil", "útil", "útiles", "útiles"), "sin ningún defecto"],
}
# relative phrases: (singular, plural)
BOLT = {
    "train": [("con perno", "con perno"), ("con perno", "con pernos"), ("con un perno", "con perno"),
              ("que tiene perno", "que tienen perno"), ("que lleva perno", "que llevan perno"),
              ("que trae perno", "que traen perno"), ("con perno puesto", "con perno puesto"),
              ("con el perno puesto", "con el perno puesto"), ("que tiene un perno", "que tienen un perno")],
    "calib": [("que incorpora perno", "que incorporan perno"), ("con perno montado", "con perno montado")],
    "test": [("que incluye perno", "que incluyen perno"), ("con perno colocado", "con perno colocado")],
}
NO_BOLT = {
    "train": [("sin perno", "sin perno"), ("sin perno", "sin pernos"), ("sin ningún perno", "sin ningún perno"),
              ("que no tiene perno", "que no tienen perno"), ("que no lleva perno", "que no llevan perno"),
              ("sin el perno", "sin el perno"), ("que carece de perno", "que carecen de perno"),
              ("sin perno puesto", "sin perno puesto")],
    "calib": [("que no trae perno", "que no traen perno"), ("sin perno colocado", "sin perno colocado")],
    "test": [("que no incorpora perno", "que no incorporan perno"), ("sin perno montado", "sin perno montado")],
}
PRONOUN_BOLT = ["de perno", "con perno"]
PRONOUN_NO_BOLT = ["sin perno"]

DETS = {  # (feminine, masculine) by count
    "all": (["las"] * 4 + ["todas las"] * 3 + ["esas", "estas", "todas esas", ""],
            ["los"] * 4 + ["todos los"] * 3 + ["esos", "estos", "todos esos", ""]),
    "one": (["una", "la", "la", "esa", "esta", "una sola", "cualquier"],
            ["un", "el", "el", "ese", "este", "un solo", "cualquier"]),
    "two": (["dos", "dos", "las dos", "esas dos", "estas dos"], ["dos", "dos", "los dos", "esos dos", "estos dos"]),
    "three": (["tres", "tres", "las tres", "esas tres"], ["tres", "tres", "los tres", "esos tres"]),
}

F2M = {"las": "los", "todas las": "todos los", "esas": "esos", "estas": "estos", "todas esas": "todos esos",
       "la": "el", "una": "un", "esa": "ese", "esta": "este", "una sola": "un solo", "las dos": "los dos",
       "esas dos": "esos dos", "estas dos": "estos dos", "las tres": "los tres", "esas tres": "esos tres"}
M2F = {m: f for f, m in F2M.items()}

# --- boxes -----------------------------------------------------------------------------------------------

BOX_NOUNS = {  # noun, gender
    "train": [("caja", "f")] * 6 + [("contenedor", "m"), ("recipiente", "m"), ("cubeta", "f"), ("cajón", "m")],
    "calib": [("depósito", "m"), ("cesto", "m")],
    "test": [("receptáculo", "m"), ("caja contenedora", "f")],
}
BIG = {"train": [("grande", "grande")] * 4 + [("más grande", "más grande"), ("de mayor tamaño", "de mayor tamaño"),
                                              ("de tamaño grande", "de tamaño grande"), ("de gran tamaño", "de gran tamaño"),
                                              ("mayor", "mayor")],
       "calib": [("enorme", "enorme"), ("más amplio", "más amplia")],
       "test": [("gigante", "gigante"), ("voluminoso", "voluminosa")]}
WHITE = {"train": [("blanco", "blanca")] * 5 + [("de color blanco", "de color blanco")],
         "calib": [("blanco nieve", "blanca nieve")], "test": [("blanquecino", "blanquecina")]}
OTHER = {
    "train": [("pequeño", "pequeña"), ("chico", "chica"), ("más pequeño", "más pequeña"), ("de menor tamaño", "de menor tamaño"),
              ("negro", "negra"), ("rojo", "roja"), ("verde", "verde"), ("azul", "azul"), ("amarillo", "amarilla"),
              ("de la izquierda", "de la izquierda"), ("de la derecha", "de la derecha"), ("izquierdo", "izquierda"),
              ("derecho", "derecha"), ("del fondo", "del fondo"), ("de arriba", "de arriba")],
    "calib": [("mediano", "mediana"), ("gris", "gris")],
    "test": [("marrón", "marrón"), ("naranja", "naranja"), ("morado", "morada"), ("de en medio", "de en medio")],
}
ORDINALS = ["otra", "primera", "segunda", "tercera"]

PREPS = {"move": ["a"] * 4 + ["en"] * 4 + ["dentro de"] * 2 + ["hacia", "al interior de"],
         "sort": ["en"] * 5 + ["a"] * 3 + ["dentro de"] * 2, "pick": ["en"]}

# --- fillers -----------------------------------------------------------------------------------------------

PREFIXES = ["por favor, ", "oye, ", "oye robot, ", "robot, ", "vale, ", "venga, ", "ahora ", "ahora mismo ", "bien, ",
            "a ver, ", "perfecto, ", "entonces ", "hola, ", "ok, ", "primero ", "y ahora ", "vamos a ver, ", "eh, "]
SUFFIXES = [", por favor", " por favor", " gracias", ", gracias", " ahora", " ya", " rápido", " cuando puedas",
            " si puedes", " de una vez", " cuanto antes", " con cuidado"]

CONTROL = {
    "stop": {"train": ["para", "para ya", "para ahora", "para todo", "stop", "alto", "alto ahí", "detente", "detente ya",
                       "detén el robot", "detén todo", "basta", "basta ya", "frena", "frena ahora", "quieto", "no te muevas",
                       "no sigas", "no sigas moviendo", "deja de moverte", "para el brazo", "parada de emergencia",
                       "emergencia", "stop stop", "para, para", "detente ahora mismo"],
             "calib": ["frénate", "interrumpe", "interrumpe el movimiento", "detén el movimiento"],
             "test": ["paraliza todo", "no continúes", "suspende el movimiento", "alto el robot", "cesa el movimiento"]},
    "cancel": {"train": ["cancela", "cancela la orden", "cancela eso", "cancélalo", "olvídalo", "olvida lo que he dicho",
                         "olvida eso", "anula la orden", "anula eso", "déjalo", "déjalo, no importa", "no importa",
                         "ignora eso", "ignora la orden", "mejor no", "da igual", "borra la orden", "dejémoslo"],
               "calib": ["descarta la orden", "anula todo", "borra eso"],
               "test": ["desiste", "retira la orden", "olvida la instrucción", "descarta eso"]},
    "repeat": {"train": ["repite", "repítelo", "repite eso", "repite la orden", "otra vez", "de nuevo", "hazlo otra vez",
                         "hazlo de nuevo", "vuelve a hacerlo", "una vez más", "repite la última orden", "repite lo anterior",
                         "otra vez lo mismo", "hazlo una vez más", "dilo otra vez"],
               "calib": ["otra vez, por favor", "haz lo mismo otra vez"],
               "test": ["reitera la orden", "ejecútalo de nuevo", "repítemelo", "vuelve a intentarlo"]},
    "yes": {"train": ["sí", "si", "sí, eso", "vale", "de acuerdo", "correcto", "eso es", "exacto", "adelante", "perfecto",
                      "claro", "afirmativo", "sí, hazlo", "hazlo", "muy bien", "ok", "okey", "sí, por favor", "confirmado",
                      "así es", "eso mismo", "sí señor", "sí, correcto"],
            "calib": ["efectivamente", "confirmo", "sí, claro"],
            "test": ["por supuesto", "exactamente", "afirmo", "desde luego"]},
    "no": {"train": ["no", "no, eso no", "negativo", "incorrecto", "esa no", "no es eso", "no, otra", "tampoco",
                     "para nada", "no, gracias", "nada de eso", "no es esa", "eso no", "no, no", "qué va"],
           "calib": ["ni hablar", "no señor"],
           "test": ["de ninguna manera", "en absoluto", "jamás", "ni de broma"]},
    "hello": {"train": ["hola", "hola robot", "buenos días", "buenas tardes", "buenas noches", "buenas", "hey", "hey robot",
                        "qué tal", "¿qué tal?", "hola, ¿cómo estás?", "¿cómo estás?", "saludos", "adiós", "hasta luego",
                        "hasta pronto", "gracias", "muchas gracias", "gracias robot", "buen trabajo", "muy bien hecho",
                        "chao", "nos vemos", "hasta mañana"],
              "calib": ["buen día", "qué hay", "gracias por todo"],
              "test": ["muy buenas", "encantado de verte", "hasta la vista", "que tengas buen día", "buena faena"]},
    "chat": {"train": ["¿qué hora es?", "¿qué tiempo hace hoy?", "cuéntame un chiste", "¿cómo te llamas?", "¿quién te ha creado?",
                       "pon música", "pon la radio", "pon una canción", "¿qué día es hoy?", "¿cuál es la capital de Francia?",
                       "¿cuánto es dos más dos?", "¿te gusta el fútbol?", "¿qué vamos a comer?", "apaga las luces",
                       "enciende la luz", "sube el volumen", "llama a mi madre", "¿qué hay para cenar?", "canta una canción",
                       "hace mucho calor aquí", "el examen ha sido difícil", "¿quién ganó el partido?", "abre la ventana",
                       "pon el despertador a las siete", "mañana llueve", "tengo hambre", "¿sabes bailar?",
                       "¿cuántos años tienes?", "traduce esto al inglés", "¿dónde está el baño?", "necesito un café",
                       "recuérdame llamar al médico", "cierra la puerta", "qué bonito día", "esto es muy aburrido",
                       "¿me ayudas con los deberes?", "mueve el volumen", "coloca la mesa para comer", "pon la tele",
                       "lleva esto a mi casa en coche", "¿qué opinas del tráfico?", "baja la persiana", "¿cómo está el gato?"],
             "calib": ["¿qué película me recomiendas?", "pon el telediario", "¿me prestas un bolígrafo?"],
             "test": ["¿crees que va a nevar?", "pon un podcast", "¿a qué hora cierran las tiendas?", "recomiéndame un libro",
                      "hoy toca pizza", "¿cuándo son las vacaciones?"]},
}

STATIC_ASK = {
    "train": ["¿qué ves?", "¿qué estás haciendo?", "¿qué puedes hacer?", "¿has terminado?", "¿ya está?", "¿cuánto falta?",
              "¿qué cajas hay?", "¿cuántas cajas hay?", "¿cuántas piezas quedan?", "¿estás listo?", "¿me oyes?",
              "¿qué piezas hay?", "dime qué ves", "dime cuántas piezas quedan", "cuéntame qué estás haciendo",
              "¿qué tipos de piezas hay?", "¿cómo se llaman las cajas?", "¿hay alguna pieza mala?",
              "¿hay alguna pieza con perno?", "¿qué has hecho hasta ahora?", "¿cuál es la mejor caja para las malas?"],
    "calib": ["¿cuántas piezas tienes delante?", "¿puedes decirme qué ves?"],
    "test": ["¿qué tal va la clasificación?", "¿cuánto tiempo queda?", "¿en qué estado está todo?", "informa de lo que ves"],
}
ASK_TEMPLATES = {
    "train": ["¿{cuant} {npb} hay?", "¿{cuant} {npb} quedan?", "¿{cuant} {npb} has movido?", "¿hay {npb}?",
              "¿dónde está {box}?", "¿cuál es {box}?", "¿a qué caja van {npd}?", "¿qué hay en {box}?",
              "¿cuántas piezas hay en {box}?", "dime {cuant} {npb} quedan", "dime dónde está {box}",
              "¿ves {npd}?", "¿queda alguna caja {boxadj}?"],
    "calib": ["¿{cuant} {npb} tienes delante?", "¿puedes decirme {cuant} {npb} hay?"],
    "test": ["¿me dices dónde está {box}?", "¿cuál de las cajas es {box}?", "necesito saber {cuant} {npb} quedan",
             "¿sabes {cuant} {npb} hay?"],
}
STATEMENTS = {
    "train": ["{npd} ya están en {box}", "ayer vi {npb} en {box}", "antes había {npb}", "{box} está llena",
              "{box} está vacía", "ya he movido {npd}", "hoy hay muchas {npb}", "me gustan {npd}", "{npd} son muy bonitas",
              "ayer clasificamos {npd}", "mañana clasificaremos {npd}", "ya moví {npd} a {box}"],
    "calib": ["{box} se ha caído", "el otro día había {npb}"],
    "test": ["{npd} estaban en {box} anoche", "{box} me parece bonita"],
}
DESTROY = {"train": ["tira", "lanza", "arroja", "rompe", "destroza", "aplasta", "golpea", "machaca", "quema", "pisa"],
           "calib": ["revienta"], "test": ["estampa", "hace trizas"]}
DESTROY_TAIL = ["al suelo", "a la basura", "por la ventana", "contra la pared", "fuera", "lejos", "al fuego"]
NEG_INTROS = ["no", "nunca", "en ningún caso", "bajo ningún concepto", "por favor no"]
IMPOSSIBLE = ["el coche", "la silla", "la mesa", "el ordenador", "la puerta", "mi móvil", "el gato", "a mi hermano",
              "el edificio", "el monitor", "la caja grande", "la caja blanca", "el brazo", "la cámara", "la lámpara",
              "la caja entera", "ambas cajas"]


class Spanish:
    code = "es"
    markers = MARKERS

    # -- noun phrases -----------------------------------------------------------------------------------

    @staticmethod
    def _agree(adj, g: str, plural: bool) -> str:
        return adj if isinstance(adj, str) else adj[(2 if plural else 0) + (1 if g == "f" else 0)]

    def np(self, ctx: Ctx, f: str, q: str, count: str, det: str | None = None, pron: bool = True):
        """The pieces: ``(text, gender)``. ``det=""`` forces a bare plural ("piezas malas con perno")."""
        noun_sg, noun_pl, g = ctx.pick(NOUNS)
        plural = count != "one"
        if det is None:
            options = DETS["all" if count == "none" else count][0 if g == "f" else 1]
            det = ctx.rng.choice(options)
        det = (F2M.get(det, det) if g == "m" else M2F.get(det, det))
        if count == "one" and det == "":
            det = "la" if g == "f" else "el"
        adj = self._agree(ctx.pick(BAD if q == "bad" else GOOD), g, plural) if q != "any" else ""
        rel_sg, rel_pl = ctx.pick(BOLT if f == "bolt" else NO_BOLT) if f != "any" else ("", "")
        rel = rel_pl if plural else rel_sg
        if pron and (f != "any" or q != "any") and ctx.chance(0.12) and det:
            extra = (PRONOUN_BOLT if f == "bolt" else PRONOUN_NO_BOLT) if f != "any" else []
            if rel and rel.startswith("que") and ctx.chance(0.6):
                feat = rel
            else:
                feat = ctx.rng.choice(extra) if extra else ""
            return " ".join(p for p in (det, adj, feat) if p), g
        noun = noun_pl if plural else noun_sg
        if f != "any" and q != "any":
            pattern = ctx.weighted({"qf": 0.4, "fq": 0.12, "q y f": 0.2, "que": 0.2, "pre": 0.08})
            pre_ok = adj.split()[0] in ("malo", "mala", "malos", "malas", "bueno", "buena", "buenos", "buenas")
            if pattern == "pre" and not pre_ok:
                pattern = "qf"
            if pattern == "qf":
                parts = [det, noun, adj, rel]
            elif pattern == "fq":
                parts = [det, noun, rel, adj]
            elif pattern == "q y f":
                parts = [det, noun, adj, "y", rel]
            elif pattern == "que":
                parts = [det, noun, rel, "que " + ("están" if plural else "está"), adj]
            else:
                parts = [det, adj, noun, rel]
        else:
            parts = [det, noun, adj, rel]
        return " ".join(p for p in parts if p), g

    def box_np(self, ctx: Ctx, label: str) -> str:
        noun, g = ctx.pick(BOX_NOUNS)
        art = ctx.rng.choice(["la", "la", "la", "esa", "esa"] if g == "f" else ["el", "el", "el", "ese", "ese"])
        if label == "other" and ctx.chance(0.25):
            ordinal = ctx.rng.choice(ORDINALS)
            ordinal = ordinal if g == "f" else ordinal[:-1] + "o"
            return f"{art} {ordinal} {noun}"
        slot = {"big": BIG, "white": WHITE, "other": OTHER}[label]
        masc, fem = ctx.pick(slot)
        adj = fem if g == "f" else masc
        if ctx.chance(0.12) and not adj.startswith("de "):
            return f"{art} {adj}"
        return f"{art} {noun} {adj}"

    def box_ph(self, ctx: Ctx, label: str, intent: str = "move") -> str:
        prep = ctx.rng.choice(PREPS[intent])
        np = self.box_np(ctx, label)
        if np.startswith("el ") and prep in ("a", "dentro de", "al interior de"):
            return {"a": "al", "dentro de": "dentro del", "al interior de": "al interior del"}[prep] + np[2:]
        return f"{prep} {np}"

    # -- action sentences -------------------------------------------------------------------------------

    def action(self, ctx: Ctx, intent: str, f: str, q: str, box: str, count: str) -> str:
        verb = ctx.pick(VERBS[intent])
        bx = self.box_ph(ctx, box, intent) if box != "none" else ""
        np_text, g = self.np(ctx, f, q, count)
        weights = {"imp": 0.34, "imp2": 0.07, "boxfirst": 0.06, "clitic": 0.10, "modal": 0.17, "subj": 0.10, "inf": 0.06}
        if not bx:
            weights.pop("boxfirst")
            weights.pop("imp2")
        if intent == "move" and bx:
            weights["fragment"] = 0.06
        frame = ctx.weighted(weights)
        join = lambda *p: " ".join(x for x in p if x)
        if frame == "imp":
            return join(verb["imp"], np_text, bx)
        if frame == "imp2":
            return join(verb["imp"], bx, np_text)
        if frame == "boxfirst":
            return join(bx + ",", verb["imp"], np_text)
        if frame == "clitic":
            plural = count != "one"
            clitic = ("las" if g == "f" else "los") if plural else ("la" if g == "f" else "lo")
            text, g = self.np(ctx, f, q, count, pron=False)
            clitic = ("las" if g == "f" else "los") if plural else ("la" if g == "f" else "lo")
            return join(text + ",", verb["acc"] + clitic, bx)
        if frame == "modal":
            modal = ctx.rng.choice(MODALS)
            text = join(modal, verb["inf"], np_text, bx)
            return "¿" + text + "?" if modal in ("puedes", "podrías", "puedes por favor") and ctx.chance(0.6) else text
        if frame == "subj":
            return join(ctx.rng.choice(SUBJ_INTROS), verb["subj"], np_text, bx)
        if frame == "inf":
            return join(verb["inf"], np_text, bx)
        return join(np_text, bx)  # fragment: "las piezas malas a la caja blanca"

    def fragment(self, ctx: Ctx, f: str, q: str, box: str, count: str) -> str:
        np_text, _ = self.np(ctx, f, q, count)
        return " ".join((np_text, self.box_ph(ctx, box, "move")))

    def reply(self, ctx: Ctx, f: str, q: str, box: str, count: str) -> str:
        """An answer to "which pieces?" / "which box?": a fragment with no verb."""
        if box != "none" and f == "any" and q == "any" and count == "none":
            lead = ctx.rng.choice(["", "", "", "sí, ", "pues ", "mejor ", "no, ", "la "])
            if lead == "la ":
                masc, fem = ctx.pick(BIG if box == "big" else WHITE if box == "white" else OTHER)
                return lead + fem
            return lead + self.box_ph(ctx, box, "move")
        if box == "none" and f == "any" and q == "any":
            words = {"all": ["todas", "todos", "todas ellas"], "one": ["una", "solo una", "una sola", "esa"],
                     "two": ["dos", "las dos", "solo dos"], "three": ["tres", "las tres", "solo tres"]}[count]
            return ctx.rng.choice(words)
        text, _ = self.np(ctx, f, q, count)
        return text

    def both(self, ctx: Ctx) -> str:
        """Two rules in one sentence: the first pieces to one box, the second to the other."""
        (f1, q1), (f2, q2) = self._two_groups(ctx)
        b1, b2 = ("big", "white") if ctx.chance(0.5) else ("white", "big")
        verb = ctx.pick(VERBS[ctx.rng.choice(["move", "sort"])])
        a, _ = self.np(ctx, f1, q1, "all")
        if ctx.chance(0.2):
            second = ctx.rng.choice(["las demás", "el resto", "las otras", "las restantes"])
        else:
            second, _ = self.np(ctx, f2, q2, "all", det=ctx.rng.choice(["las", "los", "todas las"]))
        glue = ctx.rng.choice(["y", "y", "y", ", y", ";", "y luego", "y después"])
        first = " ".join((verb["imp"], a, self.box_ph(ctx, b1, "move")))
        return f"{first}{glue if glue in (';', ', y') else ' ' + glue} {second} {self.box_ph(ctx, b2, 'move')}".replace(
            ", y y", ", y")

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

    def _bare(self, ctx: Ctx):
        f, q = ctx.rng.choice([("bolt", "any"), ("no bolt", "any"), ("any", "bad"), ("any", "good"), ("bolt", "bad"),
                               ("no bolt", "good"), ("any", "any")])
        return self.np(ctx, f, q, "all", det="", pron=False)

    def ask(self, ctx: Ctx) -> str:
        if ctx.chance(0.4):
            return ctx.pick(STATIC_ASK)
        template = ctx.pick(ASK_TEMPLATES)
        npb, g = self._bare(ctx)
        npd, _ = self.np(ctx, *ctx.rng.choice([("bolt", "any"), ("any", "bad"), ("any", "good"), ("no bolt", "any")]),
                         "all", det="las", pron=False)
        box = self.box_np(ctx, ctx.rng.choice(["big", "white", "white", "big", "other"]))
        boxadj = ctx.pick(ctx.rng.choice([BIG, WHITE]))[1]
        return template.format(cuant="cuántas" if g == "f" else "cuántos", npb=npb, npd=npd, box=box, boxadj=boxadj)

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
                return " ".join(p for p in (ctx.rng.choice(NEG_INTROS), verb["subj"], npt, bx) if p)
            if style == "nothing":
                return ctx.rng.choice(["no toques nada", "no muevas nada", "no hagas nada", "no cojas nada",
                                       "no pongas nada en " + self.box_np(ctx, box)])
            if style == "want":
                return " ".join(p for p in ("no quiero que", verb["subj"], npt, bx) if p)
            return " ".join(p for p in (ctx.rng.choice(["no hace falta", "no hay que", "no tienes que"]),
                                        verb["inf"], npt, bx) if p)
        if kind == "thing":
            verb = ctx.pick(VERBS[ctx.rng.choice(["move", "pick"])])
            bx = self.box_ph(ctx, box, "move") if ctx.chance(0.5) else ""
            return " ".join(p for p in (verb["imp"], ctx.rng.choice(IMPOSSIBLE), bx) if p)
        template = ctx.pick(STATEMENTS)
        npd, _ = self.np(ctx, f, q, "all", det="las", pron=False)
        npb, _ = self.np(ctx, f, q, "all", det="", pron=False)
        return template.format(npd=npd, npb=npb, box=self.box_np(ctx, box))

    def wrap(self, ctx: Ctx, text: str, intent: str) -> str:
        busy = intent in ("move", "sort", "pick", "both", "ask", "refuse")
        if ctx.chance(0.30 if busy else 0.10) and not text.startswith("¿"):
            text = ctx.rng.choice(PREFIXES) + text
        if ctx.chance(0.22 if busy else 0.08) and not text.endswith("?"):
            text = text + ctx.rng.choice(SUFFIXES)
        return text
