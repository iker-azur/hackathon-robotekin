"""Semantic-first synthetic commands. Called manually from Colab, never on import."""

import hashlib
import itertools
import json
import random
from pathlib import Path

from .voice import CATEGORIES, COLOR_WORDS, normalize, result

QUALITY_WORDS = {
    "good": ["buenas", "correctas", "aceptables", "sin defectos"],
    "defect_1": ["defectuosas tipo uno", "con defecto tipo 1", "tipo uno", "defectuosas de tipo uno"],
    "defect_2": ["defectuosas tipo dos", "con defecto tipo 2", "tipo dos", "defectuosas de tipo dos"],
}
PARTS = ["piezas", "tuercas", "arandelas", "tornillos", "soportes", "componentes"]
VERBS = ["Pon", "Coloca", "Clasifica", "Lleva", "Envia", "Distribuye", "Manda", "Deja", "Separa", "Reparte"]
# Entire grammatical families belong to one split; variants never cross it.
FAMILIES = {
    "train": ["{v} {s} en la caja {d}", "{v} {s} al contenedor {d}", "{v} {s} hacia el recipiente {d}", "{v} {s} dentro de la caja {d}"],
    "validation": ["{v} a la caja {d}: {s}", "{v} {s} a la {d}"],
    "test": ["{v} {s} en el destino {d}", "{v} hacia la caja {d} {s}"],
}
PREFIXES = ["", "Por favor, ", "Robot, ", "Ahora, ", "Cuando puedas, ", "Para esta tanda, "]
SUFFIXES = ["", ", por favor", " con cuidado", " despacio", " ahora", " para esta tanda"]


def key(text):
    return " ".join(__import__("re").sub(r"[^\w\s]", " ", normalize(text)).split())


def _render(family, verb, subject, destination):
    color = "rojo" if destination == "red" and any(word in family for word in ("contenedor", "recipiente", "destino")) else COLOR_WORDS[destination]
    return family.format(v=verb, s=subject, d=color)


def _subject(category, mode, rng):
    description = rng.choice(QUALITY_WORDS[category]) if mode == "quality" else {"red":"rojas","green":"verdes","blue":"azules"}[category]
    part = rng.choice(PARTS)
    masculine = part in ("tornillos","soportes","componentes")
    if masculine:
        for feminine,male in [("buenas","buenos"),("correctas","correctos"),("defectuosas","defectuosos"),("rojas","rojos")]:
            description = description.replace(feminine,male)
    return ("los " if masculine else "las ") + part + " " + description


def generate(n=25000, seed=42):
    if n < 20000:
        raise ValueError("The voice dataset requires at least 20,000 examples.")
    rng = random.Random(seed)
    splits, seen = {}, set()
    for split, count in zip(FAMILIES, [int(n * .70), int(n * .15), n - int(n * .70) - int(n * .15)]):
        records = []
        attempts = 0
        while len(records) < count:
            attempts += 1
            if attempts > count * 300:
                raise RuntimeError("Unique example space exhausted; expand grammatical families.")
            mode = rng.choice(list(CATEGORIES))
            cats = rng.sample(list(CATEGORIES[mode]), rng.randint(1, 3))
            family_index = rng.randrange(len(FAMILIES[split]))
            family = FAMILIES[split][family_index]
            kind = rng.choices(["valid", "missing", "conflict", "unsupported", "control", "invalid"], [.52, .16, .10, .08, .08, .06])[0]
            clauses, rules = [], []
            for i, cat in enumerate(cats):
                subject = _subject(cat,mode,rng)
                dest = rng.choice(list(COLOR_WORDS))
                clauses.append(_render(family, rng.choice(VERBS), subject, dest))
                rules.append({"category": cat, "destination": dest})
            text = rng.choice(PREFIXES) + " y ".join(clauses) + rng.choice(SUFFIXES)
            command = {"intent": "sort", "mode": mode, "rules": rules}
            expected = result(command)
            if kind == "invalid":
                text = rng.choice(PREFIXES) + _render(family,rng.choice(VERBS),f"-{rng.randint(1, 99)} piezas buenas",rng.choice(list(COLOR_WORDS)))
                expected = result(status="invalid", issues=["negative_quantity"])
            elif kind == "missing":
                # Generate independently labelled missing-destination instructions.
                cat = cats[0]
                subject = _subject(cat,mode,rng)
                missing_families = {"train":["{v} solamente {s}","{v} solo {s}"],
                    "validation":["{v} unicamente {s} para esta tanda"],
                    "test":["{v} para esta tanda solamente {s}"]}
                text = rng.choice(PREFIXES) + rng.choice(missing_families[split]).format(v=rng.choice(VERBS),s=subject) + rng.choice(SUFFIXES)
                command = {"intent": "sort", "mode": mode, "rules": [{"category": cat, "destination": None}], "filter": [cat]}
                expected = result(command, "needs_clarification", ["missing_destination"])
            elif kind == "conflict":
                cat = cats[0]
                other = rng.choice([d for d in COLOR_WORDS if d != rules[0]["destination"]])
                desc = rng.choice(QUALITY_WORDS[cat]) if mode == "quality" else {"red": "rojas", "green": "verdes", "blue": "azules"}[cat]
                text += " y " + _render(family, rng.choice(VERBS), "las piezas " + desc, other)
                command["rules"].append({"category": cat, "destination": other})
                expected = result(command, "contradictory", ["conflicting_destinations"])
            elif kind == "unsupported":
                text = rng.choice(PREFIXES) + _render(family,rng.choice(["Suelda", "Pinta", "Taladra", "Pule", "Atornilla", "Empaqueta"]),_subject(cats[0],mode,rng),rng.choice(list(COLOR_WORDS))) + rng.choice(SUFFIXES)
                expected = result(status="unsupported", issues=["unrecognized_action"])
            elif kind == "control":
                intent = rng.choice(["stop", "resume", "cancel"])
                phrase = {
                    "train": {"stop": ["Deten la clasificacion"], "resume": ["Continua clasificando"], "cancel": ["Cancela la orden"]},
                    "validation": {"stop": ["Interrumpe el proceso"], "resume": ["Reanuda la clasificacion"], "cancel": ["Anula la peticion"]},
                    "test": {"stop": ["Para el robot"], "resume": ["Sigue con la clasificacion"], "cancel": ["Cancela la clasificacion"]},
                }[split]
                text = rng.choice(["", "Por favor, ", "Robot, "]) + rng.choice(phrase[intent]) + rng.choice(["", ".", "!"])
                expected = result({"intent": intent})
            norm = key(text)
            if norm in seen:
                continue
            seen.add(norm)
            group = f"{split}:{family_index}:{kind}"
            record = {"id": hashlib.sha256(norm.encode()).hexdigest()[:16], "text": text,
                      "family": f"{split}:{family_index}", "group": group, "split": split,
                      "expected": expected, "intent": (expected["command"] or {}).get("intent", "unsupported"),
                      "mode": (expected["command"] or {}).get("mode", "none")}
            records.append(record)
        splits[split] = records
    return splits


def hypotheses():
    return [f"La orden pide {action}." for action in ("clasificar piezas", "detener la clasificacion", "continuar clasificando", "cancelar la orden")]


def pairs(record):
    """Relations and action decisions; neutral is distinct from contradiction."""
    command = record["expected"]["command"] or {}
    for action, hypothesis in zip(["sort", "stop", "resume", "cancel"], hypotheses()):
        yield record["text"], hypothesis, "entailment" if command.get("intent") == action else "neutral"
    for mode in CATEGORIES:
        yield record["text"], f"La orden pide clasificar por {'calidad' if mode == 'quality' else 'color'}.", "entailment" if command.get("mode") == mode else "neutral"
    yield record["text"], "La orden solicita una accion no admitida.", "entailment" if record["intent"] == "unsupported" else "neutral"
    for cat, dest in itertools.product(CATEGORIES.get(command.get("mode"), []), COLOR_WORDS):
        matches = [r for r in command.get("rules", []) if r["category"] == cat]
        label = "entailment" if any(r["destination"] == dest for r in matches) else "contradiction" if any(r["destination"] for r in matches) else "neutral"
        yield record["text"], f"El destino de {cat} es la caja {dest}.", label


def export(splits, output):
    import csv
    from collections import Counter
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    keys, families, report = {}, {}, {}
    for split, records in splits.items():
        keys[split] = {key(r["text"]) for r in records}
        families[split] = {r["family"] for r in records}
        assert len(keys[split]) == len(records)
        for record in records:
            json.dumps(record["expected"], ensure_ascii=False, allow_nan=False)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (out / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({**r, "expected": json.dumps(r["expected"], ensure_ascii=False)} for r in records)
        report[split] = {"size": len(records), "status": dict(Counter(r["expected"]["status"] for r in records)),
                         "intent": dict(Counter(r["intent"] for r in records))}
    for a, b in itertools.combinations(keys, 2):
        assert not keys[a] & keys[b], (a, b, "duplicate text")
        assert not families[a] & families[b], (a, b, "family leakage")
    (out / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
