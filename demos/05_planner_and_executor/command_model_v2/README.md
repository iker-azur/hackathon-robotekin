# command model v2

A copy of `demos/03_command_model/` that adds **per-item destinations**, so that one
command can send different things to different places. It exists because demo 10 needs to
execute:

> *"mete las bolas rojas en la caja roja y las azules en la bandeja azul"*
> *"pon el cubo rojo en la caja azul y la bola verde en la bandeja roja"*
> *"pásame la bola roja y déjalo en la bandeja"*

Demo 03's schema cannot express any of those, and its `sort` cannot either.

The supported geometry labels include balls, cubes, cylinders, triangular prisms and
hexagonal prisms. Explicit geometry words and the words in each object-to-destination span
are resolved against the vocabulary before accepting the learned classifier's answer. The
classifier still handles phrasing and open vocabulary; it cannot change an explicitly named
"caja roja" into a blue destination or collapse "cilindro" into the old generic `other` label.

`demos/03_command_model/` is untouched. Every path here is relative to this file, so the
copy is self-contained, and the training starts from demo 03's checkpoint read-only:

    python generate_dataset.py --output-dir data_shapes
    python train.py `
        --base models\command-model\best --data-dir data_shapes --epochs 3 `
        --batch-size 2 --gradient-accumulation-steps 4 --eval-batch-size 4 --use-lora `
        --output models\command-model-shapes-lora-candidate
    python evaluate.py test_real.jsonl test_unseen.jsonl `
        --model models\command-model-shapes-lora-candidate\best_merged --data-dir data_shapes `
        --results-dir results\shapes_candidate

## What changed, and why

| Change | Reason |
|---|---|
| `Item.dest`, and `destination_kind_word` / `destination_color_word` on the item record | The per-item destination. Demo 03 had one `place` per command and said so deliberately. |
| `sort/into_matching` template, in es/en/eu | Demo 03's generator did `place.color = None` for every sort, so **all 1500** of its sort sentences were trained with an uncoloured destination. "The red ones in the red box" was not in the data at all. |
| `sequence/each_own_place` template | Two items, two destinations, on `sequence` as well as `sort`. |
| `give` may carry a place | Demo 03 forced `place=None` for every `give`. The design guide asks for this — `docs/guia-so-101-autonomous-agent.md:1686`: handing into the hand should be avoided; for the hackathon, leave it in a tray. |
| `clear` may carry an item filter | "quita las bolas rojas de la mesa" was inexpressible. |
| `sort`, `clear` removed from `ITEMLESS`; `question` added | `sort`/`clear` now carry items. `question` was missing from this list but present in `ground.NO_ACTION`, so "¿Qué ves?" had its items grounded against the scene and could ask the person "¿cuál?" about a question that was never ambiguous. |
| `PLACE_WORDS` table | `_word_table` cannot flatten `vocabulary.yaml`'s `places`, which has a `train`/`test_only` level and then a noun list. |
| `evaluate.same_item` compares the per-item place | Otherwise a command that swapped two destinations scored as correct. |

### Two fields, not three, and not nested

GLiNER2.5 **cannot nest** — calling `.structure()` on a record builder emits a sibling
record, not a child (verified against the installed `gliner2` 2.0.0). So the destination is
two flat string fields on the existing `item` record.

They must not use `choices=`. Demo 03's README records that a multiple-choice field inside a
repeated record does not depend on which record it belongs to — the second item copied the
first one's colour — so the words are extracted and looked up, like the five existing fields.

A third field, `destination_mention`, was written and dropped: `to_records` asserts every
labelled word occurs verbatim, and a composed phrase is not a literal substring.

## Measured results

### Shape and color candidate trained for Demo 11

The RTX 5060 Ti was shared with the running simulator, so this candidate was
stopped after its first completed validation epoch (step 4,226). It is a
rank-8 LoRA adapter merged into `best_merged`; the original checkpoint remains
available as a fallback. The short run is not a replacement for a full training
study, but it was promoted locally because it improved both held-out command
sets without weakening stop handling:

| held-out set | model | object fields | full command | stop caught | false action rate |
|---|---|---:|---:|---:|---:|
| 136 hand-written real commands | base | 90.4% | 76.5% | 100% | 0.0% |
|  | one-epoch candidate | 91.9% | 79.4% | 100% | 0.0% |
| 1,500 generated unseen commands | base | 55.9% | 51.6% | — | 0.7% |
|  | one-epoch candidate | 56.3% | 53.1% | — | 0.7% |

The gains are modest. The reliable interpretation of an explicitly named
`cilindro verde` and `caja roja` still depends on the lexical safeguards in
`command_model.py`; the trained classifier adds generalization but does not
replace those checks. The four direct probes for green cylinder/red box,
triangular prism, hexagonal prism, and two differently colored destinations all
parsed with the expected geometry and colors.

`command_model.py` prefers the local merged candidate when it exists and falls
back to the original checkpoint when model weights are absent. Trained weights
are gitignored, so a clean checkout must run the commands below before it can
use this candidate.

The earlier v2 comparison against Demo 03 is separate from the shape candidate above:

| test set | | intent | items | place | sort_by | **command** |
|---|---|---|---|---|---|---|
| 98 sentences, demo 03's own test set | v2 | **98.0%** | **89.8%** | 93.9% | 100.0% | **84.7%** |
| | demo 03 | 95.9% | 87.8% | **94.9%** | 100.0% | **85.7%** |
| 38 new sentences for v2 | v2 | 86.8% | 92.1% | 100.0% | 100.0% | **81.6%** |

- **Parity on the legacy set** — 84.7% against demo 03's 85.7%, with better `intent` and
  better `items`, one point worse on `place`.
- **All 12 hand-written multi-destination sentences are correct**
  (`show_v2_failures.py`).
- **3.5x faster**: 49 ms per sentence against demo 03's 176 ms.
- `stop` caught 7/7. False action rate 0/54.

`test_legacy_only.jsonl` and `test_v2_only.jsonl` are written by `build_test_set.py` so the
two populations can be scored separately. **The 98 are copied verbatim** from demo 03; an
earlier version rewrote the `sort` rows and that was reverted — a hand-written test set
edited to agree with the model is not a measurement.

## Four things the training did not do, and what happens instead

All four are in `command_model.parse`, all four were found by measurement, and all four are
documented at the code. They exist because 2 epochs of fine-tuning from a checkpoint whose
`place_kind` head was already trained for hours does not teach a new string field inside a
repeated record.

1. **The per-item destinations are read out of the sentence.** After training, all 12
   multi-destination sentences came back with `place: None` on every item and the receptacle
   absorbed into the command-level `place` — sometimes the wrong one of the two. The model's
   item mentions are correct and correctly ordered, so the destination is taken from the
   stretch of sentence between one mention and the next. The model's own fields are consulted
   first, so training the head further makes this stop firing.

2. **An extracted `name` is only believed if it occurs in that item's own mention.** In "put
   the red cube in the blue box and the green ball in the red tray" **both** items came back
   with `name: "cube"`. This is the repeated-record bleed demo 03's README warns about,
   happening to a string field.

3. **A coordinated phrase shares one noun.** "las bolas rojas y las azules" — the second
   mention has no noun, so its kind falls back to `any` and is carried forward from the item
   before it. Only when there is no kind to carry, so a second group that names itself is
   left alone.

4. **A `sort` with no per-group destination carries no items.** "Separa las bolas de los
   cubos" is about the whole table and the dataset trains it with an empty item list; the
   model volunteered a ball group and a cube group for it.

Plus two smaller ones: a `clear` that names nothing (no noun, no colour) carries no item, and
a command whose items each have a destination has no command-level place.

## Known gaps

- **One more epoch is the obvious way to close the last point of legacy parity.** The
  remaining legacy errors are the ones demo 03 also had: test-only vocabulary (`canasta`,
  `contenedor`, `crate`, `otzara`) and colour order inside a list ("rojo, azul y rojo").
- `SORTABLE_COLORS` is red/green/blue only, because that is what perception separates. A sort
  sentence about orange would train toward a sort the robot cannot perform.
- No ASR noise round: `add_asr_noise.py` was not run, so the model has not seen
  speech-recognition mistakes the way demo 03's has.
- Laya is not copied. It owns the four command-level questions, which did not change; the
  per-item destination is GLiNER's job.
- `eval_loss` is 151.3 against demo 03's 143.8, which is not comparable across schemas: the
  loss now includes two new fields.

## Files

| | |
|---|---|
| `generate_dataset.py` | the grammar, plus the new templates |
| `command_model.py` | inference: the schema, `PLACE_WORDS`, and the four rules above |
| `train.py` | unchanged from demo 03 except that it reads this directory's `data/` |
| `evaluate.py` | scores the per-item destination |
| `build_test_set.py` | writes `test_real.jsonl` and the two subsets |
| `show_templates.py` | what the new templates produce, per language, no model needed |
| `show_v2_failures.py` | every multi-destination sentence, expected against got |
| `compare_errors.py` | separates the changed contract from real errors |
| `summarise_errors.py` | which intents and fields fail |
| `debug_items.py` | one sentence, full parsed output |
