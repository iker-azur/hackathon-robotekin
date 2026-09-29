# 03 — Command model for the hackathon

A small, fast model that turns what someone says to the robot (Spanish, English or Basque) into a command the robot can execute, built like TypeSafe's Jev: it reads the sentence together with the possible answers and scores every answer, instead of writing text. It never invents an option, and every answer has a confidence.

```text
"coge dos bolas rojas y mételas en la caja azul"
  -> {"intent": "pick_and_place",
      "items": [{"kind": "ball", "name": "bolas", "color": "red", "quantity": 2, "position": null}],
      "place": {"kind": "box", "color": "blue"}, "sort_by": "none", "confidence": 0.97}
```

It extends demo 07 (`../02_voice/07_action_classifier.py`) from one object and a handful of intents to what the hackathon may need: several objects per sentence, quantities, positions, sorting, clearing the table, stop/repeat/cancel, answers to the robot's questions, and sentences that must *not* move the robot.

## The pipeline

Every step is a script that writes files you can open and read:

| Step | Script | Writes | Time (RTX 5060 Ti) |
| --- | --- | --- | --- |
| 0 | edit `vocabulary.yaml` | the words: colours, numbers, positions, objects, places | — |
| 1 | `generate_dataset.py` | `data/train_commands.jsonl` (30,000), `data/train_item_kinds.jsonl`, `data/eval_*.jsonl`, `data/test_unseen.jsonl`, `data/stats.md` | 2 s |
| 2 | `add_asr_noise.py` (optional) | `data/train_asr.jsonl`: the same kind of sentences after text-to-speech → speech recognition | ~10 min |
| 3 | `train.py` | `models/command-model/best` (1.1 GB, git-ignored) | ~1 h |
| 4 | `evaluate.py` | `results/*_report.md`, `results/*_errors.jsonl` | 1 min |
| use | `command_model.py` | — (`CommandModel().parse(text)`) | ~70 ms per sentence |

```powershell
python .\demos\03_command_model\generate_dataset.py --show 30   # look at samples first
python .\demos\03_command_model\generate_dataset.py
python .\demos\03_command_model\add_asr_noise.py
python .\demos\03_command_model\train.py
python .\demos\03_command_model\evaluate.py
python .\demos\03_command_model\command_model.py "hartu bi bola gorri eta sartu kutxan"
```

`data/test_real.jsonl` is different from the rest: 98 sentences written and labelled by hand (not from the templates), including real transcripts from demos 06/07. It is the fairest measure of how the model will do with people.

## What the model answers

| Output | Values |
| --- | --- |
| `intent` | `pick_and_place`, `pick`, `give`, `sort`, `clear`, `sequence` (things to do); `stop`, `repeat`, `cancel` (control); `answer_yes`, `answer_no`, `answer_value` (replies to a robot question); `question`, `greeting`, `off_topic`, `not_doable` (no action) |
| `items[]` | one per object mentioned, in order: `kind` (ball / cube / any / other), `name` (as said), `color`, `quantity` (1–5 or `all`), `position` (left / right / front / back / nearest / farthest) |
| `place` | `kind` (basket / box / tray / table / left / right / other) and `color`, or null |
| `sort_by` | `color`, `kind` or `none` |
| `confidence` | the weakest of the model's decisions (0–1) |

`not_doable` covers what looks like a command but must not run: impossible or unsafe objects (*coge el coche*, *pick up John*), destruction (*tira la bola al suelo*), negation (*no toques nada*), and statements about the past (*ayer cogí la bola*). `place.kind = other` means a place the robot does not use (*la estantería*): the application should ask where to put it.

## The data format

Each line of the `*_commands.jsonl` files is one sample. `input` and `output` are what GLiNER2's trainer reads; `meta` is for you:

```json
{"input": "toma una goma roja, cuatro botellas y el lápiz rojo de detrás y mételos dentro de la bandeja morada",
 "output": {
   "classifications": [
     {"task": "intent", "labels": ["pick_and_place", "pick", "..."], "true_label": ["sequence"]},
     {"task": "place_kind", "labels": ["basket", "box", "tray", "..."], "true_label": ["tray"]},
     {"task": "place_color", "labels": ["red", "green", "..."], "true_label": ["purple"]},
     {"task": "sort_by", "labels": ["color", "kind", "none"], "true_label": ["none"]}],
   "json_structures": [
     {"item": {"mention": "una goma roja", "name": "goma", "color_word": "roja", "quantity_word": "una"}},
     {"item": {"mention": "cuatro botellas", "name": "botellas", "quantity_word": "cuatro"}},
     {"item": {"mention": "el lápiz rojo de detrás", "name": "lápiz", "color_word": "rojo", "position_word": "detrás"}}]},
 "meta": {"lang": "es", "template": "sequence/list", "command": {"intent": "sequence", "items": ["..."], "...": "..."}}}
```

About `add_asr_noise.py`: of 1,800 sentences read by Piper voices and transcribed by Whisper, about 190 come out usefully different (*"golpea"* → *"Volpea"*, *"mételos"* → *"arimételos"*) with every labelled word still present; the synthetic voices are clear, so Whisper is usually exactly right. Transcripts that only change case or punctuation, or that differ too much to trust the labels (*"chao"* → *"¡Todo!"*), are dropped. Use `--per-language 3000` for about five times more. Running it also revealed that Whisper sometimes repeats a sentence of its hint on short audio (*"Eh, recoge todo"* → *"Deja el objeto en el cesto verde. Recoge todo."*); demo 06's `transcribe()` now removes such sentences.

`train_item_kinds.jsonl` holds the item phrases alone (`"cuatro botellas"` → `item_kind: other`, `item_count: several`); `train_asr.jsonl` has the same format plus `meta.asr.original`, the sentence before speech recognition.

## Design decisions (and why)

- **GLiNER2.5-multi, fine-tuned.** Fastino's schema-driven encoder (Apache-2.0, 0.3B, mDeBERTa-v3) is the closest open model to Jev: it classifies and extracts in one forward pass, with confidences, and runs offline in ~70 ms. Zero-shot it was poor (27/64 in demo 07); training on examples is what makes it work, which is also how Jev is built.
- **Colours, numbers and positions are extracted as words, not classified per item.** A test showed that GLiNER2's multiple-choice fields inside repeated records do not depend on which record they belong to (the second item copied the first one's colour). Extracted words do work per item (`"rojas"`, `"dos"`), and `vocabulary.yaml` maps them to values. The model still does the hard part: finding which colour belongs to which object, in any of the three languages.
- **Item kind in a second pass.** For the same reason the kind (ball/cube/any/other) and one/several are classified on each item phrase separately, all item phrases in one batch.
- **One destination per command.** "Put the red ones in the red box and the blue ones in the blue box" is handled by `sort` (by colour), not by per-item destinations.
- **The test sets are kept honest.** `test_unseen.jsonl` is built only from `test_only` words (esfera, crate, otzara, grapadora...), and `test_real.jsonl` is written by hand; neither is used for training.
- **fp32 training.** The trainer's default bf16 for this architecture produced non-finite losses on this task.

## Adapting it at the hackathon

1. When you see the real objects and places, add them to `vocabulary.yaml` (`objects.train`, `places.train`), with the forms each language needs.
2. `generate_dataset.py`, then `train.py` (about an hour: run it overnight).
3. `evaluate.py`, and add a few real sentences you hear on site to `data/test_real.jsonl` to check.

Names the model has never seen are still extracted (`kind: other`, `name` as said), so a new object does not strictly need retraining; new *places* and colours do (they are classes).

## Laya: an open-source Jev, fine-tuned on the same data

[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) is an open-source "System One" decision model with Jev's question format: a sentence plus typed questions (`choice`, `score`, `noul`), each answered with calibrated probabilities in one forward pass. It cannot extract words, so it answers the sentence-level part of a command: `intent`, `place_kind`, `place_color`, `sort_by`. The items still come from the GLiNER2 model.

| Step | Script | Writes |
| --- | --- | --- |
| questions and right answers | `laya/laya_questions.py` | — |
| data | `laya/laya_prepare.py` | `data/laya_train_items.pt`: 12,187 sentences (4,000 per language + the speech-recognition ones) x 4 questions = 48,748 items |
| training | `laya/laya_train.py` | `models/laya-commands/` (650 MB), ~27 min for 2 epochs |
| evaluation | `laya/laya_evaluate.py` | `results/laya_*.md` |

`laya_train.py` is a single-GPU port of the trainer in Laya's Kaggle notebook (which needs two GPUs and NCCL, not available on Windows). The training is theirs: RLCD (noisy samples of the option scores rewarded with proper scoring rules, GRPO-style) plus a soft cross-entropy, then one calibration temperature per question type fitted on 400 held-out items. Our data has one right answer per question, so the targets are smoothed (0.9 on it, 0.1 spread over the rest).

Results on the same test sets (sentence-level decisions only):

| test_real (98 hand-written) | intent | place | place colour | all four | stop caught | false action |
| --- | --- | --- | --- | --- | --- | --- |
| Laya zero-shot (`laya-multilingual`) | 34.7% | 48.0% | 52.0% | 5.1% | 4/5 | 6.7% |
| GLiNER2 command model (3 epochs) | 95.9% | 94.9% | — | — | 5/5 | 2.2% |
| **Laya fine-tuned** | **99.0%** | **95.9%** | **99.0%** | **93.9%** | **5/5** | **0.0%** |

On `test_unseen.jsonl` the fine-tuned Laya keeps 99.1% intent (GLiNER 92.1%) but drops to 77% on places named with words it never saw (*otzara*, *crate*). It answers in about 40 ms per sentence.

The best combination is therefore Laya for the sentence-level decisions and GLiNER2 for the items.

```powershell
python .\demos\03_command_model\laya\laya_evaluate.py                                   # zero-shot
python .\demos\03_command_model\laya\laya_prepare.py
python .\demos\03_command_model\laya\laya_train.py
python .\demos\03_command_model\laya\laya_evaluate.py --model demos\03_command_model\models\laya-commands
```

## Demo 08: both models together, by voice

`08_voice_hybrid.py` puts everything together: microphone → Whisper (language detected) → **Laya** decides the intent, place, place colour and sort criterion → **GLiNER2** extracts the items → a command for the robot and a spoken reply in the language that was spoken.

```powershell
python .\demos\03_command_model\08_voice_hybrid.py                     # microphone
python .\demos\03_command_model\08_voice_hybrid.py --text "coge una bola roja y dos azules y mételas en la caja"
```

```text
heard   : 'coge una bola roja y dos azules y mételas en la caja'
command : {"intent": "sequence",
           "items": [{"kind": "ball", "name": "bola", "color": "red", "quantity": 1, ...},
                     {"kind": "any", "color": "blue", "quantity": 2, ...}],
           "place": {"kind": "box", "color": null}, "confidence": 0.89, "status": "ok"}
reply   : 'Vale, pongo la bola roja y dos bolas azules en la caja.'
time    : laya 37 ms, gliner 60-110 ms
```

`status` tells the robot what to do with the command: `ok` (execute), `no_action` (stop, greeting, question, off-topic, not doable...), or a question back to the person: `unsure` (weakest decision under 0.5), `no_object`, `no_color` (*"¿De qué color es la bola?"*), `no_target` (a place the robot does not use, or none for a pick-and-place).

Language detection here widens demo 06's word list with every word of the dataset (about 350–430 words per language), so newer Basque words such as *sailkatu* or *koloreka* are recognised: 98/103 test sentences get the right language; the misses are code-switched sentences (*"hartu la bola roja"*) and *"stop"*, which is also Spanish.

Known gaps, from the tests: object and place words that were never in the training data are not recognised as a kind (*esfera* → any, *otzara* → unknown place, so it asks where); add the real ones to `vocabulary.yaml` and retrain. `question` intents get a fixed reply until the camera demo exists.
