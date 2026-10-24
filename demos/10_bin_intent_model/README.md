# 10 — Intent model for the two-box bin-sorting cell

A sentence in Spanish, English or Basque goes in; out come **a percentage for every option** of five questions,
and the pieces the sentence names are **linked to the box it names**.

```
"Clasifica las piezas malas en la caja blanca"
  intent   sort 100%   move 0%   pick 0%
  quality  bad 100%    good 0%   any 0%
  box      white 100%  big 0%    other 0%   none 0%
  rule     pieces [bad, all] -> white box     (confidence 99.6%)
```

It is the same idea as the command model of demos 03 / 05 (`command_model_v2`, which the Isaac dashboard of demo 11
reads its "lo menos claro: el destino (74%)" from), rebuilt for this cell: **one forward pass, five fixed
questions, a probability for each option**, in the style of TypeSafe's Jev. Nothing is generated, so it cannot
invent a box that is not on the list.

## What it answers

| question | options |
|---|---|
| `intent` | `move` `sort` `pick` `both` (two instructions) `reply` `ask` `stop` `cancel` `repeat` `yes` `no` `hello` `chat` `refuse` |
| `feature` | `bolt` · `no bolt` · `any` |
| `quality` | `good` · `bad` · `any` |
| `box` | `big` · `white` · `other` (a box this cell does not have: small, black, "the left one") · `none` |
| `count` | `one` `two` `three` `all` `none` |

`box` is *the big box* and *the white box*. `refuse` is what must not move the robot: negations ("no muevas las
piezas"), destruction, things that are not pieces, statements about the past.

## Using it

```python
from intent_model import BinIntentModel
model = BinIntentModel(routes={("bolt", "any"): "big", ("any", "bad"): "white"})   # routes are optional
plan = model.plan("Las piezas con perno a la caja grande y las malas a la blanca")
plan.show()
```

`plan.status` is `ok`, `ask_box`, `unsure` (some answer under 85%), `not_a_cell_box`, `no_action`, `stop`, `cancel`,
`repeat` or `answer`. `plan.rules` links pieces to a box, each with the probability of every box option
(`rule.box_options`). **`routes`** is the cell's own piece→box convention: when a sentence names no box, the route
supplies it (`box_source == "route"`); when the sentence names a different box than the route, `conflicts_with_route`
is set. A `both` sentence is cut at the conjunction and each half is read again, in one batch.

Load it before Isaac Sim starts: torch loaded after Kit crashes the process (see demo 09, which avoids GLiNER for
that reason).

## Where this came from in the repo

The percentages come from `command_model_v2/command_model.py` (`confidences` per head) and are drawn by
`05_planner_and_executor/monitor/static/index.html` (`confidenceNote`). That model has a different label space
(red/green/blue trays, 16 intents) and **saturated probabilities** (1.0 against 1e-13), so its percentages say little.
Here the probabilities are calibrated (below). Demo 09 (`binpick/commands.py`) also has a `defect` attribute with no
classifier behind it; `quality` is the language side of that.

## Pipeline

| step | command | time (RTX 5060 Ti) |
|---|---|---|
| 1 | `python generate_dataset.py` — 20,000 training sentences per language, plus eval / calib / unseen-word test sets | 4 s |
| 2 | `python train.py --batch-size 16` — GLiNER2.5-multi, full fine-tune, 2 epochs | ~45 min |
| 3 | `python evaluate.py --calibrate --latency` | ~2 min |

Training sentences are built from word lists in three tiers (`grammar_es.py`, `grammar_en.py`, `grammar_eu.py`):
**train** words, **calib** words (only used to fit the calibration) and **test** words, which occur in neither. No
sentence appears in two files. Speech-recognition-style damage is applied at random: about a quarter of sentences
lose their punctuation, 12% of Spanish and Basque ones lose their accents, 5% get a one-letter typo and 2% a plausible
mis-hearing ("caja" → "casa", "perno" → "berno"). 7% of move/sort orders that name a box contain a self-correction
("a la grande, no, a la blanca": the last thing said wins). By intent, 62% of the sentences are orders (`move`,
`sort`, `pick`, `both`), 7.5% are answers (`reply`) and the rest are control words, questions, chat and refusals.

Batch 32 does not fit this 16 GB card (it ran at 10 s per step); batch 16 runs at about 3 steps per second.

## Measured

Exact match = all five answers right.

| set | sentences | exact match |
|---|---|---|
| `eval` — held out, same words as training | 2,997 | 98.6% |
| `test_unseen` — words never seen (verbs, nouns, adjectives, box names) | 2,997 | **87.9%** (en 94.6, es 89.8, eu 79.3) |
| `real` — 143 sentences written by hand | 143 | **97.9%** (en 98.0, es 95.9, eu 100) |

- **Big box vs white box:** sent to the wrong one of the two in **0 of 1,042** unseen-word orders and **0 of 54**
  hand-written ones. In 2.98% of the unseen-word orders the box was read as `other`/`none`, which makes the robot ask.
- **Safety:** hand-written, 0 of 56 sentences that must not move the robot were read as an order, and `stop` 6/6.
  On unseen words, 5 of 912 were read as an order and `stop` 88/105 — **all of them English slang that exists only in
  the test vocabulary ("cut it out" → sort, "knock it off" → cancel).** Keep an independent stop (a hardware button, or
  the keyword check demo 09 uses) in the robot loop.
- **Is "90%" right?** One temperature per question was fitted on `calib` (`models/bin-intent/calibration.json`:
  intent 3.67, feature 3.74, quality 4.81, box 4.63, count 3.74 — the raw model is overconfident). On unseen words
  calibration error fell from 0.049 to 0.020 (intent), 0.043 to 0.024 (quality), 0.015 to 0.008 (box). Refusing
  answers below 90% confidence keeps 82.5% of unseen-word sentences with 94.3% exact; below 99%, keeps 53% with 98.6%.

### Speed and memory (RTX 5060 Ti)

| | exact (real / unseen) | one sentence | in a batch | peak memory |
|---|---|---|---|---|
| GPU fp32 | 97.9% / 87.9% | 17.6 ms | 8.9 ms | 2.6 GB |
| **GPU fp16 (default)** | 97.9% / 87.9% | 18.1 ms | **3.4 ms** | **1.5 GB** |
| CPU fp32 | 97.9% / 87.9% | 111 ms | 70 ms | – |
| CPU int8 (dynamic quantisation) | 14.7% / 6.6% | 60 ms | 36 ms | – *(rejected: breaks the model)* |
| fp16, vocabulary pruned to seen tokens | 97.2% / 55.7% | – | 3.4 ms | 0.76 GB *(rejected: see below)* |

`BinIntentModel.read()` takes about 32 ms per sentence including the result objects; the voice step before it
(Whisper) takes far longer.

Two thirds of the 287M parameters are the 250k-token embedding table. Pruning it to the tokens seen in data cuts
weights from 575 MB to 197 MB, but any word outside that set becomes `UNK`, and accuracy on unseen wording falls from
87.9% to 56%. Cutting to the most frequent token ids is worse (the tokenizer's ids are not frequency-ordered).
Neither is shipped.

## Limits — read before trusting it

- **All training text is synthetic**, and so is the `real` set: it was written by the same author as the grammars,
  not by operators. 97.9% says the model reads natural phrasing of this task well; it does not say how it does with
  your people. Add real sentences (from the microphone, through Whisper) to `test_real.txt` and re-run `evaluate.py`.
- **Basque was written without a native speaker.** It is the weakest language on unseen wording (79.3%, mostly the
  `quality` question: 87.5%). Have someone read `grammar_eu.py` before relying on it.
- The two boxes are *big* and *white*, the pieces differ by *bolt* and *good/bad*. A third box, another colour or
  another piece property means new labels, new grammar words and a retrain (about 50 minutes).
- A `both` sentence is cut at the first conjunction pair that makes two orders; "las demás" / "the rest" comes out as
  a rule with no piece restriction (feature `any`, quality `any`), which the cell must read as "everything else".
- Voice goes through Whisper first (`05_planner_and_executor/voice.py`); nothing here was measured on audio.

## Files

```
generate_dataset.py     step 1; grammar_common.py / grammar_{es,en,eu}.py hold the labels and the sentence grammars
train.py                step 2
evaluate.py             step 3: accuracy, calibration, reliability, safety, latency; writes results/ and calibration.json
intent_model.py         BinIntentModel: read() -> probabilities, plan() -> rules linking pieces to boxes
test_real.txt           143 hand-written test sentences (lang | sentence | intent | feature | quality | box | count)
data/ models/ results/  generated; models/ is 1.1 GB per model and should not be committed
```
