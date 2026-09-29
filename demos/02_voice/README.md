# 02 — Voice commands

- `06_voice_command.py`: say an order, get a robot command and a spoken reply, in Spanish, English or Basque. The command comes from word lists and rules.
- `07_action_classifier.py`: the same, but the command comes from a trained "System One" classifier (Jev-style) instead of rules. See [07](#07--a-model-as-the-action-classifier).

```text
"Coge la bola roja y métela en el cesto verde"
  -> {"action": "pick_and_place", "object": {"type": "ball", "color": "red", "word": "bola"},
      "target": {"type": "basket", "color": "green", "word": "cesto"}}
  -> 🔊 "Vale, cojo la bola roja y la meto en el cesto verde."
```

The robot is not moved yet; the printed command is what the next demo hands to the arm.

## How it works

| Step | Tool | Where it runs |
| --- | --- | --- |
| Speech → text | faster-whisper `small` (es, en); `xezpeleta/whisper-small-eu-ct2` (eu, a Basque fine-tune, 7.6% WER on Common Voice 18) | GPU if CUDA works (int8_float16), else CPU (int8) |
| Language | detected per command (see below), or fixed with `--lang` | same |
| Text → command | Word lists per language + a few rules (no AI model) | < 1 ms |
| Text → speech | Piper voices `es_ES-davefx`, `en_US-lessac`, `eu_ES-maider` | CPU |

Everything is offline after the first run. On the GPU the two Whisper models use well under 1 GB of VRAM and only work while a command is being transcribed; `--cpu` keeps them off the GPU entirely.

### Language detection

Whisper's own language detection tells English from Spanish reliably, but in a test with two Basque voices it called Basque "Spanish" 7 times out of 10. So for each command three jobs run at once:

1. Whisper detects the language. If it is English, the audio is transcribed as English and that is it.
2. Otherwise the audio is transcribed both as Spanish (`small`) and as Basque (the Basque model).
3. The transcript that reads more like its own language wins: share of familiar words, plus a bonus if it parses into a command.

With `--text` and no `--lang`, the same word-based score picks the language. In tests: 21/21 synthetic spoken commands (five voices) and 64/64 typed sentences got the right language.

The parser understands:

| | Spanish | English | Basque |
| --- | --- | --- | --- |
| Objects | bola, pelota, cubo, dado | ball, cube, block | bola, pilota, kubo |
| Any other object | the noun after el/la/un/una/los/las/unos/unas ("coge el plátano", "coge las peras") | the noun after the/a/an ("grab the apple") | the word next to the verb ("hartu platanoa", "platanoa hartu") |
| Places | cesto, cesta, caja, mesa, izquierda, derecha | basket, box, bin, table, left, right | saski, otarre, kutxa, mahai, ezker, eskuin |
| Any object (the colour decides) | objeto, pieza, cosa, elemento, figura, or just "el rojo" | object, thing, item, piece, one, or just "the red" | objektu, gauza, pieza, or just "gorria" |
| Colours | rojo, verde, azul, amarillo | red, green, blue, yellow | gorri, berde, urdin, horia |
| Greetings | hola, buenas | hello, hi, hey | kaixo, egun on, arratsalde on, gabon |

Each object and place carries the word that was said, which the Spanish reply repeats ("la cesta roja"). A generic object comes out as `{"type": "any", "color": "red", ...}` and any other noun as `{"type": "other", "color": null, "word": "plátano"}`. Only balls and cubes need a colour, because there are several of them. Plurals ("las peras", "the pears") add `"plural": true`, meaning all of them, and the reply agrees ("las meto", "putting them"). It greets back with an example sentence, says so with an example if it cannot find anything to pick, and asks for the colour if the object has none ("¿De qué color es la bola?"). If you say where to put it but the place is unknown ("en la estantería"), it asks where instead of ignoring the destination.

Two things make it tolerant to recognition slips:

- Whisper gets example sentences with this vocabulary as context (`STT_HINT`), so it expects "en el cesto" rather than "el incesto". Silence and background noise still come out as empty text.
- Place words are matched anywhere inside a word, so a glued "incesto" still finds "cesto".

## Install

In the same environment as the simulation demos:

```powershell
python -m pip install faster-whisper==1.2.1 piper-tts==1.8.0 sounddevice==0.5.6 truststore "av>=15,<16"
```

- `av` is pinned because `lerobot 0.4.4` needs `av<16`; faster-whisper works with it.
- `truststore` makes Python trust the Windows certificate store. Norton's Web Shield re-signs HTTPS traffic, and without it Hugging Face downloads fail with `CERTIFICATE_VERIFY_FAILED`.
- numpy stays at 1.26.0.

The first run downloads about 250 MB per Whisper model and 60 MB per voice. The GPU path needs no extra install here: ctranslate2 finds the CUDA libraries that come with the PyTorch install.

## Run

```powershell
python .\demos\02_voice\06_voice_command.py              # any of the three languages, microphone
python .\demos\02_voice\06_voice_command.py --lang eu    # only Basque
python .\demos\02_voice\06_voice_command.py --text "put the blue cube in the yellow box"
python .\demos\02_voice\06_voice_command.py --cpu        # keep speech recognition off the GPU
```

Press Enter, speak, press Enter again. `q` + Enter quits. `--no-speak` skips the audio reply.

Measured on this machine (RTX 5060 Ti, 16 threads) for a 2–3 s command:

| | GPU | CPU (`--cpu`) |
| --- | --- | --- |
| Language detected automatically | ~0.35 s (max 0.65 s) | ~2.3 s |
| Language fixed with `--lang` | ~0.15 s | ~1.1 s |

Parsing takes about 0.2 ms and the spoken reply about 150 ms. Startup on the GPU takes about 12 s, because the models are warmed up so the first command is not slow.

## Using it together with Isaac Sim

Do not import the voice libraries after Isaac Sim's Kit has started: `onnxruntime` then crashes the process with a segfault. Either import them before Kit starts (like `env_loader.py` does with `h5py`), or, better, run the voice demo as its own process and pass the JSON command to the simulator.

## 07 — A model as the action classifier

`07_action_classifier.py` keeps demo 06's speech recognition, language detection and spoken replies, and replaces the rules with a classifier built like TypeSafe's Jev: it reads the sentence together with the possible answers and returns a probability for each answer in one pass, without generating text. It cannot invent an answer that is not on the list, and every decision has a confidence; a command whose weakest decision is under 0.5 is not executed (the robot asks to repeat).

```powershell
python .\demos\02_voice\train_command_model.py                       # once, ~27 min, writes models/command-gliner/best (1.1 GB)
python .\demos\02_voice\07_action_classifier.py                      # microphone, open-source classifier
python .\demos\02_voice\07_action_classifier.py --text "coge la esfera azul y déjala en la canasta verde"
python .\demos\02_voice\07_action_classifier.py --classifier jev      # needs TYPESAFE_API_KEY
```

Extra install (adds only `gliner2`, `peft`, `accelerate`; numpy, torch and transformers are untouched):

```powershell
python -m pip install "gliner2[local]==2.0.0"
```

### How Jev is used

Jev is a hosted model: `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer $TYPESAFE_API_KEY`, a `state` (our sentence) and a batch of typed questions: `choice` (pick one option; up to 255 options, each with a description), `noul` (yes/no probability) and `score` (a rubric). Answers come back keyed by question name, with `choice`, `probabilities` and `confidence` (TypeSafe's intent-routing guide sends anything under 0.5 to a human).

Demo 07 asks six questions in one request: intent, object, object colour, place, place colour (all `choice`) and plural (`noul`). Jev only picks from lists, so the words of the sentence it does not recognise are offered as candidate object names ("plátano"), the "pre-parsed value extraction" pattern from TypeSafe's cookbook. It calls the HTTP API directly with `httpx`, because the official `typesafe-sdk` would upgrade `pydantic` in the Isaac environment.

Facts from TypeSafe's docs: model `jev-1.13.0` (`jev-latest`); $0.042 per million input tokens and free output (one of our requests is a few hundred tokens); 1,200 requests/minute; 70–500 ms per call (TypeSafe's number); English is its best language, and other languages "are handled but not equally well", with Basque not mentioned. Early access only, no weights.

**Not measured:** there is no API key here, so the Jev path is only checked with a mocked response in the documented format (request shape, parsing, reply). Test it with a real key, especially in Basque, before relying on it.

### Open-source alternatives measured

Same test for all: the 64 sentences the rules were written for, plus 12 held-out sentences with words and patterns that appear neither in the rules nor in the training data (*esfera, sphere, canasta, otzara, crate, bloque*, weather and time questions, a self-correction).

| Classifier | Kind | Known 64 | Held-out 12 | Per sentence |
| --- | --- | --- | --- | --- |
| Rules (demo 06) | word lists | 64 | 4 | < 1 ms |
| bge-m3-zeroshot-v2.0 | zero-shot NLI encoder | 1 (61 unsure) | 2 | ~30 ms |
| Salamandra-2B-instruct | LLM, option scoring | 0 (all unsure) | 0 | ~180 ms |
| Qwen3-1.7B | LLM, option scoring | 17 | 5 | ~1.2 s |
| Qwen3-4B-Instruct | LLM, option scoring | 43 | 7 | ~1.5 s |
| GLiNER2.5-multi, zero-shot | schema encoder, 2 passes | 27 | 3 | ~60 ms |
| **GLiNER2.5-multi, fine-tuned (used)** | schema encoder, 1 pass | **62** | **7** (+1 asks again) | **~30 ms** |
| Jev | hosted System One | not measured | not measured | 70–500 ms (their figure) |

Notes:

- The rules' 64/64 is on the sentences they were written for; the held-out column is the fair comparison.
- The LLMs were given the same questions and scored each option's probability, with no generation. They are slow here because every option is a separate sequence, and small ones often answer confidently wrong (for example colours nobody said: plátano → yellow).
- Zero-shot GLiNER2 finds the object and place phrases well in all three languages, but its label choices are weak. Training on examples is what makes it work, which is also how Jev is built: it is trained for the decisions it makes.

### The trained model

`train_command_model.py` writes 6,000 synthetic commands (2,000 per language) from templates, with their answers, and fully fine-tunes [GLiNER2.5-multi](https://huggingface.co/fastino/gliner2.5-multi-v1) (Apache-2.0, 0.3B, mDeBERTa-v3) for 8 epochs. One pass of the result answers `intent`, `object_kind`, `object_color`, `place_kind`, `place_color` and `number`, and extracts `object_name`. Training runs in fp32: the trainer's default bf16 produced non-finite losses.

What it still gets wrong on the held-out set: *esfera* and *sphere* become "cube", *crate* becomes an unknown place (it asks where), and it ignores self-corrections ("coge el cubo azul, no, mejor el rojo" picks the blue one). Those words and corrections were kept out of the training data on purpose; adding them to the templates is the way to teach them.

Two known limitations:

- **Confidence is not calibrated:** on clear sentences the model reports 1.0, so the "ask again" safety net rarely triggers. Jev is trained for calibrated probabilities; ours is not.
- **Synthetic data only:** the training sentences come from templates written with the known sentences in mind. Record a few dozen real commands (especially Basque) before trusting the numbers.
