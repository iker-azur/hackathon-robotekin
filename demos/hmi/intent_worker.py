"""JSON-lines worker for GLiNER2, run by the isolated intent environment."""

from contextlib import redirect_stdout
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
MODEL_SOURCE = ROOT.parent / '10_bin_intent_model'
sys.path.insert(0, str(MODEL_SOURCE))


def reply(message):
    print(json.dumps(message, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision')
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            import torch
            from huggingface_hub import snapshot_download
            from intent_model import BinIntentModel

            torch.set_num_threads(2)
            path = Path(args.model)
            if not path.is_dir():
                path = Path(snapshot_download(args.model, revision=args.revision,
                                              allow_patterns=['best/*', 'calibration.json', 'labels.json']))
            if (path / 'best').is_dir():
                path = path / 'best'
            model = BinIntentModel(path, device='cpu')
        reply({'ready': True})
    except Exception:
        traceback.print_exc(file=sys.stderr)
        reply({'error': 'No se pudo cargar el modelo de intenciones. Revisa la instalación y la primera descarga.'})
        return 1
    for line in sys.stdin:
        try:
            message = json.loads(line)
            with redirect_stdout(sys.stderr):
                plan = model.plan(message['text'])
            reply({'status': plan.status, 'intent': plan.reading.intent,
                   'confidence': plan.reading.confidence, 'probabilities': plan.reading.probabilities,
                   'rules': [{**asdict(rule), 'confidence': rule.confidence} for rule in plan.rules],
                   'note': plan.note})
        except Exception:
            traceback.print_exc(file=sys.stderr)
            reply({'error': 'No se pudo interpretar la orden con el modelo de intenciones.'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
