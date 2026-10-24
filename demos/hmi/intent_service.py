"""Keep the trained intent model loaded without importing its dependencies into LeRobot."""

import json
import os
from pathlib import Path
import subprocess
import threading

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = 'IkerAzurmendi/bin-intent-gliner2'
DEFAULT_REVISION = '718261576991fc57319f39a01f6e48b075380e83'
WORKER_PYTHON = ROOT / '.venv-intent' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
INSTALL = ('uv venv --python .venv/bin/python demos/hmi/.venv-intent\n'
           'uv pip install --python demos/hmi/.venv-intent/bin/python -r demos/hmi/requirements-intent.txt')


class IntentUnavailable(RuntimeError):
    pass


class IntentService:
    def __init__(self, model=DEFAULT_MODEL, python=WORKER_PYTHON):
        self.model_name, self.python = str(model), Path(python)
        self.process = None
        self.ready = False
        self.lock = threading.Lock()

    def status(self):
        return {'available': self.python.is_file(), 'loaded': self.ready and self.process.poll() is None,
                'engine': 'GLiNER2 local', 'model': self.model_name, 'install_command': INSTALL}

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise IntentUnavailable('El proceso del modelo de intenciones se ha detenido. Revisa el registro del servidor.')
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IntentUnavailable('El modelo de intenciones devolvió una respuesta no válida.') from exc
        if message.get('error'):
            raise IntentUnavailable(message['error'])
        return message

    def load(self):
        if self.ready and self.process.poll() is None:
            return
        if not self.python.is_file():
            raise IntentUnavailable('Falta preparar el entorno del modelo de intenciones. Ejecuta:\n' + INSTALL)
        self.close()
        command = [str(self.python), '-u', str(ROOT / 'intent_worker.py'), '--model', self.model_name]
        if self.model_name == DEFAULT_MODEL:
            command += ['--revision', DEFAULT_REVISION]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding='utf-8', bufsize=1,
                                        env={**os.environ, 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                                             'TOKENIZERS_PARALLELISM': 'false'})
        try:
            if self._read().get('ready') is not True:
                raise IntentUnavailable('El modelo de intenciones no terminó de arrancar.')
            self.ready = True
        except Exception:
            self.close()
            raise

    def predict(self, text):
        if not self.lock.acquire(blocking=False):
            raise IntentUnavailable('Ya se está interpretando otra orden. Espera e inténtalo de nuevo.')
        try:
            self.load()
            self.process.stdin.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            self.process.stdin.flush()
            return self._read()
        except (OSError, ValueError) as exc:
            self.close()
            raise IntentUnavailable('No se pudo comunicar con el modelo de intenciones.') from exc
        finally:
            self.lock.release()

    def close(self):
        self.ready = False
        if self.process is not None:
            process, self.process = self.process, None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            process.stdout.close()
