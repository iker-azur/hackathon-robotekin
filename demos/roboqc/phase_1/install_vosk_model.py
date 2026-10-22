#!/usr/bin/env python3
"""Opt-in helper to download and extract the Vosk Spanish model.

This script is NOT run automatically. Execute it only if you want an
automatic download. It uses only the Python standard library.
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import shutil
import tempfile


MODELS = {
    "small": ("https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip", "models/vosk-model-small-es"),
    "es": ("https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip", "models/vosk-model-es-0.42"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga (opt-in) del modelo Vosk ES.")
    ap.add_argument("--model", choices=list(MODELS), default="small", help="small (~40 MB) o es (grande, ~1.4 GB).")
    ap.add_argument("--url", default=None, help="URL del ZIP del modelo (anula --model).")
    ap.add_argument("--dest", default=None, help="Directorio de destino (anula --model).")
    ap.add_argument("--force", action="store_true", help="Sobrescribe si ya existe.")
    ap.add_argument("--yes", action="store_true", help="Confirma la descarga sin preguntar.")
    args = ap.parse_args()
    default_url, default_dest = MODELS[args.model]
    args.url = args.url or default_url
    args.dest = args.dest or default_dest

    if os.path.exists(args.dest) and not args.force:
        print(f"El destino ya existe: {args.dest}. Usa --force para sobrescribir.")
        return

    if not args.yes:
        resp = input(f"Descargar modelo desde {args.url} a {args.dest}? [s/N]: ").strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado.")
            return

    os.makedirs(os.path.dirname(args.dest) or ".", exist_ok=True)
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, "model.zip")
    try:
        print("Descargando...")
        urllib.request.urlretrieve(args.url, zip_path)
        print("Extrayendo...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmpdir)
        # find extracted model dir
        extracted = None
        for name in os.listdir(tmpdir):
            full = os.path.join(tmpdir, name)
            if os.path.isdir(full) and name.startswith("vosk-model"):
                extracted = full
                break
        if extracted is None:
            # maybe nested
            for root, dirs, files in os.walk(tmpdir):
                for d in dirs:
                    if d.startswith("vosk-model"):
                        extracted = os.path.join(root, d)
                        break
                if extracted:
                    break
        if extracted is None:
            print("ERROR: no se encontro el directorio del modelo en el ZIP.")
            return
        if os.path.exists(args.dest):
            shutil.rmtree(args.dest)
        shutil.move(extracted, args.dest)
        print(f"Modelo instalado en: {args.dest}")
    except Exception as e:
        print(f"ERROR durante la descarga/extraccion: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
