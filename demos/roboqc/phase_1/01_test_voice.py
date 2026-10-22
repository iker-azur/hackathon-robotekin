#!/usr/bin/env python3
"""Offline Spanish speech recognition test utility (RoboQC)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from roboqc.speech.recorder import list_devices, record_until_stop, save_wav, load_wav, is_available as recorder_available
from roboqc.speech.engines import ENGINES, DEFAULT_ENGINE, available_engines, get_engine


def main() -> None:
    ap = argparse.ArgumentParser(description="Prueba de reconocimiento de voz offline (RoboQC).")
    ap.add_argument("--list-devices", action="store_true", help="Lista dispositivos de entrada.")
    ap.add_argument("--list-engines", action="store_true", help="Lista motores de voz y su disponibilidad.")
    ap.add_argument("--engine", type=str, default=DEFAULT_ENGINE, choices=list(ENGINES), help="Motor de voz.")
    ap.add_argument("--compare", action="store_true", help="Transcribe con todos los motores disponibles.")
    ap.add_argument("--mic", action="store_true", help="Graba desde microfono hasta ENTER.")
    ap.add_argument("--device", type=int, default=None, help="Indice de dispositivo de entrada.")
    ap.add_argument("--samplerate", type=int, default=None, help="Frecuencia de muestreo.")
    ap.add_argument("--file", type=str, default=None, help="WAV de entrada.")
    ap.add_argument("--save-wav", type=str, default=None, help="Guardar grabacion WAV.")
    ap.add_argument("--save-transcript", type=str, default=None, help="Guardar transcripcion en archivo.")
    args = ap.parse_args()

    if args.list_devices:
        devs = list_devices()
        if not devs:
            print("No hay dispositivos de entrada disponibles o sounddevice no esta instalado.")
            return
        for d in devs:
            print(f"[{d['index']}] {d['name']} (canales={d['channels']}, sr={d['default_samplerate']})")
        return

    if args.list_engines:
        for name, engine in ENGINES.items():
            state = "disponible" if engine.available() else "no disponible"
            print(f"{name:14} {engine.label:16} {state}")
        return

    if not args.file and not args.mic:
        ap.print_help()
        return

    if args.file:
        pcm, sr = load_wav(args.file)
    else:
        if not recorder_available():
            print("ERROR: sounddevice no disponible.")
            return
        try:
            pcm, sr = record_until_stop(device=args.device, samplerate=args.samplerate)
        except Exception as e:
            print(f"ERROR de grabacion: {e}")
            return
        if not pcm:
            print("No se capturo audio (silencio o error).")
            return
        if args.save_wav:
            save_wav(args.save_wav, pcm, sr)

    names = available_engines() if args.compare else [args.engine]
    if not names:
        print("ERROR: ningun motor de voz disponible. Revisa --list-engines.")
        return

    results = []
    for name in names:
        engine = get_engine(name)
        if not engine.available():
            print(f"[{name}] no disponible; se omite.")
            continue
        try:
            text, latency = engine.transcribe(pcm, sr)
        except Exception as e:
            print(f"[{name}] ERROR al transcribir: {e}")
            continue
        results.append((name, text, latency))

    for name, text, latency in results:
        print(f"[{name}] ({latency:.2f} s) {text if text else '(sin texto reconocido)'}")
    if args.save_transcript and results:
        with open(args.save_transcript, "w", encoding="utf-8") as f:
            f.write(results[0][1])


if __name__ == "__main__":
    main()
