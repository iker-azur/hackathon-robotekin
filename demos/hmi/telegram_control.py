"""Telegram input for sorting rules and explicitly confirmed robot starts."""

from __future__ import annotations

import json
import re
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from sorting_rules import describe_settings, normalize


MODE_ACTION = re.compile(r"\b(?:pon|ponme|poner|activa|activar|cambia|cambiar|pasa|pasar|elige|elegir|selecciona|seleccionar|quiero|usa|usar|controla|controlar)\b")
MODE_NEGATION = re.compile(r"\b(?:no|nunca|desactiva|desactivar|quita|quitar|cancela|cancelar)\b")


def control_mode_from_text(command):
    command = normalize(command)
    if MODE_NEGATION.search(command):
        return None
    manual = bool(re.search(r"\b(?:manual|maestro|leader)\b", command))
    automatic = bool(re.search(r"\b(?:automatico|automatic|programa)\b", command))
    if manual == automatic:
        return None
    exact = command in ("manual", "modo manual", "maestro", "modo maestro",
                        "automatico", "modo automatico", "automatic", "modo automatic",
                        "programa", "modo programa")
    if exact or re.search(r"\bmodo\s+(?:manual|maestro|automatico|automatic|programa)\b", command) or MODE_ACTION.search(command):
        return "manual" if manual else "automatic"
    return None


class RobotStartRequests:
    """Accept one short-lived start request only while the picker is ready."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "starting"
        self.leader_available = False
        self.control_mode = "automatic"
        self.requested_control_mode = "automatic"
        self.last_seen = 0.0
        self.pending_until = 0.0
        self.home_pending_until = 0.0
        self.home_stop_requested = False
        self.report = {"note": "", "result": "", "destination": None}

    def poll(self, phase, report=None):
        if phase not in ("starting", "idle", "running", "classifying", "routing", "result", "manual", "homing"):
            raise ValueError("invalid robot phase")
        report = report if isinstance(report, dict) else {}
        destination = report.get("destination")
        if destination not in (None, "green", "white"):
            destination = None
        try:
            config_revision = max(0, int(report.get("config_revision", 0)))
        except (TypeError, ValueError):
            config_revision = 0
        settings = report.get("settings") if isinstance(report.get("settings"), dict) else {}
        defaults = report.get("defaults") if isinstance(report.get("defaults"), dict) else {}
        leader_available = bool(report.get("leader_available"))
        leader_status = str(report.get("leader_status", ""))[:240]
        control_mode = report.get("control_mode")
        now = time.monotonic()
        with self.lock:
            self.phase = phase
            self.last_seen = now
            self.leader_available = leader_available
            if control_mode in ("automatic", "manual"):
                self.control_mode = control_mode
            self.report = {
                "note": str(report.get("note", ""))[:240],
                "result": str(report.get("result", ""))[:160],
                "destination": destination,
                "settings": settings,
                "defaults": defaults,
                "config_revision": config_revision,
                "leader_status": leader_status,
            }
            start = False
            if self.pending_until:
                if now > self.pending_until or phase not in ("idle", "result"):
                    self.pending_until = 0.0
                else:
                    start = True
                    self.pending_until = 0.0
            home = False
            if self.home_pending_until:
                if (now > self.home_pending_until or phase not in ("idle", "result")
                        or self.control_mode != "automatic"):
                    self.home_pending_until = 0.0
                else:
                    home = True
                    self.home_pending_until = 0.0
            stop_home = self.home_stop_requested and phase == "homing"
            self.home_stop_requested = False
            return {"start": start, "home": home, "stop_home": stop_home,
                    "control_mode": self.requested_control_mode}

    def request_start(self):
        now = time.monotonic()
        with self.lock:
            if now - self.last_seen > 1.5:
                return False, "La aplicación de recogida no está conectada."
            if self.phase not in ("idle", "result"):
                return False, "La aplicación está ocupada; no he iniciado otro ciclo."
            if self.control_mode != "automatic" or self.requested_control_mode != "automatic":
                return False, "Cambia a Automático antes de iniciar un ciclo."
            if self.home_pending_until:
                return False, "Ya hay una vuelta a casa pendiente."
            if self.pending_until:
                return False, "Ya hay un inicio pendiente."
            self.pending_until = now + 2.0
            return True, ""

    def request_home(self):
        now = time.monotonic()
        with self.lock:
            if now - self.last_seen > 1.5:
                return False, "La aplicación de recogida no está conectada."
            if (self.phase not in ("idle", "result") or self.control_mode != "automatic"
                    or self.requested_control_mode != "automatic"):
                return False, "Vuelve a Automático y espera a que la aplicación esté en reposo."
            if self.pending_until:
                return False, "Ya hay un inicio pendiente."
            if self.home_pending_until:
                return False, "Ya hay una vuelta a casa pendiente."
            self.home_pending_until = now + 2.0
            return True, ""

    def request_home_stop(self):
        with self.lock:
            if time.monotonic() - self.last_seen > 1.5:
                return False, "La aplicación de recogida no está conectada."
            if self.phase != "homing":
                return False, "No hay una vuelta a casa en curso."
            self.home_stop_requested = True
            return True, ""

    def request_control_mode(self, mode):
        if mode not in ("automatic", "manual"):
            return False, "Modo de control no válido."
        with self.lock:
            if time.monotonic() - self.last_seen > 1.5:
                return False, "La aplicación de recogida no está conectada."
            if self.phase not in ("idle", "result", "manual"):
                return False, "Espera a que termine el ciclo automático antes de cambiar el control."
            if self.pending_until or self.home_pending_until:
                return False, "Hay una orden de movimiento pendiente; espera a que se procese."
            if mode == "manual" and not self.leader_available:
                return False, "La aplicación no detecta un leader calibrado."
            self.requested_control_mode = mode
            return True, "Modo manual solicitado." if mode == "manual" else "Modo automático solicitado."

    def snapshot(self):
        with self.lock:
            connected = time.monotonic() - self.last_seen <= 1.5
            return {"connected": connected,
                    "phase": self.phase if connected else "offline",
                    "leader_available": self.leader_available if connected else False,
                    "leader_status": self.report.get("leader_status", ""),
                    "control_mode": self.control_mode if connected else "automatic",
                    **self.report}


class TelegramControl:
    """Long-poll Telegram and route authorized messages to the local HMI."""

    def __init__(self, token, allowed_user_id, speech, sorting_rules, robot_starts):
        self.token = token
        self.allowed_user_id = int(allowed_user_id) if allowed_user_id else None
        self.speech = speech
        self.sorting_rules = sorting_rules
        self.robot_starts = robot_starts
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="telegram-control", daemon=True)
        self.pending_start_until = 0.0
        self.pending_home_until = 0.0
        self.pending_mode_until = 0.0
        self.pending_control_mode = None

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()

    def _api(self, method, payload=None, timeout=30):
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        body = json.dumps(payload or {}).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read())
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("No se pudo comunicar con Telegram.") from exc
        if not result.get("ok"):
            raise RuntimeError(result.get("description", "Telegram rechazó la solicitud."))
        return result["result"]

    def _send(self, chat_id, text):
        self._api("sendMessage", {"chat_id": chat_id, "text": text}, timeout=10)

    def _run(self):
        try:
            bot = self._api("getMe", timeout=10)
            webhook = self._api("getWebhookInfo", timeout=10)
            if webhook.get("url"):
                raise RuntimeError("El bot tiene un webhook activo; desactívalo antes de usar long polling.")
        except Exception as exc:
            print(f"Telegram desactivado: {exc}", flush=True)
            return

        try:
            pending = self._api("getUpdates", {
                "offset": -1, "timeout": 0, "allowed_updates": ["message"],
            }, timeout=5)
            offset = pending[-1]["update_id"] + 1 if pending else 0
        except Exception as exc:
            print(f"No se pudo preparar la cola de Telegram: {exc}", flush=True)
            return
        if self.allowed_user_id is None:
            print(f"Telegram activo: @{bot.get('username', 'bot')} · falta ROBOT_TELEGRAM_USER_ID", flush=True)
        else:
            print(f"Telegram activo: @{bot.get('username', 'bot')} · cuenta autorizada configurada", flush=True)
        while not self.stop.is_set():
            try:
                updates = self._api("getUpdates", {
                    "offset": offset, "timeout": 20,
                    "allowed_updates": ["message"],
                }, timeout=25)
                for update in updates:
                    offset = max(offset, update["update_id"] + 1)
                    try:
                        self._handle_message(update.get("message", {}))
                    except Exception as exc:
                        print(f"Error procesando un mensaje de Telegram: {exc}", flush=True)
            except Exception as exc:
                print(f"Telegram temporalmente no disponible: {exc}", flush=True)
                self.stop.wait(3)

    def _handle_message(self, message):
        chat = message.get("chat", {})
        sender = message.get("from", {})
        chat_id = chat.get("id")
        user_id = sender.get("id")
        if not chat_id or not user_id or chat.get("type") != "private":
            return

        text = message.get("text", "").strip()
        command_token = text.split(maxsplit=1)[0].split("@", 1)[0].lower() if text else ""
        if command_token == "/id":
            self._send(chat_id, f"Tu ID de Telegram es {user_id}.")
            return
        if self.allowed_user_id is None:
            self._send(chat_id, "El bot aún no está vinculado a tu cuenta. Envía /id y configura ROBOT_TELEGRAM_USER_ID en el PC.")
            return
        if user_id != self.allowed_user_id:
            self._send(chat_id, "Esta cuenta no está autorizada. Comprueba que ROBOT_TELEGRAM_USER_ID coincide con tu /id.")
            return

        if text:
            self._handle_text(chat_id, text)
            return

        audio = message.get("voice") or message.get("audio")
        if audio:
            self._handle_audio(chat_id, audio)

    def _handle_audio(self, chat_id, audio):
        try:
            file_info = self._api("getFile", {"file_id": audio["file_id"]}, timeout=10)
            file_path = file_info["file_path"]
            url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
            with urlopen(url, timeout=20) as response:
                content = response.read(20_000_001)
            if len(content) > 20_000_000:
                raise ValueError("La nota de voz supera el límite de 20 MB.")
            self._send(chat_id, "Transcribiendo la nota de voz…")
            transcript = self.speech.transcribe(content)["text"]
            self._send(chat_id, f"Transcripción: {transcript}")
            self._handle_text(chat_id, transcript)
        except (ValueError, RuntimeError) as exc:
            self._send(chat_id, f"No pude procesar el audio: {exc}")
        except Exception:
            print("No se pudo descargar o transcribir un audio de Telegram.", flush=True)
            self._send(chat_id, "No pude procesar el audio. Comprueba la conexión y vuelve a enviarlo.")

    def _handle_text(self, chat_id, text):
        command = normalize(text)
        if command in ("start", "ayuda", "help"):
            self._send(chat_id, "Envía texto o una nota de voz para configurar los destinos. Escribe EMPEZAR y CONFIRMAR para iniciar un ciclo, VUELVE A CASA y CONFIRMAR para regresar a la posición guardada, o PON EL MODO MANUAL y CONFIRMAR para seguir al maestro. Para volver, escribe PON EL MODO AUTOMÁTICO. CANCELAR detiene una vuelta a casa en curso.")
            return
        requested_mode = control_mode_from_text(command)
        if requested_mode == "manual":
            self.pending_start_until = 0.0
            self.pending_home_until = 0.0
            self.pending_control_mode = "manual"
            self.pending_mode_until = time.monotonic() + 45
            self._send(chat_id, "¿Activo el modo Manual? El follower seguirá al maestro. Responde CONFIRMAR en los próximos 45 segundos.")
            return
        if requested_mode == "automatic":
            self.pending_start_until = 0.0
            self.pending_home_until = 0.0
            self.pending_mode_until = 0.0
            self.pending_control_mode = None
            _, detail = self.robot_starts.request_control_mode("automatic")
            self._send(chat_id, detail or "Modo Automático solicitado.")
            return
        if command in ("empezar", "empieza", "empezar ciclo", "empieza el ciclo",
                       "iniciar", "inicia", "iniciar ciclo", "inicia el ciclo",
                       "comenzar", "comienza", "comenzar ciclo", "comienza el ciclo"):
            self.pending_start_until = time.monotonic() + 45
            self.pending_home_until = 0.0
            self.pending_mode_until = 0.0
            self.pending_control_mode = None
            self._send(chat_id, "¿Inicio un ciclo físico ahora? Responde CONFIRMAR en los próximos 45 segundos.")
            return
        if command in ("casa", "home", "vuelve a casa", "vuelve a casa por favor", "volver a casa", "regresa a casa",
                       "ir a casa", "vuelve a la posicion de casa", "volver a la posicion de casa",
                       "regresa a la posicion de casa", "posicion de casa", "ve a casa"):
            self.pending_start_until = 0.0
            self.pending_home_until = time.monotonic() + 45
            self.pending_mode_until = 0.0
            self.pending_control_mode = None
            self._send(chat_id, "¿Muevo el follower a la posición de casa? Responde CONFIRMAR en los próximos 45 segundos.")
            return
        if command == "cancelar":
            pending = (time.monotonic() <= self.pending_start_until or
                       time.monotonic() <= self.pending_home_until or
                       self.pending_control_mode is not None)
            self.pending_start_until = 0.0
            self.pending_home_until = 0.0
            self.pending_mode_until = 0.0
            self.pending_control_mode = None
            if pending:
                self._send(chat_id, "Solicitud pendiente cancelada.")
            else:
                accepted, detail = self.robot_starts.request_home_stop()
                self._send(chat_id, detail or "Deteniendo la vuelta a casa; el follower mantendrá la posición medida.")
            return
        if command == "confirmar":
            if self.pending_control_mode and time.monotonic() <= self.pending_mode_until:
                mode = self.pending_control_mode
                self.pending_control_mode = None
                self.pending_mode_until = 0.0
                _, detail = self.robot_starts.request_control_mode(mode)
                self._send(chat_id, detail or "Modo Manual solicitado.")
                return
            if self.pending_mode_until:
                self.pending_mode_until = 0.0
                self.pending_control_mode = None
                self._send(chat_id, "La confirmación del modo ha caducado. Vuelve a solicitarlo.")
                return
            if time.monotonic() <= self.pending_home_until:
                self.pending_home_until = 0.0
                _, detail = self.robot_starts.request_home()
                self._send(chat_id, detail or "Confirmado. La aplicación moverá el follower a casa si sigue en reposo.")
                return
            if time.monotonic() > self.pending_start_until:
                self.pending_start_until = 0.0
                self._send(chat_id, "No hay una solicitud de inicio vigente. Escribe EMPEZAR para solicitar una.")
                return
            self.pending_start_until = 0.0
            _, detail = self.robot_starts.request_start()
            self._send(chat_id, detail or "Confirmación recibida. La aplicación iniciará un ciclo si sigue en reposo.")
            return

        self.pending_start_until = 0.0
        self.pending_home_until = 0.0
        self.pending_mode_until = 0.0
        self.pending_control_mode = None
        result = self.sorting_rules.apply(text)
        if result["status"] == "active":
            routes = result["rule"]["routes"]
            destination = lambda key: "verde" if routes[key]["destination"] == "green" else "blanca"
            message = f"Regla aplicada: con perno → caja {destination('with_pin')}; sin perno → caja {destination('without_pin')}."
            if result.get("changes"):
                message += f" Ajustes: {describe_settings(result['changes'])}. Se aplican al siguiente ciclo."
            self._send(chat_id, message)
        elif result["status"] == "settings_updated":
            changed = describe_settings(result.get("changes", {}))
            self._send(chat_id, f"Configuración actualizada ({changed}). Se aplicará al siguiente ciclo.")
        elif result["status"] == "restored":
            self._send(chat_id, result.get("message", "Configuración restaurada.") + " Se aplicará al siguiente ciclo.")
        elif result["status"] == "stopped":
            self._send(chat_id, "Regla de clasificación desactivada.")
        else:
            self._send(chat_id, result.get("message", "No entendí la configuración. Repítela."))
