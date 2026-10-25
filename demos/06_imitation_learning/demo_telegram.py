"""Orders from Telegram for ``30_full_demo_app.py``: a text or a voice note sent to the bot is read like a spoken order.

The bot token comes from the environment (``TELEGRAM_BOT_TOKEN``) and is never written anywhere. Only the chats listed
in ``--telegram-chats`` are obeyed; any other chat is told its id and ignored, so a stranger who finds the bot cannot
move the robot. Messages sent before the app started, or more than a minute ago, are dropped for the same reason.

The bridge long-polls ``getUpdates`` in one thread and sends replies from another, so neither blocks the control loop.
Voice notes (OGG/Opus) are decoded with the PyAV that faster-whisper already depends on.
"""

from __future__ import annotations

import io
import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.telegram.org"
POLL_S = 25
MAX_AGE_S = 60  # an order older than this is stale: the scene has changed
MAX_VOICE_S = 30
SAMPLE_RATE = 16000
HELP = ("Envíame una orden escrita o una nota de voz (es / en / eu), por ejemplo: «mueve las piezas buenas a la caja "
        "blanca y las malas a la verde», «todas a la caja verde» o «stop».")


class TelegramError(RuntimeError):
    pass


def decode_voice(data: bytes):
    """16 kHz mono float32 samples of an OGG/Opus (or any PyAV-readable) voice note."""
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(data), sampling_rate=SAMPLE_RATE)


def chat_ids(text: str | None) -> set[int]:
    """``"123, -456"`` -> {123, -456}; empty or None -> no chat."""
    ids = set()
    for part in (text or "").replace(";", ",").split(","):
        if part.strip():
            try:
                ids.add(int(part))
            except ValueError:
                raise ValueError(f"not a Telegram chat id: {part.strip()!r}") from None
    return ids


class TelegramBridge:
    def __init__(self, token: str, chats, on_text, on_audio, call=None, fetch=None, decode=decode_voice,
                 clock=time.time) -> None:
        """``on_text(text, reply)`` and ``on_audio(samples, reply)`` get each order; ``reply(text)`` answers its chat."""
        self.token, self.chats = token, set(chats)
        self.on_text, self.on_audio, self.decode, self.clock = on_text, on_audio, decode, clock
        self.call = call or self._call
        self.fetch = fetch or self._fetch
        self.name, self.error, self.offset = "", "", None
        self.outbox: queue.Queue = queue.Queue()
        self.closed = threading.Event()
        self.threads: list[threading.Thread] = []

    # -- the Bot API (errors never carry the token: it is part of every URL) ---------------------------------------
    def _call(self, method: str, **params):
        timeout = params.get("timeout", 0) + 10
        request = urllib.request.Request(f"{API}/bot{self.token}/{method}", urllib.parse.urlencode(params).encode())
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            raise TelegramError(f"{method}: HTTP {exc.code}") from None
        except Exception as exc:
            raise TelegramError(f"{method}: {type(exc).__name__}") from None
        if not body.get("ok"):
            raise TelegramError(f"{method}: {body.get('description', 'refused')}")
        return body["result"]

    def _fetch(self, file_path: str) -> bytes:
        try:
            with urllib.request.urlopen(f"{API}/file/bot{self.token}/{file_path}", timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise TelegramError(f"download: HTTP {exc.code}") from None
        except Exception as exc:
            raise TelegramError(f"download: {type(exc).__name__}") from None

    # -- life cycle -------------------------------------------------------------------------------------------------
    def start(self) -> str:
        """Check the token, skip what was sent before now and start listening. Returns ``@bot_name``."""
        self.name = "@" + self.call("getMe").get("username", "bot")
        backlog = self.call("getUpdates", offset=-1, timeout=0)
        if backlog:
            self.offset = backlog[-1]["update_id"] + 1
        for target in (self._poll, self._send_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)
        return self.name

    def close(self) -> None:
        self.closed.set()  # the poll may sit in a long request for up to POLL_S; it is a daemon thread

    def send(self, chat: int, text: str) -> None:
        """Queue a reply; the network call happens in the sender thread."""
        self.outbox.put((chat, text[:4000]))

    def _send_loop(self) -> None:
        while not self.closed.is_set():
            try:
                chat, text = self.outbox.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.call("sendMessage", chat_id=chat, text=text)
            except TelegramError as exc:
                self.error = str(exc)

    def _poll(self) -> None:
        while not self.closed.is_set():
            params = {"timeout": POLL_S, "allowed_updates": json.dumps(["message"])}
            if self.offset is not None:
                params["offset"] = self.offset
            try:
                updates = self.call("getUpdates", **params)
            except TelegramError as exc:
                self.error = str(exc)
                self.closed.wait(3)
                continue
            for update in updates:
                self.offset = update["update_id"] + 1
                try:
                    self.handle(update.get("message") or {})
                except Exception as exc:  # one bad message must not stop the bridge
                    self.error = f"{type(exc).__name__}: {exc}"

    # -- one message ------------------------------------------------------------------------------------------------
    def handle(self, message: dict) -> None:
        if "chat" not in message:
            return
        chat = message["chat"]["id"]
        if chat not in self.chats:
            self.send(chat, f"Chat no autorizado (id {chat}). Arranca la app con --telegram-chats {chat}.")
            return
        if self.clock() - message.get("date", 0) > MAX_AGE_S:
            return

        def reply(text: str) -> None:
            self.send(chat, text)

        text = message.get("text")
        media = message.get("voice") or message.get("audio")
        if text is not None:
            if text.startswith(("/start", "/help")):
                reply(HELP)
            else:
                self.on_text(text, reply)
        elif media:
            if media.get("duration", 0) > MAX_VOICE_S:
                reply(f"Nota de voz demasiado larga (máximo {MAX_VOICE_S} s).")
                return
            reply("🎧 Recibido, transcribiendo…")
            info = self.call("getFile", file_id=media["file_id"])
            self.on_audio(self.decode(self.fetch(info["file_path"])), reply)
        else:
            reply(HELP)
