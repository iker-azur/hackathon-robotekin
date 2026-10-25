"""The Telegram bridge: who may give orders, what it forwards and what it sends back, without any network."""

import importlib.util
import sys
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np


def load():
    path = Path(__file__).resolve().parent.parent / "demo_telegram.py"
    spec = importlib.util.spec_from_file_location("demo_telegram_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


telegram = load()
NOW = 1_000_000.0


class FakeApi:
    """A scripted Bot API: ``updates`` are handed out once, every call is recorded."""

    def __init__(self, updates=()):
        self.updates, self.calls, self.sent = list(updates), [], []

    def __call__(self, method, **params):
        self.calls.append((method, params))
        if method == "getMe":
            return {"username": "bin_bot"}
        if method == "getUpdates":
            if params.get("offset") == -1:
                return self.updates[-1:]  # what was sent before the app started
            offset = params.get("offset") or 0  # Telegram forgets every update below the confirmed offset
            taken = [u for u in self.updates if u["update_id"] >= offset]
            self.updates = []
            time.sleep(0.02 if taken else 0.1)
            return taken
        if method == "getFile":
            return {"file_path": f"voice/{params['file_id']}.oga"}
        if method == "sendMessage":
            self.sent.append((params["chat_id"], params["text"]))
            return {}
        raise AssertionError(method)


def message(chat=7, text=None, voice=None, date=NOW, update_id=1):
    body = {"chat": {"id": chat}, "date": date}
    if text is not None:
        body["text"] = text
    if voice is not None:
        body["voice"] = voice
    return {"update_id": update_id, "message": body}


def bridge(api=None, chats=(7,), decode=None, fetch=None):
    api = api or FakeApi()
    texts, audios = [], []
    made = telegram.TelegramBridge(
        "123:SECRET", chats, lambda text, reply: texts.append((text, reply)),
        lambda audio, reply: audios.append((audio, reply)), call=api,
        fetch=fetch or (lambda path: b"ogg:" + path.encode()),
        decode=decode or (lambda data: np.zeros(160, dtype=np.float32)), clock=lambda: NOW + 5)
    made.texts, made.audios = texts, audios
    return made, api


def wait_for(condition, seconds=2.0):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


class ChatIdTests(unittest.TestCase):
    def test_ids_are_parsed_from_a_comma_list(self):
        self.assertEqual(telegram.chat_ids("123, -456;789"), {123, -456, 789})
        self.assertEqual(telegram.chat_ids(None), set())
        self.assertEqual(telegram.chat_ids(""), set())

    def test_a_username_is_not_an_id(self):
        with self.assertRaises(ValueError):
            telegram.chat_ids("@iker")


class HandleTests(unittest.TestCase):
    def test_text_from_an_allowed_chat_is_forwarded_with_a_reply_to_that_chat(self):
        b, _ = bridge()
        b.handle(message(text="todas a la caja verde")["message"])
        text, reply = b.texts[0]
        self.assertEqual(text, "todas a la caja verde")
        reply("hecho")
        self.assertEqual(b.outbox.get_nowait(), (7, "hecho"))

    def test_an_unknown_chat_is_told_its_id_and_not_obeyed(self):
        b, _ = bridge()
        b.handle(message(chat=99, text="stop")["message"])
        self.assertEqual(b.texts, [])
        chat, text = b.outbox.get_nowait()
        self.assertEqual(chat, 99)
        self.assertIn("--telegram-chats 99", text)

    def test_with_no_allowed_chat_nobody_is_obeyed(self):
        b, _ = bridge(chats=())
        b.handle(message(text="stop")["message"])
        self.assertEqual(b.texts, [])

    def test_a_stale_message_is_dropped(self):
        b, _ = bridge()
        b.handle(message(text="stop", date=NOW - 3600)["message"])
        self.assertEqual((b.texts, b.outbox.empty()), ([], True))

    def test_start_and_help_commands_answer_with_help(self):
        b, _ = bridge()
        b.handle(message(text="/start")["message"])
        self.assertEqual(b.texts, [])
        self.assertEqual(b.outbox.get_nowait(), (7, telegram.HELP))

    def test_a_voice_note_is_downloaded_decoded_and_forwarded(self):
        decoded = []
        b, api = bridge(decode=lambda data: decoded.append(data) or np.ones(160, dtype=np.float32))
        b.handle(message(voice={"file_id": "abc", "duration": 4})["message"])
        self.assertEqual(decoded, [b"ogg:voice/abc.oga"])
        audio, reply = b.audios[0]
        self.assertEqual(audio.shape, (160,))
        self.assertIn("Recibido", b.outbox.get_nowait()[1])
        reply("ok")
        self.assertEqual(b.outbox.get_nowait(), (7, "ok"))

    def test_a_long_voice_note_is_refused_without_downloading(self):
        b, api = bridge()
        b.handle(message(voice={"file_id": "abc", "duration": 300})["message"])
        self.assertEqual(b.audios, [])
        self.assertNotIn("getFile", [method for method, _ in api.calls])

    def test_other_kinds_of_message_get_the_help(self):
        b, _ = bridge()
        body = message()["message"]
        body["photo"] = [{}]
        b.handle(body)
        self.assertEqual(b.outbox.get_nowait(), (7, telegram.HELP))


class RunTests(unittest.TestCase):
    def test_start_checks_the_token_and_skips_what_was_sent_before(self):
        api = FakeApi([message(text="old order", update_id=41)])
        b, _ = bridge(api)
        self.addCleanup(b.close)
        self.assertEqual(b.start(), "@bin_bot")
        self.assertEqual(b.offset, 42)
        time.sleep(0.2)
        self.assertEqual(b.texts, [])

    def test_polling_forwards_new_messages_and_sends_the_replies(self):
        api = FakeApi()
        b, _ = bridge(api)
        self.addCleanup(b.close)
        b.start()
        api.updates = [message(text="stop", update_id=5)]
        self.assertTrue(wait_for(lambda: b.texts))
        b.texts[0][1]("Parado")
        self.assertTrue(wait_for(lambda: api.sent))
        self.assertEqual(api.sent, [(7, "Parado")])
        self.assertEqual(b.offset, 6)

    def test_a_failing_message_does_not_stop_the_bridge(self):
        api = FakeApi()
        b, _ = bridge(api, decode=lambda data: 1 / 0)
        self.addCleanup(b.close)
        b.start()
        api.updates = [message(voice={"file_id": "x", "duration": 1}, update_id=1)]
        self.assertTrue(wait_for(lambda: b.error))
        api.updates = [message(text="stop", update_id=2)]
        self.assertTrue(wait_for(lambda: b.texts))


class HttpTests(unittest.TestCase):
    """The real urllib path against a local server that speaks the little of the Bot API the bridge uses."""

    def setUp(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen = self.seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"])).decode()
                seen.append((self.path, body))
                if self.path.endswith("/getMe"):
                    reply = {"ok": True, "result": {"username": "bin_bot"}}
                elif self.path.endswith("/sendMessage"):
                    reply = {"ok": False, "description": "chat not found"}
                else:
                    reply = {"ok": True, "result": []}
                payload = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                seen.append((self.path, ""))
                self.send_response(200)
                self.send_header("Content-Length", "3")
                self.end_headers()
                self.wfile.write(b"ogg")

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)  # cleanups run last-in first-out: shut down, then close
        self.addCleanup(self.server.shutdown)
        patcher = patch.object(telegram, "API", f"http://127.0.0.1:{self.server.server_port}")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_calls_are_form_encoded_posts_to_the_bot_url_and_results_are_unwrapped(self):
        b = telegram.TelegramBridge("123:SECRET", (), None, None)
        self.assertEqual(b.call("getMe"), {"username": "bin_bot"})
        self.assertEqual(b.call("getUpdates", offset=5, timeout=0), [])
        self.assertEqual(self.seen[0][0], "/bot123:SECRET/getMe")
        self.assertEqual(self.seen[1], ("/bot123:SECRET/getUpdates", "offset=5&timeout=0"))

    def test_a_refusal_becomes_a_telegram_error_with_the_reason(self):
        b = telegram.TelegramBridge("123:SECRET", (), None, None)
        with self.assertRaises(telegram.TelegramError) as caught:
            b.call("sendMessage", chat_id=1, text="x")
        self.assertIn("chat not found", str(caught.exception))

    def test_files_are_downloaded_from_the_file_url(self):
        b = telegram.TelegramBridge("123:SECRET", (), None, None)
        self.assertEqual(b.fetch("voice/file_1.oga"), b"ogg")
        self.assertEqual(self.seen[0][0], "/file/bot123:SECRET/voice/file_1.oga")


class TokenTests(unittest.TestCase):
    def test_http_errors_never_carry_the_token(self):
        b = telegram.TelegramBridge("123:SECRET", (), None, None)
        failure = urllib.error.HTTPError(f"{telegram.API}/bot123:SECRET/getMe", 401, "Unauthorized", {}, None)
        with patch.object(telegram.urllib.request, "urlopen", side_effect=failure):
            with self.assertRaises(telegram.TelegramError) as caught:
                b.call("getMe")
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertIn("401", str(caught.exception))

    def test_network_errors_never_carry_the_token(self):
        b = telegram.TelegramBridge("123:SECRET", (), None, None)
        with patch.object(telegram.urllib.request, "urlopen", side_effect=OSError("cannot reach /bot123:SECRET/")):
            with self.assertRaises(telegram.TelegramError) as caught:
                b.call("getMe")
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
