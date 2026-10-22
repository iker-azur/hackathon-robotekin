"""Panel: servidor, órdenes, imágenes y visualización (sin Isaac, sin micrófono)."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from binpick import config as C
from binpick import segmentation as S
from binpick import viz
from binpick.server import STATIC, build_app
from binpick.telemetry import BinTelemetry
from binpick.voice import TextOnly
from synth import piece, render, scene_and_ctx

SCENE, CTX = scene_and_ctx()


def client(voice=None, ready=True):
    tele = BinTelemetry()
    tele.set(ready=ready, status="esperando")
    return TestClient(build_app(tele, voice)), tele


def test_index_is_spanish_and_has_the_controls():
    c, _ = client()
    r = c.get("/")
    assert r.status_code == 200 and "no-store" in r.headers["cache-control"]
    html = r.text
    for needle in ('lang="es"', "Iniciar", "Detener", "Reiniciar", "Aleatorizar", "Hablar", "Segmentación",
                   "He oído", "He entendido", "caja izquierda", "caja derecha", "Registro", "Resultado"):
        assert needle in html, needle
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html   # sin recursos externos


def test_state_endpoint_has_the_fields_the_page_reads():
    c, tele = client()
    s = c.get("/api/state").json()
    for k in ("status", "heard", "order", "instances", "boxes", "capacity", "policy", "segmenter", "voice", "ready", "frames", "events"):
        assert k in s, k
    assert s["policy"]["kind"] == "guion"


def test_order_is_queued_and_rejected_when_busy_or_empty_or_not_ready():
    c, tele = client()
    assert c.post("/api/orden", json={"text": "  "}).status_code == 400
    assert c.post("/api/orden", json={"text": "las rojas a la izquierda"}).status_code == 200
    cmd = tele.commands.get_nowait()
    assert cmd.kind == "orden" and cmd.payload["text"] == "las rojas a la izquierda"
    tele.set(status="ejecutando")
    assert c.post("/api/orden", json={"text": "otra"}).status_code == 409
    c2, _ = client(ready=False)
    assert c2.post("/api/orden", json={"text": "algo"}).status_code == 409


def test_controls():
    c, tele = client()
    assert c.post("/api/control", json={"action": "detener"}).status_code == 200 and tele.stop_requested
    assert c.post("/api/control", json={"action": "pausa"}).status_code == 200 and tele.paused
    assert c.post("/api/control", json={"action": "reanudar"}).status_code == 200 and not tele.paused
    assert c.post("/api/control", json={"action": "aleatorizar"}).status_code == 200
    assert tele.commands.get_nowait().kind == "aleatorizar"
    assert c.post("/api/control", json={"action": "bailar"}).status_code == 400


def test_reset_while_running_requests_a_stop_first():
    c, tele = client()
    tele.set(status="ejecutando")
    assert c.post("/api/control", json={"action": "reiniciar"}).status_code == 200
    assert tele.stop_requested and tele.commands.get_nowait().kind == "reiniciar"


def test_voice_unavailable_is_reported_not_crashed():
    c, _ = client(voice=TextOnly("sin micrófono"))
    r = c.post("/api/voz")
    assert r.status_code == 503 and "micrófono" in r.json()["detail"]
    c, _ = client(voice=None)
    assert c.post("/api/voz").status_code == 503


def test_voice_transcription_becomes_an_order():
    class FakeVoice:
        busy = False

        def problem(self):
            return None

        def listen(self):
            return "pon las rojas a la izquierda"

    c, tele = client(voice=FakeVoice())
    assert c.post("/api/voz").status_code == 200
    for _ in range(50):
        if not tele.commands.empty():
            break
        time.sleep(0.05)
    cmd = tele.commands.get(timeout=1)
    assert cmd.kind == "orden" and cmd.payload["source"] == "voz" and "rojas" in cmd.payload["text"]
    assert tele.snapshot()["heard"] == "pon las rojas a la izquierda"


def test_frames_served_after_they_are_pushed():
    c, tele = client()
    assert c.get("/api/frame/cam.jpg").status_code == 404
    assert c.get("/api/frame/otra.jpg").status_code == 404
    tele.push_named("cam", b"\xff\xd8fake")
    r = c.get("/api/frame/cam.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert tele.snapshot()["frames"]["cam"] == 1


def test_websocket_sends_state_then_tagged_frames():
    c, tele = client()
    tele.push_named("seg", b"\xff\xd8seg")
    with c.websocket_connect("/ws") as ws:
        seen_state, seen_frame = False, False
        for _ in range(6):
            msg = ws.receive()
            if "text" in msg and msg["text"]:
                seen_state = True
            if msg.get("bytes"):
                assert msg["bytes"][0] == 2 and msg["bytes"][1:3] == b"\xff\xd8"
                seen_frame = True
            if seen_state and seen_frame:
                break
        assert seen_state and seen_frame


def test_telemetry_boxes_and_clear():
    tele = BinTelemetry()
    rgb, depth = render(SCENE, CTX, [piece(SCENE, "a", "red", 0.2, 0.0)])
    inst = S.HsvSegmenter(CTX).segment(rgb, depth)
    tele.set_boxes(inst)
    assert tele.snapshot()["boxes"]["origen"] == 1
    tele.event("ver", "hola")
    tele.set(order={"x": 1}, heard="hola")
    tele.clear_run()
    s = tele.snapshot()
    assert s["events"] == [] and s["order"] is None and s["heard"] is None


# ---------------------------------------------------------------- visualización
def test_rotated_view_puts_the_robots_left_box_on_the_left_of_the_screen():
    rgb, depth = render(SCENE, CTX, [piece(SCENE, "a", "red", 0.2, 0.0)])
    img = viz.annotate(rgb, [], SCENE, CTX.cam, rotate=True)
    h, w = rgb.shape[:2]
    assert img.shape[:2] == (w, h)                                  # girada 90°
    left = C.container(SCENE, C.DEST_IDS["izquierda"]).center_xy_m
    right = C.container(SCENE, C.DEST_IDS["derecha"]).center_xy_m
    ul, _ = CTX.cam.project(np.array([[left[0], left[1], SCENE.table.top_z_m]]))
    ur, _ = CTX.cam.project(np.array([[right[0], right[1], SCENE.table.top_z_m]]))
    xl, _ = viz.rot_pt(ul[0, 0], ul[0, 1], w)
    xr, _ = viz.rot_pt(ur[0, 0], ur[0, 1], w)
    assert xl < img.shape[1] / 2 < xr


def test_annotate_modes_and_jpeg():
    ps = [piece(SCENE, "a", "red", 0.19, -0.02), piece(SCENE, "b", "blue", 0.22, 0.025)]
    rgb, depth = render(SCENE, CTX, ps)
    inst = S.HsvSegmenter(CTX).segment(rgb, depth)
    for i, k in enumerate(inst):
        k.track = f"P{i + 1}"
    raw = viz.annotate(rgb, inst, SCENE, CTX.cam, mode="rgb", target_track="P1")
    seg = viz.annotate(rgb, inst, SCENE, CTX.cam, mode="seg", target_track="P1")
    assert raw.shape == seg.shape and not np.array_equal(raw, seg)
    assert seg.mean() < raw.mean() + 40                              # la segmentación oscurece el fondo
    jpg = viz.to_jpeg(seg)
    assert jpg[:2] == b"\xff\xd8" and len(jpg) > 1000
    assert viz.plain_jpeg(rgb, rotate=True)[:2] == b"\xff\xd8"


def test_static_page_exists():
    assert (STATIC / "index.html").is_file()
