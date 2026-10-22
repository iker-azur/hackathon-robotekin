"""Panel web sin Isaac: rutas, órdenes en la cola y validación de la configuración."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recoverykit import config  # noqa: E402
from recoverykit.perturb import LABELS  # noqa: E402
from recoverykit.telemetry import Telemetry  # noqa: E402
from server.app import build_app  # noqa: E402


def client():
    tele = Telemetry()
    cfg = config.preset("demo")
    return tele, TestClient(build_app(tele, lambda: cfg, LABELS))


def drain(tele):
    out = []
    while not tele.commands.empty():
        out.append(tele.commands.get_nowait())
    return out


def test_page_and_static():
    _, c = client()
    r = c.get("/")
    assert r.status_code == 200 and "RecoveryKit" in r.text and 'lang="es"' in r.text
    assert c.get("/static/vendor/three.module.min.js").status_code == 200
    assert c.get("/static/robot_meshes.json").json()["gripper"]["f"]


def test_kit_command_is_queued_and_rejected_while_running():
    tele, c = client()
    assert c.post("/api/kit", json={"text": "una roja"}).status_code == 200
    assert [(x.kind, x.payload["text"]) for x in drain(tele)] == [("kit", "una roja")]
    assert c.post("/api/kit", json={"text": "  "}).status_code == 400
    tele.set(status="ejecutando")
    assert c.post("/api/kit", json={"text": "una roja"}).status_code == 409


def test_controls():
    tele, c = client()
    c.post("/api/control", json={"action": "pause"})
    assert tele.paused
    c.post("/api/control", json={"action": "resume"})
    assert not tele.paused
    c.post("/api/control", json={"action": "stop"})
    assert tele.stop_requested
    c.post("/api/control", json={"action": "randomize"})
    assert [x.kind for x in drain(tele)] == ["randomize"]
    assert c.post("/api/control", json={"action": "bailar"}).status_code == 400


def test_perturbations():
    tele, c = client()
    r = c.post("/api/perturb", json={"kind": "empujar"})
    assert r.status_code == 200 and r.json()["label"] == LABELS["empujar"]
    assert drain(tele)[0].payload == {"kind": "empujar"}
    assert c.post("/api/perturb", json={"kind": "teletransportar"}).status_code == 400


def test_config_validation():
    tele, c = client()
    cfg = c.get("/api/config").json()["config"]
    cfg["pieces"][0]["size_m"] = 0.06
    r = c.post("/api/config", json={"config": cfg})
    assert r.status_code == 400 and "apertura" in r.json()["detail"]
    assert c.post("/api/config", json={"preset": "simple"}).status_code == 200
    assert drain(tele)[0].kind == "config"


def test_state_and_websocket():
    tele, c = client()
    tele.event("percepción", "Veo un cubo rojo")
    tele.push_frame(b"\xff\xd8jpeg")
    assert c.get("/api/state").json()["events"][0]["text"] == "Veo un cubo rojo"
    assert c.get("/api/frame.jpg").content == b"\xff\xd8jpeg"
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "state"
        assert ws.receive_bytes() == b"\xff\xd8jpeg"
