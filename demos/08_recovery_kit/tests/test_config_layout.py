"""Configuración y colocación de piezas, sin Isaac."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recoverykit import config, layout  # noqa: E402


def test_presets_are_valid():
    for name in ("demo", "simple", "formas"):
        cfg = config.preset(name)
        assert config.validate(cfg) == [], (name, config.validate(cfg))


def test_round_trip(tmp_path):
    cfg = config.preset("demo")
    path = tmp_path / "scene.json"
    cfg.save(path)
    again = config.load(path)
    # JSON convierte tuplas en listas: se compara la forma serializada
    assert json.dumps(again.to_dict()) == json.dumps(cfg.to_dict())
    assert [p.id for p in again.pieces] == [p.id for p in cfg.pieces]


def test_validate_rejects_bad_values():
    cfg = config.preset("simple")
    cfg.pieces[0].color = "pink"
    cfg.pieces[0].size_m = 0.05
    cfg.physics.dt = 1 / 60
    errors = config.validate(cfg)
    assert any("pink" in e for e in errors)
    assert any("apertura" in e for e in errors)
    assert any("dt" in e for e in errors)


def test_layout_is_reproducible_and_separated():
    cfg = config.preset("demo")
    a = layout.sample(cfg, seed=5)
    b = layout.sample(cfg, seed=5)
    c = layout.sample(cfg, seed=6)
    assert a == b
    assert a != c
    pts = list(a.values())
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= cfg.work_area.min_gap_m - 1e-9
    for x, y, _ in pts:
        r = math.hypot(x, y)
        a = math.degrees(math.atan2(y, x))
        assert cfg.work_area.r_m[0] - 1e-9 <= r <= cfg.work_area.r_m[1] + 1e-9
        assert cfg.work_area.angle_deg[0] - 1e-9 <= a <= cfg.work_area.angle_deg[1] + 1e-9
        assert not layout._in_box_zone(cfg, x, y)


def test_layout_many_seeds_fit():
    cfg = config.preset("demo")
    for seed in range(200):
        assert len(layout.sample(cfg, seed=seed)) == len(cfg.pieces)


def test_layout_keeps_fixed_positions():
    cfg = config.preset("simple")
    placed = layout.sample(cfg)
    assert placed["rojo_1"][:2] == (0.17, -0.07)
