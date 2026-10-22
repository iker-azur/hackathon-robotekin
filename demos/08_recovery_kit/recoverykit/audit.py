"""Auditor independiente: qué hay de verdad dentro de la caja (verdad del simulador).

Sólo para evaluar y depurar. El agente nunca importa este módulo
(``tests/test_isolation.py`` lo comprueba); su veredicto se compara con el del
agente para medir si el agente ve bien lo que ha hecho.
"""

from __future__ import annotations

from .kit import Goal


def box_contents(cell) -> list[dict]:
    cfg = cell.cfg
    out = []
    for c in cfg.containers:
        for name in cell.pieces:
            t = cell.truth(name)
            p = t["pos"]
            if (abs(p[0] - c.center_xy_m[0]) < c.inner_m[0] / 2 and abs(p[1] - c.center_xy_m[1]) < c.inner_m[1] / 2
                    and p[2] < cfg.table.top_z_m + c.outer_m[2]):
                piece = cfg.piece(name)
                out.append({"piece": name, "color": piece.color, "shape": piece.shape, "box": c.id})
    return out


def check(cell, goal: Goal) -> dict:
    """¿La caja contiene exactamente lo pedido? (una pieza "cualquiera" acepta cualquier forma)."""
    contents = box_contents(cell)
    remaining = list(contents)
    missing = []
    # primero lo que pide forma concreta
    for (color, shape), n in sorted(goal.items(), key=lambda kv: kv[0][1] is None):
        if n == "all":
            n = sum(1 for name in cell.pieces if cell.cfg.piece(name).color == color
                    and (shape is None or cell.cfg.piece(name).shape == shape)
                    and cell.truth(name)["pos"][2] > cell.cfg.table.top_z_m - 0.1)
        for _ in range(int(n)):
            hit = next((c for c in remaining if c["color"] == color and (shape is None or c["shape"] == shape)), None)
            if hit is None:
                missing.append({"color": color, "shape": shape})
            else:
                remaining.remove(hit)
    return {"contents": contents, "missing": missing, "extra": remaining, "satisfied": not missing and not remaining}
