"""Shared message shapes for Demo 12."""

from __future__ import annotations

from typing import Any, TypedDict


class StateMessage(TypedDict, total=False):
    simTimeS: float
    robots: list[dict[str, Any]]
    parts: list[dict[str, Any]]
    bins: list[dict[str, Any]]
