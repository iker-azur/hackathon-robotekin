"""High-level command understanding pipeline used by scripts and GUI."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .parser import parse_text
from .schema import StructuredCommand


def understand_text(
    text: str,
    source: str = "text",
    classifier: Optional[Callable[[str], str]] = None,
    confidence_fn: Optional[Callable[[str], float]] = None,
) -> StructuredCommand:
    cmd = parse_text(text, source=source, classifier=classifier, confidence_fn=confidence_fn)
    return cmd


def understand_transcription(
    transcription: str,
    classifier=None,
    confidence_fn=None,
) -> StructuredCommand:
    return understand_text(transcription, source="voice", classifier=classifier, confidence_fn=confidence_fn)
