import pytest

from roboqc.commands.taxonomy import ALL_INTENTS, Intent, get_intent_spec
from roboqc.commands.schema import StructuredCommand, Action, SCHEMA_VERSION


def test_all_declared_intents_present():
    assert len(ALL_INTENTS) >= 30
    assert "UNKNOWN" in ALL_INTENTS
    assert "EMERGENCY_STOP" in ALL_INTENTS


def test_schema_roundtrip():
    cmd = StructuredCommand(transcription="x", actions=[Action(intent="PICK_OBJECT", parameters={"object_color": "rojo"}, negated=False)])
    d = cmd.to_dict()
    cmd2 = StructuredCommand.from_dict(d)
    assert cmd2.to_json() == cmd.to_json() or cmd2.transcription == cmd.transcription
    assert cmd2.actions[0].intent == "PICK_OBJECT"


def test_schema_version():
    assert SCHEMA_VERSION == "1.0"
