import os
import tempfile
import json

from roboqc.dataset.validation import validate_example, check_split_leakage


def test_validate_example_valid():
    ex = {"text": "Coge la pieza roja", "actions": [{"intent": "PICK_OBJECT", "parameters": {}, "negated": False}]}
    assert validate_example(ex) == []


def test_validate_example_invalid():
    ex = {"text": "", "actions": []}
    assert validate_example(ex)


def test_split_leakage_detects_overlap():
    rec = {"text": "hola", "actions": [{"intent": "UNKNOWN", "parameters": {}, "negated": False}]}
    splits = {"train": [rec], "test": [rec]}
    res = check_split_leakage(splits)
    assert res["test"]["overlap_with_train"] == 1
