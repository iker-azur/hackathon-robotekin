"""Demo 12 -- the client's types against the server's answers.

**Bug 3 of `docs/PHYSICS.md` §11f was a wire-format mismatch that survived review,
documentation and a TypeScript typecheck.** `MoveTask.to_dict()` emitted `part_id` where
`drain_commands`, the ledger, `SceneConfig.from_dict` and `web/src/api.ts` all read
`partId`. The client was *typechecked against a fiction*: it declared fields no endpoint
produced, and nothing anywhere compared the two.

So this file compares them, mechanically. It parses `web/src/api.ts` for the field names it
declares, asks the server for the corresponding endpoint, and asserts every declared field is
actually present in the answer. A rename on either side now fails a test in under two
seconds instead of surfacing as a dead button in front of a judge.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from demo12.tests.test_server import _make, _publish

_API_TS = (Path(__file__).resolve().parents[1] / "web" / "src" / "api.ts")


def _declared_fields(type_name: str) -> set[str]:
    """The field names ``api.ts`` declares for ``type_name``.

    A deliberately small parser: it reads one ``export type X = { ... };`` block and takes
    the identifiers at the start of each line. It does not need to understand TypeScript --
    only to notice when a field name stops existing on one side of the wire, which is
    exactly the failure that is invisible otherwise.

    Both shapes in the file have to be handled. Most types are written across several lines
    and end at the first ``};`` on its own line, but a few are written on one, and matching
    only the multi-line form made the parser read ``LivePart`` as the whole file from its
    declaration to the last closing brace.
    """
    source = _API_TS.read_text(encoding="utf-8")
    start = re.search(rf"export type {re.escape(type_name)} = \{{", source)
    if start is None:
        raise AssertionError(f"{_API_TS.name} declares no type {type_name!r}")

    rest = source[start.end():]
    # The block ends at the first ``}``, whether or not it closes the line.
    end = rest.find("}")
    if end == -1:
        raise AssertionError(f"{type_name} is never closed in {_API_TS.name}")
    body = rest[:end]

    fields = set()
    for line in body.splitlines():
        line = line.strip().rstrip(";").strip()
        if not line or line.startswith("//") or line.startswith("*"):
            continue
        name = re.match(r"([A-Za-z_$][\w$]*)\??\s*:", line)
        if name:
            fields.add(name.group(1))
    return fields


@pytest.fixture
def live():
    client, hub = _make()
    _publish(client, hub)
    return client


# Each entry: the client type, and a body from the endpoint that is supposed to satisfy it.
# The endpoint is named so that a failure says which surface drifted, not just "a field".
ENDPOINTS = [
    ("Part", lambda c: c.get("/api/scene").json()["parts"][0], "/api/scene parts"),
    ("Bin", lambda c: c.get("/api/scene").json()["bins"][0], "/api/scene bins"),
    ("SceneRobot", lambda c: c.get("/api/scene").json()["robots"][0], "/api/scene robots"),
    ("LivePart", lambda c: c.get("/api/state").json()["parts"][0], "/api/state parts"),
    ("EventRecord", lambda c: c.get("/api/events", params={"since": 0}).json()["events"][0],
     "/api/events"),
    ("Metrics", lambda c: c.get("/api/metrics").json(), "/api/metrics"),
]


@pytest.mark.parametrize("type_name,get_body,where",
                         ENDPOINTS, ids=[e[2] for e in ENDPOINTS])
def test_every_field_the_client_declares_is_actually_sent(live, type_name, get_body, where):
    """The check that would have caught the `partId` defect before a judge did."""
    body = get_body(live)
    assert isinstance(body, dict), f"{where} did not return an object"

    declared = _declared_fields(type_name)
    assert declared, f"parsed no fields out of {type_name} in api.ts -- parser is wrong"

    missing = sorted(declared - set(body))
    assert not missing, (
        f"api.ts declares {type_name}.{missing} but {where} does not send it. "
        f"The client will read `undefined` and draw nothing, with no type error, because "
        f"it was compiling against a declaration rather than against the server.")


def test_the_client_and_server_agree_on_task_field_names(live):
    """The exact field names that were wrong, pinned so they cannot drift apart again."""
    source = _API_TS.read_text(encoding="utf-8")
    inline = re.search(r"tasks: \{([^}]*)\}", source)
    assert inline is not None, "api.ts no longer declares an inline task shape"

    declared = set(re.findall(r"(\w+)\??:", inline.group(1)))
    served = live.get("/api/scene").json()["tasks"][0]

    assert {"partId", "binId"} <= declared, f"api.ts task shape is now {sorted(declared)}"
    assert {"partId", "binId"} <= set(served), (
        "the server sends snake_case task ids again: "
        f"{sorted(set(served))}. drain_commands reads t['partId'], so every task submitted "
        "from the interface will raise KeyError and be swallowed into lastError.")
    assert not ({"part_id", "bin_id"} & set(served))


def test_a_task_round_trips_through_the_queue_the_way_the_client_uses_it(live):
    """Submit, then read the queue back exactly as ``drain_commands`` does.

    This is the seam that broke. A dict that the endpoint produces and the consumer reads
    are only the same dict if someone actually passes one to the other.
    """
    live.post("/api/tasks/submit", json={"moves": [{"partId": "part_3",
                                                    "binId": "bin_2"}]})
    name, payload = live.app.state.hub.commands.get_nowait()
    assert name == "set_tasks"

    # ``drain_commands`` does exactly this comprehension, with this key spelling.
    rebuilt = [{"id": str(t["id"]), "partId": str(t["partId"]),
                "binId": str(t["binId"])} for t in payload]
    assert rebuilt[0]["partId"] == "part_3"