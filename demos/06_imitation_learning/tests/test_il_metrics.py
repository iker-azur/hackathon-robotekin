"""Tests for il_metrics.py. No GPU, no Isaac.

    python -m pytest demos/06_imitation_learning/tests -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import il_metrics as metrics  # noqa: E402

TARGET = (0.45, -0.40)
REST = metrics.REST_Z_M


def judge(trace, held=None):
    """Run a list of (x, y, z) positions through the judge."""
    episode = metrics.EpisodeJudge(target_xy=TARGET)
    for index, position in enumerate(trace):
        episode.update(position, None if held is None else held[index])
    return episode.verdict()


def lift_and_place(lift=0.06, carry_steps=120, hand=0.005):
    """A clean pick: rest, rise, carry to the drop point, lower, rest."""
    start = (0.35, -0.40)
    trace, distance = [], []
    for _ in range(40):
        trace.append((*start, REST)); distance.append(0.2)
    for i in range(30):  # rise
        trace.append((start[0], start[1], REST + lift * (i + 1) / 30)); distance.append(hand)
    for i in range(carry_steps):  # carry sideways
        f = (i + 1) / carry_steps
        trace.append((start[0] + (TARGET[0] - start[0]) * f, start[1], REST + lift)); distance.append(hand)
    for i in range(30):  # lower
        trace.append((TARGET[0], TARGET[1], REST + lift * (1 - (i + 1) / 30))); distance.append(hand)
    for _ in range(60):  # at rest
        trace.append((TARGET[0], TARGET[1], REST)); distance.append(0.2)
    return trace, distance


def test_rest_height_matches_environment():
    source = (ROOT / "il_env.py").read_text(encoding="utf-8")
    declared = float(re.search(r"^CUBE_REST_Z\s*=\s*([0-9.]+)", source, re.M).group(1))
    assert declared == metrics.REST_Z_M


def test_a_clean_pick_succeeds():
    trace, distance = lift_and_place()
    verdict = judge(trace, distance)
    assert verdict.success and verdict.failure == ""
    assert verdict.lifted and verdict.stable_grasp and verdict.placed
    assert verdict.max_lift_m >= metrics.LIFT_MIN_M


def test_a_cube_that_is_only_pushed_never_lifts():
    # Ends exactly on the drop point, on the table - the old metric called this a win.
    trace = [(0.35 + 0.10 * i / 200, -0.40, REST) for i in range(200)]
    trace += [(TARGET[0], TARGET[1], REST)] * 60
    verdict = judge(trace)
    assert not verdict.success
    assert verdict.failure == "never_lifted"


def test_a_flicked_cube_is_not_a_stable_grasp():
    trace, distance = lift_and_place(carry_steps=1, lift=0.06)
    # Airborne for ~60 steps in total but the hand is nowhere near it.
    distance = [0.2 if d == 0.005 else d for d in distance]
    verdict = judge(trace, distance)
    assert verdict.lifted and not verdict.stable_grasp
    assert not verdict.success
    assert verdict.failure in {"not_held", "dropped_early"}


def test_a_cube_dropped_after_a_moment_is_dropped_early():
    start = (0.35, -0.40)
    trace = [(*start, REST)] * 20
    trace += [(*start, REST + 0.05)] * 20  # up for 20 steps, well under a second
    trace += [(TARGET[0], TARGET[1], REST)] * 80
    verdict = judge(trace)
    assert verdict.lifted and verdict.failure == "dropped_early"


def test_carried_but_put_in_the_wrong_place_is_misplaced():
    trace, distance = lift_and_place()
    trace = [(x - 0.08, y, z) if i > 100 else (x, y, z) for i, (x, y, z) in enumerate(trace)]
    verdict = judge(trace, distance)
    assert verdict.stable_grasp and not verdict.placed
    assert verdict.failure == "misplaced"


def test_a_cube_still_in_the_air_at_the_end_is_not_placed():
    trace, distance = lift_and_place()
    trace = trace[:-90]  # stop while it is being lowered / carried
    verdict = judge(trace, distance[: len(trace)])
    assert not verdict.success


def test_distance_to_hand_is_optional_and_reported():
    trace, _ = lift_and_place()
    episode = metrics.EpisodeJudge(target_xy=TARGET)
    for position in trace:
        episode.update(position)
    assert not episode.grasp_checked
    assert episode.verdict().success  # falls back to height and duration alone


def test_verdict_serialises_for_the_report():
    trace, distance = lift_and_place()
    record = judge(trace, distance).as_dict()
    assert set(record) >= {"success", "lifted", "stable_grasp", "placed", "failure",
                           "max_lift_cm", "final_error_cm", "steps"}
