"""Tests for the pure logic around datasets, spots and reports. No GPU, no Isaac.

    python -m pytest demos/06_imitation_learning/tests -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import expert as expert_module  # noqa: E402
import il_dataset  # noqa: E402
import il_env  # noqa: E402


def load_evaluator():
    """``16_evaluate_policy.py`` has a digit-first name, so it is loaded by path."""
    spec = importlib.util.spec_from_file_location("evaluate_policy", ROOT / "16_evaluate_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- plan clock and detector scaling ------------------------------------------------


def test_plan_clock_spans_minus_one_to_one_and_clamps():
    last = il_dataset.plan_steps() - 1
    assert il_dataset.plan_steps() == 640
    assert il_dataset.plan_clock(0) == -1.0
    assert il_dataset.plan_clock(last) == 1.0
    assert il_dataset.plan_clock(last + 500) == 1.0  # a rollout may outlast the plan
    values = [il_dataset.plan_clock(step) for step in range(last + 1)]
    assert values == sorted(values)


def test_normalised_uv_maps_the_frame_to_minus_one_one():
    class Detection:
        u, v = 319.5, 239.5

    assert il_dataset.normalised_uv(Detection, (480, 640, 3)) == pytest.approx((0.0, 0.0))
    Detection.u, Detection.v = 0.0, 479.0
    assert il_dataset.normalised_uv(Detection, (480, 640, 3)) == pytest.approx((-1.0, 1.0))


# -- what a checkpoint's environment state holds ------------------------------------


def test_environment_kind_comes_from_the_sidecar(tmp_path):
    checkpoint = tmp_path / "outputs" / "run" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (tmp_path / "outputs" / "run" / il_dataset.INPUTS_FILE).write_text(
        json.dumps({"env_state": "clock_cube"}), encoding="utf-8"
    )
    assert il_dataset.resolve_env_state_kind(checkpoint, 3) == "clock_cube"


def test_a_sidecar_that_disagrees_with_the_width_is_an_error(tmp_path):
    (tmp_path / il_dataset.INPUTS_FILE).write_text(json.dumps({"env_state": "phase"}), encoding="utf-8")
    with pytest.raises(ValueError):
        il_dataset.resolve_env_state_kind(tmp_path, 3)


def test_old_checkpoints_are_told_apart_by_width(tmp_path):
    assert il_dataset.resolve_env_state_kind(tmp_path, 1) == "phase"
    assert il_dataset.resolve_env_state_kind(tmp_path, 3) == "detector"
    assert il_dataset.resolve_env_state_kind(tmp_path, 7) is None


def test_perception_features_are_opt_in():
    assert il_dataset.PLAN_CLOCK_KEY not in il_dataset.features()
    extended = il_dataset.features(with_perception=True)
    assert extended[il_dataset.PLAN_CLOCK_KEY]["shape"] == (1,)
    assert extended[il_dataset.CUBE_UV_KEY]["shape"] == (2,)


# -- held-out evaluation spots ------------------------------------------------------


def test_evaluation_spots_are_inside_the_box_and_not_training_spots():
    train = {il_env.sample_cube_spot(i, il_env.TRAIN_SPOT_COUNT) for i in range(il_env.TRAIN_SPOT_COUNT)}
    for episode in range(20):
        spot = il_env.sample_eval_spot(episode)
        assert il_env.CUBE_BOX_X[0] <= spot[0] <= il_env.CUBE_BOX_X[1]
        assert il_env.CUBE_BOX_Y[0] <= spot[1] <= il_env.CUBE_BOX_Y[1]
        assert spot not in train
        assert il_env.nearest_training_spot_m(spot) > 0.0005
        assert abs(il_env.sample_eval_yaw(episode)) <= il_env.CUBE_MAX_YAW


def test_outside_spots_are_all_outside_the_training_box_but_in_reach():
    spots = [il_env.sample_outside_spot(i) for i in range(40)]
    margin = il_env.OUTSIDE_MARGIN_M
    for x, y in spots:
        assert not (il_env.CUBE_BOX_X[0] <= x <= il_env.CUBE_BOX_X[1]
                    and il_env.CUBE_BOX_Y[0] <= y <= il_env.CUBE_BOX_Y[1])
        assert il_env.CUBE_BOX_X[0] - margin <= x <= il_env.CUBE_BOX_X[1] + margin
        assert il_env.CUBE_BOX_Y[0] - margin <= y <= il_env.CUBE_BOX_Y[1] + margin
        # the arm's base is at (0.35, -0.64); the drop is 0.10 further in x. Measured reach 0.33 m.
        assert ((x + 0.10 - 0.35) ** 2 + (y + 0.64) ** 2) ** 0.5 < 0.33
    assert len(set(spots)) == 40
    assert [il_env.sample_outside_spot(i) for i in range(5)] == spots[:5]


def test_evaluation_spots_are_deterministic():
    assert [il_env.sample_eval_spot(i) for i in range(5)] == [il_env.sample_eval_spot(i) for i in range(5)]


def test_demonstrations_cannot_reach_the_evaluation_spots():
    assert il_env.TRAIN_SPOT_COUNT < il_env.EVAL_EPISODE_OFFSET


# -- the randomisation freeze -------------------------------------------------------


class _Named:
    def __init__(self, name):
        self.name = name


class _Term:
    def __init__(self, asset, pose_range):
        self.params = {"asset_cfg": _Named(asset), "pose_range": pose_range}


class _Events:
    """Like LeIsaac's event config: the terms are plain attributes, not dataclass fields."""

    def __init__(self):
        self.reset_all = _Term("robot", None)
        self.reset_all.params = {}
        self.domain_randomize_0 = _Term(
            "cube", {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (0.0, 0.0), "yaw": (-0.5, 0.5)}
        )
        self.domain_randomize_1 = _Term(
            "front",
            {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005),
             "roll": (-0.001, 0.001), "pitch": (-0.001, 0.001), "yaw": (-0.001, 0.001)},
        )


class _Cfg:
    events = _Events()


def test_the_camera_and_the_cube_are_frozen_by_set_cube_randomization():
    # Regression: the first version looped over dataclasses.fields(), matched nothing,
    # and left the camera jittering +-5 mm at every reset (5 mm of perception noise).
    cfg = _Cfg()
    il_env.set_cube_randomization(cfg)
    for term in (cfg.events.domain_randomize_0, cfg.events.domain_randomize_1):
        assert all(low == 0.0 and high == 0.0 for low, high in term.params["pose_range"].values())
    assert set(cfg.events.domain_randomize_1.params["pose_range"]) >= {"x", "y", "z", "roll", "pitch", "yaw"}


def test_nothing_to_freeze_is_an_error_not_a_silent_pass():
    class Empty:
        class events:
            pass

    with pytest.raises(RuntimeError):
        il_env.set_cube_randomization(Empty())


# -- the teacher's grasp point ------------------------------------------------------


def test_grasp_bias_stays_inside_the_measured_window():
    # Measured window around the cube centre: x in [-18, +5] mm, y in [-12, +8] mm.
    x, y, z = (value * 1000 for value in expert_module.GRASP_BIAS_M)
    assert -18 < x < 5 and -12 < y < 8 and z == 0


# -- reports are never overwritten --------------------------------------------------


def minimal_record(name):
    return {
        "run": {"name": name, "utc": "2026-10-19T18:00:00Z", "git": {"commit": "abc", "branch": "b", "dirty": False},
                "policy": "p", "type": "act", "inputs": "i", "spots_description": "s", "seed": 0, "steps": 10},
        "episodes": [{"episode": 0, "spot": [0.3, -0.4], "yaw_deg": 3.0, "nearest_train_cm": 1.0,
                      "verdict": {"success": False, "failure": "never_lifted", "max_lift_cm": 0.1,
                                  "longest_carry_steps": 0, "held_fraction": 0.0, "final_error_cm": 9.9}}],
        "summary": {"episodes": 1, "successes": 0, "rate": 0.0, "failures": {"never_lifted": 1}},
    }


def test_a_second_run_cannot_replace_a_report(tmp_path, monkeypatch):
    evaluator = load_evaluator()
    monkeypatch.setattr(evaluator, "RUNS_DIR", tmp_path)
    evaluator.write_report("benchmark", minimal_record("benchmark"))
    before = (tmp_path / "benchmark.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluator.write_report("benchmark", minimal_record("benchmark"))
    assert (tmp_path / "benchmark.json").read_text(encoding="utf-8") == before


def test_default_report_names_differ_between_policies_and_sizes():
    evaluator = load_evaluator()

    class Args:
        type = "act"
        spots = "heldout"
        episodes = 10

    first = Path("outputs/run_a/checkpoints/020000/pretrained_model")
    second = Path("outputs/run_b/checkpoints/020000/pretrained_model")
    assert evaluator.default_report_name(Args, first).startswith("run_a-020000_heldout_10ep_")
    assert evaluator.default_report_name(Args, first) != evaluator.default_report_name(Args, second)
    Args.episodes = 2
    assert "_2ep_" in evaluator.default_report_name(Args, first)


def test_report_states_what_the_policy_was_given():
    evaluator = load_evaluator()
    text = evaluator.render_markdown(minimal_record("x"))
    assert "Entradas de la política" in text and "never_lifted" in text
    assert "Sin sujeción cinemática" in text
