"""Demo 12 -- scene configuration round-trips and refuses what it cannot build."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from demo12.config import SceneConfig, SceneError, default_scene


def test_default_scenes_validate():
    """Two and three arms build a cell that passes validation."""
    for n in (2, 3):
        scene = default_scene(n)
        scene.validate()
        assert len(scene.robots) >= n
        assert len(scene.enabled_robots) == n


def test_a_scene_round_trips_through_json():
    """The editor sends a scene as JSON and the server rebuilds it from that JSON."""
    original = default_scene(3)
    rebuilt = SceneConfig.from_dict(original.to_dict())
    assert rebuilt.to_dict()["parts"] == original.to_dict()["parts"]
    assert rebuilt.to_dict()["bins"] == original.to_dict()["bins"]
    rebuilt.validate()


def test_a_bad_scene_is_rejected_with_a_message_naming_the_field():
    data = default_scene(2).to_dict()
    data["parts"][0]["sizeM"] = [0.040, 0.040, 0.040]      # wider than the gripper
    with pytest.raises(SceneError) as excinfo:
        SceneConfig.from_dict(data)
    assert "gripper" in str(excinfo.value)


def test_every_mission_task_refers_to_a_real_part_and_bin():
    """Tasks are data: every one names a part and a tray that exist.

    Not every part has a task -- the default mission is colour sorting, and the parts whose
    colour no tray carries wait to be assigned from the interface. That is a choice, and it
    is the test below that pins it.
    """
    scene = default_scene(2)
    part_ids = {p.id for p in scene.parts}
    bin_ids = {b.id for b in scene.bins}
    assert scene.mission.tasks
    for task in scene.mission.tasks:
        assert task.part_id in part_ids
        assert task.bin_id in bin_ids


def test_the_default_mission_is_the_two_colour_matched_picks(scene=None):
    """Two picks on opposite sides of the cell, which is what lets them run in parallel."""
    scene = default_scene(2)
    bins = {b.id: b for b in scene.bins}
    assert len(scene.mission.tasks) == 2
    for task in scene.mission.tasks:
        part = scene.part(task.part_id)
        assert bins[task.bin_id].color.lower() == part.color.lower()
    xs = sorted(scene.part(t.part_id).x for t in scene.mission.tasks)
    assert xs[1] - xs[0] > 0.10