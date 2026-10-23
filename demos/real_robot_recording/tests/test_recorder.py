import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import record as CLI
import recorder as R


class Follower:
    def __init__(self, cfg):
        self.cfg = cfg
        self.events = []
        self.error = None

    def get_observation(self):
        self.events.append("observe")
        if self.error:
            raise self.error
        return {**{key: float(i) for i, key in enumerate(R.POSITION_KEYS)},
                "wrist": np.full((self.cfg.height, self.cfg.width, 3), (240, 15, 30), np.uint8),
                "overhead": np.full((self.cfg.height, self.cfg.width, 3), (5, 220, 50), np.uint8)}

    def send_action(self, command):
        self.events.append("send")
        # Simulate a limit: the action actually sent is different from the leader.
        return {key: value - 2 for key, value in command.items()}


class Leader:
    def __init__(self, follower):
        self.follower = follower
        self.calls = 0

    def get_action(self):
        self.follower.events.append("leader")
        self.calls += 1
        # Reverse insertion order to catch accidental reliance on dictionary order.
        return {key: float(i + 10) for i, key in reversed(list(enumerate(R.POSITION_KEYS)))}


def test_sample_records_sent_action_and_one_leader_read(cfg):
    follower = Follower(cfg)
    leader = Leader(follower)
    frame = R.sample(cfg, follower, leader, 0.25)
    assert follower.events == ["observe", "leader", "send"]
    assert leader.calls == 1
    np.testing.assert_array_equal(frame["observation.state"], np.arange(6))
    np.testing.assert_array_equal(frame["leader_state"], np.arange(6) + 10)
    np.testing.assert_array_equal(frame["action"], np.arange(6) + 8)
    assert frame["capture_time_s"].dtype == np.float64
    assert frame["capture_time_s"].tolist() == [0.25]


@pytest.mark.parametrize("error", [TimeoutError("camera"), OSError("serial")])
def test_bad_observation_never_sends_or_records(cfg, error):
    follower = Follower(cfg)
    follower.error = error
    with pytest.raises(type(error)):
        R.sample(cfg, follower, Leader(follower), 0)
    assert follower.events == ["observe"]


def test_bad_image_and_nonfinite_joint_fail_before_moving(cfg):
    follower = Follower(cfg)
    obs = follower.get_observation()
    obs["wrist"] = np.zeros((1, 1, 3), np.uint8)
    follower.get_observation = lambda: obs
    with pytest.raises(ValueError, match="wrist"):
        R.sample(cfg, follower, Leader(follower), 0)
    obs["wrist"] = np.zeros((cfg.height, cfg.width, 3), np.uint8)
    obs[R.POSITION_KEYS[0]] = float("nan")
    with pytest.raises(ValueError, match="finita"):
        R.sample(cfg, follower, Leader(follower), 0)
    assert "send" not in follower.events


@pytest.mark.parametrize("changes", [{"fps": 0}, {"width": 63}, {"episode_time_s": float("nan")},
                                     {"max_relative_target": -1}, {"display_data": "false"}, {"task": " "}])
def test_invalid_config(cfg, changes):
    with pytest.raises(ValueError):
        replace(cfg, **changes)


def test_config_paths_resolve_relative_to_config(cfg, tmp_path):
    from dataclasses import asdict
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**asdict(cfg), "root": "data", "leader_calibration_dir": "cal/leader"}))
    actual = R.Config.load(path)
    assert actual.root == str(tmp_path / "data")
    assert actual.leader_calibration_dir == str(tmp_path / "cal/leader")


def test_same_devices_are_rejected_before_connection(cfg):
    with pytest.raises(ValueError, match="puertos"):
        CLI.device_checks(replace(cfg, follower_port="same", leader_port="same"))
    with pytest.raises(ValueError, match="cámaras"):
        CLI.device_checks(replace(cfg, follower_port="one", leader_port="two", overhead_camera=0))


def test_os_lock_prevents_second_writer_and_releases(cfg):
    with R.dataset_lock(cfg), pytest.raises(RuntimeError, match="Otro proceso"), R.dataset_lock(cfg):
        pass
    with R.dataset_lock(cfg):
        pass


class FakeDataset:
    def __init__(self):
        self.frames = []
        self.num_episodes = 0
        self.closed = False
        self.events = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def has_pending_frames(self):
        return bool(self.frames)

    def save_episode(self, parallel_encoding=False):
        self.events.append("save")
        self.num_episodes += 1

    def finalize(self):
        self.events.append("finalize")
        self.closed = True

    def clear_episode_buffer(self):
        self.events.append("discard")
        self.frames.clear()


@pytest.fixture
def session(cfg, monkeypatch):
    dataset = FakeDataset()
    monkeypatch.setattr(R, "writer", lambda *a, **kw: dataset)
    return R.Session(cfg)


def test_accept_closes_before_upload_and_retries_without_rewriting(session):
    session.begin()
    dataset = session.dataset
    session.add({"sample": 1})
    with pytest.raises(RuntimeError, match="abierto"):
        session.sync(push=Mock())
    session.accept({"frames": 1})
    calls = []

    def push():
        assert dataset.closed
        calls.append(True)
        if len(calls) < 3:
            raise ConnectionError("offline")

    sleep = Mock()
    assert session.sync(push=push, sleep=sleep)
    assert len(calls) == 3 and sleep.call_count == 2
    assert dataset.events == ["save", "finalize"]
    assert session.count == session.state["uploaded_episodes"] == 1


def test_offline_keeps_pending_and_manual_retry_does_not_duplicate(session):
    session.begin()
    session.add({"sample": 1})
    session.accept({"frames": 1})
    assert not session.sync(push=Mock(side_effect=ConnectionError), sleep=Mock())
    assert session.state["uploaded_episodes"] == 0 and session.count == 1
    assert session.sync(push=Mock(), sleep=Mock())
    assert session.count == 1 and session.state["uploaded_episodes"] == 1


def test_discard_drains_images_and_never_uploads(session):
    session.begin()
    dataset = session.dataset
    session.add({"sample": 1})
    session.discard()
    push = Mock()
    assert session.sync(push=push)
    push.assert_not_called()
    assert dataset.events == ["finalize", "discard"]
    assert session.count == 0 and session.state["phase"] == "idle"


def test_empty_cannot_be_accepted(session):
    session.begin()
    with pytest.raises(ValueError, match="fotogramas"):
        session.accept({})
    session.discard()


def test_interrupted_save_is_not_silently_resumed(session):
    session.begin()
    session.add({"sample": 1})
    session.dataset.save_episode = Mock(side_effect=OSError("disk full"))
    with pytest.raises(OSError):
        session.accept({})
    assert session.state["phase"] == "saving"
    with pytest.raises(RuntimeError, match="interrumpida"):
        R.Session(session.cfg, resume=True)


def test_resume_rejects_different_contract(session):
    with pytest.raises(ValueError, match="incompatible"):
        R.Session(replace(session.cfg, fps=15), resume=True)


def test_pause_only_session_can_be_resumed_without_a_dataset(session):
    resumed = R.Session(session.cfg, resume=True)
    assert resumed.count == 0


def test_failed_capture_discards_on_record_cleanup(cfg, monkeypatch):
    # Exercise the real record() finally path, including sensor exceptions.
    dataset = FakeDataset()
    follower = Follower(cfg)
    leader = Leader(follower)
    original = follower.get_observation
    calls = 0

    def observation():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise TimeoutError("camera unplugged")
        return original()

    follower.get_observation = observation

    import contextlib
    @contextlib.contextmanager
    def hardware(_):
        yield follower, leader, {}

    class Keys:
        def __enter__(self):
            self.keys = iter(["s", "", ""])
            return self

        def __exit__(self, *_):
            pass

        def wait(self, _):
            return "p"

        def poll(self):
            return next(self.keys)

    monkeypatch.setattr(R, "writer", lambda *a, **kw: dataset)
    monkeypatch.setattr(CLI, "hardware", hardware)
    monkeypatch.setattr(CLI, "Keys", Keys)
    with pytest.raises(TimeoutError):
        CLI.record(cfg, False)
    assert dataset.closed and dataset.events == ["finalize", "discard"]
    assert json.loads((cfg.local / "state.json").read_text())["phase"] == "idle"
