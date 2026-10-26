"""Real LeRobot/MP4/ACT round trip, no hardware, Hub token or weight downloads."""

from dataclasses import replace

import numpy as np
import pytest
import recorder as R
from test_recorder import Follower, Leader


def add_episode(session, length=6):
    follower = Follower(session.cfg)
    leader = Leader(follower)
    session.begin()
    for i in range(length):
        frame = R.sample(session.cfg, follower, leader, i / session.cfg.fps)
        frame["observation.state"] += i
        frame["action"] += i
        session.add(frame)
    session.accept({"frames": length})


@pytest.mark.parametrize("streaming", [False, True])
def test_videos_discard_resume_and_rgb_roundtrip(cfg, streaming):
    R.require_version()
    cfg = replace(cfg, streaming_encoding=streaming)
    with R.dataset_lock(cfg):
        session = R.Session(cfg)
        # Reject the very first recording, then accept without recreating a dataset.
        session.begin()
        follower = Follower(cfg)
        session.add(R.sample(cfg, follower, Leader(follower), 0))
        session.discard()
        add_episode(session)
        add_episode(session)
        assert R.validate(cfg) == {"episodes": 2, "frames": 12, "fps": cfg.fps}
    with R.dataset_lock(cfg):
        session = R.Session(cfg, resume=True)
        add_episode(session)
        assert R.validate(cfg)["episodes"] == 3
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, video_backend="pyav")
        item = dataset[0]
        wrist = item[R.IMAGE_KEYS[0]].numpy().mean(axis=(1, 2))
        overhead = item[R.IMAGE_KEYS[1]].numpy().mean(axis=(1, 2))
        assert wrist[0] > wrist[1] + 0.5 and overhead[1] > overhead[0] + 0.5
        np.testing.assert_allclose(item["leader_state"], np.arange(6) + 10)
        assert session.state["uploaded_episodes"] == 0


def test_zero_accepted_episodes_resume_after_exit(cfg):
    session = R.Session(cfg)
    session.begin()
    follower = Follower(cfg)
    session.add(R.sample(cfg, follower, Leader(follower), 0))
    session.discard()
    session = R.Session(cfg, resume=True)
    add_episode(session)
    assert R.validate(cfg)["frames"] == 6


def test_act_optimizer_step_uses_two_views_and_follower_only(cfg):
    import torch
    from lerobot.configs import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    torch.set_num_threads(2)
    session = R.Session(cfg)
    add_episode(session)
    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, video_backend="pyav",
                             delta_timestamps={"action": [i / cfg.fps for i in range(4)]})
    policy_features = dataset_to_policy_features(dataset.features)
    outputs = {k: v for k, v in policy_features.items() if v.type == FeatureType.ACTION}
    inputs = {k: v for k, v in policy_features.items() if k not in outputs}
    assert set(inputs) == {*R.IMAGE_KEYS, "observation.state"}
    assert set(outputs) == {"action"} and outputs["action"].shape == (6,)
    config = ACTConfig(input_features=inputs, output_features=outputs, device="cpu", push_to_hub=False,
                       chunk_size=4, n_action_steps=4, dim_model=32, n_heads=4, dim_feedforward=64,
                       n_encoder_layers=1, n_decoder_layers=1, use_vae=False,
                       pretrained_backbone_weights=None)
    pre, _ = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    batch = pre(batch)
    policy = ACTPolicy(config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    before = next(policy.model.parameters()).detach().clone()
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    assert not torch.equal(before, next(policy.model.parameters()).detach())


def test_missing_video_is_reported_without_downloading_or_repairing(cfg, monkeypatch):
    from pathlib import Path
    from unittest.mock import Mock

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    session = R.Session(cfg)
    add_episode(session)
    video = next((Path(cfg.root) / "videos").rglob("*.mp4"))
    parked = video.with_suffix(".parked")
    video.rename(parked)
    download = Mock(side_effect=AssertionError("Validation must stay local"))
    monkeypatch.setattr(LeRobotDataset, "_download", download)
    with pytest.raises(FileNotFoundError):
        R.validate(cfg)
    download.assert_not_called()
    assert parked.exists() and not video.exists()


def test_default_upload_boundary_is_public_closed_and_excludes_local_journal(cfg, monkeypatch):
    from unittest.mock import Mock

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    session = R.Session(cfg)
    add_episode(session)
    push = Mock()
    monkeypatch.setattr(LeRobotDataset, "push_to_hub", push)
    assert session.sync()
    assert session.dataset is None
    kwargs = push.call_args.kwargs
    assert kwargs["private"] is False and kwargs["license"] is None
    assert kwargs["allow_patterns"] == ["data/**", "videos/**", "meta/**", "recording.json", "README.md"]
    assert session.state["uploaded_episodes"] == 1


def test_hardware_adapter_uses_existing_calibrations_without_connecting(cfg, monkeypatch, tmp_path):
    import record as CLI

    monkeypatch.setattr(CLI, "device_checks", lambda _: None)
    cfg = replace(cfg, follower_calibration_dir=str(tmp_path / "follower"),
                  leader_calibration_dir=str(tmp_path / "leader"))
    with pytest.raises(RuntimeError, match="Falta calibración"), CLI.hardware(cfg):
        pytest.fail("Should not connect uncalibrated hardware")


def test_rerun_preview_accepts_real_frame_without_opening_gui(cfg):
    import record as CLI
    import rerun as rr

    rr.init("recording_api_test", spawn=False)
    memory = rr.memory_recording()
    preview = CLI.Preview(False)
    preview.rr = rr
    follower = Follower(cfg)
    preview.show(R.sample(cfg, follower, Leader(follower), 0.25), 0)
    assert memory.num_msgs() > 0
