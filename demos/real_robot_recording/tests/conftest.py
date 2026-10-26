import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder import Config


@pytest.fixture
def cfg(tmp_path):
    return Config(repo_id="test/so101", root=str(tmp_path / "dataset"), task="Put cube in tray",
                  follower_port="/dev/follower", leader_port="/dev/leader",
                  wrist_camera=0, overhead_camera=1, width=64, height=48, display_data=False)
