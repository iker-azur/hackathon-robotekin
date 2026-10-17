import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from demo12.config import default_scene

def test_default_scenes():
    for n in (1, 2, 3):
        s = default_scene(n)
        s.validate()
        assert len(s.robots) >= n