"""Phase 2 extends the Phase 1 roboqc package.

Sub-packages are looked up here first (vision, ui/camera_worker, ui/workstation) and
then in phase_1/src/roboqc (commands, speech, intent, ui/main_window, ui/workers).
"""

import os

__path__.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "phase_1", "src", "roboqc")))
