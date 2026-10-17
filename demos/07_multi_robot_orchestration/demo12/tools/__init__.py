"""Probe and measurement tools for Demo 12.

None of these are part of the demo. They exist so that every number in
``docs/PHYSICS.md`` was measured rather than asserted, and so the measurements are
reproducible.

    probe_grasp_pose.py   Step 0. World direction of the jaw's local axes at the grasp
                          pose. Settles the horizontal-vs-vertical pad axis question.
    probe_grasp.py        Step 1. The four-criterion grasp probe (tracking, span, grip,
                          lift-and-release). Writes out12/grasp_probe.json.
"""

from __future__ import annotations

__all__ = ["probe_grasp_pose", "probe_grasp"]