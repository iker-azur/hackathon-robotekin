"""Phase 2 vision pipeline: deterministic, CPU-only object detection and inspection.

Modules are layered so the robot planner can depend on world_state alone:

    camera -> detector -> tracker -> world_state (+ quality rules) -> resolver / UI
"""
