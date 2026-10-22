"""Demo 12 — multi-robot SO-101 orchestration on Isaac Lab.

Package layout, and which layer owns what:

    config.py      what the cell contains. Pure Python, no Isaac.
    materials.py   the physical parameters, with their provenance. Pure Python.
    isaac_env.py   builds the Isaac Lab scene. The ONLY physics authority.
    robot.py       one SO-101 handle: real joints, real DLS IK, real state reads.
    grasp.py       decides whether a grasp is real, from measured simulation state.
    orchestrator.py  who does what, and when. Reads the sim, never writes to it.
    worker.py      the Isaac Sim process: owns Kit, runs the loop, streams state.
    protocol.py    the message shapes shared with the interface.

The rule the whole design rests on: **the orchestrator never writes a pose into the
simulator.** It sends joint targets, exactly as a real robot's controller would, and
then reads back what PhysX actually did. If a part is not in a bin, it is because it
was not put there.
"""

__all__ = ["config", "materials", "isaac_env", "robot", "orchestrator", "worker", "server", "protocol"]