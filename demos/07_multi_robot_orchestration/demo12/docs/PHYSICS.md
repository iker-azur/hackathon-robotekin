# Demo 12 — Physics

## Constants provenance
- Masses: exact asset masses from SO-101 USD (base 0.147, shoulder 0.100, ...).
- Gravity: 9.81 m/s^2 (measured SI value).
- Friction: material-specific, bound via UsdPhysics.MaterialAPI.
- Gripper linkage travel: 12.5 mm per pad (tuning, not measurement).
- Arm actuator stiffness 900 N.m/rad (tuning).

## Approximations
- ARM_BODIES_WITH_COLLISION = () (arm self-collision off).
- Pads axis-aligned with jaw local Y, not world.
- wrist_roll held at zero.
- Linkage actuator gains are tuning.
