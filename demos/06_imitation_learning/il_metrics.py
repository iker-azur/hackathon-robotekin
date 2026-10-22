"""What counts as success for the lift-and-place task, judged on physics alone.

Why this module exists
----------------------

The first version of the evaluation asked one question: *did the cube end within
3 cm of the drop point?* That is a poor test of a pick. A cube nudged 10 cm along
the table by a gripper that never closed on it ends in the same place as a cube
that was carried there. It also hid a second weakness: the scripted teacher pins
the cube to the gripper (``expert.Expert.pin_cube``) while the jaw is closed, so a
demonstration "succeeds" whether or not the fingers could hold it.

A learned policy gets no pin. So success here is stated in terms nothing but the
simulated contacts can fake, and it is decided **before** any policy is scored:

1. **lifted**     - the cube rose at least :data:`LIFT_MIN_M` above where it rested.
                    The plan lifts it 6 cm; 4 cm leaves slack and still cannot be
                    reached by a cube that was only bumped.
2. **stable grasp** - the cube stayed more than :data:`CARRY_Z_M` above the table
                    for at least :data:`CARRY_MIN_STEPS` consecutive steps (one
                    second of simulated time), and during that stretch it stayed
                    within :data:`HELD_MAX_DIST_M` of the point between the
                    fingers. A flicked cube is airborne for a fraction of a second
                    and far from the hand; a held one is neither.
3. **placed**     - the cube ends within :data:`PLACE_TOL_M` of the drop point,
                    back on the table (within :data:`REST_Z_TOL_M` of its resting
                    height) and no longer moving.

All three must hold. The verdict says which one failed, so a failure is a
diagnosis rather than a zero.

What this does *not* certify
----------------------------

It is a statement about the simulator: friction contacts on the SO-101's jaw as
LeIsaac models them, at the default friction. It says nothing about a real gripper.
It also does not say the *policy perceived* anything - that depends on what the
policy was given, which each evaluation report states next to its number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The cube rests on the table top at this height (centre), from demo 05's
# measurement of the shipped USD. Duplicated from ``il_env.CUBE_REST_Z`` on
# purpose: this module must import without the simulator, and ``il_env`` pulls it
# in. ``tests/test_il_metrics.py`` checks the two stay equal.
REST_Z_M = 0.056

# The plan lifts the cube ~6 cm. Four centimetres cannot be reached by bumping it.
LIFT_MIN_M = 0.04
# "Off the table": the cube centre is this far above its resting height.
CARRY_Z_M = 0.03
# One second of simulated time at 60 steps per second.
CARRY_MIN_STEPS = 60
# While carried, the cube centre has to stay this close to the point between the
# fingers (``il_env.HOLD`` in the gripper frame).
HELD_MAX_DIST_M = 0.03
# Same tolerance the demonstrations were accepted with (``expert.PLACEMENT_TOLERANCE_M``).
PLACE_TOL_M = 0.03
# Back on the table, not balanced on something or still in the air.
REST_Z_TOL_M = 0.008
# The cube has stopped: it moved less than this over the last SETTLE_WINDOW steps.
SETTLE_MOVE_M = 0.005
SETTLE_WINDOW = 30
# A carried stretch may lose the hold for a few steps (the arm accelerates); more
# than this fraction of the stretch away from the hand means it was not held.
HELD_MIN_FRACTION = 0.95


@dataclass
class Verdict:
    """The outcome of one episode, split into the three things that must hold."""

    lifted: bool
    stable_grasp: bool
    placed: bool
    success: bool
    failure: str  # "" on success, otherwise the first criterion that failed
    max_lift_m: float
    longest_carry_steps: int
    held_fraction: float
    final_error_m: float
    final_height_above_rest_m: float
    steps: int

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "lifted": self.lifted,
            "stable_grasp": self.stable_grasp,
            "placed": self.placed,
            "failure": self.failure,
            "max_lift_cm": round(self.max_lift_m * 100, 2),
            "longest_carry_steps": self.longest_carry_steps,
            "held_fraction": round(self.held_fraction, 3),
            "final_error_cm": round(self.final_error_m * 100, 2),
            "final_height_above_rest_cm": round(self.final_height_above_rest_m * 100, 2),
            "steps": self.steps,
        }


@dataclass
class EpisodeJudge:
    """Feed it one observation per step, then ask for the verdict.

    ``target_xy`` is the drop point in world x, y. ``held_distance`` is the
    distance from the cube centre to the point between the fingers; pass ``None``
    when it is not available and the stable-grasp test falls back to height alone
    (the report then says so, see :attr:`grasp_checked`).
    """

    target_xy: tuple[float, float]
    rest_z: float = REST_Z_M
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    held: list[float | None] = field(default_factory=list)

    def update(self, cube_xyz, held_distance: float | None = None) -> None:
        self.positions.append((float(cube_xyz[0]), float(cube_xyz[1]), float(cube_xyz[2])))
        self.held.append(None if held_distance is None else float(held_distance))

    @property
    def grasp_checked(self) -> bool:
        """Whether the distance to the hand was available for every step."""
        return all(value is not None for value in self.held)

    def verdict(self) -> Verdict:
        steps = len(self.positions)
        if steps == 0:
            raise ValueError("no steps recorded")

        heights = [p[2] - self.rest_z for p in self.positions]
        max_lift = max(heights)
        lifted = max_lift >= LIFT_MIN_M

        # The longest unbroken stretch above CARRY_Z_M, and how much of it the
        # cube spent in the hand.
        best = (0, 0, 0)  # (length, start, end exclusive)
        run_start = None
        for index, height in enumerate(heights + [float("-inf")]):
            if height > CARRY_Z_M:
                if run_start is None:
                    run_start = index
            elif run_start is not None:
                if index - run_start > best[0]:
                    best = (index - run_start, run_start, index)
                run_start = None
        longest, start, end = best

        if longest == 0:
            held_fraction = 0.0
        elif self.grasp_checked:
            distances = self.held[start:end]
            held_fraction = sum(1 for d in distances if d <= HELD_MAX_DIST_M) / len(distances)
        else:
            held_fraction = 1.0  # not measurable; the report flags it via grasp_checked

        stable = longest >= CARRY_MIN_STEPS and held_fraction >= HELD_MIN_FRACTION

        end_x, end_y, end_z = self.positions[-1]
        error = ((end_x - self.target_xy[0]) ** 2 + (end_y - self.target_xy[1]) ** 2) ** 0.5
        final_height = end_z - self.rest_z
        window = self.positions[-SETTLE_WINDOW:]
        moved = max(
            ((p[0] - end_x) ** 2 + (p[1] - end_y) ** 2 + (p[2] - end_z) ** 2) ** 0.5 for p in window
        )
        placed = (
            error < PLACE_TOL_M
            and abs(final_height) <= REST_Z_TOL_M
            and moved <= SETTLE_MOVE_M
            and steps >= SETTLE_WINDOW
        )

        if not lifted:
            failure = "never_lifted"
        elif not stable:
            failure = "not_held" if longest >= CARRY_MIN_STEPS else "dropped_early"
        elif not placed:
            failure = "misplaced" if error >= PLACE_TOL_M else "not_at_rest"
        else:
            failure = ""

        return Verdict(
            lifted=lifted,
            stable_grasp=stable,
            placed=placed,
            success=lifted and stable and placed,
            failure=failure,
            max_lift_m=max_lift,
            longest_carry_steps=longest,
            held_fraction=held_fraction,
            final_error_m=error,
            final_height_above_rest_m=final_height,
            steps=steps,
        )
