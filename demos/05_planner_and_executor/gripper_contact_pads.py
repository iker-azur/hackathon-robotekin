"""Experimental SO-101 fingertip pads for physical-grasp A/B measurements.

The LeIsaac USD stays untouched.  Isaac authors a sibling copy in its asset cache so
relative asset paths continue to resolve exactly as they do for the stock robot.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


PAD_RADIUS_M = 0.005
PAD_THICKNESS_M = 0.002
PAD_VEE_ANGLE_DEG = 35.0
# A paired 18 mm jitter sweep improved the 20 mm ball from 3/5 with 30 steps to 4/5
# with 45. Keep the experimental override for repeatable A/B runs; the default remains
# scoped to this one validated profile.
BALL_CLOSE_RAMP_STEPS = 45
TRIANGLE_CLOSE_RAMP_STEPS = 20
TRIANGLE30_CLOSE_RAMP_STEPS = 45
CYLINDER30_CLOSE_RAMP_STEPS = 20
HEXAGON30_CLOSE_RAMP_STEPS = 20
CUBE30_CLOSE_RAMP_STEPS = 45
PAD_CENTRES_LOCAL_M = {
    "gripper": (-0.0031, 0.0064, -0.0976),
    "jaw": (-0.0112, -0.0716, 0.0124),
}
# Inward-facing local normals measured at close_command=0.15 rad on a centered 30 mm
# hexagonal prism trial. The faces use those axes and keep their inner surfaces at the
# calibrated contact centres; only the pad body extends away from the object.
PAD_NORMALS_LOCAL = {
    "gripper": (0.997140774, -0.000010749, 0.075566379),
    "jaw": (-0.997327336, 0.073062894, -0.000010616),
}


def vee_profile_for(kind: str, size_m: tuple[float, ...], yaw_deg: float = 0.0) -> bool:
    """Return only the object profiles tested with a physical V-groove grasp.

    The profile is deliberately narrow: V30 improved the 20 mm cube; V40 passed physical
    carries on the 20 mm cylinder, hexagon and 30 mm triangle; V45 with staged closure
    passed clustered 20 mm triangle trials across all three cameras and a contained side
    target. V inserts regressed a 30 mm hexagon and the ball.
    Polygon symmetry is accounted for when comparing yaw.
    """
    if not size_m:
        return False
    size_mm = round(float(size_m[0]) * 1000)
    yaw = float(yaw_deg) % 360.0
    if kind == "ball" and size_mm == 20:
        # The 5 mm / 2 mm V35 insert closes around the small sphere. Five strict,
        # randomized physical carries passed at 0.10 rad on the flat tabletop collider.
        return True
    if kind == "cube" and size_mm == 20:
        # V30 passed ten isolated physical lift-and-carry trials at 0 degrees. A square
        # presents the same face pair after a 90-degree rotation; other square yaws are
        # aligned by tool_yaw_for before the grasp.
        return True
    if kind == "hexagonal_prism" and size_mm == 20:
        # Both hexagon symmetry orientations are validated with V40: flat-on at 30 degrees
        # and vertex-on at 0 degrees. Intermediate orientations remain on stock fingertips.
        return abs(((yaw + 15.0) % 30.0) - 15.0) <= 2.0
    if kind == "cylinder" and size_mm == 20:
        # V40, stiffness 500 and close=0.20 passed five physical lift-and-carry trials.
        # The upright cylinder is rotationally symmetric.
        return True
    if kind == "triangular_prism" and size_mm == 20:
        # V45 with the 20-step closure ramp passed 3/3 side-contained randomized cluster
        # carries and one open-cluster carry on each camera. The prior V40 profile slipped
        # on a side-contained seed even when the aperture was centred on the prism. Yaw
        # robustness still needs its own varied-orientation sweep.
        return True
    if kind == "triangular_prism" and size_mm == 30:
        # The executor rotates the wrist to cancel the object's yaw before contact, so
        # every scene orientation maps to the one physically tested V-groove pose.
        return True
    return False


def scene_uses_only_vee_profiles(profiles: list[tuple]) -> bool:
    """V geometry can be loaded only for one validated, physically compatible profile."""
    if not profiles or not all(
            vee_profile_for(profile[0], profile[1], profile[2] if len(profile) > 2 else 0.0)
            for profile in profiles):
        return False
    angles = {vee_angle_for(profile[0], profile[1], PAD_VEE_ANGLE_DEG,
                            profile[2] if len(profile) > 2 else 0.0)
              for profile in profiles}
    return len(angles) == 1


def validated_profile_items(items, target_kind: str | None = None):
    """Return objects relevant to V-pad compatibility for a scene or resolved target.

    Target-scoped selection ignores unvalidated distractors, including same-kind objects
    with a different size. Scene-wide selection still considers every movable object.
    """
    return [item for item in items if not item.container
            and (target_kind is None or (item.kind == target_kind
                 and vee_profile_for(item.kind, item.size,
                                     getattr(item, "yaw_deg", 0.0))))]


def flat_profile_for(kind: str, size_m: tuple[float, ...]) -> bool:
    """Profiles that hold best on flat inward-facing pads instead of the V-groove.

    The 30 mm triangular prism: strict trials on the same 12 scenes (open and contained,
    single and crowded, 3 repetitions) gave 0/12 with the V45 insert and 12/12 with flat pads.
    A trace showed the V slabs tip the 24 mm-tall triangle on the close (tilt 5-11 degrees,
    part lifted off the table) and the vertex never seats against the fixed pad.
    """
    return bool(kind == "triangular_prism" and size_m
                and round(float(size_m[0]) * 1000) == 30)


def automatic_scene_pad_shape(items, target_kind: str | None = None) -> str | None:
    """Select flat or V pads for a validated target profile, optionally amid non-target clutter.

    Without ``target_kind`` the safe default remains scene-wide: every movable object must
    have a validated profile. A caller that has resolved one specific grasp may scope the
    decision to that target; this is useful for mixed scenes with distant, untouched
    distractors and must not be used for an unfiltered multi-object command.
    """
    selected = validated_profile_items(items, target_kind)
    profiles = [(item.kind, item.size, getattr(item, "yaw_deg", 0.0)) for item in selected]
    if not scene_uses_only_vee_profiles(profiles):
        return None
    if all(flat_profile_for(item.kind, item.size) for item in selected):
        return "flat"
    return "vee"


def vee_angle_for(kind: str, size_m: tuple[float, ...], default: float,
                  yaw_deg: float = 0.0, override_deg: float | None = None) -> float:
    """Return an experimentally measured V angle for a supported object profile."""
    if vee_profile_for(kind, size_m, yaw_deg):
        if override_deg is not None:
            angle = float(override_deg)
            if not 15.0 <= angle <= 70.0:
                raise ValueError("experimental V angle must be within [15, 70] degrees")
            return angle
        if kind == "cube" and round(float(size_m[0]) * 1000) == 20:
            return 30.0
    if kind == "ball" and round(float(size_m[0]) * 1000) == 20:
        return 35.0
    if kind == "triangular_prism" and round(float(size_m[0]) * 1000) == 20:
        return 45.0
    if kind == "triangular_prism" and round(float(size_m[0]) * 1000) == 30:
        # V45 with a 30 mm face span and 1 mm inward inset completed 17/18 strict,
        # randomized camera/scene trials; the remaining miss was carry slip and passed
        # three exact-seed repeats with a 400-step transport.
        return 45.0
    # V40 is validated for the 20 mm hexagon and other established profiles.
    return 40.0
    return float(default)


def experimental_pad_inset_m(override_mm: str | None) -> float:
    """Validate an opt-in inward pad offset used to test contact retention."""
    if override_mm is None or not str(override_mm).strip():
        return 0.0
    value = float(override_mm)
    if not 0.0 <= value <= 12.0:
        raise ValueError("experimental pad inset must be within [0, 12] mm")
    return value / 1000.0


def pad_inset_for(kind: str, size_m: tuple[float, ...],
                  override_mm: str | None = None) -> float:
    """Use the repeated-contact 1 mm insert only for the 30 mm triangular prism profile."""
    if override_mm is not None and str(override_mm).strip():
        return experimental_pad_inset_m(override_mm)
    if (kind == "triangular_prism" and size_m
            and round(float(size_m[0]) * 1000) == 30):
        return 0.001
    return 0.0


def experimental_pad_radius_m(override_mm: str | None) -> float:
    """Validate an opt-in spherical-tip radius for small round-object trials."""
    if override_mm is None or not str(override_mm).strip():
        return PAD_RADIUS_M
    value = float(override_mm)
    if not 1.0 <= value <= 10.0:
        raise ValueError("experimental spherical-pad radius must be within [1, 10] mm")
    return value / 1000.0


def experimental_pad_span_m(override_mm: str | None) -> float:
    """Validate an opt-in V-pad face span for better small-object support."""
    if override_mm is None or not str(override_mm).strip():
        return 0.018
    value = float(override_mm)
    if not 12.0 <= value <= 40.0:
        raise ValueError("experimental V-pad span must be within [12, 40] mm")
    return value / 1000.0


def pad_span_for(kind: str, size_m: tuple[float, ...],
                 override_mm: str | None = None) -> float:
    """Use the validated wide V face for 30 mm triangles, otherwise the normal span."""
    if override_mm is not None and str(override_mm).strip():
        return experimental_pad_span_m(override_mm)
    if (kind == "triangular_prism" and size_m
            and round(float(size_m[0]) * 1000) == 30):
        return 0.030
    return experimental_pad_span_m(None)


def experimental_vee_angle_deg(override_deg: str | None, default: float = PAD_VEE_ANGLE_DEG) -> float:
    """Validate an opt-in V-pad angle, including unvalidated object profiles."""
    if override_deg is None or not str(override_deg).strip():
        return float(default)
    value = float(override_deg)
    if not 5.0 <= value <= 45.0:
        raise ValueError("experimental V-pad angle must be within [5, 45] degrees")
    return value


def force_unvalidated_experimental_pads(override: str | None) -> bool:
    """Allow an explicitly requested, unvalidated V-pad experiment.

    This is intentionally opt-in and does not change the physical-grasp policy. It only
    bypasses the automatic compatibility guard that otherwise keeps V inserts off shapes
    whose fit has not been measured.
    """
    return str(override or "").strip().lower() in {"1", "true", "yes", "on"}


def vee_stiffness_for(kind: str, size_m: tuple[float, ...], default: float,
                      yaw_deg: float = 0.0) -> float:
    """Use the tested squeeze stiffness for validated V-pad grasp profiles."""
    if vee_profile_for(kind, size_m, yaw_deg):
        if (kind == "triangular_prism" and size_m
                and round(float(size_m[0]) * 1000) == 30):
            # A strict front-contained miss was centred between V40 pads but slipped on lift
            # at stiffness 500. The same pose completed a verified physical carry at 700.
            return max(float(default), 700.0)
        return 500.0
    return float(default)


def automatic_scene_grip_stiffness(items, default: float,
                                   target_kind: str | None = None) -> float:
    """Raise stock-finger stiffness for homogeneous, physically tested 30 mm profiles."""
    movable = [item for item in items if not getattr(item, "container", False)]
    if target_kind is not None:
        movable = [item for item in movable if item.kind == target_kind]
    kinds = {item.kind for item in movable}
    tested_30mm_profile = lambda item: bool(
        item.size and round(float(item.size[0]) * 1000) == 30)
    if movable and len(kinds) == 1 and all(tested_30mm_profile(item) for item in movable):
        if kinds == {"cylinder"}:
            # A 30 mm cylinder can be visually centred yet slip during lift at stiffness 500.
            # 1000 with staged closure passed randomized physical front-contained trials.
            return max(float(default), 1000.0)
        if kinds == {"hexagonal_prism"}:
            return max(float(default), 500.0)
    return float(default)


def last_visual_correction_height(kind: str, size_m: tuple[float, ...],
                                  pad_shape: str | None, default: float,
                                  yaw_deg: float = 0.0) -> float:
    """Allow low visual centering only for profiles validated to tolerate it."""
    valid_profile = vee_profile_for(kind, size_m, yaw_deg)
    yaw = float(yaw_deg) % 360.0
    effective_yaw = (yaw + tool_yaw_for(kind, size_m, yaw)) % 60.0
    hex20_low_correction_tested = (
        kind == "hexagonal_prism" and size_m
        and round(float(size_m[0]) * 1000) == 20
        and abs(((effective_yaw - 30.0 + 30.0) % 60.0) - 30.0) <= 2.0)
    triangle30_low_correction_tested = (
        kind == "triangular_prism" and size_m
        and round(float(size_m[0]) * 1000) == 30)
    if (pad_shape == "vee" and valid_profile
            and (triangle30_low_correction_tested or hex20_low_correction_tested)):
        # Only the 30 mm triangular prism and the validated flat-on 20 mm hexagon tolerate
        # visual corrections down to this height. Hexagon yaw is evaluated after tool_yaw_for:
        # yaw-zero objects are wrist-rotated by 30 degrees into this same tested flat-on pose.
        # The 20 mm triangle profile retains the conservative default until its integrated
        # approach is validated.
        return 0.13
    return float(default)


def close_command_for(kind: str, size_m: tuple[float, ...], default: float,
                      override: float | None = None,
                      pad_shape: str | None = None) -> float:
    """Return an explicit experiment/user override or the validated object profile."""
    if override is not None:
        value = float(override)
        if not 0.0 <= value <= 1.745:
            raise ValueError("close-command override must be within [0, 1.745] rad")
        return value
    if not size_m:
        return float(default)
    size_mm = round(float(size_m[0]) * 1000)
    if kind == "ball":
        if size_mm == 20 and pad_shape == "vee":
            # The isolated contact bench passed at 0.10 rad, but an E2E ball at the
            # randomized seed-2 edge pose slipped on lift with 0.10 and 0.15 rad. The
            # same pose completed strict physical lift/carry/release at 0.20 rad.
            return 0.20
        return 0.20
    if kind == "hexagonal_prism" and size_mm == 20:
        # The validated isolated V-pad profile is the stock 0.20 rad close target.
        # Reducing it to 0.10 rad in the integrated executor caused a real E2E miss:
        # the hexagon was between both pads but had insufficient normal force to lift.
        return float(default)
    if kind == "hexagonal_prism" and size_mm == 30:
        # The integrated yaw-aware grasp is less forgiving to a marginally off-centre face
        # contact than the isolated bench trial. 0.15 rad passed strict transport trials and
        # stays below the 0.20 rad setting that over-squeezed this linkage.
        return 0.15
    if kind == "triangular_prism" and size_mm == 20 and pad_shape == "vee":
        # V45 with close=0.10 and the 20-step close ramp passed randomized cluster carries.
        return 0.10
    if kind == "triangular_prism" and size_mm == 30:
        return 0.15
    return float(default)


# Distance between the two pad centres as a function of the gripper joint angle, fitted to the
# traced closes (strict-physical, 2026-10-19): 70 mm at 0.822 rad, 55 mm at 0.60, 27 mm at 0.20
# with a 30 mm part held, 22-23 mm at 0.15.
JAW_SEPARATION_INTERCEPT_MM = 14.0
JAW_SEPARATION_MM_PER_RAD = 68.0
# A part held between the pads has its surfaces about 3 mm outside the pad centres.
PAD_CENTRE_INSET_MM = 3.0
# Slack kept on each side of the part when the jaws are open for the approach. The aperture
# servo settles within 5 mm, so 8 mm per side straddles the part with room to spare.
OPEN_SLACK_PER_SIDE_MM = 8.0
MIN_OPEN_COMMAND_RAD = 0.30


def open_command_for(kind: str, size_m: tuple[float, ...], stock_open: float,
                     close_command: float) -> float:
    """Jaw angle that opens the pads just wider than the part, never wider than the stock.

    The stock open angle (0.822 rad) puts the pads 70 mm apart, which leaves 20 mm of slack on
    each side of a 30 mm part and puts the moving jaw within about 10 mm of a tray wall 90 mm
    away. Strict traces showed the jaw clipping that rim: it was pushed open to 0.91-1.02 rad
    and stayed wedged there (pads 78-83 mm apart) while its target ramped to 0.2 rad, so the
    part was never gripped. Opening only as far as the part needs keeps the jaw clear.
    """
    del kind
    if not size_m:
        return float(stock_open)
    width_mm = max(float(value) for value in size_m[:2]) * 1000.0
    separation_mm = width_mm - PAD_CENTRE_INSET_MM + 2.0 * OPEN_SLACK_PER_SIDE_MM
    angle = (separation_mm - JAW_SEPARATION_INTERCEPT_MM) / JAW_SEPARATION_MM_PER_RAD
    floor = max(MIN_OPEN_COMMAND_RAD, float(close_command) + 0.10)
    return float(min(float(stock_open), max(floor, angle)))


def staged_close_ramp_steps(kind: str, size_m: tuple[float, ...], total_steps: int,
                            ramp_steps: int | None = None) -> int:
    """Stage closure for small profiles where a rapid jaw impact can lose contact."""
    if (kind == "hexagonal_prism" and size_m
            and round(float(size_m[0]) * 1000) == 30 and int(total_steps) > 1):
        # One randomized top/open 30 mm hexagon was displaced by the stock close transient.
        # Ramp the existing stock-jaw target gradually; validate in strict trials before
        # broadening this profile to other sizes or geometries.
        requested = (HEXAGON30_CLOSE_RAMP_STEPS if ramp_steps is None
                     else max(0, int(ramp_steps)))
        return min(requested, int(total_steps) - 1)
    if (kind == "cube" and size_m
            and round(float(size_m[0]) * 1000) == 30 and int(total_steps) > 1):
        # At forward reach the SO-101 linkage moves its aperture several millimetres
        # in the first three close ticks. Ramp the jaw command so the per-tick aperture
        # servo can compensate before the pads push a 30 mm cube out of the grasp.
        requested = (CUBE30_CLOSE_RAMP_STEPS if ramp_steps is None
                     else max(0, int(ramp_steps)))
        return min(requested, int(total_steps) - 1)
    if (kind == "cylinder" and size_m
            and round(float(size_m[0]) * 1000) == 30 and int(total_steps) > 1):
        requested = CYLINDER30_CLOSE_RAMP_STEPS if ramp_steps is None else max(0, int(ramp_steps))
        return min(requested, int(total_steps) - 1)
    if (kind == "ball" and size_m
            and round(float(size_m[0]) * 1000) in (20, 30) and int(total_steps) > 1):
        # Without the ramp the stock jaw hits a 30 mm ball hard enough to move it 22 mm in the
        # first three physics ticks (reproduced twice on the central pick pose, strict-physical,
        # 2026-10-19), so the 8 mm centre-drift guard aborts the close before any grip. With the
        # 45-step ramp the same scenes held the ball and carried it to the tray.
        requested = BALL_CLOSE_RAMP_STEPS if ramp_steps is None else max(0, int(ramp_steps))
        return min(requested, int(total_steps) - 1)
    if (kind == "triangular_prism" and size_m
            and round(float(size_m[0]) * 1000) in (20, 30) and int(total_steps) > 1):
        default_ramp = (TRIANGLE30_CLOSE_RAMP_STEPS
                        if round(float(size_m[0]) * 1000) == 30
                        else TRIANGLE_CLOSE_RAMP_STEPS)
        requested = (default_ramp if ramp_steps is None
                     else max(0, int(ramp_steps)))
        return min(requested, int(total_steps) - 1)
    return 0


def tool_yaw_for(kind: str, size_m: tuple[float, ...], object_yaw_deg: float) -> float:
    """Choose a wrist angle that brings a symmetric grasp profile to its tested pose.

    A 30 mm hexagon held with stock flat fingers is stable when a jaw axis is aligned with
    opposite facets; the equivalent pick at object yaw zero requires a 30 degree wrist rotation.
    The 20 mm profile uses the same validated facet alignment. Hexagonal symmetry repeats every
    60 degrees, so choose the smallest equivalent rotation in [-30, 30].
    Square cubes align one face pair with the jaws. Tested 30 mm triangular prisms
    rotate to the validated zero-yaw V-groove pose; their symmetry repeats every 120°.
    Other geometries keep the calibrated neutral tool orientation.
    """
    if (kind == "hexagonal_prism" and size_m
            and round(float(size_m[0]) * 1000) in (20, 30)):
        yaw = (30.0 - float(object_yaw_deg) + 30.0) % 60.0 - 30.0
        # At the symmetry boundary either sign is equivalent geometrically; the positive
        # direction is the one validated by the isolated 30 mm hexagon tool-yaw trials.
        # The 20 mm V40 profile also has a validated flat-on pose at 30 degrees; move its
        # usual vertex-on yaw-zero scene to that face-pair alignment before closing.
        return 30.0 if yaw <= -29.999 else yaw
    if kind == "cube" and size_m:
        size_mm = round(float(size_m[0]) * 1000)
        if size_mm in (20, 30, 40):
            # A cube at 30° presents a 42 mm diagonal to 36.2 mm jaws. Align a face
            # pair instead of attempting an impossible diagonal pinch.
            return (float(object_yaw_deg) + 45.0) % 90.0 - 45.0
    if (kind == "triangular_prism" and size_m
            and round(float(size_m[0]) * 1000) in (20, 30)):
        if round(float(size_m[0]) * 1000) == 20:
            # Neutral wrist carried yaw 0 and 30. At yaw 60, +60 carried 3/3 while
            # the equivalent -60 pose saturated IK. Select the reachable pose, then
            # repeat the policy every 120 degrees (triangle symmetry).
            yaw = float(object_yaw_deg) % 120.0
            return 60.0 if 45.0 <= yaw <= 90.0 else 0.0
        return (60.0 - float(object_yaw_deg)) % 120.0 - 60.0
    return 0.0


def alternate_tool_yaw_for(kind: str, size_m: tuple[float, ...],
                           object_yaw_deg: float) -> float | None:
    """Return a second geometric contact orientation for a bounded physical retry.

    The default 30 mm triangle grasp uses the known neutral orientation. A retry may bisect
    its vertex/face alignment and place the jaws on the two sloping facets instead. This
    alternative is intentionally not selected on the first attempt: it helped a strict
    front-contained triangle trial but regressed some top-view placements. A 30 mm hexagon
    has two distinct symmetric jaw orientations, facet-on and vertex-on; after an observed
    physical miss, try the other orientation without changing the first attempt.
    """
    if kind in {"ball", "cylinder"} and size_m and len(size_m) >= 2:
        # A round horizontal footprint has no preferred jaw axis. Rotating the tool on a
        # failed approach preserves the grasp geometry while giving the 5-DOF IK controller
        # another wrist posture; this matters when the neutral posture saturates wrist_flex
        # or wrist_roll near the forward edge of the workspace.
        return float(object_yaw_deg) + 30.0
    if kind == "cube" and size_m and round(float(size_m[0]) * 1000) in (20, 30, 40):
        # A quarter turn presents the same pair of faces on a cube, but is a different
        # articulated SO-101 posture and can avoid a wrist joint limit.
        return tool_yaw_for(kind, size_m, object_yaw_deg) + 90.0
    if (kind == "hexagonal_prism" and size_m
            and round(float(size_m[0]) * 1000) == 30):
        primary = tool_yaw_for(kind, size_m, object_yaw_deg)
        return (primary + 60.0) % 60.0 - 30.0
    if (kind != "triangular_prism" or not size_m
            or round(float(size_m[0]) * 1000) != 30):
        return None
    primary = tool_yaw_for(kind, size_m, object_yaw_deg)
    yaw = (primary + 120.0) % 120.0 - 60.0
    return 60.0 if yaw <= -59.999 else yaw


def tool_yaw_for_workspace(kind: str, size_m: tuple[float, ...], object_yaw_deg: float,
                           radial_reach_m: float | None) -> float:
    """Select a grasp wrist pose that fits the measured arm reach annulus.

    Strict A/B trials show the symmetric facet pinch is useful at the outer forward pose,
    while the neutral pose tracks better near the inner boundary. Other geometries retain
    their established orientation policy. The crossover is deliberately inside the tested
    forward-pick annulus and remains a conservative, explicit calibration until a live IK
    residual score can choose between candidate poses.
    """
    primary = tool_yaw_for(kind, size_m, object_yaw_deg)
    alternate = alternate_tool_yaw_for(kind, size_m, object_yaw_deg)
    # A 30 mm cube at the forward-pick radius (0.241 m) drives wrist_flex and wrist_roll to
    # their limits in the neutral pose. The equivalent quarter-turn pinch was measured to bring
    # the aperture within 6.4 mm at the same pose, so begin with that reachable posture instead
    # of spending a full failed neutral approach first.
    if (kind == "cube" and size_m and round(float(size_m[0]) * 1000) == 30
            and radial_reach_m is not None and radial_reach_m >= 0.230):
        alternate = alternate_tool_yaw_for(kind, size_m, object_yaw_deg)
        if alternate is not None:
            return alternate
    # The reach-conditioned triangle alternative is validated only for 30 mm prisms. The
    # alternate hexagon pose remains a retry candidate until a live first-attempt matrix
    # validates its reach-dependent use.
    if (kind == "triangular_prism" and alternate is not None
            and radial_reach_m is not None and radial_reach_m >= 0.250):
        return alternate
    return primary


def tool_yaw_for_view(kind: str, size_m: tuple[float, ...], object_yaw_deg: float,
                      radial_reach_m: float | None, camera: str | None = None) -> float:
    """Account for camera-conditioned target-pose error in the measured triangle profile."""
    if (kind == "triangular_prism" and size_m
            and round(float(size_m[0]) * 1000) == 30 and camera == "side"):
        # A strict side-camera open-scene trial at the outer annulus missed contact after the
        # +60 degree body-offset rotation. Keep that view on the neutral pose; retry logic can
        # still test the alternate if the physical close/lift reports a miss.
        return tool_yaw_for(kind, size_m, object_yaw_deg)
    return tool_yaw_for_workspace(kind, size_m, object_yaw_deg, radial_reach_m)


def retry_tool_yaw_for(preferred_yaw: float, alternate_yaw: float | None,
                       previous_yaw: float | None) -> float:
    """Switch to the other tested orientation after a failed physical grasp."""
    if alternate_yaw is None:
        return float(preferred_yaw)
    if previous_yaw is not None and abs(float(previous_yaw) - float(alternate_yaw)) <= 1e-6:
        return float(preferred_yaw)
    return float(alternate_yaw)


def carry_steps_for(kind: str, size_m: tuple[float, ...], default_steps: int) -> int:
    """Reduce carry acceleration for the 30 mm triangular prism's narrow edge margin."""
    if (kind == "triangular_prism" and size_m
            and round(float(size_m[0]) * 1000) == 30):
        return max(int(default_steps), 400)
    return int(default_steps)


def make_padded_robot_asset(source_usd: str | Path,
                            radius_m: float = PAD_RADIUS_M,
                            shape: str = "sphere",
                            thickness_m: float = PAD_THICKNESS_M,
                            vee_angle_deg: float = PAD_VEE_ANGLE_DEG,
                            inset_m: float = 0.0,
                            span_m: float = 0.018,
                            cache_dir: str | Path | None = None) -> Path:
    """Return a cached robot USD with experimental finger contact pads.

    The pad centres use Demo 11's measured link-local finger contact centres.  Each
    pad is a child collider of the existing rigid link, so it moves with the finger and
    does not add a separate body or constraint. ``sphere`` reproduces the first A/B; ``flat``
    uses a thin, inward-facing cuboid whose inner face is flush with the calibrated contact
    centre. ``vee`` uses two thin convex slabs per finger to form a shallow contact cradle.
    The caller binds the same measured rubber material as the stock fingers.
    """
    source = Path(source_usd).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SO-101 USD not found: {source}")
    if not 0.001 <= float(radius_m) <= 0.010:
        raise ValueError("experimental pad radius must be between 1 and 10 mm")
    if shape not in {"sphere", "flat", "vee"}:
        raise ValueError("experimental contact-pad shape must be 'sphere', 'flat', or 'vee'")
    if not 0.0005 <= float(thickness_m) <= 0.005:
        raise ValueError("experimental flat-pad thickness must be between 0.5 and 5 mm")
    if shape == "vee" and not 5.0 <= float(vee_angle_deg) <= 45.0:
        raise ValueError("experimental V-pad angle must be between 5 and 45 degrees")
    if not 0.0 <= float(inset_m) <= 0.012:
        raise ValueError("pad inset must be between 0 and 12 mm")
    if shape == "vee" and not 0.012 <= float(span_m) <= 0.040:
        raise ValueError("V-pad span must be between 12 and 40 mm")

    # Hugging Face snapshots are commonly read-only. Keep generated USDs inside the demo's
    # writable cache, while retaining the assets/robots suffix so relative robot asset paths
    # keep the same layout as the source repository. The cache can be overridden for CI.
    if cache_dir is None:
        cache_dir = os.environ.get("DEMO11_ROBOT_ASSET_CACHE")
    cache_root = (Path(cache_dir).expanduser().resolve() if cache_dir else
                  Path(__file__).resolve().parent / "out11" / "robot_asset_cache")
    robot_dir = cache_root / "assets" / "robots"
    robot_dir.mkdir(parents=True, exist_ok=True)
    local_source = robot_dir / source.name
    if (not local_source.is_file()
            or local_source.stat().st_size != source.stat().st_size
            or local_source.stat().st_mtime_ns != source.stat().st_mtime_ns):
        shutil.copy2(source, local_source)

    mm_key = lambda value: f"{value * 1000:.1f}".rstrip("0").rstrip(".").replace(".", "p")
    deg_key = f"{vee_angle_deg:.1f}".rstrip("0").rstrip(".").replace(".", "p")
    profile = (f"flat_{mm_key(thickness_m)}mm" if shape == "flat"
               else f"vee_{mm_key(thickness_m)}mm_{deg_key}deg_"
                    f"inset{mm_key(inset_m)}mm_span{mm_key(span_m)}mm" if shape == "vee"
               else f"sphere_{mm_key(radius_m)}mm")
    output = robot_dir / f"{source.stem}_demo11_pads_{profile}.usd"
    if output.is_file():
        return output

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(local_source))
    if stage is None:
        raise RuntimeError(f"Could not open SO-101 USD: {local_source}")
    for link, centre in PAD_CENTRES_LOCAL_M.items():
        link_path = f"/so101_new_calib/{link}"
        if not stage.GetPrimAtPath(link_path):
            raise RuntimeError(f"Expected SO-101 rigid link is missing: {link_path}")
        pad_path = f"{link_path}/demo11_rubber_contact_pad"
        if shape == "sphere":
            pad = UsdGeom.Sphere.Define(stage, pad_path)
            pad.CreateRadiusAttr(float(radius_m))
            UsdGeom.Xformable(pad).AddTranslateOp().Set(tuple(float(v) for v in centre))
            pad.CreateDisplayColorAttr([(0.12, 0.12, 0.12)])
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        else:
            # The V profile is an opt-in retention experiment. Two thin convex slabs
            # share a central seam and tilt in opposite directions around local Y,
            # creating a shallow cradle in the finger's vertical contact section.
            # It remains two independent convex colliders for PhysX compatibility.
            pad_paths = ([pad_path] if shape == "flat" else
                         [f"{pad_path}_upper", f"{pad_path}_lower"])
            for index, child_path in enumerate(pad_paths):
                pad = UsdGeom.Cube.Define(stage, child_path)
                pad.CreateSizeAttr(1.0)
                xform = UsdGeom.Xformable(pad)
                inward = Gf.Vec3d(*PAD_NORMALS_LOCAL[link])
                inward.Normalize()
                outward_offset = tuple(float(centre[i] + inward[i] * inset_m
                                             - inward[i] * thickness_m / 2)
                                       for i in range(3))
                xform.AddTranslateOp().Set(outward_offset)
                orient_d = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), inward).GetQuat()
                if shape == "vee":
                    tilt_sign = 1.0 if index == 0 else -1.0
                    tilt_d = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0),
                                         tilt_sign * float(vee_angle_deg)).GetQuat()
                    orient_d = orient_d * tilt_d
                orient = Gf.Quatf(float(orient_d.GetReal()),
                                  Gf.Vec3f(*[float(v) for v in orient_d.GetImaginary()]))
                xform.AddOrientOp().Set(orient)
                # Keep V slabs narrower than flat pads to limit interference as the
                # two opposing fingers close around small parts.
                width = 0.020 if shape == "flat" else float(span_m)
                height = 0.018 if shape == "flat" else float(span_m)
                xform.AddScaleOp().Set((float(thickness_m), width, height))
                pad.CreateDisplayColorAttr([(0.12, 0.12, 0.12)])
                UsdPhysics.CollisionAPI.Apply(pad.GetPrim())

    if not stage.GetRootLayer().Export(str(output)):
        raise RuntimeError(f"Could not export experimental SO-101 USD: {output}")
    # Re-open the exported asset and check both colliders are present before returning it.
    check = Usd.Stage.Open(str(output))
    for link in PAD_CENTRES_LOCAL_M:
        pad_paths = ([f"/so101_new_calib/{link}/demo11_rubber_contact_pad"]
                     if shape != "vee" else
                     [f"/so101_new_calib/{link}/demo11_rubber_contact_pad_upper",
                      f"/so101_new_calib/{link}/demo11_rubber_contact_pad_lower"])
        for pad_path in pad_paths:
            prim = check.GetPrimAtPath(pad_path)
            if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
                output.unlink(missing_ok=True)
                raise RuntimeError(f"Exported pad collider failed validation: {pad_path}")
    return output
