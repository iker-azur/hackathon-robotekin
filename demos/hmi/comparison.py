"""Explicitly synthetic imitation traces for the HMI demo (metres, Z-up).

Reference and follower are two illustrative joint trajectories, not a human
recording or a learned policy. Their TCP separation is calculated, not scored.
"""
from __future__ import annotations

import math

PERIOD = 12.0


def demo_joints(elapsed, names, follower=True):
    phase = 2 * math.pi * (elapsed - (.18 if follower else 0)) / PERIOD
    offset = .018 if follower else 0
    lift = -.30 + .12 * math.sin(phase)
    elbow = .65 + .12 * math.sin(phase + .8)
    return dict(zip(names, (
        .40 * math.sin(phase) + offset,
        lift,
        elbow,
        math.pi / 2 - lift - elbow,
        .15 * math.sin(phase),
        .4 + .25 * math.sin(phase * 2),
    )))


def tcp_position(links):
    """Fixed fingertip at local (0, 0, -0.104), matching the existing mesh."""
    pos, quat = links['gripper']
    w, x, y, z = quat
    norm = w*w + x*x + y*y + z*z
    if norm < 1e-12:
        raise ValueError('Invalid gripper quaternion')
    # Third column of quaternion rotation, times the fingertip offset.
    axis = (2*(x*z + w*y)/norm, 2*(y*z - w*x)/norm, 1 - 2*(x*x + y*y)/norm)
    return [pos[i] - .104 * axis[i] for i in range(3)]


def comparison_sample(elapsed, reference_links, robot_links):
    reference = tcp_position(reference_links)
    robot = tcp_position(robot_links)
    return {'kind': 'illustrative', 'reference_tcp': reference, 'robot_tcp': robot,
            'phase': (elapsed % PERIOD) / PERIOD,
            'distance_mm': round(math.dist(reference, robot) * 1000, 2)}
