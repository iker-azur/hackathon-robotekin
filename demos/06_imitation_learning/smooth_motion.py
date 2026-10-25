"""Smoother commands for the real arm: blend the overlapping ACT chunks, then low-pass the targets.

Why the arm jerks. An ACT checkpoint predicts a chunk of 90 (or 60) future targets and, as trained, the apps run
30 of them open loop before asking again. Every new prediction starts a little away from where the old one was going,
so once a second the target jumps and the motor snaps to it. ACT's own remedy is temporal ensembling (Zhao et al.,
LeRobot's ``temporal_ensemble_coeff``): predict at every step and average all the predictions that cover the
current step. That costs one forward pass per control cycle; ``BlendedPolicy`` predicts every ``stride`` steps
instead and averages the overlapping chunks the same way (weights exp(-coeff * rank), the oldest prediction first),
so a stride of 5 costs a sixth and is almost as smooth.

``TargetFilter`` is a critically damped second-order filter on what is finally sent: it removes the remaining
frame-to-frame noise and cannot overshoot. ``analyze_motion_smoothness.py`` measures both on the held-out episodes.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEFAULT_STRIDE = 3
# ACT's paper uses +0.01 (the oldest prediction weighs most): smooth, but it averages predictions made from old
# images, so the arm drifts from the demonstrated path (15.7 deg from the leader's recorded action on the held-out
# episodes, against 10.9 for the old open loop). A negative value weighs the newest prediction most: -0.2 is smoother
# than the old way and closer to the demonstrations (9.5 deg), see analyze_motion_smoothness.py.
DEFAULT_COEFF = -0.2
DEFAULT_FILTER_HZ = 4.0  # lag ~ 1 / (pi * 4) = 80 ms; 3 Hz is smoother, 6 Hz tracks tighter (see the analysis)


class BlendedPolicy:
    """Wraps an ACT policy: ``select_action`` returns the weighted mean of every chunk that covers the current step.

    Same interface as the policy for the callers (``pre`` -> ``select_action`` -> ``post``): everything else is
    delegated, so ``config`` keeps working. ``stride`` = new prediction every this many steps. Weights are
    exp(-coeff * rank) with rank 0 for the oldest prediction, as in ACT's temporal ensembling.

    ``threaded``: the forward pass runs in a worker so the control loop never waits for it (it takes 20-120 ms,
    the loop period is 33 ms). A chunk is then used from the step of the observation it was computed from, so the
    delay does not shift the motion. Only the very first prediction is waited for.
    """

    def __init__(self, policy, stride: int = DEFAULT_STRIDE, coeff: float = DEFAULT_COEFF,
                 threaded: bool = False) -> None:
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self._policy, self.stride, self.coeff = policy, int(stride), float(coeff)
        self._pool = ThreadPoolExecutor(1, "blended-policy") if threaded else None
        self._pending = None  # (future, step of its observation)
        self._chunks: list[tuple] = []  # (chunk (L, A) tensor, step of its observation), oldest first
        self._step, self._queried = 0, None

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def reset(self) -> None:
        self._take(wait=True)  # let a running prediction finish: it must not land in the next motion
        self._chunks, self._step, self._queried = [], 0, None
        self._policy.reset()

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True)

    def _take(self, wait: bool) -> None:
        """Collect the prediction a worker has finished (or is still computing, when ``wait``)."""
        if self._pending and (wait or self._pending[0].done()):
            future, step = self._pending
            self._pending = None
            self._chunks.append((future.result(), step))

    def select_action(self, batch):
        import torch

        step, self._step = self._step, self._step + 1
        self._take(wait=False)
        if self._pending is None and (self._queried is None or step - self._queried >= self.stride):
            self._queried = step
            if self._pool:
                self._pending = (self._pool.submit(self._predict, batch), step)
            else:
                self._chunks.append((self._predict(batch), step))
        if not self._chunks:
            self._take(wait=True)  # the first prediction: nothing to send before it
        # a chunk is only usable while it still has predictions left (the newest one holds its last target if the
        # worker has been slower than a whole chunk)
        self._chunks = [c for c in self._chunks[:-1] if step - c[1] < len(c[0])] + self._chunks[-1:]
        actions = torch.stack([chunk[min(step - start, len(chunk) - 1)] for chunk, start in self._chunks])
        weights = torch.exp(-self.coeff * torch.arange(len(actions), device=actions.device, dtype=actions.dtype))
        return ((weights[:, None] * actions).sum(0) / weights.sum()).unsqueeze(0)

    def _predict(self, batch):
        return self._policy.predict_action_chunk(batch)[0]


class TargetFilter:
    """Critically damped low-pass on the six joint targets: smooth, no overshoot, lag about 1 / (pi * cutoff_hz).

    ``reset`` it with the measured pose when a motion starts, so the first command does not jump.
    """

    def __init__(self, arm_hz: float = DEFAULT_FILTER_HZ, gripper_hz: float = 10.0) -> None:
        self.cutoff = np.array([arm_hz] * 5 + [gripper_hz], dtype=np.float64)
        self.x = self.v = None

    def reset(self, pose=None) -> None:
        self.x = None if pose is None else np.asarray(pose, dtype=np.float64).copy()
        self.v = None if pose is None else np.zeros_like(self.x)

    def __call__(self, target, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        if self.x is None:
            self.x, self.v = target.copy(), np.zeros_like(target)
            return self.x.copy()
        w = 2.0 * math.pi * self.cutoff
        # exact step of x'' = w^2 (u - x) - 2 w x' for a constant target u over dt
        e = np.exp(-w * dt)
        d = self.x - target
        self.x, self.v = (target + (d + (self.v + w * d) * dt) * e,
                          (self.v - (self.v + w * d) * w * dt) * e)
        return self.x.copy()
