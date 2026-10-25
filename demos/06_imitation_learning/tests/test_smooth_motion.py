"""Chunk blending and the target filter: smooth commands that still follow the policy."""

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load():
    path = Path(__file__).resolve().parent.parent / "smooth_motion.py"
    spec = importlib.util.spec_from_file_location("smooth_motion_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sm = load()


class Stub:
    """A policy whose k-th prediction is ``chunks[k]`` (tensors of shape (1, L, A))."""

    def __init__(self, chunks, gate=None):
        self.chunks, self.calls, self.resets, self.gate = chunks, 0, 0, gate
        self.config = SimpleNamespace(n_action_steps=30, chunk_size=len(chunks[0][0]))

    def predict_action_chunk(self, batch):
        k, self.calls = self.calls, self.calls + 1
        if self.gate is not None and k >= 1:
            self.gate.wait(5)
        return self.chunks[min(k, len(self.chunks) - 1)]

    def reset(self):
        self.resets += 1


def ramp(start, length=12, slope=1.0, joints=2):
    values = start + slope * torch.arange(length, dtype=torch.float32)
    return values[:, None].repeat(1, joints)[None]


class BlendedPolicyTests(unittest.TestCase):
    def test_a_stride_of_one_is_acts_own_temporal_ensembling(self):
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        torch.manual_seed(0)
        chunks = [torch.randn(1, 20, 6) for _ in range(40)]
        policy = sm.BlendedPolicy(Stub(chunks), stride=1, coeff=0.01)
        ensembler = ACTTemporalEnsembler(0.01, 20)
        for chunk in chunks:
            torch.testing.assert_close(policy.select_action(None), ensembler.update(chunk))

    def test_chunks_that_do_not_overlap_are_the_plain_open_loop(self):
        chunks = [ramp(100.0 * k, length=5) for k in range(4)]
        policy = sm.BlendedPolicy(Stub(chunks), stride=5)
        out = [float(policy.select_action(None)[0, 0]) for _ in range(20)]
        self.assertEqual(out, [100.0 * (t // 5) + t % 5 for t in range(20)])

    def test_blending_removes_the_jump_at_each_new_prediction(self):
        # the arm should move one degree per step, but the prediction made every 5 steps is 6 degrees off, alternately
        # above and below that path
        chunks = [(5.0 * k + 6.0 * (-1) ** k + torch.arange(60.0))[None, :, None] for k in range(12)]
        open_loop = [float(chunks[t // 5][0, t % 5, 0]) for t in range(60)]
        self.assertGreaterEqual(max(np.abs(np.diff(open_loop))), 12.0)  # the snap the arm used to make
        policy = sm.BlendedPolicy(Stub(chunks), stride=5, coeff=-0.2)
        out = [float(policy.select_action(None)[0, 0]) for _ in range(60)]
        self.assertLess(max(np.abs(np.diff(out))), 6.5)

    def test_the_sign_of_the_coefficient_decides_which_prediction_weighs_most(self):
        old, new = torch.zeros(1, 4, 1), torch.full((1, 4, 1), 10.0)
        for coeff, expected in ((0.0, 5.0), (-0.2, 10 * np.exp(0.2) / (1 + np.exp(0.2))),
                                (0.2, 10 / (1 + np.exp(0.2)))):
            policy = sm.BlendedPolicy(Stub([old, new]), stride=1, coeff=coeff)
            policy.select_action(None)
            self.assertAlmostEqual(float(policy.select_action(None)), expected, places=5, msg=str(coeff))

    def test_the_forward_pass_runs_in_a_worker_and_a_late_chunk_is_used_from_its_own_step(self):
        gate = threading.Event()
        old = 10.0 * torch.arange(12, dtype=torch.float32)[None, :, None]
        new = 100.0 + old
        policy = sm.BlendedPolicy(Stub([old, new], gate), stride=1, coeff=0.0, threaded=True)
        self.assertEqual(float(policy.select_action(None)), 0.0)  # the first prediction is waited for
        done = []
        thread = threading.Thread(target=lambda: done.extend([float(policy.select_action(None)),
                                                              float(policy.select_action(None))]))
        thread.start()
        thread.join(2)  # the second prediction is stuck in the worker: the loop must go on with the first chunk
        self.assertEqual(done, [10.0, 20.0])
        gate.set()
        policy._pending[0].result(5)
        # step 3: the second chunk was computed from the observation of step 1, so it contributes its element 2
        self.assertEqual(float(policy.select_action(None)), (30.0 + (100.0 + 20.0)) / 2)
        policy.close()

    def test_reset_forgets_the_chunks_and_resets_the_policy(self):
        stub = Stub([ramp(0.0), ramp(500.0)])
        policy = sm.BlendedPolicy(stub, stride=5)
        policy.select_action(None)
        policy.reset()
        self.assertEqual(stub.resets, 1)
        self.assertEqual(float(policy.select_action(None)[0, 0]), 500.0)  # predicted again, from element 0

    def test_everything_else_is_delegated_and_a_bad_stride_is_refused(self):
        policy = sm.BlendedPolicy(Stub([ramp(0.0)]))
        self.assertEqual(policy.config.chunk_size, 12)
        with self.assertRaises(ValueError):
            sm.BlendedPolicy(Stub([ramp(0.0)]), stride=0)

    def test_a_worker_slower_than_a_whole_chunk_holds_the_last_target_instead_of_failing(self):
        policy = sm.BlendedPolicy(Stub([ramp(0.0, length=4)]), stride=100)
        out = [float(policy.select_action(None)[0, 0]) for _ in range(8)]
        self.assertEqual(out, [0, 1, 2, 3, 3, 3, 3, 3])


class TargetFilterTests(unittest.TestCase):
    def test_a_step_is_followed_without_overshoot_and_settles(self):
        f = sm.TargetFilter(4.0)
        f.reset(np.zeros(6))
        out = np.array([f(np.full(6, 10.0), 1 / 30) for _ in range(60)])
        self.assertTrue(np.all(np.diff(out[:, 0]) >= 0))
        self.assertLessEqual(out.max(), 10.0)
        self.assertLess(abs(out[-1, 0] - 10.0), 1e-3)

    def test_it_starts_from_the_given_pose_so_the_first_command_does_not_jump(self):
        f = sm.TargetFilter()
        f.reset([5, 5, 5, 5, 5, 5])
        first = f(np.full(6, 50.0), 1 / 30)
        self.assertGreater(first[0], 5.0)
        self.assertLess(first[0], 5.0 + 0.5 * 45)  # a 45 degree step is not taken in one frame

    def test_without_a_pose_the_first_target_passes_through(self):
        f = sm.TargetFilter()
        np.testing.assert_allclose(f(np.arange(6.0), 1 / 30), np.arange(6.0))

    def test_the_step_is_exact_whatever_the_period(self):
        a, b = sm.TargetFilter(3.0), sm.TargetFilter(3.0)
        for f in (a, b):
            f.reset(np.zeros(6))
        big = a(np.full(6, 7.0), 0.5)
        for _ in range(5):
            small = b(np.full(6, 7.0), 0.1)
        np.testing.assert_allclose(big, small, atol=1e-9)

    def test_the_gripper_has_its_own_faster_cutoff(self):
        f = sm.TargetFilter(arm_hz=2.0, gripper_hz=12.0)
        f.reset(np.zeros(6))
        out = f(np.full(6, 10.0), 1 / 30)
        self.assertGreater(out[5], 3 * out[0])


if __name__ == "__main__":
    unittest.main()
