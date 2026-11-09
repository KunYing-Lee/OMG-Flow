import numpy as np
import pytest

from deployment.sync_flow import SyncScheduler, TICK_NS


def scheduler(period=0.2, seed=1234):
    standing = np.zeros(36, dtype=np.float32)
    standing[2:4] = 0.79, 1
    return SyncScheduler(standing, 10, 100, 25, period, 10, seed, 123_456_789_000)


class AnalyticGenerator:
    def generate(self, history, committed, noise):
        # Distinct changing suffix, independent of every previously committed pose.
        result = np.repeat(committed[-1:], 75, axis=0)
        result[:, 7:] += noise[25:, 9:] * 0.01
        return result


@pytest.mark.parametrize("period,expected", [(0.2, [0, 10, 20, 30, 40, 50]), (0.5, [0, 25, 50])])
def test_schedule_and_absolute_committed_frames_never_change(period, expected):
    s = scheduler(period)
    committed = {}
    replan_ticks = []
    for _ in range(51):
        if s.replan_due:
            replan_ticks.append(s.tick)
            before = s.qpos[s.cursor - s.history:s.cursor + s.prefix].copy()
            times = s.timestamps_ns[s.cursor - s.history:s.cursor + s.prefix].copy()
            s.replan(AnalyticGenerator())
            np.testing.assert_array_equal(s.qpos[:35], before)
            np.testing.assert_array_equal(s.timestamps_ns[:35], times)
            assert np.all(np.diff(s.timestamps_ns) == TICK_NS)
        assert s.timestamps_ns[s.cursor] == s.epoch_ns + s.tick * TICK_NS
        # Everything exposed to HoloMotion, including its derivative stencil,
        # must remain bit-identical whenever that timestamp is seen again.
        for stamp, pose in zip(s.timestamps_ns[s.cursor:s.cursor + 12], s.qpos[s.cursor:s.cursor + 12]):
            if stamp in committed:
                np.testing.assert_array_equal(pose, committed[stamp])
            committed[stamp] = pose.copy()
        s.advance()
    assert replan_ticks == expected


def test_noise_is_reproducible_and_does_not_depend_on_other_rng_calls():
    a, b = scheduler(), scheduler()
    np.testing.assert_array_equal(a.noise(), b.noise())
    first = a.noise()
    np.random.standard_normal(1000)
    np.testing.assert_array_equal(a.noise(), first)
    a.tick = b.tick = 10
    np.testing.assert_array_equal(a.noise(), b.noise())
    assert not np.array_equal(a.noise(), first)
    assert not np.array_equal(first, scheduler(seed=2).noise())


def test_rejects_unsafe_preview_and_fractional_tick_period():
    with pytest.raises(ValueError, match="integer"):
        scheduler(0.21)
    s = scheduler()
    with pytest.raises(ValueError, match="preview"):
        SyncScheduler(s.qpos[0], 10, 100, 10, 0.2, 10, 1, 0)
    s.tick = 1
    with pytest.raises(RuntimeError, match="replan tick"):
        s.replan(AnalyticGenerator())


def test_tracker_features_of_committed_frames_do_not_change():
    from omg.tracking.holomotion.reference import precompute_reference_features

    s = scheduler()
    s.replan(AnalyticGenerator())
    before = precompute_reference_features(s.qpos, 50, np.arange(29))
    for _ in range(10):
        s.advance()
    # At tick 9, tracker saw through tick 19; tick 20 was its derivative support.
    s.replan(AnalyticGenerator())
    after = precompute_reference_features(s.qpos, 50, np.arange(29))
    for key in before:
        np.testing.assert_array_equal(before[key][20:30], after[key][10:20])


def test_deadline_failure_precedes_joint_write(monkeypatch):
    from types import SimpleNamespace

    from deployment import run_sync, sync_flow

    clock = [0]
    monkeypatch.setattr(run_sync, "perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(sync_flow, "perf_counter_ns", lambda: clock[0])

    class SlowGenerator(AnalyticGenerator):
        history, horizon = 10, 100

        def generate(self, *args):
            clock[0] += 21_000_000
            return super().generate(*args)

    class Robot:
        writes = 0

        def read_state(self):
            return SimpleNamespace(qpos=scheduler().qpos[0], timestamp_ns=0)

        def write_joint_targets(self, targets):
            self.writes += 1

    robot = Robot()
    tracker = SimpleNamespace(metadata=SimpleNamespace(context_length=1, n_fut_frames=10),
                              reference_features=lambda q: q)
    with pytest.raises(RuntimeError, match="REPLAN OVERRUN"):
        run_sync.run_control(SlowGenerator(), tracker, robot, scheduler().qpos[0], duration=0.2,
                             prefix_length=25, replan_period=0.2, seed=1234, epoch_ns=0, pace=False)
    assert robot.writes == 0


def test_capture_safe_rotation_codec_against_scipy_and_zero_gradient():
    import torch
    from scipy.spatial.transform import Rotation

    from omg.utils.rotation_conversions import axis_angle_to_quaternion, matrix_to_quaternion, quaternion_invert

    vectors = np.array([[0, 0, 0], [1e-9, 0, 0], [0, 1e-6, 0],
                        [0, 0, np.pi], [0.3, -0.5, 1.2]], dtype=np.float64)
    actual = axis_angle_to_quaternion(torch.from_numpy(vectors))
    np.testing.assert_allclose(actual.numpy(), Rotation.from_rotvec(vectors).as_quat()[:, [3, 0, 1, 2]], atol=1e-15)
    np.testing.assert_allclose(quaternion_invert(actual).numpy(), Rotation.from_rotvec(-vectors).as_quat()[:, [3, 0, 1, 2]], atol=1e-15)
    np.testing.assert_allclose(matrix_to_quaternion(torch.from_numpy(Rotation.from_rotvec(vectors).as_matrix())).numpy(), actual.numpy(), atol=1e-15)
    assert torch.autograd.gradcheck(axis_angle_to_quaternion, (torch.from_numpy(vectors).requires_grad_(),))


def test_batched_tracker_features_match_scalar_rotation_formulas():
    from scipy.spatial.transform import Rotation

    from omg.tracking.holomotion.reference import (
        body_angvel_from_quats, finite_difference, gravity_orientation_wxyz,
        precompute_reference_features, quat_rotate_inv_wxyz,
    )

    rng = np.random.default_rng(10)
    qpos = rng.standard_normal((110, 36)).astype(np.float32)
    qpos[:, 3:7] = Rotation.from_rotvec(rng.normal(0, 0.01, (110, 3)).cumsum(axis=0)).as_quat()[:, [3, 0, 1, 2]]
    features = precompute_reference_features(qpos, 50, np.arange(29))
    rotations = features["ref_root_quat_wxyz"]
    np.testing.assert_allclose(features["ref_gravity_projection"], np.stack([gravity_orientation_wxyz(q) for q in rotations]), atol=5e-7)
    np.testing.assert_allclose(features["ref_base_linvel"], np.stack([
        quat_rotate_inv_wxyz(q, v) for q, v in zip(rotations, finite_difference(qpos[:, :3], 50))
    ]), atol=3e-5, rtol=1e-6)
    np.testing.assert_allclose(features["ref_base_angvel"], body_angvel_from_quats(rotations, 50), atol=3e-6)
