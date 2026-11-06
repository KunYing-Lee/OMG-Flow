import pytest
import torch

from omg.data.bones import encode_bones_window
from omg.generation.flow import MotionFlow
from omg.generation.flow_reference import FlowReferenceGenerator
from omg.utils.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_matrix,
)


def reference_from_disk(tmp_path, *, zero_velocity=False):
    config = dict(feature_dim=38, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.2)
    model = MotionFlow(**config)
    if zero_velocity:
        with torch.no_grad():
            model.output[-1].weight.zero_()
            model.output[-1].bias.zero_()
    checkpoint = tmp_path / "reference.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": config,
        "config": {"history": 10, "horizon": 100},
        "mean": torch.randn(38) * 0.1,
        "std": torch.rand(38) + 0.2,
    }, checkpoint)
    return FlowReferenceGenerator.from_checkpoint(checkpoint)


def world_motion(batch, length):
    qpos = torch.randn(batch, length, 36)
    qpos[:, :, 3:7] = axis_angle_to_quaternion(qpos[:, :, 3:6] * 0.35)
    return qpos


def test_native38_checkpoint_world_round_trip_with_25_committed_frames(tmp_path):
    generator = reference_from_disk(tmp_path, zero_velocity=True)
    world = world_motion(2, 110)
    # Analytic oracle: zero vector field integrates the supplied starting state unchanged.
    normalized = encode_bones_window(world, 10, generator.mean, generator.std)
    noise = normalized[:, 10:].clone()
    noise[:, :25] = float("nan")
    suffix = generator.generate_suffix(world[:, :10], world[:, 10:35], noise, steps=2)
    torch.testing.assert_close(suffix, world[:, 35:], atol=2e-5, rtol=2e-5)
    assert generator.generate_suffix(world[:, :10], world[:, 10:], noise, steps=2).shape == (2, 0, 36)


def test_second_plan_keeps_raw_committed_reference_and_rebases_at_new_history(tmp_path):
    torch.manual_seed(25)
    generator = reference_from_disk(tmp_path)
    history = world_motion(2, 10)
    noise = torch.randn(2, 100, 38)
    first = generator.generate_suffix(history, history[:, :0], noise, steps=2)
    saved = first.clone()
    # Advance 25 frames; the tracker has already been shown frames 25..49.
    history, committed = first[:, 15:25], first[:, 25:50]
    noise = torch.randn_like(noise)
    second_suffix = generator.generate_suffix(history, committed, noise, steps=2)
    assert second_suffix.shape == (2, 75, 36)
    assert torch.equal(first, saved)
    assert torch.equal(torch.cat([committed, second_suffix], dim=1)[:, :25], saved[:, 25:50])
    assert torch.equal(generator.generate_suffix(history, committed, noise, steps=2), second_suffix)

    # A world translation and yaw applied to *both* known regions must produce
    # the same transformed second plan with identical normalized noise.
    rotation = axis_angle_to_quaternion(torch.tensor([[[0.0, 0.0, 0.8]], [[0.0, 0.0, -1.2]]]))
    translation = torch.tensor([[[2.0, -1.5, 0.1]], [[-4.0, 2.5, 1.4]]])

    def transform(qpos):
        return torch.cat([
            quaternion_apply(rotation, qpos[..., :3]) + translation,
            quaternion_multiply(rotation, qpos[..., 3:7]),
            qpos[..., 7:],
        ], dim=-1)

    transformed = generator.generate_suffix(transform(history), transform(committed), noise, steps=2)
    expected = transform(second_suffix)
    torch.testing.assert_close(transformed[..., :3], expected[..., :3], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        quaternion_to_matrix(transformed[..., 3:7]), quaternion_to_matrix(expected[..., 3:7]),
        atol=2e-5, rtol=2e-5,
    )
    torch.testing.assert_close(transformed[..., 7:], expected[..., 7:], atol=2e-5, rtol=2e-5)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        generator.generate_suffix(history, committed, noise, future_qpos=saved)


def test_replan_seam_probe_compares_matching_times_and_excludes_both_splice_stencils():
    from omg.generation.flow_seam import replan_seam_metrics

    # Quadratic joint motion has 1 rad/s² acceleration; the second plan's
    # 0.02 rad splice adds 50 rad/s² and then subtracts it on the next stencil.
    t = torch.arange(-10, 125, dtype=torch.float32)
    motion = torch.zeros(2, 135, 36)
    motion[..., 3] = 1.0
    motion[..., 7:] = (t.square() / (2 * 50**2))[:, None]
    first = motion[:, 10:110].clone()
    noise_a, noise_b = torch.randn(2, 2, 100, 38)

    class AnalyticGenerator:
        history = 10
        horizon = 100

        def generate_suffix(self, history, committed, noise, **conditions):
            if committed.size(1) == 0:
                assert torch.equal(history, motion[:, :10]) and noise is noise_a
                return first
            assert torch.equal(history, first[:, 15:25]) and torch.equal(committed, first[:, 25:50])
            assert noise is noise_b
            suffix = motion[:, 60:].clone()
            suffix[..., 7:] += 0.02
            return suffix

    metrics = replan_seam_metrics(AnalyticGenerator(), motion[:, :110], noise_a, noise_b,
                                  steps=2, text_tokens=torch.empty(2, 0, 8), text_mask=torch.empty(2, 0))
    for name, value in metrics.items():
        expected = 51.0 if name == "replan_seam_joint_accel_abs_rad_s2" else 1.0
        torch.testing.assert_close(value, torch.tensor(expected), atol=3e-3, rtol=0)
