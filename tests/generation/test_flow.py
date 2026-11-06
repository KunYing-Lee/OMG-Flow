import pytest
import torch

from omg.generation.flow import MotionFlow


def tiny_flow(**kwargs):
    return MotionFlow(feature_dim=4, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.0, **kwargs)


def test_linear_flow_path_and_suffix_only_loss(monkeypatch):
    flow = tiny_flow()
    history = torch.randn(3, 2, 4)
    future, noise = torch.randn(2, 3, 4, 4)
    time = torch.tensor([0.0, 0.25, 1.0])
    lengths = torch.tensor([0, 2, 3])
    committed = torch.arange(4) < lengths[:, None]

    def velocity(state, model_time, mask, context, context_mask):
        torch.testing.assert_close(model_time, time, rtol=0, atol=0)
        assert torch.equal(mask, committed)
        assert torch.equal(context_mask, torch.ones(3, 2, dtype=torch.bool))
        assert torch.equal(state[mask], future[mask])
        interpolated = time[:, None, None] * future + (1 - time[:, None, None]) * noise
        torch.testing.assert_close(state[~mask], interpolated[~mask])
        # Deliberately wrong on committed frames; exact residuals 1, 2 and 3 elsewhere.
        return torch.where(mask[..., None], 1.0e6, future - noise + torch.arange(1, 4)[:, None, None])

    monkeypatch.setattr(flow, "_velocity", velocity)
    loss = flow(history, future, prefix_lengths=lengths, time=time, noise=noise)
    torch.testing.assert_close(loss, torch.tensor((1 + 4 + 9) / 3))


def test_random_prefix_is_sampled_independently_per_window(monkeypatch):
    torch.manual_seed(7)
    flow = tiny_flow(max_prefix_length=2)
    future = torch.randn(128, 4, 4)

    def velocity(state, time, mask, context, context_mask):
        assert set(mask.sum(1).tolist()) == {0, 1, 2}
        assert torch.equal(mask, torch.arange(4) < mask.sum(1)[:, None])
        assert torch.equal(state[mask], future[mask])
        assert time.unique().numel() == time.numel()
        return torch.zeros_like(state)

    monkeypatch.setattr(flow, "_velocity", velocity)
    flow(torch.zeros(128, 2, 4), future)


def test_prefix_marker_and_clean_timestep_reach_transformer():
    flow = tiny_flow()
    future = torch.zeros(2, 4, 4)
    lengths = torch.tensor([2, 0])
    times = torch.tensor([0.375, 0.125])
    projected = []
    hook = flow.input_proj.register_forward_pre_hook(lambda _, args: projected.append(args[0].detach()))
    flow(future[:, :2], future, noise=future, time=times, prefix_lengths=lengths)
    hook.remove()
    tokens = projected[0]
    mask = torch.arange(4) < lengths[:, None]
    assert torch.equal(tokens[:, :, -1].bool(), mask)
    expected = torch.where(
        mask[..., None], flow.time_embedding(torch.tensor([1000.0])),
        flow.time_embedding(1000.0 * times)[:, None],
    )
    torch.testing.assert_close(tokens[:, :, 4:-1], expected)


@pytest.mark.parametrize("prefix", [0, 2, 3])
def test_euler_integrates_only_uncommitted_coordinates_at_every_step(prefix, monkeypatch):
    history = torch.randn(2, 2, 4)
    committed = torch.randn(2, prefix, 4)
    noise = torch.randn(2, 4, 4)
    noise[:, :prefix] = float("nan")
    flow = tiny_flow().eval()
    expected = noise[:, prefix:].clone()
    calls = []

    def velocity(state, time, mask, context, context_mask):
        nonlocal expected
        t = len(calls) / 4.0
        calls.append(t)
        assert torch.equal(mask, (torch.arange(4) < prefix).expand(2, -1))
        assert torch.equal(state[:, :prefix], committed)
        torch.testing.assert_close(state[:, prefix:], expected)
        torch.testing.assert_close(time, torch.full((2,), t))
        v = state + t
        expected = expected + (expected + t) / 4
        return v.masked_fill(mask[..., None], float("nan"))

    monkeypatch.setattr(flow, "_velocity", velocity)
    plan = flow.sample(history, committed, horizon=4, steps=4, noise=noise)
    assert len(calls) == 4
    assert torch.equal(committed, plan[:, :prefix])
    torch.testing.assert_close(plan[:, prefix:], expected)


def test_actual_backbone_text_mask_determinism_and_training_gradients():
    torch.manual_seed(9)
    flow = tiny_flow(text_dim=8).eval()
    history = torch.randn(2, 2, 4)
    committed = torch.randn(2, 1, 4)
    noise = torch.randn(2, 4, 4)
    text = torch.randn(2, 2, 8)
    mask = torch.tensor([[True, False], [False, False]])
    args = dict(horizon=4, steps=2, noise=noise, text_tokens=text, text_mask=mask)
    plan = flow.sample(history, committed, **args)
    assert torch.equal(plan[:, :1], committed)
    assert torch.equal(plan, flow.sample(history, committed, **args))
    text[~mask] += 100
    noise[:, :1] += 100
    assert torch.equal(plan, flow.sample(history, committed, **args))
    assert not torch.allclose(plan[0, 1:], flow.sample(history.flip(1), committed, **args)[0, 1:])
    assert not torch.allclose(plan[0, 1:], flow.sample(history, committed + 1, **args)[0, 1:])
    text[mask] += 1
    assert not torch.allclose(plan[0, 1:], flow.sample(history, committed, **args)[0, 1:])
    # All trainable parameters participate, including during a caption-free batch.
    flow(history, plan.detach()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())


def test_reject_untrainable_prefix_and_incorrect_noise_dimensions():
    flow = tiny_flow()
    window = torch.zeros(2, 4, 4)
    with pytest.raises(ValueError, match="prefix_lengths"):
        flow(window, window, prefix_lengths=torch.tensor([1, 4]))
    with pytest.raises(ValueError, match="sampling noise must have shape"):
        flow.sample(window, window[:, :2], 4, noise=window[:, 2:])
    assert torch.equal(flow.sample(window, window, 4), window)
