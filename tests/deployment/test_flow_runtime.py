import hashlib
import json

import numpy as np
import pytest
import torch

from deployment.flow_runtime import FlowRuntime, filter_suffix
from deployment.sync_flow import SyncScheduler
from omg.data.bones import encode_bones_window
from omg.generation.flow import MotionFlow
from omg.generation.flow_reference import FlowReferenceGenerator
from omg.utils.rotation_conversions import axis_angle_to_quaternion


@pytest.fixture
def inputs():
    torch.manual_seed(20261116)
    model = MotionFlow(hidden_dim=16, num_layers=1, num_heads=2, text_dim=8, max_prefix_length=25)
    reference = FlowReferenceGenerator(model, 10, 100, torch.randn(38) * .1, torch.rand(38) + .2)
    world = torch.randn(1, 35, 36)
    world[..., 3:7] = axis_angle_to_quaternion(world[..., 3:6] * .3)
    mask = torch.tensor([[True, True, True, True, False, False, False]])
    return reference, world, torch.randn(1, 100, 38), torch.randn(1, 7, 8), mask


@pytest.mark.parametrize("prefix", [0, 25])
@pytest.mark.parametrize("steps", [5, 10])
def test_fp32_matches_original_reference_api(inputs, prefix, steps):
    ref, world, noise, tokens, mask = inputs
    runtime = FlowRuntime(dict(device="cpu", steps=steps, prefix_length=prefix, backend="torch-fp32"),
                          reference=ref, text_tokens=tokens, text_mask=mask)
    history, committed = world[:, :10], world[:, 10:10 + prefix]
    expected = ref.generate_suffix(history, committed, noise, steps=steps, text_tokens=tokens, text_mask=mask)
    actual = runtime.generate(history[0], committed[0], noise[0])
    np.testing.assert_allclose(actual, expected[0].numpy(), atol=1e-5, rtol=1e-5)
    # PPO excludes these coordinates entirely, including from validity checks.
    noise[:, :prefix] = float("nan")
    np.testing.assert_array_equal(runtime.generate(history[0], committed[0], noise[0]), actual)
    np.testing.assert_array_equal(world[:, :10], history)


def test_filter_only_suffix_and_seed_rule(inputs):
    ref, world, noise, tokens, mask = inputs
    runtime = FlowRuntime(dict(device="cpu", steps=10, prefix_length=25, backend="torch-fp32", lowpass_hz=8),
                          reference=ref, text_tokens=tokens, text_mask=mask)
    standing = np.zeros(36, dtype=np.float32)
    standing[3] = 1
    scheduler = SyncScheduler(standing, 10, 100, 25, .2, 10, 1234, 0)
    scheduler.tick = 30
    expected_noise = scheduler.noise()
    seeded = runtime.generate(world[0, :10], world[0, 10:], seed=1234, replan_index=3)
    np.testing.assert_array_equal(seeded, runtime.generate(world[0, :10], world[0, 10:], expected_noise))
    known = encode_bones_window(world, 10, ref.mean, ref.std)
    future = ref.model.sample(known[:, :10], known[:, 10:], 100, steps=10,
                              noise=torch.from_numpy(expected_noise)[None], text_tokens=tokens, text_mask=mask)
    filtered = filter_suffix(torch.cat([known[:, :10], future], dim=1).numpy(), 10, 25, 8)
    np.testing.assert_array_equal(runtime.sample(world[0, :10], world[0, 10:], expected_noise).numpy(),
                                  filtered[0].astype(np.float32))


def test_request_validation(inputs):
    ref, world, noise, tokens, mask = inputs
    runtime = FlowRuntime(dict(device="cpu", steps=10, prefix_length=25, backend="torch-fp32"),
                          reference=ref, text_tokens=tokens, text_mask=mask)
    with pytest.raises(ValueError, match="exactly one"):
        runtime.generate(world[0, :10], world[0, 10:])
    with pytest.raises(ValueError, match="exactly one"):
        runtime.generate(world[0, :10], world[0, 10:], noise[0], seed=1)
    with pytest.raises(ValueError, match="noise must have shape"):
        runtime.generate(world[0, :10], world[0, 10:], noise[0, 25:])
    noise[0, 25, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        runtime.generate(world[0, :10], world[0, 10:], noise[0])


def test_cache_load_without_t5_and_checksum(tmp_path, monkeypatch):
    from omg.generation.bones_text import BonesTextCache, MAX_TOKENS, HIDDEN_DIM, T5_REVISION, T5_SHA256
    from omg.generation.conditions.t5 import FrozenT5TextEncoder

    def forbidden(*args, **kwargs):
        pytest.fail("cache loading must not instantiate T5")

    monkeypatch.setattr(FrozenT5TextEncoder, "__init__", forbidden)
    np.save(tmp_path / "tokens.npy", np.ones((1, MAX_TOKENS, HIDDEN_DIM), dtype=np.float16))
    np.save(tmp_path / "lengths.npy", np.array([5], dtype=np.uint8))
    (tmp_path / "text.json").write_text(json.dumps(dict(
        encoder="google-t5/t5-base", revision=T5_REVISION, encoder_sha256=T5_SHA256,
        max_length=MAX_TOKENS, captions=["walk"], files_sha256={
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ("tokens.npy", "lengths.npy")})))
    ref = FlowReferenceGenerator(MotionFlow(hidden_dim=16, num_layers=1, num_heads=2, text_dim=768),
                                 10, 100, torch.zeros(38), torch.ones(38))
    config = dict(device="cpu", steps=10, prefix_length=0, backend="torch-fp32", text_cache=tmp_path, prompt="walk")
    runtime = FlowRuntime(config, reference=ref)
    tokens, mask = BonesTextCache(tmp_path, verify_files=True).lookup(["walk"])
    torch.testing.assert_close(runtime.text_tokens, tokens.float(), atol=0, rtol=0)
    assert torch.equal(runtime.text_mask, mask)
    (tmp_path / "lengths.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        FlowRuntime(config, reference=ref)
