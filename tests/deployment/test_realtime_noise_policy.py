from pathlib import Path

import numpy as np
import pytest
import torch

from deployment.realtime.noise_policy import (
    FlowNoiseActor, encode_observation, paired_epsilon, load_noise_policy,
)


@pytest.fixture(scope="module")
def synthetic_actor():
    actor = FlowNoiseActor().eval().requires_grad_(False)
    with torch.no_grad():
        for index, (name, parameter) in enumerate(actor.named_parameters()):
            flat = torch.arange(parameter.numel(), dtype=torch.float64)
            values = 0.015 * torch.sin(flat * 0.013 + index) + 0.005 * torch.cos(flat * 0.007 + index)
            parameter.copy_((values - (0.2 if name == "log_std" else 0.0)).float().reshape(parameter.shape))
    return actor


def test_reference_numerical_parity(tmp_path, synthetic_actor):
    path = tmp_path / "policy.pt"
    torch.save({"actor": synthetic_actor.state_dict(), "format_version": 1}, path)
    with np.load(Path(__file__).parent / "fixtures/noise_policy_parity.npz", allow_pickle=False) as data:
        mean, std = torch.from_numpy(data["mean"]), torch.from_numpy(data["std"])
        for index, key in enumerate(("context10", "context35")):
            context = data[key]
            observation = encode_observation(context, mean, std)
            assert observation.numpy().tobytes() == data["observations"][index:index+1].tobytes()
            with torch.inference_mode():
                actual_mean = synthetic_actor(observation)[0].numpy()
            np.testing.assert_allclose(actual_mean, data["actor_means"][index], rtol=1e-6, atol=1e-7)
            seed, episode, request = map(int, data["keys"][index])
            policy = load_noise_policy(path, mean=mean, std=std, seed=seed, episode_id=episode)
            sample = policy.noise(context, request_id=request)
            assert sample.shape == (100, 38) and sample.dtype == torch.float32
            assert not sample.requires_grad
            np.testing.assert_allclose(sample.numpy(), data["samples"][index], rtol=1e-6, atol=1e-7)
            assert sample.numpy().tobytes() == policy.noise(context, request_id=request).numpy().tobytes()
            assert torch.count_nonzero(sample[:len(context)-10]) == 0
            assert not torch.equal(sample, torch.from_numpy(actual_mean.copy()).reshape(100, 38))
            assert not torch.equal(sample, policy.noise(context, request_id=request+1))
        for index, (seed, episode, request) in enumerate(data["keys"]):
            epsilon = paired_epsilon(seed, episode, request).flatten()
            edges = torch.cat([epsilon[:64], epsilon[-64:]]).numpy()
            assert edges.tobytes() == data["epsilon_edges"][index].tobytes()
            assert epsilon.double().sum().item() == data["epsilon_sums"][index]


@pytest.mark.parametrize("bad,match", [
    ("missing_actor", "containing actor"), ("not_mapping", "containing actor"),
    ("actor_not_mapping", "state dict"), ("missing_key", "missing=.*mean_head.bias"),
    ("extra_key", "extra=.*extra"), ("wrong_shape", "mean_head.bias.*shape"),
    ("not_tensor", "mean_head.bias.*tensor"), ("wrong_dtype", "mean_head.bias.*float32"),
    ("nonfinite", "mean_head.bias.*finite"),
    ("unsupported_version", "format_version"), ("bool_version", "format_version"),
    ("float_version", "format_version"), ("corrupt_file", "Cannot load"),
])
def test_checkpoint_schema_errors(tmp_path, synthetic_actor, bad, match):
    state = dict(synthetic_actor.state_dict())
    checkpoint = {"actor": state}
    if bad == "missing_actor":
        checkpoint = {}
    elif bad == "not_mapping":
        checkpoint = []
    elif bad == "actor_not_mapping":
        checkpoint["actor"] = []
    elif bad == "missing_key":
        del state["mean_head.bias"]
    elif bad == "extra_key":
        state["extra"] = torch.zeros(1)
    elif bad == "wrong_shape":
        state["mean_head.bias"] = torch.zeros(1)
    elif bad == "not_tensor":
        state["mean_head.bias"] = [0]
    elif bad == "wrong_dtype":
        state["mean_head.bias"] = state["mean_head.bias"].double()
    elif bad == "nonfinite":
        state["mean_head.bias"] = torch.full((3800,), float("nan"))
    elif bad.endswith("version"):
        checkpoint["format_version"] = {"unsupported_version": 2, "bool_version": True, "float_version": 1.0}[bad]
    path = tmp_path / "policy.pt"
    if bad == "corrupt_file":
        path.write_bytes(b"invalid")
    else:
        torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=match):
        load_noise_policy(path, mean=torch.zeros(38), std=torch.ones(38), seed=7, episode_id=3)


def test_unversioned_checkpoint_and_weights_only(tmp_path, synthetic_actor, monkeypatch):
    path = tmp_path / "policy.pt"
    torch.save({"actor": synthetic_actor.state_dict()}, path)
    original = torch.load
    def checked_load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", checked_load)
    policy = load_noise_policy(path, mean=np.zeros(38), std=np.ones(38), seed=1, episode_id=0)
    assert policy.actor.state_dict().keys() == synthetic_actor.state_dict().keys()
    assert all(not parameter.requires_grad for parameter in policy.actor.parameters())


@pytest.mark.parametrize("key,value", [("seed", -1), ("seed", True), ("episode_id", 2**63),
                                       ("request_id", 1.2), ("request_id", np.bool_(True))])
def test_epsilon_rejects_invalid_keys(key, value):
    with pytest.raises(ValueError, match=key):
        paired_epsilon(**{**dict(seed=0, episode_id=0, request_id=0), key: value})


def test_epsilon_does_not_consume_global_rng():
    state = torch.get_rng_state().clone()
    first = paired_epsilon(1, 2, 3)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(first, paired_epsilon(1, 2, 3))
    assert not torch.equal(first, paired_epsilon(1, 3, 3))
    assert not torch.equal(first, paired_epsilon(2, 2, 3))


@pytest.mark.parametrize("shape", [(11, 36), (35, 38), (1, 10, 36)])
def test_bad_observation_shape(shape):
    with pytest.raises(ValueError, match="context"):
        encode_observation(np.zeros(shape, np.float32), torch.zeros(38), torch.ones(38))
