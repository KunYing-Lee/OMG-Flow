"""Inference-only sampled Flow noise policy.

Checkpoint schema (loaded with weights_only=True): a mapping containing `actor`,
whose state dict has exactly these float32 tensors:
    log_std: (3800,)
    backbone.0.weight: (512, 1331), backbone.0.bias: (512,)
    backbone.2.weight: (512, 512), backbone.2.bias: (512,)
    mean_head.weight: (3800, 512), mean_head.bias: (3800,)
Optional `format_version` must be integer 1. Other top-level metadata is unused.
No optimizer, critic, or trained weights are needed by this module.

NoiseProvider.noise accepts float32 qpos [10,36] or [35,36] and returns a
float32 Torch tensor [100,38]. The first P rows are exactly zero. Used rows
are always sampled as mean + exp(log_std) * epsilon, never mean-only.
"""
from collections.abc import Mapping
import hashlib
import pickle
import struct
from typing import Protocol

import numpy as np
import torch
from torch import nn

from .wire import motion


class NoiseProvider(Protocol):
    def noise(self, context: np.ndarray, *, request_id: int) -> torch.Tensor: ...


class FlowNoiseActor(nn.Module):
    """Fixed 1331 -> 512 -> 512 -> 3800 mean with state-independent variance."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(1331, 512, dtype=torch.float32), nn.SiLU(),
            nn.Linear(512, 512, dtype=torch.float32), nn.SiLU())
        self.mean_head = nn.Linear(512, 3800, dtype=torch.float32)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.zeros(3800, dtype=torch.float32))

    def forward(self, observation):
        return self.mean_head(self.backbone(observation))


@torch.inference_mode()
def encode_observation(context, mean, std):
    """Return [1,1331] normalized anchor-relative context and the P/25 flag."""
    from omg.data.bones import encode_bones_window

    if context.shape not in ((10, 36), (35, 36)):
        raise ValueError("context must have shape (10,36) or (35,36)")
    motion(context, len(context))
    features = encode_bones_window(
        torch.as_tensor(context, device=mean.device, dtype=torch.float32)[None], 10, mean, std)
    padded = torch.zeros(1, 35, 38, device=features.device, dtype=torch.float32)
    padded[:, :features.shape[1]] = features
    flag = torch.full((1, 1), (len(context) - 10) / 25, device=features.device, dtype=torch.float32)
    return torch.cat([padded.flatten(1), flag], dim=1)


def paired_epsilon(seed, episode_id, request_id, *, device="cpu", dtype=torch.float32):
    """Full [100,38] draw keyed by (seed, episode, request, noise stream 0)."""
    for name, value in (("seed", seed), ("episode_id", episode_id), ("request_id", request_id)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**63:
            raise ValueError(f"{name} must be an integer in [0, 2**63)")
    key = struct.pack("<QQQQ", int(seed), int(episode_id), int(request_id), 0)
    digest = hashlib.blake2b(key, digest_size=8, person=b"omg-noise-eval1").digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest, "little") & (2**63 - 1))
    return torch.randn(3800, generator=generator, dtype=torch.float32).reshape(100, 38).to(device=device, dtype=dtype)


class SampledNoisePolicy:
    def __init__(self, actor, mean, std, *, seed, episode_id):
        self.actor, self.mean, self.std = actor, mean, std
        self.seed, self.episode_id = seed, episode_id

    @torch.inference_mode()
    def noise(self, context, *, request_id):
        observation = encode_observation(context, self.mean, self.std)
        mean = self.actor(observation).reshape(100, 38)
        eps = paired_epsilon(self.seed, self.episode_id, request_id, device=mean.device)
        sample = mean + self.actor.log_std.exp().reshape(100, 38) * eps
        sample[:len(context) - 10] = 0.
        return sample


def load_noise_policy(path, *, mean, std, seed, episode_id, device="cpu") -> NoiseProvider:
    """Validate the checkpoint schema before strict loading; return a sampled policy."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError(f"Cannot load noise policy: {error}") from error
    if not isinstance(checkpoint, Mapping) or "actor" not in checkpoint:
        raise ValueError("noise policy must be a mapping containing actor")
    if "format_version" in checkpoint and (type(checkpoint["format_version"]) is not int or checkpoint["format_version"] != 1):
        raise ValueError("Unsupported noise policy format_version; expected integer 1")
    state = checkpoint["actor"]
    shapes = {
        "log_std": (3800,), "backbone.0.weight": (512, 1331), "backbone.0.bias": (512,),
        "backbone.2.weight": (512, 512), "backbone.2.bias": (512,),
        "mean_head.weight": (3800, 512), "mean_head.bias": (3800,),
    }
    if not isinstance(state, Mapping):
        raise ValueError("actor must be a state dict mapping")
    missing, extra = shapes.keys() - state.keys(), state.keys() - shapes.keys()
    if missing or extra:
        raise ValueError(f"actor keys: missing={sorted(missing)}, extra={sorted(map(str, extra))}")
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"actor.{name} must be a tensor with shape {shape}")
        if value.dtype != torch.float32:
            raise ValueError(f"actor.{name} must have float32 dtype, got {value.dtype}")
        if not torch.isfinite(value).all():
            raise ValueError(f"actor.{name} must be finite")
    mean, std = (torch.as_tensor(value, device=device, dtype=torch.float32) for value in (mean, std))
    if mean.shape != (38,) or std.shape != (38,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("mean and std must be finite (38,) tensors with positive std")
    actor = FlowNoiseActor()
    actor.load_state_dict(state, strict=True)
    actor = actor.to(device).eval().requires_grad_(False)
    return SampledNoisePolicy(actor, mean, std, seed=seed, episode_id=episode_id)
