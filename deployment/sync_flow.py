"""OMG-Flow only: synchronous Euler generation and an absolute-time reference buffer."""

import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn

from omg.generation.conditions.t5 import FrozenT5TextEncoder
from omg.generation.flow_reference import FlowReferenceGenerator

TICK_NS = 20_000_000


def filter_suffix(window, history, prefix, cutoff_hz):
    """Posttrain 4f7ff92: float64 filtfilt on normalized xyz/rot6d/joints.

    Context participates in filtering, but only the suffix is returned. Never
    filter world quaternions or copy filtered history/prefix into the tape.
    """
    from scipy import signal

    b, a = signal.butter(4, cutoff_hz / 25.)
    return signal.filtfilt(b, a, window.astype(np.float64), axis=1,
                           padtype="odd", padlen=15)[:, history + prefix:].copy()


class FlowGenerator(nn.Module):
    """Explicit velocity backend; exact Euler math with fixed-shape CUDA replay."""

    def __init__(self, config):
        super().__init__()
        if config["steps"] not in (5, 10):
            raise ValueError("steps must be 5 or 10")
        self.steps, self.prefix = config["steps"], config["prefix_length"]
        self.lowpass_hz = config.get("lowpass_hz")
        if self.lowpass_hz is not None and not 0 < self.lowpass_hz < 25:
            raise ValueError("lowpass_hz must be null (off) or lie in (0, 25) Hz")
        backend = config["backend"]
        if backend not in ("torch-fp32", "torch-fp16", "tensorrt-fp16"):
            raise ValueError("backend must be torch-fp32, torch-fp16 or tensorrt-fp16")
        self.reference = FlowReferenceGenerator.from_checkpoint(config["checkpoint"], "cuda").requires_grad_(False)
        self.history, self.horizon = self.reference.history, self.reference.horizon
        if self.reference.model.text_proj is None:
            raise ValueError("A text-conditioned OMG-Flow checkpoint is required")
        if not 0 < self.prefix < self.horizon:
            raise ValueError("prefix_length must lie strictly inside the Flow horizon")
        self.contract = {
            "format": "omg-flow-sync-v2", "steps": self.steps, "prefix_length": self.prefix,
            "checkpoint_sha256": hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest(),
            "prompt": config["prompt"], "history": self.history, "horizon": self.horizon,
            "lowpass_hz": self.lowpass_hz,
        }
        encoder = FrozenT5TextEncoder(config["text_encoder"], max_length=50, frozen_eval=True).cuda().eval()
        with torch.inference_mode():
            text = encoder([config["prompt"]])
        self.register_buffer("text_tokens", text["context"])
        self.register_buffer("text_mask", text["mask"])
        self.eval()
        self.inputs = {
            "history_qpos": torch.zeros(1, self.history, 36, device="cuda"),
            "committed_qpos": torch.zeros(1, self.prefix, 36, device="cuda"),
            "noise": torch.zeros(1, self.horizon, 38, device="cuda"),
        }
        self.inputs["history_qpos"][..., 3] = 1
        self.inputs["committed_qpos"][..., 3] = 1
        self.trt = None
        if backend == "tensorrt-fp16":
            from deployment.flow_trt import FlowTRT

            with torch.inference_mode():
                self.trt = FlowTRT(self.reference, self.text_tokens, self.text_mask,
                                   self.contract["checkpoint_sha256"], config["engine_cache"])
        self.backend = backend + "-cudagraph"
        self.precision = "fp16" if backend == "torch-fp16" else "fp32"
        self.filtered = torch.empty(1, self.horizon - self.prefix, 38, device="cuda")
        # CPU filtering separates sampling and decoding into two exact graphs.
        # Warmup/capture does not consume scheduler noise or a control tick.
        torch.cuda.synchronize()
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(3):
                window = self.forward(*self.inputs.values())
                self.filtered.copy_(window[:, self.history + self.prefix:])
                self.decode(self.filtered)
        torch.cuda.synchronize()
        self.graph, self.decode_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.window = self.forward(*self.inputs.values())
        with torch.inference_mode(), torch.cuda.graph(self.decode_graph):
            self.output = self.decode(self.filtered)
        for _ in range(3):
            self.generate(*(value[0].cpu().numpy() for value in self.inputs.values()))
        torch.cuda.synchronize()

    def forward(self, history_qpos, committed_qpos, noise, *, precision=None, use_trt=True):
        from omg.data.bones import encode_bones_window

        ref = self.reference
        known = encode_bones_window(torch.cat([history_qpos, committed_qpos], dim=1),
                                    self.history, ref.mean, ref.std)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(precision or self.precision) == "fp16"):
            context, context_mask = ref.model._context(known[:, :self.history], self.text_tokens, self.text_mask)
            committed = known[:, self.history:]
            committed_mask = (torch.arange(self.horizon, device=noise.device) < self.prefix)[None]
            suffix = noise[:, self.prefix:]
            velocity = self.trt if use_trt and self.trt is not None else ref.model._velocity
            for step in range(self.steps):
                state = torch.cat([committed, suffix], dim=1)
                time = torch.full((1,), step / self.steps, device=noise.device, dtype=torch.float32)
                v = velocity(state, time, committed_mask, context, context_mask)
                suffix = suffix + v[:, self.prefix:].to(suffix.dtype) / self.steps
        return torch.cat([known, suffix], dim=1)

    def decode(self, suffix):
        import torch.nn.functional as F
        from omg.motion.feature_codec import heading_quaternions
        from omg.utils.rotation_conversions import (
            matrix_to_quaternion, quaternion_apply, quaternion_multiply, rotation_6d_to_matrix,
        )

        features = suffix * self.reference.std + self.reference.mean
        anchor = self.inputs["history_qpos"][:, -1:]
        heading, _ = heading_quaternions(anchor[..., 3:7])
        orientation = F.normalize(matrix_to_quaternion(rotation_6d_to_matrix(features[..., 3:9])), dim=-1)
        return torch.cat([quaternion_apply(heading, features[..., :3]) + anchor[..., :3],
                          F.normalize(quaternion_multiply(heading, orientation), dim=-1), features[..., 9:]], dim=-1)

    @torch.inference_mode()
    def generate(self, history, committed, noise):
        for tensor, value in zip(self.inputs.values(), (history, committed, noise)):
            tensor.copy_(torch.from_numpy(value)[None])
        self.graph.replay()
        if self.lowpass_hz is None:
            self.filtered.copy_(self.window[:, self.history + self.prefix:])
        else:
            suffix = filter_suffix(self.window.cpu().numpy(), self.history, self.prefix, self.lowpass_hz)
            self.filtered.copy_(torch.from_numpy(suffix))
        self.decode_graph.replay()
        return self.output[0].cpu().numpy().copy()


class SyncScheduler:
    """Keep original world poses and timestamps; replace only the unexposed suffix.

    At tick k, history is [k-H,k), committed is [k,k+P), and generated
    suffix is [k+P,k+T). No interpolation, blending, or timestamp rebasing.
    """

    def __init__(self, standing, history, horizon, prefix_length, replan_period, preview_frames, seed, epoch_ns):
        period_ticks = round(replan_period * 50)
        if period_ticks < 1 or not np.isclose(period_ticks / 50, replan_period, rtol=0, atol=1e-10):
            raise ValueError("replan_period must be a positive integer number of 50 Hz ticks")
        # Preview features use a one-frame forward velocity stencil.
        if not preview_frames + 1 <= prefix_length < horizon - period_ticks:
            raise ValueError("prefix must cover preview + velocity stencil and leave a full replan interval")
        standing = np.asarray(standing, dtype=np.float32)
        if standing.shape != (36,) or not np.isfinite(standing).all() or not np.isclose(np.linalg.norm(standing[3:7]), 1):
            raise ValueError("standing must be finite qpos36 with a unit wxyz quaternion")
        self.history, self.horizon, self.prefix = history, horizon, prefix_length
        self.period_ticks, self.preview_frames = period_ticks, preview_frames
        self.seed, self.epoch_ns = seed, epoch_ns
        self.tick = self.plan_tick = 0
        self.exposed_until = -1
        self.qpos = np.repeat(standing[None], history + horizon, axis=0)
        self.timestamps_ns = epoch_ns + np.arange(-history, horizon, dtype=np.int64) * TICK_NS

    @property
    def cursor(self):
        return self.history + self.tick - self.plan_tick

    @property
    def replan_due(self):
        return self.tick % self.period_ticks == 0

    def noise(self):
        # Indexed CPU noise is reproducible independently of warmup, backend and
        # global RNG state. One stream per replan, never frozen across replans.
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.tick // self.period_ticks])).standard_normal(
            (self.horizon, 38), dtype=np.float32,
        )

    def replan(self, generator):
        if not self.replan_due:
            raise RuntimeError("Generation is only allowed on a replan tick")
        if self.tick + self.prefix <= self.exposed_until:
            raise RuntimeError("Replan would overwrite a tracker-visible frame")
        started = perf_counter_ns()
        cursor = self.cursor
        known = self.qpos[cursor - self.history:cursor + self.prefix].copy()
        timestamps = self.timestamps_ns[cursor - self.history:cursor + self.prefix].copy()
        suffix = generator.generate(known[:self.history], known[self.history:], self.noise())
        if suffix.shape != (self.horizon - self.prefix, 36) or not np.isfinite(suffix).all():
            raise ValueError("Flow returned an invalid suffix")
        self.qpos = np.concatenate([known, suffix])
        self.timestamps_ns = np.concatenate([
            timestamps, self.epoch_ns + np.arange(self.tick + self.prefix, self.tick + self.horizon, dtype=np.int64) * TICK_NS,
        ])
        self.plan_tick = self.tick
        return (perf_counter_ns() - started) / 1e6

    def advance(self):
        self.exposed_until = self.tick + self.preview_frames + 1
        self.tick += 1
