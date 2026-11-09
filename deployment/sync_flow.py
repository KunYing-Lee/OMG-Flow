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


class FlowGenerator(nn.Module):
    """One fixed prompt/prefix/step count; TensorRT if present, otherwise CUDA FP16.

    CUDA graph replay removes Python launch overhead without changing Euler or
    moving generation out of the requesting control tick. CPU geometry stays FP32.
    """

    def __init__(self, config):
        super().__init__()
        if config["steps"] not in (5, 10):
            raise ValueError("steps must be 5 or 10")
        self.steps = config["steps"]
        self.prefix = config["prefix_length"]
        self.reference = FlowReferenceGenerator.from_checkpoint(config["checkpoint"], "cuda")
        self.history, self.horizon = self.reference.history, self.reference.horizon
        if self.reference.model.text_proj is None:
            raise ValueError("A text-conditioned OMG-Flow checkpoint is required")
        if not 0 < self.prefix < self.horizon:
            raise ValueError("prefix_length must lie strictly inside the Flow horizon")
        self.contract = {
            "format": "omg-flow-sync-v1", "steps": self.steps, "prefix_length": self.prefix,
            "checkpoint_sha256": hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest(),
            "prompt": config["prompt"], "history": self.history, "horizon": self.horizon,
        }
        # Encode once at startup, in FP32 as in training; no text work in the loop.
        encoder = FrozenT5TextEncoder(config["text_encoder"], max_length=50, frozen_eval=True).cuda().eval()
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
        self.engine = None
        engine_path = Path(config["engine_path"]) if config.get("engine_path") else None
        if engine_path is not None and engine_path.is_file():
            import tensorrt as trt

            if json.loads(engine_path.with_suffix(".json").read_text()) != self.contract:
                raise ValueError("TensorRT engine does not match checkpoint, prompt, steps and prefix")
            self.trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = self.trt_runtime.deserialize_cuda_engine(engine_path.read_bytes())
            self.context = self.engine.create_execution_context()
            self.output = torch.empty(1, self.horizon - self.prefix, 36, device="cuda")
            tensors = {**self.inputs, "suffix": self.output}
            if {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)} != tensors.keys():
                raise ValueError("Expected the OMG-Flow synchronous engine I/O contract")
            for name, tensor in tensors.items():
                if (tuple(self.engine.get_tensor_shape(name)) != tuple(tensor.shape)
                        or self.engine.get_tensor_dtype(name) != trt.float32):
                    raise ValueError(f"Unexpected engine binding: {name}")
                self.context.set_tensor_address(name, tensor.data_ptr())
            self.backend = "tensorrt-fp16"
        else:
            self.backend = "torch-fp16-cudagraph"
            # Fixed shapes and explicit noise make this an exact replay of the
            # same eager program. Warmup/capture never consumes scheduler noise.
            torch.cuda.synchronize()
            with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                for _ in range(3):
                    self.forward(*self.inputs.values())
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self.output = self.forward(*self.inputs.values())
            for _ in range(3):
                self.graph.replay()
            torch.cuda.synchronize()

    def forward(self, history_qpos, committed_qpos, noise):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.reference.generate_suffix(
                history_qpos, committed_qpos, noise, steps=self.steps,
                text_tokens=self.text_tokens, text_mask=self.text_mask,
            ).float()

    @torch.inference_mode()
    def generate(self, history, committed, noise):
        for tensor, value in zip(self.inputs.values(), (history, committed, noise)):
            tensor.copy_(torch.from_numpy(value)[None])
        if self.engine is None:
            self.graph.replay()
        elif not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        # This copy synchronizes the GPU before returning to the SAME control tick.
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
