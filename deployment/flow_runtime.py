"""Robot-independent, batch-one native50 Flow planning with externally supplied noise."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn

from omg.generation.flow_reference import FlowReferenceGenerator


def filter_suffix(window, history, prefix, cutoff_hz):
    """Posttrain 4f7ff92: float64 filtfilt on normalized xyz/rot6d/joints.

    Context participates in filtering, but only the suffix is returned. Never
    filter world quaternions or copy filtered history/prefix into the tape.
    """
    from scipy import signal

    b, a = signal.butter(4, cutoff_hz / 25.)
    return signal.filtfilt(b, a, window.astype(np.float64), axis=1,
                           padtype="odd", padlen=15)[:, history + prefix:].copy()


class FlowRuntime(nn.Module):
    """Explicit velocity backend; exact Euler math with fixed-shape CUDA replay."""

    def __init__(self, config, *, reference=None, text_tokens=None, text_mask=None):
        super().__init__()
        if not isinstance(config["steps"], int) or config["steps"] < 1:
            raise ValueError("steps must be a positive integer")
        self.steps, self.prefix = config["steps"], config["prefix_length"]
        self.lowpass_hz = config.get("lowpass_hz")
        if self.lowpass_hz is not None and not 0 < self.lowpass_hz < 25:
            raise ValueError("lowpass_hz must be null (off) or lie in (0, 25) Hz")
        backend = config["backend"]
        if backend not in ("torch-fp32", "torch-fp16", "tensorrt-fp16"):
            raise ValueError("backend must be torch-fp32, torch-fp16 or tensorrt-fp16")
        device = torch.device(config.get("device", "cuda"))
        if device.type != "cuda" and backend != "torch-fp32":
            raise ValueError("FP16 backends require CUDA")
        if device.type == "cuda":
            torch.cuda.init()
        self.reference = (FlowReferenceGenerator.from_checkpoint(config["checkpoint"], device)
                          if reference is None else reference.to(device)).eval().requires_grad_(False)
        self.history, self.horizon = self.reference.history, self.reference.horizon
        if (self.history, self.horizon) != (10, 100):
            raise ValueError("runtime requires history10 and horizon100")
        limit = self.reference.model.max_prefix_length
        if (not isinstance(self.prefix, int) or not 0 <= self.prefix < self.horizon
                or (limit is not None and self.prefix > limit)):
            raise ValueError("prefix_length must lie in the trained prefix range and below the horizon")
        self.contract = {
            "format": "omg-flow-sync-v2", "steps": self.steps, "prefix_length": self.prefix,
            "checkpoint_sha256": (hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest()
                                  if reference is None else None),
            "prompt": config.get("prompt"), "history": self.history, "horizon": self.horizon,
            "lowpass_hz": self.lowpass_hz,
        }
        if text_tokens is None and self.reference.model.text_proj is not None:
            if config.get("text_cache"):
                from omg.generation.bones_text import BonesTextCache

                cache = BonesTextCache(config["text_cache"], verify_files=True)
                text_tokens, text_mask = cache.lookup([config["prompt"]])
                self.contract["text_cache_signature"] = cache.signature
            else:
                from omg.generation.conditions.t5 import FrozenT5TextEncoder

                model_path = Path(config["text_encoder"]).expanduser().resolve(strict=True)
                encoder = FrozenT5TextEncoder(str(model_path), max_length=50, frozen_eval=True).to(device).eval()
                with torch.inference_mode():
                    text = encoder([config["prompt"]])
                text_tokens, text_mask = text["context"], text["mask"]
        if (text_tokens is None) != (text_mask is None):
            raise ValueError("text_tokens and text_mask must be supplied together")
        if text_tokens is not None:
            text_tokens = torch.as_tensor(text_tokens, device=device, dtype=torch.float32)
            text_mask = torch.as_tensor(text_mask, device=device)
            if (self.reference.model.text_proj is None or text_tokens.ndim != 3
                    or text_tokens.shape[0] != 1
                    or text_tokens.shape[-1] != self.reference.model.text_proj.in_features
                    or text_mask.shape != text_tokens.shape[:2] or text_mask.dtype != torch.bool
                    or not text_mask.any() or not torch.isfinite(text_tokens).all()):
                raise ValueError("text must be finite batch-one tokens [1,L,D] with a nonempty bool mask [1,L]")
        self.register_buffer("text_tokens", text_tokens)
        self.register_buffer("text_mask", text_mask)
        self.eval()
        self.inputs = {
            "history_qpos": torch.zeros(1, self.history, 36, device=device),
            "committed_qpos": torch.zeros(1, self.prefix, 36, device=device),
            "noise": torch.zeros(1, self.horizon, 38, device=device),
        }
        self.inputs["history_qpos"][..., 3] = 1
        self.inputs["committed_qpos"][..., 3] = 1
        self.trt = None
        if backend == "tensorrt-fp16":
            from deployment.flow_trt import FlowTRT

            with torch.inference_mode():
                self.trt = FlowTRT(self.reference, self.text_tokens, self.text_mask,
                                   self.contract["checkpoint_sha256"], config["engine_cache"])
        self.use_graph = device.type == "cuda" and config.get("cuda_graph", True)
        self.backend = backend + ("-cudagraph" if self.use_graph else "")
        self.precision = "fp16" if backend == "torch-fp16" else "fp32"
        self.filtered = torch.empty(1, self.horizon - self.prefix, 38, device=device)
        if not self.use_graph:
            return
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
        with torch.autocast(noise.device.type, dtype=torch.float16, enabled=(precision or self.precision) == "fp16"):
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
    def sample(self, history, committed, noise=None, *, seed=None, replan_index=0):
        """Return a borrowed device tensor [100-P,38] of normalized suffix features.

        Noise is full horizon [100,38]; rows :P are unused (even NaNs there
        are ignored). Seed mode preserves SyncScheduler's CPU NumPy stream,
        which deliberately differs from Posttrain's Torch seed+index stream.
        This batch-one instance owns mutable buffers and is not reentrant.
        """
        if (noise is None) == (seed is None):
            raise ValueError("provide exactly one of external noise or seed")
        if noise is None:
            noise = np.random.default_rng(np.random.SeedSequence([seed, replan_index])).standard_normal(
                (self.horizon, 38), dtype=np.float32)
        values = [torch.as_tensor(value, device=self.reference.mean.device, dtype=torch.float32)
                  for value in (history, committed, noise)]
        for (name, target), value in zip(self.inputs.items(), values):
            if value.shape != target.shape[1:]:
                raise ValueError(f"{name} must have shape {tuple(target.shape[1:])}")
            used = value[self.prefix:] if name == "noise" else value
            if not torch.isfinite(used).all():
                raise ValueError(f"{name} contains nonfinite used values")
            if name != "noise" and not torch.allclose(
                    torch.linalg.vector_norm(value[:, 3:7], dim=-1),
                    torch.ones(len(value), device=value.device), atol=1e-4, rtol=0):
                raise ValueError(f"{name} requires unit wxyz quaternions")
            target.copy_(value[None])
        if self.use_graph:
            self.graph.replay()
        else:
            self.window = self.forward(*self.inputs.values())
        if self.lowpass_hz is None:
            self.filtered.copy_(self.window[:, self.history + self.prefix:])
        else:
            suffix = filter_suffix(self.window.cpu().numpy(), self.history, self.prefix, self.lowpass_hz)
            self.filtered.copy_(torch.from_numpy(suffix))
        return self.filtered[0]

    @torch.inference_mode()
    def generate(self, history, committed, noise=None, *, seed=None, replan_index=0):
        """Return only new world qpos [100-P,36]; retain original committed poses."""
        self.sample(history, committed, noise, seed=seed, replan_index=replan_index)
        if self.use_graph:
            self.decode_graph.replay()
        else:
            self.output = self.decode(self.filtered)
        result = self.output[0].cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Flow produced a nonfinite plan")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("text-cache", help="Write a Posttrain-compatible BonesTextCache directory")
    cache.add_argument("--prompt", action="append", required=True)
    cache.add_argument("--text-encoder", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--device", default="cpu")
    for command in ("plan", "benchmark"):
        sub = commands.add_parser(command)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--input", type=Path, required=True, help="npz: history_qpos, committed_qpos, optional noise")
        sub.add_argument("--prompt", required=True)
        text = sub.add_mutually_exclusive_group(required=True)
        text.add_argument("--text-cache", type=Path)
        text.add_argument("--text-encoder", type=Path)
        sub.add_argument("--backend", choices=("torch-fp32", "torch-fp16", "tensorrt-fp16"), default="torch-fp32")
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--engine-cache", default="tensorrt_engine_cache/flow_sync")
        sub.add_argument("--steps", type=int, default=10)
        sub.add_argument("--lowpass-hz", type=float, default=None, help="e.g. 8; omitted means off")
        sub.add_argument("--seed", type=int)
        sub.add_argument("--replan-index", type=int, default=0)
        sub.add_argument("--no-cuda-graph", action="store_true")
        sub.add_argument("--output", type=Path, required=True)
        if command == "benchmark":
            sub.add_argument("--warmup", type=int, default=10)
            sub.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.command == "text-cache":
        from omg.generation.bones_text import prepare_cache

        prepare_cache(args.prompt, args.output, args.text_encoder, args.device, batch_size=1)
        return
    with np.load(args.input, allow_pickle=False) as data:
        history, committed = data["history_qpos"], data["committed_qpos"]
        noise = data["noise"] if "noise" in data else None
    runtime = FlowRuntime(dict(vars(args), prefix_length=len(committed), cuda_graph=not args.no_cuda_graph))
    request = dict(seed=args.seed, replan_index=args.replan_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        qpos = runtime.generate(history, committed, noise, **request)
        np.savez(args.output, suffix_qpos=qpos, committed_qpos=committed,
                 contract_json=json.dumps(runtime.contract))
        return
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations positive")
    result = {"batch": 1, "backend": runtime.backend, "contract": runtime.contract,
              "warmup": args.warmup, "iterations": args.iterations,
              "timing": "wall clock, synchronized; includes validation, copies and optional filter; excludes construction"}
    for name, call in (("without_decode", runtime.sample), ("with_decode", runtime.generate)):
        elapsed = []
        for index in range(args.warmup + args.iterations):
            if runtime.reference.mean.is_cuda:
                torch.cuda.synchronize()
            started = perf_counter_ns()
            call(history, committed, noise, **request)
            if runtime.reference.mean.is_cuda:
                torch.cuda.synchronize()
            if index >= args.warmup:
                elapsed.append((perf_counter_ns() - started) / 1e6)
        result[name] = dict(zip(("p50_ms", "p95_ms", "max_ms"),
                               map(float, np.percentile(elapsed, [50, 95, 100]))))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
