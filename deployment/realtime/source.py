"""Flow-backed timeline. Importing this module requires only NumPy."""
import hashlib
from pathlib import Path

import numpy as np

from .config import Config
from .timeline import Sync50Timeline


class FlowSource(Sync50Timeline):
    def __init__(self, config: Config):
        import torch
        from deployment.flow_runtime import FlowRuntime

        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.config = config
        self.caption = config.prompt
        with np.load(config.start_history.path, allow_pickle=False) as data:
            if config.start_history.key not in data:
                raise ValueError(f"start_history key {config.start_history.key!r} is missing")
            history = data[config.start_history.key]
        # Validate history and schedule before loading the two runtime instances.
        super().__init__(history, self.generate, play_frames=config.play_frames,
                         period=round(config.switch_seconds * 50))
        self.validate_wire_schedule()
        cfg = dict(checkpoint=config.checkpoint, prompt=config.prompt, steps=config.steps,
                   lowpass_hz=config.lowpass_hz, text_cache=config.text_cache,
                   text_encoder=config.text_encoder, backend=config.backend, device=config.device,
                   engine_cache=config.engine_directory, cuda_graph=config.device.startswith("cuda"))
        self.initial = FlowRuntime(dict(cfg, prefix_length=0))
        self.replan = FlowRuntime(dict(cfg, prefix_length=25))
        self.noise_provider = None
        self.policy_sha256 = None
        if config.noise_policy is not None:
            from .noise_policy import load_noise_policy

            self.noise_provider = load_noise_policy(config.noise_policy,
                mean=self.replan.reference.mean, std=self.replan.reference.std,
                seed=config.seed, episode_id=config.episode_id, device=config.device)
            self.policy_sha256 = hashlib.sha256(config.noise_policy.read_bytes()).hexdigest()

    def generate(self, context, request_id):
        runtime = self.initial if request_id == 0 else self.replan
        if self.noise_provider is not None:
            return runtime.generate(context[:10], context[10:],
                                    self.noise_provider.noise(context, request_id=request_id))
        return runtime.generate(context[:10], context[10:], seed=self.config.seed, replan_index=request_id)

    def metadata(self):
        return dict(**self.transport_metadata(), source="flow-noise" if self.noise_provider is not None else "flow",
                    caption=self.caption, backend=self.config.backend, seed=self.config.seed,
                    episode_id=self.config.episode_id, play_frames=self.play_frames, preview_frames=11,
                    period_frames=self.period, replan_terminal_preview=True,
                    internal_replans=getattr(self, "internal_replans", 0), generated_end=getattr(self, "valid_end", 0),
                    runtime_contract=self.initial.contract, policy_sha256=self.policy_sha256, package=None,
                    engine=None if self.initial.trt is None else dict(
                        path=str(self.initial.trt.path), cache_hit=self.initial.trt.cache_hit))

    def export(self, q, output):
        if self.config.robot_xml is None:
            raise ValueError("robot_xml is required for clip export")
        from omg.tracking.holomotion.export_clip import build_holomotion_deployment_clip

        np.savez(Path(output).with_suffix(".holomotion.npz"), **build_holomotion_deployment_clip(
            q, fps=50., robot_xml=self.config.robot_xml))
