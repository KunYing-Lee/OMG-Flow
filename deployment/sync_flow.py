"""OMG-Flow synchronous planning and an absolute-time reference buffer."""

from time import perf_counter_ns

import numpy as np

# Preserve the existing scheduler and external imports.
from deployment.flow_runtime import FlowRuntime as FlowGenerator, filter_suffix

TICK_NS = 20_000_000


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
