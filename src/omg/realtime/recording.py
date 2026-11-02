"""Per-run recorder for the realtime chain (module-level, no ROS dependency).

Records the four module outputs plus the transforms' parameters and the event
timeline, so every other quantity (timeline, tape, obs, target) is derivable:

    M1 main model   : plan block 100x36, one per accepted plan     -> <base>.plans.npz
    M2 corrector    : 25x29 joint delta, one per decision frame    -> <base>.decisions.npz
    M3 tracker      : action 29 per frame                          -> <base>.frames.csv
    M4 robot        : q/dq/tau/quat/gyro per frame                 -> <base>.frames.csv
    events          : request/accept frames, wall times, latency   -> <base>.plans.npz
    transforms      : switch_seconds, action_scale, default_angles,
                      onnx_to_real, residual manifest, tracker meta -> <base>.meta.json

Frame timings are stored next to the frame row (read/tick/encode/consume/step).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

RECORD_VERSION = 1
FRAME_COLUMNS = (
    ["frame", "t"]
    + ["action%d" % i for i in range(29)]
    + ["q%d" % i for i in range(29)]
    + ["dq%d" % i for i in range(29)]
    + ["tau%d" % i for i in range(29)]
    + ["quat%d" % i for i in range(4)]
    + ["gyro%d" % i for i in range(3)]
    + ["residual"]
    + ["read_ms", "tick_ms", "encode_ms", "consume_ms", "step_ms"]
)


class Recorder:
    """Writes one directory-less file family next to `path`:

    <path>.frames.csv, <path>.plans.npz, <path>.decisions.npz, <path>.meta.json
    """

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.base = Path(path)
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.frames_path = Path(str(self.base) + ".frames.csv")
        self.plans_path = Path(str(self.base) + ".plans.npz")
        self.decisions_path = Path(str(self.base) + ".decisions.npz")
        self.meta_path = Path(str(self.base) + ".meta.json")

        self.meta = dict(meta or {})
        self.meta.setdefault("record_version", RECORD_VERSION)
        self.meta.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.meta.setdefault("frame_columns", FRAME_COLUMNS)
        self._write_meta()

        self._frame_file = open(self.frames_path, "w", encoding="utf-8", newline="")
        self._frame_writer = csv.writer(self._frame_file)
        self._frame_writer.writerow(FRAME_COLUMNS)
        self._frames = 0

        self._plans: list[dict] = []
        self._decisions: list[dict] = []
        self._save_plans()
        self._save_decisions()

    # -- metadata ---------------------------------------------------------
    def update_meta(self, **fields) -> None:
        self.meta.update(fields)
        self._write_meta()

    def _write_meta(self) -> None:
        self.meta_path.write_text(json.dumps(self.meta, indent=1, default=str), encoding="utf-8")

    # -- M1 events --------------------------------------------------------
    def plan(self, *, plan_id, request_frame, accept_frame, switch_frame, deadline_frame,
             block, request_time=None, accept_time=None) -> None:
        block = np.asarray(block, dtype=np.float32)
        latency = None
        if request_time is not None and accept_time is not None:
            latency = float((accept_time - request_time) * 1000.0)
        self._plans.append(dict(
            plan_id=int(plan_id), request_frame=int(request_frame),
            accept_frame=int(accept_frame), switch_frame=int(switch_frame),
            deadline_frame=int(deadline_frame), request_time=request_time,
            accept_time=accept_time, latency_ms=latency, block=block,
        ))
        self._save_plans()

    def _save_plans(self) -> None:
        if not self._plans:
            np.savez_compressed(
                self.plans_path,
                plan_id=np.zeros(0, np.int64), request_frame=np.zeros(0, np.int64),
                accept_frame=np.zeros(0, np.int64), switch_frame=np.zeros(0, np.int64),
                deadline_frame=np.zeros(0, np.int64), request_time=np.zeros(0, np.float64),
                accept_time=np.zeros(0, np.float64), latency_ms=np.zeros(0, np.float64),
                blocks=np.zeros((0, 0, 36), np.float32),
            )
            return
        np.savez_compressed(
            self.plans_path,
            plan_id=np.array([p["plan_id"] for p in self._plans], np.int64),
            request_frame=np.array([p["request_frame"] for p in self._plans], np.int64),
            accept_frame=np.array([p["accept_frame"] for p in self._plans], np.int64),
            switch_frame=np.array([p["switch_frame"] for p in self._plans], np.int64),
            deadline_frame=np.array([p["deadline_frame"] for p in self._plans], np.int64),
            request_time=np.array([p["request_time"] if p["request_time"] is not None else np.nan
                                   for p in self._plans], np.float64),
            accept_time=np.array([p["accept_time"] if p["accept_time"] is not None else np.nan
                                  for p in self._plans], np.float64),
            latency_ms=np.array([p["latency_ms"] if p["latency_ms"] is not None else np.nan
                                 for p in self._plans], np.float64),
            blocks=np.stack([p["block"] for p in self._plans]),
        )

    # -- M2 events --------------------------------------------------------
    def decision(self, *, frame, start, delta, latent_norm=None) -> None:
        delta = np.asarray(delta, dtype=np.float32)
        self._decisions.append(dict(
            frame=int(frame), start=int(start), delta=delta,
            latent_norm=None if latent_norm is None else float(latent_norm),
        ))
        self._save_decisions()

    def _save_decisions(self) -> None:
        if not self._decisions:
            np.savez_compressed(
                self.decisions_path,
                frame=np.zeros(0, np.int64), start=np.zeros(0, np.int64),
                latent_norm=np.zeros(0, np.float64), delta=np.zeros((0, 0, 29), np.float32),
            )
            return
        np.savez_compressed(
            self.decisions_path,
            frame=np.array([d["frame"] for d in self._decisions], np.int64),
            start=np.array([d["start"] for d in self._decisions], np.int64),
            latent_norm=np.array([np.nan if d["latent_norm"] is None else d["latent_norm"]
                                  for d in self._decisions], np.float64),
            delta=np.stack([d["delta"] for d in self._decisions]),
        )

    # -- per-frame --------------------------------------------------------
    def frame(self, *, frame, t, action, q, dq, tau, quat, gyro, residual,
              read_ms=0.0, tick_ms=0.0, encode_ms=0.0, consume_ms=0.0, step_ms=0.0) -> None:
        row = [int(frame), float(t)]
        row += np.asarray(action, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(q, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(dq, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(tau, np.float32).reshape(-1)[:29].tolist()
        row += np.asarray(quat, np.float32).reshape(-1)[:4].tolist()
        row += np.asarray(gyro, np.float32).reshape(-1)[:3].tolist()
        row += [int(bool(residual)), float(read_ms), float(tick_ms), float(encode_ms),
                float(consume_ms), float(step_ms)]
        if len(row) != len(FRAME_COLUMNS):
            raise ValueError("frame row has %d fields, expected %d" % (len(row), len(FRAME_COLUMNS)))
        self._frame_writer.writerow(row)
        self._frame_file.flush()
        self._frames += 1

    def close(self) -> None:
        try:
            self._frame_file.close()
        finally:
            self._save_plans()
            self._save_decisions()
            self.update_meta(frames=self._frames,
                             plans=len(self._plans), decisions=len(self._decisions))
