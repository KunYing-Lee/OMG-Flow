"""The recorder's event path must not run compression on the 50 Hz control loop."""
from __future__ import annotations

import time

import numpy as np

from omg.realtime.recording import Recorder


def _plan(rec: Recorder, i: int) -> None:
    rec.plan(
        plan_id=i,
        request_frame=i * 25,
        accept_frame=i * 25 + 3,
        switch_frame=i * 25 + 25,
        deadline_frame=i * 25 + 14,
        block=np.zeros((100, 36), np.float32),
        request_time=0.0,
        accept_time=0.01,
    )


def _decision(rec: Recorder, i: int) -> None:
    rec.decision(
        frame=i, start=i * 25, delta=np.zeros((25, 29), np.float32), latent_norm=1.0
    )


def test_event_calls_do_not_compress_on_the_caller(monkeypatch, tmp_path):
    rec = Recorder(tmp_path / "run")
    original = np.savez_compressed

    def slow_savez(*args, **kwargs):
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", slow_savez)
    started = time.perf_counter()
    for i in range(8):
        _plan(rec, i)
        _decision(rec, i)
    elapsed = time.perf_counter() - started
    # 16 events; the slow writer is deliberately 50 ms per write.  The caller
    # must only append + set an event.
    assert elapsed < 0.05, f"event path took {elapsed:.3f}s"

    rec.close()
    assert not rec._writer.is_alive()
    plans = np.load(tmp_path / "run.plans.npz")
    decisions = np.load(tmp_path / "run.decisions.npz")
    assert len(plans["plan_id"]) == 8
    assert len(decisions["frame"]) == 8


def test_close_is_idempotent_and_keeps_counts(tmp_path):
    rec = Recorder(tmp_path / "run")
    _plan(rec, 0)
    _decision(rec, 0)
    rec.close()
    rec.close()
    plans = np.load(tmp_path / "run.plans.npz")
    decisions = np.load(tmp_path / "run.decisions.npz")
    assert len(plans["plan_id"]) == 1
    assert len(decisions["frame"]) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_many_events_are_all_written(tmp_path):
    rec = Recorder(tmp_path / "run")
    for i in range(50):
        _plan(rec, i)
        _decision(rec, i)
    rec.close()
    plans = np.load(tmp_path / "run.plans.npz")
    decisions = np.load(tmp_path / "run.decisions.npz")
    assert np.asarray(plans["plan_id"]).tolist() == list(range(50))
    assert np.asarray(decisions["frame"]).tolist() == list(range(50))
