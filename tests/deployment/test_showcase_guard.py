import json
import os
import signal
import sys
from types import SimpleNamespace

import pytest

from scripts.validation import sync_showcase


@pytest.mark.parametrize("arrives_later", [False, True])
def test_foreign_gpu_process_prevents_or_stops_our_run(tmp_path, monkeypatch, arrives_later):
    asset = tmp_path / "asset"
    asset.write_bytes(b"fixture")
    asset.with_name("4de98d3c-9d3c-11f1-aff4-06f301228322").write_bytes(b"weights")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": "tensorrt-fp16", "replan_period": 0.2,
                                 "lowpass_hz": 8.0, "steps": 10,
                                 **{key: str(asset) for key in ("checkpoint", "tracker", "scene")}}))
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"name": "test", "prompt": "walk", "seed": 1, "duration": 20}]))
    monkeypatch.setattr(sys, "argv", ["showcase", "--config", str(config), "--cases", str(cases),
                                      "--output", str(tmp_path / "out"), "--gpu", "3"])
    foreign = f"<nvidia_smi_log><gpu><processes><process_info><pid>{os.getpid()}</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>"
    responses = iter(["GPU-test", *(["<nvidia_smi_log/>"] if arrives_later else []), foreign])
    monkeypatch.setattr(sync_showcase.subprocess, "check_output", lambda *a, **k: next(responses))
    started, stopped = [], []
    child = SimpleNamespace(pid=-123, poll=lambda: None, wait=lambda: 0)
    monkeypatch.setattr(sync_showcase.subprocess, "Popen", lambda *a, **k: started.append(child) or child)
    monkeypatch.setattr(sync_showcase.os, "killpg", lambda pid, sig: stopped.append((pid, sig)))
    monkeypatch.setattr(sync_showcase.time, "sleep", lambda _: None)
    with pytest.raises(SystemExit, match="occupied"):
        sync_showcase.main()
    assert len(started) == int(arrives_later)
    assert stopped == ([(child.pid, signal.SIGTERM)] if arrives_later else [])
    assert not (tmp_path / "out" / "manifest.json").exists()
