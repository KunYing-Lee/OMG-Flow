import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.realtime.cli import bake, main, request
from deployment.realtime.config import load_config
from deployment.realtime.server import StreamSession
from deployment.realtime.source import FlowSource


@pytest.fixture
def source_setup(tmp_path, monkeypatch):
    import deployment.flow_runtime

    calls = []
    class FakeRuntime:
        def __init__(self, cfg):
            self.cfg = cfg
            self.reference = SimpleNamespace(mean=torch.zeros(38), std=torch.ones(38))
            self.contract = dict(prompt=cfg["prompt"], prefix_length=cfg["prefix_length"])
            self.trt = None
            calls.append(("construct", cfg))

        def generate(self, history, committed, noise=None, **kwargs):
            calls.append(("generate", history.copy(), committed.copy(), noise, kwargs))
            q = np.repeat(history[-1:], 100-len(committed), axis=0)
            q[:, 7] = np.arange(len(q), dtype=np.float32) * .001 + history[-1, 7]
            return q

    monkeypatch.setattr(deployment.flow_runtime, "FlowRuntime", FakeRuntime)
    history = np.zeros((10, 36), np.float32)
    history[:, 3] = 1
    history[:, 7] = .05
    np.savez(tmp_path / "start.npz", fixed=history)
    data = dict(checkpoint="flow.pt", prompt="fixed", steps=10, lowpass_hz=8.,
                start_history=dict(path="start.npz", key="fixed"), play_frames=100,
                seed=7, episode_id=3, backend="torch-fp32", device="cpu", text_cache="text")
    path = tmp_path / "planner.json"
    path.write_text(json.dumps(data))
    return path, calls, history


def test_source_seed_path_and_start_contract(source_setup, tmp_path):
    path, calls, history = source_setup
    source = FlowSource(load_config(path))
    assert [call[1]["prefix_length"] for call in calls] == [0, 25]
    assert all(call[1]["checkpoint"] == path.parent / "flow.pt" for call in calls)
    session = StreamSession(source)
    measured = history.copy()
    measured[:, 7] += .1
    first = session.plan(request(measured, 0, session.schedule, "fixed"), {})
    assert calls[2][1].tobytes() == history.tobytes()
    assert calls[2][2].shape == (0, 36)
    assert calls[2][3] is None
    assert calls[2][4] == dict(seed=7, replan_index=0)
    session.plan(request(first.qpos_36[15:25], 1, session.schedule, "fixed"), {})
    assert [call[4]["replan_index"] for call in calls[2:]] == [0, 1, 2]
    assert calls[-1][2].shape == (25, 36)
    metadata = source.metadata()
    assert metadata["caption"] == "fixed"
    assert metadata["source"] == "flow" and metadata["policy_sha256"] is None
    assert metadata["engine"] is None and metadata["package"] is None
    output = tmp_path / "baked.npz"
    baked = bake(source, output)
    with np.load(output, allow_pickle=False) as saved:
        assert saved.files == ["qpos_36", "fps", "play_frames", "history", "metadata"]
        assert saved["qpos_36"].shape == (111, 36)
        assert saved["qpos_36"].tobytes() == baked.tobytes()
        assert saved["history"].tobytes() == history.tobytes()
        assert json.loads(saved["metadata"][0])["after_end"] == "hold"
    assert not output.with_suffix(".holomotion.npz").exists()
    with pytest.raises(ValueError, match="robot_xml"):
        source.export(baked, output)


def test_source_noise_provider_receives_every_context(source_setup, monkeypatch):
    import deployment.realtime.noise_policy

    path, calls, history = source_setup
    policy_path = path.parent / "actor.pt"
    policy_path.write_bytes(b"synthetic policy loader test")
    contexts = []
    class Provider:
        def noise(self, context, *, request_id):
            contexts.append((request_id, context.copy()))
            return torch.full((100, 38), float(request_id))
    def load_policy(file, **kwargs):
        assert file == policy_path
        assert kwargs["seed"] == 7 and kwargs["episode_id"] == 3
        assert kwargs["device"] == "cpu"
        return Provider()
    monkeypatch.setattr(deployment.realtime.noise_policy, "load_noise_policy", load_policy)
    source = FlowSource(load_config(path, noise_policy="actor.pt"))
    source.reset()
    source.chunk(0, 0)
    source.chunk(1, 25)
    assert [row[0] for row in contexts] == [0, 1, 2]
    assert contexts[0][1].tobytes() == history.tobytes()
    assert contexts[1][1].shape == contexts[2][1].shape == (35, 36)
    for index, call in enumerate(calls[2:]):
        assert call[4] == {}
        assert torch.equal(call[3], torch.full((100, 38), float(index)))
    assert source.metadata()["source"] == "flow-noise"
    assert len(source.metadata()["policy_sha256"]) == 64


def test_optional_clip_export(source_setup, tmp_path, monkeypatch):
    from omg.tracking.holomotion import export_clip

    path, _, _ = source_setup
    source = FlowSource(load_config(path, robot_xml="robot.xml"))
    def build(q, *, fps, robot_xml):
        assert fps == 50 and robot_xml == path.parent / "robot.xml"
        return dict(qpos=q)
    monkeypatch.setattr(export_clip, "build_holomotion_deployment_clip", build)
    q = bake(source, tmp_path / "clip.npz")
    with np.load(tmp_path / "clip.holomotion.npz", allow_pickle=False) as data:
        assert data["qpos"].tobytes() == q.tobytes()


@pytest.mark.parametrize("change,match", [
    ({"start_history": {"path": "start.npz", "key": "missing"}}, "missing"),
    ({"play_frames": 100, "switch_seconds": 2.}, "terminal generated support"),
])
def test_source_preflight_before_runtime(source_setup, change, match):
    path, calls, _ = source_setup
    with pytest.raises(ValueError, match=match):
        FlowSource(load_config(path, **change))
    assert calls == []


def test_cli_bake_overrides(source_setup, tmp_path, monkeypatch):
    path, calls, _ = source_setup
    monkeypatch.setattr("sys.argv", ["realtime", "bake", "--config", str(path),
        "--output", str(tmp_path / "cli.npz"), "--backend", "torch-fp32", "--device", "cpu",
        "--engine-directory", str(tmp_path / "engines"), "--switch-seconds", "1", "--after-end", "error"])
    main()
    with np.load(tmp_path / "cli.npz", allow_pickle=False) as data:
        metadata = json.loads(data["metadata"][0])
        assert metadata["period_frames"] == 50 and metadata["after_end"] == "error"
    assert calls[0][1]["engine_cache"] == tmp_path / "engines"


def test_cli_serve_passes_gate_and_warms(source_setup, monkeypatch):
    import deployment.realtime.cli

    path, calls, _ = source_setup
    data = json.loads(path.read_text())
    data["start_gate"] = dict(joint_rad=.2, height_m=.03, yaw_deg=4.)
    path.write_text(json.dumps(data))
    received = []
    def fake_serve(source, endpoint, **kwargs):
        assert len(calls) == 3  # Two runtimes and an initial warmup request.
        received.append((source, endpoint, kwargs))
    monkeypatch.setattr(deployment.realtime.cli, "serve", fake_serve)
    monkeypatch.setattr("sys.argv", ["realtime", "serve", "--config", str(path),
        "--bind", "inproc://cli-test", "--log", str(path.parent / "server.jsonl"),
        "--max-plans", "2", "--allow-start-mismatch"])
    main()
    assert received[0][1] == "inproc://cli-test"
    assert received[0][2]["max_plans"] == 2
    assert received[0][2]["allow_start_mismatch"]
    assert received[0][2]["start_gate"].yaw_deg == 4.
