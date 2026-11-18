import json
from pathlib import Path

import pytest

from deployment.realtime.config import load_config


@pytest.fixture
def config_data():
    return dict(checkpoint="assets/flow.pt", text_cache="assets/text", prompt="walk", steps=10,
                lowpass_hz=8., start_history=dict(path="start.npz", key="history"), play_frames=364,
                seed=7, episode_id=3, backend="torch-fp32", device="cpu")


def test_config_paths_and_defaults(tmp_path, config_data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_data))
    config = load_config(path)
    assert config.checkpoint == tmp_path / "assets/flow.pt"
    assert config.text_cache == tmp_path / "assets/text"
    assert config.start_history.path == tmp_path / "start.npz"
    assert config.start_history.key == "history"
    assert (config.start_gate.joint_rad, config.start_gate.height_m, config.start_gate.yaw_deg) == (.3, .1, 10.)
    assert config.switch_seconds == .5 and config.after_end == "hold"
    config_data.pop("text_cache")
    config_data.update(text_encoder="encoder", noise_policy="actor.pt", robot_xml="robot.xml",
                       backend="tensorrt-fp16", device="cuda", engine_directory="engines",
                       start_gate={"joint_rad": .2}, switch_seconds=1., after_end="error")
    path.write_text(json.dumps(config_data))
    config = load_config(path)
    for key in ("text_encoder", "noise_policy", "robot_xml", "engine_directory"):
        assert getattr(config, key) == tmp_path / config_data[key]
    assert config.text_cache is None and config.start_gate.joint_rad == .2
    assert config.start_gate.height_m == .1
    config = load_config(path, backend="torch-fp32", device="cpu", engine_directory=str(tmp_path))
    assert config.engine_directory == tmp_path
    assert config.backend == "torch-fp32"


@pytest.mark.parametrize("key", ["checkpoint", "prompt", "steps", "lowpass_hz", "start_history",
                                     "play_frames", "seed", "episode_id", "backend", "device"])
def test_missing_required_key(tmp_path, config_data, key):
    del config_data[key]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_data))
    with pytest.raises(ValueError, match="Missing required.*" + key):
        load_config(path)


@pytest.mark.parametrize("updates,match", [
    ({"unknown": 1}, "Unknown.*unknown"),
    ({"prompt": ""}, "prompt"), ({"prompt": 1}, "prompt"),
    ({"backend": []}, "backend"), ({"backend": "other"}, "backend"),
    ({"device": None}, "device"), ({"device": ""}, "device"),
    ({"steps": True}, "steps"), ({"steps": 1.5}, "steps"), ({"steps": 0}, "steps"),
    ({"play_frames": 0}, "play_frames"), ({"play_frames": "100"}, "play_frames"),
    ({"seed": -1}, "seed"), ({"seed": False}, "seed"), ({"seed": 2**63}, "seed"),
    ({"episode_id": 1.2}, "episode_id"), ({"episode_id": 2**63}, "episode_id"),
    ({"lowpass_hz": True}, "lowpass_hz"), ({"lowpass_hz": 0}, "lowpass_hz"),
    ({"lowpass_hz": 25}, "lowpass_hz"), ({"lowpass_hz": "8"}, "lowpass_hz"),
    ({"lowpass_hz": float("nan")}, "lowpass_hz"), ({"lowpass_hz": float("inf")}, "lowpass_hz"),
    ({"text_cache": None}, "exactly one"), ({"text_encoder": "encoder"}, "exactly one"),
    ({"text_cache": []}, "text_cache"), ({"checkpoint": None}, "checkpoint"),
    ({"engine_directory": False}, "engine_directory"), ({"noise_policy": 4}, "noise_policy"),
    ({"robot_xml": ""}, "robot_xml"),
    ({"start_history": "start.npz"}, "start_history"),
    ({"start_history": {"path": "start.npz"}}, "start_history"),
    ({"start_history": {"path": "start.npz", "key": "history", "extra": 1}}, "start_history"),
    ({"start_history": {"path": 3, "key": "history"}}, "start_history.path"),
    ({"start_history": {"path": "start.npz", "key": ""}}, "start_history.key"),
    ({"start_gate": []}, "start_gate"), ({"start_gate": {"other": 0}}, "start_gate"),
    ({"start_gate": {"joint_rad": -1}}, "joint_rad"),
    ({"start_gate": {"height_m": True}}, "height_m"),
    ({"start_gate": {"yaw_deg": float("inf")}}, "yaw_deg"),
    ({"switch_seconds": "0.5"}, "switch_seconds"), ({"switch_seconds": .501}, "switch_seconds"),
    ({"after_end": None}, "after_end"), ({"after_end": []}, "after_end"),
    ({"backend": "tensorrt-fp16", "device": "cuda"}, "engine_directory"),
    ({"backend": "torch-fp16"}, "CUDA"),
])
def test_invalid_config(tmp_path, config_data, updates, match):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**config_data, **updates}))
    with pytest.raises(ValueError, match=match):
        load_config(path)


@pytest.mark.parametrize("text", ["[]", "null", "1", "{"])
def test_invalid_json(tmp_path, text):
    path = tmp_path / "config.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_override_and_null_filter(tmp_path, config_data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_data))
    with pytest.raises(ValueError, match="Unknown.*typo"):
        load_config(path, typo=1)
    assert load_config(path, lowpass_hz=None).lowpass_hz is None


def test_example_config():
    config = load_config(Path("deployment/realtime/example_config.json"))
    assert config.backend == "tensorrt-fp16"
    assert config.engine_directory is not None
